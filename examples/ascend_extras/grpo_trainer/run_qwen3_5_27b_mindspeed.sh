#!/bin/bash
set -xeuo pipefail
# Project Configuration
project_name='GRPO-Qwen3-5-27B'
exp_name='GRPO-Qwen3-5-27B-FSDPTurbo-VLLM'

# environment
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export OMP_NUM_THREADS=1

gen_tp=8
turbo_cp_size=2

FSDP_TURBO_ROOT=${FSDP_TURBO_ROOT:-/Users/albert/code/ascend/FSDPTurbo}
if [[ ! -d "${FSDP_TURBO_ROOT}/fsdp_turbo" ]]; then
    echo "FSDP-Turbo was not found at ${FSDP_TURBO_ROOT}; set FSDP_TURBO_ROOT to its source or install path."
    exit 1
fi
export PYTHONPATH="${FSDP_TURBO_ROOT}:${PYTHONPATH:-}"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3.5-27B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/geo3k/test.parquet"}

start_time=$(date +%Y%m%d)_$(date +%H%M%S)


DATA_CONFIG=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=64
    data.max_prompt_length=1024
    data.max_response_length=2048
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    data.image_key=images
)

TRAINER_CONFIG=(
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=16
    trainer.nnodes=1
    trainer.resume_from_path=checkpoints/
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_epochs=10
)

MODEL_CONFIG=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR_CONFIG=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.lr_warmup_ratio=-1
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=AdamW
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
)

ROLLOUT_CONFIG=(
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.max_model_len=32768
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=8192
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.n=5
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes="[4,8,12,16,24,32,48,56,64]"
)

MINDSPEED_CONFIG=(
    actor_rollout_ref.actor.mindspeed.strategy=mindspeed_fsdp
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.fsdp_plan.ignored_modules=[]
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.fsdp_plan.hook_modules="['model.language_model.layers.{*}']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.tp_plan.colwise_parallel="['*.q_proj', '*.k_proj', '*.v_proj']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.tp_plan.rowwise_parallel="['*.o_proj']"
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.fully_shard_parallel_size=8
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.ulysses_parallel_size=${turbo_cp_size}
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.distributed.cp_plan.ulysses_function_patches="[{target_functions:['transformers.models.qwen3_5.modeling_qwen3_5.eager_attention_forward'],type:full_attention},{target_functions:['transformers.loss.loss_utils.ForCausalLMLoss'],type:causal_lm_loss}]"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.module_patches="[{target_function:transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward,patch_function:examples.qwen3_5.modeling_qwen3_5.qwen3_5_gated_delta_net_forward},{target_function:transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5Model.forward,patch_function:examples.qwen3_5.modeling_qwen3_5.qwen3_5_model_forward}]"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.memory.recompute_plan="['model.language_model.layers.{*}','model.visual.blocks.{*}']"
    # The FSDP engine default keeps verl's parent Ulysses path disabled; CP is
    # owned by FSDP-Turbo.
    # FSDP-Turbo's native FSDP2 wrappers own placement; verl CPU offload is not
    # combined with them in this experiment.
    actor_rollout_ref.actor.mindspeed.param_offload=False
    actor_rollout_ref.actor.mindspeed.optimizer_offload=False
    actor_rollout_ref.actor.mindspeed.offload_policy=False
    actor_rollout_ref.ref.mindspeed.param_offload=False
    actor_rollout_ref.ref.mindspeed.offload_policy=False
)

REF_CONFIG=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.use_torch_compile=False
)


mkdir -p logs
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    algorithm.adv_estimator=grpo \
    "${DATA_CONFIG[@]}" \
    "${MODEL_CONFIG[@]}" \
    "${ACTOR_CONFIG[@]}" \
    "${REF_CONFIG[@]}" \
    "${ROLLOUT_CONFIG[@]}" \
    "${MINDSPEED_CONFIG[@]}" \
    "${TRAINER_CONFIG[@]}" \
    algorithm.use_kl_in_reward=False \
    $@ 2>&1 | tee logs/qwen3_5-27b-${start_time}.log
