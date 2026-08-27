#!/usr/bin/env bash
# Kimi-K3 CountBench GRPO: vLLM-Ascend rollout + Megatron/MindSpeed actor/ref.

set -euo pipefail

BACKEND_ROOT=${BACKEND_ROOT:-/mnt/share/w00848461/kimi-k3/megatron-backend}
MODEL_PATH=${MODEL_PATH:-/mnt/share/w00848461/weights/kimi-k3-4layer-16expert}
EXPECTED_NUM_HIDDEN_LAYERS=${EXPECTED_NUM_HIDDEN_LAYERS:-4}
EXPECTED_NUM_EXPERTS=${EXPECTED_NUM_EXPERTS:-16}
TRAIN_FILE=${TRAIN_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train_448.parquet}
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-${BACKEND_ROOT}/.python_deps/transformers-5.10.4}
MODELOPT_SITE=${MODELOPT_SITE:-${BACKEND_ROOT}/.python_deps/modelopt-0.46.0}
VLLM_METADATA_SITE=${VLLM_METADATA_SITE:-${BACKEND_ROOT}/.python_deps/vllm-0.26.0}
VLLM_SOURCE=${VLLM_SOURCE:-${BACKEND_ROOT}/vllm}
VLLM_ASCEND_SOURCE=${VLLM_ASCEND_SOURCE:-${BACKEND_ROOT}/vllm-ascend}
CANN_HOME=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
CANN_PYTHON_SITE=${CANN_PYTHON_SITE:-${CANN_HOME}/python/site-packages}
CANN_TBE_SITE=${CANN_TBE_SITE:-${CANN_HOME}/opp/built-in/op_impl/ai_core/tbe}

NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
ACTOR_TP=${ACTOR_TP:-4}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-1}
ACTOR_EP=${ACTOR_EP:-2}
ACTOR_ETP=${ACTOR_ETP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-8}
ROLLOUT_EP=${ROLLOUT_EP:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
MAX_VISUAL_TOKENS=${MAX_VISUAL_TOKENS:-256}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-1408}

PROJECT_NAME=${PROJECT_NAME:-verl_kimi_k3}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi_k3_4layer_countbench_8npu}
OUTPUT_ROOT=${OUTPUT_ROOT:-${BACKEND_ROOT}/verl/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${BACKEND_ROOT}/verl/logs}

TOTAL_EPOCHS=10
TOTAL_TRAINING_STEPS=500

required_paths=(
    "$MODEL_PATH/config.json"
    "$TRAIN_FILE"
    "$TRANSFORMERS_SITE/transformers/__init__.py"
    "$MODELOPT_SITE/modelopt/__init__.py"
    "$MODELOPT_SITE/nvidia_modelopt-0.46.0.dist-info/METADATA"
    "$VLLM_METADATA_SITE/vllm-0.26.0.dist-info/METADATA"
    "$VLLM_SOURCE/vllm/__init__.py"
    "$VLLM_ASCEND_SOURCE/vllm_ascend/__init__.py"
    "$CANN_PYTHON_SITE/acl.so"
    "$CANN_TBE_SITE"
)
for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required path: $path" >&2
        exit 2
    fi
done

required_mm_length=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH - 1 + MAX_VISUAL_TOKENS))
if ((MAX_MODEL_LENGTH < required_mm_length)); then
    echo "MAX_MODEL_LENGTH=$MAX_MODEL_LENGTH is smaller than the multimodal budget $required_mm_length" >&2
    exit 2
fi

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}
unset NON_MEGATRON || true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export MINDSPEED_BRIDGE_AUTOREG_MODE=${MINDSPEED_BRIDGE_AUTOREG_MODE:-off}
export VERL_USE_MEGATRON_ADAPTOR=enabled
export EXPECTED_NUM_HIDDEN_LAYERS EXPECTED_NUM_EXPERTS
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp48s3u1u1}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-17340}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-256}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_VERSION=${VLLM_VERSION:-0.26.0}
export VLLM_ASCEND_ENABLE_FLASHCOMM=${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}

source_roots=(
    "$TRANSFORMERS_SITE"
    "$MODELOPT_SITE"
    "$VLLM_METADATA_SITE"
    "$VLLM_SOURCE"
    "$VLLM_ASCEND_SOURCE"
    "$BACKEND_ROOT/verl"
    "$BACKEND_ROOT/MS-Bridge-KIMI-K3"
    "$BACKEND_ROOT/Megatron-Bridge/src"
    "$BACKEND_ROOT/Megatron-LM-KIMI-K3"
    "$BACKEND_ROOT/MegatronAdaptor"
    "$BACKEND_ROOT/TransformerEngineNPU"
    "$BACKEND_ROOT/MindSpeed-Ops"
    "$CANN_PYTHON_SITE"
    "$CANN_TBE_SITE"
)
joined_pythonpath=$(IFS=:; echo "${source_roots[*]}")
export PYTHONPATH="$joined_pythonpath"

python3 - "$MODEL_PATH/config.json" "$VLLM_SOURCE" "$VLLM_ASCEND_SOURCE" "$MODELOPT_SITE" <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import torch
import transformers
import modelopt
from modelopt.torch.quantization.utils import is_quantized
import vllm
import vllm._version
import vllm_ascend
import vllm_ascend.models.kimi_k3 as kimi_k3_model
from vllm_ascend.utils import enable_custom_op

module_version = transformers.__version__
metadata_version = importlib.metadata.version("transformers")
if (module_version, metadata_version) != ("5.10.4", "5.10.4"):
    raise SystemExit(
        "Kimi-K3 vLLM 0.26 migration requires transformers 5.10.4 from TRANSFORMERS_SITE, "
        f"got module={module_version}, metadata={metadata_version}"
    )
print(f"transformers version passed: module={module_version}, metadata={metadata_version}")

vllm_module_version = vllm.__version__
vllm_metadata_version = importlib.metadata.version("vllm")
if (vllm_module_version, vllm_metadata_version) != ("0.26.0", "0.26.0"):
    raise SystemExit(
        "Kimi-K3 requires vLLM 0.26.0 from VLLM_SOURCE, "
        f"got module={vllm_module_version}, metadata={vllm_metadata_version}"
    )
print(f"vllm version passed: module={vllm_module_version}, metadata={vllm_metadata_version}")

def require_source(module_file: str, source_root: str, name: str) -> None:
    actual_path = Path(module_file).resolve()
    expected_root = Path(source_root).resolve()
    if not actual_path.is_relative_to(expected_root):
        raise SystemExit(f"{name} resolved outside {expected_root}: {actual_path}")
    print(f"{name} source passed: {actual_path}")

require_source(vllm.__file__, sys.argv[2], "vllm")
require_source(vllm._version.__file__, sys.argv[2], "vllm._version")
require_source(vllm_ascend.__file__, sys.argv[3], "vllm_ascend")
require_source(kimi_k3_model.__file__, sys.argv[3], "vllm_ascend.models.kimi_k3")
modelopt_version = importlib.metadata.version("nvidia-modelopt")
if modelopt_version != "0.46.0" or not callable(is_quantized):
    raise SystemExit(f"expected usable nvidia-modelopt 0.46.0, got {modelopt_version}")
require_source(modelopt.__file__, sys.argv[4], "modelopt")

for module_name, module in tuple(sys.modules.items()):
    if not (module_name == "vllm" or module_name.startswith(("vllm.", "vllm_ascend"))):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file and ("/vllm-workspace/vllm/" in module_file or "/vllm-ascend-kimi-k3/" in module_file):
        raise SystemExit(f"legacy source leaked into {module_name}: {module_file}")

required_kda_ops = ("recurrent_kda", "chunk_kda_fwd", "kda_gate_cumsum")
if not enable_custom_op():
    raise SystemExit("vLLM-Ascend custom operator extension could not be enabled")
missing_kda_ops = [name for name in required_kda_ops if not hasattr(torch.ops._C_ascend, name)]
if missing_kda_ops:
    raise SystemExit(f"vLLM-Ascend KDA operator schemas are missing: {missing_kda_ops}")
print(f"KDA operator schemas passed: {', '.join(required_kda_ops)}")

with open(sys.argv[1], encoding="utf-8") as config_file:
    config = json.load(config_file)
text_config = config.get("text_config", config)
actual = (text_config.get("num_hidden_layers"), text_config.get("num_experts"))
expected = (
    int(os.environ["EXPECTED_NUM_HIDDEN_LAYERS"]),
    int(os.environ["EXPECTED_NUM_EXPERTS"]),
)
if actual != expected:
    raise SystemExit(f"expected a {expected[0]}-layer/{expected[1]}-expert checkpoint, got {actual}")
print(f"model config passed: layers={actual[0]}, experts={actual[1]}")
PY

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.image_key=images
    data.train_batch_size=$TRAIN_BATCH_SIZE
    data.max_prompt_length=$MAX_PROMPT_LENGTH
    data.max_response_length=$MAX_RESPONSE_LENGTH
    data.filter_overlong_prompts=True
    # Kimi-K3's remote-code tokenizer retains an SSLContext, so even
    # datasets' one-worker pool cannot pickle it.
    # A null num_proc keeps this lightweight filter in the Ray actor process.
    data.filter_overlong_prompts_workers=null
    data.truncation=error
    data.shuffle=False
    data.validation_shuffle=False
    data.trust_remote_code=True
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True
    actor_rollout_ref.actor.optim.main_grads_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_dtype=bf16
    actor_rollout_ref.actor.optim.exp_avg_sq_dtype=bf16
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$ACTOR_TP
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$ACTOR_PP
    actor_rollout_ref.actor.megatron.context_parallel_size=$ACTOR_CP
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$ACTOR_EP
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ACTOR_ETP
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=$MAX_MODEL_LENGTH
    # The current Ascend fused KDA produces non-finite actor logits for this
    # checkpoint. Keep the portable implementation as the correctness default.
    +actor_rollout_ref.actor.megatron.override_transformer_config.kimi_use_fused_kda=${KIMI_USE_FUSED_KDA:-False}
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.megatron.use_mbridge=True
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False
    actor_rollout_ref.ref.megatron.use_remove_padding=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$ACTOR_TP
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$ACTOR_PP
    actor_rollout_ref.ref.megatron.context_parallel_size=$ACTOR_CP
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=$ACTOR_EP
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=$ACTOR_ETP
    actor_rollout_ref.ref.megatron.sequence_parallel=True
    actor_rollout_ref.ref.megatron.param_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP
    actor_rollout_ref.rollout.expert_parallel_size=$ROLLOUT_EP
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-2}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LENGTH
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS:-2048}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-True}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.load_format=dummy
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
)

TRAINER=(
    trainer.logger='["console"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.nnodes=$NNODES
    trainer.n_gpus_per_node=$NPUS_PER_NODE
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ:--1}
    trainer.test_freq=${TEST_FREQ:--1}
    trainer.total_epochs=${TOTAL_EPOCHS:-1}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-1}
    trainer.default_local_dir="$OUTPUT_ROOT"
)

RAY=(
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$PYTHONPATH"
    ++ray_kwargs.ray_init.runtime_env.env_vars.MINDSPEED_BRIDGE_AUTOREG_MODE="$MINDSPEED_BRIDGE_AUTOREG_MODE"
    ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_MEGATRON_ADAPTOR="$VERL_USE_MEGATRON_ADAPTOR"
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_VERSION="$VLLM_VERSION"
    ++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME"
    ++ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
    ++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_HOST_SOCKET_PORT_RANGE="$HCCL_HOST_SOCKET_PORT_RANGE"
    ++ray_kwargs.ray_init.runtime_env.env_vars.HCCL_NPU_SOCKET_PORT_RANGE="$HCCL_NPU_SOCKET_PORT_RANGE"
)

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)
cd "$BACKEND_ROOT/verl"

python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=megatron \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@" 2>&1 | tee "$LOG_DIR/kimi-k3-megatron-$timestamp.log"
