#!/usr/bin/env bash
# Kimi K3 pure-language SFT with MindSpeed FSDP-Turbo on Ascend NPUs.
#
# Example:
#   MODEL_PATH=/models/Kimi-K3-Base \
#   TRAIN_FILE=/data/kimi-sft/train.parquet \
#   VAL_FILE=/data/kimi-sft/val.parquet \
#   bash examples/sft/kimi_k3/run_kimi_k3_fsdp_turbo.sh 16 /checkpoints/kimi-k3-sft

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <nproc_per_node> <save_path> [Hydra overrides...]"
    exit 1
fi

NPROC_PER_NODE=$1
SAVE_PATH=$2
shift 2

# ---- user-adjustable model and data settings ----
FSDP_TURBO_ROOT=${FSDP_TURBO_ROOT:-/path/to/FSDPTurbo}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to a KimiLinearForCausalLM checkpoint}
TRAIN_FILE=${TRAIN_FILE:?Set TRAIN_FILE to a messages-format parquet file}
VAL_FILE=${VAL_FILE:-null}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

FSDP_SIZE=${FSDP_SIZE:-${NPROC_PER_NODE}}
EP_SIZE=${EP_SIZE:-1}
EFSDP_SIZE=${EFSDP_SIZE:-1}
TURBO_CP_SIZE=${TURBO_CP_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${NPROC_PER_NODE}}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PROJECT_NAME=${PROJECT_NAME:-kimi-k3-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi-k3-fsdp-turbo}
# ---- end user-adjustable settings ----

if [[ ! -d "${FSDP_TURBO_ROOT}/fsdp_turbo" ]]; then
    echo "FSDP-Turbo was not found at ${FSDP_TURBO_ROOT}; set FSDP_TURBO_ROOT to its source root." >&2
    exit 1
fi

if (( NPROC_PER_NODE % FSDP_SIZE != 0 )); then
    echo "FSDP_SIZE (${FSDP_SIZE}) must divide nproc_per_node (${NPROC_PER_NODE})." >&2
    exit 1
fi

if (( FSDP_SIZE % EP_SIZE != 0 )); then
    echo "EP_SIZE (${EP_SIZE}) must divide FSDP_SIZE (${FSDP_SIZE})." >&2
    exit 1
fi

if (( TURBO_CP_SIZE != 1 )); then
    echo "KimiLinearForCausalLM has no FSDP-Turbo Ulysses function-patch plan yet; TURBO_CP_SIZE must be 1." >&2
    exit 1
fi

export PYTHONPATH="${FSDP_TURBO_ROOT}:${PYTHONPATH:-}"
export NON_MEGATRON=${NON_MEGATRON:-true}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}

mkdir -p "${SAVE_PATH}"

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    --config-name sft_trainer_mindspeed_fsdp_turbo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.use_dynamic_bsz=true \
    model.path="${MODEL_PATH}" \
    model.enable_gradient_checkpointing=false \
    model.random_init=true \
    optim.lr="${LR}" \
    optim.optimizer=AdamW \
    engine.fsdp_kwargs.distributed.fully_shard_parallel_size="${FSDP_SIZE}" \
    engine.fsdp_kwargs.distributed.expert_parallel_size="${EP_SIZE}" \
    engine.fsdp_kwargs.distributed.expert_fully_shard_parallel_size="${EFSDP_SIZE}" \
    engine.fsdp_kwargs.distributed.ulysses_parallel_size="${TURBO_CP_SIZE}" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
