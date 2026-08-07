#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_ROOT="$(cd -- "${SCRIPT_DIR}/verl/tools" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/" && pwd)"

###############################################################################
# 只需要修改这一段。运行本脚本时不要再在命令行追加训练脚本和 Hydra 参数。
###############################################################################
TRAINING_COMMAND=(
    bash "${REPO_ROOT}/run_kimi_k3_mindspeed.sh"
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=false
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.calculate_log_probs=true
    trainer.use_v1=true
    trainer.v1.trainer_mode=sync
    trainer.critic_warmup=0
)

# synthetic reward 也在这里固定；需要调整时直接修改脚本中的数值。
SYNTHETIC_REWARD_SCALE="1.0"

if [[ "$#" -ne 0 ]]; then
    echo "本脚本不接收命令行参数，请直接编辑脚本顶部的 TRAINING_COMMAND 后重新运行。" >&2
    exit 2
fi

TRAINING_SCRIPT="${TRAINING_COMMAND[1]:-}"
if [[ "${TRAINING_COMMAND[0]:-}" == "bash" && ! -f "${TRAINING_SCRIPT}" ]]; then
    echo "找不到训练脚本: ${TRAINING_SCRIPT}" >&2
    echo "请编辑 ${BASH_SOURCE[0]} 顶部的 TRAINING_COMMAND。" >&2
    exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}/_bootstrap:${TOOLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_SYNTHETIC_REWARD_DEBUG="1"
export VERL_SYNTHETIC_REWARD_DEBUG_STRICT="1"
export VERL_SYNTHETIC_REWARD_SCALE="${SYNTHETIC_REWARD_SCALE}"
exec "${TRAINING_COMMAND[@]}"
