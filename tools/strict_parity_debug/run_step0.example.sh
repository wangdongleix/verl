#!/usr/bin/env bash
# One-command initial-weight parity run. Copy to a private path, adjust the
# paths below, and execute it with bash. Do not source it.
set -euo pipefail

VERL_ROOT=${VERL_ROOT:-/mnt/share/w00848461/kimi-k3/verl}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-/mnt/share/w00848461/kimi-k3/run_kimi_k3_mindspeed.sh}
OUT=${STRICT_PARITY_DIR:-/mnt/share/w00848461/kimi-k3/strict_parity_step0}
TARGET_GLOBAL_STEP=${STRICT_PARITY_GLOBAL_STEP:-1}
VLLM_ACTOR_NAME=${VLLM_ACTOR_NAME:-vllm_server_0_0}
VLLM_ACTOR_NAMESPACE=${VLLM_ACTOR_NAMESPACE:-}
WAIT_TIMEOUT_SEC=${STRICT_PARITY_AUTOMATION_TIMEOUT_SEC:-7200}
REPLAY_KILL_GRACE_SEC=${STRICT_PARITY_REPLAY_KILL_GRACE_SEC:-1800}
DIAGNOSTIC_TRAIN_BATCH_SIZE=${STRICT_PARITY_TRAIN_BATCH_SIZE:-}
DIAGNOSTIC_ROLLOUT_N=${STRICT_PARITY_ROLLOUT_N:-}
DIAGNOSTIC_MAX_MODEL_LEN=${STRICT_PARITY_MAX_MODEL_LEN:-}

TRAIN_MSPROBE_CONFIG=${TRAIN_MSPROBE_CONFIG:-$VERL_ROOT/tools/strict_parity_debug/msprobe_train.json}
ROLLOUT_MSPROBE_CONFIG=${ROLLOUT_MSPROBE_CONFIG:-$VERL_ROOT/tools/strict_parity_debug/msprobe_rollout.json}
TRAIN_MSPROBE_DUMP_PATH=${TRAIN_MSPROBE_DUMP_PATH:-$OUT/train/msprobe}
ROLLOUT_MSPROBE_DUMP_PATH=${ROLLOUT_MSPROBE_DUMP_PATH:-$OUT/rollout/msprobe}
READY_FILE="$OUT/READY.json"
CONTINUE_FILE="$OUT/CONTINUE"
REPLAY_FILE="$OUT/replay.pt"
MANIFEST_FILE="$OUT/manifest.json"
MEDIA_DIR="$OUT/media"
COMPARE_REPORT="$OUT/msprobe_compare.json"
REPLAY_RESULT_FILE="$OUT/replay_result.json"

log() {
    printf '\n========== [STRICT-PARITY] %s ==========\n' "$*"
}

for path in "$VERL_ROOT" "$TRAIN_SCRIPT" "$TRAIN_MSPROBE_CONFIG" "$ROLLOUT_MSPROBE_CONFIG"; do
    if [[ ! -e "$path" ]]; then
        echo "missing required path: $path" >&2
        exit 1
    fi
done

for path in \
    "$READY_FILE" "$CONTINUE_FILE" "$REPLAY_FILE" "$MANIFEST_FILE" "$MEDIA_DIR" \
    "$TRAIN_MSPROBE_DUMP_PATH" "$ROLLOUT_MSPROBE_DUMP_PATH" "$COMPARE_REPORT" "$REPLAY_RESULT_FILE"; do
    if [[ -e "$path" ]]; then
        echo "output path contains a previous run: $path" >&2
        echo "move the old output away, then rerun; stale msprobe files must not be merged." >&2
        exit 1
    fi
done
mkdir -p "$OUT"

export STRICT_PARITY_DIR="$OUT"
export STRICT_PARITY_READY_FILE="$READY_FILE"
export STRICT_PARITY_CONTINUE_FILE="$CONTINUE_FILE"
export STRICT_PARITY_REPLAY_PATH="$REPLAY_FILE"
export STRICT_PARITY_MANIFEST="$MANIFEST_FILE"
export STRICT_PARITY_INPUT_MODE=dataset
# verl starts its first training iteration at global_steps=1. The model weights
# are still the initial (step-0) weights before that iteration's actor update.
export STRICT_PARITY_GLOBAL_STEP="$TARGET_GLOBAL_STEP"
export STRICT_PARITY_CAPTURE=1
export STRICT_PARITY_CAPTURE_MEDIA=1
export STRICT_PARITY_PAUSE_AFTER_CAPTURE=1
export STRICT_PARITY_STRICT=1
export STRICT_PARITY_ROLLOUT_MSPROBE_CONFIG="$ROLLOUT_MSPROBE_CONFIG"
export STRICT_PARITY_ROLLOUT_MSPROBE_DUMP_PATH="$ROLLOUT_MSPROBE_DUMP_PATH"
unset VERL_WEIGHT_SYNC_DEBUG

TRAIN_OVERRIDES=(
    trainer.resume_mode=disable
    "trainer.total_training_steps=$TARGET_GLOBAL_STEP"
    # This run only needs actor_compute_log_prob.  Skipping actor_update makes
    # msprobe finalize the target stage and avoids spending another full
    # backward/update on a diagnostic input after both forwards are complete.
    "trainer.critic_warmup=$((TARGET_GLOBAL_STEP + 1))"
    "global_profiler.steps=[$TARGET_GLOBAL_STEP]"
    # Enabling actor.profiler alone is insufficient: without selecting the
    # backend DistProfiler silently falls back to a no-op implementation.
    global_profiler.tool=precision_debugger
    "global_profiler.save_path=$TRAIN_MSPROBE_DUMP_PATH"
    "global_profiler.global_tool_config.precision_debugger.config_path=$TRAIN_MSPROBE_CONFIG"
    'global_profiler.global_tool_config.precision_debugger.stages=[actor_compute_log_prob]'
    global_profiler.global_tool_config.precision_debugger.strict=True
    actor_rollout_ref.actor.profiler.enable=True
)
if [[ -n "$DIAGNOSTIC_TRAIN_BATCH_SIZE" ]]; then
    TRAIN_OVERRIDES+=("data.train_batch_size=$DIAGNOSTIC_TRAIN_BATCH_SIZE")
fi
if [[ -n "$DIAGNOSTIC_ROLLOUT_N" ]]; then
    TRAIN_OVERRIDES+=("actor_rollout_ref.rollout.n=$DIAGNOSTIC_ROLLOUT_N")
fi
if [[ -n "$DIAGNOSTIC_MAX_MODEL_LEN" ]]; then
    TRAIN_OVERRIDES+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$DIAGNOSTIC_MAX_MODEL_LEN")
fi

TRAIN_PID=""
TRAIN_PROCESS_GROUP=0
cleanup() {
    status=$?
    trap - EXIT INT TERM
    if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        log "aborting orchestration; releasing and terminating the diagnostic training process"
        date +%s%N > "$CONTINUE_FILE"
        if [[ "$TRAIN_PROCESS_GROUP" == 1 ]]; then
            kill -TERM -- "-$TRAIN_PID" 2>/dev/null || true
        else
            kill -TERM "$TRAIN_PID" 2>/dev/null || true
        fi
        wait "$TRAIN_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

log "starting diagnostic training; waiting for global_steps=$TARGET_GLOBAL_STEP replay capture"
if command -v setsid >/dev/null 2>&1; then
    setsid bash "$VERL_ROOT/tools/strict_parity_debug/run_capture.sh" -- \
        bash "$TRAIN_SCRIPT" "${TRAIN_OVERRIDES[@]}" &
    TRAIN_PROCESS_GROUP=1
else
    bash "$VERL_ROOT/tools/strict_parity_debug/run_capture.sh" -- \
        bash "$TRAIN_SCRIPT" "${TRAIN_OVERRIDES[@]}" &
fi
TRAIN_PID=$!

started_at=$SECONDS
next_notice=$SECONDS
while [[ ! -f "$READY_FILE" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        set +e
        wait "$TRAIN_PID"
        status=$?
        set -e
        TRAIN_PID=""
        echo "training exited before producing $READY_FILE (status=$status)" >&2
        if [[ "$status" == 0 ]]; then
            status=1
        fi
        exit "$status"
    fi
    if (( SECONDS - started_at >= WAIT_TIMEOUT_SEC )); then
        echo "timed out waiting for $READY_FILE after ${WAIT_TIMEOUT_SEC}s" >&2
        exit 1
    fi
    if (( SECONDS >= next_notice )); then
        log "training is initializing/running; READY.json has not been produced yet"
        next_notice=$((SECONDS + 30))
    fi
    sleep 1
done

if [[ -e "$ROLLOUT_MSPROBE_DUMP_PATH" ]]; then
    echo "the base training script wrote ordinary rollout data into the strict replay output: $ROLLOUT_MSPROBE_DUMP_PATH" >&2
    echo "give its startup dump_config_path a different dump_path; replay output must contain only the fixed request" >&2
    exit 1
fi

log "PAUSED: fixed training input captured; automatically replaying it on vLLM"
REPLAY_ARGS=(
    --actor-name "$VLLM_ACTOR_NAME"
    --ray-address auto
    --replay "$REPLAY_FILE"
    --sample-index 0
    --msprobe-config "$ROLLOUT_MSPROBE_CONFIG"
    --msprobe-dump-path "$ROLLOUT_MSPROBE_DUMP_PATH"
)
if [[ -n "$VLLM_ACTOR_NAMESPACE" ]]; then
    REPLAY_ARGS+=(--actor-namespace "$VLLM_ACTOR_NAMESPACE")
fi
set +e
bash "$VERL_ROOT/tools/strict_parity_debug/run_replay_vllm.sh" "${REPLAY_ARGS[@]}"
replay_status=$?
set -e

if [[ "$replay_status" == 137 ]]; then
    log "replay client received SIGKILL (status=137); waiting for the in-flight vLLM actor replay"
    replay_wait_started=$SECONDS
    next_notice=$SECONDS
    while [[ ! -f "$CONTINUE_FILE" ]]; do
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            echo "training exited while the killed replay client had an in-flight request" >&2
            exit 137
        fi
        if (( SECONDS - replay_wait_started >= REPLAY_KILL_GRACE_SEC )); then
            echo "vLLM actor did not release training within ${REPLAY_KILL_GRACE_SEC}s after replay client SIGKILL" >&2
            echo "check host/cgroup OOM logs before retrying; the diagnostic training will now be cleaned up" >&2
            exit 137
        fi
        if (( SECONDS >= next_notice )); then
            log "replay is still running inside vLLM; training remains safely paused"
            next_notice=$((SECONDS + 30))
        fi
        sleep 1
    done
    log "vLLM actor completed replay and released training despite replay client SIGKILL"
elif [[ "$replay_status" != 0 ]]; then
    echo "vLLM replay client failed (status=$replay_status); actor did not report a successful replay" >&2
    exit "$replay_status"
fi

if [[ ! -f "$REPLAY_RESULT_FILE" ]]; then
    echo "vLLM actor released training without writing replay identity: $REPLAY_RESULT_FILE" >&2
    exit 1
fi
if ! find "$ROLLOUT_MSPROBE_DUMP_PATH" -type f -name dump.json -print -quit 2>/dev/null | grep -q .; then
    echo "vLLM actor returned replay identity but no rollout dump.json was written under: $ROLLOUT_MSPROBE_DUMP_PATH" >&2
    exit 1
fi

log "vLLM replay completed and training released; waiting for the one-step run to finish"
set +e
wait "$TRAIN_PID"
training_status=$?
set -e
TRAIN_PID=""
if [[ "$training_status" != 0 ]]; then
    echo "training failed after replay (status=$training_status)" >&2
    exit "$training_status"
fi
if ! find "$TRAIN_MSPROBE_DUMP_PATH" -type f -name dump.json -print -quit 2>/dev/null | grep -q .; then
    echo "training completed but no train dump.json was written under: $TRAIN_MSPROBE_DUMP_PATH" >&2
    exit 1
fi

log "training and rollout dumps completed; comparing msprobe outputs"
set +e
bash "$VERL_ROOT/tools/strict_parity_debug/run_compare.sh" \
    --train "$TRAIN_MSPROBE_DUMP_PATH" \
    --rollout "$ROLLOUT_MSPROBE_DUMP_PATH" \
    --output "$COMPARE_REPORT" \
    --atol 1e-5 \
    --rtol 1e-3
compare_status=$?
set -e

trap - EXIT INT TERM
if [[ "$compare_status" == 0 ]]; then
    log "PASS: msprobe outputs match; report: $COMPARE_REPORT"
elif [[ "$compare_status" == 2 ]]; then
    log "DIFF FOUND: comparison completed; report: $COMPARE_REPORT"
else
    echo "msprobe comparison failed (status=$compare_status)" >&2
fi
exit "$compare_status"
