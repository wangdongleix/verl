#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TOOLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# The actor was patched when the training command started.  The client itself
# only calls the RPC and should not import the heavy training/vLLM modules.
export STRICT_PARITY_ENABLE="0"
exec python3 -m strict_parity_debug.replay_vllm "$@"
