#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ "${1:-}" == "--" ]]; then
    shift
fi
if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 -- <verl training command>" >&2
    exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}/_bootstrap:${TOOLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_WEIGHT_SYNC_DEBUG="1"
exec "$@"

