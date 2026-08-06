#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${TOOLS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export STRICT_PARITY_ENABLE="0"
exec python3 -m strict_parity_debug.compare_msprobe "$@"
