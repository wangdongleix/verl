"""Explicit bootstrap for synthetic reward injection.

Usage::

    python -m synthetic_reward_debug.bootstrap -- bash run_train.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command after --")
    os.environ["VERL_SYNTHETIC_REWARD_DEBUG"] = "1"
    os.environ["VERL_SYNTHETIC_REWARD_DEBUG_STRICT"] = "1"
    os.environ.setdefault("VERL_SYNTHETIC_REWARD_SCALE", "1.0")
    package_root = Path(__file__).resolve().parent
    tools_root = package_root.parent
    bootstrap_root = package_root / "_bootstrap"
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(
        value for value in (str(bootstrap_root), str(tools_root), pythonpath) if value
    )
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    sys.exit(main())
