"""Explicit bootstrap entry point.

Usage:
    python -m strict_parity_debug.bootstrap -- bash run_train.sh
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("missing command after --")
    os.environ["STRICT_PARITY_ENABLE"] = "1"
    os.execvpe(command[0], command, os.environ)
    return 127


if __name__ == "__main__":
    sys.exit(main())
