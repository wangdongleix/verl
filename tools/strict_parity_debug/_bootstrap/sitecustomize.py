"""Private Python startup shim for strict-parity Ray/vLLM child processes.

The wrapper scripts put this directory first on ``PYTHONPATH``.  Keeping the
shim private to this tool prevents ``tools/`` from globally changing the
startup behavior of unrelated utilities.
"""

import os

try:
    from strict_parity_debug.patches import install_import_hook

    install_import_hook()
except Exception:
    if os.getenv("STRICT_PARITY_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise
