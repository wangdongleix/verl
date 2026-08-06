"""Private startup shim for weight-sync diagnostic child processes."""

import os

try:
    from weight_sync_debug.patches import install_import_hook

    install_import_hook()
except Exception:
    if os.getenv("VERL_WEIGHT_SYNC_DEBUG_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise

