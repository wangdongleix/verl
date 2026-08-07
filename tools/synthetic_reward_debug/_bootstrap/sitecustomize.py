"""Private Python startup shim for synthetic reward child processes."""

import os

try:
    from synthetic_reward_debug.patches import install_import_hook

    install_import_hook()
except Exception:
    if os.getenv("VERL_SYNTHETIC_REWARD_DEBUG_STRICT", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise
