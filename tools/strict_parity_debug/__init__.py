"""External monkey-patch tooling for strict verl train/rollout parity checks.

This package is intentionally outside the ``verl`` Python package.  It is loaded
through ``PYTHONPATH`` and changes methods in memory only; it does not modify
verl, vLLM, or vLLM-Ascend source files.
"""

from .patches import install_import_hook as install

__all__ = ["install"]
