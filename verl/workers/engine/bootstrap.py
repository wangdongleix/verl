# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0.
"""Explicit early runtime bootstrap for Megatron-based engines."""

from __future__ import annotations

import os


_ENABLED_VALUES = {"1", "true", "enabled"}


def bootstrap_megatron_runtime() -> None:
    """Register MegatronAdaptor before any Megatron engine imports."""
    enabled = os.getenv("VERL_USE_MEGATRON_ADAPTOR", "").lower() in _ENABLED_VALUES
    if not enabled:
        return

    try:
        from mindspeed_bridge.runtime import state
        from mindspeed_bridge.runtime.auto_register import register_mindspeed_adaptor
    except ImportError as exc:
        raise RuntimeError(
            "VERL_USE_MEGATRON_ADAPTOR=enabled requires MS-Bridge and MegatronAdaptor"
        ) from exc
    if not state.MEGATRON_ADAPTOR_REGISTERED_BY_RUNTIME:
        register_mindspeed_adaptor(strict=True)
