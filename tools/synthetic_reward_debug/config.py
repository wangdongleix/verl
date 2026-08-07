"""Environment configuration for synthetic reward injection."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    enabled: bool
    scale: float
    strict: bool


def load_settings() -> Settings:
    raw_scale = os.getenv("VERL_SYNTHETIC_REWARD_SCALE", "1.0")
    try:
        scale = float(raw_scale)
    except ValueError as exc:
        raise ValueError(f"VERL_SYNTHETIC_REWARD_SCALE must be a float, got {raw_scale!r}") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"VERL_SYNTHETIC_REWARD_SCALE must be finite and positive, got {scale}")
    return Settings(
        enabled=_bool("VERL_SYNTHETIC_REWARD_DEBUG"),
        scale=scale,
        strict=_bool("VERL_SYNTHETIC_REWARD_DEBUG_STRICT"),
    )
