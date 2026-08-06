"""Configuration for the external weight-sync diagnostic patches."""

from __future__ import annotations

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
    strict: bool


def load_settings() -> Settings:
    return Settings(
        enabled=_bool("VERL_WEIGHT_SYNC_DEBUG"),
        strict=_bool("VERL_WEIGHT_SYNC_DEBUG_STRICT"),
    )

