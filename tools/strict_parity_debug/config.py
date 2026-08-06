"""Environment configuration for the external parity patches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float(name: str, default: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    enabled: bool
    root: Path
    input_mode: str
    capture: bool
    capture_media: bool
    pause_after_capture: bool
    ready_path: Path
    continue_path: Path
    wait_timeout_sec: float
    target_global_step: int | None
    sample_index: int
    replay_path: Path
    manifest_path: Path
    rollout_msprobe_config: Path | None
    rollout_msprobe_dump_path: Path | None
    strict: bool


def load_settings() -> Settings:
    root = Path(os.getenv("STRICT_PARITY_DIR", "strict_parity_output")).expanduser().resolve()
    rollout_msprobe_config = os.getenv("STRICT_PARITY_ROLLOUT_MSPROBE_CONFIG", "").strip()
    rollout_msprobe_dump_path = os.getenv("STRICT_PARITY_ROLLOUT_MSPROBE_DUMP_PATH", "").strip()
    input_mode = os.getenv("STRICT_PARITY_INPUT_MODE", "dataset").strip().lower()
    if input_mode not in {"dataset", "custom"}:
        raise ValueError("STRICT_PARITY_INPUT_MODE must be 'dataset' or 'custom'")
    return Settings(
        enabled=_bool("STRICT_PARITY_ENABLE"),
        root=root,
        input_mode=input_mode,
        capture=_bool("STRICT_PARITY_CAPTURE", True),
        capture_media=_bool("STRICT_PARITY_CAPTURE_MEDIA", True),
        pause_after_capture=_bool("STRICT_PARITY_PAUSE_AFTER_CAPTURE"),
        ready_path=Path(os.getenv("STRICT_PARITY_READY_FILE", root / "READY.json")).expanduser(),
        continue_path=Path(os.getenv("STRICT_PARITY_CONTINUE_FILE", root / "CONTINUE")).expanduser(),
        wait_timeout_sec=_float("STRICT_PARITY_WAIT_TIMEOUT_SEC"),
        target_global_step=_int("STRICT_PARITY_GLOBAL_STEP"),
        sample_index=_int("STRICT_PARITY_SAMPLE_INDEX", 0) or 0,
        replay_path=Path(os.getenv("STRICT_PARITY_REPLAY_PATH", root / "replay.pt")).expanduser(),
        manifest_path=Path(os.getenv("STRICT_PARITY_MANIFEST", root / "manifest.json")).expanduser(),
        rollout_msprobe_config=(Path(rollout_msprobe_config).expanduser().resolve() if rollout_msprobe_config else None),
        rollout_msprobe_dump_path=(
            Path(rollout_msprobe_dump_path).expanduser().resolve() if rollout_msprobe_dump_path else None
        ),
        strict=_bool("STRICT_PARITY_STRICT"),
    )
