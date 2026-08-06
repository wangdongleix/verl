"""Save and restore raw multimodal data for strict parity replay."""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping


MEDIA_FORMAT_VERSION = 1

try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None


def _media_path(root: str | os.PathLike[str], key: str) -> Path:
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    return Path(root).expanduser() / "media" / f"{digest}.pkl"


def has_media_payload(value: Any) -> bool:
    """Return whether a nested value contains non-empty media data."""
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(has_media_payload(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_media_payload(item) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bool(value)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.numel() > 0
    if np is not None and isinstance(value, np.ndarray):
        return value.size > 0
    return True


def _copy_path(value: str | os.PathLike[str], output: Path, counter: list[int]) -> str:
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"strict parity media path does not exist: {source}")
    target = output / f"payload_{counter[0]:04d}{source.suffix or '.bin'}"
    counter[0] += 1
    if source != target:
        shutil.copy2(source, target)
    return str(target)


def _prepare(value: Any, output: Path, counter: list[int]) -> Any:
    """Make media portable while leaving its original Python representation intact."""
    if isinstance(value, str):
        return _copy_path(value, output, counter) if Path(value).expanduser().is_file() else value
    if isinstance(value, Path):
        return _copy_path(value, output, counter)
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {str(key): _prepare(item, output, counter) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prepare(item, output, counter) for item in value)
    if isinstance(value, list):
        return [_prepare(item, output, counter) for item in value]
    return value


def save_media_for_key(root: str | os.PathLike[str], key: str, data: Any) -> str:
    """Atomically save one sample's raw media and return its sidecar path."""
    target = _media_path(root, key)
    target.parent.mkdir(parents=True, exist_ok=True)
    prepared = _prepare(data, target.parent, [0])
    payload = {"format_version": MEDIA_FORMAT_VERSION, "key": str(key), "data": prepared}
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return str(target)


def media_ref_for_key(root: str | os.PathLike[str], key: str) -> str | None:
    path = _media_path(root, key)
    return str(path) if path.is_file() else None


def load_media_ref(ref: str | os.PathLike[str]) -> Any:
    path = Path(ref).expanduser()
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("format_version") != MEDIA_FORMAT_VERSION:
        raise ValueError(f"unsupported strict parity media sidecar: {path}")
    return payload["data"]


def media_ref_fingerprint(ref: str | os.PathLike[str]) -> str:
    """Hash the sidecar and copied file payloads it references."""
    path = Path(ref).expanduser()
    digest = hashlib.sha256(path.read_bytes())
    media_root = path.parent.resolve()
    for value in _walk(load_media_ref(path)):
        if isinstance(value, str):
            candidate = Path(value).expanduser()
            try:
                candidate.relative_to(media_root)
            except ValueError:
                continue
            if candidate.is_file():
                digest.update(str(candidate).encode("utf-8"))
                digest.update(candidate.read_bytes())
    return digest.hexdigest()


def _walk(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)
    else:
        yield value
