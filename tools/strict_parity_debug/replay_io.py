"""Replay artifact serialization and exact tensor fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


CORE_FIELDS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "response_mask",
    "loss_mask",
    "prompts",
    "responses",
    "uid",
    "multi_modal_inputs",
    "mm_processor_kwargs",
)


def _torch():
    import torch

    return torch


def to_cpu(value: Any, *, padding: int = 0) -> Any:
    """Convert tensors and nested tensors to CPU while preserving dictionaries."""
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if getattr(value, "is_nested", False):
            try:
                value = value.to_padded_tensor(padding)
            except TypeError:
                value = value.to_padded_tensor(padding=padding)
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {str(k): to_cpu(v, padding=padding) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [to_cpu(v, padding=padding) for v in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return value


def _nested_attention_mask(value: Any, padded_input_ids: Any) -> Any:
    """Build the dense attention mask lost when a nested token batch is padded."""
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not getattr(value, "is_nested", False):
        return None
    if not isinstance(padded_input_ids, torch.Tensor) or padded_input_ids.ndim != 2:
        return None

    rows = value.unbind()
    if any(row.ndim != 1 for row in rows):
        return None
    lengths = torch.tensor([row.numel() for row in rows], dtype=torch.int64)
    positions = torch.arange(padded_input_ids.shape[1], dtype=torch.int64)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def _array_bytes(value: Any) -> tuple[str, tuple[int, ...], bytes] | None:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().cpu().contiguous()
    if getattr(tensor, "is_nested", False):
        tensor = tensor.to_padded_tensor(0)
    # numpy is used only for a stable byte representation.  Older numpy builds
    # cannot expose bfloat16/float8 directly; viewing the contiguous storage as
    # uint8 keeps the fingerprint independent of the scalar dtype support.
    try:
        raw = tensor.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return str(tensor.dtype), tuple(tensor.shape), raw


def tensor_fingerprint(value: Any) -> str | None:
    item = _array_bytes(value)
    if item is None:
        return None
    dtype, shape, raw = item
    digest = hashlib.sha256()
    digest.update(dtype.encode())
    digest.update(json.dumps(shape, separators=(",", ":")).encode())
    digest.update(raw)
    return digest.hexdigest()


def select_sample(value: Any, index: int) -> Any:
    """Select one row from a batch-like value without importing TensorDict.

    TransferQueue stores variable-length fields as nested tensors.  They are
    padded by :func:`save_replay`, but callers can also use this helper with a
    plain list or a regular tensor.  A one-dimensional token vector is treated
    as a single sample rather than as a batch of scalar samples.
    """
    if value is None:
        return None
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.ndim <= 1:
            return value
        if index >= value.shape[0]:
            raise IndexError(f"sample_index={index} is outside tensor batch of size {value.shape[0]}")
        return value[index]
    if isinstance(value, Mapping):
        return {key: select_sample(item, index) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return value
        first = value[0]
        if isinstance(first, (int, float, bool, str)):
            return value
        if index >= len(value):
            raise IndexError(f"sample_index={index} is outside list batch of size {len(value)}")
        return value[index]
    return value


def _fingerprints(fields: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in fields.items():
        fingerprint = tensor_fingerprint(value)
        if fingerprint is not None:
            result[name] = fingerprint
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_replay(
    path: str | os.PathLike[str],
    fields: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    media_refs: list[str | None] | None = None,
) -> dict:
    """Atomically write a replay artifact and its JSON-compatible manifest."""
    torch = _torch()
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    cpu_fields = {name: to_cpu(value) for name, value in fields.items() if value is not None}
    if "attention_mask" not in cpu_fields and "input_ids" in cpu_fields:
        # V1 TransferQueue stores variable-length input_ids as a nested tensor
        # and does not persist an attention_mask.  Padding the nested tensor
        # without recording its row lengths silently turns padding into real
        # prompt tokens during vLLM replay.
        attention_mask = _nested_attention_mask(fields.get("input_ids"), cpu_fields["input_ids"])
        if attention_mask is not None:
            cpu_fields["attention_mask"] = attention_mask
    payload = {
        "format_version": 1,
        "fields": cpu_fields,
        "metadata": _json_safe(dict(metadata)),
        "fingerprints": _fingerprints(cpu_fields),
    }
    if media_refs is not None:
        payload["media_refs"] = list(media_refs)
    with tempfile.NamedTemporaryFile(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "format_version": 1,
        "replay_path": str(target),
        "metadata": payload["metadata"],
        "fingerprints": payload["fingerprints"],
        "media_refs": payload.get("media_refs"),
        "fields": {
            name: {
                "type": type(value).__name__,
                "shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
                "dtype": str(value.dtype) if isinstance(value, torch.Tensor) else None,
            }
            for name, value in cpu_fields.items()
        },
    }
    return manifest


def load_replay(path: str | os.PathLike[str]) -> dict:
    torch = _torch()
    load_kwargs = {"map_location": "cpu"}
    # ``weights_only`` is not available in older PyTorch versions used by some
    # Ascend images.  The artifact is produced by this tool and contains no
    # executable object, so retain compatibility with both APIs.
    try:
        payload = torch.load(Path(path).expanduser(), weights_only=False, **load_kwargs)
    except TypeError:
        payload = torch.load(Path(path).expanduser(), **load_kwargs)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported replay artifact: {path}")
    return payload


def write_manifest(path: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(dict(manifest)), ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, target)
