"""Monkey patches for capturing the exact training-side replay batch."""

from __future__ import annotations

import logging
import json
import time
from functools import wraps
from typing import Any

from .config import Settings
from .media_io import has_media_payload, media_ref_for_key, save_media_for_key
from .replay_io import CORE_FIELDS, load_replay, save_replay, write_manifest

logger = logging.getLogger("strict_parity_debug.train")
_PATCHED: set[str] = set()
_CAPTURED = False


def _step_matches(settings: Settings, step: Any) -> bool:
    if settings.target_global_step is None:
        return True
    if step is None:
        return False
    try:
        return int(step) == settings.target_global_step
    except (TypeError, ValueError):
        return False


def _value_from_tensordict(data: Any, name: str) -> Any:
    try:
        return data.get(name, None)
    except Exception:
        try:
            return data[name]
        except Exception:
            return None


def _unwrap_ray_actor_class(value: Any) -> Any:
    """Get the Python class behind a ``@ray.remote`` ActorClass wrapper."""
    modified = getattr(value, "_modified_class", None)
    if modified is not None:
        return modified
    metadata = getattr(value, "__ray_metadata__", None)
    modified = getattr(metadata, "modified_class", None)
    return modified if modified is not None else value


def _sample_value(value: Any, index: int, batch_size: int) -> Any:
    """Extract one custom replay sample while preserving nested dictionaries."""
    import torch

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim <= 1:
            return value
        return value[index]
    if isinstance(value, dict):
        return {key: _sample_value(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size and value and not isinstance(value[0], (int, float, bool, str)):
            return value[index]
        return value
    return value


def _custom_tensordict(replay_path: Any, expected_batch_size: int) -> Any:
    """Build the same TensorDict shape that agent_loop_tq writes to TQ."""
    import torch
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    payload = load_replay(replay_path)
    fields = payload.get("fields", {})
    input_ids = fields.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("custom replay input_ids must be a torch.Tensor")
    replay_batch_size = 1 if input_ids.ndim <= 1 else input_ids.shape[0]
    if replay_batch_size != expected_batch_size:
        raise ValueError(
            "custom replay batch size does not match the current verl batch: "
            f"replay={replay_batch_size}, verl={expected_batch_size}. "
            "Set data.train_batch_size/rollout.n to the custom batch size."
        )

    names = (
        "input_ids",
        "attention_mask",
        "position_ids",
        "response_mask",
        "loss_mask",
        "prompts",
        "responses",
        "multi_modal_inputs",
        "mm_processor_kwargs",
    )
    samples = []
    for index in range(replay_batch_size):
        sample = {}
        for name in names:
            value = _sample_value(fields.get(name), index, replay_batch_size)
            if value is None:
                continue
            if name in {"prompts", "responses"} and not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.int64)
            sample[name] = value
        missing = {"input_ids", "position_ids", "response_mask"} - sample.keys()
        if missing:
            raise ValueError(f"custom replay is missing required fields: {sorted(missing)}")
        if "loss_mask" not in sample:
            sample["loss_mask"] = sample["response_mask"]
        samples.append(sample)
    return list_of_dict_to_tensordict(samples)


def _inject_custom_batch(batch: Any, settings: Settings) -> None:
    import transfer_queue as tq

    fields = _custom_tensordict(settings.replay_path, len(batch))
    tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=fields)
    logger.warning(
        "[strict-parity] replaced dataset batch with custom replay: path=%s batch_size=%d",
        settings.replay_path,
        len(batch),
    )


def _capture_fields(fields: Any, settings: Settings, metadata: dict[str, Any]) -> bool:
    global _CAPTURED
    if _CAPTURED or not settings.capture:
        return False
    normalized = {name: _value_from_tensordict(fields, name) for name in CORE_FIELDS}
    if normalized.get("input_ids") is None:
        logger.warning("strict parity: training batch has no input_ids; replay was not written")
        return False
    media_keys = metadata.get("media_keys", metadata.get("keys", []))
    media_refs = None
    if media_keys:
        media_refs = [media_ref_for_key(settings.root, str(key)) for key in media_keys]
    manifest = save_replay(settings.replay_path, normalized, metadata, media_refs=media_refs)
    write_manifest(settings.manifest_path, manifest)
    _CAPTURED = True
    settings.ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_time_ns = time.time_ns()
    settings.ready_path.write_text(
        json.dumps(
            {
                "ready_time_ns": ready_time_ns,
                "replay_path": str(settings.replay_path),
                "manifest_path": str(settings.manifest_path),
                "metadata": manifest["metadata"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.warning(
        "[strict-parity] captured training replay: path=%s input_ids=%s",
        settings.replay_path,
        manifest["fingerprints"].get("input_ids"),
    )
    if settings.pause_after_capture:
        _wait_for_release(settings, ready_time_ns)
    return True


def _wait_for_release(settings: Settings, ready_time_ns: int) -> None:
    logger.warning(
        "[STRICT-PARITY][PAUSED] replay input is ready. Training is intentionally paused; "
        "waiting for vLLM replay and release marker %s",
        settings.continue_path,
    )
    deadline = time.monotonic() + settings.wait_timeout_sec if settings.wait_timeout_sec > 0 else None
    next_reminder = time.monotonic() + 30
    while True:
        if settings.continue_path.exists():
            try:
                marker = settings.continue_path.read_text(encoding="utf-8").strip()
            except OSError:
                marker = ""
            marker_time = 0
            try:
                marker_time = int(marker.splitlines()[0]) if marker else 0
            except ValueError:
                marker_time = 0
            if marker_time >= ready_time_ns or settings.continue_path.stat().st_mtime_ns >= ready_time_ns:
                logger.warning("[STRICT-PARITY][RESUMED] vLLM replay completed; continuing training forward")
                return
        now = time.monotonic()
        if now >= next_reminder:
            logger.warning(
                "[STRICT-PARITY][PAUSED] still waiting for replay release: %s",
                settings.continue_path,
            )
            next_reminder = now + 30
        if deadline is not None and now >= deadline:
            message = f"strict parity release timeout: {settings.continue_path}"
            if settings.strict:
                raise TimeoutError(message)
            logger.warning(message)
            return
        time.sleep(0.5)


def _patch_v1(settings: Settings, trainer_base: Any = None) -> bool:
    if trainer_base is None:
        try:
            from verl.trainer.ppo.v1 import trainer_base
        except Exception as exc:
            logger.debug("strict parity: V1 trainer unavailable: %s", exc)
            return False
    cls = getattr(trainer_base, "PPOTrainer", None)
    original = getattr(cls, "_compute_old_log_prob", None) if cls is not None else None
    key = "v1.PPOTrainer._compute_old_log_prob"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    def wrapped(self, batch, metrics, *args, **kwargs):
        if _step_matches(settings, getattr(self, "global_steps", None)):
            try:
                import transfer_queue as tq

                if settings.input_mode == "custom":
                    _inject_custom_batch(batch, settings)

                fields = list(CORE_FIELDS)
                try:
                    data = tq.kv_batch_get(
                        keys=batch.keys,
                        partition_id=batch.partition_id,
                        select_fields=fields,
                    )
                except Exception:
                    # Older TransferQueue versions reject absent optional fields.
                    data = tq.kv_batch_get(
                        keys=batch.keys,
                        partition_id=batch.partition_id,
                        select_fields=["input_ids", "attention_mask", "position_ids", "response_mask", "prompts", "responses"],
                    )
                _capture_fields(
                    data,
                    settings,
                    {
                        "backend": "verl_v1",
                        "stage": "actor_compute_log_prob_input",
                        "global_steps": getattr(self, "global_steps", None),
                        "partition_id": getattr(batch, "partition_id", None),
                        "keys": list(getattr(batch, "keys", [])),
                    },
                )
            except Exception:
                logger.exception("strict parity: failed to capture V1 TransferQueue batch")
                if settings.strict:
                    raise
        return original(self, batch, metrics, *args, **kwargs)

    cls._compute_old_log_prob = wrapped
    _PATCHED.add(key)
    logger.warning("[strict-parity] patched %s", key)
    return True


def _patch_v1_sync_sleep(settings: Settings, trainer_sync: Any) -> bool:
    """Keep vLLM awake until the target fixed-input replay has finished."""
    cls = getattr(trainer_sync, "PPOTrainerSync", None)
    original = getattr(cls, "on_sample_end", None) if cls is not None else None
    key = "v1.PPOTrainerSync.on_sample_end"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        if (
            settings.capture
            and settings.pause_after_capture
            and not _CAPTURED
            and _step_matches(settings, getattr(self, "global_steps", None))
        ):
            logger.warning("[strict-parity] deferring rollout sleep until fixed-input replay completes")
            return None
        return original(self, *args, **kwargs)

    cls.on_sample_end = wrapped
    _PATCHED.add(key)
    logger.warning("[strict-parity] patched %s", key)
    return True


def _patch_v1_media_capture(settings: Settings, agent_loop_tq: Any = None) -> bool:
    """Capture raw media before AgentLoopWorkerTQ removes it from the field."""
    if not settings.capture_media or not settings.capture:
        return False
    if agent_loop_tq is None:
        try:
            from verl.trainer.ppo.v1 import agent_loop_tq
        except Exception as exc:
            logger.debug("strict parity: V1 agent-loop module unavailable: %s", exc)
            return False
    cls = _unwrap_ray_actor_class(getattr(agent_loop_tq, "AgentLoopWorkerTQ", None))
    original = getattr(cls, "_agent_loop_postprocess", None) if cls is not None else None
    key = "v1.AgentLoopWorkerTQ._agent_loop_postprocess.media"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    async def wrapped(self, output, validate, **kwargs):
        if _step_matches(settings, kwargs.get("global_steps")):
            outputs = output if isinstance(output, list) else [output]
            uid, session_id = kwargs.get("uid"), kwargs.get("session_id")
            for index, item in enumerate(outputs):
                media = getattr(item, "multi_modal_data", None)
                if has_media_payload(media):
                    if uid is None or session_id is None:
                        raise ValueError("strict parity cannot correlate V1 multimodal output without uid/session_id")
                    sample_key = f"{uid}_{session_id}_{index}"
                    save_media_for_key(settings.root, sample_key, media)
        return await original(self, output, validate, **kwargs)

    cls._agent_loop_postprocess = wrapped
    _PATCHED.add(key)
    logger.warning("[strict-parity] patched %s", key)
    return True


def _legacy_media_keys(batch: Any, settings: Settings, step: Any) -> list[str] | None:
    """Save raw media when a legacy DataProto still carries it."""
    non_tensor = getattr(batch, "non_tensor_batch", {}) or {}
    raw = non_tensor.get("multi_modal_data")
    if raw is None:
        return None
    batch_size = len(batch)
    if isinstance(raw, (list, tuple)):
        samples = list(raw)
    else:
        try:
            converted = raw.tolist()
        except AttributeError:
            converted = raw
        samples = converted if isinstance(converted, (list, tuple)) else [converted]
    if len(samples) == 1 and batch_size > 1:
        samples *= batch_size
    if len(samples) != batch_size:
        raise ValueError(f"legacy multi_modal_data batch size {len(samples)} != DataProto size {batch_size}")
    uids = non_tensor.get("uid")
    if uids is None:
        uids = [f"legacy_{step}_{index}" for index in range(batch_size)]
    else:
        try:
            uids = uids.tolist()
        except AttributeError:
            uids = list(uids) if isinstance(uids, (list, tuple)) else [uids]
        if not isinstance(uids, (list, tuple)):
            uids = [uids]
    keys = []
    for index, media in enumerate(samples):
        key = f"legacy_{step}_{uids[index] if index < len(uids) else index}_{index}"
        keys.append(key)
        if has_media_payload(media):
            save_media_for_key(settings.root, key, media)
    return keys


def _patch_legacy(settings: Settings, ray_trainer: Any = None) -> bool:
    if ray_trainer is None:
        try:
            from verl.trainer.ppo import ray_trainer
        except Exception as exc:
            logger.debug("strict parity: legacy trainer unavailable: %s", exc)
            return False
    cls = getattr(ray_trainer, "RayPPOTrainer", None)
    original = getattr(cls, "_compute_old_log_prob", None) if cls is not None else None
    key = "legacy.RayPPOTrainer._compute_old_log_prob"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    def wrapped(self, batch, *args, **kwargs):
        step = getattr(self, "global_steps", None)
        if _step_matches(settings, step):
            try:
                if settings.input_mode == "custom":
                    custom_data = _custom_tensordict(settings.replay_path, len(batch))
                    batch.batch.update(custom_data)
                    logger.warning(
                        "[strict-parity] replaced legacy dataset batch with custom replay: path=%s batch_size=%d",
                        settings.replay_path,
                        len(batch),
                    )
                fields = {name: _value_from_tensordict(batch.batch, name) for name in CORE_FIELDS}
                metadata = {
                    "backend": "verl_legacy",
                    "stage": "actor_compute_log_prob_input",
                    "global_steps": step,
                }
                media_keys = (
                    _legacy_media_keys(batch, settings, step)
                    if settings.capture and settings.capture_media
                    else None
                )
                if media_keys is not None:
                    metadata["media_keys"] = media_keys
                _capture_fields(fields, settings, metadata)
            except Exception:
                logger.exception("strict parity: failed to capture legacy DataProto batch")
                if settings.strict:
                    raise
        return original(self, batch, *args, **kwargs)

    cls._compute_old_log_prob = wrapped
    _PATCHED.add(key)
    logger.warning("[strict-parity] patched %s", key)
    return True


def install_loaded(settings: Settings, module_name: str, module: Any) -> bool:
    """Patch one target module without importing any other verl modules."""
    if not settings.enabled:
        return False
    if module_name == "verl.trainer.ppo.v1.agent_loop_tq":
        return _patch_v1_media_capture(settings, module)
    if module_name == "verl.trainer.ppo.v1.trainer_base":
        return _patch_v1(settings, module)
    if module_name == "verl.trainer.ppo.v1.trainer_sync":
        return _patch_v1_sync_sleep(settings, module)
    if module_name == "verl.trainer.ppo.ray_trainer":
        return _patch_legacy(settings, module)
    return False
