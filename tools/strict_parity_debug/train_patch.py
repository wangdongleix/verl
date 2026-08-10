"""Monkey patches for capturing the exact training-side replay batch."""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from functools import wraps
from typing import Any

from .config import Settings
from .media_io import has_media_payload, load_media_ref, media_ref_for_key, save_media_for_key
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


def _load_strict_rollout_fields(
    fields: dict[str, Any],
    result_root: Any,
    sample_index: int,
    batch_size: int,
) -> tuple[Any, Any] | None:
    """Build fixed-input rollout log-probs for custom replay diagnostics.

    A custom replay replaces the actor input after the ordinary rollout has
    already populated ``rollout_log_probs`` in TransferQueue.  Keeping that
    field would compare the fixed actor batch with the old random rollout
    batch (and, in practice, often with a different response width).  The
    strict vLLM RPC writes top-1 prompt log-probs for the same captured
    response suffix; use those values for the selected sample and mask the
    other diagnostic rows.
    """
    import torch

    result_path = Path(result_root) / "replay_result.json" if result_root is not None else None
    if result_path is None or not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        extra = result.get("extra_fields", {})
        processed_ids = [int(value) for value in extra.get("strict_parity_processed_prompt_ids", [])]
        prompt_logprobs = extra.get("prompt_logprobs", [])
        responses = fields.get("responses")
        response_mask = fields.get("response_mask")
        if not isinstance(responses, torch.Tensor) or not isinstance(response_mask, torch.Tensor):
            return None
        if responses.ndim == 1:
            responses = responses.unsqueeze(0)
        if response_mask.ndim == 1:
            response_mask = response_mask.unsqueeze(0)
        if not processed_ids or not prompt_logprobs:
            return None
        sample_index = int(sample_index)
        if sample_index < 0 or sample_index >= batch_size:
            raise ValueError(f"strict parity sample_index={sample_index} outside custom batch size {batch_size}")
        response_ids = responses[sample_index][response_mask[sample_index].bool()].detach().cpu().tolist()
        response_ids = [int(value) for value in response_ids]
        if not response_ids:
            return None
        start = next(
            (offset for offset in range(len(processed_ids) - len(response_ids) + 1)
             if processed_ids[offset : offset + len(response_ids)] == response_ids),
            None,
        )
        if start is None:
            raise ValueError(
                "strict parity replay processed prompt does not end with the captured response; "
                "refusing to fabricate rollout log-probs"
            )
        # verl's extract_prompt_logprobs drops the first prompt position and
        # appends one dummy element at the end.  Thus processed token i maps to
        # extracted entry i-1 (for i >= 1).
        log_values = []
        for item in prompt_logprobs:
            if isinstance(item, (list, tuple)):
                item = item[0] if item else 0.0
            log_values.append(0.0 if item is None else float(item))
        offset = max(start - 1, 0)
        if offset + len(response_ids) > len(log_values):
            raise ValueError(
                f"strict parity replay prompt log-probs are too short: offset={offset}, "
                f"response={len(response_ids)}, available={len(log_values)}"
            )
        rollout = torch.zeros(
            (batch_size, response_mask.shape[-1]), dtype=torch.float32, device=response_mask.device
        )
        values = torch.tensor(log_values[offset : offset + len(response_ids)], dtype=rollout.dtype)
        rollout[sample_index, : values.numel()] = values.to(rollout.device)
        metric_mask = response_mask.clone()
        metric_mask[torch.arange(batch_size, device=metric_mask.device) != sample_index] = 0
        return rollout, metric_mask
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        logger.exception("strict parity: failed to restore fixed-input rollout log-probs")
        raise


def _custom_tensordict(
    replay_path: Any, expected_batch_size: int, result_root: Any = None, sample_index: int = 0
) -> Any:
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

    # ``multi_modal_inputs`` is intentionally not embedded in replay.pt: the
    # capture path stores raw images in per-sample pickle sidecars so the main
    # artifact stays small and portable.  Restore those sidecars here before
    # constructing the TensorDict.  Without this step the token ids are
    # present but Kimi's ``grid_thws`` is missing, so the training-side image
    # expansion/logit mapping cannot be reconstructed.
    media_refs = payload.get("media_refs") or []
    media_values = []
    if media_refs:
        if len(media_refs) != replay_batch_size:
            raise ValueError(
                "custom replay media_refs length does not match batch size: "
                f"refs={len(media_refs)}, batch={replay_batch_size}"
            )
        for index, ref in enumerate(media_refs):
            if not ref:
                media_values.append(None)
                continue
            try:
                media_values.append(load_media_ref(ref))
            except Exception as exc:
                raise ValueError(f"failed to restore custom replay media sidecar {ref!r}") from exc

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
    strict_rollout_fields = _load_strict_rollout_fields(
        fields, result_root, sample_index=sample_index, batch_size=replay_batch_size
    )

    samples = []
    for index in range(replay_batch_size):
        sample = {}
        for name in names:
            if name == "multi_modal_inputs":
                # Prefer the processor-ready tensors captured from the V1
                # TransferQueue.  Raw media sidecars are retained for vLLM
                # replay, but are not valid training ``multi_modal_inputs``:
                # ``extract_multi_modal_inputs`` expects a dict of tensors.
                value = _sample_value(fields.get(name), index, replay_batch_size)
                if (value is None or value == {}) and media_values:
                    value = media_values[index]
            else:
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
        if strict_rollout_fields is not None:
            rollout_log_probs, metric_mask = strict_rollout_fields
            sample["rollout_log_probs"] = rollout_log_probs[index]
            # Do not let the ordinary rollout rows contaminate a custom
            # fixed-input metric.  The selected row is compared against the
            # prompt log-probs returned by the strict vLLM replay.
            sample["response_mask"] = metric_mask[index]
        samples.append(sample)
    return list_of_dict_to_tensordict(samples)


def _inject_custom_batch(batch: Any, settings: Settings) -> None:
    import transfer_queue as tq

    fields = _custom_tensordict(settings.replay_path, len(batch), settings.root, settings.sample_index)
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
    # In custom mode the replay artifact is the input, not an output.  Do not
    # overwrite it after injecting the fixed batch: doing so replaced the
    # original vLLM media references with the custom run's random sample.
    if settings.input_mode == "custom" and Path(settings.replay_path).is_file():
        payload = load_replay(settings.replay_path)
        manifest = {
            "format_version": payload.get("format_version", 1),
            "replay_path": str(settings.replay_path),
            "metadata": payload.get("metadata", {}),
            "fingerprints": payload.get("fingerprints", {}),
            "media_refs": payload.get("media_refs"),
            "fields": {name: {"type": type(value).__name__,
                              "shape": list(value.shape) if hasattr(value, "shape") else None,
                              "dtype": str(value.dtype) if hasattr(value, "dtype") else None}
                       for name, value in payload.get("fields", {}).items()},
        }
        logger.warning("[strict-parity] custom replay input is immutable: %s", settings.replay_path)
    else:
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
                captured = _capture_fields(
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
                # The first injection necessarily happens before the replay
                # RPC, because capture pauses the training worker.  Once the
                # RPC has written replay_result.json and released the pause,
                # inject the fixed-input rollout log-probs as well; otherwise
                # the old random rollout field remains in TransferQueue.
                if settings.input_mode == "custom" and captured:
                    _inject_custom_batch(batch, settings)
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
                    custom_data = _custom_tensordict(settings.replay_path, len(batch), settings.root, settings.sample_index)
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
                captured = _capture_fields(fields, settings, metadata)
                if settings.input_mode == "custom" and captured:
                    _inject_custom_batch(batch, settings)
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
