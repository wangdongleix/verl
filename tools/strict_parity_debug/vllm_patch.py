"""Monkey patch for the vLLM server replay RPC."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from .config import Settings
from .media_io import has_media_payload, load_media_ref, media_ref_fingerprint
from .replay_io import load_replay, select_sample, tensor_fingerprint

logger = logging.getLogger("strict_parity_debug.vllm")
_PATCHED = False
_WORKER_PATCHED = False
_ENABLE_MSPROBE_METHOD = "strict_parity_enable_replay_msprobe"
_DISABLE_MSPROBE_METHOD = "strict_parity_disable_replay_msprobe"
_REPLAY_REQUEST_ID = "strict-parity-replay"


def _write_replay_result(settings: Settings, result: dict[str, Any]) -> None:
    """Persist actor-side replay identity before releasing paused training."""
    path = settings.root.expanduser() / "replay_result.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _release_paused_training(settings: Settings) -> None:
    """Release capture only after the actor-side replay completed successfully."""
    if not settings.pause_after_capture:
        return
    marker_path = settings.continue_path.expanduser()
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(f"{time.time_ns()}\n", encoding="utf-8")
    logger.warning("[strict-parity] replay completed; released training via %s", marker_path)


def _replay_model_runner(worker: Any) -> Any:
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("strict parity: vLLM worker has no model_runner")
    return runner


def _enable_replay_msprobe(worker: Any, config_path: str, dump_path: str) -> None:
    """Install a fresh msprobe debugger only for the replay request."""
    runner = _replay_model_runner(worker)
    if getattr(runner, "_debugger_started", False):
        runner._finalize_dump_data()
    from msprobe.pytorch import PrecisionDebugger

    runner.debugger = PrecisionDebugger(config_path=config_path, dump_path=dump_path)
    runner._debugger_started = False


def _disable_replay_msprobe(worker: Any) -> None:
    runner = _replay_model_runner(worker)
    if getattr(runner, "_debugger_started", False):
        runner._finalize_dump_data()
    runner.debugger = None
    runner._debugger_started = False


def _logical_sequence(input_ids: Any, attention_mask: Any = None) -> list[int]:
    values = input_ids
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    if attention_mask is not None:
        mask = attention_mask
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu()
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if mask and isinstance(mask[0], list):
            mask = mask[0]
        values = [int(token) for token, keep in zip(values, mask) if int(keep)]
    return [int(token) for token in values]


def _selected_media_ref(payload: dict, sample_index: int) -> str | None:
    refs = payload.get("media_refs")
    if refs is None:
        return None
    if not isinstance(refs, list) or sample_index >= len(refs):
        raise ValueError(
            "strict parity replay media_refs does not match the captured batch: "
            f"sample_index={sample_index}, refs={len(refs) if isinstance(refs, list) else type(refs).__name__}"
        )
    return refs[sample_index]


def _media_value(data: Any, *keys: str) -> Any:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if has_media_payload(value):
            return value
    return None


def _prepare_replay_prompt_ids(
    server: Any,
    fields: dict[str, Any],
    sample_index: int,
    captured_ids: list[int],
    image_data: Any,
) -> tuple[list[int], int, int, bool]:
    """Adapt only the actor prompt and preserve sampled response IDs exactly.

    Kimi K3's vLLM adapter removes the image-size annotation through a
    decode/regex/encode pass. Applying that pass to a strict replay's full
    prompt+response sequence is invalid: arbitrary sampled token IDs are not
    guaranteed to survive tokenizer decode/encode round trips.
    """
    adapter = getattr(server, "_vllm_adapter", None)
    if adapter is None:
        return captured_ids, len(captured_ids), 0, False

    responses = select_sample(fields.get("responses"), sample_index)
    response_mask = select_sample(fields.get("response_mask"), sample_index)
    response_ids = _logical_sequence(responses, response_mask) if responses is not None else []
    if response_ids:
        if len(response_ids) > len(captured_ids) or captured_ids[-len(response_ids) :] != response_ids:
            raise ValueError(
                "strict parity replay responses are not the exact suffix of attention-masked input_ids; "
                "cannot isolate the prompt adapter from sampled response IDs"
            )
        actor_prompt_ids = captured_ids[: -len(response_ids)]
    else:
        actor_prompt_ids = captured_ids

    prepared_actor_prompt_ids = adapter.prepare_vllm_prompt_ids(
        actor_prompt_ids,
        server.model_config.tokenizer,
        image_data,
    )
    return prepared_actor_prompt_ids + response_ids, len(actor_prompt_ids), len(response_ids), True


class _PreparedPromptAdapter:
    """Delegate an adapter while preserving prompt IDs supplied by replay."""

    def __init__(self, adapter: Any):
        self._adapter = adapter

    def prepare_vllm_prompt_ids(self, prompt_ids: Any, tokenizer: Any, image_data: Any) -> Any:
        del tokenizer, image_data
        return prompt_ids

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)


def _install_prepared_prompt_generate_patch(cls: Any, original: Any, vllm_async_server: Any) -> Any:
    """Wrap ``generate`` without changing verl's tracked server source."""
    normalize_token_ids = vllm_async_server.normalize_token_ids
    dedup_image_tokens = vllm_async_server.qwen2_5_vl_dedup_image_tokens

    async def _strict_parity_generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        priority: int = 0,
        kv_transfer_params: dict | None = None,
        prompt_ids_are_prepared: bool = False,
    ) -> Any:
        if not prompt_ids_are_prepared:
            return await original(
                self,
                prompt_ids,
                sampling_params,
                request_id,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
                kv_transfer_params=kv_transfer_params,
            )

        submitted_prompt_ids = list(normalize_token_ids(prompt_ids))
        processed_prompt_ids = list(dedup_image_tokens(submitted_prompt_ids, self.model_config.processor))
        original_adapter = getattr(self, "_vllm_adapter", None)
        original_engine = getattr(self, "engine", None)
        original_engine_generate = getattr(original_engine, "generate", None)
        captured_prompt_ids: dict[str, list[int]] = {}

        if original_adapter is not None:
            self._vllm_adapter = _PreparedPromptAdapter(original_adapter)

        if original_engine is not None and original_engine_generate is not None:

            async def _capture_engine_generate(*args: Any, **kwargs: Any):
                async for result in original_engine_generate(*args, **kwargs):
                    engine_prompt_ids = getattr(result, "prompt_token_ids", None)
                    if engine_prompt_ids is not None:
                        captured_prompt_ids["processed"] = list(normalize_token_ids(engine_prompt_ids))
                    yield result

            original_engine.generate = _capture_engine_generate

        try:
            output = await original(
                self,
                prompt_ids,
                sampling_params,
                request_id,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
                kv_transfer_params=kv_transfer_params,
            )
        finally:
            if original_adapter is not None:
                self._vllm_adapter = original_adapter
            if original_engine is not None and original_engine_generate is not None:
                original_engine.generate = original_engine_generate

        extra_fields = getattr(output, "extra_fields", None)
        if isinstance(extra_fields, dict):
            extra_fields["strict_parity_submitted_prompt_ids"] = submitted_prompt_ids
            extra_fields["strict_parity_processed_prompt_ids"] = captured_prompt_ids.get(
                "processed", processed_prompt_ids
            )
        return output

    _strict_parity_generate.__name__ = getattr(original, "__name__", "generate")
    _strict_parity_generate.__qualname__ = getattr(original, "__qualname__", "generate")
    return _strict_parity_generate


async def _strict_replay(self) -> dict:
    settings = self._strict_parity_settings
    msprobe_config_path = str(settings.rollout_msprobe_config) if settings.rollout_msprobe_config else None
    msprobe_dump_path = str(settings.rollout_msprobe_dump_path) if settings.rollout_msprobe_dump_path else None
    if bool(msprobe_config_path) != bool(msprobe_dump_path):
        raise ValueError(
            "strict parity replay requires both STRICT_PARITY_ROLLOUT_MSPROBE_CONFIG "
            "and STRICT_PARITY_ROLLOUT_MSPROBE_DUMP_PATH"
        )
    resolved_replay_path = settings.replay_path.expanduser().resolve()
    payload = load_replay(resolved_replay_path)
    fields = payload["fields"]
    selected_index = settings.sample_index
    input_ids = select_sample(fields.get("input_ids"), selected_index)
    attention_mask = select_sample(fields.get("attention_mask"), selected_index)
    if input_ids is None:
        raise ValueError("strict parity replay requires input_ids")
    prompt_ids = _logical_sequence(input_ids, attention_mask)
    if not prompt_ids:
        raise ValueError("strict parity replay input_ids is empty after applying attention_mask")

    # The TQ batch intentionally keeps processed multimodal tensors for the
    # actor, but vLLM's public generate API requires the original media object
    # (image/video/audio).  Do not silently run a text-only forward with media
    # placeholder IDs: that would invalidate the parity experiment.
    multimodal_inputs = select_sample(fields.get("multi_modal_inputs"), selected_index)
    media_ref = _selected_media_ref(payload, selected_index)
    if has_media_payload(multimodal_inputs) and not media_ref:
        raise ValueError(
            "strict parity replay found processed multimodal inputs but no raw media sidecar. "
            "Recapture the target step with STRICT_PARITY_CAPTURE_MEDIA=1."
        )
    raw_media = load_media_ref(media_ref) if media_ref else {}
    image_data = _media_value(raw_media, "images", "image")
    video_data = _media_value(raw_media, "videos", "video")
    audio_data = _media_value(raw_media, "audios", "audio")
    mm_processor_kwargs = select_sample(fields.get("mm_processor_kwargs"), selected_index)
    engine_prompt_ids, actor_prompt_length, response_length, prompt_ids_are_prepared = _prepare_replay_prompt_ids(
        self,
        fields,
        selected_index,
        prompt_ids,
        image_data,
    )

    # This deliberately uses the original server generate path.  vLLM's
    # existing vLLM-Ascend ModelRunner msprobe hooks therefore observe the
    # replay forward without any model/backend source change.
    sampling_params = {
        "max_tokens": 1,
        # verl's extractor treats 0 as the target token's log-prob; a
        # positive value requests top-k candidates, which is not the value
        # used by the training-side actor metric.
        "prompt_logprobs": 0,
        "temperature": 0.0,
        "detokenize": False,
        "ignore_eos": True,
    }
    replay_generate = getattr(type(self), "_strict_parity_generate", type(self)._strict_parity_original_generate)
    manage_sleep = bool(getattr(self.config, "free_cache_engine", False))
    try:
        # The same image was already used by the ordinary rollout that produced
        # the captured response.  Clear prefix, multimodal-processor, and encoder
        # caches before enabling msprobe; otherwise strict replay can skip the
        # vision path and the resulting "all-layer" profile starts only at the
        # language model with cached image embeddings.
        logger.warning("[strict-parity] clearing vLLM prefix and multimodal caches before replay")
        await self.clear_kv_cache()
        if msprobe_config_path:
            logger.warning("[strict-parity] enabling replay msprobe on vLLM workers")
            await self.collective_rpc(_ENABLE_MSPROBE_METHOD, args=(msprobe_config_path, msprobe_dump_path))
        try:
            logger.warning("[strict-parity] running fixed-input vLLM prefill")
            output = await replay_generate(
                self,
                prompt_ids=engine_prompt_ids,
                sampling_params=sampling_params,
                request_id=_REPLAY_REQUEST_ID,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=0,
                prompt_ids_are_prepared=prompt_ids_are_prepared,
            )
        finally:
            if msprobe_config_path:
                logger.warning("[strict-parity] finalizing replay msprobe on vLLM workers")
                await self.collective_rpc(_DISABLE_MSPROBE_METHOD)
    finally:
        if manage_sleep:
            # The normal NPU rollout path deliberately uses sleep(level=1),
            # because vLLM-Ascend historically did not support level-2 sleep.
            # That leaves the model weights resident on every colocated NPU,
            # however, and the following FSDP actor forward can OOM before its
            # profile is written.  Strict replay is a terminal diagnostic
            # request for this iteration, so release weights as well.  Keep a
            # compatibility fallback for engines that reject the level kwarg.
            logger.warning("[strict-parity] releasing vLLM weights after replay")
            try:
                await self.engine.sleep(level=2)
            except (AttributeError, TypeError, RuntimeError) as exc:
                logger.warning("[strict-parity] level-2 sleep unavailable (%s); falling back to normal sleep", exc)
                await self.sleep()
    result = {
        "request_id": _REPLAY_REQUEST_ID,
        "replay_path": str(resolved_replay_path),
        "global_steps": getattr(self, "global_steps", None),
        "input_ids_fingerprint": tensor_fingerprint(input_ids),
        "batch_input_ids_fingerprint": payload.get("fingerprints", {}).get("input_ids"),
        "sample_index": selected_index,
        "prompt_length": len(prompt_ids),
        "captured_sequence_length": len(prompt_ids),
        "actor_prompt_length": actor_prompt_length,
        "response_length": response_length,
        "submitted_sequence_length": len(engine_prompt_ids),
        "media_ref": media_ref,
        "media_fingerprint": media_ref_fingerprint(media_ref) if media_ref else None,
        "msprobe_config_path": msprobe_config_path,
        "msprobe_dump_path": msprobe_dump_path,
        "token_ids": list(getattr(output, "token_ids", []) or []),
        "extra_fields": dict(getattr(output, "extra_fields", {}) or {}),
    }
    _write_replay_result(settings, result)
    _release_paused_training(settings)
    return result


def install_worker_extension(settings: Settings, rollout_utils: Any) -> bool:
    """Add named worker methods so vLLM RPC uses secure string serialization."""
    global _WORKER_PATCHED
    if not settings.enabled or _WORKER_PATCHED:
        return False
    cls = getattr(rollout_utils, "vLLMColocateWorkerExtension", None)
    if cls is None:
        return False
    setattr(cls, _ENABLE_MSPROBE_METHOD, _enable_replay_msprobe)
    setattr(cls, _DISABLE_MSPROBE_METHOD, _disable_replay_msprobe)
    _WORKER_PATCHED = True
    logger.warning("[strict-parity] patched replay msprobe methods onto %s", cls.__name__)
    return True


def install(settings: Settings, vllm_async_server: Any = None) -> bool:
    global _PATCHED
    if not settings.enabled or _PATCHED:
        return False
    if vllm_async_server is None:
        try:
            from verl.workers.rollout.vllm_rollout import vllm_async_server
        except Exception as exc:
            logger.debug("strict parity: vLLM server module unavailable: %s", exc)
            return False
    cls = getattr(vllm_async_server, "vLLMHttpServer", None)
    original = getattr(cls, "generate", None) if cls is not None else None
    if cls is None or original is None:
        return False

    patched_generate = _install_prepared_prompt_generate_patch(cls, original, vllm_async_server)
    cls._strict_parity_original_generate = original
    cls._strict_parity_generate = patched_generate
    cls.generate = patched_generate
    cls.strict_parity_replay = _strict_replay
    cls._strict_parity_settings = settings
    _PATCHED = True
    logger.warning("[strict-parity] added replay RPC to vLLM server %s", cls.__name__)
    return True
