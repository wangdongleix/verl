"""Monkey patches for vLLM-side weight receive and model-load diagnostics."""

from __future__ import annotations

import contextvars
import logging
from functools import wraps
from typing import Any

from .config import Settings

logger = logging.getLogger("weight_sync_debug.vllm")
_PATCHED: set[str] = set()
_NO_CONTEXT = object()
_RECEIVE_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "weight_sync_debug_receive_context", default=_NO_CONTEXT
)


def _context(global_steps: Any, base_sync_done: Any, quant_prepared: Any = None) -> dict[str, Any]:
    return {
        "global_steps": global_steps,
        "base_sync_done": base_sync_done,
        "quant_prepared": quant_prepared,
    }


def _patch_load_weights(worker: Any, context: dict[str, Any], quantized: bool) -> list[tuple[Any, str, Any]]:
    try:
        models = list(worker._iter_all_models())
    except (AttributeError, TypeError):
        return []
    if quantized:
        models = models[:1]

    patches: list[tuple[Any, str, Any]] = []
    for model_index, model in enumerate(models):
        original = getattr(model, "load_weights", None)
        if not callable(original):
            continue
        instance_dict = getattr(model, "__dict__", {})
        had_instance_attribute = "load_weights" in instance_dict
        old_instance_attribute = instance_dict.get("load_weights")

        def wrapped(*args: Any, _original=original, _model=model, _index=model_index, **kwargs: Any):
            loaded_names = _original(*args, **kwargs)
            from .recorder import log_loaded_model_parameters

            log_loaded_model_parameters(
                _model,
                loaded_names,
                stage="vllm_loaded",
                context={
                    **context,
                    "model_index": _index,
                    "quantized": quantized,
                },
            )
            return loaded_names

        try:
            setattr(model, "load_weights", wrapped)
        except (AttributeError, TypeError):
            logger.debug("[weight-sync-debug] cannot patch model.load_weights", exc_info=True)
            continue
        patches.append((model, "load_weights", (had_instance_attribute, old_instance_attribute)))
    return patches


def _restore_load_weights(patches: list[tuple[Any, str, Any]]) -> None:
    for model, name, state in reversed(patches):
        had_instance_attribute, old_instance_attribute = state
        try:
            if had_instance_attribute:
                setattr(model, name, old_instance_attribute)
            else:
                delattr(model, name)
        except (AttributeError, TypeError):
            logger.debug("[weight-sync-debug] cannot restore %s", name, exc_info=True)


def _patch_update_weights(module: Any) -> bool:
    cls = getattr(module, "vLLMColocateWorkerExtension", None)
    original = getattr(cls, "_update_weights", None) if cls is not None else None
    key = "vllm_rollout.utils.vLLMColocateWorkerExtension._update_weights"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    def wrapped(
        self,
        weights: Any,
        peft_config: Any = None,
        base_sync_done: bool = False,
        quant_prepared: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        explicit_steps = kwargs.pop("global_steps", _NO_CONTEXT)
        active = _RECEIVE_CONTEXT.get()
        if active is _NO_CONTEXT:
            active = _context(None if explicit_steps is _NO_CONTEXT else explicit_steps, base_sync_done, quant_prepared)
        elif explicit_steps is not _NO_CONTEXT:
            active = {**active, "global_steps": explicit_steps}
        stage = "vllm_receive_lora" if peft_config and base_sync_done else "vllm_receive"

        from .recorder import log_received_weights

        log_received_weights(weights, stage=stage, context=active)
        try:
            quantized = bool(module.is_fp8_model(self.model_runner.vllm_config))
        except (AttributeError, TypeError, RuntimeError):
            quantized = False
        patches = _patch_load_weights(self, active, quantized)
        try:
            return original(self, weights, peft_config, base_sync_done, quant_prepared, *args, **kwargs)
        finally:
            _restore_load_weights(patches)

    cls._update_weights = wrapped
    _PATCHED.add(key)
    logger.warning("[weight-sync-debug] patched %s", key)
    return True


def _patch_update_weights_from_ipc(module: Any) -> bool:
    cls = getattr(module, "vLLMColocateWorkerExtension", None)
    original = getattr(cls, "update_weights_from_ipc", None) if cls is not None else None
    key = "vllm_rollout.utils.vLLMColocateWorkerExtension.update_weights_from_ipc"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    def wrapped(self, *args: Any, **kwargs: Any):
        global_steps = kwargs.pop("global_steps", None)
        base_sync_done = kwargs.get("base_sync_done", args[1] if len(args) > 1 else False)
        quant_prepared = kwargs.get("quant_prepared")
        token = _RECEIVE_CONTEXT.set(_context(global_steps, base_sync_done, quant_prepared))
        try:
            return original(self, *args, **kwargs)
        finally:
            _RECEIVE_CONTEXT.reset(token)

    cls.update_weights_from_ipc = wrapped
    _PATCHED.add(key)
    logger.warning("[weight-sync-debug] patched %s", key)
    return True


def install_loaded(settings: Settings, module: Any) -> bool:
    if not settings.enabled:
        return False
    installed = _patch_update_weights_from_ipc(module)
    installed = _patch_update_weights(module) or installed
    return installed
