"""Monkey patches for actor-side weight export and vLLM RPC arguments."""

from __future__ import annotations

import contextvars
import inspect
import logging
from functools import wraps
from typing import Any

from .config import Settings

logger = logging.getLogger("weight_sync_debug.train")
_PATCHED: set[str] = set()
_NO_CONTEXT = object()
_RPC_GLOBAL_STEP: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "weight_sync_debug_rpc_global_step", default=_NO_CONTEXT
)


def _trace(weights: Any, stage: str, context: dict[str, Any]) -> Any:
    from .recorder import trace_weight_stream

    return trace_weight_stream(weights, stage=stage, context=context)


def _patch_server_adapter(module: Any) -> bool:
    cls = getattr(module, "ServerAdapter", None)
    update_original = getattr(cls, "update_weights", None) if cls is not None else None
    execute_original = getattr(cls, "_execute_method", None) if cls is not None else None
    if update_original is None or execute_original is None:
        return False
    update_key = "vllm_rollout.ServerAdapter.update_weights"
    execute_key = "vllm_rollout.ServerAdapter._execute_method"
    if update_key in _PATCHED or execute_key in _PATCHED:
        return False

    @wraps(execute_original)
    async def execute_wrapped(self, method: str, *args: Any, **kwargs: Any):
        step = _RPC_GLOBAL_STEP.get()
        if method == "update_weights_from_ipc" and step is not _NO_CONTEXT:
            rpc_kwargs = dict(kwargs.get("kwargs") or {})
            rpc_kwargs["global_steps"] = step
            kwargs["kwargs"] = rpc_kwargs
        return await execute_original(self, method, *args, **kwargs)

    @wraps(update_original)
    async def update_wrapped(self, weights: Any, global_steps: Any = None, **kwargs: Any):
        stage = "actor_export_base" if kwargs.get("base_sync_done") is False else "actor_export"
        context = {
            "global_steps": global_steps,
            "base_sync_done": kwargs.get("base_sync_done"),
        }
        token = _RPC_GLOBAL_STEP.set(global_steps)
        try:
            traced = _trace(weights, stage, context)
            return await update_original(self, traced, global_steps=global_steps, **kwargs)
        finally:
            _RPC_GLOBAL_STEP.reset(token)

    cls._execute_method = execute_wrapped
    cls.update_weights = update_wrapped
    _PATCHED.update({update_key, execute_key})
    logger.warning("[weight-sync-debug] patched %s and %s", update_key, execute_key)
    return True


def _patch_checkpoint_send_weights(module: Any) -> bool:
    cls = getattr(module, "ActorRolloutRefWorker", None)
    original = getattr(cls, "update_weights", None) if cls is not None else None
    key = "engine_workers.ActorRolloutRefWorker.update_weights"
    if original is None or key in _PATCHED:
        return False

    @wraps(original)
    async def wrapped(self, *args: Any, **kwargs: Any):
        engine = getattr(self, "checkpoint_engine", None)
        send_original = getattr(engine, "send_weights", None) if engine is not None else None
        if send_original is None:
            return await original(self, *args, **kwargs)

        had_instance_attribute = "send_weights" in getattr(engine, "__dict__", {})
        old_instance_attribute = getattr(engine, "__dict__", {}).get("send_weights")

        async def send_wrapped(weights: Any, *send_args: Any, **send_kwargs: Any):
            global_steps = send_kwargs.get("global_steps")
            if global_steps is None and send_args:
                global_steps = send_args[0]
            traced = _trace(weights, "actor_export", {"global_steps": global_steps})
            result = send_original(traced, *send_args, **send_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            setattr(engine, "send_weights", send_wrapped)
        except (AttributeError, TypeError):
            logger.debug("[weight-sync-debug] cannot patch checkpoint engine instance", exc_info=True)
            return await original(self, *args, **kwargs)

        try:
            return await original(self, *args, **kwargs)
        finally:
            try:
                if had_instance_attribute:
                    setattr(engine, "send_weights", old_instance_attribute)
                else:
                    delattr(engine, "send_weights")
            except (AttributeError, TypeError):
                logger.debug("[weight-sync-debug] cannot restore checkpoint engine instance", exc_info=True)

    cls.update_weights = wrapped
    _PATCHED.add(key)
    logger.warning("[weight-sync-debug] patched %s", key)
    return True


def install_loaded(settings: Settings, module_name: str, module: Any) -> bool:
    if not settings.enabled:
        return False
    if module_name == "verl.workers.rollout.vllm_rollout.vllm_rollout":
        return _patch_server_adapter(module)
    if module_name == "verl.workers.engine_workers":
        return _patch_checkpoint_send_weights(module)
    return False

