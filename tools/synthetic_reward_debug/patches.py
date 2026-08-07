"""Deferred import hook for the synthetic reward V1 patch."""

from __future__ import annotations

import logging
import os
import sys
from importlib.machinery import PathFinder
from typing import Any

from .config import load_settings

logger = logging.getLogger("synthetic_reward_debug")
_DEFERRED_FINDER: Any = None
_PATCHED_MODULES: set[str] = set()
_TARGET_MODULES = frozenset({"verl.trainer.ppo.v1.trainer_base"})


def _patch_loaded_module(module_name: str, module: Any) -> bool:
    if module_name in _PATCHED_MODULES:
        return False
    settings = load_settings()
    if not settings.enabled:
        return False
    try:
        from .train_patch import install_loaded

        installed = install_loaded(settings, module_name, module)
    except Exception:
        logger.exception("synthetic reward debug: failed to patch %s", module_name)
        if settings.strict:
            raise
        return False
    _PATCHED_MODULES.add(module_name)
    return installed


class _DeferredLoader:
    def __init__(self, module_name: str, loader: Any):
        self._module_name = module_name
        self._loader = loader

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self._loader.exec_module(module)
        _patch_loaded_module(self._module_name, module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


class _DeferredPatchFinder:
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in _TARGET_MODULES:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None and not isinstance(spec.loader, _DeferredLoader):
            spec.loader = _DeferredLoader(fullname, spec.loader)
        return spec


def install_import_hook() -> bool:
    global _DEFERRED_FINDER
    settings = load_settings()
    if not settings.enabled:
        return False
    for module_name in _TARGET_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            _patch_loaded_module(module_name, module)
    if _DEFERRED_FINDER is None:
        _DEFERRED_FINDER = _DeferredPatchFinder()
        sys.meta_path.insert(0, _DEFERRED_FINDER)
    return True
