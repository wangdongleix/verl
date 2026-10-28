from functools import partial
from unittest.mock import patch

import torch

from verl.utils.megatron.dist_checkpointing import (
    _MemoryEfficientSyncSaveStrategy,
    _preload_tensors_for_sync_save,
)


def test_sync_preloader_reuses_cpu_tensor():
    tensor = torch.ones(4)
    buckets = [("part", "key", (b"metadata", [("tensor", tensor)]))]

    staged = _preload_tensors_for_sync_save(buckets, non_blocking=False)

    assert staged[0][2][1][0][1] is tensor


def test_sync_strategy_replaces_only_its_request_preloader(tmp_path):
    tensor = torch.ones(4)
    buckets = [("part", "key", (b"metadata", [("tensor", tensor)]))]

    def upstream_preloader(write_buckets, non_blocking=True):
        return write_buckets, non_blocking

    class Request:
        def __init__(self):
            self.preload_fn = partial(upstream_preloader, buckets, False)
            self.executed = None

        def _replace(self, **changes):
            self.preload_fn = changes["preload_fn"]
            return self

        def execute_sync(self):
            self.executed = self.preload_fn()

    request = Request()
    strategy = _MemoryEfficientSyncSaveStrategy()
    with patch.object(strategy, "async_save", return_value=request):
        strategy.save({}, tmp_path)

    assert request.executed[0][2][1][0][1] is tensor
