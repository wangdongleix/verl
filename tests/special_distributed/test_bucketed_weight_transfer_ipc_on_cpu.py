"""CPU-only regression tests for vLLM weight-transfer IPC dispatch."""

import torch
from torch.multiprocessing.reductions import reduce_tensor

from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as transfer


def test_rebuild_cpu_handle_does_not_inject_accelerator_device(monkeypatch):
    source = torch.arange(4, dtype=torch.float32)
    handle = reduce_tensor(source)
    assert len(handle[1]) < 7

    monkeypatch.setattr(transfer, "get_device_name", lambda: "cpu")
    rebuilt = transfer.rebuild_ipc(handle, device_id=0)

    assert rebuilt.device.type == "cpu"
    torch.testing.assert_close(rebuilt, source)


def test_rebuild_accelerator_handle_uses_named_rebuilder(monkeypatch):
    captured = {}

    def rebuild_npu_tensor(*args):
        captured["args"] = args
        return torch.ones(1)

    monkeypatch.setattr(transfer, "get_device_name", lambda: "cpu")
    transfer.rebuild_ipc((rebuild_npu_tensor, (0, 1, 2, 3, 4, 5, 6)), device_id=9)

    assert captured["args"][6] == 9
