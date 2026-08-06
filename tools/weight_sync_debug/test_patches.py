import asyncio
import sys
import types
import unittest
from unittest.mock import patch

from weight_sync_debug.config import Settings
from weight_sync_debug import train_patch, vllm_patch


def _fake_recorder(traces, received, loaded):
    recorder = types.ModuleType("weight_sync_debug.recorder")

    def trace(weights, stage, context):
        traces.append((stage, dict(context)))
        return iter(weights)

    recorder.trace_weight_stream = trace
    recorder.log_received_weights = lambda weights, stage, context: received.append(
        (stage, dict(context), list(weights))
    )
    recorder.log_loaded_model_parameters = lambda model, names, stage, context: loaded.append(
        (stage, dict(context), list(names))
    )
    return recorder


class TestTrainPatches(unittest.TestCase):
    def setUp(self):
        train_patch._PATCHED.clear()

    def test_server_adapter_traces_and_forwards_global_step(self):
        traces = []

        class ServerAdapter:
            async def _execute_method(self, method, **kwargs):
                self.rpc = (method, kwargs)

            async def update_weights(self, weights, global_steps=None, **kwargs):
                self.weights = list(weights)
                await self._execute_method("update_weights_from_ipc", kwargs={**kwargs, "use_shm": False})

        module = types.SimpleNamespace(ServerAdapter=ServerAdapter)
        recorder = _fake_recorder(traces, [], [])
        with patch.dict(sys.modules, {"weight_sync_debug.recorder": recorder}):
            self.assertTrue(
                train_patch.install_loaded(
                    Settings(enabled=True, strict=True),
                    "verl.workers.rollout.vllm_rollout.vllm_rollout",
                    module,
                )
            )
            server = ServerAdapter()
            asyncio.run(server.update_weights([("weight", object())], global_steps=7, base_sync_done=False))

        self.assertEqual(traces, [("actor_export_base", {"global_steps": 7, "base_sync_done": False})])
        self.assertEqual(server.rpc[1]["kwargs"]["global_steps"], 7)

    def test_checkpoint_engine_stream_is_traced(self):
        traces = []

        class Engine:
            async def send_weights(self, weights, global_steps=None):
                self.weights = list(weights)

        class ActorRolloutRefWorker:
            def __init__(self):
                self.checkpoint_engine = Engine()

            async def update_weights(self, global_steps=None, mode="auto"):
                await self.checkpoint_engine.send_weights([("weight", object())], global_steps=global_steps)

        module = types.SimpleNamespace(ActorRolloutRefWorker=ActorRolloutRefWorker)
        recorder = _fake_recorder(traces, [], [])
        with patch.dict(sys.modules, {"weight_sync_debug.recorder": recorder}):
            self.assertTrue(
                train_patch.install_loaded(
                    Settings(enabled=True, strict=True), "verl.workers.engine_workers", module
                )
            )
            worker = ActorRolloutRefWorker()
            asyncio.run(worker.update_weights(global_steps=11, mode="other"))

        self.assertEqual(traces, [("actor_export", {"global_steps": 11})])


class TestVllmPatches(unittest.TestCase):
    def setUp(self):
        vllm_patch._PATCHED.clear()

    def test_receive_and_loaded_records_keep_rpc_step(self):
        received = []
        loaded = []

        class Model:
            def __init__(self):
                self.parameter = object()

            def named_parameters(self):
                return [("weight", self.parameter)]

            def load_weights(self, weights):
                self.weights = weights
                return ["weight"]

        model = Model()

        class WorkerExtension:
            def _iter_all_models(self):
                return [model]

            def update_weights_from_ipc(self, peft_config=None, base_sync_done=False, use_shm=False):
                self._update_weights(
                    [("weight", object())],
                    peft_config=peft_config,
                    base_sync_done=base_sync_done,
                )

            def _update_weights(self, weights, peft_config=None, base_sync_done=False, quant_prepared=False):
                for item in self._iter_all_models():
                    item.load_weights(weights)

        module = types.SimpleNamespace(vLLMColocateWorkerExtension=WorkerExtension, is_fp8_model=lambda _: False)
        recorder = _fake_recorder([], received, loaded)
        with patch.dict(sys.modules, {"weight_sync_debug.recorder": recorder}):
            self.assertTrue(vllm_patch.install_loaded(Settings(enabled=True, strict=True), module))
            worker = WorkerExtension()
            worker.update_weights_from_ipc(base_sync_done=False, global_steps=13)

        self.assertEqual(received[0][0], "vllm_receive")
        self.assertEqual(received[0][1]["global_steps"], 13)
        self.assertEqual(
            loaded[0],
            (
                "vllm_loaded",
                {
                    "global_steps": 13,
                    "base_sync_done": False,
                    "quant_prepared": None,
                    "model_index": 0,
                    "quantized": False,
                },
                ["weight"],
            ),
        )
        self.assertEqual(model.weights[0][0], "weight")


if __name__ == "__main__":
    unittest.main()
