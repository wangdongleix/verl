"""Dependency-light tests for deferred strict-parity patch installation."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    from . import patches
except ImportError:  # unittest discover with this directory as the top level
    from strict_parity_debug import patches
from strict_parity_debug import train_patch, vllm_patch
from strict_parity_debug.replay_io import load_replay, save_replay
from strict_parity_debug.replay_vllm import _invoke_replay, _select_actor_entry


class _FakeLoader:
    def exec_module(self, module):
        module.loaded = True


class DeferredPatchTest(unittest.TestCase):
    def test_step0_launcher_preserves_base_rollout_runtime(self):
        source = Path(__file__).with_name("run_step0.example.sh").read_text(encoding="utf-8")
        self.assertNotIn("dump_config_path=null", source)
        self.assertIn('TRAIN_MSPROBE_DUMP_PATH=${TRAIN_MSPROBE_DUMP_PATH:-$OUT/train/msprobe}', source)
        self.assertIn('ROLLOUT_MSPROBE_DUMP_PATH=${ROLLOUT_MSPROBE_DUMP_PATH:-$OUT/rollout/msprobe}', source)
        self.assertIn('"global_profiler.save_path=$TRAIN_MSPROBE_DUMP_PATH"', source)
        self.assertIn('--msprobe-dump-path "$ROLLOUT_MSPROBE_DUMP_PATH"', source)
        self.assertIn('--train "$TRAIN_MSPROBE_DUMP_PATH"', source)
        self.assertIn('--rollout "$ROLLOUT_MSPROBE_DUMP_PATH"', source)

    def test_msprobe_templates_have_no_output_paths(self):
        for name in ("msprobe_train.json", "msprobe_rollout.json"):
            payload = json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))
            self.assertNotIn("dump_path", payload)

    def test_target_module_is_patched_after_its_loader_runs(self):
        module_name = "verl.trainer.ppo.v1.trainer_base"
        module = types.ModuleType(module_name)
        loader = patches._DeferredLoader(module_name, _FakeLoader())

        with mock.patch.object(patches, "_patch_loaded_module") as patch_loaded:
            loader.exec_module(module)

        self.assertTrue(module.loaded)
        patch_loaded.assert_called_once_with(module_name, module)

    def test_unrelated_module_is_not_intercepted(self):
        finder = patches._DeferredPatchFinder()
        with mock.patch.object(patches.PathFinder, "find_spec") as find_spec:
            self.assertIsNone(finder.find_spec("json"))
        find_spec.assert_not_called()

    def test_replay_msprobe_is_installed_and_removed_on_worker(self):
        debugger = object()
        pytorch_module = types.ModuleType("msprobe.pytorch")
        pytorch_module.PrecisionDebugger = mock.Mock(return_value=debugger)
        msprobe_module = types.ModuleType("msprobe")
        runner = types.SimpleNamespace(
            debugger=object(),
            _debugger_started=True,
            _finalize_dump_data=mock.Mock(),
        )
        worker = types.SimpleNamespace(model_runner=runner)

        with mock.patch.dict(
            "sys.modules",
            {"msprobe": msprobe_module, "msprobe.pytorch": pytorch_module},
        ):
            vllm_patch._enable_replay_msprobe(worker, "/tmp/rollout.json", "/tmp/rollout-dump")

        self.assertIs(runner.debugger, debugger)
        runner._finalize_dump_data.assert_called_once_with()
        pytorch_module.PrecisionDebugger.assert_called_once_with(
            config_path="/tmp/rollout.json",
            dump_path="/tmp/rollout-dump",
        )
        runner._debugger_started = True
        vllm_patch._disable_replay_msprobe(worker)
        self.assertEqual(runner._finalize_dump_data.call_count, 2)
        self.assertEqual(runner._finalize_dump_data.call_args, mock.call())
        self.assertIsNone(runner.debugger)

    def test_named_msprobe_methods_are_added_to_worker_extension(self):
        class WorkerExtension:
            pass

        module = types.SimpleNamespace(vLLMColocateWorkerExtension=WorkerExtension)
        settings = types.SimpleNamespace(enabled=True)
        with mock.patch.object(vllm_patch, "_WORKER_PATCHED", False):
            self.assertTrue(vllm_patch.install_worker_extension(settings, module))

        self.assertIs(
            WorkerExtension.strict_parity_enable_replay_msprobe,
            vllm_patch._enable_replay_msprobe,
        )
        self.assertIs(
            WorkerExtension.strict_parity_disable_replay_msprobe,
            vllm_patch._disable_replay_msprobe,
        )

    def test_actor_selection_finds_requested_name_in_anonymous_namespace(self):
        entries = [
            {"name": "unrelated", "namespace": "other"},
            {"name": "vllm_server_0_0", "namespace": "job-anonymous-namespace"},
        ]
        self.assertEqual(_select_actor_entry(entries, "vllm_server_0_0"), entries[1])

    def test_actor_selection_prefers_replica_zero(self):
        entries = [
            {"name": "vllm_server_1_0", "namespace": "job"},
            {"name": "vllm_server_0_0_suffix", "namespace": "job"},
        ]
        self.assertEqual(_select_actor_entry(entries, "missing"), entries[1])

    def test_replay_rpc_has_zero_argument_server_signature(self):
        parameters = list(inspect.signature(vllm_patch._strict_replay).parameters)
        self.assertEqual(parameters, ["self"])
        source = inspect.getsource(vllm_patch._strict_replay)
        self.assertIn("collective_rpc(_ENABLE_MSPROBE_METHOD", source)
        self.assertIn("collective_rpc(_DISABLE_MSPROBE_METHOD", source)
        self.assertNotIn("collective_rpc(_enable_replay_msprobe", source)

    def test_multimodal_replay_adapts_prompt_without_reencoding_response(self):
        class Adapter:
            @staticmethod
            def prepare_vllm_prompt_ids(prompt_ids, tokenizer, image_data):
                self.assertIs(tokenizer, tokenizer_marker)
                self.assertEqual(image_data, ["image"])
                self.assertEqual(prompt_ids, [10, 11, 12])
                return [10, 99, 12]

        tokenizer_marker = object()
        server = types.SimpleNamespace(
            _vllm_adapter=Adapter(),
            model_config=types.SimpleNamespace(tokenizer=tokenizer_marker),
        )
        fields = {
            "responses": [[21, 22, 0]],
            "response_mask": [[1, 1, 0]],
        }

        with mock.patch.object(vllm_patch, "select_sample", side_effect=lambda value, index: value[index]):
            prepared, prompt_length, response_length, already_prepared = vllm_patch._prepare_replay_prompt_ids(
                server,
                fields,
                0,
                [10, 11, 12, 21, 22],
                ["image"],
            )

        self.assertEqual(prepared, [10, 99, 12, 21, 22])
        self.assertEqual(prompt_length, 3)
        self.assertEqual(response_length, 2)
        self.assertTrue(already_prepared)

    def test_replay_rejects_response_that_is_not_input_suffix(self):
        server = types.SimpleNamespace(
            _vllm_adapter=mock.Mock(),
            model_config=types.SimpleNamespace(tokenizer=object()),
        )
        fields = {"responses": [[31]], "response_mask": [[1]]}

        with (
            mock.patch.object(vllm_patch, "select_sample", side_effect=lambda value, index: value[index]),
            self.assertRaisesRegex(ValueError, "exact suffix"),
        ):
            vllm_patch._prepare_replay_prompt_ids(server, fields, 0, [10, 21], None)

    def test_actor_side_release_marker_survives_client_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "CONTINUE"
            settings = types.SimpleNamespace(pause_after_capture=True, continue_path=marker)

            vllm_patch._release_paused_training(settings)

            self.assertTrue(int(marker.read_text(encoding="utf-8").strip()) > 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for nested tensor serialization")
    def test_nested_input_ids_persist_attention_mask(self):
        import torch

        input_ids = torch.nested.nested_tensor(
            [torch.tensor([11, 12, 13]), torch.tensor([21])],
            dtype=torch.int64,
        )
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "replay.pt"
            save_replay(replay_path, {"input_ids": input_ids}, {})
            fields = load_replay(replay_path)["fields"]

        self.assertEqual(fields["input_ids"].tolist(), [[11, 12, 13], [21, 0, 0]])
        self.assertEqual(fields["attention_mask"].tolist(), [[True, True, True], [True, False, False]])

    def test_target_step_defers_rollout_sleep_until_replay(self):
        events = []

        class Trainer:
            global_steps = 1

            def on_sample_end(self):
                events.append("sleep")

        module = types.SimpleNamespace(PPOTrainerSync=Trainer)
        settings = types.SimpleNamespace(capture=True, pause_after_capture=True, target_global_step=1)
        with (
            mock.patch.object(train_patch, "_CAPTURED", False),
            mock.patch.object(train_patch, "_PATCHED", set()),
        ):
            self.assertTrue(train_patch._patch_v1_sync_sleep(settings, module))
            Trainer().on_sample_end()
            self.assertEqual(events, [])
            with mock.patch.object(train_patch, "_CAPTURED", True):
                Trainer().on_sample_end()
        self.assertEqual(events, ["sleep"])

    def test_replay_uses_awake_vllm_and_resleeps_after_msprobe_forward(self):
        events = []

        class Server:
            config = types.SimpleNamespace(free_cache_engine=True)
            _vllm_adapter = None
            _strict_parity_settings = types.SimpleNamespace(
                rollout_msprobe_config=Path("/tmp/msprobe.json"),
                rollout_msprobe_dump_path=Path("/tmp/msprobe-dump"),
                replay_path=Path("/tmp/replay.pt"),
                sample_index=0,
                pause_after_capture=False,
            )

            async def sleep(self):
                events.append("sleep")

            async def clear_kv_cache(self):
                events.append("clear_kv_cache")

            async def collective_rpc(self, method, args=()):
                events.append((method, args))

            async def _original_generate(self, **kwargs):
                events.append("generate")
                return types.SimpleNamespace(token_ids=[1], extra_fields={})

        Server._strict_parity_generate = Server._original_generate
        payload = {"fields": {"input_ids": [1, 2], "attention_mask": [1, 1]}, "fingerprints": {}}
        with (
            mock.patch.object(vllm_patch, "load_replay", return_value=payload),
            mock.patch.object(vllm_patch, "select_sample", side_effect=lambda value, _: value),
            mock.patch.object(vllm_patch, "tensor_fingerprint", return_value={}),
            mock.patch.object(vllm_patch, "_write_replay_result"),
        ):
            asyncio.run(vllm_patch._strict_replay(Server()))

        self.assertEqual(
            events,
            [
                "clear_kv_cache",
                (
                    vllm_patch._ENABLE_MSPROBE_METHOD,
                    ("/tmp/msprobe.json", "/tmp/msprobe-dump"),
                ),
                "generate",
                (vllm_patch._DISABLE_MSPROBE_METHOD, ()),
                "sleep",
            ],
        )

    def test_prepared_prompt_replay_is_injected_without_changing_server_source(self):
        seen_prompt_ids = []
        adapter_calls = []

        class Adapter:
            def prepare_vllm_prompt_ids(self, prompt_ids, tokenizer, image_data):
                adapter_calls.append((prompt_ids, tokenizer, image_data))
                return [99]

        class Engine:
            async def generate(self, **kwargs):
                yield types.SimpleNamespace(prompt_token_ids=[7, 8])

        class Server:
            _vllm_adapter = Adapter()
            model_config = types.SimpleNamespace(processor=object())
            engine = Engine()

            async def generate(
                self,
                prompt_ids,
                sampling_params,
                request_id,
                image_data=None,
                video_data=None,
                audio_data=None,
                mm_processor_kwargs=None,
                priority=0,
                kv_transfer_params=None,
            ):
                del sampling_params, request_id, video_data, audio_data, mm_processor_kwargs, priority
                del kv_transfer_params
                prompt_ids = self._vllm_adapter.prepare_vllm_prompt_ids(prompt_ids, object(), image_data)
                seen_prompt_ids.append(prompt_ids)
                async for _ in self.engine.generate(prompt_token_ids=prompt_ids):
                    pass
                return types.SimpleNamespace(extra_fields={})

        module = types.SimpleNamespace(
            vLLMHttpServer=Server,
            normalize_token_ids=lambda value: list(value),
            qwen2_5_vl_dedup_image_tokens=lambda value, processor: list(value) + [42],
        )
        settings = types.SimpleNamespace(enabled=True)
        with mock.patch.object(vllm_patch, "_PATCHED", False):
            self.assertTrue(vllm_patch.install(settings, module))

        server = Server()
        output = asyncio.run(
            server.generate(
                prompt_ids=[1, 2],
                sampling_params={},
                request_id="replay",
                prompt_ids_are_prepared=True,
            )
        )
        self.assertEqual(seen_prompt_ids, [[1, 2]])
        self.assertEqual(adapter_calls, [])
        self.assertEqual(output.extra_fields["strict_parity_submitted_prompt_ids"], [1, 2])
        self.assertEqual(output.extra_fields["strict_parity_processed_prompt_ids"], [7, 8])

    def test_replay_client_triggers_rpc_without_arguments(self):
        remote = mock.Mock(return_value="reference")
        actor = types.SimpleNamespace(strict_parity_replay=types.SimpleNamespace(remote=remote))
        ray = types.SimpleNamespace(get=mock.Mock(return_value={"ok": True}))

        self.assertEqual(_invoke_replay(ray, actor), {"ok": True})
        remote.assert_called_once_with()
        ray.get.assert_called_once_with("reference")


if __name__ == "__main__":
    unittest.main()
