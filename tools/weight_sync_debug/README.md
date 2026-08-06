# Weight-sync debug

This is an opt-in external diagnostic for comparing actor-exported tensors with
the tensors received and loaded by a vLLM rollout. It is injected through a
private `sitecustomize.py` and monkey patches methods after the relevant verl
modules load. No file under the `verl` Python package is changed at runtime.

Run the training command through the tool entry point so Ray/vLLM child
processes inherit the bootstrap:

```bash
bash tools/weight_sync_debug/run.sh -- bash run_kimi_k3_mindspeed.sh
```

The tool enables `VERL_WEIGHT_SYNC_DEBUG=1`. To write records to a file, set:

```bash
export VERL_WEIGHT_SYNC_DEBUG_OUTPUT='./debug/weight_sync.rank{rank}.pid{pid}.jsonl'
export VERL_WEIGHT_SYNC_DEBUG_STEPS=0,1,5-7
```

The remaining filters retain the existing names:

```bash
export VERL_WEIGHT_SYNC_DEBUG_NAMES='embed_tokens|lm_head|experts\.(0|831)\.w[123]\.weight'
export VERL_WEIGHT_SYNC_DEBUG_TARGET_NAMES='embed_tokens|lm_head|gate_up_proj|down_proj|w13_weight|w2_weight'
export VERL_WEIGHT_SYNC_DEBUG_MAX_TENSORS=8
export VERL_WEIGHT_SYNC_DEBUG_HASH=sample       # sample or full
export VERL_WEIGHT_SYNC_DEBUG_STATS=sample      # sample or full
export VERL_WEIGHT_SYNC_DEBUG_SAMPLE_SIZE=4096
```

The patches observe these stages:

| Stage | Injection point |
| --- | --- |
| `actor_export` | actor generator passed to colocated vLLM or checkpoint engine |
| `actor_export_base` | LoRA base generator passed to colocated vLLM |
| `vllm_receive` | vLLM worker before parameter mapping |
| `vllm_receive_lora` | vLLM worker before LoRA installation |
| `vllm_loaded` | target parameter immediately after `model.load_weights()` |

Compare a combined log with:

```bash
python3 tools/compare_weight_sync_debug.py training.log
```

The external patch also carries `global_steps` through the in-memory
`ServerAdapter` RPC wrapper, so no verl method signature needs to be changed.
Set `VERL_WEIGHT_SYNC_DEBUG_STRICT=1` when a patching error should fail the
process instead of being logged and ignored.

