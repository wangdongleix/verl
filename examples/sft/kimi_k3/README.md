# Kimi K3 pure-language SFT with FSDP-Turbo

This example runs text-only SFT on Ascend NPUs. The Kimi checkpoints currently
available through Transformers load `KimiK3ForConditionalGeneration`, so the
wrapper is instantiated; however, the SFT data path, FSDP plan, EP plan, and
recompute plan target only its `language_model` submodule. No images, media
processor inputs, vision tower, projector, or vLLM rollout integration are
used in training.

## Prerequisites

- Install compatible `torch`, `torch_npu`, MindSpeed, and verl packages in the
  training environment.
- Set `FSDP_TURBO_ROOT` to the FSDP-Turbo source root. The launcher adds it to
  `PYTHONPATH`.
- Use a Kimi checkpoint compatible with `trust_remote_code=True`. The current
  recipe supports the `KimiK3ForConditionalGeneration` wrapper while training
  only its language model.
- Run on Ascend NPUs. `mindspeed_fsdp` is registered by verl only for `npu`.

## Dataset

`TRAIN_FILE` and optional `VAL_FILE` are Parquet files with a `messages`
column. Each row is a conversation:

```json
{
  "messages": [
    {"role": "user", "content": "Solve 2 + 2."},
    {"role": "assistant", "content": "4"}
  ]
}
```

`MultiTurnSFTDataset` invokes Kimi's XTML chat template and builds labels only
for assistant tokens. The example sets `data.ignore_input_ids_mismatch=true`
because the per-turn tokenizer rendering can differ from a single full-chat
render. Although the collator uses `pad_mode=no_padding`, this recipe sets
`model.use_remove_padding=false`: Kimi Linear currently has no packed-token
sequence-boundary implementation compatible with verl's flattened input path.
The engine therefore reconstructs a padded batch and attention mask before
the model forward, avoiding cross-conversation attention. Optional Kimi
thinking controls can be passed as Hydra overrides, for example:

```bash
data.apply_chat_template_kwargs.thinking_effort=low
```

## Launch

```bash
FSDP_TURBO_ROOT=/opt/FSDPTurbo \
MODEL_PATH=/models/Kimi-K3-Base \
TRAIN_FILE=/data/kimi-sft/train.parquet \
VAL_FILE=/data/kimi-sft/val.parquet \
FSDP_SIZE=16 EP_SIZE=4 TURBO_CP_SIZE=1 \
bash examples/sft/kimi_k3/run_kimi_k3_fsdp_turbo.sh 16 /checkpoints/kimi-k3-sft
```

For multinode training, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` identically with the appropriate per-node rank before invoking
the script on every node.

## FSDP-Turbo plan

The selected Hydra engine config is
`verl/trainer/config/engine/mindspeed_fsdp_turbo.yaml`. Its module paths are
for the wrapper's language model:

- FSDP: `language_model.model.embed_tokens`,
  `language_model.model.layers.{*}`, `language_model.lm_head`
- expert parallel:
  `language_model.model.layers.{*}.block_sparse_moe.experts`
- recomputation and forward hooks: `language_model.model.layers.{*}`

No vision or projector paths are configured.

`TURBO_CP_SIZE` maps to
`engine.fsdp_kwargs.distributed.ulysses_parallel_size`. The launcher currently
requires it to remain `1`: Kimi Linear does not yet ship the FSDP-Turbo
Ulysses function-patch plan required to split its attention and loss across CP
ranks. This check prevents an invalid partially-parallel run. When that plan
is added, keep `engine.ulysses_sequence_parallel_size` at one (otherwise verl
and FSDP-Turbo Ulysses would both be enabled), and add a Kimi packed-token
sequence-boundary implementation before enabling `model.use_remove_padding`.

## Architecture and distributed data flow

```mermaid
flowchart LR
  parquet[ParquetMessages] --> dataset[MultiTurnSFTDataset]
  dataset --> collator[SFTTensorCollator]
  collator --> trainer[SFTTrainer]
  trainer --> worker[TrainingWorker]
  worker --> engine[MindSpeedFSDPEngineWithLMHead]
  engine --> turbo[FSDPTurbo]
  turbo --> model[KimiK3LanguageModel]
  model --> loss[sft_loss]
  loss --> optimizer[MindSpeedOptimizer]
  optimizer --> checkpoint[CheckpointHandler]
```

`SFTTrainer` obtains data-parallel rank, size, and process group from the
engine. `MindSpeedFSDPEngineWithLMHead` delegates these methods to
FSDP-Turbo's parallel state, so `DistributedSampler`, validation loss
all-reduce, and logging use the true DP group after FSDP, EP, TP, and CP
partitioning. FSDP-Turbo wraps and shards model parameters, EP shards expert
weights, and CP splits the token dimension when enabled.

## GRPO-to-SFT mapping

| GRPO component | SFT equivalent |
| --- | --- |
| `actor_rollout_ref.model.*` | `model.*` |
| `actor_rollout_ref.actor.optim.*` | `optim.*` |
| `actor_rollout_ref.actor.mindspeed.*` | `engine.*` |
| PPO mini/micro batch | `data.train_batch_size` / `data.micro_batch_size_per_gpu` |
| prompt and response length limits | `data.max_token_len_per_gpu` |
| actor, reference model, rollout, reward, advantage, KL, entropy | Removed |
| Ray `main_ppo` launch and vLLM settings | Removed; use `torchrun -m verl.trainer.sft_trainer` |

SFT performs a single causal-LM forward/backward pass with token-level
cross-entropy. It does not generate responses, evaluate rewards, synchronize
actor weights to a rollout engine, or hold a reference model. Unlike the GRPO
baseline, the SFT launcher loads pretrained weights and does not set
`random_init=true`.

## Validation sequence

1. Run a one-step smoke job with a small text-only Parquet shard.
2. Confirm each rank reports the intended FSDP/EP/CP groups and that loss is
   finite.
3. Verify a checkpoint save and resume before scaling to the full A3B run.
