# 严格训推一致性采集

这个 `tools/strict_parity_debug` 目录主体是一组可移除的 monkey patch。为了让 strict replay 能绕过普通
rollout 的 Kimi prompt 适配器，并取得 vLLM 多模态处理前后的真实 token ID，本次还给 verl 的
`vllm_async_server.generate()` 增加了一个默认关闭的诊断参数；FSDPTurbo、vLLM 和 vLLM-Ascend 的公共接口
不受影响。它完成三件事：

1. 在 verl V1 的 `_compute_old_log_prob` 入口抓取完整 TransferQueue batch，保存
   `input_ids`、`attention_mask`、`position_ids`、`response_mask`、`prompts`、
   `responses`、处理后的多模态字段和 `mm_processor_kwargs`，并为每个张量计算
   fingerprint。
   对 V1，还会在 `AgentLoopWorkerTQ` 丢弃原始媒体前保存
   `multi_modal_data` sidecar。
2. 给 verl 的 vLLM Ray server 动态增加 `strict_parity_replay` RPC。RPC 从保存的
   `input_ids` 中分离 actor prompt 与 response，只允许 Kimi adapter 改写 actor prompt，随后逐 ID
   拼回 response，再把完整序列送入 vLLM prefill，并设置 `prompt_logprobs=1`；
   RPC 在 replay 前先清空 prefix、KV、encoder 和 multimodal processor cache，确保视觉塔也在
   诊断 Profile 内真实执行，然后只为这次 replay 替换一份新的 msprobe debugger，结束后立即移除。训练
   脚本原有的 rollout debugger 配置保持不变，不让诊断启动器改变普通 rollout 的
   执行条件；新 debugger 的 `step: [0]` 仍只对应固定输入 replay。
3. 递归读取 train/rollout 的原生 msprobe `dump.json`，识别以算子/模块名作为键、
   以 `Max/Min/Mean/Norm` 表示统计值的结构，输出缺失、额外、统计值差异和第一批
   不一致记录到 JSON 文件。

replay RPC 支持真实的 image/video/audio batch。它从 sidecar 恢复原始媒体，并通过
verl 现有 vLLM server 的 `image_data`、`video_data`、`audio_data` 和
`mm_processor_kwargs` 参数重新送入 vLLM。若处理后的多模态字段存在但 sidecar
缺失，工具会直接失败，不会把媒体 placeholder 当成普通文本运行。

## 0. 一个完整的 dataset 示例

下面的例子对比初始权重下的第一轮前向。verl 的训练循环从
`global_steps=1` 开始，因此脚本捕获 `global_steps=1`，但此时 actor 尚未执行该轮
更新，仍然是 step-0 初始权重。先复制
[`run_step0.example.sh`](run_step0.example.sh) 到一个私有路径，修改
`VERL_ROOT`、`TRAIN_SCRIPT` 和 `OUT`，然后直接执行：

```bash
cp tools/strict_parity_debug/run_step0.example.sh /tmp/run_strict_parity_step0.sh
vim /tmp/run_strict_parity_step0.sh
bash /tmp/run_strict_parity_step0.sh
```

不要使用 `source /tmp/run_strict_parity_step0.sh`。脚本里的 `export` 只会传给
训练命令及其 Ray 子进程，不会修改当前终端、`.bashrc` 或 `.zshrc`。

这是一条命令完成的流程：脚本启动一次单步训练，等待输入捕获，自动调用 vLLM
replay，释放训练侧前向，等待 msprobe 落盘，最后生成比较报告。无需另开终端。
成功的 replay 会由 vLLM Ray actor 自己写入 `CONTINUE`；因此即使只负责等待结果的
replay 客户端被系统 `SIGKILL`，远端 replay 仍可完成并释放训练。一键脚本会保留训练
暂停状态并等待该 marker，默认最多等待 1800 秒，可通过
`STRICT_PARITY_REPLAY_KILL_GRACE_SEC` 调整。
在目标 step，工具会把 `PPOTrainerSync.on_sample_end()` 的 rollout sleep 延后到
固定输入 replay 结束，因此不会额外执行一次容易扰动 EngineCore 状态的 sleep/wake
往返。replay RPC 依次清空 replay 相关缓存、安装 msprobe、执行固定输入 prefill、卸载 msprobe 并让 vLLM
sleep，最后才写入 `CONTINUE`。日志会分别打印 `deferring`、`enabling`、`running`、
`finalizing` 和 `returning ... to sleep`，可直接判断卡在哪个阶段。
运行过程中会打印 `initializing`、`PAUSED`、`replay completed` 和最终 `PASS/DIFF`
横幅。输出目录必须是干净目录，避免旧 msprobe 文件混入。
replay 客户端会通过 `ray.util.list_named_actors(all_namespaces=True)` 自动找到
`vllm_server_0_0` 所在的匿名 namespace；不需要让第二个 Ray driver 猜测训练 job 的
namespace。如果集群中同时存在多个同名 server，可设置 `VLLM_ACTOR_NAMESPACE` 明确
选择。

训练到目标 step 后会短暂暂停并生成：

```text
$OUT/replay.pt
$OUT/manifest.json
$OUT/READY.json
$OUT/replay_result.json
$OUT/media/<sha256>.pkl
```

`replay_result.json` 由 vLLM actor 在释放训练前写入，记录实际使用的
`sample_index`、捕获序列长度、actor prompt/response 长度、提交序列长度、vLLM 多模态处理后的精确
token IDs、媒体引用和 fingerprint。严格检查时应确认：提交序列的 response 后缀与 replay 中
`response_mask` 选出的 ID 完全相同；处理后序列除 media placeholder 展开外与训练序列完全相同。
V1 TransferQueue
没有显式 `attention_mask` 时，采集器会在把 nested `input_ids` 补齐前保存其真实行长，
避免把补齐位置当成 vLLM prompt token。

### 0.1 配置两侧 msprobe

可直接使用工具目录中的两份配置：

- [`msprobe_train.json`](msprobe_train.json)
- [`msprobe_rollout.json`](msprobe_rollout.json)

如果统计量已经把问题缩小到某一层，需要逐元素计算 Pearson、cosine、RMSE 或按 token 区域比较，
可改用只保存 decoder layer 0 输入/输出的 tensor 配置：

- [`msprobe_train_layer0_tensor.json`](msprobe_train_layer0_tensor.json)
- [`msprobe_rollout_layer0_tensor.json`](msprobe_rollout_layer0_tensor.json)

如果 layer 0 分区结果证明只有视觉 embedding 不同，可进一步使用视觉路径配置。它们只保存 patch
embedding、若干代表性 vision block、final encoder 和 multimodal projector，不会 dump 全模型：

- [`msprobe_train_vision_tensor.json`](msprobe_train_vision_tensor.json)
- [`msprobe_rollout_vision_tensor.json`](msprobe_rollout_vision_tensor.json)

`statistics` 配置中的 `Norm/Mean/Max/Min` 只适合定位量级和第一处可疑层，不能证明两个张量逐元素
一致；特别是两边 shape 不同或 token 身份未验证时，不应计算 Pearson。tensor 配置用于第二阶段，
必须先由 `replay_result.json` 证明 submitted/processed token ID 与训练 token ID 只有 media placeholder
展开差异，再去掉训练侧 singleton batch 维后逐元素比较相同位置。

两份 JSON 只描述采集行为，不包含 `dump_path` 或任何机器路径。所有输入、输出路径均
集中在 `run_step0.example.sh` 开头，由 `OUT`、`TRAIN_MSPROBE_CONFIG`、
`ROLLOUT_MSPROBE_CONFIG`、`TRAIN_MSPROBE_DUMP_PATH`、
`ROLLOUT_MSPROBE_DUMP_PATH` 和 `COMPARE_REPORT` 控制。

两侧输出目录分别是：

训练侧配置的 dump 目录应指向：

```text
$OUT/train/msprobe
```

verl 会继续在该目录下增加 `step_1/actor_compute_log_prob/step0` 等层级；比较工具会
递归读取，因此传入 `$OUT/train/msprobe` 根目录即可。

推理侧配置的 dump 目录应指向：

```text
$OUT/rollout/msprobe
```

配置内容如下；train 和 rollout 模板可以完全相同：

```json
{
  "task": "statistics",
  "step": [0],
  "level": "L0",
  "tensor": {
    "scope": [],
    "list": [],
    "tensor_list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

JSON 中的 `step: [0]` 是每个新 debugger 的第一次前向，不是 verl 的
`global_steps=1`。一键脚本会覆盖原训练脚本中的 profiler 参数：训练侧只启用
`actor_compute_log_prob`，但不会覆盖原训练脚本的 rollout `dump_config_path`；固定输入
replay RPC 会把 worker 切换到工具目录中的专用 rollout 配置，并把脚本中的
`ROLLOUT_MSPROBE_DUMP_PATH` 直接传给 `PrecisionDebugger`。如果原训练脚本也会采集普通
rollout，其输出必须与该路径不同，否则一键脚本会在 replay 前报错，避免混入目标结果。

### 0.2 手工重放（仅用于自动流程失败后的调试）

正常运行 `run_step0.example.sh` 不需要执行本节。若要手工重试，另开终端：

```bash
ray list actors | grep vllm_server

bash /mnt/share/w00848461/kimi-k3/verl/tools/strict_parity_debug/run_replay_vllm.sh \
  --actor-name vllm_server_0_0 \
  --ray-address auto \
  --replay /mnt/share/w00848461/kimi-k3/strict_parity_step0/replay.pt \
  --sample-index 0 \
  --msprobe-config /mnt/share/w00848461/kimi-k3/verl/tools/strict_parity_debug/msprobe_rollout.json \
  --msprobe-dump-path /mnt/share/w00848461/kimi-k3/strict_parity_step0/rollout/msprobe
```

RPC 成功后，vLLM actor 会在 replay 所在目录写入 `CONTINUE`，训练继续执行 actor
前向；客户端不会写入 release marker。两侧 msprobe 分别保存这次 vLLM 和
FSDP-Turbo 前向。

如果终端只显示 `Killed`，且 shell 返回 137，没有 Python traceback，说明 replay
客户端收到了外部 `SIGKILL`，最常见原因是宿主机或容器 cgroup 内存压力。这不等价于
vLLM replay 失败：若 worker 已打印 msprobe 初始化日志，RPC 往往仍在 actor 内执行。
新版一键脚本会明确打印等待状态，不会立刻清理训练。可在节点上检查
`/sys/fs/cgroup/memory.events`，以及有权限时检查内核 OOM 日志来确认外部杀进程原因。

如果 RPC 失败，不要写入 `CONTINUE`，训练会继续保持暂停。确认不再重试时才手工
释放：

```bash
date +%s%N > /mnt/share/w00848461/kimi-k3/strict_parity_step0/CONTINUE
```

### 0.3 手工比较

一键脚本会自动执行以下比较；只有需要重新调整容差时才手工运行：

```bash
bash /mnt/share/w00848461/kimi-k3/verl/tools/strict_parity_debug/run_compare.sh \
  --train /mnt/share/w00848461/kimi-k3/strict_parity_step0/train/msprobe \
  --rollout /mnt/share/w00848461/kimi-k3/strict_parity_step0/rollout/msprobe \
  --output /mnt/share/w00848461/kimi-k3/strict_parity_step0/msprobe_compare.json \
  --atol 1e-5 \
  --rtol 1e-3
```

只有 msprobe 输出确实落在上述两个目录后，比较命令才有意义。若目录为空，先检查
两侧 profiler 是否启用、`run_step0.example.sh` 中的输出路径，以及 msprobe 的内部
`step` 是否匹配。报告中的 `inputs.*.dump_files` 是发现的 `dump.json` 数量，
`inputs.*.files_with_records` 是成功解析出张量统计值的文件数量；两者可区分“路径下
没有 dump 文件”和“dump 文件存在但内容未被识别”。

## 1. 选择输入来源

支持两种输入来源：

- `dataset`（默认）：直接从真实 verl 数据集和 TransferQueue 捕获指定
  `global_step` 的实际训练 batch。这是最接近线上行为的模式，推荐先用它定位问题。
- `custom`：提前生成固定的 `replay.pt`，训练 monkey patch 在目标 step 把当前 TQ
  batch 替换成这份输入，随后 vLLM replay 使用同一文件。它适合缩短序列、固定 token、
  做 CP/EP=1 的最小复现。训练进程仍需一个可启动的最小数据集来初始化 verl 和
  rollout；真正用于比较的 log-prob batch 会被 custom replay 替换。

### 1.1 真实数据集模式

保持默认配置即可。若要显式设置，应放在上面的独立启动脚本中：

```bash
export STRICT_PARITY_INPUT_MODE=dataset
```

训练侧会从 V1 `TransferQueue` 中读取实际 `input_ids`、mask 和 position ids，保存
到 replay 文件；不需要手工准备输入。

### 1.2 自定义 token 模式

严格实验优先直接指定 token IDs，不要让 tokenizer/chat template 在两侧各自重新构造
输入：

```bash
bash tools/strict_parity_debug/run_make_replay.sh \
  --input-ids '[101, 202, 303, 404, 505]' \
  --prompt-length 3 \
  --output /mnt/share/w00848461/kimi-k3/custom_step0/replay.pt
```

也可以使用 JSON spec，支持 batch 和显式 mask：

仓库内有可直接复制的模板：`tools/strict_parity_debug/custom_input.example.json`。

```json
{
  "input_ids": [[101, 202, 303, 404, 505]],
  "attention_mask": [[1, 1, 1, 1, 1]],
  "position_ids": [[0, 1, 2, 3, 4]],
  "response_mask": [[0, 0, 0, 1, 1]]
}
```

```bash
bash tools/strict_parity_debug/run_make_replay.sh \
  --spec /path/to/custom_input.json \
  --output /mnt/share/w00848461/kimi-k3/custom_step0/replay.pt
```

简单文本输入也支持，但它只是生成 token IDs 的便利入口：

```bash
bash tools/strict_parity_debug/run_make_replay.sh \
  --prompt-text '问题：1+1等于多少？' \
  --response-text '2' \
  --tokenizer /path/to/tokenizer \
  --output /mnt/share/w00848461/kimi-k3/custom_step0/replay.pt
```

文本入口不能替代 Kimi 的多模态 processor 或 chat template 严格复现；Kimi/多模态
场景应使用已经由真实数据流程生成的 `input_ids`，或者在 spec 中提供最终 token IDs。

生成后，在独立的 custom 启动脚本中开启 custom 注入：

```bash
export STRICT_PARITY_INPUT_MODE=custom
export STRICT_PARITY_REPLAY_PATH=/mnt/share/w00848461/kimi-k3/custom_step0/replay.pt
```

custom batch size 必须和当前 verl batch size 一致。训练 batch 还必须满足当前 world size、DP/CP/EP
和 mini-batch 划分的最小整除约束，不能默认设为 1。例如本次 16-rank Kimi-K3 配置的最小可用
`data.train_batch_size` 是 16；要减少 rollout 工作量可保持训练 batch 为 16，同时设置
`actor_rollout_ref.rollout.n=1`。`run_step0.example.sh` 分别通过
`STRICT_PARITY_TRAIN_BATCH_SIZE` 和 `STRICT_PARITY_ROLLOUT_N` 提供这两个诊断覆盖项。
当模型配置的默认最大上下文远大于诊断序列、KV cache 因而无法分配时，可另外设置
`STRICT_PARITY_MAX_MODEL_LEN`；该值必须覆盖 prompt、视觉展开和 response 的总长度。

## 2. 准备环境

这个目录必须在训练 driver、Ray vLLM server actor 以及 vLLM worker 所在的每个
节点上可见。共享目录是最简单的方式。所有进程都需要把本目录的父目录加入
`PYTHONPATH`；下面的脚本会自动加入当前 checkout 的 `tools` 目录，并只为本工具
临时加入私有 `_bootstrap` 路径，不会影响 `tools` 下的其他工具。启动 shim 只注册
轻量的延迟导入钩子；只有训练 trainer 或 vLLM server 的目标模块实际加载后才安装
对应补丁，不会让 Ray dashboard、API server 等辅助进程提前导入 Torch/vLLM。

一键脚本使用工具目录内的两份 msprobe 配置。训练侧通过 verl
PrecisionDebugger profiler 加载配置；一键脚本保留训练脚本已有的
`additional_config.dump_config_path`，避免诊断入口改变普通 rollout 的运行条件。
replay RPC 内部会替换为一份新的 debugger，因此原 debugger 的计数不会消耗 replay
配置的 `step: [0]`。两份工具 JSON 都使用 `step: [0]`，表示各自新 debugger 看到的
第一次目标前向。rollout 配置和输出路径分别由训练启动环境中的
`STRICT_PARITY_ROLLOUT_MSPROBE_CONFIG`、`STRICT_PARITY_ROLLOUT_MSPROBE_DUMP_PATH`
传给 server。客户端触发 RPC 时不传任何参数，replay 路径、sample index、msprobe
配置和输出路径均以 actor 启动时加载的
设置为准；返回结果会再次校验这些值，防止静默重放错误输入。
vLLM engine worker 的开关通过 `vLLMColocateWorkerExtension` 上动态注册的具名方法调用，
`collective_rpc` 只传方法名字符串，不传 Python function，也不要求开启不安全的 pickle
序列化。

## 3. 比较结果说明

完整命令见 0.3。比较结果同时打印到终端并写入 JSON。返回码为 `0` 表示所有发现的 name、重复
次数、shape/dtype/hash 和数值统计均通过；返回码为 `2` 表示存在 missing、extra
或 mismatch。结果中的 `mismatches` 按 tensor name 和 occurrence 给出 train/rollout
来源文件及 `mean/min/max/absmax/l2` 等字段的差异。

如果两侧 msprobe 对同一层使用了不同名字，准备一个简单的 JSON 名称映射：

```json
{
  "train.layer.0.output": "layers.0.output",
  "rollout.layers.0.output": "layers.0.output"
}
```

然后增加 `--name-map path/to/name_map.json`。默认是严格名称匹配，不会擅自把
不同的层合并。

## 4. 多模态运行要求

- 真实多模态数据请使用 `STRICT_PARITY_INPUT_MODE=dataset`。V1 的原始媒体是在
  agent-loop worker 中产生的，因此 `STRICT_PARITY_DIR` 必须对 agent-loop worker、
  trainer 和 vLLM server/worker 节点可见，且路径在各节点上相同。
- 不要只复制 `replay.pt`。其中的 `media_refs` 指向 `media/` sidecar 和媒体文件；
  需要整体共享 `$STRICT_PARITY_DIR`。
- `multi_modal_inputs` 是给 actor 的处理后张量，`media_refs` 是给 vLLM 的原始
  媒体，两者必须同时存在。`manifest.json` 记录前者的 tensor fingerprint，RPC
  返回 `media_fingerprint` 记录后者的 sidecar fingerprint。
- legacy 路径只有在 `DataProto.non_tensor_batch` 仍包含
  `multi_modal_data` 时才能自动保存原始媒体；如果该路径只有
  `multi_modal_inputs`，工具会明确提示重新在能取得原始媒体的入口采集。
- `custom` 模式当前用于固定 token 的 text-only/CP/EP 最小复现；它不会凭几组
  token 自动重建 processor 的图像、视频或音频输入。真实多模态严格对齐应使用
  `dataset` 模式。

## 5. 环境变量

下表中的 strict 工具变量应放在一个独立的训练启动脚本中，并执行该脚本；不要放进
shell 初始化文件。`run_replay_vllm.sh` 和 `run_compare.sh` 优先使用命令行参数，
不需要复用训练脚本的环境变量。

| 变量 | 默认值 | 作用 |
|---|---|---|
| `STRICT_PARITY_ENABLE` | `0` | 总开关；脚本自动设为 `1` |
| `STRICT_PARITY_DIR` | `strict_parity_output` | replay、manifest 和 vLLM 请求的输出目录 |
| `STRICT_PARITY_INPUT_MODE` | `dataset` | `dataset` 使用真实 TQ batch；`custom` 注入指定 replay |
| `STRICT_PARITY_REPLAY_PATH` | `$DIR/replay.pt` | custom 输入或指定 replay 文件 |
| `STRICT_PARITY_ROLLOUT_MSPROBE_CONFIG` | 空 | replay 时在 vLLM worker 临时安装的 msprobe 配置 |
| `STRICT_PARITY_ROLLOUT_MSPROBE_DUMP_PATH` | 空 | replay 专用 msprobe 输出根目录；一键脚本自动设置 |
| `STRICT_PARITY_CAPTURE_MEDIA` | `1` | V1 是否保存原始 image/video/audio sidecar |
| `STRICT_PARITY_GLOBAL_STEP` | 不过滤 | 只在这个 verl `global_steps` 捕获 |
| `STRICT_PARITY_SAMPLE_INDEX` | `0` | replay RPC 默认重放整批中的哪个样本 |
| `STRICT_PARITY_PAUSE_AFTER_CAPTURE` | `0` | 捕获后等待 replay RPC 释放 |
| `STRICT_PARITY_CONTINUE_FILE` | `$DIR/CONTINUE` | 释放训练的 marker |
| `STRICT_PARITY_WAIT_TIMEOUT_SEC` | `0` | 等待超时；`0` 表示一直等待 |
| `STRICT_PARITY_STRICT` | `0` | patch/捕获失败时抛异常；严格实验建议设为 `1` |
| `STRICT_PARITY_TRAIN_BATCH_SIZE` | 不覆盖训练脚本 | 一键脚本的诊断 train/gen batch；必须满足分布式整除约束 |
| `STRICT_PARITY_ROLLOUT_N` | 不覆盖训练脚本 | 一键脚本的诊断 rollout response 数量；layer tensor 调试可设为 `1` |
| `STRICT_PARITY_MAX_MODEL_LEN` | 不覆盖训练脚本 | 诊断 vLLM 的最大上下文；只用于约束 KV cache，必须大于 processed sequence length |

## 6. 解释边界

- `global step` 是 verl trainer 的权重版本筛选；msprobe 配置里的 `step` 是
  msprobe 自己的采集步，两者不是同一个计数器。
- replay 文件保存整个 batch，RPC 的 `sample-index` 只选择其中一个样本进行
  vLLM prefill；manifest 的 batch fingerprint 和 RPC 返回的 sample fingerprint
  都应保留。
- 本工具不改变 attention、position ids 或权重加载逻辑。它保存
  `position_ids` 用于审计，但 vLLM 会依据自身输入路径重新构造位置编码；如果
  两侧 position-id 生成规则不同，msprobe 的第一处差异正是定位证据。
- V1 的 bypass 模式直接复用 rollout log-prob，不会执行 actor 前向；做严格对齐
  时必须关闭 `rollout_correction.bypass_mode`，并确保 `calculate_log_probs` 等
  配置不会绕过待比较的 forward。
- 这是诊断工具，不是训练逻辑补丁。权重版本应通过现有的
  `VERL_WEIGHT_SYNC_DEBUG` 另行确认。不要在内存紧张的大模型 msprobe 采集任务中同时
  开启 `VERL_WEIGHT_SYNC_DEBUG_STATS=full`；全量统计会产生额外的 FP32 临时张量。
