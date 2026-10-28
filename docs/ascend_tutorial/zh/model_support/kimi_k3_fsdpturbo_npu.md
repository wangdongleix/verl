# Kimi K3 FSDPTurbo NPU 使用指南

本文用于指导在 Ascend A3 上使用 verl + FSDPTurbo + vLLM-Ascend 运行 Kimi K3 多模态 MoE GRPO 训练。适配支持动态批量、训练权重同步和 rollout decode 图模式。

## 版本要求

| software | version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.10.0（CPU wheel，配合 torch_npu） |
| torch_npu | 2.10.0.post7（官方 A3 Dockerfile 指定的 daily 包） |
| CANN | 9.2.0（与 vLLM-Ascend main 的 A3 Dockerfile 配套） |
| vLLM | vLLM-Ascend main 验证的提交 `c0b6f74bee518aeaf5d72b200a0216e72fbb5e7c` |
| Transformers | 5.19.0 |
| Triton Ascend | 3.6.0（官方 A3 Dockerfile 指定的构建） |
| FSDPTurbo | 官方 main 基线 `1fa40ad350eb681ed413ca64bffedc166c986e0f` 及 Kimi 增量 |
| verl | 官方 main 基线 `3c6d9b43b67c74ec343d77874cd8fa6e841523da` 及 Kimi 增量 |
| vLLM-Ascend | 官方 main 基线 `4e6eac4e476d9be3476e99440f4cabed67b253ea` 及 Kimi 增量 |

请使用满足上述版本要求的 Ascend NPU 环境，并同时准备配套源码、Python 依赖及编译产物。源码版本与算子编译产物需要匹配。

容器还需提供与宿主驱动匹配的 `npu-smi`，可只读挂载宿主的
`/usr/local/sbin/npu-smi`。verl 使用它检查 IPC 支持；仅挂载驱动库并不能替代该工具。

main 的 AuxOutput 路由输出要求启用底层前缀缓存机制。配置
`rollout.enable_prefix_caching=False` 时，verl 会为每次生成分配独立的
`cache_salt`，保留不跨请求复用前缀的行为，同时支持完整路由记录。
权重更新后的缓存清理仍由 rollout 生命周期执行。图模式使用
`additional_config.rl_config.sleep_mode_extra_cleanup=False`；旧的顶层
`enable_sleep_mode_extra_cleanup` 字段不再受支持。
脚本通过 `additional_config.enable_flashcomm1` 和 `weight_nz_mode` 显式设置
FlashComm1 与权重格式，默认分别为开启和 `0`。兼容读取旧的
`VLLM_ASCEND_ENABLE_FLASHCOMM`、`VLLM_ASCEND_ENABLE_NZ` 启动参数；main
运行库本身已不再读取这两个旧变量。RL 权重同步必须禁用 NZ。

使用 [kimi-k3-build.zip](https://modelscope.cn/models/S1eeeep/test/file/view/master/kimi-k3-build.zip) 的构建配方升级 main 时，保留其 CANN 9.0.1 基础镜像，并加入 main 配套的 CANN 9.2 工具包和隔离 Python 环境。运行入口显式加载 `/usr/local/Ascend/cann-9.2.0/set_env.sh`。FLA 使用官方 wheel，`triton-ascend-kernels` 使用下述主仓提交，不应用压缩包中的旧 KDA 补丁。

KDA 与短卷积依赖 `flash-linear-attention-npu-a3`，版本沿用对应 vLLM-Ascend A3 Dockerfile 的 `26.9.1+dev5cc6990`。vLLM-Ascend 的 C++ 扩展及自定义 OPP 必须基于当前 main 源码和 CANN 重新构建，不能沿用 0.26 的二进制。

FSDPTurbo main 的 KDA 还需要源码依赖
[`triton-ascend-kernels`](https://gitcode.com/Ascend/triton-ascend-kernels)，
本次固定提交为 `7acb9c40a6ca827e885d902b7eaee5d45457b230`。
该模块与 Triton Ascend 编译器分别安装；运行前应确认 KDA 选择加速后端。
main 开发提交应使用开发版本元数据，不能把 `VLLM_VERSION` 标为 `0.31.0`，
否则会选择该 release 对应的接口。脚本默认 `0.31.1.dev0`，允许显式覆盖。

其他运行依赖包括 Ray、Hydra/OmegaConf、TransferQueue、datasets、pyarrow、pandas、NumPy、Pillow、qwen_vl_utils、tensordict、torchdata、accelerate、peft，以及 reward 使用的 mathruler、pylatexenc。其他库的精确版本尚未在本文锁定。不要直接用 verl 的 CUDA extras 覆盖这套 NPU 环境。

## 模型和脚本

| model | 本地权重 | script |
| --- | --- | --- |
| Kimi K3 top16 Geo3K SFT | `/path/to/models/kimi-k3-geo3k-sft` | [run_kimi_k3_fsdpturbo.sh](../../../../../examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh) |


## 硬件和并行配置

以下为四机全层基线，共 64 张逻辑 NPU：

| nnodes | devices per node | FSDP | CP | EP | expert FSDP | rollout TP | rollout EP | rollout replicas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 64 | 1 | 16 | 4 | 16 | 16 | 4 |

global prompt batch 为 16，rollout n=4，生成 64 条训练样本，作为四机 bsz=1 的配置示例。开启动态批量后，实际 micro batch 还受序列长度及 token budget 影响，不能仅依据 `ppo_micro_batch_size_per_gpu` 判断实际批量。

Kimi 的 `gate_up_proj` 保持输入优先布局并使用 `Shard(2)`，`down_proj` 使用
`Shard(1)`。对于 `[E,3584,6144]` 和 `[E,3072,3584]` 的参数，expert FSDP
并行度必须同时整除 6144 和 3072，因此支持三机 48 卡、训练 EP=16、eFSDP=3。
训练与推理 EP 不同时，权重同步按专家、行、列区间直接交换分片。

### FSDPTurbo Ulysses CP（关闭 TND）

设置 `TURBO_CP_SIZE=2` 或 `4`，启动脚本会同时配置 actor 和 ref 的
`fsdp_config.turbo_config.distributed.ulysses_parallel_size`、MLA/KDA 通信替换以及
Kimi 多模态融合后的序列切分。保持 `actor_rollout_ref.model.use_remove_padding=False`，
不要同时开启 verl 的 `ulysses_sequence_parallel_size`。

单机 16 卡示例：

```bash
NNODES=1 NPUS_PER_NODE=16 TURBO_EP_SIZE=8 TURBO_CP_SIZE=2 \
  bash examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh
```

CP 大小需要整除设备数和 attention/KDA 的 head 数。CP 分组内处理同一批样本，
数据并行大小为设备数除以 CP 大小；FSDP 仍覆盖全部训练设备。
本配置保持 rollout routing replay 开启。verl 将完整展开序列的路由 ID、权重和重放 mask
对齐到图文融合位置，然后使用与模型隐藏状态相同的 CP padding 和切分。
模型先完成图文融合，再切分展开后的序列。verl 在各分片选择其负责的原始 token 位置，
只同步 log probability、entropy 和 sum_pi_squared 等逐 token 标量，不聚合完整词表 logits。

CPU 输出和梯度回归测试：

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29458 \
  tests/workers/engine/test_kimi_turbo_cp.py
```

## 数据和模型准备

使用已准备好的 Geo3K Parquet 数据：

```bash
export BACKEND_ROOT=/path/to/kimi-k3
export MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export TRAIN_FILE=/path/to/datasets/geo3k/train.parquet
export TEST_FILE=/path/to/datasets/geo3k/test.parquet
```

训练脚本会在 `$BACKEND_ROOT/models/fsdpturbo-<权重目录名>` 创建模型副本目录，链接原始权重，并复制 FSDPTurbo 的 modeling 代码。必须通过该准备流程，使实际加载的模型使用本次适配实现；只修改仓库内 modeling、仍直接加载旧模型目录，不能保证算子生效。

main 的 Kimi processor 按文本批次匹配图片占位符。verl 适配层保留单样本文本列表，确保 Geo3K 的图片与占位符一一对应；准备数据时应检查过滤后的数据集非空。

## 启动训练

先在所有节点配置相同依赖并加载 CANN 环境。执行前将 `/path/to/...` 和 `<...>` 占位符替换为实际配置。

### 启动 Ray 集群

在主节点执行：

```bash
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
ray start --head --port="$MASTER_PORT" \
  --node-ip-address="$MASTER_ADDR" \
  --resources='{"NPU":16}'
```

在其余三个 worker 节点分别执行：

```bash
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
export NODE_IP="<current-node-ip>"
ray start --address="$MASTER_ADDR:$MASTER_PORT" \
  --node-ip-address="$NODE_IP" \
  --resources='{"NPU":16}'
```

在主节点执行 `ray status --address="$MASTER_ADDR:$MASTER_PORT"`，确认四个节点、共 64 个 NPU 资源已就绪。

### 启动 FSDPTurbo

以下命令仅在主节点执行：

```bash
export BACKEND_ROOT=/path/to/kimi-k3
export MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export TRAIN_FILE=/path/to/datasets/geo3k/train.parquet
export TEST_FILE=/path/to/datasets/geo3k/test.parquet
export KIMI_RUN_MODE=multinode
export NNODES=4
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
export RAY_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
export NPUS_PER_NODE=16
export TURBO_EP_SIZE=16
export ROLLOUT_TP=16
export TRAIN_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export MAX_VISUAL_TOKENS=1024
export VERL_VLLM_MULTIMODAL_TOKEN_MARGIN=1024
export VERL_MAX_RESPONSE_LENGTH=2048
export VERL_VLLM_MAX_MODEL_LEN=4096
export VERL_VLLM_MAX_NUM_SEQS=16
export VERL_VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VERL_TOTAL_TRAINING_STEPS=3
export VERL_SAVE_FREQ=-1
export VERL_VLLM_ENFORCE_EAGER=False

bash "$BACKEND_ROOT/verl/examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh" \
  "++ray_kwargs.ray_init.address=$RAY_ADDRESS" \
  data.max_prompt_length=1024 \
  trainer.balance_batch=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=2147483648'
```

这是 prompt=1024、response=2048 的三步训练示例，不保存 checkpoint，也不主动开启 profiling。网络接口默认值与环境有关，迁移时应设置 `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 为实际互通网卡。

## eFSDP 参数分片配置

`actor_rollout_ref.{actor,ref}.fsdp_config.turbo_config.distributed.ep_plan.efsdp_shard_placements` 接收 `(参数名, 维度)` 列表。
YAML/JSON用二元素数组，配置初始化时转为元组：

```yaml
distributed:
  ep_plan:
    efsdp_shard_placements:
      - [gate_up_proj, 2]
      - [down_proj, 1]
```

参数名相对于每个被eFSDP包装的专家模块，ModuleList则相对于每个成员，
可写成 `projection.weight`。维度从0开始；Kimi配置保持gate/up切列、down切行。
未配置的参数沿用二维及以上切dim1、一维切dim0；低精度all-gather仅支持dim0。
重复名称、负数和非整数维度在配置初始化时报错；不存在的参数、维度越界及
共享参数冲突在包装模型时报错。模型不再提供 `efsdp_shard_placement` 回调；
外部自定义启动配置需要显式传入这两项。

## 权重同步与上游能力复用

非量化权重同步使用 vLLM 的 `initialize_layerwise_reload` 和
`finalize_layerwise_reload`。verl 在接收 IPC bucket 时复制一次输入张量，
保证延迟到后续 bucket 才加载的权重不会被覆盖。Ascend 不再重复复制输入，
只保留 callable loader 和非 checkpoint 运行时状态的兼容处理。
训推 log-prob 差异直接使用上游指标 `training/rollout_logprobs_diff_mean`。
