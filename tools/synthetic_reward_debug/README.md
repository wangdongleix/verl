# Synthetic Reward Debug 注入工具

这个工具用于随机初始化模型的长跑实验：不修改 `verl` 源码，通过 Python
启动时的 monkey-patch 注入一个确定性的 synthetic reward，让 actor 真正执行
多轮更新，然后观察训练模型和 rollout 模型是否持续一致。

使用专用启动脚本就会自动开启注入和严格失败模式，不需要手动设置开关环境变量。

## 一、基本用法

先编辑 `run_synthetic_reward_debug.sh` 顶部的 `TRAINING_COMMAND`，把训练脚本路径
和全部 Hydra 参数写进去。例如：

```bash
TRAINING_COMMAND=(
  bash "${REPO_ROOT}/run_kimi_k3_mindspeed.sh"
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  actor_rollout_ref.rollout.n=4
  actor_rollout_ref.rollout.calculate_log_probs=true
  trainer.use_v1=true
  trainer.v1.trainer_mode=sync
  trainer.critic_warmup=0
)
```

上面只是配置示例；实际修改应直接编辑脚本文件。配置好之后，从仓库根目录直接运行：

```bash
bash tools/synthetic_reward_debug/run_synthetic_reward_debug.sh
```

本启动器不接收尾随命令行参数。这样可以保证每次长跑实验使用同一套训练脚本和参数，
需要调整时只改启动器顶部的配置块。

也可以使用 Python bootstrap：

```bash
PYTHONPATH=tools python3 -m synthetic_reward_debug.bootstrap -- <训练命令> [参数...]
```

Python bootstrap 仍然保留为调试 patch 的通用入口，但不会读取启动器里的
`TRAINING_COMMAND`；长跑实验推荐使用专用脚本。它会自动设置 `PYTHONPATH`、开启
synthetic reward、开启严格失败模式，并让 Ray/vLLM 子进程加载这个工具的 patch。

## 二、环境变量

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `VERL_SYNTHETIC_REWARD_DEBUG` | 脚本自动设为 `1` | 是否开启注入 |
| `VERL_SYNTHETIC_REWARD_SCALE` | 脚本中的 `SYNTHETIC_REWARD_SCALE` | reward 的范围，必须是有限正数；需要调整时编辑启动器 |
| `VERL_SYNTHETIC_REWARD_DEBUG_STRICT` | 脚本自动设为 `1` | patch 失败时直接终止 |

需要修改 reward 范围时，直接编辑 `run_synthetic_reward_debug.sh` 中的
`SYNTHETIC_REWARD_SCALE`。

## 三、必须满足的训练配置

工具当前只支持 V1 同步 GRPO：

```text
algorithm.adv_estimator=grpo
actor_rollout_ref.rollout.n>=2
trainer.use_v1=true
trainer.v1.trainer_mode=sync
algorithm.use_kl_in_reward=false
```

`rollout.n` 必须至少为 2。因为 GRPO 会在同一 prompt 的 response 组内做中心化，
只有一条 response 时 synthetic reward 仍然会变成零 advantage。

如果开启了 `algorithm.use_kl_in_reward=true`，工具会直接报错，避免 synthetic
reward 静默覆盖 KL reward。这个实验建议先关闭 KL reward。

另外需要确保 actor 确实会更新：

```text
trainer.critic_warmup=0
actor_rollout_ref.actor.ppo_epochs=1
```

如果使用 `run_step0.example.sh`，需要特别注意它为了初始 parity 会设置较大的
`critic_warmup`，会禁止 actor update，不能直接用于本实验。

## 四、注入的 reward 形式

对于同一个 prompt UID，工具按照 `session_id` 排序并分配均匀间隔的 reward。
例如 `rollout.n=4`、scale 为 1 时：

```text
session 0: -1.000
session 1: -0.333
session 2: +0.333
session 3: +1.000
```

同一个 session 的多条 trajectory 输出共享相同 reward；padding 行保持为 0；
reward 被放在每条 response 的最后一个有效 token 上。

工具会在原始 advantage 计算前写入 synthetic `rm_scores`，并在原始 advantage
计算后把 `rm_scores` 和 `token_level_rewards` 写回 TransferQueue。因此后续仍然
使用 verl 原本的 GRPO、PPO loss、optimizer 和权重同步流程。

这个 reward 是为了制造稳定的非零更新，不代表真实任务质量。由于 GRPO 会去掉
组内 reward 均值，`reward mean` 接近 0 是正常现象。

## 五、观察哪些指标

奖励和 advantage 曲线：

```text
debug/synthetic_reward/mean
debug/synthetic_reward/std
debug/synthetic_reward/max
debug/synthetic_reward/min

critic/score/mean
critic/rewards/mean
critic/advantages/mean
critic/advantages/max
critic/advantages/min
```

更新是否真的发生：

```text
actor/grad_norm
actor/pg_loss
```

训推一致性：

```text
training/rollout_probs_diff_mean
training/rollout_probs_diff_max
training/rollout_actor_probs_pearson_corr
rollout_corr/log_ppl_abs_diff
rollout_corr/ppl_ratio
```

建议同时开启：

```text
actor_rollout_ref.rollout.calculate_log_probs=true
```

判断标准是：`actor/grad_norm` 持续有限且非零，模型确实发生更新；更新之后的
rollout-vs-actor 差异不持续扩大，Pearson correlation 不持续下降，且没有
NaN/Inf。

## 六、step 对齐

V1 同步训练的时序是：

```text
global_steps=1: 使用初始权重进行 rollout 和 actor forward
step 1: actor update，然后同步新权重到 vLLM
global_steps=2: 观察第一次更新后的训推一致性
```

因此如果要观察完成 N 次更新后的状态，应重点查看 `global_steps=N+1` 的
rollout-vs-actor 指标。

## 七、结合权重同步诊断

这个工具只负责 reward 注入，不记录参数 hash。参数传输需要时可以另外使用：

```bash
VERL_WEIGHT_SYNC_DEBUG_STEPS=0,1,2,5 \
bash tools/weight_sync_debug/run.sh -- bash run_kimi_k3_mindspeed.sh
```

重点检查同一个 `global_steps` 下的：

```text
actor_export
vllm_receive
vllm_loaded
```

对于 Kimi 的 TP/EP 参数，`actor_export` 和 `vllm_receive` 更适合直接比较；
`vllm_loaded` 可能已经经过 TP shard 或 fused expert 映射，不能简单要求名称和
shape 完全相同。

## 八、实验后关闭

直接停止使用这个工具即可；它不会修改 `verl` 源码。若当前 shell 中曾经手动设置过
环境变量，可以执行：

```bash
unset VERL_SYNTHETIC_REWARD_DEBUG
unset VERL_SYNTHETIC_REWARD_SCALE
unset VERL_SYNTHETIC_REWARD_DEBUG_STRICT
```
