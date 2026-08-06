# Tools

`tools/` 只保存诊断和开发辅助工具，不自动修改 verl 的运行环境。

目录约定：

- 单文件、无运行时注入的工具可以直接放在本目录，例如
  `compare_weight_sync_debug.py`。
- 包含多个 Python 模块、启动脚本、配置文件或 monkey patch 的工具必须使用
  独立子目录，例如 `strict_parity_debug/`，并在子目录内提供自己的 `README.md`
  和启动脚本。
- 不在仓库根目录或 `tools/` 根目录放置 `sitecustomize.py`、全局导入副作用或
  自动 monkey patch。需要注入时，只能由对应工具的启动脚本临时设置
  `PYTHONPATH`，并加载该工具私有的 bootstrap。

新增工具时，优先遵循下面的结构：

```text
tools/<tool_name>/
├── README.md
├── run_*.sh          # 可选：用户入口
├── __init__.py
└── *.py
```

当前多文件工具：

- [`strict_parity_debug`](strict_parity_debug/README.md)：固定输入并采集训练、推理
  前向，用于严格训推一致性定位。
- [`weight_sync_debug`](weight_sync_debug/README.md)：通过外部 monkey patch 记录
  actor 到 vLLM 的权重同步张量签名，不修改 `verl` 源码。
