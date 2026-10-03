# IDC — 数据中心能源协同调度

当前主链为安全 PPO + 联合滚动修正器，使用因果预测、任务服务约束和共同终点库存口径。

**当前状态：seed 0 长训已结束，但后期故障根因未闭合，尚未达到正式验证条件。**
固定 final 策略的 48 条 train-only 诊断全部通过，不能据此宣称长期训练问题已修复。
换电脑迁移不意味着批准重训、seed 1/2 或 validation/test。

- [最新诊断与证据限制](docs/audits/M6_P2b_S0_DIAGNOSIS.md)
- [换机与目录说明](docs/PROJECT_TRANSFER.md)
- [Mac→WSL 远程任务与全部回传](docs/REMOTE_EXECUTION.md)
- [项目背景交接](docs/CHATGPT_HANDOFF_2026-10-03.md)
- [协作规则](AGENTS.md)

## 目录

| 路径 | 内容 |
| --- | --- |
| scenario/、data_io/、contracts/ | 数据场景、因果预测、契约 |
| planning/、safe_rl/、safe_rl_v2/ | 规划修正器、wrapper、当前 PPO 主链 |
| checkpointing/、evaluation/ | 严格模型绑定与评估接口 |
| configs/ | 冻结参考值、训练配置、发布、实验矩阵 |
| tests/、scripts/ | 回归测试与诊断入口 |
| docs/task_cards/、docs/audits/ | 任务边界与审计证据 |
| docs/archive/2026-08-14/ | 原根目录四份历史审计，已归档 |
| data/、runs/ | 原始/处理数据与完整运行证据；大产物不进 Git |
| local_archive/ | 本地文献侧项及临时 manifest，随换机包保留 |
| marl/、grid_model/、legacy/、根目录兼容入口 | 受保护的历史实现，保留原路径 |

## 安装与检查

需要 Python 3.12 和 uv。在新机器创建环境，不复制旧 .venv：

```sh
uv sync --frozen
make check
```

必须先按换机说明恢复数据与证据并校验 hash。当前发布为
`configs/release/idc_formal_train_release_v2_r7.json`，训练配置 `v2_r5`、矩阵 `v4_r5`。
不要改后缀、改绑定或用较早 checkpoint 替换登记的 final 模型。
