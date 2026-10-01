# M6-P2b：终点库存修复、train-only 验收与三种子重训

## 开卡 / 回滚点

- 起点：`55be5e53330636d9f7d21cf26631d47b220dbd91`；开卡工作树为空，diff --check exit 0。
- 独立分支：`p5-eval-viz-m6-p2b-runtime`（从失败测试提交 d7ccf56 分出；历史诊断分支 p5-eval-viz-m6-p2b/c936f53 保留，未改写历史）。公共 forecast 契约不修改，避免冻结 B6 revision 失效。
- 回滚：在独立 worktree 从最终 HEAD newest-first `git revert` 本卡提交，核对起点 tree；不合并 paper-baseline。
- 授权：用户明确要求实现本卡，包括短跑、版本发布、验收通过后的 3×512 批重训及 636 条 train 诊断。

## 边界

允许：新增 contracts/inventory.py（公共 forecast 契约保持原字节）；planning/model.py、planning/snapshot_adapter.py、planning/corrector.py；safe_rl/corrector_wrapper.py；safe_rl_v2/rollout.py、formal_train_loop.py，以及新增 inventory 训练/诊断模块；新增 scenario/inventory_release.py 与 checkpointing/inventory_eval_input.py；scripts/m6p2b_inventory_repair.py；Makefile；本卡相关测试、docs 文档、新 v2 配置/发布与 v4 矩阵；全新 runs/m6p2b_* 产物。

旧配置/发布/矩阵/运行产物保留。新正式入口须绑定训练语义，不允许旧 checkpoint 跨版本恢复。历史重放只读权重，不进行更新。

本卡暴露失败后的诊断入口还包括 scripts/m6p2b_short_trace.py、
scripts/m6p2b_budget_probe.py、scripts/m6p2b_reward_counterfactual.py；允许补充
planning/service_guard.py 与 tests/test_m6p2b_service_guard.py，研究在既有 B6
信息边界内为当前已到达任务保留服务功率。不得借此改变任务分配优先级、读取未来
具体任务或把保守规划假设当作实际物理不可达证明。服务保护尚未实现；若采用新
语义，先建立失败测试，重新标定并写全新 v2_r2 / v4_r2 资产及 run-id，保留首轮资产。

## 禁止项

不读或运行 validation/test；不改冻结服务标准、refs_v4、日期、批次顺序或种子；不放宽物理/任务约束，不重置 SOC、清空队列或跳过任务。不可达、超时、预测误差分别记录，不伪造成功。不改受保护目录，不把 exec 写入 raw 或 log-prob。原奖励优先；禁止逐步保持 50% SOC 的惩罚。若涉及环境 step，先提交失败测试。

## 改前证据与验收命令

改前：`uv run python -m scripts.m6p2b_inventory_repair --help` → No module named scripts.m6p2b_inventory_repair，exit 1。现有 snapshot 缺 episode 终点，滚动 cap=24，投影无终点库存约束。

```sh
uv run pytest tests/test_m6p2b_terminal_inventory.py
uv run python -m scripts.m6p2b_inventory_repair --help
uv run python -m scripts.m6p2b_inventory_repair --phase replay --run-id m6p2b_historical_replay_v2
uv run python -m scripts.m6p2b_inventory_repair --phase calibrate --run-id m6p2b_calibration_v2
uv run python -m scripts.m6p2b_inventory_repair --phase release --run-id m6p2b_release_v2
uv run python -m safe_rl_v2.inventory_train --short --seed 0 --run-id m6p2b_short_seed0_v2
uv run python -m safe_rl_v2.inventory_train --short --seed 1 --run-id m6p2b_short_seed1_v2
uv run python -m safe_rl_v2.inventory_train --short --seed 2 --run-id m6p2b_short_seed2_v2
uv run python -m scripts.m6p2b_inventory_repair --phase gate --run-id m6p2b_short_gate_v2
# gate 通过后逐 seed 从头完整训练；不得用短跑 checkpoint 冒充正式恢复
uv run python -m safe_rl_v2.inventory_train --seed 0 --run-id m6p2b_formal_seed0_v2
uv run python -m safe_rl_v2.inventory_train --seed 1 --run-id m6p2b_formal_seed1_v2
uv run python -m safe_rl_v2.inventory_train --seed 2 --run-id m6p2b_formal_seed2_v2
uv run python -m scripts.m6p2b_inventory_repair --phase audit --run-id m6p2b_trainonly_212x3_v2
make check
git diff --check
git status --short
```

失败定位及奖励依据补充命令（train-only，权重更新为零）：

```sh
uv run python -m scripts.m6p2b_short_trace --seed 0 --batch-index 0 --run-id m6p2b_short_trace_seed0_batch0_v2_r1
uv run python -m scripts.m6p2b_budget_probe --trace-run m6p2b_short_trace_seed0_batch2_v2 --baseline-revision ed184bf --run-id m6p2b_runtime_budget_verified_v2_r8
uv run python -m scripts.m6p2b_reward_counterfactual --run-id m6p2b_reward_counterfactual_v2
uv run pytest tests/test_m6p2b_sparse_assembly.py
```

## 证据产物与验收标准

每个 run 必须 config.yaml、metrics.parquet、report.json、figures/、manifest.json，包含 revision、lock/data/scenario hash、种子、真实命令、失败状态。失败产物不得删除或覆盖。

测试覆盖真实终点、末步、上下偏差、不可达安全执行、共享阶段约束、因果来源、raw/log-prob、新版恢复与旧 checkpoint 拒绝。24 个既有 train origin 的历史三 seed 重放、三种参考提案标定；三 seed 各 8 批真实短跑，共 4608 transitions，最终写盘读回；gate 检查完整服务、物理、库存以及学习信号与求解预算。任何未解决失败不得启动长训。

通过后按冻结 512 批 × seed 0/1/2 顺序从新初始化训练，审核每 seed 98304 transitions、8192 Adam step、512 乘子更新，再完成 636 条 train-only 诊断。真实终点对准现有 50% 目标；45–55% 资格及方法终点电量差 ≤1e-6 kWh 的配对规则不变。报告 validation readiness 与缺失四方法；本卡不运行 validation/test。

## 状态

执行中。首轮三 seed ×8 批已完成最终 checkpoint / 五类产物与读回验签，
但服务 35/96、库存区间 66/96、共同目标 57/96、物理违规 0、回退 135 次；
短跑闸门失败，未启动长训、636 条新训练诊断或 validation/test。原奖励未改。
已修复部分求解预算与 Stage B 原生数值错误；该候选代码尚未重新发布，首轮
release/checkpoint 绑定保留为历史证据，不能在代码 hash 不一致时用于新正式训练。
