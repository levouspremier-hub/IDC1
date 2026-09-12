# AGENTS.md — 项目规则书（本仓库所有 Agent 与 vibecoding 助手共用）

> 本文件是 DSH agent 与 VSCode 助手**共同遵守**的唯一规则源。改动前先读本节。
> 上游方案：`docs/EXECUTION_PLAN.md`（模块/步骤）与 `docs/AUDIT_EXECUTION_CHAIN.md`（现状行号）。

## 0. 一句话边界

这是“考虑业务服务约束与新能源不确定性的数据中心计算负荷—储能协同优化调度”研究项目。**任何 Agent 不得改动研究主线、不得放松物理约束换取“可行”、不得让评估通过而牺牲业务语义。**

---

## 1. 项目结构（模块 M0–M9）

| 模块 | 目录 | 说明 |
|---|---|---|
| M0 工程外壳 | `Makefile` `AGENTS.md` `contracts/` `tests/` | 环境/门禁/契约版本 |
| M2 数据契约 | `contracts/` | 6 个 pydantic 模型 + 版本常量 |
| M3 物理链 | `envs/idc_price_env.py` `idc_model/` | 逐组执行/功耗/SOC/碳（**首版核心**） |
| M4 联合规划器 | `planning/` `safe_rl/corrector*.py` | 前瞻检查 + 动作修正 |
| M5 安全 PPO v2 | `safe_rl/` | 三套价值 + 多约束乘子 |
| M6 统一评估 | `eval/eval_base.py` | 全方法统一指标 |
| M7 可视化/报告 | `viz/` `runs/` `scripts/build_report.py` | 图/面板/机器报告 |
| M8 电网验证 | `grid_model/` | IEEE-14 离线 AC（后置） |
| M9 实验统计 | `scripts/` `runs/` | 对照/消融/敏感性/多种子 |

**不动**：`marl/`（多智能体对照基线）、`grid_model/` 主逻辑、`legacy/`、顶层 shim。

---

## 2. 唯一命令（经 `make` 闸门）

```bash
uv sync && make check      # 建环境 + 门禁
make test                  # tests/ 全量
make probe                 # planning/probe.py 求解耗时/规模/内存
make smoke                 # scripts/vertical_slice.py 最小切片
make train / make eval     # 训练 / 评估
make figures / make report # 出图 / 机器生成报告
make contract              # 契约版本一致性
```

> 每阶段结束时对应门禁必须全绿（`docs/EXECUTION_PLAN.md` §7）。

---

## 3. 契约版本规则

- `contracts/CONTRACT_VERSION_ID` 在**任何破坏性变更**（契约形状、动作空间、检查点 schema）时必须 +1。
- 旧 checkpoint/模型**不得静默加载**到新环境：无 `contract_version` 即显式报错。

---

## 4. `runs/` 产物规范

- 每次运行写 `runs/<run_id>/`：`config.yaml`、`metrics.parquet`、`report.json`、`figures/`。
- `runs/` 不入库；机器可读结论（`report.json` + `metrics.parquet`）保留并归档。
- 报告结论**由 `runs/*/metrics.parquet` 生成，不手工填数**。

---

## 5. 任务卡四要素（每张卡必须齐备）

**边界 / 禁止项 / 可执行验收 / 回滚点**。一次一卡；验收命令先于改动存在；禁止为绿而改验收标准。

---

## 6. 八条红线（任何 Agent / 助手都不得违反）

1. 不得放松接入容量约束来让结果“可行”。
2. 不得清空任务队列、重置 SOC、跳过任务来让评估通过。
3. 不得逐测试日重新计算归一化参考值。
4. 不得在测试/评估时读取未来真值。
5. 不得改动 `marl/`。
6. 不得静默保留被废弃的动作维度。
7. 不得在没有对应验收测试的情况下修改 `envs/idc_price_env.py` 的 `step()`。
8. 不得把修正后的执行动作写进原始动作的 log-prob 记录。

---

## 7. 分支与提交

- `main` 只接受人工确认过的合并。
- 分支：`p0-bootstrap` / `p1-contracts` / `p2-physics/<card>` / `p3-corrector/<card>` / `p4-safeppo/<card>` / `p5-eval-viz` / `p6-experiments`。
- 语义变更（改变仿真物理或实验语义）**必须逐卡 + 人工看 diff**，不 commit 未经确认的语义改动；纯新增文件可直接提交。
