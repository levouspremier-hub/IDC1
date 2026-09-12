# 数据中心能源协同调度 — 模块化执行方案（v2 详细版）

> 版本：v2（执行蓝图，**不代表任何代码已被修改**；实际改动按 §5 任务卡门禁逐卡落地）
> 上游依据：《数据中心能源协同调度研究与实施计划：审核修订版》
> 分工依据：五类工作（**agent 直接做 / vibecoding / agent 小步验证 / vibecoding / 本机先测吞吐**）
> 关系：本文件是主方案；`docs/AGENT_EXECUTION_PLAN.md` 为按 Phase 展开的任务卡底稿，两者保持一致，冲突以本文件为准。

---

## 0. 现状基线（本会话已核实，非假设）

| 项 | 事实 |
|---|---|
| 机器 | Apple M5 (arm64) / macOS 26.6 / 16 GB 统一内存 / 10 核 / 磁盘剩余约 154 GB |
| 代码 | 177 个 `.py`；主战场 `envs/idc_price_env.py`(1726 行)、`eval/eval_base.py`(1563 行)、`safe_rl/`(8 文件) |
| 工具链 | `uv 0.12.12` ✅、`git` ✅、`node/npm` ✅、`code`(VSCode CLI) ✅、`docker` CLI ✅(daemon 未确认)；无 conda/brew/git-lfs（均非必需） |
| **环境** | **已建立并验证**：`pyproject.toml` + `uv.lock` + `.python-version`(3.12) + `.venv`；157 包就位 |
| 已验证 | `torch 2.5.1`(MPS 可用)、`gymnasium 1.2.3`、`stable-baselines3 2.8.0`、`scipy 1.15.3`(**HiGHS MILP/LP 求解通过**)、`pandapower 3.4.0`、`matplotlib/plotly/streamlit/duckdb/pyarrow/mlflow/tensorboard` |
| 遗留阻塞 | B1 旧 `environment.yml`(Windows+CUDA) 已由 uv 环境取代；B3 `.gitignore` 白名单已补 3 个新文件；**B4 HARL 未 pin**、**B5 缺 `AGENTS.md`/`Makefile`**、**B6 数据仅 1 个月 USEP** |

**结论**：环境与可视化栈就绪，工程外壳（`pyproject.toml`/`uv.lock`/`.gitignore`）已部分完成；剩余工作从 §3 的 M1 继续。

---

## 1. 目标模块架构

统一以 **6 个数据契约** 为模块间唯一交换语言，模块内部实现可替换、可单独测试、可单独回滚。

```
                    ┌──────────────────────── 数据契约层 (contracts) ────────────────────────┐
                    │  ScenarioBundle │ SystemSnapshot │ DispatchProposal │                  │
                    │  TaskAllocation │ DispatchResult │ EvaluationRecord │  + 版本常量      │
                    └──────────────▲───────────────────────────────────────────▲─────────────┘
                                   │                                           │
  ┌──────────────┐   ┌─────────────┴──────────┐    ┌───────────────┐   ┌───────┴────────┐
  │ 场景/预测提供器 │   │   任务管理器 + 环境      │    │ 联合可行性规划器 │   │   统一评估器     │
  │ data/ 场景    │──▶│  envs/idc_price_env.py  │──▶ │ planning/      │──▶│ eval/eval_base  │
  │ (真值/预测/单位) │   │  (逐组执行/功耗/SOC/碳)  │    │ (前瞻检查/修正) │   │ (成本/业务/碳/峰值)│
  └──────────────┘   └─────────────┬──────────┘    └───────┬────────┘   └───────┬────────┘
                                   │                       │                   │
                                   ▼                       ▼                   ▼
                          ┌──────────────┐         ┌──────────────┐    ┌──────────────┐
                          │ 安全 PPO (v2) │ ──a_raw──▶│ 修正器 wrapper │    │  可视化/报告    │
                          │ safe_rl/      │         │ safe_rl/      │    │ viz/ + runs/  │
                          │ 三套价值+乘子  │ ◀─(s,r,c)─│ a_exec=S(s,·) │    │ 图/面板/报告   │
                          └──────────────┘         └──────────────┘    └──────────────┘
                                   ▲                                        ▲
                          ┌────────┴────────┐                       ┌──────┴───────┐
                          │ 电网验证器(后置)  │                       │ 运行产物库     │
                          │ grid_model/     │                       │ runs/<id>/   │
                          │ IEEE-14 AC/OPF  │                       │ parquet/json │
                          └─────────────────┘                       └──────────────┘
```

### 模块职责表（对齐计划书 §1）

| # | 模块 | 责任 | 现有/新建 | 主文件 |
|---|---|---|---|---|
| M0 | 工程外壳 | 环境、命令闸门、规则书、契约版本、测试骨架 | 部分完成 | `pyproject.toml` `uv.lock` `Makefile` `AGENTS.md` `contracts/__init__.py` |
| M1 | 场景与预测提供器 | 当前真值、可见预测、来源、单位、哈希 | 改造 | `data_io/`、`idc_model/task_forecast.py`、新增 `contracts/scenario.py` |
| M2 | 数据契约 | 6 个 pydantic 模型 + 版本常量 + 往返序列化 + 哈希 | **新建** | `contracts/` |
| M3 | 任务执行与物理链 | 逐组分配、逐组实际功耗、SOC、碳、逾期三分类 | **改造核心** | `envs/idc_price_env.py` `idc_model/{task,task_model,power_model}.py` |
| M4 | 联合可行性规划器 | 前瞻检查 + 动作修正 + 失败分类 + 求解探针 | **新建** | `planning/{window,model,solver,probe}.py` `safe_rl/corrector*.py` |
| M5 | 安全 PPO v2 | 三套价值、多约束乘子、rollout buffer 扩展、恢复一致性 | **改造+新增** | `safe_rl/{safe_ppo_v2,safe_costs,lagrangian,corrector_wrapper}.py` |
| M6 | 统一评估器 | 所有方法统一输出成本/业务/碳/消纳/峰值/可靠性/耗时 | 扩展 | `eval/eval_base.py` |
| M7 | 可视化与报告 | 静态图、交互图、面板、产物库、机器生成报告 | **新建** | `viz/` `runs/` `scripts/build_report.py` |
| M8 | 电网验证(后置) | 代表性轨迹 AC 潮流/OPF 离线验证 | 已有，核查 | `grid_model/` |
| M9 | 实验与统计 | 端到端/机制对照、消融、敏感性、多种子 | 编排 | `scripts/`、`runs/` |

> 红线（不变）：不改 `marl/`、不改 `grid_model/` 主逻辑、不碰 `legacy/` 与顶层 shim；`envs/idc_price_env.py` 的 `step()` 无对应验收测试不得改。

---

## 2. 数据契约（M2，全部方法共享的“语言”）

| 契约 | 关键字段 | 说明 |
|---|---|---|
| `ScenarioBundle` | 数据、预测、任务、容量、单位、`hash` | 一个场景的不可变输入包 |
| `SystemSnapshot` | 时间、任务、SOC、预算、修正器上下文 | 决策时点的完整状态 |
| `DispatchProposal` | 原始计算建议(20 维) + 储能建议(1 维) | 策略原始输出 `a_raw` |
| `TaskAllocation` | **任务 × 服务器组(20) 执行量矩阵** | 分配器产物，`sum(group_exec)==完成量`，逐组≤容量，逐任务≤上限 |
| `DispatchResult` | 执行动作、能量流、修正原因、求解状态 | 修正器+环境执行后的结果 |
| `EvaluationRecord` | 业务、经济、碳、消纳、峰值、耗时 | 评估器统一输出 |

- 版本常量 `contracts/CONTRACT_VERSION`；环境/动作/检查点契约版本化，**旧模型不得静默加载到新环境**。
- `hash` 确定性（同输入同哈希）；单位缺失即构造报错。

---

## 3. 执行步骤（模块 × 步骤 × 角色 × 验收 × 回滚）

> 角色代号：**A**=agent 直接做（只读/纯新增/脚手架）；**V**=vibecoding（人 + VSCode 助手，逐卡审 diff）；**T**=本机实测（先测吞吐再定外移）。语义变更一律 `A|V` 逐卡 + 人工看 diff。

### M0 工程外壳（剩余部分）｜角色 A

| 步 | 内容 | 验收 | 回滚 |
|---|---|---|---|
| M0.1 | `Makefile`：`setup/check/test/probe/smoke/train/eval/figures/report/contract` | `make check` 可运行（允许空过） | 删除 |
| M0.2 | `AGENTS.md`：项目结构、每阶段唯一命令、契约版本规则、`runs/` 规范、§5.3 八条红线 | 内容覆盖全部条目 | 删除 |
| M0.3 | `contracts/__init__.py` 版本常量 | `uv run python -c "from contracts import CONTRACT_VERSION"` | 删除 |
| M0.4 | `tests/` 骨架 + `conftest.py`（复用 `marl/tests/conftest.py` 的 RNG 隔离） | `uv run pytest` 可运行 | 删除 |

### M1 场景/预测 + 数据核验｜角色 A（审计）+ V（实现）

| 步 | 内容 | 验收 |
|---|---|---|
| M1.1 | 现有链路审计：数据来源、口径、时区、单位、哈希（**只读**） | 审计报告 + 依赖/数据口径确认 |
| M1.2 | 数据获取与核验：≥1 全年匹配价格+负荷；风光/温度/任务轨迹；核验来源 | 数据清单 + 来源 + 哈希（见 §8） |
| M1.3 | `ScenarioBundle` 提供器：真值/预测/单位/来源 | 单位缺失报错；哈希确定 |

### M2 数据契约｜角色 V（人审 `TaskAllocation` 维度）

| 步 | 内容 | 验收 |
|---|---|---|
| M2.1 | 6 个契约模型（pydantic） | dict/JSON 往返；`hash` 确定 |
| M2.2 | 单位/来源字段强制 | 缺失即报错 |
| M2.3 | `tests/test_contracts.py` | `uv run pytest tests/test_contracts.py -v` 全过 |
| M2.4 | 旧 checkpoint 拦截（仅加校验不改逻辑） | 无 `contract_version` 的旧模型显式报错 |

### M3 物理链改造｜角色 A|V 逐卡（首版核心贡献）

| 步 | 内容 | 验收（关键断言） |
|---|---|---|
| M3.0 | 改造前先建探针 `scripts/probe_physics.py` + `tests/test_physics_invariants.py` | 输出当前基线（允许红） |
| M3.1 | 取消 20 维→标量 `np.sum` 压缩，逐组容量下传 | `len(planned_capacity_vec)==20`；本卡不动维度 |
| M3.2 | 分配器输出任务×组执行量 | `sum(group_exec)==完成量±1e-9`；逐组≤容量；**逐任务≤上限** |
| M3.3 | 废除 `_actual_loads_from_completed_work`，逐组实际负载→功耗 | 能量平衡断言；`_compute_reward*` 与 `configs/` diff 零改动 |
| M3.4 | α 诊断对照（α=0 vs 0.25 旧链） | 成本/功耗对照表 |
| M3.5 | 逾期三分类：未到期积压/逾期积压/逾期完成 | “截止后才完成”计入逾期完成 |
| M3.6 | 动作维度 23→21（20 计算 + 1 储能） | `action_dim==21`；旧 23 维 checkpoint 报错 |
| M3.7 | 观测补齐（期限分布/剩余工作/逐组容量/SOC/接入/资源/预测/预算） | 改资源容量时观测必须可见 |

### M4 联合可行性规划器｜角色 A 小步验证（语义核心）

| 步 | 内容 | 验收 |
|---|---|---|
| M4.1 | 修正器接口 + 单步版 `safe_rl/corrector.py` | 只收 `(SystemSnapshot, DispatchProposal)`，不调 policy/value |
| M4.2 | 修正器 wrapper `safe_rl/corrector_wrapper.py` | 训练/评估同语义；`info` 记 `correction_reason/raw_action/exec_action` |
| M4.3 | **最小求解探针 `planning/probe.py`（第 1 周硬要求）** | 记录：纯环境步耗时、修正器中位数/P95/超时率/不可行率、变量规模、整数变量数、训练吞吐与内存 |
| M4.4 | 滚动前瞻 `planning/{window,model,solver}.py` | 24h 基础窗，超窗延至最晚期限；缺预测区间用边界假设，**不读未来真值** |
| M4.5 | LP/MIP 双后端 | 默认 MIP 保证充放电互斥；LP 松弛仅显式开启且物理一致性验证通过 |
| M4.6 | 失败四分类 + 显式业务缺口 | 超时/数学不可行/物理复核失败/预测超界；失败时维持物理边界并记缺口 |

### M5 安全 PPO v2｜角色 V（审原始/执行动作一致性）

| 步 | 内容 | 验收 |
|---|---|---|
| M5.1 | rollout buffer 扩展 `safe_rl/safe_ppo_v2.py` | 同时存原始动作+log-prob；执行动作不覆盖原始概率记录 |
| M5.2 | 三套价值 `safe_rl/safe_costs.py` | 收益/业务约束/碳代价独立；不重复计入奖励项 |
| M5.3 | 多约束乘子 `safe_rl/lagrangian.py` | `state_dict`/`load_state_dict` 往返一致 |
| M5.4 | 恢复一致性 | CPU + `torch.use_deterministic_algorithms(True)` 位级一致 |
| M5.5 | 泄漏回归 | 未来信息隔离测试成为观测改动的必过回归项 |

### M6 统一评估器｜角色 V

| 步 | 内容 | 验收 |
|---|---|---|
| M6.1 | 扩展 `eval/eval_base.py` | 所有方法统一输出成本/业务/碳/消纳/峰值/可靠性/耗时 |
| M6.2 | 归一化参考值冻结 `scripts/freeze_refs.py` | 仅训练集或预定物理尺度；带哈希冻结；不逐测试日重算 |

### M7 可视化与报告｜角色 A（纯新增）

| 步 | 内容 | 验收 |
|---|---|---|
| M7.1 | `runs/<id>/` 产物规范（config/parquet/report.json/figures） | 每次运行齐套 |
| M7.2 | 静态图 `viz/figures.py`（matplotlib） | 落 PNG 文件供共同审阅 |
| M7.3 | 交互图 `viz/figures_html.py`（plotly） | 落 HTML 文件 |
| M7.4 | 面板 `viz/app.py`（streamlit，读 duckdb/parquet） | 跨 run 对比、调度回放、修正幅度、失败分类 |
| M7.5 | 报告 `scripts/build_report.py` | 结论由 `runs/*/metrics.parquet` 机器生成，不手填 |

### M8 电网验证（后置）｜角色 A（只读核查 + 离线验证）

| 步 | 内容 | 验收 |
|---|---|---|
| M8.1 | 设备容量/字段含义核查 | 字段语义确认 |
| M8.2 | 代表性完整轨迹 IEEE-14 离线 AC/OPF | 不包装成新加坡真实网络 |

### M9 实验与统计｜角色 T（本机先测吞吐，再决定外移）

| 步 | 内容 | 验收 |
|---|---|---|
| M9.1 | 本机吞吐/内存/耗时预算（依赖 M4.3 探针） | 预算表；超预算先讨论 |
| M9.2 | 端到端 5 对照 | 规则 / 独立 MPC / 原惩罚式安全 PPO / 新+单步修正 / 新+联合修正 |
| M9.3 | 机制对照 3 组 | 同修正器异策略 / 同策略单步 vs 前瞻 / 联合预留 vs 分别预留 |
| M9.4 | 资源/算法消融分开 | 无储能、固定调度等与算法比较分离 |
| M9.5 | 敏感性 | 固定策略泛化（改预测误差/资源/接入）≠ 配置最优可达，分开报告 |
| M9.6 | 多种子 + 分层统计 | 首轮 ≥3 训练种子；测试 ≥30 日历日；60/20/20 时间划分 |

---

## 4. 角色分工矩阵（对应任务图五类工作）

| 工作包 | 角色 | 边界 | 验收重点 |
|---|---|---|---|
| 环境、数据来源、现有链路审计 | **A** 直接做 | 只读侦察，不改代码 | 确认依赖与数据口径 |
| ScenarioBundle 等契约 + 最小垂直切片 | **V** vibecoding | 纯新增 + 人审维度语义 | 审核动作—任务—功耗语义 |
| 联合可行性规划器 + 物理测试 | **A** 实现、小步验证 | 逐卡、一屏 diff、人看 diff | 审核可行/失败定义 |
| Safe PPO 接入 + 回归测试 | **V** vibecoding | 动 `safe_rl/` 与 `step()`，回归必过 | 审核原始动作/log-prob/执行动作一致性 |
| 正式多种子实验 | **T** 本机先测吞吐再定外移 | 预算由 M4.3 实测决定 | 冻结配置与预算 |

**让 A/V 协作不打架的唯一纪律**：共用同一份 `AGENTS.md` + 同一任务卡四要素（边界/禁止项/可执行验收/回滚点）+ 同一红线；语义变更不 commit 未经人确认的 diff。

### 最小垂直切片（M2→M3→M4→M6 各取一最小件串通）

`scripts/vertical_slice.py` 单脚本贯通：1 天场景 → `ScenarioBundle` → 一步 `env.step` 产出 `TaskAllocation`/`DispatchResult` → 一次 LP 修正 `a_exec` → 一条 `EvaluationRecord` → 一张图 + 一行 parquet。挂 `make smoke` 门禁。**切片绿 = 全链路契约成立**，之后再横向扩展每个模块。

---

## 5. 工程化规范

### 5.1 环境与锁定（已落地）
- `pyproject.toml` + `uv.lock` + `.python-version`(3.12) + `.venv`。
- 核心包锚定 `==`：`torch 2.5.1`、`gymnasium 1.2.3`、`SB3 2.8.0`、`numpy 2.2.6`、`pandas 2.3.3`、`scipy 1.15.3`、`pandapower 3.4.0`、`tensorboard 2.20.0`、`matplotlib 3.10.9`；其余由 `uv.lock` 锁定。
- 一条命令复现：`uv sync && make check`。

### 5.2 命令闸门（Makefile 目标）
```
setup  uv sync           check  ruff+mypy+pytest -m "not slow"
test   pytest            probe  求解/规模/内存探针
smoke  端到端最小切片      train  runs/<id>   eval  统一评估
figures 出图             report 机器可读报告   contract 契约版本一致性
```

### 5.3 八条红线（AGENTS.md 必含）
1. 不得放松接入约束换“可行”；2. 不得清队列/重置 SOC/跳任务换评估通过；3. 不得逐测试日重算归一化参考；4. 不得在测试/评估读未来真值；5. 不改 `marl/`；6. 不得静默保留废弃动作维度；7. 无验收测试不改 `step()`；8. 不得把执行动作写进原始动作 log-prob 记录。

### 5.4 任务卡四要素 + 分支
- 每张卡：**边界 / 禁止项 / 可执行验收 / 回滚点**；一次一卡，验收命令先于改动存在，禁止为绿而改验收标准。
- 分支：`main` 只收确认过的合并；`p0-bootstrap`、`p1-contracts`、`p2-physics/<card>`、`p3-corrector/<card>`、`p4-safeppo/<card>`、`p5-eval-viz`、`p6-experiments`。

### 5.5 版本化与复现产物
- 环境/动作/检查点契约版本化；旧模型不静默加载。
- 归档：配置、代码版本、依赖版本、数据来源+哈希、场景+种子、checkpoint、逐步评价、测试结果、运行命令、失败记录。
- 小型必要产物白名单；大型产物按清单独立归档；`runs/` 不入库但保留 `report.json` + `metrics.parquet`。

---

## 6. 可视化方案（已装栈，按层落地）

| 层 | 工具（版本） | 用途 | 产出 |
|---|---|---|---|
| 训练曲线 | tensorboard 2.20.0 / mlflow 3.16.0 | 奖励/约束价值/乘子/修正率随训练变化 | `runs/<id>/tboard` |
| 静态图 | matplotlib 3.10.9 | 功耗曲线、SOC、成本/碳对比、修正幅度分布 | `runs/<id>/figures/*.png` |
| 交互图 | plotly 7.0.0 | 调度回放、帕累托权衡、分层散点 | `*.html` |
| 产物库 | duckdb 1.5.5 + pyarrow 25.0.1 | `metrics.parquet` 统一指标仓库 | `runs/*/metrics.parquet` |
| 面板 | streamlit 1.63.0 | 跨 run 对比、回放、修正/失败分类（导师/团队易读） | `viz/app.py` |
| 报告 | `scripts/build_report.py` | 由 parquet 机器生成结论（不手填） | `report.json`/md |

---

## 7. 门禁清单（每 Phase 结束全绿）

| 门禁 | 命令 | 覆盖 |
|---|---|---|
| 静态检查 | `make check` | ruff + mypy |
| 单元测试 | `uv run pytest -m "not slow"` | 契约/分配器/修正器 |
| 物理不变量 | `uv run pytest tests/test_physics_invariants.py` | SOC 边界/充放电互斥/能量平衡/逐组容量 |
| 泄漏回归 | `uv run pytest -m leakage` | 未来信息隔离 |
| 恢复一致性 | `uv run pytest -m resume` | 位级恢复(CPU) |
| 契约一致性 | `make contract` | 旧 checkpoint 不静默加载 |
| 计时预算 | `make probe` | 不超预算，超了先讨论 |

---

## 8. 数据任务（并行推进，不阻塞代码）

| 项 | 目标 | 现状 |
|---|---|---|
| 新加坡系统负荷 + USEP | ≥1 全年、匹配 | 仅 `USEP_May-2026.csv`（1 月） |
| 风光/温度时序 | 与地点年份匹配 | 缺失 |
| 任务轨迹 | 公开轨迹或统计特征生成 | 缺失 |

规则：先恢复/重新取得并核验来源，不假设历史文件仍在；明确时区/对齐/聚合；SGD/MWh→SGD/kWh；正式运行缺数据**直接报错**，合成数据只能显式启用；不重复日期充当独立样本。

---

## 9. 待人工拍板决策点

| # | 决策 | 建议 |
|---|---|---|
| D1 | 参考运行平台 | 正式跑分锁 CPU+确定性算法；MPS 仅加速探索，记录 backend/线程数 |
| D2 | `TaskAllocation` 维度 | `(n_tasks, n_groups)` 执行量矩阵 |
| D3 | 是否保留 legacy 23 维适配层 | 不保留（旧模型报错） |
| D4 | HARL | pin 成 `[marl]` 可选 extra，不 vendor |
| D5 | 可视化宿主 | Streamlit |
| D6 | 逐组分配公平性/优先级规则 | 明确书面定义，写入 `contracts/` |
| D7 | 归一化参考值来源 | 仅训练集或预定物理尺度 |

---

## 10. 里程碑时间线（对齐计划书四阶段）

| 阶段 | 主要交付 | 覆盖步骤 |
|---|---|---|
| 第 1 周 | 环境+数据核验、逐组执行探针、物理检查、**最小滚动求解耗时** | M0–M3、**M4.3** |
| 第 2 周 | 主算法闭环、统一对照、恢复与泄漏回归、本机实验预算 | M4(余)、M5、M6、M9.1 |
| 第 3 周 | 主实验、共享修正器与前瞻机制消融 | M9.2–M9.3 |
| 第 4 周 | 泛化和敏感性、失败分析、贡献证据表与论文细纲 | M9.4–M9.6、M8、M7.5 |

> 周期是目标窗口；预算不足时优先保留核心比较与证据质量，不以删失败场景或省略强基线换完成。**M4.3 探针数据出来前，不承诺任何本机工期。**

---

## 11. 下一步（待你点头后执行）

1. 补 M0 剩余：`Makefile`、`AGENTS.md`、`contracts/__init__.py`、`tests/` 骨架（纯新增，可整体委派 agent）。
2. 然后 M1.1 链路审计（只读）→ M2 契约 → M3.0 探针，逐卡推进。
