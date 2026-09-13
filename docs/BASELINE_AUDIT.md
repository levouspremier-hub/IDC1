# 基线审计（M0.1）

> **性质：基线历史，非当前实现。** 本文件描述改造**之前**的代码状态（基线 commit `787a3c8`）。
> 其中涉及的旧 α 预留损耗（`planned_load_reserve_alpha`）已在 M3.3 退出正式链、在 M3.4a 被彻底删除；
> 现行实现中已无该参数、属性与方法。引用本文时请以改造后的代码与任务卡为准。
>
> 角色：Agent。本文件为只读审计，不修改任何算法、配置或数据。
> 生成日期：2026-09-12。基线 commit：`787a3c8`。分支：`p0-bootstrap`（自 `paper-baseline` 切出）。
> 依据：`docs/IMPLEMENTATION_PLAN.md` §2、`docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md` §5 M0.1。

## 1. 硬件

| 项 | 值 |
|---|---|
| 芯片 | Apple M5（arm64） |
| CPU 核 | 10 |
| 内存 | 16 GB（17179869184 B） |
| 系统 | macOS 26.6.2（Darwin 25.6.0） |
| 后端 | CPU（正式复核平台）；MPS 仅探索，须记录 backend 与非确定性风险（协议 §4.6） |

## 2. Python 环境

| 项 | 值 |
|---|---|
| 系统 Python | 3.9.6（`/usr/bin/python3`） |
| 项目虚拟环境 | `.venv/`，Python 3.12.14 |
| 锁定环境 | 无 `pyproject.toml` / `.python-version` / `uv.lock`（M0.2 待建） |
| 旧环境文件 | `environment.yml`（历史，M0.2 保留不改写） |

## 3. 当前 commit 与分支

- 基线 SHA：`787a3c8 docs: add executable Claude Code protocol`
- 计划声明的基线：`a90a06f`（`IMPLEMENTATION_PLAN.md` §0）
- 默认分支：`paper-baseline`（本仓库 main）
- M0.1 分支：`p0-bootstrap`

## 4. 入口文件（顶层 shim，红线保护，不可修改）

| 顶层文件 | 指向 |
|---|---|
| `train_ppo_ultimate.py` | `train.train_ppo_ultimate.main` |
| `eval_base.py` | `eval.eval_base.main` |
| `ga_base.py` | `algorithms.baselines.ga_base.main` |
| `pso_base.py` | `algorithms.baselines.pso_base.main` |
| `config_ultimate.py` | `configs.config_ultimate.*` |
| `IDCPriceEnv20D_ultimate.py` | `envs.idc_price_env.IDCPriceEnv20D` |
| `task.py` | `idc_model.task.Task` |
| `task_model.py` | `idc_model.task_model.IDCEnergyTaskModel` |
| `power_model.py` | `idc_model.power_model.IDCPowerModel` |
| `data_loader.py` | `data_io.data_loader.*` |

## 5. 保护目录（红线，不可修改）

- `marl/`（含 `marl/checkpointing/`、`marl/tests/`）
- `grid_model/` 主逻辑
- `legacy/`
- 顶层兼容 shim（§4 列表）
- `envs/idc_price_env.py::step()`：修改前须先提交该卡的失败测试（协议 §1.1.7）

## 6. 数据状态

### 6.1 已跟踪数据（`git ls-files data/`）

- `data/grid_node_sensitivity/`：IEEE-14 静态节点灵敏度（`ieee14_node_sensitivity_static.csv` + `ieee14_node_sensitivity_detail.json`）
- `data/grid_scenarios/nems_singapore/`：NEMS 24h 负荷标度、逐小时剖面，及 `raw/USEP_May-2026.csv`（仅 2026-05 一个月的统一新加坡能源价格）

### 6.2 缺口（相对 M1.2 目标）

- 无连续一年（价格 + 系统负荷）对齐时间轴；可再生数据（温度/辐照/PV/风速/风电）未下载。
- `data/raw/` 不存在；`data/manifest/` 不存在。
- 环境默认合成数据兜底；风电 `wt_t` 默认全零（`envs/idc_price_env.py:300`）。

### 6.3 未跟踪 / 忽略（`git status --ignored`，顶层）

- 工具与缓存：`.venv/`、`.claude/`、`.mypy_cache/`、`.pytest_cache/`、`.ruff_cache/`、`.DS_Store`
- 空目录残留（revert 后仅剩 `__pycache__`）：`contracts/`、`planning/`、`tests/`
- 生成的运行产物：`runs/`（含 `runs/smoke/`）
- 未入 git 的文档/侧项：`docs/AGENT_EXECUTION_PLAN.md`、`carbon-lit-search/`、`litsearch/`

## 7. 关键代码事实（行号定位）

### 7.1 23 维动作空间

- `envs/idc_price_env.py:242-246`：`action_dim = server_action_dim(N) + extra_action_dim(3) = 20 + 3 = 23`；`Box` 上界 `ones(23)`。
- `envs/idc_price_env.py:551-556`：`step()` 校验 `action.shape[0] != action_dim` 即抛 `ValueError`。
- 动作语义（`envs/idc_price_env.py:559-562`）：`[0:20]` 服务器组强度、`[20]` urgent、`[21]` continuity、`[22]` BESS。

### 7.2 标量容量（逐组被压成标量）

- `envs/idc_price_env.py:570`：`planned_capacity = float(np.sum(planned_task_loads * self.model.C_server))` —— 逐组计划负载被求和成单一标量。
- `envs/idc_price_env.py:592-593`：以标量 `available_capacity=planned_capacity` 传入 `_execute_tasks_action_guided`。
- 对应 M3.1 目标：分配器应收到长度 20 的 `planned_capacity_vec`，不再只有标量输入。

### 7.3 每日 reset（任务与 SOC 均按 episode 重置）

- `envs/idc_price_env.py:457-530`：`reset()` 开始新 24h episode。
- `envs/idc_price_env.py:465-466`：`bess_soc` 重置为 `bess_soc_init`。
- `envs/idc_price_env.py:490-494`：`self.tasks = self.model.create_demo_tasks(...)` 每 episode 重新生成任务，并插入初始积压任务。
- 对应 M3.6 目标：跨日不重置 SOC、不新建任务队列、统一尾段结算。

### 7.4 旧 α 预留损耗

- `envs/idc_price_env.py:1286-1320`：`_actual_loads_from_completed_work` 以 `planned_load_reserve_alpha` 作为未用计划负载的预留损耗（`actual = used + α·unused`）。
- 对应 M3.4 目标：α 隔离为诊断，不进入正式主链。

### 7.5 风电未接入能量平衡

- `envs/idc_price_env.py:576-577`：读取 `pv_now` / `wt_now`。
- `envs/idc_price_env.py:689-723`：仅 PV 参与本地母线净负荷抵消（无反送电）；`wt_now` 仅写入 info（`:896`），未进入能量平衡，无 `wind_used/curtailed` 统计。
- 对应 M3.5 目标：风/PV/储能各工况守恒，风进入无反送电平衡。

## 8. 与首版目标的差距（指向下游卡）

| 现状 | 差距 | 下游卡 |
|---|---|---|
| 23 维动作 | 需 21 维（20 compute + 1 signed storage） | M3.9 |
| 标量容量 | 需逐组 `planned_capacity_vec`（长度 20） | M3.1 / M3.2 |
| 比例回分功耗 + α | 需由 A 矩阵导出逐组实际功耗 | M3.3 / M3.4 |
| 风电只记录不入平衡 | 需接入无反送电平衡 | M3.5 |
| 每日 reset 任务与 SOC | 需跨日队列 + 尾段结算 | M3.6 |
| 无版本契约 / checkpoint | 需 frozen 契约 + 版本化新主链 | M2.1–M2.3 |
| 无正式运行产物库 | 需 config/parquet/report/figures/manifest | M7.1 |
| 数据仅 1 个月价格 + NEMS 负荷 | 需连续一年对齐 + 可再生数据 | M1.2 |

## 9. .gitignore 说明（基线事实）

`.gitignore` 采用「`*` 默认全忽略 + `!` 白名单」策略。`docs/` 下仅白名单了 `dev_reports/`、`IMPLEMENTATION_PLAN.md`、`VSCODE_CLAUDE_EXECUTION_PROTOCOL.md`、两个 MULTI_AGENT 文档。新增 docs（本卡 `BASELINE_AUDIT.md`、`task_cards/`）默认被忽略，需在白名单补行才能 `git add`（本卡已补，见任务卡「允许范围外修改」）。
