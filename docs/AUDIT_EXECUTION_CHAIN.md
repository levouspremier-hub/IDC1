# 现有执行链审计报告（只读，M1.1 交付物）

> 性质：**只读侦察**，不改任何代码。结论供 M2–M6 任务卡引用。
> 审计对象：`envs/idc_price_env.py`、`data_io/data_loader.py`、`idc_model/`、`safe_rl/` 现状。
> 依据：《审核修订版》§2–§7。

---

## 1. 动作 → 任务 → 功耗 链路（首版核心，需改造）

### 1.1 动作空间现状（23 维）

| 位置 | 现状 |
|---|---|
| `envs/idc_price_env.py:243-245` | `server_action_dim = model.N`(默认 20)；`extra_action_dim = 3`；`action_dim = 23` |
| `:559-563` | `action[0:20]`=逐组计算强度；`action[20]`=urgent 偏好；`action[21]`=continuity 偏好；`action[22]`=BESS(映射到 [-1,1]) |

**结论**：计划书要求收敛为 **20 计算 + 1 储能 = 21 维**，urgent/continuity 两维由“明确任务约束”取代。当前仍是 23 维。

### 1.2 逐组信息在何处丢失（关键）

```
:566  planned_task_loads = server_action * max_task_load_per_server   # 20 维向量
:567  planned_total_loads = clip(base_load + planned_task_loads, 0, 1)
:570  planned_capacity = float(np.sum(planned_task_loads * C_server)) # ← 20 维压成标量
:592  _execute_tasks_action_guided(available_capacity=planned_capacity, ...)  # 只传标量
```

**结论**：20 个组级建议在 `:570` 被 `sum(planned_task_loads * C_server)` 压成**带权标量**（不是纯 `np.sum`）。因 `C_server` 异构（`server_capacity_variation=0.25`），组间分布仍会影响加权总量——探针实测 `uniform_0.5` 与 `split_10×1.0` 的完成量/功耗不同（`scripts/probe_physics.py`）——但分配器只拿到一个总量，**没有任务×组分配**，且实际功耗由计划负载按单一比例回分（`:1313-1318`），与实际逐组执行脱钩。这正是 M3.1–M3.3 要拆的点。

### 1.3 分配器现状

`_execute_tasks_action_guided`（`:1124-1224`）：
- 接收**标量** `available_capacity` + urgent/continuity 两个偏好；
- 按 `_task_selection_score` 排序后**逐个任务**从标量余量里扣 `task.execute(remaining_capacity)`；
- **无逐组分配、无每任务最大执行速率显式约束**（`task.execute` 直接吃剩余容量，任务上限是否生效取决于 `Task.execute` 内部实现，需在 M3.2 单测确认）；
- 暂停/恢复/不可中断统计挂在 `_update_pause_events`（`:1226-1267`）。

**结论**：当前是“标量容量 + 优先级排序”的任务选择器，不是“任务×服务器组”分配器。计划书 §2 要求的“分配器输出任务—组执行量、逐组≤容量、逐任务≤上限”不成立。

### 1.4 实际功耗反推（计划书点名废除）

`_actual_loads_from_completed_work`（`:1286-1320`）：
```
:1313  usage_ratio = clip(completed_work / planned_capacity, 0, 1)
:1314  used_task_loads = planned_task_loads * usage_ratio       # 总量按计划负载比例回分
:1316  reserve_alpha = clip(planned_load_reserve_alpha, 0, 1)   # 默认 0.25 (:70)
:1318  actual_task_loads = used_task_loads + reserve_alpha * unused_planned_loads
```

**结论**：实际负载 = “总完成量按计划负载统一回分” + α×未用计划负载。计划书 §3 要求改为**逐组实际执行→实际功耗**，并把 α 仅作敏感性参考（先做 α=0 vs 原设定诊断对照，M3.4）。

> ⚠️ α 基准：环境默认 `planned_load_reserve_alpha=0.25`（`:70`），但 `configs/config_ultimate.py:64` 覆盖为 **0.40**。M3.4 诊断对照须明确以哪个为“原设定”，避免口径漂移。

### 1.5 逾期完成漏记（bug 已确认）

`_update_deadline_miss`（`:1269-1284`）：
```
:1278  if task.status in ["not_arrived", "finished", "failed"]:
:1279      continue        # ← finished 被跳过
```
**结论**：任务一旦 `finished` 即被跳过，因此“**截止后才完成**”的任务永远不被计入逾期。计划书 §2 要求区分“未到期积压 / 逾期积压 / 逾期完成”三分类（M3.5）。

---

## 2. 风光、储能、接入、跨日

### 2.1 风电是空壳（计划书要求接通）

- `:577` 读 `wt_now = self.wt_t[t]`，但 `wt_now` **只进入 info（`:896 "WT"`），不进入能量平衡**；
- 能量平衡只走 PV（`:689-699`）：`pv_available/used/curtail` 有统计；
- `data_io/data_loader.py:146` 注释明确 “WT remains an unused reserved interface”，默认全零；
- `renewable_share`（`:874-875`）只算 PV。

**结论**：风电**未接入能源平衡、无利用/弃电统计**。计划书 §4 要求“接通风电输入、预测、观测、能源平衡、利用与弃电统计”（M 模块对应 M3 改造）。

### 2.2 光伏与反送电

- PV 已接入（可用/利用/弃电三量齐，`:689-723`）；`allow_pv_export=True` 直接报错（`:298-299`），即**不反送电**——与计划书 §4 一致。

### 2.3 储能（基线良好）

- 充放电互斥由 `if bess_raw_action < 0 else` 分支保证（`:626-631`）；
- SOC 上下界硬裁剪（`:633-659`）、效率（`:663`）、退化成本（`:198-209`）均有。

**结论**：储能是独立可控资源、核算基本完整，是后续扩展（多约束、退化成本口径统一）的可用基线。

### 2.4 跨日状态：当前无（计划书要求多日连续）

- `reset()`（`:457-530`）每次**新建 24h episode**：`bess_soc` 重置到 `bess_soc_init`（`:465`）、`tasks` 重新生成（`:491-494`）、累计量清零（`:468-488`）。

**结论**：**当前是单日独立 episode，无跨日连续性**。计划书 §4 要求“连续多日运行、不每日清空任务/恢复电量、统一结算尾段”——这是 M3 之外的**结构性改动**（环境需支持多日滚动与尾段结算）。

### 2.5 接入容量

- 未见“接入物理上限 vs 削峰/计费阈值分离”的明确建模；峰值项在 reward（`:756 peak_load_norm`、`peak_power_ref_kW`）中体现。

**结论**：接入容量作为模型边界与场景维度尚未系统化（计划书 §4），需在 M3/M4 规划模型中显式化。

---

## 3. 数据与归一化

### 3.1 数据现状

| 外部序列 | 现状 |
|---|---|
| 价格/碳/温度 | 缺 CSV 时**回退到内置默认曲线**（`data_io/data_loader.py:150-164` → env `_create_price_curve` 等） |
| 光伏 | 缺 CSV 时用默认日光钟形曲线（`data_loader.py:71-83`） |
| 风电 | 默认全零（`default_zero=True`） |
| 实际数据文件 | `data/` 仅 `USEP_May-2026.csv`（1 个月）+ NEMS 处理结果；**无全年、无风光/温度/任务轨迹** |

**结论**：正式运行现在基本靠**合成/默认曲线**，与计划书 §5“真实数据驱动、缺数据直接报错、合成只能显式启用”相悖。数据获取（§8 数据任务）仍是硬依赖。

### 3.2 归一化参考值现状

- `*_ref` 均为**构造函数默认值**（`price_ref=1.50`、`lambda_ref=1000`、`cost_ref=30`、`carbon_ref=15`、`sla_ref=50`、`peak_power_ref_kW=10` 等，`:45-98`）；
- `reset()` **不重算**这些 ref（好事）；但它们是**硬编码经验值，非“训练数据或预定物理尺度”导出**；
- `pv_ref_kw` 例外，由装机/峰值导出（`data_loader.py:136`）。

**结论**：ref 已冻结（不会逐日重算），但**来源无据**。计划书 §5 要求 ref 仅由训练数据或物理尺度确定、对全部方法冻结——M6.2 `freeze_refs` 需把 ref 改为有据来源并带哈希冻结。

### 3.3 关键配置核对（`configs/config_ultimate.py`）

| 项 | 实测 | 计划书对应 |
|---|---|---|
| 规模 | `num_server_groups=20`、25 MW 正式定义（`:70`）、`server_group_size=1841` | 25 MW / 20 组 ✅ |
| 储能 | `bess_capacity_kWh=100` × `bess_scale_factor=100` = **10 MWh**；功率 20 kW×100 = **2 MW** | “2 MW／10 MWh 历史参考” ✅ |
| 数据接入 | `DATA_CONFIG` 五个 csv path **全 None**（`:81-95`），价格/碳/温度/风光全走默认/合成 | §5 真实数据驱动 **不符**（N6） |
| 参考值 | `ENV_CONFIG` 硬编码 `price_ref/lambda_ref/queue_ref/cost_ref/carbon_ref/...`（`:35-64`） | §5 ref 须有据（N5） |
| 奖励 | `REWARD_CONFIG` 为 **21 项加权和**（`:98-123`，成本/碳/SLA/队列/逾期/峰值/暂停/退化…） | §6 “奖励惩罚式”→ 拆约束价值，减少重复项（M5） |
| 训练 | `PPO_CONFIG` n_steps=768、batch=256、lr=3e-4、net [256,256]、`device="auto"` | 本机吞吐需 M4.3 实测 |

> `GRID_CONFIG` 里 `use_mef=True`、`enable_grid_obs=True` 属于**多智能体/网格耦合**路径（`env_wrappers/`、`marl/`），不是首版单智能体主链；计划书 §8 已把 ACOPF/MEF 后置为离线验证，主训练不叠加。

---

## 4. 契约与依赖现状

| 项 | 现状 | 计划书要求 |
|---|---|---|
| 6 个数据契约 | **不存在**（无 `contracts/`） | M2 新建 |
| 契约版本化 | 无 | 旧模型不得静默加载（M2.4） |
| 修正器 `a_exec=S(s,a_raw)` | **不存在**（`safe_rl/safe_wrapper.py:105` 是奖励惩罚器，非修正器） | M4 新建 |
| HARL | 计划文档记载约 25 处 `import harl`，无 pin（待 M1.1 复核） | 可选 extra + pin sha |
| 打包/锁 | ✅ 已由 uv 环境解决（`pyproject.toml`+`uv.lock`） | 已完成 |

---

## 5. 阻塞项汇总（编号沿用 `AGENT_EXECUTION_PLAN.md` §1.2，附本次核实）

| # | 问题 | 本次核实证据 | 影响 |
|---|---|---|---|
| B1 | 旧 `environment.yml` Windows+CUDA | **已解决**（uv 环境 + HiGHS 已验证） | 无 |
| B2 | 无打包/锁 | **已解决**（pyproject/uv.lock） | 无 |
| B3 | `.gitignore` 白名单 | **已补** 3+2 个新文件 | 无 |
| B4 | HARL 裸依赖 | 未复核，待 M1.1 扫描 | 复现风险 |
| B5 | 无 `AGENTS.md`/`Makefile` | 仍未建（M0.2/M0.1） | 门禁缺失 |
| B6 | 数据仅 1 月 USEP | 已核实 `data/` ≈312 KB | 主实验阻塞 |
| B7 | MPS 非确定性 | 参考运行须锁 CPU（D1） | 复现审计 |
| **N1** | 20 维→带权标量压缩（C_server 异构）+ 无逐组分配 | `:570` / `:1286` | 功耗与实际执行脱钩 |
| **N2** | 逾期完成漏记 | `:1278` | 业务指标失真 |
| **N3** | 风电空壳 | `:577/:896` 不入能量平衡 | 消纳/弃电无法评估 |
| **N4** | 无跨日连续性 | `reset():465/491` | 多日统计不成立 |
| **N5** | 归一化 ref 无据 | 硬编码默认值 | 比较公平性风险 |

---

## 6. 审计结论（下一步建议，仍不改代码）

1. **M0**：先补 `Makefile`、`AGENTS.md`（八条红线）、`contracts/__init__.py`、`tests/` 骨架 —— 纯新增，可整体委派。
2. **M3**：按行号拆卡，逐卡改 `:570`(N1)、`:1278`(N2)、`:577` 风电链路(N3)、`:465/491` 跨日(N4)。
3. **M4.3 探针**：第 1 周硬要求，先于任何工期承诺。
4. **数据（N6）**：并行推进全年匹配价格/负荷 + 风光/温度/任务轨迹获取，是主实验的前置。

> 本报告为只读结论；所有行号以本文件生成时刻的仓库为准，M3/M4 改卡前请再 `grep` 复核一次行号。
