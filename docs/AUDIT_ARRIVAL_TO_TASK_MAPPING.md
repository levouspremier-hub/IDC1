# M1.3g-e-a 审计：arrival-to-task 映射契约

> 任务卡：`docs/task_cards/M1.3g.md` §aa–ab
> 分支：`p4-safeppo-m51a-rollout-contract-m12-integration`
> 开始 SHA：`2c5e215`
> 性质：**只读审计 + 契约设计**。本文件**不**实现 mapper、**不**接线 env/train、
> **不**改变任何 readiness。

**审计问题（唯一）**：Singapore 2024 外生驱动中的每半小时 **aggregate arrival
workload**，如何在不改变物理容量、SOC、任务约束与信息边界的前提下，
**确定性、可复现、因果**地转换为 `IDCPriceEnv20D` 使用的离散 `Task` stream？

**结论摘要（详见 §I）**：

① **映射本身是 1:1 恒等的**：mapper **必须**把原始 aggregate arrival
**原样**分割成任务，**不得**缩放、不得改变场景强度（§G.3 不可变式 1）。
② **在推荐的 rate-based 物理语义下**（§C.0），正式 aggregate arrival
`1000.172 work / 0.5h` 对应服务能力 `402.521 work/hour × 0.5h
= 201.261 work / 0.5h`，**arrival/service ≈ 4.970**（§D.7）——
即当前 arrival 数据是一个**过载（重负载）场景**。这**如实登记**，
**不**通过缩放 arrival 或放大容量来「修好」它。
③ **步长**：env 默认 1 h 与正式数据 0.5 h 不一致；本审计给出**唯一一致**的
半小时物理语义（§C），并说明旧的 work/step 方案是**非物理备选**。
④ 存在三处 future-information 暴露面（§E）。

**推荐 M-1**：原始 aggregate **1:1 守恒** + 确定性分割（允许结果是过载场景）；
**备选 M-2**：暂停实现并**回到 M1.3f** 重新定义、版本化并人工批准有**物理依据**
的 arrival intensity（**不是**在 mapper 里乘一个系数）。

**⛔ 在「接受当前 arrival 作为正式重负载场景」vs「返回 M1.3f 重定强度」
这一决定完成之前，不得开始 g-e-b**（§I.2 **D-INTENSITY**）。

---

## A. 单位账本

> **禁止只写 “work units”。** 下表对每一项明确：名称、单位、时间尺度、来源、
> 使用位置，并指出**当前代码自相矛盾之处**。

### A.1 `exogenous_drivers_v2.parquet` 的 `arrival`

| 项 | 值 |
|---|---|
| 单位 | **work-units / 半小时槽**（每 30 分钟一个标量） |
| 时间尺度 | **30 min**（17,568 行 = 366 天 × 48 槽） |
| dtype | `int64`（Poisson 实现值） |
| 来源 | `scenario/exogenous_drivers.py:574-605` `generate_arrival()`；B5-ARRIVAL 批准 |
| 生成 | `Poisson(rate_template[slot] × 1000)`，seed `20240916`（`:94-95`） |
| 物化 | `data/manifest/singapore_2024_exogenous_v2.json` → `columns.arrival` |
| 实测 | min **803** / max **1230** / mean **1000.1721**（§D.1） |

### A.2 formal scenario 的 **arrival forecast**

| 项 | 值 |
|---|---|
| 单位 | **work-units / 槽**，但取的是**期望值**而非抽样值 |
| 定义 | `scenario/formal_scenario.py:336-353` `arrival_forecast()` =
  `rate_template[slot] × 1000`，**D3**：不调用 `generate_arrival`、不用 Poisson |
| 实测 | min **896.23** / max **1115.77** / mean **1000.0**（§D.2） |
| 与 A.1 的关系 | 同一 template 的**期望**；A.1 是同一 template 的**一次实现** |

> **这是本审计第一个必须澄清的单位问题**：`local_pv_kw` / `wind_generation_kw` /
> `carbon_intensity` 在 v2 驱动表里是**确定性物理量**，而 `arrival` 是
> **随机过程的实现值**。三者的「每行一个标量」形状相同，但语义不同类。

### A.3 Poisson 实现值 vs `lambda(slot) × 1000` 期望

| 项 | Poisson 实现值 | 期望值 |
|---|---|---|
| 位置 | `exogenous_drivers_v2.parquet` 的 `arrival` 列 | `rate_template × 1000` |
| 用途 | 模拟场景的**实际** arrival（历史/评估） | **formal forecast**（D3） |
| 可复现性 | 需 seed + 调用顺序 | 确定性 |
| 方差 | `Var = λ`（λ≈1000 → σ≈31.6，实测 803–1230） | 0 |

**红线**：formal forecast **不得**使用 A.1 的实现值（那是「已抽样的未来」）。

### A.4 `Task.workload`

| 项 | 值 |
|---|---|
| 定义 | `idc_model/task_model.py:163` `workload = float(np.sum(load_profile) * self._task_workload_capacity_ref())` |
| 参考量 | `:63-66` `_task_workload_capacity_ref() = C_IDC_base × task_workload_scale` |
| **当前实现** | `workload = Σ(load_profile) × C_IDC_base × scale`，**没有乘 `delta_t_hours`** |
| **推荐语义（§C.2）** | **物理 work 数量** = `Σ(load_profile × C_IDC × delta_t_hours)`，单位 **work** |
| **时间语义** | `load_profile` 的第 i 项是**第 i 个仿真步内的负载率（速率）**；步内工作量 = `负载率 × C_IDC × delta_t_hours` |
| 实测参考量 | `C_IDC_base = 469.556874725279`，`task_workload_scale = 1.0`（§D.3） |

> **⚠️ 矛盾登记 ①（本卡 P1 的根源之一）**：`load_profile[i]` 是**无量纲速率**，
> 但当前 `workload = Σ load_profile × ref` **没有乘 `delta_t_hours`**。
> 因此在 `delta_t_hours = 0.5` 下：
>
> - 作为「**每步工作量计数**」它**不变**，但该计数**不再等于物理 work**；
> - 作为「物理 work」它**必须**乘 `delta_t_hours`，否则每步工作量被**高估 2 倍**。
>
> 二者**当前不可区分**，因为 env 的 `delta_t_hours` 默认恰为 `1.0`。
> **推荐**采用 §C.2：`workload` 一律是**物理 work 数量**，
> 且 `max_rate` 与 `capacity_per_step` 同处 **work/step** 单位。

### A.5 `Task.duration`

| 项 | 值 |
|---|---|
| 单位 | **整数步数**（`int`，`:35` `self.duration = int(self.duration)`） |
| 取值范围 | A `(1,2)`、B `(2,5)`、C `(4,8)`、D `(1,4)`（`task_model.py:81/91/101/111`） |
| 与 `load_profile` 的关系 | `len(load_profile) == duration`（`:162`） |
| 期望 | **2.7 步**（§D.3） |
| **时间语义** | 「步」= `delta_t_hours`。**1 h step → 2.7 h；0.5 h step → 1.35 h** |
| **推荐（§C.2）** | profile 的业务时长**按物理小时声明**；`duration_steps = round_half_up(hours / delta_t_hours)`，**最小 1 步** |

### A.6 `Task.deadline` / `latest_finish_time`

| 项 | 值 |
|---|---|
| `deadline` 单位 | **相对延迟步数**：`latest_finish_time = arrival_time + deadline`（`task.py:44-46`） |
| `latest_finish_time` 单位 | **绝对步索引**（同一时间轴上） |
| 取值范围 | A `(2,4)`、B `(8,16)`、C `(14,24)`、D `(5,10)` |
| 生成约束 | `real_deadline_min = max(deadline_min, duration)`（`task_model.py:167`） |
| 使用位置 | env `:1256` `deadline_left = latest_finish_time - current_time`；`:1319` 传给分配器 **绝对**值；`planning/snapshot_adapter.py:231` `deadline=int(task.latest_finish_time)` **绝对**值 |
| **时间语义** | 同样是**步数**；0.5 h step 下同一数值代表**一半的物理时间** |
| **推荐（§C.2）** | 业务 deadline **按物理小时声明**；`deadline_steps = round_half_up(hours / delta_t_hours)`，且**不得小于** `duration_steps` |

> **⚠️ 矛盾登记 ②**：`idc_model/allocation.py:24-27` 的调度口径写
> 「按 (priority 降序, **deadline 升序**, …) 排序」，读的是 `tasks[i]["deadline"]`
> **原值**；而给它喂数据的 `snapshot_adapter.py:231` 已经把 `deadline` 换成
> **绝对** `latest_finish_time`。两者在「绝对 vs 相对」上**命名相同、语义不同**。
> 当前恰好可用（同一批任务里相对序与绝对序一致），但**契约上必须写清楚**。

### A.7 `load_profile`

| 项 | 值 |
|---|---|
| 单位 | **无量纲负载率**（占单组算力的比例），元素 ∈ `[0.08, 0.45]` |
| 形状 | `(duration,)` |
| 来源 | `task_model.py:162` `self.task_rng.uniform(load_min, load_max, size=duration)` |
| 使用位置 | 仅用于算 `workload`（`:163`）与 `avg_load`（`task.py:52-57`）；**不直接进功耗** |
| 注意 | env 的执行链用 `A[i,g]` 完成量反推负载（`:738-739` `_loads_from_group_completion`），**不读** `load_profile` |

### A.8 `C_IDC`、`C_server`、`planned_capacity_vec`

| 项 | 值 |
|---|---|
| `C_IDC` | `idc_model/power_model.py:90` `self.C_IDC_base = sum(self.single_server_C_server)`；`:91` `C_server = single_server_C_server × server_group_size` |
| 实测 | `C_IDC_base = 469.556874725279`；`Σ C_server = 469.556874725279`（scale=1、group model 关闭） |
| `planned_capacity_vec`（**当前实现**） | `envs/idc_price_env.py:622` `planned_task_loads × C_server`（逐组），**不含** `delta_t_hours` |
| **推荐语义（§C.2）** | `planned_capacity_rate`（**work/hour**）→ `capacity_per_step = rate × delta_t_hours`（**work/step**） |
| 全动作上限 | `max_task_load_per_server(0.80) × Σ C_server = 402.521` **work/hour**（§D.7 计算 1） |

> **⚠️ 矛盾登记 ③**：`planned_capacity_vec` 与 `Task.workload` 都是「每步」量，
> 二者**自洽**；但两者与**物理功率/能量**之间的关系需要 `delta_t_hours`
> （`kWh = kW × delta_t_hours`，见 `:234` `× self.delta_t_hours`）。

### A.9 `completed_work`、`remaining_work`、`Q_t`

| 项 | 值 |
|---|---|
| `remaining_work` | `task.py:41` 初始 = `workload`；`:85` 每步减去 `actual_work` |
| `completed_work` | env `:713` 由 `_execute_tasks_action_guided` 返回，**每步**完成量 |
| `Q_t` | env `:1558-1563` `_compute_backlog_work()`：已到达（`status != not_arrived`）且未完成任务的 `remaining_work` 之和 |
| 单位 | 与 `Task.workload` 相同 = **capacity-ref × 步** |

### A.10 `lambda_ref`、`queue_ref`、`queue_capacity_ref`

| 参考值 | `refs_v3.json`（D2 裁决：formal 用 refs 侧） | env 默认 | 单位（refs 声明） |
|---|---|---|---|
| `lambda_ref` | **2000.0** | **1000.0** | `work-units/hour` |
| `queue_ref` | **6000.0** | 1500.0 | `work-units` |
| `queue_capacity_ref` | **6000.0** | 1500.0 | `work-units` |
| `cost_ref` | **60.0** | 30.0 | `SGD` |

env 侧统一乘 `task_workload_scale`（`:198-201`）。

> **⚠️ 单位冲突（本审计最关键的归一化问题）**：`refs_v3` 声明
> `lambda_ref = 2000 **work-units/hour**`，而正式 aggregate arrival 的均值是
> **1000.17 work-units / 半小时**（§D.1）。若把 30 分钟量直接除以
> 「每小时」参考值，归一化结果会**系统性减半**（≈0.5 而非 ≈1.0）。
> M1.3f-c 的 `scale_basis` 写的正是
> `lambda_ref = 2000 work-units/hour × 0.5 hour = 1000`（`exogenous_drivers.py:99`），
> 即 **1000 就是「每半小时」的参考值**。因此
> **`lambda_ref` 的时间基准必须与映射后的步长一致**，否则观测归一化错 2 倍。

### A.11 `delta_t_hours`

| 项 | 值 |
|---|---|
| 定义 | `envs/idc_price_env.py:81` `delta_t_hours: float = 1.0`；`:207` `self.delta_t_hours = float(delta_t_hours)` |
| 用途 | **仅**用于 kW ↔ kWh 换算（`:234`、`:777-778`、`:653-656`） |
| **不**影响 | `Task.workload` / `duration` / `deadline` / `planned_capacity_vec`（这些全是「每步」量） |

### A.12 每步功率 kW、能量 kWh 与 workload 的关系

```text
A[i,g] 完成量 (work-units)  →  _loads_from_group_completion  →  负载率 L[g] (无量纲)
L[g]                        →  calc_pue_and_total_power      →  P_IDC (kW)
P_IDC (kW) × delta_t_hours  →  本步电量 (kWh)                  ← env :234 / :777-778
```

**结论**：`workload` 与 **kW** 之间**没有直接换算**；中间必须经过
`load_profile` 的负载率语义 + `delta_t_hours`。任何「1 work-unit = x kW」
的说法都是错的。

---

## B. 完整调用链

### B.1 当前（demo）链路

```text
[构造] IDCPriceEnv20D.__init__                      envs/idc_price_env.py:67
  └─ IDCEnergyTaskModel.__init__                    idc_model/task_model.py:18
       ├─ task_rng = default_rng(task_seed)          idc_model/power_model.py:60
       └─ _init_task_profiles()                      task_model.py:68   ← 四类 profile + 概率

[reset] env.reset()                                  envs/idc_price_env.py:502
  ├─ model.create_demo_tasks(num_tasks=12, h=24)     :547 → task_model.py:254
  │    └─ create_random_tasks(...)                   task_model.py:207
  │         ├─ arrival_time ~ U[1, horizon-6]         :241   ← **随机**，非因果
  │         ├─ profile_key ~ Categorical(prob)        :240
  │         └─ create_task → _sample_task_parameters  :179 / :144
  │              └─ duration/load_profile/workload/deadline/priority 全部 RNG 抽样
  ├─ create_initial_backlog_task(initial_Q)          :553 → task_model.py:272
  ├─ _initialize_task_runtime_state()                :556
  ├─ build_task_arrival_curve(tasks, horizon)        :557 → task_model.py:294
  │    └─ lambda_t[arrival_time] += task.workload    :307   ← **TRUTH**
  ├─ generate_task_arrival_forecast(...,"noisy",0.2) :562 → task_forecast.py:41
  │    └─ truth × (1 + N(0,0.2))                     :64-65 ← **读全段 truth 的 oracle 损坏**
  ├─ _activate_arrivals(0)                           :570
  └─ Q_t = _compute_backlog_work()                   :571

[step t] env.step(action)                            envs/idc_price_env.py:588
  ├─ planned_capacity_vec = action × max_load × C_server        :622
  ├─ lambda_now = true_task_arrival_profile[t]                 :631   ← **TRUTH**
  ├─ _activate_arrivals(t)                                     :634
  ├─ 物理投影（access_limit / 可再生 / BESS）→ scale           :637-711
  ├─ _execute_tasks_action_guided(scaled_vec, …)               :715-727
  │    └─（分配器：allocate_tasks / A[i,g]）
  ├─ _loads_from_group_completion → 负载率 → calc_pue_and_total_power  :738-744
  ├─ 能量 kWh = P × delta_t_hours                               :234/:777-778
  ├─ reward / deadline_miss / terminal settlement               :819/:1434+
  └─ obs = _get_obs()                                           :1964
       ├─ lambda_norm = true_task_arrival_profile[t] / lambda_ref :1975  ← **TRUTH**
       ├─ _get_task_pool_features(t)  → 只含 arrival_time<=t        :1739-1742 ✓
       └─ _get_forecast_features()    → 只含 [t, t+cutoff)          :1900-1962 ✓
```

### B.2 正式链路（M1.3g 目标）

```text
exogenous_drivers_v2.parquet  arrival (int64, 1000.17/半小时, 实现值)
   │
   └─► formal arrival forecast  = rate_template × 1000  （期望值，D3）
         scenario/formal_scenario.py:336-353
            │
            └─► 【缺失：arrival-to-task mapper】  ← **本审计设计的对象**
                  │
                  └─► Task stream（离散、带 workload/duration/deadline/profile）
                        │
                        └─► env 注入（g-e-c 设计）→ 与 B.1 相同的下游链
```

### B.3 语义分叉点（**现在**就存在）

| # | 分叉 | 位置 |
|---|---|---|
| 1 | demo task 的 `arrival_time` 是 **U[1, horizon−6]** 随机数，与任何 `arrival` 数据**无关** | `task_model.py:241` |
| 2 | `task_arrival_forecast` 由 **truth × 噪声**生成（`"noisy"` 默认，误差 0.20），**不是**因果预测 | `task_forecast.py:64-65`、env `:135` |
| 3 | `lambda_t` **别名**为 `true_task_arrival_profile`（`:343`），观测里直接用 truth 的当步值（`:1975`） | env `:343`、`:1975` |
| 4 | `info["true_task_arrival_profile"]` 暴露**整段** truth | env `:496` |

**结论**：当前 env 的 arrival 语义与正式链**完全不同来源**；
mapper 不能「接上」它，必须**替换**它（在 g-e-c 中，且不得保留 fallback）。

---

## C. 半小时步长影响

### C.0 三个**必须分开**的概念（本卡 P1 修正的核心）

> 修复前的 §G.1 让 `arrival_scale` 同时承担了三件事，这是错的。三者**必须分开**：

| # | 概念 | 定义 | 影响 arrival/service 比？ | 允许出现在哪里 |
|---|---|---|---|---|
| 1 | **单位换算** | 对 arrival、capacity、queue refs 等**同维度量**做**一致**换算（例如 work/hour ↔ work/step） | **否**（分子分母同比例） | mapper / env 注入层 |
| 2 | **归一化** | 只改**观测数值尺度**（如 `λ / lambda_ref`） | **否** | 观测层 |
| 3 | **场景强度修改** | 改 arrival / service 的**比值**本身 | **是** | **只能**在上游场景定义（M1.3f + manifest + refs + 人工批准 + 版本化） |

**红线**：第 3 类**不得**藏在 mapper 里。「为了让环境可行而缩小 arrival」等于
**放松约束 / 改变任务负载来制造可行**（AGENTS §1.1、IMPLEMENTATION_PLAN §1.1.1）。
`arrival_scale` **不是**单位换算量，**不得**作为 mapper 参数或 train-only 校准参数。

### C.1 `delta_t_hours=1.0` vs 数据 0.5 h

`envs/idc_price_env.py:81` 默认 `delta_t_hours = 1.0`；正式 canonical 与 v2 驱动表
是 **30 min**（17,568 行，`FREQUENCY="30min"`）。**D4 已裁决**：formal 路径**显式**
使用 `delta_t_hours = 0.5`；旧路径默认 `1.0` **不改**。

### C.2 **推荐默认：唯一一致的 rate-based 半小时物理语义**

> 这是本审计**推荐的默认**，取代修复前的「保持 `planned_capacity_vec` 为
> work/step、不乘 `delta_t_hours`」。

| 量 | 语义 | 单位 | 与 `delta_t_hours` 的关系 |
|---|---|---|---|
| `C_IDC` / `C_server` | **服务速率** | **work-units / hour** | 与步长**无关** |
| `planned_capacity_rate` | `action × max_task_load × C_server` | **work / hour** | 与步长**无关** |
| `capacity_per_step` | `planned_capacity_rate × delta_t_hours` | **work / step** | **乘 `delta_t_hours`** |
| `Task.workload` | `Σ(load_profile × C_IDC × delta_t_hours)` | **work**（物理工作数量） | **乘 `delta_t_hours`** |
| `load_profile` | **速率 / 利用率**语义（该步内占算力的比例） | 无量纲 | 与步长**无关** |
| `duration` / `deadline`（业务参数） | 按**物理小时**声明 | **hour** | 与步长**无关** |
| `duration_steps` / `deadline_steps` | `physical_hours / delta_t_hours` | **step** | **除 `delta_t_hours`** |
| per-task `max_rate`（env / allocation） | **work / step** | **work/step** | 与 `capacity_per_step` **同单位** |
| `lambda_ref` | **work-units / hour**（refs 声明） | work/hour | — |
| `lambda_ref_per_step` | `lambda_ref_work_per_hour × delta_t_hours` | work/step | **乘 `delta_t_hours`** |
| `queue_ref` / `queue_capacity_ref` | **存量** work-units | work | **不**因步长直接缩放 |
| `price` / `carbon` / `cost` | 由 `kW × delta_t_hours` 得 kWh 再计价 | — | **乘 `delta_t_hours`** |

**非整数 `duration_steps` 的确定性取整规则（必须冻结并登记）**：
`duration_steps = round_half_up(physical_hours / delta_t_hours)`，**最小 1 步**；
`deadline_steps` 同样取整，且**不得小于** `duration_steps`
（与 `task_model.py:167` 的既有约束一致）。

### C.3 为什么该语义在 1 h → 0.5 h 下保持自洽

| # | 必须保持的不变量 | 为什么成立 |
|---|---|---|
| 1 | **每小时服务能力不变** | `rate` 与步长无关；`rate × delta_t_hours` 只是把速率**折算到该步** |
| 2 | **同一物理持续时间不变** | `duration_steps = hours / delta_t_hours` ⇒ `steps × delta_t_hours == hours` |
| 3 | **同一负载曲线产生的总工作量不变** | `Σ(load_profile) × C_IDC × delta_t_hours`，步数翻倍与单步时长减半**互相抵消**（§C.4 微例） |
| 4 | **task `max_rate` 与 per-step capacity 同单位** | 二者都是 **work/step** |
| 5 | **arrival/service 比不因离散步长改变** | arrival 与 capacity 都换算到**同一** work/step 基准 |

### C.4 数值微例：同一物理 profile 在 1 h / 0.5 h 下的总工作量

```text
设定：物理持续时间 1 小时，平均负载率 0.2，C_IDC = 100 work/hour

1h step : duration_steps = 1/1.0 = 1
          workload = 1 × 0.2 × 100 × 1.0 h = 20 work        ← 1 步 × 20 work/step

0.5h step: duration_steps = 1/0.5 = 2
          workload = 2 × 0.2 × 100 × 0.5 h = 20 work        ← 2 步 × 10 work/step

⇒ 总工作量 20 work 不变；每步工作量由 20 降到 10（因为每步只占 0.5 h）
⇒ task 的 work/step 速率 = workload / duration_steps = 10 work/step（两种步长一致）
```

> **修复前的问题**：旧稿说「duration 翻倍、workload **数值不变**、capacity 每步不变」，
> 那会让 `workload / duration_steps` **减半**，即改变任务的**最大速率语义**。
> 采用 §C.2 后不再发生。

### C.5 `duration` / `deadline` 从 1 h 切到 0.5 h 的换算

**必须**换算，否则任务物理时长减半：

| 项 | 业务声明（物理小时） | 1 h step | 0.5 h step（正确） | 0.5 h step（不换算，❌） |
|---|---|---|---|---|
| `duration` | 2 h | 2 步 = 2 h | **4 步 = 2 h** | 2 步 = **1 h** |
| `deadline`（相对） | 8 h | 8 步 = 8 h | **16 步 = 8 h** | 8 步 = **4 h** |

**profile 常量按小时声明**：`task_model.py:81/91/101/111` 的
`duration_range` / `deadline_range` 目前**隐含**以小时为单位
（因为 env 默认 1 h step）。mapper 必须显式按 §C.2 转换为**槽数**。

### C.6 `lambda_ref = 2000 work-units/hour` 与 ≈1000 work-units/half-hour

见 §A.10。**推荐**：formal 链使用
`lambda_ref_per_step = 2000 × delta_t_hours`（0.5 h → **1000**），
归一化时**分子分母同基准**。**不得**用「每小时参考值」去除「每半小时量」
（那会系统性减半，见 §A.10 ⚠️）。注意这属于 **§C.0 第 1/2 类**（单位换算 / 归一化），
**不改变** arrival/service 比。

### C.7 `queue_ref` / `cost_ref` / SLA 是否受步长影响？

| 参考值 | 语义 | 是否随步长变化 | 理由 |
|---|---|---|---|
| `queue_ref` / `queue_capacity_ref` | **存量** work-units | **否**（不因步长缩放） | 队列是工作量**存量**，其数值由 arrival 与 service 的**速率差**随时间累积决定；把参考值本身乘步长会改变归一化语义 |
| `cost_ref` | 货币 | **否** | 总额与步长无关 |
| `price_ref` | SGD/kWh | **否** | 单位是 kWh 计价 |
| `carbon_ref` / `carbon_factor_ref` | kgCO2 / kgCO2·kWh⁻¹ | **否** | — |
| `sla_penalty_ref` | 货币 | **否** | — |
| `lambda_ref` | **速率** work/hour | **是**（换算到每步：`× delta_t_hours`） | 速率量必须与步长配对 |

### C.8 未来可能触及 `envs/idc_price_env.py::step()` 的变更

| 变更 | 是否触及 `step()` |
|---|---|
| mapper 纯函数（g-e-b） | **否** |
| env 注入 arrival / 任务表（g-e-c） | **是**（reset 与观测的注入点） |
| `delta_t_hours` 传参 | **否**（构造参数，不改 `step` 体） |
| `planned_capacity_vec` 语义（C.3） | **是**（若改语义则 `:622` 变） |
| `_get_task_pool_features` 归一化分母（§E.2） | **是** |

> **⚠️ 强制登记**：**若未来任何卡需要修改 `envs/idc_price_env.py::step()`，
> 该实现卡必须先提交失败测试（AGENTS.md §1.7 / IMPLEMENTATION_PLAN §1.1.7），
> 之后才能修改 `step()`。**

---

## D. 数值量级核验

> 全部为**只读**命令；本节记录完整命令与输出摘要。

### D.1 arrival truth

```bash
uv run python -c "
import pandas as pd, numpy as np
ex = pd.read_parquet('data/processed/singapore_2024/exogenous_drivers_v2.parquet')
a = ex['arrival'].to_numpy(dtype=float)
print(ex['arrival'].dtype, a.min(), a.max(), round(a.mean(),4))
print(np.percentile(a,[1,25,50,75,99]).round(2).tolist())
print(a.sum(), round(a.sum()/366,2))
"
```

```text
int64 | min 803.0 | max 1230.0 | mean 1000.1721
q01/q25/q50/q75/q99 = [857.0, 949.75, 1001.0, 1050.0, 1146.33]
annual total = 17,571,024.0 | per-day mean = 48,008.26
```

### D.2 48-slot template 期望量级

```text
template shape (48,) mean 1.0 min 0.896228 max 1.115769
期望 λ×1000: min 896.23 / max 1115.77 / mean 1000.0
scale param 1000.0 | realized annual mean 1000.1721
```

### D.3 四类 task profile

```text
C_IDC = C_IDC_base = 469.556874725279 | task_workload_scale = 1.0
Σ C_server = 469.556874725279

profile          prob  duration  load_range     deadline_range   wl_rep    wl_range
A_inference      0.40  (1,2)     (0.08,0.18)    (2,4)            122.1     (37.6, 169.0)
B_rl_training    0.25  (2,5)     (0.14,0.30)    (8,16)           413.2     (131.5, 704.3)
C_dl_training    0.10  (4,8)     (0.22,0.45)    (14,24)          943.8     (413.2, 1690.4)
D_preprocess     0.25  (1,4)     (0.08,0.20)    (5,10)           131.5     (37.6, 375.6)
Σ type_probability = 1.0
```

### D.4 期望单任务 workload

```text
E[workload per task] = 262.482   （ref = 469.556874725279）
E[duration per task] = 2.7 步
```

### D.5 每半小时 aggregate arrival 的隐含任务数

```text
1000.1721 / 262.482 = 3.8104 任务 / 半小时
⇒ 7.62 任务/小时；182.90 任务/天
```

### D.6 与 env 现状的差距

```text
env demo : num_tasks=12 over horizon=24 (1h step) → 0.500 任务/步 = 131.2 work/步
formal   : 2000.344 work/小时 → 7.62 任务/小时 → 182.9 任务/天
差距     : 任务数 ≈ 7.6×；工作量/小时 ≈ 15.2×（**rate-based**，见 §D.7 计算 1）
```

> 本节的「工作量/小时」按 **rate-based 物理语义**（§C.2 / §D.7 计算 1）表达。
> 若按非物理的 work/step 语义，分母会翻倍，差距会**看起来**小一半 ——
> **不得**混用两种基准。

### D.7 服务容量 vs 到达量：**两种计算必须分开写**

> **不得混用。** 两种语义的差别完全来自「`planned_capacity_vec` 是否乘
> `delta_t_hours`」，即 §C.2 与旧 work/step 方案的差别。

**计算 1（★ 推荐：rate-based 物理语义，§C.2）**

```bash
uv run python -c "
arr_hh = 1000.1721   # work / 0.5 h
cap_rate = 402.521   # work / hour  (@ action=1.0)
delta = 0.5
print(arr_hh, cap_rate * delta, arr_hh / (cap_rate * delta))
print(arr_hh*2, cap_rate, (arr_hh*2)/cap_rate)
"
```

```text
arrival = 1000.172 work / 0.5h
service = 402.521 work/hour × 0.5h = 201.261 work / 0.5h
ratio   ≈ 4.9695  → 4.970

按小时表达：
arrival ≈ 2000.344 work/hour
service ≈  402.521 work/hour
ratio   ≈ 4.9695  → 4.970
```

**计算 2（当前代码若只传 `delta_t_hours=0.5`、但**不**缩放 planned capacity）**

```text
arrival = 1000.172 work/step
service =  402.521 work/step
ratio   ≈ 2.4848  → 2.485

其隐含的每小时服务能力 = 402.521 / 0.5 = 805.042 work/hour
```

> **结论**：计算 2 的比值虽然「看起来好一半」，但它是**非物理**的 ——
> 它把每小时服务能力**翻倍**（402.521 → 805.042）。**不得**用它作为正式推荐。

**★ 硬结论（rate-based，计算 1）**：正式 aggregate arrival 是满负荷服务能力的
**≈4.970 倍**。把 aggregate **原样 1:1** 映射成 task workload（这正是
本审计的推荐，§G.1）会得到**过载（重负载）场景** —— 队列持续增长、
大量任务无法在 deadline 内完成。

**如实登记，不「修好」**：

1. mapper **保持原始 aggregate 1:1 守恒**（§G.3 不可变式 1）；
2. **不**通过改 `C_server` / `max_task_load` / 接入容量 / SOC / deadline / queue
   来制造可行；
3. **不**因队列增长或 deadline miss 删除、截断、提前完成任务或不计入队列；
4. 若论文主场景需要稳定负载或一定服务达标率，则**当前 arrival 数据的尺度
   缺乏物理校准** ⇒ **阻塞**，并回到上游 arrival 场景口径（M1.3f）
   **重新批准并版本化**（§I.2 **D-INTENSITY**）；
5. **mapper 不承担**修正上游尺度的职责。

### D.8 初始 backlog 数量级

```text
env initial_Q (Q0=50, scale=1) = 50.0
占正式一小时到达量的比例 = 50 / 2000.34 = 2.50%   ← 可忽略
create_initial_backlog_task: workload=Q0=50, duration=1, deadline=24 步, priority=1.0
                             load_profile=[min(Q0/C_IDC,1)] = [0.1065]
```

### D.9 refs 与 env 默认对照

见 §A.10 表：`lambda_ref` 2000 vs 1000、`queue_ref` 6000 vs 1500、
`queue_capacity_ref` 6000 vs 1500、`cost_ref` 60 vs 30 —— **D2 已裁决 formal 用
refs 侧**，但 env 构造默认值仍是旧值，g-e-c 必须**显式**传参。

---

## E. 未来信息（future information）与可见性

### E.1 reset 是否预先创建整段 episode 的 future Task？

**是。** `env.reset()` → `create_demo_tasks()`（`:547`）一次性生成
**整段 horizon** 的任务；未来任务以 `status="not_arrived"` 存在。这是
「任务列表里存在未来任务」的根源。

### E.2 future task 的属性是否可能泄漏？

| 通道 | 是否泄漏 | 证据 |
|---|---|---|
| 观测 `_get_task_pool_features` | **否**（按 `arrival_time <= current_time` 过滤） | env `:1739-1742` ✓ |
| `_compute_backlog_work` | **否**（排除 `not_arrived`） | `:1558-1563` ✓ |
| `snapshot_adapter.build_snapshot` | **否**（`task.status != "not_arrived"`） | `snapshot_adapter.py:236` ✓ |
| **观测归一化分母** | **⚠️ 是**：`total_task_ref = max(len(self.tasks), 1)` 用的是**整段**任务数，含未来任务 | env `:1761` |
| **`info["true_task_arrival_profile"]`** | **⚠️ 是**：`reset` 的 `info` 直接给出**整段 truth**（`include_profiles=True`），即 future arrival 真值 | env `:493-497`、`:580` |
| **观测当步 truth** | **⚠️ 是**：`lambda_norm = true_task_arrival_profile[t] / lambda_ref` 用的是 **truth** 而非 forecast | env `:1975` |

> **对 E.2 第 3 行的判断**：`true_task_arrival_profile[t]` 是**当步**到达量，
> 当步任务已在 `_activate_arrivals(t)`（`:634`）中激活，因此 agent
> 「本来就能从已到达任务算出它」。**但**：(a) 它是**truth**，而 formal 链要求
> 观测只暴露 **causal forecast**；(b) 它与 `_get_forecast_features` 用的
> `task_arrival_forecast` 是**两个不同来源**，同一步里出现两套 arrival 语义。
> **登记为实现卡必须处理的一致性问题**，不在此定性为「泄漏」。

### E.3 future task 的**存在**本身是否影响当前分配/规划/奖励？

| 路径 | 影响 | 证据 |
|---|---|---|
| 分配 `_execute_tasks_action_guided` | **否**（只取已激活任务） | env `:1211-1218`、`:550-554` |
| 规划 `snapshot_adapter` | **否**（过滤 `not_arrived`） | `:236` |
| 奖励 | **否**（`_compute_backlog_work` 排除 `not_arrived`） | `:1558` |
| **观测归一化** | **是**（分母含 future task 数） | `:1761` |

### E.4 正式 mapper 应一次生成全年、一次生成 episode，还是逐槽生成？

| 方案 | 评价 |
|---|---|
| 一次生成**全年** | 便于一次性校验总量守恒；但会把「未来」物化在同一个对象里，泄漏面最大 |
| 一次生成 **episode** | 与现有 env 生命周期一致（reset 生成整段）；仍需 §E.2 的观测归一化修复 |
| **逐槽生成** | 泄漏面最小；但需要 mapper 无状态化 + 明确的 RNG 推进规则 |

**推荐**：**以 episode 为单位生成**（与 env 生命周期一致、便于守恒校验），
**同时**要求 mapper 的输出**只**通过「已到达任务」进入决策（§E.5），
并修复 §E.2 的两处 truth/分母暴露。理由见 §G.1。

### E.5 无论哪种实现，当前决策只能看到已到达任务与 causal forecast

**契约要求**（写入 §G.3 不可变式 8）：

- 决策输入 = 已到达任务（`arrival_time <= t`）+ 已冻结的 causal forecast；
- **不得**读取 `[t, ...)` 的 task 属性（`workload`/`type`/`deadline`/`priority`）；
- 未来任务的**个数**同样不得进入归一化分母。

### E.6 resume/checkpoint 时如何恢复 task stream？

**当前不可恢复。** `reset()` **不**重播种 `task_rng`
（`idc_model/power_model.py:60` 只在构造时 seed）；
`reset()` 每次都推进 RNG，因此**同一 env 实例连续两次 `reset()` 得到不同任务流**，
且任务流**不是** (seed, episode) 的纯函数。

**契约要求**（§G.3 不可变式 5）：task stream 必须是
`f(split, origin, params, revision, seed)` 的**纯函数**；resume 时**恢复**
而非重抽样。

---

## F. 边界与跨日语义

| # | 问题 | 结论 / 推荐 |
|---|---|---|
| F.1 | initial backlog 是否属于 arrival trace？ | **不属于**。它是 `Q0`（`task_model.py:272-292`），与 `arrival` 数据无关 |
| F.2 | initial backlog 如何记账？ | **单独账本**：`task_id=0`、`profile_key="initial_backlog"`；不得混入 arrival 守恒式 |
| F.3 | episode 末未完成任务如何结算？ | env 已有 terminal settlement（`:1494-1495` `leftover = _compute_backlog_work()`），**保留**，mapper 不改 |
| F.4 | deadline 超出 episode 的任务？ | **不得**为对齐 horizon 而缩短 deadline（§G.3 不可变式 9）。允许任务在 episode 结束时仍 `waiting`，由 F.3 结算（future episode 不受影响） |
| F.5 | 不得为对齐 horizon 丢弃/提前到达/缩短任务 | **红线**，写进 §G.3 |
| F.6 | split/episode 边界能否用前一 split 的任务状态？ | **不得**把前一 split 的任务带进后一 split 的**episode**；但 **arrival 的历史窗口**（§E.4 的 causal forecast）可以使用前一 split 的 `arrival` 数据（已发生） |
| F.7 | train/validation/test 的 task seed 与 realization 如何隔离？ | 每 split 独立 seed；**不得**用 validation/test 校准任何参数（§H） |
| F.8 | candidate origin + H 与 `split_end_exclusive` | 沿用 M1.3d：`origin_index + H <= row_end_exclusive`；**`H >= C` 时不重复扣 `C`** |
| F.9 | forecast cutoff 不得造成 `H+C` 双重 purge | 同上；mapper 只负责 arrival→task，**不**改 origin 门禁 |

---

## G. 映射契约

### G.1 **推荐方案 M-1：原始 aggregate 1:1 守恒 + 确定性任务分割**

**核心思想**：mapper 把每槽的**原始** aggregate arrival **原样**分割成整数个
`Task`。**没有**缩放系数、**没有**「有效 workload」、**没有**场景强度修改。

**输入（全部显式、全部冻结）**

| 名称 | 说明 |
|---|---|
| `split` | `train` / `validation` / `test` |
| `episode_origin` | 全局步索引（经 M1.3d `validate_episode_origin`） |
| `aggregate_workload_truth` | 该 episode 覆盖的每槽 aggregate（**原始值，不缩放**） |
| `causal_workload_forecast` | 同一窗口的**因果** forecast（**唯一**可进决策的版本） |
| `mapper_revision` | 冻结的 Git SHA |
| `params` | 冻结参数集（§H） |
| `seed` | 每 episode seed |
| `delta_t_hours` | **0.5**（D4） |

**输出（每个 task 一行）**

`task_id`（稳定：`f(seed, slot, k)`）、`arrival_slot`、`workload`（**work**）、
`profile_key`、`duration_steps`、`deadline_steps`、`priority`、`interruptible`、
`parallelizable`、`provenance`（mapper revision + 参数 hash + 输入 hash）。

**分割规则（确定性，**不含任何缩放**）**

```text
total = aggregate_workload[slot]                      # ← 原始值，1:1
n     = max(1, round(total / E[workload_per_task]))   # 冻结的期望
w_k   = 按冻结的 profile 概率与 profile 期望 workload 加权，
        用 largest-remainder（Hamilton）法分配 total 到 n 个 task
        ⇒ Σ w_k == total   （**对原始 aggregate 精确守恒**，无浮点余数丢失）
```

**为什么推荐它**

1. **数据保真**：`Σ w_k == total == 原始 aggregate`（不可变式 1/2/3）；
2. **确定性**：`n` 与 `w_k` 都是 `total` 与冻结参数的纯函数（不可变式 5）；
3. **不碰物理**：mapper 只产出任务表；容量 / SOC / 约束全在 env 侧不变；
4. **不碰场景**：不改变 arrival/service 比（§C.0 第 3 类**不出现**在 mapper 中）；
5. **可审计**：每个 task 都带 provenance，可与**原始** aggregate 对账。

**它允许的结果是过载场景**（§D.7 计算 1：≈4.970）。这是**如实**的结论，
不是缺陷；是否适合作为论文主实验场景，是**另一层决策**（§I.2 D-INTENSITY）。

### G.2 **备选方案 M-2：阻塞并回到 M1.3f 重定 arrival intensity**

**内容**：暂停 mapper 实现，回到上游 M1.3f 的 arrival 口径，
重新定义并冻结一个**有物理依据**的 arrival intensity
（新的数据版本、manifest、refs、人工批准与版本化）。

**这不是**「在 mapper 里乘一个系数」：

- 新的 intensity 属于 **§C.0 第 3 类**（场景强度修改），
  **必须**在上游完成并被版本化；
- 一旦上游产出新版本 arrival，mapper 仍按 **§G.1 的 1:1 规则**处理**新**数据；
- 旧版本 arrival 与旧 mapper 证据**保留**、标记 superseded。

**触发条件**：人工判定「当前 arrival ≈4.970 倍的过载场景不适合作为
论文主实验场景」（§I.2 **D-INTENSITY**）。

> **修复前**的 M-2（「不缩放，仅在人工接受长期溢出时可选」）**已删除**：
> 1:1 是**数据保真要求**，不是可选项；能否作为主场景是**另一层**决策。

### G.3 不可变式（**13 条**，任何方案都必须满足）

| # | 不变式 |
|---|---|
| 1 | 每槽离散 Task workload 之和**精确等于**该槽 aggregate arrival truth（**原始值，无任何限定词**：**不得**写「缩放后」、「有效 workload」、`scaled truth`、「映射后 truth」等任何限定） |
| 2 | 全 episode workload **守恒**（同一 work-unit 基准） |
| 3 | 不因浮点余数丢工作量（largest-remainder 收尾） |
| 4 | 不创建 `workload <= 0` 的任务 |
| 5 | 固定输入 + revision + 参数 + seed ⇒ **相同** task stream / hash |
| 6 | validation/test **不**拟合或重算参数 |
| 7 | forecast **始终**是 aggregate expected workload，**不**伪装未来 task realization |
| 8 | mapper **不**读取 origin 之后对当前决策不可见的 task 属性（`future` task 的 `workload`/`type`/`deadline`/`priority`，含**任务个数**） |
| 9 | 不清队列、不丢任务、不缩短 deadline |
| 10 | 不放松每任务最大速率、接入容量、SOC、充放电互斥 |
| 11 | initial backlog 与 arrivals **分账**（backlog **不**进入不变式 1/2） |
| 12 | 旧 demo task generator（`create_demo_tasks` / `create_random_tasks`）**不得**作为 formal fallback |
| 13 | **mapper 不改变场景强度**：不得出现 `arrival_scale` 或任何等价缩放；若需改强度，走 §G.2（上游版本化） |

> 若把 aggregate 表达成**另一单位**（例如 work/hour），**必须**同时给出
> **双向换算**与**原始值**，并证明换算前后 arrival/service 比**不变**
> （§C.0 第 1 类）。**不得**只展示换算后的量。

## H. 参数冻结方案

> **`arrival_scale` 已从本表彻底删除**（本卡 P1 修正）。任何「改变 arrival 总强度」
> 的参数都**不属于** mapper 的参数空间（§C.0 第 3 类、§G.3 不变式 13）。

| 类别 | 参数 | 来源 |
|---|---|---|
| **可由 train split 校准** | profile 概率**候选**、`E[workload_per_task]` 的口径候选 | 仅 train 行 `[0, 10224)` |
| **必须人工批准** | 四类 profile 的 `duration_range`/`load_range`/`deadline_range`/`priority_range`（按**物理小时**声明）、`type_probability`、task count 规则、residual 规则、`duration_steps` 取整规则 | 人工裁决（§I.2） |
| **预定物理尺度** | `C_IDC`（**work/hour**）、`C_server`（**work/hour**）、`max_task_load_per_server`、`access_limit_kw`、BESS 参数、`queue_ref`/`queue_capacity_ref`（**存量 work-units**） | 既有声明值，**不改** |
| **每 episode seed** | task realization seed | 显式传入，**不**留空 |
| **禁止从 validation/test 计算** | 上表所有「校准」类 | — |
| **⛔ 禁止出现在 mapper** | **任何**改变 arrival 总强度的量（含 `arrival_scale` 及其等价物） | 只能走上游 M1.3f 版本化（§G.2） |

**未来冻结资产至少应包含**：`schema`/`version`、`mapper_revision`、`units`
（**逐项**声明 work/hour、work/step、work、hour、step）、`delta_t_hours`、
`training_range`、`source_hashes`（canonical / split / exogenous v2 /
**v4 triad** / `refs_v3`）、`profile_probabilities`、`workload_partition_rule`、
`duration_deadline_rule`（含**物理小时 → 槽数**的取整规则）、`seed_policy`、
`residual_handling`、`task_stream_content_hash`。

**建议落盘位置**：`data/manifest/m1.3g_arrival_mapper_v1.json`（g-e-b 产出，
本卡**不**创建）。

## I. 推荐结论与人工决定

### I.1 推荐

**推荐 §G.1（M-1：原始 aggregate 1:1 守恒 + 确定性任务分割）**。
它**允许**结果是**过载场景**（§D.7 计算 1：≈4.970）；这是**数据保真**的必然结果，
**不**通过缩放 arrival 或放大容量来回避。

**备选 §G.2（M-2）**：若人工判定过载场景不适合作为论文主实验场景，
则**暂停实现**并回到 **M1.3f** 重新定义、版本化并批准有物理依据的
arrival intensity。**这不是**在 mapper 里乘系数。

**明确不推荐**：① `arrival_scale`（arrival-only 缩放，改变 arrival/service 比）；
② 「demo tasks 继续充当 formal 来源」；③ 全零 / 默认曲线填充；
④ 为让队列可行而修改容量 / SOC / deadline / queue。

### I.2 需要人工批准的最小决定集合

| # | 决定 | 推荐默认 | 替代项 | 影响 |
|---|---|---|---|---|
| **D1** | aggregate arrival 的语义 | **原始每槽 workload**（work-units/槽）；**mapper 不得改变其强度** | 无（缩放已被否决） | 决定整个契约；缩放 = 改变场景强度 |
| **D2** | `Task.workload` 语义 | **物理 work 数量** = `Σ(load_profile × C_IDC × delta_t_hours)` | 无量纲「每步计数」（**非物理**，不推荐） | 守恒式与能量口径 |
| **D3** | `duration` / `deadline` 的槽位换算 | profile 业务参数按**物理小时**声明；`steps = hours / delta_t_hours`，非整数按 §C.2 取整 | 把常量直接当槽数（物理时长减半，**禁止**） | 任务物理时长差 2× |
| **D4** | `C_server` / planned capacity 语义 | **rate**（work/hour）；`capacity_per_step = rate × delta_t_hours` | 保持 work/step（**非物理**，每小时能力翻倍） | 与 D2 必须成对；决定 §D.7 用哪种比值 |
| **D5** | task partition 规则 | 只**分割原始 aggregate**：`n = max(1, round(total / E[workload]))` + largest-remainder；**不得缩放总量** | 固定 n；按 profile 概率逐类采样 | task 数与 workload 分布 |
| **D6** | profile/type 分配规则 | 确定性加权（按冻结 `type_probability`） | RNG 采样（须固定 seed 顺序） | 可复现性 |
| **D7** | residual workload 处理 | largest-remainder **收尾到最后一个 task**，**保持原始总量** | 丢弃（违反不变式 3，**禁止**） | 守恒 |
| **D8** | seed 粒度 | 每 `(split, episode_origin)` 一个显式 seed | 每 split 一个 | resume 可复现性 |
| **D9** | initial backlog | 保持独立账本（`task_id=0`），**不进入** arrival 守恒式 | 并入第一槽（**不推荐**） | 守恒式正确性 |
| **D10** | 跨 episode deadline | 允许任务在 episode 末仍未完成，由既有 terminal settlement 结算 | 截断 deadline（**禁止**，不变式 9） | 语义保真 |
| **D11** | train-only 校准参数 | **只允许** profile 概率候选与 `E[workload_per_task]` 口径候选；**删除 `arrival_scale` 候选**；任何改变 arrival 总强度的量**一律禁止** | 全部人工给定 | 防泄漏 + 防止悄悄改场景 |
| **D-INTENSITY** | **独立人工决定** | **接受当前 arrival 作为正式「重负载 / 过载」场景**（≈4.970），按 §G.1 实施 | **返回 M1.3f** 重新批准并版本化有物理依据的 arrival intensity（§G.2） | **决定 g-e-b 能否开工** |

**在 D1–D11 与 D-INTENSITY 全部裁决之前**：

- **不得开始 g-e-b**；
- **不得**把本 mapper 设计称为「已获准实施」；
- `formal_env_ready` / `formal_training_ready` **保持 false**。

## J. 后续实现拆卡（**只设计，不执行**）

### J.1 M1.3g-e-b：纯 arrival-to-task mapper

| 项 | 内容 |
|---|---|
| 建议允许文件 | 新增 `scenario/arrival_mapper.py`、新增 `tests/test_m13geb_arrival_mapper.py`、新增冻结参数 manifest、两份 docs |
| 前置 | **D-INTENSITY 与 D1–D11 必须先裁决**（§I.2）；**未裁决不得开工** |
| 必须先红 | **对原始 aggregate 的逐槽守恒**（`Σ w_k == aggregate_truth[t]`，**无限定词**）、确定性（同 seed 同 hash）、无零/负 workload、`delta_t_hours=0.5` 的 duration/deadline **物理小时→槽数**换算、**不读** `[t, …)` 属性、demo generator 不得被调用、**不得出现任何缩放系数** |
| 停点 | mapper 为**纯函数**；不接 env；readiness 不变 |

### J.2 M1.3g-e-c：formal env 注入与 0.5 h 对齐

| 项 | 内容 |
|---|---|
| 建议允许文件 | 新增 `scenario/env_injection.py`、修改 `envs/idc_price_env.py`（**注入点**）、新增测试、两份 docs |
| 必须先红 | 注入数组与 v4 triad / refs_v3 一致；`delta_t_hours=0.5` 显式；`capacity_per_step = rate × delta_t_hours`；**永不**落回 `create_demo_tasks` / 全零 / 默认曲线；`task_forecast_mode` 不得为 `perfect`/`noisy` |
| 停点 | **若需改 `step()`，该卡必须先提交失败测试**（§C.8）；观测归一化分母修复（§E.2） |
| 红线 | 不改物理链、SOC、互斥、接入上限 |

### J.3 M1.3g-e-d：跨层守恒、泄漏与 resume 回归

| 项 | 内容 |
|---|---|
| 建议允许文件 | 新增 `tests/test_m13ged_arrival_conservation.py`、两份 docs |
| 必须先红 | `@pytest.mark.leakage`：改 `[t, …)` 的真值不改变当步决策输入；task stream 是 `(seed, origin, params, revision)` 的纯函数；resume 后 task stream **逐位相同**；`Σ` 跨层守恒（mapper → env → reward → settlement），且**守恒的是原始 aggregate** |
| 停点 | 只加测试，不改实现（除被测缺陷） |

### J.4 M1.3g-f：formal train entry 门禁与接线

| 项 | 内容 |
|---|---|
| 建议允许文件 | 修改 `safe_rl_v2/train.py`（`_require_frozen_real_scenario` 与正式分支）、新增测试、两份 docs |
| 必须先红 | `validate_forecast_purpose(purpose="training")` 被调用；缺 provenance / 不完整链 fail closed；`start` 取自 v4 manifest 而非硬编码 `2023-01-01`；错误信息归因 **M1.3** 而非 M1.2 |
| 停点 | **`make train` 的真实路径只有在 g-e 全部完成后才可能真正跑起来**；在人工放行前不得宣称训练可用 |

---

## K. 范围外修改

**无。** 本卡只新增/修改四份 markdown。
