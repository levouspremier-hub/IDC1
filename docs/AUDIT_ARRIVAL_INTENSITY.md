# M1.3f-d 审计：arrival intensity 物理口径与场景分层

> 任务卡：`docs/task_cards/M1.3f.md` §M–§N
> 分支：`p4-safeppo-m51a-rollout-contract-m12-integration`
> 开始 SHA：`c02f6e7`
> 前置：**M1.3g-e-a/R2 已通过人工复审**；**`D-INTENSITY` 已裁决**（保留 1000 为
> `stress` 候选 + 返回上游 M1.3f 建立 main intensity）
> 性质：**只读审计 + 契约设计**。**不**修改数据、**不**生成新 intensity、
> **不**开始 mapper、**不**改变 readiness。

**审计问题**：现有 arrival intensity（`1000 work-units / 半小时`）
**有没有**物理或实证依据？若没有，**main scenario** 与 **stress scenario**
应如何分层？上游重新版本化需要什么契约？需要人工批准哪些参数？

---

## A. 当前 `1000` 尺度的来源链

### A.1 Azure/public trace 实际提供什么

| 项 | 值 | 证据 |
|---|---|---|
| 资产 | `azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz` | `data/manifest/singapore_2024_exogenous_v2.json` → `azure.url` |
| 字节数 | `142,968,140` | 同上 `azure.content_length`（B3 批准值） |
| SHA-256 | `aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89` | 同上 |
| 许可 | **CC-BY-4.0** | 同上 `azure.license` |
| 使用的成员 | **14** 个 `invocations_per_function_md.anon.d01..d14.csv` | `azure.used_members` |
| 未使用成员 | **26** 个（`app_memory_*` 12 + `function_durations_*` 14） | `azure.unused_members` |

**每个成员的实际字段**（`scenario/exogenous_drivers.py:377-411` `trace_minute_totals`）：
**`1440` 个按分钟的列（1..1440）+ 逐函数的行**，值是该函数在该分钟的
**invocation count（次数）**。

**单位**：**次数 / 分钟**（per-function invocation count）。
**不是** work-unit、**不是** CPU 利用率、**不是** kW/kWh。
**成员的名称与内容都不含任何日期、星期或时区**
（`uses_archive_dates = False`；`source_aggregation` 明确声明）。

### A.2 48-slot template 如何生成

`scenario/exogenous_drivers.py:377-411`（`trace_minute_totals`）
→ `:512-530`（`arrival_slot_counts`）→ `:533-547`（`arrival_rate_template`）：

```text
1. 流式读取 14 个成员，各自按分钟列求和 → 每个成员一条 1440 维「每分钟总调用数」
2. 成员按文件名 d01..d14 显式升序聚合（顺序不影响结果）
3. 1440 分钟按 minute//30 聚合为 48 个半小时槽
4. counts / counts.mean()  → 48 槽归一化 rate template
```

### A.3 template 是否只保留形状？

**是。只剩形状。** 决定性证据：第 4 步是 `counts / counts.mean()`
（`scenario/exogenous_drivers.py:544`），**绝对调用量被除掉了**。

只读实测：

```bash
uv run python -c "
import json, numpy as np
m = json.load(open('data/manifest/singapore_2024_exogenous_v2.json'))
tpl = np.asarray(m['columns']['arrival']['rate_template'], dtype=float)
print('mean =', repr(float(tpl.mean())))
print('min/max =', round(float(tpl.min()),6), round(float(tpl.max()),6))
print('tpl*1000 mean =', float((tpl*1000).mean()), '| tpl*5 mean =', float((tpl*5).mean()))
"
```

```text
mean = 1.0
min/max = 0.896228 1.115769
argmin/argmax slot = 7 28
tpl*1000 mean = 1000.0 | tpl*5 mean = 5.0
```

**结论**：template 的均值**精确为 1**；同一形状可以乘**任意**绝对尺度。
**template 不携带任何绝对强度信息。**

### A.4 `mean_arrival_work_units_per_half_hour = 1000` 的来源

| 项 | 值 | 来源 |
|---|---|---|
| 常量 | `ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR = 1000.0` | `scenario/exogenous_drivers.py:95` |
| 依据字符串 | `"lambda_ref=2000 work-units/hour × 0.5 hour"` | `:99` `ARRIVAL_SCALE_BASIS` |
| 人工批准 | `decision_id = B5-ARRIVAL`，`approved_on = 2026-09-16` | v2 manifest `columns.arrival.b5_approval` |

**只读实测**（同一命令族）：

```text
ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR = 1000.0
lambda_ref = 2000 (declared) x 0.5 h = 1000.0
=> 1000 == lambda_ref x delta_t_hours : True
```

**关键**：`1000` 的**唯一**来源是 `lambda_ref = 2000`。而

```bash
uv run python -c "
import json
r = json.load(open('configs/frozen_refs/refs_v3.json'))
print(json.dumps(r['references']['lambda_ref'], ensure_ascii=False))
"
```

```text
{"binding": [], "derived_from": "declared_physical_scale",
 "method": "declared_physical_scale", "source_kind": "declared_physical_scale",
 "training_range": "not_applicable", "unit": "work-units/hour", "value": 2000.0}
```

`lambda_ref = 2000` 是 **`declared_physical_scale`**：
`binding` 为空、`training_range = "not_applicable"`、**没有任何数据或外部来源支撑**。

**因此来源链是**：

```text
【declared, 无实证来源】lambda_ref = 2000 work-units/hour
        │  × 0.5 hour（B5-ARRIVAL 人工批准）
        ▼
【human-approved modeled scenario】1000 work-units / 半小时
        │  × 归一化 template（均值 1，只有形状）
        ▼
formal arrival forecast（期望值，D3）
        │  Poisson(seed=20240916)
        ▼
exogenous_drivers_v2.parquet 的 arrival 列（803–1230 的实现值）
```

### A.5 seed `20240916` 的作用

**只决定 Poisson 的随机实现**（`scenario/exogenous_drivers.py:94`、
`generate_arrival` `:574-605`）。它**不**影响期望强度，
**不**携带任何物理信息，**不**是对外部数据的引用。
formal forecast **不使用**该 seed（D3：forecast 取期望值，不抽样）。

### A.6 Poisson realization 与 formal expected forecast

| | Poisson 实现值 | formal expected forecast |
|---|---|---|
| 位置 | `exogenous_drivers_v2.parquet` 的 `arrival` 列（`int64`） | `rate_template × 1000` |
| 实测 | min **803** / max **1230** / mean **1000.1721** | min **896.23** / max **1115.77** / mean **1000.0** |
| 用途 | 模拟场景的**实际** arrival（历史/评估） | **formal forecast**（D3，确定性） |
| 方差 | `Var = λ` ⇒ σ ≈ 31.6 | 0 |

### A.7 哪些是 external evidence

**只有形状**：

| 内容 | 来源 | 证据 |
|---|---|---|
| 日内 48 槽的**相对**形状 | Azure Functions 2019 trace（CC-BY-4.0） | `sha256 aff8b3ca…`、`142,968,140 B`、14 个成员、**绝对量被归一化掉** |

### A.8 哪些只是 human-approved modeled scenario

| 内容 | 批准 |
|---|---|
| 绝对尺度 `1000`（= `lambda_ref × 0.5 h`） | **B5-ARRIVAL**（2026-09-16） |
| `lambda_ref = 2000 work-units/hour` | `refs` 的 `declared_physical_scale`（D2） |
| `process_family = Poisson` | B5-ARRIVAL |
| `seed = 20240916` | B5-ARRIVAL |
| 槽映射 `0..47 → 00:00..23:30` | B5-ARRIVAL |
| `uses_archive_dates = False` | B5-ARRIVAL |

分类：**`modeled_scenario_calibrated_from_benchmark_trace`**
（v2 manifest `columns.arrival.classification`）。

### A.9 `1000` 是否存在任何独立物理校准？

**没有。** 三条只读证据：

1. `lambda_ref` 的 `source_kind = declared_physical_scale`、`binding: []`；
2. `training_range = "not_applicable"` ⇒ **没有**从 train 数据推导；
3. template 均值精确为 1 ⇒ Azure 的绝对调用量已在归一化中被除掉。

### A.10 **不得等同**的量（红线）

| 量 | 单位 | 与 work-unit 的关系 |
|---|---|---|
| Azure invocation count | **次 / 分钟** | **无**已知换算；本审计**未**发现任何可追溯的转换依据 |
| task count | **个 / 槽** | 由 mapper 从 aggregate workload 分割得出（尚未实现） |
| CPU utilization | **无量纲比例** | 无换算 |
| work-unit | work | 本项目自定义量；`lambda_ref` 是**声明**尺度 |
| kW / kWh | 功率 / 能量 | 经 `load_profile` + `delta_t_hours` 间接相关（§B） |

**任何把上表任意两行画等号的表述都必须给出明确、可追溯的转换依据；
本审计未发现这样的依据。**

---

## B. 服务能力基准（rate-based capacity ledger）

> 只使用**预定物理尺度**、**train split** 与**已冻结上游资产**；
> **不**读取 validation/test。以下为只读探针。

### B.1 账本

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from envs.idc_price_env import IDCPriceEnv20D
env = IDCPriceEnv20D(horizon=24, task_seed=0, server_seed=0, forecast_seed=300000)
m = env.model
C = float(np.asarray(m.C_server).sum())
print('N', m.N, '| C_IDC_base', m.C_IDC_base, '| sum C_server', C)
print('max_task_load_per_server', env.max_task_load_per_server, '| base_load', env.base_load)
print('access_limit_kw', env.access_limit_kw, '| delta_t_hours', env.delta_t_hours)
print('facility_rated_power_mw', env.facility_rated_power_mw)
"
```

```text
N 20 | C_IDC_base 485.64689687541835 | sum C_server 485.64689687541835
max_task_load_per_server 0.8 | base_load 0.05
access_limit_kw 18.0 | delta_t_hours 1.0
facility_rated_power_mw None
```

| 量 | 值 | 单位 | 来源 |
|---|---|---|---|
| `C_IDC` / `C_server` | `Σ C_server = 485.647`（`server_seed=0`） | **work/hour** | `idc_model/power_model.py:90-91`（**由 seed 实现**，见 B.5） |
| `max_task_load_per_server` | `0.80` | 无量纲 | env 默认 `:71` |
| nominal compute service rate | `0.80 × 485.647 = 388.518` | **work/hour** | 只读探针 |
| `capacity_per_half_hour` | `388.518 × 0.5 = 194.259` | **work / 0.5h** | §C.2 的 rate-based 换算 |
| `base_load` | `0.05` | 无量纲 | env 默认 `:70` |
| `access_limit_kw` | `18.0` | kW | env 默认 `:85`；`refs.grid_power_limit_kW` 亦为 `18.0` |
| 温度 / PUE | `T_amb` 来自 canonical；PUE `1.486–1.579`（load 1.0→0.0，28 °C） | — | 只读探针 |
| PV / wind | `pv_t` / `wt_t`（M1.3g-e 才接线） | kW | 尚未接入 env |
| BESS | `100 kWh`、充放各 `20 kW`、`η=0.95`、退化 `0.02 SGD/kWh` | — | env 默认 |
| no-export | `allow_pv_export=False`（`:332` 硬拒绝 `True`） | — | env |
| task `max_rate` | **work/step**（`contracts/models.py:860`） | work/step | 契约 |
| queue / deadline | `queue_ref=6000`（存量 work）；`deadline` 为**相对步数** | work / step | `refs_v3` / `idc_model/task.py:46` |

### B.2 四类能力（**必须区分**）

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from envs.idc_price_env import IDCPriceEnv20D
env = IDCPriceEnv20D(horizon=24, task_seed=0, server_seed=0, forecast_seed=300000)
m = env.model; C = float(np.asarray(m.C_server).sum())
print('theoretical compute upper = %.4f work/hour' % (env.max_task_load_per_server*C))
lo,hi=0.0,1.0
for _ in range(60):
    mid=(lo+hi)/2
    L=np.full((1,m.N),mid,dtype=float)
    P=float(m.calc_pue_and_total_power(L_matrix=L,T_amb=np.array([28.0]))[0][0])/1000.0
    if P<=env.access_limit_kw: lo=mid
    else: hi=mid
print('load where P_IDC==access_limit_kw(18kW): %.4f' % lo)
print('access-limited sustainable (task-load only) = %.4f work/hour' % ((lo-env.base_load)*C))
"
```

```text
theoretical compute upper = 388.5175 work/hour
load where P_IDC==access_limit_kw(18kW): 0.2468
access-limited sustainable (task-load only) = 95.5994 work/hour
```

| # | 能力 | 值（work/hour） | 用途 |
|---|---|---|---|
| 1 | **理论计算上限**（纯算力，action=1.0，**无**接入/基础负载约束） | **388.518**（`server_seed=0` 探针） | 上界参考；**不**是可持续服务能力 |
| 2 | **受接入容量与基础负载限制的持续服务能力** | **95.599**（**只读示例探针**） | **不**是已声明/已冻结的正式容量；正式分母待 D3 批准 |
| 3 | **有可再生与 BESS 的时变短期能力** | 未接线（M1.3g-e 才接） | 只能作**逐槽**容量，**不**作恒定 intensity 锚点 |
| 4 | **瞬时峰值能力**（如 PV 满发 + BESS 全放那一刻） | 未接线 | **不得**用作 main intensity 锚点 |

**⚠️ 关键发现（经 R1 修正）**：`access_limit_kw = 18.0 kW` 把**任务**服务能力
压到**约理论计算上限的 1/4**。但表内 **95.599** 是
**`server_seed=0`、`T_amb=28°C`、`access_limit_kw=18`、`base_load=0.05`、
无 PV/风电/BESS 支持时的只读示例探针值**——**不是**已声明/已冻结的正式容量，
**不得**直接作为论文物理常数或正式 `rho` 分母。该探针随温度与 `server_seed`
显著变化（见 §B.6），因此正式容量在人工批准前只用符号
`DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`，**不得给该符号赋值 95.599**。
`env.step()` 的 M3.7a 投影（`envs/idc_price_env.py:637-711`）会按接入预算
**限缩** `planned_capacity_vec`，因此接入上限是**实际约束**，不是纸面数字。

### B.3 arrival vs 能力（rate-based，work/hour）

| 对比 | arrival | 能力（`server_seed=0` 探针） | 比值（**探针，非冻结**） |
|---|---|---|---|
| vs **理论计算上限** | 2000.344 | 388.518 | **5.1487** |
| vs **接入受限可持续**（seed 0、28°C、18 kW） | 2000.344 | **95.599** | **20.9242** |

> **⚠️ 历史比值退役（经 R1 修正）**：`4.970` 的分母
> （`402.521 work/hour` 的 full-action 计划容量）来自**早先未冻结 `server_seed`
> 的环境实例**，**不可复现**，已**退役**为 `superseded_unfrozen_probe`。
> 本轮固定 `server_seed=0` 的理论 full-action 是 **388.518 work/hour**，
> 对应比值是 **5.1487**，**不是 4.970**；**不得**再写「4.970 与 20.9242 两者都对」。
> **5.1487 与 20.9242 也都只是带 seed、温度与约束条件的示例探针**，
> 随 `server_seed` / `T_amb` 漂移（§B.6），**不得**称为正式冻结比值。
> 正式比值只能写
> `stress_ratio = mean_stress_arrival_work_per_hour /
> DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`，
> 在 D3 冻结前**不得**给正式数值。

### B.4 结论

- **第 2 类（受接入限制的持续能力）在「量纲」上**是 intensity 锚点的唯一**候选**，
  但**其具体数值尚未决定**：§B.2 的 95.599 只是 `server_seed=0`、28°C 的单点探针，
  §B.6 证明它随 `server_seed` 与 `T_amb` 显著变化；
- 第 3 类只能作**逐槽**容量；第 4 类**不得**用作锚点；
- 第 1 类只能作**上界**参考，**不**得冒充可持续能力；
- 正式分母在人工批准前只用符号 `DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`。

### B.5 **额外发现：`C_IDC` 本身随 `server_seed` 变化**

```bash
uv run python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from envs.idc_price_env import IDCPriceEnv20D
v=[float(np.asarray(IDCPriceEnv20D(horizon=24,task_seed=0,server_seed=s,forecast_seed=300000).model.C_server).sum()) for s in range(8)]
v=np.array(v); print(np.round(v,3).tolist()); print('spread (max-min)/mean =', round((v.max()-v.min())/v.mean(),3))
"
```

```text
[485.647, 521.626, 511.933, 511.702, 506.909, 507.471, 498.646, 506.778]
spread (max-min)/mean = 0.071
```

`server_capacity_variation = 0.25`（`idc_model/task_model.py:31`）使
`C_IDC` **随 `server_seed` 变化 7.1%**。

> **hard 结论**：任何 `rho` 的**分母**都**必须**是一个**声明（declared）**的
> 容量常数，**不得**用某个 seed 实现出来的 `C_IDC` —— 否则「intensity 是否可接受」
> 会随 `server_seed` 漂移，违反「冻结后所有方法共享」。

### B.6 持续服务能力的**依赖清单**与多温度 / 多 seed 只读探针（**R1 新增**）

**持续服务能力至少依赖以下 8 项**（任何一项未冻结，就不能称为「已声明容量」）：

| # | 依赖 | 影响 |
|---|---|---|
| 1 | `server_seed` 生成的 `P_idle` | 改变每台服务器 idle 功耗 |
| 2 | `server_seed` 生成的 `P_max` | 改变满载功耗 |
| 3 | `server_seed` 生成的 `C_server` | 改变总算力（§B.5 实测 7.1% 漂移） |
| 4 | `T_amb` 与 PUE（COP 随温度变化） | 改变冷却功耗，接入预算可服务的任务负载随之变 |
| 5 | `access_limit_kw` | 硬接入上限（18 kW） |
| 6 | `base_load` | 需先从接入预算中扣除（`(lo − base_load) × C`） |
| 7 | 动作与服务器负载分配口径 | 本探针取**均匀负载** `L=mid` 的 full-action 语义 |
| 8 | 是否排除 PV、风电和 BESS | 本探针**无 PV/风电/BESS 支持**（`pv_t`/`wt_t` 全零、BESS 未放电） |

**探针口径**（只读，未新增脚本/缓存/run/CSV/JSON/parquet）：

- **task-load capacity** = `max_task_load_per_server(0.80) × Σ C_server`；
- **`base_load` 先扣除**：对负载 `mid` 二分，使
  `calc_pue_and_total_power(L=mid, T_amb)[P_IDC]/1000 == access_limit_kw`，
  得到 `lo` 后，持续任务服务能力 = `(lo − base_load) × ΣC_server`（work/hour）；
- **action/load 含义**：`mid` 是逐服务器统一负载率（full-action 语义，
  `calc_it_power` 按 `P_idle + (P_max−P_idle)·(2L − L^k)` 计 IT 功耗）；
- **PUE 与 access-limit 的关系**：`calc_pue_and_total_power` 的 `P_IDC` 含
  IT + 冷却（COP 随 `T_amb`/负载变化）+ `P_others`，接入预算约束的是这个**总功率**；
- **无 PV/风电/BESS 假设**：本探针只评估「纯计算任务服务能力」，不叠加可再生/储能。

**探针 1：`server_seed=0`，多温度**（命令与 §B.2 相同，仅 `T_amb` 变化）：

```text
T_amb=20°C  sustainable=116.5965 work/hour
T_amb=24°C  sustainable=106.5175 work/hour
T_amb=28°C  sustainable= 95.5994 work/hour
T_amb=30°C  sustainable= 89.7949 work/hour
T_amb=32°C  sustainable= 83.7421 work/hour
T_amb=35°C  sustainable= 74.1653 work/hour
```

**探针 2：`T_amb=28°C`，多 `server_seed`**：

```text
seed=0  sustainable= 95.5994 work/hour
seed=1  sustainable=107.4348 work/hour
seed=2  sustainable=104.1109 work/hour
seed=3  sustainable=109.3575 work/hour
seed=4  sustainable= 93.2696 work/hour
seed=5  sustainable=108.3390 work/hour
seed=6  sustainable=103.8888 work/hour
seed=7  sustainable=114.5974 work/hour
```

**结论**：95.599 只能称为「`server_seed=0`、`T_amb=28°C` 下的示例探针值」；
温度从 20°C 到 35°C 使持续能力从 116.60 降到 74.17 work/hour，
`server_seed` 0→7 使 28°C 下的持续能力在 93.27 ~ 114.60 work/hour 间漂移。
**这些探针只证明「正式分母仍未决定」，不得从中挑选最有利的值**；正式容量在
D3 批准前只用符号 `DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`。

---

## C. 可用的 intensity 依据（逐项评估）

> 每项按 8 个维度评估。**「能否决定 absolute intensity」是核心判据。**

### C.1 Azure trace 的**绝对** invocation 数

| 维度 | 结论 |
|---|---|
| source kind | `benchmark_trace`（Azure Functions 2019，CC-BY-4.0） |
| 单位 | **次 / 分钟 / 函数** |
| 带 task-level 标签？ | **否**（只有 invocation 计数，无 task workload/type/duration） |
| 与本 IDC 容量可比？ | **否**（2019 Microsoft 云函数负载 ≠ 本 IDC 的 20 组算力） |
| 只能决定 shape？ | **是**（当前实现的唯一用法） |
| **能决定 absolute intensity？** | **否** —— 且实现**已把绝对量归一化掉**（§A.3） |
| 可否用于 formal main scenario？ | **不可**（无单位换算依据；把 invocation count 当 work-unit 是本审计 §A.10 明令禁止的等同） |
| 允许的论文表述 | 「日内**相对**形状校准自 Azure Functions 2019 公开 trace（CC-BY-4.0）；**绝对强度不是**该 trace 提供的」 |

### C.2 Azure trace 的**相对日内形状**

| 维度 | 结论 |
|---|---|
| source kind | 同上（形状） |
| 单位 | **无量纲**（均值 1 的 48 槽 rate template） |
| 带 task-level 标签？ | **否** |
| 与本 IDC 容量可比？ | **形状**可比（形状不需要容量可比性） |
| 只能决定 shape？ | **是**（这正是它的用途） |
| **能决定 absolute intensity？** | **否** |
| 可否用于 formal main scenario？ | **可以**（作为 **shape** 来源；绝对尺度另行确定） |
| 允许的论文表述 | 「日内形状取自公开 benchmark trace」 |

### C.3 现有四类 modeled task profiles

| 维度 | 结论 |
|---|---|
| source kind | **human-approved modeled scenario**（`idc_model/task_model.py:68-142`） |
| 单位 | `load_range` 无量纲；`duration_range` **物理小时**（本审计 §C 建议） |
| 带 task-level 标签？ | **是**（profile 自带 duration/load/deadline/priority/interruptible） |
| 与本 IDC 容量可比？ | **是**（`workload = Σ load_profile × C_IDC × delta_t_hours`，与 `C_IDC` 同基准） |
| 只能决定 shape？ | **否** —— 它还决定**每任务** workload 的**可行区间** |
| **能决定 absolute intensity？** | **不能单独决定** —— 它给出 `[w_min, w_max]`，但要定「每小时多少 work」仍需**外部强度依据** |
| 可否用于 formal main scenario？ | **可以**（但必须整体人工批准；`type_probability` 不得按结果调） |
| 允许的论文表述 | 「任务类型分布为人工批准的 **modeled scenario** 参数，**不是** 2024 观测」 |

### C.4 IDC 的 nominal / sustainable service capacity

| 维度 | 结论 |
|---|---|
| source kind | **预定物理尺度**（`C_server`、`max_task_load_per_server`、`access_limit_kw`） |
| 单位 | **work/hour**（rate-based，§B） |
| 带 task-level 标签？ | n/a |
| 与本 IDC 容量可比？ | **是**（就是本 IDC 自己的容量） |
| 只能决定 shape？ | 否 |
| **能决定 absolute intensity？** | **能提供分母**，但**分子**仍需外部依据 |
| 可否用于 formal main scenario？ | **可以**（作分母；且必须用**声明**容量，不用 seed 实现值，§B.5） |
| 允许的论文表述 | 「服务能力为模型声明值；接入限制下的持续任务服务率随 `server_seed`/温度漂移，**正式**口径待人工批准 `DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`（seed 0、28°C 探针约 95.599 work/hour，仅示例）」 |

### C.5 文献或公开基准中的利用率区间

| 维度 | 结论 |
|---|---|
| source kind | **UNKNOWN** —— **本审计未在仓库内找到任何声称的文献利用率区间** |
| 单位 | n/a |
| 带 task-level 标签？ | n/a |
| 与本 IDC 容量可比？ | **UNKNOWN** |
| 只能决定 shape？ | n/a |
| **能决定 absolute intensity？** | **不能**（当前资产中不存在；若将来引入，必须带来源、许可、hash） |
| 可否用于 formal main scenario？ | **不可**（当前不存在） |
| 允许的论文表述 | **不得**声称（无来源） |

### C.6 人工声明的 offered-load ratio

| 维度 | 结论 |
|---|---|
| source kind | **human-approved modeled scenario** |
| 单位 | 无量纲 `rho`（§E） |
| 带 task-level 标签？ | n/a |
| 与本 IDC 容量可比？ | **是**（`rho` 定义为「arrival / 本 IDC 声明容量」） |
| 只能决定 shape？ | 否 |
| **能决定 absolute intensity？** | **能** —— 这正是把「声明的 `rho`」变成「绝对 arrival 强度」的唯一途径 |
| 可否用于 formal main scenario？ | **可以**，**但** `rho` 必须是**预先批准**的 modeled scenario 参数，**不得**按训练/测试结果调节 |
| 允许的论文表述 | 「main scenario 的 offered-load ratio 为**预先批准**的 modeled 参数 `rho = …`」 |

### C.7 结论

**没有任何候选能"独立"确定 absolute intensity 且有实证依据。**

唯一**可操作**的路径是 **C.6（人工声明的 `rho`）+ C.4（声明容量）+ C.2（外部形状）**：

```text
absolute main intensity（work/hour） = rho_target × DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR（work/hour）
```

其中 **`rho_target` 是 modeled scenario 参数**（人工预先批准，见 §E），
**容量是声明物理尺度**，**形状来自外部 benchmark trace**。

**若人工不批准某个 `rho`，则必须如实写**：

> **main intensity 是预先批准的 modeled scenario parameter，不是 2024 实测 workload。**

且**推荐维持阻塞**（§F）。

---

## D. main / stress 场景分层

### D.1 main scenario（用于训练与主比较）

| 要求 | 规定 |
|---|---|
| 用途 | **训练**与**主比较** |
| intensity 冻结时机 | **训练之前** |
| 不得按结果调整 | **不得**根据策略表现、validation/test 成本或服务率调整 |
| offered-load 定义 | **必须**有明确的 `rho` 定义（§E） |
| 形状来源 | Azure trace 的相对形状（§C.2） |
| 绝对强度 | `rho_target × DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`，**`rho_target` 人工预先批准** |
| 标记 | `modeled_scenario`（**不是** 2024 观测） |

### D.2 stress scenario

| 要求 | 规定 |
|---|---|
| 内容 | **保留当前 `1000 work-units/半小时` 候选** |
| 定量 | 相对**可持续**能力（seed 0、28°C、18 kW 探针）约为 **20.9242 倍**；相对**理论计算上限**（seed 0 探针）约为 **5.1487 倍**；**两者都是带 seed/温度/约束的示例探针**，**必须写明分母与条件**。历史 `4.970` 已**退役**为 `superseded_unfrozen_probe`（§B.3） |
| 不得混入 | **不得混入 main 场景统计** |
| 不得用于 | **不得用于包装主方法性能** |
| 失败必须保留 | 积压、deadline miss、SLA 违约**必须保留**（不得清理、不得丢弃任务、不得缩短 deadline） |
| 标记 | 显式 `stress` / `overload` |

### D.3 若建议多个 main load level

**必须区分三层并分别命名**：

| 层 | 用途 | 是否进主比较 |
|---|---|---|
| **主场景**（main） | 训练与主比较 | **是**（唯一） |
| **敏感性场景**（sensitivity） | 稳健性检验 | 单独报告，**不**替换主场景 |
| **压力场景**（stress） | 过载行为 | **否** |

**红线**：**不得**在看到训练或测试结果后挑选「最好看」的层级
（AGENTS §1.4 / IMPLEMENTATION_PLAN §1.1.4：冻结后共享，不得按测试日重算）。

---

## E. offered-load ratio `rho` 的定义

### E.1 `rho_target` 与 `rho_realized` 两个口径（**R1 修正：必须分开**）

`rho` 的**分子**有两种可能——**formal expected arrival rate** 或 **Poisson
realization**——二者**不是同一个量**，**不得**互相混称「truth」或「calibration」：

```text
rho_target  =  train 范围内 formal expected arrival rate（lambda_t）的均值（work/hour）
               ──────────────────────────────────────────────────────────────────────────
               DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR（work/hour）

rho_realized =  固定 Poisson seed 后 train arrival realization 的均值（work/hour）
               ──────────────────────────────────────────────────────────────────────────
               同一 DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR（work/hour）
```

- **main intensity 由 `rho_target` 决定**；
- **`rho_target` 是训练前人工批准的 modeled scenario 参数**（不是训练后调参）；
- **不得用 `rho_realized` 反向调整 main intensity**（否则形成 seed-conditioned /
  circular calibration）；
- **`rho_realized` 只作物化后诊断**（登记实现值与期望值的差，不改变场景强度）。

### E.2 `rho_target` 的逐项规定（**不得省略任何一项**）

| # | 问题 | 推荐规定 |
|---|---|---|
| 1 | 分子用哪个时间范围？ | **train split `[0, 10224)` 的 30 分钟槽**（归一化到 per-hour） |
| 2 | 分母是理论上限还是持续容量？ | **持续物理容量**（**可持续**，即受 `access_limit_kw` 与 `base_load` 限制的那一类，§B.2 第 2 类）—— **不**用理论计算上限（388.518）。**但该类的具体数值尚未决定**：§B.2 的 95.599 只是 seed 0、28°C 的单点探针，正式分母在 D3 批准前只用符号 `DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR` |
| 3 | 是否含 access-limit 影响？ | **含**（这正是「可持续」的定义） |
| 4 | 是否含可再生与 BESS？ | **不含**（PV/wind/BESS 是**逐槽**时变能力，§B.2 第 3 类；把它们并入分母会让 `rho` 随天气漂移，破坏「冻结后共享」） |
| 5 | 是否按 train split 计算？ | **是**（分子）；分母是**声明常数**，与 split 无关 |
| 6 | 均值、分位数还是保守下界？ | **均值**（`rho` 是平均负载率的定义）；**另**记录 `p95` 作为**敏感性**信息，但**不**改变 `rho` |
| 7 | `rho` 是否改变物理约束？ | **不改变**任何容量 / SOC / 接入上限 / 任务约束 |
| 8 | `rho` 的定位 | **场景输入**（预先批准的 modeled 参数），**不是**训练后调参 |

### E.3 由 `rho_target` 反推绝对强度（**唯一**允许的换算）

```text
main_intensity_work_per_hour = rho_target × DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR
main_intensity_work_per_half_hour = main_intensity_work_per_hour × delta_t_hours
```

**这属于 §C.0 的「场景强度修改」**：**必须**在**上游**完成、写入新版本
manifest/refs、并由人工预先批准；**不得**在 mapper 里实现（mapper 仍只做
**1:1 原始 aggregate 守恒**）。

> **⚠️ 不得只写「取 0.8」而没有分母定义和来源。** 任何 `rho` 值都必须附带
> 分母（哪一类容量）、分子的时间范围与 split、以及批准记录。

### E.4 `rho_target` / `rho_realized` 必须写清的字段（**逐项，缺一不可**）

| # | 字段 | 规定 |
|---|---|---|
| 1 | **train 范围** | 两者分子都只取 train split `[0, 10224)` 的 30 分钟槽 |
| 2 | **work/hour 与 work/half-hour 换算** | `lambda_t` 与 realization 都是**每半小时**量；换算到 per-hour 需**除以 `delta_t_hours`（0.5）**，即 `× 2`；分子分母必须同一时间基准 |
| 3 | **`delta_t_hours`** | 显式 `0.5`（D4 裁决：formal 路径 0.5，legacy 1.0 不改） |
| 4 | **seed provenance** | `rho_target` 的分子是**期望值**（不用 seed）；`rho_realized` 的分子是 **Poisson seed `20240916`** 的实现值，必须登记该 seed |
| 5 | **分母 revision** | 都是 `DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`，其 revision 待 **D3** 冻结（§H） |
| 6 | **validation/test 不参与** | 两个量都**不得**读 validation/test 来选择或校准 |
| 7 | **expected 与 realization 不混称** | formal expected `lambda_t` 与 Poisson realization **不得**互相称为 truth / calibration；`rho_realized` 不得反调 `rho_target` |

**红线复述**：`rho_target` 是训练前批准的 modeled 参数；`rho_realized` 只是
物化后诊断。任何「用 realization 均值反推 main intensity」都是
**seed-conditioned / circular calibration**，**禁止**。

---

## F. 推荐方案与备选

### F.1 推荐（**明确**）

> **推荐：在人工批准 §E 的 `rho` 定义与**具体数值**之前，`main intensity`
> 继续 `blocked`。** 本审计**不**给出任何具体 `rho` 数值，
> 因为**现有资产中没有任何能独立支撑绝对强度的实证依据**（§C.7）。

**「继续阻塞」的精确含义**（推荐方案的**内容**，不是「什么都不做」）：

1. **保留**当前 `1000 work-units/半小时` 为明确标记的 **`stress` / overload 候选**；
2. **main scenario** 采用 **§E 的 `rho_target` 公式**，但**要求人工预先批准
   `rho_target` 的具体数值**（`rho_target` 是 modeled scenario 参数，
   **不**从 validation/test 推导，**不**按策略表现调节）；
3. 在上游**新版本化**（§G）之前，**不得**开始 g-e-b；
4. 形状继续取 Azure trace 的相对形状（§C.2），**绝对强度不取自该 trace**。

**允许该推荐给出的公式化表达**（**符号形式**，不含数值）：

```text
main_intensity_work_per_half_hour
    = rho_target × DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR × delta_t_hours
```

### F.2 备选 A：**先获取/批准实证依据，再定 intensity**

若论文需要「有实证依据的绝对强度」，则必须先**获取并批准**：

- 本 IDC（或可比设施）的**真实任务负载/到达记录**（带许可、hash、provenance）；或
- 有明确来源与适用性论证的**公开 IDC workload trace**，
  且**能**给出「该 trace 的 workload 单位 ↔ 本项目 work-unit」的**可追溯换算依据**。

**在拿到该依据之前**：维持推荐（阻塞）。

### F.3 备选 B：**多 load-level 敏感性设计**（**不**替换 main）

在备选 A 或推荐方案的 `rho` 获批后，可另设 **sensitivity levels**
（例如 `rho/2`、`2rho`），但：

- **必须**预先冻结、**不得**事后挑选；
- **不得**替换 main scenario；
- **必须**单独报告。

### F.4 **明确不推荐**

- 任何「为让环境可行而缩小 arrival」的系数（= §C.0 第 3 类，且违反红线）；
- 把 Azure invocation count 直接当 work-unit；
- 把当前 `1000` 静默改名为 main；
- 用 `C_IDC` 的**某个 seed 实现值**作 `rho` 分母。

**若现有证据不足以给出具体数值，推荐就是「继续阻塞并获取/批准所需依据」——
本审计**不**凭空给一个可行系数。**

---

## G. 版本化影响清单（**只设计，不实施**）

若未来批准新的 main intensity，至少需要**新版本**：

| # | 资产 | 新版内容 |
|---|---|---|
| 1 | arrival **source / approval manifest** | 新 `decision_id`（如 `B6-INTENSITY`）、`rho`、分母定义、批准日期 |
| 2 | **exogenous driver manifest** | 新 `schema`、`mean_arrival_work_units_per_half_hour`、`scale_basis`、classification（仍 `modeled_scenario…`） |
| 3 | **exogenous parquet** | **新文件**（如 `exogenous_drivers_v3.parquet`），旧 v2 **保留** |
| 4 | **formal scenario provenance** | 新 revision / 新输入 hash |
| 5 | **forecast policy** 或相关输入绑定 | 若 policy 引用了旧 exogenous hash，必须**重新生成** |
| 6 | **refs** | 若 `rho` 的分母是 `lambda_ref`，需新 `refs` 版本 |
| 7 | **formal split manifest triad** | **新版本目录**（如 `formal_splits_v5/`），v4 **保留** |
| 8 | **asset hashes** | 全部重新登记 |
| 9 | **schema / version** | 逐项 bump |
| 10 | **readiness** | 只有在新证据齐备后才可能变化；**本卡不改** |

**硬性要求**（沿用既有纪律）：

1. **旧 v2 exogenous 与当前 v4 split 证据全部保留**；
2. **当前 `1000` trace 标记为 `stress candidate`**，**不**静默改名；
3. **不原地覆盖**；**不自动迁移**；
4. **formal loader 不得 fallback**（沿用 canonical-only 与 fail-closed）；
5. **新证据必须晚于最后一次相关代码修改**（evidence-must-postdate-code）；
6. **mapper 仍只执行 1:1 原始 aggregate 守恒**——intensity 只改**输入**，
   不改 mapper 契约。

---

## H. 人工决定表

| # | 决定 | 推荐值 / 规则 | 替代项 | 风险 | 阻塞哪张卡 |
|---|---|---|---|---|---|
| **D1** | main intensity 的 **source kind** | `modeled_scenario`（人工批准 `rho`）+ 外部**shape** | 真实 IDC trace（需许可/hash/provenance） | 无实证依据时会被称为实测 | **g-e-b**、**M1.3f-e** |
| **D2** | **`rho_target` 公式** | §E.1 的 `rho_target`（分子 = train `[0,10224)` 的 per-hour **formal expected arrival** 均值；分母 = 声明可持续容量 `DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`） | 用 `p95` 或保守下界作分子 | 公式不唯一会导致不可比 | **M1.3f-e** |
| **D3** | **sustainable capacity 分母（以下 6 项必须共同批准，缺一不可）** | **待人工批准**：① 正式硬件 realization、固定 `server_seed`，或确定性硬件参数；② 温度口径（声明设计温度 / train 均值 / train worst-case / 其它明确且预先批准的规则）；③ 持续容量计算公式；④ 是否采用保守下界；⑤ 是否以及如何排除 PV、风电与 BESS；⑥ 最终容量常数 + 单位 + 来源 + revision + provenance | 理论计算上限 388.518（亦为 seed 0 探针） | 用未冻结的单点探针（如 95.599）作分母会把过载藏起来 / 随 seed 漂移 | **M1.3f-e**、**g-e-b** |
| **D4** | **main load level（`rho` 数值）** | **待人工批准**（本审计不给数值） | 多 load-level | 凭空给数 = 以可行性反推 | **g-e-b** |
| **D5** | **sensitivity levels** | 主场景确定后再定（如 `rho/2`、`2rho`） | 不设 | 事后挑选层级 | **g-e-b** |
| **D6** | **stress 标签** | 当前 `1000` 显式标 `stress` / `overload` | — | 隐式混入主比较 | **g-e-b** |
| **D7** | **seed policy** | 每 `(split, episode_origin)` 一个显式 seed；`C_IDC` 用**声明**容量 | 每 split 一个 | seed 漂移改变容量（§B.5） | **g-e-b** |
| **D8** | **train-only 统计范围** | 仅 train `[0, 10224)`；**不**读 validation/test | — | 泄漏 | **M1.3f-e** |
| **D9** | **refs 更新策略** | 若 `rho` 分母为 `lambda_ref` 则**新版本** `refs` | 复用旧 `refs` | 旧 refs 隐含旧强度 | **M1.3f-e** |
| **D10** | **新旧资产命名 / version** | 旧保留为 `stress` 候选；新文件新名 + 新 schema（§G） | 原地覆盖 | 覆盖 = 丢失历史证据 | **M1.3f-e** |
| **D11** | **允许的论文表述** | 「main intensity 为**预先批准**的 modeled scenario `rho`；**不是** 2024 实测 workload」 | — | 把 modeled 当实测 | 论文 |

---

## I. 后续拆卡（**只设计，不执行**）

| 卡 | 范围 | 前置 |
|---|---|---|
| **M1.3f-e** | **若获批准**，版本化 main arrival intensity（新 manifest / parquet / refs / triad 版本，§G） | §H **D1–D11**，特别是 **D3/D4** |
| **M1.3g-e-b** | 纯 arrival-to-task mapper（1:1 原始 aggregate 守恒；§G.2 有界分割；§G.6 固定点账本） | **M1.3f-e 完成**（若结论是证据不足，**仍不得开始**） |
| **M1.3g-e-c** | formal env 注入与 0.5h 对齐 | g-e-b |
| **M1.3g-e-d** | 守恒、泄漏、resume 回归 | g-e-c |
| **M1.3g-f** | formal train gate | g-e-d |

> **若 M1.3f-d 的结论是证据不足（本审计的结论），则下一步必须仍是
> 数据/口径决策（M1.3f-e 或备选 A 的数据获取），不得跳到 mapper。**

---

## J. 范围外修改

**除 `.gitignore` 新增一行白名单外：无。**
本卡只新增/修改四份 markdown（+ `.gitignore` 一行，单独提交并登记）。
**未**修改任何 `.py` / 测试 / manifest / refs / parquet / raw / 配置；
**未**创建新数据资产、mapper、fixture、checkpoint 或 run；
**未**训练或评估；**未**修改 readiness。
