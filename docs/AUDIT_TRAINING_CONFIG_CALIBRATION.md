# 正式训练配置候选与 train-only 预算标定（M1.3g-f-c-h）

> **本文件是配置**候选**与标定证据**，**不是**冻结、**不是**训练结果。
> 三种固定 raw 提案都是**无学习参考策略**；`claims` 三项恒 `false`。
>
> 标定时 HEAD：`8141a04e751c82c0dfe83e38837867619811c8a2`
> 运行产物：`runs/m13gch_calibration/{config.yaml,metrics.parquet,report.json,figures/,manifest.json}`
> 机器可读候选：`configs/training/idc_training_config_candidate_v1.json`

## 1. 配置决策（已选定，本卡执行）

| 项 | 值 | 依据 |
|---|---|---|
| corrector | **on**，`0.25 s`，来源 **`production_default`** | 经 `planning.corrector.resolve_corrector_budget(None, enabled=True)` 解析（**非**本地硬编码）；M5.4i 已审核。训练/评估同一修正语义。`off` 留作日后独立重训的机制对照 |
| 后端 | CPU 单进程，`torch_num_threads=1` | 求解器既有确定性选项**不改** |
| policy | 单隐层 `Tanh`、`hidden=64`、`log_std` 初值 0、`action_dim=21`、三头 critic | 保持现状 |
| `obs_dim` | **520**（实测，`horizon=48`） | **从 formal env 取得，不硬编码**；= `6 + 10 + 6×20 + 8×48` |
| `forecast_cutoff` | **48** | 因果预测来源为 B6 `ScenarioBundle` 的 causal forecast（可见窗口 `[t, t+48)`）。<br>⚠️ **当前 `safe_rl_v2/train.py` 预检仍为 4；后续训练接线必须改为同值 48**（本卡**未**改 `train.py`） |
| Adam | `lr=3e-4`、`betas=(0.9,0.999)`、`eps=1e-8`、`weight_decay=0`、不退火 | 本卡选定 |
| PPO | `clip=0.2`、`lambda=0.95` | 本卡选定 |
| gamma | **每小时 0.99** ⇒ 半小时一步取 **√0.99 = 0.99498743710662** | 按小时定义，避免步长歧义 |
| 批采样 | `horizon=48`（24 h）；每批 **4** episode = **192** transitions；**4** 轮 × **48** 条 mini-batch | 本卡选定 |
| 批内固定 | `old_raw_log_prob` / 三头 advantage / critic target **全部轮次固定**；Lagrangian **每采样批只更新 1 次** | 本卡选定 |
| 规模 | 每 seed **512** 批 = **98,304** transitions；主种子 **[0,1,2]** | 本卡选定 |
| RNG 派生 | `task=+0`、`server=+1`、`forecast=+300000`（相对主种子），显式传入 | **不重播种全局 RNG** |

> ⚠️ **长训前仍须按实际完整链吞吐完成 M9.1 工时预算**；超本机预算则**报告外移**，
> **不得**暗中缩减样本、种子或日期。

## 2. 24 个 train origin 的选择规则（**预先写定**，可重算）

```text
SLOTS_PER_DAY = 24 / 0.5 = 48
日对齐 origin = {48 * (1 + k)}，上界满足 origin + 48 <= 10224   ⇒ 共 212 个
index_k       = floor(k * (212 - 1) / 23),  k = 0 .. 23
origin        = 48 + 48 * index_k
```

**实测 24 个 origin**：

```text
[48, 480, 912, 1344, 1776, 2208, 2688, 3120, 3552, 3984, 4416, 4848,
 5328, 5760, 6192, 6624, 7056, 7488, 7968, 8400, 8832, 9264, 9696, 10176]
```

- **最小间隔 9 天**（不重叠 ✅）；
- 末项 `10176 + 48 = 10224` = `split_rows.end_exclusive`（**恰在边界上**）；
- 全部落在 verified **train** `candidate_origins = [48, 10224)`；**未读 validation/test**；
- `origin -> start` 已**往返校验**（`local_origin_from_start` 回映射 == origin）。

## 3. 三种固定 raw 提案与实测均值

20 个计算维全取 **1.0 / 0.5 / 0.25**，储能维 **0**；每提案 24 origin × 48 步 = **1152 transitions**。

```text
提案 compute=1.00: transitions=1152  business_mean=1.9704861111111112  carbon_mean=1.02089273465195
                   zero_action_fallbacks=0  timeouts=0  deadline_shortfall=1043
提案 compute=0.50: transitions=1152  business_mean=1.9704861111111112  carbon_mean=1.0285893052437065
                   zero_action_fallbacks=0  timeouts=0  deadline_shortfall=1043
提案 compute=0.25: transitions=1152  business_mean=1.9704861111111112  carbon_mean=1.0242255650312615
                   zero_action_fallbacks=0  timeouts=0  deadline_shortfall=1043
```

**`reasons` 分布（三提案一致）**：`{deadline_shortfall: 1043, none: 109}`。

> `correction_reason` 分类依据 `planning/corrector.py:1-11` 的权威语义：
> `none` / `deadline_shortfall` 为 **MIP 最优、可执行**（后者仅业务风险标记）；
> `timeout` / `base_shortage` / `solver_failure` / `proposal_invalid` 为**零动作回退**。
> **三提案的零动作回退与超时均为 0。**

### 3.1 ⚠️ 实测发现：business 在三种计算强度下**饱和**（如实登记）

`business_mean`（`sla_violation_count` = **已过期仍未完成的任务数**，
`envs/idc_price_env.py:1830-1854`）在三个提案上**完全相同**（逐位一致）。

对照实测（同一 origin、同 6 步）：

```text
compute=1.00: exec_action[:3] 首步 = [0.     0.       0.    ]   completed_work = [35.2, 21.12, 27.072, ...]
compute=0.25: exec_action[:3] 首步 = [0.25   0.00140557 0.25 ]  completed_work = [35.2, 21.12, 27.072, ...]
```

即：**`exec_action` 确实随提案不同**，但**完成工作量相同** ⇒ 业务违约数**饱和**
（90.5% 的步为 `deadline_shortfall`）。

> ⚠️ **原表述已降级（M1.3g-f-c-h1）**：本节初稿写「最紧的约束（任务可用量 / 接入容量）
> 在三种计算强度下**都先绑定**」—— 该因果**当时没有直接证据**，现降级为
> **待验证假设**。该假设已在 **§7** 用实测**部分证实**（任务可用量绑定成立；
> 接入容量经实测**不是**绑定层）。

**后果（必须如实理解）**：`business_budget = 1.9704861…` **并不代表"多做就能更低"**，
而是**该参考策略族下已饱和的违约水平**。它是**训练集上已展示可达到的参考水平**，
**不宣称**逐步硬安全。

## 4. 标定结果

```text
business_budget = 1.9704861111111112   violation_task_steps / transition
                  （三提案并列最小 ⇒ 按规则取提案顺序最早者 compute=1.0）
carbon_budget   = 1.02089273465195   kgCO2e / transition
                  （达标提案 {1.0, 0.5, 0.25} 中最低 ⇒ compute=1.0）
qualified_proposals = [1.0, 0.5, 0.25]
```

**乘子（按 `scale = max(1, 该约束参考均值)` 计算，**非**沿用 smoke 的 0.01 / 100）**：

| 约束 | 参考均值 | scale | `learning_rate = 0.01/scale²` | `max_multiplier = 10/scale` | 初值 |
|---|---|---|---|---|---|
| business | 1.9704861111111112 | 1.9704861111111112 | **0.00257545071707194** | **5.074889867841409** | 0 |
| carbon | 1.02089273465195 | 1.02089273465195 | **0.009594884999059906** | **9.79534838536124** | 0 |

**单位**：business 为 `violation_task_steps`（每 transition 均值）；
carbon 为 `kgCO2e`（每 transition 均值）。聚合口径由
`lagrangian.py:40,200-204` 固定为 `per_transition_mean`。

## 5. 资产 hash（本次实际消费）

```text
refs_v4                 b5b64ef28224b53734ef186aff67687ff0db2dbee3d7e83dbed251af864e3ea7
formal_split_v5_train   8608372fba5c560ba458c8903cc10746a4e21254b2c3e29a2c2cd3001d6233c7
m13g_arrival_mapper_v1  efea87f2b854190a9d4433541d7cf4db07bb11ff022a0bd828cdd1d1fe6b040b
env_release_v1          017575dc827043a5926d9b3b1057d91ffdead7b406dd115e97218f51f8c6adae
canonical_parquet       dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd
exogenous_v3_parquet    07b648f0a15db1d8c39838e3e501dafa2f9956155489702e3379cdb775858612
```

（以上为 `runs/m13gch_calibration/report.json` 的 `asset_hashes` 字段。）

## 6. 边界与未决

- 本卡**未**接 `safe_rl_v2/train.py` 的正式训练循环；**未**改 readiness；**未**运行 `make train`；
- `docs/training_config_candidates.json`（历史审计）的 **S/P 分级未被改写**；
- `configs/training/idc_training_config_candidate_v1.json` 的 `status` 为
  **`candidate_not_frozen`** —— 冻结须另开卡；
- **M9.1 工时预算**尚未按实际完整链吞吐完成（§1 警示）。

---

## 7. 业务敏感性诊断（M1.3g-f-c-h1）

> 诊断脚本：`scripts/diagnose_business_sensitivity.py`；
> 运行产物：`runs/m13gch1_diagnosis/`（**独立** run，历史 `runs/m13gch_calibration/` 未覆盖）。
> 仅用 verified train 的 origin **48 / 4848 / 10176**，**固定同一环境种子**
> （`task=0, server=0, forecast=300000`，即 `server_seed = **0**`）。
>
> ⚠️ **R1 更正（种子口径）**：**本文 §3–§6 的 24-origin 标定**使用
> `SEED_OFFSETS = {task: 0, server: **1**, forecast: 300000}`（`MASTER_SEED = 0`）
> ⇒ **标定 `server_seed = 1`，本诊断 `server_seed = 0`，两者不同**。
> 故 §7 的诊断结论**不能**直接外推到 §3–§6 的标定设置。

### 7.1 ⚠️ `corrector=off` 不是基线

`corrector=off` **只用于定位动作作用**，**不得**称为安全或服务合格的基线。
其 `exec_action == raw_action`（逐步实测均为 `True`），故动作差异最纯粹。

### 7.2 最早分叉步骤

**compute 0 vs 0.25（corrector on）**：

| origin | exec_action | planned_capacity | completed_work | sla_violation_count |
|---|---|---|---|---|
| 48 | **0** | **0** | **0** | 2 |
| 4848 | **0** | **0** | **0** | 2 |
| 10176 | **0** | **0** | **0** | 2 |

**compute 0.25 vs 1.0（corrector on）**：

| origin | exec_action | planned_capacity | completed_work | **sla_violation_count** |
|---|---|---|---|---|
| 48 | **0** | **0** | **0** | **None（全程不分叉）** |
| 4848 | **0** | **0** | **0** | **None** |
| 10176 | **0** | **0** | **0** | **None** |

即：`exec_action` / `planned_capacity` / `completed_work` 在 **step 0** 就分叉，
但 **`sla_violation_count` 全程零分叉** —— 业务指标对 0.25→1.0 的强度提升**完全不敏感**。

### 7.3 绑定层定位（**实测结论**）

**决定性对照（corrector = off，`exec == raw`）**：

```text
origin  48:  planned_capacity 总和  0.25 → 2331.1051 | 1.0 → 9324.4206   （差 4.00×）
             completed_work  总和    0.25 → 1569.000000 | 1.0 → 1569.000000（**完全相同**）
origin 4848: planned_capacity 总和  0.25 → 2331.1051 | 1.0 → 9324.4206   （差 4.00×）
             completed_work  总和    0.25 → 1579.000000 | 1.0 → 1579.000000（**完全相同**）
origin 10176:planned_capacity 总和  0.25 → 2331.1051 | 1.0 → 9324.4206   （差 4.00×）
             completed_work  总和    0.25 → 1567.000000 | 1.0 → 1567.000000（**完全相同**）
```

**完成量对照（**由 `runs/m13gch1_diagnosis/report.json` 重算**）**：

| origin | 账本+50.0 | **off** 0.25 | **off** 1.0 | **on** 0.25 | **on** 1.0 | sla off | sla on |
|---|---|---|---|---|---|---|---|
| 48 | 1569.0 | **1569.000000** | **1569.000000** | 1503.483884 | 1503.483884 | 0 | 100 |
| 4848 | 1579.0 | **1579.000000** | **1579.000000** | 1509.498549 | 1509.498550 | 0 | 106 |
| 10176 | 1567.0 | **1567.000000** | **1567.000000** | 1500.977430 | 1500.977430 | 0 | 114 |

**「全部可用工作量做完」只成立于 corrector=off 的 6 条轨迹**：

```text
off 0.25 / off 1.0：completed == initial_Q(50.0) + Σ mapper ledger   （三个 origin 全部成立）
on  0.25 / on  1.0：completed **低于**账本总量
  48: 1503.483884 < 1569.0 ; 4848: 1509.498549 < 1579.0 ; 10176: 1500.977430 < 1567.0
```

> ⚠️ **R1 更正**：本节初稿写「（compute=0.25 与 1.0 **各自**都满足；
> **corrector on/off 亦然**）」—— **错误**：上表显示 **on 的完成量明显低于账本总量**。
> 「全部可用工作量做完」**仅**适用于 **off** 的 6 条轨迹。

**该对照支持的结论（**严格限定范围**）**：

1. **在这些 off 轨迹中，两档动作均足以做完总工作量** —— `planned_capacity` 相差
   **4.00×**，而 `completed_work` 总量逐位相同且等于账本总量。
   ⇒ 仅说明**这两档动作在这 6 条 off 轨迹上都足以覆盖可用工作量**。
2. ⚠️ **由此**不能**推出（R1 更正，原表述越界）**：
   - ❌ 不能排除 **on** 轨迹上的**容量约束**（该对照只在 off 成立）；
   - ❌ 不能断言 **on** 的业务结果不变是**任务可用量**造成的；
   - ❌ 不能排除**修正器投影**对 **on** 的影响。
3. **on 相对 off 服务下降的机制：未定位**（见 §7.4）。

> 上列均基于上表实测；**未**放松容量、SOC、任务或期限约束。

### 7.4 未定位项（**明确记为未定位**）

**修正器 ON 相对 OFF 改变了结局**（origin 48）：

```text
corrector=off: completed_work 总 = 1569.000000   sla_total = 0
corrector=on : completed_work 总 = 1503.483884   sla_total = 100
（origin 4848: 1579.0 / 0  →  1509.498549 / 106；
  origin 10176: 1567.0 / 0  →  1500.977430 / 114）
```

即修正器 ON 使完成量下降约 **4.2%**，并出现 **100–114** 次 SLA 违约计数
（OFF 为 **0**）。**本卡未能定位其机制**（修正器是在满足若干物理约束下最小化
相对 raw 提案的偏移；为何该投影会减少完成量、引入违约，需要**单独诊断卡**）。
**记为未定位**，本节不作因果结论。

### 7.5 预测窗口 4 vs 48（**零对照**）

corrector=on、同一 origin、同一提案下，`forecast_cutoff` ∈ {4, 48}：

```text
first_divergence_sla_violation_count : 全部 None（三个 origin × 三档 compute）
completed_work 总量                  : c4 与 c48 相同（仅浮点级逐步差）
sla_total                            : 100/100、106/106、114/114（c4 与 c48 相同）
```

⚠️ **必须注明**：本节三种参考提案是**常量动作、与 forecast 无关**，
故该对照是环境动力学的**零对照（null control）**，**不能**用来说明
「策略会如何使用预测窗口」。它只说明：**改变预测窗口不影响环境对这些常量动作的响应**。
