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
| corrector | **on**，`0.25 s` | 唯一生产预算（M5.4i 已审核）；训练/评估同一修正语义。`off` 留作日后独立重训的机制对照 |
| 后端 | CPU 单进程，`torch_num_threads=1` | 求解器既有确定性选项**不改** |
| policy | 单隐层 `Tanh`、`hidden=64`、`log_std` 初值 0、`action_dim=21`、三头 critic | 保持现状 |
| `obs_dim` | **520**（实测，`horizon=48`） | **从 formal env 取得，不硬编码**；= `6 + 10 + 6×20 + 8×48` |
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
（90.5% 的步为 `deadline_shortfall`）。原因是最紧的约束（任务可用量 / 接入容量）
在三种计算强度下**都先绑定**，故提高计算强度并未改善违约。

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
