# M6-P2a：正式 checkpoint 审核与 train-only 服务／库存诊断

## 0. 开卡状态

| 字段 | 值 |
|---|---|
| 分支 | `p4-safeppo-m51a-rollout-contract-m12-integration` |
| 强制起点 SHA | `646f8cc5c3a0fc797473ef4c767360f7ea8980bb`（tree `7488437b6bcad36acaffac948aa949b695bc6072`，实测一致） |
| 开工前 `git status --short` | **空**；`git diff --check` **exit 0** |
| 状态 | 执行中 |

### 0.1 改前失败原因（实测，未改既有测试）

```text
uv run python -m scripts.audit_m6p2a_formal_checkpoints --help
  → No module named scripts.audit_m6p2a_formal_checkpoints（exit≠0）
ls scripts/ | grep -i m6p2 → 无
```

## 1. 目标

**只做两件事**：①审核 `runs/m13gfck_formal_train_seed{0,1,2}/` 三个正式训练产物；
②在**矩阵 v3 的全部 212 个 train 日 origin × 3 个 seed = 636 个完整 episode** 上做
train-only 服务／库存诊断。**不做** validation/test、**不**重训、**不**做五方法公平收益比较。

## 2. 边界（允许修改 / 新增）

```text
scripts/audit_m6p2a_formal_checkpoints.py（**新增**：只读审计 + 诊断入口）
runs/m6p2a_formal_trainonly_212x3_v1/（**新增** run 产物）
runs/m6p2a_export_seed{0,1,2}/（三个 seed 各一次导出与源/导出对照）
任务卡、handoff
```

复用既有 checkpoint loader、统一评估器、冻结服务标准 `m6-service-standard-v1` 与库存记录。
三个正式训练 checkpoint **只读**。必要时修正诊断脚本本身。

### 2.1 固定输入

```text
policy        runs/m13gfck_formal_train_seed{0,1,2}/checkpoint_final.pt（只读）
role          formal_training_policy（经评估输入契约导出后加载）
origin 池     矩阵 v3 training_schedule.origin_pool（212 个 train 日 origin）
scenario_seed 0（**实际驱动环境**：env 种子 = 0 + 冻结 seed_offsets）
service       FROZEN_PROJECT_SERVICE_STANDARD（**显式传入** ⇒ 资格为真值，不再是「未判定」）
corrector     on，生产默认 0.25 s
method 标签   safe_ppo_joint_rolling_corrector（训练 seed 只作策略标识/分层）
```

## 3. 禁止项

- 不读、不运行 validation/test；不重训；不放宽服务、物理或终点 SOC 条件。
- 不把同一方法的三个训练 seed 当作**五方法公平收益**比较；本卡不计算公平收益。
- 不因单条异常轨迹推断整体表现。
- 不为假设性边界编写大量回归测试。

## 4. 验收命令

```bash
uv run python -m scripts.audit_m6p2a_formal_checkpoints --help
# 先单 origin 走通「评估 → 写盘 → 重新读取」
uv run python -m scripts.audit_m6p2a_formal_checkpoints --run-id <smoke-id> --origins-limit 1
# 全量 636
uv run python -m scripts.audit_m6p2a_formal_checkpoints \
    --run-id m6p2a_formal_trainonly_212x3_v1
uv run python -m scripts.materialize_train_release --verify
uv run python -m scripts.verify_m92_matrix --matrix configs/experiments/m9_experiment_matrix_v3.json
make check && git diff --check && git status --short
```

## 5. 证据产物

1. 三个正式 run 的逐项审核结果（manifest / revision / 三 hash / 种子 / 命令 /
   report 与 metrics 的 512 批、episode 与 step 计数、批次顺序摘要、checkpoint SHA 与可加载性、
   scope/role/21 维/冻结配置/优化器与 RNG 状态）。
2. `m13gfck_seed0_recovered_artifacts/` **单列**为历史诊断产物，不混入三份正式 run。
3. 三个 seed 各一次 `formal_training_policy` 导出 + **同一 train origin** 的
   源/导出确定性动作与完整 episode 对照（结果一致 ⇒ 通过；**不更新参数**）。
4. run `m6p2a_formal_trainonly_212x3_v1`：`config.yaml`、**636 行**逐 episode `metrics.parquet`、
   `report.json`、`figures/`、`manifest.json`（三 hash / revision / 种子 / 命令 / 状态可核对）。
5. 每 seed 的服务合格率、终点库存合格率、两者同时满足率与原因分布。
6. 代码优化建议（含实测依据、预期影响、是否改训练语义、是否必须重训）；
   **明确给出「现有 checkpoint 是否具备进入 validation 的条件」及理由**，不宣称泛化性能。

## 6. 回滚点

逐文件 `git add`、分批提交；从**最终 HEAD** 在独立 worktree newest-first 回滚本卡全部提交，
核对 tree 与起点 `7488437b6bcad36acaffac948aa949b695bc6072` 一致；工作树干净。

---

## 7. 验收记录（2026-09-30）

### 7.1 起止与提交

| 字段 | 值 |
|---|---|
| 本卡起点 | `646f8cc`（tree `7488437b6bcad36acaffac948aa949b695bc6072`，实测一致） |
| 开卡 | `71ac8e4` |
| 审计/诊断入口（含实现期三处修正） | `261b766` |
| 最终 HEAD | 本验收记录提交 |

**改前失败原因（实测）**：`scripts.audit_m6p2a_formal_checkpoints` 不存在（ImportError）。

### 7.2 工作一：三个正式训练产物审核（**3/3 全通过**）

逐项核对（每条均为脚本内自动检查，全部 `pass`）：五类产物齐备、`manifest.status=success`、
三个来源 hash 非空、`claims` 三项 false、**512 批**、`metrics` 行数 = 批数、**2048 episode**、
**98 304 transitions**、**8192 Adam step**、**512 次乘子更新**、`metrics.batch_index` 连续 0..511、
**批次顺序摘要 == 矩阵 v3 的 `a06941d7afa104a5…`**、`env_seeds` 由 seed 派生
（0/1/300000、1/2/300001、2/3/300002）、`manifest.command` 含 run-id 与 seed、
checkpoint SHA == 报告记录值且**可加载**、schema/role/scope =
`m13gfck-formal-train-resume-v1` / `formal_training_resume` / `formal_training`、
21 维动作、冻结配置 == live 冻结配置 v1、优化器 + 两个 RNG + Lagrangian 齐备、
`next_batch_index=512`、**正式 loader 读回后 Adam=8192 / 乘子=512**、
冻结配置六项资产 hash 与 live 一致。

**历史诊断产物单列**：`runs/m13gfck_seed0_recovered_artifacts/` 标记为
`historical_diagnostic_artifact`（缺陷①后从 `checkpoint_final.pt` 恢复落盘的那一套；
该次 invocation `batches_run=0`，计数由活对象读出）。**未计入**上面三份正式 run。

**三个 seed 各一次 `formal_training_policy` 导出与源/导出对照**（`runs/m6p2a_export_seed{0,1,2}/`）：
三者均 `source_kind=formal`、role **`formal_training_policy`**、源文件只读未变、
来源账本与 live 全一致、确定性 raw action `max|Δ| = 0.0`、
**源 policy 与导出 policy 的完整 48 步 episode 逐项一致（25 + 21 字段 0 处不一致）**、
参数更新 **0**。

### 7.3 工作二：636 个 train-only episode（`m6p2a_formal_trainonly_212x3_v1`）

212 个 train 日 origin × 3 seed = **636 / 636 完成**（`episode_complete` 全 True、
0 失败、0 超时、0 零动作回退）。`scenario_seed = 0` 实际驱动环境；
**显式**传入 `FROZEN_PROJECT_SERVICE_STANDARD`（`service_standard_id = m6-service-standard-v1`）。

| seed | 服务合格 | 终点库存合格 | 同时满足 |
|---|---|---|---|
| 0 | **212 / 212 (1.000)** | **0 / 212 (0.000)** | 0 |
| 1 | **212 / 212 (1.000)** | **0 / 212 (0.000)** | 0 |
| 2 | **212 / 212 (1.000)** | **0 / 212 (0.000)** | 0 |
| 合计 | **636 / 636** | **0 / 636** | **0** |

**四项服务指标在三 seed 上完全一致**：按时任务率 **1.0000**、按时工作量率 **1.0000**、
期末剩余比例 **0.0000**、不可中断中断 **0**（min = p50 = max）。
四类物理违规合计 **0/0/0/0**；修正器超时 0、回退 0；解算中位数 0.0716 s。

**唯一的不合格项**：`final_soc_outside_env_tolerance` = **636/636**。
`final_soc` 的 min/p50/max = **0.1 / 0.10000000000000163 / 0.10000010526315796**
——**每一个 episode 都停在 `bess_soc_min = 0.10`**（目标 0.5、容差 0.05 ⇒ 偏差 0.4），
`terminal_soc_recovery_kwh` 恒约 **35.0**。
⇒ **seed 0 的 SOC 0.1 不是个例，是三 seed × 全部 212 个 train 日的普遍行为。**

原始经济量（未作公平配对）：购电费 min/p50/max = 0.000 / 15.159 / 132.040 SGD；
碳排 0.000 / 48.885 / 73.331 kgCO₂e；其中 39 条 episode 购电费为 0。

### 7.4 损失定位（代表轨迹 seed 0 / origin 48，逐段核对）

```text
t   raw_storage  exec_storage  charge_kW  discharge_kW  soc_after  grid_kW
 0     0.9866       0.8929       0.000       17.858      0.4060     0.889
 1     0.9871       0.8115       0.000       16.231      0.3206     0.169
 2     0.9879       0.8047       0.000       16.094      0.2359     0.000
 3     0.9882       0.7919       0.000       15.838      0.1525     0.000
 4     0.9880       0.4989       0.000        9.979      0.1000     2.918
 5–47  0.9866..0.9972  0.0000    0.000        0.000      0.1000     —
终止结算：terminal_soc_recovery_kwh = 35.0，penalty = 0.35，
          final_soc = 0.1（目标 0.5、容差 0.05 ⇒ 偏差 0.4000）
raw storage 全程 ≈ **+0.99（满放电请求）**；exec storage 前 5 步按计划放电到下限后归零；
逐 episode 的 raw→exec 修正中位数：storage **0.9143**、compute **0.3703**。
```

**逐段结论——损失发生在「规划目标」这一段，具体是 `planning/model.py` 的目标函数缺终点库存项**：

| 段 | 实测 | 判定 |
|---|---|---|
| policy raw action | 储能维恒 ≈ +0.99（近乎满放电请求，三 seed 一致） | 策略的储能头**退化**为常数，不随观测调节 |
| corrector exec action | 前 5 步按**递减计划**放电（0.89→0.50）到下限后归零；raw→exec 修正中位数 0.9143 | 实际储能轨迹是**规划器选的**，不是 policy 选的 |
| 环境分配/储能更新 | SOC 严格跟随 exec：0.5 → 0.1（第 4 步）后持平 44 步 | 物理链**无异常**（四类违规 0） |
| 终点结算 | `soc_excess = 0.4` → `recovery = 35 kWh`、`penalty = 0.35`，仅**一次** | 结算按现行语义正确执行 |
| 评估判定 | `final_soc_within_env_tolerance = False` | 判定正确 |

**为什么规划器会放光电池**：`planning/model.py` 的目标函数只有
`electricity_cost + degradation_cost + business_shortfall + deadline_shortfall`
（`planning/model.py:700-753`），**SOC 只有上下界约束**（`lb/ub = soc_min/max`，
`:233/:547/:950`），**没有任何「终点回到目标 SOC」的项**。在 48 步的**单日 episode** 内，
把已经存在于电池里的能量放出来服务负载是「零边际购电成本」的，
规划器因此**最优地**在头几步放到下限。
环境侧只在**最后一步**加一次 `-2.0 × soc_excess` 的 reward 惩罚
（`envs/idc_price_env.py:1124-1127`）与一次会计项 `settlement_penalty`
（`:1705-1719`，进 `total_objective_cost`，**不是 SGD**），
不足以在规划/策略层面形成终点库存约束。

**未定位的部分（如实登记）**：policy 的储能头为何收敛到常数 ≈ +0.99 —— 本卡只做了三 seed 的
**行为**核对（三者一致），**未**做梯度/损失归因，故**不作因果断言**。

### 7.5 代码优化建议（均需人工裁决；本卡**未**实施任何训练语义修改）

| # | 具体文件 / 逻辑 | 实测依据 | 预期影响 | 改训练语义？ | 必须重训？ |
|---|---|---|---|---|---|
| A | `planning/model.py` 的 H 步规划目标加入**终点 SOC 项**（例如按 `soc_target` 的偏差 × 容量折成等价能量成本，或加为约束） | 规划器现行目标无该项；exec 储能前 5 步按计划放到底 | 修正 exec 储能轨迹，使 636 条 episode 具备终点库存可比 | **是**（corrector 在训练环内） | **是** |
| B | `envs/idc_price_env.py:1124-1127` 的 `reward_soc_final` 由「末端一次性」改为**逐步 shaping**（或提高权重） | 单次惩罚 −0.7（权重 2.0 × excess 0.35）不足以对抗 48 步的放电收益；三 seed 均收敛到 SOC 下限 | 让策略学习终点库存，而非依赖 corrector 覆盖 | **是** | **是** |
| C | 评估/协议层：把「终点库存可比」写进**主比较的前置条件**（已在 M9.2-R1 的 `fair_cost_carbon_pairing` 内落地）并在训练前**预检** | 本卡 0/636 可比 ⇒ 现行 checkpoint 无法产出协议 §1 的主比较 | 避免把 validation 预算花在不可比的策略上 | 否（评估层） | 否 |
| D | `evaluation/inventory.py` 已记录终点库存；建议在**训练 run 报告**中一并记录每 episode 的终点 SOC 分布，便于训练期早发现 | 本卡才第一次系统测量终点 SOC | 训练期即可观测到该退化 | 否 | 否 |

> **A 与 B 都会改变训练语义，因此都要求重新训练**；本卡**不**实施，也不据此宣称任何性能。

### 7.6 「现有 checkpoint 是否具备进入 validation 的条件」

**结论：不具备（建议先修复终点库存问题并重训，再进入 validation）。** 依据：

1. **服务层已达标**：636/636 满足冻结服务标准 `m6-service-standard-v1`，
   物理违规 0、超时/回退 0、episode 全完成 ⇒ 就**服务资格**而言是干净的；
2. **但库存层全数不可比**：636/636 终点 SOC = 环境下限，超出环境现有容差
   ⇒ 按 M9.2-R1 已落地的 `fair_cost_carbon_pairing`，**没有一条 episode 能进入
   「同等服务下成本/碳更低」的配对差**；协议 §1 的**主比较**因此**无法产出**；
3. 该退化是**三 seed 一致的普遍行为**（不是单条异常轨迹），且已定位到规划目标缺项（§7.4）
   ⇒ 跑 validation/test 只会得到「服务全达标、主比较全不可比」的结果，**不构成有效证据**；
4. 上述判断**不涉及**任何泛化性能主张：本卡只测了 **train** 的 212 个日 origin，
   **未**读 validation/test。

最终是否进入 validation 由人工裁决；本卡只给出上述依据与建议。

### 7.7 门禁与资产

```text
uv run python -m scripts.audit_m6p2a_formal_checkpoints --help   → exit 0
uv run python -m scripts.audit_m6p2a_formal_checkpoints --run-id m6p2a_..._v1 → exit 0，636 episode
uv run python -m scripts.materialize_train_release --verify      → exit 0
    readiness={'formal_env_ready': True, 'formal_training_ready': True}
uv run python -m scripts.verify_m92_matrix --matrix .../v3.json  → 逐项一致 ✅
run manifest：status=success、revision=261b766、三个来源 hash 非空、seed=0、command 可核对
make check                                                       → exit 0（3046 passed, 0 failed,
                                                                   47 deselected, 40:28；
                                                                   ruff All checks passed! /
                                                                   mypy 170 source files 无问题）
git diff --check                                                 → exit 0
git status --short                                               → 空
```

**发布资产未变**：本卡只新增只读审计脚本与 run，未触碰 env release v1 / train release v1 /
v5 / refs_v4 / 冻结配置 / 矩阵。

### 7.8 回滚（最终 HEAD，newest-first，独立 worktree）

回滚列表 = `git log --format=%H <起点>..HEAD`（newest-first）。

```text
回滚侧 tree = 7488437b6bcad36acaffac948aa949b695bc6072
起点侧 646f8cc^{tree} = 7488437b6bcad36acaffac948aa949b695bc6072   → 一致 ✅
```

（实测细节见最终报告。）
