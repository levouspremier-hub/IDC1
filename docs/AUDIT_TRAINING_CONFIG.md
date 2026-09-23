# 正式训练配置候选审计（M1.3g-f-c-g）

> **只读审计卡**：本文件**盘点**现状与候选，**不选择、不冻结**任何数值。
> 每个待裁决项须由**人工**决定；决定后才另开冻结卡。
>
> 审计时 HEAD：`cbabfb021978e5c2cb09f2fab914ee6b9c0489fd`
> 分支：`p4-safeppo-m51a-rollout-contract-m12-integration`
> 机器可读清单：`docs/training_config_candidates.json`

## 0. 来源分级（本文件的核心判据）

| 代号 | 含义 | **可否直接作为正式训练参数** |
|---|---|---|
| **F** | **正式约束**：由冻结资产 / verified 链 / 已批准契约强制 | ✅ 可以（且必须遵守） |
| **S** | **synthetic smoke 示例**：`train.py` 合成 dry-run 路径用 | ❌ **不得**自动升级 |
| **P** | **probe 示例**：两批 / 三批测试的示例输入 | ❌ **不得**自动升级 |
| **U** | **尚未决定**：代码里有值但无任何正式依据 | ❌ 必须人工裁决 |
| **M** | **缺失**：无任何代码位置 | ❌ 必须补设计 |

> **本审计的第一结论**：目前**没有任何**一项 PPO 超参数属于 **F**。
> 现有值全部是 **S / P / U**；正式训练配置**整体尚未冻结**。

---

## 1. policy 架构

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/policy.py:83` `SafePPOPolicy.__init__(obs_dim, action_dim=21, hidden=64)` |
| **结构** | actor = `Linear(obs_dim,64)+Tanh+Linear(64,21)`；critic = `Linear(obs_dim,64)+Tanh+Linear(64,3)`；`log_std = zeros(21)`（`policy.py:86-90`） |
| **现值来源** | **U**（`hidden=64`、单隐层、`Tanh`、`log_std` 初值 0 均为代码默认，无正式依据） |
| **候选** | ① 保持 64×1 层；② 加宽/加深；③ 改激活；④ 对 `log_std` 设非零初值 |
| **待裁决** | 架构是否冻结为现状？容量是否与 `obs_dim=200`（formal horizon=8 的实测值）匹配？ |

**关系**：`obs_dim` **不是**超参数而是环境事实 —— 由 formal env 决定
（`horizon=8` 时实测 `obs_dim=200`；`train.py --horizon` 默认 24 时另算）。
架构冻结时必须**同时**声明 `obs_dim` 的来源。

---

## 2. optimizer 与学习率

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/train.py:631` `torch.optim.Adam(policy.parameters(), lr=1e-3)` |
| **现值来源** | **S**（synthetic smoke 路径内联；`lr=1e-3` 无正式依据） |
| **候选** | ① Adam 1e-3；② Adam 3e-4（PPO 常见）；③ 其他优化器 |
| **待裁决** | 优化器类别、`lr`、`betas`、`eps`、`weight_decay`；是否线性/余弦退火；是否共享 actor/critic 学习率 |

**关系**：`Policy`/`Optimizer` 的**所有权**已在 f-c-d-R1 定为**调用方**；
正式训练入口接线时必须由**配置**提供，**不得**沿用 smoke 内联值。

---

## 3. clip（clipped actor objective 的 ε）

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/ppo_objective.py:82` `clipped_surrogate(..., *, clip_epsilon)` —— **keyword-only 且无默认值** |
| **现值来源** | **U / P**：生产代码**刻意不设默认**；测试用 `0.2`（`tests/test_m13gcd_two_batch.py:28` 等，标为示例） |
| **候选** | ① 0.1（PPO 常见）；② 0.2（经典值 / 现测试值）；③ 其他 |
| **待裁决** | ε 取值；是否随训练进程退火；单次更新是否分 mini-batch 多轮 |

> ⚠️ **`0.2` 目前只是 probe 示例**，**不得**因为「测试里就是这么写的」而升级为正式值。

---

## 4. gamma / lambda（GAE）

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/train.py:66-67` `GAMMA = 0.99` / `LAM = 0.95`；`safe_rl_v2/models.py:146` `compute_three_value_targets(..., gamma=0.99, lam=0.95)`（**有默认值**） |
| **现值来源** | **S / U**（`train.py` 的模块常量与服务 smoke 共用；`models.py` 的默认值同源） |
| **候选** | ① 0.99 / 0.95；② 0.995 / 0.97；③ 其他 |
| **待裁决** | 与 **0.5 小时步长**对应的折扣语义：0.99/step 在 30 分钟步长下等价于每小时 ≈0.9801 —— **是否按小时标定**需人工明确 |

**关系**：γ 的**时间语义**依赖 `delta_t_hours=0.5`（正式约束，见 §11）。
冻结时必须写明「γ 是 per-step 还是 per-hour」。

---

## 5. rollout 长度与批次安排

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/rollout.py:102` `collect_rollout(..., steps)`；`safe_rl_v2/train.py:299` `DEFAULT_STEPS = 8`；`--horizon` 默认 24（`train.py:416`） |
| **现值来源** | **S / P**（`DEFAULT_STEPS=8` 是 smoke 默认；probe 用 `STEPS=3`、`HORIZON=8`） |
| **候选** | ① 单 episode = 一个 rollout；② 跨 episode 累积固定长度；③ 按时间预算 |
| **待裁决** | rollout 长度、是否截断跨 episode、是否 shuffle、mini-batch 大小、每批更新轮数 |

**关系**：formal episode 长度由 `horizon` 决定，而 `horizon` 受 verified split 的
`candidate_origins` 上界约束（`origin + H ≤ split_end_exclusive`）。
**未决定 `horizon` 之前无法定 rollout 长度。**

---

## 6. 更新次数

| 项 | 内容 |
|---|---|
| **代码位置** | 单次更新 `safe_rl_v2/ppo_update.py`（无迭代概念）；两批 `ppo_two_batch.py`；三批 `ppo_three_batch.py` |
| **现值来源** | **P**（现有 probe 各批恰 1 次 `optimizer.step()`，仅为连通性证据） |
| **候选** | ① 固定总更新步数；② 固定时间预算；③ 按收敛判据提前停止 |
| **待裁决** | 总步数 / 总 episode 数；**是否**允许提前停止；停止判据 |

> ⚠️ 现有 probe 的「2 次 / 3 次更新」**只是连通性证据**，
> **不得**被解释为「训练计划」。

---

## 7. business / carbon 乘子预算与单位

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/lagrangian.py:37-46` `UNIT_VIOLATION_TASK_STEPS = "violation_task_steps"`、`UNIT_KG_CO2E = "kgCO2e"`；`REQUIRED_UNITS` 强制 `business→violation_task_steps`、`carbon→kgCO2e` |
| **单位来源** | **F**（契约强制；`ConstraintSpec.__post_init__` 校验） |
| **预算现值来源** | **P**：`budget=5.0`（business，单位 violation·step）/ `budget=3.0`（carbon，单位 kgCO2e）—— 仅出现在测试示例（`test_m13gcd_two_batch.py:60-61` 等） |
| **候选** | ① 沿用 5.0 / 3.0；② 由 **train split** 的违规/碳排分布标定；③ 由人工给定的物理目标 |
| **待裁决** | 两个 budget 的**数值**与**标定方法**；是否按 episode 长度归一 |

**关系**：`carbon` 的物理量由 **realized `carbon_factor_t`** 提供（冻结常量 0.402，
来自 `refs_v4.carbon_factor_ref`，`train_derived`）。预算标定**不得**按
validation/test 重算（红线）。

---

## 8. 乘子学习率与上限

| 项 | 内容 |
|---|---|
| **代码位置** | `safe_rl_v2/lagrangian.py:149-156` `ConstraintSpec(name, budget, unit, learning_rate, max_multiplier)`；更新式 `lagrangian.py:245-249` `mult += lr*(estimate − budget)`，钳于 `[0, max_multiplier]` |
| **现值来源** | **P**（`learning_rate=0.01`、`max_multiplier=100.0`，仅测试示例） |
| **候选** | ① 沿用 0.01 / 100；② 按时间尺度标定 lr；③ 其他上限 |
| **待裁决** | 两个约束的 `learning_rate` 与 `max_multiplier`；是否双时间尺度 |

**已知行为**（f-c-d 实测）：当批内违规量 **低于** budget 时候选值为负 ⇒ 乘子被钳在
**0.0**；故「乘子是否变化」**依赖数据**，不能作为判据。

---

## 9. 种子所有权

| 项 | 内容 |
|---|---|
| **代码位置** | 环境三类种子：`train.py:305` `DEFAULT_ENV_SEED_KWARGS = {"task_seed":0,"server_seed":0,"forecast_seed":300000}`，且 `main()` 拒绝 `None`（`train.py` 参数校验）；策略采样：`collect_rollout(..., generator=...)`，`stats["policy_rng_source"]` 记录 `explicit_generator` / `global_torch_rng` |
| **现值来源** | **S**（`300000` 等为 smoke 默认）；**所有权**为 **F**（「三套种子必须同时显式给定」已是入口校验） |
| **候选** | ① 单种子派生三套；② 三种子独立给定；③ 多种子重复实验的 seeds 列表 |
| **待裁决** | 正式 seeds **列表**；是否要求 N 个 seed 重复；`forecast_seed=300000` 的语义（是否等于「无噪声」） |

**关系**：`collect_rollout` 的**采样随机性只由调用方传入的 `generator` 决定**
（红线：不得重播种全局 RNG）。`train.py` 用 `torch.random.fork_rng` 仅在**权重
初始化**时借用全局 RNG 并还原（`train.py:618`）。

---

## 10. corrector 模式与时间预算

| 项 | 内容 |
|---|---|
| **代码位置** | `planning/corrector.py:43` `PRODUCTION_CORRECTOR_TIME_LIMIT_S = 0.25`；解析器 `resolve_corrector_budget()`（`corrector.py:72-78`）区分 `production_default` / `explicit_override` / `disabled`；入口开关 `train.py:407-408` `--corrector {on,off}` 默认 `off`、`--corrector-time-limit-s` 默认 `None` |
| **现值来源** | **F（仅 0.25 这个数）**：`corrector.py:40-43` 明示「四个入口必须导入本常量，不得各自硬编码」，并有 **M5.4i 人工审核记录**；但 **`on/off` 的正式取值仍为 S/U**（smoke 默认 `off`） |
| **候选** | ① 全程 `on`；② 全程 `off`；③ 分阶段（先 off 后 on） |
| **待裁决** | 正式训练是否启用 corrector；若启用，`time_limit_s` 是否用 0.25 生产默认 |

> ⚠️ **0.25 s 是「本机生产默认值」，不代表性能或收敛结论**（M5.4i 记录）。
> 且 corrector 走 **HiGHS MILP**，在**墙钟预算**下可能走到不同最优顶点 ——
> 这是已知的不确定性来源（此前已登记过两次 flake）。

---

## 11. CPU 后端 / 求解确定性

| 项 | 内容 |
|---|---|
| **代码位置** | `planning/model.py:50-51` `DETERMINISTIC_RANDOM_SEED = 0`、`DETERMINISTIC_PARALLEL = False`；`planning/model.py:54-62` `deterministic_mip_options()`；`model.py:42-47` 注释记录「HiGHS 的 `threads` 选项经 scipy 会崩溃（实测 TypeError）」 |
| **现值来源** | **F（求解器确定性选项）**：`random_seed=0`、`parallel=False` 是**刻意**的确定性设定；**训练设备（CPU-only）为 U/M** |
| **候选** | ① CPU-only、单线程；② CPU 多线程（会破坏 bit 级可复现）；③ GPU（不可复现） |
| **待裁决** | 训练是否**强制 CPU-only**；`torch.set_num_threads` 是否固定；是否要求 bit 级可复现 |

**关系**：现有全部 probe 的可重放证据（两批 / 三批 / resume）都在 **CPU、单进程**
下取得。**换后端即失效**，须重新验证。

---

## 12. 与冻结约束的关系（汇总）

| 冻结来源 | 关系 |
|---|---|
| `configs/frozen_refs/refs_v4.json`（`b5b64ef2…`） | 提供**归一化参考值**（13 项），**不是**训练超参数。与训练配置的唯一交点是：reward/observation 的归一化尺度**必须**用它，**不得**按 validation/test 重算 |
| verified train split（`formal_splits_v5/train.json`，`8608372f…`） | 限定 **episode origins**（`candidate_origins = [48, 10224)`）⇒ 约束 `horizon` 与 rollout 长度；**validation/test 不得参与参数选择** |
| `delta_t_hours = 0.5`（`env_injection.py:48`，正式约束） | 决定 γ 的时间语义（§4）、rollout 的「步」与小时换算；`lambda_ref = 63.988 × 0.5 = 31.994 work/step` |
| `PRODUCTION_CORRECTOR_TIME_LIMIT_S = 0.25`（M5.4i 已审核） | corrector 的**生产默认预算**；是否在正式训练启用仍待裁决（§10） |

---

## 13. 汇总：来源分级与待裁决项

| # | 项 | 当前值 | 来源 | 是否可作正式参数 |
|---|---|---|---|---|
| 1 | policy 架构 | `hidden=64`，1 隐层，Tanh，`log_std=0` | **U** | ❌ |
| 2 | optimizer | Adam | **S** | ❌ |
| 2 | 学习率 | `1e-3` | **S** | ❌ |
| 3 | clip ε | 无默认（调用方传） | **U**（测试用 0.2 = P） | ❌ |
| 4 | gamma / lambda | `0.99` / `0.95` | **S/U** | ❌ |
| 5 | rollout 长度 / horizon | `steps=8` / `horizon=24` | **S**（probe 用 3 / 8 = P） | ❌ |
| 5 | 批次安排（mini-batch / shuffle） | **缺失** | **M** | ❌ |
| 6 | 更新次数 | probe 各 1 次（2/3 批） | **P** | ❌ |
| 7 | business budget | `5.0` violation·step | **P**（**单位 F**） | ❌ |
| 7 | carbon budget | `3.0` kgCO2e | **P**（**单位 F**） | ❌ |
| 8 | 乘子 learning_rate | `0.01` | **P** | ❌ |
| 8 | 乘子 max_multiplier | `100.0` | **P** | ❌ |
| 9 | 种子 | 三套显式（所有权 **F**） | **S**（数值）+ **F**（所有权） | ❌（数值） |
| 10 | corrector on/off | `off` | **S** | ❌ |
| 10 | corrector 预算 | `0.25` s | **F**（M5.4i 已审核） | ⚠️ 数值可，**启用与否**待裁决 |
| 11 | CPU 后端 | 未见显式约束 | **U/M** | ❌ |
| 11 | 求解器确定性 | `random_seed=0`、`parallel=False` | **F** | ✅（必须遵守） |

**结论**：正式训练配置**整体尚未冻结**。

**实测统计**（由脚本对本节表格条目计数，非估算）：

```text
条目总数 = 18
  证据级 F（可作正式参数）      : 2   （求解器确定性选项、corrector 0.25 s 数值）
  证据级 S / P / U / M（不可） : 16
  其中 ✅ 可直接使用            : 1   （求解器确定性选项）
       ⚠️ 数值正式但启用待裁决   : 1   （corrector 时间预算）
       ❌ 需人工裁决            : 15
       └ 其中无代码位置（缺失）  : 1   （批次安排）
```

---

## 14. 建议的裁决方式（供人工参考，**不预设结论**）

1. 逐项给出**数值 + 理由 + 标定依据**（train split 统计 / 物理尺度 / 文献 / 人工）；
2. 明确哪些项**一次性冻结**、哪些允许**后续版本化调整**；
3. 冻结产物应做成**新的版本化配置资产**（新路径 + 新 loader），
   **不得**改写既有冻结资产；
4. 冻结后，`safe_rl_v2/train.py` 的任何修改都会使 `env release v1` 的
   `release_revision` 失效 ⇒ **必须另开版本迁移卡**。
