# IDC 项目完整交接（新对话起点）

快照：2026-09-17（Asia/Shanghai）。本文是新对话的独立起点；逐卡证据、精确回滚和全部历史以 docs/WORK_HANDOFF.md、任务卡及 Git 历史为准。

## 1. 必读文件与协作协议

按优先级读取：

1. AGENTS.md：红线、Git 纪律、任务卡格式。
2. docs/IMPLEMENTATION_PLAN.md：总体里程碑和语义约束。
3. docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md：VSCode 执行—人工审核协议。
4. docs/WORK_HANDOFF.md：逐卡证据、资产历史、回滚记录。
5. docs/task_cards/M1.3g.md：当前 M1.3g 子卡范围。
6. 本文：当前有效资产、阻塞与下一步。

固定协作模式：

1. 审核者开卡：允许文件、禁止项、先红测试、验收、机器证据、回滚点缺一不可。
2. VSCode 在同一分支执行；先提交卡片和失败测试，再实现，再提交证据。
3. 结束报告必须含起止 SHA、提交、先红/转绿、门禁、范围外改动、资产 hash、回滚、工作树、阻塞项。
4. 审核通过才开下一张卡；不通过只开返修卡。
5. 禁止 git add -A、git add .、amend、rebase、reset hard、checkout --、git clean。

## 2. 当前 Git 快照

~~~
仓库：/Users/levous/Desktop/IDC
分支：p4-safeppo-m51a-rollout-contract-m12-integration
HEAD：1ea881b477177338d16601161db21e104bc4ccfc
工作树：clean（写本文前核验）
~~~

这是累计整合链，不要为重复旧卡而回退：

~~~
paper-baseline                         主线，只读
p1-data-contracts                     d72b192，保留
p4-safeppo-m51a-rollout-contract      04296db，保留
p4-safeppo-m51a-rollout-contract-m12-integration  当前累计链
~~~

每张新卡首先执行：

~~~
git status --short
git branch --show-current
git rev-parse HEAD
git diff --check
~~~

若 HEAD 不匹配旧卡前置，不能 reset/rebase 回去；先判断该卡是否已经执行，或从当前 HEAD 开新卡。

## 3. 永久红线

- 不放松接入容量、SOC、充放电互斥、任务约束来制造可行。
- 不清队列、不重置 SOC、不跳任务来通过测试或评估。
- 训练和评估不得读取未来真值；forecast 必须携带来源、生成时刻、可见窗口。
- 归一化值只能由训练集或预定物理尺度得到；冻结后不得按 validation/test 重算。
- marl、grid_model 主逻辑、legacy、顶层兼容 shim 不改；旧 MARL checkpoint 不进新主链。
- 新主链拒绝旧 23 维动作和无版本 checkpoint；不得填零或截断兼容。
- 修改 envs/idc_price_env.py 的 step 前，必须先提交该卡失败测试。
- PPO buffer 永远保留 a_raw 与 log-prob；a_exec 仅记录 transition/info。
- planning/snapshot_adapter.py 是 oracle/debug，不能转换成 formal 实现。
- 不能以全零、默认曲线或 synthetic fallback 伪装正式数据或训练已就绪。

## 4. 项目进度

| 区域 | 状态 | 结论 |
| --- | --- | --- |
| M1.2 | 完成并整合 | Singapore 2024 四个 raw 数据源、不可变 manifest、许可和 SHA 账本已进入累计链。 |
| M1.3b | 通过 | 半小时 canonical truth reader 与冻结 parquet 已完成。 |
| M1.3d | 通过 | 连续 truth split、严格 schema/timeline/origin 校验已完成。 |
| M1.3e | 通过 | causal forecast provenance、leakage 防护完成；随后 g-0 升级至 contract-v9。 |
| M1.3f | 可用候选数据 | 四项外生驱动已物化，尚未接入 env/train。 |
| M1.3g-0 | 通过 | contract-v9 carbon provenance 升级。 |
| M1.3g-b | 通过 | formal causal ScenarioBundle kernel、代码 provenance、交叉资产绑定完成。 |
| M1.3g-d | 通过 | train-only frozen refs v3 完成。 |
| M1.3g-c | 通过（含 R1/R2/R3） | formal split manifest triad **v4** 是唯一候选，canonical-only loader + live revision。 |
| M1.3g-e | 未开始 | formal env 注入和 0.5h 对齐；先解决 arrival-to-Task 语义阻塞。 |
| M1.3g-f | 未开始 | train entry 正式 gate 与接线。 |
| M5.4 | 工程门禁已解除 | corrector production budget 0.25 秒仅本机候选，不代表正式训练、性能或收敛。 |
| M6 | 未开始 | M1.3g-e/f 和正式训练完成前不得启动。 |

## 5. Canonical 数据与 truth split

### 5.1 Half-hour truth

路径：data/processed/singapore_2024/half_hour.parquet。

- 17,568 行 = 闰年 366 乘以每日 48 槽；时区 Asia/Singapore。
- 范围为 2024-01-01T00:00+08:00 至 2024-12-31T23:30+08:00。
- USEP 价格为 SGD/MWh 乘 0.001，转 SGD/kWh。
- 实际系统负荷取 SASEA，不读 USEP DEMAND。
- weather 来源时间不晚于目标：整点 age 0、半点 age 30 分钟；禁止 next-hour、backfill、centered interpolation。
- national_igs_mwh_per_half_hour 是真实净注入，可为负，约 43.6% 曾为负。逐值保留，不 clip、不 abs。
- canonical truth 不含 local PV、wind generation、carbon、arrival；它们由 M1.3f 的独立外生链提供。

### 5.2 Truth split

路径：data/manifest/singapore_2024_splits.json。

| split | 全局范围 | 行数 | 时间 |
| --- | ---: | ---: | --- |
| train | [0,10224) | 10,224 | 2024-01-01 至 2024-08-01 |
| validation | [10224,13152) | 2,928 | 2024-08-01 至 2024-10-01 |
| test | [13152,17568) | 4,416 | 2024-10-01 至 2025-01-01 |

三段完整覆盖、无重叠、无 gap、无随机打散，闰日在 train。运行时边界：

~~~
episode origin + H <= split_end_exclusive
forecast origin + C <= split_end_exclusive
~~~

不能写成 origin + H + C <= end；H 大于等于 C 时不得双重 purge。

## 6. M1.3f 外生驱动口径

当前 candidate 路径：

~~~
data/processed/singapore_2024/exogenous_drivers_v2.parquet
~~~

它有 17,568 行：timestamp、local_pv_kw、wind_generation_kw、carbon_intensity、arrival；没有改写 canonical truth parquet。

| 变量 | 口径 | 禁止误称 |
| --- | --- | --- |
| PV | modeled_scenario；pvlib 0.15.2、因果天气输入，B5 批准 ERBS/isotropic/albedo=.25，损耗只应用一次。 | 不是本地实测。 |
| wind | modeled_scenario；10m 风速以 (60/10)^(1/7) 换算，E48/800 曲线。 | 不能把 ERA5 m/s 当 kW。 |
| carbon | human_approved_external_low_resolution；2024 年常数 0.402 kgCO2/kWh。 | 不是半小时实测。 |
| arrival truth | modeled_scenario；date-free 48-slot template 的 Poisson realization，seed 20240916。 | 不等于 formal forecast。 |

Formal arrival forecast 必须是 lambda(slot) 乘 1000 的期望值，而不是 Poisson sampled truth。

## 7. Contract-v9、policy 与 formal kernel

唯一有效版本是 contract-v9。

- contract-v8、contract-v7、非字符串/未知版本均明确拒绝；不得迁移、填零、静默升级。
- 旧 v8 checkpoint/buffer 应明确拒绝。
- 新 human_approved_external_low_resolution 只允许 formal carbon_forecast；其他六个 forecast、synthetic、oracle-debug 均拒绝。
- unavailable 不得进入完整 formal bundle。

Seasonal-naïve 的唯一语义：

~~~
forecast[k] = y(origin + k - 48)
~~~

即上一日同槽位，来源只能是 [origin-48, origin)，不能退化为最后 C 个值重放。

scenario/formal_scenario.py 的 formal kernel 已构造七条 forecast：

| 序列 | source kind | 构造 |
| --- | --- | --- |
| price/load/temperature | seasonal_naive | 经验证 artifact 的上一日历史槽位。 |
| PV/wind | modeled_scenario | 由五个 causal driver forecasts 逐点做同一物理变换。 |
| carbon | human_approved_external_low_resolution | 0.402，受 contract-v9 限制。 |
| arrival | modeled_scenario | 48-slot expected rate，不是 Poisson 真值。 |

已保证：

- 改 [origin, origin+C) 真值不改变七条 forecast。
- 改 [origin-48,origin) 天气历史会改变 temperature/PV/wind forecast。
- FORMAL_SOURCE_PATHS 覆盖 provider、formal kernel、exogenous driver；dirty 时拒绝构造。
- public entry 没有 caller 可覆盖的 expected SHA/revision/trust root。
- exogenous 链与实际 canonical/split/exogenous 对象交叉绑定。
- 不得复用 planning/snapshot_adapter.py 的真值可见窗口或全零 load forecast。

## 8. 唯一可用 refs 与 formal split manifests

### 8.1 Refs

**B6 正式链只可用 `configs/frozen_refs/refs_v4.json`**
（schema `m1.3feb2b-frozen-refs-v1`，SHA-256 `b5b64ef2…`；R1 后重新物化）。

`refs_v3.json`（历史 v3）与 `refs.json`（历史 v2，标记
superseded_pre_trust_boundary_fix）**逐字节保留**，但**不得**被正式链使用（无 fallback）。
**`refs_v4` 的 `lambda_ref = 63.988 work-units/hour`**（`decision_id=B6-INTENSITY`），
**不继承旧的 2000**。

**refs_v4 的关键值**（已调用的判据以文件为准）：

~~~
price_ref=4.5                          max(abs(train price))
pv_ref_kw=350.9073696124661            train derived
wind_ref_kw=262.3178613166015          train derived
carbon_factor_ref=0.402                train derived
lambda_ref=63.988                      B6-INTENSITY（**不是** 旧的 2000）
queue_ref=6000, queue_capacity_ref=6000, cost_ref=60   declared physical scales
~~~

生成器不得接收 dataframe/frame、kwargs、expected hash 或 caller trust root。

### 8.2 Formal split triad

**B6 正式链只可用目录 `data/manifest/formal_splits_v5/`**
（schema `m1.3feb2b-formal-split-manifest-v5`，**已 bump**）。

**✅ `M1.3f-e-b2-b` / `-R1` 执行完成（等待人工复审）**：b2-b 第一轮审核不批准，R1 修三项 P1——① v5 loader 改为**整体重建比对**（改前改 `time_range` 起止即得 `FORGED_TIME_RANGE_ACCEPTED`），差异时输出差异顶层字段；② `frozen_at_utc` **锚定到已验签 refs_v4**（⇒ 必须先物化 refs_v4 再物化 v5）；③ revision 两集合各扩到 **9 项**（补入 `scenario/scenario.py` 与两个 materializer）；④ 伪事务测试换成**真实** `materialize_*()` 三阶段注入（四产物全不存在、零临时文件）。R1 后四产物：refs_v4 `b5b64ef2…`、v5 train `8608372f…` / validation `3f16ad3a…` / test `aaacd459…`；`materializer_revision = 577f1db…`；共享 `frozen_at_utc 2026-09-19T17:15:19+00:00`。`make check` **2632 passed**；**旧资产逐字节未变**。**public-entry mutation 回归仍未实现**（范围声明未改）。v1–v4 **全部 superseded**，逐字节保留但不得 fallback。

~~只可用目录 `data/manifest/formal_splits_v4/`~~（历史，superseded）
~~schema `m1.3g-formal-split-manifest-v4`~~（历史）

v5 triad：三个文件共享一个 `frozen_at_utc`（**锚定到 refs_v4**），以 triad 原子物化。

| split | split-local candidate origins |
| --- | --- |
| train | [48,10224) |
| validation | [0,2928) |
| test | [0,4416) |

candidate origin 仅说明有 48 个历史槽位；实际 H/C 仍运行时检查。triad 不固定 H/C，不含 forecast 数组、未来真值、默认曲线、单点 origin。

scenario/formal_split_manifests.py 的 manifest_relative_path 和 CLI 默认都必须指向 v4；loader 是 canonical-only（临时副本、路径别名、symlink、v1/v2/v3 位置一律拒绝），且 materializer_revision 必须等于当前 resolver。唯一 verified loader 是 load_verified_split_manifest(path, expected_split=...)。它 live-verify 八个输入角色、refs_v3、policy、truth split、exogenous 链、范围/时间/origins/readiness/contract/schema。

下列仅历史保留，正式链禁止 fallback：

~~~
data/manifest/train.json, validation.json, test.json       v1 superseded
data/manifest/formal_splits_v2/                            v2 superseded
data/manifest/formal_splits_v3/                            v3 superseded_pre_canonical_loader_trust_boundary_fix
~~~

### 9.0 **B6 mapper 可行域审计（2026-09-20，M1.3g-e-b-a）—— 最新**

**只读审计完成，停在人工决策处**。审计文档：`docs/AUDIT_ARRIVAL_TO_TASK_MAPPING.md` §ah。

**基数修正（重要）**：任务卡用的 `C = 469.556874725279 work/hour`
**扫 `server_seed` 0–63 无一匹配、不可复现**（同 `95.599`/`4.970` 一类未冻结探针）。
B6 policy 固定 `server_seed=0`，其参考量为
**`C_IDC_base = 485.646896875418349 work/hour`**。

| 量 | 任务卡 | **本审计** |
|---|---|---|
| 全局最小 `w_min`（work） | 37.56454997802232 | **38.851751750033472** |
| `< w_min` 的槽（ALL） | 14,401 / 81.9729% | **15,109 / 86.0030%** |
| train / validation / test | 8,375 / 2,377 / 3,649 | **8,797 / 2,494 / 3,818** |

**零聚合槽 = 0**（min = 12）⇒ 无「零任务覆盖」逃逸。
**expected forecast 的 48 个 slot 全部小于 `w_min`**（`max/w_min = 0.918824`）。

**证明**：合法 Task 的 `workload >= 全局最小 w_min > aggregate` ⇒ 非空集超出、
空集为 0，与 1:1 守恒矛盾 ⇒ **无合法分割**。

**人工决策（只有三个）**：**A** 批准新的/修改的 modeled micro-task profile 参数
（须逐项批准 **13 项**，§ah.7）；**B** 回上游 M1.3f 重新批准并版本化场景强度或时间语义；
**C** 保持当前参数并阻塞。**审计不自行选择。**

**六类伪选项禁止**：mapper 内缩放 arrival / 合并移动半小时槽 / 丢弃低 aggregate 槽 /
零 workload 或零数量替代 / 未经批准放宽 profile 约束 / 用 validation/test 拟合参数。

> ## ⛔ **g-e-b mapper 实现仍 BLOCKED，等待 micro-task/profile 参数或上游强度语义的人工批准。**

**R1 账本收口（2026-09-20，纯文档）—— 三项澄清**：

1. **`a740e38` 是【前置】的 `M1.3f-e-b2-b-R1` 审核收口提交**（改动 M1.3f 卡片与
   `WORK_HANDOFF`），**不属于**本审计卡，**必须保留**、**不得**纳入本卡回滚链。
2. **本 `M1.3g-e-b-a` 审计本身只修改四份文档**：
   `docs/AUDIT_ARRIVAL_TO_TASK_MAPPING.md`、`docs/task_cards/M1.3g.md`、
   `docs/WORK_HANDOFF.md`、`docs/NEW_CONVERSATION_HANDOFF.md`。
3. **数学结论已成立，但【不】构成 A / B / C 中任一项的人工作出决定**——
   86.0030% 的槽不可行是**事实**；选哪一项是**人工裁决**，尚未发生。
   **在裁决之前 mapper 仍 BLOCKED。**

**本审计卡未创建** mapper、fixture、参数 manifest、Task、run 或 checkpoint；
`make check` **2632 passed**、`make train` **exit 2**；范围外修改**无**。

**本审计卡精确 newest-first 回滚**：

```bash
git revert 316d813 71da67a da862a4 59c1fa3 a532390
# revert 后 HEAD^{tree} == a740e38^{tree} = 60349ad7d27a02831ac9043a0645dc0b0025a4f1
```

### 9.0c **13 项参数已由人工签核冻结（M1.3f-e-b-d，2026-09-20）—— 最新**

**人工已逐项批准路线 A 的 13 项**（状态 `PENDING` → **`APPROVED`**，
**批准来源 = 人工**；实现只记录、未选择任何值）：

| # | 参数 | 人工批准值 |
|---|---|---|
| 1 | profile 名称 | **新增 `E_micro_inference`** |
| 2 | duration | **`0.5 h`**（1 step） |
| 3 | load range | **`[0.04, 0.25]`** |
| 4 | deadline | **`1.0 h`**（2 steps） |
| 5 | priority | **`2.6–3.4`**（沿用 `A_inference`） |
| 6 | interruptible / parallelizable | **`false / false`** |
| 7 | 扩展 vs 新增 | **新增**；**不修改** A/B/C/D |
| 8 | `MAX_TASKS_PER_SLOT` | **`4`** |
| 9 | `MAX_GROWTH_ROUNDS` | **`4`** |
| 10 | `work_unit_scale` | **`1_000_000`** |
| 11 | `ABS_TOL_WORK` / `REL_TOL_WORK` | **`1e-9` / `1e-12`** |
| 12 | 确定性顺序 | **`E_micro_inference → A_inference → B_rl_training → C_dl_training → D_preprocess`** |
| 13 | 来源声明 | **`modeled scenario`**（不是 2024 task labels） |

**独立复算（只读）**：`C_IDC_base = 485.646896875418349`、`delta = 0.5`；
`duration_steps=1`、`deadline_steps=2`（`>=` 成立）；
`E_micro_inference` 合法区间 **`[9.712937937508368, 60.705862109427294] work`**；
**train `[12,59]` 被覆盖**（validation `[12,54]` / test `[14,55]` **仅作覆盖验证**）。

**本卡只解除「参数未批准」这一阻塞。** mapper（`g-e-b`）、env 接线（`g-e-c`）、
回归（`g-e-d`）、`g-f`/正式训练、评估、M6 **仍未开始**；readiness **仍为 false**；
旧 A/B/C/D profile **一律不变**。
**下一张卡 = `M1.3g-e-b`（mapper 实现卡）** —— 本卡不开始它。

### 9.0b **A 路线已选，（历史）13 项参数当时为 PENDING（M1.3f-e-b-c，2026-09-20）**

**人工已选择路线 A**：用 **modeled micro-task/profile 参数**解决可行域缺口。
**但 A 的选择【不】等于 13 个具体参数获批。**

审计文档 `docs/AUDIT_ARRIVAL_TO_TASK_MAPPING.md` §au 给出：

- **数学必要条件**（只用 train）：`C_IDC_base = 485.646896875418349`、
  `K = C x delta = 242.823448437709`；TRAIN `n=10,224 min=12 max=59 mean=32.019757`，
  严格正槽 10,224、零槽 0，**最低正 aggregate = 12 work**；
- **可行区间**（不选定）：`duration_steps x load_min <= 12 / K`，即
  1 step → `load_min <= 0.049418621131`、2 steps → `0.024709310565`、
  3 steps → `0.016472873710`、4 steps → `0.012354655283`；
  现有最小 `w_min = 38.851751750033 work`（比最低槽大 3.237646 倍）；
- **13 项人工签核表**（§au.3）：profile 名称 / duration / load range / deadline /
  priority / interruptible·parallelizable / 扩展旧 vs 新增 / `MAX_TASKS_PER_SLOT` /
  `MAX_GROWTH_ROUNDS` / `work_unit_scale` / `ABS_TOL_WORK`·`REL_TOL_WORK` /
  profile·type 确定性顺序 / modeled-scenario 来源声明 ——
  **状态一律 `PENDING`**，实现未写任何 `APPROVED`；
- **一次性签核模板**（§au.4，13 行「批准值 + 理由」）与
  **覆盖验证顺序**（§au.5：先 train 复核，再以 validation/test **仅验证**、
  不得回头改参数）。

> ## ⛔ **g-e-b mapper 实现仍 BLOCKED。**
> 在 **13 项逐项**获得人工批准值之前，不得开始 g-e-b / g-e-c / g-e-d / g-f、
> env 接线、训练、评估或 M6；readiness 保持 **false**。
> `git diff --name-only dfce1a8..HEAD` 仅三份文档；范围外修改**无**。

## 9. 当前硬阻塞：arrival workload 到离散 Task 的语义

现在不要直接启动 M1.3g-e 实施。

M1.3f 的 arrival 是每半小时 aggregate workload trace；当前 IDCPriceEnv20D 的 true arrivals 却来自 IDCEnergyTaskModel.create_demo_tasks 和 build_task_arrival_curve，是随机生成的离散任务。若只把 formal arrival forecast 写到 observation、真实 queue/task dynamics 继续跑另一套随机任务，预测和真值不是同一物理量，不能称为正式训练。

下一张推荐卡：

~~~
M1.3g-e-a：arrival-to-task 接口只读审计与映射契约设计
~~~

卡只能只读，目标：

1. 从 aggregate workload 的单位追到每个 Task 的工作量、能耗、服务时长、队列消耗及 IDCPriceEnv20D reset/step。
2. 找出 workload 到 discrete Task stream 的唯一、可复现、因果 mapping 所需参数。
3. 说明 mapping 是否只在 train 校准、如何冻结 revision/seed/units，以及 forecast expected rate 和 truth realization 的对应。
4. 给一个推荐、最多两个备选，列出需要人工批准的决定。
5. 不改 env/task model/train/manifest/parquet/refs/物理模型；不创建 checkpoint；不声称 readiness。

审计和人工决定完成后，才能开真正的 M1.3g-e。

**状态（2026-09-18）**：`M1.3g-e-a` 前两轮审核不通过，**已返修（R1/R2）并通过人工复审**。**`D-INTENSITY` 已裁决**：当前 1000 保留为 `stress` 候选，**返回上游 M1.3f** 为 main scenario 建立有来源、预先批准、不得按结果调节的 arrival intensity；**在该上游方案通过前不得开始 g-e-b**。
审计文档：`docs/AUDIT_ARRIVAL_TO_TASK_MAPPING.md`。**三项核心结论**：

1. **mapper 必须 1:1 守恒原始 aggregate**：**不得**缩放 arrival
   （改前推荐的 `arrival_scale` 已**删除**）；「单位换算 / 归一化 / 场景强度修改」
   三分，第 3 类**只能**在上游 M1.3f 版本化。
2. **推荐 rate-based 半小时语义**：`C_IDC`/`C_server` 是 **work/hour** 速率；
   `capacity_per_step = rate × delta_t_hours`；
   `Task.workload = Σ(load_profile × C_IDC × delta_t_hours)`；
   `duration`/`deadline` 按**物理小时**转槽。
3. **当前 arrival 是过载场景**：rate-based 下 **arrival/service ≈ 4.970**
   （另一种非物理口径为 2.485，隐含每小时服务能力翻倍，**不得**用作推荐）。
   **不**通过改容量/SOC/deadline/queue 或删任务来「修好」。
   **R1 勘误（M1.3f-d-R1）**：该 `4.970` 的分母 `402.521 work/hour` 来自**早先
   未冻结 `server_seed` 的旧环境实例**，已**退役**为 `superseded_unfrozen_probe`；
   固定 `server_seed=0` 的理论 full-action 是 388.518，比值应为 **5.1487**。

**✅ 人工决定 `D-INTENSITY`（已作出 2026-09-18）**：**(1)** 当前 `1000` 保留为
明确标记的 `stress` / overload **候选**；**(2)** **返回上游 M1.3f** 为 main scenario
建立有来源、预先批准、不得按结果调节的 intensity。
**`M1.3f-d` 审计已完成，`M1.3f-d-R1` 纯文档返修已通过人工复审**
（`docs/AUDIT_ARRIVAL_INTENSITY.md`）：`1000` **没有任何**独立物理校准（唯一来源
是声明的 `lambda_ref=2000`；Azure trace 只贡献**形状**，template 均值精确为 1）；
可持续任务服务能力 **95.599 work/hour 只是 `server_seed=0`、`T_amb=28°C`、
`access_limit_kw=18`、`base_load=0.05`、无 PV/风电/BESS 下的单点探针**（接入上限
18 kW 把能力压到理论值约 1/4，且随温度/seed 漂移），故 arrival 相对**可持续**能力
约 **20.9242×**、相对**理论计算上限**约 **5.1487×**——**两者都是 seed 0 的示例
探针，不是正式冻结比值**。正式容量只用符号
`DECLARED_SUSTAINABLE_CAPACITY_WORK_PER_HOUR`；`rho` 已拆为 `rho_target`（期望值）
与 `rho_realized`（Poisson realization），main intensity 由 `rho_target` 决定。

**✅ B6-INTENSITY 裁决（D3/D4，2026-09-18）**：`server_seed=0`、train max 温度
**33.2°C**、`access_limit_kw=18.0`、`base_load=0.05`、`max_task_load_per_server=0.80`、
排除 PV/风电/BESS、均匀逐服务器负载 + 现有接入二分公式；声明持续容量
**79.985 work/hour**（原始 ≈79.985193 向下截断）；**`rho_target=0.80`**；
main expected **63.988 work/hour** = **31.994 work/半小时**（`delta_t_hours=0.5`）；
`source_kind=modeled_scenario`（不声称实测）；`rho_realized` 仅诊断、禁止反调；
`1000` 保持 `stress_candidate`；sensitivity `none`。
**下一步 M1.3f-e-a 冻结该 policy（只冻结 policy，不物化新版 arrival）；在 main
intensity 版本化完成之前，不得开始 g-e-b / g-e / g-f。**

**✅ `M1.3f-e-a` 已通过人工复审（2026-09-18）**：canonical policy
`data/manifest/m13f_arrival_intensity_policy_v1.json`
（schema `m1.3fea-b6-arrival-intensity-policy-v1`，SHA-256 `7066a0e1…`，
`materializer_revision 3456f96…`）已冻结六个核心数
**33.2°C / 79.985 work/hour / `rho_target` 0.80 / 63.988 work/hour /
31.994 work/半小时 / `server_seed` 0**；
`source_kind=modeled_scenario`、`empirical_workload_claim=false`；
`rho_realized` 仅诊断、禁止反调；`1000` 保持 `stress_candidate`；sensitivity `none`。
canonical-only（副本 / symlink / 伪造 revision / 篡改 hash / 空目录全部拒绝，无 fallback），
`--verify` 连续 3 次 bytes/hash/mtime 不变。**该 policy 是当前唯一的 main-intensity policy**；
readiness 仍为 `false`。
**⛔ 下一张卡是 `M1.3f-e-b`**（生成新 exogenous 版本、更新 forecast/provenance/refs、
生成新 formal split triad，保留旧 v2/v4 与 stress 资产）；**本卡未物化新版 arrival**，
**不得自行开始 `M1.3f-e-b`**。

**✅ `M1.3f-e-b1` 已执行完成（等待人工复审）**：由已验证的 B6 policy 物化**版本化**
外生驱动表 **v3**——`exogenous_drivers_v3.parquet`
`07b648f0a15db1d8c39838e3e501dafa2f9956155489702e3379cdb775858612`、
`singapore_2024_exogenous_v3.json`
`5f55aaf74185ab8d1da7dea93b5da420791fa394ec1863e23d6e3e2c296214db`、
`m13f_materialization_sources_v4.json`
`360deef7fbd3371c191d039e7416acf43fc85633185c83b4c56a72eefe33c756`；
`materializer_revision = 0e16b29…`。
**PV / wind / carbon 与 v2 逐行相同**；`arrival` 用 **31.994**（shape 复用 v2 的
48-slot template，均值精确为 1，**不继承** B5 的 1000）；诊断值
expected `31.994`／realized `32.0204`／`rho_realized` `0.80066`（`diagnostic_only`，
禁止反调）。**v2 三项资产逐字节未变**；`scenario/exogenous_drivers.py` 等**未修改**。
**⛔ 下一张卡是 `M1.3f-e-b2`**（formal scenario 切换 v3、新 forecast/provenance 绑定、
refs 新版本、`formal_splits_v5`、旧版本 superseded 登记）；**本卡未切换 formal 链**
（仍绑定 v2），**不得自行开始 `M1.3f-e-b2`**，mapper / 训练仍未开始。

**✅ `M1.3f-e-b1-R1` 已通过人工复审（2026-09-19）**：**e-b1 第一轮审核不通过**——
旧 loader 只校验结构与上游绑定，**接受被伪造的嵌套业务语义**
（实测 `FORGED_SEMANTICS_ACCEPTED True 1000.0 False`）。新增**统一生产入口**
`load_verified_v3_bundle()`（`--verify` **只**调用它）：三份产物非 symlink、
**顶层与所有嵌套对象**精确键集合、policy 走正式 loader、v2 shape 按冻结 hash、
source-v4 **由 trusted constants + live hashes 重建**、两份 manifest **共享
`frozen_at_utc`**、**由 canonical 输入重算完整 DataFrame 并与 parquet 逐列逐值一致**、
再**重建 output manifest 并与文件逐字段相等**。因此 arrival 语义、`rho_realized`、
predecessor、readiness **全部由 policy 与重算结果导出**，不信任 JSON 自报。
伪造、**coordinated source/parquet 篡改**、symlink、dirty —— **全部 REJECTED**；
未篡改对照 **ACCEPTED**。最终 v3：parquet `07b648f0…`（**字节未变**）、
manifest `31cb241c…`、source-v4 `73f75cef…`，`materializer_revision = 34f2de0…`。
**上游 hash 全部未变**；`make check` **2520 passed**。
`load_verified_v3_bundle()` 是 v3 的**唯一**可信入口；**v3 三资产为已批准候选**；
**formal 链仍绑定 v2**，readiness **仍为 false**。

**✅ `M1.3f-e-b2-a` 执行完成（等待人工复审）**：新增**独立候选**入口
`build_formal_scenario_b6(...)`（`scenario/formal_scenario_b6.py`）与 **policy-v3**
`data/manifest/singapore_2024_forecast_policy_v3.json`
（`5114c80d…`，`materializer_revision = bf5f7dd…`）。policy-v3 绑定 policy-v2
（**仅** seasonal 规则）、B6 policy、exogenous v3 三项、canonical 三项，
`contract-v9` / `30min` / `period_steps=48`，`supersedes` **只登记**、**不覆盖**，
readiness **保持 false**。候选：seasonal 复用 M1.3e provider、PV/风复用同一物理函数、
carbon `0.402`、**arrival = `template[slot] × 31.994`（期望值，不读 realization）**；
train/validation/test 均可构造；tamper / **symlink** / **旧 v2 fallback** / dirty
全部 **REJECTED**；`--verify` ×3 幂等；`make check` **2547 passed**。
**旧资产 hash 全部未变**；**范围外修改无**。
**⛔ 本卡只是 candidate**，现有 formal 入口**仍**用 policy-v2；
**下一张卡是 `M1.3f-e-b2-b`**（切换正式入口、`refs_v4`、`formal_splits_v5`、
旧版本 superseded 登记、readiness 门禁）——**不得自行开始**；mapper / 训练仍未开始。

**✅ `M1.3f-e-b2-a-R1` 执行完成（等待人工复审）**：**e-b2-a 第一轮审核不通过**
（三项 P1 + 一项 P2）。修复：① `load_verified_policy_v3()` 改为**整体重建比对**
（trusted constants + live hashes + live revision → 逐字段等于文件），
`arrival_forecast_rule` / `seed_policy` / `supersedes` 等 **14 类语义伪造
ACCEPTED → REJECTED**，并有**未篡改接受性对照**；② `B6_FORMAL_SOURCE_PATHS`
扩到 **12** 项（补回 `scenario/splits.py` 等），dirty 与 revision 用**同一集合**；
③ 旧「future mutation」用例**没改任何数据**，替换为**真实**回归：仓库外完整自洽
临时链 + 生产 `build_formal_scenario_b6()`——改 target 五列后
**七条 forecast 逐位相同**，改历史窗口后 **price/load/temperature/pv/wind
全部变化**；④ 原子失败改为走**真实** `materialize_policy_v3()`。
最终 policy-v3 `92670333…`（revision `cda31ab…`）；**上游 hash 全部未变**；
`make check` **2570 passed**；范围外修改**无**。

**✅ `M1.3f-e-b2-a-R2` 已通过人工复审（2026-09-20）**：审核**暂不批准**，
但**生产实现未发现新问题**——只修测试夹具的 pandas 区间越界与文档证据。
缺陷：`_bump` 用 `frame.loc[slice(start, stop)]`，在 RangeIndex 上**两端包含**，
故 target 实改 **5** 行 `11224..11228`、history 实改 **49** 行 `11176..11224`
并**越界进入 target 首行**（两窗口重叠）。修复为
`labels = frame.index[start:stop]`（半开位置切片）后按标签赋值，
并新增「changed index **严格等于** `range(lo, hi)`」与「窗口不重叠」断言。
**修复后**：target **4** 行 `11224..11227`（exact）、history **48** 行
`11176..11223`（exact）、disjoint=True；**target 后七序列逐位相同**、
**history 后五序列全部变化**。**生产实现未改**：`scenario/`/`scripts/`/`data/`/
`.gitignore` 均未修改，policy-v3 **未重新生成**（仍 `92670333…`、
revision `cda31ab…`），受保护资产 hash 未变；`make check` **2571 passed**。
**⛔ 停止**：不开始 `M1.3f-e-b2-b`；mapper / 训练仍未开始。

**R2 追加（Task schema / 可行域 / 固定点守恒）**：mapper 输出必须覆盖
`idc_model.task.Task` 的 **11 个必填字段**（含 `name`、`load_profile`）；
每个任务 `workload ∈ [w_min, w_max]`；分割算法必须**同时**满足每任务上下界与
全局守恒（超出覆盖 **fail closed**，有冻结的 `MAX_TASKS_PER_SLOT`）；
守恒用**固定点整数账本**（float `Task` 只是运行时表示）；
profile 概率等参数**全部**是人工批准的 modeled scenario，**不得**声称由
aggregate train 数据校准。

## 9.1 **B6 arrival-to-Task mapper 已实现（M1.3g-e-b，2026-09-20）**

`scenario/arrival_mapper.py`（**纯函数**）+ `scripts/materialize_b6_arrival_mapper.py`
+ 冻结参数 manifest `data/manifest/m13g_arrival_mapper_v1.json`
（schema `m1.3g-e-b-arrival-mapper-v1`，SHA-256 `eb44608f…`，
`source_revision = 42f00e19…`）。

- 公开入口 `build_arrival_task_stream(split, start=…, horizon=…, seed=…)`
  **只**接受已验证 B6/v3/v5 链；副本 / symlink / 旧 v1–v4 / `refs_v3` /
  伪造 hash·revision / dirty **全拒绝**；
- 每槽**只读该槽 realized aggregate**；**未来 slot mutation 不影响此前 prefix**；
  **forecast 绝不进入 Task truth**；
- **fixed-point `work_unit_scale = 1e6`**：每槽与全 episode 整数账本
  **精确等于**原始 aggregate（float `Task.workload` 仅运行时表示）；
- Task 11 字段完整、`workload ∈ [9.712937937508368, 60.705862109427294]`、
  平坦 `load_profile`、`deadline = 2 ≥ duration = 1`；
  `task_id` 由 canonical 编码导出（**禁止** `hash()`）；`initial backlog` 不入账；
- 实测 train / validation / test 各 48 槽 **48 任务**、守恒 **True**；
  `61 work → 两 Task`；低于下界 / 超 4-task / 零 aggregate **均 fail closed**；
  `--verify` ×3 幂等、零临时文件；CLI **仅** `--help`/`--verify`。

**验收**：mapper **29 passed**；`make check` **2661 passed**、`make smoke` exit 0、
`make train` **exit 2**；既有资产逐字节未变；**范围外修改无**。

> ## ⛔ **mapper 完成即停，等待人工复审。**
> **未开始** g-e-c（env 接线）、g-e-d（回归）、g-f（训练门禁）、训练、评估、M6；
> readiness **仍为 false**。

**回滚**：`git revert <记录提交> cdfd236 47eb01a f9e206c 42f00e1 281b69e 5dfe3c7 5b51986 5844240 c09db56 e3a56d0`
→ `HEAD^{tree} == 7b00836^{tree}`。

**✅ R1（M1.3g-e-b-R1，2026-09-20）**：第一轮审核三处 P1——
① 伪造 manifest 被接受（`FORGED_MANIFEST_ACCEPTED`）；② 改 `priority` 等 Task 字段
后 content hash **不变**；③ 篡改 v5 链后**公开入口**仍接受。已全部修复：
公开入口每次走**唯一** verified chain 并从中取 aggregate；manifest 由 trusted live
inputs + live revision + live `C_IDC_base` + **锚定 refs_v4 冻结时刻**重建并逐字段比较；
content hash 覆盖**全部业务字段**（浮点 `float.hex()`）；三份 v5 全部按 hash 绑定。
manifest `b8a75488…`、`source_revision 53cb0c23…`、`frozen_at_utc` = refs_v4 锚点。
`make check` **2681 passed**；既有资产逐字节未变；**范围外修改无**。

> **账本勘误**：M1.3g-e-b 实际 **13 个**提交（上轮漏列 `86ca2dd`）；完整回滚见
> `docs/WORK_HANDOFF.md` §7AP 与 `docs/task_cards/M1.3g.md` §aq.7。

**✅ R2（M1.3g-e-b-R2，2026-09-20）**：审核指出 `idc_model/task_model.py`
（`C_IDC_base` 的来源）**未**进入 mapper 的 revision/dirty 集合 ⇒ 改它不会使旧
revision 失效。已**仅**把它加入统一 source 集合（**未**改 `task_model.py`、
**未**新增信任规则）；并补**全范围逐槽守恒**回归。实测 **17,568 槽全部覆盖**
（train 10,224 / validation 2,928 / test 4,416，每槽恰 1 个 E 任务，2.00 s）。
manifest `efea87f2…`、`source_revision a8065990…`。`make check` **2687 passed**；
既有资产逐字节未变；**范围外修改无**。

> **R1 账本勘误**：R1 实际 **16 个**提交（上轮写「15 个」——计数错误）。
> 两条完整回滚见 `docs/WORK_HANDOFF.md` §7AQ 与 `docs/task_cards/M1.3g.md` §as.7–§as.8。

## 9.2 **正式 env 注入已接线（M1.3g-e-c，2026-09-20）**

`scenario/env_injection.py` + `envs/idc_price_env.py` 接线：

```python
inj = build_verified_formal_env_injection(split, start=..., horizon=..., forecast_cutoff=...)
env = IDCPriceEnv20D(..., delta_t_hours=0.5, formal_injection=inj)
```

- **唯一**信任链：mapper chain（B6 policy / v3 bundle / `refs_v4` / v5）→
  origin 映射 → mapper Tasks + 整数账本 → B6 causal forecast → episode 切片；
  **不接受**调用方传入 truth/forecast/refs/task stream，非法输入 **fail closed**；
- **formal 分支**：`reset` **不**调用 demo/random；`delta_t_hours=0.5`；
  planned capacity `rate × 0.5` → **work/step**；`lambda_ref = 31.994 work/step`；
  `queue_ref`/`queue_capacity_ref` = 6000（**存量**，不缩放）；
  observation 的 arrival 通道只用 **causal forecast**；`info` 不含完整未来 truth；
- **legacy 路径行为完全不变**（既有测试全过）。

**验收**：新测试 **20 passed**、相关 focused **506 passed**、`make check`
**2707 passed**、`make smoke` exit 0、`make train` **exit 2**（无 fallback/成功 run/
checkpoint）；冻结资产逐字节未变；**范围外修改无**；readiness **仍为 false**。

**回滚**：`git revert 5333d69 9a8ddb8 a204e2a 3656c6b 960c8f0 0d978c1 044ab69`
→ `== 044ab69^{tree}`。

> ## ⛔ **g-e-c 完成即停，等待人工复审。**
> **未开始** g-e-d（回归）、g-f（训练门禁）、训练、评估、M6。

## 9.3 **carbon 观测通道已修复（M1.3g-e-c-R2，2026-09-20）**

**人工复审（卡片 §ax）判定 `g-e-c-R1` 不通过**：1 × P1 + 2 × P3。R1 的
exogenous truth、五类 forecast 中的**四类**、refs、Task reset replay 均已通过。

**唯一阻塞项 P1**：`_get_forecast_features` 的 **carbon 分组读 realized
`carbon_factor_t`**，而读入了正确 causal 值的 `carbon_forecast_t` **无人读取**
（死代码）。审核要求列的是**六**条通道（含 carbon），R1 只完成四条。
`realized == forecast == 0.402` 只是**冻结常量**造成的数值惰性，**不构成语义豁免**。

**修复**：`carbon_src` 与 price / temperature / PV / wind **同型**按 `self.formal`
分流；`step` 的物理 / 成本计算**继续**读 realized；legacy **逐字保持原行为**；
**未改 `step()`**、**未改 `scenario/env_injection.py`**。

**两项 P3 清理**：删除死属性 `arrival_forecast_t`（formal arrival 观测唯一来源仍是
`task_arrival_forecast`）；`formal_refs` 由盲 `setattr` 改为**固定白名单
`EXPECTED_FORMAL_REF_ATTRS`（13 项）+ fail-closed 集合校验**。

**验收**：新测试 **47 passed**；相关 m13f/m13g/env focused **670 passed**；
`make check` **2734 passed**；`make smoke` exit 0；`make train` **exit 2**；
readiness **仍全 false**；范围外修改**无**。

**回滚（均已实测零冲突、逐树一致）**：

~~~bash
# R1 完整回滚（6 提交；勘误此前只到 3a8bfc8 的 4 提交口径）
git revert 1f1f61a ed82aef 3a8bfc8 dd78fac 662eb51 c111a4f   # == 0d74865^{tree}

# R2 完整回滚（4 提交，含复审记录）
git revert a51d7cf 5bab66d 98f4195 1edbade                    # == 1f1f61a^{tree}
~~~

> ## ⛔ **g-e-c-R2 完成即停，等待人工复审。**
> **未开始** g-e-d（回归）、g-f（训练门禁）、训练、评估、M6。

## 9.4 **跨层守恒 / 泄漏 / 重放回归已落地（M1.3g-e-d，2026-09-21）**

**R2 已通过人工审核**（`docs/task_cards/M1.3g.md` §ba，提交 `2a15961`）。

新增 `tests/test_m13ged_arrival_conservation.py`（**27 passed**），四组：
**① 原始 aggregate 守恒**（mapper 整数账本 == `int(raw_aggregate[global_origin+k])×1e6`，
initial backlog 单列，逐步/终点守恒，3 split）、**② 半小时执行量与物理负载**、
**③ 因果性与未来任务数**、**④ 重放**（mapper 确定性、reset 重放、独立 env 动作重放）。

**本轮修复了 4 个真实生产缺陷**（均先红后转绿，仅改 `envs/idc_price_env.py`）：

1. `_loads_from_group_completion` 除以 `C_server` 而非 `C_server × delta_t_hours`
   —— formal 下**任务负载被低估 2×**（半油门完成全部计划量得 `0.2`，应为 `0.4`），
   连带 IDC 功率 / 能耗 / 成本 / 碳被低估；legacy `delta = 1.0` 为恒等变换；
2. `_get_task_pool_features` 的 `total_task_ref` 用 `len(self.tasks)`（含未到达任务）
   → 未来任务数进入 **observation**；
3. `step()` 6 处归一化分母同源 → 当步 **reward** 随未来任务总数变化
   （`0.183257 != 0.166743`，而已到达状态与动作完全相同）；
4. reset `info["total_task_count"]` 含未来任务。

统一以 `_task_count_ref()` 收口：**formal 只数已到达任务，legacy 逐字不变**
（实测 `c_server × 1.0 == c_server`、`_task_count_ref() == len(self.tasks)`）。
**未**修改 mapper 参数、B6/v3 数据、`refs_v4`、`formal_splits_v5`、冻结 manifest、
`idc_model/`、训练 / PPO / 评估 / readiness；**未**丢弃任务、缩放 arrival、
重置队列或放松容量。全仓库无写 `status = "failed"` 的路径，故守恒式**无误差项**。

**验收**：`make check` exit 0（**2761 passed**）；`make smoke` exit 0；
`make train` **exit 2**；`runs/train_real_*` **117 个全失败、0 success**、无 checkpoint；
readiness **仍全 false**。

**回滚（实测零冲突、逐树一致）**：

~~~bash
# 本卡完整范围（5 提交，含记录提交）
git revert 950c131 ffb7cc2 3f836b7 bb5cc9e 2a15961   # == f6734e6^{tree}
~~~

> ## ⛔ **g-e-d 完成即停，等待人工复审。**
> **未开始** g-f（训练门禁）、训练接线、正式训练、评估、M6；readiness **仍为 false**。

## 9.5 **formal info 未来工作量泄漏已收口（M1.3g-e-d-R1，2026-09-21）**

新增 `tests/test_m13gedr1_info_leakage.py`（**5 passed**）。`g-e-d` 已收口
observation / reward / `total_task_count`；本卡补 **info** 仅剩的两处由**整段**
任务计算的出口：

1. reset `_task_scale_info()` 的 `effective_total_workload` / `average_task_workload`
   （改前 `Obtained 306.0` vs `Expected 281.0`）；
2. step info 的 `completion_rate`（改前 `0.42038216560509556` vs `0.4697508896797153`）。

**方法**：两个状态完全相同的 formal env，只在一个里复制一个 `arrival_time`
晚于被测时点的 Task；先证 mutation 生效，再断言两出口不变，并对照当步
`completed_work`（37.0 双侧）与 **reward**（0.3868284143 双侧）相同 —— 证明差异
**只**出现在 info 出口。

**最小实现**：formal 分支只统计**已到达**任务（含 initial backlog）；
新增 `_arrived_available_work(current_time)`；step 的 `completion_rate` 用**本步
时点 `t`** 而非已自增的 `current_step`；终点与整段一致。**未改** reward / 任务执行 /
terminal settlement；legacy 走**原代码路径**（逐字不变）。

**验收**：原 `g-e-d` **27 passed** 继续通过；相关 focused **840 passed**；
`make check` exit 0（**2766 passed**）；`make smoke` exit 0；`make train` **exit 2**；
`runs/train_real_*` **120 个全失败、0 success**、无 checkpoint；readiness **仍全 false**。

**回滚（实测零冲突、逐树一致）**：

~~~bash
# 本卡完整范围（4 提交，含记录提交）
git revert fa82d78 2afb5a5 4d532bb 641cb19   # == bc22f01^{tree}
~~~

> ## ⛔ **g-e-d-R1 完成即停，等待人工复审。**
> **未开始** g-f（训练门禁）、训练接线、正式训练、评估、M6；readiness **仍为 false**。

## 9.6 **正式训练入口预检已接线（M1.3g-f-a，2026-09-21）**

新增 `tests/test_m13gfa_train_entry_gates.py`（**14 passed**）。
**只接 preflight，不实现训练循环。**

**preflight 顺序**：① `load_verified_split_manifest_v5("train")` →
② `load_verified_mapper_chain("train")`（复用**同一条** verified public 链）→
③ 由 **candidate origin** 推导 start → ④ `build_formal_scenario_b6` →
⑤ `validate_forecast_purpose(bundle, purpose="training")` → ⑥ readiness（**最后**）。
`train.py` **不重写**任何 hash 校验器（有结构性守卫）。

**origin 来源（关键）**：v5 train `candidate_origins.start = 48`；
`time_range.start`（`2024-01-01T00:00:00+08:00`）是**本地行 0**，**不是** episode
start。入口推导 `2024-01-02T00:00:00+08:00` 并**往返校验**回 origin 48。

**改前缺陷**：硬编码 `start="2023-01-01"`（M1.2 时代），从不读 v5
`candidate_origins` / `readiness`，从不调用 training purpose gate，归因写 M1.2。

**make train（实测）**：**exit 2**，`TrainEntryError: M1.3 发布门禁未放行…`
（`readiness={formal_env_ready: False, formal_training_ready: False}`）。
归因中**不再有** `M1.2` / `2023-01-01`。失败 run `status=failed`、
`synthetic=False`、claims 全 false；`runs/train_real_*` **121 个全失败、0 success**、
无 checkpoint；`formal_splits_v5` 三份 hash **逐字节未变**；readiness **仍全 false**。

**旧测试迁移**：`test_m54a_train_entry.py` 的 M1.2 措辞断言 → M1.3 归因断言，
**未删除、未放宽**。

**⚠️ `make check` 捕获的两处自身缺陷**：① mypy 的 `SplitName` Literal 不兼容；
② 初版 `readiness.get(name)` 违反「不用 `.get()` 兜底」红线（m51c 结构性守卫捕获）。

**验收**：`make check` exit 0（**2779 passed**，ruff/mypy 全通过）；`make smoke`
exit 0；`make train` exit 2。

**回滚（实测零冲突、逐树一致）**：

~~~bash
# 本卡完整范围（10 提交，含记录提交）
git revert f2457fe a23ab0e 06d8b33 c91b8e7 83051e8 53f307c 96434db a314e2c 9acdc33 48cd438
# == 5adb019^{tree}
~~~

> ## ⛔ **g-f-a 完成即停，等待人工审核。**
> **未开始** f-b、f-c、正式训练、M6；readiness **仍为 false**。

## 9.7 **冻结 mapper 参数 manifest 已纳入 preflight（M1.3g-f-a-R1，2026-09-21）**

**f-a 人工复审不通过（1 × P1）**：real preflight 调用
`load_verified_mapper_chain("train")`，但它只验证 **B6 policy / v3 bundle /
`refs_v4` / v5 split**，**不调用** `load_verified_mapper_manifest()` ——
后者才是 `m13g_arrival_mapper_v1.json` 的唯一验证入口（canonical 位置 / 冻结参数 /
来源 hash / live revision / 锚定 refs_v4 冻结时刻）。故 mapper 参数 manifest 被篡改
或 revision 陈旧时，`make train` **不会发现**。

**修复**：在 readiness 检查之前新增一行**现有公开** loader 调用
（`train.py` **+9 / −1** 行）；**未**复制参数 / hash 逻辑、**未**重物化、
**未**改 mapper manifest；`scenario/` **零改动**。

**调用证据（实测）**：未篡改链上调用 **1 次**，`path=None` → canonical 路径，
`schema=m1.3g-e-b-arrival-mapper-v1`、`approved_decision_id=M1.3f-e-b-d`。
**合法链对照**：合法 manifest → 抵达准确的 **M1.3 readiness 阻塞**（`exit 2`）；
篡改 manifest → 在 readiness 门**之前**失败并透出 `ArrivalMapperError`。

**验收**：新测试 **17 passed**（原 14 + 3）、m54a **30 passed**；`make check` exit 0
（**2782 passed**）；`make smoke` exit 0；`make train` **exit 2**；
`m13g_arrival_mapper_v1.json` `efea87f2…` 与 v5 三份 hash **逐字节未变**；
readiness **仍全 false**；`runs/train_real_*` **126 个全失败、0 success**、无 checkpoint。

**回滚（实测零冲突、逐树一致）**：

~~~bash
# 本卡完整范围（5 提交，含记录提交）
git revert cbec1a3 57d9cf1 e755951 e1a79f8 777a5da   # == 7de03c0^{tree}
~~~

> ## ⛔ **g-f-a-R1 完成即停，等待人工复审。**
> **未开始** f-b、f-c、正式训练、M6；readiness **仍为 false**。

## 9.8 **env-ready 发布边界已审计，推荐独立发布产物（M1.3g-f-b-a，2026-09-21）**

**纯文档卡**，交付物 = `docs/task_cards/M1.3g.md` **§bn**。f-a-R1 已通过人工审核
（§bl，`e087a05`）。**等待人工选择发布方案**。

**关键事实（只读实测）**：

- v5 校验器 `_require(readiness == READINESS)` **硬要求 both-false**；readiness 属于
  `build_split_manifest_v5` 的**重建比对**；`materializer_revision` 绑定
  `B6_SPLIT_SOURCE_PATHS`（9 路径）的 git revision ⇒ **改 readiness 必然使现有 v5 失效**，
  故 v5 **字节不得改写**（现为 `8608372f` / `3f16ad3a` / `aaacd459`）。
- **绑定单向、无环**：mapper manifest（`efea87f2…`）以 path+sha256 绑定三份 v5；
  而 v5 的 `inputs` 九角色**不含** mapper manifest。
- `forecast_policy_v3` / `exogenous_drivers_v3.parquet` / `m13f_arrival_intensity_policy_v1`
  **同时**是 v5 `inputs` 且被 mapper 绑定 ⇒ 在上游 manifest 翻 readiness 会双重破绑。
- **formal env 可信链已闭合**（g-e-a/b/c/d + f-a 各轮返修均通过）；
  但 `train.py` **无训练循环** ⇒ `formal_training_ready` **不得**为 true。

**两方案**：**A** 新版本 v6 triad + `m13g_arrival_mapper_v2.json`（触及 9 路径 ⇒ v5
立即不可读、爆炸半径最大、**且买不到训练能力**）；**B（推荐）** 独立可验签发布产物
（`configs/release/idc_formal_env_release_v1.json` + `scenario/env_release.py`），以
`binds` 逐字节绑定 v5 三份 + `refs_v4` + mapper manifest，**完全不触碰冻结资产**；
`train.py` 最终门改**合取**：v5 readiness 必须仍**精确等于** both-false **且**发布产物
验签声明 `formal_env_ready=true`、`formal_training_ready=false`。

**排除**：运行时覆盖 v5 readiness / 未经验证的配置开关 / 只在 `train.py` 忽略 false。

**下一张实施卡边界**：先红测试、允许文件、物化顺序、幂等与回滚边界见 §bn.4；
**不得**把正式训练成功作为 env-ready 发布的验收条件。

**验收**：`make check` exit 0（**2782 passed**，与上一卡一致）；冻结资产 hash
**逐字节未变**；范围外修改**无**。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert d0a0876 8b411fd e087a05    # == 14e6aec^{tree}
~~~

> ## ⛔ **f-b-a 完成即停，等待人工选择发布方案。**
> **未自行开始** f-b 实施、f-c、正式训练、M6；readiness **仍为 false**。

## 9.9 **formal env 独立发布产物已落地（M1.3g-f-b-b，2026-09-21）**

方案 B 已裁决（§bo）并实施。新增 `scenario/env_release.py`、
`scripts/materialize_env_release.py`、`configs/release/idc_formal_env_release_v1.json`
（唯一 canonical，`.gitignore` 仅加 2 行最窄放行）。

~~~text
产物 sha256      = 017575dc827043a5926d9b3b1057d91ffdead7b406dd115e97218f51f8c6adae
release_revision = 913754e6071b80070f96e51599894936e564ba77
readiness        = { formal_env_ready: true, formal_training_ready: false }
binds            = v5 三份 8608372f/3f16ad3a/aaacd459 + refs_v4 b5b64ef2 + mapper efea87f2
~~~

产物**无时间戳字段**；验签输入全部来自 live 文件 hash + live git revision。
**v5 三份字节未变、readiness 仍严格 both-false** —— 发布批准**只**来自独立产物。

**`train.py` 合取门**：⑥a v5 readiness 必须**严格保持** both-false（不得运行时
覆盖）；⑥b 独立产物必须验签且 `formal_env_ready=true`；⑥c `formal_training_ready=false`
**继续阻断训练**。**未**使用运行时覆盖 / 未验签开关 / 忽略 v5 false 中的任何一种。

**`make train`（实测）**：**exit 2**，归因为
`M1.3 训练未放行：发布产物声明 formal_training_ready=false…`（**env 已发布**）。
失败 run `status=failed`、`synthetic=False`、claims.trained=false；
`runs/train_real_*` **128 个全失败、0 success**、无 checkpoint。
**env 发布未被当作训练成功的理由。**

**验收**：新测试 **20 passed**、f-a **17 passed**、m54a **30 passed**；
`make check` exit 0（**2802 passed**）；`make smoke` exit 0。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert 3311943 660290b 57bf5d1 3f4b070 24dbf07 913754e 3009771 \
           e39425f bc0933c acb0448 d0c9e42 66a8da4 9df3ce0
# == f0d5407^{tree}
~~~

> ## ⛔ **f-b-b 完成即停，等待人工复审。**
> **未开始** f-c、正式训练、M6；`formal_training_ready=false` **继续阻断训练**。

## 9.10 **formal 真实 rollout 闭环已验证（M1.3g-f-c-a，2026-09-21）**

新增 `tests/test_m13gca_formal_rollout.py`（**12 passed**）与只读
`scripts/probe_formal_rollout.py`。链路：verified train v5 candidate origin
（本地行 48 → `2024-01-02T00:00:00+08:00`）→ `build_verified_formal_env_injection`
→ `IDCPriceEnv20D(formal_injection=…)` → `collect_rollout` → `RolloutBuffer`。

**⚠️ 本卡没有真实先红**：五组拟定断言在**实现前全部通过** —— formal 真实
rollout 闭环本就可用，本卡**未暴露任何生产缺陷**。按仓库纪律全部登记为
**回归守卫**，未人为制造失败。

**probe 实测**：`formal=true`、`horizon=8`、`steps_collected=6`、`obs_dim=200`、
`action_dim=21`、`contract_version=contract-v9`、`env_seed=0`、
`policy_rng_source=explicit_generator`、`ledger_micro_sum=250000000`、
`initial_backlog_work=50.0`、`reward_sum=0.1785`、`carbon_emission_sum_kg=14.004`、
`raw_equals_exec=true`、claims 三项全 false。

**验证要点**：正式注入逐位进入 env；buffer 字段来自真实 transition（与
`evaluate_raw_actions` 重算、与 `env.step` 返回值逐步对照一致）；同 seed + 同
动作路径**逐位可重放**；monkeypatch demo/random 生成器抛错后仍跑通；
未来真值不可见（含反向控制）。补两条守卫：`corrector_on=True` 下 exec 与 raw
**真实分离**；超长 rollout 提前终止且不丢 transition。

**§bs.6 越界条件未发生**：无需修改 `envs/` / `env_injection.py` / `train.py`，
**无需**发布产物升级，**未**触碰 `ENV_RELEASE_SOURCE_PATHS`，**未**重物化 v1。

**验收**：`make check` exit 0（**2814 passed**）；`make smoke` exit 0；
`make train` **exit 2**（training 未放行）；`runs/train_real_*` **133 个全失败、
0 success**、无正式 checkpoint；**未**写 `trained=true`。资产逐字节未变
（v5 三份 / `refs_v4` / mapper manifest / 发布产物 v1 `017575dc`）。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert c00af0d e720e7c 38d9632 8fc47d8    # == f122299^{tree}
~~~

> ## ⛔ **f-c-a 完成即停，等待人工复审。**
> **未开始**正式训练或 M6；`formal_training_ready=false` **继续阻断训练**。

## 9.11 **PPO clipped actor objective 核心已落地（M1.3g-f-c-b，2026-09-21）**

新增 `safe_rl_v2/ppo_objective.py`（**纯数值计算**）与
`tests/test_m13gcb_ppo_objective.py`（**20 passed**）。

~~~text
ratio             = exp(new_raw_log_prob − old_raw_log_prob)
advantage         = A_reward − λ_business·A_business − λ_carbon·A_carbon
clipped surrogate = min(ratio × A, clip(ratio, 1−ε, 1+ε) × A)
actor loss        = −mean(clipped surrogate)
~~~

**先红**：`ModuleNotFoundError: safe_rl_v2.ppo_objective` × 20（模块不存在）。

**数学微例**：`ratio = 2.000000`（`ln2`）；优势 `= −1.000000`；ε=0.2 下
`ratio=10,A=+2 → +2.4000`（上界保护）、**`ratio=10,A=−2 → −20.0000`（负优势
不被 clip 救回）**、`ratio=0.1,A=−2 → −1.6000`。

**红线执行**：API **无**新 log-prob / `exec_action` 入参（有结构性守卫）；
新 log-prob 只能由 `raw_action` 现算；`clip_epsilon` keyword-only **无默认值**；
乘子由调用方给出，本模块不选择/不更新。有测试证明传 `exec_action` 结果必须不同。

**验收**：`make check` exit 0（**2834 passed**）；`make smoke` exit 0；
`make train` **exit 2**（training 未放行）；`runs/train_real_*` **135 个全失败、
0 success**；无正式 checkpoint；**未**写 `trained=true`。

**资产未变**：v5 三份 / `refs_v4` / mapper manifest / **发布产物 v1 `017575dc`
未重物化**（`safe_rl_v2/ppo_objective.py` 不在 `ENV_RELEASE_SOURCE_PATHS` 内，
故 v1 的 `release_revision` **未失效**）。改动仅 3 个文件。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert a74f0e3 67de152 12af939 e79dc7b ae2a836   # == f960d31^{tree}
~~~

> ## ⛔ **f-c-b 完成即停，等待人工复审。**
> **未接**正式训练循环或 M6。
> **后续修改 `train.py` 会使发布产物 v1 的 revision 失效，必须另开版本迁移卡。**

## 9.12 **基于 formal buffer 的单次 PPO 更新已落地（M1.3g-f-c-c，2026-09-21）**

新增 `safe_rl_v2/ppo_update.py` 与 `tests/test_m13gcc_ppo_update.py`（**10 passed**）。
把三头 target、乘子与 f-c-b 的 clipped actor objective 接成**一次**更新。

~~~text
num_transitions=4  optimizer_steps=1
loss_total=25.7303371429  actor_loss=1.1115287542  critic_loss_total=24.6188087463
  reward MSE=0.0737459958  business MSE=0.0152603080  carbon MSE=24.5298023224
grad_norm_actor=51.3125495911  param_delta_norm=1.577241300083e-01
multipliers {0.5, 0.25} -> {0.45, 0.23936910577118398}（step 后更新）
claims 三项全 false
~~~

**红线**：old log-prob 与三头优势一律 `detach()`；新 log-prob 只由 `raw_action`
现算（corrector 开启时**不**用 `exec_action`）；空 buffer / 非有限损失**明确失败
且不 step**。生产代码**无**测试专用开关。

**数学发现**：`ratio → +inf` **不**产生非有限损失（`min` 的 `clip` 分支界定住了），
故非有限用例改用 `NaN`。

**验收**：`make check` exit 0（**2844 passed**）；`make smoke` exit 0；
`make train` **exit 2**；`runs/train_real_*` **137 个全失败、0 success**。
资产未变（v5 三份 / `refs_v4` / mapper manifest / **发布产物 v1 `017575dc`
未重物化**）。改动仅 3 个文件。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert 7b5afcb 3801f93 e40773d 58092b0 9c53071 a44d335 b1482c5 b7b0ff1 d28ec13
# == 34f63ce^{tree}
~~~

> ## ⛔ **f-c-c 完成即停，等待人工复审。**
> **未接**正式训练循环或 M6。
> **后续修改 `train.py` 会使发布产物 v1 的 revision 失效，必须另开版本迁移卡。**

## 9.13 **单次 PPO 更新验收证据已修正（M1.3g-f-c-c-R1，2026-09-22）**

f-c-c **复审不通过**（1 × P1 + 2 × P2），本卡只修证据与文档，**不改功能语义**。

**P1（唯一先红）**：`grad_norm_actor` 误标为 actor 梯度，实为**全参数**范数。

~~~text
改后：grad_norm_actor = 17.251099（只 actor 与 log_std，5 参数逐一非零）
      grad_norm_total = 51.312550（如实命名，含 critic）
~~~

**P2-a（重要数学更正）**：`ratio → +inf` 的后果**取决于优势符号** ——
正优势 `→ min(inf, 1.2×A) = 1.2×A` **有限**；负优势 `→ min(−inf, 1.2×A) = −inf`
**非有限**，由入口在 step 前拒绝。`§bz.5` 原结论**只在正优势时成立**，已更正。

**P2-b**：两文件末尾空行已删（验收改用 `git diff --check cc84c63..HEAD`）；
f-c-c 卡范围实际改动 **5 个文件**，账本已更正。

**单头扰动（回归守卫，改前已绿）**：不 step，只改一个输入字段 ⇒ 只有对应头的
target/MSE 变化，另两头逐位不变。

**验收**：**18 passed**；`make check` exit 0（**2852 passed**）；`make smoke` exit 0；
`make train` **exit 2**；`runs/train_real_*` **139 个全失败、0 success**；
`git diff --check cc84c63..HEAD` 空。资产未变（发布产物 v1 `017575dc` 未重物化）。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert d614bdc 09e4525 c5ce734 f24e801 17197e4   # == cc84c63^{tree}
~~~

> ## ⛔ **f-c-c-R1 完成即停，等待人工复审。**
> **未接**正式训练循环或 M6。
> **后续修改 `train.py` 会使发布产物 v1 的 revision 失效，必须另开版本迁移卡。**

## 9.14 **双批次 formal 更新连通性已验证（M1.3g-f-c-d，2026-09-22）**

新增 `safe_rl_v2/ppo_two_batch.py` 与 `tests/test_m13gcd_two_batch.py`（**7 passed**）。
两个**不同** formal episode（verified train v5 candidate origins **48** / **96**），
**同一** policy / optimizer / Lagrangian / **连续**采样 RNG；每批
`collect_rollout → single_ppo_update`。

~~~text
批 0: origin=48 2024-01-02  transitions=3  loss=15.001939  param_delta=1.556808e-01
批 1: origin=96 2024-01-03  transitions=3  loss=5.846148   param_delta=1.322283e-01
total_transitions=6  optimizer_steps_total=2  lagrangian_updates_total=2
claims 三项全 false
~~~

**连续性证据**：两批 `policy_state_digest` **不同**、`gen_state` **不同**
（RNG **前进**未重播种）；批 2 的 `old_raw_log_prob` 对应**批 2 采集时刻**的策略
（用批 1 采集时刻重算则不同）。

**验收**：`make check` exit 0（**2859 passed**）；`make smoke` exit 0；
`make train` **exit 2**；`runs/train_real_*` **141 个全失败、0 success**。
资产未变（发布产物 v1 `017575dc` 未重物化）。改动仅 3 个文件。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert f01f7a4 04f5476 19f8ec9 949448c b3487f7   # == 60c5e70^{tree}
~~~

> ## ⛔ **f-c-d 完成即停，等待人工复审。**
> **未接**正式训练入口或 M6。
> **后续修改 `train.py` 会使发布产物 v1 的 revision 失效，必须另开版本迁移卡。**

## 9.15 **双批次 probe 参数所有权已修正（M1.3g-f-c-d-R1，2026-09-22）**

f-c-d **复审不通过**（1 × P1）：probe **自行构造** policy / `Adam(lr=1e-3)` /
`Lagrangian` 并**自行播种** generator，等于替调用方冻结训练配置。

**修复**：`run_two_batch_probe(policy, optimizer, lagrangian, generator, *, …)`
四个对象改为**必填位置参数**；probe 只在其间**持续使用**，不构造、不播种。
测试中的数值**明确标注**为示例输入，**不是**正式训练超参数。

**所有权证据**：对象同一性四项 `is True`；仅改调用方 lr ⇒ 两批 `param_delta_norm`
与最终状态**都变**；仅改调用方 budget ⇒ 乘子轨迹**变**；内部构造函数与
`torch.optim.Adam` 被 monkeypatch 抛错后仍跑通；`torch.Generator` 不可
monkeypatch ⇒ 补**源码级守卫**。

**验收**：**13 passed**；`make check` exit 0（**2865 passed**）；`make smoke` exit 0；
**发布产物 v1 `--verify` exit 0**（`017575dc…`，未重物化）；`make train` **exit 2**；
`runs/train_real_*` **143 个全失败、0 success**。改动 **5 个文件**。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert 7ef3c60 6a28606 a5fc1f1 ebbfc99 2b8f748 7b32a9e   # == d9e6003^{tree}
~~~

> ## ⛔ **f-c-d-R1 完成即停，等待人工复审。**
> **未接**正式训练入口或 M6。
> **后续修改 `train.py` 会使发布产物 v1 的 revision 失效，必须另开版本迁移卡。**

## 9.16 **双批次 checkpoint/resume 等价性已验证（M1.3g-f-c-e，2026-09-22）**

新增 `tests/test_m13gce_two_batch_resume.py`（**10 passed**）；
`ppo_two_batch` 小幅拆分出 `run_single_batch`，并加**测试用**适配
`save_two_batch_checkpoint` / `resume_two_batch_checkpoint`（复用现有
`VersionedCheckpoint`）。

~~~text
批2 observation / raw_action / old_log_prob / batch_digest  digest 全相同
批2 loss        连续=5.846147537231      恢复=5.846147537231
批2 param_delta 连续=1.322283230107e-01  恢复=1.322283230107e-01
批2 乘子相同；批2 后 RNG 状态 digest 相同
恢复的 optimizer 已推进；恢复的 Lagrangian 更新次数 = 2
~~~

**反空洞**：重播种 ≠ 真实恢复；零状态跑批 2 的 digest ≠ 连续批 2 digest。
**契约拒收**：contract_version / action_dim=23 / obs_dim=280 / schema_hash /
缺 metadata / 缺 next_start 一律拒绝。

**⚠️ 边界**：**只**证明**批次边界**恢复，**不**声称中途恢复 / 正式训练 / 收敛。
checkpoint 只写测试 `tmp_path`。

**验收**：`make check` exit 0（**2875 passed**）；`make smoke` exit 0；
**发布产物 v1 `--verify` exit 0**（未重物化）；`make train` **exit 2**；
`runs/train_real_*` **145 个全失败、0 success**。改动 **5 个文件**。

**⚠️ 一次 flake（非本卡缺陷）**：首次 `make check` 有 1 项失败
（`test_m51c_..._independent_runs[True]`，corrector MILP 墙钟预算，与既有
`test_m54g` 同类）；隔离复跑 2 次均通过，该文件不导入 `ppo_two_batch`；
不并发负载后重跑 **exit 0 / 2875 passed**。

**回滚（卡范围 5 提交，实测零冲突）**：

~~~bash
git revert a95bc16 2ae7bea b1c5104 8ae456e 7e9c22f   # == 6cff8c6^{tree}
~~~

> ## ⛔ **f-c-e 完成即停，等待人工复审。**
> **未接**正式运行入口或 M6。
> **后续修改 `train.py` 会使发布产物 v1 的 revision 失效，必须另开版本迁移卡。**

## 9.17 **批次边界状态等价证据已补齐（M1.3g-f-c-e-R1，2026-09-22）**

**是否发现生产缺陷：否** —— 新增断言在当前实现上直接通过，如实登记为
**「验收证据补齐」**，未人为制造先红；`ppo_two_batch.py` **未改**。

~~~text
边界（批2前）：policy / optimizer / lagrangian / generator 逐项精确相同 True ×4
              optimizer state 条目=9、param_groups=1；next_start=2024-01-03T00:00:00+08:00
最终（批2后）：四项逐项精确相同；Adam step={2.0}；Lagrangian updates=2
非空洞性：边界 != 初始、边界 != 最终、全新 optimizer state 为空
~~~

**比较器严格性**：`_assert_exact` 逐项精确比较嵌套 `state_dict`；灵敏度测试证明
单元素 log_std 扰动、键集合差异、Adam step 标量、Lagrangian multiplier、RNG 字节
差异**全部被抓到**。

**名实不符测试已修正**：原 `test_resume_restores_the_pre_batch2_object_state`
实检批 2 **之后**的状态 ⇒ 改为真正检查恢复边界并重命名
`test_resume_writes_state_into_freshly_constructed_objects`。原有测试全保留。

**验收**：**14 passed**（resume 档）+ **13 passed**（f-c-d 档）= **27 passed**；
`make check` exit 0（**2879 passed**）；`make smoke` exit 0；
**发布产物 v1 `--verify` exit 0**（未重物化）。**training 仍未放行**（`make train` exit 2）。
改动 **4 个文件**。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert 3efb6a7 530e51c   # == fce9553^{tree}
~~~

> ## ⛔ **f-c-e-R1 完成即停，等待人工复审。**
> **未接**正式训练入口或 M6。

## 9.18 **边界快照别名缺陷已修（M1.3g-f-c-e-R2，2026-09-22）**

f-c-e-R1 **复审不通过**（1 × P1，**证据缺陷**）：`_full_state` 直接用
`optimizer.state_dict()`，其 `state` 内层张量（`exp_avg` / `exp_avg_sq` / `step`）
是**活引用**，被批 2 的 `step()` **原位改写** ⇒「批 2 前」的对照实为
「批 2 后 vs 批 2 后」，**空洞**。

~~~text
先红：快照被批 2 原位改写：实际 {2.0}；optimizer.state[0].exp_avg 与活状态共享存储
修复：_full_state 经 _deep_snapshot() 递归取快照（张量 detach().clone()）
~~~

**边界 step=1 / 最终 step=2（修复后实测）**：

~~~text
边界（批2前）：Adam step = [1.0]，Lagrangian updates = 1，四对象逐项精确相同 ×4
最终（批2后）：Adam step = [2.0]，Lagrangian updates = 2，四对象逐项精确相同 ×4
并加显式 step 锚点，防止「两次误比最终状态」
~~~

**验收**：resume 档 **16 passed** + f-c-d 档 **13 passed** = **29 passed**；
`make check` exit 0（**2881 passed**）；`make smoke` exit 0；
**发布产物 v1 `--verify` exit 0**。**生产代码零改动**；冻结资产未变。
未重复执行 `make train`（测试/文档卡），**training 仍未放行**（仍 exit 2）。

**回滚（先验证后登记，实测零冲突）**：

~~~bash
git revert 31797dc 747d297 52b7395 1999fee   # == 43ddcde^{tree}
~~~

> ## ⛔ **f-c-e-R2 完成即停，等待人工复审。**
> **未接**正式训练入口或 M6。

## 10. 后续顺序

~~~
M1.3g-e-a  arrival-to-task 只读审计 / 人工选择
      ↓
M1.3g-e    formal env 注入、真实物理序列、0.5h 对齐、task mapping
      ↓
M1.3g-f    train entry 正式门禁、purpose 验证、拒绝 synthetic/oracle/旧 manifest
      ↓
正式训练证据与训练后评估卡
      ↓
M6（仅此后）
~~~

真正 g-e 的硬要求：

- formal delta_t_hours=0.5；legacy 默认 1.0 不改。
- formal env 注入 canonical/exogenous truth 与 refs_v3；不得用默认曲线、全零 fallback、按 validation/test 重算 refs。
- observations 使用 build_formal_scenario 的 causal forecast。
- 改 envs/idc_price_env.py 的 step 前必须有失败测试。
- aggregate arrival 与离散 Task 必须是同一物理语义。
- g-f 才能将 purpose gate 真正接入 safe_rl_v2/train.py；现有 train 遗留旧 2023-01-01/M1.2 归因，不能以伪造字段绕过。

## 11. 当前验收预期

最近 M1.3g-c-R2 报告：

~~~
focused g-c: 81 passed
m12/m13/contracts: 1278 passed
make check: 2458 passed, exit 0
make smoke: exit 0
make train: exit 2（正式链未接线；不回退 synthetic）
~~~

新对话先做低风险复核：

~~~
uv run pytest tests/test_m13gc_scenario_manifests.py -q
uv run python scripts/materialize_singapore_scenario_manifests.py --verify
git diff --check
git status --short
~~~

make train 当前不能被解释为数据缺失：formal triad 已存在。它是 g-e/g-f 未接线的预期 fail-closed 状态；不得回退 synthetic、不得创建 checkpoint。

## 12. 冻结资产 hash

~~~
d4e24d6f4e00408b8acf8e7ed7bd570db5db0eec6d8762284e97bbafdad2a7a9  data/manifest/singapore_2024.json
e6484d6b050f100234a46811be30061f477bcc053b285854da5a882d2753b667  data/manifest/singapore_2024_half_hour.json
a096535fcdec81534f8cc05671d34d879a7e9517d06be789510dea586149af27  data/manifest/singapore_2024_splits.json
ef4dd58a88dcb34fd75690324f957a5f87d2fd7d1b7bf6afbcb528ae9b719e08  data/manifest/singapore_2024_forecast_policy_v2.json
640f26cda94b3479049fdbee56f05e1546a24fc3ecdf6286674c4fb415b484b9  data/manifest/singapore_2024_exogenous_v2.json
4203b4f399ee6433bfcdf63fa94ddd03a56a7bd1e6add45f697c3bec804da1b6  data/manifest/m13f_materialization_sources_v3.json
dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd  data/processed/singapore_2024/half_hour.parquet
11d322b2919e2180b596e6b02614acafdb3ee8d63682ae74e5a3ee1dbc8b92cf  data/processed/singapore_2024/exogenous_drivers_v2.parquet
ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f  configs/frozen_refs/refs_v3.json
0ee774e4e8f09feb1b62a9449e586d4c4bb1803bc2245a1ae4a18c2577d2a5e2  data/manifest/formal_splits_v3/train.json        (superseded)
3ac19480143b97aada0ed39ecc7b357480d0d8e991ed0afb68eced2d665a6d7a  data/manifest/formal_splits_v3/validation.json   (superseded)
62b91d0aca37c0c981db2d690146cc5486511ce51e88f051715bcf9df9cb4573  data/manifest/formal_splits_v3/test.json         (superseded)
215c20968be1b71aa5d5d4b1d22e6cf7ecbd0eb4c802d361dca061a1fbf082cb  data/manifest/formal_splits_v4/train.json
 a69cddaf282f04ebf4277dbe84e7a5d66950fca68054aacb07a811786945c258  data/manifest/formal_splits_v4/validation.json
829a0f12f042c34be43cc42891ed70ee0588939e672026e299382043859c80bf  data/manifest/formal_splits_v4/test.json
~~~

## 13. 容易犯的错误

- 不引用/覆盖 superseded 的 policy v1、exogenous v1、refs v2、formal split v1/v2。
- formal split 文件存在不等于 formal training ready；v4 readiness 仍为 false。
- 不把 carbon 0.402 写成半小时实测。
- 不把 arrival Poisson truth 当 formal forecast。
- 不把 national IGS 改称 local PV，也不把 ERA5 风速称为风电功率。
- 不宽松化 validator、自动 coercion，或允许 caller expected hash/frame/kwargs。
- 不改历史冻结 manifest 补字段；需要新 schema 时用新路径物化，旧证据保留。
- v1/v2/v3 均为 superseded，不得进入 g-e/g-f；只有审计确认真实缺陷才开新版本。
- 不得在 mapper 中缩放 arrival 或改变 arrival/service 比（改前 `arrival_scale`
  已被否决）；要改场景强度只能回上游 M1.3f 版本化。
- 不得把「缩放后 truth」当作守恒；守恒式必须对**原始 aggregate**成立。

## 14. 新对话首条消息模板

请先读取 AGENTS.md、docs/IMPLEMENTATION_PLAN.md、docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md、docs/WORK_HANDOFF.md 和 docs/NEW_CONVERSATION_HANDOFF.md。当前分支为 p4-safeppo-m51a-rollout-contract-m12-integration，HEAD 以交接文档为准。不要启动 M6 或正式训练；先复核工作树、M1.3g-c focused 测试及 **v4** triad 的 --verify，随后复核 M1.3g-e-a **R2** 的审计文档（`docs/AUDIT_ARRIVAL_TO_TASK_MAPPING.md`）与 **§7AB**。**在 `D-INTENSITY` 与 D1–D11 裁决完成、且人工明确放行之前，不要实现 g-e-b / g-e / g-f。**
