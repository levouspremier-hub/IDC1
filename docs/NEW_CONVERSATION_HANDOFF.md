# IDC 项目完整交接（新对话起点）

> **⚠️ 当前入口已迁移：** 本文件是累计历史账本，顶部快照和末尾“下一步”均可能
> 早于当前 HEAD。新 ChatGPT 对话请先读
> [`docs/CHATGPT_HANDOFF_CURRENT.md`](CHATGPT_HANDOFF_CURRENT.md)，再按需从本文追溯历史。

历史快照起点：2026-09-17（Asia/Shanghai）。本文保留累计过程；当前入口见上方链接，逐卡证据、精确回滚和全部历史以 docs/WORK_HANDOFF.md、任务卡及 Git 历史为准。

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

## 9.19 **三批 formal 更新连续性已验证（M1.3g-f-c-f，2026-09-23）**

新增 `safe_rl_v2/ppo_three_batch.py` 与 `tests/test_m13gcf_three_batch.py`
（**10 passed**）；`ppo_two_batch.py` 两处最小修改（公开 `deep_snapshot`；
`run_single_batch` 增 `transition_records`）。

~~~text
三个 origin：local 48 / 96 / 144（2024-01-02 / 01-03 / 01-04），互不相同
第三批 transition：13 字段 × 3 条**逐项精确相同**（非 digest、非近似）
边界：Adam step=[2.0]，Lagrangian updates=2（A / B 两侧）
最终：Adam step=[3.0]，Lagrangian updates=3；四对象逐项精确相同 ×4
边界快照独立于批 3 活状态（防 R2 同类别名假绿）
~~~

**验收**：新档 **10 passed**；三档 focused **39 passed**；`make check` exit 0
（**2891 passed**）；`make smoke` exit 0；env release `--verify` exit 0；
禁区零改动；冻结资产未变。未运行 `make train`；**training 仍未放行**。

**开工前收口**：复审方的 ChatGPT 入口文档等三处未提交变更单独提交为 `81fc237`。

**回滚（先验证后登记，实测零冲突）**：

~~~bash
git revert 609d794                         # 实现批次
git revert 609d794 646ab5a a53a87a 81fc237 # 卡范围
~~~

> ## ⛔ **f-c-f 完成即停，等待人工复审。**
> **未接**正式训练入口或 M6。

## 9.20 **回滚证据与账本已修正（M1.3g-f-c-f-R1，2026-09-23）**

**纯文档证据修正卡**，不改代码、不跑正式训练。

**D1｜「完整范围回滚」原为过期快照**：`8cd2772..a5f4a29` 实际是 **7 个提交**
（原写「6 个」是在**较早快照 `541498c`** 上验证的，**遗漏 `a5f4a29` 自身**）。
已在**最终 HEAD** 重验：

~~~bash
git revert --no-edit a5f4a29 541498c ab6ef31 609d794 646ab5a a53a87a 81fc237
# 7 个 revert 零冲突；HEAD^{tree} = 5b37d0d11b08082f27267cb8b445c0bb36024f5c == 8cd2772^{tree} ✅
~~~

**D2｜`runs/train_real_*` 实测 149**（原文「150」系**未经实测的推断**）：

~~~bash
ls -d runs/train_real_* | wc -l              → 149
ls -d runs/train_real_*/report.json | wc -l  → 149
→ 目录 149，含 failure 149，success 0，无 report.json 0
~~~

**D3｜全范围 `diff --check` 现 exit 0**：`docs/CHATGPT_HANDOFF_CURRENT.md` 第 3–5 行
行尾双空格改为 CommonMark 行尾反斜杠（硬换行语义不变、文字未改）。

**验收**：7 提交回滚**逐树一致**；`git diff --check 8cd2772..HEAD` **exit 0**；
`git status --short` **空**；`env release --verify` **exit 0**。改动 4 个文件；
未改 probe / 测试 / 训练入口 / 环境 / 冻结资产 / env release v1。

**回滚（2 提交，实测零冲突）**：

~~~bash
git revert 58f0fb9 f91e4a3   # == a5f4a29^{tree}
~~~

> ## ⛔ **f-c-f-R1 完成即停，等待人工复审。**
> **未接**训练配置审计卡、正式训练或 M6。

## 9.21 **正式训练配置候选审计已出（M1.3g-f-c-g，2026-09-23）**

**只读审计卡**：`docs/AUDIT_TRAINING_CONFIG.md` + `docs/training_config_candidates.json`
（schema `idc-training-config-candidates-v1`，18 条目）。

**来源分级**：F=正式约束 / S=synthetic smoke 示例 / P=probe 示例 / U=未决定 / M=缺失。
**S 与 P 均不得自动升级为正式训练参数。**

~~~text
条目总数 = 18
  F=2  S=5  P=5  U=2  M=1  S/P=1  S/U=1  U/M=1
✅ 可直接用 : 1（求解器确定性选项）
⚠️ 数值正式、启用待裁决 : 1（corrector 0.25 s）
❌ 需人工裁决 : 15；其中无代码位置（缺失）= 1（batch_arrangement）
~~~

**核心结论**：正式训练配置**整体尚未冻结**，**没有任何 PPO 超参数属于 F**；
`batch_arrangement` 与 `cpu_backend` **无代码位置**，需先补设计。

**逐项可定位**：11 域均标 `文件:行`，18 处行号**逐条核对通过**。

**与冻结约束的关系**：`refs_v4` 供归一化参考值（非超参数）；verified train split
限定 `candidate_origins=[48,10224)`；`delta_t_hours=0.5` 决定 γ 时间语义；
`corrector 0.25 s` 为 M5.4i 已审核生产默认。

**验收**：两处 `git diff --check` exit 0；`git status --short` 空；
`env release --verify` exit 0。**禁区零改动**。未运行正式训练、未改 readiness。

**回滚（实测零冲突、逐树一致）**：

~~~bash
git revert 6d5f1c4 cbabfb0 6fb10cf
# == 5e7f5ed^{tree} = e2c5ff9ec9d92c842e230690eec7f92cab97809f
~~~

> ## ⛔ **f-c-g 完成即停，等待人工裁决。**
> **未接**冻结卡、正式训练或 M6。

## 9.22 **训练配置审计证据已修正（M1.3g-f-c-g-R1，2026-09-23）**

**纯文档证据修正卡**：不改代码、不冻结数值、不重跑 PPO 全套。

**四项修正（自行复核代码后认定；卡面未点名）**：

~~~text
D1 policy_architecture : 整项 U -> U（架构）+ F（action_dim=21）
  依据 IMPLEMENTATION_PLAN.md:203、AGENTS.md:13 红线 6、buffer.py:30
D2 solver_determinism  : F -> U（常量仅在 planning/model.py:50-51，
  无批准记录 / 无冻结资产 / 无契约强制，不满足 F 定义）
D3/D4 business/carbon_budget : 补「聚合口径 F」
  lagrangian.py:40 AGGREGATION_PER_TRANSITION_MEAN，:200-204 不符即抛错，
  :245 更新式 => budget 必须与 estimate 同口径（per-transition）
D5 审计 §7 标题 :「乘子预算」->「约束预算」（budget 属 ConstraintSpec，不属乘子）
~~~

**计数由脚本从 JSON 实际条目重新生成**（命令写入审计 §13.1）：

~~~text
total_items = 18          （初稿 19 系估算，错）
口径A 逐字段 : F=6 M=1 P=5 S=5 S/P=1 S/U=1 U=2 U/M=1
口径B 逐条目 : F=1 M=1 P=5 S=5 S/P=1 S/U=1 U=2 U/M=1 mixed(见 source_grades)=1
主分级 F（整项可直接用）(1): corrector_time_limit
含任一 F 字段(5): business_budget, carbon_budget, corrector_time_limit,
                policy_architecture, seeds
「❌ 需人工裁决」= 17      （初稿 14 系估算，错）
~~~

§0 的「没有任何一项属于 F」也已更正为「有 5 个条目含 F 字段，但整项可直接用仅 1 项」。

**验收**：两处 `git diff --check` exit 0；`git status --short` 空；
`env release --verify` exit 0。未改代码 / 冻结资产 / readiness。

**回滚（实测零冲突）**：

~~~bash
git revert 8c91765 27da85d   # == 11f0bf3^{tree}
~~~

> ## ⛔ **f-c-g-R1 完成即停，等待复审。**
> 通过后才将清单交人工逐项裁决。**未接**冻结卡、正式训练或 M6。

## 9.23 **审计来源与计数已修正（M1.3g-f-c-g-R2，2026-09-23）**

f-c-g-R1 **复审不通过**；本卡**纯文档修正**。

**四项来源 `P` → `S/P`**：`business_budget` / `carbon_budget` 的
`value_source_grade`，`multiplier_learning_rate` / `multiplier_max` 的
`source_grade`。依据**同时**记两处：synthetic smoke `train.py:623-631` **与**
probe `test_m13gcd_two_batch.py:60-61`。**S/P 仍不等于正式训练参数。**

**预算口径**：改为「在**固定 `per_transition_mean` 口径下**，决定预算数值及其
train-only／物理标定依据」；聚合方式**固定**，不列为可选训练参数。

**收回范围外重分类**：`solver_determinism` **U → F**（恢复 R1 前状态），
本卡不重新裁决求解器选项；保留 `action_dim=21` 正式约束说明；
**corrector 仅 `0.25 s` 数值有依据，是否启用仍待裁决**。

**计数（JSON 实际字段脚本生成）**：

~~~text
total_items = 18
各来源分级（逐字段）: F=7 M=1 P=1 S=5 S/P=5 S/U=1 U=1 U/M=1
复合 source_grades 内 : F=1 U=4
非空 needs_human_decision = 17
code_location == null     = 2  ['batch_arrangement', 'cpu_backend']
交叉检查：total 18 ✅  null_loc 2 ✅
~~~

**账本澄清**：未提交草稿的 19 / 14 **从未进入任何提交**；已提交基线 `11f0bf3`
为 **18 条 + 汇总 15**；并更正基线「无代码位置: 1」→ **2**。

**验收**：JSON 合法；两处 `git diff --check` exit 0；`git status --short` 空；
`env release --verify` exit 0；禁区无改动。

**回滚（实测两侧 tree 一致、零冲突）**：

~~~bash
git revert 4ddc08d 3d29aa6 43f0379   # == 9bd5c25^{tree}
~~~

> ## ⛔ **f-c-g-R2 完成即停，等待复审。**
> 通过后才交人工逐项裁决。**未接**冻结卡、正式训练或 M6。

## 9.24 **审计正文一致性已修正（M1.3g-f-c-g-R3，2026-09-23）**

f-c-g-R2 **复审不通过**：JSON 已改 **S/P** 但**正文未同步**（§7/§8 仍写
「P、仅测试示例」），且 §0 仍称「求解器选项是否冻结待裁决」（与 JSON 的
`solver_determinism=F`、`needs_human_decision=null` 及 §11 矛盾）。
**文档一致性卡；JSON 未改。**

~~~text
F1 §7 预算现值来源 : P -> S/P，列明 smoke train.py:623-631 与 probe :60-61
F2 §8 现值来源     : 乘子 lr=0.01 / max_multiplier=100.0  P -> S/P，列明上述两处
F3 §0 前言         : 删除「求解器选项是否冻结待裁决」，与 JSON/§11 一致
~~~

**一致性核对**：§0 不再提求解器裁决 ✅；§7/§8 含两处出处且分级 S/P ✅；
§11 solver=F 且称不在裁决范围 ✅；**§13 表格 vs JSON 逐字段 6/6 一致** ✅。

**验收**：两处 `git diff --check` exit 0；`git status --short` 空；
`env release --verify` exit 0；training readiness 仍为 false；JSON 未改；
禁区无改动。

**回滚（实测两侧 tree 一致、零冲突）**：

~~~bash
git revert 0335614 d5d32f6 5fad927 4030cb8 88993f4   # == 0f38997^{tree}
~~~

> ## ⛔ **f-c-g-R3 完成即停，等待复审。**
> 通过后才交人工逐项裁决。**未接**冻结卡、正式训练或 M6。

## 9.25 **训练配置候选与 train-only 预算标定（M1.3g-f-c-h，2026-09-23）**

新增 `scripts/calibrate_training_config.py`、`tests/test_m13gch_training_config_calibration.py`
（**17 passed**）、`configs/training/idc_training_config_candidate_v1.json`
（`candidate_not_frozen`）、`docs/AUDIT_TRAINING_CONFIG_CALIBRATION.md`；运行产物
`runs/m13gch_calibration/`。

**24 个 train origin**：`index_k = floor(k*211/23)`，`origin = 48 + 48*index_k`；
日对齐池 212；实测 `[48, 480, …, 10176]`，最小间隔 9 天，末项恰在 `end_exclusive`；
**未读 validation/test**。

~~~text
三提案（20 计算维全 1.0/0.5/0.25，储能 0，各 1152 transitions）：
  compute=1.00  business_mean=1.9704861111111112  carbon_mean=1.02089273465195
  compute=0.50  business_mean=1.9704861111111112  carbon_mean=1.0285893052437065
  compute=0.25  business_mean=1.9704861111111112  carbon_mean=1.0242255650312615
  fallbacks=0  timeouts=0  deadline_shortfall=1043

标定：
  business_budget = 1.9704861111111112 violation_task_steps/transition（并列 ⇒ 取 1.0）
  carbon_budget   = 1.02089273465195   kgCO2e/transition（达标最低 ⇒ 1.0）
  乘子 business: scale=1.9704861111111112 lr=0.00257545071707194  cap=5.074889867841409
  乘子 carbon  : scale=1.02089273465195   lr=0.009594884999059906 cap=9.79534838536124
~~~

**⚠️ 标定有效性（如实登记）**：`business_mean` 三提案**逐位相同**；实测
`exec_action` 确实不同（0.25 首步 `[0.25, 0.00140557, 0.25]`）但 `completed_work` 相同
⇒ 违约数**饱和**（90.5% 步为 `deadline_shortfall`）。`business_budget` 是**已饱和的
参考水平**，不代表「多做就能更低」。

**验收**：17 passed；`git diff --check` exit 0；`git status --short` 空；
`env release --verify` exit 0。禁区零改动；历史审计分级未改写。
**未**接训练循环、未改 readiness、未运行 `make train`。
长训前须先完成 **M9.1 工时预算**。

**回滚（实测两侧 tree 一致、零冲突）**：

~~~bash
git revert b56adcd f9d3294 6b661bd   # == 8141a04^{tree}
~~~

> ## ⛔ **f-c-h 完成即停，等待复审。**
> **未接**训练循环、正式训练或 M6。

## 9.26 **业务敏感性已定位到任务可用量层（M1.3g-f-c-h1，2026-09-23）**

**配置口径**：候选与标定报告写明 `forecast_cutoff=48` + causal 来源
（`train.py` 预检仍为 4，接线须改同值，**未**改 `train.py`）；
标定脚本删除硬编码 `0.25`，改用 `resolve_corrector_budget(...)` ⇒
来源 `production_default`；历史 run **未覆盖**（新 run-id 重算，数值逐位相同）。

~~~text
诊断（origin 48/4848/10176，固定种子，corrector on）：
  compute 0.25 vs 1.0：exec_action=0 planned_capacity=0 completed_work=0
                       sla_violation_count=**None（全程不分叉）**
决定性对照（corrector=OFF，exec == raw）：
  planned_capacity 总和 0.25→2331.1051 | 1.0→9324.4206（差 4.00×）
  completed_work  总和 0.25→1569.0    | 1.0→1569.0   （逐位相同）
  completed_work_total == initial_Q(50.0) + Σ mapper ledger（6/6 全等）
~~~

**绑定层**：容量**不是**（差 4× 而完成量相同）；**任务可用量是**；
修正器投影**不是** 0.25-vs-1.0 不变的原因。
⇒ **未找到**低于现候选 business budget 的可执行轨迹。>

> ⚠️ **R1 更正（范围限定）**：该观察**只覆盖本次 3 个 origin（48 / 4848 / 10176）
> 与本次种子（`server_seed = 0`）**，**不能**用来判定
> **24-origin 标定预算（`server_seed = 1`）的可达下界**；换 origin 集合或换种子
> 都可能有不同结论。

**未定位**：修正器 ON 相对 OFF 使完成量 ↓4.2%、SLA 0→100–114，**机制未定位**。
预测窗口 4 vs 48 的相同结论是**零对照**（参考提案与 forecast 无关）。

**原报告**「某约束先绑定」已降级为**待验证假设**。

**验收**：**15 + 17 = 32 passed**；`git diff --check` exit 0；`git status --short` 空；
`env release --verify` exit 0。禁区零改动；**未**冻结任何 budget。

**回滚（实测两侧 tree 一致、零冲突）**：

~~~bash
git revert 462d99a dd7b544 eb10aca 8030b4a   # == cdb3152^{tree}
~~~

> ## ⛔ **f-c-h1 完成即停，等待复审。**
> **未自行修改或冻结** business / carbon budget。

## 9.27 **诊断结论口径已收窄（M1.3g-f-c-h1-R1，2026-09-23）**

f-c-h1 **复审不通过**（3 处结论强于证据）。**证据表述返修**：未重跑诊断/PPO、
未冻结预算、原始 run/脚本/测试/候选/数值**均未改**。

~~~text
F1 「全部可用工作量做完」只成立于 **corrector=off 的 6 条轨迹**；
   on 完成量低于账本总量（48: 1503.483884 < 1569.0；4848: 1509.498549 < 1579.0；
   10176: 1500.977430 < 1567.0）。原「corrector on/off 亦然」已删除。
F2 off 的 4× 能力对照**仅**支持「在这些 off 轨迹中，两档动作均足以做完总工作量」；
   删除三处越界结论（排除 on 容量约束 / 断言任务可用量造成 on 的不变 /
   排除修正器投影对 on 的影响）。on 相对 off 服务下降的机制**保持未定位**。
F3 标定 server_seed=**1**、本诊断 server_seed=**0**（两者不同，§7 结论不可外推）；
   「未找到更低 business budget」限定为**本次 3 origin + 本次种子**的观察，
   不能判定 24-origin 标定预算的可达下界。
~~~

**核对**：五个数值从原始 `report.json` **重算**并严格 token 匹配正文，全部精确入文；
四处越界表述已清除。

**验收**：两处 `git diff --check` exit 0；`git status --short` 空。

**回滚（实测两侧 tree 一致、零冲突）**：

~~~bash
git revert a296991 a5bbae3 c581e2c 28ea6d7   # == d037679^{tree}
~~~

> ## ⛔ **f-c-h1-R1 完成即停，等待复审。**
> **未冻结预算**；**未接**训练循环、正式训练或 M6。

## 9.28 **规划快照每步能力口径已修正（M1.3g-f-c-h2，2026-09-23）**

`planning/snapshot_adapter.py` 原把 `C_server`（**work/hour**）直接交给规划器，
而 `exec_compute = used / cap` 必须与环境的 `action × max_task_load × C_server × delta`
一致。

~~~text
group_work_capacity           : C_server            -> C_server × max_task_load × delta_t_hours
group_power_coeff_kw_per_work : (Pmax−Pidle)/C_server -> (Pmax−Pidle)/(C_server × delta_t_hours)
先红 9 failed / 7 passed（严格早于实现）；改后 16 passed
~~~

**服务差值（server_seed=1, compute=0.5, corrector on）**：

~~~text
48   : completed 1503.483884 -> 1551.000001 (+47.516) | sla 100 ->  75
4848 : completed 1509.498550 -> 1554.000000 (+44.501) | sla 106 ->  79
10176: completed 1500.977430 -> 1544.000000 (+43.023) | sla 114 ->  63
timeouts=0, fallbacks=0（前后均如此）
~~~

**剩余缺口**：完成量 ≈98.4–98.9%、SLA 63–79/episode；**未调整预算凑结果**。

**重标定（新 run-id，历史未覆盖）**：`business_mean` 由「三档**饱和**同值
1.9704861111」变为「**随强度变化** 1.3038/1.3090/1.4019」⇒ 强度敏感性恢复；
`business_budget` 1.9704861111111112 → **1.3038194444444444**；
`carbon_budget` 1.02089273465195 → **1.0279280449520944**；
乘子 business `lr` 0.005882542761449005 / `cap` 7.669773635153129；
carbon `lr` 0.009463996474822895 / `cap` 9.728307393798211。
候选 `status` 仍 `candidate_not_frozen`。

**新 corrector 探针（0.25 s）**：off/on **均 reproducible**；
`insufficient_evidence` 来自探针**既有**功效判据，非本卡回归；**M5.4 保持 blocked**。

**验收**：`make check` **2939 passed**（ruff/mypy 153 files 全通过）；
`make smoke` exit 0；`env release --verify` exit 0；`git status --short` 空。
未运行 `make train`、未改 readiness。

**回滚（实测两侧 tree 一致、零冲突）**：

~~~bash
git revert e0aacd0 e7c1e94 d055ba6 0d958df 1082216 8efaf60 6910edf 2c77526
# == 513b172^{tree}
~~~

> ## ⛔ **f-c-h2 完成即停，等待复审。**
> **未**接训练循环、正式训练或 M6；**未冻结**预算。

## 9.29 **正式 corrector 规划输入已改为 B6 因果预测（M1.3g-f-c-h2-R1，2026-09-23/24）**

复审不通过的原因：`planning/snapshot_adapter.py` 对 **formal** env 也把
`env.price_t` / `T_amb` / `pv_t` / `wt_t` / `carbon_factor_t`（**realized 真值**）
当作「可见预测」，`snapshot.forecast` 也恒为 `oracle_debug`。

~~~text
强制起点 ebde208（tree 0ea3ef2995e80dd86bc5a6f67f92489dc706dd16）
6bf2e95 开卡 / 8dad024 先红（16 failed, 7 passed）/ 2b31dd4 实现（23 passed）
276d954 重标定候选 / <record> 验收记录
~~~

**修复**：formal 的规划输入只取**已验签 B6 causal forecast** 通道
（`env.*_forecast_t` / `task_arrival_forecast`）；`snapshot.forecast` 由**同一条**
正式入口 `build_formal_scenario_b6(split, origin, forecast_cutoff)` 重建为
`mode="formal"`（真实 provenance、真实窗口、过 `purpose="training"` 门禁）。
**legacy / oracle_debug 语义逐字不变**（detached worktree 逐字段实测：唯一差异是
adapter **自身**的 `code_revision`）。

**因果隔离实测（origin 48，`cutoff=48`，审核方逐字复现）**：

~~~text
改前 planning price[:4] = [0.12659, 0.11916, 0.10834, 0.10797]  ← = env.price_t（未来真值）
改后 planning price[:4] = [0.10778, 0.10798, 0.10789, 0.10796]  ← = env causal price_forecast_t
env.price_t[1] = 123 → 规划快照与 bundle 逐字节不变
反向控制 price_forecast_t[1] += 5 → planning price[1] 0.10798 → 5.10798
purpose gate PASS；bundle generated_at 2024-01-02T00:00:00+08:00
→ target_end 2024-01-03T00:00:00+08:00；9 个来源 hash
~~~

**P2 修复**：`replay_corrector_service.py` 的 `carbon_total` 由 `sum([x])`
（恒等于**最后一步**）改为**逐步累加**；三步 1.0/2.5/4.0 的用例由 4.0 → **7.5**。

**三次 origin 重放（`runs/m13gch2r1_replay`，历史未覆盖）**：

~~~text
origin  48: completed=1551.000001 sla=75 carbon_total 0.193232 -> 32.285432
origin 4848: completed=1554.000000 sla=79 carbon_total 3.131196 -> 39.759661
origin 10176:completed=1544.000000 sla=63 carbon_total 3.207020 -> 57.967448
timeouts=0, fallbacks=0
~~~

completed/sla 与 h2 相同：A/B 实测 exec 动作最大差 3.07e-03，但固定提案重放受
**任务可用量**约束（h1 结论），故服务量差异仅 ≈−7e-08。**未据此调预算凑结果。**

**重标定（`runs/m13gch2r1_calibration`，24-origin，历史未覆盖）**：

~~~text
compute=1.0 : business 1.3038194444(不变)   carbon 1.0279280450 -> 1.0278989607
compute=0.5 : business 1.3090277778 -> 1.3038194444   carbon 1.0348856339 -> 1.0348850010
compute=0.25: business 1.4019097222(不变)   carbon 1.0317367381 -> 1.0317406822
business_budget 1.3038194444444444（不变）；carbon_budget 1.0279280449520944
-> 1.0278989607183693；达标提案 [1.0] -> [1.0, 0.5]
乘子 carbon: lr 0.009463996474822895 -> 0.009464532046878296
             cap 9.728307393798211   -> 9.7285826546719
候选 status 仍为 candidate_not_frozen
~~~

> ⚠️ **h2 的 `runs/m13gch2_calibration_recap/` 使用了含未来真值的规划输入**，
> 属**诊断结果，不作为正式配置依据**；原始产物保留、未删未改。

**验收**：新档 **23 passed**；focused 110 passed；`make check` **2962 passed**
（h2 为 2939）；`make smoke` exit 0；`env release --verify` exit 0
（`017575dc…`，`formal_training_ready=False`）；`probe_corrector_repro`
off/on 均 reproducible（`insufficient_evidence` 为探针既有功效判据）；
`git status --short` 空。**未**运行 `make train`、**未**启动 M6、**M5.4 保持 blocked**。

> ## ⛔ **f-c-h2-R1 完成即停，等待复审。**
> **未冻结**预算；**未**接训练循环、正式训练或 M6。回滚见任务卡 §dt。


## 9.30 **评估输入契约与受控短跑评估器已交付，并触发一次资产链前向刷新（M6-P1 + CHAIN-REFRESH，2026-09-24）**

**M6-P1**（起点 `1e3a045`，提交 `a2dda64` 开卡 → `bc1444c` 先红 → `626f97f` 实现 →
`980a149` 证据）：新增 `checkpointing/eval_input.py`（评估输入契约：版本 / 21 维动作 /
obs 维度 / policy 权重 / 策略结构 / **train-only** origin / 三环境种子 /
**12 个配置与资产来源的 role+规范路径+SHA-256**）、`evaluation/{service_standard,
metrics,sources,controlled_run}.py`，重写 `contracts.EvaluationRecord`
（成本分列购电费/退化费、混合目标量**不标 SGD**、风光分列可用/使用/弃电 +
used/available **利用率**、可再生**占比**另名、物理违规、raw→exec 修正、
`not_computable` 显式标记）。

**红线落实**：`service_qualified` **不再默认 true**，必须显式传入服务标准；
标准未冻结 / 零分母 / 失败 run ⇒ `None`（**未判定**，≠ 达标）；
`FROZEN_PROJECT_SERVICE_STANDARD is None`（95%/1% **未**冻结）；
五类正式方法一个都没评估 ⇒ 全部 `not_evaluated`（**未评估**）；
two-batch 恢复格式被**拒绝**冒充评估输入。

**受控短跑链路（`runs/m6p1_controlled_short_run`，train-only，无参数更新）**：
checkpoint sha256 `1fb4fd4fd58e7fdf…`；购电费 11.534050 SGD / 退化费 0.842105 SGD /
碳排 42.134131 kgCO₂e / 购电 104.811270 kWh；PV 利用率 0.1207、风电 0.2732、
可再生占比 **0.8456**（与利用率**不同名不同量**）；物理违规 0；修正 48 步、
耗时中位 0.079 s / P95 0.112 s、超时 0、回退 0；`service_qualified=None`。

**⚠️ 本卡触发并已完成一次冻结资产链的「前向刷新」**：M6-P1 按边界改了
`contracts/models.py`，而该路径属于 `FORECAST_SOURCE_PATHS`，而
`b6_formal_code_revision()` 的语义是**「最后触碰该路径集的提交」**——
于是实现提交把 policy-v2/v3 的 revision 锚点推进，**本分支全部 formal env 链路
（63 项回归）同时变红**。经用户裁决「前向修复、保留历史」，另开
`docs/task_cards/CHAIN_REFRESH.md`：穷举扫描得到 **11 个文件**的依赖闭包
（policy-v2 → policy-v3 → refs_v4 → formal_splits_v5×3 + formal_splits_v4×3 →
arrival mapper → env release），**逐层刷新**且每层只用 `materializer_revision` 与
指向被刷新文件的 `sha256` 变化（数值、日期、origin 集合、切分、readiness 全部实测未变）。
**已取代的历史证据（policy v1、formal_splits v1/v2/v3、refs v1/v2/v3、exogenous v2）
字节保留、未刷新。** 旧/新 SHA 账本见卡片 §8.2。

**教训（重要）**：闭包必须**穷举扫描**，不得手工列举——首轮手工列举漏了 v4 triad，
由 `make check` 抓出。另：任何改动 `contracts/**` 的卡都会推进该 revision 锚点，
**必须同时规划资产链刷新**。

**门禁**：`make check` exit 0 → **3025 passed**（ruff/mypy 全通过）；`make smoke` exit 0；
5 个资产链物化器 `--verify` 全 exit 0；`git status --short` 空。
**未**运行 `make train`、**未**读 validation/test、**未**改 readiness
（`formal_training_ready=False`）。

**回滚（实测，newest-first，独立 worktree）**：
```bash
git revert 980a149 1e0e5d7 4d16f04 da288ee 626f97f bc1444c a2dda64
# == 1e3a045^{tree} = 81e360671835f98a8f6f593a9af5955c9a27e42d
```

> ## ⛔ **M6-P1 + CHAIN-REFRESH 完成即停，等待复审。**
> **未**接正式训练、validation 或最终 test；**未冻结**服务阈值。

## 9.31 **M6-P1 返修：资格口径与来源账本（M6-P1-R1，2026-09-27）**

起点 `a6e9667`（CHAIN-REFRESH 通过；M6-P1 因三项不通过）。提交
`9b201b7`（开卡）→ `9ba1aff`/`990cb6c`（先红）→ `1b51452`（实现）→ 记录提交。

```text
1) 按时率分子改与到期分母同口径（只数 latest_finish_time < horizon 且按时完成）
   先红实测：按时任务率必须在 [0,1]，实际 2.0 / 工作量率 3.0
   （反例 = 1 个本 episode 到期 + 1 个提前完成但下 episode 才到期）
2) 新增 adapter.violations_in() / qualify_service()：接入 / SOC / 充放互斥 / 守恒
   任一违规 ⇒ False，即使服务指标达标也不得 True（不得进入同等服务成本/碳比较）
3) 受控 run 的 manifest 写入实测、可重算的三个 hash（新 run-id
   runs/m6p1r1_controlled_short_run，未覆盖旧的）：
   dependency_lock_hash=8e6bd4a67311b32b…（uv.lock）
   data_hash=90c4f7952c9c69f5…（12 个已验签来源的规范化 JSON）
   scenario_hash=bca4e8cd8aea1429…（注入 provenance_hash）
```

**未**触碰 `contracts/` 与 11 个已刷新资产（`git diff a6e9667 -- contracts/ data/manifest/ configs/` 为空），
故 B6 revision 锚点未前进、资产链仍有效。

**门禁**：`make check` exit 0 → **3032 passed**；`make smoke` exit 0；
`env release --verify` exit 0。**回滚**（newest-first，独立 worktree，本会话全部提交）
== `1e3a045^{tree}` = `81e360671835f98a8f6f593a9af5955c9a27e42d`，零冲突。

> ## ⛔ **M6-P1-R1 完成即停，等待复审。**

> ✅ **M6-P1-R1 人工复审通过（2026-09-27）**：见 `docs/task_cards/M6_P1_R1.md` §10。

## 9.32 **train-only 服务达标线可行性审计（M6-P1-F1，2026-09-27）**

起点 `1830d30`（前置：M6-P1-R1 审核通过记录 `56445c3` 已**单独提交**）。提交
`2019026`（开卡）→ `eac0efb`（脚本）→ 记录提交。

**唯一 run**：`runs/m6_service_feasibility_train_v1/`（train-only；24 个标定卡预先选定的
train origin × 三种**固定参考提案** compute=1.0/0.5/0.25（储能 0）；corrector **on**、
生产默认 **0.25 s**；M6-P1 评估器，`service_standard=None` ⇒ 正式资格恒为**未判定**）。

**结论：「本次参考轨迹未展示达标」** —— 0 / 72 联合达标，四个单项各 0 / 72：

```text
按时率(任务数)  0.6170 / 0.7234 / 0.7872  (需 ≥0.95)  差 0.163–0.333
按时率(工作量)  0.6071 / 0.7089 / 0.7781  (需 ≥0.95)  差 0.172–0.343
期末剩余比例    0.0115 / 0.0183 / 0.0261  (需 ≤0.01)  超 1.15–2.61 倍
不可中断中断    9 / 14 / 18               (需 =0)     每 episode 多 9–18 次
失败面全 0：未完成 episode 0 / 超时 0 / 回退 0 / 四类物理违规 0 / failed_tasks 0
```

**主要差距**：① **不可中断任务中断**是硬阻塞（参考提案与单步修正器无任何连续性机制）；
② 按时率与期末剩余同向 ⇒ 差距在**任务服务层**而非物理或修正器失败面。
**不外推**为「任何策略都无法达标」；门槛**未**按观测结果调整，
`FROZEN_PROJECT_SERVICE_STANDARD` 仍为 `None`。

**下一步方向 A–D（供人工裁决，均需另开卡）**：A 修改参考提案族（加入不可中断优先/
期限驱动，**只改轨迹不改门槛**）；B 复核任务/到达口径（M1.3f-d 的 arrival 强度决策
仍待裁决）；C 维持 95%/1%、把达标责任交给正式训练策略；D 由人工重新裁定门槛数值
（**本卡不提议、不执行**）。

资产 hash：`dependency_lock_hash=8e6bd4a67311b32b…`、`data_hash=90c4f7952c9c69f5…`、
`scenario_hash=bca4e8cd8aea1429…`。

> ## ⛔ **M6-P1-F1 完成即停，等待复审与裁决。**
> **未**启动 M6-P2、正式训练或 validation/test；服务标准**未冻结**。

> ✅ **M6-P1-F1 人工复审通过（2026-09-27）**：见 `docs/task_cards/M6_P1_F1.md` §8。

## 9.33 **服务缺口与不可中断中断机制诊断（M6-P1-F2，2026-09-27）**

起点 `1263ba8`（F1 审核通过记录之上）。提交 `2ddd0d0`（开卡）→ 脚本提交 → 记录提交。
run：`runs/m6_service_gap_diagnosis_v1/`（train-only；origin 48/4848/10176 ×
compute 1.0/0.5/0.25 × corrector **on/off**，同种子；**off 仅机制对照**）。

**决定性结果**：**corrector off 的 9 条轨迹全部按时率 1.0、期末剩余 0、不可中断中断 0**；
corrector on 为 0.68–0.74 / 18–25 / 12–18；**两种模式四类物理违规均为 0**。
⇒ 服务缺口与不可中断中断**由 corrector 通路引入**，不是接入/SOC/守恒约束或任务量所致。

**首次中断（o48 / c1.0 / on）**：step 4、不可中断任务 `2515224290600144462`
（`start_time=2` 已启动、当步未获工作量、剩余 3.518、**当步到期**）；
当步计划能力**用尽**（未用 ≈2e-11）但**接入余量 5.35 kW 未触顶**；
`raw→exec` 修正 **1.0（最大）**、`correction_reason=deadline_shortfall`、
**6 个已启动任务仅 1 个被执行**。

**归属**：容量/接入限制 **不支持**；**修正器投影 支持（最强）**；
任务分配顺序 **支持但为下游表现**。**未定位**：corrector 内部哪一阶段造成 6→1 收缩。

> ## ⛔ **M6-P1-F2 完成即停，等待复审与下一张实施卡的决定。**
> **未**修改调度算法或门槛；**未**启动 M6-P2 / 正式训练 / validation / test。

> ✅ **M6-P1-F2 人工复审通过（2026-09-27）**：见 `docs/task_cards/M6_P1_F2.md` §8。

> ❌ **M6-P1-F3 人工复审不通过（2026-09-27）**：见 `docs/task_cards/M6_P1_F3.md` §8。返修卡 M6-P1-F3-R1。

> ⚠️ **M6-P1-F3-R1 复审：诊断有效、修复未完成（2026-09-27）** → 见 R2 卡。

## 9.34 **M6-P1-F3-R2 开卡 + 交接（2026-09-27，上下文到上限）**

起点 `f2c7431`（tree `fd2f8308ce2fec53c318066e193494a2646c75b6`）。
提交：`0c698c3`（R1 复审：诊断有效、修复未完成）→ `60aa037`（R2 卡）→ `7d790ae`（交接）。

**决定性反例（R1 保留证据）**：origin 48 / compute=1.0 / corrector on / **step 3**：
快照与 env 任务集相同；**Stage A == Stage B == 10.5179**（到期任务全部剩余工作，slack 0），
**env 实际只给 7.0000**（差 3.5179）；exec 非零组仅 [8,12,14,15]。
corrector off 服务完美（1.0 / 0 / 0），两种模式四类物理违规均为 0。

**已启动**：改前 run `m6_service_gap_prefix_r2`（新 run-id，未覆盖既有 run）。

**续做起点（见卡 §7.3）**：在 step 3 同一前状态下取环境分配器的**输入任务与排序**、
逐组计划容量、**物理缩放后容量**与**最终逐任务分配**，查明 3.5179 去了哪里；
再做最小修复（不预设定保形目标、不改优先级/期限口径），同一前状态证明有效，
然后 9 组 on/off 配对重放 + 门禁 + 回滚核对 tree == `fd2f8308…`。

> ✅ **M6-P1-F3-R2 人工复审通过（2026-09-28）**：见 `docs/task_cards/M6_P1_F3_R2.md` §9。
> 根因 = 到达激活时机；修复后 corrector on 的 9 组配对服务与 off 完全一致（1.0 / 0 / 0），
> 四类物理违规 0/0/0/0，门禁全绿。下一卡 **M6-P1-F4**（服务达标线复审）。

> ✅ **M6-P1-F4 复审完成（2026-09-28）**：F3-R2 修复后 train-only 服务达标线
> **72/72 联合达标**（修复前 0/72），四项实测值均为 1.0 / 1.0 / 0 / 0（各自的可达上/下界值），
> 零物理违规 / 超时 / 回退。见 `docs/audits/M6_SERVICE_FEASIBILITY_REAUDIT.md`，
> run `runs/m6_service_feasibility_after_f3r2/`。
> **冻结建议：支持冻结原提案值**（可达性已正面回答），附两条限制：
> (a) 本次三种固定提案在 24 个 train origin 上四项指标均相同且全部通过，故本审计的
> 四项列没有区分出这三档提案——对其他提案与方法的资格尚无结论；(b) 只读 train，
> 泛化未验证。等待人工复审。

> ✅ **M6-P1-F4 人工复审通过（2026-09-28）**：见 `docs/task_cards/M6_P1_F4.md` §8。
> 72/72 联合达标、门禁全绿、回滚零冲突；返修要求（收窄「零余量／无区分力」类
> 无限定表述）交由 **M6-P1-F5** 执行。下一卡 **M6-P1-F5**（冻结服务标准）。

> ✅ **M6-P1-F5 完成（2026-09-28）**：业务服务资格标准已冻结为 **`m6-service-standard-v1`**
> （`frozen=True`：0.95 / 0.95 / 0.01 / 0），记录见 `docs/M6_EVALUATION_PROTOCOL.md` §2.1。
> 评估器接线**未改**：`evaluate(..., service_standard)` 仍无默认值，未传入 ⇒ 未判定。
> F4 的「零余量／无区分力」类无限定表述已按复审返修要求收窄。等待人工复审。

> ✅ **M1.3g-f-c-i 已完成（2026-09-28）**：F3-R2 修复后重跑 train-only 标定
> （新 run `runs/m13gci_calibration_after_f3r2/`，24 origin × 三档提案，超时/回退 0）。
> 按 dh.6 预注册规则重算：`business_budget` 1.3038194444 → **0.0**（三档
> `sla_violation_count` 实测全为 0）、`carbon_budget` 1.0278989607 → **1.0321066253**；
> business 乘子 `scale/lr/cap` 1.3038194/0.0058825428/7.6697736 → **1.0/0.01/10.0**，
> carbon 乘子 → **1.0321066/0.0093875198/9.6889214**。数值配置已冻结为
> **`configs/training/idc_training_config_v1.json`**（v1，frozen；候选仍
> `candidate_not_frozen`）。**「数值冻结」≠「训练放行」**：`formal_training_ready` 仍 false。
> 同卡更正了 F5 §7.6/§7.7 的回滚范围事实（最终范围 6 个提交）。等待人工复审。

> ✅ **M6-P1-F5 人工复审通过（2026-09-28）**：见 `docs/task_cards/M6_P1_F5.md` §8。
> 服务标准已冻结为 `m6-service-standard-v1`（0.95/0.95/0.01/0，frozen=True），
> 评估器接线未改（无默认值）。遗留更正（§7.6/§7.7 的回滚验证实际在 `206ee73` 执行，
> 最终 HEAD `94a94e9` 的完整范围是 6 个提交）交由 **M1.3g-f-c-i** 顺手更正。

> ✅ **M1.3g-f-c-j 完成（2026-09-28）**：冻结配置驱动的训练闭环 + 批次边界训练恢复。
> 新增 `safe_rl_v2/formal_train_loop.py` 与 `safe_rl_v2/controlled_formal_train.py`；
> 一批 = 4 episode × 48 步 = 192 transition → 4 epoch × 4 minibatch = **16 次 Adam step**
> → 整批 192 条信号更新**一次**乘子（实测 Adam 16/32/48、Lag 1/2/3）。
> 三条 short-run：3 批连续 / 2 批+checkpoint / 新对象恢复第 3 批；第 3 批 20/20 字段与
> 最终 policy/Adam/Lagrangian/两个 RNG **逐项一致**，边界对照点在第 3 批**开始前**。
> **实现中实测并修复**：策略初值原先消耗全局 Torch RNG（非 corrector 预算所致），
> 修复后同命令两次跑逐位相同。`training_scope=controlled_short_run`，claims 全 false，
> 512 批 × 3 seed 未运行，`formal_training_ready` 仍 false。等待人工复审。

> ✅ **M1.3g-f-c-i 人工复审通过（2026-09-28）**：见 `docs/task_cards/M1.3g-f-c-i.md` §8。
> 冻结配置 `configs/training/idc_training_config_v1.json`（v1）已确认；预算
> `business 0.0 / carbon 1.0321066253` 为实测结果，未加人为下限。下一卡
> **M1.3g-f-c-j**（训练闭环 + 批次边界 checkpoint）。

> ✅ **M1.3g-f-c-j-R1 完成（2026-09-28）**：三项修正落地 —— ①先在 `a0c790f` 提交代码再跑短跑，
> 四条新 run 的 `manifest.revision` 指向含实际运行代码的提交；②manifest/checkpoint 写入
> **非空可重算**的三个 hash（`dependency_lock_hash`=uv.lock、`data_hash`=现有来源摘要口径、
> `scenario_hash`=本次实际 origin 的 injection provenance），六项资产 hash 与 live **实测全等**；
> ③训练环境改由本次 master seed + 冻结 `seed_offsets` 构造（seed 0 → 0/1/300000 语义不变，
> **seed 1 → 1/2/300001**）。seed 0 恢复对照：第 3 批 22 个字段不一致 0 个，最终
> policy/Adam/Lagrangian/两个 RNG 逐项一致。旧 run 与 checkpoint 保留未动（仅历史诊断证据）。
> 等待人工复审。

> ⚠️ **M1.3g-f-c-j 复审：闭环与恢复对照有效，来源账本与环境种子接线未达标（2026-09-28）**
> → 返修卡 **M1.3g-f-c-j-R1**。须修：manifest 的 `code_revision` 必须指向含实际运行代码的
> 提交；三个来源 hash 必须非空且可重算；环境种子须随本次 master seed（seed 1 → 1/2/300001）。

> ✅ **M1.3g-f-c-j-R1 人工复审通过（2026-09-28）**：见 `docs/task_cards/M1.3g-f-c-j-R1.md` §8。
> 三项修正（提交时序、来源账本、环境种子）均达标。下一卡 **M9.1**（PPO 训练链资源预算）。
