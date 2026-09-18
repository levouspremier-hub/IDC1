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

只可用 configs/frozen_refs/refs_v3.json。

configs/frozen_refs/refs.json 是历史 v2，标记 superseded_pre_trust_boundary_fix，保留但不可被 g-c/e/f 使用。

关键值：

~~~
price_ref=4.5                          max(abs(train price))
pv_ref_kw=350.9073696124661            train derived
wind_ref_kw=262.3178613166015          train derived
carbon_factor_ref=0.402                train derived
lambda_ref=2000, queue_ref=6000,
queue_capacity_ref=6000, cost_ref=60   declared physical scales
~~~

生成器不得接收 dataframe/frame、kwargs、expected hash 或 caller trust root。

### 8.2 Formal split triad

只可用目录 data/manifest/formal_splits_v4/。

schema 是 m1.3g-formal-split-manifest-v4；三个文件共享一个 frozen_at_utc，并以 triad 原子物化。

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

**R2 追加（Task schema / 可行域 / 固定点守恒）**：mapper 输出必须覆盖
`idc_model.task.Task` 的 **11 个必填字段**（含 `name`、`load_profile`）；
每个任务 `workload ∈ [w_min, w_max]`；分割算法必须**同时**满足每任务上下界与
全局守恒（超出覆盖 **fail closed**，有冻结的 `MAX_TASKS_PER_SLOT`）；
守恒用**固定点整数账本**（float `Task` 只是运行时表示）；
profile 概率等参数**全部**是人工批准的 modeled scenario，**不得**声称由
aggregate train 数据校准。

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
