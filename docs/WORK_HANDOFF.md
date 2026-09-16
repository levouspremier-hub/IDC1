# 工程工作交接文档

> 更新时间：2026-09-16（Asia/Shanghai）
> M5.4h1 实现基线：`p4-safeppo-m51a-rollout-contract` @ `ca17d87`。
> ⚠️ **当前工作分支**：`p4-safeppo-m51a-rollout-contract-m12-integration`
> 下文的 `p4-safeppo-m51a-rollout-contract` 指针**未被移动**，仍为 `04296db`。
>
> 当前 HEAD（整合分支）：M1.3b **已通过**、M1.3c 审计通过、
> **M1.3d 已通过人工审核**（2026-09-16，含 R1/R2/R3 三轮返修）、
> **M1.3e 第一轮审核不通过，已返修（M1.3e-R1）并再次提交复审**。
> ⚠️ **在 M1.3e-R1 通过人工复审之前，M1.3e 不得视为通过**；**M1.3f 未开始**。
> 详见下方 §7K（含 §7K.8 本轮返修）。
>
> 历史记录：原 p4 分支上，M5.4i **第三次返修**实现终点 `2cab5a1`；
> `M5.4i 已通过人工审核，M5.4 工程门禁已解除`（见 §8）。
> 受保护基线：`paper-baseline` @ `787a3c8`，**绝不直接修改或自行合并**。

本文件供新的 VSCode/Claude Code 会话或人工审阅者继续工作。它记录的是当前可核查的
代码状态、审阅结论与下一步授权；任务卡中的历史表述若与本文件冲突，以本文件的
“当前门禁”与已追加的勘误为准。

## 1. 先读什么、先做什么

继续任何工作前，必须完整阅读：

1. `AGENTS.md`（仓库红线与 Git 纪律）。
2. `docs/IMPLEMENTATION_PLAN.md`（研究计划与模块依赖）。
3. `docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md`（任务卡和提交流程）。
4. 本文件与目标任务卡。

每张卡开工前必须执行并报告：

```bash
git status --short
git branch --show-current
git log -1 --oneline
git diff --check
```

工作树不为空、分支错误、或有未知文件时停止；禁止用 `reset --hard`、`checkout --`、
`git clean`、`rebase`、`commit --amend`、`git add .` 或 `git add -A` 处理。

## 2. 不可违反的工程红线

- 不得放松接入容量、SOC、充放电互斥或任务约束来制造可行。
- 不得清空队列、重置 SOC、跳过任务或日界伪造完成量。
- 测试、规划和评估不得读取未来真值；预测必须含来源、生成时刻、可见窗口。
- 归一化参考值只来自训练集或预定物理尺度，冻结后共享。
- 不改 `marl/`、`grid_model/` 主逻辑、`legacy/`、顶层兼容 shim。
- 新主链严格使用 21 维动作；拒绝旧 23 维动作、无版本或 schema 不匹配 checkpoint。
- PPO buffer 必须同时保存 `a_raw` 与其 raw log-prob；`a_exec` 只能记录在 transition/info。
- 改 `envs/idc_price_env.py::step()` 前，必须先有该卡提交的失败测试。

## 3. 分支地图与合并状态

| 分支 | HEAD | 作用 / 当前状态 |
|---|---|---|
| `paper-baseline` | `787a3c8` | 受保护主线；不改、不合并。 |
| `p0-bootstrap` | `e8c6087` | M0 工程门禁证据。 |
| `p1-data-contracts` | `d72b192` | M1/M2；M1.2c 原始数据冻结已通过，尚未并入当前 M5 链。 |
| `p2-physics` | `4e070bc` | M3 物理链卡片与实现证据。 |
| `p3-corrector` | `da0a28c` | M4 规划器证据。 |
| `p4-safeppo` | `fef12f6` | 早期 M5 链。 |
| `p4-safeppo-m51a-rollout-contract` | `2cab5a1` | **当前继续工作分支**；包含 M5.1–M5.4i 的累计链（`2cab5a1` 为 M5.4i **第三次返修**实现终点）。 |
| `p5-eval-viz` | `05612f9` | M4.4c 终点及早期评估/契约工作。 |

不同分支尚未进行人工批准的整合。不得把 `p1-data-contracts`、当前 p4 分支或其他
阶段分支直接 merge 到 `paper-baseline`。跨分支整合本身也必须单开任务卡、先列出
冲突策略和验收，再得到人工批准。

## 4. 模块进度总览

| 模块 | 当前判断 | 已有成果 | 尚未完成 / 门禁 |
|---|---|---|---|
| M0 | 已完成 | uv/Python 3.12、Makefile、AGENTS、测试骨架。 | 无。 |
| M1.1 | 已完成 | 执行链审计。 | 无。 |
| M1.2 | **原始冻结通过** | Singapore-2024 原始价格、负荷、IGS、ERA5 的 hash、时区、许可和只读核验。 | 不等于正式 ScenarioBundle；碳强度、IDC 本地 PV/风电映射、小时到半小时规则尚未冻结。 |
| M1.3 | 推进中（b/c/d **已通过**；e 执行完成待审） | M1.3b canonical 半小时事实表；M1.3d 连续 truth split + train-only 描述统计 + origin 门禁；**M1.3e contract-v8 + 因果 seasonal-naive provider + policy manifest**。 | **M1.3f**（PV/风电/碳强度/arrival 口径）→ **M1.3g**（正式 ScenarioBundle + env/train 接线，含 refs 冻结）未完成；正式训练仍不可开始。 |
| M2 | 大部分完成 | 版本化契约、21 维拒绝、checkpoint/schema 门禁。当前版本为 **`contract-v8`**（M1.3e 起；v7 buffer/checkpoint 明确拒绝）。 | 与真实 M1.2 场景的完整接线仍待 M1.3g。 |
| M3 | 有实现和大量卡片 | 21 维、A[i,g]、接入投影、尾段结算、deadline 分类、可见预测等已有卡片证据。 | 不在本轮 M5.4 工作范围内；跨分支整合前不得重新声称全链已验收。 |
| M4 | 有实现和大量卡片 | H 步 LP/MIP、raw-action projection、wrapper、性能探针。 | M5.4 的 corrector 确定性/预算发布门禁仍未解除。 |
| M5.1–M5.3 | 已有实现，未作正式训练结论 | raw/exec buffer、三 value/GAE、Lagrangian 与状态校验。 | 无真实数据正式训练；训练性能或收敛均不得声称。 |
| M5.4 | **blocked，等待人工发布确认** | 训练入口、产物账本、corrector 复现与预算归因探针；M5.4h2 账本勘误；M5.4i 生产默认 0.25 s + 跨批发布证据。 | 默认 0.05 s 在受控负载下跨进程不稳定（M5.4h/M5.4h1/M5.4h2）；M5.4i 已把默认改为 0.25 s 并给出 3 负载 × 3 批的稳定证据。**是否发布由人工决定**，本文件不代为解除 blocked。 |
| M6–M9 | 未作为当前继续范围 | 有早期骨架/卡片。 | 依赖 M1.3、M5.4 以及人工批准。 |

## 5. M1.2 原始数据冻结：已通过但范围有限

### 5.1 已接受的提交

分支 `p1-data-contracts`：

- M1.2a 证据终点：`b77594e`。
- M1.2b manifest 不可变修正：`05e7034` → `7d04d3d`。
- M1.2c raw 写入路径封锁：`16ee77b` → `d72b192`。

冻结 manifest：`data/manifest/singapore_2024.json`，schema
`m1.2-singapore-2024-v2`。原始文件均在被忽略的
`data/raw/singapore_2024/`，不可提交、镜像或重新分发。

| 文件 | SHA-256 |
|---|---|
| USEP | `3e86e162526e2c3f499571c607c9b4d414a1648ea3e96fae65461bfaaefd306a` |
| IGS | `4ff06ea9044ed17bb24c69515cebd2c518deb0c96e169463b98a028e521ae83fd` |
| SASEA load | `1264e192171c36fb86a580974c5b0ae64b41917ab7c2330e369f1791532c3e89` |
| ERA5 weather | `9732f1b2e740da1917a191191706026c1b9a2c6b99e1eee1023642425cf1ea7d` |

注意：上表 IGS hash 以 manifest 为准；继续前应运行 `--verify` 重新核验，不能手工修改
manifest 来适配本地文件。

### 5.2 M1.2 的强制语义

- USEP 原始单位是 `SGD/MWh`；后续 reader 才可按固定比例 `0.001` 转成 `SGD/kWh`。
  M1.2 **没有**物化转换数据。
- 全部时间轴为 `Asia/Singapore`；2024 闰年半小时序列为 17,568 点，天气小时序列为
  8,784 点；无插补。
- ERA5 是国家级网格替代，不是 IDC 现场观测。IGS 不是本地 PV，也不是太阳能专属。
- 缺少半小时碳强度、IDC 本地 PV 与风电发电量；不得用默认曲线、常数或重复日补造。
- manifest 存在时，任何 `--usep-zip`、`--generation-zip`、`--sasea-zip`、
  `--fetch-weather` 都必须在任何 raw I/O 之前被拒绝。`--verify` 是只读操作。

实际核验命令：

```bash
uv run python scripts/fetch_singapore_data.py \
  --raw-dir data/raw/singapore_2024 \
  --manifest data/manifest/singapore_2024.json --verify
```

正式数据训练仍被 M1.3 语义缺口阻断；`make train` 必须明确报错，绝不隐式回退合成数据。

## 6. M5.4 已审阅历史与当前结论

### 6.1 必须记住的勘误

M5.4f（终点 `46a8acc`）曾把确定性 options 接到不被 corrector 使用的
`_solve_time_indexed` 路径。M5.4g（`4682d49` → `9042ec5`）已修正：
`solve_time_indexed_mip_raw_projection()` 的 Stage A/B 实际传入
`random_seed=0`、`parallel=False` 和共享剩余 `time_limit`。

不得再引用 M5.4f 的“路径已修好”或 digest 对比作为接线证据；其任务卡追加的 §10
历史勘误已明确该结论失效。M5.4g 的运行时 spy 才是有效接线证据。

### 6.2 M5.4h / M5.4h1 / M5.4h2 的证据

- M5.4h：`96f3aa1` → `c842c34`，建立预算矩阵、受控 CPU hog、逐阶段探针。
- M5.4h1：`07e2224` → `ca17d87`，修复了初版账本中的负载记录、失败状态和节点替代
  实验语义。
- M5.4h2：`bf4b62a` → `15b8bdd`（**已通过人工审查并被接受**），勘误了 `options`
  字段名、`overall` 与 release gate 的一致性、以及未绑定节点上限的候选表述。
- M5.4i：`d6f875b` → `be26170`（**执行完成，等待人工审查**），把生产默认预算集中到
  唯一来源并改为 **0.25 s**；详见 §8。
- 默认预算现为 **0.25 s**（`production_default`，唯一来源
  `planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S`）；0.05 s 仍为显式 override。
  该常量现在同时写入 config/report/**manifest** 三处，且三者恒等（M5.4i 返修）。
  M5.4h/M5.4h1/M5.4h2 记录的 **0.05 s blocked** 是**当时**默认值的真实结论，
  不可用于当前 0.25 s 默认的判定，反之亦然。
- 在本机样本中，0.25 s 在 no-load / hogs4 / hogs8 三种负载、每负载 3 批 × 36 个
  独立进程观测下均无 observed `time_limit` 且 digest 一致（M5.4i §9.9）；这是
  **本机候选值**，不是跨机器保证。
- `mip_max_nodes=100/10000` 的替代语义臂稳定，是因为移除了 `time_limit`；所有可行解
  `mip_node_count=1`，节点上限未实际绑定。它不能被称为“节点上限导致确定性”，
  M5.4h2 起在产物中固定为 `node_cap_effective=false`、`node_cap_candidate=false`。

历史 runs 保留且不得覆盖：

```text
runs/m54h_matrix/
runs/m54h_nodecaps/
runs/m54h1_matrix/
```

M5.4h2 新增产物：`runs/m54h2_matrix/`。

## 7. M5.4h2 的账本勘误结果（已执行）

M5.4h2 已修正下列三项，专项测试 22 项、`tests/test_m54h*.py` 67 项全通过，
`make check` exit 0（998 passed），`make smoke` exit 0：

1. `summary.parquet` 的字段名由 `options_json` 改为约定的 `options`
   （规范 JSON：键排序、无多余空白、保留非 ASCII）；未执行的 skipped 阶段写 `null`。
2. `overall` **只**反映 release gate 的最终阶段状态，与 `release_gate.blocked` 恒等；
   `attribution` 只解释原因、绝不决定放行。`summary.json` 与 `manifest.status`、
   退出码共用同一推导。
3. `candidate_notes` 固定输出 `node_cap_effective=false` 与 `node_cap_candidate=false`
   （实测未绑定），并将稳定性归因于 probe 内关闭 wall clock。

### 7.1 人工复审后的返修（2026-09-14，仍等待复审）

复审指出两处语义仍不正确，已返修（`f8a548a` → `15b8bdd`，见
`docs/task_cards/M5.4h2.md` §10）：

1. **node-cap 候选必须是同一个 cap 的证据交集**。原先把「任意 cap 绑定」与
   「任意另一个 cap 稳定」拼成 `node_cap_candidate=true`；现在由纯函数
   `cap_is_candidate` 要求**同一个 cap** 同时满足 `cap_bound`、`distinct==1`、
   `processes>=6`、`steps>=8`，并输出 `candidate_caps` 列表。
2. **证据不足必须 fail closed**。`release_gate` 新增 `passed = evaluated and
   all_qualifying and not blocked`；`manifest.status` 与退出码改由 `passed` 决定
   （原先只看 `blocked`，于是未测量默认预算、`processes<6`、`steps<8` 都会得到
   `success`/退出码 0）。`overall.blocked == release_gate.blocked` 仍恒成立；
   证据不足时 `reason` 必须写明样本量不足。

最终验收证据路径（返修后，`runs/m54h2r1_matrix/`）：

```text
runs/m54h2r1_matrix/m54h2r1_hogs4_attribution    # attribution=wall_clock_budget_dominant 且门禁 blocked
runs/m54h2r1_matrix/m54h2r1_nodealt             # cap 100/10000 均未绑定 -> candidate_caps=[]
runs/m54h2r1_matrix/m54h2r1_not_evaluated       # evaluated=false -> not_evaluated / failed / exit 1
runs/m54h2r1_matrix/m54h2r1_not_evaluated_ledger # 同上，且含完整七类产物
runs/m54h2r1_matrix/m54h2r1_underpowered        # processes<6 -> insufficient_evidence / failed / exit 1
```

旧产物 `runs/m54h2_matrix/**`：修订早于返修，**`superseded_pre_fix`**，
**不属于最终验收证据**。其中
`runs/m54h2_matrix/m54h2_hogs4_attribution_20260914T114439Z` 的
`manifest=success` **必须读作**「该次样本的门禁恰好是清的」——
**它不是**当前有效的 `released` 产物，也**不表示** M5.4 已发布。
按「不挑选重试结果」的要求，这些 run 全部**保留**，未删除、未覆盖。

**必须留意的既有事实**：release gate 的判定来自**单次抽样**。0.05 s 的不稳定是间歇的，
一次抽到全部 `distinct=1` 就能让门禁通过。返修消除了「证据不足却放行」（fail open），
但**没有**解决「一次幸运抽样即可通过门禁」。是否引入跨 run 证据合并或最小重复次数，
属 M5.4h2 范围之外，**待人工决定**。

## 7J. M1.3d-R3 第三轮返修：冻结 schema 的键集合精确性（**已通过人工审核**）

**M1.3d 第三轮审核不通过**，问题与修正（详见 `docs/task_cards/M1.3d.md` §13）：

- **冻结 schema 只校验已知字段的值，不校验键集合的精确性**，于是「新增/额外的未声明
  字段」被静默接受。实测被接受的：`splits` 增加第四项 `shadow_test`；
  split entry 增加 `randomized_indices`/`source`/`ready`；
  `train_only_statistics_source` 增加未知键；**顶层 manifest** 增加未知字段。
- **修复**：新增 `_require_exact_keys()`，对**顶层 manifest（23 项）、`splits`（恰
  train/validation/test）、每个 split entry（恰 5 项）、`readiness`、
  `train_only_statistics`、每列 statistics entry、`train_only_statistics_source`（9 项）、
  `unavailable_not_materialized`** 全部强制**精确键集合**。
  **未知字段不再静默通过 —— 未来扩展必须 bump `schema`。**
- 改前 **23 failed**（全部 `DID NOT RAISE`）；改后
  `tests/test_m13d_splits.py` **246 非 slow + 1 slow 全绿**；
  `make check` **exit 0**（1586 passed）；`make smoke` **exit 0**；
  `make train` 仍 **exit 2**、不回退 synthetic、无 checkpoint、正式 split 名未创建。

**审计账本修正（本轮）**：

- `3327346..caf0c9d` **实际为 15 个提交**（不是 14 —— 上轮少算了 R2 的开卡提交 `3c851b1`）。
- **R2 实际为 7 个提交**（不是「4 + 2」）：`3c851b1 0b18471 1ab6fca 4a64077 427528e
  84e6e72 caf0c9d`。
- 删除「记录自身无法列全属正常」这类不稳定说法，改用可稳定复算的写法：
  **截至父提交 `<已知SHA>` 共 N 个；包含本次文档提交后为 N+1**。

**最终 split manifest**：`materializer_revision` = `1d63c64`
（`1d63c643c361183c17e8e208b36d3ac7dd151b35`）；SHA-256 = `a096535fcdec81534f8cc05671d34d879a7e9517d06be789510dea586149af27`；
最终 HEAD 上复跑物化命令两次均 `exit=0`，bytes/sha/`mtime_ns` 不变。

**上游未变**：`singapore_2024.json` `d4e24d6f…`、
`singapore_2024_half_hour.json` `e6484d6b…`、`half_hour.parquet` `dec76ea2…`；
raw 与 `configs/frozen_refs/refs.json` 未动；split 边界与行数未变。

**回滚（由新到旧；已在临时 detached worktree 中只读验证逐树等于 `3327346`）**：

```bash
git revert e186ce4 1d63c64 7f8d169 50c19f4 caf0c9d 84e6e72 427528e \
           4a64077 1ab6fca 0b18471 3c851b1 e3aafdc 373b392 37ca45b \
           076923b 2c0b8c1 6a2ca5d 45e5a29 d60f597
```

**账本（稳定写法）**：截至父提交 `ffb1f06`，`3327346..ffb1f06` 共 **20 个**提交；
包含本次交接提交后为 **21 个**。**回滚必须由新到旧** —— 实测由旧到新会逐步冲突；
由新到旧则**零冲突**，且回滚后 `HEAD^{tree}` 与 `3327346` 的树逐树一致。

**M1.3e 尚未开始**：`forecast_ready=false`，正式 `ScenarioBundle` 与训练**仍 blocked**；
四项缺口仍 unavailable。

## 7K. M1.3e：contract-v8 与因果 forecast provenance（**执行完成，等待人工审查**）

`docs/task_cards/M1.3e.md`。**M1.3d 已通过人工审核**（2026-09-16，含 R1/R2/R3；
最终 `materializer_revision = 1d63c64`、split manifest SHA-256 = `a096535f…`）。

| 字段 | 值 |
|---|---|
| 分支 | `p4-safeppo-m51a-rollout-contract-m12-integration` |
| 卡片登记的严格开始 SHA | `66ce93c` |
| 本会话续做起点 | `d41ee07`（其上已有开卡 `ac5f245` 与先红测试 `d41ee07`） |
| 最终 HEAD | `75e57ef` |

**账本（稳定写法）**：截至父提交 `75e57ef`，`66ce93c..75e57ef` 共 **12 个**提交；
包含本次文档提交后为 **13 个**。

```text
ac5f245 docs(card): start M1.3e contract-v8 and causal forecast provenance
d41ee07 test: add failing M1.3e contract-v8 provenance and causal forecast regressions
222f76f feat: raise the contract to contract-v8 with structured forecast provenance
cf4f763 feat: add the causal seasonal-naive forecast provider and the oracle-debug helper
abf4f24 feat: add the forecast policy materializer for Singapore-2024
4e142e0 test: migrate the existing fixtures to contract-v8 provenance
9481f5c fix: scope the provider's upstream-hash verification and content hash
b3c9da2 test: correct the history-window mutation rows in the M1.3e leakage regression
6401029 fix: let the forecast policy materializer run as a standalone script
c45b38f feat: freeze the Singapore-2024 forecast policy manifest
a124dd3 style: take the canonical frequency from the frozen split module
75e57ef feat: regenerate the forecast policy manifest after the import fix
```

### 7K.1 contract-v8 的破坏性边界

`contracts.CONTRACT_VERSION_ID` 是**全仓唯一**版本源，由 `contract-v7` 提升为
**`contract-v8`**；`checkpointing`、`safe_rl_v2/buffer`、`safe_rl_v2/lagrangian`、
`scenario/scenario.py` 仍从该常量导入，**未新增硬编码版本源**。

破坏性后果（**预期代价**，与既有 v6→v7 的处理一致）：

- 既有 **v7 rollout buffer / checkpoint 全部失效**（明确拒绝，**不**自动填充新字段、
  **不**截断、**不**静默升级）；`runs/` 中既有 v7 历史证据**原样保留、未改写**；
- `ScenarioBundle` 形状变化：`synthetic: bool` 与无 schema 的自由 dict
  `source_hashes` **退役**，改为显式 `mode` + `generated_at` + 结构化
  `forecast_provenance`；`content_hash()` 因此覆盖全部 provenance；
- 所有合法测试 fixture 已**显式**迁移为 v8 provenance，**未**借迁移弱化任何断言。

**迁移范围**（仅两类，均为卡内授权）：

| 类别 | 文件 |
|---|---|
| 直接构造 `ScenarioBundle`、需补 v8 provenance | `test_contracts`、`test_contract_validators`、`test_m43_milp`、`test_m44_corrector`、`test_m46_probe`、`test_m310b_wind_carbon_forecast` |
| 明确断言 `contract-v7`、迁移为 v8 | `test_m310c`、`test_m41a`、`test_m41b`、`test_m41c`、`test_m51a`、`test_m51_buffer`、`test_m53a`、`test_m53c`、`test_m53e`、`test_m54a` |

> **如实登记**：`tests/test_m41a_snapshot_completeness.py` 属于第 2 类，
> 卡片原文枚举遗漏，已一并迁移。

**v7 拒绝证据**（`tests/test_m13e_forecast_provenance.py`）：
`test_v7_artifacts_are_explicitly_rejected`（buffer payload 写 `contract-v7` → 拒绝）、
`test_no_second_hardcoded_version_source`（四个文件不得含 `contract-v7` 字面量）、
`test_m51a_rollout_contract` 的 v5 旧 payload **前向改成当前版本**后仍被拒绝。

### 7K.2 provenance 模型与 purpose gate

- **构造时 fail closed**：`ForecastSeriesProvenance` / `ScenarioBundle` 各自带
  `model_validator(mode="after")`；校验实现放在 `contracts.models`
  （`validators` 依赖 `models`，反向导入会成环），`contracts.validators` 以
  `validate_series_provenance` / `validate_scenario_provenance` 委托同一实现。
- **模型**：`ArtifactDigest(role, logical_path, sha256)`；
  `ForecastSeriesProvenance`（series_name / source_kind / method / generated_at /
  information_cutoff_exclusive / target_start / target_end_exclusive /
  lookback_start / lookback_end_exclusive / model_name / model_version /
  code_revision / seed / sources）；
  `ScenarioForecastProvenance` **刻意不继承** `ContractBase`，使 `model_fields`
  **精确**等于七个 forecast 字段（否则「一一对应」无法机器判定）。
- **校验规则**：带时区的**规范** ISO-8601；`generated_at <= target_start`；
  `information_cutoff_exclusive <= target_start`；
  `lookback_end_exclusive <= information_cutoff_exclusive`；
  `lookback_start <= lookback_end_exclusive`；`target_start < target_end_exclusive`；
  `series_name` 等于所在字段名；`code_revision` 40 位小写 Git SHA；
  `sources[*].sha256` 64 位小写十六进制且 `sources` 非空；**`seed` 显式拒绝 bool**
  （pydantic 宽松模式会把 `True` 变成 `1`）；未知 mode / 未知 source_kind /
  非 `ScenarioForecastProvenance` 一律拒绝，**不泄漏** `KeyError`/`TypeError`/`AttributeError`。
- **mode ↔ 来源自洽**：`formal` 只接受 external/seasonal_naive/persistence/
  modeled_scenario；`synthetic` 拒绝 external_forecast 与 oracle_debug；
  `oracle_debug` 只接受 oracle_debug/synthetic。
- **purpose gate**（`validate_forecast_purpose`）：`training`/`evaluation` **只**接受
  `mode="formal"`，拒绝 synthetic / oracle_debug，也拒绝**不是** `ScenarioBundle`
  的 forecast artifact；只有 `debug` 接受。**本卡只建立 gate**，正式训练接线属 **M1.3g**。

### 7K.3 seasonal-naive 的精确历史窗口（含一处**人为裁定**）

冻结口径：`FORECAST_PERIOD_STEPS = 48`、`method = trailing_seasonal_naive`、
`frequency = 30min`；**只**为五个 canonical driver 生成 forecast
（`price_sgd_per_kwh` / `system_load_mw` / `temperature_deg_c` /
`wind_speed_10m_mps` / `ghi_w_per_m2`）。

对全局 origin = `i`：

- 历史模板**只**读 `[i-48, i)`；**绝不**读 `i` 或 `i` 之后的 truth；
- `forecast[k] = template[k mod 48]`（模板按**时间正序**，等价于 `y(i+k-48)`）；
- `generated_at = information_cutoff_exclusive = lookback_end_exclusive = origin`；
- `target = [origin, origin+C)`，`C` 必经 M1.3d `validate_forecast_origin`
  （越界拒绝，不截断、不换段）；
- train 内 `i < 48` fail closed（不回填、不跨年环绕）；validation/test 起点可以
  使用其**之前已发生**的 canonical 历史；
- 无随机数（`seed` 明确为 `null`）；**未**引入任何拟合。

> ⚠️ **必须记住的人为裁定**：先红测试 `test_history_window_changes_the_forecast`
> 原先只改 `origin-1`（第 199 行）却要求 `cutoff=4` 下的 forecast 变化 —— 按上述
> 规则，第 k 项消费行是 `origin-48+k`，`origin-1` 落在模板下标 **47**，
> 只有 `C > 47` 才会被消费。**卡片文字与已提交测试互相矛盾**，已上报人工，
> **裁定为取卡片字面规则**，并把该测试的触发位置改为**整个历史窗口**
> `range(origin-PERIOD_STEPS, origin)`。**断言强度不变**（仍是「历史窗口的任何变化
> 都必须体现在 forecast 上」），没有删除、没有反转、没有放宽。

**另一处由 leakage 回归暴露的语义**：artifact 的 `content_hash()` 只覆盖
**预测内容本身**（contract_version / split / origin / global_origin /
forecast_cutoff / frequency / method / period_steps / generated_at /
target_timestamps / series）；上游制品的 sha256 随上游字节变化而预测未必变化，
把它们算进 content hash 会让「同一份预测」在上游重物化后得到不同身份。
上游 digest 仍完整保留在 `to_dict()["provenance"][*]["sources"]` 供审计。
provider 的上游校验只覆盖**它真正读到的字节**（split manifest 的
`canonical_parquet_sha256`、canonical manifest 的 `output_parquet_sha256` 与 parquet
实测 sha256 三者相等 + schema/行数/时区/频率 + 整条 canonical 时间轴）；
`canonical_manifest_sha256` 由 M1.3d `load_truth_split` 读取链负责。

**future-truth mutation leakage 证据**（全部 `@pytest.mark.leakage`，五个 driver 参数化）：

| 测试 | 断言 |
|---|---|
| `test_origin_and_future_truth_do_not_change_the_forecast` | 改 `[origin, end)` 全部 truth + 同步更新两个 manifest 的 parquet hash → **forecast 值与 content hash 均不变** |
| `test_history_window_changes_the_forecast` | 改整个 `[origin-48, origin)` → forecast **必变** |
| `test_validation_future_mutation_does_not_change_the_current_forecast` | validation 内 origin 之后的 truth mutation → 当前 origin 的 content hash **不变** |

### 7K.4 oracle helper 与 snapshot adapter

- `build_scenario_from_true` → **`build_oracle_debug_scenario_from_truth`**，
  `oracle_debug` 为**必填 keyword-only**：漏传 `TypeError`、传假值 `ValueError`；
  产物 `mode` 恒为 `"oracle_debug"`，`purpose=training/evaluation` 时被 gate 拒绝。
- M1.3a 的「可见 truth 改变 bundle」两条断言**改名保留为 oracle-debug 语义**
  （未删除、未反转）：把 `[t, t+cutoff)` 真值当作预测**正是 oracle-debug 的定义**；
  正式 causal provider 的 leakage 回归是**独立**的另一组测试。
- `planning/snapshot_adapter.py` 顶部明确标注 **oracle_debug / dev-only，
  不能标 formal**：其「可见预测」就是 env 真值窗口、`load_forecast` 仍为全零占位；
  产出的 bundle 恒为 `mode="oracle_debug"` 并在训练 purpose 下被拒绝。
  **本卡不在 adapter 中伪造正式预测**（M1.3g 再接正式 provider）。

### 7K.5 policy manifest 与 readiness

| 项 | 值 |
|---|---|
| 路径 | `data/manifest/singapore_2024_forecast_policy.json`（入库） |
| schema | `m1.3e-singapore-2024-forecast-policy-v1` |
| `materializer_revision` | **`a124dd3f7c50dfb92d077eb5f8c044bd849cff03`** |
| SHA-256 | **`8fcb2af01ed79d2d74d28dc85281ea67757711037def1452b0bdc032aedf3115`** |
| 含 `/Users/` 或绝对路径 | **否**（全部为仓库相对逻辑路径） |

- 冻结 19 项键集合，**未知字段拒绝**（未来扩展必须 bump schema）；
- **幂等**：最终 HEAD 上复跑 2 次均 `exit=0`，bytes / sha / `mtime_ns` 全不变；
- **原子**：首次写入失败 → 目录为空，不留半份 manifest 或临时文件；
- **已存在且不同 → 拒绝覆盖**（本卡实测触发过一次，故 manifest 有两版提交）；
- **dirty generator 拒绝**：`scenario/forecast.py` 或物化器有未提交修改即拒绝；
- **上游 hash 不符 fail closed**。

```text
readiness.available_driver_forecasts_ready   = true
readiness.complete_scenario_forecasts_ready  = false
readiness.formal_scenario_bundle_ready       = false
readiness.formal_training_ready              = false
```

**四项仍 unavailable**（未生成 forecast、未进 `ScenarioBundle`、未零填）：
`local_pv_kw`、`wind_generation_kw`、`carbon_intensity`、`arrival`。

### 7K.6 上游未变与当前门禁

`singapore_2024.json` `d4e24d6f…`、`singapore_2024_half_hour.json` `e6484d6b…`、
`singapore_2024_splits.json` `a096535f…`、
`half_hour.parquet` `dec76ea2…` —— **全部未变**；raw 未动；
`configs/frozen_refs/refs.json` 未动；**split 边界与行数未变**（10224/2928/4416）。
split manifest 的 `readiness.forecast_ready` **仍为 `false`**（有回归断言）。

| 命令 | 结果 |
|---|---|
| `pytest tests/test_m13e_forecast_provenance.py -q -m "not slow"` | **105 passed** |
| `pytest tests/test_m13e_forecast_provenance.py -q -m slow` | **1 passed** |
| `pytest tests/test_m13*.py tests/test_contracts.py tests/test_contract_validators.py -q` | **493 passed** |
| 第二组（checkpointing / m310c / m41a / m41c / m51* / m52* / m53*） | **464 passed** |
| `make check` | **exit 0**，**1691 passed**, 45 deselected |
| `make smoke` | **exit 0** |
| `make train` | **exit 2**（缺 `data/manifest/train.json`），**未**回退 synthetic、无 checkpoint |
| `train.json` / `validation.json` / `test.json` | **均未创建** |
| `git diff --check` / `git status --short` | 空 / 空 |

**正式训练仍 blocked**：`forecast_ready=false`、`formal_scenario_bundle_ready=false`、
`formal_training_ready=false`。**下一步是 M1.3f**（PV / 风电发电量 / 碳强度 /
arrival 的人工批准口径或数据接入），**不是 M6**；**不新增 M5.5**。

**本卡范围外只读登记（未修改）**：`safe_rl_v2/train.py` 的文档字符串首行仍写
「唯一数据来源是 contract-v7 `RolloutBuffer`」；其外层错误信息改为 M1.3 readiness
属 **M1.3g**（`config["contract_version"]` 实测已是 `contract-v8`）。

### 7K.7 精确回滚（**由新到旧**）与只读验证

```bash
git revert 75e57ef a124dd3 c45b38f 6401029 b3c9da2 9481f5c 4e142e0 \
           abf4f24 cf4f763 222f76f d41ee07 ac5f245
```

```text
66ce93c 的树                = a9129c3a2794bbc40137f7aa89f173255402f8e2
revert 链后 HEAD^{tree}     = a9129c3a2794bbc40137f7aa89f173255402f8e2
→ ✅ 逐树一致（零冲突；临时 detached worktree 已移除，主分支指针未移动）
```

`data/manifest/singapore_2024_forecast_policy.json` 为本卡新增，可随 revert 删除；
**不**删除 raw、**不**改 canonical/source/split manifest、**不**移动任何分支指针。
**回滚必须由新到旧**（M1.3d §13.12 实测得到的操作约束）。

### 7K.8 M1.3e-R1：第一轮审核返修（**执行完成，等待人工复审**）

**M1.3e 第一轮人工审核不通过**；返修区间 `1d9a91d..bee96f7`（**7 个**提交），
详见 `docs/task_cards/M1.3e.md` §14–§15。

**审核给出的六个阻塞项，全部修复：**

1. **policy manifest 未进入 provider 信任链** → provider 现在**必填**
   `policy_manifest_path`，逐层校验 policy → split → canonical manifest → parquet。
2. **provider 接受伪造的最小 split manifest** → split manifest 走 **M1.3d 的
   `load_truth_split` 完整严格校验**（复用，不复制宽松校验器）；最小伪造 JSON、
   extra/missing key、错误 readiness、伪造 train 统计等 13 类篡改全部拒绝。
3. **provider 接受任意 `code_revision`** → **删除**该公开参数；revision 只能由内部
   Git resolver 从 `FORECAST_SOURCE_PATHS`（provider + policy 物化器）解析，
   并与 policy 的 `materializer_revision` **恒等**。
4. **`ContractBase` 可显式声明 `contract-v7`** → 在**基底类**统一锁定
   `schema_version == CONTRACT_VERSION_ID`；旧版本、空串、bool、数字、list、dict、
   None 全部干净拒绝。
5. **`AvailableExogenousForecast` 不是严格冻结契约** → 改为 frozen Pydantic 契约
   （`AvailableSeries`/`AvailableDriverProvenance`，五个 driver 精确齐全、嵌套为
   不可变 tuple、整数语义字段显式拒绝 bool）；并拆出**两个语义不同的摘要**：
   `prediction_hash()`（只覆盖预测数值/单位/顺序）与 `content_hash()`（覆盖含审计
   provenance 的完整 artifact）。
6. **mode/source_kind 语义过宽** → `synthetic`/`oracle_debug` 要求七条**逐项**为唯一
   来源；`formal` 只允许 external/seasonal_naive/persistence/modeled_scenario；
   **任何** mode 的完整 bundle 都不得以 `unavailable` 占位；
   `ScenarioBundle.generated_at` 与七项 provenance **逐项恒等**。

**leakage 测试已按审核要求下沉**：因果性断言落在**纯函数**
`seasonal_naive_forecast(series, origin, cutoff)` 上（卡片字面规则
`forecast[k] = y(origin + k − 48)`，模板按时间正序）；provider 层则用**两条各自
完整自洽的冻结链**做对照（不是「同步改几个 hash 绕过校验」），并新增一条边界回归
固定「只有模板下标 `< C` 的行参与计算」。

**改前实测**（在临时 detached worktree 中检出「先红」提交 `ed5b877`）：
`216 collected, 115 failed, 0 errors`。

| 命令 | 结果 |
|---|---|
| `pytest tests/test_m13e_forecast_provenance.py -q -m "not slow"` | **216 passed** |
| `pytest tests/test_m13e_forecast_provenance.py -q -m slow` | **1 passed** |
| `pytest tests/test_m12*.py tests/test_m13*.py tests/test_contracts.py tests/test_contract_validators.py -q` | **626 passed** |
| `make check` | **exit 0**，**1802 passed**, 45 deselected |
| `make smoke` | **exit 0** |
| `make train` | **exit 2**，**未**回退 synthetic、无 checkpoint |
| `train.json` / `validation.json` / `test.json` | **均未创建** |

**最终 policy manifest**：`materializer_revision =
05ad5521a14e9b6e04bcdc1f00655f9a135b1574`；SHA-256 =
`d2221802dbd99381f805e6ab76c517b3d73da058be240f1e451583d99d79ce8d`；
最终 HEAD 复跑两次 bytes/hash/`mtime_ns` 不变；首冻失败原子；已存在且不同拒绝覆盖。

**上游仍未变**：`d4e24d6f…` / `e6484d6b…` / `a096535f…` / `dec76ea2…`；
`configs/frozen_refs/refs.json` 未动；split 边界未变。
**本轮未修改 `scenario/splits.py`**（因此其 materializer revision 与冻结 split
manifest 未变）；实际变更文件仅 `contracts/{__init__,models,validators}.py`、
`scenario/forecast.py`、`scripts/materialize_singapore_forecast_policy.py`、
`tests/test_m13e_forecast_provenance.py`、policy manifest 与两份 docs。

**账本勘误（审核指出）**：上一轮结束报告把已逐条列出的 **14** 个提交
（`66ce93c..1d9a91d`）写成了「共 13 个」——列表正确、计数错误。
稳定写法：**截至父提交 `1d9a91d`，`66ce93c..1d9a91d` 共 14 个**；
加本轮 8 个后，`66ce93c..HEAD` 共 **22 个**。

**回滚（由新到旧，已只读验证零冲突）**：

```bash
git revert bee96f7 05ad552 89f915b f5d5656 d681608 ed5b877 18f598e
# 1d9a91d 的树 = 75e5348f95ab369f698184f2de7f58ce64c517e3 = revert 后 HEAD^{tree}
```

**M1.3e-R1 通过人工复审之前：M1.3e 不得视为通过；M1.3f 未开始；
`forecast_ready` 不得因「只存在 policy manifest」就提前声明**
（policy 的 readiness 只有 `available_driver_forecasts_ready=true`，
其余三项 false；split manifest 的 `forecast_ready` 仍为 false）。

## 7I. M1.3d-R2 第二轮返修（已被 7J 取代）

> ⚠️ **本节的历史说法是历史错误，以 7J / R3 为准**：本节（及其引用的 R2 任务卡 §12.11）
> 曾把 `3327346..caf0c9d` 记为 **14 个**提交、并把 R2 描述为「4 + 2」。
> 实际为：`3327346..caf0c9d` 共 **15 个**提交；**R2 共 7 个**
> （`3c851b1 0b18471 1ab6fca 4a64077 427528e 84e6e72 caf0c9d`）。
> 历史提交不重写、不删除，仅在此标注为历史错误。

**M1.3d 第二轮审核不通过**，问题与修正（详见 `docs/task_cards/M1.3d.md` §12）：

1. **`train_only_statistics_source.columns` 的类型审计泄漏 `TypeError`**：
   原写法 `list(source.get("columns") or [])` 对 `int`/`bool`/`float` 等**不可迭代标量**
   直接抛 `TypeError`。现改为 `_require_str_list()`：必须是 `list`、元素全为 `str`、
   顺序与内容**严格等于** `STATISTIC_COLUMNS`。
2. **`unavailable_not_materialized` 的容器条目被接受**：原先把非字符串条目
   `json.dumps` 成文本再查关键词，于是 `["unavailable"]` / `{"status":"unavailable"}`
   可以绕过。现在每个条目**必须是字符串**，容器**即使文本含 "unavailable" 也一律拒绝**。
3. **`frozen_at_utc` 完全未校验**：缺失 / 非字符串 / 无时区 / 非规范格式此前**全部被接受**。
   现由 `_require_canonical_utc()` 在索引任何其他字段**之前**校验：必须是**规范 UTC**
   （`fromisoformat` 可解析、带 `tzinfo`、归一化后与原串逐字符相等、以 `+00:00` 结尾）。
4. **其余所有嵌套外部字段**统一走严格校验器，**任何**畸形输入都不泄漏
   `KeyError`/`TypeError`/`AttributeError`/`IndexError`（测试侧由 `_assert_no_leak` 强制）。

**审计账本勘误（本轮补记）**：

- **补列真实提交 `2c0b8c1`**（`feat: regenerate the M1.3d split manifest under strict
  validation`）—— 首轮返修链的真实成员，首轮任务卡表格当时漏列。
- **`3327346..e3aafdc` 首轮返修链实际有 8 个提交**（不是 5 个）：
  `d60f597 45e5a29 6a2ca5d 2c0b8c1 076923b 37ca45b 373b392 e3aafdc`。
- `3327346..HEAD` 的完整链共 **14 个提交**（首轮 8 + R2 实现/产物 4 + R2 证据/交接 2）；
  下方回滚命令列出前 12 个，不含记录本回滚的两个 docs 提交（记录自身的提交
  无法在记录中列全，属既有正常限制），已实测可精确恢复 `3327346` 的树。

**改前 16 failed**（直接原因：3 项 `TypeError: 'int'/'float'/'bool' object is not
iterable`，13 项 `DID NOT RAISE`）；**改后** `tests/test_m13d_splits.py`
**224 非 slow + 1 slow 全绿**（首轮 129 项全部保留、未弱化）；
`make check` **exit 0**（1514 passed）；`make smoke` **exit 0**；
`make train` 仍 **exit 2**、不回退 synthetic、无 checkpoint、
`train.json`/`validation.json`/`test.json` **未创建**。

**最终 split manifest**：`materializer_revision` = `1ab6fca`
（`1ab6fcacdc86e3114942ca2bc0c956a73acec37e`）；SHA-256 = `f8006aa0db32a80504304cd1c3d8655887e2742b6860d75ed1687fff96398be1`；
最终 HEAD 上复跑物化命令两次均 `exit=0`，bytes/sha/`mtime_ns` 不变。

**上游未变**：`singapore_2024.json` `d4e24d6f…`、
`singapore_2024_half_hour.json` `e6484d6b…`、`half_hour.parquet` `dec76ea2…`；
raw 与 `configs/frozen_refs/refs.json` 未动；**split 边界与行数未变**。

**精确回滚（含 `2c0b8c1`；已用临时 detached worktree **只读**验证可恢复 `3327346` 的树）**：

```bash
git revert 4a64077 1ab6fca 0b18471 3c851b1 \
           e3aafdc 373b392 37ca45b 076923b 2c0b8c1 6a2ca5d 45e5a29 d60f597
```

```text
3327346 的树                = 19120a3c9ca21157bc4f82568db9f6233a677ee9
revert 链后 HEAD^{tree}     = 19120a3c9ca21157bc4f82568db9f6233a677ee9   ✅ 一致
```

**M1.3e 尚未开始**：forecast provenance / contract-v8 未动；`forecast_ready=false`，
正式 `ScenarioBundle` 与训练**仍 blocked**；四项缺口仍 unavailable。

## 7H. M1.3d 第一轮审核返修（已被 7I 取代）

**M1.3d 第一轮审核不通过**，两个根因（详见 `docs/task_cards/M1.3d.md` §11）：

1. **split manifest 未被严格验证**：reader 只证明了「生成时写对了」，
   没有拒绝**随后被篡改的冻结声明**。实测 15 类篡改（`year`、`step_minutes`、
   两个路径声明、`materializer_revision` 格式、`no_overlap`/`no_gap`/`randomized`/
   `leap_day_split`、两条 origin 规则、`readiness`、`unavailable` 四项、
   `train_only_statistics` 清空/伪造）**全部被接受**。
2. **canonical 时间轴未被严格验证**：只查首/末行与总行数。
   在 canonical manifest hash 同步更新、自洽的前提下，
   实测**内部交换相邻两行、重复时间戳、非网格时间戳、非单调、边界换行**
   **全部被接受**（物化器与 reader 同）。

**修复**：`validate_canonical_timeline()` 校验**整表**（tz-aware、严格 30min 网格、
严格递增、唯一、首末精确），**首次冻结前**与 **reader 返回切片前**各调用一次；
`_read_split_manifest()` 现在校验上述**全部**字段，并要求**路径声明与调用者实际提供的
logical repo path 一致**、`materializer_revision` 为 40 位小写 Git SHA；
`_verify_train_statistics()` 把 `train_only_statistics` 与 canonical **train 段重算值**
逐字段比对（伪造/清空/`bool`/`NaN` 均拒绝）；`no_overlap`/`no_gap`/`randomized`/
`leap_day_split`/两条 origin 规则/`readiness` 四态/`unavailable` 四项**必须为严格期望值**。

**证据**：改前 **43 failed**（全为 `DID NOT RAISE`）；改后
`tests/test_m13d_splits.py` **128 非 slow + 1 slow 全绿**；
`make check` **exit 0**（1418 passed）；`make smoke` **exit 0**；
`make train` 仍 **exit 2** 且不回退 synthetic，无 checkpoint，
`train.json`/`validation.json`/`test.json` **未创建**。

**最终 split manifest**：`materializer_revision` = `076923b603e961652593e40f5e3b8869315b81e9`；
SHA-256 = `b1eab71aa92f01093fb396519eb85d44e89641bc0fbee646ba607548edced4ca`；最终 docs 提交后复跑物化命令两次均 `exit=0`，bytes/sha/`mtime_ns` 不变。
**上游未变**：`singapore_2024.json` `d4e24d6f…`、`singapore_2024_half_hour.json`
`e6484d6b…`、`half_hour.parquet` `dec76ea2…` 全部未变；raw 与
`configs/frozen_refs/refs.json` 未动；**split 边界与行数未变**（10224/2928/4416）。

**一次未复现的既有 flake（与本卡无关）**：`make check` 首跑有 1 次
`test_m54g_raw_projection_deterministic_options.py` 失败；重跑 exit 0，
该文件单独连跑 5 次 4 绿 1 红 —— 属 M5.4g corrector 跨进程确定性既有的偶发 flake，
本卡未触碰 corrector/planning/求解路径，且该文件不在允许范围内，故只登记不修改。

**M1.3e 尚未开始**：forecast provenance / contract-v8 仍未动；
`forecast_ready=false`，正式 `ScenarioBundle` 与训练**仍 blocked**；
四项缺口仍 unavailable。

## 7G. M1.3d：连续 truth split 首轮冻结（已被 7H 取代）

`docs/task_cards/M1.3d.md`（`49084b6` → `4d82317`）。**M1.3c 审计通过**，
人工**批准月对齐方案 B**。本卡**只**冻结 truth 切分与 train-only 描述统计。

- **冻结切分**（半开区间、`Asia/Singapore`、半小时）：

  | split | 区间 | 行段 | 行数 |
  |---|---|---|---|
  | `train` | `[2024-01-01T00:00+08:00, 2024-08-01T00:00+08:00)` | [0, 10224) | 10224 |
  | `validation` | `[2024-08-01T00:00+08:00, 2024-10-01T00:00+08:00)` | [10224, 13152) | 2928 |
  | `test` | `[2024-10-01T00:00+08:00, 2025-01-01T00:00+08:00)` | [13152, 17568) | 4416 |

  合计 **17,568** 行；无重叠、无缺口、不随机打散；**Feb 29 只在 train**。
- **api**：`scenario/splits.py` 的 `load_truth_split(split, *, canonical_parquet_path,
  canonical_manifest_path, split_manifest_path)`（逐级校验 hash，返回副本，不截断/不换段）
  与 `validate_episode_origin` / `validate_forecast_origin`。
- **origin 门禁（取代草案的「统一 `H+C` purge」）**：`origin_index + H <= row_end_exclusive`
  且 `origin_index + C <= row_end_exclusive`；**`H >= C` 不重复扣除 `C`**（有专门回归）；
  `origin` 为 **split 本地** step，越界**报错不截断**；`H`/`C` 必须是严格正整数。
- **train-only 描述统计**（真实数据实测）：`national_igs_mwh_per_half_hour`
  count=10224、negative=**4519**、min=**-0.1140**；`price_sgd_per_kwh`
  count=10224、negative=**3**、min=**-0.0202** —— **signed IGS 与负电价原值保留**。
  只由 train 行参与（validation/test mutation 不改变统计，有回归）。
  **这些只是描述统计，不得冒充正式 normalization refs**。
- **manifest**：`data/manifest/singapore_2024_splits.json`，`materializer_revision`
  由 Git 解析（= 最后修改 split 生成实现的提交），路径**仓库相对**、无绝对路径，
  **幂等**（重跑 bytes/sha/`mtime_ns` 不变）、**原子**（失败回滚不留半成品）、
  已存在且不同则拒绝覆盖、generator dirty 时拒绝。
- **readiness**：`truth_splits_ready=true`，而 `forecast_ready=false`、
  `formal_scenario_bundle_ready=false`、`formal_training_ready=false`。
  **`train.json` / `validation.json` / `test.json` 一律未创建**（保留名）。

**truth 仍然不能当 forecast**：本卡只冻结历史事实切分；
四项缺口（`local_pv_kw` / `wind_generation_kw` / `carbon_intensity` / `arrival`）
**仍 unavailable**；`configs/frozen_refs/refs.json` **未动**，
其正式冻结**最迟必须在 M1.3g 训练接线前**完成（**不是 M6**）。
**正式 `ScenarioBundle`、训练与 M6 仍未开始**；`make train` 仍 exit 2。

**下一步是 M1.3e（forecast provenance / contract-v8），不是 M6。不新增 M5.5。**

## 7F. M1.3c：正式 ScenarioBundle 接线前置审计（**已通过**）

`docs/task_cards/M1.3c.md`。**M1.3b 第二次返修已正式通过**；本卡是**只读**审计 ——
不改任何代码/数据/契约/manifest，不创建 `train.json`，不开始训练。

### 7F.1 最重要的发现：正式链会把「未来真值」当作 forecast

- `scenario/scenario.py:65-66` 的 `_window()` 把传入 `true` 数组的**前 `forecast_cutoff` 个
  元素**直接当作 `*_forecast`；函数**无法**保证这些值来自「时刻 t 可获得的信息」。
- `planning/snapshot_adapter.py:128-134` 的「可见预测」**就是**
  `env.price_t/pv_t/wt_t/T_amb/carbon_factor_t` 在 `[t, t+cutoff)` 的**真值**；
  其中 `load_forecast=[0.0]*cutoff`（129 行）是**全零数组**静默冒充系统负荷预测。
- 现有 leakage 测试**只**验证 cutoff **之外**不泄漏
  （`tests/test_m13a_canonical_bundle.py:109-119`）；而
  `test_visible_truth_mutation_changes_bundle`（122-128 行）**断言 cutoff 内的真值进入
  forecast 会改变 bundle** —— 即**把该缺陷固化成了正确行为**。
- **结论：在 M1.3e 重写该测试之前，不得开始正式训练。**

### 7F.2 契约判断：`ScenarioBundle` 不足以承载 provenance

`contracts/models.py:25-56` 缺 `generated_at`、forecast origin、model/version、seed、
visible window 语义，以及 truth/forecast/oracle-debug 分类（只有 `synthetic: bool`）；
`source_hashes` 是无 schema 的自由 dict，无法逐序列区分
external/modeled/persistence/zero-fill。`contracts/validators.py:72-84` 的
`validate_scenario()` **只**校验长度与单位，**完全不校验 provenance**。
**推荐 bump 到 `contract-v8`**（完整影响面见卡片 §7.1：单一版本源、≈50 个引用文件、
v7 buffer/checkpoint 全部失效 —— 属预期代价）。卡内另列了更窄的替代方案供人工选择。

### 7F.3 连续切分：推荐方案 B（月对齐）

| 方案 | train | val | test |
|---|---|---|---|
| A 严格 60/20/20 | 01-01→08-07（220 天，60.1%） | 08-08→10-13（67 天，18.3%） | 10-14→12-31（79 天，21.6%） |
| **B 月对齐（推荐）** | **01-01→07-31（213 天，58.2%）** | **08-01→09-30（61 天，16.7%）** | **10-01→12-31（92 天，25.1%）** |
| C 长训练 | 01-01→09-11（255 天，69.7%） | 09-12→10-26（45 天，12.3%） | 10-27→12-31（66 天，18.0%） |

三者合计均为 366 天 / 17,568 行；闰日 Feb 29 均落在 train；test 分别为 79/92/66 天，
均 ≥ 30 天。**共同约束**：不随机打散；episode 不得跨 split 边界（`t ≤ split_end − H`）；
边界丢弃 `H + C` 的 **purge gap**；refs **只在 train** 冻结。
推荐 B 的理由：切分点落在自然月边界、可肉眼审计、闰日无歧义、三段季节不重叠；
代价是 58.2/16.7/25.1 偏离计划书字面的 60/20/20（该值为**指导值**）。
**本卡不创建 split manifest，需人工确认后再由 M1.3d 落地。**

### 7F.4 forecast 口径与四项缺口（**全部仍需人工决定**）

- **forecast 三类严格区分**：truth（M1.3b canonical 历史事实，只作执行真值与评估参照）、
  forecast（带 `generated_at`/窗口/模型 revision/seed）、oracle（**仅 debug**）。
  方案 A（外部真实预测产品）**当前不可得**；推荐先以 **B（仅用 train 拟合/校准的
  persistence / seasonal-naive）** 起步。
- **四项缺口目前全部 unavailable**：`local_pv_kw`、`wind_generation_kw`、
  `carbon_intensity`、`arrival`。硬约束：national IGS **不得**改名 IDC local PV；
  ERA5 wind speed **不得**当作风电发电量；carbon **不得**用未注明来源的常数/默认日曲线；
  arrival **不得**用已抽样出的未来真实任务序列生成正式 forecast；
  **全零数组不得**作为缺失兼容方案静默进入正式链；不满足时**保持 blocked**。
- 链上现存的默认值构造（`data_io/data_loader.py:71` 默认 PV 曲线、
  `snapshot_adapter.py:129` 全零负荷、`idc_model/task_forecast.py` 的 `perfect` oracle 模式）
  在正式模式下必须禁用。

### 7F.5 后续实现卡（最小、可回滚）

**M1.3d**（split manifest + train 段 refs + 边界校验，需先人工确认方案 B）→
**M1.3e**（forecast artifact/provenance 契约 + **重写** leakage 测试）→
**M1.3f**（PV/wind/carbon/arrival 的人工批准口径或数据接入）→
**M1.3g**（正式 `ScenarioBundle` + env/train 接线，并把 `train.py` 的「需要 M1.2 数据」
错误信息改为 M1.3 readiness）。**不新增 M5.5，不提前进入 M6。**

**正式 `ScenarioBundle`、训练与 M6 均未开始**；`data/manifest/train.json` **仍不存在**；
`make train` 仍非零失败。**下一步必须等待人工确认 split 与 forecast / 缺失字段口径。**

## 7E. M1.3b 第二次审核返修（**已通过**）

**第二次人工审核仍未通过**：上一轮的 revision 语义、路径可移植、正常幂等、
已有 manifest 的失败保护与测试标记整改**均已确认通过**；只剩**首冻失败原子性**一个阻断缺陷 ——
原实现先把候选 parquet 安装到正式路径、再写 manifest，manifest 失败会留下**孤立正式 parquet**。
返修提交 `f364bed` → `3b235f1`；详见 `docs/task_cards/M1.3b.md` §12。

**修复要点**：

- **首冻原子回滚**：parquet 与 manifest 的安装包在同一个 `try` 内，记录本次**创建**的
  文件；任一异常即删除本次创建的文件后重抛。回滚只处理本次创建的路径，不误删调用前已存在的文件。
- **孤立 parquet fail closed**：manifest 不存在但正式 parquet 已存在 → 直接拒绝，
  不覆盖/删除/修改，不创建 manifest，不留临时文件。
- **frozen manifest 结构校验**：索引字段前先校验顶层为 object、
  `output_parquet_sha256` 为 64 位小写十六进制、`materializer_revision` 为 40 位小写十六进制；
  畸形一律 `ValueError`，CLI 干净 `exit 1`，不泄漏 `KeyError`/`TypeError`。
- **dirty generator 拒绝**：生成实现文件有未提交修改时**拒绝生成**
  （不能用旧 revision 为未提交代码背书）。测试侧按审核指南 monkeypatch Git 查询，
  另有 `test_dirty_generator_is_rejected` 显式验证。

**零半成品实测**：首冻时注入 manifest 临时写入失败 / `os.replace` 失败 / 一般 `OSError`，
三种情况均**完全回滚**（parquet 不存在、manifest 不存在、无临时文件）。

**数据语义零变化**：parquet SHA-256 仍为
`dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd`（逐字节相同）。
`materializer_revision` 自然更新为 `7007f36`；canonical manifest 已基于该提交重新生成，
路径仍为仓库相对形式，`/Users/` = False。最终 HEAD 幂等连跑两次 `exit=0` 且 mtime 不变。

**M1.3b 仍只是半小时事实表**：`ScenarioBundle` 接线、连续切分、forecast 可见窗口与
泄漏门禁属 **M1.3c**；`data/manifest/train.json` **仍不存在**，正式 `make train` 仍非零失败
（原因已正确指向 M1.3），**未**产生 checkpoint；**M6 尚未开始**。

## 7D. M1.3b 第一次审核返修（已被 7E 取代）

**第一次人工审核未通过**（五个问题：最终 HEAD 幂等失败、docs-only 提交使冻结数据失效、
manifest 不一致时先改写 parquet、完全相同也重写、路径不可移植；
另有 27 项测试被标 slow 导致 `make check` 只跑 6 项）。
返修提交 `c14f349` → `64bed88`；详见 `docs/task_cards/M1.3b.md` §11。

**修复要点**：

- **revision 语义**：字段由随 HEAD 漂移的 `code_revision` 改为
  **`materializer_revision`** —— 由 Git 解析「最后修改 `scenario/singapore_2024.py`
  或 `scripts/materialize_singapore_2024.py` 的提交」。**docs-only 提交不再使冻结数据失效。**
- **失败原子性**：候选 parquet 只写同文件系统临时文件；先算候选 hash 与候选 manifest
  并**比较完毕**，才在安全时 `os.replace`。不一致 → 正式 parquet/manifest
  **bytes/hash/mtime_ns 全不变**、无临时文件残留；一致 → **不重写**；
  正式 parquet 损坏 → 仅当候选 hash 与冻结值相符时原子恢复。
- **路径可移植**：入库 manifest 只记**仓库相对逻辑路径**
  （`data/raw/singapore_2024/...`、`data/manifest/singapore_2024.json`），
  **无** `/Users/...`、无任何绝对或 home 路径。
- **数据语义零变化**（可证明）：返修前后 parquet SHA-256 **完全相同**
  （`dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd`）——
  17,568 行、signed IGS 原值、USEP ×0.001、SASEA 唯一实际负荷、天气 age 0/30 全部不变。
- **测试标记整改**：slow 由 **27 → 1**（只剩真实冻结 raw 全量验收）；
  全年 fixture 与读取结果模块级缓存；**未删除测试、未弱化断言**。
  测试耗时 **3m46s → 13.9s**。

**⚠️ 一处越界说明（供审阅者判断）**：为使非 slow 门禁可用，本次还修改了审核范围外的
`scenario/singapore_2024.py`，**仅**修 `_require_complete` 的 O(n²) 性能缺陷
（循环内重建 `set(expected)`，实测每调用 ≈2.5 s）。该修正**输出保持**，
并由上述 parquet 字节同一性证明未改变任何数据语义。

**最终 HEAD 幂等**：同一物化命令连跑两次均 `exit=0`，parquet 与 manifest 的
`mtime_ns` **均不变**，无临时文件残留。

**M1.3b 仍只是半小时事实表**：`ScenarioBundle` 接线、连续 train/validation/test 切分、
forecast 可见窗口与泄漏门禁属 **M1.3c**；`data/manifest/train.json` **仍不存在**，
正式 `make train` 仍非零失败（原因已正确指向 M1.3），**未**产生 checkpoint；**M6 尚未开始**。

## 7C. M1.3b：Singapore-2024 半小时 canonical reader（首轮，已被 7D 取代）

`docs/task_cards/M1.3b.md`（`ac8ba3a` → `485491d`）。**M1.2e 已正式通过**，
本分支即后续累计工程链（**不**反向合并到旧 `p4-safeppo-m51a-rollout-contract`）。

- **只读 canonical reader**：`scenario/singapore_2024.py`
  （`load_singapore_2024_half_hour(raw_dir, source_manifest_path)`）。
- **物化**：`scripts/materialize_singapore_2024.py`。
  Parquet 落 `data/processed/singapore_2024/half_hour.parquet`（**gitignore，不入库**）；
  canonical manifest 入库：`data/manifest/singapore_2024_half_hour.json`。
- **实测**：**17,568 行**，`2024-01-01T00:00+08:00` → `2024-12-31T23:30+08:00`，
  `frequency=30min`，时区 `Asia/Singapore`（tz-aware），output parquet SHA-256
  `dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd`。
- **天气映射规则（已写入 manifest，非仅藏在实现里）**：
  `latest source timestamp not later than target timestamp` —— 目标 `h:00` 用天气 `h:00`
  （age=0），目标 `h:30` 仍用天气 `h:00`（age=30）；逐行 `weather_source_timestamp <= timestamp`。
  禁止下一小时填充 / 线性插值 / centered resampling / backfill。
- **⚠️ 人工授权的契约修正**：卡原要求 IGS **非负**，但冻结的真实数据有
  **7,657/17,568（43.6%）为负**（最小 -0.124 MWh）—— 设施净耗电时段的净注入本就为负。
  经**人工批准**改为「**有限**（允许负值）」，**逐值原值保留、未 clip、未取绝对值**；
  manifest 的 `observed_value_ranges` 记录 min/max/负值行数。
  `system_load_mw` / `wind_speed_10m_mps` / `ghi_w_per_m2` **仍保持非负**（实测均非负）。
- **unavailable（绝不伪造）**：`local_pv_kw`、`wind_generation_kw`、`carbon_intensity`、
  `arrival` 逐项登记为 `unavailable`；national IGS ≠ IDC 本地 PV、ERA5 风速 ≠ 本地
  风电实测、无冻结碳强度、arrival 未入口径。
- **未改动**：raw 四文件的 SHA/字节/mtime 与 `singapore_2024.json` **字节完全不变**；
  未联网；未生成新时间戳。幂等重跑 OK；篡改 canonical manifest 后重跑**拒绝覆盖**（exit 1）。
- **`scenario/scenario.py`** 只改过时的阻塞措辞：错误信息现在指向
  **M1.3 正式数据集/split manifest 尚未完成**，不再误称「M1.2 阻塞」。

**M1.3b 只完成 canonical 半小时事实表**：仍**没有** `ScenarioBundle` 接线、
连续 train/validation/test 切分、forecast 可见窗口与泄漏门禁 —— 那些属 **M1.3c**。
`data/manifest/train.json` **仍不存在**，正式 `make train` 仍非零失败（原因已正确指向 M1.3），
**未**产生 checkpoint。**M6 尚未开始。**

**下一步：M1.3c**（正式 `ScenarioBundle`、连续切分、forecast 可见窗口、泄漏门禁）。

## 7B. M1.2e：冻结数据链**已整合**（独立分支，**已通过人工审查**）

`docs/task_cards/M1.2e.md`。已获人工批准执行**方案 A**。

- **整合分支**：`p4-safeppo-m51a-rollout-contract-m12-integration`，起点 `04296db`。
- **20 个源提交按序移植完毕**（源→整合 SHA 完整映射见任务卡 §4）：
  M1.2a `a4a788c`…`b77594e` → `88adbde`…`e33d26c`；
  M1.2b `05e7034`…`7d04d3d` → `f46cc34`…`476aa4f`；
  M1.2c `16ee77b`…`d72b192` → `df608d9`…`a475ce3`。
  **无空提交、无 `--skip`、无授权外冲突。**
- **实际发生 3 次冲突**（`.gitignore` 自动合并，无需人工）：
  1. `0731c92` × `data/raw/README.md` —— 采用 p1 的冻结叙述，另立 §1.1–§1.3 保留
     p4 的只读语义、语义隔离与「未验证 ≠ 已证明不存在」；
  2. `0731c92` × `docs/task_cards/M1.2.md` —— p1 的验收记录与 p4 的认知保真修正
     （`53b54f9`）**同时保留**，并标注 raw freeze ≠ ScenarioBundle、M1.3 未完成；
  3. `0d4ed73` × `data/raw/README.md` —— **同一文件再次冲突**（纯追加），两块都保留。
  > **这证实了 M1.2d 的勘误 2**：`merge-tree` 只预测「最终树」的 2 个冲突文件，
  > **不能**用于推断逐提交的冲突次数。
- **逐字节一致**：`data/manifest/singapore_2024.json` 等 **8 个文件**与
  `p1-data-contracts` blob 完全相同（manifest `cmp` exit 0）。
- **分阶段验收**：M1.2a 端点 **7 passed**、M1.2b 端点 **16 passed**、
  M1.2c 端点 **22 passed**（均在临时 `git worktree` 中执行，未移动任何分支指针）。
- **raw 只读核验**：`--verify` exit 0，四个文件的 SHA-256/字节数/mtime
  **整合前后完全一致**；未联网、未改 manifest、未生成新时间戳。
- **不可回退验收**：`make check` exit 0（**1235 passed**），`make smoke` exit 0，
  contract-v7 / 21 维动作 / 旧 23 维拒绝 / canonical `ScenarioBundle` 全部保持。
- **原三个分支指针均未移动**：`p4-safeppo-m51a-rollout-contract` = `04296db`、
  `p1-data-contracts` = `d72b192`、`paper-baseline` = `787a3c8`。

**M1.2 raw freeze 已进入当前累计链，但这仍不等于 M1.3 或正式训练可用**：
`data/manifest/train.json` **仍不存在**，正式 `make train` 仍明确失败（exit 2，
**不**回退合成数据），未产生任何 checkpoint。**M6 尚未开始。**

**下一步：M1.3b**（半小时 canonical reader、单位与时间对齐）。
整合分支**尚未合回** `p4-safeppo-m51a-rollout-contract`，需人工批准。

## 7A. M1.2d：冻结数据链跨分支整合审计（只读，**未实际整合**）

`docs/task_cards/M1.2d.md`（`62d72b6`）。**本卡只产出可审核的整合方案，
没有执行任何 merge / cherry-pick，没有修改任何代码、测试、配置或数据**。

- **数据来源**：`p1-data-contracts` @ `d72b192`；**共同祖先（merge-base）** `3a75d0e`。
- **需要移植的完整提交序列**：`merge-base..p1-data-contracts` 共 **20 个**，
  按序为 `a4a788c`（M1.2 前置）→ M1.2a（`f79c94a`…`b77594e`，终点 ✅）→
  M1.2b（`05e7034`…`7d04d3d`，终点 ✅）→ M1.2c（`16ee77b`…`d72b192`，终点 ✅）。
  逐提交清单见任务卡 §6.1。
- **p1 只改动了 11 个文件**，且**完全未触碰** `contracts/`、`scenario/`、`Makefile`、
  `uv.lock`、`pyproject.toml`、`tests/test_m13*` —— 因此**本批移植不需要针对
  `contract-v7` 重写任何文件**。
- **冲突面极小**：双方都改过的只有 3 个文件。只读 `git merge-tree` 实测恰好
  **2 处文本冲突**（`data/raw/README.md`、`docs/task_cards/M1.2.md`，均为文档），
  `.gitignore` 自动合并。
- **raw 只读核验**：本机 `data/raw/singapore_2024/` 的 4 个文件**全部存在且
  SHA-256 与 p1 冻结 manifest 逐一一致**（含字节数），未改动任何时间戳或内容，
  **未 blocked**。
- **推荐方案（唯一）**：**A —— 从 `6c5ed50` 新建
  `p4-safeppo-m51a-rollout-contract-m12-integration`，按序逐提交移植那 20 个提交**；
  明确不推荐 B（整体 merge，会引入 merge commit、破坏逐卡审计）与
  C（重新实现，会迫使重新冻结 manifest，违反「不改 hash/时间戳」红线）。
- **冲突处理**：`data/raw/README.md` 以 p1 为主但**必须保留** p4 的
  「未验证 ≠ 已证明不存在」表述；`docs/task_cards/M1.2.md` 同时保留 p1 的验收记录
  与 p4 的软化勘误（`53b54f9` **不得被整合回退**）。
- **后续卡顺序**：M1.2e（实际整合）→ M1.3b（半小时 canonical reader）→
  M1.3c（正式 ScenarioBundle、连续切分与泄漏门禁）→ M6.1（冻结归一化参考值）→
  M6.2（统一评估适配器）。**不新增 M5.5。**

**M1.2 已完成**：USEP / 系统负荷 / national intermittent generation / ERA5 weather
四源冻结，含来源、许可、时区、行数与 SHA 账本；manifest 不可变（M1.2b）；
冻结后 raw 路径只读（M1.2c）。

**M1.2 仍不等于**：正式训练 ScenarioBundle、IDC 本地 PV/风电实测、碳强度、
半小时 canonical dataset、train/val/test 切分 —— 这些属 **M1.3**（缺口清单见任务卡 §10.3）。

**正式训练仍未开始**，`make train` 在本分支仍因 `data/manifest/` 整个目录缺失
（M1.2 未整合）与 M1.3 未完成而明确失败，**绝不回退合成数据**。**M6 尚未开始。**

## 8. 当前授权状态：M5.4i **已通过人工审核，M5.4 工程门禁已解除**

**M5.4i 第三次返修通过第四轮人工审核（2026-09-15）→ M5.4 工程门禁正式解除。**
详见 `docs/task_cards/M5.4i.md` §13。

- **最终有效证据**：`runs/m54i_release_v5/`（13 个 run，全部修订 `2cab5a1`）。
- **历史证据保留但 superseded**，一律不得作为发布证据引用：
  v1 `runs/m54i_release/`（`61ba510`）、v2 `runs/m54i_release_v2/`（`aa3294f`）、
  v3 `runs/m54i_release_v3/`（`771a33d`）、v4 `runs/m54i_release_v4/`（`4f759fb`）。
- **生产默认 corrector 预算为 0.25 s**（唯一来源
  `planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S`）。
- **效力边界**：这是**本机工程门禁**通过，**不是跨机器保证**。
  **没有**完成正式训练、性能评估或收敛结论。
  正式 `make train` 仍被 **M1.2 当前分支缺失**（无 `data/manifest/train.json`）与
  **M1.3 未完成**阻塞，绝不回退合成数据。

### 8.1 M5.4i 三轮审核未通过的历史（存档）

**三轮审核未通过的三个原因**（详见任务卡 §10.1/§11.1/§12.1）：聚合器 fail-open →
证据不足 fail-open 与 manifest provenance 缺失 → 外部产物类型强转（容器 provenance
泄漏 `TypeError`、非字符串 digest 被 `str()` 洗白、宽松 `int()` 截断）。


`M5.4h2` **已通过人工审查并被接受**（2026-09-14）。

`M5.4i` **首轮执行后人工审核未通过**（2026-09-14），已按审核意见在原卡返修
（`d6f875b` → `771a33d`；见 `docs/task_cards/M5.4i.md` §9 与 §10）。
**返修仍等待再次人工审核。**

**第三轮审核未通过的三个原因**（详见任务卡 §12.1）：

1. **容器型 provenance 在 set 构造时泄漏 `TypeError`**：三处同为 `list`/`dict` 时，
   逐字段比较不会报错（三处相等），但随后 `{...}` 集合构造因不可哈希而抛
   `unhashable type: 'list'/'dict'` 并泄漏给调用方。
2. **非字符串 digest 被 `str()` 后错误放行**：`digests.extend(str(d) ...)` 把错误类型
   洗成看似合法的证据；三批全为 `[1]*6` 时 `aggregate_distinct=1` → `passed=true`。
3. **`processes`/`steps`/`status` 的宽松 `int()` 会截断非整数**：
   `processes=6.9`、`steps=8.9`、`processes="6"`、`status=0.5` 全部 `passed=true`。
   另有 `manifest.run_id` 与输入目录名不一致时未被检查。

**第三轮返修要点**：新增严格校验器（`_strict_int` 拒绝 bool/字符串/浮点、
`_strict_finite_float` 用 `math.isfinite` 并显式拒绝 bool、`_strict_sha256` 固定
`^[0-9a-f]{64}$`、`_strict_load`/`_strict_machine` 校验类型与语义）；
`_check_batch` 写入摘要前把不合法值归一为 `None`，后续 `set`/`sorted`/`json.dumps`
不再接触未验证容器；新增原因码 `provenance_not_scalar`、`digest_malformed`、
`count_not_integer`、`manifest_run_id_mismatch`。合法产物语义不变。

**第二轮审核未通过的三个原因**（详见任务卡 §11.1）：

1. **config provenance 未参与「三处恒等」校验**：只比较了 manifest 与 report，
   config 里的矛盾值不会被发现（实测 `passed=true`）。
2. **`production_default_only=false` 仍可能 `passed=true`**：顶层
   `production_corrector_time_limit_s` / `effective_corrector_time_limit_s` /
   顶层 `corrector_time_limit_source` 与 `production_default_only` 从未进入判据；
   三批**全部缺** `load`/`machine` 时三个空签名互相相等，同样通过。
3. **合法 JSON/YAML/Parquet 中的错误类型会抛异常**：`report`/`manifest` 是 list →
   `AttributeError`；`processes="abc"` → `ValueError`；`summary.parquet` 的
   `status=NaN` → `ValueError` —— 异常从 `aggregate_release_batches` 泄漏。

**第二轮返修要点**：三方恒等改为 config/report/manifest 逐字段比较；
`passed` 显式依赖 `failures` 为空 ∧ batch 数达标 ∧ `production_default_only` ∧
每批与跨批 `distinct=1` ∧ `time_limit_failures=0` ∧ `every_batch_passed` ∧
`production_default_satisfied`；`load`/`machine` 必须是非空 dict 且键完整；
每个输入**只读一次**，签名比较基于已验证摘要；所有类型/数值异常转成 fail-closed
结果（新增 `malformed_batch_artifact`、`load_signature_invalid`、
`machine_signature_invalid`、`summary_parquet_malformed_status`、
`production_default_not_satisfied`）。

**首轮审核未通过的三个原因**（详见任务卡 §10.1）：

1. **跨批聚合器 fail-open**：只要求批次非空，于是「单批次」「同一 run 传三次」
   「三种负载混合」都能冒充三个独立同负载批次。
2. **聚合器读 manifest 却不校验它**：`manifest.status`、run_id 唯一性、revision 一致性、
   样本量、输入产物完整性、`summary.parquet` 存在性都未要求；缺 `summary.parquet`
   反而被当作「零次 time-limit」。
3. **provenance 未写入 manifest**，且聚合产物只有 `aggregate.json`，
   不符合标准 run 产物契约，聚合命令还会覆盖同名结果。

**返修要点**：聚合器改为 fail closed 并返回机器可读 `failures`（批次不足 / 路径或
run_id 重复 / 批间负载或机器不一致 / manifest 非 success / revision 不一致 / 门禁未过 /
样本不足 / digest 数与进程数不符 / 缺 `summary.parquet` / 非生产默认预算 等一律失败）；
`runs/writer.py` 增加**受控** `manifest_metadata`（禁止覆盖 9 个保留字段）；
四入口把 provenance 写入 manifest 且与 config/report **恒等**；
`--aggregate-runs` 经统一 writer 写标准 run 且不覆盖既有成功聚合。

**M5.4 整体仍为 blocked**：M5.4i 提供的是**发布候选证据**，是否发布必须由
**下一轮人工审查**决定。在人工发布确认之前，任何产物与文档都**不得**表述为
M5.4 released。`M5.4h2` 的执行通过**不构成**正式训练、性能评估或收敛结论。

### M5.4i（执行完成，等待人工审查）：采纳 0.25 s 为正确器生产默认预算

授权为“选择 B”，已实现：`planning/corrector.py` 的
`PRODUCTION_CORRECTOR_TIME_LIMIT_S = 0.25` 是**唯一**来源，四个入口
（train / smoke / probe_rollout_deterministic / probe_corrector_repro）全部导入它，
不再各自硬编码。`correct()` 仍要求调用方**显式**传入 `time_limit_s`。

- **provenance 三字段**：`production_corrector_time_limit_s`、
  `effective_corrector_time_limit_s`、`corrector_time_limit_source`
  ∈ {`production_default`, `explicit_override`, `disabled`}；source 由「**是否显式给出
  预算**」决定，与数值无关（显式给 0.25 也记 `explicit_override`）。
- **release gate 只认生产默认**：只有 `production_default` 观测能进
  `release_gate.observations`；显式 override 观测进入 `override_observations`，
  如实记录但绝不参与放行判定。M5.4h2 的 fail-closed 规则保留。
- **Stage A/B 共享 0.25 s deadline**：由 runtime spy 证明（`planning/model.py` 未改）。

**发布批次证据（第三次返修后，`runs/m54i_release_v5/`，全部修订 `2cab5a1`）**：
no-load / hogs4 / hogs8 各 3 个独立批次 + 1 个显式 0.05 诊断 run +
每种负载 1 个标准聚合 run（共 13 个 run）。三组聚合均 `passed=true`、`failures=[]`、
`production_default_satisfied=true`、每负载 36 个进程观测、`aggregate distinct=1`、
`time_limit failures=0`。
`runs/m54i_release_v4/` 标记 `superseded_pre_strict_external_type_validation`（保留不删）。

**上一代（已作废的）发布批次证据**（`runs/m54i_release_v4/`，修订 `4f759fb`）：
no-load / hogs4 / hogs8 各 3 个独立批次 + 1 个显式 0.05 诊断 run +
每种负载 1 个标准聚合 run（共 13 个 run）。三组聚合均 `passed=true`、
`failures=[]`、`production_default_satisfied=true`、每负载 36 个进程观测、
`aggregate distinct=1`、`time_limit failures=0`。
`runs/m54i_release_v3/` 标记 `superseded_pre_second_aggregation_gate_fix`（保留不删）。

**上一代（已作废的）发布批次证据**（`runs/m54i_release_v3/`，修订 `771a33d`）：
no-load / hogs4 / hogs8 各 3 个独立批次 + 1 个显式 0.05 诊断 run +
每种负载 1 个**标准聚合 run**（共 13 个 run）。三组聚合均 `passed=true`、
`failures=[]`、每负载 36 个进程观测、`aggregate distinct=1`、`time_limit failures=0`。
**v2 已标记 `superseded_pre_aggregation_gate_fix`**（保留不删，不属于最终验收证据）。

**首轮（已作废的）发布批次证据**（`runs/m54i_release_v2/`，修订 `aa3294f`）：
no-load / hogs4 / hogs8 三种负载各 3 个独立批次，共 **9 个生产默认 run**
（另 1 个显式 0.05 诊断 run）。每批 `--modes on --runs 6 --steps 8`、**不传任何显式
预算**。三种负载的跨批聚合**全部通过**：每负载 36 个进程观测、`aggregate distinct=1`、
每单批 `distinct=1`、`time_limit failure=0`、`production_default` 且预算 0.25。

`runs/m54i_release/`（v1，修订 `61ba510`）因随后一处 **typing-only** 改动而重生成，
标记为 **`superseded_pre_typing_fix`**，不属于最终验收证据，保留不删。

**这仍不等于 M5.4 已发布**：0.25 s 是**本机**候选值，跨机器无保证；
本轮人工审核**未通过**，发布决定权在人工。**下一步仍是停止并等待审核。**

- **允许文件**：`planning/corrector.py`（若仅为唯一常量来源所需）、
  `safe_rl_v2/train.py`、`scripts/smoke_main_chain.py`、
  `scripts/probe_rollout_deterministic.py`、`scripts/probe_corrector_repro.py`、对应测试、
  `docs/task_cards/M5.4i.md`。
- **禁止**：改 `planning/model.py` 的目标/约束/options 构造，启用生产 `mip_max_nodes`
  替代语义，改环境、动作、物理链、契约或 M1.2 数据。
- **必须**：建立唯一生产默认来源；train/smoke/rollout 默认路径不再各自硬编码 0.05；
  Stage A/B runtime spy 证明共享 0.25 s deadline；显式 0.05 的运行仍如实记录为 override。
- **M5.4 release 验收**：synthetic-smoke corrector-on 在 no-load、hogs4、hogs8 三种口径下，
  每种至少 6 个独立进程、每进程 8 步；每组 `distinct=1` 且 `time_limit=0`。所有 run
  产物完整并记录 0.25 s。
- **训练边界**：`make train` 正式路径仍必须因 M1.3 未完成而明确失败；只有显式
  synthetic-smoke 可运行，且不得宣称训练、性能或收敛。

## 9. 统一任务卡、提交和报告标准

每张卡至少按以下顺序创建独立提交：

1. `docs(card): start Mx.y ...`：边界、禁止项、失败命令、证据、回滚点。
2. `test: ...`：实现前失败测试，失败原因必须与卡目标直接相关。
3. `fix/feat/refactor: ...`：最小实现批次；显式 `git add <file...>`。
4. `test: ...`：必要的回归覆盖。
5. `docs(card): record Mx.y evidence`：最终命令、runs 路径、风险、完整 revert。

卡完成报告必须包含：分支、开始 SHA、全部 SHA、文件清单、失败转绿证据、验收命令、
机器可读 runs 路径、范围外修改、风险/blocked、完整 `git revert` 序列和空工作树证据。

## 10. 禁止的结论与术语

- 不得说“正式训练已完成”“PPO 已收敛”“性能已评估”；目前从未完成真实数据正式训练。
- 不得说 M1.2 已使正式训练可用；它只完成 raw freeze。
- 不得说 M5.4 已解除 blocked；只有在**人工**发布确认后才可以这么说。
- 不得把 M5.4i 的 3 负载 × 3 批稳定证据表述为「已发布」或「跨机器保证」；
  它是**本机**候选证据，发布决定权在人工。
- 不得引用 `runs/m54i_release_v2/`（`superseded_pre_aggregation_gate_fix`）、
  `runs/m54i_release_v3/`（`superseded_pre_second_aggregation_gate_fix`）或
  `runs/m54i_release_v4/`（`superseded_pre_strict_external_type_validation`）
  作为发布证据；最终证据是 `runs/m54i_release_v5/`。
- 不得把 M5.4h/M5.4h1/M5.4h2 的 0.05 s blocked 结论套用到当前的 0.25 s 默认上。
- 不得把 `mip_max_nodes` 的未绑定实验称为确定性的根因或生产方案。
- 不得以任务卡提交数、测试数或绿灯替代真实集成验收。
- 不得把 M1.3e 的五个 driver forecast 说成「完整场景预测」或
  「forecast 已就绪」：`forecast_ready=false`、
  `complete_scenario_forecasts_ready=false`、`formal_scenario_bundle_ready=false`、
  `formal_training_ready=false`；四项缺口仍 unavailable。
- 不得把 `planning/snapshot_adapter` 的 oracle-debug bundle（其「可见预测」就是 env
  真值窗口、`load_forecast` 为全零占位）表述为正式 forecast 或用于训练/评估。
- 不得把 M1.3a 的「可见 truth 改变 bundle」断言表述为正式 forecast 的泄漏门禁；
  它是 **oracle-debug 语义**，正式 causal provider 的 leakage 回归是**另一组独立**测试。
- 不得在 **M1.3e-R1 通过人工复审之前**把 M1.3e 说成已通过；不得说 M1.3f 已开始。
- 不得因为「`data/manifest/singapore_2024_forecast_policy.json` 存在」就声明
  `forecast_ready`：该 manifest 的 readiness 明确只有
  `available_driver_forecasts_ready=true`，其余三项 false。
- 不得把 `prediction_hash()` 与 `content_hash()` 混为一谈：前者只覆盖预测数值、
  单位与顺序（用于证明未来真值变化不改变预测），后者覆盖含审计 provenance 的
  完整 artifact。

## 11. 当前可安全执行的命令

```bash
# 当前分支基础健康检查
make check
make smoke
git diff --check
git status --short

# M1.2 数据只读核验（必须在 p1 或已批准的整合分支上使用）
uv run python scripts/fetch_singapore_data.py \
  --raw-dir data/raw/singapore_2024 \
  --manifest data/manifest/singapore_2024.json --verify
```

`make train` 的正式路径预期失败，直到 M1.3 完成；这是正确门禁，而不是要修掉的错误。
