# 工程工作交接文档

> 更新时间：2026-09-14（Asia/Shanghai）
> M5.4h1 实现基线：`p4-safeppo-m51a-rollout-contract` @ `ca17d87`。
> 当前 HEAD：`p4-safeppo-m51a-rollout-contract`，M5.4i **第三次返修**实现终点 `2cab5a1`
> （其后仅有本卡的证据与交接 docs 提交）；`M5.4h2 已通过人工审查并被接受`，
> **M5.4i 已连续三轮人工审核未通过**，已三轮返修，**仍等待再次人工审核**。
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
| M1.3 | 部分 | 统一 `ScenarioBundle` 契约/泛化提供器已有早期实现。 | 真实 frozen manifest 读取与完整语义接线未验收；正式训练仍不可开始。 |
| M2 | 大部分完成 | 版本化契约、21 维拒绝、checkpoint/schema 门禁。当前版本为 `contract-v7`。 | 与真实 M1.2 场景的完整接线仍待 M1.3。 |
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

## 8. 当前授权状态：M5.4i 三轮审核均未通过，第三次返修后等待再次审核

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
