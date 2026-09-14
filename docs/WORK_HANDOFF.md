# 工程工作交接文档

> 更新时间：2026-09-14（Asia/Shanghai）
> 当前代码工作分支：`p4-safeppo-m51a-rollout-contract` @ `ca17d87`
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
| `p4-safeppo-m51a-rollout-contract` | `ca17d87` | **当前继续工作分支**；包含 M5.1–M5.4h1 的累计链。 |
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
| M5.4 | **blocked，正在收尾** | 训练入口、产物账本、corrector 复现与预算归因探针。 | 默认 0.05 s 在受控负载下跨进程不稳定；必须执行本文件第 8 节两卡。 |
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

### 6.2 M5.4h / M5.4h1 的证据

- M5.4h：`96f3aa1` → `c842c34`，建立预算矩阵、受控 CPU hog、逐阶段探针。
- M5.4h1：`07e2224` → `ca17d87`，修复了初版账本中的负载记录、失败状态和节点替代
  实验语义。
- 当前默认预算仍是 **0.05 s**；在受控 hogs4/hogs8 下，独立进程 digest 可分叉，
  release gate 因此为 blocked。
- 在本机样本中，0.25 s 及以上没有 observed `time_limit` 且 digest 一致；这是
  **本机候选值**，不是跨机器保证。
- `mip_max_nodes=100/10000` 的替代语义臂稳定，是因为移除了 `time_limit`；所有可行解
  `mip_node_count=1`，节点上限未实际绑定。它不能被称为“节点上限导致确定性”。

历史 runs 保留且不得覆盖：

```text
runs/m54h_matrix/
runs/m54h_nodecaps/
runs/m54h1_matrix/
```

## 7. 当前 M5.4h1 的未决账本问题

当前 HEAD 是 `ca17d87`，M5.4h1 已修复大部分账本问题，但尚有两项必须先用
M5.4h2 修正：

1. `summary.parquet` 使用 `options_json`，而约定字段名是 `options`。新卡须将其改为
   `options`（规范 JSON 字符串），并明确 encoding。
2. 某些 report 同时有 `release_gate.blocked=true` 与 `overall.blocked=false`。
   `overall` 必须反映 M5.4 phase/release 状态；预算归因仅位于 `attribution`。
   否则下游会误将“归因成功”理解为“阶段已放行”。

此外，`candidate_notes` 必须明确 `node_cap_effective=false` 与
`node_cap_candidate=false`，将稳定性归因于 probe 中停用了 wall clock，不得将 cap
100/10000 表述为生产候选。

## 8. 已授权的下一批：严格两张连续卡

只在卡 1 验收通过后才能开始卡 2。两卡完成后停止，提交阶段验收，等待人工审阅。

### M5.4h2：诊断账本语义最终勘误

- **基线**：`ca17d87`；分支保持
  `p4-safeppo-m51a-rollout-contract`。
- **允许文件**：`scripts/probe_corrector_repro.py`、`tests/test_m54h*.py`、
  `docs/task_cards/M5.4h2.md`、新 `runs/m54h2_*` 产物。
- **禁止**：改 `planning/`、`envs/`、`safe_rl_v2/`、`contracts/`、默认预算或任何
  求解/物理语义。
- **必须**：以失败测试证明第 7 节三项旧语义错误存在；`options` 作为规范 JSON 字段
  持久化；`overall` 与 release gate 一致；node-cap 不绑定时不能成为 candidate。
  不得覆盖历史 runs。
- **验收**：专项测试、`make check`、`make smoke`、`git diff --check`；新 run 的
  manifest/release gate/summary 三者必须一致。

### M5.4i：采纳 0.25 s 为正确器生产默认预算

只有 M5.4h2 全绿才开始。本卡已获得“选择 B”的授权：生产默认值从 0.05 s 改为
0.25 s；0.05 s 仍为显式 override/诊断值，绝不静默改写。

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
- 不得说 M5.4 已解除 blocked；在 M5.4i 前默认 0.05 s 仍 blocked。
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
