# VSCode Claude Code 唯一执行者协议

> 主方案：`docs/IMPLEMENTATION_PLAN.md`。本文件规定 Claude Code 的 Git 纪律、执行顺序和必须停下等待用户的节点。
>
> 当前授权仅限写本协议；没有安装、下载数据、训练或任何研究代码改造授权。

## 1. 启动提示词

将下面全文发给 VSCode 中的 Claude Code：

```text
你是本仓库唯一代码执行者。先完整阅读 docs/IMPLEMENTATION_PLAN.md 和
docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md。严格按照 M0 到 M9 的依赖顺序，
一次只执行一张任务卡，从 M0.1 开始；不并行、不跳卡、不把 mock 或原型称为集成完成。

所有修改都必须有 Git 记录：开卡前提交任务卡和改前证据；每个可验证编辑批次在测试后
提交；结束时提交验收记录。一个提交只能属于一张卡。禁止在 main 上修改、git add -A、
git add .、commit --amend、rebase、reset --hard、checkout --、git clean、强制推送。

不要修改 marl/、grid_model/ 主逻辑、legacy/ 或顶层 shim。不得放松接入容量、SOC、
充放电互斥或任务约束；不得清空任务、重置 SOC、读取未来真值；不得把 a_exec 写入
a_raw 的 log-prob。未获逐项授权，不安装依赖、不下载数据、不训练。

每卡结束报告：任务卡、分支、开始 SHA、新 SHA、修改文件、验收命令和结果、证据路径、
风险、回滚命令、git status --short 输出。M3/M5/M6 的语义卡测试通过后停止，等待用户
审阅 diff 后才进入下一张卡或合并。
```

## 2. 强制 Git 工作流

### 2.1 开卡检查

每张卡开始前必须运行并在对话中贴出摘要：

```bash
git status --short
git branch --show-current
git log -1 --oneline
git diff --check
```

发现工作树不为空、未知 untracked 文件、当前在 `main`、或 staged 文件不属于当前卡：立即停止。不得删除、stash、重置或“顺手修复”这些文件。

分支格式固定为：

```text
p0-bootstrap
p1-data-contracts
p2-physics/m3-<card>
p3-corrector/m4-<card>
p4-safeppo/m5-<card>
p5-eval-viz
p6-experiments
```

分支已存在时不得重建或覆盖；报告分支 HEAD，等用户指示。

### 2.2 每张卡的 Git 记录

每张卡至少有以下三类提交，所有 `git add` 必须显式列出文件：

1. `docs(card): start Mx.y <title>`：新增 `docs/task_cards/Mx.y.md`，写明目标、允许文件、禁止项、开始 SHA、改前命令与结果、预期失败、验收命令、证据产物、回滚命令。
2. `test(...)` / `feat(...)` / `refactor(...)`：一次可验证编辑批次一个提交；提交前运行最小相关测试。不可把多张卡压成一个提交。
3. `docs(card): record Mx.y evidence`：在任务卡中追加完整验收输出摘要、新 commit SHA、run id、剩余风险与回滚命令。

完成每次提交后必须运行：

```bash
git status --short
git diff --check HEAD~1..HEAD
git show --stat --oneline HEAD
```

一张卡结束时工作树必须为空。要撤销已完成卡，只能 `git revert <SHA>`，保留 revert 记录。Claude Code 不得自行合并到 `main`。

### 2.3 不得以 Git 规避审阅

- 禁止删除或弱化测试、修改 expected 值、缩小测试范围来转绿。
- 禁止在同一提交混进格式化、依赖升级、无关修复或生成产物。
- 禁止提交原始数据、大模型 checkpoint、缓存和大图；运行结果按 manifest 归档。
- 语义卡的 Git 提交用于审阅和回滚，不代表获准进入下一张卡。

## 3. 每卡固定报告格式

```text
任务卡：M?.?
分支：...
开始基线：<SHA>
新增提交：<SHA 列表>
变更文件：<完整列表>
验收：<命令> -> <退出码和关键结果>
证据：<测试、runs/<id>/、报告路径>
允许范围外修改：无 / <说明>
风险或阻塞：...
回滚：git revert <SHA ...>
工作树：git status --short -> 空
下一步：等待用户是否放行 M?.?
```

## 4. 全局红线和工程产物

1. 新主链不能接受旧 23 维动作，也不能加载无 `contract_version` 或 schema 不同的 checkpoint；必须明确报错，不能截断或填零。
2. 新 checkpoint loader 只能在新目录或新 Safe PPO v2 入口实现；绝不能修改 `marl/checkpointing/`。
3. 修改 `envs/idc_price_env.py::step()` 前，相关失败测试必须已在开卡提交中存在。
4. 场景/预测/配置/评估均需 version、hash 和来源；正式模式缺数据直接失败。合成数据只能显式开关并写入 manifest。
5. 每个 run 必须写 `runs/<id>/config.yaml`、`metrics.parquet`、`report.json`、`figures/`、`manifest.json`。manifest 包含 revision、锁文件 hash、数据/场景 hash、种子、命令和失败状态。
6. CPU 是正式复核平台；MPS 仅探索，必须记录 backend 与非确定性风险。

## 5. 串行任务队列

除非用户显式放行，Claude Code 一次会话只完成表中一张卡。任何标为“审阅停点”的卡完成后必须停止。

### M0：工程门禁

| 卡 | 允许文件 | 完成标准 | 停点 |
|---|---|---|---|
| M0.1 | `docs/BASELINE_AUDIT.md`, `docs/task_cards/` | 记录当前 commit、M5/16GB 机器、Python、入口、保护目录、数据状态；可定位现有 23 维动作、标量容量、日 reset。 | 否 |
| M0.2 | `pyproject.toml`, `.python-version`, `uv.lock` | Python 3.12 arm64 锁定环境；旧 `environment.yml` 保持原样；仅在用户授权安装后 `uv sync --locked`、导入核心包、HiGHS 二元互斥微例通过。 | 是：安装授权 |
| M0.3 | `Makefile`, ruff/mypy/pytest 配置 | `make check` 真运行 ruff+mypy+非 slow pytest；test/contract/probe/smoke/train/eval/figures/report 目标存在，未实现项显式失败。 | 否 |
| M0.4 | `AGENTS.md`, `tests/conftest.py` | 规则覆盖红线、分支、产物、任务卡和 `leakage`/`resume`/`slow` markers；`pytest tests -q` 可发现骨架。 | 否 |

M0 全绿前，不得开始 M1 的实现或 M3 的代码改动。

### M1/M2：数据、预测、契约

| 卡 | 允许文件 | 完成标准 | 停点 |
|---|---|---|---|
| M1.1 | `docs/AUDIT_EXECUTION_CHAIN.md` | 只读审计并以行号定位动作→任务→功耗、风电、SOC、接入、预测、数据、refs 与 checkpoint。 | 否 |
| M1.2 | `data/raw/README.md`, `data/manifest/`, `scripts/fetch_*` | 仅用户授权数据访问后：连续一年价格/负荷对齐、气象/可再生/任务来源、时区/单位/许可/sha256/缺口全记录。 | 是：数据口径 |
| M1.3 | `scenario/`, `tests/test_scenario*.py` | `build_scenario` 只返回当前真值和可见预测；改变未来真值不改变当前输入；缺单位/来源/hash 报错。 | 审阅停点 |
| M2.1 | `contracts/`, `tests/test_contracts*.py` | 六个 frozen Pydantic 契约；任务×组矩阵矩形、非负、带单位/来源；JSON 往返和确定性 hash 通过。 | 审阅停点 |
| M2.2 | `contracts/`, `tests/test_contract_validators.py` | 容量、任务最大速率、SOC、时间轴与数组长度跨契约校验；每种错误明确失败。 | 审阅停点 |
| M2.3 | 新 `checkpointing/` 或 `safe_rl_v2/`, tests | 新主链保存/加载含版本和 schema；无版本、23 维、schema 不同均失败；同版本状态往返。 | 审阅停点 |

### M3：物理链（每张均为审阅停点）

| 卡 | 允许范围 | 必须转绿的验收 |
|---|---|---|
| M3.0 | 新探针与 baseline tests | 固定 seed 记录 23 维、互斥、SOC、能量平衡、deadline 漏记和组分配基线；未来要求使用 strict xfail。 |
| M3.1 | `envs/idc_price_env.py` 调用处、专属测试 | 分配器接收长度 20 的 `planned_capacity_vec`；不再只有标量容量输入。 |
| M3.2 | 新 `idc_model/allocation.py`、契约接线、测试 | `A[i,g]>=0`；逐组不超容量；逐任务不超 max rate/remaining；总完成量等于矩阵和。 |
| M3.3 | `idc_model/power_model.py`、实际 load 接线、测试 | 实际功耗由 A 导出；相同工作量不同组分配可产生不同功耗；能量守恒；本卡不改 reward/config。 |
| M3.4 | 新诊断、测试 | 同场景/动作/seed 对照 α=0 与历史 α；α 不进入正式主链。 |
| M3.5 | 风电输入/观测/能量流、测试 | 风电可用/利用/弃电进入无反送电平衡；PV/风/储能各工况守恒。 |
| M3.6 | reset/跨日状态、窗口、测试 | 48/72h 日界不清队列/不重置 SOC；统一尾段结算；恢复与连续轨迹等价。 |
| M3.7 | 容量配置、执行前检查、测试 | 物理接入上限与计费阈值独立；超限保留缺口/修正，不放宽上限。 |
| M3.8 | task status/metrics、测试 | 未到期积压、逾期积压、逾期完成、按时完成互斥完备；逾期完成只计一次。 |
| M3.9 | 动作解析/config/tests | 严格 21 维：20 compute + 1 signed storage；23 维立即失败；无 dummy 动作。 |
| M3.10 | observation/forecast、leakage tests | 观测含任务、容量、SOC、接入、资源、预算和可见预测；未来 mutation 不影响当前观察。 |

M3 禁止项：不能以 α 模拟未来预留；不能新增 DVFS、开关机、通信代价或亲和性；不能改 reward 掩盖物理变更；不能通过清队列、日界重置 SOC 或增大接入上限通过。

### M4：联合可行性规划器

| 卡 | 允许范围 | 必须转绿的验收 | 停点 |
|---|---|---|---|
| M4.1 | `planning/snapshot_adapter.py`, tests | snapshot 只含当前状态/可见预测，不访问 policy、value 或未来真值。 | 否 |
| M4.2 | `safe_rl/corrector.py`, tests | 单步修正仅收 snapshot/proposal；检查计算范围、SOC、互斥、无反送电、接入。 | 否 |
| M4.3 | `planning/window.py`, `model.py`, `solver.py`, tests | 默认 MIP 二元互斥；24h 按 deadline 延伸；任务×组、SOC、风光、接入、效率、退化齐全。 | 审阅停点 |
| M4.4 | `planning/corrector.py`, tests | 最小修改；timeout/数学不可行/物理复核失败/预测超界四分类；失败使用已验证边界动作+gap，绝不返回未经检验 raw action。 | 审阅停点 |
| M4.5 | `safe_rl/corrector_wrapper.py`, integration tests | 训练/评估同修正器；reward/next state 对 exec；buffer log-prob 只对 raw。 | 审阅停点 |
| M4.6 | `planning/probe.py`, `runs/` writer | 真实 20组、24h/超窗记录环境步、median/P95、失败率、规模、内存、随机 rollout 吞吐。 | 审阅停点 |

### M5：安全 PPO v2（全部审阅停点）

| 卡 | 允许范围 | 必须转绿的验收 |
|---|---|---|
| M5.1 | 新 `safe_rl_v2/`、tests | buffer 同存 obs/raw action/raw log-prob/reward/业务 cost/碳 cost/exec action/修正信息。 |
| M5.2 | models/buffer/tests | 收益、业务、碳各有 value/GAE；改变一种 cost 不污染另两种 target。 |
| M5.3 | lagrangian/checkpoint/tests | 独立乘子/预算/日志；网络、优化器、RNG、乘子、schema 均可恢复。 |
| M5.4 | train/eval entry/tests | CPU 下中断恢复逐步一致；未来信息隔离回归通过；短 dry rollout 可更新一次。 |

### M6/M7/M8：评估、产物和离线电网

| 卡 | 允许范围 | 必须转绿的验收 | 停点 |
|---|---|---|---|
| M6.1 | `freeze_refs.py`, frozen JSON | refs 只从训练切分/物理尺度产生，带 hash；测试日无法重算。 | 审阅停点 |
| M6.2 | 新 evaluation adapter | 五方法同 schema；不达服务标准明确标记，不能进入等服务成本比较。 | 审阅停点 |
| M7.1 | `runs/` writer | 每 run 有 config/parquet/report/figures/manifest；失败 run 也有失败状态。 | 否 |
| M7.2 | `viz/`, `build_report.py` | 图/报告只从 Parquet+manifest 生成，并回链 run id。 | 否 |
| M8.1 | 只读核查脚本/报告 | 明确 IEEE-14 字段、容量与“不代表新加坡真实网络”。 | 否 |
| M8.2 | 离线轨迹验证脚本 | 正常/临界/失败轨迹均输出 AC/OPF 成功率和失败原因。 | 否 |

### M9：实验与统计（需要单独训练授权）

| 卡 | 必须转绿的验收 |
|---|---|
| M9.1 | 新主链 rollout/s、内存、磁盘、每 seed 工时实测；超预算停止报告。 |
| M9.2 | 冻结 manifest：60/20/20 连续切分、至少 30 测试日、至少 3 训练种子、服务/容量/碳/refs 固定。 |
| M9.3 | 五方法共享业务、物理、信息和资产；机制实验区分重训与运行干预；失败 run 不删除。 |
| M9.4 | 日期块和 seed 分层统计；自然分布为主；固定策略泛化与重训性能分开；结论由 Parquet 自动生成。 |

## 6. 用户授权检查点

Claude Code 必须停下等待明确授权：

1. M0.2 前的安装或锁文件更新；
2. M1.2 前的外部数据下载、登录或 API；
3. 每一张 M3、M5、M6 卡测试通过后；
4. M4.3 完成数学模型后；
5. M9.1 前的训练、批量运行或长期占用本机；
6. 合并 `main`、发布外部内容、删除任何文件前。

在未获下一张卡明确放行时，Claude Code 的正确行为是报告当前状态并停止。
