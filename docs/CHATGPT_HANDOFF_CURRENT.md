# IDC 项目交接：新 ChatGPT 对话当前入口

更新时间：2026-09-23（Asia/Shanghai）\
代码快照：`8cd2772e9b2c8d1c8d8bdc3f0215a201f79a59f8`\
分支：`p4-safeppo-m51a-rollout-contract-m12-integration`\
仓库：`/Users/levous/Desktop/IDC`

> 本文是新 ChatGPT 对话的**当前入口**。它提供上下文，不自动授权执行修改；
> 用户在新对话中的明确请求优先。逐卡原始证据见 `docs/task_cards/M1.3g.md`，
> 完整历史见 `docs/WORK_HANDOFF.md` 和 Git。旧的
> `docs/NEW_CONVERSATION_HANDOFF.md` 已成为累计历史，不应再用其顶部快照判断现状。

> **2026-09-24 计划补充**：本文顶部 SHA 是 2026-09-23 的历史入口快照，
> 当前 HEAD 以 `git rev-parse HEAD` 为准。M6-P0 评估协议可现在准备，
> 见 `docs/M6_EVALUATION_PROTOCOL.md`；checkpoint 契约稳定后再补评估器并用
> 受控短跑 checkpoint 验证。正式 validation/test 仍须等待训练产物审核通过。


> **2026-09-24（M6-P1 + CHAIN-REFRESH）**：B6 正式资产链做了一次**前向刷新**
> （改 `contracts/**` 会推进 `b6_formal_code_revision()`，从而失效整条链）。
> 上表 SHA 已更新为刷新后的值；旧值见 `docs/task_cards/CHAIN_REFRESH.md` §8.2 的账本。
> 已取代的历史证据（policy v1、formal_splits v1/v2/v3、refs v1/v2/v3、exogenous v2）
> **字节保留、未刷新**，仍保持其原始 SHA。

## 1. 先读什么

新对话先按以下顺序读取：

1. `AGENTS.md`：不可违反的工程红线、Git 纪律和任务卡格式。
2. 本文：当前状态、阻塞、计划和协作方式。
3. `docs/task_cards/M1.3g.md`：当前 M1.3g 各子卡的完整边界与证据。
4. `docs/WORK_HANDOFF.md`：累计卡片历史、回滚和资产迁移记录。
5. 只有需要追溯总体设计时，再读 `docs/IMPLEMENTATION_PLAN.md` 和
   `docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md`。

不要从旧交接文件中的早期 “下一步” 或旧 readiness 推断当前状态；以本文、当前
Git 和 verified loader 的结果为准。

## 2. 项目目标与当前结论

项目正在构建 IDC 安全 PPO 的正式数据—场景—环境—训练链。目标不是做一个能跑的
demo，而是让每一层都具备可复现、可验签、无未来泄漏、单位一致、可回滚的证据。

当前已完成：

- Singapore 2024 半小时 canonical 数据与连续 train/validation/test split；
- B6 arrival intensity、exogenous v3、causal forecast policy v3；
- `refs_v4`、formal split triad v5、arrival-to-Task mapper；
- formal 环境注入、0.5 小时单位修正、守恒/泄漏/重放回归；
- 独立 formal env 发布产物 v1，环境已正式发布；
- formal rollout、PPO clipped objective、单次更新、连续两批更新；
- 两批边界 checkpoint/resume 的完整状态等价，包括 policy、Adam、Lagrangian、RNG。

当前未完成：

- `safe_rl_v2/train.py` 还没有正式训练循环；
- `formal_training_ready` 仍为 `false`；
- 没有正式 checkpoint、成功训练 run、性能结论或收敛结论；
- M6 正式评估尚未开始；当前只允许 M6-P0 协议准备，不能提前运行 validation/test。

因此现在的核心结论是：**formal env 已发布，但 formal training 尚未发布。**

## 3. 当前 Git 与验收状态

在写本文前核验：

```text
HEAD   = f3a1fd2cf8a2d901770302a90129c3988c9766d4
branch = p4-safeppo-m51a-rollout-contract-m12-integration
工作树 = clean（创建本文之前）
```

最新通过人工审核的卡是 `M1.3g-f-c-e-R2`：修正测试快照对 Adam 活张量的别名，
使“批 2 前”与“批 2 后”的 optimizer 状态对照真正有效。

```text
边界（批 2 前）：Adam step={1.0}，Lagrangian updates=1
最终（批 2 后）：Adam step={2.0}，Lagrangian updates=2
连续路径与恢复路径：policy / optimizer / Lagrangian / RNG 全部逐项精确相同
```

最近一轮已报告并复核的门禁：

```text
resume focused     16 passed
two-batch focused  13 passed
make check         exit 0，2881 passed，47 deselected
make smoke         exit 0
env release verify exit 0
```

当前仓库有 148 个 `runs/train_real_*`；全部 report 含 `failure`，没有 success，
`claims.trained` 全为 false，仓库内没有 `.pt` checkpoint。这些失败 run 是正式入口
fail-closed 的证据，不得包装成训练结果。

## 4. 当前唯一正式资产

以下为当前链的 canonical 资产；旧版本逐字节保留，但禁止 fallback：

| 资产 | SHA-256 | 状态 |
| --- | --- | --- |
| `data/manifest/m13f_arrival_intensity_policy_v1.json` | `7066a0e127bc28f6a56ad4e62810c34536e1eb13b3c3c134baa3a1e6c4bca251` | B6 intensity policy |
| `data/processed/singapore_2024/exogenous_drivers_v3.parquet` | `07b648f0a15db1d8c39838e3e501dafa2f9956155489702e3379cdb775858612` | B6 realized exogenous |
| `data/manifest/singapore_2024_forecast_policy_v3.json` | `23863ea44b3a882449f8930370b1e02182f5467e4c4d52acbfadaf675acfe754` | causal forecast policy |
| `configs/frozen_refs/refs_v4.json` | `0889dcae5d896963301b3f645fdb3ceab0584105b01b990f009f957625a9b7c8` | 唯一正式 refs |
| `data/manifest/formal_splits_v5/train.json` | `077e0ea21725f9e42be40f774217b385fd2eed4cf99ff6658ba85725a1571865` | 唯一正式 train split |
| `data/manifest/formal_splits_v5/validation.json` | `ac1c50db28a8d6376513c6a2e6023cd7e45eaa6f4b8ad34f2b474a55a55d9933` | 唯一正式 validation split |
| `data/manifest/formal_splits_v5/test.json` | `76c6bb79446fbbc5dd1566c4e617d6b0d746f2ae87caa2eb26a6a18b853a827f` | 唯一正式 test split |
| `data/manifest/m13g_arrival_mapper_v1.json` | `ef401999ce9f43d0f03d31e4a96de5867581eac3d8d007a29054a7f97f4fb371` | approved mapper parameters |
| `configs/release/idc_formal_env_release_v1.json` | `be16e08e78ee9b445ceb5ca386f1de0ab4ee700ec25d690e7412ebca07cb92ce` | env 发布产物 |

任何 SHA 都应以 `shasum -a 256` 的实时结果为准，不要根据历史报告前缀手抄或拼接。

发布产物实时验证命令：

```bash
uv run python -m scripts.materialize_env_release --verify
```

当前结果：

```text
formal_env_ready      = true
formal_training_ready = false
sha256                = be16e08e78ee9b445ceb5ca386f1de0ab4ee700ec25d690e7412ebca07cb92ce
```

`formal_splits_v5` 三份文件内的 readiness 仍严格保持 both-false，这是冻结资产的
既定语义；不得改写或运行时覆盖。`formal_env_ready=true` 只由上面的独立验签发布产物
表达，training 继续由 `formal_training_ready=false` 阻断。

## 5. 必须保持的语义

### 5.1 时间、负载与单位

- 正式时间步为 30 分钟，`delta_t_hours=0.5`。
- `C_server`、`C_IDC` 和 planned capacity 是 `work/hour`；每步执行量必须乘 0.5。
- `queue_ref`、`queue_capacity_ref` 是存量，不随步长缩放。
- `lambda_ref=63.988 work/hour`，即 `31.994 work/step`；不得恢复旧值 2000。
- arrival mapper 必须对原始 aggregate 做 1:1 整数账本守恒，不得在 mapper 内缩放
  arrival、移动槽、丢弃低负载槽或制造零工作量任务。
- Route A 的 13 项 micro-task 参数已人工批准；当前正式 mapper 使用批准的
  `E_micro_inference` 可行域。

### 5.2 因果性

- observation 中的 formal forecast 必须来自 causal forecast，不得读取未来 realized truth。
- carbon 的物理成本使用 realized `carbon_factor_t`；formal forecast observation 使用
  causal `carbon_forecast_t`。当前二者均为 0.402 也不能混接。
- 任务数、工作量、completion rate 和 info 只能统计被测时点已到达任务；未来任务不得
  影响 observation、reward 或 info。
- validation/test 不参与参数选择、归一化或训练配置选择。

### 5.3 PPO 与 checkpoint

- buffer 永远保存 `raw_action` 与对应旧 log-prob；`exec_action` 只用于执行和审计。
- 新 log-prob 只可通过 `evaluate_raw_actions(observation, raw_action)` 现算。
- clipped objective 对负优势具有 PPO 的非对称行为，不能误称所有 `ratio=inf` 都会被
  clip 成有限值。
- policy、optimizer、Lagrangian、generator 由调用方拥有；probe 不得自行选择学习率、
  budget、隐藏层或种子。
- checkpoint 目前只证明**批次边界**恢复，不证明单批中途恢复，也尚未进入正式业务路径。

## 6. 当前代码能力

| 层 | 当前能力 | 关键入口 |
| --- | --- | --- |
| verified scenario | v5 split、refs_v4、B6 exogenous/policy 严格验签 | `scenario/` loaders |
| mapper | aggregate → Task，整数账本守恒 | `scenario/arrival_mapper.py` |
| formal env | verified injection、causal observation、0.5h 物理语义 | `scenario/env_injection.py`、`envs/idc_price_env.py` |
| env release | 独立验签产物，env=true、training=false | `scenario/env_release.py` |
| rollout | formal real-data transitions，支持 corrector on/off | `safe_rl_v2/rollout.py` 与 probe |
| PPO objective | clipped actor objective + 三 critic target | `safe_rl_v2/ppo_objective.py` |
| single update | 一次真实 buffer 更新 | `safe_rl_v2/ppo_update.py` |
| multi-batch probe | 同一调用方对象连续两批 | `safe_rl_v2/ppo_two_batch.py` |
| resume probe | 批次边界完整状态保存/恢复 | `safe_rl_v2/ppo_two_batch.py` 测试适配 |
| formal training entry | 只有 verified preflight，随后明确阻断 | `safe_rl_v2/train.py` |

`make train` 当前预期 exit 2。正确错误归因应是 training release 未放行，而不是 M1.2、
缺数据或 2023 硬编码起点；不得回退 synthetic，也不得写成功 run/checkpoint。

## 7. 下一阶段计划

### 7.1 已经开出的下一张卡

下一卡为 `M1.3g-f-c-f`：**三批 formal 更新连续性 probe**。

目的：在不改 `train.py` 的前提下，将已有的两批和恢复证据扩展到至少三个合法
train origins，并对照：

```text
连续三批
vs
前两批 → checkpoint → 新对象恢复 → 第三批
```

第 3 批 transition 以及最终 policy、完整 optimizer、Lagrangian、RNG 必须逐项一致。
所有对象和超参数仍由调用方提供；不得接正式训练入口或宣称训练有效。

开卡时应先提交：

```text
docs(review): approve M1.3g-f-c-e-R2 boundary snapshot repair
```

再提交 f-c-f 任务卡。当前仓库尚未包含这条审核收口提交。

### 7.2 f-c-f 之后的建议顺序

1. **训练配置人工审计/冻结**：明确 optimizer、clip、gamma/lambda、批次/rollout、
   Lagrangian budget 与种子所有权；不得由 probe 或入口偷偷选值。
2. **正式训练运行与 checkpoint 契约设计**：明确 run 目录、周期 checkpoint、恢复点、
   失败原子性和 claims；先做设计/门禁卡，再写入口。
3. **发布版本迁移**：`safe_rl_v2/train.py` 在 `ENV_RELEASE_SOURCE_PATHS` 内，任何修改都会
   使 env release v1 的 live revision 失效。必须保留 v1，物化新版本；不得覆盖 v1。
4. **正式训练入口接线**：只消费 verified train split 和发布后的训练配置；禁止
   synthetic/oracle/旧 manifest fallback。先红测试必须早于实现。
5. **受控正式训练**：先证明 run/checkpoint/resume 和 claims 正确，再讨论性能；单次短跑
   不足以宣称有效或收敛。
6. **M6 准备提前**：现在确定评估协议；checkpoint 契约稳定后补齐评估器，
   用受控短跑 checkpoint 与 train-only 场景验证。正式训练产物经审核后才运行
   validation，并在选择锁定后对预定矩阵运行一次最终 test。

后续卡号可在审计后确定，不要为了延续编号而把多个风险层塞进同一张卡。

## 8. 固定协作工作模式

当前实际工作流是“三方接力”：

```text
ChatGPT 审核/开卡
    ↓ 用户复制任务卡
VSCode 执行实现与测试
    ↓ 用户复制简短报告
ChatGPT 复审
    ├─ 通过：给出下一张卡
    └─ 不通过：只给最小返修卡
```

### 8.1 ChatGPT 的职责

- 先读当前 Git 和任务卡，不只信报告文字。
- 审核要**非防御性**：围绕卡片目标、真实生产风险和证据闭合，不为理论上所有可能性
  增加通用防御系统。
- 区分三类问题：生产缺陷、测试/证据缺陷、文档账本错误；定级和返修范围不能混淆。
- 不因单次、已隔离且与改动无关的 MILP 墙钟 flake 否决整卡。
- 发现问题时给出可复制的最小返修卡；不要顺便扩展范围。
- 通过时明确说“审核通过”，再给下一卡；不能让用户猜结论。

### 8.2 VSCode 的职责

- 每卡先做 preflight，再提交任务卡；若修改 `step()`，失败测试必须先于实现提交。
- 显式 `git add <files>`；禁止 `git add .`、`git add -A`、amend、rebase、reset hard、
  checkout --、git clean。
- 改前已绿的测试登记为回归守卫，不得人为制造先红。
- 自己发现的假绿、错误断言、flake 或实现偏差必须如实登记。
- 卡结束工作树必须干净，并在 detached worktree 验证 newest-first 回滚。

### 8.3 用户的职责

- 在 ChatGPT 和 VSCode 之间复制任务卡及简短报告。
- 人工决定会改变研究语义、训练超参数、readiness 或发布状态的事项。
- 未明确授权时，不让任一代理直接进入下一阶段、正式训练或 M6。

### 8.4 VSCode 回传报告模板

报告保持短、可审核，至少包含：

```text
卡号；起止 SHA；时间正序提交；完整修改文件
真实先红原文（或明确写“只有回归守卫”）
核心行为/数值证据
focused、make check、make smoke、release --verify、make train（按卡要求）
资产是否改变；范围外修改；readiness/claims/checkpoint 状态
从最终 HEAD 验证的 newest-first 回滚
git status / git diff --check
发现的假绿、flake、自身缺陷和未开始项
```

不要粘贴大段普通日志；只保留失败原文、关键数值和可复核命令。

## 9. 审核尺度：避免过度审核

应该返修：

- 卡片点名的语义未实现；
- 证据声称逐位一致但实际只做非空/digest/近似检查；
- 测试因 monkeypatch 失效、别名快照、错误序列化等原因假绿；
- future truth、旧资产或 synthetic fallback 进入正式链；
- 单位、守恒、raw/exec action、版本或发布门被破坏。

通常不应返修：

- 与改动无关、可隔离复现为环境波动的一次 MILP 墙钟 flake；
- 卡外既有 lint 债务且数量未增加；
- 没有现实调用路径、也不在卡片验收范围内的假想攻击面；
- 为了“更安全”而要求新增一整套通用框架；
- 要求每个回归守卫必须先红。

标准是：**该问题是否会让本卡声称的结论不成立，或让正式链在现实路径上失真。**

## 10. 每次新卡的起手检查

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git diff --check
```

若 HEAD 已前进，不得 reset/rebase 回任务书旧 SHA；先判断旧卡是否已经执行，再从当前
HEAD 开新卡。工作树若不干净，必须分辨是用户修改还是本卡遗留，不能覆盖。

低风险当前状态核验：

```bash
uv run pytest -q \
  tests/test_m13gce_two_batch_resume.py \
  tests/test_m13gcd_two_batch.py -m 'not slow'
uv run python -m scripts.materialize_env_release --verify
git diff --check
git status --short
```

## 11. 永久红线摘要

- 不放松容量、SOC、充放电互斥或任务约束制造可行。
- 不清队列、不重置 SOC、不跳任务来通过测试或评估。
- 不读取未来真值；forecast 必须有来源、生成时刻和可见窗口。
- 归一化只来自 train 或预定物理尺度；冻结后不得按 validation/test 重算。
- 不改 `marl/`、`grid_model/` 主逻辑、`legacy/` 和顶层兼容 shim。
- 新链拒绝旧 23 维动作和无版本 checkpoint；不得填零或截断兼容。
- 修改 `envs/idc_price_env.py::step()` 前必须先提交失败测试。
- PPO buffer 永远保留 `a_raw` 与其 log-prob；`a_exec` 不得覆盖概率记录。
- 旧 manifest/refs/policy 必须保留但不得 fallback；新语义必须新版本、新路径。
- `formal_env_ready=true` 不等于 `formal_training_ready=true`。

## 12. 给新 ChatGPT 的首条指令模板

```text
请把 docs/CHATGPT_HANDOFF_CURRENT.md 当作当前交接入口，再读取 AGENTS.md 和
docs/task_cards/M1.3g.md。先只读核对分支、HEAD、工作树和 env release --verify。
不要把交接文档当作自动执行授权，不要启动正式训练或 M6。

当前代码快照应为 8cd2772e9b2c8d1c8d8bdc3f0215a201f79a59f8；最新通过审核的是
M1.3g-f-c-e-R2。下一张卡是 M1.3g-f-c-f 三批 formal 更新连续性 probe；开卡前
先提交 R2 审核通过记录。延续“你开卡—我复制给 VSCode—VSCode 回短报告—你审核”
模式；审核不要防御性扩张，通过才开下一卡，不通过只开最小返修卡。
```
