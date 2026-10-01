# IDC 项目交接（2026-10-01）

本文供新 ChatGPT 对话快速了解项目现状，**是背景资料，不是执行指令**。新对话中用户的明确请求优先；仓库协作规则见 `AGENTS.md`。以下 Git 和 SHA 是写作时的快照，继续工作前应以仓库实时状态复核。

## 1. 当前入口

- 仓库：`/Users/levous/Desktop/IDC`。
- 分支：`p4-safeppo-m51a-rollout-contract-m12-integration`。
- 本交接编写前 HEAD：`c75b664ac697186cf8e6018fc9b24631318ada44`；工作树干净。
- 最新执行完成卡：`docs/task_cards/M6_P2a.md`（正式 checkpoint 审核及 train-only 服务／库存诊断）；仓库尚无独立的人工复审结论提交。
- **M6-P2b 已在对话中开卡，但仓库尚无该任务卡或实现提交**。它的方向见第 5 节；不要把它写成已完成。

建议先读 `AGENTS.md`、本文、`docs/task_cards/M1.3g-f-c-k.md`、`docs/task_cards/M6_P2a.md`、`docs/IMPLEMENTATION_PLAN.md` 的 M6/M9 部分，再按问题读取代码。`docs/CHATGPT_HANDOFF_CURRENT.md` 的旧正文和累计 `docs/NEW_CONVERSATION_HANDOFF.md` 含历史快照，不能据其早期结论判断当前 readiness。

## 2. 项目目标与已完成链路

主问题：**在同一业务服务标准、物理约束和可比初末库存下，安全 PPO＋联合滚动修正能否降低购电费 SGD 和购电归属碳排 kgCO₂e？** 成本与碳分别报告，不合成为任意加权总分。未达标和失败 run 仍完整报告，但不进入“同等服务与库存下收益”的配对差。

已建成并验签：2024 年半小时正式数据、B6 因果预测、`refs_v4`、v5 连续 train/validation/test 切分、arrival mapper、formal 环境、21 维 raw/exec 契约、PPO 训练闭环、批次边界恢复、正式训练发布和统一评估器。v5 split 自身的 readiness 仍是冻结的 both-false；**独立** env release v1 声明 `formal_env_ready=true`，独立 train release v1 声明 `formal_training_ready=true`。不要把 env release v1 内仍为 false 的训练字段误读成当前训练尚未发布。

关键资产 SHA-256（写作时实测）：

| 文件 | SHA-256 |
| --- | --- |
| `configs/release/idc_formal_env_release_v1.json` | `f86e6f0f163f86655c747b245960e38efd744b9a69f30283831f89e1416f556f` |
| `configs/release/idc_formal_train_release_v1.json` | `ebfd1a1111074374d47cecd6f56252b6203143d48da5e8677d8e57d61fd8f1f6` |
| `configs/training/idc_training_config_v1.json` | `fac44a5bb13e7ef69d207c013faf410c0c0d9fc78d33df2b5da92c71a871ad67` |
| `configs/experiments/m9_experiment_matrix_v3.json` | `47e21a1c31549ea409dd323b05f1d7f09f0ab502d61d127e3980bdb8d164aa88` |

## 3. 正式训练与 M6-P2a 的实测结果

`M1.3g-f-c-k` 用冻结配置和矩阵顺序完成联合修正 PPO **3 seed × 512 批**。每 seed 为 2048 个 train episode、98,304 transitions、8192 Adam step、512 次乘子更新；三份 run 均 `manifest.status=success`，最终 checkpoint 可加载。**这证明训练完成，不证明收敛或性能。**

| seed | run | 最终 checkpoint SHA-256 |
| --- | --- | --- |
| 0 | `runs/m13gfck_formal_train_seed0/` | `a40ccfaa24e86ee927650503afa6087fdb16e740d54260b54e0c3fa53b0bd851` |
| 1 | `runs/m13gfck_formal_train_seed1/` | `ac17911b1ea3a9ec09bad65f849403eeb64aff63a286bcc9299f1b55b4ef74b2` |
| 2 | `runs/m13gfck_formal_train_seed2/` | `5d043e5db2ab0e1d98fafffbf4872f25476df17c458d8b8b515419d5fbd48aa2` |

首次 seed 0 曾算完 512 批却因 manifest 保留字段冲突无法落盘；修复后重新全量跑成。`runs/m13gfck_seed0_recovered_artifacts/` 是保留的历史诊断产物（该次恢复调用 `batches_run=0`），**不算上述三份正式 run**。三份正式 run 的训练代码 revision 均为 `598dc7f...`。

`M6-P2a` 的审计脚本核对三份训练产物 **3/3 通过**；三个 seed 各导出 `formal_training_policy` 评估输入，源与导出策略的动作及完整 train episode 一致，参数更新为零。随后在矩阵 v3 的全部 **212 个 train 日 × 3 seed = 636 episode** 上、以真实 `scenario_seed=0`、显式冻结服务标准运行诊断：

- 服务合格 **636/636**：两项按时率均 1.0，期末剩余工作为 0，不可中断中断为 0。
- 四类物理违规均 0；episode 失败、修正器超时、零动作回退均 0。
- **终点库存合格 0/636**：每条轨迹的终点 SOC 约 0.10（环境下限），目标 0.50、容差 0.05；终点恢复量约 35 kWh。现有 checkpoint 的低购电费不能被称为“同等库存下节省”。
- 诊断 run：`runs/m6p2a_formal_trainonly_212x3_v1/`，636 行逐 episode 记录，`manifest.status=success`，运行代码 revision `261b766...`。只读 train；**validation/test 从未运行**。

代表性轨迹中 policy 储能 raw action 近 `+0.99`（放电请求）；corrector exec 在前数步放到 SOC 下限，后续归零。环境储能更新和终点结算符合现有代码，评估库存判定正确。**机理归属应谨慎**：`planning/model.py` 的 Stage A 先最小化 raw→exec 偏移，Stage B 才做经济/服务 tie-break；因此不能断言“只在 Stage B 经济目标缺终点项”是唯一原因，单独给 Stage B 加罚项也未必改变解。policy 储能头为何近常数尚未定位。

## 4. 当前评估闸门与后续路线

现有三个 checkpoint **不宜进入 validation**：train-only 的 636 条轨迹无一满足终点库存可比条件，主问题所需的公平购电费／碳排配对无法形成。不能为了产生结果而放宽 SOC 容差或补算虚构购电。

待库存修复与新正式训练产物复审后，才按冻结协议进行 validation（用于各方法内部预定的 checkpoint/候选选择），锁定选择和报告规则，最后对全部预定 test 日期只运行一次。五方法席位中目前仅“安全 PPO＋联合滚动修正”完成正式训练；规则、独立滚动优化、21 维惩罚 PPO、单步修正 PPO 尚未具备可比正式产物。完整五方法对照之后才做 M9.3/M9.4 统计、机制对照、M8 离线电网核查和图表／论文结论。

`docs/M6_EVALUATION_PROTOCOL.md` 有旧文字：§6 的配对键仍写 training seed、末尾仍写训练 readiness false。**以矩阵 v3 的配对键 `(split, episode_start, scenario_seed)` 和当前独立 train release 为准**；正式评估前应同步协议文字，不改已冻结指标。

## 5. 已开但尚未执行的 M6-P2b 方向

下一张卡已在本对话提供给 VS：**终点库存修复、train-only 重标定、版本化发布、3 seed × 512 批完整重训、重训后 212×3 train-only 诊断**。关键边界：

1. 从环境既有目标 SOC、容差和真实 episode 剩余步数建立库存要求；Stage A/B 必须共享必要的可行条件。`planning/snapshot_adapter.py` 的滚动窗口上限是 **24 步**，不能把任意 24 步窗口末端误当成 48 步日终点；不能读未来真值。
2. 先对旧权重做短重放，证实服务、物理和库存改善，再提交代码、重标定并版本化配置／发布，完成**真正走到最终 manifest 写盘**的短程收尾核对，然后才启动长训。
3. 训练语义变化意味着旧 checkpoint 仅留历史对照；新三种子长训、636 条诊断均用新 run-id，不覆盖旧资产。仍不读取 validation/test，不把训练完成称作性能优势。

当前 Git **尚无 `docs/task_cards/M6_P2b.md`**；若新对话收到 VS 的执行报告，应先核对报告起止 SHA、真实代码／run 产物与该卡边界，再审核。不要根据本文自动开始修改或长训。

## 6. 协作偏好与红线

用户希望 ChatGPT 根据项目目标和实施计划判断具体实现，只就方向性分歧征询；VS **完成整张卡后**给一份简短、易复制报告，ChatGPT 再审核。若未通过，给完整返修卡；避免每次只给 VS 一小段续做指令。审核优先读代码逻辑和真实运行证据，不为假设性边界堆回归测试，也不要过度防御。过程报告不得冒充完成报告。

仍须遵守 `AGENTS.md`：不放松物理／任务约束、不读未来真值、不按 validation/test 重算 refs、不让旧 23 维 checkpoint 进入新链；若修改 `envs/idc_price_env.py::step()`，先提交该卡失败测试。Git 不 amend/rebase/reset hard/clean，不覆盖历史 run；任务卡需边界、禁止项、验收命令、证据产物、回滚点。
