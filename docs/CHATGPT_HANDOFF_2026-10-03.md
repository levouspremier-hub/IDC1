# IDC 项目交接（2026-10-03）

本文是供新对话理解项目的背景快照，不是执行指令，也不授予启动训练、修改代码
或运行评估的权限。用户的新请求优先；仓库规则以 `AGENTS.md` 为准。
与10月1日交接不同：M6-P2b已实现，受控短跑通过，修复后的seed0正式长训已结束，
但长训暴露后期回退与服务失败，尚未通过质量复审。旧交接保留作历史资料。

## 1. 当前结论与授权边界

- 仓库：`/Users/levous/Desktop/IDC`；分支：`p5-eval-viz-m6-p2b-runtime`。
- 本次文档开卡前HEAD：`e2c69051eb48e03a1e0ad2dd0b7027e4580ff485`，工作树干净。
  文档开卡提交：`c350e78`。训练入口最后实现提交：`73fcc09`；r7发布提交：`b4f49a5`。
- 用户先要求修复库存/学习信号问题并只做一个seed短跑，随后明确授权
  **只启动seed0的512批正式训练**。该次训练从新初始化开始，现已结束。
- 新版seed1/2未运行，新版三seed产物复审及636条独立train诊断未完成。
  validation/test均未运行，validation readiness仍为false。
- 当前请求仅为交接文档。本文的后续建议不等于用户已授权返修、重训或评估。

核心问题仍是：在同等服务、物理约束和初末库存下，安全PPO＋联合滚动修正能否
降低购电费SGD及购电归属碳排kgCO₂e。两项分别报告；不合格样本完整保留，
不用于声称“同等服务与库存下节省”。不设置训练成本必须足够漂亮的验证门槛。

## 2. 已完成的库存与学习信号修复

详细证据见 [M6-P2b复审记录](audits/M6_P2b_R3_REPAIR_REVIEW.md)，
任务卡见 [原卡](task_cards/M6_P2b.md)、[返修卡](task_cards/M6_P2b_R3.md)。

1. 规划快照携带真实episode结束步、剩余步数、目标及合格上下界。正式路径规划
   覆盖当前至真实终点的全部剩余步数（日episode最多48步），取消24步截断。
   终点50%是既有公平比较共同目标，不是物理定律；45%–55%仍为库存合格区间，
   方法间终点电量差上限仍是1e-6kWh，不放宽。
2. 在相同物理/任务可行域先求最小终点偏差R，固定该偏差给A/B共同使用，再按
   最小raw→exec偏移、经济/服务tie-break排序。三阶段共享0.25秒配置预算。
   超时/求解失败、安全回退与规划不可达分开记录，不把超时当不可达证明。
3. service reserve v3把当前已到达任务的优先级、逐步进度、不可中断连续性接入
   全部时域；允许首次部分启动、完成末步剩余量，与实际分配器契约一致。
   因果预测的未来到达只用聚合工作量和队列守恒/最早处理，不读未到达任务实例。
   充电功率预留用已到达任务的优化分配、预测温度加冻结train裕量4.4°C及零
   可再生预留。该裕量是声明的规划假设，不是已证明的实际未来误差上界。
4. wrapper逐步核对计划/实际进度、SOC与可见输入变动，记录库存缺口来源。
   不重置SOC、不清空队列、不放松物理约束，不凭库存失败推定预测误差。
   20/22/23维raw在进入规划和环境前明确拒绝；正式21维raw与log-prob保留在buffer。
5. 退化费奖励改为与购电费同SGD边际尺度，修复旧退化斜率600倍问题。
   固定24-origin实测对照又发现原负载/动作平滑罚项抵消了经济收益，随后改为
   势函数塑形，版本为 `common-sgd-potential-smooth-v1`：

   `Phi(s)=-0.05*mean(abs(prev_loads-base_load))-0.03*mean(abs(prev_exec_action))`

   `r_new=r_common-r_load_smooth-r_action_smooth+gamma*Phi(next)-Phi(current)`

   终止Phi=0；gamma严格取冻结PPO的0.99498743710662。完整折扣塑形和=-Phi(initial)，
   当前正式初态为共同常数+0.015，不作为储能收益，同初态策略间差不变。
   未新增逐步“偏离50%”惩罚，未靠提高已被修正器消除的终端权重解决信号问题。
   没有修改 `envs/idc_price_env.py::step()` 或受保护目录。

## 3. 长训前验收与当前发布

| 证据 | 实际结论 |
| --- | --- |
| `runs/m6p2b_calibration_v2_r5_retry1/` | 固定24-origin×三参考提案共72例合格、无回退；继承原预算/乘子公式，refs_v4不变 |
| `runs/m6p2b_reward_counterfactual_v2_r5_final/` | 72条完整轨迹合格；48个公平配对中9个净获利，9个实际训练奖励增益均为正；实测与公式误差约1.24e-15 |
| `runs/m6p2b_storage_state_probe_v2_r4/` | 固定24-origin×时刻0/12/24/36，共96状态、288探针；64状态提案能改变实际储能执行，32坍缩；无回退 |
| `runs/m6p2b_historical_replay_v2_r6/` | 旧三权重×24日×前后两臂共144条；两臂服务/物理均合格，旧臂库存0/72，修复臂72/72；权重hash未改 |
| `runs/m6p2b_short_seed0_v2_r6/` | 新初始化seed0×8批，1536 transitions；32/32完整、服务/物理/库存/目标合格，无回退；最终五类产物与checkpoint验签 |
| `runs/m6p2b_short_gate_seed0_v2_r6/` | 单seed闸门passed=true，formal_three_seed_gate=false |
| `runs/m6p2b_full_check_v2_r6/` | 完整make check：3127通过，0失败/错误/跳过，47按not-slow规则排除 |
| `runs/m6p2b_related_check_v2_r7/` | 93项相关测试通过；r7只改变入口与运行状态账目，未重跑完整make check，不能冒充r7全仓通过 |

旧权重历史重放中储能头约91.7%饱和；修复臂几乎不充放电，只说明修正器能保障
库存，不能说旧PPO学会了调度。受控新seed0短跑raw→exec平均差约0.514，
32条都有充放电、储能头梯度/有效优势非零；同样不证明已收敛或每次循环均合理。

当前资产为**配置v2_r5、矩阵v4_r5、发布v2_r7**，不是三个同后缀版本：

| 文件 | SHA-256 |
| --- | --- |
| `configs/release/idc_formal_train_release_v2_r7.json` | `412057ca837cbea5b4e08017317eae6c1d797897f32f7389fa38553354ddf22f` |
| `configs/training/idc_training_config_v2_r5.json` | `31e1ce5b83e0fb75e41f150c7079d64d0907ccddd823ea9313bf03c4239b0e39` |
| `configs/experiments/m9_experiment_matrix_v4_r5.json` | `eb48aa0dfcfdd82f1a1abe77849b5a876cce1582870d5a502fed1b479e516e77` |

r7发布继承已验签r5标定和r6单seed短跑，只新增显式seed0正式启动授权及运行中
五类产物记录。继承校验要求除 `safe_rl_v2/inventory_train.py` 和
`scenario/inventory_release.py` 外，规划/奖励/优化器/政策/buffer及配置/矩阵绑定
全部字节一致。短跑checkpoint只作资格证据，没有恢复其权重开始长训。
默认三seed闸门仍保留，显式 `--seed0-formal` 不能用于seed1/2或short。

版本化checkpoint绑定实际规划/guard/wrapper/奖励/523维观测、配置、矩阵和发布。
旧520维策略只进入显式历史诊断；不能通过改标签、补观测或改binding冒充新策略。
若后续改变训练内修正器或奖励，本轮r7 checkpoint也须按新语义区分历史对照，
不能自动混入新版恢复或正式评估。

## 4. 新seed0正式训练：完成，但质量未通过

run：`runs/m6p2b_formal_train_seed0_v2_r7/`。
真实命令：`python -m safe_rl_v2.inventory_train --seed0-formal --seed 0 --run-id m6p2b_formal_train_seed0_v2_r7`。

512批、98,304 transitions、2,048个完整episode、8192 Adam步、512次乘子更新；
耗时10968.17秒（约3.05小时）。五类产物及final checkpoint均写盘，artifact receipt
的全部hash已再次核对。manifest.status=success只表示运行与收尾完成，不表示
质量合格、收敛或性能优势。旧PID56898/56900已退出，不是仍在训练。

| 指标 | 实测 |
| --- | --- |
| 完整episode | 2048/2048 |
| 冻结服务合格 | 2009/2048，98.10%；39次不合格，涉及37个origin |
| 45%–55%库存合格 | 2048/2048 |
| 共同50%目标合格 | 2046/2048 |
| 物理违规 | 0 |
| 安全回退 | 1676步，49个episode |
| 可达性未证明 | 1511步，35个episode |
| 规划不可达标记 | 32步；不等于证明实际未来物理不可达 |
| 按inventory_episode_acceptance全部条件严格合格 | 1999/2048，97.61% |

失败条件可重叠，不能相加当作不合格episode总数。当前严格不合格为49条，
其中39条服务失败，35条出现未证明可达性，2条共同目标缺口未解释。
库存区间100%合格不能替代服务、求解状态或共同终点电量公平配对。

按批次区间分布（训练批次重复采样，不能把2048条当作2048个独立日期）：

| 批次（从1计数） | 服务失败episode | 回退步 |
| --- | --- | --- |
| 1-128 | 0 | 0 |
| 129-256 | 0 | 1 |
| 257-384 | 0 | 1 |
| 385-512 | 39 | 1674 |

首个回退发生在第185批；两次孤立早期回退后，大量异常集中在最后128批。
末批第512批的192/192步全部回退，服务0/4。实测corrector总调用耗时中位数
0.41167秒、P95 0.50756秒、最大0.56544秒，超过配置的0.25秒量级。
这要求分解建模、R/A/B求解及计时边界；仅凭总调用时间不能确定哪阶段耗尽预算，
也不能证明根因是预测误差、机器负载或奖励尺度。

末批exec储能标准差为0，raw→exec平均差0.5902，储能头梯度均值473.59，
有效优势均值-857.54、标准差259.35，奖励和-3392.68。它们是故障阶段的实测信号，
不是学习改善或收敛证据；本次交接没有进行根因定位或新的返修。

两条目标不合格轨迹（批次从1计数）：

| 批次 | origin | 最终SOC | 目标缺口kWh | 回退步 | 未证明步 |
| --- | --- | --- | --- | --- | --- |
| 504 | 5088 | 0.4810552770 | 1.894472304 | 32 | 9 |
| 506 | 5568 | 0.4812438519 | 1.875614806 | 47 | 45 |

两例都在库存合格区间内，terminal分类为unproven_reachability，explanation_complete=false，
prediction_error_proven=false、physical_unreachability_proven=false。不能称作“预测误差
导致不可达已经解释清楚”。全run终端审计分类target_met有2013条、unproven有35条；
这与实际共同目标2046条合格口径不同，因为中途缺失证明也会保留为unproven。

最终关键hash：

- final checkpoint：`472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935`。
- report.json：`bde53d02ed41cf24973b66919395a278dbe90e5077515c8dc41b253a851dc0d4`。

可读入口：[最终报告](../runs/m6p2b_formal_train_seed0_v2_r7/report.json)、
[逐批记录](../runs/m6p2b_formal_train_seed0_v2_r7/batches.jsonl)、
[日志](../runs/m6p2b_formal_train_seed0_v2_r7/console.log)、
[来源manifest](../runs/m6p2b_formal_train_seed0_v2_r7/manifest.json)、
[产物验签](../runs/m6p2b_formal_train_seed0_v2_r7/artifact_verification.json)。
正式run实际只保存checkpoint_latest.pt与checkpoint_final.pt，未见逐批历史checkpoint。
journal中的policy/optimizer摘要不能重构历史权重；后续诊断不可假定存在第185批或
第385批之前的完整checkpoint。launch_verification.json只是起跑时第3批的验签快照。

## 5. 旧三seed与失败资产仍须保留

旧正式run `runs/m13gfck_formal_train_seed0/1/2/` 各512批，属于旧语义历史产物。
旧636日独立train诊断 `runs/m6p2a_formal_trainonly_212x3_v1/` 服务636/636、
库存0/636，不能当作新r7的636复审。当前2048条是训练采样日志，不是新版
212 train日×3 seed的独立诊断。

已有异常证据继续保留：r4首次service trace索引异常；r4/r5标定偶发超时；
r5完整门禁因接口返修主动中止；本次r7长训服务失败/回退；启动收尾误用全仓
Ruff扫描的667项既有旧链问题（之后按Makefile主链范围0问题）。
不能因manifest为success、一次复跑成功或末端SOC合格删除失败、缩小分母或
将超时改写为不可达。旧日期交接与阶段性“running”记录是历史快照；最新最终
run报告优先，不能把当时短跑通过外推成长训通过。

## 6. 待决问题与后续建议（未自动获授权）

建议后续先开独立诊断/返修卡，而非直接启动seed1/2或评估：

1. 围绕末128批定位回退：核对具体failure class、R/A/B状态、任务数量/进度、
   raw提案、调用耗时与0.25秒预算边界；区分库存不可达、求解超时/失败、
   计划/执行不一致和观察输入变动。现有数据不足以给出根因结论。
2. 在train-only、相同当前状态与可见因果输入上复现异常，先提交失败回归再修改。
   若需要过去策略状态，先核实实际checkpoint可用性；不能用摘要或最终权重
   冒充失败发生时的策略。必要新增诊断均使用新run-id并保留异常日期。
3. 排查服务失败与异常优势/梯度的关系，先判断修正器回退是否造成信号异常，再
   判断是否另需奖励修复。原始动作、执行动作、实际储能动作应分开报告，
   不靠提高终端权重、逐步保持50%罚项、扩大预算或放松物理域制造通过。
4. 返修后的受控短跑必须真正完成final checkpoint/report/manifest写盘与验签，
   再由用户决定是否重跑seed0及恢复原三seed计划；修改语义需新版本和新资产。

validation readiness=false。缺少可信新版三seed正式产物复审、636条train独立
诊断，以及四个方法席位：rule_baseline、independent_rolling_optimization、
penalty_ppo、safe_ppo_single_step_corrector。联合修正单方法完成不等于五方法
公平比较已经就绪。后续顺序仍为修复/产物复审→validation→锁定选择/报告规则→test。

矩阵当前冻结配对键为(split, episode_start, scenario_seed)，scenario_seed=0；
training seed仅分层，不是配对键。无训练seed的规则/独立优化基线每场景只跑一次，
被三PPO分层引用不增加独立样本数。旧评估协议文字与矩阵不一致处应在正式
评估前核对同步，不改冻结指标、日期、服务标准或refs_v4。

## 7. 接手时的证据索引与Git纪律

- 最新启动/授权：[M6_P2b_S0.md](task_cards/M6_P2b_S0.md)。
- 修复/奖励/历史诊断：[M6_P2b_R3_REPAIR_REVIEW.md](audits/M6_P2b_R3_REPAIR_REVIEW.md)。
- 历史缺陷：[M6_P2b_DEFECT_AUDIT.md](audits/M6_P2b_DEFECT_AUDIT.md)。
- 全面工程计划：[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)。
- 仓库规则：[AGENTS.md](../AGENTS.md)；执行纪律：[VSCODE_CLAUDE_EXECUTION_PROTOCOL.md](VSCODE_CLAUDE_EXECUTION_PROTOCOL.md)。

源码入口：scenario/inventory_release.py；safe_rl_v2/inventory_train.py及
formal_train_loop.py；planning/snapshot_adapter.py、model.py、service_guard.py、
corrector.py；safe_rl/corrector_wrapper.py；safe_rl_v2/inventory_diagnostics.py；
contracts/inventory.py；checkpointing/inventory_eval_input.py；相关test_m6p2b_*。

启动卡回滚点d69c91a；单seed入口测试02bda65、实现73fcc09、发布b4f49a5；
库存/奖励卡各独立回滚点及提交列表见复审记录。历史产物始终保留，不覆盖。
规则包括：独立分支先提交五项齐全任务卡；显式git add文件；测试后提交；
收尾git status为空。不修改paper-baseline主分支，不amend/rebase/reset hard/
checkout--/clean/强推，不改受保护目录和兼容shim。若修改env.step，须先提交
该卡失败测试；raw与log-prob永远不被exec覆盖。

本交接只新增文档，没有运行新的训练/诊断实验、validation或test，没有修改
代码、配置、发布、checkpoint或既有run。写作中的统计来自最终原生产物只读核对。
