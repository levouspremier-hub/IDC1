# M6-P2b 复审返修记录（受控单seed验收通过）

本轮从519d01a继续；任务卡初始起点c3e37fb，分支p5-eval-viz-m6-p2b-runtime。
用户已将后续范围改为先修复并只观察一个seed，本轮不启动三种子长训、validation/test。
共同50%目标来自库存公平配对协议，45%–55%仍是库存合格区间；它不是物理定律。
方法间终点电量差1e-6kWh保持不变。

## 已修复及证据

- 已到达任务的服务与终点库存使用同一时域可行域，不再用可被优化器延后的
  “最早执行”静态预留证明库存；全时域执行优先级、不可中断连续性已接入。
  首次启动允许部分执行，启动后按已有任务速率连续执行，完成末步允许剩余量。
- 因果预测的聚合到达保留队列平衡与最早处理；不可服务部分保留积压，不创建
  尚未到达任务实例，不把预测任务积压丢掉来增加回充空间。
- R/A/B继续共享0.25秒；冗余累计进度界及变量界均由原任务量、速率、截止时间
  推导。Stage A/B共用R证明的最小终点缺口。超时与不可达分别记录。
- 库存缺口扩大逐步对照预测输入、已到达任务新增、计划/执行任务进度及SOC。
  同输入、同进度下的可达性损失阻止放行；输入变动不自动证明预测误差。
- gate核对完整episode、服务、物理、库存、证明与回退；不能仅以末步R=False
  绕过失败。单seed短跑报告不具备三种子长训放行资格。trace也以episode资格
  判定状态和退出码，不再仅凭“未捕获异常”写success。

6d22ab6/6447eee为连续执行与求解界限修复；18项调度回归通过。
`runs/m6p2b_service_trace_v2_r4_retry1/`：4416/6624共96步，服务、目标、物理通过，
终点均50%，无回退、超时、未证明可达性。首次索引异常保留于r4首次trace，
其旧manifest状态判定存在缺陷，报告两例IndexError不能算作成功。

`runs/m6p2b_calibration_v2_r4/`的72例服务/目标通过，但7次超时，passed=false。
收紧推导界后`runs/m6p2b_calibration_v2_r4_retry1/`的72例均通过且无回退；
r4配置/矩阵已冻结，发布绑定到实际代码。后续奖励语义改变后，r4仅作历史资产，
不能把旧live绑定当作新发布或新训练恢复依据。

## 学习信号与具体奖励返修

`runs/m6p2b_reward_counterfactual_v2_r4_final/`完整72例均合格、无回退。
48个同服务/同库存公平配对中9个净获利，9个原环境奖励均下降；采用已有
common-sgd-degradation-v1后仅1个实测奖励增益为正。购电和退化单位奖励斜率
均为1/120，不再有此前600倍退化尺度错误。主要抵消项是负载和动作平滑惩罚；
单独去掉这两项的算术对照中，9个净获利配对的奖励增益均为正。
这只是诊断依据，不用算术替代新语义的实际对照。

24个日初探针全部坍缩，不能推广到全天。
`runs/m6p2b_storage_state_probe_v2_r4/`固定24-origin、时刻0/12/24/36，共96个
当前状态、288次同状态探针。64个状态非坍缩，32个坍缩；全部探针无回退，
24条基础轨迹完整且服务/物理/目标合格。记录实际任务进度、购电和退化费、
奖励分项；实际当前外生值仅用于环境执行，规划仍只取验签预测与当前已到达任务。
因此不是全程raw提案对执行完全无效，但不能用执行改善证明PPO学会了调度。

用户已授权在诊断证明需要时进行具体奖励返修。任务卡补充公式后，c24ceec先提交
6个失败回归；d927480在wrapper新增common-sgd-potential-smooth-v1：

Phi(s)=-0.05*mean(abs(prev_loads-base_load))-0.03*mean(abs(prev_exec_action))，
终止Phi=0；r_new=r_common-r_load_smooth-r_action_smooth+gamma*Phi(next)-Phi(current)。

系数沿用原环境值，gamma严格等于冻结PPO的0.99498743710662。完整折扣塑形
和恒等于-Phi(initial)，不改变同初态的策略回报差。现有reset的prev_action=0.5，
正式初态Phi=-0.015，所以共同塑形常数为+0.015，不能当作储能收益。两项原始
平滑值保留在审计中，其他奖励项和所有物理/服务/归一化标准不变。
不新增逐步SOC惩罚，不提高已经被修正器消除的终端SOC权重，不修改env.step。

新奖励8项直接回归通过；r5标定首次72例服务/目标均通过，折扣塑形和范围
[0.014999999999999987,0.015000000000000012]；6624/compute=1一次超时，
因此passed=false、未冻结。随后相同代码/预算的retry1与实测收益对照已通过，
详见下方r5证据；完整make check及seed0短跑随后均通过，详见最终受控验收；不宣称覆盖全部未来情况。

## 后续闸门

r5标定、发布验签、原生收益对照与真实折扣塑形核对、完整make check及
新初始化seed0×8批（1536 transitions、32完整episode）已经完成；最终
checkpoint/report/manifest已重读验签。下文报告raw/exec差距、头饱和、梯度和有效优势。
不以训练成本是否漂亮决定是否值得验证，也不把物理合格的回退隐去。

目前validation readiness=false：单seed受控短跑已通过，但未运行新版三种子正式
重训及636条新训练诊断。四个方法席位仍缺：rule_baseline、
independent_rolling_optimization、penalty_ppo、safe_ppo_single_step_corrector。
单方法修复不能宣称五方法公平比较已就绪。后续仍按validation后test顺序。

回滚采用独立提交逆序revert，不覆盖历史runs，不改受保护目录或主分支。
最终验收将补充起止SHA、实际命令及结果、证据hash、回滚列表和干净工作树。

## r5 受控发布进展

`runs/m6p2b_calibration_v2_r5_retry1/`：相同代码和预算复跑72例，完整、服务、
目标全部合格，回退为0；共同折扣塑形常数仍为0.015。50a1a90冻结全新配置/
矩阵，db4c53e提交绑定实际代码、gamma和塑形证据的发布。r4及r5首次失败均保留，
不能因一次复跑成功宣称所有偶发超时已被消除。

`runs/m6p2b_reward_counterfactual_v2_r5_final/`：72/72完整、服务/物理/目标合格，
无回退。48个公平配对中9个净获利，9个实测训练奖励增益均为正；实测与公式最大
误差1.2351231148954867e-15。折扣塑形项范围
[0.014999999999999987,0.01500000000000001]，同初态策略间差为浮点精度。
这是实际运行对照，不用旧r4的算术补偿代替新奖励的执行结果。

`runs/m6p2b_related_check_v2_r5/`：78项相关测试通过，0失败、0跳过，原生JUnit/
console与来源hash均保存且读回验签。完整`runs/m6p2b_full_check_v2_r5/`的
ruff/mypy通过，pytest随后因r6接口修复主动中止，详情见下节；不作为完整通过证据。

## r6 最后接口校验与完整门禁

额外复审合成环境确认：wrapper曾接受23维raw、截成21维exec并推进环境。
checkpoint/buffer拒绝旧维度并不能代替wrapper入口拒绝。336b2ca先提交20/22/23维
失败回归，3d5aaaf添加进入规划前的明确ValueError，所有错误维度不推进环境。
这只改变非法输入的拒绝行为，合法21维的执行、奖励、物理与标定参数不变。
ac87616提交r6发布，绑定实际代码并显式继承已验签的r5配置/矩阵/标定。
配置仍为v2_r5、矩阵v4_r5，发布与受控run-id为r6；不把继承写成重新标定。

`runs/m6p2b_full_check_v2_r5/`因补此红线修复主动中止：2622项已执行测试无
失败，但退出码2，原生JUnit/console及五类产物保留、passed=false。这不是完整
门禁通过。新`runs/m6p2b_related_check_v2_r6/`81项全部通过、0失败、0跳过，
包含缺失/错误折扣率、旧checkpoint、旧动作、恢复、泄漏及真实终点回归。
最终`runs/m6p2b_full_check_v2_r6/`已完成：ruff/mypy通过，3127项测试通过、
0失败/错误/跳过，47项按既有not-slow规则排除；来源未变，原生JUnit、五类
产物及读回验签通过。没有启动长训。

日内探针补充核对：64个非坍缩状态的物理实际储能动作也均非坍缩，288次
修正器储能输出与物理实际执行的最大差为0；不是只检查修正器提交动作。

## 已完成诊断的报告验签

以下均为原生报告文件的SHA256，尚不代替完整门禁和训练验收。

- `runs/m6p2b_calibration_v2_r5_retry1/report.json`：`0702a47e5cc81e51565da86b444b0ccd314b372b8ab49c2154615371a78304f6`。
- `runs/m6p2b_reward_counterfactual_v2_r5_final/report.json`：`ab56d733e48cf3780f2a357e123a233870961eb698b5fdb620c3d9f56aaf1e39`。
- `runs/m6p2b_storage_state_probe_v2_r4/report.json`：`c8a5470fceffa14d8b39527a0d28f0982d0bafb979c5aa73c888d076bb4ba232`。
- `runs/m6p2b_related_check_v2_r6/report.json`：`86054c2388e6dac034a8c9c0b49182779f13113e533827cbfa1d14ca566df584`。

## 完整门禁与下一步

完整make check退出码0，耗时约41分钟；报告SHA256：
`6751530d530d636d16c43fb3dba72159785e46f4f1ac2960e1197bc503197ba6`。
历史重放和seed0短跑随后完整结束并验签，结果见下文。

## 旧三种子历史重放（不更新权重）

`runs/m6p2b_historical_replay_v2_r6/`144条记录、24-origin×3旧seed×前后两臂，
五类产物已写盘并重读；旧三份checkpoint hash均不变。两臂各72条完整/服务合格、
物理违规0、回退0。旧臂库存/共同目标0/72，终点SOC约0.1；修复臂均72/72、
终点恰0.5、超时与未证明可达性0。原始头饱和均值0.9169560185；
raw到exec平均差从0.9099051613变为0.9890795764。旧臂日均放电38.0000131kWh，
修复臂几乎不充放电。该结果证明修正器库存保障，不能称为旧PPO已学会调度；
终点库存不同，前后成本不作公平收益结论。
报告SHA256：`863ca67d6913baf361c7e6228f832bb74c6f8e1244610af17c7431b3dc8db0ab`。

新初始化seed0×8短跑及独立单seed闸门随后完成，结果如下。

## 最终受控验收（2026-10-03）

本轮实现起点519d01a，最后实现提交3d5aaaf，发布ac87616；验收证据执行时
HEAD为e91234b。本任务卡起点c3e37fb，原始M6-P2b项目起点55be5e5。
最终文档提交不改变发布绑定的实现；完整起止提交在交付消息及git log可查。

实际命令：

- `uv run python -m scripts.m6p2b_quality_gate --run-id m6p2b_full_check_v2_r6 --full`
- `uv run python -m scripts.m6p2b_inventory_repair --phase replay --run-id m6p2b_historical_replay_v2_r6`
- `uv run python -m safe_rl_v2.inventory_train --short --seed 0 --run-id m6p2b_short_seed0_v2_r6`
- `uv run python -m scripts.m6p2b_inventory_repair --phase gate --seed0-gate --run-id m6p2b_short_gate_seed0_v2_r6`

单seed短跑8批、1536 transitions、128 Adam步、8次乘子更新、32完整episode。
完整、冻结服务、物理、45%–55%库存与共同50%目标全部32/32通过；回退、
未证明可达性、不可达记录均0。SOC范围[0.4999999992363155,0.5000000014197105]，
最大目标缺口1.4197105e-7kWh，小于既有1e-6kWh；没有放宽阈值。
共充电367.1931473kWh、放电331.3918155kWh，32条轨迹均出现双向储能，
这不是要求每天循环，也不证明每一次随机策略充放电均经济最优。

8批储能raw→exec平均绝对差0.5138398293；头饱和均值0.0201822920，执行储能
标准差均值0.0659291593；修正器输出与物理实际储能执行差为0。储能头梯度
均值0.2721442551（范围0.1713343153–0.3934231056），有效优势标准差均值
1.0318231806，raw储能与优势协方差均值-0.0006928836、正负均有。终端惩罚和0，
购电/退化奖励的8批累计分别-1.7488217659/-0.1164308271；终端惩罚已被修正器
消除，不能靠加大其权重解决学习信号。短跑信号存在，但未证明PPO已形成稳定
经济调度；收益机会与奖励方向的证据来自固定24-origin的9/9实测公平获利配对。

最终checkpoint hash：
`8f6061c4b0bdb677acbe572206d1ed92a6f4af80aa2739a8b5e198412bce3803`。
最终报告hash：
`489388c2866dd6a85789d97c1088c2ee4eaca239a5afaf8b6983fd83001d992a`。
config/metrics/report/manifest/figures与checkpoint均完成写盘及重读验签。
单seed闸门passed=true、failed_checks为空、formal_three_seed_gate=false。
旧checkpoint仍只允许历史诊断；没有从旧权重恢复新版训练。

统一原生验收报告：`runs/m6p2b_repair_acceptance_v2_r6/report.json`，
SHA256 `f6fa418a4c2a38b235012867a82468c0d8b49b659dfcedefbc95698bf0c228f4`。
该run保存六组输入证据的四类文件hash、各自来源manifest、实际实现绑定与
最终checkpoint来源；五类产物及读回receipt齐全，汇总生成源码另存
finalization_source.py。报告scope仅train-only修复及受控seed0验收；未运行
三种子正式重训、636新诊断、validation或test。第一次汇总命令因/tmp模块路径
缺失在写run前报错，随后以显式PYTHONPATH=.执行成功；没有重跑或覆盖训练资产。

仍保留r4/r5标定偶发超时、r4首次trace索引异常及被中止r5完整门禁。
有限验收不证明所有未来求解都能满足预算，冻结4.4°C与聚合工作量假设也不是
实际未来可达性保证。预测或实际任务输入改变时仍须逐项记录，不凭SOC推定原因。
validation readiness=false，缺少新版三种子正式训练复审、636日诊断及前述
四个方法席位；不增加训练成本门槛，不称五方法比较已就绪。

独立回滚点（逆序revert、先保留全部runs，不使用破坏性Git操作）：
6d22ab6、6447eee、6c94c0e、08d2e56、4d37639、989970b、c24ceec、d927480、
7b5a36a、50a1a90、db4c53e、336b2ca、3d5aaaf、ac87616。
受保护目录及env.step均无本轮修改。最终git diff --check与干净工作树检查
在验收记录提交后执行；完整测试后仅追加验收文档和原生证据，没有改实现/测试。
