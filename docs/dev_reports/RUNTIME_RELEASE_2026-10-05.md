# IDC runtime修复、提速与主机放行审核

主机资格已完整通过，选定0.50秒；交付包验签尚未结束，当前formal_training_ready=false，512批未启动。所有重型检查、性能对照、恢复和资格任务由SSH主机单槽串行执行；Mac仅开发、轻量验证和回传审核。未使用validation/test进行策略评估或选择。

## 版本与输入冻结

- 主机job：runtime-qualification-v8；完整revision：99e00c218604b1fc18667398a77b2226c0925f2d。
- 执行闭包最后修改revision：cb4b694ec584c6e2b46a697399fa00e234ed2982；138个执行文件、95项资产逐项绑定。后续文档提交不改变资格执行字节。
- execution_version：verified-a-stop-batch-v1。只有.25/.50/1.00秒三档；A/B共享原预算，不给B补时。
- uv.lock：8e6bd4a67311b32b2369b2f2b9c1fc8780214b59c452982ecddd2cf9cbe51e0a。
- 半小时数据：dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd。
- train manifest：077e0ea21725f9e42be40f774217b385fd2eed4cf99ff6658ba85725a1571865。
- 资产集合：5005eb32bc2b9971c3153d3d6be0999039572bfa25694fcadffef4af9fce7eed。

|预算秒|候选文件|候选SHA256|
|---|---|---|
|.25|configs/release/idc_runtime_candidate_v8_025.json|f4293d7d771e145fb994863780c692397325c7b21bdadce20914a426da1e2ed5|
|.50|configs/release/idc_runtime_candidate_v8_050.json|cf90ac80f903b286aa223e1ccb9478c22556fd1759c0e6d4569f25b452939b2d|
|1.00|configs/release/idc_runtime_candidate_v8_100.json|293764315715f91fa04571707ec4d5ebdde00f524447febb62a2e9ed7db4fe06|

继承r5训练尺度、奖励、网络、优化器与物理/服务配置；原标定测量仅.25秒，.50/1.00明确不是已标定预算。旧r7绑定不刷新，旧final仅专用train-only只读策略诊断，SHA256为472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935；不恢复旧Adam或乘子。新版checkpoint必须绑定候选/新发布执行闭包，旧训练状态不得跨语义恢复。

## 修复范围与证据

保留294b59c已有核心修复：A最优且完整检查合格后，B超时/无剩余预算保留A；无合格方案在环境执行及更新前停止，保存原始raw/logprob、来源、实际阶段状态、快照与部分采集。批前checkpoint、最近成功状态、每16批历史checkpoint继续沿用。没有重写env.step或放松接入、SOC、互斥、任务、服务和终点要求。

新增候选签名、严格资格诊断、真实批边界恢复、正式release验签与eval-input接线。完整门禁还发现并修复正优势ratio溢出时exp反向0*inf的问题：仅在数学上已被clip的正优势溢出分支计算有限等价值与零梯度；有限输入loss/梯度逐位对照通过，负优势溢出仍在Adam前拒绝。

新A降级类别在calibration和benchmark中独立计数，不冒充最优或失败fallback；确定性探针仅排除新增纯计时读数，动作、logprob、阶段状态、预算和业务字段仍进入语义指纹。以上缺口均先有提交失败回归，再实现修复。

## 性能与剩余瓶颈

见[RUNTIME_PERFORMANCE_2026-10-05.md](RUNTIME_PERFORMANCE_2026-10-05.md)。低风险优化移除factory外重复验证，完整内部校验、每次全字节fingerprint、LRU和deepcopy隔离保留。

冷构造profile均值下降23.9%；同前5批实测总耗时下降4.72%。优化版一次8批完成32/32合格回合、1536 transitions、128 Adam、8乘子更新，A降级7次、fallback0。基线第6批A超时停止且失败批更新0，保留现场，不补成功样本。

主瓶颈仍为采集与corrector，优化8批均值环境10.347秒、采集35.060秒、PPO .117秒。末7批外推512批约6.49小时；212个独立origin适配既有256LRU的条件缓存模型约5.18小时。两者不是长训实测承诺，不将预算提高当作提速。未减少512批/4回合/48步/既定PPO更新量，未关闭hash、检查或日志。细分计时嵌套关系和未独立测量残差在性能报告中说明。

## 历史失败保留

- profile-baseline-v1：候选误收入忽略的历史runner，预检失败，未执行性能探针；源码闭包改为Git跟踪文件后新版本完成。
- short-baseline-v2：5成功批后第6批A超时失败；原批前状态及144条partial transitions完整。
- qualification-v4：1683 passed、2 failed后因源码过期中断，不能作完整门禁；v5排队阶段中断，未执行测试；v6接线过期中断，不能作完整通过。
- qualification-v7：完整门禁3220 passed、4 failed、47 deselected，尚未进入恢复/预算资格。修正审计消费者后使用新v8，未重跑替换v7证据。

以上和四个性能profile/short对照的回传receipt已逐文件大小与SHA256重新核对，全部一致。完整日志、XML、失败现场和checkpoint位于runs/remote_<job>/；不自动清理历史证据。

## 最终资格结果与放行条件

runtime-qualification-v8已成功终止，exit_code=0；完整回传receipt和主机交付预检完成前仍不生成正式ready文件。

|阶段|实测结果|
|---|---|
|完整make check|3226 passed、47 deselected、440 warnings，1520.09秒；Ruff/mypy通过|
|真实恢复|批边界0/1/2，Adam 0/16/32、乘子0/1/2；游标、策略、Adam、乘子、采样/洗牌RNG、日期与来源账本一致|
|0.25诊断|24独立进程+24共享进程，共48/48合格|
|0.25短训|seed0八批通过；seed1第6批origin1008第0步A=time_limit、B=not_run，无合格动作，环境执行前停止；本失败批0 transitions、0更新，累计Adam80/乘子5。seed2和soak不执行，不补采|
|0.50诊断|24独立进程+24共享进程，共48/48合格|
|0.50短训|seed0/1/2各新初始化8批，合计96/96合格回合、4608 transitions、384 Adam、24乘子更新；A降级0、fallback0|
|0.50 soak|共享阶段14404.280516678秒、1649回合；前后各24独立回合，共1697/1697合格，0失败、0参数更新、参数哈希不变、A降级0|
|1.00|未执行：按登记顺序，首个完整合格预算0.50选定后停止增加|

全部资格保持预登记train origins和动作模式、统一50%终点误差<=1e-6，无初始不可达或band豁免。通过固定旧final的4h只读稳定性不证明后期PPO权重的可靠性、全部212 origins覆盖或学习有效性；旧优化器/乘子未恢复。

## 服务与经济账审核

逐步cost和carbon_emission分别求和，与每回合purchase_cost_sgd/carbon_kg核对误差<1e-6。0.50诊断与soak的全部完成案例合格，business_gap总计0，无A降级观测，因而没有该预算A降级的成本/碳排比较样本。0.25短训seed0的A降级2次仅有短训计数，短训未保存完整逐步经济账，不能据此声称经济优势。

soak前后24个配对案例：购电成本均651.077145674 SGD，碳排均1910.365607115 kg，逐案例最大绝对差均0。共享1649回合因末轮未凑满24而不可直接用总平均比较；按24个origin/mode组合等权后，每回合成本27.128214403 SGD、碳排79.598566963 kg，和前后对照等权值相同。每组合68或69次，未删去末轮案例。共享总成本44714.733557581 SGD、碳排131264.920767382 kg只是重复诊断汇总，不是实际运营账单。未建立本次修复的因果降本/减排结论。

## 交付预检与产物审核（进行中）

等待完整回传逐文件大小/SHA256验签。资格完成后将保留旧5005集合，建立包含新资格证据的不可变输入集合，在独立主机worktree只读构建并验证正式release。执行源码138文件和原95项输入必须保持与资格完全一致，元数据需在Mac再次验签；该预检不运行环境、PPO或512批。最终标准审核产物、交付job和正式release摘要将在此补齐。

本次停止边界固定为放行报告；即使资格通过也不自动启动512批。

配置本体另按UTF-8、JSON sort_keys=True、separators=(',', ':')计算SHA256：.25为45e0cd46f91b8f413b8c384b07f6cf7981e02a0751266cbdc9a59365c317460a；.50为6e65255cae3b9ad1df7c8f11ef1abee2953c1896931bb1551b58aa21bc0e14f1；1.00为9705421de1cabadbb5095078daa113581b84b8d9c4c74faf58ed22b252741002。候选文件hash与配置本体hash范围不同，均独立记录。

尚未独立精确测量的性能项包括队列/环境准备/资产拷贝的稳定均值，以及采集内部策略推理、稀疏构模、环境step的完整独立计时；现有嵌套分项和残差不能冒充这些测量。没有改变WSL/Windows系统安全设置，未做跨平台重构或求解器替换。剩余采集瓶颈仍须在独立任务中继续细化。
