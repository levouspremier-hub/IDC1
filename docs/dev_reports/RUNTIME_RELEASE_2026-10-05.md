# IDC runtime修复、提速与主机放行审核

修复、性能优化和主机资格验收已完成，选定0.50秒；正式release两端验签通过，formal_training_ready=true，仅放行seed0新初始化。512批未启动。所有重型检查、性能对照、恢复和资格任务由SSH主机单槽串行执行；Mac仅开发、轻量验证和回传审核。未使用validation/test进行策略评估或选择。

## 版本与输入冻结

- 主机job：runtime-qualification-v8；完整revision：99e00c218604b1fc18667398a77b2226c0925f2d。
- 执行闭包最后修改revision：cb4b694ec584c6e2b46a697399fa00e234ed2982；138个执行文件、95项资产逐项绑定。后续文档提交不改变资格执行字节。
- execution_version：verified-a-stop-batch-v1。只有.25/.50/1.00秒三档；A/B共享原预算，不给B补时。
- uv.lock：8e6bd4a67311b32b2369b2f2b9c1fc8780214b59c452982ecddd2cf9cbe51e0a。
- 半小时数据：dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd。
- train manifest：077e0ea21725f9e42be40f774217b385fd2eed4cf99ff6658ba85725a1571865。
- 资产集合：5005eb32bc2b9971c3153d3d6be0999039572bfa25694fcadffef4af9fce7eed。

资格主机为idc-y9000p，i9-14900HX / WSL2，冻结torch单线程；代码、venv和工作数据位于/home/w1877/idc-host的Linux文件系统。旧资产集合用于原资格，保持封存；新的交付集合和两端验签记录见下文。

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

最终0.50秒v8三种子8批均值43.991367/44.128463/43.114658秒，末7批512外推6.266798/6.271905/6.120763小时。正式seed0采用约6.27小时冷构造外推；跨批缓存持续有效的条件模型约4.98小时，详见性能报告假设。未实测512，不承诺硬性小时上限，采集/求解仍是主要剩余瓶颈。

## 历史失败保留

- profile-baseline-v1：候选误收入忽略的历史runner，预检失败，未执行性能探针；源码闭包改为Git跟踪文件后新版本完成。
- short-baseline-v2：5成功批后第6批A超时失败；原批前状态及144条partial transitions完整。
- qualification-v4：1683 passed、2 failed后因源码过期中断，不能作完整门禁；v5排队阶段中断，未执行测试；v6接线过期中断，不能作完整通过。
- qualification-v7：完整门禁3220 passed、4 failed、47 deselected，尚未进入恢复/预算资格。修正审计消费者后使用新v8，未重跑替换v7证据。

以上和四个性能profile/short对照的回传receipt已逐文件大小与SHA256重新核对，全部一致。完整日志、XML、失败现场和checkpoint位于runs/remote_<job>/；不自动清理历史证据。

## 最终资格结果与放行条件

runtime-qualification-v8已成功终止，exit_code=0；完整回传receipt核验和独立主机交付预检均已通过，随后生成正式ready文件。原资格报告的ready=false/release_freeze_pending保留为该阶段事实，不回写历史产物；正式放行以新冻结release及审核run为准。

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

## 交付预检与产物审核

原资格回传5488文件全部通过大小/SHA256核验，receipt SHA256为1aad14760348e16b731ec7821d82d2f5ffc0d8e99e1d07e0ce223f1dfd69e4a6。回传时冗余手动同步与自动watcher同时写同一目录，一个本地steps.jsonl未通过checksum；停止冗余同步后单次重同步全部通过。原主机封存产物和资格案例未改变，也没有重跑资格。

已登记添加15个零字节figures/.gitkeep，仅解决既有files-only资产打包丢失必需空目录的问题，原5488文件和receipt不改。新不可变交付资产集合f20ffe34f6db6cb7f3f31e38fa4f1169f92e4292bb3d43f8521028b74143fb71，共10115文件，两端全部校验；旧5005集合保留。新asset_receipt文件SHA256为08de61eb1f23df2b8200eee6d02340e502e5d24c3689619beb73d8eaaa8a0fcb，记录字典规范化摘要也与集合ID一致。本机.idc_remote.json已选择新集合。

独立主机job runtime-release-evidence-preflight-v1固定revision 036551ed1c95ab9a9d463483650059e041dc85d9，终态succeeded/exit0。它以新输入包重读完整门禁、真实恢复、48回合、三种子短训及4h原始证据，构建并验签release；只做文件/签名审核，环境与PPO均未运行、训练更新0。执行源码138文件及原95项输入与资格完全相同；新增文档、目录占位和签名元数据不改变执行路径，不继承不同源码的资格。

正式文件configs/release/idc_runtime_formal_release_v1.json与主机已验签文件逐字节相同，SHA256为5bec763e700c040054e3ba620289ac7888c173ac69d8263fb3f550057453917d，绑定5294项必要证据。Mac重新build/verify_runtime_release通过。主机预检回传9文件全部验签，receipt SHA256为109285adbbd1f865f1fa20cdebbf0b77cfbdb411f59bc6722b4ba18c90fae675。

标准审核产物：runs/m6p2c_release_review_v1/，包含config.yaml、metrics.parquet、report.json、manifest.json、figures/，并保存可复核脚本、完整经济账、交付资产receipt及目录补充receipt。manifest记录本地revision、资格revision、锁/数据/场景、命令与go决策；report记录formal_512_started=false。artifact_receipt SHA256为120d594f46b903d31d550fe73493e129ff35c87e12296bf5b7f0af30b8f903c3。全部历史性能与资格job receipt也已重新核对，失败证据保留。

正式release仅允许seed0、新初始化及当前执行闭包；旧r7训练状态不能恢复。后续若获启动授权，应使用新唯一job/run ID，在主机沿用单槽执行和每批前/每16批checkpoint，不使用旧final初始化正式策略。当前放行是运行资格，不等于后期策略质量或论文经济结论成立。

本次停止边界固定为放行报告；即使资格通过也不自动启动512批。

配置本体另按UTF-8、JSON sort_keys=True、separators=(',', ':')计算SHA256：.25为45e0cd46f91b8f413b8c384b07f6cf7981e02a0751266cbdc9a59365c317460a；.50为6e65255cae3b9ad1df7c8f11ef1abee2953c1896931bb1551b58aa21bc0e14f1；1.00为9705421de1cabadbb5095078daa113581b84b8d9c4c74faf58ed22b252741002。候选文件hash与配置本体hash范围不同，均独立记录。

尚未独立精确测量的性能项包括队列/环境准备/资产拷贝的稳定均值，以及采集内部策略推理、稀疏构模、环境step的完整独立计时；现有嵌套分项和残差不能冒充这些测量。没有改变WSL/Windows系统安全设置，未做跨平台重构或求解器替换。剩余采集瓶颈仍须在独立任务中继续细化。
