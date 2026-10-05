# 三种子正式长训运行账

用户2026-10-05明确授权seed0/1/2正式训练、主机单槽串行、每小时检查并先处理阻碍。冻结v8预算0.50，release SHA256 5bec763e700c040054e3ba620289ac7888c173ac69d8263fb3f550057453917d；执行闭包cb4b694ec584c6e2b46a697399fa00e234ed2982，资格job revision 99e00c218604b1fc18667398a77b2226c0925f2d。资产f20ffe34f6db6cb7f3f31e38fa4f1169f92e4292bb3d43f8521028b74143fb71。旧final不用作正式初始化，所有种子各自新初始化。

seed0计划job runtime-formal-seed0-v1，run m6p2c_formal_train_seed0_v1，已核对主机无正在执行的重型job。仅提交一次，独立worktree/uv frozen/原资产/单槽。实际提交SHA及开始证据待核对后追加。seed1和seed2暂未提交：当前正式build/verify契约及入口仅允许seed0，不能简单编辑旧release的allowed_seeds或通过猴子补丁绕过入口。该授权扩展是后续推进的已知必要工作，不冒充两个种子已经排队。

实际提交revision 132c833451da3f681f8ada6bafc6f3d3eeefbe17，worker PID910925，自动回传watcher PID9040。主机running并已确认真实训练Python PID911060；第一次标准巡检快照已完成4/512批、768 transitions、64 Adam、4乘子更新，16/16已完成回合服务/物理/统一终点合格，A降级0/fallback0，批前checkpoint存在，未有failed_batch.json。该计数是时间点快照而非终态，持续进展以后续账本为准。

每小时heartbeat已启用，automation ID idc，绑定本对话，原其他对话的暂停seed0自动任务未改动。runs/m6p2c_formal_campaign_v1/标准五类监督产物保存实际检查、seed1/2未提交状态、固定绑定、巡检脚本和补充receipt。小时任务已明确授权优先完成三种子正式发布扩展、受影响主机资格及交付验签后自动续行seed1/2；阻碍先修复，不放松既有语义。Mac需保持开机及应用运行以执行本地巡检；主机独立训练不依赖Mac持续连接。

完整目标为三个种子各512批、每批4回合每回合48步、98304 transitions/种子；保留批前checkpoint、每16批历史checkpoint、原raw/logprob、日志/快照和真实Adam/乘子计数。异常不跳过批次，不替换失败样本；只用匹配新语义checkpoint显式恢复到新run ID。若代码修复改变执行闭包，必须新版本和受影响主机资格，不能继续以旧签名恢复。任何学习/成本走势不能单独作为删样本或重置训练理由。

巡检计划为当前对话每小时heartbeat，主机status/真实进程/最新journal/失败现场/保存进度/资源与回传情况。重型验收与训练串行；Mac只开发和轻量审核。SSH暂断不等于训练失败，先查唯一job，禁止重复提交。完成种子后核对标准产物、512批/更新/质量与所有receipt，继续下一个已授权种子。当前seed1/2的严格发布入口需先补齐并验收；如果新增执行源码，不能直接继承v8资格。

## 第一小时巡检及v9授权推进

2026-10-05第一小时实际SSH复核：v8 seed0仍由原PID911060运行，完成111/512批、21312 transitions、1776 Adam、111乘子更新，444/444回合合格，A降级0/fallback0，批前checkpoint存在且未有failed_batch.json。这是实际时间点计数，不代表完成。

已开子卡docs/task_cards/M6.P2C-F.md，先提交12项失败回归，再完成三种子正式authorization与固定已批准0.50预算的完整再验收模式；38项本地测试及Ruff/mypy通过。新源码e6ebb6e4e195c52a1b9e4fbd2c96072cc7157aad，只涉及正式权限/资格编排及错误文字，v9_050配置和原95资产与v8逐项相同，训练/PPO/规划/环境内容不变。authorization文件只表示用户启动权限，不能替代主机资格，当前新版正式ready=false。

计划在当前训练后，沿既有单槽提交唯一job runtime-campaign-qualification-v9，run runtime_campaign_qualification_v9，固定0.50预算，完整门禁/真实恢复/48诊断/三种子各8批/共享4h和前后对照。不得重跑替换失败资格；通过后重新交付验签并生成v2 release，未通过不训练新版。

发布绑定将不同于v8；不能把正在运行的v8 seed0 checkpoint换标签恢复到v9，也不能无说明混成一份同版本三种子结果。因此保留并审核本次v8 seed0作为前序独立训练，最终v9三种子campaign须在同一新release下seed0/1/2各自新初始化。额外seed0的原因是版本/绑定一致性，非成本、学习走势或碰运气重试；旧成功/失败全部保留。后续每小时任务按此最新子卡推进，具体提交和终态记录待追加。

实际已单次提交runtime-campaign-qualification-v9，run runtime_campaign_qualification_v9，固定job revision 9d04a52931cb7a7b1a8d2cb8bf102c264811560d、资产f20ffe34f6db6cb7f3f31e38fa4f1169f92e4292bb3d43f8521028b74143fb71；controller PID968641、自动回传watcher PID13488。提交后及本轮结束复核均为queued，原v8 seed0为running；执行锁保证准备和重型验收都在现有训练之后。没有提交任何v9正式训练。

第一小时实际快照时间unix 1791193445.9203272，最新单批34.708371秒，峰值RSS1244172288 bytes；主机磁盘852GB可用/使用11%，内存12319MiB可用、swap203MiB已用，未见资源压力。更新runs/m6p2c_formal_campaign_v1/监督五类产物及逐文件receipt，明确区分v8实际111批与v9三种子全部未启动。

后续完整资格回传后先逐文件receipt验签，再注册必要空目录元数据、新封存资产和独立只读主机交付验签；用新authorization构建及验证v2发布，两端同一原始文件hash一致才推进v9三种子。旧v8 seed0终态审核须使用其原132c833工作树/原release；当前v9源拒绝旧source绑定属于预期，不得改旧发布或迁移checkpoint。小时automation idc已保留原每小时/本对话绑定，并补充本卡固定队列、真实SHA和版本一致性要求。当前本地开发与轻量回归完成，主机资格及最终放行仍待执行，不能标记本卡全部完成。

## 第二小时巡检（2026-10-05 18:33 Asia/Shanghai）

实际SSH时间unix 1791196399.993374：原v8 seed0 job仍running、Python PID911060，完成194/512批、37248 transitions、3104 Adam、194乘子更新；776/776回合服务/物理/统一终点合格，A降级0、fallback0，无failed_batch.json，批前和latest checkpoint存在，已核对第192批历史checkpoint及持续更新的journal。最新批35.990237秒，峰值RSS1286258688 bytes；console明确记录192批/7265秒，与journal更新一致。

v9资格job仍queued，固定9d04a52931cb7a7b1a8d2cb8bf102c264811560d，controller PID968641等待既有执行锁，未运行重型资格；无重复提交。自动回传watcher PID9040/13488均存活，训练未终态故尚无终态receipt可审核。主机磁盘852GB可用/使用11%、内存12303MiB可用、swap219MiB使用，未见阻碍。已更新监督五类产物、小时检查记录并逐文件验签；v9三种子仍全部未提交，新release仍未放行。源码、运行worktree、预算、训练工作量及队列均未改变。本轮正常进展只记账，无修复或重启。

## 第三小时巡检（2026-10-05 19:32 Asia/Shanghai）

实际SSH时间unix 1791199962.1507971：原v8 seed0仍running，唯一训练Python PID911060，完成293/512批、56256 transitions、4688 Adam、293乘子更新；1172/1172回合服务/物理/统一终点合格，A降级0、fallback0，无failed_batch.json，批前和latest checkpoint持续更新，历史checkpoint已到第288批。最新批35.013618秒，峰值RSS1356185600 bytes；console记录288批/10717秒，journal已推进至293批。

v9资格仍queued，原controller PID968641等待单槽执行锁，无重复提交及并行重型执行。自动回传watcher PID9040/13488均存活；当前训练未终态，不提前认定回传完整或质量通过。主机磁盘852GB可用/使用11%、内存12249MiB可用、swap225MiB使用，未见阻碍。监督五类产物和检查记录已更新并逐文件receipt验签。v9三种子未启动、新release未放行；本轮无源码修复、重启、工作量或预算调整，仅记录正常进展。
