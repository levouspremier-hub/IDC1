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

## 第四轮巡检（2026-10-05 20:45—20:49 Asia/Shanghai）

原v8 seed0终态failed/exit1，实际结束unix1791202777.475543（20:19:37）；巡检unix1791204446.2872007。完成371批、71232 transitions、5936 Adam、371乘子，1484/1484完成回合合格，A降级0/fallback0；下一批origin48 step20 A=time_limit/B=not_run，原audit solve_time_s=.5665697420045035，reachability unproven。失败步未执行环境、失败批参数更新0、20条partial及原snapshot/proposal/raw/logprob保留，不能补采或把未完成141批算通过。原39文件回传receipt全部核验，receipt SHA c0ac8e07c9deacd3f50883032b2f21e8d27088baf16e100034597696d0eb805b，failed_batch SHA 1d2b97f270ce00253dcaee6cd5605d390cd925a81fe988818a8e2436e12ad749。before/latest checkpoint都next_batch_index371、旧release匹配、采样/洗牌RNG相同，日期来源账本仍在；没有重启、换签或跨绑定恢复。

已开M6.P2C-G诊断卡；Mac仅一次轻量原快照profile，约.132535秒可执行，但不能据此称主机问题解决；失败前无证明可执行A，现有安全停止行为正确，根因未确定。已固定132c833单次提交runtime-formal-seed0-failure-probe-v1/run m6p2c_formal_seed0_failure_probe_v1，controller1113176、自动watcher20077，queued在现有v9资格之后。五次预定只读.50快照重放（第五次profile）全部保留，无环境/PPO，不以成功重放替代原失败、不无限重试。标准本地失败审核runs/m6p2c_formal_seed0_failure_audit_v1/及receipt完整，原产物不改。

v9资格自动取得原槽后running，Python992715；门禁pytest已3242 passed/47 deselected/439 warnings、1686.69秒，仍须完成后续真实恢复/48/3x8/4h，不能先宣称全部通过。watcher13488存活，主机磁盘843GB可用/12%、内存10130MiB可用、swap571MiB使用，无磁盘或内存阻碍。已更新监督五类产物和receipt及小时自动任务。正式阻碍未解决，新正式ready保持false；即使v9旧策略资格通过，也需先完成晚批主机根因分析和必要修复/新资格，再推进正式三种子。可自主诊断工作仍在进行，尚不是确认无法修复。

## 第五轮巡检（2026-10-05 21:44 Asia/Shanghai）

实际SSH快照unix1791207855.3273804：v9资格仍running，门禁、真实边界恢复、0.50诊断48回合及seed0/1/2各8批已各自成功终止；三短训各1536 transitions/128 Adam/8乘子更新。当前soak Python1154455存活，本轮核对before24/24及shared72/72合格，A降级/fallback/参数更新均0，after尚0；共享阶段刚开始，未满14400秒，不认定完整资格通过。只读失败快照probe仍queued、controller1113176，等待现有单槽，未重复提交。watcher13488/20077均存活。

原v8正式seed0仍failed，371成功批终态与封存39文件receipt及本地失败审核receipt再次核验通过，原checkpoint/RNG/partial未更改。晚批超时根因仍待已排队主机诊断，不能以目前短训/旧策略soak进展替代修复或重新启动正式训练。主机磁盘842GB可用/12%、内存12231MiB可用、swap444MiB使用，无资源阻碍；监督产物/小时记录及receipt已更新。未改源码/预算/工作量、未恢复正式训练、未新建job；当前新版ready保持false。

## 互动修复 v10：主机连接阻碍

用户要求直接修复后恢复，已中断并保留未完成v9资格，原终态failed是明确人工SIGINT、不作为自然solver失败或4h通过。原失败求解计时全0，定位为共享预算在求解前耗尽；准确历史耗时来源无法由原日志确定。修复04a4007减少每非零元tuple/GC压力并正确标记求解前阶段，52e2589先红，24项本地回归/Ruff/mypy通过；任何物理/服务、raw、优化器、.50共享预算均不变。

8389f89冻结v10，runtime-timeout-heap-repair-v1与runtime-campaign-qualification-v10各唯一提交且单槽串行；随后Tailscale peer离线/SSH超时，现有任务实际状态待连接恢复核对。新release尚未生成，正式ready=false，原v8 371批失败不改，三种子新发布freshinit尚未启动。具体证据与恢复步骤见RUNTIME_TIMEOUT_REPAIR_2026-10-05.md；小时任务只读辅助不替代互动闭环。

2026-10-06用户再次授权继续：主机当前C144GB/D97GB、D上的Ubuntu root rw/ext4错误0/服务与fsync读回通过；旧迁移length误报已更正并保留日志。新唯一probe-v3及qualification-v10-r3同源8389f89/.50单槽串行已提交，当前等待完整资格及放行；原371批失败保留，新正式三种子依冻结campaign freshinit约定，尚未启动。小时任务仅只读辅助，本互动继续执行修复验收发布流程。

2026-10-06当前对话继续：主机实际容量、根rw及Git交付修复已验收；新版v10-r4 gate3245及真实resume通过、48诊断通过，seed0短训8批/32回合/128Adam通过，A降级0/fallback0、326.7265秒；seed1执行中，seed2/完整4h/晚批212缓存验收待，不提前放行。I卡直接控制器独立等待两验收及完整回传后发布/交付/串行三seed，新正式job尚未提交。小时旧automation已不存在（update明确报not exist），按仍有效用户授权恢复idc ACTIVE hourly至当前对话01a10c70-b4e5-7f10-9b8f-9b70f01c4e53；仅辅助当前直接流程，正常安静，正式新阻碍先修复无法处理再报，旧seed-0 PAUSED未改。

最新：三seed v10短训全部通过，共96合格回合，完整4h验收流程已启动（before隔离阶段），晚批212-cache补验在其后。正式v2 release/job尚未生成；直接控制器等待验收并负责之后完整发布/交付/串行启动，不需再向用户索取已有训练授权。每seed当前短训约327—329秒；512工作量粗略外推约5.8—5.9h，不是实测长训或跨主机条件不变的因果提速证明。

2026-10-06 J卡已接续容量修复：原parent正式启动0批即D14.6GB守卫失败，全receipt验签；既有资格/release有效。37.3GB已验SHA的退役输入副本共享存储而不删任何历史路径/内容；最小完整资产6013文件/8.16GB已全验签。TRIM exit0但非稀疏VHDX未归还Windows物理空间，D仍14.6GB，新投递预留需48.96GB。当前Windows非管理员，已准备管理员备份/离线compact/全验签脚本（尚未执行），详见RUNTIME_DELIVERY_CAPACITY_2026-10-06.md。没有r2投递，没有科学重复重训或降低容量门槛。

2026-10-06 12:27恢复实训：新job runtime-formal-three-seed-v2-v10-r2/controller308291，固定revision15d65b505f0881203240be79b7b43a1f5d4d1dfe，最小asset6fa5ab...7c50ca，原release edc30e5b...666e20c不变，watcher59855自动完整回传。主机实际seed0 Python309920，已完成1/512批、192transitions、16Adam/1乘子、4/4回合通过服务/物理/库存/统一终点全部既定标准，A降级0/fallback0，批前/latest存在、无failed_batch。旧failed parent/原v8失败保留且不重用；当前三seed都从新发布fresh初始化，seed1/2由同一parent在前seed完整512/更新/质量与产物验签通过后串行启动，不独立重投。当前仅启动已核实，不宣称三seed训练完成。标准新恢复验收runs/m6p2c_delivery_capacity_acceptance_v2/及receipt。未执行管理员compact或更改科学执行源码/预算/工作量。

2026-10-06 17:22—17:34阶段验收：seed0已完成512批/98304 transitions/8192 Adam/512乘子/2048回合，全部既定服务/物理/库存/统一终点合格，A降级0/fallback0；实际训练elapsed17089.047秒（约4h45m），不宣称经济表现/收敛或三seed全部完成。源工作树revision15d65b505f0881203240be79b7b43a1f5d4d1dfe，report SHA029e244b19e30ec81c57f82ea2f171a18e0ea47267f8043e37dd284e96dd2a94，final checkpoint SHA9c3e40d494cf3e31ddc6a01e5fb86a21fdaa2b7524ddb3c9e896c28b2b7ec8ea。41文件/354930086字节独立完整回传全部验签、core artifact_verification重验、final21/523/schema/原release绑定正确，CP实际next512、Adam参数step均8192、Lagrangian512、策略参数finite、来源212、fresh init无resume。标准Mac终态审计runs/m6p2c_formal_seed0_terminal_audit_v10_v1/与receipt；该独立回传仅seed0，parent三seed job仍running，不能混成parent终态receipt。

同一parent验证seed0成功后已自动fresh启动seed1（Python386283）；最新已32/512批、512Adam/32乘子、128回合全部合格，A降级0/fallback0，无失败；seed2未启动。资源C约142GB/D211GB、根rw，无新增阻碍。用户讨论并发后明确继续原计划，保持单槽串行，不开并发实验或独投seed1/2。

2026-10-06 18:25后：v10-r2 seed1在100完整批后Aoptimal/Binfeasible停止，seed2被同一launcher阻断；完整parent69文件回传验签，原seed0成功保留。本互动已开K卡、提交原快照红回归、最小收紧内部A/B精度1e-9（物理/服务/终点/预算/验收容差不变），97回归通过。v11固定.50完整资格与五次原快照验收唯一提交，主机单槽。必须两者通过并验签才新发布/同源fresh三seed，旧CP不跨绑定恢复。详见RUNTIME_SEED1_NUMERIC_REPAIR_2026-10-06.md；小时巡检只读辅助直接控制器，不竞争修复或正式job。

2026-10-06 20:30后：v11-r2完整门禁3项legacy .05确定性比较失败，发布控制器v2已阻断；18文件回传验签。M卡固定观察定位至少M5.4f差异为B超时执行A与B最优分支，当前原生Windows存在Overwatch等额外CPU负载，历史唯一因果不能确认。未改测试/预算或科学设置，准备直接闲置恢复控制器：用户退出游戏且CPU连续60秒<=15%后，原三个测试唯一验收通过才新v11-r3完整资格；全验收成功并签字节后才正式fresh三seed。详情RUNTIME_V11_GATE_DETERMINISM_2026-10-06.md。既有K数值/L输入修复不重写，旧失败全部保留，当前正式仍未启动。

2026-10-07 05:42—05:44：r3完整资格5740文件/7766653176字节已全部回传SHA验签，严格release v3再次验证通过。完整门禁3247、三边界真实恢复、48诊断、三seed各8批/32合格回合及14400.857778秒shared1789+前后各24共1837回合全部通过；K/L/idle补验附录及5760 evidence绑定已验签。新release SHA ad084824beebbec644dfbcee8cc508de0202a323826ea40c82dbd0ae7dd1b3b6，发布提交5403f2181675031163cbc99ac0a8a706d28a0dc0。交付最小5882文件/7932659849字节asset99148cbd7c2fa79069e88dc8a7d37da2751b815c62a8565a66067bcc327ad7d0，实际D预留48734849161通过；5866同SHA链接/16新文件，worker实际copy隔离，原所有失败/历史不删。

唯一runtime-formal-three-seed-v3-v11已真实running（host controller921336，自动全回传watcher81752）；同一parent m6p2c_formal_three_seed_v3_v11 fresh串行，seed0 Python923142，seed1/2尚未创建，不能独投。实时只读标准验收runs/m6p2c_v11_formal_start_acceptance_v1/与receipt通过：seed0实际56/512批、10752 transitions、896 Adam/56乘子、224/224回合所有冻结服务/物理/库存/统一终点合格，A降级0/fallback0/无failed_batch；before/latest CP真next56、Adam所有参数step896、乘子56、finite policy、212来源/原release binding匹配。实际C138779885568/D195363700736、根rw，无资源阻碍。直接交付控制器73958已完成首批核实并success退出；后续由原host launcher串行推进，全部三个seed终态仍待，小时巡检ACTIVE。放行/启动不代表完整训练或收敛结论，validation/test封存。

2026-10-07 10:46后：v11 seed0已完整512成功并独立41文件/354851755字节全验签和final真实状态/全部2048质量重验。seed1在58完整批后origin1008/step18再次Aoptimal/Binfeasible停止，18partial/raw/logprob/CP/RNG保留、失败批0更新，seed2未创建，parent66文件全回传验签。N卡两历史输入固定对照复现当前1e-9首步mode z=8.13e-10数值见证；1e-10两例都A/Boptimal/整数残差0/矩阵目标边界逐字节不变。先红后最小收紧内部精度，98回归通过，唯一v12两失败补验及完整资格主机串行已提交，全部验收验签后才新v4同源三seed fresh。旧v10/v11所有失败与两seed0成功均独立保留，不重复科学失败批挑成功、换签或混旧结果。详见RUNTIME_SEED1_SECOND_NUMERIC_REPAIR_2026-10-07.md及新直接控制器。
