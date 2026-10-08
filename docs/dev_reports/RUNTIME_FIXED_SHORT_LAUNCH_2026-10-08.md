# 修复版短跑与正式训练推进

旧v14 seed0在32批后6192/step47失败。v15整数证书并未覆盖终点微差；v16尾界实现仍被1008/+1e-9严谨A证书拦下；v17网络蕴含充电界仍在1008/+ulp被B误报不可行。各原输入、变体与失败全部保留，没有挑选通过版本覆盖失败。

最终v18追加原约束推导的H=1储能域、网络充电界/有效整数cut，并以已认证A解为B连续变量原点。原行/目标/整数域/物理业务边界保留；B恢复原坐标后重新检查原矩阵，原失败语义与共享.50预算不变，无额外求解或重试。108项回归和Ruff/mypy通过，4份历史失败×11变体共44固定观察全部A/Boptimal、整数残差0、严格原行证书通过，耗时.016286—.371440秒；这仅是本机数值验收。

用户已明确要求短跑通过后开始正式训练。唯一直接控制器先主机固定数值+seed0/1/2各8批短跑，实际Checkpoint/128Adam/8乘子/32回合/来源/全部SHA通过后，开展同源完整门禁、真实恢复、48固定诊断、每seed106批两周期学习覆盖、4h稳定性及完整回传。三个已验签短跑只在同一候选下复用，其余全部实际执行。资格全部通过才生成严格v7，最小不可变资产/真实C/D余量/根rw通过后fresh三个512批串行正式，每前seed完整真实CP与2048回合验签通过才开始下一seed。不跨版本resume，validation/test封存。

源码6119650；冻结4cdf826。主机状态及PID、固定revision、传输与启动实际证据位于runs/m6p2c_v18_direct_launch_v1/report.json。当前准备阶段不能称为正式启动。按最新运行及后续正式授权，现有小时巡检已更新到当前对话ACTIVE，只在有意义变化时通知；持久直接控制器执行条件后续，任一门禁失败停止并完整保留。

回滚使用git revert逆序撤销本卡提交，不改变paper-baseline或历史证据。

实际投递：固定revision626bef23cb4859b55f2c0c49ff024883e3e9f97e，runtime-fixed-short-v18-v1已running，主机PID1677439；Mac直控62570、全回传watcher62751。主机44观察亦全部通过，seed0短跑已开始（尚未完成）。实际C136250920960/D168219095040字节，根rw，输入468217305字节及30GiB+输入+8GiB预留通过。直控PID用进程文件及ps核验，不以历史PID当实际。

最终修复v19：红fe7a797/5016480先提交，源码2c85c1e，冻结571a4db。精确见证0额外调用；仅严格原始整数归一见证失败且整数差小于既有1e-10、原粗校验1e-6范围及共享deadline未过时，固定全部原整数模式一次highs-ds连续LP、保留原完整行/边界/阶段目标，返回再严格1e-10原矩阵验签。LP infeasible/time-limit或超预算仍拒绝，不重试MIP；A原下界/主目标证书保持。调用选项/数学输入SHA与真实LP观察计数进入资格核验。112去重相关回归（111组合+新增失败不重试1）通过、64 wiring及30 polish独立重验通过、Ruff/mypy通过；5份历史失败×11变体55观察全部A/Boptimal、整数0、原行证书通过/.01625775—.36682488秒，仅5例各1次LP。v18真实before/latest CPnext1/全Adam16/lag1/finite/原候选binding及注册origins均验证。

新的唯一流程为runs/m6p2c_v19_direct_launch_v1/controller.py，主机runtime-fixed-short-v19-v1→runtime-campaign-qualification-v19→严格v7/fresh runtime-formal-three-seed-v7-v19；旧v18 controller与watcher均已终态退出，不重启，不继承其short或资格，v7仍未生成。主机运行状态只能以新直控和实际job为准。

主机实际v19运行与首批审核：固定dc0a823310c6e25ba64c09b2b24c119221e0af32；job runtime-fixed-short-v19-v1 running，host1681646、Mac直控66405/watcher66598。主机55数值观察通过；seed0已完成2批，实际latest CPnext2/所有Adam32/lag2/384transitions/8合格回合/原候选binding/注册origins/finite审计通过，当前无failed_batch；已越过v18失败位置。审核时短跑未完成、v7未生成、正式未启动。证据runs/m6p2c_v19_short_first_batch_audit_v1/和runs/m6p2c_v19_direct_launch_v1/；后续以持续增长的实际状态为准。当前代码/候选/文档均显式提交推送、git status干净，活动科学任务仅一个；已授权后续由持久直控和当前对话小时巡检继续，不用用户再次确认。

最终v20：红673499c，等价原能量整数cover b9fb88d，冻结11ea6e0。114相关回归/Ruff/mypy通过，6原输入×11变体66观察全部A/Boptimal、完全整数及原矩阵严格证书通过/.01607429—.31835075秒，仅5案例一次连续LP；未改原行/目标/预算或任何容差。v19完整26文件及真实before/latestCPnext2/所有Adam32/lag2/finite/原binding已验证，93partial与失败0更新0env保留，旧直控/watchers终态退出。新唯一短跑runtime-fixed-short-v20-v1→同源runtime-campaign-qualification-v20→严格v7/fresh runtime-formal-three-seed-v7-v20，实际状态以runs/m6p2c_v20_direct_launch_v1/report.json与进程文件为准。仍未正式启动、v7未生成，不把新本机验收当主机成功。

v20第一次投递仅Git fetch setup失败（Recv failure: Connection reset by peer），0数值观察/0环境/0PPO；全部7文件native SHA保全。原Mac SSH reverse relay34510已消失、WSL19065无监听；只恢复本任务已有127.0.0.1:19065反向转发（不改Windows/全局proxy或防火墙），恢复后同裸仓库Git ls-remote验证通过。采用新唯一job runtime-fixed-short-v20-v1-r2，源码/候选不改、不覆盖旧ID，pipeline新直控runs/m6p2c_v20_direct_launch_r2_v1/controller.py；qualification/formal仍原计划唯一v20（尚未存在），未进入科学执行的网络故障不算重试科学失败。

最终v21：红e9618b8/53c9e03，源e9ae630，冻结53caef0。117相关回归/Ruff/mypy通过；7原输入×11变体77观察全部A/Boptimal、整数0、原矩阵严格证书及原主目标证书通过/.01598221—.29003271秒，10例各一次LP。A保持原MIP主目标下界/offset证书；B LP抛光后额外验证原MIP下界与incumbent，不能劣化原返回incumbent gap（仍1e-10数值精度），资格必须重开该证书，避免将任意可行mode冒充经济最优。无精度/预算/原目标或物理放宽。v20-r2全部36文件native SHA及实际seed0 final next8/所有Adam128/lag8、seed1 latest next2/所有Adam32/lag2/原候选binding/finite已核验，历史short成功不跨source继承。新唯一流程runs/m6p2c_v21_direct_launch_v1/controller.py：runtime-fixed-short-v21-v1→runtime-campaign-qualification-v21→严格v7/fresh runtime-formal-three-seed-v7-v21；当前主机短跑和formal尚未启动，v7未生成，以后续实际状态为准。既有本任务relay在每次submit前核验，断开仅恢复本任务loopback，不改全局网络。

实际v21主机77观察及三seed短跑全部通过、全部47文件native SHA；Mac重开真实CP/128Adam/8乘子/32回合/角色/来源/RNG/注册origin/fresh/0fallback全部验证，审计runs/m6p2c_v21_three_short_terminal_audit_v1/。原完整资格runtime-campaign-qualification-v21在make check Ruff阶段因本卡新tests文件7处import分组缺空行失败，0其余准入观察，17文件native SHA保全。本地之前只Ruff改动源码漏tests，承认检查范围不全；现仅增加空行，AST逐字等价、测试断言/源码/候选/原short全部保持，Makefile相同scope Ruff通过、mypy201sourcefiles通过、33相关测试通过。不得修受保护旧shim/遗留代码来处理本卡格式。新唯一runtime-campaign-qualification-v21-r2 / runtime_campaign_qualification_v21_r2，从全gate重新实际执行全部准入，重开同一候选short家族/47receipt及已sealed输入验签，复用已过short而不重做、不伪造旧gate通过。源e9ae630/候选53caef0不变，不重新冻结；新测试格式commit只影响门禁。继续直控runs/m6p2c_v21_qualification_resume_v1/controller.py，v7及runtime-formal-three-seed-v7-v21仍未生成/启动，严格条件满足后自动推进。

当前实际状态：三seed短跑全部完整通过（每seed8批/1536transitions/128Adam/8乘子/32合格、0fallback、fresh、真CP/来源/角色/RNG/排序），47文件native SHA、独立terminal audit全部通过。修复测试import格式后，仅tests及文档revision c8bfd7efcdaf7f53ca428da97cbb1a13a5930bbc；short固定c6b7c8cc76ecbe34933fc6edb2f0df6ab23a5c00，执行源e9ae630和v21候选全部绑定保持同一。新完整资格runtime-campaign-qualification-v21-r2 / runtime_campaign_qualification_v21_r2已running，host1733017、Mac直控78919/watcher79860（查实际进程文件及ps）；重新验签原short47文件/canonical家族/真实CP和已有sealed输入a8f708276632fabf33e58edc4a5943cdcf3dc4c0ef1e449de7781880caf4beed，未重跑short。全部gate/resume/77/48/3×106/4h实际通过、完整回传后仍自动严格v7/fresh三个seed串行formal；当前formal未启动、v7未生成，主任务继续，不以short通过宣称完整准入/正式完成。Git源/候选/测试/文档已显式提交，工作树干净；以下长期进度只由当前直控及当前对话巡检跟进，不竞争collect或重跑科学任务。
