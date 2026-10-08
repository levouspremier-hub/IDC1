# 修复版短跑与正式训练推进

旧v14 seed0在32批后6192/step47失败。v15整数证书并未覆盖终点微差；v16尾界实现仍被1008/+1e-9严谨A证书拦下；v17网络蕴含充电界仍在1008/+ulp被B误报不可行。各原输入、变体与失败全部保留，没有挑选通过版本覆盖失败。

最终v18追加原约束推导的H=1储能域、网络充电界/有效整数cut，并以已认证A解为B连续变量原点。原行/目标/整数域/物理业务边界保留；B恢复原坐标后重新检查原矩阵，原失败语义与共享.50预算不变，无额外求解或重试。108项回归和Ruff/mypy通过，4份历史失败×11变体共44固定观察全部A/Boptimal、整数残差0、严格原行证书通过，耗时.016286—.371440秒；这仅是本机数值验收。

用户已明确要求短跑通过后开始正式训练。唯一直接控制器先主机固定数值+seed0/1/2各8批短跑，实际Checkpoint/128Adam/8乘子/32回合/来源/全部SHA通过后，开展同源完整门禁、真实恢复、48固定诊断、每seed106批两周期学习覆盖、4h稳定性及完整回传。三个已验签短跑只在同一候选下复用，其余全部实际执行。资格全部通过才生成严格v7，最小不可变资产/真实C/D余量/根rw通过后fresh三个512批串行正式，每前seed完整真实CP与2048回合验签通过才开始下一seed。不跨版本resume，validation/test封存。

源码6119650；冻结4cdf826。主机状态及PID、固定revision、传输与启动实际证据位于runs/m6p2c_v18_direct_launch_v1/report.json。当前准备阶段不能称为正式启动。按最新运行及后续正式授权，现有小时巡检已更新到当前对话ACTIVE，只在有意义变化时通知；持久直接控制器执行条件后续，任一门禁失败停止并完整保留。

回滚使用git revert逆序撤销本卡提交，不改变paper-baseline或历史证据。

实际投递：固定revision626bef23cb4859b55f2c0c49ff024883e3e9f97e，runtime-fixed-short-v18-v1已running，主机PID1677439；Mac直控62570、全回传watcher62751。主机44观察亦全部通过，seed0短跑已开始（尚未完成）。实际C136250920960/D168219095040字节，根rw，输入468217305字节及30GiB+输入+8GiB预留通过。直控PID用进程文件及ps核验，不以历史PID当实际。

最终修复v19：红fe7a797/5016480先提交，源码2c85c1e，冻结571a4db。精确见证0额外调用；仅严格原始整数归一见证失败且整数差小于既有1e-10、原粗校验1e-6范围及共享deadline未过时，固定全部原整数模式一次highs-ds连续LP、保留原完整行/边界/阶段目标，返回再严格1e-10原矩阵验签。LP infeasible/time-limit或超预算仍拒绝，不重试MIP；A原下界/主目标证书保持。调用选项/数学输入SHA与真实LP观察计数进入资格核验。112去重相关回归（111组合+新增失败不重试1）通过、64 wiring及30 polish独立重验通过、Ruff/mypy通过；5份历史失败×11变体55观察全部A/Boptimal、整数0、原行证书通过/.01625775—.36682488秒，仅5例各1次LP。v18真实before/latest CPnext1/全Adam16/lag1/finite/原候选binding及注册origins均验证。

新的唯一流程为runs/m6p2c_v19_direct_launch_v1/controller.py，主机runtime-fixed-short-v19-v1→runtime-campaign-qualification-v19→严格v7/fresh runtime-formal-three-seed-v7-v19；旧v18 controller与watcher均已终态退出，不重启，不继承其short或资格，v7仍未生成。主机运行状态只能以新直控和实际job为准。
