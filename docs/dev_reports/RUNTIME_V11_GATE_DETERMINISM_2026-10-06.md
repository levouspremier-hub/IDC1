# v11资格门禁确定性阻碍

runtime-campaign-qualification-v11-r2已failed，完整18文件/614321字节回传验签。gate在2456.97秒后3 failed/3244 passed/47 deselected；失败为M5.1c两个执行payload/动作逐项一致测试及M5.4f原0.05秒三次digest一致测试。旧直接交付控制器v2已正确阻断，无正式release v3或新512任务生成；源/候选仍固定3b1033fa5ecfdc634ea2063f27adc655a957657a，不修改测试/预算来通过。

M卡d8ba2f4/087a72c先登记固定观察，唯一runtime-v11-gate-determinism-probe-v1已全回传验签。默认Torch16和正式CPU1两组各三次原snapshot .05全部Aoptimal/Btime_limit/执行A，其digest5526e47d…与原门禁失败其中一个完全相同；既有宽裕2.0仅诊断的三次均A/Boptimal、digest4d651fcc…与原另一digest完全相同。说明至少该失败来自真实时间分支，不能把两种行为差异当成纯墙钟字段忽略。两组各两个3步arm当前指纹一致（fed6cf…），未复现原8a1bc9…，因此不以重放成功覆盖原失败，也不证明另外两项唯一原因。原3失败没有逐步solver options/状态快照；新probe保留全部snapshot/proposal/raw/logprob/动作/状态/阶段时间，但未额外插桩记录每次SciPy options，相关缺口明确保留，不伪造计时。

当前Windows原生主机额外负载：Overwatch 2秒增加7.015625 CPU秒、NahimicSvc64增加2.046875 CPU秒，CPU utility36%；平衡电源GUID381b4222-f694-41f0-9685-ff5bb260df2e、AC电源/电量96%。存在额外CPU竞争，但原失败时没有Windows原生遥测，不能认定游戏是历史唯一因果。不停止用户游戏/Windows服务，不改主机全局电源或CPU配置。主机C142409121792/D202789732352字节、rw、MemAvailable14018100kB，容量/内存正常。

已准备直接恢复控制器runs/m6p2c_v11_idle_restore_v1/controller.py：只读等待Overwatch关闭且原生CPU utility连续60秒均<=15%，随后唯一runtime-v11-gate-idle-acceptance-v1执行原三个失败测试（仍.05/完全同一比较，不增加预算或重试），完整回传验签通过才提交新唯一runtime-campaign-qualification-v11-r3，所有原全门禁/真实resume/48/3x8/4h阶段完整执行。任一失败立即终止，不自动反复寻找成功。全r3资格与已有K数值补验、L资格输入预检和本次idle原测试验收都通过/验签后，独立runs/m6p2c_v11_release_delivery_v3/controller.py才严格发布v3并启动原已授权fresh三seed串行。恢复/发布控制器及嵌入主机代码compile通过，现有host launcher内容不变；只增加外部资源前置条件及补验证据，不修改执行源码/候选。任务仍未完成，小时巡检辅助直接恢复流程。

当前需要用户结束主机上的游戏、留出空闲主机；主机外部负载未解除时仅等待，不重复投递资格或通知同一阻碍。外部条件满足后的验收和恢复已有授权，将由上述控制器自动执行，无需再次索取启动许可。所有旧失败、旧seed0成功、原seed1 partial/raw/RNG/更新和seal输入均保留，validation/test仍封存。
