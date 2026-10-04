# Y9000P seed 0 吞吐实测

固定SHA：6d3afa266e8cf80d3f8f2d444e3b2a8f75c3d89f；job host-throughput-seed0-v1。
主机：Intel i9-14900HX / WSL2 / Python3.12.15。原冻结v2_r5配置与v2_r7发布，CPU、torch单线程、corrector=on、预算0.25s；GPU未使用。

调用已实现inventory_train --short --seed 0 --run-id host_throughput_seed0_v1：新初始化策略，预登记前8批，32条train轨迹×48步，共1536步，真实128次Adam更新及8次Lagrangian更新，checkpoint与日志照常写入。外层只计时原入口子进程，不插桩求解器，不更改参数/配置/日期。没有启动正式512批训练，未加载替换旧final模型。

| 项目 | 实测 |
|---|---:|
| 入口端到端墙钟 | 436.657s（7分16.7秒） |
| 内层elapsed | 422.818s |
| 首批batch timer | 51.428s |
| 后7批平均 / 中位数 | 53.031 / 53.059s |
| 后7批min / max | 52.286 / 53.697s |
| 端到端吞吐 | 3.518步/s |
| env build合计 | 120.765s |
| rollout合计 | 300.577s |
| PPO更新合计 | 1.250s |
| checkpoint/journal等内层未拆分差值 | 0.175s（非逐次checkpoint计时） |
| peak RSS | 1227788288 bytes（约1.14GiB） |

完整512批（98304步）外推：
`outside_inner + first_batch + 511*steady_mean + 512*(inner_elapsed-sum(batch_timer))/8`
= **7.549小时，约7小时33分**。按观察到的后7批min/max替换均值，情景值7.443–7.643小时；**不是置信区间或保证上限**。外层13.839s包含启动、前置校验和最终验证，未分开计时；job排队、创建环境、资产拷贝和回传不含在入口时间内。短程特有checkpoint/finalization摊销及完整训练不同写盘节奏存在估计误差。

质量结果：**12步回退（0.78125%），库存32/32、服务31/32合格**。回退逐批为0/1/1/2/1/1/2/4；第8批origin1440服务不合格，physical_violation_count=0，该轨迹fallbacks=1。不能从汇总认定回退导致服务失败，更不能认定这是历史index503突变的同一原因。原统计含建模/结果处理的规划总耗时，末批p95=0.282292s、max=0.308639s；超过0.25s不能单独证明实际求解器共享预算失效。

**吞吐测量完成，正式训练不放行**：本短程未满足零回退、全轨迹服务合格要求。仅前8批初始策略和早期日期，末期策略/日期/长期进程状态可能改变成本；未重训新版本或seed1/2，未跑validation/test。下一步若要推进训练应先诊断上述回退/服务失败，不增加预算或放松约束。

全部18个远端文件已回传验签，严格checkpoint绑定、配置、batch schedule和训练计数核对通过；策略输出是独立短程产物，保留失败轨迹与checkpoint。证据：runs/remote_host-throughput-seed0-v1/，其中runs/host_throughput_seed0_v1/为原训练五类产物、checkpoint及artifact_verification；runs/host_throughput_estimate_v1/为五类估算汇总及receipt。大产物不上传GitHub。
