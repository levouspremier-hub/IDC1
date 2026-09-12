# 数据来源侦察（M1.2，只读）

> 目的：为计划书 §8 的“≥1 个完整年度匹配价格与负荷 + 风光/温度 + 任务轨迹”定位来源。
> 状态：**侦察结论，尚未下载/核验**。正式接入前需逐源核验可下载性、时区、单位与聚合口径。

## 需求清单（来自 `docs/EXECUTION_PLAN.md` §8 与审计 B6）

| 数据 | 目标 | 现状 |
|---|---|---|
| 系统负荷 + USEP | ≥1 全年、时间匹配 | 仅 `USEP_May-2026.csv`（1 月） |
| 风光时序 | 与地点/年份匹配 | 缺失 |
| 温度时序 | 冷却模型输入 | 缺失 |
| 任务轨迹 | 公开轨迹或统计特征生成 | 缺失 |

## 已识别的来源（待核验）

**价格 / 系统负荷（新加坡）**
- EMC NEMS 价格页：[nems.emcsg.com/nems-prices](https://www.nems.emcsg.com/nems-prices)
- EMC 数据订阅（含数据目录 PDF，列明数据类型/频率/来源）：[home.emcsg.com/services/nems-services/data-subscription](https://www.home.emcsg.com/services/nems-services/data-subscription)
- 新加坡开放数据 portal：[data.ik.sg](https://data.ik.sg)（含“Peak System Demand”等集合）

**气象（温度/太阳辐照/风速）**
- ERA5 逐小时再分析（全球、免费 API，含温度/风速/地表太阳辐射）：open-meteo / Clarigrid 的 ERA5 归档
- IEA EBC Annex 80 新加坡 TMY 天气文件：[wdc-climate.de（Singapore v1.0）](https://www.wdc-climate.de/ui/entry?acronym=WDTF_Annex80_build_sing_v1.0)
- 典型/极端天气数据集（含新加坡 TMY，20 年偏差订正）：[Nature Sci Data s41597-024-03319-8](https://www.nature.com/articles/s41597-024-03319-8)

**任务轨迹**
- 待定：公开计算任务轨迹（如 Alibaba/Google 集群 trace）或其统计特征生成（计划书允许“按其统计特征生成”）。

## 接入前必须核验（计划书 §5 规则）

1. **可下载性**：USEP 是否订阅墙；data.ik.sg 是否开放 API/CSV。
2. **时区/对齐**：价格（半小时）与负荷、气象（逐小时）的时间对齐与聚合。
3. **单位**：SGD/MWh → SGD/kWh；辐照/风速单位。
4. **年份匹配**：价格/负荷与风光/温度取同一地点、同一年份。
5. **不得用重复日期充当独立样本**；正式运行缺数据直接报错。

> 结论：价格/负荷走 EMC + data.ik.sg；气象走 ERA5（逐小时、免费）或新加坡 TMY。下一步是**实际下载并核验**，这一步需要你确认数据访问方式（是否已有 EMC 订阅/账号）。
