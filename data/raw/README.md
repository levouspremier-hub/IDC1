# Singapore-2024 原始场景数据（M1.2 冻结）

本目录冻结 M1.2 的原始输入；所有大型数据文件均被 `.gitignore` 忽略，**不得**
提交、镜像或重新分发。Git 只跟踪
[`data/manifest/singapore_2024.json`](../manifest/singapore_2024.json) 中的来源、
用途、时间轴和 SHA-256。

## 1. 冻结范围

选择 2024（闰年）作为唯一的首版真实场景年。所有时间戳均为
`Asia/Singapore`（UTC+8）。核验结果：

| 原始文件 | 原始字段与语义 | 粒度 / 完整性 |
|---|---|---|
| `emc_usep_2024.zip` | 最终 USEP；`DEMAND (MW)` 只是**需求预测**，不得作为实际系统负荷 | 17,568 半小时点 |
| `sasea_demand_2024.zip` | `raw_SGP_demand.csv` 的实际系统负荷（MW） | 17,568 半小时点 |
| `emc_metered_generation_2024.zip` | NEMS IGS 的实际净注入（MWh/半小时） | 17,568 半小时点；不是太阳能专属、不是 IDC 本地 PV 实测 |
| `open_meteo_era5_2024.csv` | ERA5 温度、10m 风速、GHI | 8,784 小时点；M1.2 不重采样 |

任何重复、缺口、无穷/非数值或错误时区都会使本地核验失败；本卡**没有插补**。
天气字段为“前一小时平均”的原始 GHI，后续读入器若要映射到半小时，必须另开卡、
固定规则并记录该规则，不能静默复制或前向填充。

### 1.1 冻结后不可变与只读

四个 raw 文件**已冻结**并完成 SHA-256 核验；raw 文件**不入 Git**（仅 manifest 入库）。
**冻结后路径只读**：不得重新下载、不得覆盖、不得改动已冻结的 hash、时间戳或来源。

### 1.2 语义隔离（不得扩大解释）

- `emc_metered_generation_2024.zip` 是 **national intermittent generation**，
  **不是** IDC 本地 PV，也**不是**太阳能专属。
- ERA5 的 10m 风速是**国家级网格再分析**，**不是**本地风电实测。

### 1.3 尚未冻结的项：是「未验证/未冻结」，不是「已证明不存在」

半小时碳强度、IDC 本地 PV 实测、IDC 本地风电实测**尚未冻结**。对这些项，
本卡的结论是「**尚未验证到满足要求的数据组合**」，**不得**表述为
「已证明不存在」或「无数据源」——两者是不同的认识论状态，混用会误导下游决策。
它们**不得**由默认曲线、常数或重复日伪造。
M1.2b 的 manifest 是不可变的：已存在时，`--verify` 会把四个本机原始文件的
SHA-256、字节数、年份、时区、行数和“拒绝缺失、无插补”策略逐项对照；任何差异都
会失败。相同内容的 `--write-manifest` 仅作只读幂等验证，绝不改写文件。

## 2. 来源与权利

| 来源 | 用途 | 权利 / 使用限制 |
|---|---|---|
| [EMC NEMS Prices](https://www.nems.emcsg.com/nems-prices) | USEP 年度 ZIP、Metered Generation by Facility Type 年度 ZIP | [EMC 条款](https://www.home.emcsg.com/terms-and-conditions)：仅个人/非商业使用；不可重新分发；禁止自动化系统性收集。研究者须用网页“By Year → 2024”手工下载，脚本不会访问 EMC。 |
| [SASEA / Zenodo](https://zenodo.org/records/17175212) | `data.zip` 内的 `data/raw/raw_SGP_demand.csv` | CC BY 4.0；原始文件和 Zenodo 包的 hash 均写入 manifest。 |
| [Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api) | ERA5 历史温度、GHI、风速 | CC BY 4.0；免费接口仅限非商业用途。查询参数及实际返回网格坐标冻结在 manifest。 |

本卡没有冻结可公开追溯的**半小时新加坡碳强度**、IDC 本地 PV 实测或 IDC 本地
风电实测。它们不得由默认曲线或常数伪造；若下游需要，必须有单独的数据口径卡。

Open-Meteo 查询坐标 `1.3521, 103.8198` 与实际返回 ERA5 网格坐标是不同概念，二者均
记录在 manifest。这是新加坡**国家级场景的 ERA5 网格替代**，不是 IDC 站点观测，也
不能据此宣称存在本地 PV 或风电发电量实测。每个来源只记录可证明的本地取得/冻结
时刻；上游下载时刻无法证明时明确标为 `unknown`，不作推断。

## 3. 本地获取和核验

将两个从 EMC 网页手工取得的 2024 年度 ZIP、Zenodo 的 `data.zip` 放在任意本地路径，
然后运行：

```bash
uv run python scripts/fetch_singapore_data.py \
  --usep-zip /path/to/USEP_from_01-Jan-2024_to_31-Dec-2024.zip \
  --generation-zip /path/to/MG_from_01-Jan-2024_to_31-Dec-2024.zip \
  --sasea-zip /path/to/data.zip \
  --raw-dir data/raw/singapore_2024 \
  --fetch-weather \
  --verify \
  --write-manifest \
  --frozen-at-utc '2026-09-14T07:28:44+00:00'
```

`--fetch-weather` 仅发起一次明确的 Open-Meteo 非商业查询；EMC 从不由脚本下载。
`--write-manifest` 必须显式提供冻结时刻；已有 manifest 不会被此参数覆盖，而是只有
逐字段完全相同时才会通过幂等验证。
成功时 stdout 是机器可读 JSON；原始文件始终留在忽略目录。

## 4. 下游边界

M1.2 只负责原始数据和可审计冻结，尚不实现 `ScenarioBundle` 的正式数据读取。
后续正式读取卡必须：校验 manifest SHA、将 USEP 除以 1000、以 SASEA 负荷作为唯一
实际系统负荷来源，并显式解决小时到半小时、GHI/IGS 到 IDC PV、风速到风电以及碳
因子的建模口径；不得退回合成数据。

具体而言，M1.2 只保存 USEP 原始单位 `SGD/MWh`；它声明 reader 所需目标单位
`SGD/kWh` 与比例 `0.001`，但没有生成任何转换后的价格序列。
