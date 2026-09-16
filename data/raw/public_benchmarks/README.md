# `data/raw/public_benchmarks/` —— M1.3f-b 公开基准来源

本目录存放 **M1.3f-b** 冻结的公开来源原始文件。它们是**模型/方法来源**，
**不是** 2024 年新加坡 IDC 的真实观测；两者的关系见下表与
`docs/task_cards/M1.3f.md`（§B 任务卡）。

- 目录内的**二进制/文本原始文件**（`*.csv`、`*.py`、`*.txt`）**不进入 Git**
  （`.gitignore` 默认忽略 `data/raw/` 下非 `*.md` 文件）；
- 入库的是**本 README** 与 `data/manifest/m13f_public_sources.json`；
- 复现：`uv run python scripts/fetch_m13f_public_sources.py --fetch`（显式联网）；
  校验：`uv run python scripts/fetch_m13f_public_sources.py --verify`（**不联网**）。

## 大小上限

单源上限 **16 MiB**（`16,777,216 B`），锚点为既有 M1.2 raw 资产的最大值
`14.47 MiB`（`sasea_demand_2024.zip`）向上取整。超限来源**一律拒绝下载**。

## 已冻结（`status=frozen`）

| # | 文件 | 来源（官方 URL） | 许可 | 字节数 | SHA-256 |
|---|---|---|---|---|---|
| B | `pvlib_v0.15.2_LICENSE.txt` | `https://raw.githubusercontent.com/pvlib/pvlib-python/v0.15.2/LICENSE` | BSD-3-Clause | 1,622 | `a02e12ddaada3cf0d5dbdd8affdd577c2eec640758a95842a304f7513b8c0be4` |
| B | `pvlib_v0.15.2_pvsystem.py` | `https://raw.githubusercontent.com/pvlib/pvlib-python/v0.15.2/pvlib/pvsystem.py` | BSD-3-Clause | 117,911 | `668afd274e69dd4741854f643640fc5d9052b86feba1b94da7ecebbaccfef8f3` |
| C | `windpowerlib_v0.2.2_LICENSE.txt` | `https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/LICENSE` | MIT | 1,083 | `140f742e061d4c8e4c8a2bb45516d1d38a0e595a7d7ab6bad02d89dd9dd93f6a` |
| C | `windpowerlib_v0.2.2_power_curves.csv` | `https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/windpowerlib/data/default_turbine_data/power_curves.csv` | MIT | 26,042 | `7d91ddde701ce6d0ac4cacb31fac04b38f0664921b75ca44c174ca26cd394add` |
| C | `windpowerlib_v0.2.2_turbine_data.csv` | `https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/windpowerlib/data/default_turbine_data/turbine_data.csv` | MIT | 22,922 | `d379379ba20fe41ec151f74ad5ddf368f9e1cce596756ccea73888d0614c3840` |

两个仓库都以**不可变 tag** 固定（`v0.15.2` / `v0.2.2`），因此 URL 可长期复现。
`pvsystem.py` 含 `pvwatts_dc` / `pvwatts_ac`（PVWatts V8 的模块与逆变器模型）。

## 未冻结（**必须按原因分别对待**）

### A. `carbon_intensity` —— `status=blocked`

| 探测 | 结果 |
|---|---|
| `https://www.ema.gov.sg/resources/singapore-energy-statistics` | HTTP 200，但内容是 **Incapsula（Imperva）反爬挑战页**（`_Incapsula_Resource`，212–848 B），**不是真实内容** |
| `https://www.ema.gov.sg/content/dam/ema/resources/singapore-energy-statistics/*.xlsx` | 同上，返回挑战页而非 xlsx |
| `data.gov.sg` EMA 数据集（`managedBy: Energy Market Authority`，id `d_3de362b580b2dd2fd50cc1006d4edd4f`） | **可达**且许可明确（Singapore Open Data Licence），但 `coverageStart=2005-01-01`、**`coverageEnd=2020-12-31`**、`lastUpdatedAt=2024-06-06` |

**结论**：**2024 年度的 grid emission factor 无法从本环境可达的官方机读来源核验。**
因此本卡**没有**冻结任何 2024 碳强度文件，**也没有**写死 `0.402 kg CO2/kWh`。
按红线，年度碳因子只能作为 `external_low_resolution`，且**不得**冒充半小时真值。

### D. `arrival` —— `status=refused_over_size_cap`

| 探测 | 结果 |
|---|---|
| 官方发布包 `azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz` | HTTP 200，**`Content-Length = 142,968,140 B`（≈136.3 MiB）** ≫ 16 MiB 上限 |
| 许可 | **CC-BY**（`Azure/AzurePublicDataset` 官方说明，`revision 2, 20200618`） |
| 容器列举 `?restype=container&comp=list` | **`PublicAccessNotPermitted`** |
| 逐文件 blob 路径（如 `invocations_per_function_md.anon.d01.csv`） | **HTTP 409**（不可公开寻址） |

**结论**：按大小上限政策**拒绝盲目下载**；官方**不存在**可用的单文件端点。
本卡**只登记元数据**（URL、许可、字节数、发布版本），未下载任何 trace 文件。

## 四条红线（已固化为可测试守卫）

1. **2026 Solar Generation Profile 不得冒充 2024 真值** → `assert_year_matches()`
2. **年度碳因子不得冒充半小时碳强度真值** → `assert_resolution_matches()`
3. **未批准参数不得生成正式 PV / 风电**（`pv_capacity_kw=500` 不得继承）
   → `assert_parameters_approved()`
4. **Azure 2019 trace 不得被静默重放成 2024 arrival** → `assert_no_silent_replay()`

## 仍然 `UNAPPROVED` 的参数

| 路线 | 参数 | 状态 |
|---|---|---|
| B `local_pv_kw` | `pv_capacity_kw`、`tilt_deg`、`azimuth_deg`、`array_type`、`losses_pct` | **UNAPPROVED** |
| C `wind_generation_kw` | `hub_height_m`、`shear_exponent`、`turbine_model`、`rated_capacity_kw` | **UNAPPROVED** |

## readiness

```text
public_source_frozen              = true    （B/C 已冻结；A/D 已登记）
local_pv_kw_ready                 = false
wind_generation_kw_ready          = false
carbon_intensity_ready            = false
arrival_ready                     = false
formal_scenario_bundle_ready      = false
formal_training_ready             = false
```

**本目录不产生任何正式 `local_pv_kw` / `wind_generation_kw` / `carbon_intensity` /
`arrival` 序列**，也不解除任何训练门禁。
