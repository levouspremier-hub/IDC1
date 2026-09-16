"""M1.3f-c：四类正式外生驱动（`local_pv_kw` / `wind_generation_kw` /
`carbon_intensity` / `arrival`）的**纯计算**。

本模块只做确定性/可复现的数值计算与冻结输入读取，**不**写文件、**不**构造
`ScenarioBundle`、**不**接入训练。物化由
`scripts/materialize_singapore_exogenous.py` 负责。

## 口径（全部为 `modeled_scenario`，**不是**现场实测）

- **`local_pv_kw`**：pvlib v0.15.2 的确定性链条
  `ERA5 GHI → solar position → erbs 分解(DNI/DHI) → isotropic 透射(POA) →
  sapm_cell 电池温度 → pvwatts_dc → pvwatts_ac`，
  参数取 B4 批准值（见 `PV_PARAMS`）。
- **`wind_generation_kw`**：ERA5 10 m 风速按切变律外推到 60 m，
  再按**冻结的** windpowerlib v0.2.2 `E48/800` 功率曲线做确定性线性插值。
- **`carbon_intensity`**：全 2024 为 **0.402 kgCO2/kWh** 年内常数（B1 人工批准）。
- **`arrival`**：由 Azure Functions 2019 trace 校准的 **48-slot
  day-of-benchmark-period** rate template（**date-free**：不使用、也不声称
  archive 提供日期/星期/时区），按固定 seed 用 Poisson **前向生成**；
  2019 trace 只校准**一个 24 小时周期内的分布形状**，**不**重放、
  **不**改称 2024 真实到达。

## 无未来泄漏

`arrival` 的生成只依赖**冻结 template**、`timestamp` 的星期与时刻、以及固定 seed；
**不读取**任何 2024 truth。`local_pv_kw` / `wind_generation_kw` 只用**同 timestamp**
的已冻结 GHI / 温度 / 风速。
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pvlib import inverter, irradiance, pvsystem, solarposition, temperature

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- 冻结参数（B4 人工批准；不得在本模块内自行改动） -------------------------

PV_PARAMS: dict[str, Any] = {
    "latitude": 1.5,
    "longitude": 103.75,
    "timezone": "Asia/Singapore",
    "pv_capacity_kw": 500.0,
    "tilt_deg": 10.0,
    "azimuth_deg": 180.0,
    "array_type": "fixed_open_rack",
    "losses_pct": 14.0,
    "gamma_pdc_per_deg_c": -0.004,
    "dc_ac_ratio": 1.2,
    "eta_inv_nom": 0.96,
    "temperature_model": "open_rack_glass_polymer",
}
# **B5-PV 人工批准**（2026-09-16）：PV 链的两处模型选择与地面反照率
B5_PV_APPROVAL: dict[str, Any] = {
    "decision_id": "B5-PV",
    "approved_on": "2026-09-16",
    "decomposition_model": "pvlib.irradiance.erbs",
    "transposition_model": "isotropic",
    "albedo": 0.25,
    "classification": "modeled_scenario",
    "note": "**modeled**；不是 IDC 本地 PV 现场实测",
}

WIND_PARAMS: dict[str, Any] = {
    "turbine_model": "E48/800",
    "hub_height_m": 60.0,
    "shear_exponent": 1.0 / 7.0,
    "rated_capacity_kw": 800.0,
    "reference_height_m": 10.0,
}
# 冻结功率曲线的来源（M1.3f-b 已冻结文件）
WIND_CURVE_FILE = "windpowerlib_v0.2.2_power_curves.csv"
WIND_CURVE_SHA256 = (
    "7d91ddde701ce6d0ac4cacb31fac04b38f0664921b75ca44c174ca26cd394add"
)
WIND_CURVE_COLUMN = "turbine_type"
WIND_CURVE_UNIT = "W"          # 冻结曲线以 W 给出
WIND_CURVE_W_TO_KW = 0.001

CARBON_KG_PER_KWH = 0.402
CARBON_CLASSIFICATION = "human_approved_external_low_resolution"
CARBON_RESOLUTION = "annual_constant"
CARBON_SOURCE_URL = "https://www.ema.gov.sg/resources/singapore-energy-statistics"

ARRIVAL_SEED = 20240916
ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR = 1000.0
ARRIVAL_CLASSIFICATION = "modeled_scenario_calibrated_from_benchmark_trace"
ARRIVAL_UNIT = "work-units/step"
# 依据：已批准的 benchmark 尺度 lambda_ref = 2000 work-units/hour × 0.5 hour
ARRIVAL_SCALE_BASIS = "lambda_ref=2000 work-units/hour × 0.5 hour"

PV_UNIT = "kW"
WIND_UNIT = "kW"
CARBON_UNIT = "kgCO2/kWh"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
EXOGENOUS_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v2.json"
PUBLIC_BENCHMARKS_DIR = REPO_ROOT / "data/raw/public_benchmarks"
ARRIVAL_ARCHIVE = PUBLIC_BENCHMARKS_DIR / "azurefunctions_dataset2019.tar.xz"


class ExogenousDriverError(ValueError):
    """四类外生驱动的**明确失败**。"""


def _as_array(value: Any) -> np.ndarray:
    """pvlib 在不同输入下会返回 `DataFrame` 或 `ndarray`；统一成数组。"""
    if isinstance(value, np.ndarray):
        return value.astype(float)
    return np.asarray(value.to_numpy(dtype=float), dtype=float)


# --- 冻结输入 ----------------------------------------------------------------

@dataclass(frozen=True)
class FrozenInputs:
    """已通过 M1.3d 严格链条校验的 canonical 表（只读）。"""

    frame: pd.DataFrame
    canonical_parquet_sha256: str


def load_frozen_inputs(
    *,
    canonical_parquet_path: Path | str = CANONICAL_PARQUET,
    canonical_manifest_path: Path | str = CANONICAL_MANIFEST,
    split_manifest_path: Path | str = SPLIT_MANIFEST,
) -> FrozenInputs:
    """读取 canonical 表，并走 **M1.3d 的完整严格校验**（不复制宽松校验器）。"""
    from scenario.splits import load_truth_split

    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    # 复用既有 reader 触发完整校验（manifest 键集合 / 时间轴 / train 统计）
    load_truth_split(
        "train",
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    frame = pd.read_parquet(canonical_parquet_path)
    if len(frame) == 0:
        raise ExogenousDriverError("canonical 表为空")
    actual = hashlib.sha256(canonical_parquet_path.read_bytes()).hexdigest()
    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if actual != split_payload["canonical_parquet_sha256"]:
        raise ExogenousDriverError(
            "canonical parquet SHA-256 与 split manifest 不符："
            f"split={split_payload['canonical_parquet_sha256']} 实际={actual}"
        )
    return FrozenInputs(frame=frame, canonical_parquet_sha256=actual)


# --- PV：pvlib 确定性链条 ----------------------------------------------------

def pv_ac_limit_kw(
    pv_capacity_kw: float = PV_PARAMS["pv_capacity_kw"],
    dc_ac_ratio: float = PV_PARAMS["dc_ac_ratio"],
) -> float:
    """由批准的 DC/AC ratio 决定的逆变器 AC 上限（`pv_capacity_kw / dc_ac_ratio`）。"""
    if pv_capacity_kw <= 0 or dc_ac_ratio <= 0:
        raise ExogenousDriverError("pv_capacity_kw 与 dc_ac_ratio 必须为正")
    return float(pv_capacity_kw) / float(dc_ac_ratio)


def pv_loss_multiplier(losses_pct: float) -> float:
    """DC 侧**一次性**损耗乘子：`1 - losses_pct / 100`（只应用一次）。"""
    if isinstance(losses_pct, bool) or not isinstance(losses_pct, (int, float)):
        raise ExogenousDriverError(f"losses_pct 必须是数值，实际 {losses_pct!r}")
    value = float(losses_pct)
    if not (0.0 <= value < 100.0):
        raise ExogenousDriverError(f"losses_pct 必须落在 [0, 100)，实际 {value!r}")
    return 1.0 - value / 100.0


def assert_pv_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """PV 参数块必须**逐字段等于**冻结的 `PV_PARAMS`（缺失或篡改一律拒绝）。"""
    if params is None:
        return dict(PV_PARAMS)
    if not isinstance(params, dict):
        raise ExogenousDriverError(f"PV 参数块必须是 object，实际 {type(params).__name__}")
    if set(params) != set(PV_PARAMS):
        raise ExogenousDriverError(
            "PV 参数块键集合必须精确等于冻结参数；"
            f"多出={sorted(set(params) - set(PV_PARAMS))} "
            f"缺少={sorted(set(PV_PARAMS) - set(params))}"
        )
    for name, expected in PV_PARAMS.items():
        if params[name] != expected:
            raise ExogenousDriverError(
                f"PV 参数 {name} 被篡改：期望 {expected!r}，实际 {params[name]!r}"
            )
    return dict(params)


def pv_dc_before_losses(
    times: pd.DatetimeIndex,
    ghi_w_per_m2: Sequence[float] | np.ndarray,
    temp_air_deg_c: Sequence[float] | np.ndarray,
    wind_speed_10m_mps: Sequence[float] | np.ndarray,
    *,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """pvlib 链条到 **DC**（`pvwatts_dc`）为止，**尚未应用损耗**。"""
    return _pv_dc_stage(times, ghi_w_per_m2, temp_air_deg_c, wind_speed_10m_mps, params)


def pv_ac_from_dc(pdc_before_losses: np.ndarray, *,
                  params: dict[str, Any] | None = None,
                  ghi_w_per_m2: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    """DC → （一次性损耗）→ `inverter.pvwatts` → 双重上限 → kW。"""
    p = assert_pv_params(params)
    pdc = np.clip(np.asarray(pdc_before_losses, dtype=float), 0.0, None)
    pdc = pdc * pv_loss_multiplier(p["losses_pct"])
    ac = inverter.pvwatts(pdc, p["pv_capacity_kw"], eta_inv_nom=p["eta_inv_nom"])
    ac = np.clip(np.nan_to_num(np.asarray(ac, dtype=float), nan=0.0), 0.0, None)
    if ghi_w_per_m2 is not None:
        ac = np.where(np.asarray(ghi_w_per_m2, dtype=float) <= 0.0, 0.0, ac)
    limit = min(pv_ac_limit_kw(p["pv_capacity_kw"], p["dc_ac_ratio"]),
                p["pv_capacity_kw"] * p["eta_inv_nom"])
    result = np.clip(ac, 0.0, limit)
    if not np.isfinite(result).all():
        raise ExogenousDriverError("local_pv_kw 含非有限值")
    return result


def local_pv_kw(
    times: pd.DatetimeIndex,
    ghi_w_per_m2: Sequence[float] | np.ndarray,
    temp_air_deg_c: Sequence[float] | np.ndarray,
    wind_speed_10m_mps: Sequence[float] | np.ndarray,
    *,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """按冻结链条计算 `local_pv_kw`（kW）。

    链条：solar position → `erbs` 分解 → `isotropic` 透射 →
    `sapm_cell`（`open_rack_glass_polymer`）→ `pvwatts_dc`
    → **一次性 DC 侧损耗** `× (1 - losses_pct/100)` → `inverter.pvwatts`。

    **损耗只应用一次**：DC 阶段不施加任何损耗，乘子在进入逆变器**之前**施加一次。

    **双重上限**：pvlib 的 AC 模型饱和于 `pdc0 × eta_inv_nom`（= 480 kW），
    而批准的 `dc_ac_ratio=1.2` 给出更严格的 416.67 kW；
    实现取两者的**较小值**——只可能**降低**出力，绝不制造发电。
    """
    p = assert_pv_params(params)
    dc = pv_dc_before_losses(
        times, ghi_w_per_m2, temp_air_deg_c, wind_speed_10m_mps, params=p
    )
    return pv_ac_from_dc(dc, params=p, ghi_w_per_m2=ghi_w_per_m2)


def _pv_dc_stage(
    times: pd.DatetimeIndex,
    ghi_w_per_m2: Sequence[float] | np.ndarray,
    temp_air_deg_c: Sequence[float] | np.ndarray,
    wind_speed_10m_mps: Sequence[float] | np.ndarray,
    params: dict[str, Any] | None,
) -> np.ndarray:
    """pvlib 链条到 `pvwatts_dc` 为止（**不施加损耗**）。"""
    p = assert_pv_params(params)
    ghi = np.clip(np.asarray(ghi_w_per_m2, dtype=float), 0.0, None)
    temp_air = np.asarray(temp_air_deg_c, dtype=float)
    wind = np.clip(np.asarray(wind_speed_10m_mps, dtype=float), 0.0, None)
    index = pd.DatetimeIndex(times)
    if not (len(ghi) == len(temp_air) == len(wind) == len(index)):
        raise ExogenousDriverError("PV 输入长度不一致")

    position = solarposition.get_solarposition(index, p["latitude"], p["longitude"])
    zenith = _as_array(position["apparent_zenith"])
    azimuth = _as_array(position["azimuth"])

    decomposed = irradiance.erbs(ghi, zenith, index)
    dni = np.clip(np.nan_to_num(_as_array(decomposed["dni"]), nan=0.0), 0.0, None)
    dhi = np.clip(np.nan_to_num(_as_array(decomposed["dhi"]), nan=0.0), 0.0, None)

    poa = irradiance.get_total_irradiance(
        surface_tilt=p["tilt_deg"],
        surface_azimuth=p["azimuth_deg"],
        solar_zenith=zenith,
        solar_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        albedo=B5_PV_APPROVAL["albedo"],
        model=B5_PV_APPROVAL["transposition_model"],
    )
    poa_global = np.clip(
        np.nan_to_num(_as_array(poa["poa_global"]), nan=0.0), 0.0, None
    )
    cell = temperature.sapm_cell(
        poa_global, temp_air, wind,
        **temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][p["temperature_model"]],
    )
    dc = pvsystem.pvwatts_dc(
        poa_global, np.asarray(cell, dtype=float),
        pdc0=p["pv_capacity_kw"], gamma_pdc=p["gamma_pdc_per_deg_c"],
    )
    return np.clip(np.nan_to_num(np.asarray(dc, dtype=float), nan=0.0), 0.0, None)


# --- 风电：切变律 + 冻结功率曲线 ---------------------------------------------

@dataclass(frozen=True)
class WindPowerCurve:
    """冻结的确定性功率曲线（定义点按风速严格递增）。"""

    turbine_type: str
    speeds_mps: tuple[float, ...]
    power_kw: tuple[float, ...]
    source_sha256: str

    @property
    def cut_in_mps(self) -> float:
        return self.speeds_mps[0]

    @property
    def last_defined_mps(self) -> float:
        return self.speeds_mps[-1]


def load_wind_power_curve(
    path: Path | str = PUBLIC_BENCHMARKS_DIR / WIND_CURVE_FILE,
    *,
    turbine_type: str = WIND_PARAMS["turbine_model"],
    expected_sha256: str | None = WIND_CURVE_SHA256,
) -> WindPowerCurve:
    """读取**冻结**功率曲线的指定机组（W → kW）。"""
    path = Path(path)
    if not path.is_file():
        raise ExogenousDriverError(f"缺少冻结功率曲线 {path}")
    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ExogenousDriverError(
                f"功率曲线 SHA-256 不符：期望 {expected_sha256} 实际 {actual}"
            )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or rows[0][0] != WIND_CURVE_COLUMN:
        raise ExogenousDriverError("功率曲线首列必须是 turbine_type")
    header = rows[0]
    match = [row for row in rows[1:] if row and row[0] == turbine_type]
    if len(match) != 1:
        raise ExogenousDriverError(
            f"功率曲线中 {turbine_type!r} 的行数必须是 1，实际 {len(match)}"
        )
    row = match[0]
    speeds: list[float] = []
    power: list[float] = []
    for index, column in enumerate(header[1:], start=1):
        # 冻结 CSV 的某些行可能比表头短/长；按表头逐列取，越界视为未定义
        value = row[index] if index < len(row) else ""
        if value in ("", None):
            continue
        speeds.append(float(column))
        power.append(float(value) * WIND_CURVE_W_TO_KW)
    if len(speeds) < 2:
        raise ExogenousDriverError(f"{turbine_type} 的定义点不足")
    if any(b <= a for a, b in zip(speeds, speeds[1:], strict=False)):
        raise ExogenousDriverError("功率曲线的风速定义点必须严格递增")
    return WindPowerCurve(
        turbine_type=turbine_type, speeds_mps=tuple(speeds),
        power_kw=tuple(power), source_sha256=expected_sha256 or "",
    )


def hub_wind_speed(
    wind_speed_10m_mps: Sequence[float] | np.ndarray,
    *,
    hub_height_m: float = WIND_PARAMS["hub_height_m"],
    reference_height_m: float = WIND_PARAMS["reference_height_m"],
    shear_exponent: float = WIND_PARAMS["shear_exponent"],
) -> np.ndarray:
    """`v_hub = v_ref × (hub / ref) ** shear`（冻结的切变律）。"""
    v10 = np.asarray(wind_speed_10m_mps, dtype=float)
    if (v10 < 0).any():
        raise ExogenousDriverError("风速不得为负")
    return v10 * (float(hub_height_m) / float(reference_height_m)) ** float(shear_exponent)


def power_from_curve(
    v_hub_mps: Sequence[float] | np.ndarray,
    *,
    curve: WindPowerCurve,
    rated_capacity_kw: float = WIND_PARAMS["rated_capacity_kw"],
) -> np.ndarray:
    """按冻结曲线做**确定性线性插值**（kW）。

    边界规则（全部来自冻结曲线/模型定义，**不得**自行制造发电）：

    - `v < 首个定义点`（低于切入）→ `0`；
    - 定义域内 → 相邻定义点线性插值；
    - `v > 末个定义点`（**超出曲线定义域**，冻结曲线未定义切出行为）→ `0`（保守停机）；
    - 结果截到 `rated_capacity_kw`（**只减不增**）。
    """
    v = np.asarray(v_hub_mps, dtype=float)
    if (v < 0).any():
        raise ExogenousDriverError("轮毂风速不得为负")
    speeds = np.asarray(curve.speeds_mps, dtype=float)
    power = np.asarray(curve.power_kw, dtype=float)
    if speeds[0] > 0.0:
        speeds = np.concatenate(([0.0], speeds))
        power = np.concatenate(([0.0], power))
    interpolated = np.interp(v, speeds, power, left=0.0, right=0.0)
    beyond = v > curve.last_defined_mps
    interpolated = np.where(beyond, 0.0, interpolated)
    interpolated = np.where(v < curve.cut_in_mps, 0.0, interpolated)
    result = np.clip(interpolated, 0.0, float(rated_capacity_kw))
    if not np.isfinite(result).all():
        raise ExogenousDriverError("wind_generation_kw 含非有限值")
    return result


def wind_generation_kw(
    wind_speed_10m_mps: Sequence[float] | np.ndarray,
    *,
    curve: WindPowerCurve | None = None,
    hub_height_m: float = WIND_PARAMS["hub_height_m"],
    rated_capacity_kw: float = WIND_PARAMS["rated_capacity_kw"],
) -> np.ndarray:
    """`ERA5 10 m 风速 → 轮毂高度 → 冻结功率曲线 → kW`。"""
    loaded = load_wind_power_curve() if curve is None else curve
    hub = hub_wind_speed(wind_speed_10m_mps, hub_height_m=hub_height_m)
    return power_from_curve(hub, curve=loaded, rated_capacity_kw=rated_capacity_kw)


# --- 碳强度 -------------------------------------------------------------------

def carbon_intensity(rows: int) -> np.ndarray:
    """全 2024 逐行相同的**年内常数** 0.402 kgCO2/kWh（B1 人工批准）。"""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ExogenousDriverError(f"rows 必须是正整数，实际 {rows!r}")
    return np.full(rows, CARBON_KG_PER_KWH, dtype=float)


# --- arrival：由 benchmark trace 校准的 **date-free** 48-slot template ---------
#
# ⚠️ 官方 Azure archive 的**成员名不含任何日期**，官方说明也只写「collected in
# July of 2019」「14 files, one file per 24-h period」。因此本模块**不使用、也不
# 声称** archive 提供日期、星期或时区：模板只是**一个 24 小时周期内的
# 48 个半小时槽**（day-of-benchmark-period），再映射到 Singapore 2024 本地
# 00:00..23:30。**不得**称其为 IDC 真实 arrival、Azure 2019 replay 或 2024 观测。

ARRIVAL_TEMPLATE_SLOTS = 48
ARRIVAL_TEMPLATE_SHAPE = (ARRIVAL_TEMPLATE_SLOTS,)
ARRIVAL_SLOT_MAPPING = "slot 0..47 → Singapore 2024 本地 00:00..23:30"
ARRIVAL_PROCESS_FAMILY = "Poisson"
ARRIVAL_SOURCE_AGGREGATION = (
    "14 个 invocation 文件按 minute-of-24h 聚合（不使用、也不声称 archive 提供"
    "日期、星期或时区）"
)
ARRIVAL_TEMPLATE_KIND = "48-slot day-of-benchmark-period template"

# **B5-ARRIVAL 人工批准**（2026-09-16）
B5_ARRIVAL_APPROVAL: dict[str, Any] = {
    "decision_id": "B5-ARRIVAL",
    "approved_on": "2026-09-16",
    "template": ARRIVAL_TEMPLATE_KIND,
    "source_aggregation": ARRIVAL_SOURCE_AGGREGATION,
    "uses_archive_dates": False,
    "slot_mapping": ARRIVAL_SLOT_MAPPING,
    "process_family": ARRIVAL_PROCESS_FAMILY,
    "seed": ARRIVAL_SEED,
    "mean_arrival_work_units_per_half_hour": ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR,
    "scale_basis": ARRIVAL_SCALE_BASIS,
    "classification": ARRIVAL_CLASSIFICATION,
    "note": (
        "**不得**称为 IDC 真实 arrival、Azure 2019 replay 或 2024 observation；"
        "2019 trace 只校准 24 小时周期内的分布形状"
    ),
}


def assert_arrival_approved(record: dict[str, Any] | None) -> dict[str, Any]:
    """arrival 的过程/尺度/seed/slot mapping 必须**逐字段等于** B5-ARRIVAL 批准记录。"""
    if record is None:
        raise ExogenousDriverError("缺少 B5-ARRIVAL 人工批准记录：拒绝物化 arrival")
    if not isinstance(record, dict):
        raise ExogenousDriverError(
            f"B5-ARRIVAL 批准记录必须是 object，实际 {type(record).__name__}"
        )
    if set(record) != set(B5_ARRIVAL_APPROVAL):
        raise ExogenousDriverError(
            "B5-ARRIVAL 批准记录键集合必须精确等于冻结集合；"
            f"多出={sorted(set(record) - set(B5_ARRIVAL_APPROVAL))} "
            f"缺少={sorted(set(B5_ARRIVAL_APPROVAL) - set(record))}"
        )
    for name, expected in B5_ARRIVAL_APPROVAL.items():
        if record[name] != expected:
            raise ExogenousDriverError(
                f"B5-ARRIVAL 的 {name} 与批准记录不符："
                f"期望 {expected!r}，实际 {record[name]!r}"
            )
    return dict(record)


def arrival_slot_counts(minute_totals: pd.Series) -> np.ndarray:
    """把 `minute-of-24h` 的调用总量聚合为 **48 个半小时槽**。

    **只**使用 minute（0..1439）；**不**使用、也不推断任何日期、星期或时区。
    输入顺序不影响结果（按 minute **显式升序**聚合）。
    """
    if not isinstance(minute_totals, pd.Series) or minute_totals.empty:
        raise ExogenousDriverError("minute_totals 必须是非空 Series")
    minutes = np.asarray(minute_totals.index, dtype=int)
    if ((minutes < 0) | (minutes >= 1440)).any():
        raise ExogenousDriverError("minute 必须落在 [0, 1440)")
    values = np.asarray(minute_totals.to_numpy(dtype=float))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ExogenousDriverError("minute 调用量必须有限且非负")
    order = np.argsort(minutes, kind="stable")
    counts = np.zeros(ARRIVAL_TEMPLATE_SLOTS, dtype=float)
    for index in order:
        counts[int(minutes[index]) // 30] += float(values[index])
    return counts


def arrival_rate_template(counts: np.ndarray | Sequence[float]) -> np.ndarray:
    """把 48 槽计数归一化为**均值 1** 的 rate template。"""
    counts = np.asarray(counts, dtype=float)
    if counts.shape != ARRIVAL_TEMPLATE_SHAPE:
        raise ExogenousDriverError(
            f"计数向量形状必须是 {ARRIVAL_TEMPLATE_SHAPE}，实际 {counts.shape}"
        )
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ExogenousDriverError("计数向量必须有限且非负")
    if counts.sum() <= 0:
        raise ExogenousDriverError("计数向量总量必须为正")
    template = counts / counts.mean()
    if not np.isclose(template.mean(), 1.0, rtol=1e-12):
        raise ExogenousDriverError("rate template 未归一化为均值 1")
    return template


def arrival_template_slot(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """由 timestamp 的 **hour/minute** 选槽（**不读星期**）。"""
    stamps = pd.DatetimeIndex(timestamps)
    return (stamps.hour * 2 + stamps.minute // 30).to_numpy()


def arrival_rate_template_from_cache(
    manifest_path: Path | str = EXOGENOUS_MANIFEST,
) -> np.ndarray:
    """从已物化的 manifest 读取**冻结**的 48 槽 rate template（**不读 2024 truth**）。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ExogenousDriverError(f"缺少 {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrival = payload["columns"]["arrival"]
    assert_arrival_approved(arrival["b5_approval"])
    template = np.asarray(arrival["rate_template"], dtype=float)
    if template.shape != ARRIVAL_TEMPLATE_SHAPE:
        raise ExogenousDriverError(
            f"冻结 template 形状必须是 {ARRIVAL_TEMPLATE_SHAPE}，实际 {template.shape}"
        )
    return template


def generate_arrival(
    timestamps: pd.DatetimeIndex,
    template: np.ndarray | Sequence[float],
    *,
    seed: int = ARRIVAL_SEED,
    mean_per_half_hour: float = ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR,
) -> np.ndarray:
    """按 `Poisson(rate_template × mean)` **前向生成** arrival（work-units/step）。

    只依赖：冻结的 `template`、每个 timestamp 的 **hour/minute**、固定 `seed`。
    **不读取**任何 truth、价格、负荷、PV、风电、温度或未来 arrival——
    因此不存在未来信息泄漏，也不依赖星期。
    """
    template = np.asarray(template, dtype=float)
    if template.shape != ARRIVAL_TEMPLATE_SHAPE:
        raise ExogenousDriverError(
            f"template 形状必须是 {ARRIVAL_TEMPLATE_SHAPE}，实际 {template.shape}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExogenousDriverError(f"seed 必须是整数，实际 {seed!r}")
    if mean_per_half_hour <= 0:
        raise ExogenousDriverError("mean_per_half_hour 必须为正")
    stamps = pd.DatetimeIndex(timestamps)
    if len(stamps) == 0:
        raise ExogenousDriverError("timestamps 不得为空")
    slots = arrival_template_slot(stamps)
    rates = template[slots] * float(mean_per_half_hour)
    rng = np.random.default_rng(seed)
    arrival = rng.poisson(rates).astype(np.int64)
    if (arrival < 0).any():
        raise ExogenousDriverError("arrival 不得为负")
    return arrival


# --- 组装 ---------------------------------------------------------------------

def build_drivers(inputs: FrozenInputs, *, template: np.ndarray | None = None,
                  seed: int = ARRIVAL_SEED) -> dict[str, np.ndarray]:
    """由冻结输入 + 冻结 template 计算四列（**纯函数**，无 I/O 副作用）。"""
    frame = inputs.frame
    stamps = pd.DatetimeIndex(frame["timestamp"])
    if template is None:
        template = arrival_rate_template_from_cache()
    drivers: dict[str, np.ndarray] = {
        "local_pv_kw": local_pv_kw(
            stamps,
            frame["ghi_w_per_m2"].to_numpy(dtype=float),
            frame["temperature_deg_c"].to_numpy(dtype=float),
            frame["wind_speed_10m_mps"].to_numpy(dtype=float),
        ),
        "wind_generation_kw": wind_generation_kw(
            frame["wind_speed_10m_mps"].to_numpy(dtype=float)
        ),
        "carbon_intensity": carbon_intensity(len(frame)),
        "arrival": generate_arrival(stamps, template, seed=seed),
    }
    for name, values in drivers.items():
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise ExogenousDriverError(f"{name} 含非有限值")
    return drivers


def build_frame(inputs: FrozenInputs, *, template: np.ndarray | None = None,
                seed: int = ARRIVAL_SEED) -> pd.DataFrame:
    """组装完整输出表（五列，行数等于 canonical 行数）。"""
    drivers = build_drivers(inputs, template=template, seed=seed)
    return pd.DataFrame({
        "timestamp": pd.DatetimeIndex(inputs.frame["timestamp"]),
        **{name: np.asarray(values) for name, values in drivers.items()},
    })
