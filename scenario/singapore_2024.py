"""M1.3b：Singapore-2024 半小时 canonical reader（**只读、可审计**）。

把 M1.2 已冻结的四个 raw 文件读成一张统一的 **17,568 行**半小时事实表。

本模块**只**负责：raw hash/manifest 验证 → 原始字段解析 → 单位转换 →
半小时时间轴对齐 → 语义与来源记录。

本模块**不**负责：正式 `ScenarioBundle` 接线、train/validation/test 切分、
forecast 构造、arrival 数据、正式训练或评估。也**不**生成任何
本地 PV / 风电发电量 / 碳强度 —— 这些在 M1.2 中是**未冻结**的，
本模块只把它们登记为 `unavailable`，**绝不伪造**。

设计红线：

- 任一 raw 的 SHA-256 / 字节数 / schema / 年份 / 时区不匹配即 **fail closed**；
- **不插值、不补缺失、不重复日、不读取未来值**；
- 时间戳必须是 **tz-aware `Asia/Singapore`**，不得用 naive 时间冒充；
- 实际系统负荷**只能**来自 SASEA，**不得**读 USEP 的 `DEMAND (MW)`（那是预测）。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

YEAR = 2024
TIMEZONE = "Asia/Singapore"
SUPPORTED_SOURCE_SCHEMA = "m1.2-singapore-2024-v2"
EXPECTED_ROWS = 366 * 48  # 2024 是闰年：366 天 × 48 个半小时
FREQUENCY = "30min"

# 逐列单位与语义（canonical manifest 直接引用同一张表，避免两处漂移）
COLUMN_SPECS: dict[str, dict[str, str]] = {
    "timestamp": {
        "unit": "tz-aware Asia/Singapore",
        "semantic": "半小时结算点，严格递增且无重复",
    },
    "price_sgd_per_kwh": {
        "unit": "SGD/kWh",
        "semantic": "最终批发电价 USEP（SGD/MWh × 0.001）；允许为负，只要求有限",
    },
    "system_load_mw": {
        "unit": "MW",
        "semantic": "新加坡实际系统负荷；唯一来源是 SASEA raw_SGP_demand.csv",
    },
    "national_igs_mwh_per_half_hour": {
        "unit": "MWh per half-hour settlement period",
        "semantic": (
            "national intermittent generation（EMC metered generation, FACILITY TYPE=IGS）"
            "的**净注入**；**不是** IDC 本地 PV，也**不是**太阳能专属。"
            "约束是**有限**（允许负值）：设施净耗电的半小时净注入本就为负，"
            "这是真实计量语义，**不得** clip 或取绝对值"
        ),
        "constraint": "finite; negative values preserved as-is (net injection)",
    },
    "temperature_deg_c": {"unit": "degC", "semantic": "ERA5 2m 气温（国家级网格再分析）"},
    "wind_speed_10m_mps": {
        "unit": "m/s",
        "semantic": "ERA5 10m 风速；**不是**本地风电实测，也**不是**发电量",
    },
    "ghi_w_per_m2": {
        "unit": "W/m2",
        "semantic": "ERA5 前一小时平均 GHI；M1.2 未重采样",
    },
    "weather_source_timestamp": {
        "unit": "tz-aware Asia/Singapore",
        "semantic": "该行天气字段的**实际来源**小时时刻",
    },
    "weather_age_minutes": {
        "unit": "minutes",
        "semantic": "目标时刻 − 天气来源时刻；只能为 0 或 30",
    },
}

CANONICAL_COLUMNS: tuple[str, ...] = tuple(COLUMN_SPECS)

# 天气映射规则（**必须**同时写进 canonical manifest 与文档）
WEATHER_MAPPING_RULE = (
    "latest source timestamp not later than target timestamp"
)
ALLOWED_WEATHER_AGES_MINUTES = (0, 30)

# M1.2 中**未冻结**、本模块**绝不伪造**的字段
UNAVAILABLE_COLUMNS: dict[str, str] = {
    "local_pv_kw": (
        "unavailable: M1.2 未冻结 IDC 本地 PV 实测；national IGS "
        "**不等于** IDC 本地 PV，也**不是** solar-only"
    ),
    "wind_generation_kw": (
        "unavailable: M1.2 未冻结风电发电量；ERA5 wind_speed_10m "
        "**不等于**本地风电实测，也**不等于**发电量"
    ),
    "carbon_intensity": (
        "unavailable: 尚无冻结的**半小时碳强度**；"
        "不得由默认曲线或常数伪造"
    ),
    "arrival": (
        "unavailable: arrival（任务到达）尚未进入 M1.3 数据口径"
    ),
}

MISSING_DATA_POLICY = "reject; no imputation"

REQUIRED_RAW_FILES = {
    "emc_usep": "emc_usep_2024.zip",
    "emc_metered_generation": "emc_metered_generation_2024.zip",
    "sasea_demand": "sasea_demand_2024.zip",
    "open_meteo_weather": "open_meteo_era5_2024.csv",
}

_SASEA_MEMBER = "data/raw/raw_SGP_demand.csv"
_HALF_HOUR = timedelta(minutes=30)


class SingaporeReaderError(ValueError):
    """canonical reader 的**明确失败**（不得静默降级或插补）。"""


# --- 单位与数值 -------------------------------------------------------------

def usep_sgd_per_mwh_to_reader_unit(value_sgd_per_mwh: float) -> float:
    """USEP 单位换算：SGD/MWh × 0.001 = SGD/kWh（M1.2 未物化，转换只发生在这里）。"""
    return float(value_sgd_per_mwh) * 0.001


def _finite(value: Any, *, field: str) -> float:
    """必须是有限实数；NaN / ±inf / 非数值一律失败。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SingaporeReaderError(f"{field} 不是数值：{value!r}") from error
    if not math.isfinite(number):
        raise SingaporeReaderError(f"{field} 不是有限值：{value!r}")
    return number


def _non_negative(value: Any, *, field: str) -> float:
    number = _finite(value, field=field)
    if number < 0.0:
        raise SingaporeReaderError(f"{field} 不得为负：{number!r}")
    return number


# --- raw 核验 ---------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_source_manifest(source_manifest_path: Path) -> dict:
    try:
        manifest = json.loads(Path(source_manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SingaporeReaderError(f"无法读取 source manifest：{error}") from error
    if not isinstance(manifest, dict):
        raise SingaporeReaderError("source manifest 顶层必须是对象")
    if manifest.get("schema") != SUPPORTED_SOURCE_SCHEMA:
        raise SingaporeReaderError(
            f"source manifest schema 必须是 {SUPPORTED_SOURCE_SCHEMA!r}，"
            f"实际 {manifest.get('schema')!r}"
        )
    if manifest.get("year") != YEAR:
        raise SingaporeReaderError(
            f"source manifest year 必须是 {YEAR}，实际 {manifest.get('year')!r}"
        )
    if manifest.get("timezone") != TIMEZONE:
        raise SingaporeReaderError(
            f"source manifest timezone 必须是 {TIMEZONE!r}，实际 {manifest.get('timezone')!r}"
        )
    return manifest


def verify_raw_files(raw_dir: Path | str, source_manifest: dict) -> dict[str, Path]:
    """按冻结 manifest 核验四个 raw 文件（SHA-256 + 字节数）；不匹配即失败。"""
    raw_path = Path(raw_dir)
    recorded = (source_manifest.get("verification") or {}).get("raw_files")
    if not isinstance(recorded, dict):
        raise SingaporeReaderError("source manifest 缺少 verification.raw_files")
    resolved: dict[str, Path] = {}
    for name, filename in REQUIRED_RAW_FILES.items():
        entry = recorded.get(name)
        if not isinstance(entry, dict) or "sha256" not in entry or "bytes" not in entry:
            raise SingaporeReaderError(f"source manifest 未记录 {name} 的 sha256/bytes")
        path = raw_path / filename
        if not path.is_file():
            raise SingaporeReaderError(f"缺少 raw 文件：{path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["bytes"]:
            raise SingaporeReaderError(
                f"{filename} 字节数不符：manifest={entry['bytes']} 实际={actual_bytes}"
            )
        actual_sha = _sha256(path)
        if actual_sha != entry["sha256"]:
            raise SingaporeReaderError(
                f"{filename} SHA-256 不符：manifest={entry['sha256']} 实际={actual_sha}"
            )
        resolved[name] = path
    return resolved


# --- 原始解析 ---------------------------------------------------------------

def _nems_timestamp(row: dict[str, str], *, label: str) -> datetime:
    """NEMS 的 (DATE, PERIOD) → naive 本地半小时时刻；越界即失败。"""
    try:
        day = datetime.strptime(row["DATE"], "%d-%b-%Y").date()
        period = int(row["PERIOD"])
    except (KeyError, TypeError, ValueError) as error:
        raise SingaporeReaderError(f"{label} 行 DATE/PERIOD 非法：{row!r}") from error
    if day.year != YEAR:
        raise SingaporeReaderError(f"{label} 行不属于 {YEAR}：{row!r}")
    if not 1 <= period <= 48:
        raise SingaporeReaderError(f"{label} 行 PERIOD 越界：{row!r}")
    return datetime.combine(day, datetime.min.time()) + timedelta(minutes=(period - 1) * 30)


def _iter_zip_rows(path: Path, prefix: str):
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if Path(name).name.startswith(prefix) and name.endswith(".csv")
        )
        if len(members) != 12:
            raise SingaporeReaderError(
                f"{path.name} 必须含 12 个 {prefix}*.csv，实际 {len(members)}"
            )
        for member in members:
            with archive.open(member) as raw:
                reader = csv.DictReader(
                    io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                )
                if reader.fieldnames is None:
                    raise SingaporeReaderError(f"{path.name}:{member} 没有表头")
                for row in reader:
                    yield member, row


def _parse_usep(path: Path) -> dict[datetime, float]:
    """USEP 价格（SGD/MWh）。**只读** `USEP ($/MWh)`；绝不使用 `DEMAND (MW)`。"""
    prices: dict[datetime, float] = {}
    for member, row in _iter_zip_rows(path, "USEP_"):
        if row.get("INFORMATION TYPE") != "USEP":
            raise SingaporeReaderError(f"{member} 含非 USEP 行：{row!r}")
        timestamp = _nems_timestamp(row, label="USEP")
        if timestamp in prices:
            raise SingaporeReaderError(f"USEP 出现重复时间戳：{timestamp}")
        # 注意：`DEMAND (MW)` 是**需求预测**，不是实际负荷，故刻意不读。
        prices[timestamp] = _finite(row.get("USEP ($/MWh)"), field="USEP ($/MWh)")
    return prices


def _parse_igs(path: Path) -> dict[datetime, float]:
    """national intermittent generation（EMC `FACILITY TYPE=IGS`）。"""
    igs: dict[datetime, float] = {}
    for member, row in _iter_zip_rows(path, "MG_"):
        if row.get("FACILITY TYPE") != "IGS":
            continue  # 只取 IGS；其他设施类型不属于 national intermittent generation
        if row.get("INFORMATION TYPE") != "MG":
            raise SingaporeReaderError(f"{member} 含非 MG 的 IGS 行：{row!r}")
        timestamp = _nems_timestamp(row, label="IGS")
        if timestamp in igs:
            raise SingaporeReaderError(f"IGS 出现重复时间戳：{timestamp}")
        # **有限**即可：净注入在设施净耗电时段为负，是真实计量语义。
        # 绝不 clip 到 0、绝不取绝对值（那会伪造数据）。
        igs[timestamp] = _finite(
            row.get("NET INJECTION (MWh)"), field="NET INJECTION (MWh)"
        )
    if not igs:
        raise SingaporeReaderError(f"{path.name} 没有 IGS 行")
    return igs


def _parse_system_load(path: Path) -> dict[datetime, float]:
    """实际系统负荷（MW）—— **唯一**来源是 SASEA `raw_SGP_demand.csv`。"""
    loads: dict[datetime, float] = {}
    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.open(_SASEA_MEMBER)
        except KeyError as error:
            raise SingaporeReaderError(f"{path.name} 缺少 {_SASEA_MEMBER}") from error
        with raw:
            reader = csv.DictReader(
                io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            )
            for row in reader:
                try:
                    timestamp = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
                except (KeyError, TypeError, ValueError) as error:
                    raise SingaporeReaderError(f"系统负荷行 datetime 非法：{row!r}") from error
                if timestamp.year != YEAR:
                    continue  # SASEA 文件覆盖多年，只取 2024
                if timestamp in loads:
                    raise SingaporeReaderError(f"系统负荷出现重复时间戳：{timestamp}")
                loads[timestamp] = _non_negative(
                    row.get("system_demand"), field="system_demand"
                )
    if not loads:
        raise SingaporeReaderError(f"{path.name} 没有 {YEAR} 年的系统负荷行")
    return loads


def _parse_weather(path: Path) -> dict[datetime, tuple[float, float, float]]:
    """ERA5 小时天气。保留原单位；**不做任何重采样**。"""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.startswith("time,"))
    except StopIteration as error:
        raise SingaporeReaderError(f"{path.name} 没有 Open-Meteo 小时表头") from error
    if header_index < 2:
        raise SingaporeReaderError(f"{path.name} 缺少 Open-Meteo metadata 行")
    metadata = dict(zip(
        next(csv.reader([lines[0]])), next(csv.reader([lines[1]])), strict=True
    ))
    if metadata.get("timezone") != TIMEZONE:
        raise SingaporeReaderError(
            f"天气时区必须是 {TIMEZONE!r}，实际 {metadata.get('timezone')!r}"
        )
    expected = {"time", "temperature_2m (°C)", "wind_speed_10m (m/s)",
                "shortwave_radiation (W/m²)"}
    reader = csv.DictReader(lines[header_index:])
    if set(reader.fieldnames or ()) != expected:
        raise SingaporeReaderError(f"{path.name} 天气列与冻结查询不符：{reader.fieldnames}")

    weather: dict[datetime, tuple[float, float, float]] = {}
    for row in reader:
        try:
            timestamp = datetime.strptime(row["time"], "%Y-%m-%dT%H:%M")
        except (KeyError, TypeError, ValueError) as error:
            raise SingaporeReaderError(f"天气行 time 非法：{row!r}") from error
        if timestamp.year != YEAR:
            continue
        if timestamp in weather:
            raise SingaporeReaderError(f"天气出现重复时间戳：{timestamp}")
        weather[timestamp] = (
            _finite(row["temperature_2m (°C)"], field="temperature_2m"),
            _non_negative(row["wind_speed_10m (m/s)"], field="wind_speed_10m"),
            _non_negative(row["shortwave_radiation (W/m²)"], field="shortwave_radiation"),
        )
    if not weather:
        raise SingaporeReaderError(f"{path.name} 没有 {YEAR} 年的天气行")
    return weather


# --- 时间轴与映射 -----------------------------------------------------------

def expected_half_hours(year: int = YEAR) -> list[datetime]:
    """该年的完整半小时轴（naive 本地时刻）。"""
    start = datetime(year, 1, 1)
    days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    return [start + _HALF_HOUR * i for i in range(days * 48)]


def _localize(naive: datetime, *, field: str) -> pd.Timestamp:
    """naive 本地墙钟 → tz-aware `Asia/Singapore`。

    Asia/Singapore 自 1982 年起无夏令时，不存在歧义/不存在时刻；
    若上游给出无法定位的时刻，pandas 会抛错 —— 这正是我们要的 fail closed。
    """
    try:
        return pd.Timestamp(naive).tz_localize(ZoneInfo(TIMEZONE))
    except Exception as error:  # noqa: BLE001 - 无法定位即失败，不得静默假设
        raise SingaporeReaderError(f"{field} 无法定位到 {TIMEZONE}：{naive}") from error


def _require_complete(observed: dict, *, label: str) -> None:
    """必须是**恰好**完整闰年半小时轴：无缺失、无多余。"""
    expected = expected_half_hours()
    missing = [t for t in expected if t not in observed]
    extra = [t for t in observed if t not in set(expected)]
    if missing:
        raise SingaporeReaderError(
            f"{label} 缺少 {len(missing)} 个半小时点（不插补），"
            f"首个缺失：{missing[0]}"
        )
    if extra:
        raise SingaporeReaderError(
            f"{label} 含 {len(extra)} 个轴外时刻（不重复日），首个：{extra[0]}"
        )


def load_singapore_2024_half_hour(
    raw_dir: Path | str,
    source_manifest_path: Path | str,
) -> pd.DataFrame:
    """把 M1.2 冻结的四个 raw 文件读成 17,568 行半小时 canonical 事实表。

    **只读**：不写入任何文件，也不改动 source manifest 的任何字节。
    """
    source_manifest = _read_source_manifest(Path(source_manifest_path))
    paths = verify_raw_files(raw_dir, source_manifest)

    prices = _parse_usep(paths["emc_usep"])
    igs = _parse_igs(paths["emc_metered_generation"])
    loads = _parse_system_load(paths["sasea_demand"])
    weather = _parse_weather(paths["open_meteo_weather"])

    _require_complete(prices, label="USEP")
    _require_complete(igs, label="IGS")
    _require_complete(loads, label="系统负荷")

    weather_hours = sorted(weather)
    if len(weather_hours) != 366 * 24:
        raise SingaporeReaderError(
            f"天气必须是 {YEAR} 的 8,784 个小时点，实际 {len(weather_hours)}"
        )

    rows: list[dict] = []
    cursor = 0
    for naive in expected_half_hours():
        # 冻结映射规则：latest source timestamp **not later than** target
        while cursor + 1 < len(weather_hours) and weather_hours[cursor + 1] <= naive:
            cursor += 1
        if weather_hours[cursor] > naive:
            raise SingaporeReaderError(
                f"天气来源晚于目标时刻（禁止用下一小时填充）："
                f"target={naive} source={weather_hours[cursor]}"
            )
        source_hour = weather_hours[cursor]
        age_minutes = int((naive - source_hour).total_seconds() // 60)
        if age_minutes not in ALLOWED_WEATHER_AGES_MINUTES:
            raise SingaporeReaderError(
                f"天气 age 必须是 {ALLOWED_WEATHER_AGES_MINUTES} 分钟，"
                f"实际 {age_minutes}（target={naive} source={source_hour}）"
            )
        temperature, wind, ghi = weather[source_hour]
        rows.append({
            "timestamp": _localize(naive, field="timestamp"),
            "price_sgd_per_kwh": usep_sgd_per_mwh_to_reader_unit(prices[naive]),
            "system_load_mw": loads[naive],
            "national_igs_mwh_per_half_hour": igs[naive],
            "temperature_deg_c": temperature,
            "wind_speed_10m_mps": wind,
            "ghi_w_per_m2": ghi,
            "weather_source_timestamp": _localize(source_hour, field="weather_source_timestamp"),
            "weather_age_minutes": age_minutes,
        })

    frame = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    _validate_frame(frame)
    return frame


def _validate_frame(frame: pd.DataFrame) -> None:
    """输出的最终自检：任一不满足即失败（不得产出不合格的表）。"""
    if len(frame) != EXPECTED_ROWS:
        raise SingaporeReaderError(f"canonical 表必须是 {EXPECTED_ROWS} 行，实际 {len(frame)}")
    if tuple(frame.columns) != CANONICAL_COLUMNS:
        raise SingaporeReaderError(f"canonical 列不符：{tuple(frame.columns)}")
    if str(frame["timestamp"].dt.tz) != TIMEZONE:
        raise SingaporeReaderError("timestamp 时区语义必须是 Asia/Singapore")
    if not frame["timestamp"].is_monotonic_increasing:
        raise SingaporeReaderError("timestamp 必须严格递增")
    if frame["timestamp"].duplicated().any():
        raise SingaporeReaderError("timestamp 不得重复")
    if not (frame["weather_source_timestamp"] <= frame["timestamp"]).all():
        raise SingaporeReaderError("weather_source_timestamp 不得晚于 timestamp")
    if not set(frame["weather_age_minutes"].unique()) <= set(ALLOWED_WEATHER_AGES_MINUTES):
        raise SingaporeReaderError("weather_age_minutes 只能为 0 或 30")
    for column in ("price_sgd_per_kwh", "temperature_deg_c"):
        if not frame[column].map(math.isfinite).all():
            raise SingaporeReaderError(f"{column} 必须是有限值")
    if not frame["national_igs_mwh_per_half_hour"].map(math.isfinite).all():
        raise SingaporeReaderError("national_igs_mwh_per_half_hour 必须是有限值")
    for column in ("system_load_mw", "wind_speed_10m_mps", "ghi_w_per_m2"):
        if not frame[column].map(math.isfinite).all() or (frame[column] < 0).any():
            raise SingaporeReaderError(f"{column} 必须非负且有限")
