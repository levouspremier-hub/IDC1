"""M1.3b 测试：Singapore-2024 半小时 canonical reader。

本卡只做**只读、可审计**的半小时事实表：raw 核验 → 解析 → 单位转换 → 时间对齐 →
本地物化。**不做** ScenarioBundle 接线、切分、forecast、arrival、训练或评估。

改前缺陷（本文件在实现前必须为红）：`scenario/singapore_2024.py` 不存在，
没有任何半小时 canonical reader，也没有物化与 canonical manifest。
"""

import hashlib
import io
import json
import math
import pathlib
import zipfile
from datetime import datetime, timedelta

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIMEZONE = "Asia/Singapore"
YEAR = 2024
EXPECTED_ROWS = 366 * 48
# fixture 中 USEP 的 DEMAND 列写死一个可辨识的假值；实际负荷只能来自 SASEA
USEP_DEMAND_FORECAST = 9999.0
CANONICAL_COLUMNS = (
    "timestamp",
    "price_sgd_per_kwh",
    "system_load_mw",
    "national_igs_mwh_per_half_hour",
    "temperature_deg_c",
    "wind_speed_10m_mps",
    "ghi_w_per_m2",
    "weather_source_timestamp",
    "weather_age_minutes",
)
UNAVAILABLE_COLUMNS = ("local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival")

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# 与**冻结 raw 文件实测表头**一致（不得自创列名；见 M1.3b 卡 §实现记录）
USEP_HEADER = (
    "INFORMATION TYPE,DATE,PERIOD,USEP ($/MWh),LCP ($/MWh),DEMAND (MW),"
    "SOLAR(MW),TCL (MW),RUSEP ($/MWh),MAP ($/MWh),MAPT ($/MWh),TPC Applied"
)
MG_HEADER = (
    "INFORMATION TYPE,DATE,PERIOD,FACILITY TYPE,GROSS INJECTION (MWh),"
    "NET INJECTION (MWh)"
)
SASEA_HEADER = "datetime,system_demand"
WEATHER_HEADER = "time,temperature_2m (°C),wind_speed_10m (m/s),shortwave_radiation (W/m²)"


# --- fixture 构造 -----------------------------------------------------------

def _half_hours() -> list[datetime]:
    start = datetime(YEAR, 1, 1)
    return [start + timedelta(minutes=30 * i) for i in range(EXPECTED_ROWS)]


def _month_files(timestamps, usep_values, igs_values):
    """按月份分组，生成 zip 内的成员内容。"""
    usep_by_month: dict[str, list[str]] = {}
    igs_by_month: dict[str, list[str]] = {}
    for ts, price, igs in zip(timestamps, usep_values, igs_values, strict=True):
        key = f"{ts.month:02d}"
        date_text = ts.strftime("%d-%b-%Y")
        period = (ts.hour * 2) + (1 if ts.minute == 30 else 0) + 1
        usep_by_month.setdefault(key, []).append(
            f"USEP,{date_text},{period},{price},0.00,{USEP_DEMAND_FORECAST},"
            f"0.000,0.000,{price},166.23,556.02,No"
        )
        igs_by_month.setdefault(key, []).append(
            f"MG,{date_text},{period},IGS,{igs},{igs}"
        )
        # 非 IGS 行（BATTERY）必须被过滤掉，不得混入 national IGS
        igs_by_month[key].append(
            f"MG,{date_text},{period},BATTERY,99999.0,99999.0"
        )
    return usep_by_month, igs_by_month


def _zip_bytes(members: dict[str, list[str]], headers: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, rows in members.items():
            archive.writestr(name, headers + "\n" + "\n".join(rows) + "\n")
    return buffer.getvalue()


def write_raw_fixture(
    root: pathlib.Path,
    *,
    rows: int = EXPECTED_ROWS,
    price_scale: float = 1.0,
    igs_scale: float = 1.0,
    drop_half_hour: bool = False,
    duplicate_half_hour: bool = False,
    nan_price: bool = False,
    inf_load: bool = False,
    negative_igs: bool = False,
    weather_shift_minutes: int = 0,
    weather_year: int = YEAR,
    weather_timezone: str = TIMEZONE,
) -> dict:
    """构造四个最小 raw fixture + 冻结 manifest，返回路径字典。"""
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    stamps = _half_hours()[:rows]
    prices = [100.0 * price_scale] * len(stamps)
    igs_values = [250.0 * igs_scale] * len(stamps)
    loads = [6000.0] * len(stamps)

    if drop_half_hour:
        stamps = stamps[:-1]
        prices = prices[:-1]
        igs_values = igs_values[:-1]
        loads = loads[:-1]
    if duplicate_half_hour:
        stamps = stamps + [stamps[0]]
        prices = prices + [prices[0]]
        igs_values = igs_values + [igs_values[0]]
        loads = loads + [loads[0]]
    if nan_price:
        prices[0] = float("nan")
    if inf_load:
        loads[0] = float("inf")
    if negative_igs:
        igs_values[0] = -1.0

    usep_months, igs_months = _month_files(stamps, prices, igs_values)
    usep_zip = raw_dir / "emc_usep_2024.zip"
    usep_zip.write_bytes(_zip_bytes(
        {f"USEP_{m}{YEAR}.csv": v for m, v in usep_months.items()},
        USEP_HEADER,
    ))
    igs_zip = raw_dir / "emc_metered_generation_2024.zip"
    igs_zip.write_bytes(_zip_bytes(
        {f"MG_{m}{YEAR}.csv": v for m, v in igs_months.items()},
        MG_HEADER,
    ))

    # SASEA
    sasea_months: dict[str, list[str]] = {}
    for ts, load in zip(stamps, loads, strict=True):
        sasea_months.setdefault(f"{ts.month:02d}", []).append(
            f"{ts.strftime('%Y-%m-%d %H:%M:%S')},{load}"
        )
    sasea_zip = raw_dir / "sasea_demand_2024.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "data/raw/raw_SGP_demand.csv",
            SASEA_HEADER + "\n"
            + "2014-01-06 00:00:00,4980.1787\n"
            + "2023-12-31 23:30:00,5000.0\n"
            + "\n".join(r for m in sorted(sasea_months) for r in sasea_months[m])
            + "\n",
        )
    sasea_zip.write_bytes(buffer.getvalue())

    # ERA5
    weather_rows = []
    base = datetime(weather_year, 1, 1) + timedelta(minutes=weather_shift_minutes)
    for i in range(366 * 24):
        ts = base + timedelta(hours=i)
        weather_rows.append(f"{ts.strftime('%Y-%m-%dT%H:%M')},28.0,2.5,180.0")
    weather_csv = raw_dir / "open_meteo_era5_2024.csv"
    weather_csv.write_text(
        "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
        f"1.5,103.75,46.0,28800,{weather_timezone},GMT+8\n"
        "\n"
        + WEATHER_HEADER + "\n"
        + "\n".join(weather_rows) + "\n",
        encoding="utf-8",
    )

    files = {
        "emc_usep": usep_zip,
        "emc_metered_generation": igs_zip,
        "sasea_demand": sasea_zip,
        "open_meteo_weather": weather_csv,
    }
    source_manifest = root / "singapore_2024.json"
    source_manifest.write_text(json.dumps({
        "schema": "m1.2-singapore-2024-v2",
        "year": YEAR,
        "timezone": TIMEZONE,
        "verification": {
            "raw_files": {
                name: {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
                for name, path in files.items()
            }
        },
    }, indent=2), encoding="utf-8")
    return {"raw_dir": raw_dir, "source_manifest": source_manifest, "files": files}


LOADER = "scenario.singapore_2024"


def _load(fixture, **kwargs):
    from scenario.singapore_2024 import load_singapore_2024_half_hour

    return load_singapore_2024_half_hour(
        fixture["raw_dir"], fixture["source_manifest"], **kwargs
    )


# --- 1. 形状与时间轴 --------------------------------------------------------

def test_module_is_importable():
    """先决条件：canonical reader 模块必须存在。"""
    import importlib

    importlib.import_module(LOADER)


@pytest.mark.slow
def test_full_leap_year_produces_17568_rows(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    assert len(frame) == EXPECTED_ROWS
    assert tuple(frame.columns) == CANONICAL_COLUMNS


@pytest.mark.slow
def test_timeline_is_strictly_increasing_and_complete(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    stamps = frame["timestamp"]
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    assert stamps.iloc[0].isoformat().startswith("2024-01-01T00:00")
    assert stamps.iloc[-1].isoformat().startswith("2024-12-31T23:30")
    assert len(stamps) == EXPECTED_ROWS


@pytest.mark.slow
def test_timestamps_are_timezone_aware(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    stamps = frame["timestamp"]
    assert str(stamps.dt.tz) == TIMEZONE, "不得用 naive timestamp 冒充带时区时间"
    assert frame["weather_source_timestamp"].dt.tz is not None


# --- 2. 单位与来源 ----------------------------------------------------------

@pytest.mark.slow
def test_usep_is_converted_exactly_to_sgd_per_kwh(tmp_path):
    frame = _load(write_raw_fixture(tmp_path, price_scale=1.0))
    assert frame["price_sgd_per_kwh"].iloc[0] == pytest.approx(0.1, abs=0)
    assert (frame["price_sgd_per_kwh"] == 0.1).all()


def test_negative_price_is_allowed_but_must_be_finite():
    from scenario.singapore_2024 import usep_sgd_per_mwh_to_reader_unit

    assert usep_sgd_per_mwh_to_reader_unit(-50.0) == pytest.approx(-0.05)
    assert usep_sgd_per_mwh_to_reader_unit(100.0) == pytest.approx(0.1)


@pytest.mark.slow
def test_system_load_comes_from_sasea_not_from_the_usep_demand_column(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    # fixture 的 USEP DEMAND 列写死 9999；实际负荷必须是 SASEA 的 6000
    assert (frame["system_load_mw"] == 6000.0).all()
    assert not (frame["system_load_mw"] == 9999.0).any()


@pytest.mark.slow
def test_igs_keeps_national_semantics_and_is_not_local_pv(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    assert (frame["national_igs_mwh_per_half_hour"] == 250.0).all()
    lowered = " ".join(frame.columns).lower()
    assert "local_pv" not in lowered and "solar" not in lowered
    assert "national" in lowered


# --- 3. 天气映射规则 --------------------------------------------------------

@pytest.mark.slow
def test_weather_uses_the_latest_source_not_later_than_the_target(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    ages = frame["weather_age_minutes"]
    assert set(ages.unique()) <= {0, 30}
    at_h00 = frame[frame["timestamp"].dt.minute == 0]
    at_h30 = frame[frame["timestamp"].dt.minute == 30]
    assert (at_h00["weather_age_minutes"] == 0).all()
    assert (at_h30["weather_age_minutes"] == 30).all()
    assert (at_h30["weather_source_timestamp"].dt.minute == 0).all()


@pytest.mark.slow
def test_weather_source_is_never_later_than_the_target(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    assert (frame["weather_source_timestamp"] <= frame["timestamp"]).all()


@pytest.mark.slow
def test_future_weather_source_fails_closed(tmp_path):
    """天气来源晚于目标时间 -> 必须失败，不得用下一小时填充。"""
    fixture = write_raw_fixture(tmp_path, weather_shift_minutes=30)
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


@pytest.mark.slow
def test_weather_not_shifted_cannot_silently_backfill(tmp_path):
    """规则必须是「不晚于目标」；用 h:30 的天气填 h:00 属 backfill，必须失败。"""
    fixture = write_raw_fixture(tmp_path, weather_shift_minutes=-30)
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


# --- 4. 缺失与非法值 fail closed -------------------------------------------

@pytest.mark.slow
def test_missing_half_hour_fails_without_imputation(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, drop_half_hour=True))


@pytest.mark.slow
def test_duplicate_timestamp_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, duplicate_half_hour=True))


@pytest.mark.slow
def test_nan_price_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, nan_price=True))


@pytest.mark.slow
def test_infinite_load_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, inf_load=True))


@pytest.mark.slow
def test_negative_igs_is_preserved_not_clipped(tmp_path):
    """净注入为负是真实计量语义：必须**原值保留**，不得 clip 到 0 或取绝对值。

    （M1.3b 卡 §3 的「非负」已按人工授权改为「有限」——冻结数据中
    43.6% 的 IGS 值为负，最小 -0.124 MWh。）
    """
    frame = _load(write_raw_fixture(tmp_path, negative_igs=True))
    assert frame["national_igs_mwh_per_half_hour"].iloc[0] == pytest.approx(-1.0)
    assert (frame["national_igs_mwh_per_half_hour"] < 0).sum() == 1
    assert (frame["national_igs_mwh_per_half_hour"] >= 0).sum() == EXPECTED_ROWS - 1


def test_infinite_igs_fails():
    """非有限值仍然必须失败（只是不再要求非负）。"""
    from scenario.singapore_2024 import _finite

    with pytest.raises(ValueError):
        _finite(float("inf"), field="NET INJECTION (MWh)")
    with pytest.raises(ValueError):
        _finite(float("nan"), field="NET INJECTION (MWh)")


@pytest.mark.slow
def test_wrong_year_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, weather_year=2023))


@pytest.mark.slow
def test_wrong_timezone_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, weather_timezone="UTC"))


@pytest.mark.slow
def test_raw_hash_mismatch_fails(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    (fixture["files"]["emc_usep"]).write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


@pytest.mark.slow
def test_raw_byte_count_mismatch_fails(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    manifest = json.loads(fixture["source_manifest"].read_text(encoding="utf-8"))
    entry = manifest["verification"]["raw_files"]["emc_usep"]
    entry["bytes"] = entry["bytes"] + 1          # sha 仍对，仅 bytes 不符
    fixture["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


@pytest.mark.slow
def test_unsupported_schema_fails(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    manifest = json.loads(fixture["source_manifest"].read_text(encoding="utf-8"))
    manifest["schema"] = "m1.2-singapore-2024-v1"
    fixture["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


@pytest.mark.slow
def test_source_manifest_is_byte_identical_before_and_after(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    before = fixture["source_manifest"].read_bytes()
    _load(fixture)
    assert fixture["source_manifest"].read_bytes() == before


# --- 5. unavailable 字段 ----------------------------------------------------

def test_unavailable_columns_are_declared_not_invented():
    from scenario.singapore_2024 import UNAVAILABLE_COLUMNS as declared

    assert set(declared) == set(UNAVAILABLE_COLUMNS)


@pytest.mark.slow
def test_canonical_table_does_not_invent_unavailable_columns(tmp_path):
    frame = _load(write_raw_fixture(tmp_path))
    for column in UNAVAILABLE_COLUMNS:
        assert column not in frame.columns, f"不得伪造 {column}"


# --- 6. 真实 raw 的 slow acceptance ----------------------------------------

@pytest.mark.slow
def test_real_frozen_raw_materializes_17568_rows():
    """真实 M1.2 冻结数据必须能被读出 17,568 行半小时表。"""
    raw_dir = REPO_ROOT / "data/raw/singapore_2024"
    source_manifest = REPO_ROOT / "data/manifest/singapore_2024.json"
    if not source_manifest.exists():
        pytest.skip("M1.2 冻结 manifest 不在本分支")
    from scenario.singapore_2024 import load_singapore_2024_half_hour

    frame = load_singapore_2024_half_hour(raw_dir, source_manifest)
    assert len(frame) == EXPECTED_ROWS
    assert str(frame["timestamp"].dt.tz) == TIMEZONE
    assert (frame["weather_source_timestamp"] <= frame["timestamp"]).all()
    assert frame["system_load_mw"].notna().all()
    assert (frame["system_load_mw"] >= 0).all()
    # IGS 净注入按人工授权为「有限」：真实数据约 43.6% 为负，必须**原值保留**
    assert frame["national_igs_mwh_per_half_hour"].notna().all()
    assert (frame["national_igs_mwh_per_half_hour"] < 0).any(), "真实 IGS 含负净注入"
    assert frame["national_igs_mwh_per_half_hour"].map(math.isfinite).all()


# --- 7. canonical manifest（物化） -----------------------------------------

def _materialize(fixture, out_root, *, frozen_at="2026-09-15T00:00:00+00:00"):
    from scripts.materialize_singapore_2024 import materialize

    return materialize(
        raw_dir=fixture["raw_dir"],
        source_manifest_path=fixture["source_manifest"],
        output_dir=out_root / "processed",
        manifest_path=out_root / "singapore_2024_half_hour.json",
        frozen_at_utc=frozen_at,
    )


@pytest.mark.slow
def test_canonical_manifest_records_inputs_outputs_and_mapping(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    out = tmp_path / "out"
    result = _materialize(fixture, out)
    manifest_path = pathlib.Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"]
    assert manifest["year"] == 2024
    assert manifest["timezone"] == TIMEZONE
    assert manifest["row_count"] == EXPECTED_ROWS
    assert manifest["frequency"] == "30min"
    assert manifest["missing_data_policy"] == "reject; no imputation"
    assert manifest["frozen_at_utc"] == "2026-09-15T00:00:00+00:00"

    # 四个 raw 输入的 SHA-256 与 bytes
    raw_files = manifest["raw_files"]
    assert set(raw_files) == {"emc_usep", "emc_metered_generation",
                              "sasea_demand", "open_meteo_weather"}
    for name, entry in raw_files.items():
        assert entry["sha256"] == hashlib.sha256(
            fixture["files"][name].read_bytes()
        ).hexdigest()
        assert entry["bytes"] == fixture["files"][name].stat().st_size

    # source manifest 的 SHA-256
    assert manifest["source_manifest_sha256"] == hashlib.sha256(
        fixture["source_manifest"].read_bytes()
    ).hexdigest()

    # 输出 parquet 的 SHA-256
    parquet = pathlib.Path(result["parquet_path"])
    assert manifest["output_parquet_sha256"] == hashlib.sha256(
        parquet.read_bytes()
    ).hexdigest()

    # 天气映射规则必须**写在 manifest 里**，不能只藏在实现中
    policy = manifest["weather_mapping_policy"]
    assert "latest source timestamp not later than target" in policy["rule"]
    assert policy["allowed_weather_age_minutes"] == [0, 30]

    # 每列单位与语义
    for column in CANONICAL_COLUMNS:
        assert column in manifest["columns"], column
        assert manifest["columns"][column]["unit"] is not None
        assert manifest["columns"][column]["semantic"]


@pytest.mark.slow
def test_canonical_manifest_declares_the_unavailable_fields(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    result = _materialize(fixture, tmp_path / "out")
    manifest = json.loads(pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8"))

    unavailable = manifest["unavailable_not_materialized"]
    assert set(unavailable) == set(UNAVAILABLE_COLUMNS)
    for column, entry in unavailable.items():
        assert entry["status"] in ("unavailable", "not_materialized"), column
        assert entry["reason"], column
    blob = json.dumps(unavailable, ensure_ascii=False).lower()
    assert "idc 本地 pv" in blob          # national IGS ≠ IDC 本地 PV
    assert "local_pv_kw" in blob and "wind_generation_kw" in blob
    assert "carbon" in blob


@pytest.mark.slow
def test_existing_different_canonical_manifest_is_not_silently_overwritten(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    out = tmp_path / "out"
    _materialize(fixture, out)
    manifest_path = out / "singapore_2024_half_hour.json"
    manifest_path.write_text('{"schema": "tampered"}', encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        _materialize(fixture, out)


@pytest.mark.slow
def test_idempotent_rematerialization_with_the_same_freeze_time(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    out = tmp_path / "out"
    first = _materialize(fixture, out)
    second = _materialize(fixture, out)
    assert first["output_parquet_sha256"] == second["output_parquet_sha256"]


# --- 8. 过时的阻塞说明不得残留 ----------------------------------------------

def test_error_messages_do_not_claim_m12_is_unfinished():
    """M1.2 已冻结；错误信息不得再误称「M1.2 尚未完成/仍阻塞」。"""
    source = (REPO_ROOT / "scenario/scenario.py").read_text(encoding="utf-8")
    assert "M1.2 阻塞" not in source, "不得再称 M1.2 阻塞"
    assert "待 M1.2 解除阻塞后实现" not in source
    assert "M1.3" in source, "必须指向真正未完成的 M1.3"


def test_build_scenario_still_fails_closed_without_an_m13_manifest(tmp_path):
    """正式路径仍必须明确失败，且原因指向 M1.3，而非 M1.2。"""
    from scenario.scenario import build_scenario

    with pytest.raises((FileNotFoundError, ValueError)) as excinfo:
        build_scenario("train", start="2024-01-01", horizon=24, forecast_cutoff=4,
                       manifest_dir=str(tmp_path / "no_such_manifest_dir"))
    message = str(excinfo.value)
    assert "M1.3" in message
    assert "M1.2 阻塞" not in message
