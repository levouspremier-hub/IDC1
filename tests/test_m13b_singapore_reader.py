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
import subprocess
import sys
import tempfile
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

# 全年 fixture 只构造/读取一次：17,568 行 × 4 源，重复构造会让门禁变慢。
_CACHED: dict = {}


def cached_valid_fixture() -> dict:
    if "fixture" not in _CACHED:
        _CACHED["fixture"] = write_raw_fixture(
            pathlib.Path(tempfile.mkdtemp(prefix="m13b_valid_"))
        )
    return _CACHED["fixture"]


def cached_valid_frame():
    if "frame" not in _CACHED:
        _CACHED["frame"] = _load(cached_valid_fixture())
    return _CACHED["frame"]


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


def test_full_leap_year_produces_17568_rows(tmp_path):
    frame = cached_valid_frame()
    assert len(frame) == EXPECTED_ROWS
    assert tuple(frame.columns) == CANONICAL_COLUMNS


def test_timeline_is_strictly_increasing_and_complete(tmp_path):
    frame = cached_valid_frame()
    stamps = frame["timestamp"]
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    assert stamps.iloc[0].isoformat().startswith("2024-01-01T00:00")
    assert stamps.iloc[-1].isoformat().startswith("2024-12-31T23:30")
    assert len(stamps) == EXPECTED_ROWS


def test_timestamps_are_timezone_aware(tmp_path):
    frame = cached_valid_frame()
    stamps = frame["timestamp"]
    assert str(stamps.dt.tz) == TIMEZONE, "不得用 naive timestamp 冒充带时区时间"
    assert frame["weather_source_timestamp"].dt.tz is not None


# --- 2. 单位与来源 ----------------------------------------------------------

def test_usep_is_converted_exactly_to_sgd_per_kwh(tmp_path):
    frame = _load(write_raw_fixture(tmp_path, price_scale=1.0))
    assert frame["price_sgd_per_kwh"].iloc[0] == pytest.approx(0.1, abs=0)
    assert (frame["price_sgd_per_kwh"] == 0.1).all()


def test_negative_price_is_allowed_but_must_be_finite():
    from scenario.singapore_2024 import usep_sgd_per_mwh_to_reader_unit

    assert usep_sgd_per_mwh_to_reader_unit(-50.0) == pytest.approx(-0.05)
    assert usep_sgd_per_mwh_to_reader_unit(100.0) == pytest.approx(0.1)


def test_system_load_comes_from_sasea_not_from_the_usep_demand_column(tmp_path):
    frame = cached_valid_frame()
    # fixture 的 USEP DEMAND 列写死 9999；实际负荷必须是 SASEA 的 6000
    assert (frame["system_load_mw"] == 6000.0).all()
    assert not (frame["system_load_mw"] == 9999.0).any()


def test_igs_keeps_national_semantics_and_is_not_local_pv(tmp_path):
    frame = cached_valid_frame()
    assert (frame["national_igs_mwh_per_half_hour"] == 250.0).all()
    lowered = " ".join(frame.columns).lower()
    assert "local_pv" not in lowered and "solar" not in lowered
    assert "national" in lowered


# --- 3. 天气映射规则 --------------------------------------------------------

def test_weather_uses_the_latest_source_not_later_than_the_target(tmp_path):
    frame = cached_valid_frame()
    ages = frame["weather_age_minutes"]
    assert set(ages.unique()) <= {0, 30}
    at_h00 = frame[frame["timestamp"].dt.minute == 0]
    at_h30 = frame[frame["timestamp"].dt.minute == 30]
    assert (at_h00["weather_age_minutes"] == 0).all()
    assert (at_h30["weather_age_minutes"] == 30).all()
    assert (at_h30["weather_source_timestamp"].dt.minute == 0).all()


def test_weather_source_is_never_later_than_the_target(tmp_path):
    frame = cached_valid_frame()
    assert (frame["weather_source_timestamp"] <= frame["timestamp"]).all()


def test_future_weather_source_fails_closed(tmp_path):
    """天气来源晚于目标时间 -> 必须失败，不得用下一小时填充。"""
    fixture = write_raw_fixture(tmp_path, weather_shift_minutes=30)
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


def test_weather_not_shifted_cannot_silently_backfill(tmp_path):
    """规则必须是「不晚于目标」；用 h:30 的天气填 h:00 属 backfill，必须失败。"""
    fixture = write_raw_fixture(tmp_path, weather_shift_minutes=-30)
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


# --- 4. 缺失与非法值 fail closed -------------------------------------------

def test_missing_half_hour_fails_without_imputation(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, drop_half_hour=True))


def test_duplicate_timestamp_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, duplicate_half_hour=True))


def test_nan_price_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, nan_price=True))


def test_infinite_load_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, inf_load=True))


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


def test_wrong_year_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, weather_year=2023))


def test_wrong_timezone_fails(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(write_raw_fixture(tmp_path, weather_timezone="UTC"))


def test_raw_hash_mismatch_fails(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    (fixture["files"]["emc_usep"]).write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


def test_raw_byte_count_mismatch_fails(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    manifest = json.loads(fixture["source_manifest"].read_text(encoding="utf-8"))
    entry = manifest["verification"]["raw_files"]["emc_usep"]
    entry["bytes"] = entry["bytes"] + 1          # sha 仍对，仅 bytes 不符
    fixture["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


def test_unsupported_schema_fails(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    manifest = json.loads(fixture["source_manifest"].read_text(encoding="utf-8"))
    manifest["schema"] = "m1.2-singapore-2024-v1"
    fixture["source_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        _load(fixture)


def test_source_manifest_is_byte_identical_before_and_after(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    before = fixture["source_manifest"].read_bytes()
    _load(fixture)
    assert fixture["source_manifest"].read_bytes() == before


# --- 5. unavailable 字段 ----------------------------------------------------

def test_unavailable_columns_are_declared_not_invented():
    from scenario.singapore_2024 import UNAVAILABLE_COLUMNS as declared

    assert set(declared) == set(UNAVAILABLE_COLUMNS)


def test_canonical_table_does_not_invent_unavailable_columns(tmp_path):
    frame = cached_valid_frame()
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


def test_existing_different_canonical_manifest_is_not_silently_overwritten(tmp_path):
    fixture = write_raw_fixture(tmp_path)
    out = tmp_path / "out"
    _materialize(fixture, out)
    manifest_path = out / "singapore_2024_half_hour.json"
    manifest_path.write_text('{"schema": "tampered"}', encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        _materialize(fixture, out)


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


# --- 9. 返修：冻结可复现性、失败原子性、路径可移植性 -------------------------

REPO_ROOT_STR = str(REPO_ROOT)
MATERIALIZER = "scripts.materialize_singapore_2024"
MATERIALIZER_SOURCES = (
    "scenario/singapore_2024.py",
    "scripts/materialize_singapore_2024.py",
)


def _snap(path: pathlib.Path) -> tuple:
    st = path.stat()
    return (st.st_size, hashlib.sha256(path.read_bytes()).hexdigest(), st.st_mtime_ns)


def test_materializer_revision_is_git_verified():
    """`materializer_revision` 必须由 Git 解析，指向最后修改生成实现的提交。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    revision = module.resolve_materializer_revision()
    assert isinstance(revision, str) and len(revision) == 40, revision
    assert all(c in "0123456789abcdef" for c in revision)

    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *MATERIALIZER_SOURCES],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert revision == expected


def test_materializer_revision_ignores_docs_only_commits():
    """docs-only 提交不得改变 `materializer_revision`（否则冻结数据会被误判失效）。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    revision = module.resolve_materializer_revision()

    # 该 revision 之后的提交**不得**触及任何生成实现文件
    touched = subprocess.run(
        ["git", "log", "--format=%H", "--name-only", f"{revision}..HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    for source in MATERIALIZER_SOURCES:
        assert source not in touched, f"{revision} 之后又改动了 {source}"


def test_materializer_revision_is_not_a_caller_supplied_value(tmp_path):
    """不得接受调用者传入的未经验证 revision 冒充来源。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    module.materialize(
        raw_dir=fixture["raw_dir"],
        source_manifest_path=fixture["source_manifest"],
        output_dir=out,
        manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    manifest = json.loads((out / "canon.json").read_text(encoding="utf-8"))
    assert manifest["materializer_revision"] == module.resolve_materializer_revision()
    assert "code_revision" not in manifest, "不得再使用随 HEAD 漂移的 code_revision"


def test_manifest_records_repo_relative_logical_paths(tmp_path):
    """入库 manifest 只记仓库相对逻辑路径，不得出现绝对/home 路径。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    module.materialize(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    text = (out / "canon.json").read_text(encoding="utf-8")
    assert REPO_ROOT_STR not in text
    assert "/Users/" not in text
    assert str(pathlib.Path.home()) not in text
    manifest = json.loads(text)
    for entry in manifest["raw_files"].values():
        assert not entry["path"].startswith("/"), entry["path"]
        assert ".." not in entry["path"], entry["path"]
    assert not manifest["source_manifest_path"].startswith("/")


def test_logical_paths_are_stable_across_different_roots(tmp_path):
    """不同临时 root 下生成的逻辑 provenance 必须一致（可移植）。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    first = write_raw_fixture(tmp_path / "root_a")
    second = write_raw_fixture(tmp_path / "root_b")
    manifests = []
    for index, fixture in enumerate((first, second)):
        out = tmp_path / f"out{index}"
        module.materialize(
            raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
            output_dir=out, manifest_path=out / "canon.json",
            frozen_at_utc="2026-09-15T00:00:00+00:00",
        )
        manifests.append(json.loads((out / "canon.json").read_text(encoding="utf-8")))
    assert {entry["path"] for entry in manifests[0]["raw_files"].values()} == \
           {entry["path"] for entry in manifests[1]["raw_files"].values()}


def test_identical_rematerialization_does_not_rewrite_anything(tmp_path):
    """完全相同时不得重写：parquet 与 manifest 的 mtime_ns 都必须不变。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    kwargs = dict(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    module.materialize(**kwargs)
    parquet, manifest = out / "half_hour.parquet", out / "canon.json"
    before = (_snap(parquet), _snap(manifest))
    module.materialize(**kwargs)
    after = (_snap(parquet), _snap(manifest))
    assert after[0][1] == before[0][1], "parquet hash 变了"
    assert after[0][2] == before[0][2], "parquet mtime_ns 变了（被无谓重写）"
    assert after[1][1] == before[1][1], "manifest hash 变了"
    assert after[1][2] == before[1][2], "manifest mtime_ns 变了（被无谓重写）"


def test_manifest_mismatch_leaves_both_artifacts_untouched(tmp_path):
    """manifest 不一致时 fail closed，且 parquet/manifest 的 bytes/hash/mtime 全不变。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    kwargs = dict(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    module.materialize(**kwargs)
    parquet, manifest = out / "half_hour.parquet", out / "canon.json"
    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["row_count"] = 999
    manifest.write_text(json.dumps(tampered), encoding="utf-8")

    before = (_snap(parquet), _snap(manifest))
    entries_before = sorted(p.name for p in out.iterdir())
    with pytest.raises((ValueError, FileExistsError)):
        module.materialize(**kwargs)
    after = (_snap(parquet), _snap(manifest))

    assert after[0] == before[0], "失败的物化改写了正式 parquet"
    assert after[1] == before[1], "失败的物化改写了正式 manifest"
    assert sorted(p.name for p in out.iterdir()) == entries_before, "留下了临时文件"


def test_first_freeze_is_atomic_and_leaves_no_temp_files(tmp_path):
    """首次冻结：原子安装，不留下临时文件。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    module.materialize(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    assert sorted(p.name for p in out.iterdir()) == ["canon.json", "half_hour.parquet"]


def test_corrupt_parquet_is_restored_only_when_the_candidate_hash_matches(tmp_path):
    """正式 parquet 损坏：候选 hash 与冻结 manifest 相符时原子恢复。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    kwargs = dict(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    module.materialize(**kwargs)
    parquet = out / "half_hour.parquet"
    frozen_sha = json.loads((out / "canon.json").read_text(encoding="utf-8"))[
        "output_parquet_sha256"]

    parquet.write_bytes(b"corrupted")
    module.materialize(**kwargs)
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == frozen_sha
    assert sorted(p.name for p in out.iterdir()) == ["canon.json", "half_hour.parquet"]


def test_corrupt_parquet_with_mismatching_frozen_hash_fails_closed(tmp_path):
    """冻结 manifest 的 hash 与候选不符时不得"恢复"，必须失败。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    kwargs = dict(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    module.materialize(**kwargs)
    parquet = out / "half_hour.parquet"
    manifest = json.loads((out / "canon.json").read_text(encoding="utf-8"))
    manifest["output_parquet_sha256"] = "0" * 64
    (out / "canon.json").write_text(json.dumps(manifest), encoding="utf-8")

    parquet.write_bytes(b"corrupted")
    before_pq = _snap(parquet)
    before_mf = _snap(out / "canon.json")
    with pytest.raises((ValueError, FileExistsError)):
        module.materialize(**kwargs)
    assert _snap(parquet) == before_pq
    assert _snap(out / "canon.json") == before_mf


def test_frozen_manifest_and_parquet_agree_after_materialization(tmp_path):
    """冻结不变量：manifest 的 output hash 必须与实际 parquet 完全一致。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    module.materialize(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )
    manifest = json.loads((out / "canon.json").read_text(encoding="utf-8"))
    assert manifest["output_parquet_sha256"] == hashlib.sha256(
        (out / "half_hour.parquet").read_bytes()
    ).hexdigest()


# --- 10. 第二次返修：首冻失败原子性、外部 manifest 严格校验、dirty generator ---

def _materialize_kwargs(fixture, out):
    return dict(
        raw_dir=fixture["raw_dir"], source_manifest_path=fixture["source_manifest"],
        output_dir=out, manifest_path=out / "canon.json",
        frozen_at_utc="2026-09-15T00:00:00+00:00",
    )


def test_first_freeze_manifest_failure_leaves_no_half_state(tmp_path, monkeypatch):
    """首冻时 manifest 安装失败：正式 parquet/manifest 都不得存在，且无临时文件。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"

    def boom(path, text):
        raise OSError("injected manifest install failure")

    monkeypatch.setattr(module, "_atomic_write_text", boom)
    with pytest.raises(OSError):
        module.materialize(**_materialize_kwargs(fixture, out))

    assert not (out / "canon.json").exists(), "manifest 不得存在"
    assert not (out / "half_hour.parquet").exists(), "首冻失败不得留下孤立 parquet"
    assert [p.name for p in out.iterdir()] == [], "不得留下任何半成品或临时文件"


@pytest.mark.parametrize("stage", ("write", "replace"))
def test_first_freeze_manifest_install_rollback(tmp_path, monkeypatch, stage):
    """manifest 临时写入失败与 os.replace 失败都必须完整回滚。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"

    if stage == "write":
        def broken_mkstemp(*args, **kwargs):
            raise OSError("injected manifest temp write failure")
        monkeypatch.setattr(module.tempfile, "mkstemp", broken_mkstemp)
    else:
        real_replace = module.os.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):
            calls["n"] += 1
            if str(dst).endswith("canon.json"):
                raise OSError("injected manifest os.replace failure")
            return real_replace(src, dst)
        monkeypatch.setattr(module.os, "replace", flaky_replace)

    with pytest.raises(OSError):
        module.materialize(**_materialize_kwargs(fixture, out))

    assert not (out / "half_hour.parquet").exists(), f"{stage}: 不得留下孤立 parquet"
    assert not (out / "canon.json").exists(), f"{stage}: 不得留下 manifest"
    assert [p.name for p in out.iterdir()] == [], f"{stage}: 不得留下临时文件"


def test_orphan_parquet_without_manifest_fails_closed(tmp_path):
    """manifest 不存在但正式 parquet 已存在 -> fail closed，且原 parquet 完全不动。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    orphan = out / "half_hour.parquet"
    orphan.write_bytes(b"pre-existing unknown parquet")
    before = _snap(orphan)

    with pytest.raises((ValueError, FileExistsError)):
        module.materialize(**_materialize_kwargs(fixture, out))

    assert orphan.exists(), "不得删除既有 parquet"
    assert _snap(orphan) == before, "既有 parquet 的 bytes/hash/mtime_ns 不得变化"
    assert not (out / "canon.json").exists(), "不得创建 manifest"
    assert sorted(p.name for p in out.iterdir()) == ["half_hour.parquet"], "不得留下临时文件"


@pytest.mark.parametrize("mutation,label", [
    (lambda m: [], "顶层不是 object"),
    (lambda m: {k: v for k, v in m.items() if k != "output_parquet_sha256"},
     "缺 output_parquet_sha256"),
    (lambda m: {**m, "output_parquet_sha256": "not-hex"}, "hash 不是 64 位小写十六进制"),
    (lambda m: {**m, "output_parquet_sha256": "A" * 64}, "hash 含大写"),
    (lambda m: {k: v for k, v in m.items() if k != "materializer_revision"},
     "缺 materializer_revision"),
    (lambda m: {**m, "materializer_revision": "zzz"}, "revision 非法"),
])
def test_malformed_frozen_manifest_fails_closed(tmp_path, mutation, label):
    """畸形 frozen manifest 必须 fail closed，且不改动任何正式产物。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    module.materialize(**_materialize_kwargs(fixture, out))

    parquet, manifest = out / "half_hour.parquet", out / "canon.json"
    good = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(json.dumps(mutation(good)), encoding="utf-8")
    before = (_snap(parquet), _snap(manifest))

    with pytest.raises(ValueError):
        module.materialize(**_materialize_kwargs(fixture, out))

    assert _snap(parquet) == before[0], f"{label}: parquet 被改动"
    assert _snap(manifest) == before[1], f"{label}: manifest 被改动"


def test_malformed_frozen_manifest_exits_cleanly_via_cli(tmp_path):
    """畸形 frozen manifest 必须由 CLI 干净 exit 1，不得泄漏 traceback。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"
    module.materialize(**_materialize_kwargs(fixture, out))
    (out / "canon.json").write_text('{"output_parquet_sha256": "short"}', encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/materialize_singapore_2024.py",
         "--raw-dir", str(fixture["raw_dir"]),
         "--source-manifest", str(fixture["source_manifest"]),
         "--output-dir", str(out), "--manifest", str(out / "canon.json"),
         "--frozen-at-utc", "2026-09-15T00:00:00+00:00"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr and "Traceback" not in result.stdout
    assert "KeyError" not in result.stderr and "TypeError" not in result.stderr


def test_dirty_generator_is_rejected(tmp_path, monkeypatch):
    """生成实现文件有未提交修改时必须拒绝正式物化。"""
    import importlib

    module = importlib.import_module(MATERIALIZER)
    fixture = cached_valid_fixture()
    out = tmp_path / "out"

    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        module.materialize(**_materialize_kwargs(fixture, out))
    assert not (out / "half_hour.parquet").exists()
    assert not (out / "canon.json").exists()
