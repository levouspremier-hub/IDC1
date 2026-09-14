#!/usr/bin/env python
"""M1.2a：核验并冻结 Singapore-2024 原始场景数据。

EMC 的 USEP 与按设施类型发电年度包必须由研究者在官方网页手工下载。
该脚本绝不会访问 EMC，也不会尝试登入或批量抓取；它只核验给定的本地
ZIP。SASEA 负荷 ZIP 同样以本地文件传入。Open-Meteo 天气可用一次、明确
的非商业 API 请求下载，或传入已有 CSV。

原始数据不入 Git。脚本只将可审计的元数据和 SHA-256 写入 Git 跟踪的
``data/manifest/singapore_2024.json``。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

YEAR = 2024
SINGAPORE_TIMEZONE = "Asia/Singapore"
EXPECTED_HALF_HOUR_ROWS = 366 * 48
EXPECTED_HOURLY_ROWS = 366 * 24

EMC_PRICES_URL = "https://www.nems.emcsg.com/nems-prices"
EMC_TERMS_URL = "https://www.home.emcsg.com/terms-and-conditions"
SASEA_URL = "https://zenodo.org/records/17175212"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

REQUIRED_FILES = {
    "emc_usep": "emc_usep_2024.zip",
    "emc_metered_generation": "emc_metered_generation_2024.zip",
    "sasea_demand": "sasea_demand_2024.zip",
    "open_meteo_weather": "open_meteo_era5_2024.csv",
}


def _expected_half_hours(year: int = YEAR) -> list[datetime]:
    start = datetime.combine(date(year, 1, 1), time.min)
    end = datetime.combine(date(year + 1, 1, 1), time.min)
    values: list[datetime] = []
    current = start
    while current < end:
        values.append(current)
        current += timedelta(minutes=30)
    return values


def _expected_hours(year: int = YEAR) -> list[datetime]:
    start = datetime.combine(date(year, 1, 1), time.min)
    end = datetime.combine(date(year + 1, 1, 1), time.min)
    values: list[datetime] = []
    current = start
    while current < end:
        values.append(current)
        current += timedelta(hours=1)
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: str, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric, got {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return parsed


def _require_timeline(
    label: str, timestamps: Iterable[datetime], expected: list[datetime]
) -> None:
    actual = list(timestamps)
    expected_set = set(expected)
    actual_set = set(actual)
    if len(actual) != len(actual_set):
        raise ValueError(f"{label} contains duplicate timestamps")
    if actual_set != expected_set:
        missing = len(expected_set - actual_set)
        unexpected = len(actual_set - expected_set)
        raise ValueError(
            f"{label} is not continuous: expected {len(expected)} timestamps, "
            f"got {len(actual)} (missing={missing}, unexpected={unexpected})"
        )


def _csv_rows_from_zip(path: Path, prefix: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if Path(name).name.startswith(prefix) and name.endswith(".csv")
        )
        if len(members) != 12:
            raise ValueError(
                f"{path.name} must contain 12 {prefix} monthly CSV files, got {len(members)}"
            )
        rows: list[dict[str, str]] = []
        for member in members:
            with archive.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                if reader.fieldnames is None:
                    raise ValueError(f"{path.name}:{member} has no CSV header")
                rows.extend(dict(row) for row in reader)
    return rows


def _nems_timestamp(row: dict[str, str], *, label: str, year: int) -> datetime:
    try:
        day = datetime.strptime(row["DATE"], "%d-%b-%Y").date()
        period = int(row["PERIOD"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} has invalid DATE/PERIOD row: {row!r}") from error
    if day.year != year or not 1 <= period <= 48:
        raise ValueError(f"{label} timestamp outside {year} or invalid period: {row!r}")
    return datetime.combine(day, time.min) + timedelta(minutes=(period - 1) * 30)


def _validate_usep(path: Path, year: int) -> dict[str, Any]:
    rows = _csv_rows_from_zip(path, "USEP_")
    timestamps: list[datetime] = []
    for row in rows:
        if row.get("INFORMATION TYPE") != "USEP":
            raise ValueError(f"{path.name} contains non-USEP row")
        timestamps.append(_nems_timestamp(row, label="USEP", year=year))
        _finite_float(row.get("USEP ($/MWh)", ""), field="USEP ($/MWh)")
        _finite_float(row.get("DEMAND (MW)", ""), field="DEMAND (MW)")
    _require_timeline("USEP", timestamps, _expected_half_hours(year))
    return {
        "rows": len(rows),
        "raw_unit": "SGD/MWh",
        "unit_conversion": "SGD/MWh -> SGD/kWh (/1000)",
        "timezone": SINGAPORE_TIMEZONE,
        "semantic": "final wholesale USEP; DEMAND column is forecast only",
    }


def _validate_igs(path: Path, year: int) -> dict[str, Any]:
    rows = _csv_rows_from_zip(path, "MG_")
    igs_rows = [row for row in rows if row.get("FACILITY TYPE") == "IGS"]
    if not igs_rows:
        raise ValueError(f"{path.name} has no IGS rows")
    timestamps: list[datetime] = []
    for row in igs_rows:
        if row.get("INFORMATION TYPE") != "MG":
            raise ValueError(f"{path.name} contains non-MG IGS row")
        timestamps.append(_nems_timestamp(row, label="IGS", year=year))
        _finite_float(row.get("NET INJECTION (MWh)", ""), field="NET INJECTION (MWh)")
    _require_timeline("IGS", timestamps, _expected_half_hours(year))
    return {
        "rows": len(igs_rows),
        "facility_type": "IGS",
        "raw_unit": "MWh per half-hour settlement period",
        "timezone": SINGAPORE_TIMEZONE,
        "semantic": (
            "metered national intermittent generation; "
            "not solar-only or a local IDC PV measurement"
        ),
    }


def _validate_actual_system_load(path: Path, year: int) -> dict[str, Any]:
    member = "data/raw/raw_SGP_demand.csv"
    with zipfile.ZipFile(path) as archive:
        try:
            raw = archive.open(member)
        except KeyError as error:
            raise ValueError(f"{path.name} is missing {member}") from error
        with raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            timestamps: list[datetime] = []
            for row in reader:
                try:
                    timestamp = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"actual system load has invalid datetime: {row!r}") from error
                if timestamp.year != year:
                    continue
                _finite_float(row.get("system_demand", ""), field="system_demand")
                timestamps.append(timestamp)
    _require_timeline("actual system load", timestamps, _expected_half_hours(year))
    return {
        "source": "SASEA raw_SGP_demand.csv",
        "rows": len(timestamps),
        "raw_unit": "MW",
        "timezone": SINGAPORE_TIMEZONE,
        "semantic": "actual system demand",
    }


def _read_weather(path: Path, year: int) -> tuple[list[datetime], dict[str, str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith("time,"))
    except StopIteration as error:
        raise ValueError(f"{path.name} has no Open-Meteo hourly header") from error
    if header_index < 2:
        raise ValueError(f"{path.name} has no Open-Meteo metadata row")
    metadata_keys = next(csv.reader([lines[0]]))
    metadata_values = next(csv.reader([lines[1]]))
    metadata = dict(zip(metadata_keys, metadata_values, strict=True))
    reader = csv.DictReader(lines[header_index:])
    expected_columns = {
        "time",
        "temperature_2m (°C)",
        "wind_speed_10m (m/s)",
        "shortwave_radiation (W/m²)",
    }
    if set(reader.fieldnames or ()) != expected_columns:
        raise ValueError(f"{path.name} weather columns do not match frozen query")
    timestamps: list[datetime] = []
    for row in reader:
        try:
            timestamp = datetime.strptime(row["time"], "%Y-%m-%dT%H:%M")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"weather has invalid time: {row!r}") from error
        if timestamp.year != year:
            continue
        _finite_float(row["temperature_2m (°C)"], field="temperature_2m")
        _finite_float(row["wind_speed_10m (m/s)"], field="wind_speed_10m")
        _finite_float(row["shortwave_radiation (W/m²)"], field="shortwave_radiation")
        timestamps.append(timestamp)
    return timestamps, metadata


def _validate_weather(path: Path, year: int) -> dict[str, Any]:
    timestamps, metadata = _read_weather(path, year)
    if metadata.get("timezone") != SINGAPORE_TIMEZONE:
        raise ValueError(
            f"weather timezone must be {SINGAPORE_TIMEZONE}, got {metadata.get('timezone')!r}"
        )
    _require_timeline("weather", timestamps, _expected_hours(year))
    return {
        "rows": len(timestamps),
        "timezone": SINGAPORE_TIMEZONE,
        "raw_units": {
            "temperature_2m": "degC",
            "wind_speed_10m": "m/s",
            "shortwave_radiation": "W/m2 (hourly preceding-hour mean GHI)",
        },
        "model": "ERA5",
        "coordinates_returned": {
            "latitude": metadata.get("latitude"),
            "longitude": metadata.get("longitude"),
        },
        "resampling": "not performed in M1.2",
    }


def validate_singapore_2024(raw_dir: Path | str) -> dict[str, Any]:
    """核验本地四个原始文件；任何缺失、重复或时间间断均明确失败。"""
    raw_path = Path(raw_dir)
    paths = {name: raw_path / filename for name, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required raw files: " + ", ".join(missing))
    return {
        "year": YEAR,
        "timezone": SINGAPORE_TIMEZONE,
        "verified_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "files": {
            name: {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
        "usep": _validate_usep(paths["emc_usep"], YEAR),
        "actual_system_load": _validate_actual_system_load(paths["sasea_demand"], YEAR),
        "intermittent_generation": _validate_igs(paths["emc_metered_generation"], YEAR),
        "weather": _validate_weather(paths["open_meteo_weather"], YEAR),
    }


def _parse_freeze_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"--frozen-at-utc must be ISO-8601, got {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError("--frozen-at-utc must include an explicit UTC offset")
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _manifest_from_report(report: dict[str, Any], *, frozen_at_utc: str) -> dict[str, Any]:
    return {
        "schema": "m1.2-singapore-2024-v1",
        "year": report["year"],
        "timezone": report["timezone"],
        "verification": {
            "frozen_at_utc": frozen_at_utc,
            "missing_data_policy": "reject; no imputation",
            "raw_files": report["files"],
        },
        "sources": {
            "emc_usep": {
                "url": EMC_PRICES_URL,
                "rights": (
                    "EMC website terms: personal/non-commercial use only; do not redistribute"
                ),
                "terms_url": EMC_TERMS_URL,
                "automation": "manual browser download only",
                "downloaded_product": "USEP and Demand Forecast annual ZIP",
                "raw_file_frozen_at_utc": frozen_at_utc,
                "timezone": SINGAPORE_TIMEZONE,
                "raw_units": {"usep": "SGD/MWh", "demand_forecast": "MW"},
            },
            "emc_metered_generation": {
                "url": EMC_PRICES_URL,
                "rights": (
                    "EMC website terms: personal/non-commercial use only; do not redistribute"
                ),
                "terms_url": EMC_TERMS_URL,
                "automation": "manual browser download only",
                "downloaded_product": "Metered Generation by Facility Type annual ZIP",
                "raw_file_frozen_at_utc": frozen_at_utc,
                "timezone": SINGAPORE_TIMEZONE,
                "raw_units": {"net_injection": "MWh per half-hour settlement period"},
            },
            "sasea_demand": {
                "url": SASEA_URL,
                "license": "CC-BY-4.0",
                "downloaded_product": "data.zip / data/raw/raw_SGP_demand.csv",
                "raw_file_frozen_at_utc": frozen_at_utc,
                "timezone": SINGAPORE_TIMEZONE,
                "raw_units": {"system_demand": "MW"},
            },
            "open_meteo_weather": {
                "url": OPEN_METEO_URL,
                "license": "CC-BY-4.0; free API non-commercial use only",
                "query": {
                    "latitude": 1.3521,
                    "longitude": 103.8198,
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                    "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
                    "wind_speed_unit": "ms",
                    "timezone": SINGAPORE_TIMEZONE,
                    "models": "era5",
                },
                "raw_file_frozen_at_utc": frozen_at_utc,
                "timezone": SINGAPORE_TIMEZONE,
                "raw_units": report["weather"]["raw_units"],
            },
        },
        "series": {
            "price": {
                "source": "emc_usep",
                "semantic": report["usep"]["semantic"],
                "raw_unit": report["usep"]["raw_unit"],
                "stored_unit": "SGD/kWh",
                "transform": report["usep"]["unit_conversion"],
                "rows": report["usep"]["rows"],
            },
            "load": {
                "source": "sasea_demand",
                "semantic": report["actual_system_load"]["semantic"],
                "raw_unit": report["actual_system_load"]["raw_unit"],
                "rows": report["actual_system_load"]["rows"],
            },
            "intermittent_generation": {
                "source": "emc_metered_generation",
                "semantic": report["intermittent_generation"]["semantic"],
                "raw_unit": report["intermittent_generation"]["raw_unit"],
                "rows": report["intermittent_generation"]["rows"],
            },
            "weather": report["weather"],
        },
        "missing_data_policy": "reject; no imputation",
        "resampling_policy": (
            "not performed in M1.2; later reader must declare any hourly-to-half-hour mapping"
        ),
        "not_frozen": {
            "carbon": (
                "No publicly sourced half-hourly Singapore carbon-intensity series "
                "is frozen by M1.2a."
            ),
            "local_pv_kw": (
                "No local IDC PV measurement is claimed; "
                "weather GHI and national IGS are raw inputs only."
            ),
            "wind_generation_kw": (
                "No local IDC wind-generation measurement is claimed; "
                "wind speed is retained as weather input."
            ),
        },
    }


def write_manifest(report: dict[str, Any], output: Path | str, *, frozen_at_utc: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _manifest_from_report(report, frozen_at_utc=_parse_freeze_timestamp(frozen_at_utc)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def fetch_open_meteo_weather(output: Path, *, year: int = YEAR) -> str:
    """用单次、显式的非商业 Open-Meteo 查询获取原始小时天气 CSV。"""
    query = {
        "latitude": "1.3521",
        "longitude": "103.8198",
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "temperature_2m,wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "timezone": SINGAPORE_TIMEZONE,
        "models": "era5",
        "format": "csv",
    }
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "IDC-M1.2-research-data-freeze/1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed HTTPS source
        content = response.read()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return url


def _copy_input(source: Path | None, destination: Path) -> None:
    if source is None:
        return
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/singapore_2024"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest/singapore_2024.json"))
    parser.add_argument("--usep-zip", type=Path, help="manually downloaded EMC USEP annual ZIP")
    parser.add_argument(
        "--generation-zip", type=Path, help="manually downloaded EMC metered-generation ZIP"
    )
    parser.add_argument("--sasea-zip", type=Path, help="downloaded Zenodo SASEA data.zip")
    parser.add_argument(
        "--fetch-weather", action="store_true", help="make one non-commercial Open-Meteo request"
    )
    parser.add_argument("--verify", action="store_true", help="verify local raw files")
    parser.add_argument(
        "--write-manifest", action="store_true", help="write manifest after successful verification"
    )
    parser.add_argument(
        "--frozen-at-utc",
        help="required ISO-8601 timestamp when writing the immutable manifest",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        _copy_input(args.usep_zip, args.raw_dir / REQUIRED_FILES["emc_usep"])
        _copy_input(args.generation_zip, args.raw_dir / REQUIRED_FILES["emc_metered_generation"])
        _copy_input(args.sasea_zip, args.raw_dir / REQUIRED_FILES["sasea_demand"])
        if args.fetch_weather:
            url = fetch_open_meteo_weather(args.raw_dir / REQUIRED_FILES["open_meteo_weather"])
            print(f"[downloaded] Open-Meteo weather: {url}", file=sys.stderr)
        if not args.verify and not args.write_manifest:
            print(
                "No verification requested; use --verify --write-manifest "
                "after supplying all raw files."
            )
            return 0
        report = validate_singapore_2024(args.raw_dir)
        if args.write_manifest:
            if args.frozen_at_utc is None:
                raise ValueError("--write-manifest requires --frozen-at-utc")
            write_manifest(report, args.manifest, frozen_at_utc=args.frozen_at_utc)
            print(f"[manifest] {args.manifest}", file=sys.stderr)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, ValueError, zipfile.BadZipFile, OSError) as error:
        print(f"M1.2 verification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
