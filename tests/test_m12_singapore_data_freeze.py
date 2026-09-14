"""M1.2a：Singapore-2024 原始数据冻结的本地核验契约。"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

from scripts.fetch_singapore_data import (  # type: ignore[import-not-found]
    EXPECTED_HALF_HOUR_ROWS,
    EXPECTED_HOURLY_ROWS,
    SINGAPORE_TIMEZONE,
    main,
    validate_singapore_2024,
    write_manifest,
)


def _half_hours_2024() -> list[datetime]:
    start = datetime.combine(date(2024, 1, 1), time.min)
    return [start + timedelta(minutes=30 * index) for index in range(EXPECTED_HALF_HOUR_ROWS)]


def _hours_2024() -> list[datetime]:
    start = datetime.combine(date(2024, 1, 1), time.min)
    return [start + timedelta(hours=index) for index in range(EXPECTED_HOURLY_ROWS)]


def _csv_text(header: list[str], rows: list[list[str]]) -> str:
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue()


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _write_complete_raw_bundle(raw_dir: Path) -> None:
    raw_dir.mkdir()
    half_hours = _half_hours_2024()

    usep_by_month: dict[int, list[list[str]]] = {month: [] for month in range(1, 13)}
    generation_by_month: dict[int, list[list[str]]] = {month: [] for month in range(1, 13)}
    for timestamp in half_hours:
        period = (timestamp.hour * 2) + (2 if timestamp.minute == 30 else 1)
        date_text = timestamp.strftime("%d-%b-%Y")
        usep_by_month[timestamp.month].append(
            ["USEP", date_text, str(period), "100.0", "0.0", "5000.0", "10.0"]
        )
        generation_by_month[timestamp.month].append(
            ["MG", date_text, str(period), "IGS", "2.0", "1.0"]
        )

    usep_header = [
        "INFORMATION TYPE",
        "DATE",
        "PERIOD",
        "USEP ($/MWh)",
        "LCP ($/MWh)",
        "DEMAND (MW)",
        "SOLAR(MW)",
    ]
    generation_header = [
        "INFORMATION TYPE",
        "DATE",
        "PERIOD",
        "FACILITY TYPE",
        "GROSS INJECTION (MWh)",
        "NET INJECTION (MWh)",
    ]
    _write_zip(
        raw_dir / "emc_usep_2024.zip",
        {
            f"USEP_{date(2024, month, 1).strftime('%b')}-2024.csv": _csv_text(
                usep_header, usep_by_month[month]
            )
            for month in range(1, 13)
        },
    )
    _write_zip(
        raw_dir / "emc_metered_generation_2024.zip",
        {
            f"MG_{date(2024, month, 1).strftime('%b')}-2024.csv": _csv_text(
                generation_header, generation_by_month[month]
            )
            for month in range(1, 13)
        },
    )
    _write_zip(
        raw_dir / "sasea_demand_2024.zip",
        {
            "data/raw/raw_SGP_demand.csv": _csv_text(
                ["datetime", "system_demand"],
                [[timestamp.strftime("%Y-%m-%d %H:%M:%S"), "5000.0"] for timestamp in half_hours],
            )
        },
    )
    weather_rows = [
        [timestamp.strftime("%Y-%m-%dT%H:%M"), "27.0", "2.0", "100.0"]
        for timestamp in _hours_2024()
    ]
    weather = (
        "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
        "1.35,103.82,15.0,28800,Asia/Singapore,GMT+8\n\n"
        + _csv_text(
            [
                "time",
                "temperature_2m (°C)",
                "wind_speed_10m (m/s)",
                "shortwave_radiation (W/m²)",
            ],
            weather_rows,
        )
    )
    (raw_dir / "open_meteo_era5_2024.csv").write_text(weather, encoding="utf-8")


@pytest.fixture
def complete_bundle(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    _write_complete_raw_bundle(raw_dir)
    return raw_dir


def test_expected_calendar_sizes_and_timezone_are_fixed() -> None:
    assert EXPECTED_HALF_HOUR_ROWS == 17_568
    assert EXPECTED_HOURLY_ROWS == 8_784
    assert SINGAPORE_TIMEZONE == "Asia/Singapore"


def test_complete_bundle_passes_and_keeps_actual_load_semantics(complete_bundle: Path) -> None:
    report = validate_singapore_2024(complete_bundle)

    assert report["year"] == 2024
    assert report["actual_system_load"]["source"] == "SASEA raw_SGP_demand.csv"
    assert report["actual_system_load"]["rows"] == EXPECTED_HALF_HOUR_ROWS
    assert report["usep"]["unit_conversion"] == "SGD/MWh -> SGD/kWh (/1000)"
    assert report["weather"]["rows"] == EXPECTED_HOURLY_ROWS
    assert report["weather"]["timezone"] == SINGAPORE_TIMEZONE
    assert report["intermittent_generation"]["facility_type"] == "IGS"


def test_missing_weather_hour_is_rejected_without_imputation(complete_bundle: Path) -> None:
    weather_path = complete_bundle / "open_meteo_era5_2024.csv"
    lines = weather_path.read_text(encoding="utf-8").splitlines()
    weather_path.write_text("\n".join(lines[:4] + lines[5:]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="weather.*continuous"):
        validate_singapore_2024(complete_bundle)


def test_missing_half_hour_load_is_rejected_without_imputation(complete_bundle: Path) -> None:
    demand_path = complete_bundle / "sasea_demand_2024.zip"
    with zipfile.ZipFile(demand_path) as archive:
        rows = archive.read("data/raw/raw_SGP_demand.csv").decode("utf-8").splitlines()
    _write_zip(demand_path, {"data/raw/raw_SGP_demand.csv": "\n".join(rows[:-1]) + "\n"})

    with pytest.raises(ValueError, match="actual system load.*continuous"):
        validate_singapore_2024(complete_bundle)


def test_manifest_records_permissions_and_no_silent_resampling(
    complete_bundle: Path, tmp_path: Path
) -> None:
    report = validate_singapore_2024(complete_bundle)
    manifest_path = tmp_path / "singapore_2024.json"
    write_manifest(report, manifest_path, frozen_at_utc="2026-09-14T07:28:44+00:00")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["series"]["load"]["semantic"] == "actual system demand"
    assert manifest["series"]["weather"]["resampling"] == "not performed in M1.2"
    assert manifest["sources"]["emc_usep"]["automation"] == "manual browser download only"
    assert manifest["missing_data_policy"] == "reject; no imputation"
    for source in manifest["sources"].values():
        assert source["local_acquired_or_frozen_at_utc"]
        assert source["upstream_downloaded_at_utc"] == "unknown"
        assert source["timezone"] == SINGAPORE_TIMEZONE
        assert source["raw_units"]


def test_cli_verification_output_is_machine_readable_json(
    complete_bundle: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        [
            "--raw-dir",
            str(complete_bundle),
            "--manifest",
            str(tmp_path / "singapore_2024.json"),
            "--verify",
            "--write-manifest",
            "--frozen-at-utc",
            "2026-09-14T07:28:44+00:00",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["year"] == 2024


def test_cli_refuses_to_mutate_frozen_manifest_without_explicit_freeze_time(
    complete_bundle: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(
        [
            "--raw-dir",
            str(complete_bundle),
            "--manifest",
            str(tmp_path / "singapore_2024.json"),
            "--verify",
            "--write-manifest",
        ]
    ) == 2
    assert "--frozen-at-utc" in capsys.readouterr().err


def test_cli_verify_rejects_raw_hash_mismatch_against_existing_manifest(
    complete_bundle: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "singapore_2024.json"
    write_manifest(
        validate_singapore_2024(complete_bundle),
        manifest_path,
        frozen_at_utc="2026-09-14T07:28:44+00:00",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["verification"]["raw_files"]["emc_usep"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert (
        main(["--raw-dir", str(complete_bundle), "--manifest", str(manifest_path), "--verify"])
        == 2
    )
    assert "sha256" in capsys.readouterr().err
