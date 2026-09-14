"""M1.2c：已有成功 manifest 后，所有 raw 写入入口必须在调用前被封锁。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import fetch_singapore_data as singapore_data  # type: ignore[import-not-found]

FROZEN_AT = "2026-09-14T07:28:44+00:00"


def _report() -> dict[str, object]:
    return {
        "year": 2024,
        "timezone": singapore_data.SINGAPORE_TIMEZONE,
        "files": {
            "emc_usep": {"path": "raw/usep.zip", "sha256": "a" * 64, "bytes": 101},
            "emc_metered_generation": {"path": "raw/mg.zip", "sha256": "b" * 64, "bytes": 102},
            "sasea_demand": {"path": "raw/demand.zip", "sha256": "c" * 64, "bytes": 103},
            "open_meteo_weather": {"path": "raw/weather.csv", "sha256": "d" * 64, "bytes": 104},
        },
        "usep": {
            "rows": 17_568,
            "raw_unit": "SGD/MWh",
            "timezone": singapore_data.SINGAPORE_TIMEZONE,
            "semantic": "final wholesale USEP; DEMAND column is forecast only",
        },
        "actual_system_load": {
            "rows": 17_568,
            "raw_unit": "MW",
            "timezone": singapore_data.SINGAPORE_TIMEZONE,
            "semantic": "actual system demand",
            "source": "SASEA raw_SGP_demand.csv",
        },
        "intermittent_generation": {
            "rows": 17_568,
            "raw_unit": "MWh per half-hour settlement period",
            "timezone": singapore_data.SINGAPORE_TIMEZONE,
            "facility_type": "IGS",
            "semantic": "metered national intermittent generation; not solar-only",
        },
        "weather": {
            "rows": 8_784,
            "timezone": singapore_data.SINGAPORE_TIMEZONE,
            "model": "ERA5",
            "raw_units": {
                "temperature_2m": "degC",
                "wind_speed_10m": "m/s",
                "shortwave_radiation": "W/m2 (hourly preceding-hour mean GHI)",
            },
            "coordinates_returned": {"latitude": "1.5", "longitude": "103.75"},
            "resampling": "not performed in M1.2",
        },
    }


@pytest.mark.parametrize(
    "write_args",
    [
        ("--usep-zip", "candidate-usep.zip"),
        ("--generation-zip", "candidate-generation.zip"),
        ("--sasea-zip", "candidate-demand.zip"),
        ("--fetch-weather",),
    ],
)
def test_existing_manifest_rejects_every_raw_write_option_before_any_write_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    write_args: tuple[str, ...],
) -> None:
    manifest = tmp_path / "singapore_2024.json"
    singapore_data.write_or_verify_manifest(_report(), manifest, local_frozen_at_utc=FROZEN_AT)
    before = manifest.read_bytes()
    copy_input = Mock()
    fetch_weather = Mock(return_value="https://example.invalid/weather")
    monkeypatch.setattr(singapore_data, "_copy_input", copy_input)
    monkeypatch.setattr(singapore_data, "fetch_open_meteo_weather", fetch_weather)

    assert singapore_data.main(
        [
            "--raw-dir",
            str(tmp_path / "raw"),
            "--manifest",
            str(manifest),
            "--verify",
            *write_args,
        ]
    ) == 2
    copy_input.assert_not_called()
    fetch_weather.assert_not_called()
    assert "existing manifest" in capsys.readouterr().err
    assert manifest.read_bytes() == before


def test_existing_manifest_verify_and_idempotent_write_are_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "singapore_2024.json"
    report = _report()
    singapore_data.write_or_verify_manifest(report, manifest, local_frozen_at_utc=FROZEN_AT)
    before = manifest.read_bytes()
    validate = Mock(return_value=report)
    copy_input = Mock()
    fetch_weather = Mock()
    monkeypatch.setattr(singapore_data, "validate_singapore_2024", validate)
    monkeypatch.setattr(singapore_data, "_copy_input", copy_input)
    monkeypatch.setattr(singapore_data, "fetch_open_meteo_weather", fetch_weather)

    assert singapore_data.main(
        [
            "--raw-dir",
            str(tmp_path / "raw"),
            "--manifest",
            str(manifest),
            "--verify",
            "--write-manifest",
            "--frozen-at-utc",
            FROZEN_AT,
        ]
    ) == 0
    assert "verified existing" in capsys.readouterr().err
    validate.assert_called_once()
    copy_input.assert_not_called()
    fetch_weather.assert_not_called()
    assert manifest.read_bytes() == before


def test_missing_manifest_keeps_first_freeze_copy_then_verify_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    manifest = tmp_path / "singapore_2024.json"
    report = _report()
    copy_input = Mock()
    validate = Mock(return_value=report)
    monkeypatch.setattr(singapore_data, "_copy_input", copy_input)
    monkeypatch.setattr(singapore_data, "validate_singapore_2024", validate)

    assert singapore_data.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--manifest",
            str(manifest),
            "--usep-zip",
            "candidate-usep.zip",
            "--generation-zip",
            "candidate-generation.zip",
            "--sasea-zip",
            "candidate-demand.zip",
            "--verify",
            "--write-manifest",
            "--frozen-at-utc",
            FROZEN_AT,
        ]
    ) == 0
    assert copy_input.call_count == 3
    validate.assert_called_once_with(raw_dir)
    assert manifest.is_file()
