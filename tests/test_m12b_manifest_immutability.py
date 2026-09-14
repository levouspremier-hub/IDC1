"""M1.2b：冻结 manifest 不可变性、原始指纹与未物化单位语义。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.fetch_singapore_data import (  # type: ignore[import-not-found]
    SINGAPORE_TIMEZONE,
    build_manifest,
    usep_sgd_mwh_to_reader_unit,
    verify_existing_manifest,
    write_or_verify_manifest,
)

FROZEN_AT = "2026-09-14T07:28:44+00:00"


def _report() -> dict[str, object]:
    return {
        "year": 2024,
        "timezone": SINGAPORE_TIMEZONE,
        "files": {
            "emc_usep": {"path": "raw/usep.zip", "sha256": "a" * 64, "bytes": 101},
            "emc_metered_generation": {"path": "raw/mg.zip", "sha256": "b" * 64, "bytes": 102},
            "sasea_demand": {"path": "raw/demand.zip", "sha256": "c" * 64, "bytes": 103},
            "open_meteo_weather": {"path": "raw/weather.csv", "sha256": "d" * 64, "bytes": 104},
        },
        "usep": {
            "rows": 17_568,
            "raw_unit": "SGD/MWh",
            "timezone": SINGAPORE_TIMEZONE,
            "semantic": "final wholesale USEP; DEMAND column is forecast only",
        },
        "actual_system_load": {
            "rows": 17_568,
            "raw_unit": "MW",
            "timezone": SINGAPORE_TIMEZONE,
            "semantic": "actual system demand",
            "source": "SASEA raw_SGP_demand.csv",
        },
        "intermittent_generation": {
            "rows": 17_568,
            "raw_unit": "MWh per half-hour settlement period",
            "timezone": SINGAPORE_TIMEZONE,
            "facility_type": "IGS",
            "semantic": "metered national intermittent generation; not solar-only",
        },
        "weather": {
            "rows": 8_784,
            "timezone": SINGAPORE_TIMEZONE,
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


def test_existing_identical_manifest_is_idempotent_and_never_rewritten(tmp_path: Path) -> None:
    report = _report()
    path = tmp_path / "manifest.json"

    assert write_or_verify_manifest(report, path, local_frozen_at_utc=FROZEN_AT) is True
    before = path.read_bytes()
    assert write_or_verify_manifest(report, path, local_frozen_at_utc=FROZEN_AT) is False
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda report: report["files"]["emc_usep"].update({"sha256": "e" * 64}), "sha256"),
        (lambda report: report["usep"].update({"semantic": "wrong semantic"}), "semantic"),
        (lambda report: report.update({"year": 2023}), "year"),
        (lambda report: report["weather"].update({"rows": 8_783}), "rows"),
    ],
)
def test_existing_manifest_rejects_any_raw_or_semantic_difference(
    tmp_path: Path, mutate: object, match: str
) -> None:
    report = _report()
    path = tmp_path / "manifest.json"
    write_or_verify_manifest(report, path, local_frozen_at_utc=FROZEN_AT)
    changed = copy.deepcopy(report)
    mutate(changed)  # type: ignore[operator]

    with pytest.raises(ValueError, match=match):
        verify_existing_manifest(path, changed)
    with pytest.raises(ValueError, match=match):
        write_or_verify_manifest(changed, path, local_frozen_at_utc=FROZEN_AT)


def test_price_contract_is_not_materialized_but_has_numeric_reader_regression() -> None:
    manifest = build_manifest(_report(), local_frozen_at_utc=FROZEN_AT)
    price = manifest["series"]["price"]

    assert "stored_unit" not in price
    assert price == {
        "source": "emc_usep",
        "semantic": "final wholesale USEP; DEMAND column is forecast only",
        "raw_unit": "SGD/MWh",
        "reader_required_target_unit": "SGD/kWh",
        "scale_factor": 0.001,
        "materialized_in_m12": False,
        "rows": 17_568,
    }
    assert usep_sgd_mwh_to_reader_unit(1000.0) == pytest.approx(1.0)


def test_manifest_distinguishes_local_freeze_from_unknown_upstream_and_era5_grid() -> None:
    manifest = build_manifest(_report(), local_frozen_at_utc=FROZEN_AT)
    for source in manifest["sources"].values():
        assert source["local_acquired_or_frozen_at_utc"] == FROZEN_AT
        assert source["upstream_downloaded_at_utc"] == "unknown"

    weather = manifest["sources"]["open_meteo_weather"]
    assert weather["requested_coordinates"] == {"latitude": 1.3521, "longitude": 103.8198}
    assert weather["returned_grid_coordinates"] == {"latitude": "1.5", "longitude": "103.75"}
    assert weather["spatial_representation"] == (
        "ERA5 grid-cell proxy for a Singapore national scenario; not an IDC on-site observation."
    )
    assert "local PV" in weather["not_claimed"]
    assert "wind generation" in weather["not_claimed"]


def test_manifest_is_canonical_json_for_idempotence(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    report = _report()
    write_or_verify_manifest(report, path, local_frozen_at_utc=FROZEN_AT)
    assert json.loads(path.read_text(encoding="utf-8")) == build_manifest(
        report, local_frozen_at_utc=FROZEN_AT
    )
