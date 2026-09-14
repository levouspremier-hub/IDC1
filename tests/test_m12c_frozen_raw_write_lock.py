"""M1.2c：已有成功 manifest 后，所有 raw 写入入口必须在调用前被封锁。"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import fetch_singapore_data as singapore_data  # type: ignore[import-not-found]
from tests.test_m12_singapore_data_freeze import _write_complete_raw_bundle
from tests.test_m12b_manifest_immutability import FROZEN_AT, _report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir = tmp_path / "raw"
    _write_complete_raw_bundle(raw_dir)
    manifest = tmp_path / "singapore_2024.json"
    singapore_data.write_or_verify_manifest(
        singapore_data.validate_singapore_2024(raw_dir),
        manifest,
        local_frozen_at_utc=FROZEN_AT,
    )
    manifest_hash_before = _sha256(manifest)
    raw_hashes_before = sorted(_sha256(path) for path in raw_dir.iterdir())

    assert singapore_data.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--manifest",
            str(manifest),
            "--verify",
            "--write-manifest",
            "--frozen-at-utc",
            FROZEN_AT,
        ]
    ) == 0
    assert "verified existing" in capsys.readouterr().err
    assert _sha256(manifest) == manifest_hash_before
    assert sorted(_sha256(path) for path in raw_dir.iterdir()) == raw_hashes_before


def test_missing_manifest_keeps_first_freeze_copy_then_verify_flow(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    raw_dir = tmp_path / "raw"
    _write_complete_raw_bundle(source)
    raw_dir.mkdir()
    shutil.copyfile(source / "open_meteo_era5_2024.csv", raw_dir / "open_meteo_era5_2024.csv")
    manifest = tmp_path / "singapore_2024.json"

    assert singapore_data.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--manifest",
            str(manifest),
            "--usep-zip",
            str(source / "emc_usep_2024.zip"),
            "--generation-zip",
            str(source / "emc_metered_generation_2024.zip"),
            "--sasea-zip",
            str(source / "sasea_demand_2024.zip"),
            "--verify",
            "--write-manifest",
            "--frozen-at-utc",
            FROZEN_AT,
        ]
    ) == 0
    assert manifest.is_file()
