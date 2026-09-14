"""M1.2c：已有成功 manifest 后，所有 raw 写入入口必须在调用前被封锁。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import fetch_singapore_data as singapore_data  # type: ignore[import-not-found]
from tests.test_m12b_manifest_immutability import FROZEN_AT, _report


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
