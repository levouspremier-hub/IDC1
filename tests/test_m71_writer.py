"""M7.1 测试：run 产物齐全、Parquet 可被 DuckDB 读、不覆盖成功、失败也有 manifest。"""

import json

import duckdb
import pandas as pd
import pytest

from runs.writer import write_run


def _metrics() -> pd.DataFrame:
    return pd.DataFrame({"step": [0, 1, 2], "cost": [1.0, 2.0, 3.0]})


def test_write_run_creates_all_files(tmp_path):
    run_dir = write_run("r1", config={"a": 1}, metrics=_metrics(), report={"x": 1}, base_dir=str(tmp_path))
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.parquet").exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "figures").is_dir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["run_id"] == "r1"
    assert manifest["status"] == "success"
    assert "revision" in manifest


def test_parquet_readable_by_duckdb(tmp_path):
    run_dir = write_run("r2", config={}, metrics=_metrics(), report={}, base_dir=str(tmp_path))
    df = duckdb.sql(f"SELECT * FROM '{run_dir / 'metrics.parquet'}'").df()
    assert len(df) == 3
    assert list(df.columns) == ["step", "cost"]


def test_no_overwrite_existing_success(tmp_path):
    write_run("r3", config={}, metrics=_metrics(), report={}, base_dir=str(tmp_path))
    with pytest.raises(FileExistsError):
        write_run("r3", config={}, metrics=_metrics(), report={}, base_dir=str(tmp_path))


def test_failed_run_has_manifest(tmp_path):
    run_dir = write_run(
        "r4",
        config={},
        metrics=_metrics(),
        report={},
        base_dir=str(tmp_path),
        status="failed",
        failure_classification="infeasible",
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure_classification"] == "infeasible"
