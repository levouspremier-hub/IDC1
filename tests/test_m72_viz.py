"""M7.2 测试：从 metrics.parquet 生成图与报告，报告可回链 run id。"""

import pandas as pd

from runs.writer import write_run
from scripts.build_report import build_report


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [0, 1, 2, 3],
            "cost": [1.0, 1.5, 2.0, 2.5],
            "carbon": [0.5, 0.6, 0.7, 0.8],
            "business_gap": [0.0, 0.1, 0.0, 0.2],
            "peak_kw": [10.0, 11.0, 12.0, 11.5],
            "renewable_utilization": [0.0, 0.1, 0.2, 0.1],
            "solve_time_s": [0.001, 0.001, 0.002, 0.001],
        }
    )


def test_build_report_generates_figures_and_report(tmp_path):
    run_dir = write_run(
        "r1", config={}, metrics=_metrics(), report={}, base_dir=str(tmp_path)
    )
    report = build_report(run_dir)
    assert report["run_id"] == "r1"
    assert report["n_figures"] >= 4  # cost_carbon + business_gap + peak + renewable + solve_time
    assert any("r1" in line for line in report["summary_lines"])
