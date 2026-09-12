#!/usr/bin/env python
"""M7.2 报告生成：从 run 目录（metrics.parquet + report.json + manifest.json）生成图与报告。

报告只汇总可追溯数值，每句可回链 run id / metric column。
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from viz.figures import generate_all_figures


def build_report(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    metrics = pd.read_parquet(run_dir / "metrics.parquet")

    figures = generate_all_figures(metrics, run_dir / "figures", run_id)

    lines: list[str] = []
    for col in metrics.columns:
        mean = float(metrics[col].mean()) if pd.api.types.is_numeric_dtype(metrics[col]) else None
        lines.append(f"{col}: mean={mean} (run_id={run_id}, metrics.parquet column '{col}')")

    report = {
        "run_id": run_id,
        "n_figures": len(figures),
        "figure_paths": [str(p) for p in figures],
        "summary_lines": lines,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    import sys

    build_report(sys.argv[1] if len(sys.argv) > 1 else "runs/example")
