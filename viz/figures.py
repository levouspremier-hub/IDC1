"""M7.2 静态图：从 metrics.parquet 生成成本—碳、业务缺口、峰值、可再生利用等图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def generate_all_figures(metrics: pd.DataFrame, out_dir: str | Path, run_id: str) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _save(name: str, fig) -> None:
        p = out / f"{name}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    if "cost" in metrics.columns and "carbon" in metrics.columns:
        fig, ax = plt.subplots()
        ax.scatter(metrics["cost"], metrics["carbon"])
        ax.set_xlabel("cost_sgd")
        ax.set_ylabel("carbon_kg")
        _save("cost_carbon", fig)

    for col, name in [
        ("business_gap", "business_gap"),
        ("peak_kw", "peak"),
        ("renewable_utilization", "renewable"),
        ("solve_time_s", "solve_time"),
    ]:
        if col in metrics.columns:
            fig, ax = plt.subplots()
            ax.plot(metrics[col])
            ax.set_xlabel("step")
            ax.set_ylabel(col)
            _save(name, fig)

    return paths
