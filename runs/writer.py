"""M7.1 运行产物 writer。

任一 smoke/eval 输出完整 §1.4 文件；Parquet 可由 DuckDB 读取；同 run id 不覆盖已有成功结果；
失败运行也有 manifest 与失败分类。大 checkpoint/raw data 不入 git。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_run(
    run_id: str,
    *,
    config: dict,
    metrics: pd.DataFrame,
    report: dict,
    base_dir: str = "runs",
    seed: int = 0,
    command: str = "",
    data_hash: str | None = None,
    scenario_hash: str | None = None,
    dependency_lock_hash: str | None = None,
    status: str = "success",
    failure_classification: str | None = None,
) -> Path:
    run_dir = Path(base_dir) / run_id
    manifest_path = run_dir / "manifest.json"

    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        # 已有**成功**结果不得被**任何**后续状态覆盖（含 status="failed"）。
        # 仅拦 success→success 会让一次失败的复跑销毁此前的成功产物（M5.4c 修复）。
        if existing.get("status") == "success":
            raise FileExistsError(
                f"run {run_id} 已存在成功结果，不得被 status={status!r} 覆盖"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    metrics.to_parquet(run_dir / "metrics.parquet")
    (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "revision": git_revision(),
        "dependency_lock_hash": dependency_lock_hash,
        "data_hash": data_hash,
        "scenario_hash": scenario_hash,
        "seed": seed,
        "command": command,
        "status": status,
        "failure_classification": failure_classification,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir
