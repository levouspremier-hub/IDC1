"""M1.3g-f-c-h2：corrector-on 服务轨迹重放（修复前 / 修复后对照）。

固定 `server_seed = 1`（与 24-origin 标定一致），在 origin 48 / 4848 / 10176 上
跑 corrector-on 轨迹，报告完成量、SLA、timeout/失败。

**`corrector=off` 不在此脚本内**（它只用于定位动作作用，不是基线）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT = "train"
ORIGINS: tuple[int, ...] = (48, 4848, 10176)
HORIZON = 48
FORECAST_CUTOFF = 48
DELTA_T_HOURS = 0.5
COMPUTE = 0.5          # 固定 raw 提案（corrector-on 轨迹）
FIXED_SEED = 0
SEED_OFFSETS = {"task": 0, "server": 1, "forecast": 300000}
CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}


def start_for_origin(origin: int) -> str:
    from scenario.formal_scenario_b6 import _canonical_parquet_path  # noqa: PLC2701

    stamps = pd.DatetimeIndex(pd.read_parquet(_canonical_parquet_path())["timestamp"])
    return stamps[origin].isoformat()


def run(origin: int, budget: float) -> dict[str, Any]:
    import importlib

    from safe_rl.corrector_wrapper import CorrectorWrapper
    from safe_rl_v2.rollout import BUSINESS_VIOLATION_INFO_KEY, CARBON_EMISSION_INFO_KEY

    injection_module = importlib.import_module("scenario.env_injection")
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    inj = injection_module.build_verified_formal_env_injection(
        SPLIT, start=start_for_origin(origin), horizon=HORIZON,
        forecast_cutoff=FORECAST_CUTOFF)
    env = env_cls(horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF,
                  delta_t_hours=DELTA_T_HOURS, formal_injection=inj,
                  task_seed=FIXED_SEED + SEED_OFFSETS["task"],
                  server_seed=FIXED_SEED + SEED_OFFSETS["server"],
                  forecast_seed=FIXED_SEED + SEED_OFFSETS["forecast"])
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=float(budget))
    wrapped.reset(seed=FIXED_SEED)
    action = np.zeros(env.action_dim, dtype=np.float64)
    action[: env.model.N] = COMPUTE

    completed = 0.0
    sla = 0
    timeouts = 0
    fallbacks = 0
    steps = 0
    # M1.3g-f-c-h2-R1（P2）：碳排必须**逐步累加**；旧写法 `sum([x])` 恒等于最后一步的值。
    carbon_total = 0.0
    for _ in range(HORIZON):
        _o, _r, term, trunc, info = wrapped.step(action)
        completed += float(info["completed_work"])
        sla += int(info[BUSINESS_VIOLATION_INFO_KEY])
        carbon_total += float(info[CARBON_EMISSION_INFO_KEY])
        if str(info.get("correction_reason")) == "timeout":
            timeouts += 1
        if str(info.get("correction_reason")) in (
                "timeout", "base_shortage", "solver_failure", "proposal_invalid"):
            fallbacks += 1
        steps += 1
        if term or trunc:
            break
    return {"origin": origin, "steps": steps, "completed_work": completed,
            "sla_total": sla, "carbon_total": carbon_total,
            "timeouts": timeouts, "zero_action_fallbacks": fallbacks,
            "ledger_plus_backlog": sum(inj.ledger_micro) / 1e6 + float(env.initial_Q)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.replay_corrector_service",
        description="corrector-on 服务轨迹重放（M1.3g-f-c-h2）")
    parser.add_argument("--run-id", default="m13gch2_corrector_replay")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from planning.corrector import resolve_corrector_budget
    from runs.writer import write_run

    budget, source = resolve_corrector_budget(None, enabled=True)
    base = REPO_ROOT / args.base_dir

    try:
        rows = [run(o, float(budget)) for o in ORIGINS]
        report = {"entry": "python -m scripts.replay_corrector_service",
                  "statement": ("corrector-on 固定提案轨迹重放；**不是**训练结果。"),
                  "claims": dict(CLAIMS), "origins": list(ORIGINS),
                  "compute": COMPUTE, "server_seed": SEED_OFFSETS["server"],
                  "corrector_time_limit_s": float(budget),
                  "corrector_time_limit_source": source, "rows": rows}
    except Exception as exc:  # noqa: BLE001
        write_run(args.run_id, config={"run_id": args.run_id, "status": "failed"},
                  metrics=pd.DataFrame(), base_dir=str(base), seed=FIXED_SEED,
                  command=f"python -m scripts.replay_corrector_service --run-id {args.run_id}",
                  report={"entry": "python -m scripts.replay_corrector_service",
                          "claims": dict(CLAIMS),
                          "failure": f"{type(exc).__name__}: {exc}"},
                  status="failed", failure_classification=type(exc).__name__)
        print(f"重放失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    run_dir = write_run(args.run_id, config=report, metrics=pd.DataFrame(rows),
                        report=report, base_dir=str(base), seed=FIXED_SEED,
                        command=("python -m scripts.replay_corrector_service "
                                 f"--run-id {args.run_id}"),
                        status="success")
    print(f"run 产物：{run_dir}")
    for r in rows:
        print(f"  origin {r['origin']}: completed={r['completed_work']:.6f} "
              f"sla={r['sla_total']} timeouts={r['timeouts']} "
              f"fallbacks={r['zero_action_fallbacks']} steps={r['steps']} "
              f"| 账本+积压={r['ledger_plus_backlog']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
