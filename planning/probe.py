"""M4.6 真实性能门禁：环境步耗时、修正器耗时、变量/整数数、内存、rollout 吞吐。

不训练正式策略；结果打印为 JSON。Parquet 产物在 M7.1 统一写入。
"""

from __future__ import annotations

import json
import time
import tracemalloc
from typing import Any

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from planning.model import build_milp, solve_time_indexed_lp, solve_time_indexed_mip
from planning.snapshot_adapter import build_snapshot
from safe_rl.corrector_wrapper import CorrectorWrapper


def main() -> None:
    # 测量参数（**不是**研究结论或生产默认值）：仅用于本次探针的单步时间预算
    probe_corrector_time_limit_s = 5.0
    env = CorrectorWrapper(
        IDCPriceEnv20D(), corrector_time_limit_s=probe_corrector_time_limit_s
    )
    env.reset(seed=0)
    raw: Any = env.env  # 未包装环境
    action = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])

    tracemalloc.start()
    step_times: list[float] = []
    solve_times: list[float] = []
    stage_a_times: list[float] = []
    stage_b_times: list[float] = []
    total_times: list[float] = []
    reasons: list[str] = []
    for _ in range(raw.horizon):
        t0 = time.perf_counter()
        _, _, terminated, truncated, info = env.step(action)
        step_times.append(time.perf_counter() - t0)
        solve_times.append(float(info.get("correction_solve_time_s", 0.0)))
        stage_a_times.append(float(info.get("stage_a_solve_time_s", 0.0)))
        stage_b_times.append(float(info.get("stage_b_solve_time_s", 0.0)))
        total_times.append(
            float(info.get("stage_a_solve_time_s", 0.0))
            + float(info.get("stage_b_solve_time_s", 0.0))
        )
        reasons.append(str(info.get("correction_reason", "")))
        if terminated or truncated:
            break
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    snapshot = build_snapshot(raw)
    model = build_milp(snapshot)

    # M4.3a：时间索引 LP 规划核心——在 episode 中段（非终止步）测规模/耗时/残差
    probe_env = IDCPriceEnv20D()
    probe_env.reset(seed=0)
    probe_env.step(action)
    lp_snapshot = build_snapshot(probe_env)
    lp_result = solve_time_indexed_lp(lp_snapshot)
    mip_result = solve_time_indexed_mip(lp_snapshot)

    report = {
        "n_server_groups": 20,
        "horizon": raw.horizon,
        "n_tasks": len(snapshot.tasks),
        # [遗留] 旧单步 build_milp：仍被 solver.py / corrector.py 使用，
        # **不是**新 H 步规划器规模，仅作对照
        "legacy_single_step_milp_n_vars": int(model.c.shape[0]),
        "legacy_single_step_milp_n_integer": int(model.integrality.sum()),
        "lp_planning_horizon_steps": lp_result.horizon_steps,
        "lp_n_vars": lp_result.n_variables,
        "lp_n_constraints": lp_result.n_constraints,
        "lp_solve_time_s": lp_result.solve_time_s,
        "lp_max_constraint_residual": lp_result.max_constraint_residual,
        "lp_solver_status": lp_result.solver_status,
        "lp_failure_class": lp_result.failure_class,
        "lp_power_approximation_used": lp_result.power_approximation_used,
        "lp_storage_relaxation_active": lp_result.storage_relaxation_active,
        "lp_backend": lp_result.backend,
        "lp_objective_sgd": lp_result.total_objective_sgd,
        "mip_planning_horizon_steps": mip_result.horizon_steps,
        "mip_n_vars": mip_result.n_variables,
        "mip_n_integer": mip_result.n_integer_variables,
        "mip_n_constraints": mip_result.n_constraints,
        "mip_solve_time_s": mip_result.solve_time_s,
        "mip_solver_status": mip_result.solver_status,
        "mip_failure_class": mip_result.failure_class,
        "mip_objective_sgd": mip_result.total_objective_sgd,
        "mip_max_constraint_residual": mip_result.max_constraint_residual,
        "mip_storage_mutual_exclusion_ok": all(
            c * d <= 1e-9
            for c, d in zip(mip_result.charge_kw, mip_result.discharge_kw, strict=True)
        ),
        "mip_backend": mip_result.backend,
        "env_step_mean_s": float(np.mean(step_times)),
        "env_step_p95_s": float(np.percentile(step_times, 95)),
        "probe_corrector_time_limit_s": probe_corrector_time_limit_s,
        "probe_corrector_time_limit_is_measurement_parameter": True,
        "corrector_solve_mean_s": float(np.mean(solve_times)),
        "corrector_solve_p95_s": float(np.percentile(solve_times, 95)),
        "stage_a_solve_mean_s": float(np.mean(stage_a_times)),
        "stage_b_solve_mean_s": float(np.mean(stage_b_times)),
        "corrector_total_mean_s": float(np.mean(total_times)),
        "corrector_timeout_steps": int(sum(1 for r in reasons if r == "timeout")),
        "rollout_steps_per_s": float(len(step_times) / max(sum(step_times), 1e-9)),
        "peak_memory_bytes": int(peak_memory),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
