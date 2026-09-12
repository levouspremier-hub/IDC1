"""M6.2 统一评估适配器。

规则、独立滚动优化、旧惩罚 PPO、新 PPO+单步、新 PPO+联合修正均输出同一 EvaluationRecord。
不达服务标准的运行带 service_qualified=false，不混入同服务成本比较。
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from contracts.models import EvaluationRecord


def neutral_rule(obs: np.ndarray, action_dim: int = 21) -> np.ndarray:
    """规则基线：中性动作（compute 0.5、storage 0）。"""
    action = np.full(action_dim, 0.5, dtype=np.float32)
    action[20] = 0.0
    return action


def evaluate(
    env,
    method: str,
    action_fn: Callable[[np.ndarray], np.ndarray],
    *,
    run_id: str = "eval-0",
    service_qualified: bool = True,
    seed: int = 0,
) -> EvaluationRecord:
    """跑一个 episode，聚合环境累计量输出 EvaluationRecord。"""
    env.reset(seed=seed)
    solve_times: list[float] = []

    for _ in range(env.unwrapped.horizon if hasattr(env, "unwrapped") else env.horizon):
        obs = env._get_obs()
        t0 = time.perf_counter()
        action = action_fn(np.asarray(obs, dtype=np.float32))
        solve_times.append(time.perf_counter() - t0)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    renewable_utilization = (env.total_pv_used_kWh + env.total_wind_used_kWh) / max(
        env.total_idc_energy_kWh, 1e-9
    )
    reliability = env.total_completed_work / max(
        sum(t.remaining_work for t in env.tasks) + env.total_completed_work, 1e-9
    )

    return EvaluationRecord(
        method=method,
        run_id=run_id,
        service_qualified=service_qualified,
        total_cost_sgd=float(env.total_cost),
        total_carbon_kg=float(env.total_carbon_emission),
        renewable_utilization=float(renewable_utilization),
        peak_kw=float(env.episode_peak_power_kW),
        reliability=float(reliability),
        solve_time_avg_s=float(np.mean(solve_times)) if solve_times else 0.0,
        failure_classification=None,
    )
