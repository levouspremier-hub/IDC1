"""M4.1 从环境构建受限 SystemSnapshot。

红线：adapter 不访问 policy、value 或未来真值；可见预测仅取 [0, t+forecast_cutoff] 窗口，
其余置 0；缺失预测段（系统负荷）使用预先声明的边界假设（全 0）。
"""

from __future__ import annotations

import numpy as np

from contracts.models import ScenarioBundle, SystemSnapshot, TaskState


def _visible_mask(horizon: int, t: int, cutoff: int) -> np.ndarray:
    mask = np.zeros(horizon, dtype=np.float64)
    mask[: min(t + cutoff + 1, horizon)] = 1.0
    return mask


def build_snapshot(env) -> SystemSnapshot:
    """从环境当前状态构建 SystemSnapshot（只读当前状态 + 可见预测）。"""
    t = int(env.current_step)
    horizon = int(env.horizon)
    cutoff = int(getattr(env, "forecast_cutoff", 4))
    visible = _visible_mask(horizon, t, cutoff)

    tasks = [
        TaskState(
            task_id=str(task.task_id),
            remaining_work=float(task.remaining_work),
            deadline=int(task.latest_finish_time),
            priority=float(task.priority),
            status=str(task.status),
        )
        for task in env.tasks
        if task.status != "not_arrived"
    ]

    forecast = ScenarioBundle(
        split="train",
        start=str(t),
        horizon=horizon,
        forecast_cutoff=cutoff,
        price_forecast=(np.asarray(env.price_t, dtype=np.float64) * visible).tolist(),
        load_forecast=[0.0] * horizon,  # 系统负荷未建模，声明边界假设
        pv_forecast=(np.asarray(env.pv_t, dtype=np.float64) * visible).tolist(),
        wind_forecast=(np.asarray(env.wt_t, dtype=np.float64) * visible).tolist(),
        temperature_forecast=(np.asarray(env.T_amb, dtype=np.float64) * visible).tolist(),
        source_hashes={"adapter": "planning.snapshot_adapter"},
        synthetic=True,
    )

    return SystemSnapshot(
        step=t,
        soc_kwh=float(env.bess_energy_kWh),
        soc_min_kwh=float(env.bess_soc_min * env.bess_capacity_kWh),
        soc_max_kwh=float(env.bess_soc_max * env.bess_capacity_kWh),
        group_capacity_kw=[float(c) for c in np.asarray(env.model.C_server, dtype=np.float64)],
        access_limit_kw=float(env.access_limit_kw),
        budget_remaining_sgd=float(getattr(env, "budget_remaining_sgd", 1e9)),
        tasks=tasks,
        forecast=forecast,
    )
