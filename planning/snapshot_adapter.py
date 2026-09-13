"""M4.1 从环境构建受限 SystemSnapshot。

红线：adapter 不访问 policy、value 或未来真值；可见预测仅取 [t, t+forecast_cutoff) 窗口
（M3.10a 与环境观测共用 `visible_window_slice` 同一定义），长度 == forecast_cutoff
（与 contracts.validators 一致）；缺失预测段（系统负荷）用声明边界假设。
"""

from __future__ import annotations

import numpy as np

from contracts.models import ScenarioBundle, SystemSnapshot, TaskState
from envs.idc_price_env import visible_window_slice


def _visible_window(series: np.ndarray, t: int, cutoff: int, horizon: int) -> list[float]:
    """固定长度 cutoff 的可见窗口；horizon 尾部缺失部分显式零填充。"""
    start, end = visible_window_slice(t, cutoff, horizon)
    window = np.zeros(cutoff, dtype=np.float64)
    seg = np.asarray(series, dtype=np.float64)[start:end]
    window[: len(seg)] = seg
    return window.tolist()


def build_snapshot(env) -> SystemSnapshot:
    """从环境当前状态构建 SystemSnapshot（只读当前状态 + 可见预测）。"""
    t = int(env.current_step)
    horizon = int(env.horizon)
    cutoff = int(getattr(env, "forecast_cutoff", 4))

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
        price_forecast=_visible_window(env.price_t, t, cutoff, horizon),
        load_forecast=[0.0] * cutoff,  # 系统负荷未建模，声明边界假设
        pv_forecast=_visible_window(env.pv_t, t, cutoff, horizon),
        wind_forecast=_visible_window(env.wt_t, t, cutoff, horizon),
        temperature_forecast=_visible_window(env.T_amb, t, cutoff, horizon),
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
