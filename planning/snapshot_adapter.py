"""M4.1/M4.1a 从环境构建滚动规划输入 SystemSnapshot。

红线：
- adapter 不访问 policy、value 或未来真值；
- 预测只取 `[t, t + forecast_cutoff)` 可见窗口（与 env 观测共用 `visible_window_slice`），
  窗口外零填充；基础负载功率预测超出可见窗口时用**最后可见温度**的持久化假设。
"""

from __future__ import annotations

import numpy as np

from contracts.models import ScenarioBundle, SystemSnapshot, TaskState
from envs.idc_price_env import visible_window_slice

PLANNING_HORIZON_CAP = 24
POWER_APPROXIMATION_NOTE = (
    "规划近似：逐组功率由基础负载功耗与逐组 work capacity 线性化得到；"
    "执行前后必须由环境物理链复核。"
)
_DEFAULT_PLANNING_HORIZON = 24


def _visible_window(series: np.ndarray, t: int, cutoff: int, horizon: int) -> list[float]:
    """固定长度 cutoff 的可见窗口；horizon 尾部缺失部分显式零填充。"""
    start, end = visible_window_slice(t, cutoff, horizon)
    window = np.zeros(cutoff, dtype=np.float64)
    seg = np.asarray(series, dtype=np.float64)[start:end]
    window[: len(seg)] = seg
    return window.tolist()


def _planning_horizon(env, cap: int = PLANNING_HORIZON_CAP) -> int:
    remaining = int(env.horizon) - int(env.current_step)
    return max(min(cap, remaining), 0)


def _base_power_forecast(env, t: int, cutoff: int, n_steps: int) -> list[float]:
    """基础负载 IDC 功率预测（kW）：窗口内用真实温度；窗口外用最后可见温度持久化。"""
    base_load = np.clip(env.base_load, 0.0, 1.0)
    horizon = int(env.horizon)
    last_visible = min(t + cutoff - 1, horizon - 1) if cutoff > 0 else t
    last_visible = max(last_visible, t)
    out: list[float] = []
    for k in range(n_steps):
        idx = t + k
        idx = idx if idx < t + cutoff and idx < horizon else last_visible
        idx = min(max(idx, 0), horizon - 1)
        out.append(env._idc_power_kw(base_load, env.T_amb[idx]))
    return out


def build_snapshot(env) -> SystemSnapshot:
    """从环境当前状态构建规划输入快照（只读当前状态 + 可见预测）。"""
    t = int(env.current_step)
    horizon = int(env.horizon)
    cutoff = int(getattr(env, "forecast_cutoff", 4))
    n_steps = _planning_horizon(env)

    tasks = [
        TaskState(
            task_id=str(task.task_id),
            remaining_work=float(task.remaining_work),
            deadline=int(task.latest_finish_time),
            priority=float(task.priority),
            status=str(task.status),
            max_rate_work_per_step=float(task.workload / max(int(task.duration), 1)),
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
        carbon_forecast=_visible_window(env.carbon_factor_t, t, cutoff, horizon),
        arrival_forecast=_visible_window(env.task_arrival_forecast, t, cutoff, horizon),
        source_hashes={"adapter": "planning.snapshot_adapter"},
        synthetic=True,
    )

    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    p_max_kw = np.asarray(env.model.P_max, dtype=np.float64) / 1000.0
    p_idle_kw = np.asarray(env.model.P_idle, dtype=np.float64) / 1000.0
    coeff = (p_max_kw - p_idle_kw) / np.maximum(c_server, 1e-6)

    return SystemSnapshot(
        step=t,
        delta_t_hours=float(env.delta_t_hours),
        planning_horizon_steps=n_steps,
        soc_kwh=float(env.bess_energy_kWh),
        soc_min_kwh=float(env.bess_soc_min * env.bess_capacity_kWh),
        soc_max_kwh=float(env.bess_soc_max * env.bess_capacity_kWh),
        soc_capacity_kwh=float(env.bess_capacity_kWh),
        bess_charge_power_max_kw=float(env.bess_charge_power_max_kW),
        bess_discharge_power_max_kw=float(env.bess_discharge_power_max_kW),
        bess_charge_efficiency=float(env.bess_charge_efficiency),
        bess_discharge_efficiency=float(env.bess_discharge_efficiency),
        bess_degradation_cost_per_kwh=float(env.bess_degradation_cost_per_kWh),
        access_limit_kw=float(env.access_limit_kw),
        base_idc_power_forecast_kw=_base_power_forecast(env, t, cutoff, n_steps),
        tasks=tasks,
        forecast=forecast,
        group_work_capacity=[float(c) for c in c_server],
        group_power_coeff_kw_per_work=[float(c) for c in coeff],
        group_power_upper_kw=[float(p) for p in p_max_kw],
        power_approximation_note=POWER_APPROXIMATION_NOTE,
        budget_remaining_sgd=float(getattr(env, "budget_remaining_sgd", 1e9)),
    )
