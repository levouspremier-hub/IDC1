"""M4.1/M4.1a 从环境构建滚动规划输入 SystemSnapshot。

红线：
- adapter 不访问 policy、value 或未来真值；
- 预测只取 `[t, t + forecast_cutoff)` 可见窗口（与 env 观测共用 `visible_window_slice`），
  窗口外零填充；基础负载功率预测超出可见窗口时用**最后可见温度**的持久化假设。
"""

from __future__ import annotations

import numpy as np

from contracts.models import (
    PlanningExogenousForecast,
    ScenarioBundle,
    SystemSnapshot,
    TaskState,
)
from envs.idc_price_env import visible_window_slice

PLANNING_HORIZON_CAP = 24
POWER_APPROXIMATION_NOTE = (
    "规划近似：逐组功率由基础负载功耗与逐组 work capacity 线性化得到；"
    "执行前后必须由环境物理链复核。"
)
EXTENSION_POLICY = (
    "窗口外规划假设（[t, t+forecast_cutoff) 之外，不读未来真值）："
    "pv/wind -> 0（保守）；price/carbon/temperature -> 最后可见值持久化；"
    "arrival -> 0（未来具体任务不进入规划）；base_idc_power -> 按持久化温度重算。"
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


def _planning_extension(
    env, t: int, cutoff: int, n_steps: int
) -> tuple[dict[str, list[float]], list[bool], list[bool]]:
    """时域展开（M4.1c）：窗口内取可见真值，窗口外按 `EXTENSION_POLICY` 假设。

    返回 (各外生量向量, visible_mask, assumed_mask)。窗口外**不读取任何真值**。
    """
    horizon = int(env.horizon)
    start, end = visible_window_slice(t, cutoff, horizon)
    n_visible = min(end - start, n_steps)  # 尾部按实际 planning_horizon_steps 截断
    visible_mask = [k < n_visible for k in range(n_steps)]
    assumed_mask = [not v for v in visible_mask]

    def _series(values) -> list[float]:
        arr = np.asarray(values, dtype=np.float64)
        return [float(arr[t + k]) for k in range(n_visible)]

    price_v = _series(env.price_t)
    pv_v = _series(env.pv_t)
    wind_v = _series(env.wt_t)
    temp_v = _series(env.T_amb)
    carbon_v = _series(env.carbon_factor_t)
    arrival_v = _series(env.task_arrival_forecast)

    # 持久化基准 = 最后一个可见值；若窗口为空则由调用方在此之前拒绝（cutoff<=0）。
    last_price = price_v[-1] if price_v else 0.0
    last_temp = temp_v[-1] if temp_v else 0.0
    last_carbon = carbon_v[-1] if carbon_v else 0.0

    base_load = np.clip(env.base_load, 0.0, 1.0)
    vectors: dict[str, list[float]] = {
        "price": list(price_v),
        "pv": list(pv_v),
        "wind": list(wind_v),
        "temperature": list(temp_v),
        "carbon": list(carbon_v),
        "arrival": list(arrival_v),
        "base_idc_power": [env._idc_power_kw(base_load, temp) for temp in temp_v],
    }
    for _ in range(n_visible, n_steps):
        vectors["price"].append(last_price)
        vectors["pv"].append(0.0)          # 保守
        vectors["wind"].append(0.0)        # 保守
        vectors["temperature"].append(last_temp)
        vectors["carbon"].append(last_carbon)
        vectors["arrival"].append(0.0)     # 未来具体任务不进入规划
        vectors["base_idc_power"].append(env._idc_power_kw(base_load, last_temp))
    return vectors, visible_mask, assumed_mask


def build_snapshot(env) -> SystemSnapshot:
    """从环境当前状态构建规划输入快照（只读当前状态 + 可见预测）。"""
    t = int(env.current_step)
    horizon = int(env.horizon)
    cutoff = int(getattr(env, "forecast_cutoff", 4))
    if cutoff <= 0:
        raise ValueError(
            f"forecast_cutoff={cutoff} 非法：无可见窗口时拒绝构建滚动规划快照"
            "（不得读取 t 时刻真值填补）。"
        )
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

    vectors, visible_mask, assumed_mask = _planning_extension(env, t, cutoff, n_steps)
    planning_forecast = PlanningExogenousForecast(
        horizon_steps=n_steps,
        price=vectors["price"],
        pv=vectors["pv"],
        wind=vectors["wind"],
        temperature=vectors["temperature"],
        carbon=vectors["carbon"],
        arrival=vectors["arrival"],
        base_idc_power=vectors["base_idc_power"],
        visible_mask=visible_mask,
        assumed_mask=assumed_mask,
        extension_policy=EXTENSION_POLICY,
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
        base_idc_power_forecast_kw=vectors["base_idc_power"],
        tasks=tasks,
        forecast=forecast,
        planning_forecast=planning_forecast,
        group_work_capacity=[float(c) for c in c_server],
        group_power_coeff_kw_per_work=[float(c) for c in coeff],
        group_power_upper_kw=[float(p) for p in p_max_kw],
        power_approximation_note=POWER_APPROXIMATION_NOTE,
        budget_remaining_sgd=float(getattr(env, "budget_remaining_sgd", 1e9)),
    )
