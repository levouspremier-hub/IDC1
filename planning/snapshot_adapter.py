"""M4.1/M4.1a 从环境构建滚动规划输入 SystemSnapshot。

> ⚠️ **M1.3e：本 adapter 是 `oracle_debug` / dev-only，不能标 formal。**
> 它读取的「可见预测」**就是** `env` 的真值数组 `[t, t+forecast_cutoff)`，
> 且 `load_forecast` 仍为全零占位。因此它产出的 `ScenarioBundle` 恒为
> `mode="oracle_debug"`，并被 `contracts.validators.validate_forecast_purpose`
> 在 `purpose=training` / `purpose=evaluation` 时**拒绝**。
> 本卡（M1.3e）**不**在 adapter 中伪造正式预测：正式 causal forecast 接线属 **M1.3g**。

红线：
- adapter 不访问 policy、value 或未来真值；
- 预测只取 `[t, t + forecast_cutoff)` 可见窗口（与 env 观测共用 `visible_window_slice`），
  窗口外零填充；基础负载功率预测超出可见窗口时用**最后可见温度**的持久化假设。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from contracts.models import (
    BUNDLE_FORECAST_FIELDS,
    ArtifactDigest,
    ForecastSeriesProvenance,
    PlanningExogenousForecast,
    ScenarioBundle,
    ScenarioForecastProvenance,
    SystemSnapshot,
    TaskState,
)
from envs.idc_price_env import visible_window_slice

# oracle-debug 快照没有真实时间轴：`env.current_step` 是步索引，不是墙上时钟。
# 这里使用一个**显式、非物理**的 dev 锚点，绝不冒充 canonical 时间轴。
ORACLE_DEBUG_ANCHOR = "2024-01-01T00:00:00+08:00"
ORACLE_DEBUG_NOTE = (
    "oracle_debug / dev-only：可见预测直接取自 env 真值窗口，load_forecast 为全零占位；"
    "不能标 formal，不得用于训练或评估（M1.3e）。正式 causal forecast 接线属 M1.3g。"
)
ORACLE_DEBUG_DIGEST_ROLE = "oracle_debug_env_truth_window"
ORACLE_DEBUG_DIGEST_PATH = "envs/idc_price_env.py://visible_truth_window"
ADAPTER_LOGICAL_PATH = "planning/snapshot_adapter.py"

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


def _adapter_revision() -> str:
    """本 adapter 实现的 Git revision（40 位小写 SHA）；**不用**漂移的 HEAD。"""
    revision = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", ADAPTER_LOGICAL_PATH],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"adapter revision 无效：{revision!r}")
    return revision


def _oracle_debug_bundle(
    env, *, t: int, horizon: int, cutoff: int, delta_t_hours: float
) -> ScenarioBundle:
    """构造**明确标注** `mode="oracle_debug"` 的可见窗口 bundle（dev-only）。

    七个序列的来源类别**全部**是 `oracle_debug`：这里的「可见预测」就是 env 真值窗口，
    `load_forecast` 仍是全零占位。它**不是**正式 forecast，`purpose=training/evaluation`
    时会被 purpose gate 拒绝。
    """
    windows: dict[str, list[float]] = {
        "price_forecast": _visible_window(env.price_t, t, cutoff, horizon),
        "load_forecast": [0.0] * cutoff,  # 系统负荷未建模：显式零占位（dev-only）
        "pv_forecast": _visible_window(env.pv_t, t, cutoff, horizon),
        "wind_forecast": _visible_window(env.wt_t, t, cutoff, horizon),
        "temperature_forecast": _visible_window(env.T_amb, t, cutoff, horizon),
        "carbon_forecast": _visible_window(env.carbon_factor_t, t, cutoff, horizon),
        "arrival_forecast": _visible_window(env.task_arrival_forecast, t, cutoff, horizon),
    }

    anchor = datetime.fromisoformat(ORACLE_DEBUG_ANCHOR) + timedelta(
        hours=float(t) * delta_t_hours
    )
    generated_at = anchor.isoformat()
    target_end_exclusive = (
        anchor + timedelta(hours=max(int(cutoff), 1))
    ).isoformat()
    window_digest = hashlib.sha256(
        json.dumps(windows, sort_keys=True).encode("utf-8")
    ).hexdigest()
    digest = ArtifactDigest(
        role=ORACLE_DEBUG_DIGEST_ROLE,
        logical_path=ORACLE_DEBUG_DIGEST_PATH,
        sha256=window_digest,
    )
    revision = _adapter_revision()
    provenance = {
        name: ForecastSeriesProvenance(
            series_name=name,
            source_kind="oracle_debug",
            method="oracle_debug_env_truth_window",
            generated_at=generated_at,
            information_cutoff_exclusive=generated_at,
            target_start=generated_at,
            target_end_exclusive=target_end_exclusive,
            lookback_start=None,
            lookback_end_exclusive=generated_at,
            model_name="planning.snapshot_adapter._oracle_debug_bundle",
            model_version="v1",
            code_revision=revision,
            seed=None,
            sources=[digest],
        )
        for name in BUNDLE_FORECAST_FIELDS
    }

    return ScenarioBundle(
        split="train",
        start=str(t),
        horizon=horizon,
        forecast_cutoff=cutoff,
        price_forecast=windows["price_forecast"],
        load_forecast=windows["load_forecast"],
        pv_forecast=windows["pv_forecast"],
        wind_forecast=windows["wind_forecast"],
        temperature_forecast=windows["temperature_forecast"],
        carbon_forecast=windows["carbon_forecast"],
        arrival_forecast=windows["arrival_forecast"],
        mode="oracle_debug",
        generated_at=generated_at,
        forecast_provenance=ScenarioForecastProvenance(**provenance),
    )


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

    forecast = _oracle_debug_bundle(
        env,
        t=t,
        horizon=horizon,
        cutoff=cutoff,
        delta_t_hours=float(env.delta_t_hours),
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
