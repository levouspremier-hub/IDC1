"""M4.1/M4.1a 从环境构建滚动规划输入 SystemSnapshot。

**两条路径互不混用**（M1.3g-f-c-h2-R1）：

- **legacy**（`env.formal is False`）：规划输入取 realized 真值
  `price_t` / `T_amb` / `pv_t` / `wt_t` / `carbon_factor_t`；
  `snapshot.forecast` 是 `mode="oracle_debug"` 的占位 bundle。
- **正式链**（`env.formal is True`）：规划输入只取**已验签 B6 causal forecast**
  通道 `env.*_forecast_t` / `env.task_arrival_forecast`；
  `snapshot.forecast` 是经正式入口重建的 `mode="formal"` bundle。

> ⚠️ **legacy 路径是 `oracle_debug` / dev-only，不能标 formal。** 它读取的「可见预测」
> **就是** `env` 的真值数组 `[t, t+forecast_cutoff)`（legacy 下 `*_forecast_t` 只是
> 真值的别名），因此它产出的 `ScenarioBundle` 恒为 `mode="oracle_debug"`，并被
> `contracts.validators.validate_forecast_purpose` 在 `purpose=training` /
> `purpose=evaluation` 时**拒绝**。该路径语义**逐字保持不变**。

**formal 路径（本卡修复）**：formal 的规划输入**只能**取自 `env` 上由
`build_verified_formal_env_injection` → `build_formal_scenario_b6` 注入的
**causal forecast** 通道；`snapshot.forecast` 则由同一条正式入口**重建**为
`mode="formal"` 的 bundle（真实 provenance、真实时间窗口，可通过 training purpose gate）。
**不得**仅把 `oracle_debug` 改名成 formal。

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
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np

from contracts.inventory import InventorySnapshot, TerminalInventory
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
# M1.3e-R3：`logical_path` 必须是规范 POSIX 逻辑路径（不得含 `://` 这类空片段）
ORACLE_DEBUG_DIGEST_PATH = "envs/idc_price_env.py"
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

# --- 规划输入的两个来源通道（M1.3g-f-c-h2-R1） --------------------------------
#
# formal 环境上 `*_forecast_t` 由 `build_verified_formal_env_injection` 从
# **已验证的 B6 ScenarioBundle** 注入，是**决策可见**的因果预测；而
# `price_t` / `T_amb` / `pv_t` / `wt_t` / `carbon_factor_t` 是 **realized 真值**
# （其 `[t+1, ...)` 部分对未来不可见）。两条通道**不得混用**。
_FORMAL_SOURCE_ATTRS: dict[str, str] = {
    "price": "price_forecast_t",
    "pv": "pv_forecast_t",
    "wind": "wind_forecast_t",
    "temperature": "temperature_forecast_t",
    "carbon": "carbon_forecast_t",
    "arrival": "task_arrival_forecast",
}
# legacy：读 realized 真值窗口（**逐字保持原语义**，既有 leakage 回归依赖它）
_LEGACY_SOURCE_ATTRS: dict[str, str] = {
    "price": "price_t",
    "pv": "pv_t",
    "wind": "wt_t",
    "temperature": "T_amb",
    "carbon": "carbon_factor_t",
    "arrival": "task_arrival_forecast",
}


def _is_formal(env) -> bool:
    """是否走正式（B6 因果预测）链；缺失 `formal` 属性即 legacy。"""
    return bool(getattr(env, "formal", False))


def _source_attrs(env) -> dict[str, str]:
    return _FORMAL_SOURCE_ATTRS if _is_formal(env) else _LEGACY_SOURCE_ATTRS


@lru_cache(maxsize=256)
def _verified_formal_bundle(
    split: str, local_origin: int, horizon: int, source_fingerprint: str | None = None
) -> ScenarioBundle:
    """按**同一条正式入口**重建该 episode 的 formal `ScenarioBundle`。

    `build_verified_formal_env_injection` 构造环境时正是用
    `(split, origin=local_origin, forecast_cutoff=horizon)` 调 `build_formal_scenario_b6`；
    这里用同一组参数重建，因此产出的 bundle 携带**真实** provenance
    （九个上游资产的 path+sha256、真实 `generated_at` / target 窗口、
    当前实现的 40 位 revision），可通过 `validate_forecast_purpose(purpose="training")`。

    未篡改的生产路径上它与注入到 env 的 causal forecast **逐位相同**
    （`tests/test_m13gch2r1_formal_causal_snapshot.py`
    `::test_formal_planning_window_slices_the_verified_bundle` 逐位断言）。
    本函数**不**用相等性做门禁：规划输入按其**通道**取自 `env.*_forecast_t`，
    以便回归测试能验证「改可见因果预测 ⇒ 快照字段变」。

    缓存键只含已验证的不可变输入；同一 episode 内多次构快照不会重复重算。
    """
    from scenario.b6_split_manifests import default_inputs
    from scenario.formal_scenario_b6 import build_formal_scenario_b6
    from scenario.splits import SplitName

    inputs = default_inputs()
    return build_formal_scenario_b6(
        cast("SplitName", split),
        origin=int(local_origin),
        forecast_cutoff=int(horizon),
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
        policy_manifest_path=inputs["forecast_policy_manifest"],
    )


def _formal_bundle(env) -> ScenarioBundle:
    """formal env 的 `snapshot.forecast`：本 episode 的**已验签**正式 bundle。"""
    inj = getattr(env, "formal_injection", None)
    if inj is None:
        raise ValueError(
            "env.formal 为真但没有 formal_injection：无法确定正式因果预测的来源与时间窗口"
        )
    if int(inj.horizon) != int(env.horizon):
        raise ValueError(
            "formal 注入的 horizon 必须与环境的 horizon 一致："
            f"{inj.horizon} != {env.horizon}"
        )
    fingerprint = getattr(env, "verified_train_input_fingerprint", None)
    if fingerprint is not None:
        return _verified_formal_bundle(
            str(inj.split), int(inj.local_origin), int(inj.horizon), str(fingerprint))
    return _verified_formal_bundle(str(inj.split), int(inj.local_origin), int(inj.horizon))


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
            # M1.3e-R2：`sources` 与七个 forecast 序列均为不可变 tuple
            sources=(digest,),
        )
        for name in BUNDLE_FORECAST_FIELDS
    }

    return ScenarioBundle(
        split="train",
        start=str(t),
        horizon=horizon,
        forecast_cutoff=cutoff,
        price_forecast=tuple(windows["price_forecast"]),
        load_forecast=tuple(windows["load_forecast"]),
        pv_forecast=tuple(windows["pv_forecast"]),
        wind_forecast=tuple(windows["wind_forecast"]),
        temperature_forecast=tuple(windows["temperature_forecast"]),
        carbon_forecast=tuple(windows["carbon_forecast"]),
        arrival_forecast=tuple(windows["arrival_forecast"]),
        mode="oracle_debug",
        generated_at=generated_at,
        forecast_provenance=ScenarioForecastProvenance(**provenance),
    )


def _planning_horizon(env, cap: int = PLANNING_HORIZON_CAP) -> int:
    remaining = int(env.horizon) - int(env.current_step)
    if bool(getattr(env, "terminal_inventory_enabled", False)):
        return max(remaining, 0)
    return max(min(cap, remaining), 0)


def _planning_extension(
    env, t: int, cutoff: int, n_steps: int
) -> tuple[dict[str, list[float]], list[bool], list[bool]]:
    """时域展开（M4.1c）：窗口内取**该路径的可见来源**，窗口外按 `EXTENSION_POLICY` 假设。

    formal 的可见来源是 `env.*_forecast_t`（已验签 B6 causal forecast），
    legacy 的是 realized 真值窗口。返回 (各外生量向量, visible_mask, assumed_mask)。
    窗口外**不读取任何来源**。
    """
    horizon = int(env.horizon)
    start, end = visible_window_slice(t, cutoff, horizon)
    n_visible = min(end - start, n_steps)  # 尾部按实际 planning_horizon_steps 截断
    visible_mask = [k < n_visible for k in range(n_steps)]
    assumed_mask = [not v for v in visible_mask]
    attrs = _source_attrs(env)

    def _series(name: str) -> list[float]:
        arr = np.asarray(getattr(env, attrs[name]), dtype=np.float64)
        return [float(arr[t + k]) for k in range(n_visible)]

    price_v = _series("price")
    pv_v = _series("pv")
    wind_v = _series("wind")
    temp_v = _series("temperature")
    carbon_v = _series("carbon")
    arrival_v = _series("arrival")

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

    # formal：`snapshot.forecast` 是本 episode 的**已验签** B6 正式 bundle；
    # legacy：逐字保持 `oracle_debug` 语义（真值窗口 + 全零 load 占位）。
    forecast = (
        _formal_bundle(env)
        if _is_formal(env)
        else _oracle_debug_bundle(
            env,
            t=t,
            horizon=horizon,
            cutoff=cutoff,
            delta_t_hours=float(env.delta_t_hours),
        )
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

    # **M1.3g-f-c-h2 单位口径**：`C_server` 是 **work/hour rate**，而规划器的
    # `exec_compute[g] = used / cap[g]`（`planning/model.py:1123`）必须落到
    # 环境的 **action 语义**上。环境每步能力为
    # `action × max_task_load_per_server × C_server × delta_t_hours`
    # （`envs/idc_price_env.py:778-784`），故：
    #   - `group_work_capacity` 必须是 **work/step**（raw action=1 对应的能力）；
    #   - `group_power_coeff_kw_per_work` 的分母是 `C_server × delta_t_hours`
    #     （**不**乘 `max_task_load_per_server`）。
    # 自洽性：`coeff × cap = max_task_load_per_server × (P_max − P_idle)`
    # = raw action=1 时的功率摆幅。
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    p_max_kw = np.asarray(env.model.P_max, dtype=np.float64) / 1000.0
    p_idle_kw = np.asarray(env.model.P_idle, dtype=np.float64) / 1000.0
    group_capacity_per_step = c_server * float(env.delta_t_hours)
    work_capacity = group_capacity_per_step * float(env.max_task_load_per_server)
    coeff = (p_max_kw - p_idle_kw) / np.maximum(group_capacity_per_step, 1e-6)

    inventory_fields: dict = {}
    if bool(getattr(env, "terminal_inventory_enabled", False)):
        from planning.service_guard import build_service_guard
        inventory_fields = {"terminal_inventory": TerminalInventory(
            episode_end_step=horizon, remaining_steps=horizon - t,
            target_kwh=float(env.bess_soc_target * env.bess_capacity_kWh),
            lower_kwh=float((env.bess_soc_target - env.bess_soc_final_tolerance)
                            * env.bess_capacity_kWh),
            upper_kwh=float((env.bess_soc_target + env.bess_soc_final_tolerance)
                            * env.bess_capacity_kWh),
        ), "service_guard": build_service_guard(
            env, tasks, work_capacity, temperature=vectors["temperature"],
            arrival=vectors["arrival"])}
    snapshot_type = InventorySnapshot if inventory_fields else SystemSnapshot
    return snapshot_type(
        step=t,
        delta_t_hours=float(env.delta_t_hours),
        planning_horizon_steps=n_steps,
        **inventory_fields,
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
        group_work_capacity=[float(c) for c in work_capacity],
        group_power_coeff_kw_per_work=[float(c) for c in coeff],
        group_power_upper_kw=[float(p) for p in p_max_kw],
        power_approximation_note=POWER_APPROXIMATION_NOTE,
        budget_remaining_sgd=float(getattr(env, "budget_remaining_sgd", 1e9)),
    )
