"""M2.2 跨契约不变量校验器。

每个校验器在违反不变量时抛带具体字段名的 `ValueError`，绝不静默裁剪或补默认值。
"""

from __future__ import annotations

from contracts.models import (
    BUNDLE_FORECAST_FIELDS,
    FORECAST_PURPOSES,
    NON_FORMAL_SOURCE_KINDS,
    AvailableExogenousForecast,
    DispatchProposal,
    DispatchResult,
    PlanningExogenousForecast,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
    validate_available_forecast_artifact,
    validate_bundle_forecast_provenance,
    validate_forecast_series_provenance,
)

_PLANNING_VECTORS = (
    "price",
    "pv",
    "wind",
    "temperature",
    "carbon",
    "arrival",
    "base_idc_power",
)
# 窗口外必须保守为 0 的量（M4.1c）
_ASSUMED_ZERO_VECTORS = ("pv", "wind", "arrival")


def validate_planning_forecast(
    planning: PlanningExogenousForecast, expected_steps: int
) -> None:
    """规划时域外生量校验（M4.1c）。"""
    if planning.horizon_steps != expected_steps:
        raise ValueError(
            f"planning_forecast.horizon_steps {planning.horizon_steps} != {expected_steps}"
        )
    lengths = {name: len(getattr(planning, name)) for name in _PLANNING_VECTORS}
    lengths["visible_mask"] = len(planning.visible_mask)
    lengths["assumed_mask"] = len(planning.assumed_mask)
    if set(lengths.values()) != {expected_steps}:
        raise ValueError(f"planning_forecast 向量长度不一致或 != horizon_steps：{lengths}")
    if not planning.extension_policy:
        raise ValueError("planning_forecast.extension_policy 不可为空")
    for name in _PLANNING_VECTORS:
        if name not in PlanningExogenousForecast.UNITS:
            raise ValueError(f"planning_forecast 缺少单位声明: {name}")
    for idx, (visible, assumed) in enumerate(
        zip(planning.visible_mask, planning.assumed_mask, strict=True)
    ):
        if visible == assumed:
            raise ValueError(f"planning_forecast mask 非互补：index {idx}")
    for name in _ASSUMED_ZERO_VECTORS:
        values = getattr(planning, name)
        for idx, assumed in enumerate(planning.assumed_mask):
            if assumed and values[idx] != 0.0:
                raise ValueError(
                    f"planning_forecast.{name} 在假设段必须为 0（index {idx}），实际 {values[idx]}"
                )

_FORECAST_FIELDS = (
    "price_forecast",
    "load_forecast",
    "pv_forecast",
    "wind_forecast",
    "temperature_forecast",
    "carbon_forecast",
    "arrival_forecast",
)


def validate_scenario(scenario: ScenarioBundle) -> None:
    """时间轴长度一致 + 单位齐备 + **结构化 provenance** 完备（contract-v8）。"""
    lengths = {name: len(getattr(scenario, name)) for name in _FORECAST_FIELDS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"预测时间轴长度不一致: {lengths}")
    if lengths["price_forecast"] != scenario.forecast_cutoff:
        raise ValueError(
            "price_forecast 长度 "
            f"{lengths['price_forecast']} != forecast_cutoff {scenario.forecast_cutoff}"
        )
    for name in _FORECAST_FIELDS:
        if name not in scenario.UNITS:
            raise ValueError(f"缺少单位声明: {name}")
    validate_scenario_provenance(scenario)


def validate_snapshot(snapshot: SystemSnapshot) -> None:
    """规划输入契约校验（M4.1a/M4.1b）：时间、SOC、效率、功率上限、逐组向量与近似系数。"""
    if snapshot.delta_t_hours <= 0:
        raise ValueError(f"delta_t_hours 必须 > 0，实际 {snapshot.delta_t_hours}")
    if snapshot.planning_horizon_steps <= 0:
        raise ValueError(f"planning_horizon_steps 必须 > 0，实际 {snapshot.planning_horizon_steps}")
    if not snapshot.power_approximation_note:
        raise ValueError("power_approximation_note 不可为空（需声明规划近似与复核要求）")
    if not (snapshot.soc_min_kwh <= snapshot.soc_kwh <= snapshot.soc_max_kwh):
        raise ValueError(
            f"soc_kwh {snapshot.soc_kwh} 超出 [{snapshot.soc_min_kwh}, {snapshot.soc_max_kwh}]"
        )
    if snapshot.soc_max_kwh > snapshot.soc_capacity_kwh + 1e-9:
        raise ValueError(
            f"soc_max_kwh {snapshot.soc_max_kwh} 超过 soc_capacity_kwh {snapshot.soc_capacity_kwh}"
        )
    if snapshot.soc_min_kwh < 0 or snapshot.soc_capacity_kwh <= 0:
        raise ValueError("soc_min_kwh / soc_capacity_kwh 非法（soc_min_kwh 需 >=0，容量需 >0）")
    for name in ("bess_charge_efficiency", "bess_discharge_efficiency"):
        value = getattr(snapshot, name)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} efficiency 必须在 (0, 1]，实际 {value}")
    for name in ("bess_charge_power_max_kw", "bess_discharge_power_max_kw"):
        value = getattr(snapshot, name)
        if value < 0:
            raise ValueError(f"{name} power 上限为负：{value}")
    if snapshot.bess_degradation_cost_per_kwh < 0:
        raise ValueError("bess_degradation_cost_per_kwh 为负")
    if any(c < 0 for c in snapshot.group_work_capacity):
        raise ValueError("group_work_capacity 含负值")
    if any(c < 0 for c in snapshot.group_power_coeff_kw_per_work):
        raise ValueError("group_power_coeff_kw_per_work coeff 含负值")
    if any(c < 0 for c in snapshot.group_power_upper_kw):
        raise ValueError("group_power_upper_kw 含负值")
    group_lengths = {
        "group_work_capacity": len(snapshot.group_work_capacity),
        "group_power_coeff_kw_per_work": len(snapshot.group_power_coeff_kw_per_work),
        "group_power_upper_kw": len(snapshot.group_power_upper_kw),
    }
    if len(set(group_lengths.values())) != 1:
        raise ValueError(f"逐组向量长度不一致：{group_lengths}")
    if len(snapshot.base_idc_power_forecast_kw) != snapshot.planning_horizon_steps:
        raise ValueError("base_idc_power_forecast_kw 长度 != planning_horizon_steps")
    if snapshot.access_limit_kw < 0:
        raise ValueError("access_limit_kw 为负")
    if snapshot.budget_remaining_sgd < 0:
        raise ValueError("budget_remaining_sgd 为负")
    for task in snapshot.tasks:
        if task.status == "not_arrived":
            raise ValueError(f"未来任务 {task.task_id} 不得进入规划快照")
        if task.max_rate_work_per_step < 0:
            raise ValueError(f"任务 {task.task_id} max_rate_work_per_step 为负")
        if task.remaining_work < 0:
            raise ValueError(f"任务 {task.task_id} remaining_work 为负")
    validate_scenario(snapshot.forecast)
    validate_planning_forecast(snapshot.planning_forecast, snapshot.planning_horizon_steps)
    if snapshot.planning_forecast.base_idc_power != snapshot.base_idc_power_forecast_kw:
        raise ValueError(
            "planning_forecast.base_idc_power 与 base_idc_power_forecast_kw 不一致（疑似静默分叉）"
        )


def validate_dispatch_proposal(proposal: DispatchProposal, snapshot: SystemSnapshot) -> None:
    """动作维度与组数一致 + 储能/计算动作范围。"""
    n_group = len(snapshot.group_work_capacity)
    if len(proposal.compute_actions) != n_group:
        raise ValueError(f"compute_actions 长度 {len(proposal.compute_actions)} != 组数 {n_group}")
    if not (-1.0 <= proposal.storage_action <= 1.0):
        raise ValueError(f"storage_action {proposal.storage_action} 超出 [-1,1]")
    if any(x < 0 or x > 1 for x in proposal.compute_actions):
        raise ValueError("compute_actions 超出 [0,1]")


def validate_task_allocation(
    allocation: TaskAllocation, snapshot: SystemSnapshot, max_rate: dict[str, float]
) -> None:
    """任务×组矩阵列数与组数一致 + 逐任务不超 max_rate + 逐组不超容量。"""
    n_group = len(snapshot.group_work_capacity)
    if len(allocation.group_ids) != n_group:
        raise ValueError(f"group_ids 数 {len(allocation.group_ids)} != 组数 {n_group}")
    for i, task_id in enumerate(allocation.task_ids):
        row_sum = sum(allocation.matrix[i])
        rate = max_rate.get(task_id)
        if rate is None:
            raise ValueError(f"任务 {task_id} 缺少 max_rate")
        if row_sum > rate + 1e-9:
            raise ValueError(f"任务 {task_id} 分配 {row_sum} 超 max_rate {rate}")
    for g in range(n_group):
        col_sum = sum(allocation.matrix[i][g] for i in range(len(allocation.task_ids)))
        if col_sum > snapshot.group_work_capacity[g] + 1e-9:
            cap = snapshot.group_work_capacity[g]
            raise ValueError(f"组 {allocation.group_ids[g]} 分配 {col_sum} 超容量 {cap}")


def validate_dispatch_result(result: DispatchResult) -> None:
    """raw/exec 动作维度一致 + 非负物理量。"""
    if len(result.raw_compute_actions) != len(result.exec_compute_actions):
        raise ValueError("raw/exec compute_actions 长度不一致")
    if result.p_grid_kw < 0:
        raise ValueError("p_grid_kw 为负")
    if result.soc_next_kwh < 0:
        raise ValueError("soc_next_kwh 为负")
    if result.cost_sgd < 0:
        raise ValueError("cost_sgd 为负")


# --- M1.3e：contract-v8 provenance 与 purpose gate --------------------------
#
# 逐序列 / 整体的 provenance 校验实现放在 `contracts.models`：`ScenarioBundle` 与
# `ForecastSeriesProvenance` 必须在**构造时**就 fail closed，而 `models` 不能反过来
# 导入 `validators`（会成环）。此处提供对外入口，语义完全一致。

validate_series_provenance = validate_forecast_series_provenance
validate_scenario_provenance = validate_bundle_forecast_provenance


def validate_available_forecast(artifact: AvailableExogenousForecast) -> None:
    """driver forecast artifact 的显式完整性校验（**防御性第二道**）。

    模型在构造时已自校验；此入口用于对「经由 `model_copy(update=...)` 直接写入
    `__dict__`」的对象再验一次。实现委托给 `contracts.models` 的
    `validate_available_forecast_artifact()`：它把原始属性值重新过一遍**构造路径的
    同一套规则**（而不是另写一套更弱的重复规则），且**不经过序列化器**
    ——序列化器可能把值洗白（M1.3e-R3 §18.5）。
    """
    validate_available_forecast_artifact(artifact)


def validate_forecast_purpose(scenario: ScenarioBundle, *, purpose: str) -> None:
    """**purpose gate**：`training`/`evaluation` 必须拒绝 synthetic 与 oracle_debug，
    只有 `debug` 接受。

    正式训练接线属 **M1.3g**；本卡只建立 gate 本身。
    """
    if purpose not in FORECAST_PURPOSES:
        raise ValueError(
            f"未知 purpose：{purpose!r}，必须属于 {list(FORECAST_PURPOSES)}"
        )
    if purpose == "debug":
        return
    # training/evaluation **只**接受 mode=formal；synthetic、oracle_debug 以及任何
    # 不是正式 ScenarioBundle 的对象（例如 available-exogenous forecast artifact）一律拒绝。
    mode = getattr(scenario, "mode", None)
    if mode != "formal":
        raise ValueError(
            f"purpose={purpose} 只接受 mode=formal 的场景，实际 mode={mode!r}（仅 debug 可用）"
        )
    kinds = {
        getattr(scenario.forecast_provenance, name).source_kind
        for name in BUNDLE_FORECAST_FIELDS
    }
    forbidden = sorted(kinds & set(NON_FORMAL_SOURCE_KINDS))
    if forbidden:
        raise ValueError(f"purpose={purpose} 不得使用 {forbidden} 来源（仅 debug 可用）")
