"""M2.2 跨契约不变量校验器。

每个校验器在违反不变量时抛带具体字段名的 `ValueError`，绝不静默裁剪或补默认值。
"""

from __future__ import annotations

from contracts.models import (
    DispatchProposal,
    DispatchResult,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
)

_FORECAST_FIELDS = (
    "price_forecast",
    "load_forecast",
    "pv_forecast",
    "wind_forecast",
    "temperature_forecast",
)


def validate_scenario(scenario: ScenarioBundle) -> None:
    """时间轴长度一致 + 单位/来源齐备。"""
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
    if not scenario.source_hashes:
        raise ValueError("缺少来源 hash（source_hashes 为空）")


def validate_snapshot(snapshot: SystemSnapshot) -> None:
    """SOC 范围 + 容量/接入非负。"""
    if not (snapshot.soc_min_kwh <= snapshot.soc_kwh <= snapshot.soc_max_kwh):
        raise ValueError(
            f"soc_kwh {snapshot.soc_kwh} 超出 [{snapshot.soc_min_kwh}, {snapshot.soc_max_kwh}]"
        )
    if any(c < 0 for c in snapshot.group_capacity_kw):
        raise ValueError("group_capacity_kw 含负值")
    if snapshot.access_limit_kw < 0:
        raise ValueError("access_limit_kw 为负")
    if snapshot.budget_remaining_sgd < 0:
        raise ValueError("budget_remaining_sgd 为负")


def validate_dispatch_proposal(proposal: DispatchProposal, snapshot: SystemSnapshot) -> None:
    """动作维度与组数一致 + 储能/计算动作范围。"""
    n_group = len(snapshot.group_capacity_kw)
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
    n_group = len(snapshot.group_capacity_kw)
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
        if col_sum > snapshot.group_capacity_kw[g] + 1e-9:
            cap = snapshot.group_capacity_kw[g]
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
