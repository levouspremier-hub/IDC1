"""M9.2-R1：**库存记录**与**公平成本/碳配对判定**（评估层，不改 `contracts/`）。

问题：服务达标 + 四类物理违规为零**仍不足以**声称「同等库存下成本更低」——
若两条轨迹的初始储能不同、或终点 SOC 差得很远，购电费/碳排的差值里就混进了
「把电池用掉」的收益。本模块把这件事**显式记录并显式判定**。

三件事：

1. `InventoryRecord`：episode 的**初始**与**终点**库存，以及环境**现行**的
   目标 SOC / 终点容差 / 尾段结算量；
2. `evaluate_with_inventory()`：跑一次统一评估并**同时**返回
   `EvaluationRecord`（原样，`service_qualified` 语义**不变**）与 `InventoryRecord`；
3. `assess_fair_pairing()`：按 M9.2-R1 预登记的六个条件判定两条轨迹能否进入
   「公平成本/碳收益」配对差。

口径纪律：

- `service_qualified` **原义保留**；库存可比性是**另一个**字段
  （`inventory_comparable` / `fair_cost_carbon_pair_eligible`），
  **不得**把库存不合格混写成服务合格或服务不合格；
- `terminal_soc_recovery_kwh` 是**尾段结算诊断量**，**不**冒充已发生的购电或 SGD；
- 现行正式评估单位是**独立的 48 步日 episode**（每次 reset 重新开始队列与 SOC，
  终点只执行一次既有尾段结算）——**不是**跨日连续库存实验；
- 本模块**不**改环境、物理、任务或尾段语义，也**不**用罚项凑货币化成本。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evaluation.adapter import EvaluationRecord, evaluate, violations_in
from evaluation.metrics import PHYSICS_ABS_TOL

__all__ = [
    "FAIR_PAIRING_CONDITIONS",
    "FairPairingDecision",
    "InventoryRecord",
    "assess_fair_pairing",
    "capture_inventory",
    "evaluate_with_inventory",
]

#: 预登记的公平配对条件（M9.2-R1；顺序即报告顺序）。
FAIR_PAIRING_CONDITIONS: tuple[str, ...] = (
    "episode_complete",
    "service_qualified",
    "no_physical_violation",
    "capacity_matches",
    "initial_energy_matches",
    "final_soc_within_env_tolerance",
    "final_energy_gap_within_physics_tol",
)


@dataclass(frozen=True)
class InventoryRecord:
    """一次 episode 的库存记录（初末 SOC/储能量 + 环境现行的目标与容差）。"""

    method: str
    run_id: str
    horizon: int
    steps: int
    capacity_kwh: float
    soc_min: float
    soc_max: float
    soc_target: float
    soc_final_tolerance: float
    initial_soc: float
    initial_energy_kwh: float
    final_soc: float
    final_energy_kwh: float
    target_energy_kwh: float
    final_soc_deviation: float
    final_soc_within_env_tolerance: bool
    terminal_soc_recovery_kwh: float
    terminal_settlement_penalty: float
    terminal_leftover_work: float
    episode_complete: bool
    failure_classification: str | None
    #: 诊断说明：日 episode 语义 + 结算量不是货币
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "run_id": self.run_id,
            "horizon": self.horizon,
            "steps": self.steps,
            "capacity_kwh": self.capacity_kwh,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "soc_target": self.soc_target,
            "soc_final_tolerance": self.soc_final_tolerance,
            "initial_soc": self.initial_soc,
            "initial_energy_kwh": self.initial_energy_kwh,
            "final_soc": self.final_soc,
            "final_energy_kwh": self.final_energy_kwh,
            "target_energy_kwh": self.target_energy_kwh,
            "final_soc_deviation": self.final_soc_deviation,
            "final_soc_within_env_tolerance": self.final_soc_within_env_tolerance,
            "terminal_soc_recovery_kwh": self.terminal_soc_recovery_kwh,
            "terminal_settlement_penalty": self.terminal_settlement_penalty,
            "terminal_leftover_work": self.terminal_leftover_work,
            "episode_complete": self.episode_complete,
            "failure_classification": self.failure_classification,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FairPairingDecision:
    """两条轨迹能否进入「公平成本/碳收益」配对差。"""

    eligible: bool
    conditions: dict[str, bool]
    reasons: tuple[str, ...]
    final_energy_gap_kwh: float | None
    physics_abs_tol: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "conditions": dict(self.conditions),
            "reasons": list(self.reasons),
            "final_energy_gap_kwh": self.final_energy_gap_kwh,
            "physics_abs_tol": self.physics_abs_tol,
            "on_failure": ("不满足 ⇒ 不生成公平成本/碳收益；两侧成本/碳与原因仍完整保留"),
        }


def _unwrapped(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def capture_inventory(env: Any, record: EvaluationRecord) -> InventoryRecord:
    """在 `evaluate()` **返回之后**读取环境终态，组装库存记录。"""
    u = _unwrapped(env)
    capacity = float(u.bess_capacity_kWh)
    target = float(u.bess_soc_target)
    tolerance = float(u.bess_soc_final_tolerance)
    initial_soc = float(np.clip(u.bess_soc_init, u.bess_soc_min, u.bess_soc_max))
    final_soc = float(u.bess_soc)
    deviation = abs(final_soc - target)
    horizon = int(u.horizon)
    return InventoryRecord(
        method=str(record.method),
        run_id=str(record.run_id),
        horizon=horizon,
        steps=int(record.steps),
        capacity_kwh=capacity,
        soc_min=float(u.bess_soc_min),
        soc_max=float(u.bess_soc_max),
        soc_target=target,
        soc_final_tolerance=tolerance,
        initial_soc=initial_soc,
        initial_energy_kwh=initial_soc * capacity,
        final_soc=final_soc,
        final_energy_kwh=float(u.bess_energy_kWh),
        target_energy_kwh=target * capacity,
        final_soc_deviation=deviation,
        final_soc_within_env_tolerance=bool(deviation <= tolerance),
        terminal_soc_recovery_kwh=float(u.terminal_soc_recovery_kwh),
        terminal_settlement_penalty=float(u.terminal_settlement_penalty),
        terminal_leftover_work=float(u.terminal_leftover_work),
        episode_complete=bool(record.steps == horizon
                              and record.failure_classification is None),
        failure_classification=record.failure_classification,
        notes=(
            "评估单位是**独立的 48 步日 episode**：reset 后队列与 SOC 重新开始，"
            "终点只执行一次既有尾段结算；**不是**跨日连续库存实验。",
            "`terminal_soc_recovery_kwh` 与 `terminal_settlement_penalty` 是**尾段结算诊断量**，"
            "**不**冒充已发生的购电或 SGD。",
        ),
    )


def evaluate_with_inventory(
    env: Any,
    method: str,
    action_fn: Callable[[np.ndarray], np.ndarray],
    **kwargs: Any,
) -> tuple[EvaluationRecord, InventoryRecord]:
    """跑一次统一评估，**同时**返回 `EvaluationRecord` 与 `InventoryRecord`。

    `EvaluationRecord` 与 `service_qualified` 的语义**逐字不变**；库存是**附加**记录。
    """
    # `evaluate()` 内部会 `reset()`：库存的**初始**值由环境的 `bess_soc_init` 决定，
    # 故在调用**之前**取一次容量用于一致性核对。
    before_capacity = float(_unwrapped(env).bess_capacity_kWh)
    record = evaluate(env, method, action_fn, **kwargs)
    inventory = capture_inventory(env, record)
    if abs(inventory.capacity_kwh - before_capacity) > PHYSICS_ABS_TOL:
        raise ValueError("评估前后 BESS 容量发生变化：库存记录不可信")
    return record, inventory


def assess_fair_pairing(
    left: EvaluationRecord, left_inv: InventoryRecord,
    right: EvaluationRecord, right_inv: InventoryRecord,
) -> FairPairingDecision:
    """按预登记的六个条件判定两条轨迹能否进入公平成本/碳配对差。

    不满足时**不生成**收益；调用方仍应完整报告两侧成本/碳与原因。
    """
    gap = abs(left_inv.final_energy_kwh - right_inv.final_energy_kwh)
    conditions = {
        "episode_complete": bool(left_inv.episode_complete and right_inv.episode_complete),
        "service_qualified": bool(left.service_qualified is True
                                  and right.service_qualified is True),
        "no_physical_violation": bool(not violations_in(left.physical)
                                      and not violations_in(right.physical)),
        "capacity_matches": bool(abs(left_inv.capacity_kwh - right_inv.capacity_kwh)
                                 <= PHYSICS_ABS_TOL),
        "initial_energy_matches": bool(
            abs(left_inv.initial_energy_kwh - right_inv.initial_energy_kwh)
            <= PHYSICS_ABS_TOL),
        "final_soc_within_env_tolerance": bool(
            left_inv.final_soc_within_env_tolerance
            and right_inv.final_soc_within_env_tolerance),
        "final_energy_gap_within_physics_tol": bool(gap <= PHYSICS_ABS_TOL),
    }
    reasons = tuple(reason_ for reason_ in (
        None if conditions["episode_complete"] else
        f"episode 未完整运行 48 步或异常终止（左侧 {left_inv.steps} 步 / "
        f"右侧 {right_inv.steps} 步）",
        None if conditions["service_qualified"] else
        f"服务不合格或未判定（左侧 {left.service_qualified} / "
        f"右侧 {right.service_qualified}）",
        None if conditions["no_physical_violation"] else
        f"存在物理违规（左侧 {list(violations_in(left.physical))} / "
        f"右侧 {list(violations_in(right.physical))}）",
        None if conditions["capacity_matches"] else
        f"BESS 容量不同（左侧 {left_inv.capacity_kwh} / 右侧 {right_inv.capacity_kwh}）",
        None if conditions["initial_energy_matches"] else
        f"初始储能量不同（左侧 {left_inv.initial_energy_kwh} / "
        f"右侧 {right_inv.initial_energy_kwh}）",
        None if conditions["final_soc_within_env_tolerance"] else
        f"终点 SOC 超出环境现有目标容差（左侧偏差 {left_inv.final_soc_deviation} / "
        f"容差 {left_inv.soc_final_tolerance}；右侧偏差 {right_inv.final_soc_deviation} / "
        f"容差 {right_inv.soc_final_tolerance}）",
        None if conditions["final_energy_gap_within_physics_tol"] else
        f"终点储能量之差 {gap} 超过物理数值容差 {PHYSICS_ABS_TOL}",
    ) if reason_ is not None)
    return FairPairingDecision(
        eligible=all(conditions.values()),
        conditions=conditions,
        reasons=reasons,
        final_energy_gap_kwh=float(gap),
        physics_abs_tol=float(PHYSICS_ABS_TOL),
    )
