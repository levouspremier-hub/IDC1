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
    "FairPairingReport",
    "PairingKey",
    "InventoryRecord",
    "assess_fair_pairing",
    "capture_inventory",
    "evaluate_with_inventory",
]

#: 预登记的公平配对条件（M9.2-R1；顺序即报告顺序）。
FAIR_PAIRING_CONDITIONS: tuple[str, ...] = (
    "pairing_key_matches",
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
class PairingKey:
    """公平配对的**唯一键**：`(split, episode_start, scenario_seed)`（M9.2-R1/R2）。

    **训练 seed 不是**配对键的一部分（它只作结果分层）；无训练 seed 的基线在同一
    `scenario_seed` 下只运行一次，并被多个 PPO 训练 seed 的配对引用。
    """

    split: str
    episode_start: str
    scenario_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "episode_start": self.episode_start,
                "scenario_seed": int(self.scenario_seed)}

    def same_as(self, other: PairingKey) -> bool:
        return (self.split == other.split
                and self.episode_start == other.episode_start
                and int(self.scenario_seed) == int(other.scenario_seed))


@dataclass(frozen=True)
class FairPairingReport:
    """公平成本/碳配对的**报告出口**（M9.2-R2）。

    同时保留**双方原始**购电费 / 碳排 / 服务与库存记录；**只有**同键且既有服务、物理、
    初末库存条件**全部通过**，才填 `fair_purchase_cost_delta_sgd` /
    `fair_carbon_delta_kg_co2e`；否则两项为 **`None`** 并写原因。
    """

    left_key: PairingKey
    right_key: PairingKey
    key_matches: bool
    eligible: bool
    conditions: dict[str, bool]
    reasons: tuple[str, ...]
    final_energy_gap_kwh: float | None
    physics_abs_tol: float
    left_purchase_cost_sgd: float
    right_purchase_cost_sgd: float
    left_carbon_kg_co2e: float
    right_carbon_kg_co2e: float
    left_service_qualified: bool | None
    right_service_qualified: bool | None
    left_inventory: dict[str, Any]
    right_inventory: dict[str, Any]
    fair_purchase_cost_delta_sgd: float | None
    fair_carbon_delta_kg_co2e: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairing_key": {
                "left": self.left_key.to_dict(),
                "right": self.right_key.to_dict(),
                "matches": bool(self.key_matches),
                "rule": "键不同即拒绝配对（训练 seed 不是配对键）",
            },
            "eligible": self.eligible,
            "conditions": dict(self.conditions),
            "reasons": list(self.reasons),
            "final_energy_gap_kwh": self.final_energy_gap_kwh,
            "physics_abs_tol": self.physics_abs_tol,
            "raw_results_retained": {
                "left": {"purchase_cost_sgd": self.left_purchase_cost_sgd,
                         "carbon_kg_co2e": self.left_carbon_kg_co2e,
                         "service_qualified": self.left_service_qualified,
                         "inventory": self.left_inventory},
                "right": {"purchase_cost_sgd": self.right_purchase_cost_sgd,
                          "carbon_kg_co2e": self.right_carbon_kg_co2e,
                          "service_qualified": self.right_service_qualified,
                          "inventory": self.right_inventory},
            },
            "fair_purchase_cost_delta_sgd": self.fair_purchase_cost_delta_sgd,
            "fair_carbon_delta_kg_co2e": self.fair_carbon_delta_kg_co2e,
            "delta_rule": ("只有 eligible=True 时才填写；否则两项为 null，"
                           "两侧原始成本/碳与服务/库存仍完整保留"),
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
    left: EvaluationRecord, left_inventory: InventoryRecord, left_key: PairingKey,
    right: EvaluationRecord, right_inventory: InventoryRecord, right_key: PairingKey,
) -> FairPairingReport:
    """按预登记的键与条件判定两条轨迹能否进入公平成本/碳配对差。

    **键不同即拒绝**（`key_matches=False`）：即使其余条件全部满足，也**不**填公平收益。
    不满足时**不生成**收益；两侧原始成本/碳与服务/库存记录仍完整保留。
    """
    key_matches = left_key.same_as(right_key)
    gap = abs(left_inventory.final_energy_kwh - right_inventory.final_energy_kwh)
    conditions = {
        "pairing_key_matches": bool(key_matches),
        "episode_complete": bool(
            left_inventory.episode_complete and right_inventory.episode_complete),
        "service_qualified": bool(left.service_qualified is True
                                  and right.service_qualified is True),
        "no_physical_violation": bool(not violations_in(left.physical)
                                      and not violations_in(right.physical)),
        "capacity_matches": bool(abs(left_inventory.capacity_kwh - right_inventory.capacity_kwh)
                                 <= PHYSICS_ABS_TOL),
        "initial_energy_matches": bool(
            abs(left_inventory.initial_energy_kwh - right_inventory.initial_energy_kwh)
            <= PHYSICS_ABS_TOL),
        "final_soc_within_env_tolerance": bool(
            left_inventory.final_soc_within_env_tolerance
            and right_inventory.final_soc_within_env_tolerance),
        "final_energy_gap_within_physics_tol": bool(gap <= PHYSICS_ABS_TOL),
    }
    reasons = tuple(reason_ for reason_ in (
        None if key_matches else
        f"配对键不同（左 {left_key.to_dict()} / 右 {right_key.to_dict()}）⇒ 拒绝配对",
        None if conditions["episode_complete"] else
        f"episode 未完整运行 48 步或异常终止（左侧 {left_inventory.steps} 步 / "
        f"右侧 {right_inventory.steps} 步）",
        None if conditions["service_qualified"] else
        f"服务不合格或未判定（左侧 {left.service_qualified} / "
        f"右侧 {right.service_qualified}）",
        None if conditions["no_physical_violation"] else
        f"存在物理违规（左侧 {list(violations_in(left.physical))} / "
        f"右侧 {list(violations_in(right.physical))}）",
        None if conditions["capacity_matches"] else
        f"BESS 容量不同（左侧 {left_inventory.capacity_kwh} / "
        f"右侧 {right_inventory.capacity_kwh}）",
        None if conditions["initial_energy_matches"] else
        f"初始储能量不同（左侧 {left_inventory.initial_energy_kwh} / "
        f"右侧 {right_inventory.initial_energy_kwh}）",
        None if conditions["final_soc_within_env_tolerance"] else
        f"终点 SOC 超出环境现有目标容差（左侧偏差 {left_inventory.final_soc_deviation} / "
        f"容差 {left_inventory.soc_final_tolerance}；"
        f"右侧偏差 {right_inventory.final_soc_deviation} / "
        f"容差 {right_inventory.soc_final_tolerance}）",
        None if conditions["final_energy_gap_within_physics_tol"] else
        f"终点储能量之差 {gap} 超过物理数值容差 {PHYSICS_ABS_TOL}",
    ) if reason_ is not None)
    eligible = all(conditions.values())
    return FairPairingReport(
        left_key=left_key,
        right_key=right_key,
        key_matches=key_matches,
        eligible=eligible,
        conditions=conditions,
        reasons=reasons,
        final_energy_gap_kwh=float(gap),
        physics_abs_tol=float(PHYSICS_ABS_TOL),
        left_purchase_cost_sgd=float(left.purchase_cost_sgd),
        right_purchase_cost_sgd=float(right.purchase_cost_sgd),
        left_carbon_kg_co2e=float(left.carbon_kg_co2e),
        right_carbon_kg_co2e=float(right.carbon_kg_co2e),
        left_service_qualified=left.service_qualified,
        right_service_qualified=right.service_qualified,
        left_inventory=left_inventory.to_dict(),
        right_inventory=right_inventory.to_dict(),
        fair_purchase_cost_delta_sgd=(
            float(right.purchase_cost_sgd - left.purchase_cost_sgd) if eligible else None),
        fair_carbon_delta_kg_co2e=(
            float(right.carbon_kg_co2e - left.carbon_kg_co2e) if eligible else None),
    )
