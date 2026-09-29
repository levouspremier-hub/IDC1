"""M9.2-R2：公平配对的**报告入口**（把判定接到实际出口，而不是只留一个 helper）。

一个「被评估的一侧」= 完整 `EvaluationRecord` + `InventoryRecord` + `PairingKey`。
本模块负责：

- 把两侧的**原始**结果（购电费 / 碳排 / 服务资格 / 库存）原样保留进报告；
- 调 `assess_fair_pairing()`（内含配对键检查）给出判定；
- **只有** `eligible=True` 时才有 `fair_purchase_cost_delta_sgd` /
  `fair_carbon_delta_kg_co2e`；否则两项为 **`None`** 并写原因。

本模块**不**改环境、物理、任务或尾段语义，也**不**产生任何货币化尾段成本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.adapter import EvaluationRecord
from evaluation.inventory import (
    FairPairingReport,
    InventoryRecord,
    PairingKey,
    assess_fair_pairing,
)

__all__ = ["EvaluatedSide", "build_fair_pairing_report"]


@dataclass(frozen=True)
class EvaluatedSide:
    """公平配对的一侧：记录 + 库存 + 配对键（缺一不可）。"""

    label: str
    record: EvaluationRecord
    inventory: InventoryRecord
    pairing_key: PairingKey


def build_fair_pairing_report(
    left: EvaluatedSide, right: EvaluatedSide
) -> dict[str, Any]:
    """产出可直接写进 run 报告的公平配对字典。

    返回体同时含**判定**、**两侧原始结果**与**公平收益字段**（不通过时为 `None`）。
    """
    report: FairPairingReport = assess_fair_pairing(
        left.record, left.inventory, left.pairing_key,
        right.record, right.inventory, right.pairing_key)
    payload = report.to_dict()
    payload["sides"] = {
        left.label: {
            "run_id": left.record.run_id,
            "method": left.record.method,
            "steps": int(left.record.steps),
            "checkpoint_role": left.record.checkpoint_role,
        },
        right.label: {
            "run_id": right.record.run_id,
            "method": right.record.method,
            "steps": int(right.record.steps),
            "checkpoint_role": right.record.checkpoint_role,
        },
    }
    payload["conclusion"] = (
        f"eligible={report.eligible}："
        + ("允许比较公平成本/碳收益" if report.eligible
           else "**不**生成公平收益（原因见 reasons），两侧原始结果仍完整保留"))
    return payload
