"""M6-P1 统一评估适配器：所有方法输出同一 `EvaluationRecord` schema。

评估输入契约在 `checkpointing.eval_input`；服务标准的「未判定」语义在
`evaluation.service_standard`；指标聚合在 `evaluation.metrics`。
"""

from evaluation.adapter import (
    ACTION_MODES,
    PLANNED_METHODS,
    evaluate,
    neutral_rule,
    planned_method_rows,
)
from evaluation.service_standard import (
    FROZEN_PROJECT_SERVICE_STANDARD,
    ServiceStandard,
)

__all__ = [
    "ACTION_MODES",
    "FROZEN_PROJECT_SERVICE_STANDARD",
    "PLANNED_METHODS",
    "ServiceStandard",
    "evaluate",
    "neutral_rule",
    "planned_method_rows",
]
