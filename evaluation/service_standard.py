"""M6-P1 / M6-P1-F5：业务服务标准与「未判定」判定。

`docs/M6_EVALUATION_PROTOCOL.md` §2 的 95%／95%／1%／0 已由 **M6-P1-F5 人工冻结**
为**唯一项目标准** `m6-service-standard-v1`（`frozen=True`，见 §2.1 的冻结记录）。因此：

- `FROZEN_PROJECT_SERVICE_STANDARD` 是该冻结标准的**实际 `ServiceStandard` 实例**；
- **它不会被隐式采用**：评估器仍要求调用方**显式传入**标准。未传入（`service_standard=None`）
  或传入未冻结标准 ⇒ `service_qualified is None`（**未判定**），**绝不默认达标**；
- 任一门限分量为 `None`（零分母 = 不可判定）⇒ 整体未判定；
- 测试仍可用**明确标记的临时标准**验证计算路径，那**不是**项目阈值。

冻结的**依据范围**：M6-P1-F4 的 train-only 复审
（`runs/m6_service_feasibility_after_f3r2/`，24 个 train origin × 三种固定参考提案）
证明**所测参考轨迹可达**这四项门槛；它**不**证明 validation/test 上的泛化，
也**不**证明五种方法全部合格。
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.models import ServiceMetrics

# 协议 §2 的**历史提案值**。这四个数值现已被下方 `FROZEN_PROJECT_SERVICE_STANDARD`
# 冻结为项目标准；保留常量名是因为历史审计脚本
# （`scripts/audit_m6_service_feasibility.py`）以其输出「提案门槛诊断」列。
PROPOSED_ON_TIME_TASK_RATE_MIN = 0.95
PROPOSED_ON_TIME_WORK_RATE_MIN = 0.95
PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX = 0.01
PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX = 0

#: 冻结项目标准的唯一 ID（写入 `EvaluationRecord.service_standard_id`）。
FROZEN_PROJECT_SERVICE_STANDARD_ID = "m6-service-standard-v1"

NOT_UNDETERMINED = "已按**显式传入**的服务标准计算"
NO_STANDARD_NOTE = "未判定：未显式传入服务标准（**不**隐式采用项目标准）"
UNFROZEN_NOTE = "未判定：传入的服务标准未冻结（frozen=False）"
ZERO_DENOMINATOR_NOTE = "未判定：门限分量存在零分母（不可判定），不自动合格"
NO_QUALIFICATION_KEY = "service_qualified"


@dataclass(frozen=True)
class ServiceStandard:
    """一条**显式**的业务服务标准（阈值 + 分母口径的声明身份）。"""

    standard_id: str
    frozen: bool
    on_time_task_rate_min: float
    on_time_work_rate_min: float
    end_leftover_work_fraction_max: float
    non_interruptible_interruption_max: int

    def __post_init__(self) -> None:
        if not isinstance(self.standard_id, str) or not self.standard_id:
            raise ValueError("standard_id 必须是非空字符串")
        if not isinstance(self.frozen, bool):
            raise ValueError(f"frozen 必须是 bool，实际 {self.frozen!r}")
        for name in ("on_time_task_rate_min", "on_time_work_rate_min",
                     "end_leftover_work_fraction_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} 必须是数值（bool 不算），实际 {value!r}")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} 必须在 [0, 1]，实际 {value!r}")
        limit = self.non_interruptible_interruption_max
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError(
                f"non_interruptible_interruption_max 必须是非负整数，实际 {limit!r}")


#: 项目的**唯一**冻结服务标准（M6-P1-F5 人工冻结；见协议 §2.1 的冻结记录）。
#:
#: 它是**显式传入**才生效的常量——`evaluation.adapter.evaluate` 的
#: `service_standard` 参数**没有默认值**，故不存在「默认达标」的调用路径。
#: 四个阈值与协议 §2 的提案值取同一常量，避免同一决策出现两份数值来源。
FROZEN_PROJECT_SERVICE_STANDARD: ServiceStandard = ServiceStandard(
    standard_id=FROZEN_PROJECT_SERVICE_STANDARD_ID,
    frozen=True,
    on_time_task_rate_min=PROPOSED_ON_TIME_TASK_RATE_MIN,
    on_time_work_rate_min=PROPOSED_ON_TIME_WORK_RATE_MIN,
    end_leftover_work_fraction_max=PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX,
    non_interruptible_interruption_max=PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX,
)


def qualify(
    metrics: ServiceMetrics, standard: ServiceStandard | None
) -> tuple[bool | None, str]:
    """按**显式**标准判定业务服务资格；不可判定时返回 `None`（未判定）。"""
    if standard is None:
        return None, NO_STANDARD_NOTE
    if not standard.frozen:
        return None, UNFROZEN_NOTE

    names = ("on_time_task_rate", "on_time_work_rate", "end_leftover_work_fraction")
    components = (
        metrics.on_time_task_rate,
        metrics.on_time_work_rate,
        metrics.end_leftover_work_fraction,
    )
    missing = [name for name, value in zip(names, components, strict=True) if value is None]
    if missing:
        return None, f"{ZERO_DENOMINATOR_NOTE}（{missing}）"

    task_rate, work_rate, leftover_fraction = components
    assert task_rate is not None and work_rate is not None and leftover_fraction is not None
    passed = (
        task_rate >= standard.on_time_task_rate_min
        and work_rate >= standard.on_time_work_rate_min
        and leftover_fraction <= standard.end_leftover_work_fraction_max
        and metrics.non_interruptible_interruption_count
        <= standard.non_interruptible_interruption_max
    )
    return bool(passed), f"{NOT_UNDETERMINED}：{standard.standard_id}"


__all__ = [
    "FROZEN_PROJECT_SERVICE_STANDARD",
    "FROZEN_PROJECT_SERVICE_STANDARD_ID",
    "NO_STANDARD_NOTE",
    "PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX",
    "PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX",
    "PROPOSED_ON_TIME_TASK_RATE_MIN",
    "PROPOSED_ON_TIME_WORK_RATE_MIN",
    "ServiceStandard",
    "qualify",
]
