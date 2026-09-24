"""M6-P1：业务服务标准与「未判定」判定。

`docs/M6_EVALUATION_PROTOCOL.md` §2 的 95%／1% 是**提案值**，
**尚未人工冻结**。因此：

- `FROZEN_PROJECT_SERVICE_STANDARD` 目前**是 `None`**；
- 评估器**必须**显式收到标准才计算资格；标准为 `None` **或** 未冻结
  ⇒ `service_qualified is None`（**未判定**），**绝不默认达标**；
- 任一门限分量为 `None`（零分母 = 不可判定）⇒ 整体未判定；
- 测试可以用**明确标记的临时标准**验证计算路径，那**不是**冻结项目阈值。
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.models import ServiceMetrics

# 协议 §2 的提案值（**未冻结**，仅供人工作参考；不得据此判定达标）。
PROPOSED_ON_TIME_TASK_RATE_MIN = 0.95
PROPOSED_ON_TIME_WORK_RATE_MIN = 0.95
PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX = 0.01
PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX = 0

#: 项目的**唯一**冻结服务标准。人工冻结之前**恒为 None**。
FROZEN_PROJECT_SERVICE_STANDARD: ServiceStandard | None = None

NOT_UNDETERMINED = "已按**显式传入**的服务标准计算"
NO_STANDARD_NOTE = "未判定：未显式传入服务标准（项目标准未冻结）"
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
    "NO_STANDARD_NOTE",
    "PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX",
    "PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX",
    "PROPOSED_ON_TIME_TASK_RATE_MIN",
    "PROPOSED_ON_TIME_WORK_RATE_MIN",
    "ServiceStandard",
    "qualify",
]
