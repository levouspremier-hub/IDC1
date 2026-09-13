"""M2.1 六个不可变（frozen）Pydantic 契约。

M3.10c：`CONTRACT_VERSION_ID` 是本仓库**唯一**的契约版本源。
`ContractBase.schema_version` 默认值、`checkpointing.CURRENT_CONTRACT_VERSION`、
`safe_rl_v2.buffer.CONTRACT_VERSION` 都必须导入该常量，禁止各自硬编码版本字符串。
必须先于 `contracts.models` 导入定义，以避免循环导入。
"""

CONTRACT_VERSION_ID = "contract-v6"

from contracts.models import (  # noqa: E402
    DispatchProposal,
    DispatchResult,
    EvaluationRecord,
    PlanningExogenousForecast,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
    TaskState,
)

__all__ = [
    "CONTRACT_VERSION_ID",
    "ScenarioBundle",
    "SystemSnapshot",
    "DispatchProposal",
    "TaskAllocation",
    "DispatchResult",
    "PlanningExogenousForecast",
    "EvaluationRecord",
    "TaskState",
]
