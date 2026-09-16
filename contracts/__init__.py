"""M2.1 六个不可变（frozen）Pydantic 契约。

M3.10c：`CONTRACT_VERSION_ID` 是本仓库**唯一**的契约版本源。
`ContractBase.schema_version` 默认值、`checkpointing.CURRENT_CONTRACT_VERSION`、
`safe_rl_v2.buffer.CONTRACT_VERSION` 都必须导入该常量，禁止各自硬编码版本字符串。
**M1.3e-R1 起 `ContractBase` 在基底类统一锁定 `schema_version` 必须等于该常量**，
显式声明旧版本（如 `contract-v7`）或任意其它字符串一律拒绝。
必须先于 `contracts.models` 导入定义，以避免循环导入。
"""

CONTRACT_VERSION_ID = "contract-v8"

from contracts.models import (  # noqa: E402
    AVAILABLE_DRIVER_SERIES,
    ArtifactDigest,
    AvailableDriverProvenance,
    AvailableExogenousForecast,
    AvailableSeries,
    DispatchProposal,
    DispatchResult,
    EvaluationRecord,
    ForecastSeriesProvenance,
    PlanningExogenousForecast,
    ScenarioBundle,
    ScenarioForecastProvenance,
    SystemSnapshot,
    TaskAllocation,
    TaskState,
)

__all__ = [
    "CONTRACT_VERSION_ID",
    "AVAILABLE_DRIVER_SERIES",
    "ArtifactDigest",
    "AvailableDriverProvenance",
    "AvailableExogenousForecast",
    "AvailableSeries",
    "ForecastSeriesProvenance",
    "ScenarioForecastProvenance",
    "ScenarioBundle",
    "SystemSnapshot",
    "DispatchProposal",
    "TaskAllocation",
    "DispatchResult",
    "PlanningExogenousForecast",
    "EvaluationRecord",
    "TaskState",
]
