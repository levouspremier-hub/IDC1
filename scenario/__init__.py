"""M1.3 场景提供器：可见预测 + 未来信息隔离。

场景类型统一为 `contracts.ScenarioBundle`（M1.3a；M1.3e 起为 contract-v8）。

正式（`mode="formal"`）场景**尚未接通**：`build_scenario(synthetic=False)` 仍然
明确失败（M1.3f/M1.3g）。可用的两条路径都是**非正式**的：
`build_scenario(..., synthetic=True)` 与 `build_oracle_debug_scenario_from_truth(...)`。
"""

from contracts.models import ScenarioBundle
from scenario.scenario import (
    SERIES_KEYS,
    build_oracle_debug_scenario_from_truth,
    build_scenario,
)

__all__ = [
    "SERIES_KEYS",
    "ScenarioBundle",
    "build_oracle_debug_scenario_from_truth",
    "build_scenario",
]
