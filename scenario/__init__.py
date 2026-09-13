"""M1.3 场景提供器：可见预测 + 未来信息隔离。

场景类型统一为 `contracts.ScenarioBundle`（M1.3a），此处仅做再导出。
"""

from contracts.models import ScenarioBundle
from scenario.scenario import SERIES_KEYS, build_scenario, build_scenario_from_true

__all__ = ["SERIES_KEYS", "ScenarioBundle", "build_scenario", "build_scenario_from_true"]
