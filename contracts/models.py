"""M2.1 六个不可变（frozen）Pydantic 契约。

- 所有模型 frozen（不可变），禁止 extra 字段。
- 功率/能量/货币/碳字段单位在各类 `UNITS` 中声明。
- `TaskAllocation.matrix` 为 n_task × n_group 矩形、非负。
"""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="contract-v1")


class ScenarioBundle(ContractBase):
    """可见预测场景切片（与 M1.3 提供器对齐，此处为冻结契约）。"""

    split: str
    start: str
    horizon: int
    forecast_cutoff: int
    price_forecast: list[float]
    load_forecast: list[float]
    pv_forecast: list[float]
    wind_forecast: list[float]
    temperature_forecast: list[float]
    source_hashes: dict[str, str]
    synthetic: bool = False

    UNITS: ClassVar[dict[str, str]] = {
        "price_forecast": "SGD/kWh",
        "load_forecast": "MW",
        "pv_forecast": "kW",
        "wind_forecast": "kW",
        "temperature_forecast": "degC",
    }

    def content_hash(self) -> str:
        """确定性内容 hash（与 dict 键顺序无关）。"""
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode("utf-8")
        ).hexdigest()


class TaskState(ContractBase):
    task_id: str
    remaining_work: float
    deadline: int
    priority: float
    status: str

    UNITS: ClassVar[dict[str, str]] = {"remaining_work": "work-units"}


class SystemSnapshot(ContractBase):
    """当前系统状态（供修正器/规划器）。"""

    step: int
    soc_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    group_capacity_kw: list[float]
    access_limit_kw: float
    budget_remaining_sgd: float
    tasks: list[TaskState]
    forecast: ScenarioBundle

    UNITS: ClassVar[dict[str, str]] = {
        "soc_kwh": "kWh",
        "soc_min_kwh": "kWh",
        "soc_max_kwh": "kWh",
        "group_capacity_kw": "kW",
        "access_limit_kw": "kW",
        "budget_remaining_sgd": "SGD",
    }


class DispatchProposal(ContractBase):
    """原始动作提案（21 维：20 compute + 1 有符号储能）。"""

    compute_actions: list[float]
    storage_action: float

    UNITS: ClassVar[dict[str, str]] = {
        "compute_actions": "normalized [0,1]",
        "storage_action": "normalized [-1,1]",
    }


class TaskAllocation(ContractBase):
    """任务×组分配矩阵 A[i,g]，矩形、非负。"""

    task_ids: list[str]
    group_ids: list[int]
    matrix: list[list[float]]

    UNITS: ClassVar[dict[str, str]] = {"matrix": "work-units"}

    @model_validator(mode="after")
    def _validate_matrix(self) -> TaskAllocation:
        n_group = len(self.group_ids)
        if len(self.matrix) != len(self.task_ids):
            raise ValueError("matrix 行数 != len(task_ids)")
        for row in self.matrix:
            if len(row) != n_group:
                raise ValueError("matrix 非矩形：每行长度 != len(group_ids)")
            if any(x < 0 for x in row):
                raise ValueError("matrix 必须非负")
        return self


class DispatchResult(ContractBase):
    """单步执行结果：raw 与 exec 分开记录。"""

    raw_compute_actions: list[float]
    raw_storage_action: float
    exec_compute_actions: list[float]
    exec_storage_action: float
    correction_reason: str
    business_gap: float
    solve_time_s: float
    p_grid_kw: float
    soc_next_kwh: float
    cost_sgd: float
    carbon_kg: float

    UNITS: ClassVar[dict[str, str]] = {
        "business_gap": "work-units",
        "solve_time_s": "s",
        "p_grid_kw": "kW",
        "soc_next_kwh": "kWh",
        "cost_sgd": "SGD",
        "carbon_kg": "kgCO2",
    }


class EvaluationRecord(ContractBase):
    """统一评估记录（M6 五方法同 schema）。"""

    method: str
    run_id: str
    service_qualified: bool
    total_cost_sgd: float
    total_carbon_kg: float
    renewable_utilization: float
    peak_kw: float
    reliability: float
    solve_time_avg_s: float
    failure_classification: str | None = None

    UNITS: ClassVar[dict[str, str]] = {
        "total_cost_sgd": "SGD",
        "total_carbon_kg": "kgCO2",
        "renewable_utilization": "fraction [0,1]",
        "peak_kw": "kW",
        "reliability": "fraction [0,1]",
        "solve_time_avg_s": "s",
    }
