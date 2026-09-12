"""M2 gates: the six contracts round-trip, hash deterministically, and reject bad units/shapes."""

import pytest
from pydantic import ValidationError

from contracts import (
    DispatchProposal,
    DispatchResult,
    EvaluationRecord,
    ScenarioBundle,
    SystemSnapshot,
    TaskAllocation,
    TaskSpec,
    TaskState,
)


def make_scenario() -> ScenarioBundle:
    return ScenarioBundle(
        scenario_id="s1",
        units={"price": "SGD/kWh", "power": "kW", "energy": "kWh", "workload": "work"},
        sources={"price": "USEP"},
        price=[0.1, 0.2, 0.3],
        tasks=[
            TaskSpec(
                task_id=0,
                workload=10.0,
                arrival_time=0,
                latest_finish_time=5,
                max_rate=4.0,
                is_critical=True,
            )
        ],
        capacity={"n_groups": 20, "group_capacity_work_per_hour": [100.0] * 20},
    )


def test_scenario_roundtrip_and_deterministic_hash() -> None:
    a = make_scenario()
    b = ScenarioBundle.from_json(a.to_json())
    assert b == a
    assert a.content_hash() == b.content_hash()
    assert len(a.content_hash()) == 64


def test_scenario_freeze_stores_hash() -> None:
    a = make_scenario()
    frozen = a.freeze()
    assert frozen.hash == frozen.content_hash()
    assert frozen.hash


def test_scenario_missing_unit_raises() -> None:
    with pytest.raises(ValidationError):
        ScenarioBundle(
            scenario_id="s1",
            units={"price": "SGD/kWh", "power": "kW", "energy": "kWh"},  # missing workload
            price=[0.1],
            tasks=[],
            capacity={"n_groups": 20},
        )


def test_proposal_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DispatchProposal(compute=[0.5, 1.5], storage=0.0)
    with pytest.raises(ValidationError):
        DispatchProposal(compute=[0.5, 0.5], storage=2.0)


def test_proposal_roundtrip() -> None:
    p = DispatchProposal(compute=[0.1] * 20, storage=-0.3)
    assert DispatchProposal.from_json(p.to_json()) == p
    assert p.n_groups == 20


def test_allocation_shape_and_nonnegativity() -> None:
    a = TaskAllocation(matrix=[[1.0, 2.0], [0.0, 3.0]], unit="work/hour")
    assert a.n_tasks == 2 and a.n_groups == 2
    assert TaskAllocation.from_json(a.to_json()) == a
    with pytest.raises(ValidationError):
        TaskAllocation(matrix=[[-1.0, 2.0]], unit="work/hour")


def test_allocation_missing_unit_raises() -> None:
    with pytest.raises(ValidationError):
        TaskAllocation(matrix=[[1.0]], unit="   ")


def test_snapshot_roundtrip() -> None:
    s = SystemSnapshot(
        t=3,
        task_states=[TaskState(task_id=0, remaining_work=4.0, deadline_status="on_time_backlog")],
        soc=0.5,
        budgets={"carbon": 100.0},
    )
    assert SystemSnapshot.from_json(s.to_json()) == s


def test_result_roundtrip() -> None:
    r = DispatchResult(
        raw_compute=[0.5] * 20,
        raw_storage=0.4,
        exec_compute=[0.4] * 20,
        exec_storage=0.4,
        energy_flows={"grid_purchase": 12.0, "pv_curtail": 1.0},
        correction_reason="access_cap",
        solve_status="feasible",
    )
    assert DispatchResult.from_json(r.to_json()) == r


def test_evaluation_roundtrip() -> None:
    e = EvaluationRecord(
        scenario_id="s1",
        business={"completed_work": 100.0, "deadline_misses": 1},
        economic={"total_cost": 500.0},
        carbon={"emissions_kg": 300.0},
    )
    assert EvaluationRecord.from_json(e.to_json()) == e
