"""M4.4 测试：失败分类与安全回退。"""

from contracts.models import DispatchProposal, ScenarioBundle, SystemSnapshot, TaskState
from planning.corrector import FailureClass, correct


def _forecast() -> ScenarioBundle:
    return ScenarioBundle(
        split="train",
        start="0",
        horizon=24,
        forecast_cutoff=4,
        price_forecast=[0.2] * 24,
        load_forecast=[0.0] * 24,
        pv_forecast=[0.0] * 24,
        wind_forecast=[0.0] * 24,
        temperature_forecast=[28.0] * 24,
        carbon_forecast=[0.0] * 24,
        source_hashes={"x": "y"},
    )


def _snapshot(access_limit_kw=12.0, n_group=2) -> SystemSnapshot:
    tasks = [
        TaskState(task_id="t0", remaining_work=5.0, deadline=8, priority=1.0, status="waiting"),
        TaskState(task_id="t1", remaining_work=3.0, deadline=6, priority=2.0, status="waiting"),
    ]
    return SystemSnapshot(
        step=0,
        soc_kwh=50.0,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
        group_capacity_kw=[4.0] * n_group,
        access_limit_kw=access_limit_kw,
        budget_remaining_sgd=1000.0,
        tasks=tasks,
        forecast=_forecast(),
    )


def test_success_is_reviewed():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.0)
    c = correct(snap, p)
    assert c.failure == FailureClass.NONE
    assert c.reviewed is True


def test_infeasible_returns_boundary_action(monkeypatch):
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5, 0.5], storage_action=0.0)
    # 模拟求解器返回不可行
    stub = type("R", (), {"success": False})
    monkeypatch.setattr("planning.corrector.solve_milp", lambda s: stub())
    c = correct(snap, p)
    assert c.failure == FailureClass.INFEASIBLE
    assert c.exec_compute_actions == [0.0, 0.0]
    assert c.business_gap == 8.0


def test_forecast_oob_returns_boundary_action():
    snap = _snapshot()
    p = DispatchProposal(compute_actions=[0.5, 0.5, 0.5], storage_action=0.0)  # 维度不符
    c = correct(snap, p)
    assert c.failure == FailureClass.FORECAST_OOB
    assert c.exec_compute_actions == [0.0, 0.0]
    assert c.reviewed is True
