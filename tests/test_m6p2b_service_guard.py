"""Causal current service reserves retain physics and the real terminal target."""

import numpy as np
import pytest

from contracts.models import DispatchProposal
from idc_model.task import Task
from planning.model import solve_time_indexed_mip_raw_projection
from planning.snapshot_adapter import build_snapshot
from tests.test_m6p2b_terminal_inventory import _env


def guarded_env():
    env = _env(horizon=8)
    env.access_limit_kw = 20.
    env.terminal_service_guard_version = 'arrived-service-reserve-v1'
    task = Task(1, 'A', 'known running task', 0, 2, np.array([.1, .1]),
                8., 4, 3., False, False)
    task.start_time = 0
    task.status = 'waiting'
    env.tasks = [task]
    env.wt_t[:] = 30.
    return env


def test_storage_proposal_cannot_starve_known_running_noninterruptible_task():
    env = guarded_env()
    snapshot = build_snapshot(env)
    proposal = DispatchProposal(compute_actions=[0.] * 20, storage_action=-1.)
    plan = solve_time_indexed_mip_raw_projection(snapshot, proposal)
    assert plan.solver_status == 'optimal'
    assert sum(np.array(plan.exec_compute_actions) * snapshot.group_work_capacity) >= 4. - 1e-6
    assert plan.inventory_audit['stage_a_terminal_kwh'] == pytest.approx(50., abs=1e-6)
    assert plan.soc_kwh[-1] == pytest.approx(50., abs=1e-6)
    guard = snapshot.service_guard
    assert plan.charge_kw[0] <= guard.charge_limit_kw + 1e-6
    assert guard.renewable_reserve_assumption == 'zero-current-renewables'
    assert 'planning assumptions' in plan.inventory_audit['reachability_scope']


def test_unknown_service_guard_is_rejected_instead_of_silently_disabled():
    env = guarded_env()
    env.terminal_service_guard_version = 'unknown'
    with pytest.raises(ValueError, match='service guard'):
        build_snapshot(env)


@pytest.mark.leakage
def test_service_reserve_uses_only_arrived_tasks_and_forecast_channels():
    from safe_rl_v2.formal_train_loop import build_train_env, load_frozen_training_config

    env, _ = build_train_env(48, master_seed=0, config=load_frozen_training_config())
    env.terminal_inventory_enabled = True
    env.terminal_service_guard_version = 'arrived-service-reserve-v1'
    env.reset(seed=0)
    before = build_snapshot(env)
    assert before.service_guard is not None
    for series in (env.price_t, env.pv_t, env.wt_t, env.T_amb, env.carbon_factor_t):
        series[:] += 1000.
    for task in env.tasks:
        if task.status == 'not_arrived':
            task.workload *= 1000
            task.remaining_work *= 1000
    assert build_snapshot(env) == before


def test_plan_does_not_defer_recovery_into_unreserved_future_charging_capacity():
    env = guarded_env()
    env.task_arrival_forecast[:] = 0.  # Controlled known-only reachable recovery case.
    env.bess_soc = .4
    env.bess_energy_kWh = 40.
    snapshot = build_snapshot(env)
    guard = snapshot.service_guard
    assert len(guard.charge_limits_kw) == snapshot.planning_horizon_steps
    assert guard.charge_limits_kw[0] == guard.charge_limit_kw
    plan = solve_time_indexed_mip_raw_projection(
        snapshot, DispatchProposal(compute_actions=[1.] * 20, storage_action=1.))
    assert plan.solver_status == 'optimal'
    assert all(c <= limit + 1e-6 for c, limit in zip(
        plan.charge_kw, guard.charge_limits_kw, strict=True))
    assert plan.soc_kwh[-1] == pytest.approx(50., abs=1e-6)
