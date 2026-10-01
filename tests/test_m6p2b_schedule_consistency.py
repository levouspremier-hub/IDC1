"""r3 regressions: schedule-coupled recovery and non-vacuous acceptance."""

import numpy as np
import pytest

from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task
from planning.model import solve_time_indexed_mip_raw_projection
from planning.snapshot_adapter import build_snapshot


def fixture_env(*, margin=4.4, task=True):
    from planning.service_guard import SERVICE_GUARD_VERSION

    env = IDCPriceEnv20D(
        horizon=3, forecast_cutoff=3, delta_t_hours=.5,
        bess_soc_init=.481, access_limit_kw=1000., base_load=.3,
        price_t=np.full(3, .2), pv_t=np.zeros(3), wt_t=np.zeros(3),
        carbon_factor_t=np.full(3, .4), T_amb=np.full(3, 25.),
        server_seed=0, task_seed=0, forecast_seed=0)
    env.terminal_inventory_enabled = True
    env.reset(seed=0)
    env.terminal_service_guard_version = SERVICE_GUARD_VERSION
    env.terminal_service_temperature_margin_c = margin
    env.task_arrival_forecast[:] = 0.
    env.access_limit_kw = env._idc_power_kw(
        np.full(env.model.N, env.base_load), 25. + margin) + 2.
    env.tasks = []
    if task:
        known = Task(1, "A", "known interruptible", 0, 1, np.array([.1]),
                     2., 3, 3., True, False)
        known.status = "waiting"
        env.tasks = [known]
    return env


@pytest.mark.parametrize("margin", [0., 4.4])
def test_accurate_forecast_does_not_lose_recovery_by_deferring_known_service(margin):
    from safe_rl.corrector_wrapper import CorrectorWrapper

    env = fixture_env(margin=margin)
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=.25)
    raw = np.zeros(21, dtype=np.float32)
    for _ in range(3):
        _, _, _, _, info = wrapped.step(raw)
        assert info["stage_a_status"] == info["stage_b_status"] == "optimal"
        assert info["inventory_audit"]["target_reachable"] is True
        np.testing.assert_array_equal(info["raw_action"], raw)
    assert env.tasks[0].remaining_work <= 1e-6
    assert env.bess_energy_kWh == pytest.approx(50., abs=1e-6)


def test_target_proof_requires_feasible_known_service_not_arbitrary_slack():
    snapshot = build_snapshot(fixture_env(margin=0.))
    plan = solve_time_indexed_mip_raw_projection(
        snapshot, DispatchProposal(compute_actions=[0.] * 20, storage_action=0.),
        time_limit_s=.25)
    assert plan.solver_status == "optimal"
    assert plan.inventory_audit["target_reachable"] is True
    assert plan.business_gap_work <= 1e-6
    assert plan.deadline_shortfall_work <= 1e-6
    assert plan.inventory_audit["stage_a_terminal_kwh"] == pytest.approx(50., abs=1e-6)
    assert plan.soc_kwh[-1] == pytest.approx(50., abs=1e-6)


def test_forecast_arrival_overflow_is_retained_through_terminal():
    env = fixture_env(task=False)
    capacity = float(np.sum(env.model.C_server) * .5 * env.max_task_load_per_server)
    env.task_arrival_forecast[:] = [0., capacity + 1., 0.]
    guard = build_snapshot(env).service_guard
    assert guard.aggregate_backlog_work[2] == pytest.approx(1.)
    assert guard.aggregate_service_work[2] == pytest.approx(1.)
    assert guard.aggregate_backlog_work[-1] == pytest.approx(0.)


def test_registered_power_upper_bound_covers_nonlinear_idc_and_group_mix():
    from planning.service_guard import power_upper_envelope

    env = fixture_env(task=False)
    rng = np.random.default_rng(0)
    for temp in (20., 25., 35., 80.):
        base, coefficients = power_upper_envelope(env, temp)
        rates = np.asarray(env.model.C_server) * env.delta_t_hours
        for _ in range(20):
            work = rng.uniform(0., env.max_task_load_per_server, env.model.N) * rates
            actual = env._idc_power_kw(np.clip(env.base_load + work / rates, 0., 1.), temp)
            assert actual <= base + np.dot(coefficients, work) + 1e-8


@pytest.mark.leakage
def test_schedule_coupling_never_reads_invisible_future_task_content():
    env = fixture_env()
    before = build_snapshot(env)
    future = Task(2, "A", "unseen", 2, 1, np.array([.1]), 100., 1, 100., False, False)
    env.tasks.append(future)
    assert build_snapshot(env) == before


def test_last_step_unreachable_cannot_hide_planner_consistency_loss():
    from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance

    episode = {
        "episode_complete": True, "service_qualified": True,
        "inventory_qualified": True, "target_qualified": False,
        "physical_violation_count": 0, "fallbacks": 0, "inventory_unproven_steps": 0,
        "final_planning_audit": {"target_reachable": False},
        "terminal_gap_assessment": {"classification": "planner_consistency_loss",
                                    "explanation_complete": True},
    }
    checks = inventory_episode_acceptance(episode)
    assert checks["terminal_gap_explained"] is False
    assert not all(checks.values())


def test_unknown_reachability_is_not_an_unreachable_exception():
    from safe_rl_v2.inventory_diagnostics import inventory_episode_acceptance

    episode = {
        "episode_complete": True, "service_qualified": True,
        "inventory_qualified": True, "target_qualified": False,
        "physical_violation_count": 0, "fallbacks": 0, "inventory_unproven_steps": 1,
        "final_planning_audit": {"target_reachable": None},
        "terminal_gap_assessment": {"classification": "registered_initial_gap",
                                    "explanation_complete": True},
    }
    assert not all(inventory_episode_acceptance(episode).values())


def test_readiness_requires_fallback_gap_review_and_verified_fair_pairing():
    from safe_rl_v2.inventory_diagnostics import validation_ready

    row = {"episode_complete": True, "service_qualified": True,
           "inventory_qualified": True, "target_qualified": True,
           "physical_violation_count": 0, "fallbacks": 1, "inventory_unproven_steps": 0,
           "final_planning_audit": {"target_reachable": True}}
    assert validation_ready([row], expected_episodes=1, pairing_verified=True) is False
    row["fallbacks"] = 0
    assert validation_ready([row], expected_episodes=1, pairing_verified=False) is False
    assert validation_ready([row], expected_episodes=1, pairing_verified=True) is True
