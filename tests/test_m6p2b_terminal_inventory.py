"""M6-P2b: real episode terminal inventory, without weakening physics."""

import numpy as np
import pytest

from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from planning.model import solve_time_indexed_mip_raw_projection
from planning.snapshot_adapter import build_snapshot


def _env(horizon=48, soc=0.5):
    requested_horizon = horizon
    horizon = max(horizon, 2)  # demo generator requires at least two steps
    env = IDCPriceEnv20D(
        horizon=horizon, forecast_cutoff=4, delta_t_hours=0.5,
        bess_soc_init=soc, access_limit_kw=1000.0, base_load=0.3,
        price_t=np.full(horizon, 0.2), pv_t=np.zeros(horizon),
        wt_t=np.zeros(horizon), carbon_factor_t=np.full(horizon, 0.4),
        T_amb=np.full(horizon, 25.0),
    )
    env.terminal_inventory_enabled = True
    env.reset(seed=0)
    if requested_horizon == 1:
        env.current_step = horizon - 1
    return env


def _proposal(storage):
    return DispatchProposal(compute_actions=[0.0] * 20, storage_action=storage)


def test_snapshot_targets_real_episode_end_and_preserves_visible_window():
    env = _env()
    snap = build_snapshot(env)
    assert snap.planning_horizon_steps == 48
    terminal = snap.terminal_inventory
    assert terminal.episode_end_step == 48
    assert terminal.remaining_steps == 48
    assert terminal.target_kwh == 50.0
    assert terminal.lower_kwh == pytest.approx(45.0)
    assert terminal.upper_kwh == pytest.approx(55.0)
    assert sum(snap.planning_forecast.visible_mask) == 4
    env.current_step = 25
    tail = build_snapshot(env)
    assert tail.planning_horizon_steps == 23
    assert tail.terminal_inventory.episode_end_step == 48
    assert tail.terminal_inventory.remaining_steps == 23


@pytest.mark.parametrize('storage', [-1.0, 1.0])
def test_shared_terminal_condition_overrides_last_step_raw(storage):
    snap = build_snapshot(_env(horizon=1))
    result = solve_time_indexed_mip_raw_projection(snap, _proposal(storage))
    assert result.solver_status == 'optimal'
    assert result.inventory_audit['target_reachable'] is True
    assert result.soc_kwh[-1] == pytest.approx(50.0, abs=1e-7)
    assert result.exec_storage_action == pytest.approx(0.0, abs=1e-7)
    assert result.stage_a_objective == pytest.approx(1.0, abs=1e-6)
    assert result.inventory_audit['stage_a_terminal_kwh'] == pytest.approx(50.0)


@pytest.mark.parametrize('soc, direction', [(0.1, -1), (0.9, 1)])
def test_unreachable_target_executes_best_recovery_without_reset(soc, direction):
    snap = build_snapshot(_env(horizon=1, soc=soc))
    result = solve_time_indexed_mip_raw_projection(snap, _proposal(-direction))
    assert result.solver_status == 'optimal'
    assert result.failure_class == 'none'
    assert result.inventory_audit['target_reachable'] is False
    assert result.inventory_audit['band_reachable'] is False
    assert result.inventory_audit['target_gap_kwh'] > 0
    assert result.inventory_audit['band_gap_kwh'] > 0
    assert direction * result.exec_storage_action > 0.99
    assert snap.soc_min_kwh <= result.soc_kwh[-1] <= snap.soc_max_kwh
    assert not (result.charge_kw[0] > 1e-6 and result.discharge_kw[0] > 1e-6)
    assert result.max_constraint_residual < 1e-6


def test_target_unreachable_can_still_reach_existing_band():
    snap = build_snapshot(_env(horizon=1, soc=0.44)).model_copy(
        update={'bess_charge_power_max_kw': 8.0})
    result = solve_time_indexed_mip_raw_projection(snap, _proposal(1.0))
    assert result.inventory_audit['target_reachable'] is False
    assert result.inventory_audit['band_reachable'] is True
    assert 45.0 <= result.soc_kwh[-1] < 50.0


def test_enabled_terminal_metadata_cannot_be_missing():
    snap = build_snapshot(_env()).model_copy(update={'terminal_inventory': None})
    with pytest.raises(ValueError, match='terminal'):
        solve_time_indexed_mip_raw_projection(snap, _proposal(1.0))


@pytest.mark.leakage
def test_outside_visible_window_truth_does_not_change_terminal_plan_input():
    env = _env()
    before = build_snapshot(env)
    for series in (env.price_t, env.pv_t, env.wt_t, env.T_amb, env.carbon_factor_t):
        series[4:] += 1000.0
    after = build_snapshot(env)
    assert before == after


def test_timeout_is_not_a_proof_of_inventory_unreachability(monkeypatch):
    monkeypatch.setattr('planning.model._base_only_terminal_certificate', lambda snapshot: None)
    snap = build_snapshot(_env(horizon=1))
    class Timeout:
        status = 1
        x = None
    monkeypatch.setattr('scipy.optimize.milp', lambda **kwargs: Timeout())
    result = solve_time_indexed_mip_raw_projection(snap, _proposal(1.0))
    assert result.failure_class == 'timeout'
    assert result.inventory_audit['target_reachable'] is None


def test_wrapper_keeps_raw_and_records_actual_inventory():
    from safe_rl.corrector_wrapper import CorrectorWrapper
    wrapped = CorrectorWrapper(_env(horizon=1), corrector_time_limit_s=0.25)
    raw = np.zeros(21, dtype=np.float32)
    raw[-1] = 0.99
    _, _, done, _, info = wrapped.step(raw)
    assert done
    np.testing.assert_array_equal(info['raw_action'], raw)
    assert info['exec_action'][-1] == pytest.approx(0.0, abs=1e-6)
    assert info['inventory_audit']['actual_energy_kwh'] == pytest.approx(50.0)


def test_final_step_timeout_records_unknown_prediction_and_actual_inventory(monkeypatch):
    monkeypatch.setattr('planning.model._base_only_terminal_certificate', lambda snapshot: None)
    from safe_rl.corrector_wrapper import CorrectorWrapper
    class Timeout:
        status = 1
        x = None
    monkeypatch.setattr('scipy.optimize.milp', lambda **kwargs: Timeout())
    wrapped = CorrectorWrapper(_env(horizon=1), corrector_time_limit_s=.25)
    _, _, done, _, info = wrapped.step(np.zeros(21, dtype=np.float32))
    assert done
    assert info['correction_reason'] == 'timeout'
    assert info['inventory_audit']['target_reachable'] is None
    assert info['inventory_audit']['terminal_execution_residual_kwh'] is None
    assert info['inventory_audit']['actual_soc'] == pytest.approx(.5)


def test_observed_half_hour_roundoff_does_not_trigger_zero_fallback():
    import json
    from pathlib import Path

    from contracts.inventory import InventorySnapshot
    fixture = json.loads((Path(__file__).parent / 'fixtures'
                          / 'm6p2b_terminal_roundoff.json').read_text())
    snap = InventorySnapshot.model_validate(fixture['snapshot'])
    raw = fixture['raw_action']
    result = solve_time_indexed_mip_raw_projection(
        snap, DispatchProposal(compute_actions=raw[:20], storage_action=raw[-1]))
    assert result.solver_status == 'optimal'
    assert result.inventory_audit['target_reachable'] is True
    assert result.soc_kwh[-1] == pytest.approx(50.0, abs=1e-6)


@pytest.mark.leakage
def test_formal_inventory_inputs_ignore_future_truth_and_unarrived_task_details():
    from safe_rl_v2.formal_train_loop import build_train_env, load_frozen_training_config

    env, _ = build_train_env(48, master_seed=0, config=load_frozen_training_config())
    env.terminal_inventory_enabled = True
    env.reset(seed=0)
    before = build_snapshot(env)
    for series in (env.price_t, env.pv_t, env.wt_t, env.T_amb, env.carbon_factor_t):
        series[1:] += 1000.0
    for task in env.tasks:
        if task.status == 'not_arrived':
            task.workload *= 1000
            task.remaining_work *= 1000
    after = build_snapshot(env)
    assert before == after
    assert after.planning_horizon_steps == 48
