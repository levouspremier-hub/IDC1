"""M4.4a 测试：原始动作最小偏移 MIP 投影 + wrapper 接线。

核心判据：两个不同但都物理可行的 raw proposal，必须得到相应不同的 exec action。
"""

import numpy as np
import pytest

import planning.corrector as corrector_mod
from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from planning.corrector import FailureClass, correct
from planning.model import (
    FAILURE_BASE_SHORTAGE,
    FAILURE_TIMEOUT,
    solve_time_indexed_mip_raw_projection,
)
from planning.snapshot_adapter import build_snapshot
from safe_rl.corrector_wrapper import CorrectorWrapper

HORIZON = 24
CUTOFF = 4
N_GROUP = 20
TOL = 1e-6


def _env(access_limit_kw: float = 1000.0, soc_init: float = 0.5, horizon: int = HORIZON):
    h = horizon
    env = IDCPriceEnv20D(
        horizon=h, forecast_cutoff=CUTOFF, access_limit_kw=access_limit_kw,
        bess_soc_init=soc_init,
        price_t=0.12 + 0.01 * np.arange(h),
        pv_t=np.full(h, 0.5),
        wt_t=np.full(h, 0.4),
        carbon_factor_t=np.full(h, 0.5),
        T_amb=np.full(h, 25.0),
    )
    env.reset(seed=0)
    return env


def _proposal(compute, storage=0.0) -> DispatchProposal:
    return DispatchProposal(compute_actions=list(compute), storage_action=float(storage))


# --- 1. 核心判据：不同可行 raw → 不同 exec ---

def test_two_feasible_raw_proposals_give_different_exec():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    raw_a = _proposal([0.2] * N_GROUP, 0.0)
    raw_b = _proposal([0.8] * N_GROUP, 0.0)
    res_a = solve_time_indexed_mip_raw_projection(snap, raw_a)
    res_b = solve_time_indexed_mip_raw_projection(snap, raw_b)
    assert res_a.solver_status == "optimal" and res_b.solver_status == "optimal"
    assert res_a.exec_compute_actions != res_b.exec_compute_actions
    # 高 raw 应给出不低于低 raw 的 exec 强度（单调性）
    assert sum(res_b.exec_compute_actions) > sum(res_a.exec_compute_actions)


def test_feasible_raw_proposal_offset_is_near_zero():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    raw = _proposal([0.35] * N_GROUP, -0.2)
    res = solve_time_indexed_mip_raw_projection(snap, raw)
    assert res.solver_status == "optimal"
    assert res.projection_offset <= 1e-3


def test_corrector_no_longer_ignores_proposal():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    a = correct(snap, _proposal([0.15] * N_GROUP, 0.0))
    b = correct(snap, _proposal([0.75] * N_GROUP, 0.0))
    assert a.failure == FailureClass.NONE and b.failure == FailureClass.NONE
    assert a.exec_compute_actions != b.exec_compute_actions


# --- 2. 映射正确性 ---

def test_compute_mapping_matches_definition():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 0.0))
    n_task = len(snap.tasks)
    for g in range(N_GROUP):
        cap = float(snap.group_work_capacity[g])
        if cap <= 0:
            assert res.exec_compute_actions[g] == 0.0
            continue
        expected = sum(res.allocation[i][g][0] for i in range(n_task)) / cap
        assert res.exec_compute_actions[g] == pytest.approx(expected, abs=1e-9)


def test_compute_mapping_zero_capacity_no_division_by_zero():
    snap = build_snapshot(_env()).model_copy(
        update={"group_work_capacity": [0.0] + [1.0] * (N_GROUP - 1)}
    )
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 0.0))
    assert res.exec_compute_actions[0] == 0.0
    assert all(np.isfinite(v) for v in res.exec_compute_actions)


def test_storage_mapping_sign_convention():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 0.0))
    expected = (
        res.discharge_kw[0] / snap.bess_discharge_power_max_kw
        - res.charge_kw[0] / snap.bess_charge_power_max_kw
    )
    assert res.exec_storage_action == pytest.approx(expected, abs=1e-9)
    assert -1.0 - TOL <= res.exec_storage_action <= 1.0 + TOL


def test_storage_raw_sign_drives_direction():
    """raw storage 为正（放电）应倾向放电；为负（充电）应倾向充电。"""
    snap = build_snapshot(_env(access_limit_kw=1000.0, soc_init=0.5))
    dis = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 1.0))
    chg = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, -1.0))
    assert dis.exec_storage_action >= chg.exec_storage_action


# --- 3. 约束内最小偏移（而非旧模型的最大完成量） ---

def test_infeasible_raw_is_projectable_and_low_offset():
    """接入受限时 raw 可能物理不可行；投影应落在约束内且偏移尽量小。"""
    snap = build_snapshot(_env(access_limit_kw=16.0, soc_init=0.1))
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([1.0] * N_GROUP, 0.0))
    assert res.solver_status == "optimal"
    # 每个组的 exec 强度不得违反逐组容量导出的上界
    n_task = len(snap.tasks)
    for g in range(N_GROUP):
        used = sum(res.allocation[i][g][0] for i in range(n_task))
        assert used <= snap.group_work_capacity[g] + TOL
    assert res.projection_offset >= 0.0


# --- 4. 失败语义 ---

def test_invalid_proposal_returns_zero_action_with_reason():
    snap = build_snapshot(_env())
    bad = DispatchProposal(compute_actions=[0.5] * (N_GROUP - 1), storage_action=0.0)
    c = correct(snap, bad)
    assert c.failure == FailureClass.PROPOSAL_INVALID
    assert all(v == 0.0 for v in c.exec_compute_actions)
    assert c.exec_storage_action == 0.0
    assert c.business_gap > 0.0


def test_timeout_zero_action(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))

    class _Fake:
        status = 1
        success = False
        message = "fake time limit"
        fun = 0.0
        x = None

    monkeypatch.setattr("scipy.optimize.milp", lambda *a, **k: _Fake())
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 0.0))
    assert res.solver_status == "time_limit"
    assert res.failure_class == FAILURE_TIMEOUT
    assert all(v == 0.0 for v in res.exec_compute_actions)
    assert res.exec_storage_action == 0.0


def test_base_shortage_zero_action():
    env = _env(access_limit_kw=1.0, soc_init=0.1)
    env.pv_t[:] = 0.0
    env.wt_t[:] = 0.0
    snap = build_snapshot(env)
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 0.0))
    assert res.failure_class == FAILURE_BASE_SHORTAGE
    assert all(v == 0.0 for v in res.exec_compute_actions)
    assert res.business_gap_work > 0.0


def test_deadline_shortfall_still_executable():
    """期限不足属业务风险，不等于物理失败：第 0 步物理可行候选仍可执行。"""
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.5] * N_GROUP, 0.0))
    assert res.failure_class in ("none", "deadline_shortfall")
    assert res.solver_status == "optimal"


# --- 5. 旧 solver 不再被 corrector 调用 ---

def test_corrector_does_not_call_legacy_solver(monkeypatch):
    import planning.solver as legacy_solver

    def _boom(*a, **k):
        raise AssertionError("corrector 仍在调用 legacy planning.solver.solve")

    monkeypatch.setattr(legacy_solver, "solve", _boom)
    monkeypatch.setattr(corrector_mod, "solve", _boom, raising=False)
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    c = correct(snap, _proposal([0.5] * N_GROUP, 0.0))
    assert c.failure == FailureClass.NONE


def test_corrector_module_does_not_import_legacy_solver():
    import inspect

    src = inspect.getsource(corrector_mod)
    assert "planning.solver" not in src


# --- 6. 两阶段可审计 ---

def test_two_stage_audit_fields_present():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    res = solve_time_indexed_mip_raw_projection(snap, _proposal([0.4] * N_GROUP, 0.1))
    assert res.backend == "mip"
    assert res.stage_a_status == "optimal"
    assert res.stage_b_status == "optimal"
    assert res.stage_a_solve_time_s >= 0.0
    assert res.stage_b_solve_time_s >= 0.0
    assert res.stage_a_objective >= 0.0
    assert res.stage_b_objective >= 0.0
    assert res.projection_offset >= 0.0


# --- 7. wrapper 接线 ---

def test_wrapper_exec_is_corrector_output_and_raw_preserved():
    env = CorrectorWrapper(IDCPriceEnv20D(horizon=HORIZON, access_limit_kw=1000.0))
    env.reset(seed=0)
    raw = np.concatenate([np.full(N_GROUP, 0.6, dtype=np.float32), np.array([0.2], dtype=np.float32)])
    _, _, _, _, info = env.step(raw)
    np.testing.assert_allclose(np.asarray(info["raw_action"]), raw, atol=0.0)
    assert np.asarray(info["exec_action"]).shape == (N_GROUP + 1,)
    for key in ("planner_backend", "stage_a_status", "stage_b_status",
                "projection_offset", "business_gap"):
        assert key in info, key
    assert info["planner_backend"] == "mip"


def test_wrapper_env_constraints_hold_after_exec():
    env = CorrectorWrapper(IDCPriceEnv20D(horizon=HORIZON, access_limit_kw=18.0))
    env.reset(seed=0)
    raw = np.concatenate([np.full(N_GROUP, 0.9, dtype=np.float32), np.array([0.5], dtype=np.float32)])
    _, _, _, _, info = env.step(raw)
    snap_access = info["access_limit_kw"]
    assert -TOL <= info["P_grid_kW"] <= snap_access + TOL
    assert info["bess_soc"] <= env.env.bess_soc_max + TOL
    assert info["bess_soc"] >= env.env.bess_soc_min - TOL
