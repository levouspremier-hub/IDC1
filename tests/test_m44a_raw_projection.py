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
    FAILURE_SOLVER_FAILURE,
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


def _with_big_task(env, workload: float = 5000.0, duration: int = 1):
    """注入速率与剩余工作都充裕的任务，使中等 raw 强度确实可达。"""
    from idc_model.task import Task

    task = Task(
        task_id=999, profile_key="k", name="t", arrival_time=0, duration=duration,
        load_profile=np.array([0.5]), workload=workload, deadline=HORIZON + 10,
        priority=1.0, interruptible=True, parallelizable=False,
    )
    task.status = "waiting"
    env.tasks = [task]
    env.Q_t = workload
    return env


def test_feasible_raw_proposal_offset_is_near_zero():
    """raw 确实物理可达时，投影偏移应 ≈ 0（投影不扭曲可行 raw）。"""
    snap = build_snapshot(_with_big_task(_env(access_limit_kw=1000.0)))
    raw = _proposal([0.35] * N_GROUP, 0.0)
    res = solve_time_indexed_mip_raw_projection(snap, raw)
    assert res.solver_status == "optimal"
    assert res.projection_offset <= 1e-3
    assert res.exec_compute_actions == pytest.approx([0.35] * N_GROUP, abs=1e-3)


def test_corrector_no_longer_ignores_proposal():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    a = correct(snap, _proposal([0.15] * N_GROUP, 0.0), time_limit_s=5.0)
    b = correct(snap, _proposal([0.75] * N_GROUP, 0.0), time_limit_s=5.0)
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
    c = correct(snap, bad, time_limit_s=5.0)
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
    c = correct(snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=5.0)
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
    env = CorrectorWrapper(
        IDCPriceEnv20D(horizon=HORIZON, access_limit_kw=1000.0),
        corrector_time_limit_s=5.0,
    )
    env.reset(seed=0)
    raw = np.concatenate(
        [np.full(N_GROUP, 0.6, dtype=np.float32), np.array([0.2], dtype=np.float32)]
    )
    _, _, _, _, info = env.step(raw)
    np.testing.assert_allclose(np.asarray(info["raw_action"]), raw, atol=0.0)
    assert np.asarray(info["exec_action"]).shape == (N_GROUP + 1,)
    for key in ("planner_backend", "stage_a_status", "stage_b_status",
                "projection_offset", "business_gap"):
        assert key in info, key
    assert info["planner_backend"] == "mip"


def test_wrapper_env_constraints_hold_after_exec():
    env = CorrectorWrapper(
        IDCPriceEnv20D(horizon=HORIZON, access_limit_kw=18.0),
        corrector_time_limit_s=5.0,
    )
    env.reset(seed=0)
    raw = np.concatenate(
        [np.full(N_GROUP, 0.9, dtype=np.float32), np.array([0.5], dtype=np.float32)]
    )
    _, _, _, _, info = env.step(raw)
    snap_access = info["access_limit_kw"]
    assert -TOL <= info["P_grid_kW"] <= snap_access + TOL
    assert info["bess_soc"] <= env.env.bess_soc_max + TOL
    assert info["bess_soc"] >= env.env.bess_soc_min - TOL


# --- 8. 全局时间预算与 timeout 保真（M4.4a1） ---

import planning.model as model_mod  # noqa: E402


class _FakeMilp:
    """记录每次调用收到的 options.time_limit，并按脚本返回状态。"""

    def __init__(self, statuses, clock):
        self.statuses = list(statuses)
        self.clock = clock
        self.calls: list[float | None] = []

    def __call__(self, **kw):
        opts = kw.get("options") or {}
        self.calls.append(opts.get("time_limit"))

        class _R:
            pass

        r = _R()
        r.status = self.statuses.pop(0) if self.statuses else 0
        r.success = r.status == 0
        r.message = f"fake {r.status}"
        r.fun = 0.0
        cons = kw.get("constraints") or []
        n_vars = cons[0].A.shape[1] if cons else 0
        r.x = np.zeros(n_vars) if r.status == 0 else None
        return r


def _fixed_clock(values):
    """返回一个每次调用推进既定步长的单调时钟。"""
    state = {"i": 0}

    def _now():
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        return values[i]

    return _now


def test_stage_b_timeout_is_faithfully_reported(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([0, 1], None))
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.stage_a_status == "optimal"
    assert res.stage_b_status == "time_limit"
    assert res.solver_status == "time_limit"
    assert res.failure_class == FAILURE_TIMEOUT      # 不得被改写为 solver_failure
    assert all(v == 0.0 for v in res.exec_compute_actions)
    assert res.exec_storage_action == 0.0
    assert res.business_gap_work > 0.0


def test_timeout_does_not_run_base_diagnostic(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    called = {"n": 0}
    real = model_mod.diagnose_base_feasibility

    def _spy(s):
        called["n"] += 1
        return real(s)

    monkeypatch.setattr(model_mod, "diagnose_base_feasibility", _spy)
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([0, 1], None))
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.failure_class == FAILURE_TIMEOUT
    assert called["n"] == 0


def test_stage_b_budget_leq_remaining_after_stage_a(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    clock = _fixed_clock([0.0, 2.0, 2.0, 2.0, 2.0, 3.0])
    monkeypatch.setattr(model_mod, "_monotonic", clock)
    fake = _FakeMilp([0, 0], None)
    monkeypatch.setattr("scipy.optimize.milp", fake)
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.solver_status == "optimal"
    assert len(fake.calls) == 2
    a_budget, b_budget = fake.calls
    assert a_budget is not None and b_budget is not None
    assert a_budget <= 10.0 + 1e-9
    assert b_budget <= a_budget + 1e-9          # 阶段 B 只能用剩余预算


def test_stage_b_not_started_when_budget_exhausted(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    # deadline=1.0；阶段 A 结束时已 5.0 → 预算耗尽
    clock = _fixed_clock([0.0, 5.0, 5.0])
    monkeypatch.setattr(model_mod, "_monotonic", clock)
    fake = _FakeMilp([0, 0], None)
    monkeypatch.setattr("scipy.optimize.milp", fake)
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=1.0
    )
    assert len(fake.calls) == 1                  # 阶段 B 根本未被调用
    assert res.solver_status == "time_limit"
    assert res.failure_class == FAILURE_TIMEOUT
    assert res.stage_b_status == "not_run"


def test_stage_b_solver_error_is_not_timeout(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([0, 4], None))
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.failure_class == FAILURE_SOLVER_FAILURE
    assert res.failure_class != FAILURE_TIMEOUT
    assert all(v == 0.0 for v in res.exec_compute_actions)


def test_failure_keeps_structural_audit(monkeypatch):
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([0, 1], None))
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.backend == "mip"
    assert res.horizon_steps == snap.planning_horizon_steps
    assert res.n_variables > 0
    assert res.n_integer_variables == snap.planning_horizon_steps
    assert res.stage_a_solve_time_s >= 0.0
    assert res.stage_b_solve_time_s >= 0.0


def test_corrector_requires_positive_time_budget():
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    p = _proposal([0.5] * N_GROUP, 0.0)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="time_limit"):
            correct(snap, p, time_limit_s=bad)


def test_wrapper_passes_explicit_budget(monkeypatch):
    seen = {}
    import safe_rl.corrector_wrapper as wrap_mod
    real = wrap_mod.correct

    def _spy(snapshot, proposal, *, time_limit_s):
        seen["budget"] = time_limit_s
        return real(snapshot, proposal, time_limit_s=time_limit_s)

    monkeypatch.setattr(wrap_mod, "correct", _spy)
    env = CorrectorWrapper(
        IDCPriceEnv20D(horizon=HORIZON, access_limit_kw=1000.0),
        corrector_time_limit_s=2.5,
    )
    env.reset(seed=0)
    a = np.concatenate([np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    env.step(a)
    assert seen["budget"] == 2.5


def test_wrapper_timeout_zero_action_and_raw_preserved(monkeypatch):
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([0, 1], None))
    env = CorrectorWrapper(
        IDCPriceEnv20D(horizon=HORIZON, access_limit_kw=1000.0),
        corrector_time_limit_s=5.0,
    )
    env.reset(seed=0)
    raw = np.concatenate(
        [np.full(N_GROUP, 0.7, dtype=np.float32), np.array([0.3], dtype=np.float32)]
    )
    _, _, _, _, info = env.step(raw)
    # 向环境发送完整 21 维零动作
    assert np.asarray(info["exec_action"]).shape == (N_GROUP + 1,)
    assert np.allclose(np.asarray(info["exec_action"]), 0.0)
    # raw 不被改写
    np.testing.assert_allclose(np.asarray(info["raw_action"]), raw, atol=0.0)
    assert info["correction_reason"] == "timeout"
    assert info["business_gap"] > 0.0


# --- 9. 诊断纳入全局 deadline（M4.4a2） ---


class _FakeLinprog:
    """模拟诊断用的 linprog：记录 options.time_limit，按脚本返回状态。"""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.budgets: list[float | None] = []

    def __call__(self, **kw):
        self.budgets.append((kw.get("options") or {}).get("time_limit"))

        class _R:
            pass

        r = _R()
        r.status = self.statuses.pop(0) if self.statuses else 0
        r.success = r.status == 0
        r.message = f"fake lp {r.status}"
        r.fun = 0.0
        r.x = None
        return r


def test_projection_main_infeasible_budget_exhausted_no_diagnostic(monkeypatch):
    """主阶段 infeasible 且预算耗尽 → 不启动诊断，整体 timeout。"""
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    called = {"n": 0}
    real = model_mod.diagnose_base_feasibility

    def _spy(s, **kw):
        called["n"] += 1
        return real(s, **kw)

    monkeypatch.setattr(model_mod, "diagnose_base_feasibility", _spy)
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([2], None))
    clock = _fixed_clock([0.0, 5.0, 5.0, 5.0])
    monkeypatch.setattr(model_mod, "_monotonic", clock)
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=1.0
    )
    assert called["n"] == 0                     # 诊断未被启动
    assert res.diagnostic_status == "not_started"
    assert res.solver_status == "time_limit"
    assert res.failure_class == FAILURE_TIMEOUT # 不得误报 base_shortage


def test_projection_diagnostic_timeout_is_overall_timeout(monkeypatch):
    """诊断 LP timeout → 整体 timeout、零动作、业务缺口显式。"""
    snap = build_snapshot(_env(access_limit_kw=1000.0))
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([2], None))
    monkeypatch.setattr(model_mod, "linprog", _FakeLinprog([1]))
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.solver_status == "time_limit"
    assert res.failure_class == FAILURE_TIMEOUT
    assert all(v == 0.0 for v in res.exec_compute_actions)
    assert res.exec_storage_action == 0.0
    assert res.business_gap_work > 0.0
    assert "time_limit" in res.diagnostic_status   # 诊断自身 timeout 被如实记录


def test_projection_in_budget_diagnostic_confirms_base_shortage(monkeypatch):
    """预算内诊断确认 base-only 不可行 → base_shortage。"""
    snap = build_snapshot(_env(access_limit_kw=1.0, soc_init=0.1))
    real_diag = model_mod.diagnose_base_feasibility
    seen = {}

    def _spy(s, **kw):
        seen.update(kw)
        return real_diag(s, **kw)

    monkeypatch.setattr(model_mod, "diagnose_base_feasibility", _spy)
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([2], None))
    res = solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert res.failure_class == FAILURE_BASE_SHORTAGE
    assert "time_limit_s" in seen and seen["time_limit_s"] is not None


def test_diagnostic_budget_within_global_deadline(monkeypatch):
    """断言诊断获得的预算不超过同一全局 deadline 的剩余量。"""
    snap = build_snapshot(_env(access_limit_kw=1.0, soc_init=0.1))
    fake_lp = _FakeLinprog([2, 0])
    monkeypatch.setattr("scipy.optimize.milp", _FakeMilp([2], None))
    monkeypatch.setattr(model_mod, "linprog", fake_lp)
    # deadline = 0 + 10 = 10；主 milp 前后时钟推进到 3
    clock = _fixed_clock([0.0, 3.0, 3.0, 4.0, 4.0, 5.0])
    monkeypatch.setattr(model_mod, "_monotonic", clock)
    solve_time_indexed_mip_raw_projection(
        snap, _proposal([0.5] * N_GROUP, 0.0), time_limit_s=10.0
    )
    assert fake_lp.budgets, "诊断未被调用"
    for b in fake_lp.budgets:
        assert b is not None and b <= 10.0 + 1e-9
