"""M3.6a 测试：total_objective_cost 每步累计（账务连续性）。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task


def _env(horizon=6, **kw) -> IDCPriceEnv20D:
    e = IDCPriceEnv20D(horizon=horizon, access_limit_kw=kw.pop("access_limit_kw", 1000.0), **kw)
    e.reset(seed=0)
    return e


def _task(tid, workload, duration, deadline) -> Task:
    t = Task(
        task_id=tid, profile_key="k", name="t", arrival_time=0, duration=duration,
        load_profile=np.array([0.5]), workload=workload, deadline=deadline,
        priority=1.0, interruptible=True, parallelizable=False,
    )
    t.status = "waiting"
    return t


def _action(env) -> np.ndarray:
    return np.concatenate([np.ones(20, dtype=np.float32), np.array([0.0], dtype=np.float32)])


def test_first_non_terminal_step_objective():
    env = _env(horizon=24)
    _, _, term, _, info = env.step(_action(env))
    assert term is False
    assert info["terminal_settlement_penalty"] == 0.0
    assert info["total_objective_cost"] == pytest.approx(
        env.total_cost + env.total_bess_degradation_cost, abs=1e-9
    )
    assert info["total_objective_cost"] > 0.0


def test_multi_step_objective_ledger_accumulates():
    env = _env(horizon=24)
    a = _action(env)
    for _ in range(5):
        _, _, term, _, info = env.step(a)
        assert term is False
        assert info["total_objective_cost"] == pytest.approx(
            env.total_cost + env.total_bess_degradation_cost + env.terminal_settlement_penalty,
            abs=1e-9,
        )
    assert env.total_objective_cost == pytest.approx(
        env.total_cost + env.total_bess_degradation_cost, abs=1e-9
    )


def test_terminal_step_adds_penalty_exactly_once():
    env = _env(horizon=6)
    env.tasks = [_task(1, workload=1000.0, duration=100, deadline=100)]
    env.Q_t = 1000.0
    a = _action(env)
    info = None
    for _ in range(env.horizon):
        _, _, term, _, info = env.step(a)
        if term:
            break
    assert info["terminal_settlement_penalty"] > 0
    assert info["total_objective_cost"] == pytest.approx(
        env.total_cost + env.total_bess_degradation_cost + info["terminal_settlement_penalty"],
        abs=1e-9,
    )
    # 幂等：重复结算不改变累计
    before = env.total_objective_cost
    env._apply_terminal_settlement()
    assert env.total_objective_cost == before


def test_ledger_consistent_after_resume():
    env = _env(horizon=6)
    env.tasks = [_task(1, workload=1000.0, duration=100, deadline=100)]
    env.Q_t = 1000.0
    a = _action(env)
    for _ in range(3):
        env.step(a)
    snap = env.state_dict()
    cont = []
    for _ in range(3):
        _, _, _, _, info = env.step(a)
        cont.append(info["total_objective_cost"])
    env.load_state_dict(snap)
    resumed = []
    for _ in range(3):
        _, _, _, _, info = env.step(a)
        resumed.append(info["total_objective_cost"])
    assert cont == pytest.approx(resumed, abs=1e-9)
