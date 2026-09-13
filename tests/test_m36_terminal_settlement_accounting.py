"""M3.6 测试：终止结算账务与违规记录。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task


def _env(**kw) -> IDCPriceEnv20D:
    e = IDCPriceEnv20D(**kw)
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


def _full_compute(env) -> np.ndarray:
    return np.concatenate([np.ones(20, dtype=np.float32), np.array([0.0], dtype=np.float32)])


def _run_to_terminal(env, action_fn):
    info = None
    for _ in range(env.horizon):
        _, _, term, trunc, info = env.step(action_fn(env))
        if term or trunc:
            break
    return info


def test_non_terminal_no_settlement():
    env = _env(horizon=24, access_limit_kw=1000.0)
    _, _, _, _, info = env.step(_full_compute(env))
    assert info["terminal_settlement_penalty"] == 0.0
    assert info["terminal_leftover_work"] == 0.0
    assert info["terminal_service_violation"] == 0


def test_remaining_work_settlement():
    env = _env(horizon=6, access_limit_kw=1000.0)
    env.tasks = [_task(1, workload=1000.0, duration=100, deadline=100)]  # max_rate=10，6步仅60
    env.Q_t = 1000.0
    info = _run_to_terminal(env, _full_compute)
    assert info["terminal_leftover_work"] > 0
    assert info["terminal_service_violation"] == 1
    assert info["total_objective_cost"] == pytest.approx(
        info["total_cost"] + info["total_bess_degradation_cost"]
        + info["terminal_settlement_penalty"],
        abs=1e-6,
    )


def test_deadline_miss_settlement():
    env = _env(horizon=6, access_limit_kw=1000.0)
    # max_rate=1，latest_finish=1，无法按时完成
    env.tasks = [_task(1, workload=100.0, duration=100, deadline=1)]
    env.Q_t = 100.0
    info = _run_to_terminal(env, _full_compute)
    assert info["terminal_deadline_miss_count"] > 0


def test_soc_recovery_settlement():
    env = _env(horizon=6, access_limit_kw=1000.0, bess_soc_init=0.6)

    def discharge(e):
        return np.concatenate([np.ones(20, dtype=np.float32), np.array([0.5], dtype=np.float32)])

    info = _run_to_terminal(env, discharge)
    assert info["terminal_soc_recovery_kwh"] > 0


def test_settlement_accumulates_once():
    env = _env(horizon=4, access_limit_kw=1000.0)
    env.tasks = [_task(1, workload=1000.0, duration=100, deadline=100)]
    env.Q_t = 1000.0
    a = _full_compute(env)
    info = _run_to_terminal(env, lambda e: a)
    penalty = info["terminal_settlement_penalty"]
    before = env.total_objective_cost
    env._apply_terminal_settlement()  # 幂等：不改变累计
    assert env.total_objective_cost == before
    assert env.total_objective_cost == pytest.approx(
        env.total_cost + env.total_bess_degradation_cost + penalty, abs=1e-6
    )


def test_state_dict_resume_settlement_identical():
    env = _env(horizon=6, access_limit_kw=1000.0)
    env.tasks = [_task(1, workload=1000.0, duration=100, deadline=100)]
    env.Q_t = 1000.0
    a = _full_compute(env)
    for _ in range(3):
        env.step(a)
    snap = env.state_dict()
    for _ in range(3):
        _, _, _, _, info_cont = env.step(a)
    env.load_state_dict(snap)
    for _ in range(3):
        _, _, _, _, info_resume = env.step(a)
    assert info_cont["terminal_settlement_penalty"] == pytest.approx(
        info_resume["terminal_settlement_penalty"], abs=1e-9
    )
    assert info_cont["total_objective_cost"] == pytest.approx(
        info_resume["total_objective_cost"], abs=1e-9
    )
