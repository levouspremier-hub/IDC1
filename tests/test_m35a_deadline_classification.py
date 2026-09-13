"""M3.5a 测试：期限分类 / 违约记录 / 终止结算的一致语义。"""

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task

CLASS_KEYS = ("not_due_backlog", "overdue_backlog", "overdue_completed", "on_time_completed")


def _task(tid, workload, duration, deadline, arrival=0, status="waiting") -> Task:
    t = Task(
        task_id=tid, profile_key="k", name="t", arrival_time=arrival, duration=duration,
        load_profile=np.array([0.5]), workload=workload, deadline=deadline,
        priority=1.0, interruptible=True, parallelizable=False,
    )
    t.status = status
    return t


def _action(frac=0.3) -> np.ndarray:
    return np.concatenate([np.full(20, frac, dtype=np.float32), np.array([0.0], dtype=np.float32)])


def _run(env, frac=0.3):
    info = None
    for _ in range(env.horizon):
        _, _, term, _, info = env.step(_action(frac))
        if term:
            break
    return info


# --- 1/2. 互斥完备 ---

def test_classification_mutually_exclusive_and_complete():
    env = IDCPriceEnv20D(horizon=24, access_limit_kw=1000.0)
    env.reset(seed=0)
    for _ in range(24):
        _, _, term, _, info = env.step(_action())
        if term:
            break
    c = info["task_classification"]
    assert set(c) == set(CLASS_KEYS)
    arrived = sum(1 for t in env.tasks if t.status != "not_arrived")
    assert sum(c.values()) == arrived
    assert all(v >= 0 for v in c.values())


def test_not_arrived_tasks_excluded():
    env = IDCPriceEnv20D(horizon=24, access_limit_kw=1000.0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(_action())
    not_arrived = sum(1 for t in env.tasks if t.status == "not_arrived")
    assert not_arrived > 0
    assert sum(info["task_classification"].values()) == len(env.tasks) - not_arrived
    for t in env.tasks:
        if t.status == "not_arrived":
            assert t.task_id not in env.deadline_miss_task_ids


# --- 3. 边界 ---

def test_boundary_finish_equals_deadline_is_on_time():
    """finish_time == latest_finish_time 必须归为按时完成。"""
    env = IDCPriceEnv20D(horizon=10, access_limit_kw=1000.0)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=10.0, duration=1, deadline=2)]  # lft=2
    env.tasks[0].finish_time = 2
    env.tasks[0].remaining_work = 0.0
    env.tasks[0].status = "finished"
    c = env._compute_task_classification(current_time=2)
    assert c["on_time_completed"] == 1 and c["overdue_completed"] == 0


def test_boundary_finish_after_deadline_is_overdue():
    """finish_time == latest_finish_time + 1 必须归为逾期完成。"""
    env = IDCPriceEnv20D(horizon=10, access_limit_kw=1000.0)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=10.0, duration=1, deadline=2)]  # lft=2
    env.tasks[0].finish_time = 3
    env.tasks[0].remaining_work = 0.0
    env.tasks[0].status = "finished"
    c = env._compute_task_classification(current_time=3)
    assert c["overdue_completed"] == 1 and c["on_time_completed"] == 0


def test_late_finish_on_crossing_step_is_recorded():
    """期限跨越那一步正好完成（fin=lft+1）的任务必须被记入违约集合。"""
    env = IDCPriceEnv20D(horizon=10, access_limit_kw=1000.0)
    env.reset(seed=0)
    env.tasks = [_task(1, workload=10.0, duration=1, deadline=1)]  # lft=1, max_rate=10 → 1 步完成
    env.Q_t = 10.0
    _, _, _, _, info = env.step(_action(frac=1.0))  # 在 t=0 完成，finish_time=1 == lft → 按时
    assert info["task_classification"]["on_time_completed"] == 1

    env2 = IDCPriceEnv20D(horizon=10, access_limit_kw=1000.0)
    env2.reset(seed=0)
    # lft=0：任务已逾期，且在 t=0 完成（finish_time=1 > 0）→ 逾期完成且必须记入违约集合
    env2.tasks = [_task(1, workload=10.0, duration=1, deadline=0)]
    env2.Q_t = 10.0
    _, _, _, _, info2 = env2.step(_action(frac=1.0))
    assert info2["task_classification"]["overdue_completed"] == 1
    assert 1 in env2.deadline_miss_task_ids


# --- 4/5/6. 单调去重 + 终止一致性 ---

@pytest.mark.parametrize("frac", [0.1, 0.3, 0.5, 0.9])
def test_terminal_miss_equals_overdue_classification(frac):
    """终止时：去重违约集合大小 == overdue_backlog + overdue_completed。"""
    env = IDCPriceEnv20D(horizon=24, access_limit_kw=1000.0)
    env.reset(seed=0)
    info = _run(env, frac)
    c = info["task_classification"]
    assert info["terminal_deadline_miss_count"] == len(env.deadline_miss_task_ids)
    assert info["terminal_deadline_miss_count"] == c["overdue_backlog"] + c["overdue_completed"]


def test_miss_set_monotonic_and_deduped():
    env = IDCPriceEnv20D(horizon=24, access_limit_kw=1000.0)
    env.reset(seed=0)
    prev: set[int] = set()
    for _ in range(24):
        _, _, term, _, _ = env.step(_action(0.3))
        cur = set(env.deadline_miss_task_ids)
        assert prev <= cur, "违约集合必须单调不减"
        prev = cur
        if term:
            break
    assert len(env.deadline_miss_task_ids) == len(set(env.deadline_miss_task_ids))


def test_terminal_settlement_matches_classification():
    env = IDCPriceEnv20D(horizon=24, access_limit_kw=1000.0)
    env.reset(seed=0)
    info = _run(env, 0.3)
    c = info["task_classification"]
    assert info["settlement"]["deadline_miss_total"] == info["terminal_deadline_miss_count"]
    assert info["terminal_deadline_miss_count"] == c["overdue_backlog"] + c["overdue_completed"]


# --- 连续 / 恢复一致性 ---

def test_resume_preserves_miss_records():
    env = IDCPriceEnv20D(horizon=24, access_limit_kw=1000.0)
    env.reset(seed=0)
    a = _action(0.3)
    for _ in range(12):
        env.step(a)
    snap = env.state_dict()
    mid_miss = set(env.deadline_miss_task_ids)

    for _ in range(12):
        _, _, term, _, info_c = env.step(a)
        if term:
            break
    cont_final = set(env.deadline_miss_task_ids)
    assert mid_miss <= cont_final  # 单调

    env.load_state_dict(snap)
    for _ in range(12):
        _, _, term, _, info_r = env.step(a)
        if term:
            break
    resume_final = set(env.deadline_miss_task_ids)

    assert resume_final == cont_final
    assert info_c["terminal_deadline_miss_count"] == info_r["terminal_deadline_miss_count"]
