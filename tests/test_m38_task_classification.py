"""M3.8 测试：任务四分类互斥完备，逾期完成只计一次。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task


def _task(tid: int, arrival: int, deadline: int, workload: float = 10.0) -> Task:
    return Task(
        task_id=tid,
        profile_key="k",
        name="t",
        arrival_time=arrival,
        duration=1,
        load_profile=np.array([0.5]),
        workload=workload,
        deadline=deadline,
        priority=1.0,
        interruptible=True,
        parallelizable=False,
    )


def test_classification_mutually_exclusive_and_complete():
    env = IDCPriceEnv20D()
    t1 = _task(1, 0, 5)
    t1.remaining_work = 5.0
    t1.status = "waiting"  # 未到期积压（t=3）
    t2 = _task(2, 0, 5)
    t2.remaining_work = 0.0
    t2.status = "finished"
    t2.finish_time = 4  # 按时完成
    t3 = _task(3, 0, 1)
    t3.remaining_work = 5.0
    t3.status = "waiting"  # 逾期积压（t=3）
    t4 = _task(4, 0, 1)
    t4.remaining_work = 0.0
    t4.status = "finished"
    t4.finish_time = 5  # 逾期完成
    env.tasks = [t1, t2, t3, t4]

    counts = env._compute_task_classification(current_time=3)
    assert counts == {
        "not_due_backlog": 1,
        "overdue_backlog": 1,
        "overdue_completed": 1,
        "on_time_completed": 1,
    }


def test_overdue_completed_counted_once():
    env = IDCPriceEnv20D()
    t = _task(1, 0, 1)
    t.remaining_work = 0.0
    t.status = "finished"
    t.finish_time = 5
    env.tasks = [t]

    counts = env._compute_task_classification(current_time=3)
    assert counts["overdue_completed"] == 1
    assert counts["overdue_backlog"] == 0  # 不在积压中重复计数
