"""M3.2 接线测试：env 通过 allocate_tasks 施加每任务速率约束（真实 A[i,g] 执行）。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from idc_model.task import Task


def test_env_applies_per_task_rate_limit():
    env = IDCPriceEnv20D()
    env.reset(seed=0)
    # workload=10, duration=2 → max_rate = 5
    t = Task(
        task_id=999,
        profile_key="k",
        name="t",
        arrival_time=0,
        duration=2,
        load_profile=np.array([0.5]),
        workload=10.0,
        deadline=10,
        priority=1.0,
        interruptible=True,
        parallelizable=False,
    )
    t.status = "waiting"
    env.tasks = [t]
    env.Q_t = 10.0
    a = np.concatenate([np.full(20, 1.0, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    _, _, _, _, info = env.step(a)
    assert info["completed_work"] == 5.0  # 单步完成量受速率约束
    assert float(np.asarray(info["completed_work_by_group"]).sum()) == 5.0  # A 列和一致
