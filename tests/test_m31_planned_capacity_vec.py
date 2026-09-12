"""M3.1 测试：分配器应接收长度 20 的 planned_capacity_vec（改造前失败）。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D


def _env() -> IDCPriceEnv20D:
    e = IDCPriceEnv20D()
    e.reset(seed=0)
    return e


def test_planned_capacity_vec_in_info():
    env = _env()
    a = np.full(env.action_dim, 0.5, dtype=np.float32)
    _, _, _, _, info = env.step(a)
    vec = np.asarray(info["planned_capacity_vec"])
    assert vec.shape == (20,)
    by_group = np.asarray(info["completed_work_by_group"])
    assert by_group.shape == (20,)


def test_group_distribution_affects_completed_by_group():
    env = _env()
    neutral = np.full(env.action_dim, 0.5, dtype=np.float32)
    concentrated = np.full(env.action_dim, 0.5, dtype=np.float32)
    concentrated[:20] = 0.0
    concentrated[0] = 1.0
    _, _, _, _, i1 = env.step(neutral)
    env.reset(seed=0)
    _, _, _, _, i2 = env.step(concentrated)
    v1 = np.asarray(i1["completed_work_by_group"])
    v2 = np.asarray(i2["completed_work_by_group"])
    assert not np.allclose(v1, v2)
