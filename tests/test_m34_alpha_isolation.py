"""M3.4 测试：旧 α（0.40）在正式主链（_loads_from_group_completion）中不产生影响。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D

OLD_ALPHA = 0.40  # 旧设定值


def test_loads_from_group_completion_ignores_alpha():
    env = IDCPriceEnv20D()
    cg = np.full(env.model.N, 100.0, dtype=np.float64)

    env.planned_load_reserve_alpha = 0.0
    r0 = env._loads_from_group_completion(cg)

    env.planned_load_reserve_alpha = OLD_ALPHA
    r1 = env._loads_from_group_completion(cg)

    assert np.allclose(r0, r1)
