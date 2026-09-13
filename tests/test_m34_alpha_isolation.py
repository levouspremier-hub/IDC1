"""M3.4a 收尾：α 已退役，_loads_from_group_completion 仅由完成工作/组能力导出。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D


def test_loads_from_group_completion_is_pure_capacity_ratio():
    env = IDCPriceEnv20D()
    cg = np.full(env.model.N, 100.0, dtype=np.float64)
    loads = env._loads_from_group_completion(cg)
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    expected = np.clip(100.0 / np.maximum(c_server, 1e-6), 0.0, env.max_task_load_per_server)
    assert np.allclose(loads, expected)
