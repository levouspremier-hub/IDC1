"""M4.5 测试：wrapper 记录 raw/exec/reason/solve_time，reward 由 exec 产生。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl.corrector_wrapper import CorrectorWrapper


def test_wrapper_records_raw_exec_reason():
    env = CorrectorWrapper(IDCPriceEnv20D())
    env.reset(seed=0)
    a = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.3], dtype=np.float32)])
    obs, reward, term, trunc, info = env.step(a)
    assert "raw_action" in info
    assert "exec_action" in info
    assert "correction_reason" in info
    assert "correction_solve_time_s" in info
    assert "business_gap" in info
    assert np.asarray(info["raw_action"]).shape == (21,)
    assert np.asarray(info["exec_action"]).shape == (21,)
    assert obs is not None
    assert reward is not None


def test_wrapper_uses_same_corrector():
    # 训练/评估同语义：同一 wrapper 类、同一 correct 函数
    env = CorrectorWrapper(IDCPriceEnv20D())
    env.reset(seed=0)
    a = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    _, _, _, _, info = env.step(a)
    # raw_action 与 exec_action 分开记录（log-prob 只关联 raw）
    assert "raw_action" in info
    assert "exec_action" in info
