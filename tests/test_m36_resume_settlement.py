"""M3.6 测试：跨日连续性、尾段结算、中断恢复等价。"""

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D


def _neutral(env) -> np.ndarray:
    return np.full(env.action_dim, 0.5, dtype=np.float32)


def test_72h_continuity():
    env = IDCPriceEnv20D(horizon=72)
    env.reset(seed=0)
    ids_0 = {t.task_id for t in env.tasks}
    a = _neutral(env)
    for _ in range(72):
        _, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    assert {t.task_id for t in env.tasks} == ids_0


def test_terminal_settlement_present():
    env = IDCPriceEnv20D(horizon=4)
    env.reset(seed=0)
    a = _neutral(env)
    info = None
    for _ in range(4):
        _, _, _, _, info = env.step(a)
    assert info is not None
    assert "settlement" in info
    s = info["settlement"]
    assert "leftover_work" in s
    assert "deadline_miss_total" in s
    assert "soc_recovery_energy_kwh" in s


def test_resume_equivalence():
    env = IDCPriceEnv20D(horizon=12)
    env.reset(seed=0)
    a = _neutral(env)
    for _ in range(6):
        env.step(a)
    snap = env.state_dict()

    traj_orig = []
    for _ in range(6):
        _, r, _, _, info = env.step(a)
        traj_orig.append((float(r), float(info["total_completed_work"])))

    env.load_state_dict(snap)
    traj_resumed = []
    for _ in range(6):
        _, r, _, _, info = env.step(a)
        traj_resumed.append((float(r), float(info["total_completed_work"])))

    assert traj_orig == traj_resumed
