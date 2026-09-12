"""M5.2 测试：三套价值独立，改变一种 cost 不影响另两种 target。"""

import numpy as np

from safe_rl_v2.models import compute_gae, compute_three_value_targets


def _values() -> dict[str, np.ndarray]:
    return {
        "reward": np.zeros(4),
        "business": np.zeros(4),
        "carbon": np.zeros(4),
    }


def test_compute_gae_shape():
    adv, tgt = compute_gae([1.0, 1.0, 1.0], np.zeros(4))
    assert adv.shape == (3,)
    assert tgt.shape == (3,)


def test_three_value_independence():
    rewards = np.array([1.0, 1.0, 1.0])
    business = np.array([0.5, 0.5, 0.5])
    carbon = np.array([0.3, 0.3, 0.3])
    r1 = compute_three_value_targets(rewards, business, carbon, _values())

    business_changed = np.array([0.5, 0.5, 9.9])
    r2 = compute_three_value_targets(rewards, business_changed, carbon, _values())

    # 改变业务 cost：收益与碳 target 不变，业务 target 变
    assert np.allclose(r1["reward"][1], r2["reward"][1])
    assert np.allclose(r1["carbon"][1], r2["carbon"][1])
    assert not np.allclose(r1["business"][1], r2["business"][1])


def test_three_value_independence_carbon():
    rewards = np.array([1.0, 1.0, 1.0])
    business = np.array([0.5, 0.5, 0.5])
    carbon = np.array([0.3, 0.3, 0.3])
    r1 = compute_three_value_targets(rewards, business, carbon, _values())

    carbon_changed = np.array([0.3, 0.3, 8.8])
    r2 = compute_three_value_targets(rewards, business, carbon_changed, _values())

    assert np.allclose(r1["reward"][1], r2["reward"][1])
    assert np.allclose(r1["business"][1], r2["business"][1])
    assert not np.allclose(r1["carbon"][1], r2["carbon"][1])
