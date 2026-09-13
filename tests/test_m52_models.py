"""M5.2 测试：三套价值独立，改变一种 cost 不影响另两种 target。

M5.2a 迁移：`compute_gae` 已移除，改为 `compute_signal_gae`；三套 target API 的
`terminated` / `truncated` 为必填关键字参数。本文件**只补掩码**（全非终止，
等价于迁移前的纯 GAE 语义），**未改动、未删除、未弱化任何断言**。
"""

import numpy as np

from safe_rl_v2.models import compute_signal_gae, compute_three_value_targets

N_STEPS = 3


def _masks(n: int = N_STEPS):
    return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)


def _values() -> dict[str, np.ndarray]:
    return {
        "reward": np.zeros(4),
        "business": np.zeros(4),
        "carbon": np.zeros(4),
    }


def test_compute_gae_shape():
    terminated, truncated = _masks()
    adv, tgt = compute_signal_gae(
        [1.0, 1.0, 1.0], np.zeros(4),
        terminated=terminated, truncated=truncated,
    )
    assert adv.shape == (3,)
    assert tgt.shape == (3,)


def test_three_value_independence():
    terminated, truncated = _masks()
    rewards = np.array([1.0, 1.0, 1.0])
    business = np.array([0.5, 0.5, 0.5])
    carbon = np.array([0.3, 0.3, 0.3])
    r1 = compute_three_value_targets(
        rewards, business, carbon, _values(),
        terminated=terminated, truncated=truncated,
    )

    business_changed = np.array([0.5, 0.5, 9.9])
    r2 = compute_three_value_targets(
        rewards, business_changed, carbon, _values(),
        terminated=terminated, truncated=truncated,
    )

    # 改变业务 cost：收益与碳 target 不变，业务 target 变
    assert np.allclose(r1["reward"][1], r2["reward"][1])
    assert np.allclose(r1["carbon"][1], r2["carbon"][1])
    assert not np.allclose(r1["business"][1], r2["business"][1])


def test_three_value_independence_carbon():
    terminated, truncated = _masks()
    rewards = np.array([1.0, 1.0, 1.0])
    business = np.array([0.5, 0.5, 0.5])
    carbon = np.array([0.3, 0.3, 0.3])
    r1 = compute_three_value_targets(
        rewards, business, carbon, _values(),
        terminated=terminated, truncated=truncated,
    )

    carbon_changed = np.array([0.3, 0.3, 8.8])
    r2 = compute_three_value_targets(
        rewards, business, carbon_changed, _values(),
        terminated=terminated, truncated=truncated,
    )

    assert np.allclose(r1["reward"][1], r2["reward"][1])
    assert np.allclose(r1["business"][1], r2["business"][1])
    assert not np.allclose(r1["carbon"][1], r2["carbon"][1])
