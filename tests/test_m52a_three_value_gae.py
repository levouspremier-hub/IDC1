"""M5.2a 测试：三套 value 的终端感知 GAE 数学。

数值样例全部可手算（gamma=lam=0.5），逐条固定期望值，不做近似。
本卡不测策略、不测训练：只测 target/GAE 的数学与严格校验。
"""

import inspect

import numpy as np
import pytest

from safe_rl_v2 import models


def _masks(n: int, *, terminated=None, truncated=None):
    term = np.zeros(n, dtype=bool) if terminated is None else np.asarray(terminated, dtype=bool)
    trunc = np.zeros(n, dtype=bool) if truncated is None else np.asarray(truncated, dtype=bool)
    return term, trunc


def _values3(reward, business, carbon) -> dict[str, np.ndarray]:
    return {
        "reward": np.asarray(reward, dtype=np.float64),
        "business": np.asarray(business, dtype=np.float64),
        "carbon": np.asarray(carbon, dtype=np.float64),
    }


def _reference_gae(signal, values, gamma, lam):
    """独立的朴素 GAE 参考实现（仅用于全非终止情形）。"""
    signal = np.asarray(signal, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(signal)
    gae = np.zeros(n)
    last = 0.0
    for t in reversed(range(n)):
        delta = signal[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        gae[t] = last
    return gae, gae + values[:n]


# --- 1. terminated：bootstrap = 0，递推中断 ---

def test_terminated_blocks_bootstrap_and_breaks_recursion():
    """手算：gamma=lam=0.5，signal=[1,1,1]，terminated 在末步。

    t=2: delta = 1 + 0.5*99*0 - 0 = 1            (bootstrap 被掩掉)
         carry = 0 -> gae[2] = 1
    t=1: delta = 1 + 0.5*0 - 0 = 1; gae[1] = 1 + 0.25*1 = 1.25
    t=0: delta = 1            ; gae[0] = 1 + 0.25*1.25 = 1.3125
    """
    term, trunc = _masks(3, terminated=[False, False, True])
    adv, tgt = models.compute_signal_gae(
        [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 99.0],
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    np.testing.assert_allclose(adv, [1.3125, 1.25, 1.0])
    np.testing.assert_allclose(tgt, [1.3125, 1.25, 1.0])


def test_terminated_ignores_bootstrap_value_entirely():
    """终止步的 values[T] 必须完全不参与计算。"""
    term, trunc = _masks(3, terminated=[False, False, True])
    results = [
        models.compute_signal_gae(
            [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, bootstrap],
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )
        for bootstrap in (0.0, 99.0, -99.0, 1.0e6)
    ]
    first = results[0]
    for other in results[1:]:
        np.testing.assert_array_equal(first[0], other[0])
        np.testing.assert_array_equal(first[1], other[1])


def test_mid_rollout_termination_does_not_leak_next_episode_advantage():
    """第 t 步终止时，第 t+1 步的优势不得反向传播进本条 episode。"""
    # terminated 在第 1 步；第 2、3 步属于下一条 episode
    term, trunc = _masks(4, terminated=[False, True, False, False])
    adv, _ = models.compute_signal_gae(
        [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0],
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    # t=1 终止 -> 不继承 t=2；t=0 只继承 t=1
    np.testing.assert_allclose(adv, [1.25, 1.0, 1.25, 1.0])


# --- 2. truncated：允许 bootstrap，但后续递推中断 ---

def test_truncated_allows_bootstrap_from_next_observation():
    """手算：末步 truncated 时 values[T] 必须被使用。

    truncated 在 t=1: delta = 1 + 0.5*4 - 0 = 3; gae[1] = 3
    t=0: delta = 1 + 0.5*0 - 0 = 1; gae[0] = 1 + 0.25*3 = 1.75
    """
    term, trunc = _masks(2, truncated=[False, True])
    adv, tgt = models.compute_signal_gae(
        [1.0, 1.0], [0.0, 0.0, 4.0],
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    np.testing.assert_allclose(adv, [1.75, 3.0])
    np.testing.assert_allclose(tgt, [1.75, 3.0])


def test_truncated_bootstraps_where_terminated_would_not():
    """同样输入下，truncated 与 terminated 的末步优势必须不同。"""
    signal, values = [1.0, 1.0], [0.0, 0.0, 4.0]
    term_t, trunc_f = _masks(2, terminated=[False, True])
    term_f, trunc_t = _masks(2, truncated=[False, True])

    adv_trunc, _ = models.compute_signal_gae(
        signal, values, terminated=term_f, truncated=trunc_t, gamma=0.5, lam=0.5
    )
    adv_term, _ = models.compute_signal_gae(
        signal, values, terminated=term_t, truncated=trunc_f, gamma=0.5, lam=0.5
    )
    assert adv_trunc[1] == pytest.approx(3.0)
    assert adv_term[1] == pytest.approx(1.0)
    assert adv_trunc[1] != adv_term[1]


def test_truncated_breaks_recursion_after_the_step():
    """手算：truncated 在 t=1，t=0 不得继承 t=1 再往前的递推。

    t=3: delta = 1 + 0.5*0 - 0 = 1        ; gae[3] = 1
    t=2: delta = 1 + 0.5*0 - 5 = -4       ; gae[2] = -4 + 0.25*1 = -3.75
    t=1: delta = 1 + 0.5*5 - 0 = 3.5      ; carry=0 -> gae[1] = 3.5
    t=0: delta = 1 + 0.5*0 - 0 = 1        ; gae[0] = 1 + 0.25*3.5 = 1.875
    """
    term, trunc = _masks(4, truncated=[False, True, False, False])
    adv, tgt = models.compute_signal_gae(
        [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 5.0, 0.0, 0.0],
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    np.testing.assert_allclose(adv, [1.875, 3.5, -3.75, 1.0])
    np.testing.assert_allclose(tgt, [1.875, 3.5, 1.25, 1.0])

    # 反事实：若截断不中断递推，t=1 会继承 t=2，从而 t=0 结果不同
    carry_gae_1 = 3.5 + 0.25 * (-3.75)
    assert 1.0 + 0.25 * carry_gae_1 != pytest.approx(1.875)


# --- 3. 非终止 rollout 截断：正常 bootstrap ---

def test_non_terminal_runout_matches_plain_gae():
    """rollout 因 steps 上限结束（两个掩码全 False）时等于朴素 GAE。

    t=2: 1 ; t=1: 1.25 ; t=0: 1.3125
    """
    term, trunc = _masks(3)
    adv, tgt = models.compute_signal_gae(
        [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    np.testing.assert_allclose(adv, [1.3125, 1.25, 1.0])
    np.testing.assert_allclose(tgt, [1.3125, 1.25, 1.0])


def test_non_terminal_matches_independent_reference():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=7)
    values = rng.normal(size=8)
    term, trunc = _masks(7)

    adv, tgt = models.compute_signal_gae(
        signal, values, terminated=term, truncated=trunc, gamma=0.97, lam=0.9
    )
    ref_adv, ref_tgt = _reference_gae(signal, values, 0.97, 0.9)
    np.testing.assert_allclose(adv, ref_adv)
    np.testing.assert_allclose(tgt, ref_tgt)


def test_target_equals_advantage_plus_value():
    term, trunc = _masks(5, terminated=[False, False, False, False, True])
    values = np.array([0.5, -0.25, 1.0, 0.0, 2.0, 7.0])
    adv, tgt = models.compute_signal_gae(
        [1.0, -1.0, 0.5, 0.0, 3.0], values,
        terminated=term, truncated=trunc, gamma=0.99, lam=0.95,
    )
    np.testing.assert_allclose(tgt, adv + values[:5])


# --- 4. 三套 signal 完全独立 ---

def test_changing_one_signal_leaves_other_targets_untouched():
    rewards = np.array([1.0, 1.0, 1.0, 1.0])
    business = np.array([0.5, 0.5, 0.5, 0.5])
    carbon = np.array([0.3, 0.3, 0.3, 0.3])
    values = _values3(np.zeros(5), np.zeros(5), np.zeros(5))
    term, trunc = _masks(4)

    base = models.compute_three_value_targets(
        rewards, business, carbon, values,
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    for head, mutated in (
        ("reward", rewards.copy()),
        ("business", business.copy()),
        ("carbon", carbon.copy()),
    ):
        mutated[-1] = 9.9
        args = {"reward": rewards, "business": business, "carbon": carbon}
        args[head] = mutated
        out = models.compute_three_value_targets(
            args["reward"], args["business"], args["carbon"], values,
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )
        for other in ("reward", "business", "carbon"):
            if other == head:
                assert not np.allclose(base[other][1], out[other][1]), f"{head} 应变化"
            else:
                np.testing.assert_array_equal(base[other][1], out[other][1])


def test_three_value_heads_do_not_share_critic_values():
    """改动某一头的 critic value，只能影响该头的 target。"""
    rewards = np.array([1.0, 1.0, 1.0])
    business = np.array([0.5, 0.5, 0.5])
    carbon = np.array([0.3, 0.3, 0.3])
    term, trunc = _masks(3)

    base_values = _values3(np.zeros(4), np.zeros(4), np.zeros(4))
    base = models.compute_three_value_targets(
        rewards, business, carbon, base_values,
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )

    mutated_values = _values3(np.zeros(4), np.zeros(4), np.zeros(4))
    # 改碳水位的 bootstrap 估计（values[T]）；注意改 values[0] 会与 target=gae+values 抵消
    mutated_values["carbon"][-1] = 8.0
    mutated = models.compute_three_value_targets(
        rewards, business, carbon, mutated_values,
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    assert not np.allclose(base["carbon"][1], mutated["carbon"][1])
    np.testing.assert_array_equal(base["reward"][1], mutated["reward"][1])
    np.testing.assert_array_equal(base["business"][1], mutated["business"][1])


def test_three_value_api_has_no_electricity_signal():
    """电费不得作为任何一位 target 的输入。"""
    params = set(inspect.signature(models.compute_three_value_targets).parameters)
    assert params == {
        "rewards", "business_violations", "carbon_emissions", "values",
        "terminated", "truncated", "gamma", "lam",
    }
    assert not any("electric" in name or "price" in name for name in params)


# --- 5. gamma / lambda 边界与严格校验 ---

@pytest.mark.parametrize("gamma,lam", [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5)])
def test_accepts_boundary_gamma_lambda(gamma, lam):
    term, trunc = _masks(2)
    adv, tgt = models.compute_signal_gae(
        [1.0, 1.0], [0.0, 0.0, 0.0],
        terminated=term, truncated=trunc, gamma=gamma, lam=lam,
    )
    assert np.all(np.isfinite(adv)) and np.all(np.isfinite(tgt))


@pytest.mark.parametrize(
    "gamma,lam",
    [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1), (np.nan, 0.5), (0.5, np.nan),
     (np.inf, 0.5), (0.5, -np.inf)],
)
def test_rejects_out_of_range_gamma_lambda(gamma, lam):
    term, trunc = _masks(2)
    with pytest.raises((ValueError, TypeError)):
        models.compute_signal_gae(
            [1.0, 1.0], [0.0, 0.0, 0.0],
            terminated=term, truncated=trunc, gamma=gamma, lam=lam,
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_signal_and_values(bad):
    term, trunc = _masks(3)
    signal = np.array([1.0, 1.0, 1.0])
    signal[1] = bad
    with pytest.raises(ValueError):
        models.compute_signal_gae(
            signal, [0.0, 0.0, 0.0, 0.0],
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )

    values = np.array([0.0, 0.0, 0.0, 0.0])
    values[2] = bad
    with pytest.raises(ValueError):
        models.compute_signal_gae(
            [1.0, 1.0, 1.0], values,
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )


@pytest.mark.parametrize(
    "signal_len,values_len,mask_len",
    [(3, 3, 3), (3, 5, 3), (3, 4, 2), (2, 4, 3)],
)
def test_rejects_length_mismatch(signal_len, values_len, mask_len):
    term, trunc = _masks(mask_len)
    with pytest.raises(ValueError):
        models.compute_signal_gae(
            np.ones(signal_len), np.zeros(values_len),
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )


def test_rejects_empty_signal():
    term, trunc = _masks(0)
    with pytest.raises(ValueError):
        models.compute_signal_gae(
            [], [0.0], terminated=term, truncated=trunc, gamma=0.5, lam=0.5
        )


@pytest.mark.parametrize(
    "mask",
    [
        np.array([0, 1, 0]),              # 整数数组
        np.array([0.0, 1.0, 0.0]),        # 浮点数组
        np.array([True, False, None], dtype=object),
    ],
)
def test_rejects_non_bool_masks(mask):
    term = np.array([False, False, False])
    with pytest.raises((TypeError, ValueError)):
        models.compute_signal_gae(
            [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
            terminated=mask, truncated=term, gamma=0.5, lam=0.5,
        )


def test_rejects_terminated_and_truncated_together():
    term = np.array([False, True, False])
    trunc = np.array([False, True, False])
    with pytest.raises(ValueError, match="terminated|truncated|互斥"):
        models.compute_signal_gae(
            [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )


@pytest.mark.parametrize("shape", [(3, 1), (1, 3)])
def test_rejects_non_1d_inputs(shape):
    term, trunc = _masks(3)
    with pytest.raises(ValueError):
        models.compute_signal_gae(
            np.ones(shape), np.zeros(4),
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )


def test_three_value_api_validates_every_head():
    term, trunc = _masks(3)
    values = _values3(np.zeros(4), np.zeros(4), np.zeros(4))
    bad = np.array([1.0, 1.0, np.nan])
    with pytest.raises(ValueError):
        models.compute_three_value_targets(
            np.ones(3), np.ones(3), bad, values,
            terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
        )


# --- 6. 旧无掩码 API 必须已移除（不得静默默认终止语义）---

def test_legacy_maskless_gae_is_removed():
    assert not hasattr(models, "compute_gae"), "旧的无掩码 compute_gae 必须移除"


def test_three_value_api_requires_masks():
    """缺少 terminated/truncated 必须报错，不得静默默认终止语义。"""
    values = _values3(np.zeros(4), np.zeros(4), np.zeros(4))
    with pytest.raises(TypeError):
        models.compute_three_value_targets(
            np.ones(3), np.ones(3), np.ones(3), values
        )
    with pytest.raises(TypeError):
        models.compute_signal_gae(np.ones(3), np.zeros(4))


def test_masks_are_keyword_only():
    sig = inspect.signature(models.compute_signal_gae)
    for name in ("terminated", "truncated"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters[name].default is inspect.Parameter.empty


# --- 7. 回归：跨 episode 的分解不变量（独立于递推实现的交叉验证）---

SIGNAL_5 = [1.0, 2.0, 3.0, 4.0]
VALUES_5 = [0.1, 0.2, 0.3, 0.4, 0.5]
GAMMA, LAM = 0.9, 0.8
# 第 1 步是 episode A 的末步（终止或截断），第 2、3 步属于 episode B
EPISODE_A = slice(0, 2)
EPISODE_B = slice(2, 4)


@pytest.mark.parametrize(
    "terminated,truncated",
    [
        ([False, True, False, False], [False, False, False, False]),   # 中间步终止
        ([False, False, False, False], [False, True, False, False]),   # 中间步截断
    ],
)
def test_mid_rollout_boundary_decomposition(terminated, truncated):
    """跨 episode 边界时，逐 episode 单独计算的结果必须与整段 rollout 一致。

    这是对边界处理（bootstrap 掩码 + 递推中断）的**独立**交叉验证：
    它不依赖递推实现本身，只依赖「episode 之间互不影响」这一语义。
    """
    term, trunc = _masks(4, terminated=terminated, truncated=truncated)
    adv, tgt = models.compute_signal_gae(
        SIGNAL_5, VALUES_5, terminated=term, truncated=trunc, gamma=GAMMA, lam=LAM
    )

    # episode A 单独计算（其末步的 bootstrap 来自 values[2]）
    a_term, a_trunc = _masks(2, terminated=[False, True] if terminated[1] else [False, False],
                             truncated=[False, True] if truncated[1] else [False, False])
    adv_a, tgt_a = models.compute_signal_gae(
        SIGNAL_5[EPISODE_A], VALUES_5[0:3],
        terminated=a_term, truncated=a_trunc, gamma=GAMMA, lam=LAM,
    )

    # episode B 单独计算（非终止 run-out）
    b_term, b_trunc = _masks(2)
    adv_b, tgt_b = models.compute_signal_gae(
        SIGNAL_5[EPISODE_B], VALUES_5[2:5],
        terminated=b_term, truncated=b_trunc, gamma=GAMMA, lam=LAM,
    )

    np.testing.assert_allclose(adv[:2], adv_a)
    np.testing.assert_allclose(tgt[:2], tgt_a)
    np.testing.assert_allclose(adv[2:], adv_b)
    np.testing.assert_allclose(tgt[2:], tgt_b)


def test_mid_rollout_boundary_decomposition_has_teeth():
    """若边界不生效（把边界步当作普通步），分解结果必须不同。"""
    term, trunc = _masks(4, terminated=[False, True, False, False])
    adv_boundary, _ = models.compute_signal_gae(
        SIGNAL_5, VALUES_5, terminated=term, truncated=trunc, gamma=GAMMA, lam=LAM
    )
    term_f, trunc_f = _masks(4)  # 反事实：把边界步当普通步
    no_mask, _ = models.compute_signal_gae(
        SIGNAL_5, VALUES_5, terminated=term_f, truncated=trunc_f, gamma=GAMMA, lam=LAM
    )
    assert not np.allclose(adv_boundary, no_mask)


def test_all_ones_gamma_lambda_with_zero_signal_is_zero():
    term, trunc = _masks(3)
    adv, tgt = models.compute_signal_gae(
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
        terminated=term, truncated=trunc, gamma=1.0, lam=1.0,
    )
    np.testing.assert_array_equal(adv, np.zeros(3))
    np.testing.assert_array_equal(tgt, np.zeros(3))


def test_returns_exactly_the_three_named_heads():
    term, trunc = _masks(2)
    values = _values3(np.zeros(3), np.zeros(3), np.zeros(3))
    out = models.compute_three_value_targets(
        np.ones(2), np.ones(2), np.ones(2), values,
        terminated=term, truncated=trunc, gamma=0.5, lam=0.5,
    )
    assert set(out) == set(models.SIGNAL_HEADS) == {"reward", "business", "carbon"}


def test_module_exposes_only_the_two_mask_requiring_apis():
    """不得再出现任何无掩码的 GAE 入口。"""
    assert set(models.__all__) == {"compute_signal_gae", "compute_three_value_targets"}
    for name in dir(models):
        if name.startswith("_"):
            continue
        obj = getattr(models, name)
        if not callable(obj):
            continue
        params = inspect.signature(obj).parameters
        assert "terminated" in params and "truncated" in params, f"{name} 缺少终端掩码"
