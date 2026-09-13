"""M5.3b 测试：独立 multiplier 接入 actor objective。

有效优势必须是 A_reward − λ_b·A_business − λ_c·A_carbon；
本轮用**更新前**乘子，乘子更新发生在 optimizer.step() 之后。
不测 PPO ratio/clip（本卡不实现）。
"""

import ast
import copy
import pathlib

import numpy as np
import pytest
import torch

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2 import models as models_mod
from safe_rl_v2 import train as train_mod
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, Transition
from safe_rl_v2.lagrangian import (
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.train import dry_run_update

OBS_DIM = 8
HEADS = ("reward", "business", "carbon")

# 夹具的每 transition 量（单位：违规任务·步 / kgCO2e）
REWARDS = [1.0, -0.5, 2.0]
BUSINESS = [1.0, 3.0, 2.0]
CARBON = [4.0, 6.0, 5.0]
ELECTRICITY = [7.0, 8.0, 9.0]

# 预置乘子：λ_b = 0.1*(9-5) = 0.4，λ_c = 0.2*(13-3) = 2.0
SEED_BUSINESS = 9.0
SEED_CARBON = 13.0
LAMBDA_BUSINESS = 0.4
LAMBDA_CARBON = 2.0


def make_specs(business_budget=5.0, carbon_budget=3.0,
               business_lr=0.1, carbon_lr=0.2) -> tuple[ConstraintSpec, ...]:
    return (
        ConstraintSpec(
            name="business", budget=business_budget, unit=UNIT_VIOLATION_TASK_STEPS,
            learning_rate=business_lr, max_multiplier=100.0,
        ),
        ConstraintSpec(
            name="carbon", budget=carbon_budget, unit=UNIT_KG_CO2E,
            learning_rate=carbon_lr, max_multiplier=100.0,
        ),
    )


def seeded_lagrangian() -> Lagrangian:
    """预置非零乘子，使 actor objective 中的约束项非平凡。"""
    lag = Lagrangian(make_specs())
    lag.update({"business": [SEED_BUSINESS], "carbon": [SEED_CARBON]})
    return lag


def make_policy(seed: int = 0) -> SafePPOPolicy:
    torch.manual_seed(seed)
    return SafePPOPolicy(obs_dim=OBS_DIM)


def make_transition(
    index: int, *, exec_value: float = 0.5, electricity: float | None = None
) -> Transition:
    return Transition(
        observation=np.full(OBS_DIM, 0.1 * (index + 1), dtype=np.float32),
        next_observation=np.full(OBS_DIM, 0.2 * (index + 1), dtype=np.float32),
        raw_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
        old_raw_log_prob=-1.0,
        exec_action=np.full(ACTION_DIM, exec_value, dtype=np.float32),
        reward=REWARDS[index],
        business_cost=BUSINESS[index],
        carbon_cost=CARBON[index],
        electricity_cost_sgd=ELECTRICITY[index] if electricity is None else electricity,
        terminated=False,
        truncated=False,
        correction_info={},
    )


def install_fake_collector(monkeypatch, transitions: list[Transition]) -> None:
    def fake(env, policy, buffer, *, steps, seed=0, corrector_on=False,
             corrector_time_limit_s=None, generator=None):
        for t in transitions:
            buffer.add(t)
        return {
            "corrector_on": corrector_on,
            "corrector_time_limit_s": corrector_time_limit_s,
            "steps_requested": steps,
            "transitions": len(transitions),
            "raw_exec_difference_count": 0,
            "terminated_count": 0,
            "truncated_count": 0,
            "contract_version": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
            "env_seed": seed,
            "policy_rng_source": (
                "explicit_generator" if generator is not None else "global_torch_rng"
            ),
        }

    monkeypatch.setattr(train_mod, "collect_rollout", fake)


def run_dry(policy, lagrangian, optimizer):
    return dry_run_update(
        None, policy, lagrangian, optimizer,
        steps=len(REWARDS), seed=0, corrector_on=False, generator=None,
    )


def independent_heads(policy: SafePPOPolicy, transitions: list[Transition]) -> dict:
    """独立重算三头优势与 raw likelihood（只依赖 buffer 字段与 M5.2c 数学）。"""
    obs = torch.as_tensor(
        np.stack([t.observation for t in transitions]), dtype=torch.float32
    )
    next_obs = torch.as_tensor(
        np.stack([t.next_observation for t in transitions]), dtype=torch.float32
    )
    raw = torch.as_tensor(
        np.stack([t.raw_action for t in transitions]), dtype=torch.float32
    )
    with torch.no_grad():
        _, current_v = policy.forward(obs)
        _, next_v = policy.forward(next_obs)
    targets = models_mod.compute_three_value_targets(
        np.array([t.reward for t in transitions]),
        np.array([t.business_cost for t in transitions]),
        np.array([t.carbon_cost for t in transitions]),
        {h: current_v[:, i].numpy() for i, h in enumerate(HEADS)},
        {h: next_v[:, i].numpy() for i, h in enumerate(HEADS)},
        terminated=np.zeros(len(transitions), dtype=bool),
        truncated=np.zeros(len(transitions), dtype=bool),
        gamma=train_mod.GAMMA,
        lam=train_mod.LAM,
    )
    log_probs = policy.evaluate_raw_actions(obs, raw)
    return {
        "advantages": {h: targets[h][0] for h in HEADS},
        "log_probs": log_probs.detach().numpy(),
    }


def setup(monkeypatch, *, exec_value: float = 0.5):
    transitions = [make_transition(i, exec_value=exec_value) for i in range(len(REWARDS))]
    install_fake_collector(monkeypatch, transitions)
    policy = make_policy()
    snapshot = copy.deepcopy(policy)
    lagrangian = seeded_lagrangian()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return transitions, policy, snapshot, lagrangian, optimizer


# --- 1. 公式与数值 ---

def test_actor_objective_source_is_declared(monkeypatch):
    _, policy, _, lag, opt = setup(monkeypatch)
    result = run_dry(policy, lag, opt)
    assert result["actor_objective_source"] == (
        "A_reward - lambda_business * A_business - lambda_carbon * A_carbon"
    )
    assert result["effective_advantage"]["formula"] == result["actor_objective_source"]
    assert result["effective_advantage"]["multipliers_used"] == "pre_update"


def test_effective_advantage_matches_the_formula(monkeypatch):
    transitions, policy, snapshot, lag, opt = setup(monkeypatch)
    pre = dict(lag.multipliers())
    assert pre["business"] == pytest.approx(LAMBDA_BUSINESS)
    assert pre["carbon"] == pytest.approx(LAMBDA_CARBON)

    result = run_dry(policy, lag, opt)

    ref = independent_heads(snapshot, transitions)
    expected = (
        ref["advantages"]["reward"]
        - LAMBDA_BUSINESS * ref["advantages"]["business"]
        - LAMBDA_CARBON * ref["advantages"]["carbon"]
    )
    assert result["effective_advantage"]["mean"] == pytest.approx(float(expected.mean()))
    assert result["effective_advantage"]["min"] == pytest.approx(float(expected.min()))
    assert result["effective_advantage"]["max"] == pytest.approx(float(expected.max()))


def test_actor_loss_equals_negative_mean_of_effective_advantage_times_logprob(monkeypatch):
    transitions, policy, snapshot, lag, opt = setup(monkeypatch)
    result = run_dry(policy, lag, opt)

    ref = independent_heads(snapshot, transitions)
    expected = -float(
        np.mean(
            (
                ref["advantages"]["reward"]
                - LAMBDA_BUSINESS * ref["advantages"]["business"]
                - LAMBDA_CARBON * ref["advantages"]["carbon"]
            )
            * ref["log_probs"]
        )
    )
    assert result["actor_loss"] == pytest.approx(expected)


def test_terms_decompose_the_effective_advantage(monkeypatch):
    _, policy, _, lag, opt = setup(monkeypatch)
    result = run_dry(policy, lag, opt)
    terms = result["effective_advantage_terms"]
    total = terms["reward_mean"] + terms["business_mean"] + terms["carbon_mean"]
    assert total == pytest.approx(result["effective_advantage"]["mean"])


def test_without_multipliers_actor_loss_reduces_to_reward_only(monkeypatch):
    """两个乘子都为 0 时，有效优势必须退化为 A_reward。"""
    transitions = [make_transition(i) for i in range(len(REWARDS))]
    install_fake_collector(monkeypatch, transitions)
    policy = make_policy()
    snapshot = copy.deepcopy(policy)
    lag = Lagrangian(make_specs())  # 未更新 -> 乘子全 0
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(policy, lag, opt)
    ref = independent_heads(snapshot, transitions)
    assert result["effective_advantage"]["mean"] == pytest.approx(
        float(ref["advantages"]["reward"].mean())
    )
    assert result["effective_advantage_terms"]["business_mean"] == pytest.approx(0.0)
    assert result["effective_advantage_terms"]["carbon_mean"] == pytest.approx(0.0)


# --- 2. 时序：pre-update 乘子，step 之后才更新 ---

def test_actor_uses_pre_update_multipliers(monkeypatch):
    _, policy, _, lag, opt = setup(monkeypatch)
    assert lag.multipliers()["carbon"] == pytest.approx(LAMBDA_CARBON)

    result = run_dry(policy, lag, opt)

    assert result["multipliers_pre_update"]["carbon"] == pytest.approx(LAMBDA_CARBON)
    # 本轮 rollout 的 carbon mean = 5.0 > budget 3.0 -> 乘子上升
    assert result["multipliers_post_update"]["carbon"] == pytest.approx(2.4)
    assert result["multipliers_post_update"] != result["multipliers_pre_update"]
    # 既有键保持 post-update 语义
    assert result["multipliers"] == result["multipliers_post_update"]


def test_multiplier_update_happens_after_optimizer_step(monkeypatch):
    _, policy, _, lag, opt = setup(monkeypatch)
    order: list[str] = []

    real_step = opt.step
    real_update = lag.update

    def step_spy(*args, **kwargs):
        order.append("optimizer.step")
        return real_step(*args, **kwargs)

    def update_spy(*args, **kwargs):
        order.append("lagrangian.update")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(opt, "step", step_spy)
    monkeypatch.setattr(lag, "update", update_spy)

    run_dry(policy, lag, opt)
    assert order == ["optimizer.step", "lagrangian.update"]


def test_constraint_means_are_per_transition_means(monkeypatch):
    _, policy, _, lag, opt = setup(monkeypatch)
    result = run_dry(policy, lag, opt)
    assert result["constraint_means"]["business"] == pytest.approx(float(np.mean(BUSINESS)))
    assert result["constraint_means"]["carbon"] == pytest.approx(float(np.mean(CARBON)))
    assert result["constraint_budgets"] == {"business": 5.0, "carbon": 3.0}
    assert result["constraint_units"] == {
        "business": UNIT_VIOLATION_TASK_STEPS,
        "carbon": UNIT_KG_CO2E,
    }


# --- 3. 约束独立性 ---

def test_zero_multiplier_leaves_its_term_zero(monkeypatch):
    transitions = [make_transition(i) for i in range(len(REWARDS))]
    install_fake_collector(monkeypatch, transitions)
    policy = make_policy()
    lag = Lagrangian(make_specs())
    lag.update({"business": [SEED_BUSINESS], "carbon": [3.0]})  # carbon 乘子 = 0，business = 0.4
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(policy, lag, opt)
    assert result["multipliers_pre_update"]["carbon"] == 0.0
    assert result["effective_advantage_terms"]["carbon_mean"] == pytest.approx(0.0)
    assert result["effective_advantage_terms"]["business_mean"] != pytest.approx(0.0)


def test_increasing_one_multiplier_changes_only_its_own_term(monkeypatch):
    """只改 carbon 乘子：reward 项与 business 项必须逐值不变。"""
    results = []
    for carbon_seed in (13.0, 30.0):  # λ_c = 2.0 / 5.4
        transitions = [make_transition(i) for i in range(len(REWARDS))]
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, transitions)
            policy = make_policy()
            lag = Lagrangian(make_specs())
            lag.update({"business": [SEED_BUSINESS], "carbon": [carbon_seed]})
            opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
            results.append(run_dry(policy, lag, opt))

    low, high = results
    assert low["effective_advantage_terms"]["reward_mean"] == pytest.approx(
        high["effective_advantage_terms"]["reward_mean"]
    )
    assert low["effective_advantage_terms"]["business_mean"] == pytest.approx(
        high["effective_advantage_terms"]["business_mean"]
    )
    assert low["effective_advantage_terms"]["carbon_mean"] != pytest.approx(
        high["effective_advantage_terms"]["carbon_mean"]
    )


# --- 4. exec_action / 电费 / 未来真值 不得进入 ---

def test_exec_action_does_not_affect_actor_objective_or_multipliers(monkeypatch):
    def run(exec_value: float) -> dict:
        transitions = [
            make_transition(i, exec_value=exec_value) for i in range(len(REWARDS))
        ]
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, transitions)
            policy = make_policy()
            lag = seeded_lagrangian()
            opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
            return run_dry(policy, lag, opt)

    base, mutated = run(0.5), run(-1.0)
    assert base["actor_loss"] == pytest.approx(mutated["actor_loss"])
    assert base["effective_advantage"] == mutated["effective_advantage"]
    assert base["multipliers_post_update"] == mutated["multipliers_post_update"]


def test_electricity_does_not_affect_actor_objective_or_multipliers(monkeypatch):
    def run(electricity: float) -> dict:
        transitions = [
            make_transition(i, electricity=electricity) for i in range(len(REWARDS))
        ]
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, transitions)
            policy = make_policy()
            lag = seeded_lagrangian()
            opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
            return run_dry(policy, lag, opt)

    base, mutated = run(ELECTRICITY[0]), run(9999.0)
    assert base["actor_loss"] == pytest.approx(mutated["actor_loss"])
    assert base["effective_advantage"] == mutated["effective_advantage"]
    assert base["multipliers_post_update"] == mutated["multipliers_post_update"]


@pytest.mark.leakage
def test_future_truth_does_not_affect_actor_objective_or_multipliers():
    def run(mutate: bool) -> dict:
        env = IDCPriceEnv20D(
            task_seed=0, server_seed=0, forecast_seed=300000, forecast_cutoff=2
        )
        if mutate:
            env.price_t[20] = 9999.0
            env.pv_t[20] = 9999.0
            env.carbon_factor_t[20] = 9999.0
        torch.manual_seed(0)
        policy = SafePPOPolicy(obs_dim=env.obs_dim)
        lag = Lagrangian(make_specs())
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        gen = torch.Generator()
        gen.manual_seed(0)
        return dry_run_update(
            env, policy, lag, opt, steps=4, seed=0, corrector_on=False, generator=gen
        )

    clean, mutated = run(False), run(True)
    assert clean["actor_loss"] == pytest.approx(mutated["actor_loss"])
    assert clean["effective_advantage_terms"] == mutated["effective_advantage_terms"]
    assert clean["multipliers_post_update"] == mutated["multipliers_post_update"]


# --- 5. 静态与诊断标签 ---

def test_train_module_has_no_ratio_clip_or_entropy():
    tree = ast.parse(pathlib.Path(train_mod.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    literals = {
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    for token in ("ratio", "clip", "entropy", "entropy_coef"):
        assert token not in names | attrs, f"train.py 不得实现 {token}"
        assert token not in literals, f"train.py 不得出现 {token}"


def test_metrics_are_labelled_dry_run_diagnostics(monkeypatch):
    _, policy, _, lag, opt = setup(monkeypatch)
    result = run_dry(policy, lag, opt)

    assert result["claims"] == {
        "trained": False,
        "performance_evaluated": False,
        "convergence_claimed": False,
    }
    for key in (
        "multipliers_pre_update",
        "multipliers_post_update",
        "constraint_means",
        "constraint_budgets",
        "constraint_units",
        "effective_advantage",
        "effective_advantage_terms",
        "actor_objective_source",
    ):
        assert key in result

    forbidden = ("converg", "performance", "improve", "gain", "accuracy", "return_mean")
    offenders = [
        key
        for key in result
        if key != "claims" and any(s in key.lower() for s in forbidden)
    ]
    assert offenders == []


# --- 6. 回归：乘子必须真正进入损失，且跨轮连续 ---

def test_multipliers_actually_change_the_actor_loss(monkeypatch):
    """不能只改上报值：换一组乘子，actor_loss 本身必须改变。"""
    losses, terms = [], []
    for carbon_seed in (13.0, 30.0):
        transitions = [make_transition(i) for i in range(len(REWARDS))]
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, transitions)
            policy = make_policy()
            lag = Lagrangian(make_specs())
            lag.update({"business": [SEED_BUSINESS], "carbon": [carbon_seed]})
            opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
            result = run_dry(policy, lag, opt)
        losses.append(result["actor_loss"])
        terms.append(result["effective_advantage_terms"])

    assert losses[0] != pytest.approx(losses[1]), "乘子必须真正进入 actor loss"
    assert terms[0]["reward_mean"] == pytest.approx(terms[1]["reward_mean"])
    assert terms[0]["business_mean"] == pytest.approx(terms[1]["business_mean"])
    assert terms[0]["carbon_mean"] != pytest.approx(terms[1]["carbon_mean"])


def test_zero_multipliers_reproduce_the_reward_only_objective(monkeypatch):
    """λ 全为 0 时 actor_loss 必须精确等于只用 reward 优势的旧目标。"""
    transitions = [make_transition(i) for i in range(len(REWARDS))]
    install_fake_collector(monkeypatch, transitions)
    policy = make_policy()
    snapshot = copy.deepcopy(policy)
    lag = Lagrangian(make_specs())
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_dry(policy, lag, opt)
    ref = independent_heads(snapshot, transitions)
    expected = -float(np.mean(ref["advantages"]["reward"] * ref["log_probs"]))
    assert result["actor_loss"] == pytest.approx(expected)


def test_multiplier_chain_is_continuous_across_rounds(monkeypatch):
    """连续三轮：第 i+1 轮的 pre-update 乘子必须等于第 i 轮的 post-update。"""
    transitions = [make_transition(i) for i in range(len(REWARDS))]
    install_fake_collector(monkeypatch, transitions)
    lag = Lagrangian(make_specs())

    pre, post = [], []
    for _ in range(3):
        policy = make_policy()  # 每轮全新同权重策略，避免上一轮 step 的干扰
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        result = run_dry(policy, lag, opt)
        pre.append(result["multipliers_pre_update"])
        post.append(result["multipliers_post_update"])

    assert pre[0] == {"business": 0.0, "carbon": 0.0}
    for index in range(1, 3):
        assert pre[index] == post[index - 1], "乘子链必须连续"
    # business mean=2.0 < budget=5.0 -> 恒被截断为 0；carbon mean=5.0 > budget=3.0 -> 单调上升
    assert all(state["business"] == 0.0 for state in post)
    assert post[0]["carbon"] < post[1]["carbon"] < post[2]["carbon"]


def test_constraint_terms_scale_linearly_with_their_multiplier(monkeypatch):
    """约束项必须与其乘子成正比：λ 翻倍则该项均值翻倍。"""
    def term_for(carbon_seed: float) -> float:
        transitions = [make_transition(i) for i in range(len(REWARDS))]
        with monkeypatch.context() as ctx:
            install_fake_collector(ctx, transitions)
            policy = make_policy()
            lag = Lagrangian(make_specs())
            lag.update({"business": [5.0], "carbon": [carbon_seed]})  # λ_b = 0
            opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
            return run_dry(policy, lag, opt)["effective_advantage_terms"]["carbon_mean"]

    one = term_for(13.0)   # λ_c = 2.0
    two = term_for(23.0)   # λ_c = 4.0
    assert one != pytest.approx(0.0)
    assert two == pytest.approx(2.0 * one)
