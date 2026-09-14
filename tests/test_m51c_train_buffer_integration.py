"""M5.1c 测试：dry_run_update 真实消费 contract-v7 RolloutBuffer，采样 RNG 显式可注入。

本卡不测 PPO 数学，也不声称训练有效：dry-run 仍只是「闭环可更新」的冒烟验证。
"""

import ast
import json
import pathlib
from typing import Any

import numpy as np
import pytest
import torch

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2 import train as train_mod
from safe_rl_v2.buffer import ACTION_DIM, CONTRACT_VERSION, RolloutBuffer
from safe_rl_v2.lagrangian import (
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout
from safe_rl_v2.train import dry_run_update

# 环境三类种子必须同时显式给定（既有 env 在任一为 None 时用 default_rng(None) 熵源）
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
CORRECTOR_TIME_LIMIT_S = 0.05
STEPS = 4



def make_default_lagrangian() -> Lagrangian:
    """M5.3a 迁移夹具：显式约束定义（替换 `Lagrangian({"business":5.0,"carbon":3.0})`）。"""
    return Lagrangian((
        ConstraintSpec(
            name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
            learning_rate=0.01, max_multiplier=100.0,
        ),
        ConstraintSpec(
            name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
            learning_rate=0.01, max_multiplier=100.0,
        ),
    ))

def make_env(**over):
    kwargs = dict(ENV_SEED_KWARGS)
    kwargs.update(over)
    return IDCPriceEnv20D(**kwargs)


def make_generator(seed: int) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def make_policy(env, seed: int = 0) -> SafePPOPolicy:
    torch.manual_seed(seed)  # 仅测试夹具：保证两次构造权重逐元素相同
    return SafePPOPolicy(obs_dim=env.obs_dim)


def make_lagrangian() -> Lagrangian:
    return make_default_lagrangian()


def make_optimizer(policy) -> torch.optim.Optimizer:
    return torch.optim.Adam(policy.parameters(), lr=1e-3)


def make_setup(**over):
    env = make_env(**over)
    policy = make_policy(env)
    return env, policy, make_lagrangian(), make_optimizer(policy)


# --- 1. dry_run_update 必须经 collect_rollout 并消费真实 buffer ---

def test_dry_run_delegates_sampling_to_collector(monkeypatch):
    env, policy, lag, opt = make_setup()
    gen = make_generator(0)

    recorded: list[dict[str, Any]] = []
    real_collect = train_mod.collect_rollout

    def spy(*args, **kwargs):
        recorded.append({"args": args, "kwargs": kwargs})
        return real_collect(*args, **kwargs)

    monkeypatch.setattr(train_mod, "collect_rollout", spy)

    result = dry_run_update(env, policy, lag, opt, steps=STEPS, seed=0, generator=gen)

    assert len(recorded) == 1, "dry_run_update 必须恰好调用一次 collect_rollout"
    call = recorded[0]
    assert call["kwargs"]["steps"] == STEPS
    assert call["kwargs"]["generator"] is gen
    assert call["kwargs"]["corrector_on"] is False

    # 返回的 buffer 就是 collector 实际写入的那一个实例
    buffer = result["buffer"]
    assert isinstance(buffer, RolloutBuffer)
    assert buffer is call["args"][2]
    assert len(buffer) == result["steps_collected"] == STEPS


def test_dry_run_consumes_buffer_transitions_not_its_own_collection():
    """返回的 buffer 必须带完整 v6 字段，且统计量与其逐条一致。"""
    env, policy, lag, opt = make_setup()
    result = dry_run_update(env, policy, lag, opt, steps=STEPS, seed=0, generator=make_generator(0))
    buffer = result["buffer"]

    assert result["stats"]["contract_version"] == CONTRACT_VERSION
    assert result["stats"]["transitions"] == STEPS
    for t in buffer.transitions:
        assert t.contract_version == CONTRACT_VERSION
        assert t.raw_action.shape == (ACTION_DIM,)
        assert t.next_observation.shape == (env.obs_dim,)
        assert isinstance(t.terminated, bool) and isinstance(t.truncated, bool)

    assert result["business_violation_sum"] == pytest.approx(
        float(np.sum([t.business_cost for t in buffer.transitions]))
    )
    assert result["carbon_emission_sum_kg"] == pytest.approx(
        float(np.sum([t.carbon_cost for t in buffer.transitions]))
    )
    assert result["electricity_cost_sum_sgd"] == pytest.approx(
        float(np.sum([t.electricity_cost_sgd for t in buffer.transitions]))
    )


def test_returned_quantities_are_labelled_with_their_units():
    """业务违规量 / 碳排放量 / 电费必须分别存取并自带单位标签，不得相加或互相顶替。"""
    from safe_rl_v2.buffer import UNIT_METADATA

    env, policy, lag, opt = make_setup()
    result = dry_run_update(env, policy, lag, opt, steps=STEPS, seed=0, generator=make_generator(0))

    assert result["business_violation_sum"] >= 0.0
    assert result["carbon_emission_sum_kg"] > 0.0
    assert result["electricity_cost_sum_sgd"] > 0.0
    units = {
        UNIT_METADATA["business_cost"],
        UNIT_METADATA["carbon_cost"],
        UNIT_METADATA["electricity_cost_sgd"],
    }
    assert len(units) == 3, "三类量纲必须互不相同"


# --- 2. 可导 likelihood 只对 raw_action ---

def test_likelihood_is_evaluated_on_raw_action_not_exec_action(monkeypatch):
    env, policy, lag, opt = make_setup()
    recorded: list[np.ndarray] = []
    real_eval = SafePPOPolicy.evaluate_raw_actions

    def spy(self, obs, raw_action):
        value = raw_action.detach().cpu().numpy() if torch.is_tensor(raw_action) else np.asarray(
            raw_action
        )
        recorded.append(np.asarray(value, dtype=np.float64))
        return real_eval(self, obs, raw_action)

    monkeypatch.setattr(SafePPOPolicy, "evaluate_raw_actions", spy)

    result = dry_run_update(
        env, policy, lag, opt, steps=STEPS, seed=0,
        corrector_on=True, corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S,
        generator=make_generator(0),
    )
    buffer = result["buffer"]
    raw_batch = np.stack([t.raw_action for t in buffer.transitions])
    exec_batch = np.stack([t.exec_action for t in buffer.transitions])

    assert not np.allclose(raw_batch, exec_batch), "corrector 未改变动作，本测试无法区分 raw/exec"

    batched = [r for r in recorded if r.ndim == 2]
    assert batched, "训练损失必须对整批 observation 求 raw likelihood"
    assert any(
        r.shape == raw_batch.shape and np.allclose(r, raw_batch) for r in batched
    ), "可导 likelihood 必须对 buffer 的 raw_action 计算"
    assert not any(
        r.shape == exec_batch.shape and np.allclose(r, exec_batch) for r in batched
    ), "绝不允许对 exec_action 求概率"


def test_old_raw_log_prob_is_reported_as_audit_value_only():
    env, policy, lag, opt = make_setup()
    result = dry_run_update(env, policy, lag, opt, steps=STEPS, seed=0, generator=make_generator(0))
    buffer = result["buffer"]
    expected = float(np.mean([t.old_raw_log_prob for t in buffer.transitions]))
    assert result["old_raw_log_prob_mean"] == pytest.approx(expected)
    assert result["logprob_source"] == "evaluate_raw_actions(raw_action)"


# --- 3. corrector 开/关均真实可运行 ---

@pytest.mark.parametrize("corrector_on", [False, True])
def test_dry_run_runs_with_and_without_corrector(corrector_on):
    env, policy, lag, opt = make_setup()
    result = dry_run_update(
        env, policy, lag, opt, steps=STEPS, seed=0,
        corrector_on=corrector_on,
        corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S if corrector_on else None,
        generator=make_generator(0),
    )
    assert result["corrector_on"] is corrector_on
    assert result["steps_collected"] == STEPS
    if corrector_on:
        assert result["stats"]["raw_exec_difference_count"] > 0
        assert all(t.correction_info["corrector_on"] for t in result["buffer"].transitions)
    else:
        assert result["stats"]["raw_exec_difference_count"] == 0


def test_corrector_on_requires_explicit_time_limit():
    env, policy, lag, opt = make_setup()
    with pytest.raises(ValueError, match="corrector_time_limit_s"):
        dry_run_update(env, policy, lag, opt, steps=1, seed=0, corrector_on=True,
                       generator=make_generator(0))


def test_corrector_off_rejects_unused_time_limit():
    env, policy, lag, opt = make_setup()
    with pytest.raises(ValueError, match="corrector_time_limit_s"):
        dry_run_update(env, policy, lag, opt, steps=1, seed=0, corrector_on=False,
                       corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S,
                       generator=make_generator(0))


# --- 4. 结构性回归：train.py 不得回到旧写法 ---

def _train_ast() -> ast.Module:
    source = pathlib.Path(train_mod.__file__).read_text(encoding="utf-8")
    return ast.parse(source)


def test_train_module_exposes_no_clip_helper():
    assert not hasattr(train_mod, "_clip_action"), "采样动作必须原样执行，不得 clip"


def test_train_source_has_no_info_get_with_default():
    offenders = [
        node.lineno
        for node in ast.walk(_train_ast())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert offenders == [], f"train.py 不得使用 .get() 兜底（行 {offenders}）"


def test_train_source_does_not_reference_raw_env_info_keys():
    """train.py 不得再引用原始 env info 的混单位/兜底键。

    旧实现用 `info.get("business_gap", 0.0) + info.get("cost", 0.0)` 与
    `info.get("carbon_emission", 0.0)`；这些键名在 AST 里是字符串常量，故按常量扫描。
    """
    literals = {
        node.value
        for node in ast.walk(_train_ast())
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for forbidden in ("business_gap", "cost", "carbon_emission", "electricity_cost"):
        assert forbidden not in literals, f"train.py 不得引用原始 info 键 {forbidden!r}"


def test_train_source_does_not_step_env_directly():
    """采样必须经 collect_rollout；`optimizer.step()` 是唯一允许的 .step() 调用。"""
    offenders = []
    for node in ast.walk(_train_ast()):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "step":
            continue
        receiver = node.func.value
        if not (isinstance(receiver, ast.Name) and receiver.id == "optimizer"):
            offenders.append(node.lineno)
    assert offenders == [], f"train.py 不得直接调用 env.step（行 {offenders}）"


def _fork_rng_line_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "fork_rng"
            ):
                ranges.append((node.lineno, max(node.end_lineno or node.lineno, node.lineno)))
    return ranges


def test_train_source_never_reseeds_the_global_torch_rng_without_restoring_it():
    """`torch.manual_seed` 只允许用于**权重初始化**，且必须在 `fork_rng` 内（状态还原）。

    M5.4a 迁移：此前禁止 train.py 出现任何 `manual_seed`；CLI 的权重初始化需要一个
    确定性的起点，故放宽为「必须包在 `torch.random.fork_rng` 里」——
    **采样仍只消耗调用方注入的显式 generator**（见
    `test_collector_with_explicit_generator_leaves_global_rng_untouched`）。
    """
    tree = _train_ast()
    ranges = _fork_rng_line_ranges(tree)
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        # 只针对**全局**的 torch.manual_seed；`generator.manual_seed(...)` 是显式
        # Generator 实例的正规用法，正是红线要求的那种写法。
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and not any(start <= node.lineno <= end for start, end in ranges)
    ]
    assert offenders == [], (
        f"train.py 的 torch.manual_seed 必须包在 torch.random.fork_rng 内（还原全局状态），"
        f"违规行 {offenders}"
    )


# --- 5. RNG 注入语义 ---

def test_collector_with_explicit_generator_leaves_global_rng_untouched():
    env, policy, _, _ = make_setup()
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    collect_rollout(
        env, policy, RolloutBuffer(), steps=STEPS, seed=0,
        corrector_on=False, generator=make_generator(7),
    )
    assert torch.equal(before, torch.get_rng_state()), "显式 generator 不得消耗全局 Torch RNG"


def test_collector_stats_record_rng_provenance():
    env, policy, _, _ = make_setup()
    explicit = collect_rollout(
        env, policy, RolloutBuffer(), steps=1, seed=0,
        corrector_on=False, generator=make_generator(7),
    )
    assert explicit["env_seed"] == 0
    assert explicit["policy_rng_source"] == "explicit_generator"


def test_dry_run_records_env_seed_and_rng_source():
    env, policy, lag, opt = make_setup()
    result = dry_run_update(env, policy, lag, opt, steps=STEPS, seed=0, generator=make_generator(0))
    assert result["env_seed"] == 0
    assert result["policy_rng_source"] == "explicit_generator"
    assert result["stats"]["policy_rng_source"] == "explicit_generator"


def _collect_payload(generator_seed: int) -> dict:
    """同环境种子、同策略权重、同 generator 初始状态 → 完整 payload。"""
    env = make_env()
    policy = make_policy(env, seed=0)
    buffer = RolloutBuffer()
    collect_rollout(
        env, policy, buffer, steps=STEPS, seed=0,
        corrector_on=False, generator=make_generator(generator_seed),
    )
    return buffer.to_dict()


def test_same_generator_state_gives_elementwise_identical_payload():
    first = json.dumps(_collect_payload(11), sort_keys=True)
    second = json.dumps(_collect_payload(11), sort_keys=True)
    assert first == second, "同一 generator 初始状态必须产生逐元素相同的完整 payload"


def test_different_generator_state_gives_different_payload():
    assert json.dumps(_collect_payload(11), sort_keys=True) != json.dumps(
        _collect_payload(12), sort_keys=True
    ), "不同 generator 状态应产生不同采样"


# --- 6. 终端语义 ---

def test_bootstrap_is_masked_at_terminal_step():
    env, policy, lag, opt = make_setup(horizon=3)
    result = dry_run_update(env, policy, lag, opt, steps=50, seed=0, generator=make_generator(0))
    buffer = result["buffer"]
    assert len(buffer) == 3, "episode 在 horizon 处终止，采集必须停止"
    assert buffer.transitions[-1].terminated is True
    assert result["bootstrap_is_terminal"] is True


def test_bootstrap_is_not_masked_before_terminal():
    env, policy, lag, opt = make_setup(horizon=24)
    result = dry_run_update(env, policy, lag, opt, steps=3, seed=0, generator=make_generator(0))
    assert result["bootstrap_is_terminal"] is False


# --- 7. 完整 payload 的跨进程确定性证据（不只是计数）---

def _arm_buffer(corrector_on: bool, steps: int = 3) -> RolloutBuffer:
    env = make_env()
    policy = make_policy(env, seed=0)
    buffer = RolloutBuffer()
    collect_rollout(
        env, policy, buffer, steps=steps, seed=0,
        corrector_on=corrector_on,
        corrector_time_limit_s=CORRECTOR_TIME_LIMIT_S if corrector_on else None,
        generator=make_generator(0),
    )
    return buffer


@pytest.mark.parametrize("corrector_on", [False, True])
def test_full_decision_payload_is_identical_across_independent_runs(corrector_on):
    """完整决策相关 payload 必须逐元素一致——比较整数哈希，而不是只比计数。"""
    from scripts.probe_rollout_deterministic import (
        decision_payload_fingerprint,
        payload_fingerprint,
    )

    first = _arm_buffer(corrector_on)
    second = _arm_buffer(corrector_on)

    assert decision_payload_fingerprint(first) == decision_payload_fingerprint(second)
    if not corrector_on:
        # 无修正器时连墙钟字段都没有，连完整 payload 也必须一致
        assert payload_fingerprint(first) == payload_fingerprint(second)


def test_corrector_arm_differs_only_in_wall_clock_audit_fields():
    """corrector 臂的跨进程差异必须**只**落在墙钟计时审计字段上。"""
    from scripts.probe_rollout_deterministic import WALL_CLOCK_ONLY_KEYS

    first, second = _arm_buffer(True), _arm_buffer(True)
    seen_wall_clock = set()
    for x, y in zip(first.transitions, second.transitions, strict=True):
        np.testing.assert_array_equal(x.raw_action, y.raw_action)
        np.testing.assert_array_equal(x.exec_action, y.exec_action)
        assert x.old_raw_log_prob == y.old_raw_log_prob
        assert set(x.correction_info) == set(y.correction_info)
        for key, value in x.correction_info.items():
            if key in WALL_CLOCK_ONLY_KEYS:
                seen_wall_clock.add(key)
                continue
            assert value == y.correction_info[key], f"决策相关字段 {key} 不应随进程变化"
    assert seen_wall_clock == set(WALL_CLOCK_ONLY_KEYS), "本测试必须覆盖全部墙钟字段"


def test_decision_fingerprint_is_sensitive_to_action_changes():
    """指纹必须有齿：改动一个 raw 动作分量即改变指纹。"""
    from scripts.probe_rollout_deterministic import decision_payload_fingerprint

    baseline = _arm_buffer(False)
    mutated = _arm_buffer(False)
    mutated.transitions[0].raw_action = mutated.transitions[0].raw_action.copy()
    mutated.transitions[0].raw_action[0] += 0.01

    assert decision_payload_fingerprint(baseline) != decision_payload_fingerprint(mutated)
