"""M5.3f 测试：训练轮次的约束信号预检与失败原子性。

预检必须在**任何** forward / backward / optimizer.step **之前**完成；
失败时 policy、optimizer、Lagrangian 三者都不得被改动。
"""

import ast
import copy
import pathlib
from typing import Any

import numpy as np
import pytest
import torch

from safe_rl_v2 import lagrangian as lag_mod
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

SPECS = (
    ConstraintSpec(
        name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
        learning_rate=0.1, max_multiplier=100.0,
    ),
    ConstraintSpec(
        name="carbon", budget=3.0, unit=UNIT_KG_CO2E,
        learning_rate=0.2, max_multiplier=100.0,
    ),
)

GOOD_BUSINESS = [1.0, 3.0]
GOOD_CARBON = [2.0, 4.0]


def make_policy(seed: int = 0) -> SafePPOPolicy:
    torch.manual_seed(seed)
    return SafePPOPolicy(obs_dim=OBS_DIM)


def make_transition(index: int, business: float, carbon: float) -> Transition:
    return Transition(
        observation=np.full(OBS_DIM, 0.1 * (index + 1), dtype=np.float32),
        next_observation=np.full(OBS_DIM, 0.2 * (index + 1), dtype=np.float32),
        raw_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
        old_raw_log_prob=-1.0,
        exec_action=np.full(ACTION_DIM, 0.5, dtype=np.float32),
        reward=1.0,
        business_cost=business,
        carbon_cost=carbon,
        electricity_cost_sgd=3.0,
        terminated=False,
        truncated=False,
        correction_info={},
    )


def install_fake_collector(
    monkeypatch, business=GOOD_BUSINESS, carbon=GOOD_CARBON, *, empty: bool = False
) -> None:
    transitions = [] if empty else [
        make_transition(i, b, c) for i, (b, c) in enumerate(zip(business, carbon, strict=True))
    ]

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


def snapshot(policy, lagrangian, optimizer) -> dict:
    return {
        "params": [p.detach().clone() for p in policy.parameters()],
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lagrangian": copy.deepcopy(lagrangian.state_dict()),
    }


def assert_untouched(policy, lagrangian, optimizer, before: dict) -> None:
    for old, new in zip(before["params"], policy.parameters(), strict=True):
        assert torch.equal(old, new.detach()), "policy 参数被改动"
    assert optimizer.state_dict() == before["optimizer"], "optimizer state 被改动"
    assert lagrangian.state_dict() == before["lagrangian"], "Lagrangian state 被改动"


def run(monkeypatch, *, business=GOOD_BUSINESS, carbon=GOOD_CARBON, empty=False):
    install_fake_collector(monkeypatch, business, carbon, empty=empty)
    policy = make_policy()
    lagrangian = Lagrangian(SPECS)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    return policy, lagrangian, optimizer


# --- 1. 预检必须在任何 forward 之前 ----------------------------------------

@pytest.mark.parametrize(
    "business,carbon,expected_name",
    [
        ([-3.0, 1.0], GOOD_CARBON, "business"),
        (GOOD_BUSINESS, [1.0, -2.0], "carbon"),
        ([float("nan"), 1.0], GOOD_CARBON, "business"),
        (GOOD_BUSINESS, [float("inf"), 1.0], "carbon"),
    ],
)
def test_preflight_fails_before_any_forward(monkeypatch, business, carbon, expected_name):
    policy, lagrangian, optimizer = run(monkeypatch, business=business, carbon=carbon)
    forwards: list[int] = []
    real_forward = SafePPOPolicy.forward

    def spy(self, obs):
        forwards.append(1)
        return real_forward(self, obs)

    monkeypatch.setattr(SafePPOPolicy, "forward", spy)

    with pytest.raises(ValueError) as excinfo:
        dry_run_update(None, policy, lagrangian, optimizer, steps=len(business), seed=0,
                       corrector_on=False, generator=None)

    assert forwards == [], "预检失败时不得发生任何 critic/actor 前向"
    assert expected_name in str(excinfo.value), "异常必须指出具体约束"


def test_empty_buffer_fails_before_any_forward(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch, empty=True)
    forwards: list[int] = []
    real_forward = SafePPOPolicy.forward

    def spy(self, obs):
        forwards.append(1)
        return real_forward(self, obs)

    monkeypatch.setattr(SafePPOPolicy, "forward", spy)
    with pytest.raises(RuntimeError):
        dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                       corrector_on=False, generator=None)
    assert forwards == []


# --- 2. 预检失败时三态皆不变 ------------------------------------------------

@pytest.mark.parametrize(
    "business,carbon",
    [
        ([-3.0, 1.0], GOOD_CARBON),
        (GOOD_BUSINESS, [1.0, -2.0]),
        ([-1.0, -1.0], GOOD_CARBON),
        ([float("nan"), 1.0], GOOD_CARBON),
        (GOOD_BUSINESS, [float("-inf"), 1.0]),
    ],
)
def test_preflight_failure_leaves_policy_optimizer_and_lagrangian_untouched(
    monkeypatch, business, carbon
):
    policy, lagrangian, optimizer = run(monkeypatch, business=business, carbon=carbon)
    before = snapshot(policy, lagrangian, optimizer)

    with pytest.raises(ValueError):
        dry_run_update(None, policy, lagrangian, optimizer, steps=len(business), seed=0,
                       corrector_on=False, generator=None)

    assert_untouched(policy, lagrangian, optimizer, before)


def test_optimizer_step_is_never_called_on_preflight_failure(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch, business=[-3.0, 1.0])
    calls: list[str] = []
    real_step = optimizer.step

    def spy(*args, **kwargs):
        calls.append("step")
        return real_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", spy)
    with pytest.raises(ValueError):
        dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                       corrector_on=False, generator=None)
    assert calls == []


def test_backward_is_never_called_on_preflight_failure(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch, carbon=[2.0, -4.0])
    calls: list[str] = []
    real_backward = torch.Tensor.backward

    def spy(self, *args, **kwargs):
        calls.append("backward")
        return real_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", spy)
    with pytest.raises(ValueError):
        dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                       corrector_on=False, generator=None)
    assert calls == []


# --- 3. 正常路径的顺序 ------------------------------------------------------

def test_normal_path_order_is_preflight_backward_step_update(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch)
    order: list[str] = []

    real_validate = lag_mod.validate_constraint_signals
    real_backward = torch.Tensor.backward
    real_step = optimizer.step
    real_update = lagrangian.update

    def validate_spy(*args, **kwargs):
        order.append("preflight")
        return real_validate(*args, **kwargs)

    def backward_spy(self, *args, **kwargs):
        order.append("backward")
        return real_backward(self, *args, **kwargs)

    def step_spy(*args, **kwargs):
        order.append("optimizer.step")
        return real_step(*args, **kwargs)

    def update_spy(*args, **kwargs):
        order.append("lagrangian.update")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(train_mod, "validate_constraint_signals", validate_spy)
    monkeypatch.setattr(torch.Tensor, "backward", backward_spy)
    monkeypatch.setattr(optimizer, "step", step_spy)
    monkeypatch.setattr(lagrangian, "update", update_spy)

    dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                   corrector_on=False, generator=None)

    assert order == ["preflight", "backward", "optimizer.step", "lagrangian.update"], order


# --- 4. 共用同一份校验规则 --------------------------------------------------

def test_train_calls_the_shared_validator_with_buffer_signals(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch)
    recorded: list[dict[str, Any]] = []
    real_validate = lag_mod.validate_constraint_signals

    def spy(batch_signals, expected):
        recorded.append({"batch_signals": batch_signals, "expected": expected})
        return real_validate(batch_signals, expected)

    monkeypatch.setattr(train_mod, "validate_constraint_signals", spy)
    dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                   corrector_on=False, generator=None)

    assert len(recorded) == 1, "必须恰好预检一次"
    batch = recorded[0]["batch_signals"]
    assert set(batch) == {"business", "carbon"}
    np.testing.assert_array_equal(np.asarray(batch["business"], dtype=float), GOOD_BUSINESS)
    np.testing.assert_array_equal(np.asarray(batch["carbon"], dtype=float), GOOD_CARBON)
    assert set(recorded[0]["expected"]) == {"business", "carbon"}


def test_train_module_imports_the_shared_validator():
    """不得在 train.py 里另写一套校验：必须导入 lagrangian 的共享函数。"""
    tree = ast.parse(pathlib.Path(train_mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert "validate_constraint_signals" in imported


def test_train_module_has_no_local_negative_check():
    """train.py 不得出现自己写的「< 0」约束校验（那会绕过共享规则）。"""
    tree = ast.parse(pathlib.Path(train_mod.__file__).read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(isinstance(op, ast.Lt) for op in node.ops)
        and any(
            isinstance(side, ast.Constant) and side.value == 0 for side in node.comparators
        )
    ]
    assert offenders == [], f"train.py 出现本地负值校验（行 {offenders}）"


# --- 5. 合法路径不受影响 ----------------------------------------------------

def test_legal_signals_still_train_normally(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch)
    before = snapshot(policy, lagrangian, optimizer)

    result = dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                            corrector_on=False, generator=None)

    assert result["steps_collected"] == 2
    assert lagrangian.state_dict() != before["lagrangian"], "合法路径必须更新乘子"
    assert result["multipliers_post_update"] == lagrangian.multipliers()
    assert result["claims"]["trained"] is False


def test_zero_signals_are_legal_and_train_normally(monkeypatch):
    policy, lagrangian, optimizer = run(monkeypatch, business=[0.0, 0.0], carbon=[0.0, 0.0])
    result = dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                            corrector_on=False, generator=None)
    assert result["constraint_means"] == {"business": 0.0, "carbon": 0.0}
    assert lagrangian.multipliers() == {"business": 0.0, "carbon": 0.0}


# --- 6. 顶部过时表述必须被修正 ----------------------------------------------

def test_train_docstring_no_longer_calls_multipliers_a_placeholder():
    source = pathlib.Path(train_mod.__file__).read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    assert "拉格朗日乘子仍是 M5.4 的占位实现" not in docstring
    assert "M5.3 待重建" not in docstring
    # 必须记录 M5.3 已完成乘子目标接线
    assert "M5.3" in docstring
    assert "lambda_business" in docstring or "λ" in docstring


def test_train_docstring_records_what_m54_still_owes():
    source = pathlib.Path(train_mod.__file__).read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source)) or ""
    for token in ("M5.4", "入口", "泄漏", "corrector"):
        assert token in docstring, f"顶部说明缺少 M5.4 未完成项：{token}"


# --- 7. 回归：随机批下「要么成功、要么完全不改」 ----------------------------

def _random_signal(rng, size: int) -> list[float]:
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return rng.uniform(0.0, 20.0, size=size).tolist()
    if kind == 1:
        return rng.uniform(-20.0, 0.0, size=size).tolist()
    if kind == 2:
        return rng.uniform(-20.0, 20.0, size=size).tolist()
    if kind == 3:
        return [0.0] * size
    values = rng.uniform(0.0, 20.0, size=size)
    values[0] = float("nan")
    return values.tolist()


def _signal_ok(values) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(arr.size > 0 and np.all(np.isfinite(arr)) and np.all(arr >= 0.0))


def test_dry_run_is_all_or_nothing_under_random_signals(monkeypatch):
    """随机批：dry_run_update 要么成功并更新乘子，要么抛错且三态逐位不变。"""
    rng = np.random.default_rng(20240914)
    succeeded = failed = 0
    for _ in range(60):
        size = int(rng.integers(1, 5))
        business = _random_signal(rng, size)
        carbon = _random_signal(rng, size)
        expected_ok = _signal_ok(business) and _signal_ok(carbon)

        policy, lagrangian, optimizer = run(monkeypatch, business=business, carbon=carbon)
        before = snapshot(policy, lagrangian, optimizer)

        try:
            result = dry_run_update(
                None, policy, lagrangian, optimizer,
                steps=len(business), seed=0, corrector_on=False, generator=None,
            )
        except (ValueError, TypeError) as exc:
            failed += 1
            assert not expected_ok, f"合法信号被拒绝：{business} / {carbon}：{exc}"
            assert_untouched(policy, lagrangian, optimizer, before)
        else:
            succeeded += 1
            assert expected_ok, f"非法信号被接受：{business} / {carbon}"
            assert result["constraint_means"]["business"] == pytest.approx(
                float(np.mean(business))
            )
            assert result["constraint_means"]["carbon"] == pytest.approx(float(np.mean(carbon)))
            assert lagrangian.state_dict() != before["lagrangian"]
    assert succeeded > 0 and failed > 0, f"样本必须同时覆盖成功与失败：{succeeded}/{failed}"


def test_preflight_also_guards_the_corrector_path(monkeypatch):
    """corrector_on=True 时同样必须预检（该分支不得绕过检查）。"""
    policy, lagrangian, optimizer = run(
        monkeypatch, business=[-3.0, 1.0], carbon=GOOD_CARBON
    )
    before = snapshot(policy, lagrangian, optimizer)
    with pytest.raises(ValueError, match="business"):
        dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                       corrector_on=True, corrector_time_limit_s=0.05, generator=None)
    assert_untouched(policy, lagrangian, optimizer, before)


def test_preflight_runs_exactly_once_regardless_of_buffer_size(monkeypatch):
    calls: list[int] = []
    for size in (1, 2, 5):
        business = [1.0] * size
        carbon = [2.0] * size
        policy, lagrangian, optimizer = run(monkeypatch, business=business, carbon=carbon)
        real_validate = lag_mod.validate_constraint_signals

        def spy(batch_signals, expected, _real=real_validate):
            calls.append(len(np.asarray(batch_signals["business"])))
            return _real(batch_signals, expected)

        with monkeypatch.context() as ctx:
            ctx.setattr(train_mod, "validate_constraint_signals", spy)
            dry_run_update(None, policy, lagrangian, optimizer, steps=size, seed=0,
                           corrector_on=False, generator=None)
    assert calls == [1, 2, 5]


def test_preflight_failure_message_names_constraint_and_reason(monkeypatch):
    """预检的报错必须指出具体约束与失败原因（供定位上游缺陷）。"""
    for business, carbon, expected in (
        ([1.0, -2.0], GOOD_CARBON, "business"),
        (GOOD_BUSINESS, [1.0, -2.0], "carbon"),
    ):
        policy, lagrangian, optimizer = run(monkeypatch, business=business, carbon=carbon)
        before = snapshot(policy, lagrangian, optimizer)
        with pytest.raises(ValueError) as excinfo:
            dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                           corrector_on=False, generator=None)
        message = str(excinfo.value)
        assert expected in message, message
        assert "batch_signals" in message, message
        assert_untouched(policy, lagrangian, optimizer, before)


def test_non_finite_signals_are_rejected_even_earlier_by_the_buffer(monkeypatch):
    """非有限信号由 `RolloutBuffer.add` 更早拦下，根本到不了预检。

    这是两层防御：buffer 保证入库数据有限，预检保证约束物理域。
    两者都必须在任何 forward / optimizer.step 之前完成。
    """
    policy, lagrangian, optimizer = run(monkeypatch, carbon=[2.0, float("nan")])
    before = snapshot(policy, lagrangian, optimizer)
    with pytest.raises((ValueError, TypeError), match="finite|非有限"):
        dry_run_update(None, policy, lagrangian, optimizer, steps=2, seed=0,
                       corrector_on=False, generator=None)
    assert_untouched(policy, lagrangian, optimizer, before)
