"""M5.4b 测试：真实环境泄漏门禁与 corrector 开/关的跨进程可复现性。

**已登记的 blocker**：corrector **开启**时（任何「求解器真的在跑」的预算）跨进程
逐位可复现**不成立** —— 时限内求解路径不同会得到**代价相同但元素不同**的最优解。
详见 `CORRECTOR_ON_REPRO_BLOCKER`。本文件**不**为 corrector 开启断言稳定，
也不重试挑选结果。
"""

import ast
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2.buffer import RolloutBuffer
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
STEPS = 4
NORMAL_BUDGET_S = 0.05
GENEROUS_BUDGET_S = 2.0
ALWAYS_TIMEOUT_BUDGET_S = 1.0e-6

# 未来真值数组（均在 __init__ 中生成，reset 不重建）
TRUTH_ATTRS = ("price_t", "T_amb", "lambda_t", "pv_t", "wt_t", "carbon_factor_t")

# 纯墙钟字段：**不得**纳入跨进程逐字节比较
WALL_CLOCK_KEYS = (
    "correction_solve_time_s",
    "stage_a_solve_time_s",
    "stage_b_solve_time_s",
)

# ---------------------------------------------------------------------------
# 已登记的 blocker（M5.4b 实测）
# ---------------------------------------------------------------------------
CORRECTOR_ON_REPRO_BLOCKER = {
    "id": "M5.4b-corrector-on-not-cross-process-reproducible",
    "scope": "corrector 开启且求解器真的运行（即任何非「恒超时」的预算）",
    "summary": (
        "corrector 是有时限的 MILP。时限内走到的最优解取决于求解路径，"
        "而**多个最优解的元素可以不同**（同代价、均已通过物理校验）。"
        "因此 corrector 开启时 exec 决策随进程/负载变化，跨进程逐位可复现不成立；"
        "`correction_reason` 仍为 `none`（即求解器确实收敛），不是超时退化。"
    ),
    "measured": {
        "budget_0.05_6cpu_hogs_4_procs": {
            "digests": ["8468db982fa6", "842b06d3bf97"], "distinct": 2,
        },
        "budget_0.05_no_load_2_procs": {
            "digests": ["8468db982fa6", "28258c1a5eba"], "distinct": 2,
        },
        "budget_0.020_6cpu_hogs_4_procs": {"distinct": 3},
        "budget_0.010_6cpu_hogs_4_procs": {"distinct": 2},
        "budget_0.005_6cpu_hogs_4_procs": {"distinct": 2},
        "budget_0.003_6cpu_hogs_4_procs": {
            "distinct": 1, "note": "恒超时 → 恒返回边界动作，反而确定",
        },
        "corrector_off_10_procs_mixed_load": {
            "distinct": 1, "note": "不受负载影响",
        },
    },
    "measurement": "同一代码/seed，比较**排除**三个墙钟字段后的语义 sha256",
    "consequence": (
        "本文件只对 corrector **关闭**断言跨进程逐位一致；"
        "corrector 开启不做任何相等断言，也不重试挑选取样。"
    ),
    "fix_owner": (
        "需要在 planning/ 中固定求解路径或改用确定性求解配置"
        "（超出 M5.4b 允许范围，属后续工作）"
    ),
}


def make_env(**over) -> IDCPriceEnv20D:
    kwargs = dict(ENV_SEED_KWARGS)
    kwargs.update(over)
    return IDCPriceEnv20D(**kwargs)


def make_policy(env, seed: int = 0) -> SafePPOPolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SafePPOPolicy(obs_dim=env.obs_dim)


def make_generator(seed: int = 0) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def collect(env, *, corrector_on: bool = False, budget: float | None = None) -> RolloutBuffer:
    policy = make_policy(env)
    buffer = RolloutBuffer()
    collect_rollout(
        env, policy, buffer, steps=STEPS, seed=0,
        corrector_on=corrector_on,
        corrector_time_limit_s=budget if corrector_on else None,
        generator=make_generator(0),
    )
    return buffer


def mutate_future(env, value: float = 9999.0, *, start: int) -> None:
    """把六类未来真值从下标 `start` 起逐项突变。

    可见窗口是**滑动**的：第 t 步可见 `[t, t + forecast_cutoff)`。
    因此对长度为 `steps` 的 rollout，安全区是
    `[forecast_cutoff + steps, horizon)`；只从 `forecast_cutoff` 开始突变
    只能保护**第 0 步**。
    """
    for attr in TRUTH_ATTRS:
        arr = getattr(env, attr).copy()
        arr[start:] = value
        setattr(env, attr, arr)


# --- 1/2. 未来真值突变不得影响本 rollout 可见的量 ---------------------------

@pytest.mark.leakage
def test_future_truth_beyond_the_executed_window_does_not_leak():
    """突变整个「本 rollout 绝不会看到」的区域，observation/raw/log-prob 必须逐位不变。"""
    clean = make_env()
    mutated = make_env()
    # 滑动窗口在第 t 步可见 [t, t+cutoff)，故 steps 步内最大可见下标为 cutoff+steps-1
    mutate_future(mutated, start=clean.forecast_cutoff + STEPS)

    obs_clean = np.asarray(clean.reset(seed=0)[0], dtype=np.float32).copy()
    obs_mutated = np.asarray(mutated.reset(seed=0)[0], dtype=np.float32).copy()
    assert obs_clean.shape == obs_mutated.shape
    np.testing.assert_array_equal(obs_clean, obs_mutated, err_msg="未来真值泄漏进 observation")

    buf_clean = collect(clean)
    buf_mutated = collect(mutated)

    for a, b in zip(buf_clean.transitions, buf_mutated.transitions, strict=True):
        np.testing.assert_array_equal(a.observation, b.observation)
        np.testing.assert_array_equal(a.raw_action, b.raw_action)
        assert a.old_raw_log_prob == b.old_raw_log_prob, "log-prob 受未来真值影响"
        assert a.reward == b.reward


def test_visible_window_mutation_does_change_observation():
    """反例：改动可见窗口**之内**的值，observation 必须变化（证明门禁非恒真）。"""
    clean = make_env()
    mutated = make_env()
    mutate_future(mutated, start=clean.forecast_cutoff - 1)  # 3 < 4，落在可见窗口内

    obs_clean = np.asarray(clean.reset(seed=0)[0], dtype=np.float32)
    obs_mutated = np.asarray(mutated.reset(seed=0)[0], dtype=np.float32)
    assert not np.array_equal(obs_clean, obs_mutated), "可见窗口内的改动竟然没有影响 observation"

    buf_clean = collect(clean)
    buf_mutated = collect(mutated)
    assert not np.array_equal(
        buf_clean.transitions[0].raw_action, buf_mutated.transitions[0].raw_action
    )


def test_boundary_is_exactly_the_forecast_cutoff():
    """固定边界语义：从 cutoff 起突变不改 observation，从 cutoff-1 起必须改。"""
    cutoff = make_env().forecast_cutoff
    base = np.asarray(make_env().reset(seed=0)[0], dtype=np.float32)

    at_cutoff = make_env()
    mutate_future(at_cutoff, start=cutoff)
    np.testing.assert_array_equal(
        base, np.asarray(at_cutoff.reset(seed=0)[0], dtype=np.float32)
    )

    inside = make_env()
    mutate_future(inside, start=cutoff - 1)
    assert not np.array_equal(base, np.asarray(inside.reset(seed=0)[0], dtype=np.float32))


# --- 4/5. corrector 关闭 / 开启的语义 ---------------------------------------

def test_corrector_off_exec_equals_raw_and_reports_no_solver_success():
    buffer = collect(make_env(), corrector_on=False)
    for t in buffer.transitions:
        np.testing.assert_array_equal(t.exec_action, t.raw_action, err_msg="exec 必须等于 raw")
        assert t.correction_info == {"corrector_on": False}
        for key in ("correction_reason", "planner_backend", "stage_a_status", "exec_action"):
            assert key not in t.correction_info, f"关闭时不得伪造 solver 信息：{key}"


def test_corrector_on_preserves_raw_action_and_logprob():
    buffer = collect(make_env(), corrector_on=True, budget=NORMAL_BUDGET_S)
    for t in buffer.transitions:
        # raw 必须被原样保留（由 wrapper 回写的 info["raw_action"] 佐证）
        received = np.asarray(t.correction_info["raw_action"], dtype=np.float32)
        np.testing.assert_array_equal(received, t.raw_action.astype(np.float32))
        assert np.isfinite(t.old_raw_log_prob)
        assert "correction_reason" in t.correction_info
        assert "planner_backend" in t.correction_info


def test_corrector_on_exec_action_comes_only_from_corrector():
    buffer = collect(make_env(), corrector_on=True, budget=NORMAL_BUDGET_S)
    for t in buffer.transitions:
        from_corrector = np.asarray(t.correction_info["exec_action"], dtype=np.float32)
        np.testing.assert_array_equal(t.exec_action.astype(np.float32), from_corrector)
    assert any(
        not np.array_equal(t.exec_action, t.raw_action) for t in buffer.transitions
    ), "正常预算下修正器应当至少改动一步"


def test_always_timeout_budget_returns_the_verified_boundary_action():
    """恒超时预算是确定的：返回已验证边界动作（零计算、零储能），不返回未检验的 raw。"""
    buffer = collect(make_env(), corrector_on=True, budget=ALWAYS_TIMEOUT_BUDGET_S)
    for t in buffer.transitions:
        assert t.correction_info["correction_reason"] == "timeout"
        np.testing.assert_allclose(t.exec_action, 0.0, atol=0.0)
        assert not np.array_equal(t.raw_action, t.exec_action)
        received = np.asarray(t.correction_info["raw_action"], dtype=np.float32)
        np.testing.assert_array_equal(received, t.raw_action.astype(np.float32))


def test_corrector_budget_is_semantically_load_bearing():
    """blocker 的机制证据：预算改变 ⇒ exec 决策改变（同一 seed/权重）。

    若宽裕预算在当前负载下仍未收敛，则本测量对预算语义**不构成证据**，
    此时 skip 而不是断言 —— 不得靠重试挑选结果。
    """
    tight = collect(make_env(), corrector_on=True, budget=ALWAYS_TIMEOUT_BUDGET_S)
    generous = collect(make_env(), corrector_on=True, budget=GENEROUS_BUDGET_S)

    assert all(
        t.correction_info["correction_reason"] == "timeout" for t in tight.transitions
    )
    if all(t.correction_info["correction_reason"] == "timeout" for t in generous.transitions):
        pytest.skip(
            "宽裕预算在当前负载下仍未收敛，本测量对「预算有语义影响」不构成证据"
            "（见 CORRECTOR_ON_REPRO_BLOCKER）"
        )
    assert not np.array_equal(
        tight.transitions[0].exec_action, generous.transitions[0].exec_action
    )


# --- 跨进程语义摘要 ---------------------------------------------------------

_PROBE_SCRIPT = r'''
import hashlib, json, sys
import torch
from envs.idc_price_env import IDCPriceEnv20D
from safe_rl_v2.buffer import RolloutBuffer
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

WALL = {"correction_solve_time_s", "stage_a_solve_time_s", "stage_b_solve_time_s"}
mode = sys.argv[1]
budget = {"off": None, "normal": 0.05}[mode]
env = IDCPriceEnv20D(task_seed=0, server_seed=0, forecast_seed=300000)
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(0)
    policy = SafePPOPolicy(obs_dim=env.obs_dim)
gen = torch.Generator(); gen.manual_seed(0)
buf = RolloutBuffer()
collect_rollout(env, policy, buf, steps=4, seed=0, corrector_on=(mode != "off"),
                corrector_time_limit_s=budget, generator=gen)
digest = [
    {
        "obs": [float(x) for x in t.observation],
        "next_obs": [float(x) for x in t.next_observation],
        "raw": [float(x) for x in t.raw_action],
        "exec": [float(x) for x in t.exec_action],
        "logp": t.old_raw_log_prob,
        "reward": t.reward,
        "business": t.business_cost,
        "carbon": t.carbon_cost,
        "electricity": t.electricity_cost_sgd,
        "terminated": bool(t.terminated),
        "truncated": bool(t.truncated),
        "info": {k: v for k, v in t.correction_info.items() if k not in WALL},
    }
    for t in buf.transitions
]
blob = json.dumps(digest, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(blob).hexdigest())
'''


def _probe(mode: str) -> str:
    """在**独立进程**中跑一次采集，返回语义摘要（已排除墙钟字段）。"""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, mode],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, f"{mode} 探针失败：{result.stderr}"
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_cross_process_off_is_bit_identical():
    """corrector 关闭：跨进程语义摘要必须逐位一致（10/10 实测，含 6 路 CPU 争用）。"""
    digests = {_probe("off") for _ in range(3)}
    assert len(digests) == 1, f"corrector 关闭时跨进程摘要不一致：{digests}"


@pytest.mark.slow
def test_cross_process_corrector_on_is_not_claimed_reproducible():
    """corrector 开启：**不**断言跨进程稳定 —— 该 blocker 已登记。

    本测试不比较 corrector 开启的摘要是否相等（那会随负载 flaky），
    只断言：blocker 已登记、记录了实测不一致、且本文件只对
    corrector **关闭**保留相等断言。
    """
    blocked = CORRECTOR_ON_REPRO_BLOCKER
    assert blocked["id"].startswith("M5.4b-")
    assert "corrector 开启" in blocked["scope"]

    measured = blocked["measured"]
    assert measured["budget_0.05_6cpu_hogs_4_procs"]["distinct"] > 1, "必须记录实测不一致"
    assert measured["budget_0.05_no_load_2_procs"]["distinct"] > 1
    assert measured["corrector_off_10_procs_mixed_load"]["distinct"] == 1

    tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
    cross_process = sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_cross_process_")
    )
    assert cross_process == [
        "test_cross_process_corrector_on_is_not_claimed_reproducible",
        "test_cross_process_off_is_bit_identical",
    ], f"只应对 corrector 关闭断言相等，实际 {cross_process}"


def test_wall_clock_fields_are_excluded_from_the_digest():
    """跨进程比较必须排除纯耗时字段，否则会把墙钟读数当成语义差异。"""
    assert (
        'WALL = {"correction_solve_time_s", "stage_a_solve_time_s", "stage_b_solve_time_s"}'
        in _PROBE_SCRIPT
    )
    assert "if k not in WALL" in _PROBE_SCRIPT
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    for key in WALL_CLOCK_KEYS:
        assert key in source
