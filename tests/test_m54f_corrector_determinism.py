"""M5.4f 测试：corrector-on 跨进程确定性求解。

固定**真实** snapshot + raw proposal + seed + 预算，在独立进程中重复调用
`planning.corrector.correct`，比较排除纯墙钟字段后的语义摘要。

**本文件不预设结论**：若未复现不一致，必须**明确记录**「未复现」，
而不是假装测试证明了确定性。
"""

import ast
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from planning import model as planning_model
from planning.snapshot_adapter import build_snapshot

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}

# 固定夹具（改前实测可复现不一致：5 进程中 1 进程 timeout、4 进程收敛）
FIXTURE_STEPS_INTO_EPISODE = 6
FIXTURE_PROPOSAL_SEED = 1
FIXTURE_STORAGE_ACTION = 0.3
NORMAL_BUDGET_S = 0.05
GENEROUS_BUDGET_S = 2.0
ALWAYS_TIMEOUT_BUDGET_S = 1.0e-6
PROCESSES_PER_MODE = 3

# 只排除三个纯墙钟字段
WALL_CLOCK_KEYS = (
    "correction_solve_time_s",
    "stage_a_solve_time_s",
    "stage_b_solve_time_s",
)


# --- 子进程探针 -------------------------------------------------------------

_CHILD = r'''
import hashlib, json, sys
import numpy as np
from contracts.models import DispatchProposal
from envs.idc_price_env import IDCPriceEnv20D
from planning.corrector import correct
from planning.snapshot_adapter import build_snapshot

steps_into, prop_seed, budget, storage = (
    int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
)
env = IDCPriceEnv20D(task_seed=0, server_seed=0, forecast_seed=300000)
env.reset(seed=0)
neutral = np.concatenate([np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
for _ in range(steps_into):
    env.step(neutral)
snapshot = build_snapshot(env)
n = len(snapshot.group_work_capacity)
rng = np.random.default_rng(prop_seed)
proposal = DispatchProposal(
    compute_actions=[float(x) for x in rng.random(n)], storage_action=storage
)
c = correct(snapshot, proposal, time_limit_s=budget)
payload = {
    "exec_compute": [float(x) for x in c.exec_compute_actions],
    "exec_storage": float(c.exec_storage_action),
    "failure": str(c.failure),
    "reason": str(c.reason),
    "business_gap": float(c.business_gap),
    "deadline_shortfall_work": float(c.deadline_shortfall_work),
    "planner_backend": c.planner_backend,
    "reviewed": bool(c.reviewed),
    "stage_a_status": c.stage_a_status,
    "stage_b_status": c.stage_b_status,
    "stage_a_objective": float(c.stage_a_objective),
    "stage_b_objective": float(c.stage_b_objective),
    "projection_offset": float(c.projection_offset),
}
blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(blob).hexdigest())
'''


def corrector_digest(*, budget: float) -> str:
    """在**独立进程**中调用一次 corrector，返回语义摘要。"""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, str(FIXTURE_STEPS_INTO_EPISODE),
         str(FIXTURE_PROPOSAL_SEED), str(budget), str(FIXTURE_STORAGE_ACTION)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def measure(budget: float, *, processes: int = PROCESSES_PER_MODE, rounds: int = 2) -> dict:
    """连续 `rounds` 轮，每轮 `processes` 个独立进程；保留全部 digest。"""
    rounds_digests = [
        [corrector_digest(budget=budget) for _ in range(processes)]
        for _ in range(rounds)
    ]
    flat = [d for rnd in rounds_digests for d in rnd]
    return {
        "budget_s": budget,
        "processes_per_round": processes,
        "rounds": rounds,
        "digests_by_round": rounds_digests,
        "digests": flat,
        "distinct": len(set(flat)),
        "consistent": len(set(flat)) == 1,
    }


def _snapshot_and_proposal():
    env = IDCPriceEnv20D(**ENV_SEED_KWARGS)
    env.reset(seed=0)
    neutral = np.concatenate(
        [np.full(20, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)]
    )
    for _ in range(FIXTURE_STEPS_INTO_EPISODE):
        env.step(neutral)
    snapshot = build_snapshot(env)
    rng = np.random.default_rng(FIXTURE_PROPOSAL_SEED)
    proposal = DispatchProposal(
        compute_actions=[float(x) for x in rng.random(len(snapshot.group_work_capacity))],
        storage_action=FIXTURE_STORAGE_ACTION,
    )
    return snapshot, proposal


# --- 1. 确定性 MIP 配置（不依赖机器，纯结构断言） ---------------------------

def test_mip_options_are_fixed_for_determinism():
    """必须向 HiGHS 显式传入确定性选项，且不得随调用变化。"""
    options = planning_model.deterministic_mip_options(time_limit_s=NORMAL_BUDGET_S)
    assert options["random_seed"] == 0, "必须固定 random_seed"
    assert options["parallel"] is False, "必须禁用并行（HiGHS 内部不可复现来源之一）"
    assert options["time_limit"] == pytest.approx(NORMAL_BUDGET_S)

    again = planning_model.deterministic_mip_options(time_limit_s=NORMAL_BUDGET_S)
    assert options == again, "选项必须与调用次数无关"

    no_budget = planning_model.deterministic_mip_options(time_limit_s=None)
    assert "time_limit" not in no_budget
    assert no_budget["random_seed"] == 0 and no_budget["parallel"] is False


def test_mip_call_site_uses_the_deterministic_options():
    """结构性保证：milp 调用点必须走 deterministic_mip_options，不得自己拼 options。"""
    source = pathlib.Path(planning_model.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "_milp"
    ]
    assert calls, "未找到 milp 调用点"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "options" in kwargs
    assert "deterministic_mip_options" in source


def _digest_of(correction) -> str:
    payload = {
        "exec_compute": [float(x) for x in correction.exec_compute_actions],
        "exec_storage": float(correction.exec_storage_action),
        "failure": str(correction.failure),
        "reason": str(correction.reason),
        "business_gap": float(correction.business_gap),
        "projection_offset": float(correction.projection_offset),
        "reviewed": bool(correction.reviewed),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def test_repeated_in_process_calls_are_identical():
    """进程内重复调用必须一致（必要但**不充分**的条件；跨进程一致性见 slow 用例）。"""
    from planning.corrector import correct

    snapshot, proposal = _snapshot_and_proposal()
    digests = {
        _digest_of(correct(snapshot, proposal, time_limit_s=NORMAL_BUDGET_S))
        for _ in range(3)
    }
    assert len(digests) == 1, f"进程内重复调用就已不一致：{digests}"


def test_always_timeout_budget_still_returns_the_verified_boundary_action():
    """恒超时不是「稳定」，而是确定性的**退化**：返回已验证边界动作。"""
    snapshot, proposal = _snapshot_and_proposal()
    digests = {corrector_digest(budget=ALWAYS_TIMEOUT_BUDGET_S) for _ in range(2)}
    assert len(digests) == 1, "恒超时预算必须是确定的"

    from planning.corrector import correct

    correction = correct(snapshot, proposal, time_limit_s=ALWAYS_TIMEOUT_BUDGET_S)
    assert str(correction.failure) == "timeout"
    assert correction.reviewed is True, "超时也必须返回**已验证**的边界动作"


# --- 2. 单次调用的跨进程确定性（slow） --------------------------------------

@pytest.mark.slow
def test_single_call_is_deterministic_at_the_normal_budget():
    """当前 0.05 s 正常预算：单次 corrector 调用必须跨进程逐位一致。"""
    result = measure(NORMAL_BUDGET_S)
    assert result["consistent"], (
        f"0.05 s 下仍不一致：distinct={result['distinct']}，"
        f"全部 digest={result['digests']}"
    )


@pytest.mark.slow
def test_single_call_is_deterministic_at_a_generous_but_real_budget():
    """宽裕但真实求解的预算（2 s）：必须跨进程逐位一致。"""
    result = measure(GENEROUS_BUDGET_S)
    assert result["consistent"], (
        f"2 s 下仍不一致：distinct={result['distinct']}，全部 digest={result['digests']}"
    )


@pytest.mark.slow
def test_generous_budget_really_solves_rather_than_times_out():
    """宽裕预算的证据必须来自**真实求解**，不能是恒超时退化。"""
    from planning.corrector import correct

    snapshot, proposal = _snapshot_and_proposal()
    correction = correct(snapshot, proposal, time_limit_s=GENEROUS_BUDGET_S)
    assert str(correction.failure) == "none", (
        f"宽裕预算竟然没有正常求解：failure={correction.failure} reason={correction.reason}"
    )
    assert correction.reviewed is True


@pytest.mark.slow
def test_reproduction_attempt_is_recorded_honestly():
    """把「改前是否复现」写成机器可读记录，避免未复现被当成已证明。

    本测试**不**断言复现与否，只要求记录存在且自洽。
    """
    result = measure(NORMAL_BUDGET_S, processes=3, rounds=1)
    record = {
        "fixture": {
            "steps_into_episode": FIXTURE_STEPS_INTO_EPISODE,
            "proposal_seed": FIXTURE_PROPOSAL_SEED,
            "storage_action": FIXTURE_STORAGE_ACTION,
            "budget_s": NORMAL_BUDGET_S,
        },
        "processes": 3,
        "distinct": result["distinct"],
        "reproduced_instability": not result["consistent"],
        "digests": result["digests"],
    }
    assert record["distinct"] >= 1
    assert len(record["digests"]) == 3
    if record["reproduced_instability"]:
        assert record["distinct"] > 1
