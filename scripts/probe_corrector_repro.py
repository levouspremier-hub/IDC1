#!/usr/bin/env python
"""M5.4d corrector 跨进程可复现性探针（**真实测量，不预设结论**）。

在**独立 Python 进程**中重复同一段采集，比较排除纯墙钟字段后的语义摘要：

- `off`：corrector 关闭 —— **期望**跨进程逐位一致；
- `on` ：corrector 开启、明确固定且**非恒超时**的预算 —— 结论**由测量决定**。

**检测力说明（实测）**：`on` 的不稳定是**间歇**的。以 `--runs 4` 连做 5 次调用，
其中 1 次得到 `distinct=2`（blocked）、4 次得到 `distinct=1`。
也就是说**单次调用**在 `--runs 4` 下只有约 1/5 的概率撞上不稳定 ——
因此**单次 `reproducible` 不足以断言正确**，需要多次调用或更大的 `--runs`
（每个子进程约 10 s，`--runs` 越大越慢）。`off` 在同 5 次调用中始终 `distinct=1`。

本探针**不**注入 CPU hog、**不**重试挑选取样、**不**把 timeout 当作正常求解成功，
源码中**不含**任何预设 digest 或预设布尔结论。

产出的 `runs/<id>/` 记录代码 revision、solver 证据、seed、horizon、预算、
每次 digest 与 distinct 数。**本探针不代表任何训练或性能结果。**
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from envs.idc_price_env import IDCPriceEnv20D
from runs.writer import write_run
from safe_rl_v2.buffer import RolloutBuffer
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

REPO_ROOT = Path(__file__).resolve().parent.parent

ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
POLICY_SEED = 0
GENERATOR_SEED = 0
HORIZON = 24
STEPS = 8
DEFAULT_RUNS = 3

# 每模式的 corrector 预算（秒）；`off` 不用 corrector。
# `on` 用 0.05 s：明确、固定，且**不是**恒超时（恒超时会退化为确定性的边界动作，
# 那会把「不稳定」掩盖成「稳定」，反而失去测量意义）。
MODE_BUDGETS: dict[str, float | None] = {"off": None, "on": 0.05}

# 纯墙钟字段：**只**排除这三项
WALL_CLOCK_KEYS = (
    "correction_solve_time_s",
    "stage_a_solve_time_s",
    "stage_b_solve_time_s",
)

# digest 覆盖 transition 的顶层字段
DIGEST_FIELDS = (
    "observation",
    "next_observation",
    "raw_action",
    "exec_action",
    "old_raw_log_prob",
    "reward",
    "business_cost",
    "carbon_cost",
    "electricity_cost_sgd",
    "terminated",
    "truncated",
)
# digest 覆盖 correction_info 中的审计字段（排除墙钟）
DIGEST_INFO_FIELDS = (
    "correction_reason",
    "planner_backend",
    "stage_a_status",
    "stage_b_status",
    "stage_a_objective",
    "stage_b_objective",
    "projection_offset",
    "business_gap",
    "deadline_shortfall_work",
)

REPORT_STATEMENT = (
    "本探针只测量 corrector 的跨进程可复现性；不代表训练、评估或性能结论。"
)


# --- 采集与摘要 -------------------------------------------------------------

def make_policy(env) -> SafePPOPolicy:
    """固定权重：借用全局 RNG 但用 fork_rng 还原，采样另用显式 generator。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(POLICY_SEED)
        return SafePPOPolicy(obs_dim=env.obs_dim)


def collect(env, policy, *, mode: str, budget: float | None, steps: int, generator):
    buffer = RolloutBuffer()
    collect_rollout(
        env, policy, buffer, steps=steps, seed=0,
        corrector_on=(mode != "off"),
        corrector_time_limit_s=budget if mode != "off" else None,
        generator=generator,
    )
    return buffer


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.ravel()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def semantic_digest(buffer: RolloutBuffer) -> str:
    """跨进程比较用的语义摘要；**只**排除三个纯墙钟字段。"""
    transitions = []
    for t in buffer.transitions:
        entry = {name: _jsonable(getattr(t, name)) for name in DIGEST_FIELDS}
        entry["correction_info"] = {
            key: _jsonable(value)
            for key, value in t.correction_info.items()
            if key not in WALL_CLOCK_KEYS
        }
        transitions.append(entry)
    blob = json.dumps(transitions, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def solver_evidence(*, budget: float) -> dict:
    """记录求解器身份与选项 —— 结论必须能追溯到具体 solver 配置。"""
    import scipy

    return {
        "planner_backend": "mip",
        "solver": "scipy.optimize.milp (HiGHS)",
        "scipy_version": scipy.__version__,
        "options": {"time_limit": float(budget)},
        "integrality": "binary on/off per step",
    }


def run_once(mode: str, *, steps: int = STEPS) -> dict:
    """在当前进程中跑一次采集，返回 digest 与进程来源信息。"""
    budget = MODE_BUDGETS[mode]
    env = IDCPriceEnv20D(horizon=HORIZON, **ENV_SEED_KWARGS)
    policy = make_policy(env)
    generator = torch.Generator()
    generator.manual_seed(GENERATOR_SEED)
    buffer = collect(env, policy, mode=mode, budget=budget, steps=steps, generator=generator)
    return {
        "digest": semantic_digest(buffer),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": len(buffer),
    }


# --- 结论逻辑（纯函数；本探针不预设结果） -----------------------------------

def decide(mode: str, digests: list[str]) -> dict:
    """由**测量到的** digest 推导结论，不做任何预设。"""
    if mode not in MODE_BUDGETS:
        raise ValueError(f"未知模式 {mode!r}，必须属于 {sorted(MODE_BUDGETS)}")
    if not digests:
        raise ValueError("digests 不得为空：没有测量就没有结论")
    distinct = len(set(digests))
    reproducible = distinct == 1
    if reproducible:
        failure_classification = None
    else:
        failure_classification = (
            "corrector_nonreproducible" if mode == "on" else "corrector_off_nonreproducible"
        )
    return {
        "mode": mode,
        "runs": len(digests),
        "distinct": distinct,
        "digests": list(digests),  # **全部**保留，不挑不删
        "reproducible": reproducible,
        "blocked": not reproducible,
        "failure_classification": failure_classification,
        "conclusion": "reproducible" if reproducible else "blocked",
    }


def overall_conclusion(decisions: dict[str, dict]) -> dict:
    """跨模式汇总：**任一**模式被 blocked，整体即为 blocked（纯函数）。

    依据任务卡 M5.4d §8：只要 corrector 开启仍不一致，M5.4 阶段结论必须是 blocked。
    """
    if not decisions:
        raise ValueError("decisions 不得为空：没有测量就没有结论")
    blocked = sorted(mode for mode, decision in decisions.items() if decision["blocked"])
    return {
        "blocked": bool(blocked),
        "blocked_modes": blocked,
        "conclusion": "blocked" if blocked else "reproducible",
    }


# --- 独立进程执行 -----------------------------------------------------------

_CHILD_SNIPPET = (
    "import json,sys;"
    "from scripts.probe_corrector_repro import run_once;"
    "print(json.dumps(run_once(sys.argv[1])))"
)


def run_in_subprocess(mode: str) -> dict:
    """在**独立 Python 进程**中跑一次采集（不注入负载、不重试）。"""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SNIPPET, mode],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{mode} 子进程失败：{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="corrector 跨进程可复现性探针")
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_BUDGETS), default=["off", "on"])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--base-dir", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--emit-provenance", action="store_true")
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error(f"--runs 必须 >= 1，got {args.runs}")

    base_dir = Path(args.base_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = args.run_id or f"correctorrepro_{stamp}"

    decisions: dict[str, dict] = {}
    provenance: list[dict] = []
    rows: list[dict] = []
    for mode in args.modes:
        digests: list[str] = []
        for _ in range(args.runs):
            entry = run_in_subprocess(mode)
            digests.append(entry["digest"])
            provenance.append({**entry, "mode": mode})
            rows.append({
                "mode": mode,
                "digest": entry["digest"],
                "pid": entry["pid"],
                "steps": entry["steps"],
                "budget_s": MODE_BUDGETS[mode],
            })
        decisions[mode] = decide(mode, digests)

    overall = overall_conclusion(decisions)
    blocked = overall["blocked_modes"]

    command = shlex.join(["python", "scripts/probe_corrector_repro.py", *(
        ["--modes", *args.modes, "--runs", str(args.runs), "--run-id", run_id]
    )])
    config = {
        "probe": "m54d_corrector_reproducibility",
        "trained": False,
        "code_revision": _git_revision(),
        "solver": solver_evidence(budget=MODE_BUDGETS["on"] or 0.0),
        "env_seed_kwargs": dict(ENV_SEED_KWARGS),
        "policy_seed": POLICY_SEED,
        "generator_seed": GENERATOR_SEED,
        "horizon": HORIZON,
        "steps": args.steps,
        "modes": list(args.modes),
        "runs_per_mode": args.runs,
        "budget_s": MODE_BUDGETS["on"],
        "wall_clock_keys_excluded": list(WALL_CLOCK_KEYS),
        "digest_fields": list(DIGEST_FIELDS) + list(DIGEST_INFO_FIELDS),
        "load_injected": False,
        "retry_selection": False,
    }
    report = {
        "probe": "m54d_corrector_reproducibility",
        "trained": False,
        "statement": REPORT_STATEMENT,
        "claims": {"trained": False, "performance_evaluated": False, "convergence_claimed": False},
        "modes": decisions,
        "overall": overall,
        "provenance_note": "每个 digest 来自一个独立 Python 进程；未注入 CPU 负载、未重试挑选取样",
    }

    status = "failed" if overall["blocked"] else "success"
    run_dir = write_run(
        run_id,
        config=config,
        metrics=pd.DataFrame(rows),
        report=report,
        base_dir=str(base_dir),
        seed=POLICY_SEED,
        command=command,
        dependency_lock_hash=_dependency_lock_hash(),
        status=status,
        failure_classification=(
            decisions[blocked[0]]["failure_classification"] if blocked else None
        ),
    )
    if args.emit_provenance:
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nrun 产物：{run_dir}")
    for mode, decision in decisions.items():
        print(f"  {mode:>3}: runs={decision['runs']} distinct={decision['distinct']} "
              f"-> {decision['conclusion']}")
    return 1 if overall["blocked"] else 0


def _git_revision() -> str:
    from runs.writer import git_revision

    return git_revision()


def _dependency_lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
