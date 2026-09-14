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

# M5.4h：显式预算（默认仍是 0.05，**不得**改动默认值）
DEFAULT_CORRECTOR_TIME_LIMIT_S = 0.05
# 节点上限：**仅**探针内 runtime wrapper 注入；默认关闭
DEFAULT_NODE_CAP: int | None = None
# 测量矩阵默认档位
DEFAULT_TIME_LIMIT_MATRIX: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 2.0)
# 矩阵要求：每档独立进程数 / 每进程 rollout 步数
MATRIX_MIN_PROCESSES = 6
MATRIX_MIN_STEPS = 8

# 结论只允许三选一
ALLOWED_CONCLUSIONS = (
    "wall_clock_budget_dominant",
    "node_or_solve_path_problem",
    "insufficient_evidence",
)

# 逐阶段必须记录的字段（不可用时写 null 并在 unavailable_fields 里给出原因）
STAGE_FIELDS = (
    "options", "status", "success", "message", "elapsed_s",
    "remaining_deadline_s", "mip_node_count", "mip_dual_bound", "mip_gap",
    "corrector_failure", "corrector_reason", "digest",
)

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
    """记录求解器身份与选项 —— 结论必须能追溯到具体 solver 配置。

    M5.4g：options 必须**从 planning 的同一构造函数导出**（不再自拼），
    否则 evidence 会与真正传给 HiGHS 的选项不一致（M5.4f 的教训）。
    """
    import scipy

    from planning.model import deterministic_mip_options

    return {
        "planner_backend": "mip",
        "solver": "scipy.optimize.milp (HiGHS)",
        "scipy_version": scipy.__version__,
        "options": deterministic_mip_options(time_limit_s=float(budget)),
        "integrality": "binary on/off per step",
        "options_source": "planning.model.deterministic_mip_options",
    }


def _check_steps(steps) -> int:
    """`steps` 必须为正整数。**不得**有静默默认 —— 参数不真实是本卡要修的缺陷。"""
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError(f"steps 必须为整数，got {type(steps).__name__}={steps!r}")
    if steps <= 0:
        raise ValueError(f"steps 必须为正整数，got {steps}")
    return int(steps)


def _stage_recorder(node_cap: int | None):
    """在**当前进程内**用 runtime wrapper 观测每次 milp 调用。

    只在探针内生效：不写 planning、不改默认配置。node_cap 非 None 时，
    把 `mip_max_nodes` 注入 options（用于标定试验）。
    """
    import scipy.optimize as scipy_optimize

    records: list[dict] = []
    real_milp = scipy_optimize.milp

    def wrapper(*args, **kwargs):
        options = dict(kwargs.get("options") or {})
        if node_cap is not None:
            options["mip_max_nodes"] = int(node_cap)
            kwargs["options"] = options
        started = time.perf_counter()
        res = real_milp(*args, **kwargs)
        elapsed = time.perf_counter() - started
        unavailable: dict[str, str] = {}
        values: dict = {}
        for field in ("mip_node_count", "mip_dual_bound", "mip_gap"):
            value = getattr(res, field, None)
            if value is None:
                unavailable[field] = "scipy milp 结果未提供该字段"
                values[field] = None
            else:
                values[field] = float(value)
        records.append({
            "options": options,
            "status": int(getattr(res, "status", -1)),
            "success": bool(getattr(res, "success", False)),
            "message": str(getattr(res, "message", "")),
            "elapsed_s": float(elapsed),
            "remaining_deadline_s": (
                float(options["time_limit"]) if "time_limit" in options else None
            ),
            "node_cap_injected": node_cap,
            **values,
            "unavailable_fields": unavailable,
        })
        return res

    scipy_optimize.milp = wrapper
    return records, real_milp


def _corrector_bracket():
    """包一层 correct()，把 milp 调用按 step 归组（每步 1~2 次：先 A 后 B）。"""
    import planning.corrector as corrector_module

    real_correct = corrector_module.correct
    bounds: list[tuple[int, int]] = []

    def wrapper(*args, **kwargs):
        before = _MILP_COUNTER["n"]
        result = real_correct(*args, **kwargs)
        bounds.append((before, _MILP_COUNTER["n"]))
        return result

    corrector_module.correct = wrapper
    return bounds, real_correct


_MILP_COUNTER = {"n": 0}


def run_once(
    mode: str,
    *,
    steps: int,
    budget: float | None = None,
    record_stages: bool = False,
    node_cap: int | None = DEFAULT_NODE_CAP,
) -> dict:
    """在当前进程中跑一次采集。

    `steps` 为**必填关键字参数**。`record_stages=True` 时额外产出逐步 × 逐阶段的
    求解器事实（options / status / 节点数 / 对偶界 / 间隙 / 剩余 deadline / digest）。
    """
    steps = _check_steps(steps)
    effective_budget = MODE_BUDGETS[mode] if budget is None else float(budget)
    env = IDCPriceEnv20D(horizon=HORIZON, **ENV_SEED_KWARGS)
    policy = make_policy(env)
    generator = torch.Generator()
    generator.manual_seed(GENERATOR_SEED)

    if not record_stages:
        buffer = collect(env, policy, mode=mode, budget=effective_budget,
                         steps=steps, generator=generator)
        return {
            "digest": semantic_digest(buffer),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": len(buffer),
        }

    import planning.corrector as corrector_module

    records, real_milp = _stage_recorder(node_cap)
    real_collect = collect

    def collect_with_bracket(*args, **kwargs):
        """在 correct() 外层记录调用边界，以便把 milp 调用归到具体 step。

        注意：`safe_rl.corrector_wrapper` 在**导入时**就绑定了 `correct`，
        故必须补丁**它实际持有的那个名字**，否则 spans 恒为空。
        """
        import safe_rl.corrector_wrapper as wrapper_module

        targets = [(corrector_module, "correct"), (wrapper_module, "correct")]
        originals = [(module, name, getattr(module, name)) for module, name in targets]
        spans: list[tuple[int, int]] = []

        def counting_correct(*c_args, **c_kwargs):
            start = len(records)
            out = originals[0][2](*c_args, **c_kwargs)
            spans.append((start, len(records)))
            return out

        for module, name, _ in originals:
            setattr(module, name, counting_correct)
        try:
            buffer = real_collect(*args, **kwargs)
        finally:
            for module, name, original in originals:
                setattr(module, name, original)
        return buffer, spans

    try:
        buffer, spans = collect_with_bracket(
            env, policy, mode=mode, budget=effective_budget,
            steps=steps, generator=generator,
        )
    finally:
        import scipy.optimize as scipy_optimize

        scipy_optimize.milp = real_milp

    step_records = []
    for index, transition in enumerate(buffer.transitions):
        if index < len(spans):
            start, end = spans[index]
            calls = records[start:end]
        else:
            calls = []
        decorated = []
        for offset, call in enumerate(calls):
            stage = "A" if offset == 0 else ("B" if offset == 1 else "extra")
            entry = {"stage": stage, **call}
            entry["corrector_failure"] = str(transition.correction_info.get("correction_reason"))
            entry["corrector_reason"] = str(transition.correction_info.get("correction_reason"))
            entry["digest"] = _step_digest(transition)
            decorated.append(entry)
        step_records.append({
            "step": index,
            "digest": _step_digest(transition),
            "corrector_failure": str(transition.correction_info.get("correction_reason")),
            "corrector_reason": str(transition.correction_info.get("correction_reason")),
            "calls": decorated,
        })

    return {
        "digest": semantic_digest(buffer),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": len(buffer),
        "steps_detail": step_records,
    }


def _step_digest(transition) -> str:
    payload = {
        "exec": [float(x) for x in transition.exec_action],
        "raw_log_prob": float(transition.old_raw_log_prob),
        "terminated": bool(transition.terminated),
        "truncated": bool(transition.truncated),
        "business": float(transition.business_cost),
        "carbon": float(transition.carbon_cost),
        "info": {
            k: str(v) for k, v in transition.correction_info.items()
            if k not in WALL_CLOCK_KEYS
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def classify(matrix: dict) -> dict:
    """由测量矩阵推导结论 —— 只允许三选一，规则见 M5.4h 卡 §4.7。

    - 矩阵不完整（缺档 / 进程数 < 6 / 步数 < 8）-> insufficient_evidence
    - 未复现任何不稳定 -> insufficient_evidence
    - 有不稳定且**最大档稳定** -> wall_clock_budget_dominant
    - 有不稳定且**最大档也不稳定** -> node_or_solve_path_problem
    """
    budgets = matrix.get("budgets") or {}
    missing = [b for b in DEFAULT_TIME_LIMIT_MATRIX if str(b) not in budgets]
    if missing:
        return {
            "conclusion": "insufficient_evidence",
            "reason": f"矩阵缺少预算档 {missing}",
            "unstable_budgets": [],
        }
    for key, entry in budgets.items():
        if entry.get("processes", 0) < MATRIX_MIN_PROCESSES:
            return {
                "conclusion": "insufficient_evidence",
                "reason": (
                    f"预算 {key}s 的独立进程数 {entry.get('processes')}"
                    f" < {MATRIX_MIN_PROCESSES}"
                ),
                "unstable_budgets": [],
            }
        if entry.get("steps", 0) < MATRIX_MIN_STEPS:
            return {
                "conclusion": "insufficient_evidence",
                "reason": f"预算 {key}s 的步数 {entry.get('steps')} < {MATRIX_MIN_STEPS}",
                "unstable_budgets": [],
            }

    unstable = sorted(
        (float(key) for key, entry in budgets.items() if entry.get("distinct", 1) > 1)
    )
    if not unstable:
        return {
            "conclusion": "insufficient_evidence",
            "reason": "本矩阵未复现任何不稳定，无法归因",
            "unstable_budgets": [],
        }

    largest = max(float(key) for key in budgets)
    largest_stable = budgets[str(largest)].get("distinct", 1) == 1
    if largest_stable:
        return {
            "conclusion": "wall_clock_budget_dominant",
            "reason": (
                f"最大预算档 {largest}s 稳定，而较小档 {unstable} 不稳定 —— "
                "与「预算被耗尽」一致"
            ),
            "unstable_budgets": unstable,
        }
    return {
        "conclusion": "node_or_solve_path_problem",
        "reason": (
            f"最大预算档 {largest}s 仍不稳定 —— 非单纯预算耗尽，指向节点/求解路径"
        ),
        "unstable_budgets": unstable,
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
    "print(json.dumps(run_once(sys.argv[1], steps=int(sys.argv[2]),"
    " budget=None if sys.argv[3] == 'none' else float(sys.argv[3]),"
    " record_stages=sys.argv[4] == '1',"
    " node_cap=None if sys.argv[5] == 'none' else int(sys.argv[5]))))"
)


def _start_load(load: dict) -> list:
    """按口径拉起**受控** CPU 负载（记录程序与并发数；不依赖外部 make check）。"""
    if load["mode"] != "hogs":
        return []
    program = "x=0\nfor i in range(10**9): x+=i"
    return [
        subprocess.Popen([sys.executable, "-c", program])
        for _ in range(int(load["concurrency"]))
    ]


def _stop_load(processes: list) -> None:
    for process in processes:
        process.kill()
    for process in processes:
        process.wait()


def run_in_subprocess(
    mode: str,
    *,
    steps: int,
    budget: float | None = None,
    record_stages: bool = False,
    node_cap: int | None = DEFAULT_NODE_CAP,
    load: dict | None = None,
) -> dict:
    """在**独立 Python 进程**中跑一次采集。

    `steps` 必须真正传给子进程 —— 否则子进程会用默认值，产物自相矛盾。
    `budget` 同样透传；`load` 按口径在子进程运行期间维持受控 CPU 负载。
    """
    steps = _check_steps(steps)
    load = load or {"mode": "none", "concurrency": 0, "program": None}
    processes = _start_load(load)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_SNIPPET, mode, str(steps),
             "none" if budget is None else str(budget),
             "1" if record_stages else "0",
             "none" if node_cap is None else str(node_cap)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=1800,
        )
    finally:
        _stop_load(processes)
    if result.returncode != 0:
        raise RuntimeError(f"{mode} 子进程失败：{result.stderr}")
    entry = json.loads(result.stdout.strip().splitlines()[-1])
    entry["load"] = load
    return entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="corrector 跨进程可复现性探针")
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_BUDGETS), default=["off", "on"])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument(
        "--corrector-time-limit", type=float, default=DEFAULT_CORRECTOR_TIME_LIMIT_S,
        help="corrector 的 wall-clock 预算（秒）；默认 0.05，**不得**由本探针改动默认值",
    )
    parser.add_argument(
        "--time-limits", type=float, nargs="+", default=None,
        help="矩阵模式：多个预算档（默认单档，用 --corrector-time-limit）",
    )
    parser.add_argument("--load", choices=("none", "hogs"), default="none",
                        help="负载口径：none 无额外负载；hogs 自行拉起受控 CPU 负载")
    parser.add_argument("--load-concurrency", type=int, default=0,
                        help="--load hogs 时的并发进程数")
    parser.add_argument("--node-caps", type=int, nargs="+", default=None,
                        help="节点上限候选（**仅探针内 runtime 注入**，不写 planning/默认配置）")
    parser.add_argument("--base-dir", default="runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--emit-provenance", action="store_true")
    return parser


def machine_info() -> dict:
    """机器信息 —— 负载口径与结论必须能追溯到具体环境。"""
    import multiprocessing

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
    }


def load_info(args) -> dict:
    """负载口径 —— **不得**把 make check 并发当成唯一实验条件。"""
    if args.load == "hogs":
        return {
            "mode": "hogs",
            "concurrency": int(args.load_concurrency),
            "program": "sys.executable -c 'x=0\\nfor i in range(10**9): x+=i'",
        }
    return {"mode": "none", "concurrency": 0, "program": None}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error(f"--runs 必须 >= 1，got {args.runs}")
    if args.steps <= 0:
        parser.error(f"--steps 必须为正整数，got {args.steps}")
    if args.corrector_time_limit <= 0:
        parser.error(f"--corrector-time-limit 必须为正数，got {args.corrector_time_limit}")
    if args.load == "hogs" and args.load_concurrency < 1:
        parser.error("--load hogs 时 --load-concurrency 必须 >= 1")
    budgets = list(args.time_limits) if args.time_limits else [args.corrector_time_limit]

    base_dir = Path(args.base_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = args.run_id or f"correctorrepro_{stamp}"

    load = load_info(args)
    machine = machine_info()

    # --- 逐模式（**保持 M5.4d/M5.4e 的既有语义不变**）---
    decisions: dict[str, dict] = {}
    provenance: list[dict] = []
    base_rows: list[dict] = []
    for mode in args.modes:
        digests: list[str] = []
        for _ in range(args.runs):
            entry = run_in_subprocess(
                mode, steps=args.steps, budget=MODE_BUDGETS[mode], load=load,
            )
            digests.append(entry["digest"])
            provenance.append({**entry, "mode": mode})
            base_rows.append({
                "mode": mode,
                "digest": entry["digest"],
                "pid": entry["pid"],
                "steps": entry["steps"],
                "budget_s": MODE_BUDGETS[mode],
            })
        decisions[mode] = decide(mode, digests)
    overall = overall_conclusion(decisions)
    blocked = overall["blocked_modes"]

    # --- 矩阵（M5.4h；仅在显式 --time-limits 时执行）---
    matrix: dict[str, dict] = {"budgets": {}}
    stage_rows: list[dict] = []
    for budget in (budgets if args.time_limits else []):
        budget_digests: list[str] = []
        for _ in range(args.runs):
            entry = run_in_subprocess(
                "on", steps=args.steps, budget=budget, record_stages=True, load=load,
            )
            budget_digests.append(entry["digest"])
            provenance.append({
                "mode": "on", "budget_s": budget, "digest": entry["digest"],
                "pid": entry["pid"], "python": entry["python"],
                "started_at": entry["started_at"], "steps": entry["steps"],
            })
            for step in entry["steps_detail"]:
                for call in step["calls"]:
                    stage_rows.append({
                        "budget_s": budget, "pid": entry["pid"], "step": step["step"],
                        "stage": call["stage"], "status": call["status"],
                        "success": call["success"], "elapsed_s": call["elapsed_s"],
                        "remaining_deadline_s": call["remaining_deadline_s"],
                        "mip_node_count": call["mip_node_count"],
                        "mip_dual_bound": call["mip_dual_bound"],
                        "mip_gap": call["mip_gap"],
                        "corrector_reason": call["corrector_reason"],
                        "node_cap_injected": call["node_cap_injected"],
                        "options_json": json.dumps(call["options"], sort_keys=True),
                    })
        matrix["budgets"][str(budget)] = {
            "distinct": len(set(budget_digests)),
            "processes": len(budget_digests),
            "steps": args.steps,
            "digests": budget_digests,
        }

    attribution = classify(matrix) if args.time_limits else None
    if attribution is not None:
        overall = {
            "blocked": attribution["conclusion"] != "wall_clock_budget_dominant",
            "blocked_modes": [] if attribution["conclusion"] == "wall_clock_budget_dominant"
                              else ["on"],
            "conclusion": attribution["conclusion"],
            "reason": attribution["reason"],
        }
        blocked = overall["blocked_modes"]
    rows = base_rows + stage_rows

    # 命令账本：用 shlex.join 完整记录**全部有效参数**（M5.4c 的标准）
    _argv = ["python", "scripts/probe_corrector_repro.py",
             "--modes", *args.modes,
             "--runs", str(args.runs),
             "--steps", str(args.steps),
             "--corrector-time-limit", str(args.corrector_time_limit),
             "--load", args.load,
             "--base-dir", str(args.base_dir),
             "--run-id", run_id]
    if args.time_limits:
        _argv += ["--time-limits", *[str(b) for b in args.time_limits]]
    if args.load == "hogs":
        _argv += ["--load-concurrency", str(args.load_concurrency)]
    if args.node_caps:
        _argv += ["--node-caps", *[str(c) for c in args.node_caps]]
    if args.emit_provenance:
        _argv.append("--emit-provenance")
    command = shlex.join(_argv)
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
        "corrector_time_limit_s": float(args.corrector_time_limit),
        "time_limit_matrix_s": [float(b) for b in budgets],
        "node_caps": list(args.node_caps) if args.node_caps else [],
        "node_cap_default": DEFAULT_NODE_CAP,
        "load": load,
        "machine": machine,
        "budget_s": MODE_BUDGETS["on"],
        "wall_clock_keys_excluded": list(WALL_CLOCK_KEYS),
        "digest_fields": list(DIGEST_FIELDS) + list(DIGEST_INFO_FIELDS),
        "load_injected": False,
        "retry_selection": False,
    }
    report = {
        "probe": "m54h_budget_attribution",
        "trained": False,
        "steps": int(args.steps),
        "corrector_time_limit_s": float(args.corrector_time_limit),
        "time_limit_matrix_s": [float(b) for b in budgets],
        "load": load,
        "machine": machine,
        "statement": REPORT_STATEMENT,
        "claims": {"trained": False, "performance_evaluated": False, "convergence_claimed": False},
        "modes": decisions,
        "matrix": matrix,
        "attribution": attribution,
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

    # 汇总表：不得只在任务卡写人工摘要
    summary = {
        "probe": "m54h_budget_attribution",
        "matrix": matrix,
        "conclusion": attribution["conclusion"] if attribution else overall["conclusion"],
        "reason": attribution["reason"] if attribution else "（未请求矩阵；见 modes 结论）",
        "load": load,
        "machine": machine,
        "stage_field_count": len(stage_rows),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if stage_rows:
        pd.DataFrame(stage_rows).to_parquet(run_dir / "summary.parquet")

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
