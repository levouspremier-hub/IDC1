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
from planning.corrector import (
    CORRECTOR_TIME_LIMIT_SOURCE_DISABLED,
    CORRECTOR_TIME_LIMIT_SOURCE_EXPLICIT_OVERRIDE,
    CORRECTOR_TIME_LIMIT_SOURCE_PRODUCTION_DEFAULT,
    PRODUCTION_CORRECTOR_TIME_LIMIT_S,
    resolve_corrector_budget,
)
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
# `on` 的预算是**本 run 的有效预算**（默认即生产默认），由 `main` 解析后传入。
# 预算**不得**小到恒超时：恒超时会退化为确定性的边界动作，把「不稳定」掩盖成
# 「稳定」，反而失去测量意义。
MODE_BUDGETS: dict[str, float | None] = {"off": None, "on": PRODUCTION_CORRECTOR_TIME_LIMIT_S}

# M5.4i：corrector 生产默认预算的**唯一**来源是
# `planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S`（0.25 s）；本文件不再定义。
# 0.05 s 仍是可用的**显式 override**（诊断值），绝不静默改写。
# 节点上限：**仅**探针内 runtime wrapper 注入；默认关闭
DEFAULT_NODE_CAP: int | None = None
# 测量矩阵默认档位
DEFAULT_TIME_LIMIT_MATRIX: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 2.0)
# 矩阵要求：每档独立进程数 / 每进程 rollout 步数
MATRIX_MIN_PROCESSES = 6
MATRIX_MIN_STEPS = 8


def resolve_effective_budget(requested_time_limit_s: float | None) -> tuple[float, str]:
    """解析 `(有效预算, 来源)`；来源由**是否显式给出**决定，与数值无关。"""
    effective, source = resolve_corrector_budget(requested_time_limit_s, enabled=True)
    assert effective is not None  # enabled=True 时必为数值
    return effective, source


def resolve_budget_source(
    requested_time_limit_s: float | None, *, disabled: bool = False
) -> str:
    """只要来源字符串。`disabled=True`（corrector 关闭）时为 `disabled`。"""
    return resolve_corrector_budget(requested_time_limit_s, enabled=not disabled)[1]


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


def _stage_recorder(node_cap: int | None, *, wall_clock_disabled: bool = False):
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
        if wall_clock_disabled:
            # 替代语义臂：**移除** time_limit，让停止判据只由节点上限决定。
            # 仅在 probe 的 runtime wrapper 内生效，绝不写入 planning/ 或默认配置。
            options.pop("time_limit", None)
        if node_cap is not None or wall_clock_disabled:
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
            "wall_clock_disabled_in_probe": bool(wall_clock_disabled),
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
    wall_clock_disabled: bool = False,
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

    records, real_milp = _stage_recorder(
        node_cap, wall_clock_disabled=wall_clock_disabled
    )
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


def encode_options_json(value) -> str | None:
    """审计表 `options` 列的**固定**编码规则（M5.4h2）。

    `None`（未执行的 skipped 阶段）保持 `null`；其余一律写规范 JSON 字符串。
    **不得**同时保留 `options_json` 兼容列。
    """
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _skipped_row(*, step, stage, budget, pid, entry, node_cap, wall_clock_disabled, reason):
    """未执行的阶段也必须有一条显式行 —— 不得静默少行。"""
    return {
        "budget_s": budget, "node_cap": node_cap, "pid": pid, "step": step,
        "stage": stage, "skipped": True, "skip_reason": reason,
        "options": None, "status": None, "success": None, "message": None,
        "elapsed_s": None, "remaining_deadline_s": None,
        "mip_node_count": None, "mip_dual_bound": None, "mip_gap": None,
        "unavailable_fields": None,
        "corrector_failure": entry.get("corrector_failure"),
        "corrector_reason": entry.get("corrector_reason"),
        "digest": entry.get("digest"),
        "node_cap_injected": node_cap,
        "wall_clock_disabled_in_probe": bool(wall_clock_disabled),
    }


def build_stage_rows(
    *, entry: dict, budget: float, pid: int, node_cap: int | None,
    wall_clock_disabled: bool = False,
) -> list[dict]:
    """把一条 rollout 的逐步 × 逐阶段记录摊平成审计行。

    每个 `step × {A, B}` **恰好一行**；未执行的阶段写 `skipped=true` 与原因。
    """
    rows: list[dict] = []
    for step in entry.get("steps_detail", []):
        calls = step.get("calls") or []
        by_stage = {call.get("stage"): call for call in calls}
        for stage in ("A", "B"):
            call = by_stage.get(stage)
            if call is None:
                rows.append(_skipped_row(
                    step=step["step"], stage=stage, budget=budget, pid=pid,
                    entry=step, node_cap=node_cap,
                    wall_clock_disabled=wall_clock_disabled,
                    reason=(
                        f"Stage {stage} 未执行"
                        + (
                            f"（该步实际执行 {sorted(by_stage)}）"
                            if by_stage else "（该步无求解调用）"
                        )
                    ),
                ))
                continue
            rows.append({
                "budget_s": budget, "node_cap": node_cap, "pid": pid,
                "step": step["step"], "stage": stage, "skipped": False,
                "skip_reason": None,
                "options": encode_options_json(call.get("options")),
                "status": call.get("status"),
                "success": call.get("success"),
                "message": call.get("message"),
                "elapsed_s": call.get("elapsed_s"),
                "remaining_deadline_s": call.get("remaining_deadline_s"),
                "mip_node_count": call.get("mip_node_count"),
                "mip_dual_bound": call.get("mip_dual_bound"),
                "mip_gap": call.get("mip_gap"),
                "unavailable_fields": json.dumps(
                    call.get("unavailable_fields") or {}, sort_keys=True
                ),
                "corrector_failure": call.get("corrector_failure"),
                "corrector_reason": call.get("corrector_reason"),
                "digest": call.get("digest"),
                "node_cap_injected": call.get("node_cap_injected"),
                "wall_clock_disabled_in_probe": bool(
                    call.get("wall_clock_disabled_in_probe", wall_clock_disabled)
                ),
            })
        extra = [c for c in calls if c.get("stage") == "extra"]
        for call in extra:
            rows.append({
                "budget_s": budget, "node_cap": node_cap, "pid": pid,
                "step": step["step"], "stage": "extra", "skipped": False,
                "skip_reason": None,
                "options": encode_options_json(call.get("options")),
                "status": call.get("status"), "success": call.get("success"),
                "message": call.get("message"), "elapsed_s": call.get("elapsed_s"),
                "remaining_deadline_s": call.get("remaining_deadline_s"),
                "mip_node_count": call.get("mip_node_count"),
                "mip_dual_bound": call.get("mip_dual_bound"),
                "mip_gap": call.get("mip_gap"),
                "unavailable_fields": json.dumps(
                    call.get("unavailable_fields") or {}, sort_keys=True
                ),
                "corrector_failure": call.get("corrector_failure"),
                "corrector_reason": call.get("corrector_reason"),
                "digest": call.get("digest"),
                "node_cap_injected": call.get("node_cap_injected"),
                "wall_clock_disabled_in_probe": bool(
                    call.get("wall_clock_disabled_in_probe", wall_clock_disabled)
                ),
            })
    return rows


def evaluate_release_gate(
    observations: list[dict],
    *,
    default_budget: float | None = None,
    mode_observations: list[dict] | None = None,
    override_observations: list[dict] | None = None,
) -> dict:
    """**发布门禁**：决定 M5.4 阶段是否放行。

    与 `classify`（预算**归因**）是两个概念：归因成立**不能**让本门禁放行。

    `observations` 是**生产默认预算**的跨进程观测；`mode_observations` 是平台确定性
    观测（corrector 关闭时本应逐位一致）。两者**任一**不一致都必须拦下阶段放行 ——
    本门禁是阶段放行的**唯一**依据。

    **只认生产默认**（M5.4i）：只有 `corrector_time_limit_source ==
    "production_default"` 的观测才算生产默认证据；显式 override 的观测被挪到
    `override_observations`，如实记录但**绝不**参与放行判定。
    """
    budget = PRODUCTION_CORRECTOR_TIME_LIMIT_S if default_budget is None else default_budget
    submitted = list(observations or [])
    mode_observations = list(mode_observations or [])
    observations = [
        o for o in submitted
        if o.get("corrector_time_limit_source") == CORRECTOR_TIME_LIMIT_SOURCE_PRODUCTION_DEFAULT
    ]
    override_observations = list(override_observations or []) + [
        o for o in submitted
        if o.get("corrector_time_limit_source") != CORRECTOR_TIME_LIMIT_SOURCE_PRODUCTION_DEFAULT
    ]
    combined = observations + mode_observations
    if not combined:
        not_measured = (
            f"本 run 未测量生产默认 {budget}s corrector"
            if not override_observations
            else (
                f"本 run 只测量了显式 override（"
                f"{[o['source'] for o in override_observations]}），"
                f"**未**测量生产默认 {budget}s —— override 不得充当生产默认证据"
            )
        )
        return {
            "evaluated": False,
            "default_budget_s": budget,
            "observations": [],
            "mode_observations": [],
            "override_observations": override_observations,
            "blocked": False,
            "all_qualifying": None,
            "passed": False,
            "unstable_sources": [],
            "underpowered_sources": [],
            "reason": f"{not_measured} —— 样本量不足，**不得**据此放行该阶段（fail closed）",
        }
    unstable_sources = [o["source"] for o in combined if int(o["distinct"]) > 1]
    blocked = bool(unstable_sources)
    underpowered_sources = [
        {"source": o["source"], "processes": o.get("processes", 0), "steps": o.get("steps", 0)}
        for o in combined
        if o.get("processes", 0) < MATRIX_MIN_PROCESSES
        or o.get("steps", 0) < MATRIX_MIN_STEPS
    ]
    all_qualifying = not underpowered_sources
    evaluated = bool(observations)
    # **唯一**的阶段放行判据：测过、样本量达标、且未 blocked。缺一即 fail closed。
    passed = bool(evaluated and all_qualifying and not blocked)
    if blocked:
        reason = (
            f"发布门禁 blocked：{unstable_sources} 跨进程不一致（distinct>1）；"
            f"默认预算 {budget}s 不得放行"
        )
    elif not evaluated:
        reason = (
            f"未测量默认 {budget}s corrector —— 样本量不足（要求每个观测 "
            f"processes>={MATRIX_MIN_PROCESSES} 且 steps>={MATRIX_MIN_STEPS}），"
            "**不得**据此放行该阶段（fail closed）"
        )
    elif underpowered_sources:
        shortfalls = [s["source"] for s in underpowered_sources]
        reason = (
            f"默认预算 {budget}s 的观测样本量不足（{shortfalls} 未达 "
            f"processes>={MATRIX_MIN_PROCESSES} 且 steps>={MATRIX_MIN_STEPS}）—— "
            "**不得**据「distinct=1」放行该阶段（fail closed）"
        )
    else:
        reason = (
            f"默认预算 {budget}s 的观测全部 distinct=1，且样本量达标"
            f"（processes>={MATRIX_MIN_PROCESSES}，steps>={MATRIX_MIN_STEPS}）"
        )
    return {
        "evaluated": evaluated,
        "default_budget_s": budget,
        "observations": observations,
        "mode_observations": mode_observations,
        "override_observations": override_observations,
        "blocked": blocked,
        "all_qualifying": all_qualifying,
        "passed": passed,
        "unstable_sources": unstable_sources,
        "underpowered_sources": underpowered_sources,
        "reason": reason,
    }


def phase_status(release_gate: dict, *, attribution: dict | None = None) -> dict:
    """阶段放行状态 —— **只**由 `release_gate` 决定。

    语义分层（M5.4h2 固定）：

    - `release_gate`：决定 M5.4 阶段是否放行；
    - `overall`（本函数返回）：反映 `release_gate` 的最终阶段状态；
    - `attribution`：只解释不稳定的原因，**绝不**决定是否放行。

    故 `overall["blocked"] == release_gate["blocked"]` 恒成立；`attribution` 只作为
    上下文记录，不参与判定。

    **fail closed**：`manifest.status` / 退出码由 `release_gate["passed"]` 决定，
    **不是**只看 `blocked`。未测量、样本量不足（processes/steps 不达标）与 blocked
    一律为 `failed` / 非 0；**只有** `passed=true` 才是 `success` / 0。

    返回的 `status` / `exit_code` 供 manifest 与进程退出码共用，三者**不得**各自推导。
    """
    blocked = bool(release_gate.get("blocked"))
    evaluated = bool(release_gate.get("evaluated"))
    all_qualifying = bool(release_gate.get("all_qualifying", False))
    passed = bool(release_gate.get("passed", False))
    if blocked:
        conclusion = "blocked"
    elif not evaluated:
        conclusion = "not_evaluated"
    elif not all_qualifying:
        conclusion = "insufficient_evidence"
    else:
        # 措辞刻意保守：门禁放行**不等于** M5.4 阶段已发布（发布由 M5.4i 与人工审查决定）。
        conclusion = "not_blocked"
    status = "success" if passed else "failed"
    return {
        "overall": {
            "blocked": blocked,
            "conclusion": conclusion,
            "decided_by": "release_gate",
            "reason": release_gate.get("reason", ""),
            "unstable_sources": list(release_gate.get("unstable_sources", [])),
            "underpowered_sources": list(release_gate.get("underpowered_sources", [])),
            "attribution_conclusion": (
                attribution.get("conclusion") if attribution else None
            ),
        },
        "status": status,
        "exit_code": 0 if passed else 1,
    }


def node_cap_binding(rows: list[dict], *, cap: int) -> dict:
    """由**实测**求解事实判断节点上限是否真的绑定。

    只统计已执行、且有 `mip_node_count` 的行；未执行的 skipped 行不参与。
    `cap_bound` 仅在观测到的节点数触及 cap 时为真 —— **未绑定就不是候选**。
    """
    observed = [
        float(row["mip_node_count"])
        for row in rows
        if not row.get("skipped") and row.get("mip_node_count") is not None
    ]
    maximum = max(observed) if observed else None
    return {
        "cap": int(cap),
        "observed_solves": len(observed),
        "max_mip_node_count": maximum,
        "cap_bound": bool(maximum is not None and maximum >= float(cap)),
    }


def cap_is_candidate(entry: dict) -> bool:
    """**同一个 cap** 必须同时满足全部条件才是候选（返修要求）。

    - `cap_bound`：实测 `mip_node_count` 触及该 cap；
    - `distinct == 1`：该 cap 的独立进程 digest 一致；
    - `processes >= MATRIX_MIN_PROCESSES` 且 `steps >= MATRIX_MIN_STEPS`：样本量达标。

    不得把「某个 cap 绑定」与「另一个 cap 稳定」拼成候选。
    """
    return bool(
        entry.get("cap_bound")
        and entry.get("distinct") == 1
        and entry.get("processes", 0) >= MATRIX_MIN_PROCESSES
        and entry.get("steps", 0) >= MATRIX_MIN_STEPS
    )


def candidate_notes(
    attribution: dict | None, release_gate: dict, node_cap_matrix: dict
) -> dict:
    """候选值的**如实**表述。

    - 不得声称跨机器保证；
    - 节点上限**只有**在**同一个 cap** 上同时满足「实测确实绑定」「digest 一致」
      「独立进程数与步数达标」时才是候选；否则 `node_cap_effective`/`node_cap_candidate`
      与 `candidate_caps` 必须如实给出。
    """
    notes: dict = {
        "default_budget_still_blocked": bool(release_gate["blocked"]),
        "machine_local": True,
        "disclaimer": "以下候选值均为**本机**观测，不构成跨机器保证；M5.4 默认状态仍为 blocked。",
    }
    if attribution and attribution.get("conclusion") == "wall_clock_budget_dominant":
        notes["wall_clock_candidate"] = (
            "存在「较大预算档稳定、较小档不稳定」的证据；"
            "可采用经证明非绑定的 wall-clock 预算（候选值随机器变化，仅本机）"
        )
    alternatives = {
        cap: m for cap, m in node_cap_matrix.items()
        if m.get("wall_clock_disabled_in_probe")
    }
    bound_caps = sorted(
        (cap for cap, m in alternatives.items() if m.get("cap_bound")), key=int
    )
    stable_caps = sorted(
        (cap for cap, m in alternatives.items() if m.get("distinct") == 1), key=int
    )
    # 候选必须是**同一个 cap** 的证据交集，而不是两个 cap 各取一半。
    candidate_caps = sorted(
        (cap for cap, m in alternatives.items() if cap_is_candidate(m)), key=int
    )
    effective = bool(bound_caps)
    is_candidate = bool(candidate_caps)
    if not effective:
        reasons = [
            "observed_mip_node_count_below_cap",
            "stability_attributable_to_wall_clock_disabled_in_probe",
            "mip_max_nodes_not_a_production_candidate",
        ]
    elif is_candidate:
        reasons = ["node_cap_observed_to_bind", "candidate_cap_meets_all_conditions"]
    else:
        reasons = ["node_cap_observed_to_bind", "no_cap_meets_all_candidate_conditions"]
    notes["node_cap_effective"] = effective
    notes["node_cap_candidate"] = is_candidate
    notes["candidate_caps"] = candidate_caps
    notes["node_cap_reasons"] = reasons
    if alternatives:
        notes["node_cap_alternative_semantics"] = {
            "caps_tested": sorted(alternatives, key=int),
            "stable_caps": stable_caps,
            "cap_bound_caps": bound_caps,
            "candidate_caps": candidate_caps,
            "max_mip_node_count_by_cap": {
                cap: m.get("max_mip_node_count")
                for cap, m in sorted(alternatives.items(), key=lambda kv: int(kv[0]))
            },
            "node_cap_effective": effective,
            "node_cap_candidate": is_candidate,
            "note": _node_cap_note(effective=effective, candidate_caps=candidate_caps),
        }
    return notes


def _node_cap_note(*, effective: bool, candidate_caps: list[str]) -> str:
    if not effective:
        return (
            "实测 mip_node_count 未触及任何被注入的 cap，节点上限**从未绑定**；"
            "替代语义臂的稳定性只能归因于 probe 内关闭 wall-clock（移除 time_limit），"
            "**不得**把 mip_max_nodes 描述为生产候选或确定性根因 -> 不得作为候选"
        )
    if candidate_caps:
        return (
            f"cap {candidate_caps} 同时满足「实测绑定 + digest 一致 + 样本量达标」"
            "-> 仅可称为**本机**候选，不构成跨机器保证"
        )
    return (
        "虽有 cap 实测绑定，但**没有任何单个 cap** 同时满足「绑定 + distinct=1 + "
        "processes>=6 + steps>=8」-> **不得**作为候选"
    )


def classify(matrix: dict) -> dict:
    """由测量矩阵推导结论 —— 只允许三选一，规则见 M5.4h 卡 §4.7。

    - 矩阵不完整（缺档 / 进程数 < 6 / 步数 < 8）-> insufficient_evidence
    - 未复现任何不稳定 -> insufficient_evidence
    - 有不稳定且**最大档稳定** -> wall_clock_budget_dominant
    - 有不稳定且**最大档也不稳定** -> node_or_solve_path_problem
    """
    raw_budgets = matrix.get("budgets") or {}
    try:
        budgets = {float(k): v for k, v in raw_budgets.items()}
    except (TypeError, ValueError):
        return {
            "conclusion": "insufficient_evidence",
            "reason": f"预算键无法解析为数值：{sorted(raw_budgets)}",
            "unstable_budgets": [],
        }
    missing = [b for b in DEFAULT_TIME_LIMIT_MATRIX if b not in budgets]
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
        key for key, entry in budgets.items() if entry.get("distinct", 1) > 1
    )
    if not unstable:
        return {
            "conclusion": "insufficient_evidence",
            "reason": "本矩阵未复现任何不稳定，无法归因",
            "unstable_budgets": [],
        }

    largest = max(budgets)
    largest_stable = budgets[largest].get("distinct", 1) == 1
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
    " node_cap=None if sys.argv[5] == 'none' else int(sys.argv[5]),"
    " wall_clock_disabled=sys.argv[6] == '1')))"
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
    wall_clock_disabled: bool = False,
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
             "none" if node_cap is None else str(node_cap),
             "1" if wall_clock_disabled else "0"],
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
        "--corrector-time-limit", type=float, default=None,
        help="corrector 的 wall-clock 预算（秒）的**显式 override**；省略则使用生产默认"
             " planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S",
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
    parser.add_argument(
        "--node-cap-arm", choices=("with_wall_clock", "alternative_semantics"),
        default="with_wall_clock",
        help="节点上限臂：with_wall_clock 保留 time_limit；"
             "alternative_semantics **移除** time_limit（仅探针 runtime）",
    )
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
    """负载口径 —— **不得**把 make check 并发当成唯一实验条件。

    `load_injected` 由口径**推导**（不得硬编码），否则账本会自相矛盾。
    """
    if args.load == "hogs":
        return {
            "mode": "hogs",
            "concurrency": int(args.load_concurrency),
            "program": "sys.executable -c 'x=0\\nfor i in range(10**9): x+=i'",
            "load_injected": True,
        }
    return {"mode": "none", "concurrency": 0, "program": None, "load_injected": False}


def provenance_note(load: dict) -> str:
    """溯源说明必须与**实际**负载口径一致（不得写死「未注入」）。"""
    if load.get("load_injected"):
        return (
            f"每个 digest 来自一个独立 Python 进程；"
            f"**已注入受控 CPU 负载**（program={load['program']}，"
            f"concurrency={load['concurrency']}）；未重试挑选取样"
        )
    return "每个 digest 来自一个独立 Python 进程；未注入 CPU 负载、未重试挑选取样"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error(f"--runs 必须 >= 1，got {args.runs}")
    if args.steps <= 0:
        parser.error(f"--steps 必须为正整数，got {args.steps}")
    if args.corrector_time_limit is not None and args.corrector_time_limit <= 0:
        parser.error(f"--corrector-time-limit 必须为正数，got {args.corrector_time_limit}")
    if args.load == "hogs" and args.load_concurrency < 1:
        parser.error("--load hogs 时 --load-concurrency 必须 >= 1")
    effective_budget, budget_source = resolve_effective_budget(args.corrector_time_limit)
    # 显式 `--time-limits` 的档位一律是诊断用的 override；不传时默认档即生产默认。
    matrix_is_explicit = bool(args.time_limits)
    budgets = list(args.time_limits) if matrix_is_explicit else [effective_budget]

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
            mode_budget = None if mode == "off" else effective_budget
            entry = run_in_subprocess(
                mode, steps=args.steps, budget=mode_budget, load=load,
            )
            digests.append(entry["digest"])
            provenance.append({**entry, "mode": mode})
            base_rows.append({
                "mode": mode,
                "digest": entry["digest"],
                "pid": entry["pid"],
                "steps": entry["steps"],
                "budget_s": mode_budget,
            })
        decisions[mode] = decide(mode, digests)

    # --- 矩阵（M5.4h；仅在显式 --time-limits 时执行）---
    matrix: dict[str, dict] = {"budgets": {}}
    stage_rows: list[dict] = []
    # 默认档也要产出逐阶段审计行（否则 summary.parquet 缺失）；但 `--modes off`
    # 的 run 不得偷偷跑 corrector-on 子进程 —— 只有显式 `--time-limits` 或
    # `--modes` 含 `on` 时才展开矩阵。
    matrix_budgets = budgets if (matrix_is_explicit or "on" in args.modes) else []
    for budget in matrix_budgets:
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
            stage_rows.extend(build_stage_rows(
                entry=entry, budget=budget, pid=entry["pid"],
                node_cap=None, wall_clock_disabled=False,
            ))
        matrix["budgets"][str(budget)] = {
            "distinct": len(set(budget_digests)),
            "processes": len(budget_digests),
            "steps": args.steps,
            "digests": budget_digests,
        }

    # --- 节点上限标定（M5.4h §4.5；**仅探针内 runtime 注入**，不写 planning/默认配置）---
    node_cap_matrix: dict[str, dict] = {}
    for cap in (args.node_caps or []):
        cap_digests: list[str] = []
        cap_reasons: list[str] = []
        cap_rows: list[dict] = []
        wall_clock_disabled = args.node_cap_arm == "alternative_semantics"
        for _ in range(args.runs):
            entry = run_in_subprocess(
                "on", steps=args.steps, budget=args.corrector_time_limit,
                record_stages=True, node_cap=cap, load=load,
                wall_clock_disabled=wall_clock_disabled,
            )
            cap_digests.append(entry["digest"])
            rows_for_cap = build_stage_rows(
                entry=entry, budget=args.corrector_time_limit, pid=entry["pid"],
                node_cap=cap, wall_clock_disabled=wall_clock_disabled,
            )
            stage_rows.extend(rows_for_cap)
            cap_rows.extend(rows_for_cap)
            cap_reasons.extend(r["corrector_reason"] for r in rows_for_cap)
        # 绑定判据必须来自**实测**求解事实，不得由「臂稳定」反推。
        binding = node_cap_binding(cap_rows, cap=cap)
        node_cap_matrix[str(cap)] = {
            "arm": args.node_cap_arm,
            "wall_clock_disabled_in_probe": bool(wall_clock_disabled),
            "distinct": len(set(cap_digests)),
            "processes": len(cap_digests),
            # 候选判定按**单个 cap** 做交集，故每个 cap 必须自带样本量。
            "steps": args.steps,
            "digests": cap_digests,
            "reasons": sorted({r for r in cap_reasons if r}),
            "cap_bound": binding["cap_bound"],
            "max_mip_node_count": binding["max_mip_node_count"],
            "observed_solves": binding["observed_solves"],
        }

    attribution = classify(matrix) if args.time_limits else None

    # --- release_gate（与 attribution **分离**；阶段放行的**唯一**依据）---
    # 每条观测都带**来源标签**；`evaluate_release_gate` 据此只放行生产默认证据。
    gate_observations: list[dict] = []
    if "on" in decisions:
        gate_observations.append({
            "source": "modes.on", "distinct": decisions["on"]["distinct"],
            "processes": args.runs, "steps": args.steps,
            "effective_corrector_time_limit_s": effective_budget,
            "corrector_time_limit_source": budget_source,
        })
    _key = str(PRODUCTION_CORRECTOR_TIME_LIMIT_S)
    if _key in matrix["budgets"]:
        _entry = matrix["budgets"][_key]
        gate_observations.append({
            "source": f"matrix.{_key}", "distinct": _entry["distinct"],
            "processes": _entry["processes"], "steps": _entry["steps"],
            "effective_corrector_time_limit_s": float(_key),
            # 显式 `--time-limits` 给出的档位是**诊断 override**，不得充当生产默认证据。
            "corrector_time_limit_source": (
                CORRECTOR_TIME_LIMIT_SOURCE_EXPLICIT_OVERRIDE if matrix_is_explicit
                else budget_source
            ),
        })
    # 平台确定性：corrector 关闭时本应逐位一致；不一致则整个测量平台不可信，同样不放行。
    mode_observations: list[dict] = []
    if "off" in decisions:
        mode_observations.append({
            "source": "modes.off", "distinct": decisions["off"]["distinct"],
            "processes": args.runs, "steps": args.steps,
            "effective_corrector_time_limit_s": None,
            "corrector_time_limit_source": CORRECTOR_TIME_LIMIT_SOURCE_DISABLED,
        })
    release_gate = evaluate_release_gate(
        gate_observations, mode_observations=mode_observations
    )
    # overall / manifest.status / 退出码共用**同一**推导，三者不得各自计算。
    phase = phase_status(release_gate, attribution=attribution)
    overall = phase["overall"]
    status = phase["status"]
    # modes 级汇总只作审计记录，**不参与**放行判定（attribution 更不能）。
    modes_overall = {**overall_conclusion(decisions), "decides_release": False}
    rows = base_rows + stage_rows

    # 命令账本：用 shlex.join 完整记录**全部有效参数**（M5.4c 的标准）
    _argv = ["python", "scripts/probe_corrector_repro.py",
             "--modes", *args.modes,
             "--runs", str(args.runs),
             "--steps", str(args.steps),
             "--load", args.load,
             "--base-dir", str(args.base_dir),
             "--run-id", run_id]
    if args.corrector_time_limit is not None:
        # 只有**显式**给出时才进账本：未给出时伪造该参数会让重放与实测不符。
        _argv += ["--corrector-time-limit", str(args.corrector_time_limit)]
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
        "solver": solver_evidence(budget=float(effective_budget)),
        "env_seed_kwargs": dict(ENV_SEED_KWARGS),
        "policy_seed": POLICY_SEED,
        "generator_seed": GENERATOR_SEED,
        "horizon": HORIZON,
        "steps": args.steps,
        "modes": list(args.modes),
        "runs_per_mode": args.runs,
        "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
        "effective_corrector_time_limit_s": float(effective_budget),
        "corrector_time_limit_source": budget_source,
        "corrector_time_limit_s": float(effective_budget),
        "time_limit_matrix_s": [float(b) for b in budgets],
        "node_caps": list(args.node_caps) if args.node_caps else [],
        "node_cap_default": DEFAULT_NODE_CAP,
        "load": load,
        "load_injected": bool(load["load_injected"]),
        "machine": machine,
        "budget_s": float(effective_budget),
        "wall_clock_keys_excluded": list(WALL_CLOCK_KEYS),
        "digest_fields": list(DIGEST_FIELDS) + list(DIGEST_INFO_FIELDS),
        "retry_selection": False,
    }
    report = {
        "probe": "m54h_budget_attribution",
        "trained": False,
        "steps": int(args.steps),
        "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
        "effective_corrector_time_limit_s": float(effective_budget),
        "corrector_time_limit_source": budget_source,
        "corrector_time_limit_s": float(effective_budget),
        "time_limit_matrix_s": [float(b) for b in budgets],
        "load": load,
        "load_injected": bool(load["load_injected"]),
        "machine": machine,
        "release_gate": release_gate,
        "statement": REPORT_STATEMENT,
        "claims": {"trained": False, "performance_evaluated": False, "convergence_claimed": False},
        "modes": decisions,
        "matrix": matrix,
        "node_cap_matrix": node_cap_matrix,
        "attribution": attribution,
        "overall": overall,
        "modes_overall": modes_overall,
        "status": status,
        "provenance_note": provenance_note(load),
    }

    # 生产默认预算不稳定 -> manifest 必须 failed，**即使归因成立**。
    # fail closed：blocked 之外的「未测量 / 样本量不足」同样是失败，必须归类。
    failure_classification = None
    if overall["blocked"]:
        failure_classification = (
            "corrector_off_nonreproducible"
            if "modes.off" in release_gate["unstable_sources"]
            else "corrector_nonreproducible"
        )
    elif status == "failed":
        failure_classification = "release_gate_evidence_insufficient"
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
        failure_classification=failure_classification,
    )
    if args.emit_provenance:
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 汇总表：不得只在任务卡写人工摘要。
    # 阶段状态与 report / manifest **共用** phase_status 的同一推导，不得各自计算。
    summary = {
        "probe": "m54h_budget_attribution",
        "matrix": matrix,
        "conclusion": overall["conclusion"],
        "status": status,
        "exit_code": phase["exit_code"],
        "overall": overall,
        "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
        "effective_corrector_time_limit_s": float(effective_budget),
        "corrector_time_limit_source": budget_source,
        "release_gate": release_gate,
        "attribution_conclusion": attribution["conclusion"] if attribution else None,
        "attribution_reason": attribution["reason"] if attribution else "（未请求矩阵）",
        "load": load,
        "load_injected": bool(load["load_injected"]),
        "machine": machine,
        "stage_field_count": len(stage_rows),
        "candidate_notes": candidate_notes(attribution, release_gate, node_cap_matrix),
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
    print(f"  release_gate.blocked={release_gate['blocked']} -> overall={overall['conclusion']} "
          f"(manifest={status}, exit={phase['exit_code']})")
    return phase["exit_code"]


def _git_revision() -> str:
    from runs.writer import git_revision

    return git_revision()


def _dependency_lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
