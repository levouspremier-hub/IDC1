"""M4.4a 原始动作最小偏移投影器（H 步 MIP 两阶段）。

**不再**调用旧单步求解器（legacy `solver` 模块）；本模块只把 raw proposal 投影到
H 步 MIP 的物理可行域，并把**第 0 步**映射为 exec action。

失败语义（保真）：
- `none` / `deadline_shortfall`：MIP 最优，可执行第 0 步候选（期限缺口仅为业务风险）；
- `timeout` / `base_shortage` / `solver_failure`：**零动作回退**，不执行候选解；
- `proposal_invalid`：raw proposal 维度/范围非法，**零动作回退**并记录原因。

红线：不修改 raw action；不把 LP 解作为执行候选；功率为规划近似，最终执行约束仍由
环境 M3 物理链承担。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from contracts.models import DispatchProposal, SystemSnapshot
from planning.model import (
    FAILURE_BASE_SHORTAGE,
    FAILURE_DEADLINE_SHORTFALL,
    FAILURE_NONE,
    FAILURE_PROPOSAL_INVALID,
    FAILURE_SOLVER_FAILURE,
    FAILURE_TIMEOUT,
    solve_time_indexed_mip_raw_projection,
)

# 可执行（第 0 步候选可进入环境）的失败类别
_EXECUTABLE = {FAILURE_NONE, FAILURE_DEADLINE_SHORTFALL}

# --- M5.4i：corrector 生产默认预算的**唯一**来源 -----------------------------
# 依据：0.05 s 在受控负载（hogs4/hogs8）下跨进程 digest 可分叉，release gate 因此
# blocked（M5.4h/M5.4h1/M5.4h2）；0.25 s 在三种负载口径下均无 observed `time_limit`
# 且 digest 一致。它是**本机候选值**，不构成跨机器保证。
#
# **四个入口（train / smoke / probe_rollout_deterministic / probe_corrector_repro）
# 必须导入本常量，不得各自硬编码。** 本常量是数值，不是默认参数：
# `correct()` 仍要求调用方**显式**传入 `time_limit_s`。
PRODUCTION_CORRECTOR_TIME_LIMIT_S = 0.25

# 预算来源：描述「预算从哪来」，**不是**「数值等于多少」——显式给出 0.25 也记 override。
CORRECTOR_TIME_LIMIT_SOURCE_PRODUCTION_DEFAULT = "production_default"
CORRECTOR_TIME_LIMIT_SOURCE_EXPLICIT_OVERRIDE = "explicit_override"
CORRECTOR_TIME_LIMIT_SOURCE_DISABLED = "disabled"


def resolve_corrector_budget(
    requested_time_limit_s: float | None, *, enabled: bool = True
) -> tuple[float | None, str]:
    """把「调用方是否显式给出预算」解析为 `(有效预算, 来源)`。

    这是四个入口共用的**同一份**解析逻辑，避免各处再次分叉出不同默认值。
    `enabled=False`（corrector 关闭）时有效预算为 `None`。
    """
    if not enabled:
        return None, CORRECTOR_TIME_LIMIT_SOURCE_DISABLED
    if requested_time_limit_s is None:
        return (
            PRODUCTION_CORRECTOR_TIME_LIMIT_S,
            CORRECTOR_TIME_LIMIT_SOURCE_PRODUCTION_DEFAULT,
        )
    return float(requested_time_limit_s), CORRECTOR_TIME_LIMIT_SOURCE_EXPLICIT_OVERRIDE


class FailureClass(StrEnum):
    NONE = "none"
    TIMEOUT = "timeout"
    BASE_SHORTAGE = "base_shortage"
    SOLVER_FAILURE = "solver_failure"
    PROPOSAL_INVALID = "proposal_invalid"
    DEADLINE_SHORTFALL = "deadline_shortfall"


_FAILURE_MAP = {
    FAILURE_NONE: FailureClass.NONE,
    FAILURE_TIMEOUT: FailureClass.TIMEOUT,
    FAILURE_BASE_SHORTAGE: FailureClass.BASE_SHORTAGE,
    FAILURE_SOLVER_FAILURE: FailureClass.SOLVER_FAILURE,
    FAILURE_PROPOSAL_INVALID: FailureClass.PROPOSAL_INVALID,
    FAILURE_DEADLINE_SHORTFALL: FailureClass.DEADLINE_SHORTFALL,
}


@dataclass
class Correction:
    exec_compute_actions: list[float]
    exec_storage_action: float
    business_gap: float
    failure: FailureClass
    reviewed: bool
    reason: str = ""
    deadline_shortfall_work: float = 0.0
    planner_backend: str = "mip"
    stage_a_status: str = ""
    stage_b_status: str = ""
    stage_a_solve_time_s: float = 0.0
    stage_b_solve_time_s: float = 0.0
    stage_a_objective: float = 0.0
    stage_b_objective: float = 0.0
    projection_offset: float = 0.0
    solve_time_s: float = 0.0
    audit: dict = field(default_factory=dict)


def _zero_action(
    snapshot: SystemSnapshot, failure: FailureClass, reason: str, stage: dict | None = None
) -> Correction:
    """已验证的物理边界动作：零计算、零储能，缺口 = 全部剩余工作。"""
    stage = stage or {}
    remaining = float(sum(task.remaining_work for task in snapshot.tasks))
    return Correction(
        exec_compute_actions=[0.0] * len(snapshot.group_work_capacity),
        exec_storage_action=0.0,
        business_gap=remaining,
        failure=failure,
        reviewed=True,
        reason=reason,
        deadline_shortfall_work=0.0,
        planner_backend=str(stage.get("planner_backend", "mip")),
        stage_a_status=str(stage.get("stage_a_status", "")),
        stage_b_status=str(stage.get("stage_b_status", "not_run")),
        stage_a_solve_time_s=float(stage.get("stage_a_solve_time_s", 0.0)),
        stage_b_solve_time_s=float(stage.get("stage_b_solve_time_s", 0.0)),
        stage_a_objective=float(stage.get("stage_a_objective", 0.0)),
        stage_b_objective=float(stage.get("stage_b_objective", 0.0)),
        projection_offset=float(stage.get("projection_offset", 0.0)),
        solve_time_s=float(stage.get("solve_time_s", 0.0)),
    )


def correct(
    snapshot: SystemSnapshot,
    proposal: DispatchProposal,
    *,
    time_limit_s: float,
) -> Correction:
    """把 raw proposal 投影到 H 步 MIP 可行域，并返回第 0 步 exec action。

    `time_limit_s` 是**本次调用的全局预算**（阶段 A/B 共享同一 deadline），
    必须为显式正数——不得以隐式默认值掩盖运行时预算决策。
    """
    import time

    if not (isinstance(time_limit_s, (int, float)) and time_limit_s > 0.0):
        raise ValueError(f"time_limit_s 必须为显式正数预算，实际 {time_limit_s!r}")

    n_group = len(snapshot.group_work_capacity)
    if len(proposal.compute_actions) != n_group:
        return _zero_action(
            snapshot, FailureClass.PROPOSAL_INVALID,
            f"compute_actions 维度 {len(proposal.compute_actions)} != 组数 {n_group}",
        )
    if any((x < 0.0 or x > 1.0) for x in proposal.compute_actions):
        return _zero_action(
            snapshot, FailureClass.PROPOSAL_INVALID, "compute_actions 超出 [0,1]"
        )
    if not (-1.0 <= float(proposal.storage_action) <= 1.0):
        return _zero_action(
            snapshot, FailureClass.PROPOSAL_INVALID,
            f"storage_action {proposal.storage_action} 超出 [-1,1]",
        )

    t0 = time.perf_counter()
    res = solve_time_indexed_mip_raw_projection(
        snapshot, proposal, time_limit_s=float(time_limit_s)
    )
    solve_time = time.perf_counter() - t0

    failure = _FAILURE_MAP.get(res.failure_class, FailureClass.SOLVER_FAILURE)
    audit = dict(
        planner_backend=res.backend,
        stage_a_status=res.stage_a_status,
        stage_b_status=res.stage_b_status,
        stage_a_solve_time_s=res.stage_a_solve_time_s,
        stage_b_solve_time_s=res.stage_b_solve_time_s,
        stage_a_objective=res.stage_a_objective,
        stage_b_objective=res.stage_b_objective,
        projection_offset=res.projection_offset,
        solve_time_s=solve_time,
    )

    if failure not in _EXECUTABLE:
        return _zero_action(
            snapshot, failure, f"{res.solver_status}: {res.failure_class}", audit
        )

    return Correction(
        exec_compute_actions=list(res.exec_compute_actions),
        exec_storage_action=float(res.exec_storage_action),
        business_gap=float(res.business_gap_work),
        failure=failure,
        reviewed=True,
        reason=res.solver_status,
        deadline_shortfall_work=float(res.deadline_shortfall_work),
        planner_backend=res.backend,
        stage_a_status=res.stage_a_status,
        stage_b_status=res.stage_b_status,
        stage_a_solve_time_s=float(res.stage_a_solve_time_s),
        stage_b_solve_time_s=float(res.stage_b_solve_time_s),
        stage_a_objective=float(res.stage_a_objective),
        stage_b_objective=float(res.stage_b_objective),
        projection_offset=float(res.projection_offset),
        solve_time_s=float(solve_time),
    )
