"""M6-P1-F3-R2：**同一次决策必须使用同一份可见任务状态**。

缺陷（修复前）：`_activate_arrivals` 在 `step()` 内部、且在该步观测生成**之后**
才把 `arrival_time == t` 的任务置为 `waiting`。于是决策步 `t`：

- **观测**（`_get_obs` → `_get_task_pool_features(t)`，其 `arrived_tasks` 判据是
  `arrival_time <= t`）与 **corrector 快照**（`planning.snapshot_adapter`，
  判据是 `status != "not_arrived"`）都**看不到**当步到达任务；
- 而同一决策步的 **环境分配器**（`_execute_tasks_action_guided` 的 `active_tasks`，
  在 `_activate_arrivals(t)` 之后构造）**看得到**它们，并按
  `(priority 降序, deadline 升序, arrival 升序, id)` 让它们**抢先**取走容量。

本文件锁定修复后的不变量：**在每一个决策步上，观测、快照与环境分配器看到的是
同一份任务集**，且该任务集**只含 `arrival_time <= current_step` 的任务**
（当步已到达任务本就可观测，不构成未来信息泄漏）。

只读断言：本文件不修改环境、快照或 corrector 的任何语义。
"""

from __future__ import annotations

import numpy as np
import pytest

from planning.snapshot_adapter import build_snapshot
from scripts.calibrate_training_config import build_env_for_origin, reference_action

# 报告缺陷的 origin（F3-R1/F3-R2 实测：origin 48 / step 3 落差 10.5179 → 7.0000）。
ORIGINS: tuple[int, ...] = (48, 4848)

_ALLOCATABLE_STATUSES = ("waiting", "running", "paused")


def _visible_ids(env) -> set[str]:
    """**该决策步可见**的任务：`arrival_time <= current_step` 且未终结。

    与环境 `_get_task_pool_features` 的 `arrived_tasks` 判据同一定义。
    """
    now = int(env.current_step)
    return {
        str(task.task_id)
        for task in env.tasks
        if int(task.arrival_time) <= now and str(task.status) not in ("finished", "failed")
    }


def _allocatable_ids(env) -> set[str]:
    """**环境分配器**在本次决策的 `env.step()` 里会拿到的任务集。

    与 `_execute_tasks_action_guided` 的 `active_tasks` 判据同一定义。
    """
    return {
        str(task.task_id)
        for task in env.tasks
        if str(task.status) in _ALLOCATABLE_STATUSES and float(task.remaining_work) > 1e-6
    }


def _snapshot_ids(env) -> set[str]:
    """**corrector 快照**看到的任务集。"""
    return {str(task.task_id) for task in build_snapshot(env).tasks}


@pytest.mark.parametrize("origin", ORIGINS)
def test_decision_step_sees_one_task_state(origin: int) -> None:
    """每个决策步：规划器看得见环境分配器将使用的**全部**任务，且不含未来任务。

    三条断言（`snapshot.tasks` 按 M4.1a 契约还携带 `remaining_work == 0` 的已终结
    任务，它们对 `allocate_tasks` 是惰性的——`demand <= 1e-9` 直接跳过——故只要求
    单侧包含，并显式校验这些「多余项」确实是惰性的）：

    1. `allocatable == 当步已到达且未终结`：环境自身两条口径（分配器
       `active_tasks` 与观测 `arrived_tasks`）对同一次决策一致；
    2. `snapshot ⊇ allocatable`：规划器看到分配器将使用的全部任务——**这是本卡
       修复的缺陷**：修复前当步到达任务在分配器里、却不在快照里；
    3. `snapshot` 中每个任务的 `arrival_time <= current_step`，多余项一律
       `status == "finished"` 且 `remaining_work == 0`。
    """
    env, _injection = build_env_for_origin(origin)
    action = np.asarray(reference_action(1.0, env.action_dim, env.model.N), dtype=np.float32)
    env.reset(seed=0)

    arrivals_seen = 0
    for step in range(int(env.horizon)):
        assert int(env.current_step) == step
        visible, allocatable, snapshot = (
            _visible_ids(env), _allocatable_ids(env), _snapshot_ids(env),
        )
        by_id = {str(task.task_id): task for task in env.tasks}
        arrivals_seen += sum(1 for task in env.tasks if int(task.arrival_time) == step)
        context = (
            f"origin={origin} step={step}：观测/快照/分配器看到的任务集不一致\n"
            f"  当步已到达未终结 = {sorted(visible)}\n"
            f"  分配器 active_tasks = {sorted(allocatable)}\n"
            f"  快照 snapshot.tasks  = {sorted(snapshot)}\n"
            f"  分配器有而快照缺失 = {sorted(allocatable - snapshot)}\n"
            f"  已到达未终结而分配器缺失 = {sorted(visible - allocatable)}"
        )
        assert allocatable == visible, context
        assert allocatable <= snapshot, context

        future = [
            tid for tid in snapshot if int(by_id[tid].arrival_time) > step
        ]
        assert future == [], (
            f"origin={origin} step={step}：快照含未到达任务 {future}"
        )
        not_inert = [
            tid for tid in sorted(snapshot - allocatable)
            if not (
                str(by_id[tid].status) == "finished"
                and float(by_id[tid].remaining_work) == pytest.approx(0.0)
            )
        ]
        assert not_inert == [], (
            f"origin={origin} step={step}：快照含既不在分配器集合里、又非惰性的任务 "
            f"{not_inert}"
        )

        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    # 该 episode 确实存在到达事件；否则上面的不变量是空断言。
    assert arrivals_seen > 0


@pytest.mark.leakage
@pytest.mark.parametrize("origin", ORIGINS)
def test_decision_step_exposes_no_future_task(origin: int) -> None:
    """红线：决策步 `t` 不得公开 `arrival_time > t` 的任务（避免未来信息泄漏）。"""
    env, _injection = build_env_for_origin(origin)
    action = np.asarray(reference_action(1.0, env.action_dim, env.model.N), dtype=np.float32)
    env.reset(seed=0)

    for step in range(int(env.horizon)):
        now = int(env.current_step)
        leaked_env = [
            str(task.task_id)
            for task in env.tasks
            if int(task.arrival_time) > now and str(task.status) != "not_arrived"
        ]
        leaked_snapshot = [
            str(task.task_id)
            for task in env.tasks
            if int(task.arrival_time) > now and str(task.task_id) in _snapshot_ids(env)
        ]
        assert leaked_env == [], f"origin={origin} step={step} 环境提前激活了未来任务 {leaked_env}"
        assert leaked_snapshot == [], (
            f"origin={origin} step={step} 快照泄漏了未来任务 {leaked_snapshot}"
        )

        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break


@pytest.mark.parametrize("origin", ORIGINS)
def test_queue_amount_matches_visible_backlog(origin: int) -> None:
    """队列量 `Q_t` 必须等于**当步可见**未完成任务的工作量之和。

    `Q_t` 是观测的队列通道（`Q_norm = Q_t / queue_ref`）；若它与任务池特征
    （`arrival_time <= t`）口径不同，则同一次决策的两条通道自相矛盾。
    """
    env, _injection = build_env_for_origin(origin)
    action = np.asarray(reference_action(1.0, env.action_dim, env.model.N), dtype=np.float32)
    env.reset(seed=0)

    for step in range(int(env.horizon)):
        assert float(env.Q_t) == pytest.approx(env._compute_backlog_work()), (
            f"origin={origin} step={step}：Q_t={env.Q_t} 与可见积压 "
            f"{env._compute_backlog_work()} 不一致"
        )
        _obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
