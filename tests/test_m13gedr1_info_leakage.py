"""M1.3g-e-d-R1：收口 formal info 中仅剩的两处未来工作量出口。

`g-e-d` 已收口 observation / reward / `total_task_count`；本卡只补 **info** 的：

1. reset `_task_scale_info()` 的 `effective_total_workload` / `average_task_workload`
   —— 由 `_total_available_work()`（`arrival_time < horizon`，**整段**）与
   `len(self.tasks)`（**含未到达**）计算；
2. step info 的 `completion_rate` —— 分母同为整段 `_total_available_work()`。

**方法**：两个状态**完全相同**的 formal env，只在其中一个**复制**一个
`arrival_time` **晚于被测时点且仍落在 episode 内**的 Task。每条用例先证明
未来任务确实被增加，再断言两个出口不变，并对照当步 `completed_work` 与
**reward 相同**（差异只允许出现在 info 出口）。

**改前缺陷**：两处用例都必须为红。
"""

import copy
import importlib
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

SPLIT = "train"
# 有代表性的日照 episode（PV / 风电非零）。
START = "2024-04-16T10:00:00+08:00"
DELTA_HOURS = 0.5
BACKLOG_PROFILE = "initial_backlog"


def _env_cls():
    return importlib.import_module("envs.idc_price_env").IDCPriceEnv20D


def _upstream_present() -> bool:
    return all(p.exists() for p in (
        REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
        REPO_ROOT / "data/manifest/singapore_2024_half_hour.json",
        REPO_ROOT / "data/manifest/singapore_2024_splits.json",
        REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json",
        REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json",
        REPO_ROOT / "configs/frozen_refs/refs_v4.json",
        REPO_ROOT / "data/manifest/formal_splits_v5/train.json",
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机")


def _make(horizon: int = 8, cutoff: int = 4, seed: int = 7):
    """一个 formal env（同 split / origin / seed / 冻结参数）。"""
    ei = importlib.import_module("scenario.env_injection")
    inj = ei.build_verified_formal_env_injection(
        SPLIT, start=START, horizon=horizon, forecast_cutoff=cutoff)
    return _env_cls()(
        horizon=horizon, task_seed=seed, server_seed=0, forecast_seed=seed,
        delta_t_hours=DELTA_HOURS, formal_injection=inj,
    )


def _pair(horizon: int = 8, cutoff: int = 4, seed: int = 7):
    """两个配置完全相同的 formal env。"""
    return _make(horizon, cutoff, seed), _make(horizon, cutoff, seed)


def _action(env, throttle: float = 0.4) -> np.ndarray:
    action = np.zeros(env.action_dim, dtype=np.float64)
    action[:env.model.N] = throttle
    return action


def _duplicate_future_task(env, *, at_step: int):
    """复制一个 `at_step < arrival_time < horizon` 的 Task（**未来**工作量）。

    返回 `(复制前任务数, 复制后任务数, 被复制任务的 arrival_time)`。
    """
    before = len(env.tasks)
    horizon = int(env.horizon)
    future = [
        t for t in env.tasks
        if at_step < int(t.arrival_time) < horizon
    ]
    assert future, "本用例要求存在晚于被测时点、且仍落在 episode 内的任务"
    picked = future[0]
    env.tasks = env.tasks + [copy.deepcopy(picked)]
    return before, len(env.tasks), int(picked.arrival_time)


# =============================================================================
# R1-1. reset `_task_scale_info()` 的两个字段
# =============================================================================

@needs_assets
@pytest.mark.leakage
def test_reset_task_scale_info_excludes_future_workload():
    """formal reset 的 `effective_total_workload` / `average_task_workload`
    不得随**未来**任务工作量变化。"""
    ref, probe = _pair(horizon=8, cutoff=4)
    _obs_ref, info_ref = ref.reset(seed=7)
    _obs_probe, info_probe = probe.reset(seed=7)

    # 前置：两个 env 的 reset 出口必须**完全一致**，否则对照无意义
    assert info_probe["effective_total_workload"] == pytest.approx(
        info_ref["effective_total_workload"])
    assert info_probe["average_task_workload"] == pytest.approx(
        info_ref["average_task_workload"])

    before, after, arrival = _duplicate_future_task(probe, at_step=0)
    # --- 非空洞性：未来任务必须真的被增加，且确实落在 episode 内 ---
    assert after == before + 1, "mutation 未生效：未来任务未被增加"
    assert 0 < arrival < int(probe.horizon)
    assert any(int(t.arrival_time) == arrival for t in probe.tasks)

    now = probe._task_scale_info()
    assert now["effective_total_workload"] == pytest.approx(
        info_ref["effective_total_workload"]), (
        "formal reset 的 effective_total_workload 不得含未来任务工作量")
    assert now["average_task_workload"] == pytest.approx(
        info_ref["average_task_workload"]), (
        "formal reset 的 average_task_workload 不得含未来任务工作量")


# =============================================================================
# R1-2. step info 的 `completion_rate`
# =============================================================================

@needs_assets
@pytest.mark.leakage
def test_step_completion_rate_excludes_future_workload():
    """formal 同一步的 `completion_rate` 不得随未来任务工作量变化；
    当步 `completed_work` 与 **reward** 必须相同（差异只允许在 info 出口）。"""
    ref, probe = _pair(horizon=8, cutoff=4)
    ref.reset(seed=7)
    probe.reset(seed=7)
    action = _action(ref)

    for _ in range(2):
        ref.step(action)
        probe.step(action)

    # 前置：走到同一步后两侧状态一致
    assert ref.current_step == probe.current_step
    assert np.allclose(ref._get_obs(), probe._get_obs())

    before, after, arrival = _duplicate_future_task(probe, at_step=int(probe.current_step))
    # --- 非空洞性 ---
    assert after == before + 1, "mutation 未生效：未来任务未被增加"
    assert int(probe.current_step) < arrival < int(probe.horizon)

    _o_ref, reward_ref, _t_ref, _tr_ref, info_ref = ref.step(action)
    _o_probe, reward_probe, _t_probe, _tr_probe, info_probe = probe.step(action)

    # 对照：当步完成量与 reward 必须相同（未来任务未到达，不应影响执行或奖励）
    assert info_probe["completed_work"] == pytest.approx(info_ref["completed_work"])
    assert reward_probe == pytest.approx(reward_ref), \
        "g-e-d 已收口 reward；此处必须保持一致"

    assert info_probe["completion_rate"] == pytest.approx(
        info_ref["completion_rate"]), (
        "formal 的 completion_rate 分母不得含未来任务工作量")


# =============================================================================
# R1-3. 反向控制 / 边界
# =============================================================================

@needs_assets
@pytest.mark.leakage
def test_reset_task_scale_info_still_counts_all_arrived_work():
    """**反向控制**：已到达工作量变了，formal 的 effective_total_workload **必变**。"""
    env = _make(horizon=8, cutoff=4)
    _obs, info = env.reset(seed=7)
    base = info["effective_total_workload"]
    assert base > 0.0

    arrived = [t for t in env.tasks if int(t.arrival_time) <= 0]
    assert arrived, "reset 后必须已有到达任务"
    env.tasks = env.tasks + [copy.deepcopy(arrived[0])]

    assert env._task_scale_info()["effective_total_workload"] != pytest.approx(base), (
        "已到达工作量增加后该字段必须变化 —— 否则断言过强（把真实工作量也屏蔽了）")


@needs_assets
def test_legacy_task_scale_info_and_completion_rate_are_unchanged():
    """legacy 语义逐字不变：仍按**整段**任务计算两个出口。"""
    env = _env_cls()(horizon=8, task_seed=7, server_seed=0)
    _obs, info = env.reset(seed=7)

    total = env._total_available_work()
    count = len(env.tasks)
    assert info["effective_total_workload"] == pytest.approx(total)
    assert info["average_task_workload"] == pytest.approx(total / count)

    _o, _r, _t, _tr, step_info = env.step(_action(env, 0.5))
    assert step_info["completion_rate"] == pytest.approx(
        env.total_completed_work / env._total_available_work())


@needs_assets
def test_formal_terminal_completion_rate_matches_the_whole_episode():
    """episode 终点：formal 的 `completion_rate` 分母可与**整段**总量一致。

    走到终点时所有任务都已到达，故「截至该步已到达」== 「整段」。
    """
    env = _make(horizon=8, cutoff=4)
    env.reset(seed=7)
    action = _action(env, 0.4)
    info = None
    for _ in range(env.horizon):
        _o, _r, term, _tr, info = env.step(action)
        if term:
            break
    assert term
    assert info["completion_rate"] == pytest.approx(
        env.total_completed_work / env._total_available_work())
    # 非空洞性：整段总量必须大于 0
    assert env._total_available_work() > 0.0
