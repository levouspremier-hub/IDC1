"""M1.3g-f-c-h2：规划快照的**每步能力口径**。

**改前缺陷（本文件对应先红）**：`planning/snapshot_adapter.py` 把
`C_server` 原值（**work/hour**）直接交给规划器：

```python
group_work_capacity = [float(c) for c in c_server]        # work/hour，未折算 work/step
coeff = (p_max_kw - p_idle_kw) / np.maximum(c_server, 1e-6)   # 分母缺 delta_t_hours
```

而规划器的映射是 `exec_compute[g] = used / cap[g]`（`planning/model.py:1123`），
环境的每步能力是 `action × max_task_load_per_server × C_server × delta_t_hours`
（`envs/idc_price_env.py:778-784`）⇒ **两侧口径不一致**。

**目标**：

```text
group_work_capacity[g]           = C_server[g] × max_task_load_per_server × delta_t_hours
group_power_coeff_kw_per_work[g] = (P_max[g] − P_idle[g]) / (C_server[g] × delta_t_hours)
```

覆盖 `delta_t_hours ∈ {0.5, 1.0}` × `max_task_load_per_server ∈ {0.8, 1.0}`。
"""

import importlib
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

HORIZON = 24
CUTOFF = 4
FLOAT_TOL = 1e-9


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


def _env_cls():
    return importlib.import_module("envs.idc_price_env").IDCPriceEnv20D


def build_snapshot(env):
    return importlib.import_module("planning.snapshot_adapter").build_snapshot(env)


def _env(*, delta_t_hours: float, max_task_load_per_server: float):
    """构造一个可用环境（**不改 env 语义**，只设构造参数）。"""
    return _env_cls()(
        horizon=HORIZON,
        task_seed=0,
        server_seed=0,
        forecast_seed=300000,
        delta_t_hours=delta_t_hours,
        max_task_load_per_server=max_task_load_per_server,
    )


def _expected_capacity(env) -> np.ndarray:
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    return c_server * float(env.max_task_load_per_server) * float(env.delta_t_hours)


def _expected_coeff(env) -> np.ndarray:
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    p_max_kw = np.asarray(env.model.P_max, dtype=np.float64) / 1000.0
    p_idle_kw = np.asarray(env.model.P_idle, dtype=np.float64) / 1000.0
    return (p_max_kw - p_idle_kw) / np.maximum(
        c_server * float(env.delta_t_hours), 1e-6)


CASES = (
    (0.5, 0.8),
    (0.5, 1.0),
    (1.0, 0.8),
    (1.0, 1.0),
)


# =============================================================================
# 1. group_work_capacity 必须是 **work/step**
# =============================================================================

@pytest.mark.parametrize("delta,maxload", CASES)
def test_group_work_capacity_is_work_per_step(delta, maxload):
    env = _env(delta_t_hours=delta, max_task_load_per_server=maxload)
    snap = build_snapshot(env)

    got = np.asarray(snap.group_work_capacity, dtype=np.float64)
    want = _expected_capacity(env)
    assert got.shape == want.shape
    assert np.allclose(got, want, rtol=0.0, atol=FLOAT_TOL), (
        f"group_work_capacity 必须是 work/step = C_server × {maxload} × {delta}；"
        f"实际首项 {got[0]!r}，期望 {want[0]!r}")

    # **非空洞性**：与「直接用 C_server 原值」必须不同（否则本用例测不出缺陷）
    raw = np.asarray(env.model.C_server, dtype=np.float64)
    if abs(maxload * delta - 1.0) > 1e-12:
        assert not np.allclose(got, raw, rtol=1e-9, atol=0.0), \
            "必须与未折算的 C_server 原值不同"


def test_group_work_capacity_is_the_upper_bound_of_the_env_planned_capacity():
    """`group_work_capacity` 是**上界**：env 的 `planned_capacity_vec` 不得超过它。

    ⚠️ 不得断言两者**相等** —— env 的 `planned_capacity_vec` 是经
    物理可行性投影后的**实际计划**（受可用任务量等约束），而 snapshot 的
    `group_work_capacity` 是 raw action=1 对应的**能力上界**。
    """
    env = _env(delta_t_hours=1.0, max_task_load_per_server=0.8)
    env.reset(seed=0)
    snap = build_snapshot(env)

    full = np.zeros(env.action_dim, dtype=np.float64)
    full[: env.model.N] = 1.0
    _obs, _r, _term, _trunc, info = env.step(full)
    env_vec = np.asarray(info["planned_capacity_vec"], dtype=np.float64)
    cap = np.asarray(snap.group_work_capacity, dtype=np.float64)

    # 明确浮点容差：能力量级约 20，实测超出约 3.6e-7（求解/线性化噪声）
    assert np.all(env_vec <= cap * (1.0 + 1e-9) + 1e-6), (
        "env 的实际计划不得超出 snapshot 的每步能力："
        f"超出 {np.max(env_vec - cap)!r}")


# =============================================================================
# 2. group_power_coeff_kw_per_work 的分母
# =============================================================================

@pytest.mark.parametrize("delta,maxload", CASES)
def test_group_power_coeff_divides_by_capacity_per_step(delta, maxload):
    env = _env(delta_t_hours=delta, max_task_load_per_server=maxload)
    snap = build_snapshot(env)

    got = np.asarray(snap.group_power_coeff_kw_per_work, dtype=np.float64)
    want = _expected_coeff(env)
    assert np.allclose(got, want, rtol=0.0, atol=1e-12), (
        f"coeff 分母必须是 C_server × {delta}（**不**乘 max_task_load）；"
        f"实际首项 {got[0]!r}，期望 {want[0]!r}")


@pytest.mark.parametrize("delta,maxload", CASES)
def test_coeff_times_capacity_is_the_group_power_swing(delta, maxload):
    """自洽性：`coeff × cap` == raw action=1 时的功率摆幅。"""
    env = _env(delta_t_hours=delta, max_task_load_per_server=maxload)
    snap = build_snapshot(env)

    cap = np.asarray(snap.group_work_capacity, dtype=np.float64)
    coeff = np.asarray(snap.group_power_coeff_kw_per_work, dtype=np.float64)
    p_max_kw = np.asarray(env.model.P_max, dtype=np.float64) / 1000.0
    p_idle_kw = np.asarray(env.model.P_idle, dtype=np.float64) / 1000.0

    assert np.allclose(coeff * cap, maxload * (p_max_kw - p_idle_kw),
                       rtol=1e-9, atol=1e-12), (
        "coeff × cap 必须等于 max_task_load_per_server × (P_max − P_idle)")


# =============================================================================
# 3. 端到端映射：规划第 0 步分配量 vs env.step(exec_action)
# =============================================================================

@pytest.mark.parametrize("delta,maxload", ((1.0, 0.8), (1.0, 1.0)))
def test_planner_step0_allocation_matches_the_env_planned_capacity(delta, maxload):
    """**legacy**（`formal=False`，`delta_t_hours=1.0`）上的端到端映射。

    用仓库既有的模型层入口 `solve_time_indexed_mip_raw_projection`
    （`correct()` 的公开结果不暴露 `allocation`，见 `tests/test_m54g_*` 的同一做法）。

    ⚠️ 只取 `delta=1.0`：**非 formal 环境不乘 `delta_hours`**，故把 `delta=0.5`
    的非 formal env 与 formal 口径相比属**类别错误**；delta 因子由
    `test_formal_env_step0_allocation_matches_the_env_planned_capacity` 覆盖。
    """
    from contracts.models import DispatchProposal
    from planning import model as planning_model

    env = _env(delta_t_hours=delta, max_task_load_per_server=maxload)
    env.reset(seed=0)
    snapshot = build_snapshot(env)

    proposal = DispatchProposal(
        compute_actions=[1.0] * env.model.N, storage_action=0.0)
    result = planning_model.solve_time_indexed_mip_raw_projection(
        snapshot, proposal, time_limit_s=0.25)

    exec_action = np.zeros(env.action_dim, dtype=np.float64)
    exec_action[: env.model.N] = np.asarray(
        result.exec_compute_actions, dtype=np.float64)

    # 规划第 0 步的逐组分配量（work/step）
    allocation = np.asarray(result.allocation, dtype=np.float64)  # (n_task, n_group, H)
    planned_step0 = allocation[:, :, 0].sum(axis=0)

    _obs, _r, _term, _trunc, info = env.step(exec_action)
    env_vec = np.asarray(info["planned_capacity_vec"], dtype=np.float64)

    assert np.allclose(planned_step0, env_vec, rtol=1e-6, atol=1e-6), (
        "规划第 0 步分配量与 env 的 planned_capacity_vec 必须逐组一致；"
        f"最大差 {np.max(np.abs(planned_step0 - env_vec))!r}")


@pytest.mark.skipif(not _upstream_present(), reason="真实冻结上游资产不在本机")
def test_formal_env_step0_allocation_matches_the_env_planned_capacity():
    """**formal 链**（会按 `delta_t_hours` 折算）上的端到端映射。

    ⚠️ 非 formal 环境**不**乘 `delta_t_hours`（`envs/idc_price_env.py:783` 仅在
    `self.formal` 时折算），故上面那组参数化用例只真正检验了
    `max_task_load_per_server`；**delta 因子必须由本用例覆盖**。
    """
    from contracts.models import DispatchProposal
    from planning import model as planning_model

    inj = importlib.import_module("scenario.env_injection") \
        .build_verified_formal_env_injection(
            "train", start="2024-01-02T00:00:00+08:00", horizon=HORIZON,
            forecast_cutoff=CUTOFF)
    env = _env_cls()(horizon=HORIZON, forecast_cutoff=CUTOFF, task_seed=0,
                     server_seed=0, forecast_seed=300000, delta_t_hours=0.5,
                     formal_injection=inj)
    assert env.formal is True and env.delta_t_hours == 0.5
    env.reset(seed=0)
    snapshot = build_snapshot(env)

    cap = np.asarray(snapshot.group_work_capacity, dtype=np.float64)
    expect_cap = (np.asarray(env.model.C_server, dtype=np.float64)
                  * float(env.max_task_load_per_server) * float(env.delta_t_hours))
    assert np.allclose(cap, expect_cap, rtol=0.0, atol=FLOAT_TOL), \
        "formal 链的 group_work_capacity 必须是 work/step"

    result = planning_model.solve_time_indexed_mip_raw_projection(
        snapshot, DispatchProposal(compute_actions=[1.0] * env.model.N,
                                   storage_action=0.0), time_limit_s=0.25)
    exec_action = np.zeros(env.action_dim, dtype=np.float64)
    exec_action[: env.model.N] = np.asarray(result.exec_compute_actions,
                                            dtype=np.float64)
    planned_step0 = np.asarray(result.allocation, dtype=np.float64)[:, :, 0].sum(axis=0)

    _obs, _r, _term, _trunc, info = env.step(exec_action)
    env_vec = np.asarray(info["planned_capacity_vec"], dtype=np.float64)

    assert np.allclose(planned_step0, env_vec, rtol=1e-6, atol=1e-6), (
        "formal 链上规划第 0 步分配量必须逐组等于 env 的 planned_capacity_vec；"
        f"最大差 {np.max(np.abs(planned_step0 - env_vec))!r}")
