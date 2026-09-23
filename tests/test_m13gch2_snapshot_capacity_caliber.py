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


def test_group_work_capacity_matches_the_env_per_step_capacity():
    """口径必须与 env 的 `planned_capacity_vec`（action=1）一致。"""
    env = _env(delta_t_hours=0.5, max_task_load_per_server=0.8)
    env.reset(seed=0)
    snap = build_snapshot(env)

    full = np.zeros(env.action_dim, dtype=np.float64)
    full[: env.model.N] = 1.0
    _obs, _r, _term, _trunc, info = env.step(full)
    env_vec = np.asarray(info["planned_capacity_vec"], dtype=np.float64)

    assert np.allclose(np.asarray(snap.group_work_capacity, dtype=np.float64),
                       env_vec, rtol=0.0, atol=1e-9), (
        "action=1 时 snapshot.group_work_capacity 必须等于 env 的 planned_capacity_vec")


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

@pytest.mark.parametrize("delta,maxload", CASES)
def test_planner_step0_allocation_matches_the_env_planned_capacity(delta, maxload):
    """规划第 0 步逐组分配量必须等于 env.step(exec_action) 的 `planned_capacity_vec`。"""
    from contracts.models import DispatchProposal
    from planning.corrector import correct

    env = _env(delta_t_hours=delta, max_task_load_per_server=maxload)
    env.reset(seed=0)
    snapshot = build_snapshot(env)

    proposal = DispatchProposal(
        compute_actions=[1.0] * env.model.N, storage_action=0.0)
    correction = correct(snapshot, proposal, time_limit_s=0.25)

    exec_action = np.zeros(env.action_dim, dtype=np.float64)
    exec_action[: env.model.N] = np.asarray(
        correction.exec_compute_actions, dtype=np.float64)

    # 规划第 0 步的逐组分配量
    allocation = np.asarray(correction.allocation, dtype=np.float64)  # (n_task, n_group, H)
    planned_step0 = allocation[:, :, 0].sum(axis=0)

    _obs, _r, _term, _trunc, info = env.step(exec_action)
    env_vec = np.asarray(info["planned_capacity_vec"], dtype=np.float64)

    assert np.allclose(planned_step0, env_vec, rtol=1e-6, atol=1e-6), (
        "规划第 0 步分配量与 env 的 planned_capacity_vec 必须逐组一致；"
        f"最大差 {np.max(np.abs(planned_step0 - env_vec))!r}")
