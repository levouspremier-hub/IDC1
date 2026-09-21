"""M1.3g-e-d：B6 formal env 跨层守恒、泄漏与重放回归。

四组：

1. **原始 aggregate 守恒** —— mapper 整数账本 → env Task → 执行 → 终止结算，
   是同**一套**工作量；initial backlog 单列，不得混入 arrival 总量。
2. **半小时执行量与物理负载** —— planned capacity 与负载换算都必须用 0.5 h 单位。
3. **因果性** —— formal 决策输入不得含未来任务数（observation / reward / info）。
4. **重放** —— mapper stream 确定性、reset 重放、独立 env 动作重放。

**改前缺陷（本文件对应先红，逐条见 docs/task_cards/M1.3g.md §bb.1）**：

- `_loads_from_group_completion` 除以 `C_server`（work/hour）而非
  `C_server × delta_t_hours`，formal 下任务负载被低估 **2×**；
- `_get_task_pool_features` / `step()` 的 7 处归一化分母与 reset `info` 的
  `total_task_count` 使用 `len(self.tasks)`，**含未到达任务** → 未来信息泄漏。

**改前已绿**（回归守卫，不得为了「先红」而故意失败）：逐槽账本恒等式、
逐步守恒、terminal leftover、可见窗口内 forecast 的**必变**断言。
"""

import copy
import importlib
import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

STARTS = {
    "train": "2024-01-02T00:00:00+08:00",
    "validation": "2024-08-01T00:00:00+08:00",
    "test": "2024-10-01T00:00:00+08:00",
}
SPLITS = ("train", "validation", "test")

# 有代表性的日照 episode（PV / 风电非零），避免夜间全零假绿。
DAYLIGHT_START = "2024-04-16T10:00:00+08:00"
DAYLIGHT_SPLIT = "train"

DELTA_HOURS = 0.5
MICRO = 1_000_000
BACKLOG_PROFILE = "initial_backlog"

# 压力 episode：低油门让积压真实累积（守恒式才不会退化成 0 == 0）。
STRESS_HORIZON = 48
STRESS_THROTTLE = 0.05


def _ei():
    return importlib.import_module("scenario.env_injection")


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


def _injection(split: str = DAYLIGHT_SPLIT, horizon: int = 8, cutoff: int = 4,
               start: str | None = None):
    return _ei().build_verified_formal_env_injection(
        split, start=start or STARTS[split], horizon=horizon, forecast_cutoff=cutoff)


def _env(split: str = DAYLIGHT_SPLIT, horizon: int = 8, cutoff: int = 4, seed: int = 7,
         start: str | None = None):
    inj = _injection(split, horizon, cutoff, start)
    env = _env_cls()(horizon=horizon, task_seed=seed, server_seed=0,
                     forecast_seed=seed, delta_t_hours=DELTA_HOURS,
                     formal_injection=inj)
    return env, inj


def _daylight_env(horizon: int = 8, cutoff: int = 4, seed: int = 7):
    return _env(DAYLIGHT_SPLIT, horizon, cutoff, seed, start=DAYLIGHT_START)


def _throttled(action_dim: int, n_groups: int, throttle: float) -> np.ndarray:
    action = np.zeros(action_dim, dtype=np.float64)
    action[:n_groups] = throttle
    return action


def _mapper_tasks(env):
    """env.tasks 中来自 mapper 的任务（**排除** env 另插入的 initial backlog）。"""
    return [t for t in env.tasks if t.profile_key != BACKLOG_PROFILE]


# =============================================================================
# 第 1 组：原始 aggregate 守恒
# =============================================================================

@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_slot_ledger_equals_the_raw_realized_aggregate(split):
    """每槽整数账本 == `int(原始 realized aggregate[global_origin+k]) × 1e6`。"""
    am = importlib.import_module("scenario.arrival_mapper")
    inj = _injection(split, horizon=8)
    raw = am._aggregate_from_verified_chain(am.load_verified_mapper_chain(split))

    expected = tuple(int(raw[inj.global_origin + k]) * MICRO for k in range(inj.horizon))
    assert inj.ledger_micro == expected
    # 非空洞性：账本必须逐槽为正，否则恒等式退化成 0 == 0。
    assert all(int(v) > 0 for v in inj.ledger_micro)

    # Σ 账本 == Σ mapper Task.workload（**不**混入 initial backlog）
    assert sum(inj.ledger_micro) / MICRO == pytest.approx(
        sum(t.workload for t in inj.tasks))


@needs_assets
def test_initial_backlog_is_separate_from_the_arrival_ledger():
    """initial backlog 是 env 另插入的 Task，**单独**成账，不进 arrival 总量。"""
    env, inj = _daylight_env()
    env.reset(seed=7)
    assert env.initial_Q > 0.0, "本用例要求非零初始积压，否则无意义"

    backlog = [t for t in env.tasks if t.profile_key == BACKLOG_PROFILE]
    assert len(backlog) == 1
    assert float(backlog[0].workload) == pytest.approx(env.initial_Q)

    # arrival 总量只由 mapper 账本构成
    assert sum(inj.ledger_micro) / MICRO == pytest.approx(
        sum(t.workload for t in inj.tasks))
    assert env.initial_Q != pytest.approx(
        sum(inj.ledger_micro) / MICRO), "backlog 不得被并入 arrival 总量"

    # env 的 mapper 任务总量 == 账本（backlog 已排除）
    assert sum(t.workload for t in _mapper_tasks(env)) == pytest.approx(
        sum(inj.ledger_micro) / MICRO)


@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_env_arrived_tasks_match_the_slot_ledgers(split):
    """env 中每槽实际到达的 mapper 工作量 == 该槽整数账本。"""
    env, inj = _env(split, horizon=8)
    env.reset(seed=7)

    tasks = _mapper_tasks(env)
    assert len(tasks) == len(inj.tasks)
    for k in range(inj.horizon):
        got = sum(t.workload for t in tasks if int(t.arrival_time) == k)
        assert got == pytest.approx(inj.ledger_micro[k] / MICRO, abs=1e-9)
        # 非空洞性：每槽都必须真的有任务到达
        assert any(int(t.arrival_time) == k for t in tasks)


@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_every_step_conserves_the_original_aggregate(split):
    """逐步守恒：`initial_Q + Σ_{k≤t} 账本 == 累计完成 + 当步 backlog`（无误差项）。"""
    env, inj = _env(split, horizon=STRESS_HORIZON)
    env.reset(seed=7)
    ledger_work = [m / MICRO for m in inj.ledger_micro]

    arrived = 0.0
    info = None
    for k in range(env.horizon):
        _obs, _r, term, _trunc, info = env.step(
            _throttled(env.action_dim, env.model.N, STRESS_THROTTLE))
        arrived += ledger_work[k]
        expected_backlog = env.initial_Q + arrived - env.total_completed_work
        assert info["Q"] == pytest.approx(expected_backlog, abs=1e-6), (
            f"第 {k} 步守恒被打破：Q={info['Q']} vs 期望 {expected_backlog}")
        assert info["completed_work"] >= 0.0
        if term:
            break

    assert term, "压力 episode 必须能跑完"
    # 非空洞性：低油门必须留下真实积压，否则守恒式退化成「总是清空」。
    assert info["Q"] > 0.0, "压力 episode 未留下积压，本用例无意义"


@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_terminal_leftover_matches_the_final_backlog(split):
    """终点：`terminal_leftover == 最终 backlog`；完成量不重复计入也不遗失。"""
    env, inj = _env(split, horizon=STRESS_HORIZON)
    env.reset(seed=7)

    step_completed = 0.0
    for _k in range(env.horizon):
        _obs, _r, term, _trunc, info = env.step(
            _throttled(env.action_dim, env.model.N, STRESS_THROTTLE))
        step_completed += float(info["completed_work"])
        if term:
            break
    assert term

    # reward / info 的完成量既不重复计入、也不遗失
    assert env.total_completed_work == pytest.approx(step_completed, abs=1e-9)

    assert env.terminal_leftover_work == pytest.approx(
        env._compute_backlog_work(), abs=1e-9)
    assert env.terminal_leftover_work == pytest.approx(
        env.initial_Q + sum(inj.ledger_micro) / MICRO - env.total_completed_work,
        abs=1e-6)
    # 非空洞性：终点必须真的留下未完成工作量
    assert env.terminal_leftover_work > 0.0


# =============================================================================
# 第 2 组：半小时执行量与物理负载
# =============================================================================

@needs_assets
def test_planned_capacity_is_the_half_hour_rate():
    """`planned_capacity == Σ(计划负载 × C_server) × delta_t_hours`。"""
    env, _inj = _daylight_env()
    env.reset(seed=7)
    throttle = 0.5
    _obs, _r, _term, _trunc, info = env.step(
        _throttled(env.action_dim, env.model.N, throttle))

    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    rate = float((np.full(env.model.N, throttle * env.max_task_load_per_server)
                  * c_server).sum())
    assert info["planned_capacity"] == pytest.approx(rate * DELTA_HOURS)
    # 记录在案的「每小时 rate」必须与 × 0.5 之前的一致
    assert env.last_planned_capacity_rate_work_per_hour == pytest.approx(rate)
    assert env.last_planned_capacity_per_step == pytest.approx(rate * DELTA_HOURS)


@needs_assets
def test_group_load_fraction_uses_the_half_hour_step_length():
    """负载 = 完成工作 / **每步**能力（`C_server × delta_t_hours`）。

    半油门（`server_action = 0.5`）下若完成**全部**计划量，负载必须等于
    `0.5 × max_task_load_per_server`。改前除以 `C_server`（work/hour）会得到
    恰好**一半**的值 —— 该 2× 差值就是 formal 任务负载被低估的证据。
    """
    env, _inj = _daylight_env()
    env.reset(seed=7)
    c_server = np.asarray(env.model.C_server, dtype=np.float64)
    throttle = 0.5
    planned_task_loads = np.full(env.model.N, throttle * env.max_task_load_per_server)

    capacity_per_step = planned_task_loads * c_server * env.delta_t_hours
    loads = env._loads_from_group_completion(capacity_per_step)
    expected = throttle * env.max_task_load_per_server
    assert loads[0] == pytest.approx(expected), (
        f"完成全部计划量时组负载应为 {expected}（throttle × max_task_load_per_server），"
        f"实际 {loads[0]}：比值 {loads[0] / expected:.3f}，说明漏乘 delta_t_hours")

    # 往返一致：负载 × 组能力 × 步长 必须还原出该步完成的工作量
    recovered = float(np.sum(loads * c_server * env.delta_t_hours))
    assert recovered == pytest.approx(float(np.sum(capacity_per_step)))
    # 非空洞性：输入非零且未触顶（否则 clip 会掩盖差异）
    assert float(np.sum(capacity_per_step)) > 0.0
    assert loads[0] < env.max_task_load_per_server


@needs_assets
def test_group_load_fraction_is_self_consistent_with_the_reported_mean():
    """info 报出的 `actual_task_load_mean` 必须就是该步组负载的均值（自洽守卫）。"""
    env, _inj = _daylight_env()
    env.reset(seed=7)
    _obs, _r, _term, _trunc, info = env.step(
        _throttled(env.action_dim, env.model.N, 0.5))
    by_group = np.asarray(info["completed_work_by_group"], dtype=np.float64)
    assert float(np.mean(env._loads_from_group_completion(by_group))) == pytest.approx(
        float(info["actual_task_load_mean"]))


@needs_assets
def test_step_energy_converts_power_once_with_the_half_hour_step():
    """功率 → 能量只乘**一次** delta_t_hours（不得重复乘 0.5）。"""
    env, _inj = _daylight_env()
    env.reset(seed=7)
    _obs, _r, _term, _trunc, info = env.step(
        _throttled(env.action_dim, env.model.N, 0.5))
    assert info["idc_energy_kWh"] == pytest.approx(
        float(info["P_IDC_kW"]) * env.delta_t_hours, rel=1e-9)
    assert env.delta_t_hours == DELTA_HOURS


# =============================================================================
# 第 3 组：因果性与未来任务数
# =============================================================================

def _duplicate_a_future_task(env):
    """复制一个**尚未到达**的 Task；返回 (原任务数, 新任务数)。"""
    before = len(env.tasks)
    future = [t for t in env.tasks if int(t.arrival_time) > int(env.current_step)]
    assert future, "本用例要求存在未来任务，否则无意义"
    env.tasks = env.tasks + [copy.deepcopy(future[0])]
    return before, len(env.tasks)


@needs_assets
@pytest.mark.leakage
def test_future_task_count_does_not_change_the_observation():
    """复制**未来**任务（已到达集合不变）→ formal observation 必须不变。"""
    env, _inj = _daylight_env(horizon=8, cutoff=4)
    env.reset(seed=7)
    base_obs = env._get_obs().copy()
    base_pool = env._get_task_pool_features(current_time=0).copy()

    before, after = _duplicate_a_future_task(env)
    # --- 非空洞性：mutation 必须真的改到任务总数，且未到达任务确实存在 ---
    assert after == before + 1
    assert any(int(t.arrival_time) > int(env.current_step) for t in env.tasks)

    assert np.allclose(base_pool, env._get_task_pool_features(current_time=0)), \
        "任务池特征不得随未来任务数变化（len(self.tasks) 含未到达任务）"
    assert np.allclose(base_obs, env._get_obs()), \
        "formal observation 不得随未来任务数变化"


@needs_assets
@pytest.mark.leakage
def test_future_task_count_does_not_change_the_step_reward():
    """已到达状态与动作相同 → 当步 reward 不得随未来任务数量变化。"""
    action = _throttled(_daylight_env()[0].action_dim, _daylight_env()[0].model.N, 0.3)

    ref, _inj_a = _daylight_env(horizon=8, cutoff=4)
    ref.reset(seed=7)
    for _ in range(2):
        ref.step(action)

    probe, _inj_b = _daylight_env(horizon=8, cutoff=4)
    probe.reset(seed=7)
    for _ in range(2):
        probe.step(action)

    # 前置：两侧状态必须**完全相同**（否则对照无意义）
    assert ref.current_step == probe.current_step
    assert np.allclose(ref._get_obs(), probe._get_obs())
    assert np.allclose(ref.true_task_arrival_profile, probe.true_task_arrival_profile)

    before, after = _duplicate_a_future_task(probe)
    assert after == before + 1, "mutation 未生效"

    _o1, reward_ref, _t1, _r1, info_ref = ref.step(action)
    _o2, reward_probe, _t2, _r2, info_probe = probe.step(action)

    # 已到达侧的可观测量必须一致
    assert info_ref["newly_finished_count"] == info_probe["newly_finished_count"]
    assert info_ref["Q"] == pytest.approx(info_probe["Q"])
    assert reward_ref == pytest.approx(reward_probe), (
        "当步 reward 不得随未来任务总数变化（分母用了 len(self.tasks)）")


@needs_assets
@pytest.mark.leakage
def test_reset_info_task_count_excludes_future_tasks():
    """formal reset `info` 中的任务数不得含未来任务。"""
    env, _inj = _daylight_env(horizon=8, cutoff=4)
    _obs, info = env.reset(seed=7)

    arrived = sum(1 for t in env.tasks if int(t.arrival_time) <= int(env.current_step))
    total = len(env.tasks)
    assert total > arrived, "本用例要求存在未来任务，否则无意义"

    assert info["total_task_count"] == arrived, (
        f"formal info 的 total_task_count 含未来任务：{info['total_task_count']} "
        f"vs 已到达 {arrived} / 总数 {total}")


@needs_assets
@pytest.mark.leakage
def test_visible_forecast_mutation_must_change_the_observation():
    """**反向控制**：改可见窗口内的 causal forecast → observation **必变**。"""
    env, _inj = _daylight_env(horizon=8, cutoff=4)
    env.reset(seed=7)
    base = env._get_forecast_features().copy()

    env.task_arrival_forecast = np.asarray(
        env.task_arrival_forecast, dtype=np.float64) + 5.0
    assert not np.allclose(base, env._get_forecast_features()), \
        "可见窗口内的 forecast 改变了，observation 却不变 —— 断言过强"


@needs_assets
@pytest.mark.leakage
def test_current_step_physical_input_mutation_must_change_the_observation():
    """**区分**：当步**已可观测**的真实物理输入变了，observation **应当**改变。

    这不是泄漏 —— 不得把「未来真值不变性」误用成「任何输入都不变」。
    """
    env, _inj = _daylight_env(horizon=8, cutoff=4)
    env.reset(seed=7)
    base_obs = env._get_obs().copy()

    t = int(env.current_step)
    before = float(env.price_t[t])
    env.price_t[t] = before + 1.0
    assert float(env.price_t[t]) != before, "mutation 未生效"

    assert not np.allclose(base_obs, env._get_obs()), \
        "当步 price 是已可观测输入，改变它必须改变 observation"


# =============================================================================
# 第 4 组：重放
# =============================================================================

def _stream_fingerprint(stream):
    return (
        tuple(stream.ledger_micro),
        tuple(
            (int(t.task_id), str(t.profile_key), str(t.name), int(t.arrival_time),
             int(t.duration), float(t.workload), int(t.deadline), float(t.priority),
             bool(t.interruptible), bool(t.parallelizable),
             tuple(float(v) for v in np.asarray(t.load_profile)))
            for t in stream.tasks
        ),
    )


@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_mapper_stream_is_bit_identical_on_rebuild(split):
    """同一 split / origin / 冻结参数 / seed → mapper stream **逐位一致**。"""
    am = importlib.import_module("scenario.arrival_mapper")
    a = am.build_arrival_task_stream(split, start=STARTS[split], horizon=8, seed=0)
    b = am.build_arrival_task_stream(split, start=STARTS[split], horizon=8, seed=0)
    assert _stream_fingerprint(a) == _stream_fingerprint(b)
    # 非空洞性
    assert len(a.tasks) > 0 and sum(a.ledger_micro) > 0


@needs_assets
def test_reset_replay_restores_tasks_and_observation():
    """`reset → 多步 → 同 seed reset` 的初始 Task 状态与 observation 一致。"""
    env, _inj = _daylight_env(horizon=8, cutoff=4)
    obs_before, _ = env.reset(seed=7)
    before = [(t.task_id, t.status, float(t.remaining_work), t.start_time, t.finish_time)
              for t in env.tasks]

    action = _throttled(env.action_dim, env.model.N, 0.9)
    for _ in range(3):
        env.step(action)
    assert [(t.task_id, t.status, float(t.remaining_work)) for t in env.tasks] != before, \
        "执行后任务状态必须变化，否则重放无意义"

    obs_after, _ = env.reset(seed=7)
    after = [(t.task_id, t.status, float(t.remaining_work), t.start_time, t.finish_time)
             for t in env.tasks]
    assert after == before
    assert np.allclose(obs_before, obs_after)


@needs_assets
def test_independent_env_replays_the_same_episode_and_actions():
    """独立新 env 重建同一 episode、重放同一动作序列 → 逐步状态 / reward / 结算一致。"""
    action = _throttled(_daylight_env()[0].action_dim, _daylight_env()[0].model.N, 0.4)

    left, _a = _daylight_env(horizon=8, cutoff=4)
    right, _b = _daylight_env(horizon=8, cutoff=4)
    obs_l, _ = left.reset(seed=7)
    obs_r, _ = right.reset(seed=7)
    assert np.allclose(obs_l, obs_r)

    for step in range(left.horizon):
        _ol, rl, tl, _trl, il = left.step(action)
        _or, rr, tr, _trr, ir = right.step(action)

        assert rl == pytest.approx(rr), f"第 {step} 步 reward 不一致"
        assert tl == tr
        assert il["Q"] == pytest.approx(ir["Q"], abs=1e-9)
        assert il["completed_work"] == pytest.approx(ir["completed_work"], abs=1e-9)
        assert il["total_completed_work"] == pytest.approx(
            ir["total_completed_work"], abs=1e-9)
        assert [(t.task_id, t.status, float(t.remaining_work))
                for t in left.tasks] == \
               [(t.task_id, t.status, float(t.remaining_work))
                for t in right.tasks]

        if tl:
            assert left.terminal_leftover_work == pytest.approx(
                right.terminal_leftover_work, abs=1e-9)
            assert left.terminal_settlement_penalty == pytest.approx(
                right.terminal_settlement_penalty, abs=1e-9)
            break
    else:
        raise AssertionError("episode 未终止")
