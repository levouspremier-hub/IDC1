"""M1.3g-f-c-f：**三批** formal PPO 更新连续性 probe。

三个互不相同的合法 train origins（本地 48 / 96 / 144）上对照：

```text
A 连续：      batch1(48) → batch2(96) → batch3(144)
B 两批+恢复： batch1(48) → batch2(96) → 保存批次边界 checkpoint
              → 调用方构造的**全新**对象恢复 → batch3(144)
```

**第三批的每个 `Transition` 字段逐项精确对照**（不以 digest 或近似比较代替）；
**边界 Adam step == 2**、**最终 step == 3** 显式锚定。

**本卡不改** `train.py`、env、`checkpointing` 契约、冻结资产、env release v1；
**不**接正式训练入口；**不**声明训练有效。

**改前状态**：`safe_rl_v2/ppo_three_batch.py` 与本文件均不存在。
"""

import importlib
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "safe_rl_v2.ppo_three_batch"

HORIZON = 8
FORECAST_CUTOFF = 4
DELTA_HOURS = 0.5
STEPS = 3
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAM = 0.95
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
EXAMPLE_ENV_SEED = 0

# 示例输入（由**调用方**构造并传入；**不是**正式训练超参数）
EXAMPLE_POLICY_SEED = 0
EXAMPLE_ADAM_LR = 1e-3
EXAMPLE_BUSINESS = {"budget": 5.0, "learning_rate": 0.01, "max_multiplier": 100.0}
EXAMPLE_CARBON = {"budget": 3.0, "learning_rate": 0.01, "max_multiplier": 100.0}


def tb():
    return importlib.import_module(MODULE)


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


def _objects(obs_dim: int, *, adam_lr: float = EXAMPLE_ADAM_LR):
    """**调用方**构造的一组对象（示例超参数，非正式训练配置）。"""
    from safe_rl_v2.lagrangian import (
        UNIT_KG_CO2E,
        UNIT_VIOLATION_TASK_STEPS,
        ConstraintSpec,
        Lagrangian,
    )
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(EXAMPLE_POLICY_SEED)
        policy = SafePPOPolicy(obs_dim=obs_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=adam_lr)
    lagrangian = Lagrangian((
        ConstraintSpec(name="business", unit=UNIT_VIOLATION_TASK_STEPS,
                       **EXAMPLE_BUSINESS),
        ConstraintSpec(name="carbon", unit=UNIT_KG_CO2E, **EXAMPLE_CARBON),
    ))
    generator = torch.Generator()
    generator.manual_seed(0)
    return policy, optimizer, lagrangian, generator


def _obs_dim() -> int:
    from safe_rl_v2.ppo_two_batch import build_formal_env

    env, _inj = build_formal_env(
        tb().three_batch_starts()[0], horizon=HORIZON,
        forecast_cutoff=FORECAST_CUTOFF, delta_t_hours=DELTA_HOURS,
        env_seed_kwargs=dict(ENV_SEED_KWARGS))
    return int(env.obs_dim)


def _probe(tmp_path):
    m = tb()
    obs_dim = _obs_dim()
    path_a = _objects(obs_dim)
    resume_objects = _objects(obs_dim)      # **全新的**另一组，供路径 B 恢复
    out = m.run_three_batch_probe(
        *path_a, resume_objects=resume_objects,
        checkpoint_path=tmp_path / "boundary_after_batch2.pt",
        clip_epsilon=CLIP_EPSILON, gamma=GAMMA, lam=LAM, steps=STEPS,
        horizon=HORIZON, forecast_cutoff=FORECAST_CUTOFF,
        delta_t_hours=DELTA_HOURS, env_seed=EXAMPLE_ENV_SEED,
        env_seed_kwargs=dict(ENV_SEED_KWARGS), obs_dim=obs_dim, action_dim=21)
    return out, path_a, resume_objects, obs_dim


# --- 逐项精确比较器（本档独立实现，不复用其他测试文件的） ---------------------

def _assert_exact(a, b, path: str = "state") -> None:
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b), f"{path}: 类型不同"
        assert a.dtype == b.dtype, f"{path}: dtype"
        assert a.shape == b.shape, f"{path}: shape"
        assert torch.equal(a, b), f"{path}: 张量元素不逐位相等"
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), f"{path}: 类型"
        assert a.dtype == b.dtype, f"{path}: dtype"
        assert a.shape == b.shape, f"{path}: shape"
        assert np.array_equal(a, b), f"{path}: 数组元素不逐位相等"
        return
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), f"{path}: 类型不同"
        assert set(a) == set(b), f"{path}: 键集合不同 {sorted(set(a) ^ set(b))}"
        for key in sorted(a):
            _assert_exact(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        assert isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)), f"{path}: 类型"
        assert len(a) == len(b), f"{path}: 长度"
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            _assert_exact(left, right, f"{path}[{index}]")
        return
    assert type(a) is type(b), f"{path}: 类型 {type(a)} != {type(b)}"
    assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_transition_exact(left, right, label: str) -> None:
    """**每个** `Transition` 契约字段逐项精确比较。"""
    for field in tb().TRANSITION_FIELDS:
        _assert_exact(getattr(left, field), getattr(right, field), f"{label}.{field}")


def _adam_steps(state: dict) -> set:
    return {float(v["step"]) for v in state["optimizer"]["state"].values()
            if isinstance(v, dict) and "step" in v}


# =============================================================================
# 1. 三个 origin 均经 verified 链校验；每批都有真实 formal transition
# =============================================================================

@needs_assets
def test_three_origins_are_verified_and_distinct():
    from scenario.b6_split_manifests import local_origin_from_start

    starts = tb().three_batch_starts()
    assert len(set(starts)) == 3, "三个 origin 必须互不相同"
    origins = [local_origin_from_start("train", start) for start in starts]
    assert origins == [48, 96, 144], f"实测 origins = {origins}"
    assert len(set(origins)) == 3


@needs_assets
def test_every_batch_carries_real_formal_transitions(tmp_path):
    out, _a, _b, obs_dim = _probe(tmp_path)
    starts = tb().three_batch_starts()

    for index, batch in enumerate(out["batches"]):
        assert batch["start"] == starts[index]
        assert batch["index"] == index
        assert batch["formal"] is True, f"批 {index} 必须来自 formal env"
        assert batch["transitions"] == STEPS, f"批 {index} 采集步数不足"
        assert len(batch["transition_records"]) == STEPS
        for record in batch["transition_records"]:
            obs = np.asarray(record.observation, dtype=np.float32)
            assert obs.shape == (obs_dim,)
            assert np.all(np.isfinite(obs))
            assert float(np.max(np.abs(obs))) > 0.0, "观测不得是恒零占位"
            assert record.contract_version, "Transition 必须带契约版本"
        assert any(abs(float(r.reward)) > 0.0 for r in batch["transition_records"])
    assert out["batches"][2]["origin"] == 144
    assert out["third_batch_resumed"]["origin"] == 144


# =============================================================================
# 2. 第三批 Transition **逐项**精确相同
# =============================================================================

@needs_assets
def test_third_batch_transitions_match_field_by_field(tmp_path):
    out, _a, _b, _obs_dim = _probe(tmp_path)
    live = out["batches"][2]["transition_records"]
    resumed = out["third_batch_resumed"]["transition_records"]

    assert len(live) == len(resumed) == STEPS
    for index, (left, right) in enumerate(zip(live, resumed, strict=True)):
        _assert_transition_exact(left, right, f"batch3[{index}]")

    # 非空洞性：字段本身必须**携带信息**（不是全零/占位）
    sample = live[0]
    assert np.any(np.asarray(sample.observation) != 0.0)
    assert np.any(np.asarray(sample.raw_action) != 0.0)
    # 契约：corrector 关闭时 exec 与 raw 逐元素相同（既非空洞，也非被改写）
    for index, record in enumerate(live):
        assert np.array_equal(record.raw_action, record.exec_action), (
            f"batch3[{index}]：corrector 关闭时 exec_action 必须等于 raw_action")
    # 至少一条 transition 的 reward 或 cost 非零
    assert any(abs(float(r.reward)) > 0.0 or float(r.carbon_cost) > 0.0
               for r in live)


@needs_assets
def test_third_batch_field_comparator_detects_a_single_element_difference(tmp_path):
    """**比较器灵敏度**：单元素扰动必须被抓到（防逐项对照空洞）。"""
    import copy as _copy

    out, _a, _b, _obs_dim = _probe(tmp_path)
    left = out["batches"][2]["transition_records"][0]
    right = out["third_batch_resumed"]["transition_records"][0]

    _assert_transition_exact(left, right, "sanity")   # 原样必须通过

    tampered = _copy.deepcopy(right)
    tampered.observation[0] = float(tampered.observation[0]) + 1e-6
    with pytest.raises(AssertionError, match="不逐位相等"):
        _assert_transition_exact(left, tampered, "tampered.observation")

    tampered_reward = _copy.deepcopy(right)
    tampered_reward.reward = float(tampered_reward.reward) + 1e-9
    with pytest.raises(AssertionError):
        _assert_transition_exact(left, tampered_reward, "tampered.reward")

    tampered_logprob = _copy.deepcopy(right)
    tampered_logprob.old_raw_log_prob = float(tampered_logprob.old_raw_log_prob) + 1e-9
    with pytest.raises(AssertionError):
        _assert_transition_exact(left, tampered_logprob, "tampered.old_raw_log_prob")

    tampered_version = _copy.deepcopy(right)
    tampered_version.contract_version = "contract-v0"
    with pytest.raises(AssertionError):
        _assert_transition_exact(left, tampered_version, "tampered.contract_version")


# =============================================================================
# 3. 第三批后：policy / 完整 Adam / Lagrangian / RNG 逐项精确相同
# =============================================================================

@needs_assets
def test_final_state_after_batch3_is_exactly_equal(tmp_path):
    out, _a, _b, _obs_dim = _probe(tmp_path)
    fa, fb = out["final_a"], out["final_b"]

    _assert_exact(fa["policy"], fb["policy"], "final.policy")
    _assert_exact(fa["optimizer"], fb["optimizer"], "final.optimizer")
    _assert_exact(fa["lagrangian"], fb["lagrangian"], "final.lagrangian")
    _assert_exact(fa["generator"], fb["generator"], "final.generator")

    # 非空洞性：完整 Adam 状态必须**真的**含 step 与动量张量
    state = fa["optimizer"]["state"]
    assert state, "最终 optimizer 状态不得为空"
    for value in state.values():
        assert isinstance(value, dict)
        assert "step" in value and "exp_avg" in value and "exp_avg_sq" in value
        assert torch.any(value["exp_avg"] != 0.0), "exp_avg 不应恒零"
        assert torch.any(value["exp_avg_sq"] != 0.0), "exp_avg_sq 不应恒零"
    assert fa["lagrangian"]["updates"] == 3


@needs_assets
def test_third_batch_final_state_differs_from_the_boundary(tmp_path):
    """**非空洞性**：最终状态必须已从批 2 边界**前进**，否则对照无意义。"""
    out, _a, _b, _obs_dim = _probe(tmp_path)
    boundary, final = out["boundary_a"], out["final_a"]

    assert not torch.equal(boundary["policy"]["log_std"], final["policy"]["log_std"])
    assert _adam_steps(boundary) != _adam_steps(final)
    assert boundary["lagrangian"]["updates"] != final["lagrangian"]["updates"]
    assert not torch.equal(boundary["generator"], final["generator"])


# =============================================================================
# 4. 显式锚点：边界 step=2、最终 step=3；边界快照独立于批 3 活状态
# =============================================================================

@needs_assets
def test_boundary_is_step_two_and_final_is_step_three(tmp_path):
    """**显式锚点**：批 2 边界 Adam step == 2，最终 step == 3（两条路径各自）。"""
    out, _a, _b, _obs_dim = _probe(tmp_path)

    assert _adam_steps(out["boundary_a"]) == {2.0}, \
        f"边界（路径 A）Adam step 应为 2，实际 {_adam_steps(out['boundary_a'])}"
    assert _adam_steps(out["boundary_b"]) == {2.0}, \
        f"边界（路径 B 恢复后）Adam step 应为 2，实际 {_adam_steps(out['boundary_b'])}"
    assert _adam_steps(out["final_a"]) == {3.0}, \
        f"最终（路径 A）Adam step 应为 3，实际 {_adam_steps(out['final_a'])}"
    assert _adam_steps(out["final_b"]) == {3.0}, \
        f"最终（路径 B）Adam step 应为 3，实际 {_adam_steps(out['final_b'])}"

    assert out["boundary_a"]["lagrangian"]["updates"] == 2
    assert out["boundary_b"]["lagrangian"]["updates"] == 2
    assert out["final_a"]["lagrangian"]["updates"] == 3
    assert out["final_b"]["lagrangian"]["updates"] == 3


@needs_assets
def test_boundary_snapshot_is_independent_of_the_third_batch(tmp_path):
    """**防别名假绿**：批 2 边界的快照不得被批 3 的原位更新改写。

    与 M1.3g-f-c-e-R2 同类：`optimizer.state_dict()` 的内层张量是活引用，
    若快照不是真深快照，「批 3 **之前**」的对照会退化成「之后 vs 之后」。
    """
    out, _a, _b, _obs_dim = _probe(tmp_path)
    boundary, final = out["boundary_a"], out["final_a"]

    # 边界已锚定为 step 2（若被批 3 改写，这里会是 3）
    assert _adam_steps(boundary) == {2.0}
    assert _adam_steps(final) == {3.0}

    # 张量级别：边界快照的 exp_avg 必须**独立**于最终活状态的 exp_avg
    def first_exp_avg(snapshot):
        for value in snapshot["optimizer"]["state"].values():
            if isinstance(value, dict) and "exp_avg" in value:
                return value["exp_avg"]
        raise AssertionError("找不到 exp_avg")

    boundary_exp_avg = first_exp_avg(boundary)
    final_exp_avg = first_exp_avg(final)
    assert not torch.equal(boundary_exp_avg, final_exp_avg), \
        "边界 exp_avg 与最终 exp_avg 相同 —— 快照可能被批 3 原位改写"
    # 边界快照必须**独立持有存储**（不是活张量的别名）
    assert boundary["policy"]["log_std"].data_ptr() != \
        out["policy"].state_dict()["log_std"].data_ptr(), \
        "边界快照与活参数共享存储"


# =============================================================================
# 5. 参数所有权 / claims
# =============================================================================

@needs_assets
def test_probe_uses_caller_objects_and_never_chooses_configuration(tmp_path):
    out, path_a, resume_objects, _obs_dim = _probe(tmp_path)
    assert out["policy"] is path_a[0]
    assert out["optimizer"] is path_a[1]
    assert out["lagrangian"] is path_a[2]
    assert out["generator"] is path_a[3]
    assert out["resume_objects"] is resume_objects
    assert len(path_a[1].state) > 0, "调用方 optimizer 必须被推进"
    assert path_a[2]._updates == 3, "调用方 Lagrangian 必须累计 3 次更新"

    import inspect

    params = inspect.signature(tb().run_three_batch_probe).parameters
    for forbidden in ("adam_lr", "learning_rate", "budget", "max_multiplier",
                      "hidden", "policy_seed", "sampling_seed"):
        assert forbidden not in params, f"probe 不得拥有超参数入参 {forbidden}"
    # probe 不得自行构造训练对象
    source = (REPO_ROOT / "safe_rl_v2" / "ppo_three_batch.py").read_text(
        encoding="utf-8")
    for forbidden in ("SafePPOPolicy(", "torch.optim.Adam(", "Lagrangian((",
                      "ConstraintSpec(", "manual_seed("):
        assert forbidden not in source, f"probe 不得在内部构造/播种：{forbidden!r}"


@needs_assets
def test_probe_does_not_claim_training_success(tmp_path):
    out, _a, _b, _obs_dim = _probe(tmp_path)
    assert out["claims"] == {"trained": False, "performance_evaluated": False,
                             "convergence_claimed": False}
    assert out["probe_only"] is True
    # 非空洞性：确实做了三次真实更新
    assert out["batches"][2]["optimizer_steps_cumulative"] == 3
    assert out["third_batch_resumed"]["optimizer_steps_cumulative"] == 3
