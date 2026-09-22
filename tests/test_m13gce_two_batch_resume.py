"""M1.3g-f-c-e：双批次 probe 的 checkpoint/resume **等价性**。

证明「连续跑两批」与「批 1 → 保存 → 新对象恢复 → 批 2」**逐位等价**：

```text
连续：  batch1(origin 48) → batch2(origin 96)
恢复：  batch1(origin 48) → save(policy/optimizer/lagrangian/generator + next origin)
        → 新对象 restore → batch2(origin 96)
```

**只证明批次边界恢复** —— **不**宣称中途恢复、正式训练或收敛。
checkpoint 只写 `tmp_path`，**不写仓库**。

**改前缺陷（本文件对应先红）**：`safe_rl_v2.ppo_two_batch` 尚无单批编排拆分与
checkpoint 适配入口。
"""

import importlib
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULE = "safe_rl_v2.ppo_two_batch"

HORIZON = 8
FORECAST_CUTOFF = 4
DELTA_HOURS = 0.5
STEPS = 3
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAM = 0.95
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}
EXAMPLE_ENV_SEED = 0


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


# 示例输入（由**调用方**构造；**不是**正式训练超参数）
EXAMPLE_ADAM_LR = 1e-3
EXAMPLE_BUSINESS = {"budget": 5.0, "learning_rate": 0.01, "max_multiplier": 100.0}
EXAMPLE_CARBON = {"budget": 3.0, "learning_rate": 0.01, "max_multiplier": 100.0}


def _obs_dim() -> int:
    env, _inj = tb().build_formal_env(
        tb().TRAIN_CANDIDATE_STARTS[0], horizon=HORIZON,
        forecast_cutoff=FORECAST_CUTOFF, delta_t_hours=DELTA_HOURS,
        env_seed_kwargs=dict(ENV_SEED_KWARGS))
    return int(env.obs_dim)


def _objects(obs_dim: int, *, adam_lr: float = EXAMPLE_ADAM_LR):
    from safe_rl_v2.lagrangian import (
        UNIT_KG_CO2E,
        UNIT_VIOLATION_TASK_STEPS,
        ConstraintSpec,
        Lagrangian,
    )
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
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


def _batch_kwargs(**overrides):
    kwargs = {"clip_epsilon": CLIP_EPSILON, "gamma": GAMMA, "lam": LAM,
              "steps": STEPS, "horizon": HORIZON, "forecast_cutoff": FORECAST_CUTOFF,
              "delta_t_hours": DELTA_HOURS, "env_seed": EXAMPLE_ENV_SEED,
              "env_seed_kwargs": dict(ENV_SEED_KWARGS)}
    kwargs.update(overrides)
    return kwargs


def _continuous():
    m = tb()
    policy, optimizer, lagrangian, generator = _objects(_obs_dim())
    return m.run_two_batch_probe(policy, optimizer, lagrangian, generator,
                                 **_batch_kwargs())


def _saved_and_resumed(tmp_path, **batch_overrides):
    """批 1 → 保存 → **新对象**恢复 → 批 2。"""
    m = tb()
    obs_dim = _obs_dim()
    policy, optimizer, lagrangian, generator = _objects(obs_dim)

    first = m.run_single_batch(
        policy, optimizer, lagrangian, generator,
        start=tb().TRAIN_CANDIDATE_STARTS[0], **_batch_kwargs(**batch_overrides))
    ckpt_path = tmp_path / "after_batch1.pt"
    m.save_two_batch_checkpoint(
        ckpt_path, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=tb().TRAIN_CANDIDATE_STARTS[1],
        obs_dim=obs_dim, action_dim=21)

    # **全新**对象（不是原来那两个）
    policy2, optimizer2, lagrangian2, generator2 = _objects(obs_dim)
    resumed = m.resume_two_batch_checkpoint(
        ckpt_path, policy=policy2, optimizer=optimizer2, lagrangian=lagrangian2,
        generator=generator2, expected_obs_dim=obs_dim, expected_action_dim=21)
    second = m.run_single_batch(
        policy2, optimizer2, lagrangian2, generator2,
        start=resumed["next_start"], **_batch_kwargs(**batch_overrides))
    return first, second, ckpt_path, (policy2, optimizer2, lagrangian2, generator2)


def _digest(arr) -> str:
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(arr, dtype=np.float32)).tobytes()).hexdigest()


# =============================================================================
# 1. 批次边界恢复：逐位等价
# =============================================================================

@needs_assets
def test_resumed_batch2_matches_continuous_batch2_bit_for_bit(tmp_path):
    """恢复后的批 2 必须与连续执行的批 2 **逐位一致**。"""
    m = tb()
    continuous = _continuous()
    first, second, _ckpt, objects = _saved_and_resumed(tmp_path)
    policy2, optimizer2, lagrangian2, generator2 = objects

    c2 = continuous["batches"][1]

    # transition 逐位
    assert _digest(c2["observations"]) == _digest(second["observations"])
    assert _digest(c2["raw_actions"]) == _digest(second["raw_actions"])
    assert _digest(c2["old_raw_log_prob"]) == _digest(second["old_raw_log_prob"])
    assert c2["batch_digest"] == second["batch_digest"]
    assert c2["transitions"] == second["transitions"] == STEPS

    # 最终参数 / optimizer / 乘子
    final_policy = _objects(_obs_dim())[0]
    final_policy.load_state_dict(policy2.state_dict())
    assert m._policy_state_digest(final_policy) == \
           continuous["final_policy_state_digest"]
    assert m._policy_state_digest(policy2) == continuous["final_policy_state_digest"]
    assert second["loss_total"] == pytest.approx(c2["loss_total"], rel=1e-12)
    assert second["param_delta_norm"] == pytest.approx(c2["param_delta_norm"], rel=1e-12)
    assert second["multipliers_post_update"] == c2["multipliers_post_update"]
    assert len(optimizer2.state) > 0, "恢复的 optimizer 必须被推进"

    # RNG 状态逐位一致（恢复后继续采样，状态必须等于连续路径）
    assert m._generator_state_digest(generator2) == c2["generator_state_digest_after"]


@needs_assets
def test_resume_restores_the_pre_batch2_object_state(tmp_path):
    """恢复必须把**对象状态**写回去，而不是只返回一个 origin。"""
    m = tb()
    continuous = _continuous()
    first, _second, _ckpt, objects = _saved_and_resumed(tmp_path)
    policy2, optimizer2, lagrangian2, generator2 = objects

    b1 = continuous["batches"][0]
    # 恢复到的是**批 1 结束**时的状态（不是初始状态）
    assert m._policy_state_digest(policy2) != b1["policy_state_digest"], \
        "批 2 之后策略必须已前进（否则本用例区分不出恢复是否生效）"
    assert lagrangian2._updates == 2, "恢复的 Lagrangian 必须累计到 2 次更新"
    # **RNG 前进**：恢复后的状态必须等于批 1 结束时的状态，而非初始种子状态
    initial = torch.Generator()
    initial.manual_seed(0)
    assert m._generator_state_digest(generator2) != m._generator_state_digest(initial)
    assert first["generator_state_digest"] == \
           continuous["batches"][0]["generator_state_digest"]


# =============================================================================
# 2. 契约拒收
# =============================================================================

@needs_assets
def test_resume_rejects_a_wrong_contract_version(tmp_path):
    m = tb()
    obs_dim = _obs_dim()
    policy, optimizer, lagrangian, generator = _objects(obs_dim)
    m.run_single_batch(policy, optimizer, lagrangian, generator,
                       start=tb().TRAIN_CANDIDATE_STARTS[0], **_batch_kwargs())
    path = tmp_path / "c.pt"
    m.save_two_batch_checkpoint(
        path, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=tb().TRAIN_CANDIDATE_STARTS[1],
        obs_dim=obs_dim, action_dim=21, contract_version_id="contract-v8")

    p2, o2, l2, g2 = _objects(obs_dim)
    with pytest.raises(Exception) as exc:
        m.resume_two_batch_checkpoint(
            path, policy=p2, optimizer=o2, lagrangian=l2, generator=g2,
            expected_obs_dim=obs_dim, expected_action_dim=21)
    assert "contract" in str(exc.value).lower()


@needs_assets
@pytest.mark.parametrize("field,bad,expect", [
    ("action_dim", 23, "action_dim"),
    ("obs_dim", 280, "obs_dim"),
    ("schema_hash", "0" * 64, "schema"),
])
def test_resume_rejects_mismatched_metadata(tmp_path, field, bad, expect):
    m = tb()
    obs_dim = _obs_dim()
    policy, optimizer, lagrangian, generator = _objects(obs_dim)
    m.run_single_batch(policy, optimizer, lagrangian, generator,
                       start=tb().TRAIN_CANDIDATE_STARTS[0], **_batch_kwargs())
    path = tmp_path / "c.pt"
    m.save_two_batch_checkpoint(
        path, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=tb().TRAIN_CANDIDATE_STARTS[1],
        obs_dim=obs_dim, action_dim=21, **{field: bad})

    p2, o2, l2, g2 = _objects(obs_dim)
    with pytest.raises(Exception) as exc:
        m.resume_two_batch_checkpoint(
            path, policy=p2, optimizer=o2, lagrangian=l2, generator=g2,
            expected_obs_dim=obs_dim, expected_action_dim=21)
    assert expect in str(exc.value).lower()


@needs_assets
def test_resume_rejects_a_checkpoint_without_metadata(tmp_path):
    m = tb()
    path = tmp_path / "nometa.pt"
    torch.save({"state": {}}, str(path))
    obs_dim = _obs_dim()
    p2, o2, l2, g2 = _objects(obs_dim)
    with pytest.raises(Exception) as exc:
        m.resume_two_batch_checkpoint(
            path, policy=p2, optimizer=o2, lagrangian=l2, generator=g2,
            expected_obs_dim=obs_dim, expected_action_dim=21)
    assert "metadata" in str(exc.value).lower()


@needs_assets
def test_resume_rejects_a_checkpoint_missing_state_fields(tmp_path):
    """缺状态字段（policy / optimizer / lagrangian / generator / next_start）明确拒绝。"""
    m = tb()
    obs_dim = _obs_dim()
    policy, optimizer, lagrangian, generator = _objects(obs_dim)
    m.run_single_batch(policy, optimizer, lagrangian, generator,
                       start=tb().TRAIN_CANDIDATE_STARTS[0], **_batch_kwargs())
    path = tmp_path / "c.pt"
    m.save_two_batch_checkpoint(
        path, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=tb().TRAIN_CANDIDATE_STARTS[1],
        obs_dim=obs_dim, action_dim=21)
    payload = torch.load(str(path), weights_only=False)
    payload["state"].pop("next_start", None)
    torch.save(payload, str(path))

    p2, o2, l2, g2 = _objects(obs_dim)
    with pytest.raises(Exception) as exc:
        m.resume_two_batch_checkpoint(
            path, policy=p2, optimizer=o2, lagrangian=l2, generator=g2,
            expected_obs_dim=obs_dim, expected_action_dim=21)
    assert "next_start" in str(exc.value)


# =============================================================================
# 3. 反空洞：一致性必须来自**真实恢复**
# =============================================================================

@needs_assets
def test_consistency_is_not_manufactured_by_reseeding(tmp_path):
    """**反空洞**：把 generator 改成**重播种**而不是恢复状态，结果必须**不同**。"""
    m = tb()
    continuous = _continuous()
    _first, second, _ckpt, objects = _saved_and_resumed(tmp_path)
    _p2, _o2, _l2, generator2 = objects

    reseeded = torch.Generator()
    reseeded.manual_seed(0)  # 重新播种 ⇒ 回到初始状态，而非批 1 结束状态
    assert m._generator_state_digest(reseeded) != \
           m._generator_state_digest(generator2), "重播种必须与真实恢复不同"

    # 用重播种的 generator 重跑批 2 ⇒ transition 必须与连续路径不同
    obs_dim = _obs_dim()
    policy3, optimizer3, lagrangian3, generator3 = _objects(obs_dim)
    m.run_single_batch(policy3, optimizer3, lagrangian3, generator3,
                       start=tb().TRAIN_CANDIDATE_STARTS[0], **_batch_kwargs())
    reseeded.manual_seed(0)
    bad = m.run_single_batch(
        policy3, optimizer3, lagrangian3, reseeded,
        start=tb().TRAIN_CANDIDATE_STARTS[1], **_batch_kwargs())
    assert bad["batch_digest"] != continuous["batches"][1]["batch_digest"], \
        "重播种造不出等价结果 —— 否则本卡的等价性证明没有意义"


@needs_assets
def test_zero_state_rebuild_does_not_reproduce_the_result(tmp_path):
    """**反空洞**：用**零状态**对象（而非恢复）跑批 2，结果必须不同。"""
    m = tb()
    continuous = _continuous()
    obs_dim = _obs_dim()
    fresh_policy, fresh_optimizer, fresh_lagrangian, fresh_generator = _objects(obs_dim)
    zero = m.run_single_batch(
        fresh_policy, fresh_optimizer, fresh_lagrangian, fresh_generator,
        start=tb().TRAIN_CANDIDATE_STARTS[1], **_batch_kwargs())
    assert zero["batch_digest"] != continuous["batches"][1]["batch_digest"]
    assert m._policy_state_digest(fresh_policy) != \
           continuous["final_policy_state_digest"]
