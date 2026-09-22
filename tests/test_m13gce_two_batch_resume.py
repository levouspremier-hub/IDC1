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
def test_resume_writes_state_into_freshly_constructed_objects(tmp_path):
    """**恢复边界**：`load` 必须把状态**写进调用方新建的对象**，而不是只返回 origin。

    ⚠️ **R1 重命名**：原名为 `..._restores_the_pre_batch2_object_state`，但它实际
    检查的是**批 2 之后**的状态（`_saved_and_resumed` 在返回前已跑完批 2），
    名实不符。现改为在**批 2 开始之前**检查，并取一个准确的名字。
    """
    m = tb()
    obs_dim = _obs_dim()
    policy, optimizer, lagrangian, generator = _objects(obs_dim)
    m.run_single_batch(policy, optimizer, lagrangian, generator,
                       start=_start(0), **_batch_kwargs())
    ckpt = tmp_path / "boundary.pt"
    m.save_two_batch_checkpoint(
        ckpt, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=_start(1), obs_dim=obs_dim, action_dim=21)
    boundary = _full_state(policy, optimizer, lagrangian, generator)

    # 全新对象：构造后、load 前，状态必须是**初始**的
    p2, o2, l2, g2 = _objects(obs_dim)
    fresh = _full_state(p2, o2, l2, g2)
    assert not fresh["optimizer"]["state"], "load 前 optimizer 状态应为空"
    assert l2._updates == 0, "load 前 Lagrangian 应零更新"

    resumed = m.resume_two_batch_checkpoint(
        ckpt, policy=p2, optimizer=o2, lagrangian=l2, generator=g2,
        expected_obs_dim=obs_dim, expected_action_dim=21)

    # load 之后、**批 2 之前**：四个对象必须已等于批 1 结束时的状态
    loaded = _full_state(p2, o2, l2, g2)
    _assert_exact(boundary, loaded, "resumed")
    assert resumed["next_start"] == _start(1)
    assert l2._updates == 1, "load 后 Lagrangian 应已恢复到 1 次更新"

    # **RNG 前进**：恢复后的状态必须等于批 1 结束时的状态，而非初始种子状态
    initial = torch.Generator()
    initial.manual_seed(0)
    assert m._generator_state_digest(g2) != m._generator_state_digest(initial)
    assert not torch.equal(loaded["generator"], fresh["generator"]), \
        "恢复后的 RNG 状态必须已从初始状态前进"


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
    meta = {"obs_dim": obs_dim, "action_dim": 21}
    meta[field] = bad
    m.save_two_batch_checkpoint(
        path, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        generator=generator, next_start=tb().TRAIN_CANDIDATE_STARTS[1], **meta)

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


# =============================================================================
# 5. M1.3g-f-c-e-R1：**完整状态**逐项精确对照
# =============================================================================

def _full_state(policy, optimizer, lagrangian, generator) -> dict:
    """四个对象的**完整**状态快照（含嵌套 state_dict 的全部张量与标量）。"""
    return {
        "policy": {k: v.detach().clone() for k, v in policy.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "lagrangian": lagrangian.state_dict(),
        "generator": generator.get_state(),
    }


def _assert_exact(a, b, path: str = "state") -> None:
    """**逐项精确**比较嵌套结构：张量按 dtype / shape / 元素；标量与键集合亦精确。

    「非空」或「digest 相同」**不**能替代本函数。
    """
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b), f"{path}: 类型不同"
        assert a.dtype == b.dtype, f"{path}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{path}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert torch.equal(a, b), f"{path}: 张量元素不逐位相等"
        return
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), f"{path}: 类型不同"
        assert set(a) == set(b), f"{path}: 键集合不同 {sorted(set(a) ^ set(b))}"
        for key in sorted(a):
            _assert_exact(a[key], b[key], f"{path}.{key}")
        return
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        assert isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)), \
            f"{path}: 类型不同"
        assert len(a) == len(b), f"{path}: 长度 {len(a)} != {len(b)}"
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            _assert_exact(left, right, f"{path}[{index}]")
        return
    assert type(a) is type(b), f"{path}: 类型 {type(a)} != {type(b)}"
    assert a == b, f"{path}: {a!r} != {b!r}"


def _start(index: int) -> str:
    return tb().TRAIN_CANDIDATE_STARTS[index]


def _boundary_and_final_states(tmp_path) -> dict:
    """连续路径与恢复路径的**边界**（批 2 前）与**最终**（批 2 后）完整状态。"""
    m = tb()
    obs_dim = _obs_dim()

    # --- 连续：批1 → 边界快照 → 批2 ---
    cp, co, cl, cg = _objects(obs_dim)
    m.run_single_batch(cp, co, cl, cg, start=_start(0), **_batch_kwargs())
    continuous_boundary = _full_state(cp, co, cl, cg)
    continuous_second = m.run_single_batch(
        cp, co, cl, cg, start=_start(1), **_batch_kwargs())
    continuous_final = _full_state(cp, co, cl, cg)

    # --- 恢复：批1 → 保存 → **全新对象** load → 边界快照 → 批2 ---
    rp, ro, rl, rg = _objects(obs_dim)
    m.run_single_batch(rp, ro, rl, rg, start=_start(0), **_batch_kwargs())
    ckpt = tmp_path / "after_batch1.pt"
    m.save_two_batch_checkpoint(
        ckpt, policy=rp, optimizer=ro, lagrangian=rl, generator=rg,
        next_start=_start(1), obs_dim=obs_dim, action_dim=21)

    p2, o2, l2, g2 = _objects(obs_dim)
    resumed = m.resume_two_batch_checkpoint(
        ckpt, policy=p2, optimizer=o2, lagrangian=l2, generator=g2,
        expected_obs_dim=obs_dim, expected_action_dim=21)
    resumed_boundary = _full_state(p2, o2, l2, g2)
    resumed_second = m.run_single_batch(
        p2, o2, l2, g2, start=resumed["next_start"], **_batch_kwargs())
    resumed_final = _full_state(p2, o2, l2, g2)

    return {
        "continuous_boundary": continuous_boundary,
        "resumed_boundary": resumed_boundary,
        "continuous_final": continuous_final,
        "resumed_final": resumed_final,
        "continuous_second": continuous_second,
        "resumed_second": resumed_second,
        "next_start": resumed["next_start"],
        "resumed_objects": (p2, o2, l2, g2),
    }


@needs_assets
def test_batch_boundary_state_is_exactly_restored(tmp_path):
    """**恢复边界对照**：`load` 完、**批 2 开始前**，四个对象的**完整状态**
    必须与连续路径在**批 1 结束时**的状态逐项精确一致。"""
    s = _boundary_and_final_states(tmp_path)

    _assert_exact(s["continuous_boundary"]["policy"], s["resumed_boundary"]["policy"],
                  "boundary.policy")
    _assert_exact(s["continuous_boundary"]["optimizer"],
                  s["resumed_boundary"]["optimizer"], "boundary.optimizer")
    _assert_exact(s["continuous_boundary"]["lagrangian"],
                  s["resumed_boundary"]["lagrangian"], "boundary.lagrangian")
    _assert_exact(s["continuous_boundary"]["generator"],
                  s["resumed_boundary"]["generator"], "boundary.generator")
    assert s["next_start"] == _start(1), "恢复必须给出**下一** origin"

    # 非空洞性：边界状态既不是初始状态，也不是最终状态
    initial_policy, *_ = _objects(_obs_dim())
    _assert_exact(initial_policy.state_dict()["log_std"],
                  initial_policy.state_dict()["log_std"], "sanity")
    assert not torch.equal(
        s["continuous_boundary"]["policy"]["log_std"],
        initial_policy.state_dict()["log_std"]), \
        "批 1 之后 log_std 必须已前进（否则边界对照无意义）"
    assert not torch.equal(
        s["continuous_boundary"]["policy"]["log_std"],
        s["continuous_final"]["policy"]["log_std"]), \
        "边界状态必须不同于最终状态（证明对照发生在批 2 **之前**）"
    # 非空洞性：optimizer 状态必须**真的**含逐参数张量（不是空 dict）
    assert s["continuous_boundary"]["optimizer"]["state"], \
        "边界 optimizer 状态不得为空"


@needs_assets
def test_final_state_after_batch2_is_exactly_equal(tmp_path):
    """**批 2 之后**：完整 optimizer / Lagrangian 状态与 policy / RNG 状态
    必须与连续路径逐项精确一致。"""
    s = _boundary_and_final_states(tmp_path)

    _assert_exact(s["continuous_final"]["policy"], s["resumed_final"]["policy"],
                  "final.policy")
    _assert_exact(s["continuous_final"]["optimizer"], s["resumed_final"]["optimizer"],
                  "final.optimizer")
    _assert_exact(s["continuous_final"]["lagrangian"],
                  s["resumed_final"]["lagrangian"], "final.lagrangian")
    _assert_exact(s["continuous_final"]["generator"],
                  s["resumed_final"]["generator"], "final.generator")

    # 过渡证据（不是替代）：loss 与参数变化量
    assert s["resumed_second"]["loss_total"] == pytest.approx(
        s["continuous_second"]["loss_total"], rel=1e-12)
    assert s["resumed_second"]["param_delta_norm"] == pytest.approx(
        s["continuous_second"]["param_delta_norm"], rel=1e-12)

    # 非空洞性：最终状态必须已从边界**前进**
    assert not torch.equal(s["continuous_boundary"]["policy"]["log_std"],
                           s["continuous_final"]["policy"]["log_std"])
    # Adam 的 step 计数必须真的推进到 2
    steps = {float(v["step"]) for v in s["continuous_final"]["optimizer"]["state"].values()
             if isinstance(v, dict) and "step" in v}
    assert steps == {2.0}, f"两次更新后 Adam step 应为 2，实际 {steps}"


@needs_assets
def test_boundary_restore_is_not_explainable_by_fresh_or_reseeded_objects(tmp_path):
    """**反空洞**：全新（未 load）对象的边界状态与恢复后的边界状态必须**不同**。"""
    s = _boundary_and_final_states(tmp_path)
    fresh_policy, fresh_optimizer, fresh_lagrangian, fresh_generator = \
        _objects(_obs_dim())
    fresh = _full_state(fresh_policy, fresh_optimizer, fresh_lagrangian,
                        fresh_generator)

    with pytest.raises(AssertionError):
        _assert_exact(s["resumed_boundary"], fresh, "boundary")
    assert not torch.equal(s["resumed_boundary"]["policy"]["log_std"],
                           fresh["policy"]["log_std"])
    assert s["resumed_boundary"]["optimizer"]["state"], "恢复的 optimizer 必须非空"
    assert not fresh["optimizer"]["state"], "全新 optimizer 的 state 应为空"
    _ = s["resumed_objects"]


@needs_assets
def test_full_state_comparator_detects_a_single_element_difference(tmp_path):
    """**比较器灵敏度**（防证据空洞）：`_assert_exact` 必须抓到**单元素**差异。

    否则上面「完整状态逐项精确一致」的结论可能只是因为比较器太松。
    """
    s = _boundary_and_final_states(tmp_path)
    base = s["continuous_boundary"]

    # 单元素扰动：policy 的 log_std 第 0 个元素 +1e-9
    import copy as _copy

    perturbed_policy = {k: v.clone() for k, v in base["policy"].items()}
    perturbed_policy["log_std"][0] += 1e-9
    with pytest.raises(AssertionError, match="张量元素不逐位相等"):
        _assert_exact(base["policy"], perturbed_policy, "policy")

    # 键集合差异
    missing = {k: v for k, v in base["policy"].items() if k != "log_std"}
    with pytest.raises(AssertionError, match="键集合不同"):
        _assert_exact(base["policy"], missing, "policy")

    # Adam 的 step 标量差异（嵌套在 optimizer.state.<i>.step）
    perturbed_optimizer = _copy.deepcopy(base["optimizer"])
    index = next(iter(perturbed_optimizer["state"]))
    perturbed_optimizer["state"][index]["step"] = (
        perturbed_optimizer["state"][index]["step"] + 1)
    with pytest.raises(AssertionError, match="张量元素不逐位相等"):
        _assert_exact(base["optimizer"], perturbed_optimizer, "optimizer")

    # Lagrangian 的量级差异（嵌套在 constraints.<name>.multiplier）
    perturbed_lagrangian = _copy.deepcopy(base["lagrangian"])
    if perturbed_lagrangian.get("constraints"):
        name = sorted(perturbed_lagrangian["constraints"])[0]
        perturbed_lagrangian["constraints"][name]["multiplier"] = (
            perturbed_lagrangian["constraints"][name]["multiplier"] + 1e-9)
        with pytest.raises(AssertionError):
            _assert_exact(base["lagrangian"], perturbed_lagrangian, "lagrangian")

    # RNG 状态差异
    perturbed_generator = base["generator"].clone()
    perturbed_generator[0] = (int(perturbed_generator[0]) + 1) % 256
    with pytest.raises(AssertionError, match="张量元素不逐位相等"):
        _assert_exact(base["generator"], perturbed_generator, "generator")

    # 自身比较必须通过（非空洞性的另一半）
    _assert_exact(base, {k: v for k, v in base.items()}, "self")
