"""M6-P1：受控短跑链路（train-only checkpoint → 评估 → runs 产物）。

需要本机冻结上游资产；资产缺失时整体 skip。
**只证明契约与指标计算可用**，不断言任何算法优劣。
"""

import importlib
import json
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_ID = "m6p1_controlled_short_run_test"

_UPSTREAM = (
    REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
    REPO_ROOT / "data/manifest/singapore_2024_half_hour.json",
    REPO_ROOT / "data/manifest/singapore_2024_splits.json",
    REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json",
    REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json",
    REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v3.parquet",
    REPO_ROOT / "data/manifest/singapore_2024_exogenous_v3.json",
    REPO_ROOT / "data/manifest/m13f_materialization_sources_v4.json",
    REPO_ROOT / "configs/frozen_refs/refs_v4.json",
    REPO_ROOT / "data/manifest/formal_splits_v5/train.json",
    REPO_ROOT / "data/manifest/m13g_arrival_mapper_v1.json",
    REPO_ROOT / "configs/release/idc_formal_env_release_v1.json",
)

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _UPSTREAM), reason="真实冻结上游资产不在本机")


def _controlled_run():
    return importlib.import_module("evaluation.controlled_run")


def _eval_input():
    return importlib.import_module("checkpointing.eval_input")


def _sources_module():
    return importlib.import_module("evaluation.sources")


# =============================================================================
# 1. 确定性动作与策略语义一致（不得出现第二套动作语义）
# =============================================================================

def test_deterministic_action_matches_the_policy_squash_semantics():
    from safe_rl_v2.policy import _squash

    controlled = _controlled_run()
    policy = controlled.initialize_policy(obs_dim=8)
    obs = np.linspace(-1.0, 1.0, 8, dtype=np.float32)

    got = controlled.deterministic_action(policy, obs)
    with torch.no_grad():
        want = _squash(policy.act_mean(torch.as_tensor(obs)), policy.action_dim).numpy()

    assert got.shape == (21,)
    assert np.allclose(got, want, rtol=0.0, atol=1e-6), \
        "确定性动作必须与 safe_rl_v2.policy 的有界化语义逐位一致"
    assert np.all(got[:20] >= 0.0) and np.all(got[:20] <= 1.0)
    assert -1.0 <= got[20] <= 1.0


def test_initialize_policy_is_seed_reproducible_and_does_not_touch_global_rng():
    controlled = _controlled_run()
    torch.manual_seed(1234)
    before = torch.rand(3)
    torch.manual_seed(1234)

    p1 = controlled.initialize_policy(obs_dim=8)
    after = torch.rand(3)
    p2 = controlled.initialize_policy(obs_dim=8)

    assert torch.equal(before, after), "固定种子初始化不得改动全局 RNG 状态"
    for key, value in p1.state_dict().items():
        assert torch.equal(value, p2.state_dict()[key]), key


# =============================================================================
# 2. 端到端链路：checkpoint → 评估 → runs 产物
# =============================================================================

@pytest.fixture(scope="module")
def controlled_run_dir(tmp_path_factory):
    controlled = _controlled_run()
    run_id = RUN_ID + "_" + tmp_path_factory.getbasetemp().name
    base = tmp_path_factory.mktemp("runs")
    exit_code = controlled.main(
        ["--run-id", run_id, "--base-dir", str(base), "--steps", "2"])
    assert exit_code == 0, "受控短跑必须成功退出"
    return run_id, base / run_id


def test_checkpoint_is_written_as_a_controlled_eval_input(controlled_run_dir):
    _run_id, run_dir = controlled_run_dir
    checkpoint_path = run_dir / "eval_input_checkpoint.pt"
    assert checkpoint_path.exists()

    ckpt = _eval_input().load_evaluation_checkpoint(checkpoint_path)
    assert ckpt.is_controlled_short_run is True
    assert ckpt.artifact_role == _eval_input().CONTROLLED_ROLE
    assert ckpt.action_mode == "deterministic_mean"
    assert ckpt.train_split == "train"
    assert ckpt.seeds == {"task": 0, "server": 1, "forecast": 300000}
    assert ckpt.envelope.action_dim == 21
    assert ckpt.envelope.schema_hash == _eval_input().EVAL_INPUT_SCHEMA
    assert len(ckpt.sources) == len(_eval_input().EVAL_INPUT_REQUIRED_SOURCE_ROLES)
    # 来源对**当前**资产逐项成立
    _sources_module().verify_evaluation_input_sources(ckpt)


def test_run_artifacts_follow_the_repo_run_spec(controlled_run_dir):
    _run_id, run_dir = controlled_run_dir
    for name in ("config.yaml", "metrics.parquet", "report.json", "manifest.json"):
        assert (run_dir / name).exists(), f"缺少 runs 产物 {name}"
    assert (run_dir / "figures").is_dir()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["checkpoint_role"] == _eval_input().CONTROLLED_ROLE
    assert len(manifest["checkpoint_sha256"]) == 64


def test_report_marks_undetermined_service_and_unevaluated_methods(controlled_run_dir):
    _run_id, run_dir = controlled_run_dir
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    evaluation = report["evaluation"]
    assert evaluation["service_qualified"] is None, "项目标准未冻结 ⇒ 未判定"
    assert evaluation["service_standard_id"] is None
    assert evaluation["action_mode"] == "deterministic_mean"
    assert evaluation["seed"] == 0
    assert evaluation["failure_classification"] is None

    matrix = report["planned_method_matrix"]
    assert [row["method"] for row in matrix] == list(_controlled_run().PLANNED_METHODS)
    assert all(row["status"] == "not_evaluated" for row in matrix), \
        "五类正式方法一个都没评估 ⇒ 全部「未评估」"
    assert all(row["purchase_cost_sgd"] is None for row in matrix)

    assert report["claims"]["trained"] is False
    assert report["claims"]["performance_evaluated"] is False
    assert report["split"] == "train"


def test_report_records_the_metric_sources(controlled_run_dir):
    _run_id, run_dir = controlled_run_dir
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    evaluation = report["evaluation"]

    assert evaluation["purchase_cost_sgd"] >= 0.0
    assert evaluation["bess_degradation_cost_sgd"] >= 0.0
    assert evaluation["grid_energy_kwh"] >= 0.0
    assert evaluation["carbon_kg_co2e"] >= 0.0
    for key in ("pv", "wind"):
        block = evaluation[key]
        assert block["available_kwh"] >= block["used_kwh"] - 1e-9
    assert evaluation["physical"]["energy_conservation_violations"] == 0
    assert evaluation["correction"] is not None, "受控短跑带修正器 ⇒ 修正指标必须存在"
    assert evaluation["correction"]["timeout_count"] >= 0


# =============================================================================
# 3. 来源绑定：篡改必须被拒绝
# =============================================================================

def test_tampered_source_hash_is_rejected(controlled_run_dir, tmp_path):
    """篡改副本的 hash 必须拒绝（在**副本**上做，不动共享夹具）。"""
    import shutil

    _run_id, run_dir = controlled_run_dir
    copy = tmp_path / "tampered.pt"
    shutil.copyfile(run_dir / "eval_input_checkpoint.pt", copy)

    payload = torch.load(str(copy), weights_only=False)
    payload["metadata"]["sources"][0]["sha256"] = "b" * 64
    torch.save(payload, str(copy))

    ckpt = _eval_input().load_evaluation_checkpoint(copy)
    with pytest.raises(ValueError, match="SHA-256"):
        _sources_module().verify_evaluation_input_sources(ckpt)


def test_repointed_source_role_is_rejected():
    """把角色指向副本 / 别名必须拒绝（角色 → 规范路径是唯一绑定）。"""
    sources = _sources_module()
    digests = list(sources.canonical_source_digests())
    bad = digests[0].model_copy(update={"logical_path": "tmp/copy_of_asset.json"})

    class _Stub:
        pass

    stub = _Stub()
    stub.sources = (bad, *digests[1:])
    with pytest.raises(ValueError, match="逻辑路径"):
        sources.verify_evaluation_input_sources(stub)


def test_source_role_set_must_match_exactly():
    sources = _sources_module()
    digests = list(sources.canonical_source_digests())

    class _Stub:
        pass

    stub = _Stub()
    stub.sources = tuple(digests[:-1])
    with pytest.raises(ValueError, match="来源角色集合不符"):
        sources.verify_evaluation_input_sources(stub)
