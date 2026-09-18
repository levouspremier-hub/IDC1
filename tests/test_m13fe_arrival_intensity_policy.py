"""M1.3f-e-a：B6 arrival-intensity policy 冻结的**先红**回归。

改前缺陷（本文件在实现前必须为红）：

- `scenario.arrival_intensity_policy` 模块尚不存在（`ModuleNotFoundError`）；
- `scripts.materialize_arrival_intensity_policy` 物化器尚不存在；
- canonical policy manifest `data/manifest/m13f_arrival_intensity_policy_v1.json`
  尚不存在。

本卡**只冻结 policy**：不物化新版 arrival、不生成新 exogenous、不接线 env/train。"""

import hashlib
import importlib
import json
import pathlib
import shutil

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
POLICY_MODULE = "scenario.arrival_intensity_policy"
MATERIALIZER_MODULE = "scripts.materialize_arrival_intensity_policy"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
POLICY_PATH = REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json"

TRAIN_ROW_END = 10224


def policy_module():
    return importlib.import_module(POLICY_MODULE)


def splits_module():
    return importlib.import_module("scenario.splits")


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _assets_present() -> bool:
    return all(
        p.exists() for p in (CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST,
                             POLICY_PATH)
    )


needs_assets = pytest.mark.skipif(
    not _assets_present(), reason="真实冻结资产不在本机"
)


def _copy(src, dst) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _synced_temp_chain(tmp_path, *, mutate=None) -> dict:
    """构造**完整自洽**的临时 canonical 链（parquet / canonical manifest / split）。

    `mutate(frame)` 可选地就地修改 canonical 帧；随后 split manifest 的 parquet
    hash 与（仅 train 段的）train-only 统计会被重新对齐，因此公开入口看来完全自洽。
    """
    root = tmp_path / "chain"
    cano_dir = root / "data/processed/singapore_2024"
    man_dir = root / "data/manifest"
    cano_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(CANONICAL_PARQUET)
    if mutate is not None:
        mutate(frame)
    parquet = cano_dir / "half_hour.parquet"
    frame.to_parquet(parquet, index=False)

    canonical_manifest = man_dir / "singapore_2024_half_hour.json"
    _copy(CANONICAL_MANIFEST, canonical_manifest)
    payload = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    payload["output_parquet_sha256"] = _sha256(parquet)
    canonical_manifest.write_text(json.dumps(payload))

    split_manifest = man_dir / "singapore_2024_splits.json"
    _copy(SPLIT_MANIFEST, split_manifest)
    splits = json.loads(split_manifest.read_text(encoding="utf-8"))
    splits["canonical_parquet_sha256"] = _sha256(parquet)
    splits["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    splits["canonical_parquet_path"] = "<external>/half_hour.parquet"
    splits["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
    splits["train_only_statistics_source"]["canonical_parquet_sha256"] = _sha256(parquet)
    if mutate is not None:
        splits["train_only_statistics"] = splits_module().train_only_statistics(
            frame.iloc[: splits["splits"]["train"]["row_end_exclusive"]])
    split_manifest.write_text(json.dumps(splits))

    return {
        "canonical_parquet_path": parquet,
        "canonical_manifest_path": canonical_manifest,
        "split_manifest_path": split_manifest,
    }


def _frozen_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _tampered_policy(tmp_path, monkeypatch, mutate):
    """把 policy 复制到临时「canonical」目录并篡改，monkeypatch 私有 resolver。"""
    m = policy_module()
    temp_dir = tmp_path / "canonical_manifest_dir"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / "m13f_arrival_intensity_policy_v1.json"
    shutil.copy(POLICY_PATH, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload))
    monkeypatch.setattr(m, "_canonical_policy_dir", lambda: temp_dir)
    return m, target


# --- 1. canonical policy 存在且 schema 正确 ------------------------------------

@needs_assets
def test_canonical_policy_exists_and_loads():
    m = policy_module()
    policy = _frozen_policy()
    assert policy["schema"] == m.POLICY_SCHEMA
    assert policy["decision_id"] == "B6-INTENSITY"
    assert policy["approved_on"] == "2026-09-18"
    assert policy["source_kind"] == "modeled_scenario"
    # 逐项等于模块冻结常量
    for key in m.POLICY_MANIFEST_KEYS:
        assert key in policy


# --- 2. train max temperature 精确来自 [0,10224)，validation/test 不影响 ---------

@needs_assets
def test_train_max_temperature_derives_from_train_only(tmp_path):
    m = policy_module()

    def mutate_validation_and_test(frame):
        frame.loc[TRAIN_ROW_END:, "temperature_deg_c"] = 100.0

    chain = _synced_temp_chain(tmp_path, mutate=mutate_validation_and_test)
    manifest = m.build_b6_policy_manifest(
        **chain, frozen_at_utc="2026-09-18T00:00:00+00:00")
    assert manifest["train_max_temperature_deg_c"] == 33.2
    assert manifest["train_range"] == {
        "split": "train", "row_start": 0, "row_end_exclusive": TRAIN_ROW_END,
    }
    assert manifest["temperature_field"] == "temperature_deg_c"


# --- 3. server_seed 必须为 0 ----------------------------------------------------

@needs_assets
def test_server_seed_is_zero():
    assert _frozen_policy()["server_seed"] == 0


# --- 4. 正式容量 79.985，记录未截断值与向下截断规则 ------------------------------

@needs_assets
def test_declared_capacity_records_unrounded_and_rounding():
    m = policy_module()
    policy = _frozen_policy()
    assert policy["declared_capacity_work_per_hour"] == 79.985
    assert abs(policy["unrounded_capacity_work_per_hour"] - 79.985193) < 1e-4
    assert policy["capacity_rounding_rule"] == "truncate_down_to_3_decimal_places"
    assert m.truncate_down(policy["unrounded_capacity_work_per_hour"], 3) == 79.985
    assert m.declared_capacity_from(policy["unrounded_capacity_work_per_hour"]) == 79.985


# --- 5. rho_target = 0.80 ------------------------------------------------------

@needs_assets
def test_rho_target_is_zero_point_eight():
    assert _frozen_policy()["rho_target"] == 0.80


# --- 6. 63.988 work/hour 与 31.994 work/half-hour 关系正确 ----------------------

@needs_assets
def test_main_rate_and_amount_relationship():
    policy = _frozen_policy()
    declared = policy["declared_capacity_work_per_hour"]
    rho = policy["rho_target"]
    delta = policy["delta_t_hours"]
    assert policy["main_expected_rate_work_per_hour"] == declared * rho
    assert policy["main_expected_rate_work_per_hour"] == 63.988
    assert policy["main_expected_amount_work_per_half_hour"] == \
        policy["main_expected_rate_work_per_hour"] * delta
    assert policy["main_expected_amount_work_per_half_hour"] == 31.994
    assert policy["delta_t_hours"] == 0.5


# --- 7. rho_realized 只能 diagnostic，不得进入 target 计算 -----------------------

@needs_assets
def test_rho_realized_is_diagnostic_only():
    policy = _frozen_policy()
    assert policy["rho_realized_purpose"] == "diagnostic_only"
    assert policy["forbid_realization_feedback"] is True
    assert policy["realization_seed"] == 20240916
    # main rate 只由 declared × rho_target 决定，与 realization 无关
    assert policy["main_expected_rate_work_per_hour"] == \
        policy["declared_capacity_work_per_hour"] * policy["rho_target"]
    assert "rho_realized" not in policy or \
        policy.get("rho_realized") in (None, "diagnostic_only")


# --- 8. 1000 只标 stress_candidate ----------------------------------------------

@needs_assets
def test_old_1000_is_stress_candidate():
    policy = _frozen_policy()
    assert policy["stress_candidate"] == "1000 work-units/half-hour"
    assert "1000" in policy["stress_candidate"]
    assert policy["sensitivity_levels"] == "none"


# --- 9. 不得声称 empirical workload --------------------------------------------

@needs_assets
def test_no_empirical_workload_claim():
    policy = _frozen_policy()
    assert policy["empirical_workload_claim"] is False
    assert policy["source_kind"] == "modeled_scenario"
    assert policy["azure_contribution"] == "shape_only"


# --- 10. canonical-only 与篡改拒绝 ----------------------------------------------

@needs_assets
def test_non_canonical_path_is_rejected(tmp_path):
    m = policy_module()
    copied = tmp_path / "copied_policy.json"
    shutil.copy(POLICY_PATH, copied)
    with pytest.raises(m.ArrivalIntensityPolicyError):
        m.load_verified_b6_policy(copied)


@needs_assets
def test_symlink_path_is_rejected(tmp_path):
    m = policy_module()
    link = tmp_path / "link_policy.json"
    link.symlink_to(POLICY_PATH)
    with pytest.raises(m.ArrivalIntensityPolicyError):
        m.load_verified_b6_policy(link)


@needs_assets
def test_forged_revision_is_rejected(tmp_path, monkeypatch):
    m, target = _tampered_policy(
        tmp_path, monkeypatch, lambda p: p.update(materializer_revision="0" * 40))
    with pytest.raises(m.ArrivalIntensityPolicyError):
        m.load_verified_b6_policy(target)


@needs_assets
def test_tampered_hash_is_rejected(tmp_path, monkeypatch):
    m, target = _tampered_policy(
        tmp_path, monkeypatch, lambda p: p.update(canonical_parquet_sha256="0" * 64))
    with pytest.raises(m.ArrivalIntensityPolicyError):
        m.load_verified_b6_policy(target)


# --- 11. loader 不得 fallback 到旧 policy ---------------------------------------

@needs_assets
def test_loader_has_no_fallback(tmp_path, monkeypatch):
    m = policy_module()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(m, "_canonical_policy_dir", lambda: empty_dir)
    with pytest.raises(m.ArrivalIntensityPolicyError):
        m.load_verified_b6_policy(empty_dir / "m13f_arrival_intensity_policy_v1.json")


# --- 12. materializer 不接受任意 out-dir ----------------------------------------

@needs_assets
def test_materializer_rejects_out_dir_and_manifest_path():
    materializer = importlib.import_module(MATERIALIZER_MODULE)
    with pytest.raises(SystemExit):
        materializer.main(["--out-dir", "/tmp/anywhere"])
    with pytest.raises(SystemExit):
        materializer.main(["--manifest-path", "/tmp/other.json"])


# --- 13. --verify 连续 3 次不改 bytes/hash/mtime 且无临时文件 --------------------

@needs_assets
def test_verify_is_idempotent():
    materializer = importlib.import_module(MATERIALIZER_MODULE)
    before = POLICY_PATH.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()
    before_mtime = POLICY_PATH.stat().st_mtime_ns
    for _ in range(3):
        assert materializer.main(["--verify"]) == 0
    assert POLICY_PATH.read_bytes() == before
    assert _sha256(POLICY_PATH) == before_sha
    assert POLICY_PATH.stat().st_mtime_ns == before_mtime
    leftovers = [
        p for p in POLICY_PATH.parent.iterdir()
        if p.name.startswith(".m13f_arrival_intensity_policy_v1.json.")
    ]
    assert leftovers == []
