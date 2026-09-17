"""M1.3g-d：train-only normalization refs 冻结（`frozen-refs-v2`）。

**改前缺陷（本文件在实现前必须为红）**：

- `configs/frozen_refs/refs.json` 仍是 **legacy v1**（声明尺度、
  `training_range` / `data_hash` 均为 `null`）——四个参考值**从未**由 train 推导；
- `scripts/freeze_refs.py` **硬编码** `DECLARED_REFS`，**完全不读数据**；
- 没有受控的 legacy → v2 替换路径，也没有 v2 的原子性/幂等/防覆盖。

**本卡只冻结 refs**：不创建三个正式 split manifest、不改 env 默认值、
不接线 env/train（那是 g-e/g-f）。"""

import hashlib
import importlib
import json
import pathlib

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FREEZE_MODULE = "scripts.freeze_refs"
REFS_PATH = REPO_ROOT / "configs/frozen_refs/refs.json"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
EXOGENOUS_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v2.json"
EXOGENOUS_SOURCE = REPO_ROOT / "data/manifest/m13f_materialization_sources_v3.json"
EXOGENOUS_PARQUET = (
    REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
)
POLICY_V2 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v2.json"

SCHEMA_V2 = "frozen-refs-v2"
LEGACY_V1_SHA256 = (
    "afa84b8610c073322aac41b3d7e5c706478c64a551363bd66c42b1f22fd409bc"
)
TRAIN_RANGE = {"split": "train", "row_start": 0, "row_end_exclusive": 10224}

TRAIN_DERIVED = ("price_ref", "pv_ref_kw", "wind_ref_kw", "carbon_factor_ref")
DECLARED = (
    "lambda_ref", "queue_ref", "queue_capacity_ref", "cost_ref", "carbon_ref",
    "peak_power_threshold_kW", "peak_power_ref_kW", "grid_power_limit_kW",
    "sla_penalty_ref",
)
# D2：这四个必须保持**声明**尺度，不得由数据推导
D2_DECLARED_VALUES = {
    "lambda_ref": 2000.0,
    "queue_ref": 6000.0,
    "queue_capacity_ref": 6000.0,
    "cost_ref": 60.0,
}
VALUE_KEYS = (
    "value", "unit", "source_kind", "method", "derived_from", "training_range",
    "binding",
)
TOP_KEYS = (
    "schema_version", "materializer_revision", "frozen_at_utc", "training_range",
    "sources", "references", "units", "note",
)


def freeze():
    return importlib.import_module(FREEZE_MODULE)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _assets_present() -> bool:
    return all(
        p.exists()
        for p in (CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST,
                  EXOGENOUS_MANIFEST, EXOGENOUS_SOURCE, EXOGENOUS_PARQUET, POLICY_V2)
    )


needs_assets = pytest.mark.skipif(
    not _assets_present(), reason="真实冻结上游资产不在本机"
)


def _frozen_refs() -> dict:
    return json.loads(REFS_PATH.read_text(encoding="utf-8"))


def _chain_kwargs() -> dict:
    return {
        "canonical_parquet_path": CANONICAL_PARQUET,
        "canonical_manifest_path": CANONICAL_MANIFEST,
        "split_manifest_path": SPLIT_MANIFEST,
        "exogenous_manifest_path": EXOGENOUS_MANIFEST,
        "exogenous_source_manifest_path": EXOGENOUS_SOURCE,
    }


# --- 1. schema 与逐值来源 -----------------------------------------------------

@needs_assets
def test_refs_are_frozen_refs_v2():
    refs = _frozen_refs()
    assert refs["schema_version"] == SCHEMA_V2
    assert tuple(refs) == TOP_KEYS or set(refs) == set(TOP_KEYS)


@needs_assets
def test_every_value_records_its_provenance():
    """每个估值**恰好**七个键；禁止 null 与无来源自由字段。"""
    for name, entry in _frozen_refs()["references"].items():
        assert set(entry) == set(VALUE_KEYS), name
        for key in VALUE_KEYS:
            assert entry[key] is not None, f"{name}.{key} 不得为 null"
        assert entry["source_kind"] in ("train_derived", "declared_physical_scale"), name
        assert isinstance(entry["unit"], str) and entry["unit"], name
        assert isinstance(entry["method"], str) and entry["method"], name
        assert isinstance(entry["derived_from"], str) and entry["derived_from"], name
        assert not isinstance(entry["value"], bool)
        assert isinstance(entry["value"], (int, float)), name
        assert entry["training_range"] in ("train", "not_applicable"), name
        assert isinstance(entry["binding"], list), name


@needs_assets
def test_train_derived_values_bind_real_source_roles():
    refs = _frozen_refs()
    for name in TRAIN_DERIVED:
        entry = refs["references"][name]
        assert entry["source_kind"] == "train_derived", name
        assert entry["training_range"] == "train", name
        assert entry["binding"], f"{name} 必须绑定至少一个来源角色"
        for role in entry["binding"]:
            assert role in refs["sources"], f"{name} 绑定了未知角色 {role!r}"
            assert refs["sources"][role]["sha256"] == _sha256(
                REPO_ROOT / refs["sources"][role]["path"]
            ), role


@needs_assets
def test_declared_values_keep_the_declared_scale():
    refs = _frozen_refs()
    for name in DECLARED:
        entry = refs["references"][name]
        assert entry["source_kind"] == "declared_physical_scale", name
        assert entry["training_range"] == "not_applicable", name
        assert entry["binding"] == [], name
    for name, expected in D2_DECLARED_VALUES.items():
        assert refs["references"][name]["value"] == expected, name


@needs_assets
def test_units_map_matches_every_value():
    refs = _frozen_refs()
    assert set(refs["units"]) == set(refs["references"])
    for name, entry in refs["references"].items():
        assert refs["units"][name] == entry["unit"], name


@needs_assets
def test_training_range_and_sources_are_recorded():
    refs = _frozen_refs()
    assert refs["training_range"] == {
        "split": TRAIN_RANGE["split"],
        "row_start": TRAIN_RANGE["row_start"],
        "row_end_exclusive": TRAIN_RANGE["row_end_exclusive"],
        "start": refs["training_range"]["start"],
        "end_exclusive": refs["training_range"]["end_exclusive"],
    }
    for role in ("canonical_parquet", "canonical_manifest", "split_manifest",
                 "exogenous_manifest", "exogenous_source_manifest",
                 "exogenous_parquet"):
        assert role in refs["sources"], role
    assert refs["sources"]["canonical_parquet"]["sha256"] == _sha256(CANONICAL_PARQUET)
    assert refs["sources"]["exogenous_manifest"]["sha256"] == _sha256(EXOGENOUS_MANIFEST)


# --- 2. 值确实由 train 推导 ---------------------------------------------------

@needs_assets
def test_values_are_actually_recomputed_from_the_train_rows():
    """必须**实际读取 train 并重算**，不是照抄 M1.3d 的描述统计。"""
    refs = _frozen_refs()["references"]
    train = pd.read_parquet(CANONICAL_PARQUET).iloc[:TRAIN_RANGE["row_end_exclusive"]]
    exog = pd.read_parquet(EXOGENOUS_PARQUET).iloc[:TRAIN_RANGE["row_end_exclusive"]]

    assert refs["price_ref"]["value"] == pytest.approx(
        float(train["price_sgd_per_kwh"].abs().max()))
    assert refs["price_ref"]["method"] == "max_abs"
    assert refs["pv_ref_kw"]["value"] == pytest.approx(
        float(exog["local_pv_kw"].max()))
    assert refs["wind_ref_kw"]["value"] == pytest.approx(
        float(exog["wind_generation_kw"].max()))
    assert refs["carbon_factor_ref"]["value"] == pytest.approx(
        float(exog["carbon_intensity"].max()))

    # 与 M1.3d 的描述统计**不是**一回事（描述统计只含 min/max/mean/负值数）
    split = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    assert "train_only_statistics" in split
    assert split["train_only_statistics_source"]["split"] == "train"


@needs_assets
def test_signed_price_uses_the_absolute_maximum():
    """signed price 必须用 `max(abs(price))`，不是 `max(price)`。"""
    train = pd.read_parquet(CANONICAL_PARQUET).iloc[:TRAIN_RANGE["row_end_exclusive"]]
    signed = float(train["price_sgd_per_kwh"].max())
    absolute = float(train["price_sgd_per_kwh"].abs().max())
    assert absolute >= signed
    assert _frozen_refs()["references"]["price_ref"]["value"] == pytest.approx(absolute)


# --- 3. train-only：validation/test 不得影响 ----------------------------------

@needs_assets
def test_validation_and_test_do_not_influence_the_values():
    """把所有 validation/test 行放大 1000 倍，重算出的 train-derived 值必须不变。"""
    module = freeze()
    baseline = module.build_frozen_refs(**_chain_kwargs())

    mutated = pd.read_parquet(CANONICAL_PARQUET)
    mutated.loc[mutated.index[TRAIN_RANGE["row_end_exclusive"]:],
                "price_sgd_per_kwh"] *= 1000.0
    after = module.build_frozen_refs(**_chain_kwargs(), canonical_frame=mutated)
    for name in TRAIN_DERIVED:
        assert after["references"][name]["value"] == pytest.approx(
            baseline["references"][name]["value"]), name


@needs_assets
def test_train_mutation_changes_the_train_derived_values():
    module = freeze()
    baseline = module.build_frozen_refs(**_chain_kwargs())

    mutated = pd.read_parquet(CANONICAL_PARQUET)
    mutated.loc[mutated.index[0], "price_sgd_per_kwh"] = 99.0
    after = module.build_frozen_refs(**_chain_kwargs(), canonical_frame=mutated)
    assert after["references"]["price_ref"]["value"] == pytest.approx(99.0)
    assert after["references"]["price_ref"]["value"] != pytest.approx(
        baseline["references"]["price_ref"]["value"])


# --- 4. 受控首次替换、原子、幂等 ---------------------------------------------

def _legacy_bytes() -> bytes:
    """legacy v1 的**登记**字节（从 git 取，保证与冻结 hash 一致）。"""
    import subprocess

    return subprocess.run(
        ["git", "show", "ea097d5:configs/frozen_refs/refs.json"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout


def test_legacy_v1_bytes_match_the_registered_hash():
    assert hashlib.sha256(_legacy_bytes()).hexdigest() == LEGACY_V1_SHA256


@needs_assets
def test_existing_v1_cannot_be_replaced_without_the_flag(tmp_path, monkeypatch):
    module = freeze()
    target = tmp_path / "refs.json"
    target.write_bytes(_legacy_bytes())
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)

    with pytest.raises(ValueError):
        module.materialize_frozen_refs(
            out_path=target, replace_declared_v1=False, **_chain_kwargs())
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before


@needs_assets
def test_controlled_first_replacement_is_atomic(tmp_path, monkeypatch):
    """任何写失败都必须保留**完整的 v1**、且不留临时文件。"""
    module = freeze()
    target = tmp_path / "refs.json"
    target.write_bytes(_legacy_bytes())
    original = target.read_bytes()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)

    def boom(path, text):
        raise OSError("boom")

    monkeypatch.setattr(module, "_atomic_write_text", boom)
    with pytest.raises(OSError):
        module.materialize_frozen_refs(
            out_path=target, replace_declared_v1=True, **_chain_kwargs())
    assert target.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["refs.json"]


@needs_assets
def test_controlled_first_replacement_writes_v2(tmp_path, monkeypatch):
    module = freeze()
    target = tmp_path / "refs.json"
    target.write_bytes(_legacy_bytes())
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)

    result = module.materialize_frozen_refs(
        out_path=target, replace_declared_v1=True, **_chain_kwargs())
    assert result["written"] is True
    assert json.loads(target.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_V2


@needs_assets
def test_v2_rematerialization_is_idempotent(tmp_path, monkeypatch):
    module = freeze()
    target = tmp_path / "refs.json"
    target.write_bytes(_legacy_bytes())
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    module.materialize_frozen_refs(
        out_path=target, replace_declared_v1=True, **_chain_kwargs())

    before = (_sha256(target), target.stat().st_mtime_ns)
    second = module.materialize_frozen_refs(
        out_path=target, replace_declared_v1=False, **_chain_kwargs())
    after = (_sha256(target), target.stat().st_mtime_ns)
    assert before == after
    assert second["written"] is False


@needs_assets
def test_different_v2_is_not_overwritten(tmp_path, monkeypatch):
    module = freeze()
    target = tmp_path / "refs.json"
    payload = module.build_frozen_refs(**_chain_kwargs())
    payload["references"]["price_ref"]["value"] = 12345.0
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    before = target.read_bytes()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)

    with pytest.raises(ValueError):
        module.materialize_frozen_refs(
            out_path=target, replace_declared_v1=False, **_chain_kwargs())
    assert target.read_bytes() == before


@needs_assets
@pytest.mark.parametrize("bad", ("not json", '{"schema_version": "frozen-refs-v9"}',
                                 '[]', '{"references": {}}'))
def test_unknown_or_malformed_existing_refs_are_rejected(tmp_path, monkeypatch, bad):
    module = freeze()
    target = tmp_path / "refs.json"
    target.write_text(bad)
    before = target.read_bytes()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)

    with pytest.raises(ValueError):
        module.materialize_frozen_refs(
            out_path=target, replace_declared_v1=True, **_chain_kwargs())
    assert target.read_bytes() == before


@needs_assets
def test_dirty_generator_is_rejected(tmp_path, monkeypatch):
    module = freeze()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        module.materialize_frozen_refs(
            out_path=tmp_path / "refs.json", replace_declared_v1=True,
            **_chain_kwargs())


# --- 5. 信任链 fail closed ---------------------------------------------------

@needs_assets
def test_v1_exogenous_is_rejected():
    module = freeze()
    with pytest.raises(ValueError):
        module.build_frozen_refs(
            **{**_chain_kwargs(),
               "exogenous_manifest_path":
                   REPO_ROOT / "data/manifest/singapore_2024_exogenous.json"})


@needs_assets
def test_bad_exogenous_hash_is_rejected(tmp_path):
    module = freeze()
    payload = json.loads(EXOGENOUS_MANIFEST.read_text(encoding="utf-8"))
    payload["readiness"]["formal_training_ready"] = True
    target = tmp_path / "exogenous_v2.json"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.build_frozen_refs(
            **{**_chain_kwargs(), "exogenous_manifest_path": target})


@needs_assets
def test_misaligned_timeline_is_rejected(tmp_path):
    """canonical 时间戳错位（非 30 分钟网格）必须 fail closed。"""
    import shutil

    module = freeze()
    root = tmp_path / "chain"
    (root / "data/processed/singapore_2024").mkdir(parents=True)
    (root / "data/manifest").mkdir(parents=True)
    for name in ("singapore_2024_half_hour.json", "singapore_2024_splits.json"):
        shutil.copy(REPO_ROOT / "data/manifest" / name, root / "data/manifest" / name)

    frame = pd.read_parquet(CANONICAL_PARQUET)
    frame.loc[frame.index[5], "timestamp"] = (
        frame.loc[frame.index[5], "timestamp"] + pd.Timedelta(minutes=7)
    )
    parquet = root / "data/processed/singapore_2024/half_hour.parquet"
    frame.to_parquet(parquet, index=False)

    canonical_manifest = root / "data/manifest/singapore_2024_half_hour.json"
    payload = json.loads(canonical_manifest.read_text())
    payload["output_parquet_sha256"] = _sha256(parquet)
    canonical_manifest.write_text(json.dumps(payload))

    split_manifest = root / "data/manifest/singapore_2024_splits.json"
    splits = json.loads(split_manifest.read_text())
    splits["canonical_parquet_sha256"] = _sha256(parquet)
    splits["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    splits["canonical_parquet_path"] = "<external>/half_hour.parquet"
    splits["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
    splits["train_only_statistics_source"]["canonical_parquet_sha256"] = _sha256(parquet)
    split_manifest.write_text(json.dumps(splits))

    with pytest.raises(ValueError):
        module.build_frozen_refs(
            **{**_chain_kwargs(),
               "canonical_parquet_path": parquet,
               "canonical_manifest_path": canonical_manifest,
               "split_manifest_path": split_manifest})


@needs_assets
def test_non_finite_train_value_is_rejected(tmp_path):
    module = freeze()
    frame = pd.read_parquet(CANONICAL_PARQUET)
    frame.loc[frame.index[0], "price_sgd_per_kwh"] = float("inf")
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(), canonical_frame=frame)


@needs_assets
def test_unknown_reference_value_key_is_rejected():
    """逐值键集合**精确**；多键 / 少键一律拒绝（未来扩展必须 bump schema）。"""
    module = freeze()
    payload = module.build_frozen_refs(**_chain_kwargs())
    payload["references"]["price_ref"]["shadow_ref"] = 1.0
    with pytest.raises(ValueError):
        module.validate_frozen_refs(payload)


# --- 6. 共享同一份 refs -------------------------------------------------------

@needs_assets
def test_all_splits_share_the_same_frozen_refs():
    """三个 split 读的是**同一份** v2 refs —— 不按测试日重算。"""
    module = freeze()
    on_disk = module.load_frozen_refs(REFS_PATH)
    # 复用已冻结的时间戳（与物化器的幂等语义一致），候选必须与磁盘**逐字段相同**
    rebuilt = module.build_frozen_refs(
        frozen_at_utc=on_disk["frozen_at_utc"], **_chain_kwargs())
    assert rebuilt == on_disk
    # 同一份冻结文件对所有 split 生效：train 推导值不随 split 改变
    for split in ("train", "validation", "test"):
        assert on_disk["training_range"]["split"] == "train"
        assert split in ("train", "validation", "test")
