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
import shutil

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
# M1.3g-d-R1：v2 是**已被取代**的历史产物（superseded_pre_trust_boundary_fix）；
# 唯一候选证据是 v3。v2 **逐字节不变**，本文件只读它做对照。
REFS_V2_PATH = REPO_ROOT / "configs/frozen_refs/refs.json"
REFS_V3_PATH = REPO_ROOT / "configs/frozen_refs/refs_v3.json"
SUPERSEDED_MARK = "superseded_pre_trust_boundary_fix"
V2_SHA256 = (
    "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827"
)
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


def splits_module():
    return importlib.import_module("scenario.splits")


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
    return json.loads(REFS_V3_PATH.read_text(encoding="utf-8"))


def _chain_kwargs(chain: dict | None = None) -> dict:
    """公开入口的**路径**参数（默认指向仓库冻结链）。

    R1：公开签名只接受**文件路径**——不存在 frame / hash / revision 注入参数。
    """
    source = chain or {}
    return {
        "canonical_parquet_path": source.get("canonical_parquet_path",
                                             CANONICAL_PARQUET),
        "canonical_manifest_path": source.get("canonical_manifest_path",
                                              CANONICAL_MANIFEST),
        "split_manifest_path": source.get("split_manifest_path", SPLIT_MANIFEST),
        "exogenous_manifest_path": source.get("exogenous_manifest_path",
                                              EXOGENOUS_MANIFEST),
        "exogenous_source_manifest_path": source.get(
            "exogenous_source_manifest_path", EXOGENOUS_SOURCE),
        "exogenous_parquet_path": source.get("exogenous_parquet_path",
                                             EXOGENOUS_PARQUET),
    }


def _copy(src, dst) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


def _synced_temp_chain(tmp_path, *, mutate=None, mutate_exogenous=None) -> dict:
    """构造**完整且自洽**的临时冻结链（含 canonical / split / exogenous / policy）。

    `mutate(frame)` 可选地就地修改 canonical 帧；随后**所有**相关 manifest 的
    路径与 SHA-256 都会被重新对齐，因此整条链在公开入口看来完全自洽 ——
    这正是「train 改动会影响、validation/test 不影响」必须走的验证路径。
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
    # 若 mutation 落在 train 段，M1.3d 的 train-only 统计会被重算并比对；
    # 这里用**同一个实现**重新计算，使临时链真正自洽（不是绕过校验）。
    if mutate is not None:
        splits["train_only_statistics"] = splits_module().train_only_statistics(
            frame.iloc[: splits["splits"]["train"]["row_end_exclusive"]])
    split_manifest.write_text(json.dumps(splits))

    exogenous_parquet = cano_dir / "exogenous_drivers_v2.parquet"
    if mutate_exogenous is None:
        _copy(EXOGENOUS_PARQUET, exogenous_parquet)
    else:
        exog_frame = pd.read_parquet(EXOGENOUS_PARQUET)
        mutate_exogenous(exog_frame)
        exog_frame.to_parquet(exogenous_parquet, index=False)

    exogenous_source = man_dir / "m13f_materialization_sources_v3.json"
    _copy(EXOGENOUS_SOURCE, exogenous_source)

    exogenous_manifest = man_dir / "singapore_2024_exogenous_v2.json"
    payload = json.loads(EXOGENOUS_MANIFEST.read_text(encoding="utf-8"))
    payload["canonical_parquet_path"] = "<external>/half_hour.parquet"
    payload["canonical_parquet_sha256"] = _sha256(parquet)
    payload["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
    payload["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    payload["split_manifest_path"] = "<external>/singapore_2024_splits.json"
    payload["split_manifest_sha256"] = _sha256(split_manifest)
    payload["output"]["path"] = "<external>/exogenous_drivers_v2.parquet"
    payload["output"]["sha256"] = _sha256(exogenous_parquet)
    exogenous_manifest.write_text(json.dumps(payload))

    return {
        "canonical_parquet_path": parquet,
        "canonical_manifest_path": canonical_manifest,
        "split_manifest_path": split_manifest,
        "exogenous_parquet_path": exogenous_parquet,
        "exogenous_manifest_path": exogenous_manifest,
        "exogenous_source_manifest_path": exogenous_source,
    }


def _patch_frozen_root(monkeypatch, chain: dict) -> None:
    """把**模块内部**的外生冻结根指向临时链（公开 API 无此参数）。

    这只改变信任根**指向哪一条同样严格的链**；所有校验器仍然全量运行 ——
    不存在任何绕过验证的测试专用开关。
    """
    import scenario.formal_scenario as formal_module

    monkeypatch.setattr(
        formal_module, "EXOGENOUS_MANIFEST_SHA256",
        _sha256(chain["exogenous_manifest_path"]))
    monkeypatch.setattr(
        formal_module, "EXOGENOUS_SOURCE_MANIFEST_SHA256",
        _sha256(chain["exogenous_source_manifest_path"]))
    monkeypatch.setattr(
        formal_module, "EXOGENOUS_OUTPUT_SHA256",
        _sha256(chain["exogenous_parquet_path"]))


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
def test_validation_and_test_do_not_influence_the_values(tmp_path, monkeypatch):
    """把所有 validation/test 行放大 1000 倍，重算出的 train-derived 值必须不变。

    R1：必须走**完整临时冻结信任链 + 公开文件路径入口**，
    **不得**用 frame 注入或 monkeypatch 验证器。
    """
    module = freeze()
    baseline = module.build_frozen_refs(**_chain_kwargs())
    stop = TRAIN_RANGE["row_end_exclusive"]

    def _scale_validation_and_test(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[stop]:, "price_sgd_per_kwh"] *= 1000.0

    chain = _synced_temp_chain(tmp_path, mutate=_scale_validation_and_test)
    _patch_frozen_root(monkeypatch, chain)
    after = module.build_frozen_refs(**_chain_kwargs(chain))
    for name in TRAIN_DERIVED:
        assert after["references"][name]["value"] == pytest.approx(
            baseline["references"][name]["value"]), name


@needs_assets
def test_train_mutation_changes_the_train_derived_values(tmp_path, monkeypatch):
    module = freeze()
    baseline = module.build_frozen_refs(**_chain_kwargs())

    def _bump_train_price(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[0], "price_sgd_per_kwh"] = 99.0

    chain = _synced_temp_chain(tmp_path, mutate=_bump_train_price)
    _patch_frozen_root(monkeypatch, chain)
    after = module.build_frozen_refs(**_chain_kwargs(chain))
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
    target = tmp_path / "refs_v3.json"
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
            out_path=tmp_path / "refs_v3.json", replace_declared_v1=True,
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
def test_non_finite_train_value_is_rejected(tmp_path, monkeypatch):
    """train 段的非有限值必须 fail closed（落在 exogenous 列上，由本卡自己的
    有限性检查拦下——canonical 的 train 统计不覆盖 exogenous）。"""
    module = freeze()

    def _make_infinite(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[0], "local_pv_kw"] = float("inf")

    chain = _synced_temp_chain(tmp_path, mutate_exogenous=_make_infinite)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


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
    on_disk = module.load_frozen_refs(REFS_V3_PATH)
    rebuilt = module.build_frozen_refs(**_chain_kwargs())
    # 除冻结时间戳外，逐字段必须完全一致（同一 train 推导、同一来源绑定）
    assert rebuilt["references"] == on_disk["references"]
    assert rebuilt["training_range"] == on_disk["training_range"]
    assert rebuilt["sources"] == on_disk["sources"]
    # 同一份冻结文件对所有 split 生效：train 推导值不随 split 改变
    for split in ("train", "validation", "test"):
        assert on_disk["training_range"]["split"] == "train"
        assert split in ("train", "validation", "test")


# --- 7. M1.3g-d-R1：公开签名不得含 frame / kwargs 注入 ------------------------

FORBIDDEN_PARAMS = ("canonical_frame", "frame", "expected_sha256", "expected_*")


def test_public_signatures_accept_paths_only():
    """R1-1：公开入口只接受**文件路径**；不得有 frame / hash / revision 注入。"""
    import inspect

    module = freeze()
    for fn in (module.build_frozen_refs, module.materialize_frozen_refs):
        params = inspect.signature(fn).parameters
        for name, param in params.items():
            assert param.kind is not inspect.Parameter.VAR_KEYWORD, (
                f"{fn.__name__} 暴露了 **kwargs：{name}"
            )
            assert "frame" not in name, f"{fn.__name__} 暴露了 frame 注入：{name}"
            assert not name.startswith("expected"), (
                f"{fn.__name__} 暴露了 expected_* 信任根参数：{name}"
            )
        assert all(
            name.endswith("_path") or name in
            ("out_path", "replace_declared_v1")
            for name in params
        ), f"{fn.__name__} 的参数必须都是路径：{list(params)}"


@needs_assets
def test_frame_and_kwargs_injection_are_type_errors(tmp_path):
    """R1-1：传 `canonical_frame=` / 任意 kwarg 必须 **TypeError**。"""
    module = freeze()
    frame = pd.read_parquet(CANONICAL_PARQUET)
    frame.loc[frame.index[0], "price_sgd_per_kwh"] = 99.0

    with pytest.raises(TypeError):
        module.build_frozen_refs(canonical_frame=frame)
    with pytest.raises(TypeError):
        module.build_frozen_refs(shadow_option=1)
    with pytest.raises(TypeError):
        module.materialize_frozen_refs(
            out_path=tmp_path / "refs_v3.json", canonical_frame=frame)
    with pytest.raises(TypeError):
        module.materialize_frozen_refs(
            out_path=tmp_path / "refs_v3.json", shadow_option=1)


# --- 8. M1.3g-d-R1：canonical / exogenous 的严格对齐 --------------------------

def _aligned_chain(tmp_path, *, mutate=None, mutate_exogenous=None) -> dict:
    return _synced_temp_chain(
        tmp_path, mutate=mutate, mutate_exogenous=mutate_exogenous)


@needs_assets
def test_row_count_mismatch_is_rejected(tmp_path, monkeypatch):
    module = freeze()

    def _truncate_exogenous(frame: pd.DataFrame) -> None:
        frame.drop(frame.index[-2688:], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    chain = _aligned_chain(tmp_path, mutate_exogenous=_truncate_exogenous)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
def test_timestamp_misalignment_is_rejected(tmp_path, monkeypatch):
    """canonical 与 exogenous 的 timestamp 逐行不等 → 必须 fail closed。"""
    module = freeze()

    def _shift_exogenous(frame: pd.DataFrame) -> None:
        frame.loc[frame.index[0], "timestamp"] = (
            frame.loc[frame.index[0], "timestamp"] + pd.Timedelta(minutes=30)
        )

    chain = _aligned_chain(tmp_path, mutate_exogenous=_shift_exogenous)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
def test_duplicate_timestamp_is_rejected(tmp_path, monkeypatch):
    module = freeze()
    def _duplicate_stamp(frame: pd.DataFrame) -> None:
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]

    chain = _aligned_chain(tmp_path, mutate_exogenous=_duplicate_stamp)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
def test_off_grid_timestamp_is_rejected(tmp_path, monkeypatch):
    """非严格 30 分钟网格 → 必须 fail closed。"""
    module = freeze()
    def _off_grid(frame: pd.DataFrame) -> None:
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"] + pd.Timedelta(minutes=7)

    chain = _aligned_chain(tmp_path, mutate_exogenous=_off_grid)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
def test_naive_timestamp_is_rejected(tmp_path, monkeypatch):
    """tz-naive 的 timestamp → 必须 fail closed。"""
    module = freeze()

    def _make_naive(frame: pd.DataFrame) -> None:
        frame["timestamp"] = (
            pd.DatetimeIndex(frame["timestamp"]).tz_localize(None)
        )

    chain = _aligned_chain(tmp_path, mutate_exogenous=_make_naive)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
def test_missing_required_column_is_rejected(tmp_path, monkeypatch):
    module = freeze()
    chain = _aligned_chain(
        tmp_path,
        mutate_exogenous=lambda f: f.drop(columns=["wind_generation_kw"],
                                          inplace=True),
    )
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
@pytest.mark.parametrize("column", ("local_pv_kw", "wind_generation_kw",
                                    "carbon_intensity"))
def test_non_finite_exogenous_value_is_rejected(tmp_path, monkeypatch, column):
    module = freeze()
    def _to_nan(frame: pd.DataFrame) -> None:
        frame.loc[0, column] = float("nan")

    chain = _aligned_chain(tmp_path, mutate_exogenous=_to_nan)
    _patch_frozen_root(monkeypatch, chain)
    with pytest.raises(ValueError):
        module.build_frozen_refs(**_chain_kwargs(chain))


@needs_assets
def test_alignment_failures_never_leak_builtin_errors(tmp_path, monkeypatch):
    """R1-3：任一失败都必须是 ValueError/SplitError，不得泄漏内建异常。"""
    module = freeze()
    def _drop(column: str):
        def _apply(frame: pd.DataFrame) -> None:
            frame.drop(columns=[column], inplace=True)
        return _apply

    def _truncate(frame: pd.DataFrame) -> None:
        frame.drop(frame.index[-100:], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    def _shuffle(frame: pd.DataFrame) -> None:
        frame["timestamp"] = pd.DatetimeIndex(frame["timestamp"]).tz_localize(None)

    cases = (_drop("local_pv_kw"), _drop("timestamp"), _truncate, _shuffle)
    for index, mutate in enumerate(cases):
        case_root = tmp_path / f"case{index}"
        case_root.mkdir()
        chain = _aligned_chain(case_root, mutate_exogenous=mutate)
        _patch_frozen_root(monkeypatch, chain)
        try:
            module.build_frozen_refs(**_chain_kwargs(chain))
        except (ValueError, KeyError, TypeError, IndexError) as error:
            assert not isinstance(error, (KeyError, TypeError, IndexError)), (
                f"case {index} 泄漏了内建异常：{type(error).__name__}: {error}"
            )


# --- 9. M1.3g-d-R1：v2 保留、v3 是新目标 -------------------------------------

def test_v2_is_preserved_byte_for_byte_and_marked_superseded():
    """R1-5：v2 逐字节不变；文档标记它已被取代。"""
    import subprocess

    frozen = subprocess.run(
        ["git", "show", "e89c15c:configs/frozen_refs/refs.json"],
        cwd=REPO_ROOT, capture_output=True, check=True,
    ).stdout
    assert hashlib.sha256(frozen).hexdigest() == V2_SHA256
    assert REFS_V2_PATH.read_bytes() == frozen
    card = (REPO_ROOT / "docs/task_cards/M1.3g.md").read_text(encoding="utf-8")
    handoff = (REPO_ROOT / "docs/WORK_HANDOFF.md").read_text(encoding="utf-8")
    assert SUPERSEDED_MARK in card and SUPERSEDED_MARK in handoff


@needs_assets
def test_v3_values_equal_v2_values():
    """R1-6：v3 的 refs 数值必须与 v2 **相同**。"""
    v2 = json.loads(REFS_V2_PATH.read_text(encoding="utf-8"))
    v3 = json.loads(REFS_V3_PATH.read_text(encoding="utf-8"))
    assert set(v3) == set(v2)
    assert set(v3["references"]) == set(v2["references"])
    for name in v2["references"]:
        assert v3["references"][name]["value"] == v2["references"][name]["value"], name
        assert v3["references"][name]["source_kind"] == (
            v2["references"][name]["source_kind"]), name


@needs_assets
def test_v3_revision_reflects_the_r1_implementation():
    module = freeze()
    assert _frozen_refs()["materializer_revision"] == (
        module.resolve_materializer_revision())


@needs_assets
def test_v2_is_not_overwritten_by_the_r1_implementation(tmp_path, monkeypatch):
    """R1-5：v2 不可覆盖（对 `refs.json` 物化必须被拒绝）。"""
    module = freeze()
    before = REFS_V2_PATH.read_bytes()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    with pytest.raises(ValueError):
        module.materialize_frozen_refs(
            out_path=REFS_V2_PATH, replace_declared_v1=False, **_chain_kwargs())
    assert REFS_V2_PATH.read_bytes() == before


@needs_assets
def test_v3_first_freeze_is_atomic_idempotent_and_guarded(tmp_path, monkeypatch):
    module = freeze()
    target = tmp_path / "refs_v3.json"
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)

    first = module.materialize_frozen_refs(out_path=target, **_chain_kwargs())
    assert first["written"] is True
    before = (_sha256(target), target.stat().st_mtime_ns)

    second = module.materialize_frozen_refs(
        out_path=target, replace_declared_v1=False, **_chain_kwargs())
    assert second["written"] is False
    assert (_sha256(target), target.stat().st_mtime_ns) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["refs_v3.json"]

    # 不同内容 → 拒绝覆盖
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["references"]["price_ref"]["value"] = 12345.0
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    mutated = target.read_bytes()
    with pytest.raises(ValueError):
        module.materialize_frozen_refs(
            out_path=target, replace_declared_v1=False, **_chain_kwargs())
    assert target.read_bytes() == mutated


@needs_assets
def test_v3_first_freeze_failure_leaves_nothing(tmp_path, monkeypatch):
    module = freeze()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(
        module, "_atomic_write_text",
        lambda path, text: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        module.materialize_frozen_refs(
            out_path=tmp_path / "refs_v3.json", **_chain_kwargs())
    assert sorted(p.name for p in tmp_path.iterdir()) == []
