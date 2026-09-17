"""M1.3g-c：三个正式 split manifest 的严格物化（triad）。

**改前缺陷（本文件在实现前必须为红）**：

- `data/manifest/{train,validation,test}.json` **不存在**；
- 没有正式的 split manifest schema / validator / 物化器；
- 没有任何地方把 canonical / split / policy-v2 / exogenous v2-v3 /
  **refs_v3** 的冻结信任链逐字绑定到三份 manifest 上。

**本卡只物化 triad**：不保存 forecast 数值、不启动 env 或训练、
不固定 H/C、不抽样 origin。"""

import hashlib
import importlib
import inspect
import json
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_MODULE = "scenario.formal_split_manifests"
MATERIALIZER = "scripts.materialize_singapore_scenario_manifests"

MANIFEST_DIR = REPO_ROOT / "data/manifest"
TRAIN_MANIFEST = MANIFEST_DIR / "train.json"
VALIDATION_MANIFEST = MANIFEST_DIR / "validation.json"
TEST_MANIFEST = MANIFEST_DIR / "test.json"
TRIAD = {"train": TRAIN_MANIFEST, "validation": VALIDATION_MANIFEST,
         "test": TEST_MANIFEST}

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = MANIFEST_DIR / "singapore_2024_half_hour.json"
SPLIT_MANIFEST = MANIFEST_DIR / "singapore_2024_splits.json"
POLICY_V2 = MANIFEST_DIR / "singapore_2024_forecast_policy_v2.json"
EXOGENOUS_MANIFEST = MANIFEST_DIR / "singapore_2024_exogenous_v2.json"
EXOGENOUS_SOURCE = MANIFEST_DIR / "m13f_materialization_sources_v3.json"
EXOGENOUS_PARQUET = (
    REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
)
REFS_V3 = REPO_ROOT / "configs/frozen_refs/refs_v3.json"
REFS_V2 = REPO_ROOT / "configs/frozen_refs/refs.json"

SCHEMA = "m1.3g-formal-split-manifest-v1"
CONTRACT_VERSION = "contract-v9"
REFS_V3_SHA256 = (
    "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f"
)
REFS_V2_SHA256 = (
    "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827"
)

INPUT_ROLES = (
    "canonical_parquet", "canonical_manifest", "split_manifest",
    "forecast_policy_manifest", "exogenous_manifest",
    "exogenous_source_manifest", "exogenous_parquet", "frozen_refs",
)
TOP_KEYS = (
    "schema", "contract_version", "split", "split_rows", "time_range",
    "frequency", "history_steps", "candidate_origins", "inputs", "readiness",
    "materializer_revision", "frozen_at_utc",
)

EXPECTED = {
    "train": {
        "split_rows": {"start": 0, "end_exclusive": 10224, "count": 10224},
        "start": "2024-01-01T00:00:00+08:00",
        "end_exclusive": "2024-08-01T00:00:00+08:00",
        "candidate_origins": {"start": 48, "end_exclusive": 10224},
    },
    "validation": {
        "split_rows": {"start": 10224, "end_exclusive": 13152, "count": 2928},
        "start": "2024-08-01T00:00:00+08:00",
        "end_exclusive": "2024-10-01T00:00:00+08:00",
        "candidate_origins": {"start": 0, "end_exclusive": 2928},
    },
    "test": {
        "split_rows": {"start": 13152, "end_exclusive": 17568, "count": 4416},
        "start": "2024-10-01T00:00:00+08:00",
        "end_exclusive": "2025-01-01T00:00:00+08:00",
        "candidate_origins": {"start": 0, "end_exclusive": 4416},
    },
}


def manifests():
    return importlib.import_module(MANIFEST_MODULE)


def materializer():
    return importlib.import_module(MATERIALIZER)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _assets_present() -> bool:
    return all(
        p.exists() for p in (
            CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST, POLICY_V2,
            EXOGENOUS_MANIFEST, EXOGENOUS_SOURCE, EXOGENOUS_PARQUET, REFS_V3,
        )
    )


needs_assets = pytest.mark.skipif(
    not _assets_present(), reason="真实冻结上游资产不在本机"
)


def _frozen_inputs() -> dict:
    return {
        "canonical_parquet": CANONICAL_PARQUET,
        "canonical_manifest": CANONICAL_MANIFEST,
        "split_manifest": SPLIT_MANIFEST,
        "forecast_policy_manifest": POLICY_V2,
        "exogenous_manifest": EXOGENOUS_MANIFEST,
        "exogenous_source_manifest": EXOGENOUS_SOURCE,
        "exogenous_parquet": EXOGENOUS_PARQUET,
        "frozen_refs": REFS_V3,
    }


def _payload(split: str = "train") -> dict:
    return manifests().build_split_manifest(split, inputs=_frozen_inputs())


def _triad_payloads() -> dict:
    return {split: _payload(split) for split in ("train", "validation", "test")}


def _triad_kwargs(tmp_path) -> dict:
    return {
        "out_dir": tmp_path,
        **{f"{role}_path": path for role, path in _frozen_inputs().items()},
    }


# --- 1. 三份 manifest 存在且形状正确 -------------------------------------------

@needs_assets
def test_the_three_formal_split_manifests_exist():
    for split, path in TRIAD.items():
        assert path.exists(), path
        assert json.loads(path.read_text(encoding="utf-8"))["split"] == split


@needs_assets
def test_every_manifest_uses_the_same_exact_schema():
    for split, path in TRIAD.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == set(TOP_KEYS), split
        assert payload["schema"] == SCHEMA, split
        assert payload["contract_version"] == CONTRACT_VERSION, split
        assert payload["frequency"] == "30min", split
        assert payload["history_steps"] == 48, split
        assert set(payload["inputs"]) == set(INPUT_ROLES), split
        for role, entry in payload["inputs"].items():
            assert set(entry) == {"path", "sha256"}, f"{split}.{role}"


@needs_assets
def test_split_ranges_and_time_ranges_are_exact():
    for split, expected in EXPECTED.items():
        payload = json.loads(TRIAD[split].read_text(encoding="utf-8"))
        assert payload["split_rows"] == expected["split_rows"], split
        assert payload["time_range"] == {
            "timezone": "Asia/Singapore",
            "start": expected["start"],
            "end_exclusive": expected["end_exclusive"],
        }, split


@needs_assets
def test_candidate_origins_are_the_full_history_qualified_set():
    """D5：完整候选集合（train 从 48 起，validation/test 从 0 起）。"""
    for split, expected in EXPECTED.items():
        payload = json.loads(TRIAD[split].read_text(encoding="utf-8"))
        assert payload["candidate_origins"] == expected["candidate_origins"], split


@needs_assets
def test_manifests_bind_the_frozen_trust_chain_verbatim():
    for split, path in TRIAD.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for role, actual in _frozen_inputs().items():
            entry = payload["inputs"][role]
            assert entry["path"] == manifests().logical_repo_path(actual), (split, role)
            assert entry["sha256"] == _sha256(actual), (split, role)


@needs_assets
def test_manifests_bind_refs_v3_only():
    for split, path in TRIAD.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        refs = payload["inputs"]["frozen_refs"]
        assert refs["path"] == "configs/frozen_refs/refs_v3.json", split
        assert refs["sha256"] == REFS_V3_SHA256, split
        assert refs["sha256"] != REFS_V2_SHA256, split


@needs_assets
def test_manifests_never_claim_training_or_env_readiness():
    for split, path in TRIAD.items():
        readiness = json.loads(path.read_text(encoding="utf-8"))["readiness"]
        assert readiness == {
            "formal_training_ready": False, "formal_env_ready": False,
        }, split


@needs_assets
def test_manifests_contain_no_forecast_values_or_single_origin():
    """**不得**内置七序列 forecast 数值、未来 truth 或单点 `origin_index`。"""
    forbidden = ("forecast", "price_forecast", "load_forecast", "pv_forecast",
                 "wind_forecast", "temperature_forecast", "carbon_forecast",
                 "arrival_forecast", "origin_index", "future")
    for split, path in TRIAD.items():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{split} 含 {token!r}"


# --- 2. 严格链：refs / hash / 上游绑定 ----------------------------------------

@needs_assets
def test_refs_v2_is_rejected():
    module = manifests()
    with pytest.raises(ValueError):
        module.build_split_manifest(
            "train", inputs={**_frozen_inputs(), "frozen_refs": REFS_V2})


@needs_assets
def test_wrong_refs_v3_hash_is_rejected(tmp_path):
    module = manifests()
    payload = json.loads(REFS_V3.read_text(encoding="utf-8"))
    payload["references"]["price_ref"]["value"] = 12345.0
    target = tmp_path / "refs_v3.json"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.build_split_manifest(
            "train", inputs={**_frozen_inputs(), "frozen_refs": target})


@needs_assets
@pytest.mark.parametrize("role", INPUT_ROLES)
def test_any_wrong_upstream_hash_is_rejected(tmp_path, role):
    module = manifests()
    source = _frozen_inputs()[role]
    tampered = tmp_path / f"{role}.json"
    if source.suffix == ".parquet":
        tampered = tmp_path / f"{role}.parquet"
        shutil.copy(source, tampered)
        tampered.write_bytes(tampered.read_bytes() + b"\x00")
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["shadow_field"] = True
        tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.build_split_manifest(
            "train", inputs={**_frozen_inputs(), role: tampered})


@needs_assets
def test_missing_input_role_is_rejected():
    module = manifests()
    incomplete = dict(_frozen_inputs())
    incomplete.pop("split_manifest")
    with pytest.raises(ValueError):
        module.build_split_manifest("train", inputs=incomplete)


@needs_assets
def test_unknown_split_is_rejected():
    module = manifests()
    with pytest.raises(ValueError):
        module.build_split_manifest("shadow", inputs=_frozen_inputs())


# --- 3. schema 严格性：键、类型、路径 -----------------------------------------

@needs_assets
def test_validator_rejects_schema_key_changes():
    module = manifests()
    payload = _payload()
    for mutation in ("drop", "extra"):
        candidate = json.loads(json.dumps(payload))
        if mutation == "drop":
            candidate.pop("candidate_origins")
        else:
            candidate["shadow_field"] = 1
        with pytest.raises(ValueError):
            module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
@pytest.mark.parametrize("bad", (True, "0", 0.0, None, [0], {"a": 1}))
def test_validator_rejects_non_integer_row_bounds(bad):
    """bool 冒充 int、字符串、浮点、容器一律拒绝。"""
    module = manifests()
    candidate = json.loads(json.dumps(_payload()))
    candidate["split_rows"]["start"] = bad
    with pytest.raises(ValueError):
        module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), "x", None))
def test_validator_rejects_non_finite_or_wrong_types(bad):
    module = manifests()
    candidate = json.loads(json.dumps(_payload()))
    candidate["frozen_at_utc"] = bad
    with pytest.raises(ValueError):
        module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
def test_validator_rejects_absolute_and_foreign_logical_paths():
    module = manifests()
    for bad in ("/Users/levous/x.parquet", "../x", "a//b", " lead", "a\\b"):
        candidate = json.loads(json.dumps(_payload()))
        candidate["inputs"]["canonical_parquet"]["path"] = bad
        with pytest.raises(ValueError):
            module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
def test_validator_rejects_wrong_contract_schema_split_and_bounds():
    module = manifests()
    for field, bad in (
        ("contract_version", "contract-v8"),
        ("schema", "m1.3g-formal-split-manifest-v0"),
        ("split", "shadow"),
        ("frequency", "1h"),
    ):
        candidate = json.loads(json.dumps(_payload()))
        candidate[field] = bad
        with pytest.raises(ValueError):
            module.validate_split_manifest(candidate, expected_split="train")

    candidate = json.loads(json.dumps(_payload()))
    candidate["time_range"]["timezone"] = "UTC"
    with pytest.raises(ValueError):
        module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
def test_validator_rejects_wrong_candidate_origin_range():
    module = manifests()
    candidate = json.loads(json.dumps(_payload("train")))
    candidate["candidate_origins"]["start"] = 0    # train 必须从 48 起
    with pytest.raises(ValueError):
        module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
def test_validator_rejects_forecast_and_origin_index_fields():
    module = manifests()
    for extra in ({"price_forecast": [1.0]}, {"origin_index": 48},
                  {"future_truth": []}):
        candidate = json.loads(json.dumps(_payload()))
        candidate.update(extra)
        with pytest.raises(ValueError):
            module.validate_split_manifest(candidate, expected_split="train")


@needs_assets
def test_validator_rejects_lgtmreadiness():
    module = manifests()
    candidate = json.loads(json.dumps(_payload()))
    candidate["readiness"]["formal_training_ready"] = True
    with pytest.raises(ValueError):
        module.validate_split_manifest(candidate, expected_split="train")


# --- 4. 物化：triad 原子、幂等、防覆盖 ----------------------------------------

def test_materializer_exposes_no_trust_root_override():
    """公开签名不得含 expected hash / revision / 信任根 / frame 注入。"""
    module = materializer()
    params = inspect.signature(module.materialize_split_manifest_triad).parameters
    for name, param in params.items():
        assert param.kind is not inspect.Parameter.VAR_KEYWORD, name
        assert not name.startswith("expected"), name
        assert "frame" not in name, name
        assert "revision" not in name, name


@needs_assets
def test_first_freeze_is_atomic_for_the_whole_triad(tmp_path, monkeypatch):
    """任一写失败不得留下**任一**正式 manifest。"""
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(
        module, "_atomic_write_text",
        lambda path, text: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@needs_assets
def test_triad_is_idempotent_and_preserves_mtime(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    first = module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    before = {n: (_sha256(p), p.stat().st_mtime_ns)
              for n, p in first["paths"].items()}
    second = module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    after = {n: (_sha256(p), p.stat().st_mtime_ns)
             for n, p in second["paths"].items()}
    assert first["written"] is True
    assert second["written"] is False
    assert before == after
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "test.json", "train.json", "validation.json"]


@needs_assets
def test_partial_triad_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    (tmp_path / "validation.json").unlink()
    with pytest.raises(ValueError):
        module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))


@needs_assets
def test_different_existing_manifest_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    result = module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    target = result["paths"]["train"]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["candidate_origins"]["end_exclusive"] = 1
    target.write_text(json.dumps(payload))
    mutated = target.read_bytes()
    with pytest.raises(ValueError):
        module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    assert target.read_bytes() == mutated


@needs_assets
def test_malformed_existing_manifest_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))
    for bad in ("not json", "[]", '{"schema": "other"}'):
        (tmp_path / "test.json").write_text(bad)
        with pytest.raises(ValueError):
            module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))


@needs_assets
def test_dirty_generator_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        module.materialize_split_manifest_triad(**_triad_kwargs(tmp_path))


# --- 5. 既有链校验真的被调用（不只相信 manifest 自述） -------------------------

@needs_assets
def test_build_calls_the_existing_strict_chain_checks(monkeypatch):
    module = manifests()
    seen: list[str] = []

    def _spy(name):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in ("load_truth_split", "load_verified_exogenous",
                 "load_frozen_refs"):
        monkeypatch.setattr(module, name, _spy(name))

    module.build_split_manifest("train", inputs=_frozen_inputs())
    assert set(seen) == {"load_truth_split", "load_verified_exogenous",
                         "load_frozen_refs"}


@needs_assets
def test_committed_triad_matches_a_fresh_build():
    """磁盘上的三份必须与重新构造的结果逐字段一致（同一 materializer revision
    与 frozen_at 由物化器复用）。"""
    module = manifests()
    on_disk = {split: json.loads(path.read_text(encoding="utf-8"))
               for split, path in TRIAD.items()}
    for split in TRIAD:
        fresh = module.build_split_manifest(
            split, inputs=_frozen_inputs(),
            frozen_at_utc=on_disk[split]["frozen_at_utc"],
            materializer_revision=on_disk[split]["materializer_revision"])
        assert fresh == on_disk[split], split


@needs_assets
def test_revision_is_resolvable_from_git():
    module = manifests()
    revision = module.resolve_materializer_revision()
    assert len(revision) == 40
    logged = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *module.MATERIALIZER_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert revision == logged
