"""M1.3g-c / R1：正式 split manifest 的**实时输入绑定**校验与物化。

**R1 改前缺陷（本文件在实现前必须为红）**：

- 公开 validator 只检查 `path` / `sha256` 的**语法**，**不**解析实际资产：
  实测把 `inputs.frozen_refs` 改成
  `{"path": "configs/frozen_refs/refs.json", "sha256": "0"*64}`
  后 **仍被接受** —— 于是 triad 可以把 refs 指向被取代的 v2 而不被发现；
- 没有「验证并加载」的唯一公开入口；没有固定角色路径 / 固定 `refs_v3` hash；
- 物化前的严格上游链调用**可以被绕过**（只有 `build_*` 走，reader 不走）。

**本卡只物化 triad**：不保存 forecast 数值、不启动 env 或训练、
不固定 H/C、不抽样 origin。"""

import hashlib
import importlib
import inspect
import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_MODULE = "scenario.formal_split_manifests"
MATERIALIZER = "scripts.materialize_singapore_scenario_manifests"

MANIFEST_DIR = REPO_ROOT / "data/manifest"
# **v1**：已被取代，逐字节保留（superseded_pre_live_input_binding_fix）
V1_TRIAD = {
    "train": MANIFEST_DIR / "train.json",
    "validation": MANIFEST_DIR / "validation.json",
    "test": MANIFEST_DIR / "test.json",
}
V1_SHA256 = {
    "train": "91b2d7c2efa4e8780d0a0765dbdd4f928cded5340a6c4d4c3c342b604c1f5991",
    "validation": "19f17706fb422315a2432377e0ceaa64d16e70e78155e5656554e39fd22920ca",
    "test": "a40c595ab7641fcd13de54052c0d8cc601e9d332c67c65844198f53274b106d7",
}
# **v2**：唯一候选证据
FORMAL_SPLIT_DIR = MANIFEST_DIR / "formal_splits_v2"
TRIAD = {
    "train": FORMAL_SPLIT_DIR / "train.json",
    "validation": FORMAL_SPLIT_DIR / "validation.json",
    "test": FORMAL_SPLIT_DIR / "test.json",
}

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

SCHEMA = "m1.3g-formal-split-manifest-v2"
CONTRACT_VERSION = "contract-v9"
REFS_V3_SHA256 = (
    "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f"
)
REFS_V2_SHA256 = (
    "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827"
)

# 八个角色的**固定生产 logical path**（R1：声明必须精确等于它们）
PRODUCTION_PATHS = {
    "canonical_parquet": "data/processed/singapore_2024/half_hour.parquet",
    "canonical_manifest": "data/manifest/singapore_2024_half_hour.json",
    "split_manifest": "data/manifest/singapore_2024_splits.json",
    "forecast_policy_manifest": "data/manifest/singapore_2024_forecast_policy_v2.json",
    "exogenous_manifest": "data/manifest/singapore_2024_exogenous_v2.json",
    "exogenous_source_manifest": "data/manifest/m13f_materialization_sources_v3.json",
    "exogenous_parquet": "data/processed/singapore_2024/exogenous_drivers_v2.parquet",
    "frozen_refs": "configs/frozen_refs/refs_v3.json",
}
INPUT_ROLES = tuple(PRODUCTION_PATHS)
TOP_KEYS = (
    "schema", "contract_version", "split", "split_rows", "time_range",
    "frequency", "history_steps", "candidate_origins", "inputs", "readiness",
    "materializer_revision", "frozen_at_utc",
)
SERIES_FORECAST_KEYS = (
    "price_forecast", "load_forecast", "pv_forecast", "wind_forecast",
    "temperature_forecast", "carbon_forecast", "arrival_forecast",
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


def _load(split: str = "train") -> dict:
    """经**公开 verified loader** 读取 v2 triad 的一份。"""
    return manifests().load_verified_split_manifest(
        TRIAD[split], expected_split=split)


def _tampered(tmp_path, split="train", mutate=None) -> pathlib.Path:
    """把 v2 manifest 复制到临时文件并施加 `mutate`，返回该路径。"""
    payload = json.loads(TRIAD[split].read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(payload)
    target = tmp_path / f"{split}.json"
    target.write_text(json.dumps(payload))
    return target


# --- 1. v2 triad 存在、形状正确、逐字绑定 -------------------------------------

@needs_assets
def test_v1_triad_is_preserved_byte_for_byte():
    """R1-1：v1 三份**逐字节不变**（仅历史保留，不再是候选）。"""
    for split, path in V1_TRIAD.items():
        assert _sha256(path) == V1_SHA256[split], split
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == "m1.3g-formal-split-manifest-v1", split


@needs_assets
def test_v2_triad_exists_and_is_the_only_candidate():
    for split, path in TRIAD.items():
        assert path.exists(), path
        payload = _load(split)
        assert payload["split"] == split
        assert payload["schema"] == SCHEMA


@needs_assets
def test_every_manifest_uses_the_same_exact_schema():
    for split in TRIAD:
        payload = _load(split)
        assert set(payload) == set(TOP_KEYS), split
        assert payload["contract_version"] == CONTRACT_VERSION, split
        assert payload["frequency"] == "30min", split
        assert payload["history_steps"] == 48, split
        assert set(payload["inputs"]) == set(INPUT_ROLES), split


@needs_assets
def test_split_ranges_and_time_ranges_are_exact():
    for split, expected in EXPECTED.items():
        payload = _load(split)
        assert payload["split_rows"] == expected["split_rows"], split
        assert payload["time_range"] == {
            "timezone": "Asia/Singapore",
            "start": expected["start"],
            "end_exclusive": expected["end_exclusive"],
        }, split


@needs_assets
def test_candidate_origins_are_the_full_history_qualified_set():
    for split, expected in EXPECTED.items():
        assert _load(split)["candidate_origins"] == (
            expected["candidate_origins"]), split


@needs_assets
def test_manifests_bind_the_fixed_production_paths_and_real_hashes():
    for split in TRIAD:
        payload = _load(split)
        for role, logical in PRODUCTION_PATHS.items():
            assert payload["inputs"][role]["path"] == logical, (split, role)
        assert payload["inputs"]["frozen_refs"]["sha256"] == REFS_V3_SHA256, split
        assert payload["inputs"]["canonical_parquet"]["sha256"] == (
            _sha256(CANONICAL_PARQUET)), split


@needs_assets
def test_manifests_never_claim_training_or_env_readiness():
    for split in TRIAD:
        assert _load(split)["readiness"] == {
            "formal_training_ready": False, "formal_env_ready": False,
        }, split


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


@needs_assets
def test_manifests_contain_no_forecast_values_or_single_origin():
    for split in TRIAD:
        payload = json.loads(TRIAD[split].read_text(encoding="utf-8"))
        for key in _walk(payload):
            assert key not in SERIES_FORECAST_KEYS, (split, key)
            assert key != "origin_index", split
            for token in ("future", "truth", "default_curve"):
                assert token not in key.lower(), (split, key)
        _assert_no_lists(payload, split)


def _assert_no_lists(node, split: str) -> None:
    if isinstance(node, dict):
        for value in node.values():
            _assert_no_lists(value, split)
    elif isinstance(node, list):
        raise AssertionError(f"{split} 含数组取值")


# --- 2. R1 核心：实时输入绑定（改前会**放行**的绕过） --------------------------

@needs_assets
def test_public_reader_rejects_a_tampered_binding_from_a_valid_candidate(tmp_path):
    """**审核缺陷的直接复现**（不依赖 v2 文件是否已物化）。

    先用**结构完全合法**的候选 payload，只把 `frozen_refs` 指向被取代的
    `refs.json` 并把 hash 置零；经**公开 verified loader** 必须拒绝。
    改前 `validate_split_manifest` 对此**返回成功**。
    """
    module = manifests()
    existed = hasattr(module, "load_verified_split_manifest")
    payload = module.build_split_manifest("train")
    payload["inputs"]["frozen_refs"] = {
        "path": "configs/frozen_refs/refs.json", "sha256": "0" * 64}
    target = tmp_path / "train.json"
    target.write_text(json.dumps(payload))

    assert existed, "公开 verified loader 必须存在"
    with pytest.raises(ValueError):
        module.load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_v2_path_with_all_zero_hash_is_rejected(tmp_path):
    """审核复现：把 frozen_refs 指向 v2 refs 且 hash 全零 → **必须拒绝**。"""
    def _repoint(payload):
        payload["inputs"]["frozen_refs"] = {
            "path": "configs/frozen_refs/refs.json", "sha256": "0" * 64}

    target = _tampered(tmp_path, mutate=_repoint)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_all_zero_hash_on_refs_v3_path_is_rejected(tmp_path):
    """即使 path 是 refs_v3，hash 被篡改也必须拒绝。"""
    def _zero(payload):
        payload["inputs"]["frozen_refs"]["sha256"] = "0" * 64

    target = _tampered(tmp_path, mutate=_zero)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
@pytest.mark.parametrize("role", INPUT_ROLES)
def test_every_role_rejects_a_tampered_hash(tmp_path, role):
    def _zero(payload, r=role):
        payload["inputs"][r]["sha256"] = "0" * 64

    target = _tampered(tmp_path, mutate=_zero)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
@pytest.mark.parametrize("role", INPUT_ROLES)
def test_every_role_rejects_a_foreign_or_external_path(tmp_path, role):
    """声明的 path 必须精确等于固定生产 logical path。"""
    for bad in ("configs/frozen_refs/refs.json", "<external>/x.json",
                "data/manifest/other.json", "/abs/path.json", "../escape.json"):
        def _repoint(payload, r=role, b=bad):
            payload["inputs"][r]["path"] = b

        target = _tampered(tmp_path, mutate=_repoint)
        with pytest.raises(ValueError):
            manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_swapping_two_roles_is_rejected(tmp_path):
    """把两个角色的声明**互换**（各自仍语法合法）必须拒绝。"""
    def _swap(payload):
        inputs = payload["inputs"]
        inputs["canonical_manifest"], inputs["split_manifest"] = (
            inputs["split_manifest"], inputs["canonical_manifest"])

    target = _tampered(tmp_path, mutate=_swap)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_refs_v2_file_is_rejected_outright(tmp_path):
    """`refs.json`（v2）**一律拒绝**，不得作为备选或回退。"""
    def _repoint(payload):
        payload["inputs"]["frozen_refs"] = {
            "path": "configs/frozen_refs/refs.json", "sha256": REFS_V2_SHA256}

    target = _tampered(tmp_path, mutate=_repoint)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_wrong_split_bounds_are_rejected_even_with_valid_structure(tmp_path):
    """结构完全合法、但行范围错 → 必须拒绝。"""
    def _shift(payload):
        payload["split_rows"]["end_exclusive"] -= 1
        payload["split_rows"]["count"] -= 1

    target = _tampered(tmp_path, mutate=_shift)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_wrong_candidate_origins_are_rejected(tmp_path):
    def _bad(payload):
        payload["candidate_origins"]["start"] = 0

    target = _tampered(tmp_path, mutate=_bad)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_wrong_time_range_is_rejected(tmp_path):
    def _bad(payload):
        payload["time_range"]["timezone"] = "UTC"

    target = _tampered(tmp_path, mutate=_bad)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_wrong_readiness_is_rejected(tmp_path):
    def _bad(payload):
        payload["readiness"]["formal_training_ready"] = True

    target = _tampered(tmp_path, mutate=_bad)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_wrong_contract_version_is_rejected(tmp_path):
    def _bad(payload):
        payload["contract_version"] = "contract-v8"

    target = _tampered(tmp_path, mutate=_bad)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
def test_unknown_split_is_rejected():
    module = manifests()
    with pytest.raises(ValueError):
        module.build_split_manifest("shadow")
    with pytest.raises(ValueError):
        module.load_verified_split_manifest(TRIAD["train"], expected_split="shadow")


# --- 3. 畸形输入不得泄漏内建异常 ----------------------------------------------

@needs_assets
@pytest.mark.parametrize("bad", ("not json", "[]", '{"schema": "x"}',
                                 '{"inputs": null}', '{"split_rows": []}'))
def test_malformed_input_raises_only_value_errors(tmp_path, bad):
    target = tmp_path / "bad.json"
    target.write_text(bad)
    with pytest.raises(ValueError) as info:
        manifests().load_verified_split_manifest(target, expected_split="train")
    assert not isinstance(info.value, (KeyError, TypeError, IndexError))


@needs_assets
@pytest.mark.parametrize("bad", (True, "0", 0.0, None, [0], {"a": 1}))
def test_non_integer_row_bounds_are_rejected(tmp_path, bad):
    def _bad(payload, b=bad):
        payload["split_rows"]["start"] = b

    target = _tampered(tmp_path, mutate=_bad)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


@needs_assets
@pytest.mark.parametrize("bad", ("/abs/x", "../x", "a//b", " lead", "a\\b"))
def test_foreign_logical_paths_are_rejected(tmp_path, bad):
    def _bad(payload, b=bad):
        payload["inputs"]["canonical_parquet"]["path"] = b

    target = _tampered(tmp_path, mutate=_bad)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(target, expected_split="train")


# --- 4. 公开入口不接受任何信任边界参数 ----------------------------------------

FORBIDDEN_KWARGS = ("inputs", "materializer_revision", "frozen_at_utc_override",
                    "expected_sha256", "expected_revision", "canonical_frame",
                    "frame", "trust_root", "shadow_option")


# 只有 `expected_split`（**语义选择器**，与 `expected_series_name` 同类）
# 是允许的；任何可能覆盖信任根的 `expected_*` 都不允许。
ALLOWED_EXPECTED_PARAMS = ("expected_split",)
TRUST_BOUNDARY_TOKENS = ("sha", "hash", "revision", "root", "trust", "frame",
                         "inputs")


def test_public_signatures_expose_no_trust_boundary_parameters():
    module = manifests()
    for fn in (module.build_split_manifest, module.load_verified_split_manifest):
        params = inspect.signature(fn).parameters
        for name, param in params.items():
            assert param.kind is not inspect.Parameter.VAR_KEYWORD, (
                f"{fn.__name__} 暴露了 **kwargs：{name}")
            assert name not in ALLOWED_EXPECTED_PARAMS or name == "expected_split"
            if name.startswith("expected"):
                assert name in ALLOWED_EXPECTED_PARAMS, (
                    f"{fn.__name__} 暴露了 expected_* 信任根参数：{name}")
                continue
            for token in TRUST_BOUNDARY_TOKENS:
                assert token not in name, f"{fn.__name__} 暴露了 {token}：{name}"


@needs_assets
def test_all_public_entries_raise_type_error_on_unknown_kwargs(tmp_path):
    module = manifests()
    entry = materializer()
    for kwarg in FORBIDDEN_KWARGS:
        with pytest.raises(TypeError):
            module.build_split_manifest("train", **{kwarg: 1})
        with pytest.raises(TypeError):
            module.load_verified_split_manifest(
                TRIAD["train"], expected_split="train", **{kwarg: 1})
        with pytest.raises(TypeError):
            entry.materialize_split_manifest_triad(
                out_dir=tmp_path, **{kwarg: 1})


@needs_assets
def test_materializer_signature_is_paths_and_out_dir_only():
    params = inspect.signature(
        materializer().materialize_split_manifest_triad).parameters
    for name, param in params.items():
        assert param.kind is not inspect.Parameter.VAR_KEYWORD, name
        assert name in ("out_dir", "frozen_at_utc"), name


# --- 5. 物化：triad 原子、幂等、防覆盖 ----------------------------------------

@needs_assets
def test_first_freeze_is_atomic_for_the_whole_triad(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(
        module, "_atomic_write_text",
        lambda path, text: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        module.materialize_split_manifest_triad(out_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@needs_assets
def test_second_write_failure_rolls_back_the_first_file(tmp_path, monkeypatch):
    """R1-6：**第二次** replace 失败后，第一个**已写**文件也必须回滚。"""
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    original = module._atomic_write_text
    calls: list[str] = []

    def _fail_on_second(path, text):
        calls.append(pathlib.Path(path).name)
        if len(calls) == 2:
            raise OSError("boom on the second write")
        return original(path, text)

    monkeypatch.setattr(module, "_atomic_write_text", _fail_on_second)
    with pytest.raises(OSError):
        module.materialize_split_manifest_triad(out_dir=tmp_path)
    assert len(calls) == 2, calls
    assert sorted(p.name for p in tmp_path.iterdir()) == [], sorted(
        p.name for p in tmp_path.iterdir())


@needs_assets
def test_triad_is_idempotent_and_preserves_mtime(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    first = module.materialize_split_manifest_triad(out_dir=tmp_path)
    before = {n: (_sha256(p), p.stat().st_mtime_ns)
              for n, p in first["paths"].items()}
    second = module.materialize_split_manifest_triad(out_dir=tmp_path)
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
    module.materialize_split_manifest_triad(out_dir=tmp_path)
    (tmp_path / "validation.json").unlink()
    with pytest.raises(ValueError):
        module.materialize_split_manifest_triad(out_dir=tmp_path)


@needs_assets
def test_different_existing_manifest_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    result = module.materialize_split_manifest_triad(out_dir=tmp_path)
    target = result["paths"]["train"]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["candidate_origins"]["end_exclusive"] = 1
    target.write_text(json.dumps(payload))
    mutated = target.read_bytes()
    with pytest.raises(ValueError):
        module.materialize_split_manifest_triad(out_dir=tmp_path)
    assert target.read_bytes() == mutated


@needs_assets
def test_malformed_existing_manifest_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: False)
    module.materialize_split_manifest_triad(out_dir=tmp_path)
    for bad in ("not json", "[]", '{"schema": "other"}'):
        (tmp_path / "test.json").write_text(bad)
        with pytest.raises(ValueError):
            module.materialize_split_manifest_triad(out_dir=tmp_path)


@needs_assets
def test_dirty_generator_is_rejected(tmp_path, monkeypatch):
    module = materializer()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        module.materialize_split_manifest_triad(out_dir=tmp_path)


# --- 6. 严格上游链**确实**被调用（含 reader） ---------------------------------

@needs_assets
def test_reader_calls_the_existing_strict_chain_checks(monkeypatch):
    """R1-3：**reader**（不只是 builder）必须实际调用既有严格链。"""
    module = manifests()
    seen: list[str] = []

    def _spy(name):
        original = getattr(module, name)

        def wrapper(*args, **kwargs):
            seen.append(name)
            return original(*args, **kwargs)

        return wrapper

    for name in ("load_truth_split", "load_verified_exogenous",
                 "read_forecast_policy_manifest"):
        monkeypatch.setattr(module, name, _spy(name))

    freeze_module = importlib.import_module("scripts.freeze_refs")
    original_refs = freeze_module.load_frozen_refs

    def _refs_spy(*args, **kwargs):
        seen.append("load_frozen_refs")
        return original_refs(*args, **kwargs)

    monkeypatch.setattr(freeze_module, "load_frozen_refs", _refs_spy)

    module.load_verified_split_manifest(TRIAD["train"], expected_split="train")
    assert set(seen) == {"load_truth_split", "load_verified_exogenous",
                         "load_frozen_refs", "read_forecast_policy_manifest"}


@needs_assets
def test_committed_triad_matches_a_fresh_build():
    module = manifests()
    for split in TRIAD:
        on_disk = json.loads(TRIAD[split].read_text(encoding="utf-8"))
        fresh = module.build_split_manifest(
            split, frozen_at_utc=on_disk["frozen_at_utc"])
        assert fresh == on_disk, split


@needs_assets
def test_revision_is_resolvable_from_git():
    module = manifests()
    revision = module.resolve_materializer_revision()
    assert len(revision) == 40
    assert revision == module._git(
        "log", "-1", "--format=%H", "--", *module.MATERIALIZER_SOURCE_PATHS
    ).strip()
