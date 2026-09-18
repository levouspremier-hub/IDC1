"""M1.3g-c / R1 / R2 / R3：正式 split manifest 的路径、provenance、绑定与物化。

**R3 改前缺陷（本文件在实现前必须为红）**：

- 公开 loader 接受**任意临时目录中的合法副本**（实测 `ARBITRARY_COPY_ACCEPTED`）；
- 公开 loader 接受**伪造的 `materializer_revision`**（`0`×40）
  （实测 `FORGED_PROVENANCE_ACCEPTED`）。

**R2 已修复（保留）**：`manifest_relative_path()` 曾返回 superseded 的 v1 路径。
**R1 已修复（保留）**：validator 曾只检查 `path` / `sha256` 的**语法**，
把 `inputs.frozen_refs` 指向被取代的 refs 也能通过。

**版本层次**：v1 = `superseded_pre_live_input_binding_fix`；
v2 = `superseded_pre_canonical_path_fix`；
v3 = `superseded_pre_canonical_loader_trust_boundary_fix`；**v4 = 唯一候选**。

**本卡只物化 triad**：不保存 forecast 数值、不启动 env 或训练、
不固定 H/C、不抽样 origin。

> **测试如何注入非 canonical 内容**：loader 现在**只**接受 canonical 路径。
> 要验证「内容层面的」拒绝（绑定、provenance、结构），测试 monkeypatch
> 模块**私有**的 `_canonical_split_dir()` 指向 `tmp_path`，再把内容写到
> `tmp_path/<split>.json` —— 这样路径检查通过、内容检查生效。
> 这是**私有**解析器的 monkeypatch，**不是**生产级绕过开关。
"""

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
SPLITS = ("train", "validation", "test")

# --- 三层历史版本（**逐字节保留**） -----------------------------------------
V1_TRIAD = {s: MANIFEST_DIR / f"{s}.json" for s in SPLITS}
V1_SHA256 = {
    "train": "91b2d7c2efa4e8780d0a0765dbdd4f928cded5340a6c4d4c3c342b604c1f5991",
    "validation": "19f17706fb422315a2432377e0ceaa64d16e70e78155e5656554e39fd22920ca",
    "test": "a40c595ab7641fcd13de54052c0d8cc601e9d332c67c65844198f53274b106d7",
}
V2_SPLIT_DIR = MANIFEST_DIR / "formal_splits_v2"
V2_TRIAD = {s: V2_SPLIT_DIR / f"{s}.json" for s in SPLITS}
V2_SHA256 = {
    "train": "e0084a66830aaee8d46c59d43acad586cdffb10c5d4c07d8fb6a5d77f9e77f76",
    "validation": "1f539fccefb2f2aff02b1a3f142741cc979c042fc0426d7f9e739a7825d46384",
    "test": "0b4009b87e5408b88060b25449145bc3a603ddb972cacea94db68ad6ee4a40cf",
}
V3_SPLIT_DIR = MANIFEST_DIR / "formal_splits_v3"
V3_TRIAD = {s: V3_SPLIT_DIR / f"{s}.json" for s in SPLITS}
V3_SHA256 = {
    "train": "0ee774e4e8f09feb1b62a9449e586d4c4bb1803bc2245a1ae4a18c2577d2a5e2",
    "validation": "3ac19480143b97aada0ed39ecc7b357480d0d8e991ed0afb68eced2d665a6d7a",
    "test": "62b91d0aca37c0c981db2d690146cc5486511ce51e88f051715bcf9df9cb4573",
}

# --- v4：唯一候选 -------------------------------------------------------------
FORMAL_SPLIT_DIR = MANIFEST_DIR / "formal_splits_v4"
TRIAD = {s: FORMAL_SPLIT_DIR / f"{s}.json" for s in SPLITS}

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

SCHEMA = "m1.3g-formal-split-manifest-v4"
SCHEMA_V3 = "m1.3g-formal-split-manifest-v3"
CONTRACT_VERSION = "contract-v9"
REFS_V3_SHA256 = (
    "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f"
)
REFS_V2_SHA256 = (
    "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827"
)
ZERO_SHA = "0" * 40

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
    """经**公开 verified loader** 读取 canonical v4 的一份。"""
    return manifests().load_verified_split_manifest(
        TRIAD[split], expected_split=split)


def _redirect_canonical_dir(monkeypatch, tmp_path) -> pathlib.Path:
    """把模块**私有**的 canonical 目录解析改指 `tmp_path`。

    用于在 **canonical 位置上**注入非 canonical **内容**，
    以便让内容层面的检查（绑定 / provenance / 结构）真正生效。
    """
    module = manifests()
    monkeypatch.setattr(module, "_canonical_split_dir", lambda: tmp_path)
    return tmp_path


def _place(tmp_path, split: str, mutate=None, payload=None) -> pathlib.Path:
    """把 payload（默认取自 canonical v4，缺省回退 v3）写到 `tmp_path/<split>.json`。"""
    if payload is None:
        source = TRIAD[split] if TRIAD[split].exists() else V3_TRIAD[split]
        payload = json.loads(source.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(payload)
    target = tmp_path / f"{split}.json"
    target.write_text(json.dumps(payload))
    return target


def _reject_at_canonical(tmp_path, monkeypatch, split, mutate=None) -> None:
    """在 canonical 位置放**被篡改的内容**，loader 必须拒绝。"""
    _redirect_canonical_dir(monkeypatch, tmp_path)
    _place(tmp_path, split, mutate=mutate)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(
            tmp_path / f"{split}.json", expected_split=split)


# --- 1. v4 triad 存在、形状正确、逐字绑定 -------------------------------------

@needs_assets
def test_v4_triad_exists_and_is_the_only_candidate():
    for split in SPLITS:
        assert TRIAD[split].exists(), TRIAD[split]
        payload = _load(split)
        assert payload["split"] == split
        assert payload["schema"] == SCHEMA


@needs_assets
def test_every_manifest_uses_the_same_exact_schema():
    for split in SPLITS:
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
    for split in SPLITS:
        payload = _load(split)
        for role, logical in PRODUCTION_PATHS.items():
            assert payload["inputs"][role]["path"] == logical, (split, role)
        assert payload["inputs"]["frozen_refs"]["sha256"] == REFS_V3_SHA256, split
        assert payload["inputs"]["canonical_parquet"]["sha256"] == (
            _sha256(CANONICAL_PARQUET)), split


@needs_assets
def test_manifests_never_claim_training_or_env_readiness():
    for split in SPLITS:
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
    for split in SPLITS:
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


# --- 2. R3 核心：canonical-only 路径 ------------------------------------------

@needs_assets
def test_arbitrary_copy_is_rejected_by_the_public_loader(tmp_path):
    """**审核缺陷的直接复现**（不依赖 v4 是否已物化）。

    造一份**结构完全合法**的 manifest，写到**任意**临时目录，经公开 loader 读取。
    改前为 `ARBITRARY_COPY_ACCEPTED`（认证只看内容、不看路径）。
    """
    module = manifests()
    payload = module.build_split_manifest("train")
    copy = tmp_path / "train.json"
    copy.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.load_verified_split_manifest(copy, expected_split="train")


@needs_assets
def test_forged_provenance_revision_is_rejected_by_the_public_loader(tmp_path):
    """**审核缺陷的直接复现**：伪造 `materializer_revision` 必须被拒绝。

    改前为 `FORGED_PROVENANCE_ACCEPTED 000…000`。
    """
    module = manifests()
    payload = module.build_split_manifest("train")
    payload["materializer_revision"] = ZERO_SHA
    copy = tmp_path / "train.json"
    copy.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.load_verified_split_manifest(copy, expected_split="train")


@needs_assets
def test_a_copy_in_a_temp_directory_is_rejected(tmp_path):
    """R3-1：**任意临时目录**中的合法 manifest 副本必须被拒绝。"""
    copy = tmp_path / "train.json"
    copy.write_bytes(TRIAD["train"].read_bytes())
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(copy, expected_split="train")


@needs_assets
def test_a_copy_with_a_forged_revision_is_rejected(tmp_path):
    """R3-2：临时副本带伪造 `materializer_revision` 必须被拒绝。"""
    payload = json.loads(TRIAD["train"].read_text(encoding="utf-8"))
    payload["materializer_revision"] = ZERO_SHA
    copy = tmp_path / "train.json"
    copy.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(copy, expected_split="train")


@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_cross_split_canonical_path_is_rejected(split):
    """R3-6：split 与 canonical 文件名不匹配必须明确失败。"""
    for other in SPLITS:
        if other == split:
            continue
        with pytest.raises(ValueError):
            manifests().load_verified_split_manifest(
                TRIAD[other], expected_split=split)


@needs_assets
def test_a_path_alias_outside_the_repo_is_rejected():
    """仓库外（但存在）的等价路径同样被拒绝。"""
    module = manifests()
    with pytest.raises(ValueError):
        module.load_verified_split_manifest(
            MANIFEST_DIR.parent / "manifest" / "formal_splits_v4" / "train.json",
            expected_split="train")


@needs_assets
def test_a_symlink_alias_cannot_bypass_the_canonical_path(tmp_path):
    """R3-7：symlink 指向 canonical 文件也不能绕过路径检查。"""
    link = tmp_path / "train.json"
    link.symlink_to(TRIAD["train"])
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(link, expected_split="train")


@needs_assets
def test_a_symlinked_directory_cannot_bypass_the_canonical_path(tmp_path):
    """R3-7：通过 symlink 目录指向 canonical 目录也不能绕过。"""
    link_dir = tmp_path / "linked"
    link_dir.symlink_to(FORMAL_SPLIT_DIR)
    with pytest.raises(ValueError):
        manifests().load_verified_split_manifest(
            link_dir / "train.json", expected_split="train")


@needs_assets
@pytest.mark.parametrize("split", SPLITS)
def test_all_three_historical_locations_are_rejected(split):
    """R3-4：v1、v2、v3 三个历史位置全部被拒绝。"""
    module = manifests()
    for legacy in (V1_TRIAD[split], V2_TRIAD[split], V3_TRIAD[split]):
        with pytest.raises(ValueError):
            module.load_verified_split_manifest(legacy, expected_split=split)


@needs_assets
def test_a_foreign_repo_directory_is_rejected():
    """仓库内的**其它**目录（非 canonical）也必须拒绝。"""
    target = MANIFEST_DIR / "formal_splits_v4_shadow" / "train.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(TRIAD["train"].read_bytes())
        with pytest.raises(ValueError):
            manifests().load_verified_split_manifest(
                target, expected_split="train")
    finally:
        target.unlink(missing_ok=True)
        target.parent.rmdir()


# --- 3. R3 核心：provenance revision ------------------------------------------

@needs_assets
def test_forged_revision_at_the_canonical_path_is_rejected(tmp_path, monkeypatch):
    """R3-2/3：canonical 位置的 revision 为 `0`×40 必须被拒绝。"""
    _reject_at_canonical(
        tmp_path, monkeypatch, "train",
        mutate=lambda p: p.__setitem__("materializer_revision", ZERO_SHA))


@needs_assets
def test_historical_revision_at_the_canonical_path_is_rejected(tmp_path, monkeypatch):
    """R3-3：历史（但格式合法）的 revision 必须被拒绝。"""
    _reject_at_canonical(
        tmp_path, monkeypatch, "train",
        mutate=lambda p: p.__setitem__("materializer_revision", "a" * 40))


@needs_assets
def test_the_canonical_revision_equals_the_current_resolver():
    """R3-17：canonical revision 必须精确等于当前 resolver 的结果。"""
    module = manifests()
    expected = module.resolve_materializer_revision()
    for split in SPLITS:
        assert _load(split)["materializer_revision"] == expected, split


@needs_assets
def test_loader_rejects_a_dirty_materializer_source(monkeypatch):
    """R3-8：relevant source dirty 时 loader 必须明确失败。"""
    module = manifests()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        module.load_verified_split_manifest(TRIAD["train"], expected_split="train")


# --- 4. R1 保留：实时输入绑定（在 canonical 位置上验证） -----------------------

@needs_assets
def test_public_reader_rejects_a_tampered_binding(tmp_path, monkeypatch):
    """R1 保留：`frozen_refs` 指向被取代的 refs 且 hash 全零必须被拒绝。"""
    def _repoint(payload):
        payload["inputs"]["frozen_refs"] = {
            "path": "configs/frozen_refs/refs.json", "sha256": "0" * 64}

    _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=_repoint)


@needs_assets
def test_all_zero_hash_on_refs_v3_path_is_rejected(tmp_path, monkeypatch):
    _reject_at_canonical(
        tmp_path, monkeypatch, "train",
        mutate=lambda p: p["inputs"]["frozen_refs"].__setitem__("sha256", "0" * 64))


@needs_assets
@pytest.mark.parametrize("role", INPUT_ROLES)
def test_every_role_rejects_a_tampered_hash(tmp_path, monkeypatch, role):
    def _zero(payload, r=role):
        payload["inputs"][r]["sha256"] = "0" * 64

    _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=_zero)


@needs_assets
@pytest.mark.parametrize("role", INPUT_ROLES)
def test_every_role_rejects_a_foreign_or_external_path(tmp_path, monkeypatch, role):
    for bad in ("configs/frozen_refs/refs.json", "<external>/x.json",
                "/abs/path.json", "../escape.json"):
        def _repoint(payload, r=role, b=bad):
            payload["inputs"][r]["path"] = b

        _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=_repoint)


@needs_assets
def test_swapping_two_roles_is_rejected(tmp_path, monkeypatch):
    def _swap(payload):
        inputs = payload["inputs"]
        inputs["canonical_manifest"], inputs["split_manifest"] = (
            inputs["split_manifest"], inputs["canonical_manifest"])

    _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=_swap)


@needs_assets
def test_refs_v2_file_is_rejected_outright(tmp_path, monkeypatch):
    def _repoint(payload):
        payload["inputs"]["frozen_refs"] = {
            "path": "configs/frozen_refs/refs.json", "sha256": REFS_V2_SHA256}

    _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=_repoint)


@needs_assets
@pytest.mark.parametrize("mutation,label", [
    (lambda p: p["split_rows"].__setitem__("end_exclusive", 10223), "bounds"),
    (lambda p: p["candidate_origins"].__setitem__("start", 0), "origins"),
    (lambda p: p["time_range"].__setitem__("timezone", "UTC"), "time"),
    (lambda p: p["readiness"].__setitem__("formal_training_ready", True), "readiness"),
    (lambda p: p.__setitem__("contract_version", "contract-v8"), "contract"),
])
def test_structural_tampering_at_the_canonical_path_is_rejected(
    tmp_path, monkeypatch, mutation, label
):
    _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=mutation)


@needs_assets
def test_unknown_split_is_rejected():
    module = manifests()
    with pytest.raises(ValueError):
        module.build_split_manifest("shadow")
    with pytest.raises(ValueError):
        module.load_verified_split_manifest(TRIAD["train"], expected_split="shadow")


# --- 5. 畸形输入不得泄漏内建异常 ----------------------------------------------

@needs_assets
@pytest.mark.parametrize("bad", ("not json", "[]", '{"schema": "x"}',
                                 '{"inputs": null}', '{"split_rows": []}'))
def test_malformed_canonical_content_raises_only_value_errors(
    tmp_path, monkeypatch, bad
):
    _redirect_canonical_dir(monkeypatch, tmp_path)
    (tmp_path / "train.json").write_text(bad)
    with pytest.raises(ValueError) as info:
        manifests().load_verified_split_manifest(
            tmp_path / "train.json", expected_split="train")
    assert not isinstance(info.value, (KeyError, TypeError, IndexError))


@needs_assets
@pytest.mark.parametrize("bad", (True, "0", 0.0, None, [0], {"a": 1}))
def test_non_integer_row_bounds_are_rejected(tmp_path, monkeypatch, bad):
    def _bad(payload, b=bad):
        payload["split_rows"]["start"] = b

    _reject_at_canonical(tmp_path, monkeypatch, "train", mutate=_bad)


# --- 6. 公开入口不接受任何信任边界参数 ----------------------------------------

FORBIDDEN_KWARGS = ("inputs", "materializer_revision", "out_dir", "frozen_at_utc",
                    "expected_path", "expected_sha256", "expected_revision",
                    "allow_alternate", "trust_root", "test_mode",
                    "canonical_frame", "frame", "shadow_option")
ALLOWED_EXPECTED_PARAMS = ("expected_split",)
TRUST_BOUNDARY_TOKENS = ("sha", "hash", "revision", "root", "trust", "frame",
                         "inputs", "out_dir", "frozen_at")


def test_public_signatures_expose_no_trust_boundary_parameters():
    module = manifests()
    entry = materializer()
    for fn in (module.build_split_manifest, module.load_verified_split_manifest,
               entry.materialize_split_manifest_triad):
        params = inspect.signature(fn).parameters
        for name, param in params.items():
            assert param.kind is not inspect.Parameter.VAR_KEYWORD, (
                f"{fn.__name__} 暴露了 **kwargs：{name}")
            if name.startswith("expected"):
                assert name in ALLOWED_EXPECTED_PARAMS, (
                    f"{fn.__name__} 暴露了 expected_* 信任根参数：{name}")
                continue
            for token in TRUST_BOUNDARY_TOKENS:
                assert token not in name, f"{fn.__name__} 暴露了 {token}：{name}"


@needs_assets
def test_all_public_entries_raise_type_error_on_unknown_kwargs(tmp_path, monkeypatch):
    module = manifests()
    entry = materializer()
    _redirect_canonical_dir(monkeypatch, tmp_path)
    for kwarg in FORBIDDEN_KWARGS:
        with pytest.raises(TypeError):
            module.build_split_manifest("train", **{kwarg: 1})
        with pytest.raises(TypeError):
            module.load_verified_split_manifest(
                TRIAD["train"], expected_split="train", **{kwarg: 1})
        with pytest.raises(TypeError):
            entry.materialize_split_manifest_triad(**{kwarg: 1})


@needs_assets
def test_materializer_takes_no_arguments():
    """R3-10：public materializer 不得接受 out_dir / frozen timestamp / 信任根。"""
    params = inspect.signature(
        materializer().materialize_split_manifest_triad).parameters
    assert list(params) == [], list(params)


def test_cli_has_no_out_dir_option():
    """R3-11：CLI 不得存在 `--out-dir`。"""
    entry = materializer()
    with pytest.raises(SystemExit) as info:
        entry.main(["--out-dir", "somewhere"])
    assert info.value.code == 2


def test_cli_defaults_to_the_canonical_v4_directory():
    entry = materializer()
    parser = entry.build_parser()
    assert "out_dir" not in {a.dest for a in parser._actions}
    assert "formal_splits_v4" in str(entry.canonical_out_dir())


# --- 7. 物化：triad 原子、幂等、防覆盖 ----------------------------------------

@needs_assets
def test_first_freeze_is_atomic_for_the_whole_triad(tmp_path, monkeypatch):
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    monkeypatch.setattr(
        entry, "_atomic_write_text",
        lambda path, text: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        entry.materialize_split_manifest_triad()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@needs_assets
def test_second_write_failure_rolls_back_the_first_file(tmp_path, monkeypatch):
    """R3-13：**第二次** replace 失败后，第一个**已写**文件也必须回滚。"""
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    original = entry._atomic_write_text
    calls: list[str] = []

    def _fail_on_second(path, text):
        calls.append(pathlib.Path(path).name)
        if len(calls) == 2:
            raise OSError("boom on the second write")
        return original(path, text)

    monkeypatch.setattr(entry, "_atomic_write_text", _fail_on_second)
    with pytest.raises(OSError):
        entry.materialize_split_manifest_triad()
    assert len(calls) == 2, calls
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@needs_assets
def test_triad_is_idempotent_and_preserves_mtime(tmp_path, monkeypatch):
    """R3-15：v4 幂等验证不改变 bytes / hash / mtime_ns。"""
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    first = entry.materialize_split_manifest_triad()
    before = {n: (_sha256(p), p.stat().st_mtime_ns)
              for n, p in first["paths"].items()}
    second = entry.materialize_split_manifest_triad()
    after = {n: (_sha256(p), p.stat().st_mtime_ns)
             for n, p in second["paths"].items()}
    assert first["written"] is True
    assert second["written"] is False
    assert before == after
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        f"{s}.json" for s in SPLITS)


@needs_assets
def test_partial_triad_is_rejected(tmp_path, monkeypatch):
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    entry.materialize_split_manifest_triad()
    (tmp_path / "validation.json").unlink()
    with pytest.raises(ValueError):
        entry.materialize_split_manifest_triad()


@needs_assets
def test_different_existing_manifest_is_rejected(tmp_path, monkeypatch):
    """R3-14：不同内容必须拒绝覆盖。"""
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    result = entry.materialize_split_manifest_triad()
    target = result["paths"]["train"]
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["candidate_origins"]["end_exclusive"] = 1
    target.write_text(json.dumps(payload))
    mutated = target.read_bytes()
    with pytest.raises(ValueError):
        entry.materialize_split_manifest_triad()
    assert target.read_bytes() == mutated


@needs_assets
def test_malformed_existing_manifest_is_rejected(tmp_path, monkeypatch):
    """R3-14：畸形既有文件必须拒绝。"""
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    entry.materialize_split_manifest_triad()
    for bad in ("not json", "[]", '{"schema": "other"}'):
        (tmp_path / "test.json").write_text(bad)
        with pytest.raises(ValueError):
            entry.materialize_split_manifest_triad()


@needs_assets
def test_dirty_generator_is_rejected(tmp_path, monkeypatch):
    entry = materializer()
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        entry.materialize_split_manifest_triad()


@needs_assets
def test_v4_triad_shares_one_frozen_at_utc():
    """R3-16：v4 三份必须共享**同一个** `frozen_at_utc`。"""
    stamps = {split: _load(split)["frozen_at_utc"] for split in SPLITS}
    assert len(set(stamps.values())) == 1, stamps


@needs_assets
def test_slow_first_freeze_is_still_idempotent(tmp_path, monkeypatch):
    """把每个 split 的构造**人为拖慢**跨越秒边界，仍必须共享时间戳且幂等。"""
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)

    original = entry.build_split_manifest
    ticks = iter(["2026-01-01T00:00:01+00:00", "2026-01-01T00:00:02+00:00",
                  "2026-01-01T00:00:03+00:00"])
    seen: list[str] = []

    def _slow_build(split, *, frozen_at_utc=None):
        stamped = (original(split, frozen_at_utc=frozen_at_utc)
                   if frozen_at_utc is not None
                   else original(split, frozen_at_utc=next(ticks)))
        seen.append(stamped["frozen_at_utc"])
        return stamped

    monkeypatch.setattr(entry, "build_split_manifest", _slow_build)
    first = entry.materialize_split_manifest_triad()
    assert first["written"] is True
    assert len(set(seen)) == 1, f"三个 split 未共享时间戳：{seen}"

    monkeypatch.undo()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    before = {n: (_sha256(p), p.stat().st_mtime_ns)
              for n, p in first["paths"].items()}
    second = entry.materialize_split_manifest_triad()
    after = {n: (_sha256(p), p.stat().st_mtime_ns)
             for n, p in second["paths"].items()}
    assert second["written"] is False
    assert before == after


# --- 8. 严格上游链**确实**被调用（含 reader） ---------------------------------

@needs_assets
def test_reader_calls_the_existing_strict_chain_checks(monkeypatch):
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
def test_committed_triad_matches_a_fresh_build(tmp_path, monkeypatch):
    entry = materializer()
    monkeypatch.setattr(entry, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(entry, "canonical_out_dir", lambda: tmp_path)
    rebuilt = entry.materialize_split_manifest_triad()
    for split in SPLITS:
        on_disk = json.loads(TRIAD[split].read_text(encoding="utf-8"))
        fresh = json.loads(rebuilt["paths"][split].read_text(encoding="utf-8"))
        assert fresh == on_disk, split


@needs_assets
def test_revision_equals_the_last_materializer_source_commit():
    """R3-17：v4 revision == 最后一次修改 MATERIALIZER_SOURCE_PATHS 的提交。"""
    module = manifests()
    revision = module.resolve_materializer_revision()
    assert len(revision) == 40
    assert revision == module._git(
        "log", "-1", "--format=%H", "--", *module.MATERIALIZER_SOURCE_PATHS
    ).strip()
    for split in SPLITS:
        assert _load(split)["materializer_revision"] == revision, split


# --- 9. R2 保留：唯一路径来源 ------------------------------------------------

@needs_assets
def test_manifest_relative_path_is_exactly_the_v4_candidate_path():
    module = manifests()
    for split in SPLITS:
        assert module.manifest_relative_path(split) == (
            f"data/manifest/formal_splits_v4/{split}.json"), split


@needs_assets
def test_no_public_helper_or_constant_points_at_an_older_generation():
    module = manifests()
    for name in dir(module):
        if name.startswith("_"):
            continue
        value = getattr(module, name)
        if isinstance(value, str) and "formal_splits" in value:
            assert "formal_splits_v4" in value, (name, value)
        if isinstance(value, pathlib.Path) and "formal_splits" in str(value):
            assert "formal_splits_v4" in str(value), (name, value)
    assert "formal_splits_v4" in str(module.FORMAL_SPLIT_DIR)


# --- 10. v1 / v2 / v3 逐字节保持 ----------------------------------------------

@needs_assets
def test_all_three_older_generations_are_preserved_byte_for_byte():
    for split in SPLITS:
        assert _sha256(V1_TRIAD[split]) == V1_SHA256[split], ("v1", split)
        assert _sha256(V2_TRIAD[split]) == V2_SHA256[split], ("v2", split)
        assert _sha256(V3_TRIAD[split]) == V3_SHA256[split], ("v3", split)
    assert json.loads(V1_TRIAD["train"].read_text())["schema"] == (
        "m1.3g-formal-split-manifest-v1")
    assert json.loads(V2_TRIAD["train"].read_text())["schema"] == (
        "m1.3g-formal-split-manifest-v2")
    assert json.loads(V3_TRIAD["train"].read_text())["schema"] == SCHEMA_V3


@needs_assets
def test_v4_business_semantics_match_v3():
    """除 schema / 路径 / revision / 冻结时间外，v4 与 v3 **逐字段相同**。"""
    for split in SPLITS:
        v3 = json.loads(V3_TRIAD[split].read_text(encoding="utf-8"))
        v4 = json.loads(TRIAD[split].read_text(encoding="utf-8"))
        for key in ("contract_version", "split", "split_rows", "time_range",
                    "frequency", "history_steps", "candidate_origins", "inputs",
                    "readiness"):
            assert v4[key] == v3[key], (split, key)
        assert v4["schema"] == SCHEMA
        assert v3["schema"] == SCHEMA_V3
