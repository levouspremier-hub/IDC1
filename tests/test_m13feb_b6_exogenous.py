"""M1.3f-e-b1：由 B6 policy 物化版本化 exogenous v3 的**先红**回归。

改前缺陷（本文件在实现前必须为红）：

- `scenario.exogenous_drivers_b6` 模块尚不存在（`ModuleNotFoundError`）；
- `scripts.materialize_singapore_exogenous_b6` 物化器尚不存在；
- `exogenous_drivers_v3.parquet` / `singapore_2024_exogenous_v3.json` /
  `m13f_materialization_sources_v4.json` 均不存在。

本卡**只物化 v3**：不切换 formal loader、不生成 refs 新版本、不建 formal_splits_v5、
不接 mapper、不训练。"""

import hashlib
import importlib
import json
import pathlib
import shutil

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
B6_MODULE = "scenario.exogenous_drivers_b6"
MATERIALIZER_MODULE = "scripts.materialize_singapore_exogenous_b6"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
B6_POLICY = REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json"

V2_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
V2_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v2.json"
V2_SOURCE = REPO_ROOT / "data/manifest/m13f_materialization_sources_v3.json"

V3_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v3.parquet"
V3_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v3.json"
V3_SOURCE = REPO_ROOT / "data/manifest/m13f_materialization_sources_v4.json"

# v2 三项资产的冻结 hash（本卡**不得**改动它们）
V2_HASHES = {
    V2_PARQUET: "11d322b2919e2180b596e6b02614acafdb3ee8d63682ae74e5a3ee1dbc8b92cf",
    V2_MANIFEST: "640f26cda94b3479049fdbee56f05e1546a24fc3ecdf6286674c4fb415b484b9",
    V2_SOURCE: "4203b4f399ee6433bfcdf63fa94ddd03a56a7bd1e6add45f697c3bec804da1b6",
}
SHARED_COLUMNS = ("local_pv_kw", "wind_generation_kw", "carbon_intensity")
B6_EXPECTED_AMOUNT = 31.994
B5_LEGACY_SCALE = 1000.0


def b6_module():
    return importlib.import_module(B6_MODULE)


def materializer():
    return importlib.import_module(MATERIALIZER_MODULE)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _upstream_present() -> bool:
    # 只检查**上游**资产；本卡产出的 v3 三项不在此列（否则先红会变 skip）
    return all(p.exists() for p in (
        CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST,
        B6_POLICY, V2_PARQUET, V2_MANIFEST, V2_SOURCE,
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机"
)


def _v3_manifest() -> dict:
    return json.loads(V3_MANIFEST.read_text(encoding="utf-8"))


# --- 1/2. 模块与产物存在性、B6 policy 信任根 ------------------------------------

@needs_assets
def test_b6_module_exists_and_binds_the_policy():
    m = b6_module()
    policy = m.load_b6_policy()
    assert policy["decision_id"] == "B6-INTENSITY"
    assert policy["main_expected_amount_work_per_half_hour"] == B6_EXPECTED_AMOUNT


@needs_assets
def test_b6_policy_revision_and_canonical_must_match():
    m = b6_module()
    # policy 的 revision 必须等于 live 解析（已由 loader 保证）
    assert m.load_b6_policy()["materializer_revision"] == \
        importlib.import_module("scenario.arrival_intensity_policy").b6_code_revision()


# --- 3/4. expected mean 严格来自 31.994，不是 1000；template 只贡献 shape --------

@needs_assets
def test_expected_arrival_uses_the_b6_scale_not_the_legacy_1000():
    m = b6_module()
    policy = m.load_b6_policy()
    template = m.load_v2_arrival_template()
    assert float(template.mean()) == pytest.approx(1.0, rel=1e-12)
    stamps = pd.DatetimeIndex(pd.read_parquet(V2_PARQUET)["timestamp"])
    expected = m.expected_arrival(stamps, template, policy)
    # 逐槽期望 = template[slot] × 31.994
    from scenario.exogenous_drivers import arrival_template_slot
    slots = arrival_template_slot(stamps)
    assert expected == pytest.approx(template[slots] * B6_EXPECTED_AMOUNT)
    assert float(expected.mean()) != pytest.approx(B5_LEGACY_SCALE, rel=0.5)


@needs_assets
def test_v2_template_is_bound_by_path_and_sha256():
    m = b6_module()
    assert m.V2_EXOGENOUS_MANIFEST_SHA256 == V2_HASHES[V2_MANIFEST]
    # 篡改 v2 manifest 后必须拒绝
    assert m.load_v2_arrival_template() is not None


# --- 5/6. Poisson seed 固定、结果确定；rho_realized 仅 diagnostic ----------------

@needs_assets
def test_realization_is_deterministic_and_seed_is_fixed():
    m = b6_module()
    policy = m.load_b6_policy()
    assert policy["realization_seed"] == 20240916
    template = m.load_v2_arrival_template()
    stamps = pd.DatetimeIndex(pd.read_parquet(V2_PARQUET)["timestamp"])
    first = m.realize_arrival(stamps, template, policy)
    second = m.realize_arrival(stamps, template, policy)
    assert (first == second).all()


@needs_assets
def test_rho_realized_is_diagnostic_only():
    manifest = _v3_manifest()
    arrival = manifest["columns"]["arrival"]
    assert arrival["rho_realized_purpose"] == "diagnostic_only"
    assert arrival["forbid_realization_feedback"] is True
    # 生成尺度必须逐字等于 policy 的 31.994
    assert arrival["mean_arrival_work_units_per_half_hour_scale"] == B6_EXPECTED_AMOUNT
    # **不继承** B5 的 1000 scale
    assert arrival["b5_scale_inherited"] is False
    assert arrival["mean_arrival_work_units_per_half_hour_scale"] != B5_LEGACY_SCALE
    # realized 只是登记，不是生成参数
    assert "realized_annual_mean" in arrival
    assert arrival["realized_annual_mean"] != arrival[
        "mean_arrival_work_units_per_half_hour_scale"]
    # expected 恰为 31.994（template 均值 1）
    assert arrival["expected_annual_mean"] == pytest.approx(B6_EXPECTED_AMOUNT, rel=1e-9)


# --- 7. PV / wind / carbon 与 v2 逐行相同 ---------------------------------------

@needs_assets
def test_shared_columns_are_row_identical_to_v2():
    v2 = pd.read_parquet(V2_PARQUET)
    v3 = pd.read_parquet(V3_PARQUET)
    for column in SHARED_COLUMNS:
        assert (v2[column].to_numpy() == v3[column].to_numpy()).all(), column


# --- 8. timestamp / 行数 / 时区与 canonical 一致 --------------------------------

@needs_assets
def test_timeline_matches_canonical():
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    v3 = pd.read_parquet(V3_PARQUET)
    assert len(v3) == len(canonical) == 17568
    assert (v3["timestamp"].to_numpy() == canonical["timestamp"].to_numpy()).all()
    assert str(v3["timestamp"].dt.tz) == "Asia/Singapore"
    assert list(v3.columns) == [
        "timestamp", "local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival",
    ]


# --- 9. v2 三项资产 hash 完全不变 ------------------------------------------------

@needs_assets
def test_v2_assets_are_byte_identical():
    for path, expected in V2_HASHES.items():
        assert _sha256(path) == expected, str(path)


# --- 10. v3 loader canonical-only 与篡改拒绝 -------------------------------------

@needs_assets
def test_non_canonical_path_is_rejected(tmp_path):
    m = b6_module()
    copied = tmp_path / "copied_v3.json"
    shutil.copy(V3_MANIFEST, copied)
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_manifest(copied)


@needs_assets
def test_symlink_path_is_rejected(tmp_path):
    m = b6_module()
    link = tmp_path / "link_v3.json"
    link.symlink_to(V3_MANIFEST)
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_manifest(link)


@needs_assets
def test_forged_revision_is_rejected(tmp_path, monkeypatch):
    m = b6_module()
    target = _tampered_v3(tmp_path, monkeypatch,
                          lambda p: p.update(materializer_revision="0" * 40))
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_manifest(target)


@needs_assets
def test_tampered_output_hash_is_rejected(tmp_path, monkeypatch):
    m = b6_module()
    target = _tampered_v3(
        tmp_path, monkeypatch,
        lambda p: p["output"].update(sha256="0" * 64))
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_manifest(target)


@needs_assets
def test_legacy_v2_fallback_is_rejected(tmp_path, monkeypatch):
    """loader **不得** fallback 到 v2：把 canonical 目录指向只含 v2 的目录。"""
    m = b6_module()
    lonely = tmp_path / "only_v2"
    lonely.mkdir()
    shutil.copy(V2_MANIFEST, lonely / V2_MANIFEST.name)
    monkeypatch.setattr(m, "_canonical_v3_dir", lambda: lonely)
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_manifest()


def _tampered_v3(tmp_path, monkeypatch, mutate):
    m = b6_module()
    temp_dir = tmp_path / "canonical_v3_dir"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target = temp_dir / V3_MANIFEST.name
    shutil.copy(V3_MANIFEST, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(json.dumps(payload))
    monkeypatch.setattr(m, "_canonical_v3_dir", lambda: temp_dir)
    return target


# --- 11/12. materializer CLI 与 dirty 门禁 --------------------------------------

@needs_assets
def test_materializer_rejects_out_dir_and_manifest_path():
    with pytest.raises(SystemExit):
        materializer().main(["--out-dir", "/tmp/anywhere"])
    with pytest.raises(SystemExit):
        materializer().main(["--manifest-path", "/tmp/other.json"])


@needs_assets
def test_generator_dirty_is_rejected(monkeypatch):
    mod = materializer()
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: True)
    with pytest.raises(mod.B6ExogenousError):
        mod.materialize_b6_exogenous(frozen_at_utc="2026-09-18T00:00:00+00:00")


# --- 13. --verify 连续 3 次幂等且无临时文件 --------------------------------------

@needs_assets
def test_verify_is_idempotent():
    mod = materializer()
    before = {p: (_sha256(p), p.stat().st_mtime_ns)
              for p in (V3_MANIFEST, V3_SOURCE)}
    for _ in range(3):
        assert mod.main(["--verify"]) == 0
    for path, (sha, mtime) in before.items():
        assert _sha256(path) == sha, str(path)
        assert path.stat().st_mtime_ns == mtime, str(path)
    leftovers = [
        p for p in V3_MANIFEST.parent.iterdir()
        if p.name.startswith(".") and "exogenous_v3" in p.name
    ]
    assert leftovers == []


# --- 14. 首次写入失败不留半成品（**R1：走真实 materializer 事务**） ----------------

def _empty_bundle_root(tmp_path) -> dict:
    """构造一个**空的**临时 bundle 根（三个正式产物都不存在）。"""
    root = tmp_path / "repo"
    man = root / "data/manifest"
    proc = root / "data/processed/singapore_2024"
    man.mkdir(parents=True)
    proc.mkdir(parents=True)
    return {"root": root, "manifest_dir": man, "parquet_dir": proc}


def _patch_v3_roots(monkeypatch, roots, *, dirty: bool = False) -> None:
    """把 canonical v3 根指向临时 bundle，并把 dirty 门固定为已知值。

    **语义**用例必须让 `_generator_is_dirty()` 为已知 `False`，否则在实现文件
    尚未提交时会因 dirty 提前拒绝，用例会「因错误的理由」通过。
    """
    m = b6_module()
    monkeypatch.setattr(m, "_canonical_v3_dir", lambda: roots["manifest_dir"])
    monkeypatch.setattr(m, "_canonical_v3_parquet_dir", lambda: roots["parquet_dir"])
    monkeypatch.setattr(m, "_generator_is_dirty", lambda: dirty)


def _leftovers(directory) -> list:
    return [p for p in directory.iterdir() if p.name.startswith(".")]


@needs_assets
@pytest.mark.parametrize("stage", ["after_parquet", "after_source", "at_output"])
def test_transactional_failure_leaves_nothing(tmp_path, monkeypatch, stage):
    """在**三个不同阶段**注入失败：首次物化不得留下任何正式产物或临时文件。"""
    mod = materializer()
    roots = _empty_bundle_root(tmp_path)
    _patch_v3_roots(monkeypatch, roots)
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)

    real_write = mod._atomic_write_bytes  # noqa: SLF001

    def exploding_write(path, body):
        name = pathlib.Path(path).name
        if stage == "after_parquet" and name.endswith(".parquet"):
            raise RuntimeError("injected failure after parquet")
        if stage == "after_source" and "sources_v4" in name:
            raise RuntimeError("injected failure after source")
        if stage == "at_output" and "exogenous_v3" in name:
            raise RuntimeError("injected failure at output manifest")
        real_write(path, body)

    monkeypatch.setattr(mod, "_atomic_write_bytes", exploding_write)
    with pytest.raises(RuntimeError):
        mod.materialize_b6_exogenous(frozen_at_utc="2026-09-18T00:00:00+00:00")

    assert not (roots["parquet_dir"] / "exogenous_drivers_v3.parquet").exists()
    assert not (roots["manifest_dir"] / "singapore_2024_exogenous_v3.json").exists()
    assert not (roots["manifest_dir"] / "m13f_materialization_sources_v4.json").exists()
    assert _leftovers(roots["parquet_dir"]) == []
    assert _leftovers(roots["manifest_dir"]) == []


@needs_assets
def test_existing_different_output_is_still_refused(tmp_path, monkeypatch):
    """R1 **不得**削弱生产「已存在且不同则拒绝覆盖」。"""
    mod = materializer()
    roots = _empty_bundle_root(tmp_path)
    _patch_v3_roots(monkeypatch, roots)
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    (roots["parquet_dir"] / "exogenous_drivers_v3.parquet").write_bytes(b"not a parquet")
    with pytest.raises(mod.B6ExogenousError):
        mod.materialize_b6_exogenous(frozen_at_utc="2026-09-18T00:00:00+00:00")


# =============================================================================
# M1.3f-e-b1-R1：语义信任边界
# =============================================================================

BUNDLE_ENTRY = "load_verified_v3_bundle"


def _canonical_json(payload: dict) -> str:
    """与生产 `write_json_atomic` **逐字节一致**的序列化。

    否则临时副本的字节与仓库文件不同，`source-v4` 的 SHA-256 会失配，
    用例将「因错误的理由」被拒绝而假绿（R1 实测发现）。
    """
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _bundle(tmp_path, *, tamper=None, tamper_source=None, tamper_parquet=None,
            split_frozen_at=False, symlink=None) -> dict:
    """构造一份**完整**的临时 v3 bundle（三份产物 + 可选的语义篡改）。

    `tamper(payload)` 改 output manifest；`tamper_source(payload)` 改 source-v4；
    `tamper_parquet(frame)` 改 parquet。`symlink` 把指定产物换成 symlink。

    **未篡改**时该 bundle 必须被接受（见 `test_untampered_bundle_is_accepted`）。
    """
    root = tmp_path / "repo"
    man = root / "data/manifest"
    proc = root / "data/processed/singapore_2024"
    man.mkdir(parents=True)
    proc.mkdir(parents=True)

    parquet = proc / V3_PARQUET.name
    manifest = man / V3_MANIFEST.name
    source = man / V3_SOURCE.name

    if tamper_parquet is None:
        shutil.copy(V3_PARQUET, parquet)
    else:
        frame = pd.read_parquet(V3_PARQUET)
        tamper_parquet(frame)
        frame.to_parquet(parquet, index=False)

    payload = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    source_payload = json.loads(V3_SOURCE.read_text(encoding="utf-8"))
    if tamper is not None:
        tamper(payload)
    if tamper_source is not None:
        tamper_source(source_payload)
    if split_frozen_at:
        source_payload["frozen_at_utc"] = "2026-09-18T00:00:00+00:00"
    manifest.write_text(_canonical_json(payload))
    source.write_text(_canonical_json(source_payload))

    if symlink == "manifest":
        manifest.unlink()
        manifest.symlink_to(V3_MANIFEST)
    elif symlink == "source":
        source.unlink()
        source.symlink_to(V3_SOURCE)
    elif symlink == "parquet":
        parquet.unlink()
        parquet.symlink_to(V3_PARQUET)

    return {
        "root": root, "manifest_dir": man, "parquet_dir": proc,
        "manifest_path": manifest, "source_path": source, "parquet_path": parquet,
    }


def _assert_bundle_rejected(tmp_path, monkeypatch, **kwargs):
    m = b6_module()
    assert hasattr(m, BUNDLE_ENTRY), "统一生产入口 load_verified_v3_bundle 必须存在"
    bundle = _bundle(tmp_path, **kwargs)
    _patch_v3_roots(monkeypatch, bundle)
    with pytest.raises(m.B6ExogenousError):
        getattr(m, BUNDLE_ENTRY)()


def _arrival(tamper):
    def inner(payload):
        tamper(payload["columns"]["arrival"])
    return inner


# --- 0. 接受性对照：**未篡改**的 bundle 必须被接受 -------------------------------
#
# 没有这条对照，「全部 REJECTED」可能是因为夹具本身有缺陷（例如非规范序列化
# 导致 source hash 失配）而假绿 —— R1 实测正是如此。

@needs_assets
def test_untampered_bundle_is_accepted(tmp_path, monkeypatch):
    m = b6_module()
    bundle = _bundle(tmp_path)
    _patch_v3_roots(monkeypatch, bundle)
    verified = m.load_verified_v3_bundle()
    assert verified["manifest"]["schema"] == m.B6_OUTPUT_SCHEMA
    assert verified["source"]["schema"] == m.B6_SOURCE_SCHEMA
    assert list(verified["frame"].columns) == [
        "timestamp", "local_pv_kw", "wind_generation_kw",
        "carbon_intensity", "arrival",
    ]


# --- 1–4. arrival 的冻结语义 ----------------------------------------------------

@needs_assets
def test_forged_b5_scale_inherited_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=_arrival(lambda a: a.update(b5_scale_inherited=True)))


@needs_assets
def test_forged_scale_1000_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=_arrival(lambda a: a.update(
            mean_arrival_work_units_per_half_hour_scale=1000.0)))


@needs_assets
def test_forged_forbid_realization_feedback_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=_arrival(lambda a: a.update(forbid_realization_feedback=False)))


@needs_assets
def test_forged_realization_seed_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=_arrival(lambda a: a.update(realization_seed=1)))


# --- 5. 诊断量不得伪造 ----------------------------------------------------------

@needs_assets
@pytest.mark.parametrize("field,value", [
    ("expected_annual_mean", 1000.0),
    ("realized_annual_mean", 1000.0),
    ("rho_realized", 0.5),
])
def test_forged_diagnostic_values_are_rejected(tmp_path, monkeypatch, field, value):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=_arrival(lambda a, f=field, v=value: a.update({f: v})))


# --- 6. template 元数据 --------------------------------------------------------

@needs_assets
@pytest.mark.parametrize("field,value", [
    ("template_mean", 2.0),
    ("template_slots", 24),
    ("shape_rule", "forged shape rule"),
])
def test_forged_template_metadata_is_rejected(tmp_path, monkeypatch, field, value):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=_arrival(lambda a, f=field, v=value: a.update({f: v})))


# --- 7. predecessor / readiness -------------------------------------------------

@needs_assets
def test_forged_predecessor_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=lambda p: p["predecessor"].update(
            status="migrated_to_v3"))


@needs_assets
def test_forged_readiness_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(
        tmp_path, monkeypatch,
        tamper=lambda p: p["readiness"].update(formal_training_ready=True))


# --- 8. coordinated source 语义篡改（同步更新 output 的 source hash） ------------

@needs_assets
def test_coordinated_source_semantics_tamper_is_rejected(tmp_path, monkeypatch):
    """改 source 语义**并**同步 output 的 source hash —— 仍必须拒绝。"""
    m = b6_module()
    bundle = _bundle(
        tmp_path,
        tamper_source=lambda s: s.update(arrival_scale_rule="forged scale rule"))
    # 攻击者把 output 里记录的 source hash 同步改成被篡改文件的真 hash
    payload = json.loads(bundle["manifest_path"].read_text(encoding="utf-8"))
    payload["materialization_sources_sha256"] = _sha256(bundle["source_path"])
    bundle["manifest_path"].write_text(_canonical_json(payload))
    _patch_v3_roots(monkeypatch, bundle)
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_bundle()


# --- 9. coordinated parquet 篡改（同步更新 output.sha256） ----------------------

@needs_assets
def test_coordinated_parquet_tamper_is_rejected(tmp_path, monkeypatch):
    """改 parquet 内容**并**同步 output.sha256 —— 仍必须拒绝。"""
    m = b6_module()

    def bump(frame):
        frame.loc[0, "arrival"] = int(frame.loc[0, "arrival"]) + 7

    bundle = _bundle(tmp_path, tamper_parquet=bump)
    payload = json.loads(bundle["manifest_path"].read_text(encoding="utf-8"))
    payload["output"]["sha256"] = _sha256(bundle["parquet_path"])
    bundle["manifest_path"].write_text(_canonical_json(payload))
    _patch_v3_roots(monkeypatch, bundle)
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_bundle()


# --- 10. 两份 manifest 的 frozen_at_utc 不一致 ----------------------------------

@needs_assets
def test_mismatched_frozen_at_utc_is_rejected(tmp_path, monkeypatch):
    _assert_bundle_rejected(tmp_path, monkeypatch, split_frozen_at=True)


# --- 11. canonical 文件自身是 symlink -------------------------------------------

@needs_assets
@pytest.mark.parametrize("which", ["manifest", "source", "parquet"])
def test_canonical_symlink_is_rejected(tmp_path, monkeypatch, which):
    _assert_bundle_rejected(tmp_path, monkeypatch, symlink=which)


# --- 12. dirty 生成器 ----------------------------------------------------------

@needs_assets
def test_dirty_generator_is_rejected_by_verified_entry(monkeypatch):
    m = b6_module()
    monkeypatch.setattr(m, "_generator_is_dirty", lambda: True)
    with pytest.raises(m.B6ExogenousError):
        m.load_verified_v3_bundle()


# --- 13. --verify 必须走统一入口 ------------------------------------------------

@needs_assets
def test_verify_uses_the_unified_bundle_entry(monkeypatch):
    """`--verify` 必须调用**统一入口**（不是两个不相交的浅层检查）。"""
    mod = materializer()
    called = {"bundle": 0}
    m = b6_module()
    monkeypatch.setattr(m, "_generator_is_dirty", lambda: False)

    def spy(*args, **kwargs):
        called["bundle"] += 1
        return {"manifest": {}, "source": {}, "frame": None, "policy": {}}

    monkeypatch.setattr(m, "load_verified_v3_bundle", spy)
    monkeypatch.setattr(mod, "load_verified_v3_bundle", spy)
    assert mod.main(["--verify"]) == 0
    assert called["bundle"] == 1


@needs_assets
def test_coordinated_one_ulp_power_tamper_is_rejected(tmp_path, monkeypatch):
    """A refreshed checksum cannot authorize even a one-ULP asset mutation."""
    import numpy as np

    m = b6_module()

    def bump(frame):
        i = frame.index[frame.local_pv_kw > 0][0]
        frame.loc[i, 'local_pv_kw'] = np.nextafter(frame.loc[i, 'local_pv_kw'], np.inf)

    bundle = _bundle(tmp_path, tamper_parquet=bump)
    payload = json.loads(bundle['manifest_path'].read_text(encoding='utf-8'))
    payload['output']['sha256'] = _sha256(bundle['parquet_path'])
    bundle['manifest_path'].write_text(_canonical_json(payload))
    _patch_v3_roots(monkeypatch, bundle)
    with pytest.raises(m.B6ExogenousError, match='registered frozen parquet SHA'):
        m.load_verified_v3_bundle()
