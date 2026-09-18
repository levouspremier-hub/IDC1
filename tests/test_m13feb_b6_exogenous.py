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
    # realized 只是登记，不是生成参数
    assert "realized_annual_mean" in arrival
    assert arrival["realized_annual_mean"] != arrival[
        "mean_arrival_work_units_per_half_hour_scale"]


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


# --- 14. 首次写入失败不留半成品 --------------------------------------------------

@needs_assets
def test_first_write_failure_leaves_nothing(tmp_path, monkeypatch):
    mod = materializer()
    target = tmp_path / "v3_out.json"

    def boom(path, body):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(mod, "_atomic_write_bytes", boom)
    with pytest.raises(RuntimeError):
        mod._atomic_write_bytes(target, b"{}")  # noqa: SLF001
    assert list(tmp_path.iterdir()) == []
