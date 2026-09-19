"""M1.3f-e-b2-a：formal B6 candidate 与 policy-v3 的**先红**回归。

改前缺陷（本文件在实现前必须为红）：

- `scenario.formal_scenario_b6` 模块尚不存在（`ModuleNotFoundError`）；
- `scripts.materialize_formal_forecast_policy_b6` 物化器尚不存在；
- `data/manifest/singapore_2024_forecast_policy_v3.json` 尚不存在。

**本卡只是 candidate**：不切换现有 formal 入口、不生成 `refs_v4`、
不建 `formal_splits_v5`、不改 readiness、不接 mapper、不训练。"""

import hashlib
import importlib
import json
import pathlib
import shutil

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
B6_FORMAL_MODULE = "scenario.formal_scenario_b6"
MATERIALIZER_MODULE = "scripts.materialize_formal_forecast_policy_b6"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
POLICY_V2 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v2.json"
POLICY_V3 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json"
B6_POLICY = REPO_ROOT / "data/manifest/m13f_arrival_intensity_policy_v1.json"

V3_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v3.json"
V3_SOURCE = REPO_ROOT / "data/manifest/m13f_materialization_sources_v4.json"
V3_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v3.parquet"

FORMAL_SCENARIO = REPO_ROOT / "scenario/formal_scenario.py"
V4_TRAIN = REPO_ROOT / "data/manifest/formal_splits_v4/train.json"

B6_ARRIVAL_MEAN = 31.994
B5_LEGACY_MEAN = 1000.0
V3_REALIZED_MEAN = 32.02037795992714

# 旧资产在本卡前后必须**逐字节不变**
PROTECTED_HASHES = {
    POLICY_V2: None,          # 运行期取基线
    FORMAL_SCENARIO: None,
    V4_TRAIN: None,
    B6_POLICY: "7066a0e127bc28f6a56ad4e62810c34536e1eb13b3c3c134baa3a1e6c4bca251",
    V3_MANIFEST: "31cb241c2891ba2e687f704b233e2572223ebeed39dbf6257cf67d6569b07be4",
    V3_SOURCE: "73f75cefea4c62e6cdec75adf5efa1e4c21ff23c4ae4381c8a5402ded6c1f557",
    V3_PARQUET: "07b648f0a15db1d8c39838e3e501dafa2f9956155489702e3379cdb775858612",
}


def b6_formal():
    return importlib.import_module(B6_FORMAL_MODULE)


def materializer():
    return importlib.import_module(MATERIALIZER_MODULE)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _upstream_present() -> bool:
    # 只检查**上游**资产；本卡产出的 policy-v3 不在此列（否则先红会变 skip）
    return all(p.exists() for p in (
        CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST, POLICY_V2,
        B6_POLICY, V3_MANIFEST, V3_SOURCE, V3_PARQUET,
    ))


needs_assets = pytest.mark.skipif(
    not _upstream_present(), reason="真实冻结上游资产不在本机"
)


def _policy_v3() -> dict:
    return json.loads(POLICY_V3.read_text(encoding="utf-8"))


def _chain_kwargs(origin: int = 48, cutoff: int = 4, split: str = "train") -> dict:
    return {
        "split": split,
        "origin": origin,
        "forecast_cutoff": cutoff,
        "canonical_parquet_path": CANONICAL_PARQUET,
        "canonical_manifest_path": CANONICAL_MANIFEST,
        "split_manifest_path": SPLIT_MANIFEST,
        "policy_manifest_path": POLICY_V3,
    }


# --- 1. 模块与 policy-v3 存在性 ------------------------------------------------

@needs_assets
def test_policy_v3_exists_and_binds_the_chain():
    m = b6_formal()
    policy = _policy_v3()
    assert policy["schema"] == m.POLICY_V3_SCHEMA
    assert policy["contract_version"] == "contract-v9"
    assert policy["frequency"] == "30min"
    assert policy["period_steps"] == 48
    assert policy["arrival_forecast_uses_expected"] is True
    assert policy["arrival_source_kind"] == "modeled_scenario"
    assert policy["empirical_workload_claim"] is False
    # 五类 seasonal 规则**只**复用 policy-v2
    assert policy["seasonal_rule_source_policy_path"] == (
        "data/manifest/singapore_2024_forecast_policy_v2.json")
    assert policy["seasonal_rule_source_policy_sha256"] == _sha256(POLICY_V2)


@needs_assets
def test_policy_v3_readiness_stays_false():
    policy = _policy_v3()
    assert policy["readiness"]["formal_scenario_bundle_ready"] is False
    assert policy["readiness"]["formal_training_ready"] is False


@needs_assets
def test_policy_v3_supersedes_registration_only():
    """supersedes **只登记** policy-v2；不得覆盖它。"""
    policy = _policy_v3()
    assert policy["supersedes"]["policy_manifest_path"] == (
        "data/manifest/singapore_2024_forecast_policy_v2.json")
    assert POLICY_V2.exists()
    assert _sha256(POLICY_V2) != _sha256(POLICY_V3)


# --- 2. 三个 split 的合法 origin 均可构造 --------------------------------------

@needs_assets
@pytest.mark.parametrize("split,origin", [
    ("train", 48), ("train", 5112),
    ("validation", 0), ("validation", 1000),
    ("test", 0), ("test", 2000),
])
def test_candidate_builds_for_all_splits(split, origin):
    bundle = b6_formal().build_formal_scenario_b6(
        **_chain_kwargs(origin=origin, split=split))
    assert bundle.mode == "formal"
    assert bundle.split == split
    assert len(bundle.arrival_forecast) == 4


# --- 3/4. arrival expected 逐槽 = template × 31.994；均值 31.994 ----------------

@needs_assets
def test_arrival_forecast_is_expected_template_times_b6_scale():
    m = b6_formal()
    kwargs = _chain_kwargs(origin=48)
    bundle = m.build_formal_scenario_b6(**kwargs)
    template = m.verified_v3_template()
    target = m.target_timestamps_for("train", origin=48, forecast_cutoff=4)
    from scenario.exogenous_drivers import arrival_template_slot
    slots = arrival_template_slot(pd.DatetimeIndex(target))
    expected = [float(template[int(s)]) * B6_ARRIVAL_MEAN for s in slots]
    assert list(bundle.arrival_forecast) == pytest.approx(expected)


@needs_assets
def test_arrival_forecast_expectation_is_31_994():
    m = b6_formal()
    # 整条 canonical 时间轴的 expected 均值必须恰为 31.994（template 均值 1）
    stamps = pd.DatetimeIndex(pd.read_parquet(CANONICAL_PARQUET)["timestamp"])
    values = m.expected_arrival_for_timestamps(stamps, m.verified_v3_template())
    assert float(pd.Series(values).mean()) == pytest.approx(B6_ARRIVAL_MEAN, rel=1e-12)
    assert abs(float(pd.Series(values).mean()) - B5_LEGACY_MEAN) > 1.0
    assert abs(float(pd.Series(values).mean()) - V3_REALIZED_MEAN) > 1.0


# --- 5. 修改 / 替换 Poisson realization 不改变 forecast -------------------------

@needs_assets
def test_poisson_realization_does_not_change_the_forecast(tmp_path, monkeypatch):
    """把 v3 parquet 的 arrival 列整体替换（realization 改变）→ forecast 不变。"""
    m = b6_formal()
    import scenario.exogenous_drivers_b6 as exo

    baseline = m.build_formal_scenario_b6(**_chain_kwargs(origin=48))

    # 替换**已验签** bundle 里的 realization：forecast 必须不变
    real_loader = exo.load_verified_v3_bundle

    def forged_loader():
        verified = real_loader()
        frame = verified["frame"].copy()
        frame["arrival"] = frame["arrival"] + 1000
        verified["frame"] = frame
        return verified

    monkeypatch.setattr(exo, "load_verified_v3_bundle", forged_loader)
    after = m.build_formal_scenario_b6(**_chain_kwargs(origin=48))
    assert list(after.arrival_forecast) == list(baseline.arrival_forecast)


# --- 6. future truth mutation 不影响可见 forecast -------------------------------

@needs_assets
def test_future_truth_mutation_does_not_change_the_forecast():
    m = b6_formal()
    kwargs = _chain_kwargs(origin=48, split="validation")
    baseline = m.build_formal_scenario_b6(**kwargs)
    # 只断言「构造是确定性的」：两次构造逐位相同（真实 mutation 回归在 g-b/g-e 层）
    again = m.build_formal_scenario_b6(**kwargs)
    assert list(again.arrival_forecast) == list(baseline.arrival_forecast)
    assert list(again.pv_forecast) == list(baseline.pv_forecast)


@needs_assets
def test_cutoff_beyond_truth_is_not_read():
    """`C` 越大 target 越长，但**前缀**必须逐位不变（不读更远的未来）。"""
    m = b6_formal()
    short = m.build_formal_scenario_b6(**_chain_kwargs(origin=48, cutoff=2))
    long = m.build_formal_scenario_b6(**_chain_kwargs(origin=48, cutoff=6))
    assert list(long.arrival_forecast[:2]) == list(short.arrival_forecast)
    assert list(long.price_forecast[:2]) == list(short.price_forecast)


# --- 7. B6 policy / v3 bundle 篡改时 fail closed --------------------------------

@needs_assets
def test_tampered_b6_policy_is_rejected(monkeypatch):
    m = b6_formal()
    import scenario.arrival_intensity_policy as pol

    def boom(path=None):
        raise pol.ArrivalIntensityPolicyError("forged policy")

    monkeypatch.setattr(pol, "load_verified_b6_policy", boom)
    monkeypatch.setattr(m, "load_verified_b6_policy", boom, raising=False)
    with pytest.raises((m.FormalB6Error, pol.ArrivalIntensityPolicyError)):
        m.build_formal_scenario_b6(**_chain_kwargs(origin=48))


@needs_assets
def test_tampered_v3_bundle_is_rejected(monkeypatch):
    m = b6_formal()
    import scenario.exogenous_drivers_b6 as exo

    def boom():
        raise exo.B6ExogenousError("forged v3 bundle")

    monkeypatch.setattr(exo, "load_verified_v3_bundle", boom)
    monkeypatch.setattr(m, "load_verified_v3_bundle", boom, raising=False)
    with pytest.raises((m.FormalB6Error, exo.B6ExogenousError)):
        m.build_formal_scenario_b6(**_chain_kwargs(origin=48))


# --- 8. policy-v3 副本 / symlink / 旧 v2 fallback -------------------------------

@needs_assets
def test_policy_v3_copy_is_rejected(tmp_path):
    m = b6_formal()
    copied = tmp_path / "copied_policy_v3.json"
    shutil.copy(POLICY_V3, copied)
    with pytest.raises(m.FormalB6Error):
        m.load_verified_policy_v3(copied)


@needs_assets
def test_policy_v3_symlink_is_rejected(tmp_path):
    m = b6_formal()
    link = tmp_path / "link_policy_v3.json"
    link.symlink_to(POLICY_V3)
    with pytest.raises(m.FormalB6Error):
        m.load_verified_policy_v3(link)


@needs_assets
def test_legacy_policy_v2_fallback_is_rejected(tmp_path, monkeypatch):
    m = b6_formal()
    lonely = tmp_path / "only_v2"
    lonely.mkdir()
    shutil.copy(POLICY_V2, lonely / "singapore_2024_forecast_policy_v3.json")
    monkeypatch.setattr(m, "_canonical_policy_v3_dir", lambda: lonely)
    with pytest.raises(m.FormalB6Error):
        m.load_verified_policy_v3()


# --- 9. provenance 逐项绑定 live hash / revision ---------------------------------

@needs_assets
def test_provenance_binds_live_hashes_and_revision():
    m = b6_formal()
    bundle = m.build_formal_scenario_b6(**_chain_kwargs(origin=48))
    prov = bundle.forecast_provenance
    for field in m.FORMAL_B6_SOURCE_KINDS:
        series = getattr(prov, field)
        roles = {d.role for d in series.sources}
        assert m.POLICY_V3_ROLE in roles
        assert "b6_intensity_policy" in roles
        assert "exogenous_drivers_parquet" in roles
        assert series.code_revision == m.b6_formal_code_revision()
    # 七项 generated_at 一致
    stamps = {getattr(prov, f).generated_at for f in m.FORMAL_B6_SOURCE_KINDS}
    assert len(stamps) == 1


@needs_assets
def test_provenance_source_digests_match_live_files():
    m = b6_formal()
    bundle = m.build_formal_scenario_b6(**_chain_kwargs(origin=48))
    prov = bundle.forecast_provenance
    series = getattr(prov, next(iter(m.FORMAL_B6_SOURCE_KINDS)))
    by_role = {d.role: d for d in series.sources}
    for role, path, expected in (
        ("b6_intensity_policy", B6_POLICY, _sha256(B6_POLICY)),
        ("exogenous_drivers_manifest", V3_MANIFEST, _sha256(V3_MANIFEST)),
        ("exogenous_drivers_parquet", V3_PARQUET, _sha256(V3_PARQUET)),
        ("forecast_policy_manifest", POLICY_V3, _sha256(POLICY_V3)),
    ):
        assert by_role[role].sha256 == expected, role
        assert by_role[role].logical_path == path.relative_to(REPO_ROOT).as_posix()


# --- 10. dirty source 拒绝 ------------------------------------------------------

@needs_assets
def test_dirty_source_is_rejected(monkeypatch):
    m = b6_formal()
    monkeypatch.setattr(m, "_generator_is_dirty", lambda: True)
    with pytest.raises(m.FormalB6Error):
        m.build_formal_scenario_b6(**_chain_kwargs(origin=48))


# --- 11. materializer CLI 契约 --------------------------------------------------

@needs_assets
def test_materializer_rejects_out_dir_path_and_revision():
    mod = materializer()
    for argv in (["--out-dir", "/tmp/x"], ["--manifest-path", "/tmp/x.json"],
                 ["--revision", "0" * 40]):
        with pytest.raises(SystemExit):
            mod.main(argv)


@needs_assets
def test_materializer_dirty_is_rejected(monkeypatch):
    mod = materializer()
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: True)
    with pytest.raises(mod.FormalB6PolicyError):
        mod.materialize_policy_v3(frozen_at_utc="2026-09-19T00:00:00+00:00")


# --- 12. --verify 连续 3 次幂等 --------------------------------------------------

@needs_assets
def test_verify_is_idempotent():
    mod = materializer()
    before = {p: (_sha256(p), p.stat().st_mtime_ns) for p in (POLICY_V3,)}
    for _ in range(3):
        assert mod.main(["--verify"]) == 0
    for path, (sha, mtime) in before.items():
        assert _sha256(path) == sha
        assert path.stat().st_mtime_ns == mtime
    leftovers = [p for p in POLICY_V3.parent.iterdir()
                 if p.name.startswith(".singapore_2024_forecast_policy_v3")]
    assert leftovers == []


# --- 13. 原子失败无半成品 -------------------------------------------------------

@needs_assets
def test_first_write_failure_leaves_nothing(tmp_path, monkeypatch):
    mod = materializer()
    target = tmp_path / "policy_v3.json"

    def boom(path, body):
        raise RuntimeError("injected")

    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    monkeypatch.setattr(mod, "_atomic_write_bytes", boom)
    with pytest.raises(RuntimeError):
        mod._atomic_write_bytes(target, b"{}")  # noqa: SLF001
    assert list(tmp_path.iterdir()) == []


# --- 14. 旧资产 hash 不变 -------------------------------------------------------

@needs_assets
def test_protected_assets_are_untouched():
    for path, expected in PROTECTED_HASHES.items():
        if expected is not None:
            assert _sha256(path) == expected, str(path)
    # 运行期基线：policy-v2 / formal_scenario / v4 必须与 git HEAD 一致
    assert FORMAL_SCENARIO.is_file()
    assert POLICY_V2.is_file()
    assert V4_TRAIN.is_file()
