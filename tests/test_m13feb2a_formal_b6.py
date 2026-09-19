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
from pathlib import Path

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
    mean = float(pd.Series(values).mean())
    assert mean == pytest.approx(B6_ARRIVAL_MEAN, rel=1e-12)
    # 明确不是 B5 的 1000
    assert abs(mean - B5_LEGACY_MEAN) > 1.0
    # 也**不得**被 realized（Poisson 实现）均值冒充：两者必须可区分
    assert mean != V3_REALIZED_MEAN
    assert abs(mean - V3_REALIZED_MEAN) > 0.0
    assert float(pd.Series(values).mean()) == pytest.approx(
        B6_ARRIVAL_MEAN, rel=1e-12)


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


# --- 13. 原子失败无半成品（**R1：走真实 materializer 事务**） --------------------

def _empty_policy_dir(tmp_path) -> Path:
    d = tmp_path / "repo" / "data" / "manifest"
    d.mkdir(parents=True)
    return d


@needs_assets
def test_transactional_failure_leaves_nothing(tmp_path, monkeypatch):
    """在**真实** `materialize_policy_v3()` 事务中注入写入失败。"""
    mod = materializer()
    m = b6_formal()
    manifest_dir = _empty_policy_dir(tmp_path)
    monkeypatch.setattr(m, "_canonical_policy_v3_dir", lambda: manifest_dir)
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)

    def boom(path, body):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(mod, "_atomic_write_bytes", boom)
    with pytest.raises(RuntimeError):
        mod.materialize_policy_v3(frozen_at_utc="2026-09-19T00:00:00+00:00")

    assert not (manifest_dir / "singapore_2024_forecast_policy_v3.json").exists()
    assert [p for p in manifest_dir.iterdir() if p.name.startswith(".")] == []


@needs_assets
def test_existing_different_policy_v3_is_still_refused(tmp_path, monkeypatch):
    """R1 **不得**削弱「已存在且不同则拒绝覆盖」。"""
    mod = materializer()
    m = b6_formal()
    manifest_dir = _empty_policy_dir(tmp_path)
    monkeypatch.setattr(m, "_canonical_policy_v3_dir", lambda: manifest_dir)
    monkeypatch.setattr(mod, "_generator_is_dirty", lambda: False)
    target = manifest_dir / "singapore_2024_forecast_policy_v3.json"
    target.write_text('{"schema": "forged"}')
    with pytest.raises(mod.FormalB6PolicyError):
        mod.materialize_policy_v3(frozen_at_utc="2026-09-19T00:00:00+00:00")


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


# =============================================================================
# M1.3f-e-b2-a-R1：policy-v3 语义、revision 覆盖与真实 leakage 回归
# =============================================================================

def _canonical_json(payload: dict) -> str:
    """与生产 `_canonical_json` **逐字节一致**的序列化（防止夹具假绿）。"""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _forged_policy_v3(tmp_path, monkeypatch, mutate) -> Path:
    """把 policy-v3 复制到临时 canonical 目录并篡改，monkeypatch 私有 resolver。"""
    m = b6_formal()
    d = tmp_path / "canonical_dir"
    d.mkdir(parents=True, exist_ok=True)
    target = d / "singapore_2024_forecast_policy_v3.json"
    payload = json.loads(POLICY_V3.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(_canonical_json(payload))
    monkeypatch.setattr(m, "_canonical_policy_v3_dir", lambda: d)
    return target


# --- R1-1. 未篡改接受性对照（防止全部 REJECTED 是夹具假绿） ----------------------

@needs_assets
def test_untampered_policy_v3_is_accepted(tmp_path, monkeypatch):
    m = b6_formal()
    d = tmp_path / "canonical_dir"
    d.mkdir(parents=True)
    target = d / "singapore_2024_forecast_policy_v3.json"
    shutil.copy(POLICY_V3, target)
    monkeypatch.setattr(m, "_canonical_policy_v3_dir", lambda: d)
    verified = m.load_verified_policy_v3()
    assert verified["schema"] == m.POLICY_V3_SCHEMA


# --- R1-2. policy-v3 语义伪造（改前 ACCEPTED，改后必须 REJECTED） ---------------

@needs_assets
@pytest.mark.parametrize("mutate", [
    pytest.param(lambda p: p.update(arrival_forecast_rule="lambda_t = 1000"),
                 id="arrival_forecast_rule"),
    pytest.param(lambda p: p.update(seed_policy=20240916), id="seed_policy"),
    pytest.param(lambda p: p.update(method="forged"), id="extra_business_field"),
    pytest.param(lambda p: p["supersedes"].update(policy_manifest_sha256="0" * 64),
                 id="supersedes_sha"),
    pytest.param(lambda p: p["supersedes"].update(policy_manifest_schema="forged"),
                 id="supersedes_schema"),
    pytest.param(lambda p: p["supersedes"].update(note="forged"), id="supersedes_note"),
    pytest.param(lambda p: p["supersedes"].update(status="replaced"), id="supersedes_status"),
    pytest.param(lambda p: p["readiness"].update(formal_b6_scenario_candidate_ready=False),
                 id="candidate_ready"),
    pytest.param(lambda p: p.update(information_policy="forged"), id="information_policy"),
    pytest.param(lambda p: p.update(target_policy="forged"), id="target_policy"),
    pytest.param(lambda p: p.update(arrival_forecast_uses_expected=False),
                 id="uses_expected"),
    pytest.param(lambda p: p.update(empirical_workload_claim=True), id="empirical_claim"),
    pytest.param(lambda p: p.update(contract_version="contract-v8"), id="contract"),
    pytest.param(lambda p: p.update(period_steps=47), id="period_steps"),
    pytest.param(lambda p: p.update(frequency="1h"), id="frequency"),
])
def test_forged_policy_v3_semantics_are_rejected(tmp_path, monkeypatch, mutate):
    m = b6_formal()
    target = _forged_policy_v3(tmp_path, monkeypatch, mutate)
    with pytest.raises(m.FormalB6Error):
        m.load_verified_policy_v3(target)


@needs_assets
def test_policy_v3_must_equal_the_rebuilt_candidate(tmp_path, monkeypatch):
    """整体重建比对：任何未抽查字段的伪造都会被逐字段相等拒绝。"""
    m = b6_formal()
    target = _forged_policy_v3(
        tmp_path, monkeypatch,
        lambda p: p.update(arrival_forecast_rule="lambda_t = template * 1000"))
    with pytest.raises(m.FormalB6Error):
        m.load_verified_policy_v3(target)


# --- R1-3. revision 覆盖 --------------------------------------------------------

REQUIRED_REVISION_PATHS = {
    "contracts/__init__.py",
    "contracts/models.py",
    "contracts/validators.py",
    "scenario/forecast.py",
    "scripts/materialize_singapore_forecast_policy.py",
    "scenario/splits.py",
    "scenario/formal_scenario.py",
    "scenario/exogenous_drivers.py",
    "scenario/arrival_intensity_policy.py",
    "scenario/exogenous_drivers_b6.py",
    "scenario/formal_scenario_b6.py",
    "scripts/materialize_formal_forecast_policy_b6.py",
}


@needs_assets
def test_revision_paths_cover_every_semantic_implementation_file():
    m = b6_formal()
    missing = REQUIRED_REVISION_PATHS - set(m.B6_FORMAL_SOURCE_PATHS)
    assert missing == set(), f"B6_FORMAL_SOURCE_PATHS 缺少 {sorted(missing)}"


@needs_assets
def test_dirty_check_and_revision_use_the_same_path_set(monkeypatch):
    """dirty 检查与 revision 必须由**同一**集合驱动。"""
    m = b6_formal()
    seen: list[tuple] = []
    real_git = m._git

    def spy(*args):
        seen.append(args)
        return real_git(*args)

    monkeypatch.setattr(m, "_git", spy)
    m.b6_formal_code_revision()
    m._generator_is_dirty()
    assert len(seen) == 2
    rev_paths = tuple(seen[0])[-len(m.B6_FORMAL_SOURCE_PATHS):]
    dirty_paths = tuple(seen[1])[-len(m.B6_FORMAL_SOURCE_PATHS):]
    assert rev_paths == tuple(m.B6_FORMAL_SOURCE_PATHS)
    assert dirty_paths == tuple(m.B6_FORMAL_SOURCE_PATHS)


# --- R1-4. 真实 future-truth mutation 回归 ---------------------------------------

VALIDATION_ORIGIN = 1000
VALIDATION_CUTOFF = 4
TARGET_COLUMNS = ("price_sgd_per_kwh", "system_load_mw", "temperature_deg_c",
                  "ghi_w_per_m2", "wind_speed_10m_mps")


def _temp_chain(tmp_path, monkeypatch, *, mutate=None) -> dict:
    """构造**完整自洽**的临时链：canonical / split / policy-v2 / policy-v3。

    `mutate(frame)` 就地修改 canonical；随后 canonical manifest、split manifest、
    policy-v2、policy-v3 的 path/hash **全部重新同步**，因此生产入口看到的是
    一条真实、自洽且**已改变**的链。
    """
    m = b6_formal()
    import scenario.splits as splits_module
    from scripts.materialize_singapore_forecast_policy import build_forecast_policy_manifest

    root = tmp_path / "repo"
    man = root / "data/manifest"
    proc = root / "data/processed/singapore_2024"
    man.mkdir(parents=True)
    proc.mkdir(parents=True)

    frame = pd.read_parquet(CANONICAL_PARQUET)
    if mutate is not None:
        mutate(frame)
    parquet = proc / "half_hour.parquet"
    frame.to_parquet(parquet, index=False)

    canonical_manifest = man / "singapore_2024_half_hour.json"
    cm = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    cm["output_parquet_sha256"] = _sha256(parquet)
    canonical_manifest.write_text(_canonical_json(cm))

    split_manifest = man / "singapore_2024_splits.json"
    sp = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    sp["canonical_parquet_sha256"] = _sha256(parquet)
    sp["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    sp["train_only_statistics_source"]["canonical_parquet_sha256"] = _sha256(parquet)
    if mutate is not None:
        sp["train_only_statistics"] = splits_module.train_only_statistics(
            frame.iloc[: sp["splits"]["train"]["row_end_exclusive"]])
    split_manifest.write_text(_canonical_json(sp))

    policy_v2 = man / "singapore_2024_forecast_policy_v2.json"
    p2 = build_forecast_policy_manifest(
        canonical_parquet_path=parquet,
        canonical_manifest_path=canonical_manifest,
        split_manifest_path=split_manifest,
        frozen_at_utc="2026-09-18T00:00:00+00:00",
    )
    policy_v2.write_text(_canonical_json(p2))

    # 把**生产入口**的私有 resolver 指向临时链
    monkeypatch.setattr(m, "_canonical_policy_v3_dir", lambda: man)
    monkeypatch.setattr(m, "_canonical_parquet_path", lambda: parquet)
    monkeypatch.setattr(m, "_canonical_manifest_path", lambda: canonical_manifest)
    monkeypatch.setattr(m, "_split_manifest_path", lambda: split_manifest)
    monkeypatch.setattr(m, "_seasonal_policy_v2_path", lambda: policy_v2)

    policy_v3 = man / "singapore_2024_forecast_policy_v3.json"
    p3 = m.build_policy_v3_manifest(frozen_at_utc="2026-09-18T00:00:00+00:00")
    policy_v3.write_text(_canonical_json(p3))

    return {
        "canonical_parquet_path": parquet,
        "canonical_manifest_path": canonical_manifest,
        "split_manifest_path": split_manifest,
        "policy_manifest_path": policy_v3,
        "frame": frame,
    }


def _build(chain: dict):
    return b6_formal().build_formal_scenario_b6(
        split="validation",
        origin=VALIDATION_ORIGIN,
        forecast_cutoff=VALIDATION_CUTOFF,
        canonical_parquet_path=chain["canonical_parquet_path"],
        canonical_manifest_path=chain["canonical_manifest_path"],
        split_manifest_path=chain["split_manifest_path"],
        policy_manifest_path=chain["policy_manifest_path"],
    )


def _global_window() -> tuple[int, int]:
    """validation 的 `origin=1000` 对应的全局行区间与 target 窗口。"""
    return 10224 + VALIDATION_ORIGIN, 10224 + VALIDATION_ORIGIN + VALIDATION_CUTOFF


def _mutate_target(frame) -> None:
    """修改全局 `[origin, origin+C)` 的**未来真值**（五列全部改变）。"""
    start, stop = _global_window()
    for offset, column in enumerate(TARGET_COLUMNS):
        window = slice(start, stop)
        frame.loc[window, column] = frame.loc[window, column] + (10.0 * (offset + 1))


def _mutate_history(frame) -> None:
    """反向控制：修改 `[origin-48, origin)` 的**历史窗口**。"""
    start, _ = _global_window()
    for offset, column in enumerate(TARGET_COLUMNS):
        window = slice(start - 48, start)
        frame.loc[window, column] = frame.loc[window, column] + (10.0 * (offset + 1))


@needs_assets
def test_target_future_mutation_leaves_all_seven_forecasts_unchanged(
        tmp_path, monkeypatch):
    """**真实** mutation：改 `[origin, origin+C)` 的未来真值 → 七条 forecast 逐位不变。"""
    base_chain = _temp_chain(tmp_path / "base", monkeypatch)
    baseline = _build(base_chain)

    mutated_chain = _temp_chain(tmp_path / "mut", monkeypatch, mutate=_mutate_target)
    # mutation 必须**真的**改了数据（否则用例无意义）
    start, stop = _global_window()
    before = pd.read_parquet(base_chain["canonical_parquet_path"])
    after = pd.read_parquet(mutated_chain["canonical_parquet_path"])
    for column in TARGET_COLUMNS:
        assert not (before.loc[start:stop - 1, column].to_numpy()
                    == after.loc[start:stop - 1, column].to_numpy()).all(), column

    mutated = _build(mutated_chain)
    for field in b6_formal().FORMAL_B6_SOURCE_KINDS:
        assert list(getattr(mutated, field)) == list(getattr(baseline, field)), field


@needs_assets
def test_history_mutation_changes_the_transformed_forecasts(tmp_path, monkeypatch):
    """**反向控制**：改 `[origin-48, origin)` → 对应 forecast **必须变化**。"""
    base_chain = _temp_chain(tmp_path / "base", monkeypatch)
    baseline = _build(base_chain)

    hist_chain = _temp_chain(tmp_path / "hist", monkeypatch, mutate=_mutate_history)
    start, _ = _global_window()
    before = pd.read_parquet(base_chain["canonical_parquet_path"])
    after = pd.read_parquet(hist_chain["canonical_parquet_path"])
    for column in TARGET_COLUMNS:
        assert not (before.loc[start - 48:start - 1, column].to_numpy()
                    == after.loc[start - 48:start - 1, column].to_numpy()).all(), column

    mutated = _build(hist_chain)
    for field in ("price_forecast", "load_forecast", "temperature_forecast"):
        assert list(getattr(mutated, field)) != list(getattr(baseline, field)), field


# --- R1-5. template 源经私有解析器，但生产路径下恒等 -----------------------------

@needs_assets
def test_template_source_resolver_matches_the_production_bundle():
    """`_v3_template_source()` 在生产路径下必须与 verified v3 template **恒等**。"""
    m = b6_formal()
    assert (m._v3_template_source() == m.verified_v3_template()).all()
