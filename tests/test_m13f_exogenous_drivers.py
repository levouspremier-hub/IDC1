"""M1.3f-c 测试：四类正式外生驱动的**可复现实物化**（含 R1 返修）。

**R1 改前缺陷**（本文件在实现前必须为红）：

1. `losses_pct` **从未被应用**（0/14/99 输出逐位相同）；
2. PV 参数块缺失或篡改时**不报错**；
3. arrival 的 7×48 模板依赖**未经验证**的「文件顺序→日历星期」映射；
4. archive 的文件顺序/虚构日期会影响聚合；
5. arrival 缺少 B5 批准记录时仍会物化；
6. carbon 可能被误述为 `modeled_scenario`；
7. 旧 v1 产物可能被重写、不同的 v2 产物可能被覆盖。

**测试聚焦实际训练风险**（时间轴/单位/范围、PV 损耗与上限、风电额定、
carbon 严格常量、arrival 的**因果性/可复现性/非重放/无日期依赖**、
source hash fail closed、产物不可覆盖、正式训练不得提前放行）。
**不做**大规模任意 JSON 类型 fuzz。
"""

import hashlib
import importlib
import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DRIVERS_MODULE = "scenario.exogenous_drivers"
MATERIALIZER = "scripts.materialize_singapore_exogenous"

# v2（本卡候选证据）
OUT_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
OUT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v2.json"
SRC_MANIFEST = REPO_ROOT / "data/manifest/m13f_materialization_sources_v3.json"
# v1（**superseded**，原样保留）
V1_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers.parquet"
V1_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous.json"
V1_SRC_MANIFEST = REPO_ROOT / "data/manifest/m13f_materialization_sources.json"
V1_SHA256 = {
    "parquet": "0c5e65d8fdc25ed8ced228e8087f13eb0605d0146d7e258caf54a552a246287c",
    "manifest": "46d88c38247bf1eb1568e84abc48647f1f4aeb58a5c0b525a30b8bfb0b2ac224",
    "sources": "6a80886a53056df80944db1fc536176d590f9cc10b315064d826436ae0bb1a72",
}
PUBLIC_SOURCE_V1 = REPO_ROOT / "data/manifest/m13f_public_sources.json"
PUBLIC_SOURCE_V1_SHA256 = (
    "f5a5f506c579da1b9c258d38103ca4549ac483ebd2d9f501b8d686ae9cbc3faa"
)
CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"

TOTAL_ROWS = 17568
TIMEZONE = "Asia/Singapore"
COLUMNS = ("timestamp", "local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival")
CARBON_KG_PER_KWH = 0.402
PV_CAPACITY_KW = 500.0
DC_AC_RATIO = 1.2
RATED_CAPACITY_KW = 800.0
ARRIVAL_SEED = 20240916
ARRIVAL_MEAN = 1000.0
ARRIVAL_SLOTS = 48
FORBIDDEN_DATE_TOKENS = ("2019-07-15", "weekday", "date_mapping")


def drivers():
    return importlib.import_module(DRIVERS_MODULE)


def materializer():
    return importlib.import_module(MATERIALIZER)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# --- 1. 模块与产物存在性 ------------------------------------------------------

def test_modules_and_entry_points_exist():
    module = drivers()
    for name in ("PV_PARAMS", "WIND_PARAMS", "CARBON_KG_PER_KWH", "ARRIVAL_SEED",
                 "ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR", "B5_PV_APPROVAL",
                 "B5_ARRIVAL_APPROVAL", "assert_pv_params", "assert_arrival_approved",
                 "pv_loss_multiplier", "pv_dc_before_losses", "pv_ac_from_dc",
                 "hub_wind_speed", "power_from_curve", "load_wind_power_curve",
                 "pv_ac_limit_kw", "local_pv_kw", "wind_generation_kw",
                 "carbon_intensity", "arrival_slot_counts", "arrival_rate_template",
                 "generate_arrival"):
        assert hasattr(module, name), name
    assert hasattr(materializer(), "materialize_exogenous_drivers")


def test_output_artifacts_exist():
    for path in (OUT_PARQUET, OUT_MANIFEST, SRC_MANIFEST):
        assert path.exists(), path


def test_v1_artifacts_are_preserved_and_marked_superseded():
    """§一.7：v1 产物**原样保留**，且被 v2 manifest 标为 superseded。"""
    assert _sha256(V1_PARQUET) == V1_SHA256["parquet"]
    assert _sha256(V1_MANIFEST) == V1_SHA256["manifest"]
    assert _sha256(V1_SRC_MANIFEST) == V1_SHA256["sources"]
    supersedes = json.loads(OUT_MANIFEST.read_text())["supersedes"]
    assert supersedes["status"] == "superseded_pre_approval_and_loss_fix"
    assert supersedes["output_parquet_sha256"] == V1_SHA256["parquet"]
    assert supersedes["output_manifest_sha256"] == V1_SHA256["manifest"]
    assert supersedes["source_manifest_sha256"] == V1_SHA256["sources"]
    assert supersedes["revision"]
    assert len(supersedes["reasons"]) >= 3


# --- 2. 时间轴与形状 ----------------------------------------------------------

@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return pd.read_parquet(OUT_PARQUET)


def test_exact_shape_and_columns(frame):
    assert tuple(frame.columns) == COLUMNS
    assert len(frame) == TOTAL_ROWS


def test_timeline_is_the_frozen_2024_half_hour_grid(frame):
    stamps = frame["timestamp"]
    assert isinstance(stamps.dtype, pd.DatetimeTZDtype)
    assert str(stamps.dt.tz) == TIMEZONE
    assert stamps.is_monotonic_increasing
    assert not stamps.duplicated().any()
    deltas = stamps.diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta(minutes=30)
    assert stamps.iloc[0] == pd.Timestamp("2024-01-01T00:00:00+08:00")
    assert stamps.iloc[-1] == pd.Timestamp("2024-12-31T23:30:00+08:00")


def test_all_columns_are_finite(frame):
    for column in COLUMNS[1:]:
        assert np.isfinite(frame[column].to_numpy(dtype=float)).all(), column


def test_upstream_canonical_is_untouched():
    assert _sha256(CANONICAL_PARQUET) == (
        "dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd"
    )
    assert _sha256(PUBLIC_SOURCE_V1) == PUBLIC_SOURCE_V1_SHA256


# --- 3. PV：损耗必须**恰好应用一次** ------------------------------------------

@pytest.fixture(scope="module")
def pv_inputs():
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    return (
        pd.DatetimeIndex(canonical["timestamp"]),
        canonical["ghi_w_per_m2"].to_numpy(dtype=float),
        canonical["temperature_deg_c"].to_numpy(dtype=float),
        canonical["wind_speed_10m_mps"].to_numpy(dtype=float),
    )


def test_losses_are_applied_and_strictly_ordered(pv_inputs):
    """§二.1：0% > 14% > 99%（在相同正 GHI 输入下）。"""
    module = drivers()
    times, ghi, temp, wind = pv_inputs
    base_dc = module.pv_dc_before_losses(times, ghi, temp, wind)
    day = ghi > 0.0
    at_0 = base_dc * module.pv_loss_multiplier(0.0)
    at_14 = base_dc * module.pv_loss_multiplier(14.0)
    at_99 = base_dc * module.pv_loss_multiplier(99.0)
    assert (at_0[day] > at_14[day]).all()
    assert (at_14[day] > at_99[day]).all()
    assert module.pv_loss_multiplier(0.0) == 1.0
    assert module.pv_loss_multiplier(14.0) == pytest.approx(0.86)


def test_losses_are_applied_exactly_once(pv_inputs):
    """§二.1：`local_pv_kw` 必须等于「未损耗 DC → 一次损耗 → 逆变器」。"""
    module = drivers()
    times, ghi, temp, wind = pv_inputs
    base_dc = module.pv_dc_before_losses(times, ghi, temp, wind)
    once = module.pv_ac_from_dc(base_dc, ghi_w_per_m2=ghi)
    twice = module.pv_ac_from_dc(base_dc * module.pv_loss_multiplier(14.0),
                                 ghi_w_per_m2=ghi)
    assert np.array_equal(module.local_pv_kw(times, ghi, temp, wind), once)
    assert not np.array_equal(once, twice)          # 双重套用会被发现


@pytest.mark.parametrize("mutation", ["drop", "tamper_zero", "tamper_99", "extra"])
def test_pv_param_block_must_be_the_frozen_block(pv_inputs, mutation):
    """§二.2：PV 参数块缺失或篡改（含 `losses_pct`）必须拒绝。"""
    module = drivers()
    times, ghi, temp, wind = pv_inputs
    params = dict(module.PV_PARAMS)
    if mutation == "drop":
        params.pop("losses_pct")
    elif mutation == "tamper_zero":
        params["losses_pct"] = 0.0
    elif mutation == "tamper_99":
        params["losses_pct"] = 99.0
    else:
        params["shadow_kw"] = 1.0
    with pytest.raises(module.ExogenousDriverError):
        module.local_pv_kw(times, ghi, temp, wind, params=params)


def test_pv_is_non_negative_and_within_the_ac_limit(frame):
    module = drivers()
    cap = module.pv_ac_limit_kw(PV_CAPACITY_KW, DC_AC_RATIO)
    assert cap == pytest.approx(PV_CAPACITY_KW / DC_AC_RATIO)
    pv = frame["local_pv_kw"].to_numpy(dtype=float)
    assert (pv >= 0.0).all()
    assert (pv <= cap + 1e-9).all()


def test_pv_is_zero_at_night(frame):
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    ghi = canonical["ghi_w_per_m2"].to_numpy(dtype=float)
    pv = frame["local_pv_kw"].to_numpy(dtype=float)
    night = ghi <= 0.0
    assert night.sum() > 0
    assert np.allclose(pv[night], 0.0)


def test_pv_is_positive_at_some_daytime_steps(frame):
    assert (frame["local_pv_kw"].to_numpy(dtype=float) > 0.0).sum() > 1000


def test_pv_chain_is_deterministic_and_matches_the_frozen_parameters(frame, pv_inputs):
    module = drivers()
    times, ghi, temp, wind = pv_inputs
    recomputed = module.local_pv_kw(times, ghi, temp, wind)
    assert np.array_equal(recomputed, frame["local_pv_kw"].to_numpy(dtype=float))


def test_pv_does_not_use_the_bell_curve_or_the_2026_profile():
    source = (REPO_ROOT / "scenario/exogenous_drivers.py").read_text(encoding="utf-8")
    assert "_build_default_pv_curve" not in source
    assert "use_default_pv_curve" not in source
    # 不得引用 2026 的 EMA Solar Generation Profile（B5 批准日期里的 2026 不算）
    assert "solar_generation_profile" not in source.lower()
    assert "Solar Generation Profile" not in source


# --- 4. 风电 ------------------------------------------------------------------

def test_wind_is_non_negative_and_at_most_the_rated_capacity(frame):
    wind = frame["wind_generation_kw"].to_numpy(dtype=float)
    assert (wind >= 0.0).all()
    assert (wind <= RATED_CAPACITY_KW + 1e-9).all()


def test_hub_wind_speed_follows_the_shear_law():
    module = drivers()
    v10 = np.array([0.0, 2.0, 5.0, 10.0])
    assert np.allclose(module.hub_wind_speed(v10), v10 * (60.0 / 10.0) ** (1.0 / 7.0))


def test_power_curve_below_cut_in_is_zero():
    module = drivers()
    curve = module.load_wind_power_curve()
    assert np.allclose(
        module.power_from_curve(np.array([0.0, 0.5, 0.99]), curve=curve), 0.0)


def test_power_curve_beyond_its_domain_is_zero():
    module = drivers()
    curve = module.load_wind_power_curve()
    assert np.allclose(
        module.power_from_curve(np.array([25.5, 30.0, 100.0]), curve=curve), 0.0)


def test_power_curve_interpolation_is_deterministic_and_monotone():
    module = drivers()
    curve = module.load_wind_power_curve()
    speeds = np.linspace(1.0, 25.0, 97)
    first = module.power_from_curve(speeds, curve=curve)
    assert np.array_equal(first, module.power_from_curve(speeds, curve=curve))
    assert np.all(np.diff(first) >= -1e-9)
    assert first.max() <= RATED_CAPACITY_KW + 1e-9


def test_wind_is_not_a_rename_of_wind_speed(frame):
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    v10 = canonical["wind_speed_10m_mps"].to_numpy(dtype=float)
    wind = frame["wind_generation_kw"].to_numpy(dtype=float)
    assert not np.allclose(v10, wind)
    assert (wind == 0.0).sum() > 0


# --- 5. 碳强度：分类与常量都必须准确 -----------------------------------------

def test_carbon_is_exactly_the_approved_constant(frame):
    carbon = frame["carbon_intensity"].to_numpy(dtype=float)
    assert len(carbon) == TOTAL_ROWS
    assert np.all(carbon == CARBON_KG_PER_KWH)


def test_carbon_is_not_described_as_modeled_scenario():
    """§二.6：carbon 必须维持 `human_approved_external_low_resolution`。"""
    payload = json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))
    entry = payload["columns"]["carbon_intensity"]
    assert entry["classification"] == "human_approved_external_low_resolution"
    assert entry["classification"] != "modeled_scenario"
    assert entry["resolution"] == "annual_constant"
    assert entry["human_decision"]["decision_id"] == "B1"
    source = (REPO_ROOT / "scenario/exogenous_drivers.py").read_text(encoding="utf-8")
    for forbidden in ("_create_carbon_factor_curve", "carbon_factor_ref", "0.45", "0.80"):
        assert forbidden not in source, forbidden


# --- 6. arrival：**date-free** 48-slot ----------------------------------------

def _manifest() -> dict:
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


def test_no_date_mapping_tokens_anywhere():
    """§二.3：源码与产物中都不得出现日期/星期映射的痕迹。"""
    for path in (REPO_ROOT / "scenario/exogenous_drivers.py",
                 REPO_ROOT / "scripts/materialize_singapore_exogenous.py",
                 OUT_MANIFEST, SRC_MANIFEST):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_DATE_TOKENS:
            assert token not in text, f"{path.name} 含 {token!r}"


def test_arrival_source_does_not_claim_a_weekday_dependency():
    """§五.3：源码不得声称 arrival 依赖 weekday（星期）；只依赖 hour/minute。"""
    source = (REPO_ROOT / "scenario/exogenous_drivers.py").read_text(encoding="utf-8")
    assert "星期与时刻" not in source


def test_arrival_template_is_48_slots_and_normalised():
    module = drivers()
    template = module.arrival_rate_template_from_cache()
    assert template.shape == (ARRIVAL_SLOTS,)
    assert np.isfinite(template).all() and (template >= 0).all()
    assert template.mean() == pytest.approx(1.0, rel=1e-12)
    assert _manifest()["columns"]["arrival"]["template_slots"] == ARRIVAL_SLOTS
    assert _manifest()["columns"]["arrival"]["uses_archive_dates"] is False


def test_slot_aggregation_is_independent_of_order_and_fabricated_dates():
    """§二.4：文件顺序与虚构日期都不得改变 date-free slot 聚合。"""
    module = drivers()
    minutes = np.arange(1440)
    values = np.abs(np.sin(minutes / 7.0)) * 100.0
    ordered = pd.Series(values, index=pd.Index(minutes, name="minute"))
    reversed_series = pd.Series(values[::-1], index=pd.Index(minutes[::-1], name="minute"))
    shuffled = reversed_series.sort_index(kind="stable")
    baseline = module.arrival_slot_counts(ordered)
    assert np.array_equal(baseline, module.arrival_slot_counts(shuffled))
    assert baseline.shape == (ARRIVAL_SLOTS,)
    assert baseline.sum() == pytest.approx(values.sum())


def test_arrival_slot_uses_only_hour_and_minute():
    """同一 hour/minute、不同星期的 timestamp 必须落在**同一个 slot**。"""
    module = drivers()
    monday = pd.DatetimeIndex(["2024-01-01T09:30:00+08:00"])
    sunday = pd.DatetimeIndex(["2024-01-07T09:30:00+08:00"])
    assert monday.weekday[0] != sunday.weekday[0]
    assert module.arrival_template_slot(monday) == module.arrival_template_slot(sunday)
    assert int(module.arrival_template_slot(monday)[0]) == 19  # 09:30 → slot 19


def test_missing_or_tampered_b5_arrival_approval_is_rejected():
    """§二.5：缺 B5-ARRIVAL 批准记录（或字段被篡改）时不得物化。"""
    module = drivers()
    with pytest.raises(module.ExogenousDriverError):
        module.assert_arrival_approved(None)
    with pytest.raises(module.ExogenousDriverError):
        module.assert_arrival_approved(dict(module.B5_ARRIVAL_APPROVAL, seed=123))
    dropped = dict(module.B5_ARRIVAL_APPROVAL)
    dropped.pop("slot_mapping")
    with pytest.raises(module.ExogenousDriverError):
        module.assert_arrival_approved(dropped)
    with pytest.raises(module.ExogenousDriverError):
        module.assert_arrival_approved(dict(module.B5_ARRIVAL_APPROVAL, shadow=1))
    assert module.assert_arrival_approved(module.B5_ARRIVAL_APPROVAL)


def test_b5_decisions_are_registered_in_both_manifests():
    payload = _manifest()
    source = json.loads(SRC_MANIFEST.read_text(encoding="utf-8"))
    assert payload["columns"]["arrival"]["decision_id"] == "B5-ARRIVAL"
    assert payload["columns"]["local_pv_kw"]["b5_pv_approval"]["decision_id"] == "B5-PV"
    assert source["b5_arrival_approval"]["decision_id"] == "B5-ARRIVAL"
    assert source["b5_pv_approval"]["decision_id"] == "B5-PV"
    assert source["b5_arrival_approval"]["uses_archive_dates"] is False


def test_arrival_is_non_negative_integer(frame):
    arrival = frame["arrival"].to_numpy()
    assert np.issubdtype(arrival.dtype, np.integer)
    assert (arrival >= 0).all()


def test_arrival_is_reproducible_with_the_frozen_seed(frame):
    module = drivers()
    template = module.arrival_rate_template_from_cache()
    again = module.generate_arrival(
        pd.DatetimeIndex(frame["timestamp"]), template, seed=ARRIVAL_SEED)
    assert np.array_equal(again, frame["arrival"].to_numpy())


def test_arrival_changes_with_a_different_seed(frame):
    module = drivers()
    template = module.arrival_rate_template_from_cache()
    other = module.generate_arrival(
        pd.DatetimeIndex(frame["timestamp"]), template, seed=ARRIVAL_SEED + 1)
    assert not np.array_equal(other, frame["arrival"].to_numpy())


def test_arrival_is_exactly_poisson_with_the_frozen_rate(frame):
    module = drivers()
    template = module.arrival_rate_template_from_cache()
    slots = module.arrival_template_slot(pd.DatetimeIndex(frame["timestamp"]))
    rates = template[slots] * ARRIVAL_MEAN
    rng = np.random.default_rng(ARRIVAL_SEED)
    assert np.array_equal(rng.poisson(rates), frame["arrival"].to_numpy())
    assert ARRIVAL_MEAN == 1000.0
    assert rates.mean() == pytest.approx(ARRIVAL_MEAN, rel=1e-3)


def test_arrival_is_not_a_replay_of_the_2019_trace(frame):
    arrival = frame["arrival"].to_numpy()
    assert len(np.unique(arrival)) > 50
    assert arrival.mean() == pytest.approx(ARRIVAL_MEAN, rel=0.05)


# --- 7. 失败关闭与不可覆盖 ----------------------------------------------------

def _mutated_chain(tmp_path: pathlib.Path, *, sync_hashes: bool = True) -> pathlib.Path:
    """复制上游资产并篡改 validation/test 段（train 与 trace 不变）。"""
    root = tmp_path / "root"
    (root / "data/processed/singapore_2024").mkdir(parents=True, exist_ok=True)
    (root / "data/manifest").mkdir(parents=True, exist_ok=True)
    for name in ("singapore_2024_half_hour.json", "singapore_2024_splits.json"):
        shutil.copy(REPO_ROOT / "data/manifest" / name, root / "data/manifest" / name)
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    canonical.loc[canonical.index >= 10224, "temperature_deg_c"] += 5.0
    parquet = root / "data/processed/singapore_2024/half_hour.parquet"
    canonical.to_parquet(parquet, index=False)
    if sync_hashes:
        payload = json.loads(
            (root / "data/manifest/singapore_2024_half_hour.json").read_text())
        payload["output_parquet_sha256"] = _sha256(parquet)
        (root / "data/manifest/singapore_2024_half_hour.json").write_text(
            json.dumps(payload))
        splits = json.loads(
            (root / "data/manifest/singapore_2024_splits.json").read_text())
        splits["canonical_parquet_sha256"] = _sha256(parquet)
        splits["canonical_manifest_sha256"] = _sha256(
            root / "data/manifest/singapore_2024_half_hour.json")
        splits["canonical_parquet_path"] = "<external>/half_hour.parquet"
        splits["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
        splits["train_only_statistics_source"]["canonical_parquet_sha256"] = _sha256(
            parquet)
        (root / "data/manifest/singapore_2024_splits.json").write_text(
            json.dumps(splits))
    return root


def test_source_hash_mismatch_fails_closed(tmp_path):
    module = drivers()
    root = _mutated_chain(tmp_path, sync_hashes=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.load_frozen_inputs(
            canonical_parquet_path=root / "data/processed/singapore_2024/half_hour.parquet",
            canonical_manifest_path=root / "data/manifest/singapore_2024_half_hour.json",
            split_manifest_path=root / "data/manifest/singapore_2024_splits.json",
        )


def test_validation_and_test_truth_mutation_does_not_change_arrival(tmp_path):
    """§二.8：v2 的 arrival **不读取** validation/test 的任何数值真值。"""
    module = drivers()
    baseline_inputs = module.load_frozen_inputs(
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
    )
    mutated = _mutated_chain(tmp_path)
    other_inputs = module.load_frozen_inputs(
        canonical_parquet_path=mutated / "data/processed/singapore_2024/half_hour.parquet",
        canonical_manifest_path=mutated / "data/manifest/singapore_2024_half_hour.json",
        split_manifest_path=mutated / "data/manifest/singapore_2024_splits.json",
    )
    template = module.arrival_rate_template_from_cache()
    baseline = module.build_drivers(baseline_inputs, template=template)
    after = module.build_drivers(other_inputs, template=template)
    assert not np.array_equal(baseline["local_pv_kw"], after["local_pv_kw"])
    assert np.array_equal(baseline["arrival"], after["arrival"])


def test_arrival_template_does_not_read_2024_truth(monkeypatch):
    module = drivers()
    baseline = module.arrival_rate_template_from_cache()

    def forbidden(*args, **kwargs):
        raise AssertionError("arrival template 不得读取 2024 truth")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    assert np.array_equal(baseline, module.arrival_rate_template_from_cache())


def test_existing_different_output_is_not_overwritten(tmp_path):
    module = materializer()
    target = tmp_path / "out.json"
    target.write_text('{"a": 1}', encoding="utf-8")
    before = (_sha256(target), target.stat().st_mtime_ns)
    with pytest.raises((ValueError, RuntimeError)):
        module.write_json_atomic({"a": 2}, target)
    assert (_sha256(target), target.stat().st_mtime_ns) == before


def test_first_freeze_failure_leaves_no_partial_artifacts(tmp_path, monkeypatch):
    module = materializer()
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    monkeypatch.setattr(module, "_atomic_write_bytes",
                        lambda path, body: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        module.materialize_exogenous_drivers(out_parquet=out_dir / "x.parquet",
                                             out_manifest=out_dir / "x.json")
    assert sorted(p.name for p in out_dir.iterdir()) == []


def test_rematerialization_is_idempotent(tmp_path, monkeypatch):
    module = materializer()
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    monkeypatch.setattr(module, "SOURCE_MANIFEST", tmp_path / "sources.json")
    out_parquet = out_dir / "exogenous_drivers_v2.parquet"
    out_manifest = tmp_path / "exogenous_v2.json"
    first = module.materialize_exogenous_drivers(
        out_parquet=out_parquet, out_manifest=out_manifest)
    before = {
        "parquet": (_sha256(out_parquet), out_parquet.stat().st_mtime_ns),
        "manifest": (_sha256(out_manifest), out_manifest.stat().st_mtime_ns),
        "sources": (_sha256(tmp_path / "sources.json"),
                    (tmp_path / "sources.json").stat().st_mtime_ns),
    }
    frozen_at = json.loads(out_manifest.read_text())["frozen_at_utc"]
    module.materialize_exogenous_drivers(
        out_parquet=out_parquet, out_manifest=out_manifest)
    after = {
        "parquet": (_sha256(out_parquet), out_parquet.stat().st_mtime_ns),
        "manifest": (_sha256(out_manifest), out_manifest.stat().st_mtime_ns),
        "sources": (_sha256(tmp_path / "sources.json"),
                    (tmp_path / "sources.json").stat().st_mtime_ns),
    }
    assert before == after
    assert json.loads(out_manifest.read_text())["frozen_at_utc"] == frozen_at
    assert first["rows"] == TOTAL_ROWS


# --- 8. manifest 与 readiness -------------------------------------------------

def test_output_manifest_records_the_required_fields():
    payload = _manifest()
    for key in ("schema", "contract_version", "materializer_revision",
                "canonical_parquet_path", "canonical_parquet_sha256",
                "split_manifest_path", "split_manifest_sha256",
                "public_source_manifest_path", "public_source_manifest_sha256",
                "materialization_sources_path", "materialization_sources_sha256",
                "pyproject_sha256", "uv_lock_sha256",
                "columns", "azure", "output", "supersedes", "readiness"):
        assert key in payload, key


def test_output_manifest_hashes_match_reality():
    payload = _manifest()
    assert _sha256(PUBLIC_SOURCE_V1) == payload["public_source_manifest_sha256"]
    assert _sha256(SRC_MANIFEST) == payload["materialization_sources_sha256"]
    assert _sha256(REPO_ROOT / payload["output"]["path"]) == payload["output"]["sha256"]
    assert payload["output"]["rows"] == TOTAL_ROWS
    assert payload["output"]["timezone"] == TIMEZONE


def test_source_manifest_records_the_v1_reference_and_dependencies():
    payload = json.loads(SRC_MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] != "m1.3f-public-sources-v1"
    assert payload["public_source_manifest_sha256"] == PUBLIC_SOURCE_V1_SHA256
    assert payload["uv_lock_sha256"] == _sha256(REPO_ROOT / "uv.lock")
    assert payload["pyproject_sha256"] == _sha256(REPO_ROOT / "pyproject.toml")
    assert "pvlib" in json.dumps(payload).lower()
    assert payload["azure"]["content_length"] == 142968140
    assert payload["azure"]["sha256"]


def test_readiness_is_honest():
    readiness = _manifest()["readiness"]
    for key in ("local_pv_kw_ready", "wind_generation_kw_ready",
                "carbon_intensity_ready", "arrival_ready",
                "exogenous_drivers_ready"):
        assert readiness[key] is True, key
    assert readiness["formal_scenario_bundle_ready"] is False
    assert readiness["formal_training_ready"] is False


def _triad_not_ready() -> None:
    """M1.3g-c 迁移：三份正式 split manifest 现在**存在**（g-c 的产物），
    但「存在 ≠ 就绪」——它们必须显式声明尚未就绪。"""
    import json as _json

    for name in ("train.json", "validation.json", "test.json"):
        path = REPO_ROOT / "data/manifest" / name
        if not path.exists():
            continue
        readiness = _json.loads(path.read_text(encoding="utf-8"))["readiness"]
        assert readiness == {
            "formal_training_ready": False, "formal_env_ready": False,
        }, name


def test_reserved_split_names_never_claim_readiness():
    """M1.3g-c 迁移：三份 manifest 现在存在，但**不得**声称已就绪。"""
    _triad_not_ready()


def test_upstream_assets_are_untouched():
    assert _sha256(PUBLIC_SOURCE_V1) == PUBLIC_SOURCE_V1_SHA256
    assert _sha256(CANONICAL_MANIFEST) == (
        "e6484d6b050f100234a46811be30061f477bcc053b285854da5a882d2753b667"
    )
    assert _sha256(SPLIT_MANIFEST) == (
        "a096535fcdec81534f8cc05671d34d879a7e9517d06be789510dea586149af27"
    )


def test_formal_training_is_still_blocked():
    from scenario.scenario import build_scenario

    with pytest.raises((FileNotFoundError, ValueError, NotImplementedError)):
        build_scenario("train", start="2024-01-01", horizon=24, forecast_cutoff=4)


# --- 9. slow：真实输入全量复核 ------------------------------------------------

@pytest.mark.slow
def test_real_inputs_rematerialize_identically():
    module = drivers()
    inputs = module.load_frozen_inputs(
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
    )
    rebuilt = module.build_drivers(inputs)
    frame = pd.read_parquet(OUT_PARQUET)
    for column in COLUMNS[1:]:
        assert np.array_equal(np.asarray(rebuilt[column]),
                              frame[column].to_numpy()), column
