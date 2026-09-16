"""M1.3f-c 测试：四类正式外生驱动的**可复现实物化**。

改前缺陷（本文件在实现前必须为红）：`scenario/exogenous_drivers.py` 与
`scripts/materialize_singapore_exogenous.py` 都还不存在，
`data/processed/singapore_2024/exogenous_drivers.parquet` 与
`data/manifest/singapore_2024_exogenous.json` 都还没有。

**测试聚焦实际训练风险**（按卡片 §H.8）：时间轴/单位/范围、PV 夜间与 AC 上限、
风电额定上限、carbon 严格常量、arrival 的**因果性/可复现性/非重放**、
source hash fail closed、产物不可覆盖、正式训练不得提前放行。
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

OUT_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers.parquet"
OUT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous.json"
SRC_MANIFEST = REPO_ROOT / "data/manifest/m13f_materialization_sources.json"
V1_MANIFEST = REPO_ROOT / "data/manifest/m13f_public_sources.json"
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
# M1.3f-b-R2 冻结的 v1 source manifest（**必须字节不变**）
V1_MANIFEST_SHA256 = "f5a5f506c579da1b9c258d38103ca4549ac483ebd2d9f501b8d686ae9cbc3faa"


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
                 "ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR", "hub_wind_speed",
                 "power_from_curve", "load_wind_power_curve", "pv_ac_limit_kw",
                 "local_pv_kw", "wind_generation_kw", "carbon_intensity",
                 "arrival_rate_template", "generate_arrival"):
        assert hasattr(module, name), name
    assert hasattr(materializer(), "materialize_exogenous_drivers")
    assert hasattr(materializer(), "build_manifest")


def test_output_artifacts_exist():
    assert OUT_PARQUET.exists(), "缺少 exogenous_drivers.parquet"
    assert OUT_MANIFEST.exists(), "缺少 singapore_2024_exogenous.json"
    assert SRC_MANIFEST.exists(), "缺少 m13f_materialization_sources.json"


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
        values = frame[column].to_numpy(dtype=float)
        assert np.isfinite(values).all(), column


def test_upstream_canonical_is_untouched():
    assert _sha256(CANONICAL_PARQUET) == (
        "dec76ea2e947f63d086767f443edbef5725cdfed7458a047ebba0ec7d70fddcd"
    )
    assert _sha256(V1_MANIFEST) == V1_MANIFEST_SHA256


# --- 3. PV --------------------------------------------------------------------

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
    pv = frame["local_pv_kw"].to_numpy(dtype=float)
    assert (pv > 0.0).sum() > 1000


def test_pv_chain_is_deterministic_and_matches_the_frozen_parameters(frame):
    """同一输入与同一组冻结参数必须给出**逐位相同**的结果。"""
    module = drivers()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    times = pd.DatetimeIndex(canonical["timestamp"])
    recomputed = module.local_pv_kw(
        times,
        canonical["ghi_w_per_m2"].to_numpy(dtype=float),
        canonical["temperature_deg_c"].to_numpy(dtype=float),
        canonical["wind_speed_10m_mps"].to_numpy(dtype=float),
    )
    assert np.array_equal(recomputed, frame["local_pv_kw"].to_numpy(dtype=float))


def test_pv_does_not_use_the_bell_curve_or_2026_profile():
    """`local_pv_kw` 不得来自 `_build_default_pv_curve` 等默认曲线。"""
    source = (REPO_ROOT / "scenario/exogenous_drivers.py").read_text(encoding="utf-8")
    assert "_build_default_pv_curve" not in source
    assert "use_default_pv_curve" not in source
    assert "2026" not in source


# --- 4. 风电 ------------------------------------------------------------------

def test_wind_is_non_negative_and_at_most_the_rated_capacity(frame):
    wind = frame["wind_generation_kw"].to_numpy(dtype=float)
    assert (wind >= 0.0).all()
    assert (wind <= RATED_CAPACITY_KW + 1e-9).all()


def test_hub_wind_speed_follows_the_shear_law():
    module = drivers()
    v10 = np.array([0.0, 2.0, 5.0, 10.0])
    hub = module.hub_wind_speed(v10)
    assert np.allclose(hub, v10 * (60.0 / 10.0) ** (1.0 / 7.0))


def test_power_curve_below_cut_in_is_zero():
    module = drivers()
    curve = module.load_wind_power_curve()
    below = module.power_from_curve(np.array([0.0, 0.5, 0.99]), curve=curve)
    assert np.allclose(below, 0.0)


def test_power_curve_beyond_its_domain_is_zero():
    """超出冻结曲线定义域 → 保守停机（**不得**外推制造发电）。"""
    module = drivers()
    curve = module.load_wind_power_curve()
    beyond = module.power_from_curve(np.array([25.5, 30.0, 100.0]), curve=curve)
    assert np.allclose(beyond, 0.0)


def test_power_curve_interpolation_is_deterministic_and_monotone():
    module = drivers()
    curve = module.load_wind_power_curve()
    speeds = np.linspace(1.0, 25.0, 97)
    first = module.power_from_curve(speeds, curve=curve)
    second = module.power_from_curve(speeds, curve=curve)
    assert np.array_equal(first, second)
    assert np.all(np.diff(first) >= -1e-9)          # 单调不减
    assert first.max() <= RATED_CAPACITY_KW + 1e-9  # 截到额定容量


def test_wind_is_not_a_rename_of_wind_speed(frame):
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    v10 = canonical["wind_speed_10m_mps"].to_numpy(dtype=float)
    wind = frame["wind_generation_kw"].to_numpy(dtype=float)
    assert not np.allclose(v10, wind)
    assert (wind == 0.0).sum() > 0                  # 切入以下为 0


# --- 5. 碳强度 ----------------------------------------------------------------

def test_carbon_is_exactly_the_approved_constant(frame):
    carbon = frame["carbon_intensity"].to_numpy(dtype=float)
    assert len(carbon) == TOTAL_ROWS
    assert np.all(carbon == CARBON_KG_PER_KWH)


def test_carbon_does_not_come_from_the_env_default_curve(frame):
    source = (REPO_ROOT / "scenario/exogenous_drivers.py").read_text(encoding="utf-8")
    for forbidden in ("_create_carbon_factor_curve", "carbon_factor_ref", "0.45", "0.80"):
        assert forbidden not in source, forbidden


# --- 6. arrival ---------------------------------------------------------------

def test_arrival_is_non_negative_integer(frame):
    arrival = frame["arrival"].to_numpy()
    assert np.issubdtype(arrival.dtype, np.integer)
    assert (arrival >= 0).all()


def test_arrival_is_reproducible_with_the_frozen_seed(frame):
    module = drivers()
    template = np.asarray(
        json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))["arrival"]["rate_template"]
    )
    again = module.generate_arrival(
        pd.DatetimeIndex(frame["timestamp"]), template, seed=ARRIVAL_SEED)
    assert np.array_equal(again, frame["arrival"].to_numpy())


def test_arrival_changes_with_a_different_seed(frame):
    module = drivers()
    template = np.asarray(
        json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))["arrival"]["rate_template"]
    )
    other = module.generate_arrival(
        pd.DatetimeIndex(frame["timestamp"]), template, seed=ARRIVAL_SEED + 1)
    assert not np.array_equal(other, frame["arrival"].to_numpy())


def test_arrival_template_is_normalised_and_shaped(frame):
    module = drivers()
    template = module.arrival_rate_template_from_manifest()
    assert template.shape == (7, 48)
    assert np.isfinite(template).all()
    assert (template >= 0).all()
    assert template.mean() == pytest.approx(1.0, rel=1e-9)


def test_arrival_is_exactly_poisson_with_the_frozen_rate(frame):
    module = drivers()
    template = module.arrival_rate_template_from_manifest()
    stamps = pd.DatetimeIndex(frame["timestamp"])
    rates = np.array([
        template[stamp.weekday(), stamp.hour * 2 + stamp.minute // 30] * ARRIVAL_MEAN
        for stamp in stamps
    ])
    rng = np.random.default_rng(ARRIVAL_SEED)
    expected = rng.poisson(rates)
    assert np.array_equal(expected, frame["arrival"].to_numpy())
    assert rates.mean() == pytest.approx(ARRIVAL_MEAN, rel=1e-9)


def test_arrival_is_not_a_replay_of_the_2019_trace(frame):
    """2019 trace 只校准**分布形状**，不得被直接重放成 2024 到达。"""
    arrival = frame["arrival"].to_numpy()
    assert len(np.unique(arrival)) > 50          # 不是把少数 trace 值照抄
    assert arrival.mean() == pytest.approx(ARRIVAL_MEAN, rel=0.05)


def test_arrival_template_ignores_validation_and_test_truth(tmp_path):
    """修改 validation/test truth **不得**改变 arrival rate template。"""
    module = drivers()
    baseline = module.arrival_rate_template_from_cache()
    mutated = _mutated_chain(tmp_path)
    other = module.arrival_rate_template_from_cache(root=mutated)
    assert np.array_equal(baseline, other)


def _mutated_chain(tmp_path: pathlib.Path) -> pathlib.Path:
    """复制上游资产并**篡改 validation/test 段**（train 段与 trace 不变）。"""
    root = tmp_path / "root"
    (root / "data/processed/singapore_2024").mkdir(parents=True, exist_ok=True)
    (root / "data/manifest").mkdir(parents=True, exist_ok=True)
    for name in ("singapore_2024_half_hour.json", "singapore_2024_splits.json"):
        shutil.copy(REPO_ROOT / "data/manifest" / name, root / "data/manifest" / name)
    frame = pd.read_parquet(CANONICAL_PARQUET)
    frame.loc[frame.index >= 10224, "temperature_deg_c"] += 5.0
    parquet = root / "data/processed/singapore_2024/half_hour.parquet"
    frame.to_parquet(parquet, index=False)
    payload = json.loads((root / "data/manifest/singapore_2024_half_hour.json").read_text())
    payload["output_parquet_sha256"] = _sha256(parquet)
    (root / "data/manifest/singapore_2024_half_hour.json").write_text(json.dumps(payload))
    splits = json.loads((root / "data/manifest/singapore_2024_splits.json").read_text())
    splits["canonical_parquet_sha256"] = _sha256(parquet)
    splits["canonical_manifest_sha256"] = _sha256(
        root / "data/manifest/singapore_2024_half_hour.json")
    (root / "data/manifest/singapore_2024_splits.json").write_text(json.dumps(splits))
    return root


# --- 7. 失败关闭与不可覆盖 ----------------------------------------------------

def test_source_hash_mismatch_fails_closed(tmp_path):
    module = drivers()
    root = _mutated_chain(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.load_frozen_inputs(
            canonical_parquet_path=root / "data/processed/singapore_2024/half_hour.parquet",
            canonical_manifest_path=root / "data/manifest/singapore_2024_half_hour.json",
            split_manifest_path=root / "data/manifest/singapore_2024_splits.json",
        )


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


# --- 8. manifest 与 readiness -------------------------------------------------

def _manifest() -> dict:
    return json.loads(OUT_MANIFEST.read_text(encoding="utf-8"))


def test_output_manifest_records_the_required_fields():
    payload = _manifest()
    for key in ("schema", "contract_version", "materializer_revision",
                "canonical_parquet_path", "canonical_parquet_sha256",
                "split_manifest_path", "split_manifest_sha256",
                "public_source_manifest_path", "public_source_manifest_sha256",
                "materialization_sources_path", "materialization_sources_sha256",
                "pyproject_sha256", "uv_lock_sha256",
                "columns", "azure", "output", "arrival", "readiness"):
        assert key in payload, key


def test_output_manifest_hashes_match_reality():
    payload = _manifest()
    assert _sha256(V1_MANIFEST) == payload["public_source_manifest_sha256"]
    assert _sha256(SRC_MANIFEST) == payload["materialization_sources_sha256"]
    assert _sha256(REPO_ROOT / payload["output"]["path"]) == payload["output"]["sha256"]
    assert payload["output"]["rows"] == TOTAL_ROWS
    assert payload["output"]["timezone"] == TIMEZONE


def test_source_manifest_records_the_v1_reference_and_dependencies():
    payload = json.loads(SRC_MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema"] != "m1.3f-public-sources-v1"
    assert payload["public_source_manifest_sha256"] == V1_MANIFEST_SHA256
    assert payload["uv_lock_sha256"] == _sha256(REPO_ROOT / "uv.lock")
    assert payload["pyproject_sha256"] == _sha256(REPO_ROOT / "pyproject.toml")
    assert "pvlib" in json.dumps(payload).lower()
    assert payload["azure"]["content_length"] == 142968140
    assert payload["azure"]["sha256"]


def test_readiness_is_honest():
    payload = _manifest()
    readiness = payload["readiness"]
    for key in ("local_pv_kw_ready", "wind_generation_kw_ready",
                "carbon_intensity_ready", "arrival_ready",
                "exogenous_drivers_ready"):
        assert readiness[key] is True, key
    assert readiness["formal_scenario_bundle_ready"] is False
    assert readiness["formal_training_ready"] is False


def test_no_reserved_split_names_are_created():
    for name in ("train.json", "validation.json", "test.json"):
        assert not (REPO_ROOT / "data/manifest" / name).exists(), name


def test_upstream_assets_are_untouched():
    assert _sha256(V1_MANIFEST) == V1_MANIFEST_SHA256
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
    """用真实上游资产重新计算四列，必须与已物化的 parquet **逐位一致**。"""
    module = drivers()
    inputs = module.load_frozen_inputs(
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
    )
    rebuilt = module.build_drivers(inputs)
    frame = pd.read_parquet(OUT_PARQUET)
    for column in COLUMNS[1:]:
        assert np.array_equal(
            np.asarray(rebuilt[column]), frame[column].to_numpy()
        ), column
