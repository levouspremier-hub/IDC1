"""M1.3g-b：formal causal `ScenarioBundle` 构造内核。

**改前缺陷（本文件在实现前必须为红）**：

- `scenario/formal_scenario.py` **不存在**——正式路径在
  `scenario/scenario.py` 直接抛 `NotImplementedError`；
- 没有任何把五类 causal driver forecast 变换成 PV / 风电 forecast 的正式内核；
- carbon 的 `human_approved_external_low_resolution` 与 arrival 的**期望值**口径
  都还没有被任何构造器使用。

**本卡不接线 env / train、不创建三个正式 split manifest、不冻结 refs。**
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
FORMAL_MODULE = "scenario.formal_scenario"
DRIVERS_MODULE = "scenario.exogenous_drivers"
FORECAST_MODULE = "scenario.forecast"

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
POLICY_V2 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v2.json"
ENDOGENOUS_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v2.json"
ENDOGENOUS_SOURCE = REPO_ROOT / "data/manifest/m13f_materialization_sources_v3.json"
ENDOGENOUS_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers_v2.parquet"

TOTAL_ROWS = 17568
TRAIN_ROWS = 10224
PERIOD_STEPS = 48
CARBON_VALUE = 0.402
ARRIVAL_MEAN = 1000.0
PV_AC_LIMIT_KW = 500.0 / 1.2
WIND_RATED_KW = 800.0
TIMEZONE = "Asia/Singapore"

BUNDLE_FORECAST_FIELDS = (
    "price_forecast", "load_forecast", "pv_forecast", "wind_forecast",
    "temperature_forecast", "carbon_forecast", "arrival_forecast",
)
EXPECTED_KINDS = {
    "price_forecast": "seasonal_naive",
    "load_forecast": "seasonal_naive",
    "temperature_forecast": "seasonal_naive",
    "pv_forecast": "modeled_scenario",
    "wind_forecast": "modeled_scenario",
    "arrival_forecast": "modeled_scenario",
    "carbon_forecast": "human_approved_external_low_resolution",
}

_ORIGIN = TRAIN_ROWS // 2      # split 本地 origin（train 内部，历史充足）
_CUTOFF = 4
# 因果性用例必须把 mutation 放在 **train 之外**：M1.3d 的 split manifest 会
# 重算 train-only 统计，改动 train 行会让整条信任链自洽性检查失败（与本卡无关）。
# 因此这两个用例用 validation 内的 origin，mutation 落在 validation 段。
_VALIDATION_ORIGIN = 100       # split 本地；全局 = TRAIN_ROWS + 100


def formal():
    return importlib.import_module(FORMAL_MODULE)


def drivers():
    return importlib.import_module(DRIVERS_MODULE)


def forecast():
    return importlib.import_module(FORECAST_MODULE)


def _sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _assets_present() -> bool:
    return all(
        p.exists()
        for p in (CANONICAL_PARQUET, CANONICAL_MANIFEST, SPLIT_MANIFEST, POLICY_V2,
                  ENDOGENOUS_MANIFEST, ENDOGENOUS_SOURCE, ENDOGENOUS_PARQUET)
    )


needs_assets = pytest.mark.skipif(
    not _assets_present(), reason="真实冻结上游资产不在本机"
)


def build(**over):
    """用**真实**冻结资产构造 formal bundle（默认 origin/cutoff 固定）。"""
    kwargs = dict(
        split="train",
        origin=_ORIGIN,
        forecast_cutoff=_CUTOFF,
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
        policy_manifest_path=POLICY_V2,
        exogenous_manifest_path=ENDOGENOUS_MANIFEST,
        exogenous_source_manifest_path=ENDOGENOUS_SOURCE,
    )
    kwargs.update(over)
    return formal().build_formal_scenario(**kwargs)


# --- 1. 内核存在性 -----------------------------------------------------------

def test_formal_kernel_module_exists():
    module = formal()
    for name in ("build_formal_scenario", "FORMAL_SOURCE_KINDS",
                 "EXOGENOUS_OUTPUT_SHA256"):
        assert hasattr(module, name), name


# --- 2. 形状、时间轴与 source_kind -------------------------------------------

@needs_assets
def test_bundle_is_a_formal_contract_v9_bundle():
    from contracts import CONTRACT_VERSION_ID

    bundle = build()
    assert bundle.mode == "formal"
    assert bundle.schema_version == CONTRACT_VERSION_ID
    assert bundle.split == "train"
    assert bundle.forecast_cutoff == _CUTOFF


@needs_assets
def test_source_kinds_are_exactly_the_frozen_mapping():
    bundle = build()
    for field, kind in EXPECTED_KINDS.items():
        entry = getattr(bundle.forecast_provenance, field)
        assert entry.series_name == field
        assert entry.source_kind == kind, field


@needs_assets
def test_all_seven_generated_at_equal_the_origin():
    bundle = build()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    origin_stamp = pd.Timestamp(canonical["timestamp"].iloc[_ORIGIN]).isoformat()
    assert bundle.generated_at == origin_stamp
    for field in BUNDLE_FORECAST_FIELDS:
        assert getattr(bundle.forecast_provenance, field).generated_at == origin_stamp


@needs_assets
def test_every_series_has_the_cutoff_length():
    bundle = build()
    for field in BUNDLE_FORECAST_FIELDS:
        assert len(getattr(bundle, field)) == _CUTOFF, field


@needs_assets
def test_bundle_passes_the_training_purpose_gate():
    from contracts.validators import validate_forecast_purpose

    validate_forecast_purpose(build(), purpose="training")


# --- 3. carbon 与 arrival 的口径 ---------------------------------------------

@needs_assets
def test_carbon_is_the_approved_constant_and_not_modeled_scenario():
    bundle = build()
    assert np.allclose(np.asarray(bundle.carbon_forecast), CARBON_VALUE)
    entry = bundle.forecast_provenance.carbon_forecast
    assert entry.source_kind == "human_approved_external_low_resolution"
    assert entry.source_kind != "modeled_scenario"


@needs_assets
def test_arrival_is_the_template_expectation_not_a_poisson_draw():
    """D3：arrival forecast 是 `λ(slot)`，**不是** Poisson 抽样值。"""
    module = formal()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    stamps = pd.DatetimeIndex(canonical["timestamp"].iloc[_ORIGIN:_ORIGIN + _CUTOFF])
    slots = drivers().arrival_template_slot(stamps)
    template = np.asarray(module.exogenous_rate_template(
        module.load_verified_exogenous(
            ENDOGENOUS_MANIFEST, ENDOGENOUS_SOURCE,
            exogenous_parquet_path=ENDOGENOUS_PARQUET,
            canonical_parquet_path=CANONICAL_PARQUET,
            canonical_manifest_path=CANONICAL_MANIFEST,
            split_manifest_path=SPLIT_MANIFEST,
        )))
    expected = template[slots] * ARRIVAL_MEAN

    bundle = build()
    assert np.allclose(np.asarray(bundle.arrival_forecast), expected)

    # 与 Poisson 实现值必须**不同**（否则说明用了抽样的实现值）
    realized = drivers().generate_arrival(stamps, template)
    assert not np.array_equal(np.asarray(bundle.arrival_forecast), realized)


@needs_assets
def test_arrival_is_deterministic_across_calls():
    first = np.asarray(build().arrival_forecast)
    second = np.asarray(build().arrival_forecast)
    assert np.array_equal(first, second)


# --- 4. PV / 风电：与同一物理函数逐位对照 ------------------------------------

@needs_assets
def test_pv_matches_the_frozen_physics_function():
    module = formal()
    bundle = build()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    stamps = pd.DatetimeIndex(canonical["timestamp"])
    ghi_f = forecast().seasonal_naive_forecast(
        canonical["ghi_w_per_m2"].to_numpy(), origin=_ORIGIN, forecast_cutoff=_CUTOFF)
    temp_f = forecast().seasonal_naive_forecast(
        canonical["temperature_deg_c"].to_numpy(), origin=_ORIGIN,
        forecast_cutoff=_CUTOFF)
    v10_f = forecast().seasonal_naive_forecast(
        canonical["wind_speed_10m_mps"].to_numpy(), origin=_ORIGIN,
        forecast_cutoff=_CUTOFF)
    expected = drivers().local_pv_kw(
        stamps[_ORIGIN:_ORIGIN + _CUTOFF], ghi_f, temp_f, v10_f)
    assert np.array_equal(np.asarray(bundle.pv_forecast), expected)
    assert np.array_equal(np.asarray(bundle.pv_forecast), module.pv_forecast(
        stamps[_ORIGIN:_ORIGIN + _CUTOFF], ghi_f, temp_f, v10_f))


@needs_assets
def test_wind_matches_the_frozen_physics_function():
    module = formal()
    bundle = build()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    v10_f = forecast().seasonal_naive_forecast(
        canonical["wind_speed_10m_mps"].to_numpy(), origin=_ORIGIN,
        forecast_cutoff=_CUTOFF)
    expected = drivers().wind_generation_kw(v10_f)
    assert np.array_equal(np.asarray(bundle.wind_forecast), expected)
    assert np.array_equal(
        np.asarray(bundle.wind_forecast), module.wind_forecast(v10_f))


@needs_assets
def test_pv_and_wind_respect_the_same_caps_as_truth():
    """全年前 4 个窗口之外：直接对整条序列验证上限与夜间零。"""
    module = formal()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    stamps = pd.DatetimeIndex(canonical["timestamp"])
    ghi = canonical["ghi_w_per_m2"].to_numpy()
    temp = canonical["temperature_deg_c"].to_numpy()
    v10 = canonical["wind_speed_10m_mps"].to_numpy()
    pv = np.asarray(module.pv_forecast(stamps, ghi, temp, v10))
    wind = np.asarray(module.wind_forecast(v10))

    assert (pv >= 0.0).all() and (pv <= PV_AC_LIMIT_KW + 1e-9).all()
    night = ghi <= 0.0
    assert night.sum() > 0
    assert np.allclose(pv[night], 0.0)
    assert (wind >= 0.0).all() and (wind <= WIND_RATED_KW + 1e-9).all()


# --- 5. 因果性：未来真值不得进入 forecast ------------------------------------

def _mutated_chain(tmp_path, *, mutate_from: int, mutate_to: int):
    """复制冻结链并篡改 canonical 的 `[mutate_from, mutate_to)` 行，同步所有 hash。

    mutation **必须**落在 train 之外（validation/test），否则 M1.3d 的
    train-only 统计重算会先失败。
    """
    """复制冻结链并篡改 canonical 的 `[mutate_from, mutate_to)` 行，同步两个 hash。"""
    root = tmp_path / "chain"
    (root / "data/processed/singapore_2024").mkdir(parents=True)
    (root / "data/manifest").mkdir(parents=True)
    for name in ("singapore_2024_half_hour.json", "singapore_2024_splits.json"):
        shutil.copy(REPO_ROOT / "data/manifest" / name, root / "data/manifest" / name)

    assert mutate_from >= TRAIN_ROWS, "mutation 必须落在 train 之外"
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    rows = canonical.index[mutate_from:mutate_to]
    canonical.loc[rows, "temperature_deg_c"] += 7.0
    canonical.loc[rows, "ghi_w_per_m2"] += 11.0
    canonical.loc[rows, "wind_speed_10m_mps"] += 0.9
    parquet = root / "data/processed/singapore_2024/half_hour.parquet"
    canonical.to_parquet(parquet, index=False)

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

    # policy-v2 也要随之自洽：它的三条声明路径必须等于**实际提供**的逻辑路径
    # （tmp 在仓库外 → `<external>/<name>`），两个 hash 必须等于 tmp 里的实测字节。
    policy = root / "data/manifest/singapore_2024_forecast_policy_v2.json"
    payload = json.loads(POLICY_V2.read_text(encoding="utf-8"))
    payload["canonical_parquet_path"] = "<external>/half_hour.parquet"
    payload["canonical_parquet_sha256"] = _sha256(parquet)
    payload["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
    payload["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    payload["split_manifest_path"] = "<external>/singapore_2024_splits.json"
    payload["split_manifest_sha256"] = _sha256(split_manifest)
    policy.write_text(json.dumps(payload))

    # R1-2：**外生链也必须交叉绑定**——v2 manifest 声明的 canonical / split
    # 的 path **与** hash 必须等于本次调用实际使用的对象。**保留原仓库的
    # exogenous manifest 不叫「同步链」**，因此这里把它一并改写进临时链。
    # （v2 驱动表本身未被 mutation，仍指向仓库里那份已冻结的 parquet。）
    exogenous = root / "data/manifest/singapore_2024_exogenous_v2.json"
    payload = json.loads(ENDOGENOUS_MANIFEST.read_text(encoding="utf-8"))
    payload["canonical_parquet_path"] = "<external>/half_hour.parquet"
    payload["canonical_parquet_sha256"] = _sha256(parquet)
    payload["canonical_manifest_path"] = "<external>/singapore_2024_half_hour.json"
    payload["canonical_manifest_sha256"] = _sha256(canonical_manifest)
    payload["split_manifest_path"] = "<external>/singapore_2024_splits.json"
    payload["split_manifest_sha256"] = _sha256(split_manifest)
    exogenous.write_text(json.dumps(payload))

    return {
        "canonical_parquet_path": parquet,
        "canonical_manifest_path": canonical_manifest,
        "split_manifest_path": split_manifest,
        "policy_manifest_path": policy,
        "exogenous_manifest_path": exogenous,
        "expected_exogenous_manifest_sha256": _sha256(exogenous),
    }


_GLOBAL_VALIDATION_ORIGIN = TRAIN_ROWS + _VALIDATION_ORIGIN


def _validation_build(**over):
    return build(split="validation", origin=_VALIDATION_ORIGIN, **over)


@needs_assets
def test_future_truth_mutation_does_not_change_any_forecast(tmp_path):
    """`[i, i+C)` 的 canonical 真值变化**不得**改变七条 forecast 中的任何一条。"""
    baseline = _validation_build()
    mutated = _validation_build(**_mutated_chain(
        tmp_path, mutate_from=_GLOBAL_VALIDATION_ORIGIN,
        mutate_to=_GLOBAL_VALIDATION_ORIGIN + _CUTOFF))
    for field in BUNDLE_FORECAST_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(baseline, field)),
            np.asarray(getattr(mutated, field)),
        ), field


@needs_assets
def test_history_mutation_changes_the_derived_forecasts(tmp_path):
    """`[i−48, i)` 的 driver 真值变化**必须**体现在 forecast 上。

    mutation 只改**天气三列**（温度 / GHI / 10 m 风速），因此断言的是
    由它们推导的三条：温度、PV、风电。（price / load / carbon / arrival
    的输入没有被改动，**不应**变化——这里不断言它们。）
    """
    baseline = _validation_build()
    mutated = _validation_build(**_mutated_chain(
        tmp_path, mutate_from=_GLOBAL_VALIDATION_ORIGIN - PERIOD_STEPS,
        mutate_to=_GLOBAL_VALIDATION_ORIGIN))
    for field in ("temperature_forecast", "pv_forecast", "wind_forecast"):
        assert not np.array_equal(
            np.asarray(getattr(baseline, field)),
            np.asarray(getattr(mutated, field)),
        ), field


# --- 6. 信任链 fail closed ---------------------------------------------------

@needs_assets
def test_v1_policy_is_rejected(tmp_path):
    """正式的 contract-v8 v1 policy 不得被 formal 内核接受。"""
    v1 = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy.json"
    with pytest.raises((ValueError, TypeError)):
        build(policy_manifest_path=v1)


@needs_assets
def test_v1_exogenous_manifest_is_rejected():
    """M1.3f-c 的 v1 外生驱动 manifest 不得作为正式链证据。"""
    v1 = REPO_ROOT / "data/manifest/singapore_2024_exogenous.json"
    with pytest.raises((ValueError, TypeError)):
        build(exogenous_manifest_path=v1)


@needs_assets
@pytest.mark.parametrize("field", ("carbon", "pv", "arrival"))
def test_missing_or_tampered_approval_is_rejected(tmp_path, field):
    """缺 / 篡改 B1（carbon）或 B5（PV / arrival）声明必须 fail closed。"""
    root = tmp_path / "manifests"
    root.mkdir()
    payload = json.loads(ENDOGENOUS_MANIFEST.read_text())
    if field == "carbon":
        payload["columns"]["carbon_intensity"].pop("human_decision")
    elif field == "pv":
        payload["columns"]["local_pv_kw"]["b5_pv_approval"]["albedo"] = 0.5
    else:
        payload["columns"]["arrival"]["b5_approval"].pop("seed")
    target = root / "exogenous_v2.json"
    target.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        build(exogenous_manifest_path=target)


@needs_assets
def test_bad_exogenous_hash_is_rejected(tmp_path):
    """exogenous manifest 被改写（hash 不符）必须拒绝。"""
    root = tmp_path / "manifests"
    root.mkdir()
    payload = json.loads(ENDOGENOUS_MANIFEST.read_text())
    payload["readiness"]["formal_training_ready"] = True
    target = root / "exogenous_v2.json"
    target.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        build(exogenous_manifest_path=target)


@needs_assets
def test_wrong_output_hash_is_rejected(tmp_path):
    """manifest 声明的 v2 output hash 与 parquet 实际字节不符必须拒绝。"""
    root = tmp_path / "manifests"
    root.mkdir()
    payload = json.loads(ENDOGENOUS_MANIFEST.read_text())
    payload["output"]["sha256"] = "0" * 64
    target = root / "exogenous_v2.json"
    target.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        build(exogenous_manifest_path=target)


# --- 7. 公开入口仍 fail closed ----------------------------------------------

def test_build_scenario_formal_path_still_requires_the_split_manifest():
    """g-c 之前：三个正式 split manifest 不存在 → 必须 FileNotFoundError。"""
    from scenario.scenario import build_scenario

    for name in ("train.json", "validation.json", "test.json"):
        assert not (REPO_ROOT / "data/manifest" / name).exists(), name
    with pytest.raises(FileNotFoundError):
        build_scenario("train", start="2024-01-01", horizon=24, forecast_cutoff=4)


def test_synthetic_path_is_untouched():
    from scenario.scenario import build_scenario

    bundle = build_scenario("train", start="s", horizon=24, forecast_cutoff=4,
                            synthetic=True)
    assert bundle.mode == "synthetic"


# --- 8. M1.3g-b-R1：代码 provenance 覆盖面 ------------------------------------

def test_formal_source_paths_cover_the_formal_implementation():
    """R1-1：revision 必须覆盖 provider 路径 + formal 内核 + 外生驱动实现。"""
    module = formal()
    provider_paths = set(forecast().FORECAST_SOURCE_PATHS)
    covered = set(module.FORMAL_SOURCE_PATHS)
    assert provider_paths <= covered, provider_paths - covered
    for rel in ("scenario/formal_scenario.py", "scenario/exogenous_drivers.py"):
        assert rel in covered, rel


@needs_assets
def test_code_revision_equals_the_formal_revision():
    """R1-1：七条 provenance 的 revision 必须是 formal 内核自身的冻结 revision。"""
    module = formal()
    expected = module.formal_code_revision()
    bundle = build()
    for field in BUNDLE_FORECAST_FIELDS:
        assert getattr(bundle.forecast_provenance, field).code_revision == expected, field


def test_formal_revision_is_stricter_than_the_provider_revision():
    """formal revision 必须比 provider-only revision 覆盖更多实现文件。"""
    module = formal()
    assert set(module.FORMAL_SOURCE_PATHS) > set(forecast().FORECAST_SOURCE_PATHS)


@needs_assets
def test_dirty_formal_source_fails_closed(monkeypatch):
    """R1-1：formal source 有未提交变更时不得物化（不能用旧提交背书新实现）。"""
    module = formal()
    monkeypatch.setattr(module, "formal_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        build()


@needs_assets
def test_dirty_check_covers_the_same_paths():
    """R1-1：dirty 检查与 revision 使用**同一**集合。"""
    module = formal()
    seen: list[tuple[str, ...]] = []
    original = module._git

    def spy(*args):
        if args and args[0] == "status":
            seen.append(tuple(args[-len(module.FORMAL_SOURCE_PATHS):]))
        return original(*args)

    module._git = spy  # type: ignore[assignment]
    try:
        module.formal_generator_is_dirty()
    finally:
        module._git = original  # type: ignore[assignment]
    assert seen, "formal_generator_is_dirty 必须查询 Git"
    assert set(seen[0]) == set(module.FORMAL_SOURCE_PATHS)


# --- 9. M1.3g-b-R1：外生链与本次调用的交叉绑定 --------------------------------

@needs_assets
def test_exogenous_manifest_binding_must_match_the_actual_objects(tmp_path):
    """R1-2：v2 manifest 声明的 canonical/split path 与 hash 必须等于**实际提供**的对象。

    这里让 manifest 声明的 canonical hash **故意**与本次调用实际使用的 parquet 不同——
    两边各自都自洽（manifest 内部一致、parquet 本身合法），仍必须拒绝。
    """
    root = tmp_path / "manifests"
    root.mkdir()
    payload = json.loads(ENDOGENOUS_MANIFEST.read_text(encoding="utf-8"))
    payload["canonical_parquet_sha256"] = "0" * 64
    target = root / "exogenous_v2.json"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        build(exogenous_manifest_path=target,
              expected_exogenous_manifest_sha256=_sha256(target))


@needs_assets
def test_exogenous_manifest_path_binding_must_match(tmp_path):
    """R1-2：声明的 **path** 也必须等于实际提供的路径。"""
    root = tmp_path / "manifests"
    root.mkdir()
    payload = json.loads(ENDOGENOUS_MANIFEST.read_text(encoding="utf-8"))
    payload["split_manifest_path"] = "<external>/somewhere_else.json"
    target = root / "exogenous_v2.json"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        build(exogenous_manifest_path=target,
              expected_exogenous_manifest_sha256=_sha256(target))


@needs_assets
def test_arrival_template_comes_only_from_the_verified_payload():
    """R1：不得通过未核验的 manifest 单独读取 arrival template。"""
    module = formal()
    verified = module.load_verified_exogenous(
        ENDOGENOUS_MANIFEST, ENDOGENOUS_SOURCE,
        exogenous_parquet_path=ENDOGENOUS_PARQUET,
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
    )
    template = np.asarray(module.exogenous_rate_template(verified))
    assert template.shape == (48,)
    bundle = build()
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    stamps = pd.DatetimeIndex(canonical["timestamp"].iloc[_ORIGIN:_ORIGIN + _CUTOFF])
    slots = drivers().arrival_template_slot(stamps)
    assert np.allclose(np.asarray(bundle.arrival_forecast) / ARRIVAL_MEAN,
                       template[slots])
    # 该入口**只**接受已核验 payload，不接受 manifest 路径
    with pytest.raises(TypeError):
        module.exogenous_rate_template(exogenous_manifest_path=ENDOGENOUS_MANIFEST)


# --- 10. M1.3g-b-R1：sources 必须含外生驱动表 ---------------------------------

@needs_assets
def test_sources_include_the_exogenous_parquet_digest():
    """R1-3：`sources` 必须含 `exogenous_drivers_parquet`，path/hash 等于经验证 output。"""
    bundle = build()
    for field in BUNDLE_FORECAST_FIELDS:
        entry = getattr(bundle.forecast_provenance, field)
        by_role = {d.role: d for d in entry.sources}
        assert "exogenous_drivers_parquet" in by_role, field
        digest = by_role["exogenous_drivers_parquet"]
        assert digest.logical_path == (
            "data/processed/singapore_2024/exogenous_drivers_v2.parquet")
        assert digest.sha256 == _sha256(ENDOGENOUS_PARQUET)


# --- 11. M1.3g-b-R1：seasonal 序列直接取 artifact ------------------------------

@needs_assets
def test_seasonal_series_are_the_artifact_series():
    """R1-4：price / load / temperature 必须**直接**取自 artifact 的 series。"""
    module = formal()
    artifact = module.build_available_forecast_only(
        split="train", origin=_ORIGIN, forecast_cutoff=_CUTOFF,
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
        policy_manifest_path=POLICY_V2,
    )
    bundle = build()
    for field, driver in (
        ("price_forecast", "price_sgd_per_kwh"),
        ("load_forecast", "system_load_mw"),
        ("temperature_forecast", "temperature_deg_c"),
    ):
        assert np.array_equal(np.asarray(getattr(bundle, field)),
                              np.asarray(artifact.series[driver])), field


@needs_assets
def test_seasonal_series_do_not_recompute_the_provider(monkeypatch):
    """R1-4：屏蔽 `seasonal_naive_forecast` 后仍能构造 → 没有第二次独立计算。"""
    module = formal()

    def boom(*args, **kwargs):
        raise AssertionError("formal 内核不得再次调用 seasonal_naive_forecast")

    monkeypatch.setattr(module, "seasonal_naive_forecast", boom)
    bundle = build()
    assert len(bundle.price_forecast) == _CUTOFF
