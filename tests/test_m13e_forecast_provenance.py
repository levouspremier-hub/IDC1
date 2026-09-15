"""M1.3e 测试：contract-v8 与**因果** forecast provenance。

改前缺陷（本文件在实现前必须为红）：契约仍是 `contract-v7`；
`ScenarioBundle` 用含糊的 `synthetic: bool` 与无 schema 的自由 dict `source_hashes`
承载来源语义；没有任何结构化 provenance、没有 purpose gate；
不存在只依赖 `[origin-48, origin)` 历史的因果 forecast provider，
也不存在 forecast policy manifest。

本卡**不**产生正式 `ScenarioBundle`、不开始训练或评估。
"""

import hashlib
import importlib
import json
import pathlib
import subprocess

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIMEZONE = "Asia/Singapore"
TOTAL_ROWS = 17568
TRAIN_ROWS = 10224
PERIOD_STEPS = 48
CONTRACT_V8 = "contract-v8"

DRIVERS = (
    "price_sgd_per_kwh",
    "system_load_mw",
    "temperature_deg_c",
    "wind_speed_10m_mps",
    "ghi_w_per_m2",
)
BUNDLE_FORECAST_FIELDS = (
    "price_forecast", "load_forecast", "pv_forecast", "wind_forecast",
    "temperature_forecast", "carbon_forecast", "arrival_forecast",
)
UNAVAILABLE = ("local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival")
FORECAST_MODULE = "scenario.forecast"
POLICY_MATERIALIZER = "scripts.materialize_singapore_forecast_policy"
POLICY_SCHEMA = "m1.3e-singapore-2024-forecast-policy-v1"


# --- fixture ---------------------------------------------------------------

def canonical_frame(rows: int = TOTAL_ROWS) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01T00:00:00", tz=TIMEZONE)
    stamps = pd.date_range(start, periods=rows, freq="30min")
    index = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "timestamp": stamps,
        "price_sgd_per_kwh": 0.12 + 0.001 * (index % 480),
        "system_load_mw": 6000.0 + 50.0 * np.sin(index / 48.0 * 2 * np.pi),
        "national_igs_mwh_per_half_hour": 200.0,
        "temperature_deg_c": 28.0 + 3.0 * np.sin(index / 48.0 * 2 * np.pi),
        "wind_speed_10m_mps": 2.0 + 0.5 * np.cos(index / 48.0 * 2 * np.pi),
        "ghi_w_per_m2": np.clip(300.0 * np.sin(index / 48.0 * 2 * np.pi), 0.0, None),
        "weather_source_timestamp": stamps,
        "weather_age_minutes": np.where(index % 2 == 0, 0, 30),
    })


def write_upstream(root: pathlib.Path) -> dict:
    """canonical parquet + canonical manifest + split manifest（最小自洽集合）。"""
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "half_hour.parquet"
    canonical_frame().to_parquet(parquet, index=False)
    canonical_manifest = root / "singapore_2024_half_hour.json"
    canonical_manifest.write_text(json.dumps({
        "schema": "m1.3b-singapore-2024-half-hour-v1",
        "year": 2024, "timezone": TIMEZONE, "row_count": TOTAL_ROWS,
        "frequency": "30min",
        "output_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "materializer_revision": "0" * 39 + "1",
    }), encoding="utf-8")
    split_manifest = root / "singapore_2024_splits.json"
    split_manifest.write_text(json.dumps({
        "schema": "m1.3d-singapore-2024-splits-v1",
        "year": 2024, "timezone": TIMEZONE, "frequency": "30min",
        "total_rows": TOTAL_ROWS,
        "canonical_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "canonical_manifest_sha256": hashlib.sha256(
            canonical_manifest.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return {"parquet": parquet, "canonical_manifest": canonical_manifest,
            "split_manifest": split_manifest}


def make_series_provenance(**over) -> dict:
    base = {
        "series_name": "price_forecast",
        "source_kind": "persistence",
        "method": "trailing_seasonal_naive",
        "generated_at": "2024-01-03T00:00:00+08:00",
        "information_cutoff_exclusive": "2024-01-03T00:00:00+08:00",
        "target_start": "2024-01-03T00:00:00+08:00",
        "target_end_exclusive": "2024-01-03T02:00:00+08:00",
        "lookback_start": "2024-01-02T00:00:00+08:00",
        "lookback_end_exclusive": "2024-01-03T00:00:00+08:00",
        "model_name": "trailing_seasonal_naive",
        "model_version": "v1",
        "code_revision": "a" * 40,
        "seed": None,
        "sources": [{"role": "canonical", "logical_path": "data/x.parquet",
                     "sha256": "b" * 64}],
    }
    base.update(over)
    return base


def make_provenance(*, mode: str = "formal", cutoff: int = 4) -> dict:
    """七项 provenance，`series_name` 与字段名一一对应。"""
    provenance = {}
    for field in BUNDLE_FORECAST_FIELDS:
        kind = "unavailable" if field in ("pv_forecast", "wind_forecast",
                                          "carbon_forecast", "arrival_forecast") \
            else "persistence"
        provenance[field] = make_series_provenance(series_name=field, source_kind=kind)
    return provenance


def make_bundle_kwargs(*, mode: str = "synthetic", cutoff: int = 4) -> dict:
    return {
        "split": "train", "start": "2024-01-03T00:00:00+08:00", "horizon": 24,
        "forecast_cutoff": cutoff,
        "price_forecast": [0.1] * cutoff, "load_forecast": [1.0] * cutoff,
        "pv_forecast": [0.0] * cutoff, "wind_forecast": [0.0] * cutoff,
        "temperature_forecast": [28.0] * cutoff, "carbon_forecast": [0.0] * cutoff,
        "arrival_forecast": [0.0] * cutoff,
        "mode": mode,
        "generated_at": "2024-01-03T00:00:00+08:00",
        "forecast_provenance": make_provenance(mode=mode, cutoff=cutoff),
    }


def build_bundle(**over):
    from contracts.models import ScenarioBundle

    kwargs = make_bundle_kwargs()
    kwargs.update(over)
    return ScenarioBundle(**kwargs)


def forecast_artifact(upstream, *, split="train", origin=200, cutoff=4, **over):
    module = importlib.import_module(FORECAST_MODULE)
    kwargs = dict(
        split=split, origin=origin, forecast_cutoff=cutoff,
        canonical_parquet_path=upstream["parquet"],
        canonical_manifest_path=upstream["canonical_manifest"],
        split_manifest_path=upstream["split_manifest"],
    )
    kwargs.update(over)
    return module.build_available_exogenous_forecast(**kwargs)


# --- 1. 契约版本 ------------------------------------------------------------

def test_contract_version_is_v8():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID == CONTRACT_V8


def test_v7_artifacts_are_explicitly_rejected():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID != "contract-v7"
    from safe_rl_v2.buffer import RolloutBuffer

    buffer = RolloutBuffer()
    payload = buffer.to_dict()
    payload["contract_version"] = "contract-v7"
    payload["transitions"] = []
    with pytest.raises((ValueError, KeyError, TypeError)):
        RolloutBuffer.from_dict(payload)


def test_no_second_hardcoded_version_source():
    """版本只能来自 `contracts.CONTRACT_VERSION_ID`，不得新增硬编码源。"""
    from contracts import CONTRACT_VERSION_ID

    for rel in ("checkpointing/__init__.py", "safe_rl_v2/buffer.py",
                "safe_rl_v2/lagrangian.py", "scenario/scenario.py"):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "CONTRACT_VERSION" in text or "contracts" in text, rel
        assert f'= "{CONTRACT_VERSION_ID}"' not in text, f"{rel} 硬编码了版本"
        assert "'contract-v7'" not in text and '"contract-v7"' not in text, rel


# --- 2. provenance 模型 -----------------------------------------------------

def test_provenance_models_exist_and_forbid_extra_fields():
    from contracts.models import (
        ArtifactDigest,
        ForecastSeriesProvenance,
        ScenarioForecastProvenance,
    )

    assert set(ScenarioForecastProvenance.model_fields) == set(BUNDLE_FORECAST_FIELDS)
    assert ArtifactDigest(role="r", logical_path="p", sha256="a" * 64)
    with pytest.raises((ValueError, TypeError, KeyError)):
        ForecastSeriesProvenance(**{**make_series_provenance(), "unexpected": 1})


def test_bundle_no_longer_accepts_the_free_form_source_hashes():
    from contracts.models import ScenarioBundle

    assert "source_hashes" not in ScenarioBundle.model_fields
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(source_hashes={"anything": "goes"})


def test_bundle_requires_structured_provenance():
    for missing in ("generated_at", "forecast_provenance", "mode"):
        kwargs = make_bundle_kwargs()
        kwargs.pop(missing)
        with pytest.raises((ValueError, TypeError, KeyError)):
            from contracts.models import ScenarioBundle

            ScenarioBundle(**kwargs)


def test_mode_replaces_the_ambiguous_synthetic_bool():
    from contracts.models import ScenarioBundle

    assert "mode" in ScenarioBundle.model_fields
    assert "synthetic" not in ScenarioBundle.model_fields


@pytest.mark.parametrize("bad_mode", ("formal ", "FORMAL", "", "oracle", None, 1))
def test_unknown_modes_are_rejected(bad_mode):
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode=bad_mode)


# --- 3. provenance 校验规则 -------------------------------------------------

def test_content_hash_covers_provenance():
    first = build_bundle()
    other = make_bundle_kwargs()
    other["forecast_provenance"]["price_forecast"]["model_version"] = "v2"
    assert first.content_hash() != build_bundle(**other).content_hash()


@pytest.mark.parametrize("field", (
    "series_name", "source_kind", "method", "generated_at",
    "information_cutoff_exclusive", "target_start", "target_end_exclusive",
    "model_name", "model_version", "code_revision", "sources",
))
def test_missing_provenance_field_fails_closed(field):
    provenance = make_provenance()
    provenance["price_forecast"] = {
        k: v for k, v in provenance["price_forecast"].items() if k != field}
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


def test_missing_provenance_key_for_a_series_fails_closed():
    provenance = make_provenance()
    provenance.pop("carbon_forecast")
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


def test_extra_provenance_key_without_a_matching_series_fails_closed():
    provenance = make_provenance()
    provenance["shadow_forecast"] = make_series_provenance(series_name="shadow_forecast")
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


def test_series_name_must_match_its_bundle_field():
    provenance = make_provenance()
    provenance["price_forecast"] = make_series_provenance(series_name="load_forecast")
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


@pytest.mark.parametrize("over,label", [
    ({"generated_at": "2024-01-04T00:00:00+08:00"}, "generated_at 晚于 target"),
    ({"information_cutoff_exclusive": "2024-01-04T00:00:00+08:00"}, "cutoff 晚于 target"),
    ({"lookback_end_exclusive": "2024-01-04T00:00:00+08:00"}, "lookback 晚于 cutoff"),
    ({"target_end_exclusive": "2024-01-03T00:00:00+08:00"}, "target 区间为空"),
    ({"generated_at": "2024-01-03T00:00:00"}, "naive 时间戳"),
    ({"target_start": "2024-01-03T00:00:00"}, "naive target"),
])
def test_illegal_time_orderings_fail_closed(over, label):
    provenance = make_provenance()
    provenance["price_forecast"] = make_series_provenance(**over)
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


@pytest.mark.parametrize("bad", ("A" * 64, "abc", "z" * 64, 5, None))
def test_source_hashes_must_be_lowercase_sha256(bad):
    provenance = make_provenance()
    provenance["price_forecast"] = make_series_provenance(
        sources=[{"role": "canonical", "logical_path": "p", "sha256": bad}])
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


@pytest.mark.parametrize("bad", ("A" * 40, "abc", "z" * 40, 5, None))
def test_code_revision_must_be_a_lowercase_git_sha(bad):
    provenance = make_provenance()
    provenance["price_forecast"] = make_series_provenance(code_revision=bad)
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


@pytest.mark.parametrize("bad", (True, "0", 0.5, [1]))
def test_seed_cannot_be_a_bool_or_other_type(bad):
    provenance = make_provenance()
    provenance["price_forecast"] = make_series_provenance(seed=bad)
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


def test_formal_mode_rejects_unavailable_sources():
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode="formal")


def test_formal_mode_rejects_synthetic_and_oracle_sources():
    for kind in ("synthetic", "oracle_debug"):
        provenance = make_provenance()
        for field in BUNDLE_FORECAST_FIELDS:
            provenance[field] = make_series_provenance(series_name=field,
                                                       source_kind=kind)
        with pytest.raises((ValueError, TypeError, KeyError)):
            build_bundle(mode="formal", forecast_provenance=provenance)


def test_synthetic_mode_requires_synthetic_provenance():
    provenance = make_provenance()
    provenance["price_forecast"] = make_series_provenance(
        series_name="price_forecast", source_kind="external_forecast")
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode="synthetic", forecast_provenance=provenance)


# --- 4. purpose gate --------------------------------------------------------

def _purpose_gate():
    module = importlib.import_module("contracts.validators")
    return module.validate_forecast_purpose


@pytest.mark.parametrize("mode,kind", [
    ("synthetic", "synthetic"),
    ("oracle_debug", "oracle_debug"),
])
@pytest.mark.parametrize("purpose", ("training", "evaluation"))
def test_training_and_evaluation_reject_synthetic_and_oracle(mode, kind, purpose):
    provenance = make_provenance()
    for field in BUNDLE_FORECAST_FIELDS:
        provenance[field] = make_series_provenance(series_name=field, source_kind=kind)
    bundle = build_bundle(mode=mode, forecast_provenance=provenance)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(bundle, purpose=purpose)


@pytest.mark.parametrize("mode,kind", [
    ("synthetic", "synthetic"),
    ("oracle_debug", "oracle_debug"),
])
def test_debug_purpose_accepts_synthetic_and_oracle(mode, kind):
    provenance = make_provenance()
    for field in BUNDLE_FORECAST_FIELDS:
        provenance[field] = make_series_provenance(series_name=field, source_kind=kind)
    bundle = build_bundle(mode=mode, forecast_provenance=provenance)
    _purpose_gate()(bundle, purpose="debug")


def test_unknown_purpose_is_rejected():
    bundle = build_bundle()
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(bundle, purpose="production")


# --- 5. 因果 forecast provider ---------------------------------------------

def test_provider_module_and_frozen_policy_constants():
    module = importlib.import_module(FORECAST_MODULE)
    assert module.FORECAST_PERIOD_STEPS == PERIOD_STEPS
    assert tuple(module.AVAILABLE_DRIVERS) == DRIVERS


@pytest.mark.parametrize("driver", DRIVERS)
def test_provider_returns_finite_values_of_length_cutoff(tmp_path, driver):
    upstream = write_upstream(tmp_path)
    artifact = forecast_artifact(upstream, cutoff=6)
    values = artifact.series[driver]
    assert len(values) == 6
    assert all(np.isfinite(v) for v in values)


def test_provider_records_the_window_and_cutoff(tmp_path):
    upstream = write_upstream(tmp_path)
    artifact = forecast_artifact(upstream, origin=200, cutoff=4)
    assert artifact.forecast_cutoff == 4
    assert artifact.generated_at == "2024-01-05T04:00:00+08:00"  # origin=200 → i*30min
    for driver in DRIVERS:
        provenance = artifact.provenance[driver]
        assert provenance["method"] == "trailing_seasonal_naive"
        assert provenance["generated_at"] == artifact.generated_at
        assert provenance["information_cutoff_exclusive"] == artifact.generated_at
        assert provenance["lookback_end_exclusive"] == artifact.generated_at
        assert provenance["target_start"] == artifact.generated_at
        assert provenance["lookback_start"] == "2024-01-04T04:00:00+08:00"


@pytest.mark.leakage
@pytest.mark.parametrize("driver", DRIVERS)
def test_origin_and_future_truth_do_not_change_the_forecast(tmp_path, driver):
    """改 origin 及其之后的全部 truth → forecast 值与 content hash 必须不变。"""
    upstream = write_upstream(tmp_path / "a")
    baseline = forecast_artifact(upstream, origin=200, cutoff=4)

    mutated = write_upstream(tmp_path / "b")
    frame = pd.read_parquet(mutated["parquet"])
    frame.loc[frame.index >= 200, driver] += 12345.0
    frame.to_parquet(mutated["parquet"], index=False)
    for name in ("canonical_manifest", "split_manifest"):
        path = mutated[name]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "output_parquet_sha256" in payload:
            payload["output_parquet_sha256"] = hashlib.sha256(
                mutated["parquet"].read_bytes()).hexdigest()
        else:
            payload["canonical_parquet_sha256"] = hashlib.sha256(
                mutated["parquet"].read_bytes()).hexdigest()
        path.write_text(json.dumps(payload), encoding="utf-8")
    after = forecast_artifact(mutated, origin=200, cutoff=4)

    assert after.series[driver] == baseline.series[driver]
    assert after.content_hash() == baseline.content_hash()


@pytest.mark.leakage
@pytest.mark.parametrize("driver", DRIVERS)
def test_history_window_changes_the_forecast(tmp_path, driver):
    """改 `[origin-48, origin)` → 相应 forecast 必须变化。"""
    upstream = write_upstream(tmp_path / "a")
    baseline = forecast_artifact(upstream, origin=200, cutoff=4)

    mutated = write_upstream(tmp_path / "b")
    frame = pd.read_parquet(mutated["parquet"])
    frame.loc[frame.index == 199, driver] += 999.0
    frame.to_parquet(mutated["parquet"], index=False)
    for name in ("canonical_manifest", "split_manifest"):
        path = mutated[name]
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = ("output_parquet_sha256" if "output_parquet_sha256" in payload
               else "canonical_parquet_sha256")
        payload[key] = hashlib.sha256(mutated["parquet"].read_bytes()).hexdigest()
        path.write_text(json.dumps(payload), encoding="utf-8")
    after = forecast_artifact(mutated, origin=200, cutoff=4)

    assert after.series[driver] != baseline.series[driver]


@pytest.mark.leakage
def test_validation_future_mutation_does_not_change_the_current_forecast(tmp_path):
    """validation 的未来 truth mutation 不影响当前 origin 的 forecast。"""
    upstream = write_upstream(tmp_path / "a")
    origin = TRAIN_ROWS + 100  # validation 内
    baseline = forecast_artifact(upstream, split="validation",
                                 origin=origin - TRAIN_ROWS, cutoff=4)

    mutated = write_upstream(tmp_path / "b")
    frame = pd.read_parquet(mutated["parquet"])
    frame.loc[frame.index >= origin, "price_sgd_per_kwh"] += 777.0
    frame.to_parquet(mutated["parquet"], index=False)
    for name in ("canonical_manifest", "split_manifest"):
        path = mutated[name]
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = ("output_parquet_sha256" if "output_parquet_sha256" in payload
               else "canonical_parquet_sha256")
        payload[key] = hashlib.sha256(mutated["parquet"].read_bytes()).hexdigest()
        path.write_text(json.dumps(payload), encoding="utf-8")
    after = forecast_artifact(mutated, split="validation",
                              origin=origin - TRAIN_ROWS, cutoff=4)

    assert after.content_hash() == baseline.content_hash()


def test_provider_rejects_origins_without_a_full_history_window(tmp_path):
    """train 内 origin<48 必须 fail closed —— 不回填、不跨年环绕。"""
    upstream = write_upstream(tmp_path)
    with pytest.raises(ValueError):
        forecast_artifact(upstream, split="train", origin=47, cutoff=4)


def test_provider_rejects_out_of_split_targets(tmp_path):
    upstream = write_upstream(tmp_path)
    with pytest.raises(ValueError):
        forecast_artifact(upstream, split="train", origin=TRAIN_ROWS - 2, cutoff=4)


@pytest.mark.parametrize("bad", (True, 0, -1, 1.5, "4", None))
def test_provider_rejects_bad_cutoffs(tmp_path, bad):
    upstream = write_upstream(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        forecast_artifact(upstream, origin=200, cutoff=bad)


def test_provider_hash_is_stable_for_the_same_inputs(tmp_path):
    upstream = write_upstream(tmp_path)
    first = forecast_artifact(upstream, origin=200, cutoff=4)
    second = forecast_artifact(upstream, origin=200, cutoff=4)
    assert first.content_hash() == second.content_hash()


def test_provider_artifact_carries_the_frozen_contract_version(tmp_path):
    upstream = write_upstream(tmp_path)
    artifact = forecast_artifact(upstream)
    payload = artifact.to_dict()
    assert payload["contract_version"] == CONTRACT_V8
    assert payload["split"] == "train"
    assert payload["origin"] == 200
    assert set(payload["series"]) == set(DRIVERS)


def test_provider_artifact_is_not_a_scenario_bundle(tmp_path):
    """artifact 不得伪装成完整 ScenarioBundle。"""
    upstream = write_upstream(tmp_path)
    artifact = forecast_artifact(upstream)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(artifact, purpose="training")


def test_provider_does_not_invent_the_unavailable_series(tmp_path):
    upstream = write_upstream(tmp_path)
    artifact = forecast_artifact(upstream)
    for column in UNAVAILABLE:
        assert column not in artifact.series


# --- 6. oracle helper 与 snapshot adapter -----------------------------------

def test_oracle_helper_must_be_explicitly_oracle_debug():
    module = importlib.import_module("scenario.scenario")
    assert hasattr(module, "build_oracle_debug_scenario_from_truth")
    true = {k: np.zeros(24) for k in
            ("price", "load", "pv", "wind", "temperature", "carbon", "arrival")}
    with pytest.raises(TypeError):
        module.build_oracle_debug_scenario_from_truth(  # type: ignore[call-arg]
            "train", "2024-01-01T00:00:00+08:00", 24, 4, true)
    bundle = module.build_oracle_debug_scenario_from_truth(
        "train", "2024-01-01T00:00:00+08:00", 24, 4, true, oracle_debug=True)
    assert bundle.mode == "oracle_debug"


def test_oracle_debug_bundle_is_rejected_for_training():
    module = importlib.import_module("scenario.scenario")
    true = {k: np.zeros(24) for k in
            ("price", "load", "pv", "wind", "temperature", "carbon", "arrival")}
    bundle = module.build_oracle_debug_scenario_from_truth(
        "train", "2024-01-01T00:00:00+08:00", 24, 4, true, oracle_debug=True)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(bundle, purpose="training")


def test_snapshot_adapter_declares_oracle_debug_not_formal():
    source = (REPO_ROOT / "planning/snapshot_adapter.py").read_text(encoding="utf-8")
    assert "oracle_debug" in source, "adapter 必须明确标记 oracle_debug/dev-only"
    assert "formal" not in source or "不能标 formal" in source or "非 formal" in source


def test_snapshot_adapter_snapshot_is_rejected_for_training():
    from envs.idc_price_env import IDCPriceEnv20D
    from planning.snapshot_adapter import build_snapshot

    env = IDCPriceEnv20D(horizon=24, forecast_cutoff=4, access_limit_kw=1000.0,
                         task_seed=0, server_seed=0, forecast_seed=300000)
    env.reset(seed=0)
    snapshot = build_snapshot(env)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(snapshot.forecast, purpose="training")


# --- 7. policy manifest -----------------------------------------------------

def _policy_module():
    return importlib.import_module(POLICY_MATERIALIZER)


def materialize_policy(upstream, out, **over):
    module = _policy_module()
    kwargs = dict(
        canonical_parquet_path=upstream["parquet"],
        canonical_manifest_path=upstream["canonical_manifest"],
        split_manifest_path=upstream["split_manifest"],
        manifest_path=out / "singapore_2024_forecast_policy.json",
        frozen_at_utc="2026-09-16T00:00:00+00:00",
    )
    kwargs.update(over)
    return module.materialize_forecast_policy(**kwargs)


def _policy_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def test_policy_manifest_records_the_frozen_policy(tmp_path):
    upstream = write_upstream(tmp_path)
    result = materialize_policy(upstream, tmp_path / "out")
    manifest = _policy_json(result["manifest_path"])

    assert manifest["schema"] == POLICY_SCHEMA
    assert manifest["contract_version"] == CONTRACT_V8
    assert tuple(manifest["available_drivers"]) == DRIVERS
    assert manifest["method"] == "trailing_seasonal_naive"
    assert manifest["period_steps"] == PERIOD_STEPS
    assert manifest["frequency"] == "30min"
    assert manifest["information_policy"] == "closed_open_[origin-48, origin)"
    assert manifest["target_policy"] == "half_open_[origin, origin+C)"
    assert manifest["seed_policy"] is None
    assert set(manifest["unavailable_not_materialized"]) == set(UNAVAILABLE)
    assert manifest["readiness"] == {
        "available_driver_forecasts_ready": True,
        "complete_scenario_forecasts_ready": False,
        "formal_scenario_bundle_ready": False,
        "formal_training_ready": False,
    }
    assert manifest["frozen_at_utc"] == "2026-09-16T00:00:00+00:00"


def test_policy_manifest_paths_are_repo_relative(tmp_path):
    upstream = write_upstream(tmp_path)
    result = materialize_policy(upstream, tmp_path / "out")
    text = pathlib.Path(result["manifest_path"]).read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert str(REPO_ROOT) not in text


def test_policy_manifest_is_idempotent(tmp_path):
    upstream = write_upstream(tmp_path)
    out = tmp_path / "out"
    result = materialize_policy(upstream, out)
    path = pathlib.Path(result["manifest_path"])
    st = path.stat()
    before = (st.st_size, hashlib.sha256(path.read_bytes()).hexdigest(), st.st_mtime_ns)
    materialize_policy(upstream, out)
    st = path.stat()
    assert (st.st_size, hashlib.sha256(path.read_bytes()).hexdigest(),
            st.st_mtime_ns) == before


def test_policy_manifest_rejects_tampering(tmp_path):
    upstream = write_upstream(tmp_path)
    out = tmp_path / "out"
    result = materialize_policy(upstream, out)
    path = pathlib.Path(result["manifest_path"])
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["period_steps"] = 24
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        materialize_policy(upstream, out)


def test_policy_manifest_fails_closed_on_upstream_hash_mismatch(tmp_path):
    upstream = write_upstream(tmp_path)
    upstream["parquet"].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        materialize_policy(upstream, tmp_path / "out")


def test_policy_manifest_rejects_an_unknown_field(tmp_path):
    upstream = write_upstream(tmp_path)
    out = tmp_path / "out"
    result = materialize_policy(upstream, out)
    path = pathlib.Path(result["manifest_path"])
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["future_extension"] = 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError):
        materialize_policy(upstream, out)


def test_policy_manifest_first_write_failure_leaves_no_half_state(tmp_path, monkeypatch):
    module = _policy_module()
    upstream = write_upstream(tmp_path)
    out = tmp_path / "out"

    def boom(path, text):
        raise OSError("injected policy manifest install failure")

    monkeypatch.setattr(module, "_atomic_write_text", boom)
    with pytest.raises(OSError):
        materialize_policy(upstream, out)
    assert [p.name for p in out.iterdir()] == []


def test_policy_materializer_revision_is_git_verified():
    module = _policy_module()
    revision = module.resolve_forecast_materializer_revision()
    assert isinstance(revision, str) and len(revision) == 40
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *module.FORECAST_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert revision == expected


def test_policy_dirty_generator_is_rejected(tmp_path, monkeypatch):
    module = _policy_module()
    upstream = write_upstream(tmp_path)
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        materialize_policy(upstream, tmp_path / "out")


# --- 8. 上游不变与正式路径仍 blocked -----------------------------------------

def test_upstream_manifests_are_untouched_by_this_card():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "data/raw",
         "data/manifest/singapore_2024.json",
         "data/manifest/singapore_2024_half_hour.json",
         "data/manifest/singapore_2024_splits.json",
         "data/processed", "configs/frozen_refs/refs.json"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert status == "", f"上游资产被改动：{status}"


def test_m13d_split_manifest_still_declares_forecast_not_ready():
    path = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
    if not path.exists():
        pytest.skip("split manifest 不在本机")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["readiness"]["forecast_ready"] is False


def test_no_formal_split_manifest_names_are_created():
    for name in ("train.json", "validation.json", "test.json"):
        assert not (REPO_ROOT / "data/manifest" / name).exists(), name


def test_formal_build_scenario_still_fails_closed():
    from scenario.scenario import build_scenario

    with pytest.raises((FileNotFoundError, ValueError, NotImplementedError)):
        build_scenario("train", start="2024-01-01", horizon=24, forecast_cutoff=4)


# --- 9. 真实资产的 slow 验收 ------------------------------------------------

@pytest.mark.slow
def test_real_upstream_policy_materialization(tmp_path):
    canonical_parquet = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
    canonical_manifest = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
    split_manifest = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
    if not all(p.exists() for p in (canonical_parquet, canonical_manifest, split_manifest)):
        pytest.skip("真实上游资产不在本机")
    upstream = {"parquet": canonical_parquet,
                "canonical_manifest": canonical_manifest,
                "split_manifest": split_manifest}
    artifact = forecast_artifact(upstream, origin=200, cutoff=4)
    assert set(artifact.series) == set(DRIVERS)
    result = materialize_policy(upstream, tmp_path / "out")
    manifest = _policy_json(result["manifest_path"])
    assert manifest["readiness"]["available_driver_forecasts_ready"] is True
    assert manifest["readiness"]["formal_training_ready"] is False
