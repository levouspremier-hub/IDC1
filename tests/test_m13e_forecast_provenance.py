"""M1.3e 测试：contract-v8 与**因果** forecast provenance（含 M1.3e-R1 信任链）。

改前缺陷（本文件在实现前必须为红）：

- 契约仍是 `contract-v7` 且 `ContractBase.schema_version` 可被显式覆盖；
- `ScenarioBundle` 用含糊的 `synthetic: bool` 与无 schema 的自由 dict
  `source_hashes` 承载来源语义，没有结构化 provenance、没有 purpose gate；
- 不存在只依赖 `[origin-48, origin)` 历史的因果 forecast provider；
- （M1.3e-R1）provider 不读 policy manifest、接受伪造的最小 split manifest、
  接受任意 `code_revision`；artifact 只是包裹可变 raw dict 的普通类；
  `mode`/`source_kind` 语义过宽。

**M1.3e-R1 的信任链要求**：`build_available_exogenous_forecast()` 必须逐层校验
policy → split → canonical manifest → canonical parquet，任一层不符即 fail closed。

本卡**不**产生正式 `ScenarioBundle`、不开始训练或评估。
"""

import hashlib
import importlib
import json
import math
import pathlib
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
TIMEZONE = "Asia/Singapore"
TOTAL_ROWS = 17568
TRAIN_ROWS = 10224
VALIDATION_ROWS = 2928
PERIOD_STEPS = 48
CONTRACT_V8 = "contract-v8"
CONTRACT_V7 = "contract-v7"

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
SPLIT_MATERIALIZER = "scripts.materialize_singapore_splits"
POLICY_SCHEMA = "m1.3e-singapore-2024-forecast-policy-v1"
FROZEN_AT = "2026-09-16T00:00:00+00:00"


# --- fixture: 完整、自洽的临时冻结资产链 --------------------------------------

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


def build_frozen_chain(root: pathlib.Path, frame: pd.DataFrame | None = None) -> dict:
    """构造**完整且自洽**的临时冻结资产链（M1.3e-R1 的 E.2 要求）。

    四层都用**真实的**物化器/纯构造器产出：

    1. canonical parquet（17568 行、严格 30min 网格、`Asia/Singapore`）
    2. canonical manifest（M1.3b schema）
    3. split manifest —— 由 **M1.3d 的** `build_split_manifest()` 生成
       （含精确键集合、冻结声明、readiness、unavailable、train-only 统计）
    4. policy manifest —— 由 **M1.3e 的** `build_forecast_policy_manifest()` 生成

    第 4 步刻意直接调用**纯构造器 + 原子写入**而不是 `materialize_forecast_policy()`：
    后者带 dirty-generator 门禁（「未提交的实现不得被背书」），是**生成时**的部署
    策略，不应让内容层测试在开发树上无法运行。真实入口的门禁/防覆盖/原子性/幂等
    由本文件下半部分的 `test_policy_*` 直接覆盖。
    """
    root.mkdir(parents=True, exist_ok=True)
    parquet = root / "half_hour.parquet"
    (canonical_frame() if frame is None else frame).to_parquet(parquet, index=False)

    canonical_manifest = root / "singapore_2024_half_hour.json"
    canonical_manifest.write_text(json.dumps({
        "schema": "m1.3b-singapore-2024-half-hour-v1",
        "year": 2024, "timezone": TIMEZONE, "row_count": TOTAL_ROWS,
        "frequency": "30min",
        "output_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "materializer_revision": "0" * 39 + "1",
    }), encoding="utf-8")

    split_module = importlib.import_module(SPLIT_MATERIALIZER)
    split_payload = split_module.build_split_manifest(
        canonical_parquet_path=parquet,
        canonical_manifest_path=canonical_manifest,
        frozen_at_utc=FROZEN_AT,
    )
    split_manifest = root / "singapore_2024_splits.json"
    split_manifest.write_text(
        json.dumps(split_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    policy_module = _policy_module()
    policy_payload = policy_module.build_forecast_policy_manifest(
        canonical_parquet_path=parquet,
        canonical_manifest_path=canonical_manifest,
        split_manifest_path=split_manifest,
        frozen_at_utc=FROZEN_AT,
    )
    policy_manifest = root / "singapore_2024_forecast_policy.json"
    policy_module._atomic_write_text(
        policy_manifest, policy_module._canonical_json(policy_payload)
    )

    return {
        "root": root,
        "parquet": parquet,
        "canonical_manifest": canonical_manifest,
        "split_manifest": split_manifest,
        "policy_manifest": policy_manifest,
    }


@pytest.fixture(scope="session")
def base_chain(tmp_path_factory) -> pathlib.Path:
    """会话级基链：**只构建一次**（构建含 train-only 统计重算，不便宜）。"""
    root = tmp_path_factory.mktemp("m13e_base_chain")
    build_frozen_chain(root)
    return root


@pytest.fixture
def chain(tmp_path, base_chain) -> dict:
    """把基链复制进本用例的临时目录，允许用例随意篡改。"""
    root = tmp_path / "chain"
    shutil.copytree(base_chain, root)
    return {
        "root": root,
        "parquet": root / "half_hour.parquet",
        "canonical_manifest": root / "singapore_2024_half_hour.json",
        "split_manifest": root / "singapore_2024_splits.json",
        "policy_manifest": root / "singapore_2024_forecast_policy.json",
    }


def _rewrite_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resync_policy_split_hash(chain: dict) -> None:
    """把 policy manifest 的 split hash 重新对齐到**已被篡改的** split manifest。

    这样 policy 层仍然自洽，破坏只能由 **split manifest 的严格校验**发现——
    否则用例会在 policy hash 层就被拦住，无法证明 split 校验真的在起作用。
    """
    payload = _policy_json(chain["policy_manifest"])
    payload["split_manifest_sha256"] = hashlib.sha256(
        chain["split_manifest"].read_bytes()).hexdigest()
    _rewrite_json(chain["policy_manifest"], payload)


# --- provenance / bundle 夹具（contract-v8） --------------------------------

def make_series_provenance(**over) -> dict:
    base = {
        "series_name": "price_forecast",
        "source_kind": "synthetic",
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


# 每个 mode 只接受**唯一**的来源类别（M1.3e-R1 收紧）
MODE_KINDS = {
    "synthetic": "synthetic",
    "oracle_debug": "oracle_debug",
    "formal": "persistence",
}


def make_provenance(*, mode: str = "synthetic") -> dict:
    """七项 provenance，`series_name` 与字段名一一对应，来源与该 mode 一致。"""
    kind = MODE_KINDS[mode]
    return {
        field: make_series_provenance(series_name=field, source_kind=kind)
        for field in BUNDLE_FORECAST_FIELDS
    }


def make_bundle_kwargs(*, mode: str = "synthetic", cutoff: int = 4,
                       generated_at: str = "2024-01-03T00:00:00+08:00") -> dict:
    return {
        "split": "train", "start": "2024-01-03T00:00:00+08:00", "horizon": 24,
        "forecast_cutoff": cutoff,
        "price_forecast": [0.1] * cutoff, "load_forecast": [1.0] * cutoff,
        "pv_forecast": [0.0] * cutoff, "wind_forecast": [0.0] * cutoff,
        "temperature_forecast": [28.0] * cutoff, "carbon_forecast": [0.0] * cutoff,
        "arrival_forecast": [0.0] * cutoff,
        "mode": mode,
        "generated_at": generated_at,
        "forecast_provenance": make_provenance(mode=mode),
    }


def build_bundle(*, mode: str = "synthetic", **over):
    from contracts.models import ScenarioBundle

    kwargs = make_bundle_kwargs(mode=mode)
    kwargs.update(over)
    return ScenarioBundle(**kwargs)


# --- provider / policy 入口 --------------------------------------------------

def forecast_artifact(chain, *, split="train", origin=200, cutoff=4, **over):
    module = importlib.import_module(FORECAST_MODULE)
    kwargs = dict(
        split=split, origin=origin, forecast_cutoff=cutoff,
        canonical_parquet_path=chain["parquet"],
        canonical_manifest_path=chain["canonical_manifest"],
        split_manifest_path=chain["split_manifest"],
        policy_manifest_path=chain["policy_manifest"],
    )
    kwargs.update(over)
    return module.build_available_exogenous_forecast(**kwargs)


def _policy_module():
    return importlib.import_module(POLICY_MATERIALIZER)


def materialize_policy(chain, out, **over):
    module = _policy_module()
    kwargs = dict(
        canonical_parquet_path=chain["parquet"],
        canonical_manifest_path=chain["canonical_manifest"],
        split_manifest_path=chain["split_manifest"],
        manifest_path=out / "singapore_2024_forecast_policy.json",
        frozen_at_utc=FROZEN_AT,
    )
    kwargs.update(over)
    return module.materialize_forecast_policy(**kwargs)


def _policy_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _allow_clean_generator(monkeypatch) -> None:
    """内容层测试放行生成时门禁（其拒绝行为由专门用例覆盖）。"""
    monkeypatch.setattr(_policy_module(), "_generator_is_dirty", lambda: False)


# --- 1. 契约版本（A） ---------------------------------------------------------

def test_contract_version_is_v8():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID == CONTRACT_V8


def test_v7_artifacts_are_explicitly_rejected():
    from contracts import CONTRACT_VERSION_ID

    assert CONTRACT_VERSION_ID != CONTRACT_V7
    from safe_rl_v2.buffer import RolloutBuffer

    buffer = RolloutBuffer()
    payload = buffer.to_dict()
    payload["contract_version"] = CONTRACT_V7
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
        assert f"'{CONTRACT_V7}'" not in text and f'"{CONTRACT_V7}"' not in text, rel


@pytest.mark.parametrize("bad", (
    CONTRACT_V7, "contract-v6", "", "contract-v8 ", "CONTRACT-V8",
    True, 8, 8.0, ["contract-v8"], {"schema_version": CONTRACT_V8}, None,
))
def test_contract_base_locks_schema_version(bad):
    """A.2：`schema_version` 在基底类统一锁定，任何显式覆盖都被拒绝。"""
    from contracts.models import ArtifactDigest

    with pytest.raises((ValueError, TypeError, KeyError)):
        ArtifactDigest(role="r", logical_path="p", sha256="a" * 64,
                       schema_version=bad)


def test_contract_base_default_is_the_single_version_source():
    from contracts import CONTRACT_VERSION_ID
    from contracts.models import ArtifactDigest, ScenarioBundle

    digest = ArtifactDigest(role="r", logical_path="p", sha256="a" * 64)
    assert digest.schema_version == CONTRACT_VERSION_ID
    assert build_bundle().schema_version == CONTRACT_VERSION_ID
    assert ScenarioBundle.model_fields["schema_version"].default == CONTRACT_VERSION_ID


@pytest.mark.parametrize("bad", (CONTRACT_V7, "", "contract-v6", True, 8, None))
def test_artifact_schema_version_is_locked(chain, bad):
    """A.1：新增 forecast artifact 同样锁定版本。"""
    artifact = forecast_artifact(chain)
    payload = artifact.model_dump()
    payload["schema_version"] = bad
    from contracts.models import AvailableExogenousForecast

    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**payload)


def test_version_rejections_do_not_leak_builtin_errors():
    """A.3：错误必须是干净的 ValidationError/ValueError，不泄漏内建异常。"""
    from contracts.models import ArtifactDigest

    for bad in (CONTRACT_V7, "", True, 8, ["x"], {"a": 1}, object()):
        with pytest.raises(ValueError):
            ArtifactDigest(role="r", logical_path="p", sha256="a" * 64,
                           schema_version=bad)


# --- 2. provenance 模型 -------------------------------------------------------

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
    assert "source_hashes" not in _bundle_model_fields()
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
    fields = _bundle_model_fields()
    assert "mode" in fields
    assert "synthetic" not in fields


def _bundle_model_fields() -> set:
    from contracts.models import ScenarioBundle

    return set(ScenarioBundle.model_fields)


@pytest.mark.parametrize("bad_mode", ("formal ", "FORMAL", "", "oracle", None, 1))
def test_unknown_modes_are_rejected(bad_mode):
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode=bad_mode)


# --- 3. provenance 校验规则 ---------------------------------------------------

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


# --- 4. mode / source_kind 精确语义（B） --------------------------------------

@pytest.mark.parametrize("mode", ("synthetic", "oracle_debug", "formal"))
def test_each_mode_accepts_only_its_own_kind(mode):
    bundle = build_bundle(mode=mode)
    assert bundle.mode == mode


@pytest.mark.parametrize("mode,other", (
    ("synthetic", "persistence"),
    ("synthetic", "unavailable"),
    ("synthetic", "external_forecast"),
    ("oracle_debug", "synthetic"),
    ("oracle_debug", "persistence"),
    ("formal", "synthetic"),
    ("formal", "oracle_debug"),
    ("formal", "unavailable"),
))
def test_mode_rejects_foreign_source_kinds(mode, other):
    """B.1–B.4：非 formal 要求七条**逐项**为唯一来源；formal 拒绝三类。"""
    provenance = make_provenance(mode=mode)
    provenance["carbon_forecast"] = make_series_provenance(
        series_name="carbon_forecast", source_kind=other)
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode=mode, forecast_provenance=provenance)


def test_formal_mode_rejects_unavailable_sources():
    provenance = make_provenance(mode="formal")
    provenance["load_forecast"] = make_series_provenance(
        series_name="load_forecast", source_kind="unavailable")
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode="formal", forecast_provenance=provenance)


@pytest.mark.parametrize("mode", ("synthetic", "oracle_debug", "formal"))
def test_no_bundle_mode_accepts_unavailable_placeholder(mode):
    """B.5：完整 ScenarioBundle 的任何 mode 都不得以 unavailable 占位。"""
    provenance = make_provenance(mode=mode)
    provenance["arrival_forecast"] = make_series_provenance(
        series_name="arrival_forecast", source_kind="unavailable")
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(mode=mode, forecast_provenance=provenance)


def test_bundle_generated_at_must_match_every_series(chain):
    """B.6：`ScenarioBundle.generated_at` 与七项 provenance 逐项恒等。"""
    provenance = make_provenance()
    provenance["wind_forecast"] = make_series_provenance(
        series_name="wind_forecast",
        generated_at="2024-01-03T01:00:00+08:00",
        information_cutoff_exclusive="2024-01-03T01:00:00+08:00",
        target_start="2024-01-03T01:00:00+08:00",
        lookback_end_exclusive="2024-01-03T01:00:00+08:00",
    )
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_bundle(forecast_provenance=provenance)


@pytest.mark.parametrize("mode,kind", (
    ("synthetic", "synthetic"),
    ("oracle_debug", "oracle_debug"),
))
def test_synthetic_and_oracle_modes_are_exact(mode, kind):
    bundle = build_bundle(mode=mode)
    assert bundle.mode == mode
    for field in BUNDLE_FORECAST_FIELDS:
        assert getattr(bundle.forecast_provenance, field).source_kind == kind


# --- 5. purpose gate ----------------------------------------------------------

def _purpose_gate():
    module = importlib.import_module("contracts.validators")
    return module.validate_forecast_purpose


@pytest.mark.parametrize("mode", ("synthetic", "oracle_debug"))
@pytest.mark.parametrize("purpose", ("training", "evaluation"))
def test_training_and_evaluation_reject_synthetic_and_oracle(mode, purpose):
    bundle = build_bundle(mode=mode)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(bundle, purpose=purpose)


@pytest.mark.parametrize("mode", ("synthetic", "oracle_debug"))
def test_debug_purpose_accepts_synthetic_and_oracle(mode):
    _purpose_gate()(build_bundle(mode=mode), purpose="debug")


def test_unknown_purpose_is_rejected():
    bundle = build_bundle()
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(bundle, purpose="production")


def test_formal_bundle_is_accepted_for_training_and_evaluation():
    bundle = build_bundle(mode="formal")
    _purpose_gate()(bundle, purpose="training")
    _purpose_gate()(bundle, purpose="evaluation")


# --- 6. 因果 forecast：**纯函数**层的泄漏回归（E） --------------------------

def _series_values(rows: int = 400) -> np.ndarray:
    index = np.arange(rows, dtype=float)
    return 0.12 + 0.001 * (index % 480)


@pytest.mark.leakage
def test_pure_forecast_ignores_truth_at_or_after_origin():
    """E.3：`[origin, end)` 的任何变化都不得改变预测。"""
    module = importlib.import_module(FORECAST_MODULE)
    series = _series_values()
    baseline = module.seasonal_naive_forecast(series, origin=200, forecast_cutoff=8)

    mutated = series.copy()
    mutated[200:] += 12345.0
    after = module.seasonal_naive_forecast(mutated, origin=200, forecast_cutoff=8)

    assert after == baseline
    assert all(math.isfinite(v) for v in after)


@pytest.mark.leakage
def test_pure_forecast_changes_when_the_used_history_rows_change():
    """E.4：整个历史窗口 `[origin-48, origin)` 的变化必须改变预测。"""
    module = importlib.import_module(FORECAST_MODULE)
    series = _series_values()
    baseline = module.seasonal_naive_forecast(series, origin=200, forecast_cutoff=8)

    mutated = series.copy()
    mutated[200 - PERIOD_STEPS:200] += 999.0
    after = module.seasonal_naive_forecast(mutated, origin=200, forecast_cutoff=8)

    assert after != baseline


def test_pure_forecast_is_the_frozen_card_rule():
    """E.5：卡片字面规则 `forecast[k] = y(origin + k - 48)`（模板按时间正序）。"""
    module = importlib.import_module(FORECAST_MODULE)
    index = np.arange(200, dtype=float)
    series = index * 1.0
    forecast = module.seasonal_naive_forecast(series, origin=100, forecast_cutoff=60)
    assert list(forecast[:5]) == [52.0, 53.0, 54.0, 55.0, 56.0]  # y(100+k-48)
    assert list(forecast[48:52]) == [52.0, 53.0, 54.0, 55.0]  # k mod 48 环绕
    assert forecast[0] == series[100 - PERIOD_STEPS]


@pytest.mark.parametrize("bad", (True, 0, -1, 1.5, "4", None))
def test_pure_forecast_rejects_bad_cutoffs(bad):
    module = importlib.import_module(FORECAST_MODULE)
    with pytest.raises((ValueError, TypeError)):
        module.seasonal_naive_forecast(_series_values(), origin=200, forecast_cutoff=bad)


def test_pure_forecast_fails_closed_without_a_full_history_window():
    """train 内 origin<48 必须 fail closed —— 不回填、不跨年环绕。"""
    module = importlib.import_module(FORECAST_MODULE)
    with pytest.raises(ValueError):
        module.seasonal_naive_forecast(_series_values(), origin=47, forecast_cutoff=4)


def test_pure_forecast_is_the_implementation_the_provider_uses(chain):
    """provider 的数值必须与纯函数逐项一致（不得有第二条计算路径）。"""
    module = importlib.import_module(FORECAST_MODULE)
    artifact = forecast_artifact(chain, origin=200, cutoff=6)
    frame = pd.read_parquet(chain["parquet"])
    for driver in DRIVERS:
        expected = module.seasonal_naive_forecast(
            frame[driver].to_numpy(), origin=200, forecast_cutoff=6)
        assert tuple(artifact.series[driver]) == expected


# --- 7. provider：完整信任链（C） --------------------------------------------

def test_provider_module_and_frozen_policy_constants():
    module = importlib.import_module(FORECAST_MODULE)
    assert module.FORECAST_PERIOD_STEPS == PERIOD_STEPS
    assert tuple(module.AVAILABLE_DRIVERS) == DRIVERS


@pytest.mark.parametrize("driver", DRIVERS)
def test_provider_returns_finite_values_of_length_cutoff(chain, driver):
    artifact = forecast_artifact(chain, cutoff=6)
    values = artifact.series[driver]
    assert len(values) == 6
    assert all(np.isfinite(v) for v in values)


def test_provider_records_the_window_and_cutoff(chain):
    artifact = forecast_artifact(chain, origin=200, cutoff=4)
    assert artifact.forecast_cutoff == 4
    assert artifact.origin == 200
    assert artifact.global_origin == 200
    assert artifact.generated_at == "2024-01-05T04:00:00+08:00"  # origin=200 → +100h
    assert artifact.target_timestamps[0] == artifact.generated_at
    for driver in DRIVERS:
        entry = artifact.provenance[driver]
        assert entry.method == "trailing_seasonal_naive"
        assert entry.source_kind == "seasonal_naive"
        assert entry.series_name == driver
        assert entry.generated_at == artifact.generated_at
        assert entry.information_cutoff_exclusive == artifact.generated_at
        assert entry.lookback_end_exclusive == artifact.generated_at
        assert entry.target_start == artifact.generated_at
        assert entry.lookback_start == "2024-01-04T04:00:00+08:00"
        assert entry.seed is None


def test_provider_requires_a_policy_manifest(chain):
    """C.1：`policy_manifest_path` 是必填 kwarg。"""
    module = importlib.import_module(FORECAST_MODULE)
    with pytest.raises(TypeError):
        module.build_available_exogenous_forecast(  # type: ignore[call-arg]
            "train", origin=200, forecast_cutoff=4,
            canonical_parquet_path=chain["parquet"],
            canonical_manifest_path=chain["canonical_manifest"],
            split_manifest_path=chain["split_manifest"],
        )


def test_provider_rejects_a_missing_policy_manifest(chain):
    chain["policy_manifest"].unlink()
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_rejects_a_malformed_policy_manifest(chain):
    chain["policy_manifest"].write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        forecast_artifact(chain)


@pytest.mark.parametrize("field,value", [
    ("schema", "m1.3e-other-schema"),
    ("contract_version", CONTRACT_V7),
    ("contract_version", "contract-v9"),
    ("method", "persistence"),
    ("period_steps", 24),
    ("frequency", "60min"),
    ("information_policy", "closed_open_[origin-24, origin)"),
    ("target_policy", "half_open_[origin-1, origin+C)"),
    ("seed_policy", 0),
    ("available_drivers", list(DRIVERS)[:4]),
    ("available_drivers", list(DRIVERS) + ["extra"]),
    ("unavailable_not_materialized", list(UNAVAILABLE)[:3]),
    ("materializer_revision", "0" * 40),
    ("frozen_at_utc", "2026-09-16T00:00:00"),
    ("canonical_parquet_sha256", "0" * 64),
    ("canonical_manifest_sha256", "0" * 64),
    ("split_manifest_sha256", "0" * 64),
    ("canonical_parquet_path", "data/processed/other.parquet"),
    ("canonical_manifest_path", "data/manifest/other.json"),
    ("split_manifest_path", "data/manifest/other_splits.json"),
])
def test_provider_rejects_a_tampered_policy_manifest(chain, field, value):
    """C.2/C.3/C.4/C.5/C.9：policy 的任何一项被改动都必须拒绝。"""
    payload = _policy_json(chain["policy_manifest"])
    payload[field] = value
    _rewrite_json(chain["policy_manifest"], payload)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_rejects_readiness_flips(chain):
    """C.3：readiness 必须严格等于冻结值（不得提前声明就绪）。"""
    for field, value in (("formal_training_ready", True),
                         ("complete_scenario_forecasts_ready", True),
                         ("formal_scenario_bundle_ready", True),
                         ("available_driver_forecasts_ready", False)):
        payload = _policy_json(chain["policy_manifest"])
        payload["readiness"][field] = value
        _rewrite_json(chain["policy_manifest"], payload)
        with pytest.raises(ValueError):
            forecast_artifact(chain)


def test_provider_rejects_an_unknown_policy_field(chain):
    payload = _policy_json(chain["policy_manifest"])
    payload["future_extension"] = 1
    _rewrite_json(chain["policy_manifest"], payload)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_rejects_a_minimal_forged_split_manifest(chain):
    """C.6：只含正确 parquet hash 的最小伪造 split manifest 必须拒绝。"""
    parquet_sha = hashlib.sha256(chain["parquet"].read_bytes()).hexdigest()
    chain["split_manifest"].write_text(json.dumps({
        "schema": "m1.3d-singapore-2024-splits-v1",
        "year": 2024, "timezone": TIMEZONE, "frequency": "30min",
        "total_rows": TOTAL_ROWS,
        "canonical_parquet_sha256": parquet_sha,
        "canonical_manifest_sha256": hashlib.sha256(
            chain["canonical_manifest"].read_bytes()).hexdigest(),
    }), encoding="utf-8")
    _resync_policy_split_hash(chain)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


@pytest.mark.parametrize("mutation", [
    "extra_top_level_key",
    "drop_readiness_key",
    "flip_forecast_ready",
    "forge_year",
    "forge_step_minutes",
    "forge_path",
    "forge_origin_rule",
    "drop_split",
    "extra_split",
    "extra_split_entry_key",
    "forge_train_statistic",
    "empty_unavailable",
    "unavailable_as_container",
])
def test_provider_rejects_a_tampered_split_manifest(chain, mutation):
    """C.6/C.7：split manifest 必须走 M1.3d 的完整严格校验。"""
    payload = _policy_json(chain["split_manifest"])
    if mutation == "extra_top_level_key":
        payload["future_extension"] = 1
    elif mutation == "drop_readiness_key":
        payload["readiness"].pop("truth_splits_ready")
    elif mutation == "flip_forecast_ready":
        payload["readiness"]["forecast_ready"] = True
    elif mutation == "forge_year":
        payload["year"] = 1999
    elif mutation == "forge_step_minutes":
        payload["step_minutes"] = 60
    elif mutation == "forge_path":
        payload["canonical_parquet_path"] = "data/processed/other.parquet"
    elif mutation == "forge_origin_rule":
        payload["forecast_origin_rule"] = "anything"
    elif mutation == "drop_split":
        payload["splits"].pop("test")
    elif mutation == "extra_split":
        payload["splits"]["shadow_test"] = payload["splits"]["test"]
    elif mutation == "extra_split_entry_key":
        payload["splits"]["train"]["randomized_indices"] = []
    elif mutation == "forge_train_statistic":
        payload["train_only_statistics"]["price_sgd_per_kwh"]["mean"] = 9999.0
    elif mutation == "empty_unavailable":
        payload["unavailable_not_materialized"] = {}
    elif mutation == "unavailable_as_container":
        payload["unavailable_not_materialized"]["arrival"] = ["unavailable"]
    _rewrite_json(chain["split_manifest"], payload)
    _resync_policy_split_hash(chain)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_rejects_a_canonical_manifest_byte_change(chain):
    """C.8：canonical manifest 自身字节变化（parquet 不变）也必须拒绝。"""
    payload = _policy_json(chain["canonical_manifest"])
    payload["extra_note"] = "tampered"
    _rewrite_json(chain["canonical_manifest"], payload)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_rejects_a_tampered_parquet(chain):
    chain["parquet"].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_rejects_a_policy_revision_that_is_not_the_provider_revision(chain):
    """C.10：artifact 的 revision 必须与 policy revision 恒等。"""
    payload = _policy_json(chain["policy_manifest"])
    payload["materializer_revision"] = "f" * 40
    _rewrite_json(chain["policy_manifest"], payload)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_has_no_public_code_revision_parameter(chain):
    """C.10：`code_revision` 不得是公开参数（只能由内部 Git resolver 得到）。"""
    module = importlib.import_module(FORECAST_MODULE)
    import inspect

    parameters = inspect.signature(module.build_available_exogenous_forecast).parameters
    assert "code_revision" not in parameters
    artifact = forecast_artifact(chain)
    assert artifact.code_revision == module.provider_code_revision()
    assert artifact.provenance["price_sgd_per_kwh"].code_revision == (
        artifact.code_revision
    )


def test_provider_artifact_code_revision_is_a_git_resolved_sha(chain):
    module = importlib.import_module(FORECAST_MODULE)
    revision = module.provider_code_revision()
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *module.FORECAST_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert revision == expected
    assert len(revision) == 40


def test_provider_rejects_out_of_split_targets(chain):
    with pytest.raises(ValueError):
        forecast_artifact(chain, split="train", origin=TRAIN_ROWS - 2, cutoff=4)


def test_provider_rejects_origins_without_a_full_history_window(chain):
    with pytest.raises(ValueError):
        forecast_artifact(chain, split="train", origin=47, cutoff=4)


@pytest.mark.parametrize("bad", (True, 0, -1, 1.5, "4", None))
def test_provider_rejects_bad_cutoffs(chain, bad):
    with pytest.raises((ValueError, TypeError)):
        forecast_artifact(chain, origin=200, cutoff=bad)


def test_provider_uses_preceding_canonical_history_for_validation(chain):
    """validation 的起点可以使用它**之前已经发生**的 canonical 历史。"""
    artifact = forecast_artifact(chain, split="validation", origin=100, cutoff=4)
    assert artifact.split == "validation"
    assert artifact.origin == 100
    assert artifact.global_origin == TRAIN_ROWS + 100
    assert artifact.provenance["price_sgd_per_kwh"].lookback_start is not None


# --- 8. 严格冻结的 artifact 契约（D） ----------------------------------------

def test_artifact_is_a_frozen_contract(chain):
    from contracts.models import AvailableExogenousForecast

    artifact = forecast_artifact(chain)
    assert isinstance(artifact, AvailableExogenousForecast)
    assert artifact.schema_version == CONTRACT_V8
    with pytest.raises((ValueError, TypeError)):
        artifact.origin = 1  # type: ignore[misc]


def test_artifact_nested_containers_are_immutable(chain):
    """D.4：`series` / `provenance` 不得暴露可变内部 dict/list。"""
    artifact = forecast_artifact(chain)
    values = artifact.series["price_sgd_per_kwh"]
    assert isinstance(values, tuple)
    with pytest.raises(TypeError):
        values[0] = 1.0  # type: ignore[index]
    entry = artifact.provenance["price_sgd_per_kwh"]
    with pytest.raises((ValueError, TypeError)):
        entry.method = "other"  # type: ignore[misc]
    with pytest.raises(KeyError):
        artifact.series["not_a_driver"]  # type: ignore[index]
    with pytest.raises(KeyError):
        artifact.provenance["not_a_driver"]  # type: ignore[index]


def test_artifact_rejects_extra_and_missing_drivers(chain):
    from contracts.models import AvailableExogenousForecast

    payload = forecast_artifact(chain).model_dump()
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "future_driver": 1})

    trimmed = dict(payload["series"])
    trimmed.pop("ghi_w_per_m2")
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "series": trimmed})

    trimmed_provenance = dict(payload["provenance"])
    trimmed_provenance.pop("system_load_mw")
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "provenance": trimmed_provenance})


@pytest.mark.parametrize("field,value", [
    ("forecast_cutoff", True),
    ("forecast_cutoff", 0),
    ("forecast_cutoff", -1),
    ("origin", -1),
    ("origin", True),
    ("period_steps", 0),
    ("period_steps", True),
    ("code_revision", "not-a-git-sha"),
    ("code_revision", "A" * 40),
    ("policy_manifest_sha256", "abc"),
    ("canonical_parquet_sha256", "A" * 64),
    ("frequency", ""),
    ("method", ""),
    ("split", ""),
    ("generated_at", "2024-01-05T04:00:00"),
    ("policy_manifest_path", ""),
])
def test_artifact_rejects_illegal_fields(chain, field, value):
    from contracts.models import AvailableExogenousForecast

    payload = forecast_artifact(chain).model_dump()
    payload[field] = value
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**payload)


def test_artifact_rejects_wrong_series_length_and_non_finite(chain):
    from contracts.models import AvailableExogenousForecast

    payload = forecast_artifact(chain, cutoff=4).model_dump()
    short = dict(payload["series"])
    short["price_sgd_per_kwh"] = [0.1, 0.2]
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "series": short})

    bad = dict(payload["series"])
    bad["price_sgd_per_kwh"] = [float("nan")] * 4
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "series": bad})


def test_artifact_rejects_wrong_source_kind_and_generated_at(chain):
    from contracts.models import AvailableExogenousForecast

    payload = forecast_artifact(chain).model_dump()
    provenance = {driver: dict(payload["provenance"][driver]) for driver in DRIVERS}
    provenance["wind_speed_10m_mps"]["source_kind"] = "persistence"
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "provenance": provenance})

    provenance = {driver: dict(payload["provenance"][driver]) for driver in DRIVERS}
    provenance["wind_speed_10m_mps"]["generated_at"] = "2024-01-05T05:00:00+08:00"
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**{**payload, "provenance": provenance})


def test_artifact_has_two_distinct_summaries(chain):
    """D.6：`prediction_hash` 只覆盖预测；`content_hash` 覆盖完整 artifact。"""
    artifact = forecast_artifact(chain)
    assert artifact.prediction_hash() != artifact.content_hash()
    assert len(artifact.prediction_hash()) == 64
    assert len(artifact.content_hash()) == 64


def test_artifact_prediction_hash_covers_values_units_and_order(chain):
    from contracts.models import AvailableExogenousForecast

    artifact = forecast_artifact(chain, cutoff=4)
    payload = artifact.model_dump()

    reordered = {driver: list(payload["series"][driver]) for driver in DRIVERS}
    reordered["price_sgd_per_kwh"] = list(reversed(reordered["price_sgd_per_kwh"]))
    other = AvailableExogenousForecast(**{**payload, "series": reordered})
    assert other.prediction_hash() != artifact.prediction_hash()

    bumped = {driver: list(payload["series"][driver]) for driver in DRIVERS}
    bumped["system_load_mw"] = [v + 1.0 for v in bumped["system_load_mw"]]
    other = AvailableExogenousForecast(**{**payload, "series": bumped})
    assert other.prediction_hash() != artifact.prediction_hash()


def test_artifact_content_hash_covers_the_audit_provenance(chain):
    """D.7：不得为通过 leakage 测试而让 content_hash 忽略审计 provenance。

    把顶层 hash 与五条 provenance 的对应 digest **一致地**换成另一个值——
    这样构造仍然合法（内部闭环成立），此时 content_hash 必须变化而
    prediction_hash 不变。**R2 起单改顶层字段会直接构造失败**（闭环被破坏）。
    """
    from contracts.models import AvailableExogenousForecast

    artifact = forecast_artifact(chain, cutoff=4)
    payload = artifact.model_dump()
    replacement = "c" * 64
    provenance = {name: dict(payload["provenance"][name]) for name in DRIVERS}
    for name in DRIVERS:
        sources = [dict(digest) for digest in provenance[name]["sources"]]
        sources[0]["sha256"] = replacement
        provenance[name]["sources"] = sources
    tampered = {
        **payload,
        "canonical_parquet_sha256": replacement,
        "provenance": provenance,
    }
    other = AvailableExogenousForecast(**tampered)
    assert other.content_hash() != artifact.content_hash()
    # 预测本身没变
    assert other.prediction_hash() == artifact.prediction_hash()


def test_artifact_is_not_a_scenario_bundle(chain):
    artifact = forecast_artifact(chain)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(artifact, purpose="training")
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(artifact, purpose="evaluation")
    _purpose_gate()(artifact, purpose="debug")


def test_artifact_does_not_invent_the_unavailable_series(chain):
    artifact = forecast_artifact(chain)
    for column in UNAVAILABLE:
        assert column not in artifact.series.as_dict()
    assert set(artifact.series.as_dict()) == set(DRIVERS)


def test_artifact_validator_entry_point(chain):
    from contracts.validators import validate_available_forecast

    validate_available_forecast(forecast_artifact(chain))


# --- 9. artifact 层的因果性 / 泄漏回归（E.3/E.4） ---------------------------

@pytest.mark.leakage
def test_future_truth_mutation_leaves_series_and_prediction_hash_unchanged(tmp_path):
    """E.3：改 origin 之后的未来 truth → series 与 prediction_hash 不变。

    两条链各自**完整自洽**（各自重算 hash 与 train-only 统计），因此这不是
    「同步改几个 hash 绕过校验」，而是比较两个都合法的冻结资产链。
    """
    baseline_chain = build_frozen_chain(tmp_path / "baseline")

    frame = canonical_frame()
    frame.loc[frame.index >= 200, "price_sgd_per_kwh"] += 12345.0
    mutated_chain = build_frozen_chain(tmp_path / "mutated", frame=frame)

    baseline = forecast_artifact(baseline_chain, origin=200, cutoff=4)
    after = forecast_artifact(mutated_chain, origin=200, cutoff=4)

    for driver in DRIVERS:
        assert tuple(after.series[driver]) == tuple(baseline.series[driver])
    assert after.prediction_hash() == baseline.prediction_hash()
    # 审计 provenance 不同（上游字节确实变了）——两个摘要语义因此可区分
    assert after.content_hash() != baseline.content_hash()
    assert after.canonical_parquet_sha256 != baseline.canonical_parquet_sha256


@pytest.mark.leakage
def test_history_window_mutation_changes_series_and_prediction_hash(tmp_path):
    """E.4：改**实际使用的历史行** `[origin-48, origin)` → series 与 hash 必须变化。"""
    baseline_chain = build_frozen_chain(tmp_path / "baseline")

    frame = canonical_frame()
    frame.loc[frame.index.isin(range(200 - PERIOD_STEPS, 200)), "price_sgd_per_kwh"] += 999.0
    mutated_chain = build_frozen_chain(tmp_path / "mutated", frame=frame)

    baseline = forecast_artifact(baseline_chain, origin=200, cutoff=4)
    after = forecast_artifact(mutated_chain, origin=200, cutoff=4)

    assert tuple(after.series["price_sgd_per_kwh"]) != tuple(
        baseline.series["price_sgd_per_kwh"])
    assert after.prediction_hash() != baseline.prediction_hash()


@pytest.mark.leakage
def test_mutating_only_the_unused_tail_of_the_history_window_does_not_change_it(tmp_path):
    """只有模板下标 `>= C`（即 `origin-48+C` 之后）的行不参与计算，改动无效。"""
    baseline_chain = build_frozen_chain(tmp_path / "baseline")

    frame = canonical_frame()
    frame.loc[frame.index == 199, "price_sgd_per_kwh"] += 999.0
    mutated_chain = build_frozen_chain(tmp_path / "mutated", frame=frame)

    baseline = forecast_artifact(baseline_chain, origin=200, cutoff=4)
    after = forecast_artifact(mutated_chain, origin=200, cutoff=4)
    assert after.prediction_hash() == baseline.prediction_hash()

    # 同一行在 C=48 时**确实**参与计算 → 必须改变预测
    wide = forecast_artifact(mutated_chain, origin=200, cutoff=48)
    wide_baseline = forecast_artifact(baseline_chain, origin=200, cutoff=48)
    assert wide.prediction_hash() != wide_baseline.prediction_hash()


@pytest.mark.leakage
def test_provider_fails_closed_when_frozen_sources_are_rewritten(chain):
    """E.3：正式冻结 artifact 的来源文件被改写 → provider 必须拒绝。"""
    chain["parquet"].write_bytes(b"rewritten")
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_provider_hash_is_stable_for_the_same_inputs(chain):
    first = forecast_artifact(chain, origin=200, cutoff=4)
    second = forecast_artifact(chain, origin=200, cutoff=4)
    assert first.content_hash() == second.content_hash()
    assert first.prediction_hash() == second.prediction_hash()


# --- 10. oracle helper 与 snapshot adapter -----------------------------------

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
    for field in BUNDLE_FORECAST_FIELDS:
        entry = getattr(bundle.forecast_provenance, field)
        assert entry.source_kind == "oracle_debug"
        assert entry.generated_at == bundle.generated_at


def test_oracle_debug_bundle_is_rejected_for_training():
    module = importlib.import_module("scenario.scenario")
    true = {k: np.zeros(24) for k in
            ("price", "load", "pv", "wind", "temperature", "carbon", "arrival")}
    bundle = module.build_oracle_debug_scenario_from_truth(
        "train", "2024-01-01T00:00:00+08:00", 24, 4, true, oracle_debug=True)
    with pytest.raises((ValueError, TypeError, KeyError)):
        _purpose_gate()(bundle, purpose="training")


def test_synthetic_bundle_is_all_synthetic():
    module = importlib.import_module("scenario.scenario")
    bundle = module.build_scenario("train", "s", 24, 4, synthetic=True, seed=5)
    assert bundle.mode == "synthetic"
    for field in BUNDLE_FORECAST_FIELDS:
        entry = getattr(bundle.forecast_provenance, field)
        assert entry.source_kind == "synthetic"
        assert entry.generated_at == bundle.generated_at


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


# --- 11. policy manifest（真实入口） ----------------------------------------

def test_policy_manifest_records_the_frozen_policy(chain, monkeypatch):
    _allow_clean_generator(monkeypatch)
    result = materialize_policy(chain, chain["root"])
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
    assert manifest["frozen_at_utc"] == FROZEN_AT
    assert manifest["materializer_revision"] == importlib.import_module(
        FORECAST_MODULE).provider_code_revision()


def test_policy_manifest_paths_are_repo_relative(chain):
    text = chain["policy_manifest"].read_text(encoding="utf-8")
    assert "/Users/" not in text
    assert str(REPO_ROOT) not in text


def test_policy_manifest_is_idempotent(chain, monkeypatch):
    _allow_clean_generator(monkeypatch)
    out = chain["root"]
    result = materialize_policy(chain, out)
    path = pathlib.Path(result["manifest_path"])
    st = path.stat()
    before = (st.st_size, hashlib.sha256(path.read_bytes()).hexdigest(), st.st_mtime_ns)
    materialize_policy(chain, out)
    st = path.stat()
    assert (st.st_size, hashlib.sha256(path.read_bytes()).hexdigest(),
            st.st_mtime_ns) == before


def test_policy_manifest_rejects_tampering(chain, monkeypatch):
    _allow_clean_generator(monkeypatch)
    out = chain["root"]
    tampered = _policy_json(chain["policy_manifest"])
    tampered["period_steps"] = 24
    _rewrite_json(chain["policy_manifest"], tampered)
    with pytest.raises(ValueError):
        materialize_policy(chain, out)


def test_policy_manifest_fails_closed_on_upstream_hash_mismatch(chain, monkeypatch):
    _allow_clean_generator(monkeypatch)
    chain["parquet"].write_bytes(b"tampered")
    with pytest.raises(ValueError):
        materialize_policy(chain, chain["root"])


def test_policy_manifest_rejects_a_forged_minimal_split_manifest(chain, monkeypatch):
    """物化器同样要求 split manifest 通过 M1.3d 的完整严格校验。"""
    _allow_clean_generator(monkeypatch)
    chain["split_manifest"].write_text(json.dumps({
        "schema": "m1.3d-singapore-2024-splits-v1",
        "canonical_parquet_sha256": hashlib.sha256(
            chain["parquet"].read_bytes()).hexdigest(),
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        materialize_policy(chain, chain["root"])


def test_policy_manifest_rejects_an_unknown_field(chain, monkeypatch):
    _allow_clean_generator(monkeypatch)
    out = chain["root"]
    tampered = _policy_json(chain["policy_manifest"])
    tampered["future_extension"] = 1
    _rewrite_json(chain["policy_manifest"], tampered)
    with pytest.raises(ValueError):
        materialize_policy(chain, out)


def test_policy_manifest_first_write_failure_leaves_no_half_state(chain, monkeypatch):
    _allow_clean_generator(monkeypatch)
    module = _policy_module()
    out = chain["root"] / "fresh_out"

    def boom(path, text):
        raise OSError("injected policy manifest install failure")

    monkeypatch.setattr(module, "_atomic_write_text", boom)
    with pytest.raises(OSError):
        materialize_policy(chain, out)
    assert [p.name for p in out.iterdir()] == []


def test_policy_materializer_revision_is_git_verified():
    module = _policy_module()
    revision = module.resolve_forecast_materializer_revision()
    assert isinstance(revision, str) and len(revision) == 40
    expected = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *module.FORECAST_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert revision == expected


def test_policy_dirty_generator_is_rejected(chain, monkeypatch):
    module = _policy_module()
    monkeypatch.setattr(module, "_generator_is_dirty", lambda: True)
    with pytest.raises(ValueError, match="未提交"):
        materialize_policy(chain, chain["root"] / "out")


def test_policy_dirty_generator_check_matches_git():
    """`_generator_is_dirty()` 必须真的以 `git status` 为准，而不是常量。"""
    module = _policy_module()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *module.FORECAST_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert module._generator_is_dirty() is bool(status.strip())


# --- 12. 上游不变与正式路径仍 blocked -----------------------------------------

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


# --- 13. 真实资产的 slow 验收 ------------------------------------------------

@pytest.mark.slow
def test_real_upstream_policy_materialization(tmp_path):
    canonical_parquet = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
    canonical_manifest = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
    split_manifest = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
    policy_manifest = REPO_ROOT / "data/manifest/singapore_2024_forecast_policy.json"
    if not all(p.exists() for p in (canonical_parquet, canonical_manifest,
                                    split_manifest, policy_manifest)):
        pytest.skip("真实上游资产不在本机")
    chain = {"parquet": canonical_parquet, "canonical_manifest": canonical_manifest,
             "split_manifest": split_manifest, "policy_manifest": policy_manifest}
    artifact = forecast_artifact(chain, origin=200, cutoff=4)
    assert set(artifact.series.as_dict()) == set(DRIVERS)
    assert artifact.code_revision == importlib.import_module(
        FORECAST_MODULE).provider_code_revision()
    manifest = _policy_json(policy_manifest)
    assert manifest["readiness"]["available_driver_forecasts_ready"] is True
    assert manifest["readiness"]["formal_training_ready"] is False
    assert manifest["materializer_revision"] == artifact.code_revision


# --- 14. M1.3e-R2：深度不可变 -------------------------------------------------

def _entry(**over):
    from contracts.models import ForecastSeriesProvenance

    return ForecastSeriesProvenance(**{**make_series_provenance(), **over})


def test_provenance_sources_are_an_immutable_tuple():
    entry = _entry()
    assert isinstance(entry.sources, tuple)
    with pytest.raises(AttributeError):
        entry.sources.append(  # type: ignore[attr-defined]
            entry.sources[0]
        )
    with pytest.raises(TypeError):
        entry.sources[0] = entry.sources[0]  # type: ignore[index]


def test_list_input_is_normalised_to_an_immutable_tuple():
    """JSON/list 输入可在严格验证后规范化为 tuple，但对外不得暴露可变容器。"""
    entry = _entry(sources=[dict(make_series_provenance()["sources"][0])])
    assert isinstance(entry.sources, tuple)
    assert isinstance(entry.sources[0].role, str)
    bundle = build_bundle()  # fixture 传的是 list
    assert isinstance(bundle.price_forecast, tuple)


def test_scenario_bundle_forecasts_are_immutable_tuples():
    bundle = build_bundle()
    before = bundle.content_hash()
    for field in BUNDLE_FORECAST_FIELDS:
        values = getattr(bundle, field)
        assert isinstance(values, tuple), field
        with pytest.raises(TypeError):
            values[0] = 1.0  # type: ignore[index]
    assert bundle.content_hash() == before


def test_artifact_deep_immutability_keeps_the_content_hash_stable(chain):
    """A.2：四种原地修改全部失败，且操作前后 content_hash 不变。"""
    artifact = forecast_artifact(chain, cutoff=4)
    entry = artifact.provenance["price_sgd_per_kwh"]
    before = artifact.content_hash()

    with pytest.raises(AttributeError):
        entry.sources.append(entry.sources[0])  # type: ignore[attr-defined]
    assert artifact.content_hash() == before

    with pytest.raises(TypeError):
        entry.sources[0] = entry.sources[0]  # type: ignore[index]
    assert artifact.content_hash() == before

    with pytest.raises((ValueError, TypeError)):
        entry.method = "other"  # type: ignore[misc]
    assert artifact.content_hash() == before

    with pytest.raises(TypeError):
        artifact.series["price_sgd_per_kwh"][0] = 1.0  # type: ignore[index]
    assert artifact.content_hash() == before

    assert artifact.content_hash() == forecast_artifact(chain, cutoff=4).content_hash()


# --- 15. M1.3e-R2：顶层 ↔ 逐序列证据闭环 -------------------------------------

def _artifact_payload(chain, **over):
    artifact = forecast_artifact(chain, cutoff=4, **over)
    return artifact, artifact.model_dump()


def _mutate_provenance(payload, driver, field, value):
    provenance = {name: dict(payload["provenance"][name]) for name in DRIVERS}
    provenance[driver][field] = value
    return {**payload, "provenance": provenance}


def test_artifact_carries_all_four_upstream_paths(chain):
    artifact = forecast_artifact(chain, cutoff=4)
    for field in ("canonical_parquet_path", "canonical_manifest_path",
                  "split_manifest_path", "policy_manifest_path"):
        value = getattr(artifact, field)
        assert isinstance(value, str) and value
        assert not value.startswith("/") and "\\" not in value


def test_artifact_five_provenances_carry_identical_sources(chain):
    artifact = forecast_artifact(chain, cutoff=4)
    reference = artifact.provenance[DRIVERS[0]].sources
    assert [d.role for d in reference] == [
        "canonical_parquet", "canonical_manifest", "split_manifest",
        "forecast_policy_manifest",
    ]
    for driver in DRIVERS:
        assert artifact.provenance[driver].sources == reference


@pytest.mark.parametrize("field,value", [
    ("code_revision", "b" * 40),
    ("method", "persistence"),
    ("model_name", "other_model"),
    ("model_version", "v2"),
    ("seed", 0),
    ("generated_at", "2024-01-05T05:00:00+08:00"),
    ("target_end_exclusive", "2024-01-05T07:00:00+08:00"),
    ("lookback_start", "2024-01-03T04:00:00+08:00"),
])
def test_artifact_rejects_per_series_top_level_mismatch(chain, field, value):
    """B.1/B.2/B.3/B.9：逐序列证据与顶层不一致 → **构造时**拒绝。"""
    from contracts.models import AvailableExogenousForecast

    _, payload = _artifact_payload(chain)
    mutated = _mutate_provenance(payload, "ghi_w_per_m2", field, value)
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**mutated)


@pytest.mark.parametrize("mutation", [
    "drop_canonical_parquet",
    "drop_policy",
    "duplicate_split_manifest",
    "extra_role",
    "reorder",
    "wrong_path",
    "wrong_hash",
])
def test_artifact_rejects_broken_source_role_closure(chain, mutation):
    """B.4–B.8：角色缺失、重复、额外、乱序，以及 path/hash 不一致全部拒绝。"""
    from contracts.models import AvailableExogenousForecast

    _, payload = _artifact_payload(chain)
    sources = [dict(d) for d in payload["provenance"]["wind_speed_10m_mps"]["sources"]]

    if mutation == "drop_canonical_parquet":
        sources = [d for d in sources if d["role"] != "canonical_parquet"]
    elif mutation == "drop_policy":
        sources = [d for d in sources if d["role"] != "forecast_policy_manifest"]
    elif mutation == "duplicate_split_manifest":
        sources = [sources[0], sources[1], sources[2], sources[2]]
    elif mutation == "extra_role":
        sources = sources + [{"role": "shadow", "logical_path": "x", "sha256": "c" * 64}]
    elif mutation == "reorder":
        sources = [sources[1], sources[0], sources[2], sources[3]]
    elif mutation == "wrong_path":
        sources[2] = {**sources[2], "logical_path": "<external>/other.json"}
    elif mutation == "wrong_hash":
        sources[3] = {**sources[3], "sha256": "d" * 64}

    mutated = _mutate_provenance(payload, "wind_speed_10m_mps", "sources", sources)
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**mutated)


def test_artifact_digest_validates_itself_at_construction():
    """B.10：ArtifactDigest 在构造时自校验，不等嵌套进 provenance。"""
    from contracts.models import ArtifactDigest

    with pytest.raises(ValueError):
        ArtifactDigest(role="", logical_path="p", sha256="a" * 64)
    with pytest.raises(ValueError):
        ArtifactDigest(role="r", logical_path="", sha256="a" * 64)
    with pytest.raises(ValueError):
        ArtifactDigest(role="r", logical_path="/Users/someone/x", sha256="a" * 64)
    for bad in ("abc", "A" * 64, "z" * 64, 5, None):
        with pytest.raises(ValueError):
            ArtifactDigest(role="r", logical_path="p", sha256=bad)


def test_validate_available_forecast_reuses_the_construction_rules(chain):
    """校验入口的防御性复验直接复用构造规则，不维护更弱的重复规则。"""
    from contracts.validators import validate_available_forecast

    artifact = forecast_artifact(chain, cutoff=4)
    validate_available_forecast(artifact)  # 合法 → 通过

    # `model_copy` **不重跑**构造校验，因此可以造出一个不合规对象
    weak = artifact.model_copy(update={"period_steps": 47})
    with pytest.raises(ValueError):
        validate_available_forecast(weak)
    weak = artifact.model_copy(update={"code_revision": "not-a-git-sha"})
    with pytest.raises(ValueError):
        validate_available_forecast(weak)


# --- 16. M1.3e-R2：冻结 policy 规则与时间轴 -----------------------------------

@pytest.mark.parametrize("field,value", [
    ("split", "holdout"),
    ("split", ""),
    ("frequency", "1h"),
    ("frequency", "60min"),
    ("method", "persistence"),
    ("method", "seasonal_naive"),
    ("period_steps", 47),
    ("period_steps", 49),
])
def test_artifact_rejects_non_frozen_policy_values(chain, field, value):
    """C：合法但**错误**的取值同样必须拒绝（不是只查非空/正数）。"""
    from contracts.models import AvailableExogenousForecast

    _, payload = _artifact_payload(chain)
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**{**payload, field: value})


@pytest.mark.parametrize("mutation", [
    "empty",
    "duplicate",
    "out_of_order",
    "offset_17min",
    "gap_60min",
    "first_not_generated_at",
])
def test_artifact_rejects_bad_target_timeline(chain, mutation):
    """D：重复、乱序、非 30 分钟网格、缺口、首项偏离全部拒绝，且不泄漏内建异常。"""
    from contracts.models import AvailableExogenousForecast

    _, payload = _artifact_payload(chain)
    stamps = list(payload["target_timestamps"])
    if mutation == "empty":
        stamps = []
    elif mutation == "duplicate":
        stamps[1] = stamps[0]
    elif mutation == "out_of_order":
        stamps[0], stamps[1] = stamps[1], stamps[0]
    elif mutation == "offset_17min":
        stamps[1] = "2024-01-05T04:17:00+08:00"
    elif mutation == "gap_60min":
        stamps[1] = "2024-01-05T05:00:00+08:00"
    elif mutation == "first_not_generated_at":
        stamps[0] = "2024-01-05T04:30:00+08:00"
    mutated = {**payload, "target_timestamps": stamps}
    with pytest.raises((ValueError, TypeError, KeyError)):
        AvailableExogenousForecast(**mutated)


@pytest.mark.parametrize("bad", (None, 5, 1, []))
def test_target_timeline_rejections_do_not_leak_builtin_errors(chain, bad):
    from contracts.models import AvailableExogenousForecast

    _, payload = _artifact_payload(chain)
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**{**payload, "target_timestamps": bad})


def test_artifact_lookback_and_target_end_are_locked(chain):
    """D.5/D.6：target 末端 = 最后 target + 30min；lookback 起点 = generated_at − 24h。"""
    artifact = forecast_artifact(chain, cutoff=4)
    for driver in DRIVERS:
        entry = artifact.provenance[driver]
        assert entry.target_end_exclusive == "2024-01-05T06:00:00+08:00"
        assert entry.lookback_start == "2024-01-04T04:00:00+08:00"
        assert entry.target_start == artifact.generated_at


# --- 17. M1.3e-R2：revision 覆盖真实实现面 ------------------------------------

R1_POLICY_REVISION = "05ad5521a14e9b6e04bcdc1f00655f9a135b1574"


def test_forecast_source_paths_cover_the_contract_semantics():
    module = importlib.import_module(FORECAST_MODULE)
    required = {
        "contracts/__init__.py",
        "contracts/models.py",
        "contracts/validators.py",
        "scenario/forecast.py",
        "scripts/materialize_singapore_forecast_policy.py",
    }
    assert required <= set(module.FORECAST_SOURCE_PATHS)
    assert module.FORECAST_SOURCE_PATHS == tuple(
        importlib.import_module(POLICY_MATERIALIZER).FORECAST_SOURCE_PATHS
    )


def test_revision_and_dirty_check_use_the_same_paths():
    module = _policy_module()
    assert module.FORECAST_SOURCE_PATHS == importlib.import_module(
        FORECAST_MODULE).FORECAST_SOURCE_PATHS
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *module.FORECAST_SOURCE_PATHS],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    assert module._generator_is_dirty() is bool(status.strip())


def test_every_source_path_is_tracked_by_git():
    module = importlib.import_module(FORECAST_MODULE)
    for path in module.FORECAST_SOURCE_PATHS:
        assert (REPO_ROOT / path).exists(), path
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPO_ROOT, capture_output=True, text=True).returncode
        assert tracked == 0, f"{path} 未被 Git 跟踪，revision 解析将不可靠"


def test_r1_policy_revision_is_no_longer_accepted(chain):
    """E.2：实现 revision 变了之后，旧的（R1）policy revision 必须被拒绝。"""
    module = importlib.import_module(FORECAST_MODULE)
    assert module.provider_code_revision() != R1_POLICY_REVISION
    payload = _policy_json(chain["policy_manifest"])
    payload["materializer_revision"] = R1_POLICY_REVISION
    _rewrite_json(chain["policy_manifest"], payload)
    with pytest.raises(ValueError):
        forecast_artifact(chain)


def test_r2_policy_manifest_revision_is_the_current_implementation(chain):
    module = importlib.import_module(FORECAST_MODULE)
    artifact = forecast_artifact(chain, cutoff=4)
    assert artifact.code_revision == module.provider_code_revision()
    assert tuple(d.sha256 for d in artifact.provenance[DRIVERS[0]].sources) == (
        artifact.canonical_parquet_sha256,
        artifact.canonical_manifest_sha256,
        artifact.split_manifest_sha256,
        artifact.policy_manifest_sha256,
    )


def test_scenario_split_names_match_the_frozen_split_module():
    from contracts.models import SCENARIO_SPLIT_NAMES
    from scenario.splits import SPLIT_NAMES

    assert SCENARIO_SPLIT_NAMES == SPLIT_NAMES


# --- 18. M1.3e-R3：严格外部类型与规范逻辑路径 ---------------------------------

# 每个整数字段配一个**语义上仍然合法**的错误类型值：拒绝必须来自**类型**，
# 而不是来自后续的语义/长度/范围检查（否则证明不了 coercion 前的严格验型）。
_ARTIFACT_INT_VALUES = {
    "origin": 200,
    "global_origin": 200,
    "forecast_cutoff": 4,
    "period_steps": 48,
}
_ARTIFACT_INT_CASES = [
    (field, bad)
    for field, good in _ARTIFACT_INT_VALUES.items()
    for bad in (True, str(good), float(good), None, [good], {"v": good})
]

SERIES_BAD_ELEMENTS = (
    "1.25", True, None, [1.0], {"v": 1.0},
    float("nan"), float("inf"), float("-inf"),
)
BAD_LOGICAL_PATHS = (
    ".", "..", "../escape", "a/../b", "./file", "a//b", "a/b/",
    "/abs/path", "a\\b", " lead", "trail ", "in ner", "", "   ",
)


def _artifact_payload_for(chain, **over):
    return {**forecast_artifact(chain, cutoff=4).model_dump(), **over}


@pytest.mark.parametrize("field,bad", _ARTIFACT_INT_CASES)
def test_artifact_integer_fields_are_strictly_typed_before_coercion(chain, field, bad):
    """18.1：整数语义字段必须在 coercion 前验型（bool/float/字符串/None/容器）。"""
    from contracts.models import AvailableExogenousForecast

    payload = _artifact_payload_for(chain)
    assert payload[field] == _ARTIFACT_INT_VALUES[field]
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**{**payload, field: bad})


def test_artifact_integer_fields_accept_only_plain_ints(chain):
    from contracts.models import AvailableExogenousForecast

    payload = _artifact_payload_for(chain)
    for field, good in _ARTIFACT_INT_VALUES.items():
        rebuilt = AvailableExogenousForecast(**{**payload, field: good})
        assert getattr(rebuilt, field) == good
        assert type(getattr(rebuilt, field)) is int


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("bad", SERIES_BAD_ELEMENTS)
def test_artifact_series_reject_bad_element_types(chain, driver, bad):
    """18.2：元素只接受 int/float（bool 不算数值）；字符串/None/容器/NaN/±Inf 全拒绝。"""
    from contracts.models import AvailableExogenousForecast

    payload = _artifact_payload_for(chain)
    series = {d: list(payload["series"][d]) for d in DRIVERS}
    values = list(series[driver])
    values[0] = bad
    series[driver] = values
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**{**payload, "series": series})


@pytest.mark.parametrize("bad", (None, 1.0, "abc", {"a": 1}, {1.0, 2.0}))
def test_artifact_series_container_must_be_list_or_tuple(chain, bad):
    from contracts.models import AvailableExogenousForecast

    payload = _artifact_payload_for(chain)
    series = {d: list(payload["series"][d]) for d in DRIVERS}
    series["system_load_mw"] = bad
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**{**payload, "series": series})


def test_artifact_series_normalises_lists_to_tuples_and_ints_to_floats(chain):
    from contracts.models import AvailableExogenousForecast

    payload = _artifact_payload_for(chain)
    series = {d: list(payload["series"][d]) for d in DRIVERS}
    series["temperature_deg_c"] = [28, 29, 30, 31]  # int 元素可规范化为 float
    rebuilt = AvailableExogenousForecast(**{**payload, "series": series})
    for driver in DRIVERS:
        assert isinstance(rebuilt.series[driver], tuple)
    assert rebuilt.series["temperature_deg_c"] == (28.0, 29.0, 30.0, 31.0)
    assert all(
        type(v) is float for v in rebuilt.series["temperature_deg_c"]
    )


@pytest.mark.parametrize("field", BUNDLE_FORECAST_FIELDS)
@pytest.mark.parametrize("bad", ("1.0", True, None, float("nan"), float("inf")))
def test_bundle_forecasts_apply_the_same_element_rules(field, bad):
    """18.3：ScenarioBundle 七个序列执行与 artifact **同一套**元素规则。"""
    kwargs = make_bundle_kwargs()
    values = list(kwargs[field])
    values[0] = bad
    kwargs[field] = values
    with pytest.raises(ValueError):
        build_bundle(**kwargs)


@pytest.mark.parametrize("field", BUNDLE_FORECAST_FIELDS)
def test_bundle_forecasts_normalise_lists_to_tuples(field):
    bundle = build_bundle()
    assert isinstance(getattr(bundle, field), tuple)
    with pytest.raises(TypeError):
        getattr(bundle, field)[0] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("bad", BAD_LOGICAL_PATHS)
def test_digest_logical_path_must_be_a_canonical_posix_path(bad):
    """18.4：ArtifactDigest.logical_path 必须是规范 POSIX 逻辑路径。"""
    from contracts.models import ArtifactDigest

    with pytest.raises(ValueError):
        ArtifactDigest(role="r", logical_path=bad, sha256="a" * 64)


@pytest.mark.parametrize("bad", BAD_LOGICAL_PATHS)
@pytest.mark.parametrize("field", (
    "canonical_parquet_path", "canonical_manifest_path",
    "split_manifest_path", "policy_manifest_path",
))
def test_artifact_top_level_paths_must_be_canonical_posix(chain, field, bad):
    from contracts.models import AvailableExogenousForecast

    payload = _artifact_payload_for(chain)
    with pytest.raises(ValueError):
        AvailableExogenousForecast(**{**payload, field: bad})


@pytest.mark.parametrize("good", (
    "<external>/half_hour.parquet",
    "data/processed/singapore_2024/half_hour.parquet",
    "a",
    "a/b/c.json",
))
def test_canonical_logical_paths_are_accepted(good):
    from contracts.models import ArtifactDigest

    assert ArtifactDigest(role="r", logical_path=good, sha256="a" * 64).logical_path == good


def test_provider_paths_are_canonical_and_closed(chain):
    """provider 真实产出的四条 path 必须是规范路径，且与 digest 逐项恒等。"""
    artifact = forecast_artifact(chain, cutoff=4)
    for driver in DRIVERS:
        paths = tuple(d.logical_path for d in artifact.provenance[driver].sources)
        assert paths == (
            artifact.canonical_parquet_path,
            artifact.canonical_manifest_path,
            artifact.split_manifest_path,
            artifact.policy_manifest_path,
        )
        for path in paths:
            assert path and not path.startswith("/") and "\\" not in path
            assert not any(seg in (".", "..", "") for seg in path.split("/"))


@pytest.mark.parametrize("update", [
    {"origin": "200"},
    {"global_origin": "200"},
    {"forecast_cutoff": "4"},
    {"period_steps": "48"},
    {"canonical_parquet_path": "../escape"},
    {"split_manifest_path": "a/../b"},
    {"policy_manifest_path": "./file"},
])
def test_validate_available_forecast_rejects_coerced_copy_updates(chain, update):
    """18.5：`model_copy(update=...)` 造出的对象必须被复验再次拒绝。"""
    from contracts.validators import validate_available_forecast

    artifact = forecast_artifact(chain, cutoff=4)
    with pytest.raises(ValueError):
        validate_available_forecast(artifact.model_copy(update=update))


def test_validate_available_forecast_rejects_bool_series_copy(chain):
    from contracts.validators import validate_available_forecast

    artifact = forecast_artifact(chain, cutoff=4)
    for bad_values in ((True, True, True, True), ("1.0", "2.0", "3.0", "4.0")):
        bad_series = artifact.series.model_copy(
            update={"price_sgd_per_kwh": bad_values})
        with pytest.raises(ValueError):
            validate_available_forecast(artifact.model_copy(update={"series": bad_series}))


def test_validate_available_forecast_accepts_a_legitimate_artifact(chain):
    from contracts.validators import validate_available_forecast

    validate_available_forecast(forecast_artifact(chain, cutoff=4))


def test_r3_type_strictness_does_not_change_normal_artifact(chain):
    """数值与语义不变：正常 artifact 的 series / hash 与 R2 完全一致。"""
    first = forecast_artifact(chain, cutoff=4)
    second = forecast_artifact(chain, cutoff=4)
    assert first.prediction_hash() == second.prediction_hash()
    assert first.content_hash() == second.content_hash()
    frame = pd.read_parquet(chain["parquet"])
    module = importlib.import_module(FORECAST_MODULE)
    for driver in DRIVERS:
        expected = module.seasonal_naive_forecast(
            frame[driver].to_numpy(), origin=200, forecast_cutoff=4)
        assert tuple(first.series[driver]) == expected
        assert all(type(v) is float for v in first.series[driver])
