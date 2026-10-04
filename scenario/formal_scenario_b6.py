"""M1.3f-e-b2-a：绑定 **B6 policy + exogenous v3** 的 formal scenario **候选**。

本模块提供一条**独立**于 `scenario/formal_scenario.py` 的构造入口
`build_formal_scenario_b6(...)`，并配套 **policy-v3** 的 canonical 信任链。
**本卡只是 candidate**：不切换现有 formal 入口、不生成 `refs_v4`、
不建 `formal_splits_v5`、不改 readiness。

## 与既有 formal 内核的关系

| 序列 | 来源 |
|---|---|
| price / load / temperature | **复用** M1.3e 因果 provider（规则来自 policy-v2） |
| pv / wind | **复用** `scenario/exogenous_drivers.py` 的同一物理函数 |
| carbon | 获批常数 `0.402`（`human_approved_external_low_resolution`） |
| arrival | **B6 期望值** `rate_template[slot] × 31.994` |

`scenario/formal_scenario.py` **一行未改**——本模块只**导入**它的常量。

## arrival 只取期望，不取 realization

arrival forecast **只**由三者计算：已验证的 **B6 policy**、
已验证 **v3 bundle** 中的冻结 template（其 shape 源按冻结 SHA-256 绑定 v2 manifest）、
以及 target 的**已知日历 slot**。

**绝不**读取 v3 parquet 的 `arrival` 列（那是 Poisson realization，均值 32.0204，
**不是** forecast）。因此 arrival forecast 的均值恰为 **31.994**。

## 信任链

policy-v3 → policy-v2（仅 seasonal 规则） → canonical parquet/manifest/split，
外加 B6 policy 与 v3 三项资产，逐层校验；任一不符即 **fail closed**，**无 fallback**。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.models import (
    ArtifactDigest,
    ForecastSeriesProvenance,
    ScenarioBundle,
    ScenarioForecastProvenance,
)
from scenario.exogenous_drivers import (
    ARRIVAL_TEMPLATE_SHAPE,
    CARBON_KG_PER_KWH,
    arrival_template_slot,
)
from scenario.forecast import (
    FORECAST_PERIOD_STEPS,
    FORECAST_SOURCE_PATHS,
    build_available_exogenous_forecast,
)
from scenario.formal_scenario import (
    FORMAL_SOURCE_KINDS,
    GHI_COLUMN,
    PV_TEMPERATURE_COLUMN,
    WIND_SPEED_COLUMN,
    pv_forecast,
    wind_forecast,
)
from scenario.splits import (
    SplitError,
    SplitName,
    _require_canonical_utc,
    _require_dict,
    _require_exact_keys,
    _require_git_sha40,
    logical_repo_path,
    validate_forecast_origin,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

STEP_MINUTES = 30

POLICY_V3_SCHEMA = "m1.3feb2a-singapore-2024-forecast-policy-v3"
CANONICAL_POLICY_V3_NAME = "singapore_2024_forecast_policy_v3.json"

POLICY_V2_LOGICAL = "data/manifest/singapore_2024_forecast_policy_v2.json"
B6_POLICY_LOGICAL = "data/manifest/m13f_arrival_intensity_policy_v1.json"
EXOGENOUS_V3_MANIFEST_LOGICAL = "data/manifest/singapore_2024_exogenous_v3.json"
EXOGENOUS_V3_SOURCE_LOGICAL = "data/manifest/m13f_materialization_sources_v4.json"
EXOGENOUS_V3_PARQUET_LOGICAL = (
    "data/processed/singapore_2024/exogenous_drivers_v3.parquet"
)
CANONICAL_PARQUET_LOGICAL = "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST_LOGICAL = "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST_LOGICAL = "data/manifest/singapore_2024_splits.json"

CONTRACT_VERSION = "contract-v9"
FREQUENCY = "30min"
INFORMATION_POLICY = "closed_open_[origin-48, origin)"
TARGET_POLICY = "half_open_[origin, origin+C)"
SEED_POLICY = None

# B6 尺度（**必须**与 policy 的 31.994 恒等）
B6_ARRIVAL_EXPECTED_AMOUNT = 31.994
EMPIRICAL_WORKLOAD_CLAIM = False
ARRIVAL_SOURCE_KIND = "modeled_scenario"

# provenance 的 source role
POLICY_V3_ROLE = "forecast_policy_manifest"
SEASONAL_RULE_SOURCE_ROLE = "seasonal_rule_source_policy"
B6_POLICY_ROLE = "b6_intensity_policy"
EXOGENOUS_V3_MANIFEST_ROLE = "exogenous_drivers_manifest"
EXOGENOUS_V3_SOURCE_ROLE = "exogenous_source_manifest"
EXOGENOUS_V3_PARQUET_ROLE = "exogenous_drivers_parquet"

# 七条序列的 source_kind：**唯一**来源是既有 formal 内核的冻结常量
FORMAL_B6_SOURCE_KINDS: dict[str, str] = dict(FORMAL_SOURCE_KINDS)

# **本模块实际执行文件**（code_revision 与 dirty 检查共用这一组）。
#
# R1：覆盖**全部**直接影响候选语义的文件——`FORECAST_SOURCE_PATHS` 全员
# （含 policy 物化器）、`scenario/splits.py`（origin 门禁与严格链）、
# formal 内核（source_kind 常量与 PV/风函数）、外生驱动与其 B6 变体、
# B6 policy、本模块与 policy-v3 物化器。
B6_FORMAL_SOURCE_PATHS: tuple[str, ...] = (
    *FORECAST_SOURCE_PATHS,
    "scenario/splits.py",
    "scenario/formal_scenario.py",
    "scenario/exogenous_drivers.py",
    "scenario/arrival_intensity_policy.py",
    "scenario/exogenous_drivers_b6.py",
    "scenario/formal_scenario_b6.py",
    "scripts/materialize_formal_forecast_policy_b6.py",
)

POLICY_V3_KEYS: tuple[str, ...] = (
    "schema",
    "contract_version",
    "materializer_revision",
    "frozen_at_utc",
    "canonical_parquet_path",
    "canonical_parquet_sha256",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "seasonal_rule_source_policy_path",
    "seasonal_rule_source_policy_sha256",
    "b6_policy_path",
    "b6_policy_sha256",
    "exogenous_v3_manifest_path",
    "exogenous_v3_manifest_sha256",
    "exogenous_v3_source_manifest_path",
    "exogenous_v3_source_manifest_sha256",
    "exogenous_v3_parquet_path",
    "exogenous_v3_parquet_sha256",
    "frequency",
    "period_steps",
    "information_policy",
    "target_policy",
    "seed_policy",
    "available_drivers",
    "arrival_forecast_rule",
    "arrival_forecast_uses_expected",
    "arrival_expected_amount_work_per_half_hour",
    "arrival_source_kind",
    "empirical_workload_claim",
    "readiness",
    "supersedes",
)
SUPERSEDES_KEYS: tuple[str, ...] = (
    "policy_manifest_path", "policy_manifest_sha256", "policy_manifest_schema",
    "status", "note",
)
READINESS_KEYS: tuple[str, ...] = (
    "available_driver_forecasts_ready",
    "formal_b6_scenario_candidate_ready",
    "formal_scenario_bundle_ready",
    "formal_training_ready",
)

AVAILABLE_DRIVERS: tuple[str, ...] = (
    "price_sgd_per_kwh", "system_load_mw", "temperature_deg_c",
    "wind_speed_10m_mps", "ghi_w_per_m2",
)

ARRIVAL_FORECAST_RULE = (
    "lambda_t = frozen_template[slot] * 31.994 work/half-hour"
    "（B6 expected，**不**读取 Poisson realization）"
)

READINESS: dict[str, bool] = {
    "available_driver_forecasts_ready": True,
    "formal_b6_scenario_candidate_ready": True,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}

MODEL_NAME = "formal_scenario_b6_candidate"
MODEL_VERSION = "v1"


class FormalB6Error(ValueError):
    """formal B6 candidate 的**明确失败**（无 fallback、不静默）。"""


# --- Git ----------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def b6_formal_code_revision() -> str:
    """本候选实现的冻结 revision（由 `B6_FORMAL_SOURCE_PATHS` 解析）。"""
    try:
        from scenario.portable_numeric import verified_recipe_revision

        revision = _git("log", "-1", "--format=%H", "--",
                        *B6_FORMAL_SOURCE_PATHS).strip()
        revision = _require_git_sha40(revision, field="b6 formal code_revision")
        return verified_recipe_revision(REPO_ROOT, B6_FORMAL_SOURCE_PATHS, revision, 'forecast')
    except ValueError as error:
        raise FormalB6Error(f"b6 formal code_revision 无法解析：{error}") from error


def _generator_is_dirty() -> bool:
    status = _git("status", "--porcelain", "--", *B6_FORMAL_SOURCE_PATHS)
    return bool(status.strip())


def _sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise FormalB6Error(f"冻结资产不可读：{path}：{error}") from error


# --- policy-v3 的 canonical 信任链 --------------------------------------------

def _canonical_policy_v3_dir() -> Path:
    """canonical policy-v3 目录的**私有**解析器（测试只能 monkeypatch 它）。"""
    return REPO_ROOT / "data" / "manifest"


def canonical_policy_v3_path() -> Path:
    return _canonical_policy_v3_dir() / CANONICAL_POLICY_V3_NAME


# --- 上游绑定的**私有**解析器 -------------------------------------------------
#
# 全部上游对象都经这一组解析；`build_policy_v3_manifest()` 与
# `load_verified_policy_v3()` **共用**它们，因此「重建 candidate == 文件」
# 是一致性断言，而不是两套规则。测试只能 monkeypatch **私有**解析器
# （公开 API 不暴露任何 path/hash/revision 覆盖入口）。
def _canonical_parquet_path() -> Path:
    return REPO_ROOT / CANONICAL_PARQUET_LOGICAL


def _canonical_manifest_path() -> Path:
    return REPO_ROOT / CANONICAL_MANIFEST_LOGICAL


def _split_manifest_path() -> Path:
    return REPO_ROOT / SPLIT_MANIFEST_LOGICAL


def _seasonal_policy_v2_path() -> Path:
    return REPO_ROOT / POLICY_V2_LOGICAL


def _b6_policy_path() -> Path:
    return REPO_ROOT / B6_POLICY_LOGICAL


def _exogenous_v3_manifest_path() -> Path:
    return REPO_ROOT / EXOGENOUS_V3_MANIFEST_LOGICAL


def _exogenous_v3_source_path() -> Path:
    return REPO_ROOT / EXOGENOUS_V3_SOURCE_LOGICAL


def _exogenous_v3_parquet_path() -> Path:
    return REPO_ROOT / EXOGENOUS_V3_PARQUET_LOGICAL


def _v3_template_source() -> np.ndarray:
    """冻结 48 槽 template 的**私有**来源（生产路径 = 已验证 v3 bundle）。

    测试可在临时链中改写它；生产路径下必须与 `verified_v3_template()` 恒等
    （有专门断言），因此**不是**「mock 最终 bundle」。
    """
    return verified_v3_template()


def _require_canonical_location(path: Path | str | None) -> Path:
    """路径必须**精确等于** canonical policy-v3（在读 JSON 之前）。

    比较**词法绝对路径**（`absolute()` 不做符号链接解析）：副本、别名、symlink
    一律拒绝；**不得** fallback 到 policy-v2。
    """
    given = (Path(path).absolute() if path is not None
             else canonical_policy_v3_path().absolute())
    expected = canonical_policy_v3_path().absolute()
    if given != expected:
        raise FormalB6Error(
            f"policy-v3 路径必须是**唯一 canonical** 文件 "
            f"{logical_repo_path(expected)}；实际 {logical_repo_path(given)}"
            "（不接受副本、别名、symlink；无 fallback）"
        )
    if expected.is_symlink():
        raise FormalB6Error(f"policy-v3 不得是 symlink：{expected}")
    return expected


def _require_nested_keys(payload: dict) -> None:
    """`supersedes` / `readiness` 的**精确键集合**（在读任何业务字段之前）。"""
    try:
        _require_exact_keys(_require_dict(payload.get("supersedes"),
                                          field="supersedes"),
                            field="supersedes", expected=SUPERSEDES_KEYS)
        _require_exact_keys(_require_dict(payload.get("readiness"), field="readiness"),
                            field="readiness", expected=READINESS_KEYS)
    except SplitError as error:
        raise FormalB6Error(f"policy-v3 嵌套键集合不符：{error}") from error


def load_verified_policy_v3(path: Path | str | None = None) -> dict:
    """加载并**严格校验** canonical policy-v3（路径先于 JSON 读取）。

    **R1 修复**：不再「抽查部分字段」。流程是

    1. canonical 位置、普通文件、**非 symlink**；
    2. 顶层与嵌套（`supersedes` / `readiness`）**精确键集合**；
    3. 读取文件中的 `frozen_at_utc`；
    4. `build_policy_v3_manifest(frozen_at_utc=该时间)`——由 **trusted constants +
       live hashes + live revision** 重建**完整** candidate；
    5. 文件 payload 必须**逐字段等于**重建 candidate。

    因此**任何**业务字段（规则文案、`seed_policy`、`supersedes` 全字段、
    readiness、八条绑定、revision……）的伪造都会 fail closed；不存在
    「抽查若干字段后返回 verified」的路径。
    """
    path = _require_canonical_location(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalB6Error(f"policy-v3 不可读：{error}") from error
    payload = _require_dict(payload, field="policy-v3")

    try:
        _require_exact_keys(payload, field="policy-v3", expected=POLICY_V3_KEYS)
    except SplitError as error:
        raise FormalB6Error(f"policy-v3 顶层键集合不符：{error}") from error
    _require_nested_keys(payload)

    if payload["schema"] != POLICY_V3_SCHEMA:
        raise FormalB6Error(
            f"policy-v3 schema 必须是 {POLICY_V3_SCHEMA!r}，实际 {payload['schema']!r}"
        )
    frozen_at_utc = payload["frozen_at_utc"]
    try:
        _require_canonical_utc(frozen_at_utc, field="frozen_at_utc")
    except SplitError as error:
        raise FormalB6Error(f"policy-v3 frozen_at_utc 不符：{error}") from error

    # **重建**并逐字段比对：这是唯一的「verified」判据
    rebuilt = build_policy_v3_manifest(frozen_at_utc=frozen_at_utc)
    if rebuilt != payload:
        differing = sorted(
            key for key in set(rebuilt) | set(payload)
            if rebuilt.get(key) != payload.get(key)
        )
        raise FormalB6Error(
            f"policy-v3 与由 trusted constants + live hashes + live revision "
            f"重建的 candidate 不符；差异字段={differing}"
        )
    return payload


# --- 已验证的 v3 template 与 arrival expected ---------------------------------

def verified_v3_template() -> np.ndarray:
    """来自**已验证 v3 bundle** 的冻结 48 槽 template。

    先跑 v3 bundle 的**完整语义验证**（`load_verified_v3_bundle`），
    再按冻结 SHA-256 读取其 shape 源（v2 manifest）的 template。
    """
    from scenario.exogenous_drivers_b6 import (
        load_v2_arrival_template,
        load_verified_v3_bundle,
    )

    load_verified_v3_bundle()
    template = np.asarray(load_v2_arrival_template(), dtype=float)
    if template.shape != ARRIVAL_TEMPLATE_SHAPE:
        raise FormalB6Error(f"template 形状必须是 {ARRIVAL_TEMPLATE_SHAPE}")
    if not np.isclose(template.mean(), 1.0, rtol=1e-12):
        raise FormalB6Error("冻结 template 的均值必须精确为 1（只贡献 shape）")
    return template


def expected_arrival_for_timestamps(
    timestamps: pd.DatetimeIndex, template: np.ndarray | None = None
) -> np.ndarray:
    """arrival 的**期望值**：`template[slot] × 31.994`（**不**抽样）。"""
    if template is None:
        template = verified_v3_template()
    template = np.asarray(template, dtype=float)
    slots = arrival_template_slot(pd.DatetimeIndex(timestamps))
    return template[slots] * B6_ARRIVAL_EXPECTED_AMOUNT


def target_timestamps_for(
    split: SplitName, *, origin: int, forecast_cutoff: int
) -> tuple[str, ...]:
    """target 的**已知日历时刻** `[origin, origin+C)`（由 canonical 时间轴导出）。"""
    global_origin = validate_forecast_origin(split, origin, forecast_cutoff)
    stamps = pd.DatetimeIndex(
        pd.read_parquet(_canonical_parquet_path())["timestamp"]
    )
    return tuple(
        pd.Timestamp(stamps[i]).isoformat()
        for i in range(global_origin, global_origin + forecast_cutoff)
    )


# --- 组装 ---------------------------------------------------------------------

def _roles(
    *, policy_v3_path: Path, canonical_parquet_path: Path,
    canonical_manifest_path: Path, split_manifest_path: Path,
) -> tuple[ArtifactDigest, ...]:
    entries = (
        ("canonical_parquet", canonical_parquet_path),
        ("canonical_manifest", canonical_manifest_path),
        ("split_manifest", split_manifest_path),
        (POLICY_V3_ROLE, policy_v3_path),
        (SEASONAL_RULE_SOURCE_ROLE, _seasonal_policy_v2_path()),
        (B6_POLICY_ROLE, _b6_policy_path()),
        (EXOGENOUS_V3_MANIFEST_ROLE, _exogenous_v3_manifest_path()),
        (EXOGENOUS_V3_SOURCE_ROLE, _exogenous_v3_source_path()),
        (EXOGENOUS_V3_PARQUET_ROLE, _exogenous_v3_parquet_path()),
    )
    return tuple(
        ArtifactDigest(
            role=role,
            logical_path=logical_repo_path(Path(path)),
            sha256=_sha256_file(Path(path)),
        )
        for role, path in entries
    )


def build_formal_scenario_b6(
    split: SplitName,
    *,
    origin: int,
    forecast_cutoff: int,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    policy_manifest_path: Path | str,
    horizon: int | None = None,
) -> ScenarioBundle:
    """构造绑定 **B6 policy + v3 exogenous** 的 formal 候选 bundle（**不切换**入口）。

    seasonal / PV / 风电 / carbon 的规则与既有 formal 内核**同源**；
    唯一语义差异是 **arrival 改用 B6 期望尺度 31.994**。
    """
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise FormalB6Error(f"origin 必须是整数，实际 {origin!r}")
    if isinstance(forecast_cutoff, bool) or not isinstance(forecast_cutoff, int):
        raise FormalB6Error(f"forecast_cutoff 必须是整数，实际 {forecast_cutoff!r}")

    if _generator_is_dirty():
        raise FormalB6Error(
            "B6 formal 实现有未提交修改：拒绝用旧 code_revision 为未提交代码背书"
        )

    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)
    policy_v3_path = _require_canonical_location(policy_manifest_path)

    policy = load_verified_policy_v3(policy_v3_path)
    # 声明的上游对象必须是**冻结的 canonical 逻辑路径**（其 SHA-256 已由
    # `load_verified_policy_v3` 对着 resolver 解析出的**实际文件**逐条校验）。
    for field, logical in (("canonical_parquet_path", CANONICAL_PARQUET_LOGICAL),
                           ("canonical_manifest_path", CANONICAL_MANIFEST_LOGICAL),
                           ("split_manifest_path", SPLIT_MANIFEST_LOGICAL)):
        if policy[field] != logical:
            raise FormalB6Error(
                f"policy-v3 的 {field} 必须精确等于 {logical!r}，"
                f"实际 {policy[field]!r}"
            )

    global_origin = validate_forecast_origin(split, origin, forecast_cutoff)
    history_start = global_origin - FORECAST_PERIOD_STEPS
    if history_start < 0:
        raise FormalB6Error(
            f"{split} 内 origin={origin} 不足 {FORECAST_PERIOD_STEPS} 步历史；fail closed"
        )

    # 1) 五类 driver forecast：**复用** M1.3e 的因果 provider（规则来自 policy-v2）
    artifact = build_available_exogenous_forecast(
        split,
        origin=origin,
        forecast_cutoff=forecast_cutoff,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
        policy_manifest_path=_seasonal_policy_v2_path(),
    )
    generated_at = artifact.generated_at
    target_timestamps = tuple(artifact.target_timestamps)

    # 2) B6 policy 与 v3 bundle 都必须通过各自的正式 loader
    from scenario.arrival_intensity_policy import load_verified_b6_policy

    b6_policy = load_verified_b6_policy()
    if b6_policy["main_expected_amount_work_per_half_hour"] != \
            B6_ARRIVAL_EXPECTED_AMOUNT:
        raise FormalB6Error("B6 policy 的 expected amount 与冻结 31.994 不符")
    template = _v3_template_source()

    stamps = pd.DatetimeIndex(
        pd.read_parquet(canonical_parquet_path)["timestamp"]
    )

    # 3) 七条序列：全部只用 `[i−48, i)`
    drv = artifact.series.as_dict()
    series: dict[str, tuple[float, ...]] = {
        "price_forecast": drv["price_sgd_per_kwh"],
        "load_forecast": drv["system_load_mw"],
        "temperature_forecast": drv["temperature_deg_c"],
    }
    series["pv_forecast"] = pv_forecast(
        pd.DatetimeIndex(target_timestamps), drv[GHI_COLUMN],
        drv[PV_TEMPERATURE_COLUMN], drv[WIND_SPEED_COLUMN],
    )
    series["wind_forecast"] = wind_forecast(drv[WIND_SPEED_COLUMN])
    series["carbon_forecast"] = (CARBON_KG_PER_KWH,) * forecast_cutoff
    # arrival：**只**由 verified B6 policy + verified v3 template + 已知日历 slot
    series["arrival_forecast"] = tuple(
        float(v) for v in expected_arrival_for_timestamps(
            pd.DatetimeIndex(target_timestamps), template)
    )

    code_revision = b6_formal_code_revision()
    target_end_exclusive = (
        datetime.fromisoformat(target_timestamps[-1]) + timedelta(minutes=STEP_MINUTES)
    ).isoformat()
    lookback_start = pd.Timestamp(stamps[history_start]).isoformat()

    sources = _roles(
        policy_v3_path=policy_v3_path,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    provenance = {
        field: ForecastSeriesProvenance(
            series_name=field,
            source_kind=FORMAL_B6_SOURCE_KINDS[field],
            method=_method_for(field),
            generated_at=generated_at,
            information_cutoff_exclusive=generated_at,
            target_start=generated_at,
            target_end_exclusive=target_end_exclusive,
            lookback_start=lookback_start,
            lookback_end_exclusive=generated_at,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            code_revision=code_revision,
            seed=None,
            sources=sources,
        )
        for field in FORMAL_B6_SOURCE_KINDS
    }

    bundle = ScenarioBundle(
        split=split,
        start=str(origin),
        horizon=forecast_cutoff if horizon is None else horizon,
        forecast_cutoff=forecast_cutoff,
        price_forecast=series["price_forecast"],
        load_forecast=series["load_forecast"],
        pv_forecast=series["pv_forecast"],
        wind_forecast=series["wind_forecast"],
        temperature_forecast=series["temperature_forecast"],
        carbon_forecast=series["carbon_forecast"],
        arrival_forecast=series["arrival_forecast"],
        mode="formal",
        generated_at=generated_at,
        forecast_provenance=ScenarioForecastProvenance(**provenance),
    )
    from contracts.validators import validate_forecast_purpose

    validate_forecast_purpose(bundle, purpose="training")
    return bundle


def _method_for(field: str) -> str:
    if field in ("price_forecast", "load_forecast", "temperature_forecast"):
        return "trailing_seasonal_naive"
    if field == "pv_forecast":
        return "pvlib_v0.15.2_chain_from_causal_driver_forecasts"
    if field == "wind_forecast":
        return "shear_law_and_frozen_power_curve_from_causal_wind_forecast"
    if field == "carbon_forecast":
        return "annual_constant_from_verified_v2_manifest"
    return "b6_expected_rate_template_times_declared_capacity"


def build_policy_v3_manifest(*, frozen_at_utc: str) -> dict:
    """构造 candidate policy-v3 manifest（只读上游，逐项绑定 live hash）。"""
    _require_canonical_utc(frozen_at_utc, field="frozen_at_utc")
    payload = {
        "schema": POLICY_V3_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "materializer_revision": b6_formal_code_revision(),
        "frozen_at_utc": frozen_at_utc,
        "canonical_parquet_path": CANONICAL_PARQUET_LOGICAL,
        "canonical_parquet_sha256": _sha256_file(_canonical_parquet_path()),
        "canonical_manifest_path": CANONICAL_MANIFEST_LOGICAL,
        "canonical_manifest_sha256": _sha256_file(_canonical_manifest_path()),
        "split_manifest_path": SPLIT_MANIFEST_LOGICAL,
        "split_manifest_sha256": _sha256_file(_split_manifest_path()),
        "seasonal_rule_source_policy_path": POLICY_V2_LOGICAL,
        "seasonal_rule_source_policy_sha256": _sha256_file(_seasonal_policy_v2_path()),
        "b6_policy_path": B6_POLICY_LOGICAL,
        "b6_policy_sha256": _sha256_file(_b6_policy_path()),
        "exogenous_v3_manifest_path": EXOGENOUS_V3_MANIFEST_LOGICAL,
        "exogenous_v3_manifest_sha256": _sha256_file(_exogenous_v3_manifest_path()),
        "exogenous_v3_source_manifest_path": EXOGENOUS_V3_SOURCE_LOGICAL,
        "exogenous_v3_source_manifest_sha256": _sha256_file(_exogenous_v3_source_path()),
        "exogenous_v3_parquet_path": EXOGENOUS_V3_PARQUET_LOGICAL,
        "exogenous_v3_parquet_sha256": _sha256_file(_exogenous_v3_parquet_path()),
        "frequency": FREQUENCY,
        "period_steps": FORECAST_PERIOD_STEPS,
        "information_policy": INFORMATION_POLICY,
        "target_policy": TARGET_POLICY,
        "seed_policy": SEED_POLICY,
        "available_drivers": list(AVAILABLE_DRIVERS),
        "arrival_forecast_rule": ARRIVAL_FORECAST_RULE,
        "arrival_forecast_uses_expected": True,
        "arrival_expected_amount_work_per_half_hour": B6_ARRIVAL_EXPECTED_AMOUNT,
        "arrival_source_kind": ARRIVAL_SOURCE_KIND,
        "empirical_workload_claim": EMPIRICAL_WORKLOAD_CLAIM,
        "readiness": dict(READINESS),
        "supersedes": {
            "policy_manifest_path": POLICY_V2_LOGICAL,
            "policy_manifest_sha256": _sha256_file(_seasonal_policy_v2_path()),
            "policy_manifest_schema": "m1.3g0-singapore-2024-forecast-policy-v2",
            "status": "registered_not_replaced",
            "note": (
                "policy-v3 **只登记** policy-v2 并复用其五类 seasonal driver 规则；"
                "**不覆盖**、**不迁移**；现有 formal 入口在本卡后**仍**使用 policy-v2"
            ),
        },
    }
    _require_exact_keys(payload, field="policy-v3", expected=POLICY_V3_KEYS)
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise FormalB6Error("policy-v3 含绝对路径")
    return payload


__all__ = [
    "FormalB6Error",
    "POLICY_V3_SCHEMA",
    "POLICY_V3_ROLE",
    "FORMAL_B6_SOURCE_KINDS",
    "B6_FORMAL_SOURCE_PATHS",
    "B6_ARRIVAL_EXPECTED_AMOUNT",
    "b6_formal_code_revision",
    "canonical_policy_v3_path",
    "load_verified_policy_v3",
    "verified_v3_template",
    "expected_arrival_for_timestamps",
    "target_timestamps_for",
    "build_formal_scenario_b6",
    "build_policy_v3_manifest",
]
