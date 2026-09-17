"""M1.3e：**因果** forecast provider —— 只依赖 origin 之前的历史。

本模块为五个 canonical truth driver 生成 forecast：

    price_sgd_per_kwh、system_load_mw、temperature_deg_c、
    wind_speed_10m_mps、ghi_w_per_m2

**冻结方法**：trailing seasonal-naive，周期**固定** `FORECAST_PERIOD_STEPS = 48`
（半小时 × 48 = 一个物理日周期，**不从 validation/test 选择**）。

对全局 origin = i：

- 历史模板**只允许**读取 `[i-48, i)`；**绝不**读取 i 或 i 之后的 truth；
- forecast 第 k 项取模板的 `k mod 48` 项（模板按**时间正序**，等价
  `forecast[k] = y(i + k − 48)`）；
- `generated_at = information_cutoff_exclusive = lookback_end_exclusive = origin`；
- target = `[origin, origin+C)`，`C` 必须通过 M1.3d `validate_forecast_origin`；
- 历史不足 48 步（如 train 内 `origin < 48`）**fail closed**，不回填、不跨年环绕；
- 不使用任何随机数（`seed` 明确为 `null`）；不引入任何拟合。

**M1.3e-R1：完整信任链**。`build_available_exogenous_forecast()` 逐层校验

```text
policy manifest → split manifest → canonical manifest → canonical parquet
```

- policy manifest 的 schema/契约版本/method/period/drivers/readiness/路径/三个 hash/
  revision 全部校验；声明的路径必须与调用者实际提供的 logical repo path 一致；
- split manifest 走 **M1.3d 的完整严格校验**（`load_truth_split`），不另写宽松校验器；
- canonical manifest **自身字节**的 SHA-256 必须与 policy 声明一致；
- `code_revision` **没有**公开参数：只能由内部 Git resolver 得到，并与 policy 的
  `materializer_revision` 恒等。

本模块**不**生成 PV / 风电发电量 / 碳强度 / arrival 的 forecast，
也**不**构造正式 `ScenarioBundle` —— 它只产出严格冻结的
`contracts.AvailableExogenousForecast` artifact。

**M1.3g-0（contract-v9）**：policy schema 升到 **v2**，默认路径为
`DEFAULT_POLICY_MANIFEST_PATH`；v2 manifest 的 `supersedes` 逐字段登记被取代的
**contract-v8** v1 产物（其文件**字节不变**）。任何 `contract-v8` 的 policy
（含 v1）在本模块一律 **fail closed**，不做迁移或静默升级。
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import CONTRACT_VERSION_ID
from contracts.models import (
    AVAILABLE_DRIVER_SERIES,
    AVAILABLE_FORECAST_SOURCE_KIND,
    AVAILABLE_SOURCE_ROLES,
    ArtifactDigest,
    AvailableDriverProvenance,
    AvailableExogenousForecast,
    AvailableSeries,
    ForecastSeriesProvenance,
)
from scenario.splits import (
    FREQUENCY,
    SplitName,
    _require_canonical_utc,
    _require_dict,
    _require_exact_keys,
    _require_git_sha40,
    _require_hex64,
    _require_int,
    _require_str_list,
    load_truth_split,
    logical_repo_path,
    validate_canonical_timeline,
    validate_forecast_origin,
)

FORECAST_PERIOD_STEPS = 48
# 向后兼容的短别名（同一冻结值，**不是**第二个版本源）。
PERIOD_STEPS = FORECAST_PERIOD_STEPS
METHOD = "trailing_seasonal_naive"
MODEL_NAME = "trailing_seasonal_naive"
MODEL_VERSION = "v1"
SOURCE_KIND = AVAILABLE_FORECAST_SOURCE_KIND

AVAILABLE_DRIVERS: tuple[str, ...] = AVAILABLE_DRIVER_SERIES
# 这四个字段本卡**不**生成 forecast（不得用全零或默认曲线补齐）
UNAVAILABLE_NOT_MATERIALIZED: tuple[str, ...] = (
    "local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival",
)

STEP_MINUTES = 30

# provider、policy 物化器**与 artifact 契约语义**的实现文件：revision 由这一组
# 路径解析，因此 provider 解析出的 revision 与 policy 的 `materializer_revision`
# 恒等；**M1.3e-R2 起覆盖 `contracts/`**——否则改了契约语义而 policy revision
# 不变，等于用旧 revision 为新的契约语义背书。
FORECAST_SOURCE_PATHS: tuple[str, ...] = (
    "contracts/__init__.py",
    "contracts/models.py",
    "contracts/validators.py",
    "scenario/forecast.py",
    "scripts/materialize_singapore_forecast_policy.py",
)
REPO_ROOT = Path(__file__).resolve().parent.parent

# --- forecast policy manifest 的冻结 schema（provider 与物化器共用） ---------
#
# **M1.3g-0（contract-v9）**：schema 由 `…-v1` 升到 `…-v2`，并新增 `supersedes`
# 键记录被取代的 v1 产物。v1 文件**保持字节不变**，是 contract-v8 的历史证据；
# provider 对 contract-v8 的 policy 一律 fail closed（见 §contract_version 校验）。

POLICY_SCHEMA = "m1.3g0-singapore-2024-forecast-policy-v2"
# 正式入口的**默认** policy 路径（provider 与物化器共用同一常量，避免两处漂移）
DEFAULT_POLICY_MANIFEST_PATH = (
    "data/manifest/singapore_2024_forecast_policy_v2.json"
)
# 被本 schema 取代的 contract-v8 政策产物（**不得**删除、覆盖或改写）
SUPERSEDES_POLICY_V1: dict[str, Any] = {
    "policy_manifest_path": "data/manifest/singapore_2024_forecast_policy.json",
    "policy_manifest_sha256": (
        "0bf31f80ff0c9acd67dad5082e08eee7364397a2dbfa429e4a34c3d87a702923"
    ),
    "policy_manifest_contract_version": "contract-v8",
    "superseded_by_contract_version": CONTRACT_VERSION_ID,
    "reason": (
        "contract-v9 新增 source_kind human_approved_external_low_resolution；"
        "v1 是 contract-v8 的历史产物，保留不可覆盖，仅登记取代关系"
    ),
}
SUPERSEDES_POLICY_V1_KEYS: tuple[str, ...] = tuple(SUPERSEDES_POLICY_V1)
INFORMATION_POLICY = "closed_open_[origin-48, origin)"
TARGET_POLICY = "half_open_[origin, origin+C)"
SEED_POLICY = None
POLICY_READINESS: dict[str, bool] = {
    "available_driver_forecasts_ready": True,
    "complete_scenario_forecasts_ready": False,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}
POLICY_MANIFEST_KEYS: tuple[str, ...] = (
    "schema",
    "contract_version",
    "canonical_parquet_path",
    "canonical_parquet_sha256",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "materializer_revision",
    "available_drivers",
    "method",
    "period_steps",
    "frequency",
    "information_policy",
    "target_policy",
    "seed_policy",
    "unavailable_not_materialized",
    "readiness",
    "frozen_at_utc",
    # M1.3g-0：v2 schema 新增；被取代的 v1 产物的 path / sha256 / 契约版本
    "supersedes",
)


class ForecastError(ValueError):
    """因果 forecast 的**明确失败**。"""


# --- 纯计算：trailing seasonal-naive ----------------------------------------

def seasonal_naive_forecast(
    series: Sequence[float],
    *,
    origin: int,
    forecast_cutoff: int,
    period_steps: int = FORECAST_PERIOD_STEPS,
) -> tuple[float, ...]:
    """**纯函数**：trailing seasonal-naive，`forecast[k] = y(origin + k − period)`。

    这是本卡冻结规则的**唯一**实现，`build_available_exogenous_forecast` 也调用它。
    leakage 回归因此可以直接在这一层做因果性断言（不依赖冻结资产的字节）。

    - 只读取 `[origin - period, origin)`，**绝不**读取 `origin` 或之后的值；
    - 第 k 项取模板的 `k mod period` 项（模板按时间正序）；
    - `origin < period` fail closed（不回填、不环绕）；
    - 历史窗口含非有限值即拒绝。
    """
    for field, value in (("origin", origin), ("forecast_cutoff", forecast_cutoff),
                         ("period_steps", period_steps)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ForecastError(f"{field} 必须是整数，实际 {value!r}")
    if forecast_cutoff < 1:
        raise ForecastError(f"forecast_cutoff 必须是严格正整数，实际 {forecast_cutoff!r}")
    if period_steps < 1:
        raise ForecastError(f"period_steps 必须是严格正整数，实际 {period_steps!r}")
    if origin < 0:
        raise ForecastError(f"origin 必须是非负整数，实际 {origin!r}")

    history_start = origin - period_steps
    if history_start < 0:
        raise ForecastError(
            f"origin={origin} 不足 {period_steps} 步历史；fail closed，不回填、不跨年环绕"
        )
    values = [float(x) for x in series[history_start:origin]]
    if len(values) != period_steps:
        raise ForecastError(
            f"历史模板长度必须是 {period_steps}，实际 {len(values)}"
        )
    for entry in values:
        if not math.isfinite(entry):
            raise ForecastError(f"历史模板含非有限值：{entry!r}")
    return tuple(values[k % period_steps] for k in range(forecast_cutoff))


# --- Git revision ------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def provider_code_revision() -> str:
    """provider/materializer 实现的 Git revision（40 位小写 SHA）。

    **没有**调用者入口：revision 只能由 Git 解析，且与 policy manifest 的
    `materializer_revision` 使用**同一组** `FORECAST_SOURCE_PATHS`，因此两者恒等。
    """
    revision = _git(
        "log", "-1", "--format=%H", "--", *FORECAST_SOURCE_PATHS
    ).strip()
    return _require_git_sha40(revision, field="provider code_revision")


# --- policy manifest ---------------------------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_forecast_policy_manifest(
    policy_manifest_path: Path | str,
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    expected_revision: str,
    expected_sources: dict[str, str] | None = None,
) -> dict:
    """读取并**严格**校验 forecast policy manifest。

    校验 schema/契约版本/method/period/frequency/drivers/两条 policy/seed/
    unavailable/readiness/frozen_at_utc/路径可移植性，并逐项对齐：
    - 声明的三条 logical repo path 必须与调用者实际提供的路径一致；
    - 声明的三个 SHA-256 必须与实际文件**字节**一致（`expected_sources` 已提供时
      直接比对，避免重复读盘）；
    - `materializer_revision` 必须等于 `expected_revision`（内部 Git resolver）。
    """
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    try:
        manifest = json.loads(Path(policy_manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForecastError(f"forecast policy manifest 不可读：{error}") from error
    manifest = _require_dict(manifest, field="forecast policy manifest")
    _require_exact_keys(
        manifest, field="forecast policy manifest", expected=POLICY_MANIFEST_KEYS
    )

    if manifest.get("schema") != POLICY_SCHEMA:
        raise ForecastError(
            f"policy manifest schema 必须是 {POLICY_SCHEMA!r}，"
            f"实际 {manifest.get('schema')!r}"
        )
    if manifest.get("contract_version") != CONTRACT_VERSION_ID:
        raise ForecastError(
            f"policy manifest contract_version 必须是 {CONTRACT_VERSION_ID!r}，"
            f"实际 {manifest.get('contract_version')!r}"
        )
    if manifest.get("method") != METHOD:
        raise ForecastError(
            f"policy manifest method 必须是 {METHOD!r}，实际 {manifest.get('method')!r}"
        )
    if _require_int(manifest.get("period_steps"), field="period_steps") != (
        FORECAST_PERIOD_STEPS
    ):
        raise ForecastError(
            f"policy manifest period_steps 必须是 {FORECAST_PERIOD_STEPS}"
        )
    if manifest.get("frequency") != FREQUENCY:
        raise ForecastError(
            f"policy manifest frequency 必须是 {FREQUENCY!r}，"
            f"实际 {manifest.get('frequency')!r}"
        )
    _require_str_list(
        manifest.get("available_drivers"), field="available_drivers",
        expected=AVAILABLE_DRIVERS,
    )
    for field, expected in (("information_policy", INFORMATION_POLICY),
                            ("target_policy", TARGET_POLICY)):
        if manifest.get(field) != expected:
            raise ForecastError(
                f"policy manifest {field} 必须是 {expected!r}，实际 {manifest.get(field)!r}"
            )
    if manifest.get("seed_policy") is not SEED_POLICY:
        raise ForecastError(
            f"policy manifest seed_policy 必须是 {SEED_POLICY!r}，"
            f"实际 {manifest.get('seed_policy')!r}"
        )
    _require_str_list(
        manifest.get("unavailable_not_materialized"),
        field="unavailable_not_materialized", expected=UNAVAILABLE_NOT_MATERIALIZED,
    )
    readiness = _require_dict(manifest.get("readiness"), field="readiness")
    _require_exact_keys(readiness, field="readiness", expected=tuple(POLICY_READINESS))
    if readiness != POLICY_READINESS:
        raise ForecastError(
            f"policy manifest readiness 必须严格等于 {POLICY_READINESS}，实际 {readiness!r}"
        )
    # M1.3g-0：`supersedes` 必须**逐字段恒等**于冻结的取代登记（含 v1 的 path 与 sha256）
    supersedes = _require_dict(manifest.get("supersedes"), field="supersedes")
    _require_exact_keys(
        supersedes, field="supersedes", expected=SUPERSEDES_POLICY_V1_KEYS
    )
    for name, expected in SUPERSEDES_POLICY_V1.items():
        if supersedes.get(name) != expected:
            raise ForecastError(
                f"policy manifest supersedes.{name} 与冻结登记不符："
                f"期望 {expected!r}，实际 {supersedes.get(name)!r}"
            )
    _require_canonical_utc(manifest.get("frozen_at_utc"), field="frozen_at_utc")
    _require_git_sha40(manifest.get("materializer_revision"),
                       field="materializer_revision")

    # 路径声明必须与调用者实际提供的 logical repo path 一致
    for field, actual in (
        ("canonical_parquet_path", canonical_parquet_path),
        ("canonical_manifest_path", canonical_manifest_path),
        ("split_manifest_path", split_manifest_path),
    ):
        declared = manifest.get(field)
        if not isinstance(declared, str) or not declared:
            raise ForecastError(f"policy manifest {field} 必须是非空字符串，实际 {declared!r}")
        if declared != logical_repo_path(actual):
            raise ForecastError(
                f"policy manifest 的 {field} 与实际提供的路径不符："
                f"声明={declared!r} 实际={logical_repo_path(actual)!r}"
            )

    # 声明的 hash 必须与实际文件**字节**一致
    actual_sources = expected_sources or {
        "canonical_parquet_sha256": _sha256_file(canonical_parquet_path),
        "canonical_manifest_sha256": _sha256_file(canonical_manifest_path),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
    }
    for field in (
        "canonical_parquet_sha256", "canonical_manifest_sha256", "split_manifest_sha256",
    ):
        declared = _require_hex64(manifest.get(field), field=field)
        if declared != actual_sources[field]:
            raise ForecastError(
                f"policy manifest 的 {field} 与实际文件不符："
                f"声明={declared} 实际={actual_sources[field]}"
            )

    declared_revision = _require_git_sha40(
        manifest.get("materializer_revision"), field="materializer_revision"
    )
    if declared_revision != expected_revision:
        raise ForecastError(
            "policy manifest 的 materializer_revision 与当前 provider/materializer "
            f"实现的 Git revision 不一致：policy={declared_revision} "
            f"实际={expected_revision}（实现已改动，必须重新生成 policy manifest）"
        )
    return manifest


# --- provider ----------------------------------------------------------------

def _policy_sources(
    canonical_parquet_path: Path, canonical_manifest_path: Path,
    split_manifest_path: Path, policy_manifest_path: Path,
) -> dict[str, str]:
    """四个来源文件的实测 SHA-256；**缺失一律干净 fail closed**（不泄漏 OSError）。"""
    sources: dict[str, str] = {}
    for key, path in (
        ("canonical_parquet_sha256", canonical_parquet_path),
        ("canonical_manifest_sha256", canonical_manifest_path),
        ("split_manifest_sha256", split_manifest_path),
        ("policy_manifest_sha256", policy_manifest_path),
    ):
        try:
            sources[key] = _sha256_file(path)
        except OSError as error:
            raise ForecastError(f"冻结资产不可读（{key}）：{path}：{error}") from error
    return sources


def build_available_exogenous_forecast(
    split: SplitName,
    *,
    origin: int,
    forecast_cutoff: int,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    policy_manifest_path: Path | str,
) -> AvailableExogenousForecast:
    """按**冻结**的 trailing seasonal-naive 规则生成五个 driver 的 forecast。

    `origin` 是 split-**本地** half-hour step；历史窗口取**全局** `[i-48, i)`，
    因此 validation/test 的起点可以使用其**之前已经发生**的 canonical 历史。

    信任链：policy → split → canonical manifest → canonical parquet，逐层校验，
    任一不符即 fail closed。`code_revision` **不接受**调用者输入。
    """
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)
    policy_manifest_path = Path(policy_manifest_path)

    # C 必须通过 M1.3d 的 origin 门禁（越界即拒绝，不截断、不换段）
    global_origin = validate_forecast_origin(split, origin, forecast_cutoff)

    # policy manifest 必须存在且自洽（缺失、畸形、被篡改、路径不匹配均拒绝）
    expected_revision = provider_code_revision()
    sources = _policy_sources(
        canonical_parquet_path, canonical_manifest_path, split_manifest_path,
        policy_manifest_path,
    )
    read_forecast_policy_manifest(
        policy_manifest_path,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
        expected_revision=expected_revision,
        expected_sources={
            "canonical_parquet_sha256": sources["canonical_parquet_sha256"],
            "canonical_manifest_sha256": sources["canonical_manifest_sha256"],
            "split_manifest_sha256": sources["split_manifest_sha256"],
        },
    )

    # split manifest 必须走 M1.3d 的**完整**严格校验（含 canonical 链、整条时间轴
    # 与 train-only 统计重算）；不复制宽松校验器。
    load_truth_split(
        split,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )

    frame = pd.read_parquet(canonical_parquet_path)
    validate_canonical_timeline(frame, label="canonical")
    missing = [driver for driver in AVAILABLE_DRIVERS if driver not in frame.columns]
    if missing:
        raise ForecastError(f"canonical 缺少 driver 列：{missing}")
    if len(frame) < global_origin + forecast_cutoff:
        raise ForecastError("canonical 行数不足以覆盖 target 窗口")

    stamps = frame["timestamp"]
    generated_at = _iso(stamps.iloc[global_origin])
    history_start = global_origin - FORECAST_PERIOD_STEPS
    if history_start < 0:
        raise ForecastError(
            f"{split} 内 origin={origin}（全局 {global_origin}）不足 "
            f"{FORECAST_PERIOD_STEPS} 步历史；fail closed，不回填、不跨年环绕"
        )
    lookback_start = _iso(stamps.iloc[history_start])
    target_timestamps = tuple(
        _iso(stamp) for stamp in stamps.iloc[global_origin:global_origin + forecast_cutoff]
    )
    target_end_exclusive = (
        datetime.fromisoformat(target_timestamps[-1])
        + timedelta(minutes=STEP_MINUTES)
    ).isoformat()

    # 顶层四条 logical path：与逐序列 digest **逐项恒等**（由契约在构造时校验）
    logical_paths = {
        "canonical_parquet_path": logical_repo_path(canonical_parquet_path),
        "canonical_manifest_path": logical_repo_path(canonical_manifest_path),
        "split_manifest_path": logical_repo_path(split_manifest_path),
        "policy_manifest_path": logical_repo_path(policy_manifest_path),
    }
    digest_by_role = {
        "canonical_parquet": ArtifactDigest(
            role="canonical_parquet",
            logical_path=logical_paths["canonical_parquet_path"],
            sha256=sources["canonical_parquet_sha256"],
        ),
        "canonical_manifest": ArtifactDigest(
            role="canonical_manifest",
            logical_path=logical_paths["canonical_manifest_path"],
            sha256=sources["canonical_manifest_sha256"],
        ),
        "split_manifest": ArtifactDigest(
            role="split_manifest",
            logical_path=logical_paths["split_manifest_path"],
            sha256=sources["split_manifest_sha256"],
        ),
        "forecast_policy_manifest": ArtifactDigest(
            role="forecast_policy_manifest",
            logical_path=logical_paths["policy_manifest_path"],
            sha256=sources["policy_manifest_sha256"],
        ),
    }
    # 角色**按序**等于冻结集合：缺失、重复、额外、乱序都会被契约拒绝
    all_sources = tuple(
        digest_by_role[role] for role in AVAILABLE_SOURCE_ROLES
    )

    series: dict[str, tuple[float, ...]] = {}
    provenance: dict[str, ForecastSeriesProvenance] = {}
    for driver in AVAILABLE_DRIVERS:
        values = seasonal_naive_forecast(
            frame[driver].to_numpy(),
            origin=global_origin,
            forecast_cutoff=forecast_cutoff,
        )
        series[driver] = values
        provenance[driver] = ForecastSeriesProvenance(
            series_name=driver,
            source_kind=SOURCE_KIND,
            method=METHOD,
            generated_at=generated_at,
            information_cutoff_exclusive=generated_at,
            target_start=generated_at,
            target_end_exclusive=target_end_exclusive,
            lookback_start=lookback_start,
            lookback_end_exclusive=generated_at,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            code_revision=expected_revision,
            seed=None,
            sources=all_sources,
        )

    return AvailableExogenousForecast(
        split=split,
        origin=int(origin),
        global_origin=int(global_origin),
        forecast_cutoff=forecast_cutoff,
        frequency=FREQUENCY,
        method=METHOD,
        period_steps=FORECAST_PERIOD_STEPS,
        generated_at=generated_at,
        target_timestamps=target_timestamps,
        series=AvailableSeries(**series),
        provenance=AvailableDriverProvenance(**provenance),
        policy_manifest_path=logical_paths["policy_manifest_path"],
        policy_manifest_sha256=sources["policy_manifest_sha256"],
        canonical_parquet_path=logical_paths["canonical_parquet_path"],
        canonical_manifest_path=logical_paths["canonical_manifest_path"],
        split_manifest_path=logical_paths["split_manifest_path"],
        canonical_parquet_sha256=sources["canonical_parquet_sha256"],
        canonical_manifest_sha256=sources["canonical_manifest_sha256"],
        split_manifest_sha256=sources["split_manifest_sha256"],
        code_revision=expected_revision,
    )


def _iso(stamp: Any) -> str:
    return pd.Timestamp(stamp).isoformat()


def frozen_at_utc_now() -> str:
    """物化时使用的规范 UTC 时间戳（独立函数便于测试）。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat()
