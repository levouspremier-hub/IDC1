"""M1.3e：**因果** forecast provider —— 只依赖 origin 之前的历史。

本模块为五个 canonical truth driver 生成 forecast：

    price_sgd_per_kwh、system_load_mw、temperature_deg_c、
    wind_speed_10m_mps、ghi_w_per_m2

**冻结方法**：trailing seasonal-naive，周期**固定** `PERIOD_STEPS = 48`
（半小时 × 48 = 一个物理日周期，**不从 validation/test 选择**）。

对全局 origin = i：

- 历史模板**只允许**读取 `[i-48, i)`；**绝不**读取 i 或 i 之后的 truth；
- forecast 第 k 项取模板的 `k mod 48` 项；
- `generated_at = information_cutoff_exclusive = lookback_end_exclusive = origin`；
- target = `[origin, origin+C)`，`C` 必须通过 M1.3d `validate_forecast_origin`；
- 历史不足 48 步（如 train 内 `origin < 48`）**fail closed**，不回填、不跨年环绕；
- 不使用任何随机数（`seed` 明确为 `null`）；不在本卡暗中引入任何拟合。

本模块**不**生成 PV / 风电发电量 / 碳强度 / arrival 的 forecast，
也**不**构造正式 `ScenarioBundle` —— 它只产出「available exogenous forecast artifact」。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from contracts import CONTRACT_VERSION_ID
from scenario.splits import (
    CANONICAL_SCHEMA,
    FREQUENCY as SPLIT_FREQUENCY,
    STEP_MINUTES,
    TIMEZONE,
    TOTAL_ROWS,
    SplitName,
    _require_dict,
    _require_hex64,
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
FREQUENCY = "30min"
SOURCE_KIND = "seasonal_naive"

AVAILABLE_DRIVERS: tuple[str, ...] = (
    "price_sgd_per_kwh",
    "system_load_mw",
    "temperature_deg_c",
    "wind_speed_10m_mps",
    "ghi_w_per_m2",
)
# 这四个字段本卡**不**生成 forecast（不得用全零或默认曲线补齐）
UNAVAILABLE_NOT_MATERIALIZED: tuple[str, ...] = (
    "local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival",
)

FORECAST_SOURCE_PATHS = ("scenario/forecast.py",)


class ForecastError(ValueError):
    """因果 forecast 的**明确失败**。"""


def _require_positive_int(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ForecastError(f"{field} 必须是整数（half-hour steps），实际 {value!r}")
    if value <= 0:
        raise ForecastError(f"{field} 必须是严格正整数，实际 {value!r}")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_canonical_parquet(
    *, canonical_parquet_path: Path, canonical_manifest_path: Path,
    split_manifest: dict,
) -> str:
    """逐级校验**实际读取的那份 canonical parquet 字节**（不符即 fail closed）。

    - split manifest 的 `canonical_parquet_sha256`
    - canonical manifest 的 `output_parquet_sha256`
    - parquet 文件的实测 sha256

    三者必须相等，且 canonical manifest 的 schema/行数/时区/频率必须与冻结值一致。

    **范围说明**：本函数**不**校验 canonical manifest 文件自身的 hash
    （split manifest 的 `canonical_manifest_sha256`）—— 那是 M1.3d `load_truth_split`
    读取链的职责；provider 只对「它真正读到的字节」负责，
    以免把与本次预测无关的 manifest 元数据变化误判为数据变化。
    """
    try:
        canonical = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForecastError(f"canonical manifest 不可读：{error}") from error
    canonical = _require_dict(canonical, field="canonical manifest")
    if canonical.get("schema") != CANONICAL_SCHEMA:
        raise ForecastError(
            f"canonical manifest schema 必须是 {CANONICAL_SCHEMA!r}，"
            f"实际 {canonical.get('schema')!r}"
        )
    if canonical.get("row_count") != TOTAL_ROWS:
        raise ForecastError(f"canonical manifest row_count 必须是 {TOTAL_ROWS}")
    if canonical.get("timezone") != TIMEZONE or canonical.get("frequency") != SPLIT_FREQUENCY:
        raise ForecastError("canonical manifest 的 timezone/frequency 不符")

    expected = _require_hex64(
        split_manifest["canonical_parquet_sha256"], field="canonical_parquet_sha256"
    )
    recorded = _require_hex64(
        canonical.get("output_parquet_sha256"), field="output_parquet_sha256"
    )
    actual = _sha256_file(canonical_parquet_path)
    if actual != expected:
        raise ForecastError(
            "canonical parquet SHA-256 与 split manifest 不符："
            f"split={expected} 实际={actual}"
        )
    if actual != recorded:
        raise ForecastError(
            "canonical parquet SHA-256 与 canonical manifest 不符："
            f"canonical={recorded} 实际={actual}"
        )
    return actual


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True, check=True,
    ).stdout


def _source_revision() -> str:
    """本 provider 实现的 revision（由 Git 解析，不用漂移的 HEAD）。"""
    revision = _git("log", "-1", "--format=%H", "--", *FORECAST_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ForecastError(f"provider revision 无效：{revision!r}")
    return revision


class AvailableExogenousForecast:
    """**available exogenous forecast artifact** —— 不是完整 `ScenarioBundle`。

    只含五个 driver 的 forecast 与其结构化 provenance；**不**含
    PV / 风电发电量 / 碳强度 / arrival，故**不得**被当作完整场景。
    """

    # `content_hash` 只覆盖**预测内容本身**（这套预测的确定性身份）：
    # 上游制品的 sha256 会随上游文件字节变化，而预测值未必变化——把它们算进
    # content hash，会让「同一份预测」在上游重物化后得到不同身份。
    # 上游 digest 仍完整保留在 `to_dict()["provenance"][*]["sources"]` 中供审计，
    # 并对**实际读取的 parquet 字节**做 fail-closed 校验（`_verify_canonical_parquet`）。
    _CONTENT_HASH_KEYS: tuple[str, ...] = (
        "contract_version", "split", "origin", "global_origin", "forecast_cutoff",
        "frequency", "method", "period_steps", "generated_at",
        "target_timestamps", "series",
    )

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __getattr__(self, name: str):
        try:
            return self._payload[name]
        except KeyError as error:  # pragma: no cover - 属性拼写错误
            raise AttributeError(name) from error

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self._payload))

    def content_hash(self) -> str:
        content = {key: self._payload[key] for key in self._CONTENT_HASH_KEYS}
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def build_available_exogenous_forecast(
    split: SplitName,
    *,
    origin: int,
    forecast_cutoff: int,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    code_revision: str | None = None,
) -> AvailableExogenousForecast:
    """按**冻结**的 trailing seasonal-naive 规则生成五个 driver 的 forecast。

    `origin` 是 split-**本地** half-hour step；历史窗口取**全局** `[i-48, i)`，
    因此 validation/test 的起点可以使用其**之前已经发生**的 canonical 历史。
    """
    cutoff = _require_positive_int(forecast_cutoff, field="forecast_cutoff")
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    # C 必须通过 M1.3d 的 origin 门禁（越界即拒绝，不截断、不换段）
    global_origin = validate_forecast_origin(split, origin, cutoff)

    history_start = global_origin - PERIOD_STEPS
    if history_start < 0:
        raise ForecastError(
            f"{split} 内 origin={origin}（全局 {global_origin}）不足 {PERIOD_STEPS} 步历史；"
            "fail closed，不回填、不跨年环绕"
        )

    # 逐级校验 canonical manifest → canonical parquet（任一 hash 不符即 fail closed）
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))

    frame = pd.read_parquet(canonical_parquet_path)
    validate_canonical_timeline(frame, label="canonical")
    if len(frame) <= global_origin + cutoff - 1:
        raise ForecastError("canonical 行数不足以覆盖 target 窗口")
    missing = [d for d in AVAILABLE_DRIVERS if d not in frame.columns]
    if missing:
        raise ForecastError(f"canonical 缺少 driver 列：{missing}")

    template = frame.iloc[history_start:global_origin]
    if len(template) != FORECAST_PERIOD_STEPS:
        raise ForecastError(
            f"历史模板长度必须是 {FORECAST_PERIOD_STEPS}，实际 {len(template)}"
        )

    stamps = frame["timestamp"]
    generated_at = _iso(stamps.iloc[global_origin])
    lookback_start = _iso(stamps.iloc[history_start])
    target_end_exclusive = stamps.iloc[global_origin + cutoff - 1] + pd.Timedelta(
        minutes=STEP_MINUTES
    )
    target_timestamps = [_iso(t) for t in stamps.iloc[global_origin:global_origin + cutoff]]

    revision = code_revision or _source_revision()
    parquet_sha = _verify_canonical_parquet(
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest=split_manifest,
    )
    sources = [
        {"role": "canonical_parquet",
         "logical_path": logical_repo_path(canonical_parquet_path),
         "sha256": parquet_sha},
        {"role": "canonical_manifest",
         "logical_path": logical_repo_path(canonical_manifest_path),
         "sha256": _sha256_file(canonical_manifest_path)},
        {"role": "split_manifest",
         "logical_path": logical_repo_path(split_manifest_path),
         "sha256": _sha256_file(split_manifest_path)},
    ]

    series: dict[str, list[float]] = {}
    provenance: dict[str, dict] = {}
    for driver in AVAILABLE_DRIVERS:
        values = [float(v) for v in template[driver].to_numpy()]
        if not all(math.isfinite(v) for v in values):
            raise ForecastError(f"{driver} 的历史模板含非有限值")
        # forecast 第 k 项 = 模板的第 (k mod 48) 项
        forecast = [values[k % FORECAST_PERIOD_STEPS] for k in range(cutoff)]
        series[driver] = forecast
        provenance[driver] = {
            "series_name": driver,
            "source_kind": SOURCE_KIND,
            "method": METHOD,
            "generated_at": generated_at,
            "information_cutoff_exclusive": generated_at,
            "target_start": generated_at,
            "target_end_exclusive": _iso(target_end_exclusive),
            "lookback_start": lookback_start,
            "lookback_end_exclusive": generated_at,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "code_revision": revision,
            "seed": None,
            "sources": sources,
        }

    return AvailableExogenousForecast({
        "artifact": "available_exogenous_forecast",
        "contract_version": CONTRACT_VERSION_ID,
        "split": split,
        "origin": int(origin),
        "global_origin": int(global_origin),
        "forecast_cutoff": cutoff,
        "frequency": FREQUENCY,
        "method": METHOD,
        "period_steps": FORECAST_PERIOD_STEPS,
        "generated_at": generated_at,
        "target_timestamps": target_timestamps,
        "series": series,
        "provenance": provenance,
        "unavailable_not_materialized": list(UNAVAILABLE_NOT_MATERIALIZED),
    })


def _iso(stamp) -> str:
    return pd.Timestamp(stamp).isoformat()
