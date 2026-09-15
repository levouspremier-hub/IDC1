"""M1.3d：Singapore-2024 **连续 truth split** reader 与 origin 边界门禁。

冻结的是 **truth** 三段切分（人工批准的月对齐方案 B）与 train-only 描述统计。
本模块**不生成 forecast**、**不构造 `ScenarioBundle`**、**不创建 `train.json`**。

边界语义（半开区间，`Asia/Singapore`，半小时粒度）：

```text
train      [2024-01-01T00:00+08:00, 2024-08-01T00:00+08:00)   rows [0,     10224)
validation [2024-08-01T00:00+08:00, 2024-10-01T00:00+08:00)   rows [10224, 13152)
test       [2024-10-01T00:00+08:00, 2025-01-01T00:00+08:00)   rows [13152, 17568)
```

合计 **17,568** 行，无重叠、无缺口；Feb 29 完整属于 train。

**origin 门禁**（取代早期草案里的「统一 `H + C` purge」）：

- episode：`origin_index + H <= row_end_exclusive`；
- forecast：`origin_index + C <= row_end_exclusive`；
- **`H >= C` 时不得重复扣除 `C`**（`H + C` 是错误的双重惩罚）；
- 模型 lookback 只允许读 origin **之前**的数据；
- 是否需要统计学 purge 由**具体 forecast 模型的 lookback/label 定义**决定，
  本模块**不预先写死** `H + C`。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from scenario.singapore_2024 import CANONICAL_COLUMNS, TIMEZONE

SPLIT_SCHEMA = "m1.3d-singapore-2024-splits-v1"
CANONICAL_SCHEMA = "m1.3b-singapore-2024-half-hour-v1"
FREQUENCY = "30min"
STEP_MINUTES = 30
TOTAL_ROWS = 17568
YEAR = 2024

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
SplitName = Literal["train", "validation", "test"]

# 冻结的 row 段（**唯一**来源；reader 与 manifest 都从这里派生）
_SPAN = {"train": 10224, "validation": 2928, "test": 4416}
SPLIT_ROW_STARTS: dict[str, int] = {}
SPLIT_ROW_COUNTS: dict[str, int] = {}
_cursor = 0
for _name in SPLIT_NAMES:
    SPLIT_ROW_STARTS[_name] = _cursor
    SPLIT_ROW_COUNTS[_name] = _SPAN[_name]
    _cursor += _SPAN[_name]
assert _cursor == TOTAL_ROWS, "split 必须完整覆盖 canonical 行数"

_BOUNDARIES = {
    "train": ("2024-01-01T00:00:00+08:00", "2024-08-01T00:00:00+08:00"),
    "validation": ("2024-08-01T00:00:00+08:00", "2024-10-01T00:00:00+08:00"),
    "test": ("2024-10-01T00:00:00+08:00", "2025-01-01T00:00:00+08:00"),
}
SPLIT_SPECS: dict[str, dict[str, Any]] = {
    name: {
        "start": _BOUNDARIES[name][0],
        "end_exclusive": _BOUNDARIES[name][1],
        "row_start": SPLIT_ROW_STARTS[name],
        "row_end_exclusive": SPLIT_ROW_STARTS[name] + SPLIT_ROW_COUNTS[name],
        "row_count": SPLIT_ROW_COUNTS[name],
    }
    for name in SPLIT_NAMES
}

# 可供 train-only 统计的 canonical 列（unavailable 四列**不在**其中）
STATISTIC_COLUMNS: tuple[str, ...] = (
    "price_sgd_per_kwh",
    "system_load_mw",
    "national_igs_mwh_per_half_hour",
    "temperature_deg_c",
    "wind_speed_10m_mps",
    "ghi_w_per_m2",
)
STATISTIC_FIELDS = ("count", "finite_count", "min", "max", "negative_count", "mean")

UNAVAILABLE_COLUMNS = ("local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival")

EPISODE_ORIGIN_RULE = (
    "episode origin 的最后一个执行点必须仍在该 split 内：origin_index + H <= row_end_exclusive"
)
FORECAST_ORIGIN_RULE = (
    "forecast 可见点 [origin, origin + C) 必须完全属于该 split："
    "origin_index + C <= row_end_exclusive；H >= C 时不得重复扣除 C；"
    "lookback 只允许读 origin 之前的数据"
)


class SplitError(ValueError):
    """split 读取或边界校验的**明确失败**（不截断、不环绕、不换段）。"""


# --- 严格校验辅助 -----------------------------------------------------------

def _strict_positive_int(value: Any, *, field: str) -> int:
    """严格正整数：bool / float / str / 0 / 负数一律拒绝。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitError(f"{field} 必须是整数（unit: half-hour steps），实际 {value!r}")
    if value <= 0:
        raise SplitError(f"{field} 必须是严格正整数，实际 {value!r}")
    return value


def _require_split_name(split: Any) -> str:
    if not isinstance(split, str) or split not in SPLIT_NAMES:
        raise SplitError(f"split 必须是 {list(SPLIT_NAMES)} 之一，实际 {split!r}")
    return split


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_dict(value: Any, *, field: str) -> dict:
    if not isinstance(value, dict):
        raise SplitError(f"{field} 必须是 object，实际 {type(value).__name__}")
    return value


def _require_hex64(value: Any, *, field: str) -> str:
    ok = (isinstance(value, str) and len(value) == 64
          and all(c in "0123456789abcdef" for c in value))
    if not ok:
        raise SplitError(f"{field} 必须是 64 位小写十六进制字符串，实际 {value!r}")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitError(f"{field} 必须是整数，实际 {value!r}")
    return value


# --- origin 边界门禁 --------------------------------------------------------

def _origin_index(split: str, origin: int) -> int:
    """把 origin（该 split 内、从 0 开始的 half-hour step）映射到 canonical 全局行号。"""
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise SplitError(f"origin 必须是整数 half-hour step，实际 {origin!r}")
    spec = SPLIT_SPECS[split]
    if origin < 0 or origin >= spec["row_count"]:
        raise SplitError(
            f"origin {origin} 越出 {split} 的行范围 [0, {spec['row_count']})；"
            "不得自动换段或截断"
        )
    return spec["row_start"] + origin


def validate_episode_origin(split: str, origin: int, horizon_steps: int) -> int:
    """episode 的最后一个执行点必须仍在该 split 内。返回 origin 的**全局**行号。

    约束：`origin_index + H <= row_end_exclusive`（半开区间）。
    """
    name = _require_split_name(split)
    horizon = _strict_positive_int(horizon_steps, field="horizon_steps")
    index = _origin_index(name, origin)
    end = SPLIT_SPECS[name]["row_end_exclusive"]
    if index + horizon > end:
        raise SplitError(
            f"episode origin {origin}（全局行 {index}）+ H={horizon} 越出 "
            f"{name} 的 row_end_exclusive={end}；不截断、不换段"
        )
    return index


def validate_forecast_origin(split: str, origin: int, forecast_cutoff_steps: int) -> int:
    """forecast 可见窗口 `[origin, origin + C)` 必须完全属于该 split。

    返回 origin 的**全局**行号。**不**额外执行 `H + C` 扣除：
    本函数只约束 `origin + C <= row_end_exclusive`（见模块 docstring）。
    """
    name = _require_split_name(split)
    cutoff = _strict_positive_int(forecast_cutoff_steps, field="forecast_cutoff_steps")
    index = _origin_index(name, origin)
    end = SPLIT_SPECS[name]["row_end_exclusive"]
    if index + cutoff > end:
        raise SplitError(
            f"forecast origin {origin}（全局行 {index}）+ C={cutoff} 越出 "
            f"{name} 的 row_end_exclusive={end}；不截断、不换段"
        )
    return index


# --- train-only 统计 --------------------------------------------------------

def train_only_statistics(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """计算 **train 行** 的描述统计。调用方必须只传入 train 切片。

    这些是**描述统计**，**不得**自动冒充正式 normalization refs
    （后者最迟在 M1.3g 训练接线前冻结）。
    """
    statistics: dict[str, dict[str, Any]] = {}
    for column in STATISTIC_COLUMNS:
        if column not in frame.columns:
            raise SplitError(f"train 切片缺少 canonical 列 {column}")
        series = pd.to_numeric(frame[column], errors="coerce")
        finite = series.map(lambda x: isinstance(x, (int, float)) and math.isfinite(x))
        values = series[finite]
        statistics[column] = {
            "count": int(series.notna().sum()),
            "finite_count": int(finite.sum()),
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "negative_count": int((values < 0).sum()),
            "mean": float(values.mean()) if len(values) else None,
        }
    return statistics


# --- reader -----------------------------------------------------------------

def _read_split_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"split manifest 不可读：{error}") from error
    manifest = _require_dict(manifest, field="split manifest")
    if manifest.get("schema") != SPLIT_SCHEMA:
        raise SplitError(
            f"split manifest schema 必须是 {SPLIT_SCHEMA!r}，实际 {manifest.get('schema')!r}"
        )
    if manifest.get("timezone") != TIMEZONE:
        raise SplitError(f"split manifest timezone 必须是 {TIMEZONE!r}")
    if manifest.get("frequency") != FREQUENCY:
        raise SplitError(f"split manifest frequency 必须是 {FREQUENCY!r}")
    if _require_int(manifest.get("total_rows"), field="total_rows") != TOTAL_ROWS:
        raise SplitError(f"split manifest total_rows 必须是 {TOTAL_ROWS}")
    _require_hex64(manifest.get("canonical_parquet_sha256"),
                   field="canonical_parquet_sha256")
    _require_hex64(manifest.get("canonical_manifest_sha256"),
                   field="canonical_manifest_sha256")
    splits = _require_dict(manifest.get("splits"), field="splits")
    for name in SPLIT_NAMES:
        entry = _require_dict(splits.get(name), field=f"splits.{name}")
        spec = SPLIT_SPECS[name]
        for key in ("start", "end_exclusive", "row_start", "row_end_exclusive", "row_count"):
            if entry.get(key) != spec[key]:
                raise SplitError(
                    f"splits.{name}.{key} 与冻结值不符：manifest={entry.get(key)!r} "
                    f"冻结={spec[key]!r}"
                )
    return manifest


def _verify_canonical(canonical_parquet_path: Path, canonical_manifest_path: Path,
                      split_manifest: dict) -> str:
    """逐级校验：canonical manifest → canonical parquet。返回 parquet 的 sha256。"""
    try:
        canonical = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"canonical manifest 不可读：{error}") from error
    canonical = _require_dict(canonical, field="canonical manifest")
    if canonical.get("schema") != CANONICAL_SCHEMA:
        raise SplitError(
            f"canonical manifest schema 必须是 {CANONICAL_SCHEMA!r}，"
            f"实际 {canonical.get('schema')!r}"
        )
    if _require_int(canonical.get("row_count"), field="canonical.row_count") != TOTAL_ROWS:
        raise SplitError(f"canonical manifest row_count 必须是 {TOTAL_ROWS}")
    if canonical.get("timezone") != TIMEZONE or canonical.get("frequency") != FREQUENCY:
        raise SplitError("canonical manifest 的 timezone/frequency 不符")

    expected_manifest_sha = _require_hex64(
        split_manifest["canonical_manifest_sha256"], field="canonical_manifest_sha256")
    actual_manifest_sha = _sha256_file(canonical_manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise SplitError(
            "canonical manifest SHA-256 与 split manifest 不符："
            f"split={expected_manifest_sha} 实际={actual_manifest_sha}"
        )

    expected_parquet_sha = _require_hex64(
        split_manifest["canonical_parquet_sha256"], field="canonical_parquet_sha256")
    recorded_parquet_sha = _require_hex64(
        canonical.get("output_parquet_sha256"), field="canonical.output_parquet_sha256")
    actual_parquet_sha = _sha256_file(canonical_parquet_path)
    if actual_parquet_sha != expected_parquet_sha:
        raise SplitError(
            "canonical parquet SHA-256 与 split manifest 不符："
            f"split={expected_parquet_sha} 实际={actual_parquet_sha}"
        )
    if recorded_parquet_sha != actual_parquet_sha:
        raise SplitError("canonical parquet SHA-256 与 canonical manifest 不符")
    return actual_parquet_sha


def load_truth_split(
    split: SplitName,
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
) -> pd.DataFrame:
    """读取冻结 split 的 **truth** 切片（副本）。

    读取前**逐级**校验 split manifest → canonical manifest → canonical parquet hash；
    任一不符即 `SplitError`。**不接受**调用者用任意 row range 绕开冻结 split。
    """
    name = _require_split_name(split)
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    split_manifest = _read_split_manifest(split_manifest_path)
    _verify_canonical(canonical_parquet_path, canonical_manifest_path, split_manifest)

    spec = SPLIT_SPECS[name]
    frame = pd.read_parquet(canonical_parquet_path)
    if tuple(frame.columns) != CANONICAL_COLUMNS:
        raise SplitError(f"canonical 列不符：{tuple(frame.columns)}")
    if len(frame) != TOTAL_ROWS:
        raise SplitError(f"canonical 行数必须是 {TOTAL_ROWS}，实际 {len(frame)}")
    if str(frame["timestamp"].dt.tz) != TIMEZONE:
        raise SplitError(f"canonical 时间戳时区必须是 {TIMEZONE}")

    sliced = frame.iloc[spec["row_start"]:spec["row_end_exclusive"]].copy()
    if len(sliced) != spec["row_count"]:
        raise SplitError(
            f"{name} 切片行数 {len(sliced)} != 冻结值 {spec['row_count']}"
        )
    start = pd.Timestamp(spec["start"])
    end = pd.Timestamp(spec["end_exclusive"])
    if sliced["timestamp"].iloc[0] != start:
        raise SplitError(f"{name} 首行时间戳 != {spec['start']}")
    if sliced["timestamp"].iloc[-1] >= end:
        raise SplitError(f"{name} 末行时间戳 >= end_exclusive {spec['end_exclusive']}")
    return sliced
