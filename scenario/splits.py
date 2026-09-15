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


EXPECTED_READINESS = {
    "truth_splits_ready": True,
    "forecast_ready": False,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}
# unavailable 四项必须完整存在，且**不得**被伪造成 available
EXPECTED_UNAVAILABLE_STATUS = "unavailable"
_FORBIDDEN_AVAILABLE_MARKERS = ("available", "materialized", "ready", "present")


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


REPO_ROOT = Path(__file__).resolve().parent.parent


def logical_repo_path(path: Path | str) -> str:
    """入库 manifest 只记**仓库相对逻辑路径**，绝不泄漏机器绝对路径。

    reader 与物化器共用同一实现，才能让「声明路径 vs 实际路径」的比较有意义。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


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


def _require_git_sha40(value: Any, *, field: str) -> str:
    """Git 提交 SHA：40 位小写十六进制。"""
    ok = (isinstance(value, str) and len(value) == 40
          and all(c in "0123456789abcdef" for c in value))
    if not ok:
        raise SplitError(f"{field} 必须是 40 位小写十六进制 Git 提交，实际 {value!r}")
    return value


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitError(f"{field} 必须是整数，实际 {value!r}")
    return value


# --- canonical 时间轴严格校验 -----------------------------------------------

GRID_DELTA = pd.Timedelta(minutes=STEP_MINUTES)
TIMELINE_START = pd.Timestamp("2024-01-01T00:00:00+08:00")
TIMELINE_END = pd.Timestamp("2024-12-31T23:30:00+08:00")


def validate_canonical_timeline(frame: pd.DataFrame, *, label: str = "canonical") -> None:
    """**整表**时间轴校验：tz-aware、严格 30min 网格、严格递增、唯一、首末精确。

    只查首末行是不够的 —— 内部交换、重复、缺失后用非网格时间戳补足、
    非单调等都能骗过首末检查（M1.3d §11.1 问题 2 的实测缺陷）。
    """
    if "timestamp" not in frame.columns:
        raise SplitError(f"{label} 缺少 timestamp 列")
    stamps = frame["timestamp"]
    if not isinstance(stamps.dtype, pd.DatetimeTZDtype):
        raise SplitError(f"{label} 的 timestamp 必须是 tz-aware")
    if str(stamps.dt.tz) != TIMEZONE:
        raise SplitError(f"{label} 的 timestamp 时区必须是 {TIMEZONE}")
    if len(stamps) != TOTAL_ROWS:
        raise SplitError(f"{label} 行数必须是 {TOTAL_ROWS}，实际 {len(stamps)}")
    if not stamps.is_monotonic_increasing:
        raise SplitError(f"{label} 的 timestamp 必须单调不减（非单调即失败）")
    if stamps.duplicated().any():
        raise SplitError(f"{label} 的 timestamp 不得重复")
    deltas = stamps.diff().dropna().unique()
    if len(deltas) != 1 or deltas[0] != GRID_DELTA:
        raise SplitError(
            f"{label} 的 timestamp 必须是严格的 {STEP_MINUTES} 分钟网格，"
            f"实际间隔集合={[str(d) for d in deltas]}"
        )
    if stamps.iloc[0] != TIMELINE_START:
        raise SplitError(f"{label} 首个时间戳必须是 {TIMELINE_START}")
    if stamps.iloc[-1] != TIMELINE_END:
        raise SplitError(f"{label} 末个时间戳必须是 {TIMELINE_END}")


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

def _read_split_manifest(path: Path, *, canonical_parquet_path: Path,
                         canonical_manifest_path: Path) -> dict:
    """读取并**严格**校验 split manifest。

    必须拒绝「随后被篡改的冻结声明」，而不是只证明「生成时写对了」：
    除 schema/时区/频率/行数/边界/hash 格式外，还校验 year、step_minutes、
    两个**路径声明与调用者实际提供的 logical repo path 一致**、
    `materializer_revision` 的 40 位小写 SHA 格式、no_overlap/no_gap/randomized/
    leap_day_split、两条 origin 规则、readiness 四态、unavailable 四项、
    train_only_statistics_source 与 train_only_statistics。
    """
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SplitError(f"split manifest 不可读：{error}") from error
    manifest = _require_dict(manifest, field="split manifest")

    if manifest.get("schema") != SPLIT_SCHEMA:
        raise SplitError(
            f"split manifest schema 必须是 {SPLIT_SCHEMA!r}，实际 {manifest.get('schema')!r}"
        )
    if _require_int(manifest.get("year"), field="year") != YEAR:
        raise SplitError(f"split manifest year 必须是 {YEAR}")
    if manifest.get("timezone") != TIMEZONE:
        raise SplitError(f"split manifest timezone 必须是 {TIMEZONE!r}")
    if manifest.get("frequency") != FREQUENCY:
        raise SplitError(f"split manifest frequency 必须是 {FREQUENCY!r}")
    if _require_int(manifest.get("step_minutes"), field="step_minutes") != STEP_MINUTES:
        raise SplitError(f"split manifest step_minutes 必须是 {STEP_MINUTES}")
    if _require_int(manifest.get("total_rows"), field="total_rows") != TOTAL_ROWS:
        raise SplitError(f"split manifest total_rows 必须是 {TOTAL_ROWS}")

    # 路径声明必须与调用者实际提供的 logical repo path 一致
    for field, actual in (("canonical_parquet_path", canonical_parquet_path),
                          ("canonical_manifest_path", canonical_manifest_path)):
        declared = manifest.get(field)
        if not isinstance(declared, str) or not declared:
            raise SplitError(f"split manifest {field} 必须是非空字符串，实际 {declared!r}")
        if declared != logical_repo_path(actual):
            raise SplitError(
                f"split manifest 的 {field} 与实际提供的路径不符："
                f"声明={declared!r} 实际={logical_repo_path(actual)!r}"
            )

    _require_hex64(manifest.get("canonical_parquet_sha256"),
                   field="canonical_parquet_sha256")
    _require_hex64(manifest.get("canonical_manifest_sha256"),
                   field="canonical_manifest_sha256")
    _require_git_sha40(manifest.get("materializer_revision"),
                       field="materializer_revision")

    # 冻结声明：必须为**严格**的期望值（与边界推导矛盾即失败）
    for field, expected in (("no_overlap", True), ("no_gap", True), ("randomized", False)):
        if manifest.get(field) is not expected:
            raise SplitError(
                f"split manifest {field} 必须严格为 {expected}，实际 {manifest.get(field)!r}"
            )
    if manifest.get("leap_day_split") != "train":
        raise SplitError("split manifest leap_day_split 必须严格为 'train'")
    for field, expected in (("episode_origin_rule", EPISODE_ORIGIN_RULE),
                            ("forecast_origin_rule", FORECAST_ORIGIN_RULE)):
        if manifest.get(field) != expected:
            raise SplitError(f"split manifest {field} 必须等于冻结规则")

    readiness = _require_dict(manifest.get("readiness"), field="readiness")
    if readiness != EXPECTED_READINESS:
        raise SplitError(
            f"split manifest readiness 必须严格等于 {EXPECTED_READINESS}，实际 {readiness!r}"
        )

    unavailable = _require_dict(manifest.get("unavailable_not_materialized"),
                                field="unavailable_not_materialized")
    if set(unavailable) != set(UNAVAILABLE_COLUMNS):
        raise SplitError(
            "unavailable_not_materialized 必须完整覆盖四项 "
            f"{list(UNAVAILABLE_COLUMNS)}，实际 {sorted(unavailable)}"
        )
    for column, entry in unavailable.items():
        text = entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)
        lowered = text.lower()
        if EXPECTED_UNAVAILABLE_STATUS not in lowered:
            raise SplitError(f"{column} 必须标注为 {EXPECTED_UNAVAILABLE_STATUS!r}")
        # 注意：先剔除 "unavailable" 本身，否则 "available" 会命中它的子串。
        remainder = lowered.replace(EXPECTED_UNAVAILABLE_STATUS, "")
        hit = [m for m in _FORBIDDEN_AVAILABLE_MARKERS if m in remainder]
        if hit:
            raise SplitError(f"{column} 不得被伪造成 available/materialized（命中 {hit}）")

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

    split_manifest = _read_split_manifest(
        split_manifest_path,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
    )
    _verify_canonical(canonical_parquet_path, canonical_manifest_path, split_manifest)

    frame = pd.read_parquet(canonical_parquet_path)
    if tuple(frame.columns) != CANONICAL_COLUMNS:
        raise SplitError(f"canonical 列不符：{tuple(frame.columns)}")
    # 返回切片**之前**就校验整条时间轴（只查首末行不够）
    validate_canonical_timeline(frame, label="canonical")
    _verify_train_statistics(frame, split_manifest)

    spec = SPLIT_SPECS[name]
    sliced = frame.iloc[spec["row_start"]:spec["row_end_exclusive"]].copy()
    if len(sliced) != spec["row_count"]:
        raise SplitError(f"{name} 切片行数 {len(sliced)} != 冻结值 {spec['row_count']}")
    if sliced["timestamp"].iloc[0] != pd.Timestamp(spec["start"]):
        raise SplitError(f"{name} 首行时间戳 != {spec['start']}")
    if sliced["timestamp"].iloc[-1] >= pd.Timestamp(spec["end_exclusive"]):
        raise SplitError(f"{name} 末行时间戳 >= end_exclusive {spec['end_exclusive']}")
    return sliced


def _verify_train_statistics(frame: pd.DataFrame, split_manifest: dict) -> None:
    """train-only 统计必须与 canonical **train 段重新计算值**一致。

    只检查字段存在是不够的：伪造数值、清空、bool 冒充 int、NaN 都必须被拒绝。
    """
    source = _require_dict(split_manifest.get("train_only_statistics_source"),
                           field="train_only_statistics_source")
    spec = SPLIT_SPECS["train"]
    if source.get("split") != "train":
        raise SplitError("train_only_statistics_source.split 必须严格为 'train'")
    for field, expected in (("row_start", spec["row_start"]),
                            ("row_end_exclusive", spec["row_end_exclusive"]),
                            ("start", spec["start"]),
                            ("end_exclusive", spec["end_exclusive"])):
        if source.get(field) != expected:
            raise SplitError(
                f"train_only_statistics_source.{field} 与 train 边界不符："
                f"{source.get(field)!r} != {expected!r}"
            )
    if list(source.get("columns") or []) != list(STATISTIC_COLUMNS):
        raise SplitError(
            f"train_only_statistics_source.columns 必须是 {list(STATISTIC_COLUMNS)}"
        )
    # 统计来源必须**指向真实**的 canonical parquet 与实现 revision，而不是只格式合法
    if source.get("canonical_parquet_sha256") != split_manifest.get(
        "canonical_parquet_sha256"
    ):
        raise SplitError(
            "train_only_statistics_source.canonical_parquet_sha256 必须等于 split manifest "
            "的 canonical_parquet_sha256"
        )
    if source.get("statistics_implementation_revision") != split_manifest.get(
        "materializer_revision"
    ):
        raise SplitError(
            "train_only_statistics_source.statistics_implementation_revision 必须等于 "
            "materializer_revision"
        )
    _require_hex64(source.get("canonical_parquet_sha256"),
                   field="train_only_statistics_source.canonical_parquet_sha256")
    _require_git_sha40(
        source.get("statistics_implementation_revision"),
        field="train_only_statistics_source.statistics_implementation_revision",
    )

    declared = _require_dict(split_manifest.get("train_only_statistics"),
                             field="train_only_statistics")
    if set(declared) != set(STATISTIC_COLUMNS):
        raise SplitError(
            "train_only_statistics 必须精确覆盖六列 "
            f"{list(STATISTIC_COLUMNS)}，实际 {sorted(declared)}"
        )
    train_frame = frame.iloc[spec["row_start"]:spec["row_end_exclusive"]]
    recomputed = train_only_statistics(train_frame)
    for column in STATISTIC_COLUMNS:
        entry = _require_dict(declared.get(column), field=f"train_only_statistics.{column}")
        if set(entry) != set(STATISTIC_FIELDS):
            raise SplitError(
                f"train_only_statistics.{column} 字段必须是 {list(STATISTIC_FIELDS)}，"
                f"实际 {sorted(entry)}"
            )
        for field in ("count", "finite_count", "negative_count"):
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise SplitError(
                    f"train_only_statistics.{column}.{field} 必须是整数（bool 不算），"
                    f"实际 {value!r}"
                )
        for field in ("min", "max", "mean"):
            value = entry[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SplitError(
                    f"train_only_statistics.{column}.{field} 必须是数值标量，实际 {value!r}"
                )
            if not math.isfinite(float(value)):
                raise SplitError(
                    f"train_only_statistics.{column}.{field} 必须有限，实际 {value!r}"
                )
        expected_entry = recomputed[column]
        for field in STATISTIC_FIELDS:
            if entry[field] != expected_entry[field]:
                raise SplitError(
                    f"train_only_statistics.{column}.{field} 与 canonical train 段重算值不符："
                    f"声明={entry[field]!r} 重算={expected_entry[field]!r}"
                )
