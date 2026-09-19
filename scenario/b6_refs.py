"""M1.3f-e-b2-b：**B6 专用**归一化参考值 `refs_v4`（schema `m1.3feb2b-frozen-refs-v1`）。

与 `refs_v3`（`frozen-refs-v2`）的关系：

- **train-only 与物理尺度语义完整保留**：`price_ref = max(abs(price))`、
  `pv_ref_kw = max(local_pv_kw)`、`wind_ref_kw = max(wind_generation_kw)`、
  `carbon_factor_ref = max(carbon_intensity)` 仍由 **train 行实际重算**；
- **绑定的是 B6 链**：exogenous **v3** 驱动表 + policy-v3 + B6 policy；
- **`lambda_ref` 改为 B6 批准的速率**：`63.988 work-units/hour`
  （`main_expected_rate_work_per_hour`，`decision_id = B6-INTENSITY`），
  **不再继承旧的 `2000`**。

## 红线

- **只**从 M1.3d 冻结的 **train 切分**（行 `[0, 10224)`）计算；
  validation / test / 评估**共享同一份**冻结文件，**不得**按测试日重算；
- canonical-only：普通文件 / **非 symlink**；完整**重建比对**；dirty 拒绝；
  原子写入；已存在且不同**拒绝覆盖**；`--verify` 幂等；
- **`refs_v3` 逐字节不变**，且**不得**作为 fallback。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scenario.arrival_intensity_policy import load_verified_b6_policy
from scenario.exogenous_drivers_b6 import load_verified_v3_bundle
from scenario.formal_scenario_b6 import load_verified_policy_v3
from scenario.splits import SplitName, load_truth_split, logical_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent

REFS_SCHEMA = "m1.3feb2b-frozen-refs-v1"
CANONICAL_REFS_NAME = "refs_v4.json"

# canonical 逻辑路径（manifest 中记录**仓库相对**路径）
CANONICAL_PARQUET_LOGICAL = "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST_LOGICAL = "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST_LOGICAL = "data/manifest/singapore_2024_splits.json"
POLICY_V3_LOGICAL = "data/manifest/singapore_2024_forecast_policy_v3.json"
B6_POLICY_LOGICAL = "data/manifest/m13f_arrival_intensity_policy_v1.json"
EXOGENOUS_V3_MANIFEST_LOGICAL = "data/manifest/singapore_2024_exogenous_v3.json"
EXOGENOUS_V3_SOURCE_LOGICAL = "data/manifest/m13f_materialization_sources_v4.json"
EXOGENOUS_V3_PARQUET_LOGICAL = (
    "data/processed/singapore_2024/exogenous_drivers_v3.parquet"
)

# **被取代**的 refs：明确拒绝，不得作为备选或回退
SUPERSEDED_REFS_V3_LOGICAL = "configs/frozen_refs/refs_v3.json"
SUPERSEDED_REFS_V2_LOGICAL = "configs/frozen_refs/refs.json"
SUPERSEDED_REFS_V3_SHA256 = (
    "ab7f5b58f4f49690bc2716732535431bfa2c9efbcd38c842ec16a9b3ada6094f"
)
SUPERSEDED_REFS_V2_SHA256 = (
    "aae5a03e9f09239c4f490b735e4a9ab21d872783a547e7ac6fa9264bafc66827"
)

# B6 批准的速率：**必须**与 policy 的 `main_expected_rate_work_per_hour` 恒等
B6_MAIN_EXPECTED_RATE_WORK_PER_HOUR = 63.988
B6_DECISION_ID = "B6-INTENSITY"
LEGACY_LAMBDA_REF = 2000.0

TRAIN_SPLIT: SplitName = "train"
STEP_MINUTES = 30
TRAIN_RANGE_LABEL = "train"
NOT_APPLICABLE = "not_applicable"
DECLARED_SOURCE_KIND = "declared_physical_scale"
TRAIN_DERIVED_SOURCE_KIND = "train_derived"

# --- 参考值定义 ---------------------------------------------------------------

# train 推导：参考名 → (列, 方法, 单位, 绑定角色, 是否来自 exogenous)
TRAIN_DERIVED_REFS: dict[str, tuple[str, str, str, tuple[str, ...], bool]] = {
    "price_ref": (
        "price_sgd_per_kwh", "max_abs", "SGD/kWh",
        ("canonical_parquet", "canonical_manifest", "split_manifest"), False,
    ),
    "pv_ref_kw": (
        "local_pv_kw", "max", "kW",
        ("canonical_parquet", "split_manifest", "exogenous_parquet"), True,
    ),
    "wind_ref_kw": (
        "wind_generation_kw", "max", "kW",
        ("canonical_parquet", "split_manifest", "exogenous_parquet"), True,
    ),
    "carbon_factor_ref": (
        "carbon_intensity", "max", "kgCO2/kWh",
        ("canonical_parquet", "split_manifest", "exogenous_parquet"), True,
    ),
}
# 声明物理尺度（**不得**由数据推导）；lambda_ref **不在此表**，单独处理
DECLARED_REFS: dict[str, tuple[float, str]] = {
    "queue_ref": (6000.0, "work-units"),
    "queue_capacity_ref": (6000.0, "work-units"),
    "cost_ref": (60.0, "SGD"),
    "carbon_ref": (15.0, "kgCO2"),
    "peak_power_threshold_kW": (18.0, "kW"),
    "peak_power_ref_kW": (10.0, "kW"),
    "grid_power_limit_kW": (18.0, "kW"),
    "sla_penalty_ref": (50.0, "SGD"),
}

REFERENCE_KEYS: tuple[str, ...] = (
    "value", "unit", "source_kind", "method", "derived_from", "training_range",
    "binding", "decision_id",
)
TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version", "materializer_revision", "frozen_at_utc", "training_range",
    "sources", "references", "units", "note",
)
TRAINING_RANGE_KEYS: tuple[str, ...] = (
    "split", "row_start", "row_end_exclusive", "start", "end_exclusive",
)
SOURCE_KEYS: tuple[str, ...] = ("path", "sha256")
NOTE = (
    "B6 train-only 归一化参考值：四个尺度由 canonical+exogenous v3 的 train 行实际"
    "重算；lambda_ref 绑定 B6 批准的 63.988 work-units/hour（decision_id="
    "B6-INTENSITY，**不继承旧的 2000**）；其余保持声明物理尺度。冻结后所有方法共享，"
    "禁止按测试日重算（AGENTS §1.4）。"
)

# 本模块实现文件（dirty 检查与 revision 使用**同一**集合）
B6_REFS_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/b6_refs.py",
    "scenario/formal_scenario_b6.py",
    "scenario/arrival_intensity_policy.py",
    "scenario/exogenous_drivers_b6.py",
    "scenario/splits.py",
)


class RefsV4Error(ValueError):
    """`refs_v4` 的**明确失败**（无 fallback、不静默）。"""


# --- 路径 / Git ----------------------------------------------------------------

def _data_root() -> Path:
    """资产根（**私有**；测试只能 monkeypatch 它，公开 API 不暴露路径覆盖）。"""
    return REPO_ROOT


def _canonical_refs_dir() -> Path:
    return _data_root() / "configs" / "frozen_refs"


def canonical_refs_v4_path() -> Path:
    return _canonical_refs_dir() / CANONICAL_REFS_NAME


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RefsV4Error(message)


def _sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise RefsV4Error(f"冻结资产不可读：{path}：{error}") from error


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", *B6_REFS_SOURCE_PATHS).strip())


def refs_code_revision() -> str:
    """本冻结实现的 revision（由 Git 解析，不用漂移的 HEAD）。"""
    revision = _git("log", "-1", "--format=%H", "--",
                    *B6_REFS_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RefsV4Error(f"materializer revision 无效：{revision!r}")
    return revision


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --- 原子写入 ------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --- 严格校验 helper ------------------------------------------------------------

def _require_exact_keys(mapping: object, *, field: str,
                        expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise RefsV4Error(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise RefsV4Error(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}"
            "（未来扩展必须 bump schema）"
        )
    return mapping


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RefsV4Error(f"{field} 必须是数值（bool 不算），实际 {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise RefsV4Error(f"{field} 必须是有限数值，实际 {value!r}")
    return result


def _require_str(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RefsV4Error(f"{field} 必须是非空字符串，实际 {value!r}")
    return value


def _require_non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RefsV4Error(f"{field} 必须是非负整数，实际 {value!r}")
    return value


# --- 构造 ---------------------------------------------------------------------

def _resolved_inputs() -> dict[str, Path]:
    """B6 链的**唯一**冻结输入（生产路径；**不可由调用者覆盖**）。"""
    root = _data_root()
    return {
        "canonical_parquet": root / CANONICAL_PARQUET_LOGICAL,
        "canonical_manifest": root / CANONICAL_MANIFEST_LOGICAL,
        "split_manifest": root / SPLIT_MANIFEST_LOGICAL,
        "forecast_policy_manifest": root / POLICY_V3_LOGICAL,
        "b6_intensity_policy": root / B6_POLICY_LOGICAL,
        "exogenous_manifest": root / EXOGENOUS_V3_MANIFEST_LOGICAL,
        "exogenous_source_manifest": root / EXOGENOUS_V3_SOURCE_LOGICAL,
        "exogenous_parquet": root / EXOGENOUS_V3_PARQUET_LOGICAL,
    }


def _verify_frozen_chain(inputs: dict[str, Path]) -> None:
    """**实际调用**既有严格链（不只相信自述）。"""
    from scenario.formal_split_manifests import (
        EXPECTED_REFS_V3_SHA256,
        REFS_V3,
    )

    # refs_v3 必须**逐字节不变**（并在缺位时报错）
    if REFS_V3.is_file():
        measured = _sha256_file(REFS_V3)
        if measured != EXPECTED_REFS_V3_SHA256:
            raise RefsV4Error(
                "refs_v3 的字节已改变（它必须逐字节保留）："
                f"期望 {EXPECTED_REFS_V3_SHA256} 实际 {measured}"
            )

    load_truth_split(
        TRAIN_SPLIT,
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
    )
    load_verified_policy_v3(inputs["forecast_policy_manifest"])
    policy = load_verified_b6_policy()
    if float(policy["main_expected_rate_work_per_hour"]) != \
            B6_MAIN_EXPECTED_RATE_WORK_PER_HOUR:
        raise RefsV4Error(
            "B6 policy 的 main_expected_rate_work_per_hour 与冻结 63.988 不符"
        )
    load_verified_v3_bundle()


def build_refs_v4(*, frozen_at_utc: str) -> dict:
    """构造候选 `refs_v4`（**公开签名只接受冻结时间戳**，没有路径/信任根参数）。"""
    inputs = _resolved_inputs()
    _verify_frozen_chain(inputs)

    rows = load_truth_split(
        TRAIN_SPLIT,
        canonical_parquet_path=inputs["canonical_parquet"],
        canonical_manifest_path=inputs["canonical_manifest"],
        split_manifest_path=inputs["split_manifest"],
    )
    canonical = pd.read_parquet(inputs["canonical_parquet"])
    exogenous = pd.read_parquet(inputs["exogenous_parquet"])
    _require_aligned_chain(canonical, exogenous)
    train = canonical.iloc[: len(rows)]
    exog = exogenous.iloc[: len(train)]

    stamps = pd.DatetimeIndex(train["timestamp"])
    training_range = {
        "split": TRAIN_SPLIT,
        "row_start": 0,
        "row_end_exclusive": int(len(train)),
        "start": stamps[0].isoformat(),
        "end_exclusive": (stamps[-1] + pd.Timedelta(minutes=STEP_MINUTES)).isoformat(),
    }

    sources = {
        role: {"path": logical_repo_path(path), "sha256": _sha256_file(path)}
        for role, path in inputs.items()
    }

    references: dict[str, dict[str, Any]] = {
        "lambda_ref": {
            "value": B6_MAIN_EXPECTED_RATE_WORK_PER_HOUR,
            "unit": "work-units/hour",
            "source_kind": DECLARED_SOURCE_KIND,
            "method": "b6_approved_main_expected_rate",
            "derived_from": "b6_intensity_policy",
            "training_range": NOT_APPLICABLE,
            "binding": ["b6_intensity_policy", "forecast_policy_manifest"],
            "decision_id": B6_DECISION_ID,
        },
    }
    for name, (column, method, unit, binding, from_exog) in TRAIN_DERIVED_REFS.items():
        frame = exog if from_exog else train
        references[name] = {
            "value": _train_derived_value(frame, column=column, method=method, name=name),
            "unit": unit,
            "source_kind": TRAIN_DERIVED_SOURCE_KIND,
            "method": method,
            "derived_from": column,
            "training_range": TRAIN_RANGE_LABEL,
            "binding": list(binding),
            "decision_id": None,
        }
    for name, (value, unit) in DECLARED_REFS.items():
        references[name] = {
            "value": float(value),
            "unit": unit,
            "source_kind": DECLARED_SOURCE_KIND,
            "method": "declared_physical_scale",
            "derived_from": "declared_physical_scale",
            "training_range": NOT_APPLICABLE,
            "binding": [],
            "decision_id": None,
        }

    payload = {
        "schema_version": REFS_SCHEMA,
        "materializer_revision": refs_code_revision(),
        "frozen_at_utc": frozen_at_utc,
        "training_range": training_range,
        "sources": sources,
        "references": references,
        "units": {name: entry["unit"] for name, entry in references.items()},
        "note": NOTE,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise RefsV4Error("refs_v4 含绝对路径")
    return validate_refs_v4(payload)


def _require_aligned_chain(canonical: pd.DataFrame, exogenous: pd.DataFrame) -> None:
    """`iloc` 切片**之前**的严格校验：长度、时间轴、逐行时间戳、列与有限性。"""
    if len(canonical) != len(exogenous):
        raise RefsV4Error(
            f"canonical 与 exogenous 行数不同：{len(canonical)} != {len(exogenous)}"
        )
    left = _require_tz_index(canonical, label="canonical")
    right = _require_tz_index(exogenous, label="exogenous")
    if not left.equals(right):
        raise RefsV4Error("canonical 与 exogenous 的 timestamp 必须**逐行完全相等**")
    for name, (column, _method, _unit, _binding, from_exog) in TRAIN_DERIVED_REFS.items():
        frame = exogenous if from_exog else canonical
        if column not in frame.columns:
            raise RefsV4Error(f"{name} 的列 {column!r} 不在其来源帧中")
        values = np.asarray(frame[column].to_numpy(), dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise RefsV4Error(f"{name} 的列必须非空且全部有限")


def _require_tz_index(frame: pd.DataFrame, *, label: str) -> pd.DatetimeIndex:
    if "timestamp" not in frame.columns:
        raise RefsV4Error(f"{label} 缺少 timestamp 列")
    stamps = pd.DatetimeIndex(frame["timestamp"])
    if stamps.tz is None:
        raise RefsV4Error(f"{label} 的 timestamp 必须是 tz-aware")
    return stamps


def _train_derived_value(frame: pd.DataFrame, *, column: str, method: str,
                         name: str) -> float:
    _require(column in frame.columns, f"{name} 的列 {column!r} 不在 train 帧中")
    values = np.asarray(frame[column].to_numpy(), dtype=float)
    _require(values.size > 0, f"{name} 的 train 列为空")
    if not np.isfinite(values).all():
        raise RefsV4Error(f"{name} 的 train 列含非有限值")
    if method == "max_abs":
        return float(np.abs(values).max())
    if method == "max":
        return float(values.max())
    raise RefsV4Error(f"未知的推导方法 {method!r}")


# --- 严格校验 -----------------------------------------------------------------

def validate_refs_v4(payload: object) -> dict:
    """`refs_v4` 的**唯一**严格校验入口（构造 / 读取共用）。"""
    payload = _require_exact_keys(payload, field="refs_v4", expected=TOP_LEVEL_KEYS)
    if payload["schema_version"] != REFS_SCHEMA:
        raise RefsV4Error(
            f"schema_version 必须是 {REFS_SCHEMA!r}，实际 {payload['schema_version']!r}"
        )
    revision = payload["materializer_revision"]
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise RefsV4Error(f"materializer_revision 必须是 40 位小写 Git SHA，实际 {revision!r}")
    frozen_at = payload["frozen_at_utc"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("+00:00"):
        raise RefsV4Error(f"frozen_at_utc 必须是规范 UTC ISO-8601，实际 {frozen_at!r}")
    try:
        parsed = datetime.fromisoformat(frozen_at)
    except ValueError as error:
        raise RefsV4Error(f"frozen_at_utc 不是合法 ISO-8601：{frozen_at!r}") from error
    if parsed.tzinfo is None or parsed.isoformat() != frozen_at:
        raise RefsV4Error(f"frozen_at_utc 不是规范 UTC 形式：{frozen_at!r}")

    training_range = _require_exact_keys(
        payload["training_range"], field="training_range",
        expected=TRAINING_RANGE_KEYS)
    if training_range["split"] != TRAIN_SPLIT:
        raise RefsV4Error("training_range.split 必须是 'train'")
    _require_non_negative_int(training_range["row_start"], field="training_range.row_start")
    _require_non_negative_int(training_range["row_end_exclusive"],
                              field="training_range.row_end_exclusive")
    _require(training_range["row_start"] < training_range["row_end_exclusive"],
             "training_range 的行区间必须非空")
    for field in ("start", "end_exclusive"):
        _require_str(training_range[field], field=f"training_range.{field}")

    sources = _require_exact_keys(payload["sources"], field="sources",
                                  expected=tuple(_resolved_inputs()))
    for role, entry in sources.items():
        _require_exact_keys(entry, field=f"sources.{role}", expected=SOURCE_KEYS)
        path = _require_str(entry["path"], field=f"sources.{role}.path")
        if path.startswith("/") or "\\" in path:
            raise RefsV4Error(f"sources.{role}.path 必须是仓库相对 POSIX 路径：{path!r}")
        sha = _require_str(entry["sha256"], field=f"sources.{role}.sha256")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise RefsV4Error(f"sources.{role}.sha256 必须是 64 位小写十六进制")

    references = payload["references"]
    if not isinstance(references, dict) or not references:
        raise RefsV4Error("references 必须是非空 object")
    expected_names = {"lambda_ref"} | set(TRAIN_DERIVED_REFS) | set(DECLARED_REFS)
    if set(references) != expected_names:
        raise RefsV4Error(
            f"references 的名称集合必须精确等于冻结集合；"
            f"多出={sorted(set(references) - expected_names)} "
            f"缺少={sorted(expected_names - set(references))}"
        )
    for name, entry in references.items():
        _require_exact_keys(entry, field=f"references.{name}", expected=REFERENCE_KEYS)
        _require_finite_number(entry["value"], field=f"references.{name}.value")
        for field in ("unit", "method", "derived_from"):
            _require_str(entry[field], field=f"references.{name}.{field}")
        kind = entry["source_kind"]
        if kind == TRAIN_DERIVED_SOURCE_KIND:
            expected = TRAIN_DERIVED_REFS[name]
            if entry["derived_from"] != expected[0] or entry["method"] != expected[1]:
                raise RefsV4Error(f"references.{name} 的 train 推导口径不符")
            if entry["training_range"] != TRAIN_RANGE_LABEL:
                raise RefsV4Error(f"references.{name}.training_range 必须是 'train'")
            binding = entry["binding"]
            if not isinstance(binding, list) or not binding:
                raise RefsV4Error(f"references.{name}.binding 必须是非空 list")
            for role in binding:
                if role not in sources:
                    raise RefsV4Error(f"references.{name} 绑定了未知来源角色 {role!r}")
            if entry["decision_id"] is not None:
                raise RefsV4Error(f"references.{name}.decision_id 必须为 null")
        elif kind == DECLARED_SOURCE_KIND:
            if name not in DECLARED_REFS and name != "lambda_ref":
                raise RefsV4Error(f"{name} 不得声明为 {DECLARED_SOURCE_KIND!r}")
            if entry["training_range"] != NOT_APPLICABLE:
                raise RefsV4Error(
                    f"references.{name}.training_range 必须是 {NOT_APPLICABLE!r}")
            if not isinstance(entry["binding"], list):
                raise RefsV4Error(f"references.{name}.binding 必须是 list")
            if name == "lambda_ref":
                if entry["value"] != B6_MAIN_EXPECTED_RATE_WORK_PER_HOUR:
                    raise RefsV4Error(
                        "lambda_ref 必须**显式绑定** B6 批准的 "
                        f"{B6_MAIN_EXPECTED_RATE_WORK_PER_HOUR} work-units/hour，"
                        f"实际 {entry['value']!r}"
                    )
                if entry["value"] == LEGACY_LAMBDA_REF:
                    raise RefsV4Error("lambda_ref **不得**继承旧的 2000")
                if entry["decision_id"] != B6_DECISION_ID:
                    raise RefsV4Error(
                        f"lambda_ref.decision_id 必须是 {B6_DECISION_ID!r}")
                if entry["unit"] != "work-units/hour":
                    raise RefsV4Error("lambda_ref.unit 必须是 'work-units/hour'")
            elif entry["binding"] != [] or entry["decision_id"] is not None:
                raise RefsV4Error(f"references.{name} 必须是纯声明尺度")
        else:
            raise RefsV4Error(f"未知 source_kind：{kind!r}")

    units = payload["units"]
    if not isinstance(units, dict) or set(units) != set(references):
        raise RefsV4Error("units 的键集合必须与 references 精确一致")
    for name, entry in references.items():
        if units[name] != entry["unit"]:
            raise RefsV4Error(f"units.{name} 与逐值 unit 不一致")
    _require_str(payload["note"], field="note")
    return payload


# --- 唯一加载入口 --------------------------------------------------------------

def _require_canonical_location(path: Path | str | None) -> Path:
    """路径必须**精确等于** canonical `refs_v4.json`（在读 JSON 之前）。

    比较**词法绝对路径**（`absolute()` 不做符号链接解析）：副本、别名、symlink
    一律拒绝；**refs_v3 / refs.json 明确拒绝**（无 fallback）。
    """
    given = (Path(path).absolute() if path is not None
             else canonical_refs_v4_path().absolute())
    expected = canonical_refs_v4_path().absolute()
    if given == expected:
        if expected.is_symlink():
            raise RefsV4Error(f"refs_v4 不得是 symlink：{expected}")
        return expected
    superseded = {
        (_data_root() / SUPERSEDED_REFS_V3_LOGICAL).absolute(): "refs_v3",
        (_data_root() / SUPERSEDED_REFS_V2_LOGICAL).absolute(): "refs.json (v2)",
    }
    if given in superseded:
        raise RefsV4Error(
            f"{superseded[given]} 是 superseded 的 refs："
            "正式 B6 链只接受 refs_v4.json（无 fallback）"
        )
    raise RefsV4Error(
        f"refs 路径必须是**唯一 canonical** 文件 "
        f"{logical_repo_path(expected)}；实际 {logical_repo_path(given)}"
        "（不接受副本、别名、symlink 或其它生成版本）"
    )


def load_verified_refs_v4(path: Path | str | None = None) -> dict:
    """读取并**严格校验** canonical `refs_v4`。

    先做 canonical 位置 / 非 symlink 检查，再**整体重建比对**：
    文件必须逐字段等于由 trusted inputs + live hashes + live revision
    重建的 candidate。任一不符 fail closed。**没有**任何信任边界参数。
    """
    path = _require_canonical_location(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RefsV4Error(f"{path} 不可读或不是合法 JSON：{error}") from error
    try:
        validated = validate_refs_v4(payload)
    except RefsV4Error:
        raise
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise RefsV4Error(f"{path} 结构畸形：{error}") from error

    if _generator_is_dirty():
        raise RefsV4Error(
            "B6 refs 实现有未提交修改：拒绝用旧 revision 为未提交代码背书"
        )
    live = refs_code_revision()
    if validated["materializer_revision"] != live:
        raise RefsV4Error(
            f"materializer_revision 必须是**当前**实现的 revision {live}；"
            f"实际 {validated['materializer_revision']}"
        )
    rebuilt = build_refs_v4(frozen_at_utc=validated["frozen_at_utc"])
    if rebuilt != validated:
        differing = sorted(
            key for key in set(rebuilt) | set(validated)
            if rebuilt.get(key) != validated.get(key)
        )
        raise RefsV4Error(
            f"refs_v4 与由 live hashes + revision 重建的 candidate 不符；"
            f"差异字段={differing}"
        )
    return validated


# --- 物化 ---------------------------------------------------------------------

def materialize_refs_v4(*, frozen_at_utc: str | None = None) -> dict:
    """生成 / 校验 canonical `refs_v4.json`（原子、幂等、拒绝覆盖）。"""
    out_path = canonical_refs_v4_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if _generator_is_dirty():
        raise RefsV4Error("生成器有未提交修改：拒绝用旧 revision 为未提交代码背书")

    existing_frozen = None
    if out_path.is_file():
        existing = load_verified_refs_v4(out_path)
        existing_frozen = existing["frozen_at_utc"]
    candidate = build_refs_v4(
        frozen_at_utc=frozen_at_utc or existing_frozen or _now_utc())

    if out_path.is_file():
        existing = load_verified_refs_v4(out_path)
        if existing != candidate:
            raise RefsV4Error(f"{out_path} 与候选**语义不同**：拒绝覆盖已冻结的 refs_v4")
        return {"out_path": out_path,
                "sha256": _sha256_file(out_path), "written": False}

    _atomic_write_text(out_path, _canonical_json(candidate))
    return {"out_path": out_path, "sha256": _sha256_file(out_path), "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结 B6 归一化参考值（M1.3f-e-b2-b）")
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    parser.add_argument("--frozen-at-utc", default=None)
    args = parser.parse_args(argv)

    try:
        if args.verify:
            refs = load_verified_refs_v4()
            print(f"verified {canonical_refs_v4_path()}")
            print(f"lambda_ref={json.dumps(refs['references']['lambda_ref'], sort_keys=True)}")
            return 0
        result = materialize_refs_v4(frozen_at_utc=args.frozen_at_utc)
    except (RefsV4Error, OSError, ValueError) as error:
        print(f"materialize_b6_refs: {error}", file=sys.stderr)
        return 1

    print(f"out_path={result['out_path']}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
