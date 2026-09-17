#!/usr/bin/env python
"""M1.3g-d：冻结归一化参考值（**train-only**，schema `frozen-refs-v2`）。

规则（D2 / D6，人工裁决）：

- **由 train 行推导**（`source_kind = "train_derived"`）：
  `price_ref = max(abs(price_sgd_per_kwh))`、`pv_ref_kw = max(local_pv_kw)`、
  `wind_ref_kw = max(wind_generation_kw)`、`carbon_factor_ref = max(carbon_intensity)`；
- **保持声明物理尺度**（`source_kind = "declared_physical_scale"`）：
  `lambda_ref=2000`、`queue_ref=6000`、`queue_capacity_ref=6000`、`cost_ref=60`，
  以及 carbon_ref、硬件上限、SLA 等既有声明值。

红线：

- **只**从 M1.3d 冻结的 **train 切分** 计算（行 `[0, 10224)`），
  validation / test / 评估**共享同一份**冻结文件，**不得**按测试日重算；
- **不得**把 M1.3d 的 `train_only_statistics` 直接冒充 refs —— 这里**实际读取并重算**；
- 上游链走**严格验证**（canonical + split + exogenous v2/v3），任一不符即 fail closed；
- **生成器源码有未提交改动时拒绝物化**；
- 首次把 **legacy v1** 升级为 v2 必须**显式** `--replace-declared-v1`，
  且仅当其字节 hash 恰等于已登记值时允许；**原子**替换；
- v2 已存在且**相同**则不改 `mtime_ns`；**不同**则拒绝覆盖。

用法：

```bash
# 首次受控升级（legacy v1 → v2）
uv run python scripts/freeze_refs.py --replace-declared-v1
# 校验 / 幂等重跑
uv run python scripts/freeze_refs.py --verify
```
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.formal_scenario import (
    EXOGENOUS_OUTPUT_LOGICAL_PATH,
    load_verified_exogenous,
)
from scenario.splits import SplitName, load_truth_split, logical_repo_path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "configs/frozen_refs/refs.json"

REF_SCHEMA = "frozen-refs-v2"
LEGACY_SCHEMA = "frozen-refs-v1"
# legacy v1 的**登记**字节 hash（仅此一份允许被受控替换）
LEGACY_V1_SHA256 = (
    "afa84b8610c073322aac41b3d7e5c706478c64a551363bd66c42b1f22fd409bc"
)

# 生成器实现（dirty 检查与 revision 使用**同一**集合）
GENERATOR_SOURCE_PATHS: tuple[str, ...] = (
    "scripts/freeze_refs.py",
    "scenario/splits.py",
    "scenario/formal_scenario.py",
)

CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
EXOGENOUS_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous_v2.json"
EXOGENOUS_SOURCE = REPO_ROOT / "data/manifest/m13f_materialization_sources_v3.json"
TRAIN_SPLIT: SplitName = "train"

# --- 声明物理尺度（**不得**由数据推导） -------------------------------------
DECLARED_REFS: dict[str, float] = {
    "lambda_ref": 2000.0,
    "queue_ref": 6000.0,
    "queue_capacity_ref": 6000.0,
    "cost_ref": 60.0,
    "carbon_ref": 15.0,
    "peak_power_threshold_kW": 18.0,
    "peak_power_ref_kW": 10.0,
    "grid_power_limit_kW": 18.0,
    "sla_penalty_ref": 50.0,
}
DECLARED_UNITS: dict[str, str] = {
    "lambda_ref": "work-units/hour",
    "queue_ref": "work-units",
    "queue_capacity_ref": "work-units",
    "cost_ref": "SGD",
    "carbon_ref": "kgCO2",
    "peak_power_threshold_kW": "kW",
    "peak_power_ref_kW": "kW",
    "grid_power_limit_kW": "kW",
    "sla_penalty_ref": "SGD",
}

# --- train 推导值：参考名 → (列, 方法, 单位, 绑定角色) ------------------------
TRAIN_DERIVED_REFS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "price_ref": (
        "price_sgd_per_kwh", "max_abs", "SGD/kWh",
        ("canonical_parquet", "canonical_manifest", "split_manifest"),
    ),
    "pv_ref_kw": (
        "local_pv_kw", "max", "kW",
        ("canonical_parquet", "split_manifest", "exogenous_parquet"),
    ),
    "wind_ref_kw": (
        "wind_generation_kw", "max", "kW",
        ("canonical_parquet", "split_manifest", "exogenous_parquet"),
    ),
    "carbon_factor_ref": (
        "carbon_intensity", "max", "kgCO2/kWh",
        ("canonical_parquet", "split_manifest", "exogenous_parquet"),
    ),
}
DECLARED_SOURCE_KIND = "declared_physical_scale"
TRAIN_DERIVED_SOURCE_KIND = "train_derived"
DECLARED_DERIVED_FROM = "declared_physical_scale"
NOT_APPLICABLE = "not_applicable"
TRAIN_RANGE_LABEL = "train"

REFERENCE_KEYS: tuple[str, ...] = (
    "value", "unit", "source_kind", "method", "derived_from", "training_range",
    "binding",
)
TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version", "materializer_revision", "frozen_at_utc", "training_range",
    "sources", "references", "units", "note",
)
TRAINING_RANGE_KEYS: tuple[str, ...] = (
    "split", "row_start", "row_end_exclusive", "start", "end_exclusive",
)
NOTE = (
    "train-only 归一化参考值（D2/D6）：四个尺度由 train 行实际重算，其余保持声明"
    "物理尺度；冻结后所有方法共享，禁止按测试日重算（AGENTS §1.4）。"
)


class FreezeRefsError(ValueError):
    """refs 冻结的**明确失败**。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FreezeRefsError(message)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise FreezeRefsError(f"上游资产不可读：{path}：{error}") from error


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    """生成器实现是否有未提交修改（含未跟踪的新文件）。"""
    return bool(_git("status", "--porcelain", "--", *GENERATOR_SOURCE_PATHS).strip())


def resolve_materializer_revision() -> str:
    """本冻结实现的 revision（由 Git 解析，不用漂移的 HEAD）。"""
    revision = _git("log", "-1", "--format=%H", "--", *GENERATOR_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise FreezeRefsError(f"materializer revision 无效：{revision!r}")
    return revision


# --- 原子写入 ----------------------------------------------------------------

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


# --- 构造 ---------------------------------------------------------------------

def build_frozen_refs(
    *,
    canonical_parquet_path: Path | str = CANONICAL_PARQUET,
    canonical_manifest_path: Path | str = CANONICAL_MANIFEST,
    split_manifest_path: Path | str = SPLIT_MANIFEST,
    exogenous_manifest_path: Path | str = EXOGENOUS_MANIFEST,
    exogenous_source_manifest_path: Path | str = EXOGENOUS_SOURCE,
    frozen_at_utc: str | None = None,
    canonical_frame: pd.DataFrame | None = None,
) -> dict:
    """构造候选 `frozen-refs-v2`（**只读**上游，逐层严格校验）。

    `canonical_frame` 仅供测试注入一份**已修改**的 canonical 帧；
    生产路径从不传它（因此仍走严格链校验）。
    """
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)
    exogenous_manifest_path = Path(exogenous_manifest_path)
    exogenous_source_manifest_path = Path(exogenous_source_manifest_path)

    # 1) split → canonical manifest → canonical parquet 的**完整严格校验**
    #    （含整条时间轴、精确键集合、train-only 统计重算）
    split_rows = load_truth_split(
        TRAIN_SPLIT,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    _require(len(split_rows) > 0, "train 切分为空，拒绝冻结 refs")

    # 2) exogenous v2 / source v3 的逐字节核验 + 交叉绑定
    exogenous_parquet_path = REPO_ROOT / EXOGENOUS_OUTPUT_LOGICAL_PATH
    load_verified_exogenous(
        exogenous_manifest_path,
        exogenous_source_manifest_path,
        exogenous_parquet_path=exogenous_parquet_path,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )

    # 3) 只读 **train 行**（用注入帧时同样只取 train 段）
    frame = (pd.read_parquet(canonical_parquet_path)
             if canonical_frame is None else canonical_frame)
    train = frame.iloc[: len(split_rows)]
    _require(len(train) == len(split_rows), "canonical 行数不足以覆盖 train 切分")
    exog = pd.read_parquet(exogenous_parquet_path).iloc[: len(train)]

    stamps = pd.DatetimeIndex(train["timestamp"])
    training_range = {
        "split": TRAIN_SPLIT,
        "row_start": 0,
        "row_end_exclusive": int(len(train)),
        "start": stamps[0].isoformat(),
        "end_exclusive": (stamps[-1] + pd.Timedelta(minutes=30)).isoformat(),
    }

    sources = {
        role: {"path": logical_repo_path(path), "sha256": _sha256_file(path)}
        for role, path in (
            ("canonical_parquet", canonical_parquet_path),
            ("canonical_manifest", canonical_manifest_path),
            ("split_manifest", split_manifest_path),
            ("exogenous_manifest", exogenous_manifest_path),
            ("exogenous_source_manifest", exogenous_source_manifest_path),
            ("exogenous_parquet", exogenous_parquet_path),
        )
    }

    references: dict[str, dict[str, Any]] = {}
    for name, (column, method, unit, binding) in TRAIN_DERIVED_REFS.items():
        references[name] = {
            "value": _train_derived_value(
                train if column in train.columns else exog,
                column=column, method=method, name=name,
            ),
            "unit": unit,
            "source_kind": TRAIN_DERIVED_SOURCE_KIND,
            "method": method,
            "derived_from": column,
            "training_range": TRAIN_RANGE_LABEL,
            "binding": list(binding),
        }
    for name, value in DECLARED_REFS.items():
        references[name] = {
            "value": float(value),
            "unit": DECLARED_UNITS[name],
            "source_kind": DECLARED_SOURCE_KIND,
            "method": "declared_physical_scale",
            "derived_from": DECLARED_DERIVED_FROM,
            "training_range": NOT_APPLICABLE,
            "binding": [],
        }

    payload = {
        "schema_version": REF_SCHEMA,
        "materializer_revision": resolve_materializer_revision(),
        "frozen_at_utc": frozen_at_utc or _now_utc(),
        "training_range": training_range,
        "sources": sources,
        "references": references,
        "units": {name: entry["unit"] for name, entry in references.items()},
        "note": NOTE,
    }
    return validate_frozen_refs(payload)


def _train_derived_value(
    frame: pd.DataFrame, *, column: str, method: str, name: str
) -> float:
    _require(column in frame.columns, f"{name} 的列 {column!r} 不在 train 帧中")
    values = np.asarray(frame[column].to_numpy(), dtype=float)
    _require(values.size > 0, f"{name} 的 train 列为空")
    if not np.isfinite(values).all():
        raise FreezeRefsError(f"{name} 的 train 列含非有限值")
    if method == "max_abs":
        return float(np.abs(values).max())
    if method == "max":
        return float(values.max())
    raise FreezeRefsError(f"未知的推导方法 {method!r}")


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --- 严格校验 -----------------------------------------------------------------

def _require_exact_keys(mapping: object, *, field: str, expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise FreezeRefsError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise FreezeRefsError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}"
            "（未来扩展必须 bump schema）"
        )
    return mapping


def _require_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeRefsError(f"{field} 必须是数值（bool 不算），实际 {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise FreezeRefsError(f"{field} 必须是有限数值，实际 {value!r}")
    return result


def validate_frozen_refs(payload: object) -> dict:
    """`frozen-refs-v2` 的**唯一**严格校验入口（构造、读取、替换共用）。"""
    payload = _require_exact_keys(payload, field="frozen refs", expected=TOP_LEVEL_KEYS)
    if payload["schema_version"] != REF_SCHEMA:
        raise FreezeRefsError(
            f"schema_version 必须是 {REF_SCHEMA!r}，实际 {payload['schema_version']!r}"
        )
    revision = payload["materializer_revision"]
    if (not isinstance(revision, str) or len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)):
        raise FreezeRefsError(f"materializer_revision 必须是 40 位小写 Git SHA，实际 {revision!r}")
    frozen_at = payload["frozen_at_utc"]
    if not isinstance(frozen_at, str) or not frozen_at.endswith("+00:00"):
        raise FreezeRefsError(f"frozen_at_utc 必须是规范 UTC ISO-8601，实际 {frozen_at!r}")
    try:
        parsed = datetime.fromisoformat(frozen_at)
    except ValueError as error:
        raise FreezeRefsError(f"frozen_at_utc 不是合法 ISO-8601：{frozen_at!r}") from error
    if parsed.tzinfo is None or parsed.isoformat() != frozen_at:
        raise FreezeRefsError(f"frozen_at_utc 不是规范 UTC 形式：{frozen_at!r}")

    training_range = _require_exact_keys(
        payload["training_range"], field="training_range",
        expected=TRAINING_RANGE_KEYS,
    )
    if training_range["split"] != TRAIN_SPLIT:
        raise FreezeRefsError(
            f"training_range.split 必须是 {TRAIN_SPLIT!r}，实际 {training_range['split']!r}"
        )
    for field in ("row_start", "row_end_exclusive"):
        value = training_range[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FreezeRefsError(f"training_range.{field} 必须是非负整数，实际 {value!r}")
    _require(
        training_range["row_start"] < training_range["row_end_exclusive"],
        "training_range 的行区间必须非空",
    )
    for field in ("start", "end_exclusive"):
        if not isinstance(training_range[field], str) or not training_range[field]:
            raise FreezeRefsError(f"training_range.{field} 必须是非空字符串")

    sources = payload["sources"]
    if not isinstance(sources, dict) or not sources:
        raise FreezeRefsError("sources 必须是非空 object")
    for role, entry in sources.items():
        _require_exact_keys(entry, field=f"sources.{role}", expected=("path", "sha256"))
        path, sha = entry["path"], entry["sha256"]
        if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
            raise FreezeRefsError(f"sources.{role}.path 必须是仓库相对 POSIX 路径：{path!r}")
        if (not isinstance(sha, str) or len(sha) != 64
                or any(c not in "0123456789abcdef" for c in sha)):
            raise FreezeRefsError(f"sources.{role}.sha256 必须是 64 位小写十六进制")

    references = payload["references"]
    if not isinstance(references, dict) or not references:
        raise FreezeRefsError("references 必须是非空 object")
    expected_names = set(TRAIN_DERIVED_REFS) | set(DECLARED_REFS)
    if set(references) != expected_names:
        raise FreezeRefsError(
            f"references 的名称集合必须精确等于冻结集合；"
            f"多出={sorted(set(references) - expected_names)} "
            f"缺少={sorted(expected_names - set(references))}"
        )
    for name, entry in references.items():
        _require_exact_keys(entry, field=f"references.{name}", expected=REFERENCE_KEYS)
        _require_finite_number(entry["value"], field=f"references.{name}.value")
        for field in ("unit", "method", "derived_from"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise FreezeRefsError(
                    f"references.{name}.{field} 必须是非空字符串，实际 {entry[field]!r}"
                )
        kind = entry["source_kind"]
        if kind == TRAIN_DERIVED_SOURCE_KIND:
            expected = TRAIN_DERIVED_REFS[name]
            if entry["derived_from"] != expected[0] or entry["method"] != expected[1]:
                raise FreezeRefsError(
                    f"references.{name} 的 train 推导口径必须是 "
                    f"{expected[0]!r}/{expected[1]!r}"
                )
            if entry["training_range"] != TRAIN_RANGE_LABEL:
                raise FreezeRefsError(
                    f"references.{name}.training_range 必须是 {TRAIN_RANGE_LABEL!r}"
                )
            binding = entry["binding"]
            if not isinstance(binding, list) or not binding:
                raise FreezeRefsError(f"references.{name}.binding 必须是非空 list")
            for role in binding:
                if role not in sources:
                    raise FreezeRefsError(
                        f"references.{name} 绑定了未知来源角色 {role!r}"
                    )
        elif kind == DECLARED_SOURCE_KIND:
            if name not in DECLARED_REFS:
                raise FreezeRefsError(f"{name} 不得声明为 {DECLARED_SOURCE_KIND!r}")
            if entry["derived_from"] != DECLARED_DERIVED_FROM:
                raise FreezeRefsError(f"references.{name}.derived_from 必须是声明值标记")
            if entry["method"] != "declared_physical_scale":
                raise FreezeRefsError(f"references.{name}.method 必须是声明值标记")
            if entry["training_range"] != NOT_APPLICABLE:
                raise FreezeRefsError(
                    f"references.{name}.training_range 必须是 {NOT_APPLICABLE!r}"
                )
            if entry["binding"] != []:
                raise FreezeRefsError(f"references.{name}.binding 必须为空 list")
        else:
            raise FreezeRefsError(f"未知 source_kind：{kind!r}")

    units = payload["units"]
    if not isinstance(units, dict) or set(units) != set(references):
        raise FreezeRefsError("units 的键集合必须与 references 精确一致")
    for name, entry in references.items():
        if units[name] != entry["unit"]:
            raise FreezeRefsError(f"units.{name} 与逐值 unit 不一致")
    if not isinstance(payload["note"], str) or not payload["note"]:
        raise FreezeRefsError("note 必须是非空字符串")
    return payload


def load_frozen_refs(path: Path | str = OUT_PATH) -> dict:
    """读取并**严格校验** v2 refs（畸形输入只抛 `FreezeRefsError`）。"""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeRefsError(f"{path} 不可读或不是合法 JSON：{error}") from error
    return validate_frozen_refs(payload)


# --- 物化 ---------------------------------------------------------------------

def _existing_kind(path: Path) -> str:
    """既有文件的类别：`"v2"` / `"legacy_v1"` / `"absent"`；其它一律拒绝。"""
    if not path.is_file():
        return "absent"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == LEGACY_V1_SHA256:
        return "legacy_v1"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeRefsError(f"既有的 {path} 不是合法 JSON：{error}") from error
    if not isinstance(payload, dict):
        raise FreezeRefsError(f"既有的 {path} 顶层必须是 object")
    schema = payload.get("schema_version")
    if schema != REF_SCHEMA:
        raise FreezeRefsError(
            f"既有的 {path} 的 schema_version 是未知的 {schema!r}：拒绝覆盖"
        )
    return "v2"


def materialize_frozen_refs(
    *,
    out_path: Path | str = OUT_PATH,
    replace_declared_v1: bool = False,
    **build_kwargs: Any,
) -> dict:
    """生成 / 校验 `configs/frozen_refs/refs.json`（原子、幂等、受控首替）。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise FreezeRefsError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书（未提交）"
        )

    kind = _existing_kind(out_path)
    if kind == "legacy_v1" and not replace_declared_v1:
        raise FreezeRefsError(
            f"{out_path} 仍是 legacy {LEGACY_SCHEMA}（声明尺度）："
            "首次升级必须显式 --replace-declared-v1"
        )
    if kind == "v2" and replace_declared_v1:
        raise FreezeRefsError(
            f"{out_path} 已是 {REF_SCHEMA}：--replace-declared-v1 只用于 legacy v1 首次升级"
        )

    existing_frozen_at = (
        load_frozen_refs(out_path)["frozen_at_utc"] if kind == "v2" else None
    )
    candidate = build_frozen_refs(frozen_at_utc=existing_frozen_at, **build_kwargs)
    text = _canonical_json(candidate)

    if kind == "v2":
        existing = load_frozen_refs(out_path)
        if existing != candidate:
            raise FreezeRefsError(
                f"{out_path} 与候选**语义不同**：拒绝覆盖已冻结的 v2 refs"
            )
        return {
            "out_path": out_path,
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "written": False,
        }

    _atomic_write_text(out_path, text)
    return {
        "out_path": out_path,
        "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
        "written": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结 train-only 归一化参考值（M1.3g-d）")
    parser.add_argument("--replace-declared-v1", action="store_true",
                        help="显式受控：仅当既有 refs 恰为登记的 legacy v1 时替换为 v2")
    parser.add_argument("--verify", action="store_true",
                        help="只校验 / 幂等重跑（不改变既有文件）")
    parser.add_argument("--out-path", default=str(OUT_PATH))
    args = parser.parse_args(argv)

    try:
        result = materialize_frozen_refs(
            out_path=Path(args.out_path),
            replace_declared_v1=args.replace_declared_v1,
        )
    except (FreezeRefsError, OSError, ValueError) as error:
        print(f"freeze_refs: {error}", file=sys.stderr)
        return 1

    print(f"out_path={result['out_path']}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    if args.verify:
        refs = load_frozen_refs(args.out_path)
        print(f"references={json.dumps(refs['references'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
