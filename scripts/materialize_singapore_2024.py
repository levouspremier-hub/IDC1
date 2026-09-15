#!/usr/bin/env python
"""M1.3b：把 Singapore-2024 半小时 canonical 事实表物化到本地并写可审计 manifest。

**本地派生**：Parquet 落在 `data/processed/singapore_2024/half_hour.parquet`
（被 gitignore，不入库）；入库的只有
`data/manifest/singapore_2024_half_hour.json`。

纪律：

- **不得**把运行时当前时间静默写进可复现内容；首次物化必须**显式**提供冻结时间
  （`--frozen-at-utc`）。
- 已有 canonical manifest 只能做**只读一致性验证**；内容不同即失败，**不得覆盖**。
- 本脚本不改动 source manifest、raw 文件或任何 M1.2 冻结物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from runs.writer import git_revision
from scenario.singapore_2024 import (
    ALLOWED_WEATHER_AGES_MINUTES,
    CANONICAL_COLUMNS,
    COLUMN_SPECS,
    EXPECTED_ROWS,
    FREQUENCY,
    MISSING_DATA_POLICY,
    TIMEZONE,
    UNAVAILABLE_COLUMNS,
    WEATHER_MAPPING_RULE,
    YEAR,
    load_singapore_2024_half_hour,
    verify_raw_files,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "m1.3b-singapore-2024-half-hour-v1"
DEFAULT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/singapore_2024"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _parse_freeze_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"--frozen-at-utc 必须是 ISO-8601，实际 {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError("--frozen-at-utc 必须带显式时区偏移")
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _canonical_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)


def build_manifest(
    *,
    frame: pd.DataFrame,
    parquet_sha256: str,
    raw_paths: dict[str, Path],
    source_manifest_path: Path,
    frozen_at_utc: str,
) -> dict:
    """构造 canonical manifest。**所有**可复现事实都来自显式输入，不用运行时刻。"""
    raw_files = {
        name: {
            "path": str(path),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in sorted(raw_paths.items())
    }
    return {
        "schema": SCHEMA,
        "year": YEAR,
        "timezone": TIMEZONE,
        "row_count": int(len(frame)),
        "frequency": FREQUENCY,
        "start": frame["timestamp"].iloc[0].isoformat(),
        "end": frame["timestamp"].iloc[-1].isoformat(),
        "columns": {name: dict(COLUMN_SPECS[name]) for name in CANONICAL_COLUMNS},
        "raw_files": raw_files,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": _sha256_file(source_manifest_path),
        "output_parquet_sha256": parquet_sha256,
        "weather_mapping_policy": {
            "rule": WEATHER_MAPPING_RULE,
            "description": (
                "目标 h:00 使用天气 h:00（age=0）；目标 h:30 仍使用天气 h:00（age=30）。"
                "逐行保证 weather_source_timestamp <= timestamp。"
                "禁止用下一小时天气填充、禁止线性插值、禁止 centered resampling、"
                "禁止 backfill。"
            ),
            "allowed_weather_age_minutes": list(ALLOWED_WEATHER_AGES_MINUTES),
            "granularity_mapping": "hourly ERA5 -> half-hourly canonical（最近不晚于）",
        },
        "observed_value_ranges": {
            column: {
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
                "negative_rows": int((frame[column] < 0).sum()),
            }
            for column in CANONICAL_COLUMNS
            if column not in ("timestamp", "weather_source_timestamp")
        },
        "missing_data_policy": MISSING_DATA_POLICY,
        "unavailable_not_materialized": {
            name: {"status": "unavailable", "reason": reason}
            for name, reason in sorted(UNAVAILABLE_COLUMNS.items())
        },
        "code_revision": git_revision(),
        "frozen_at_utc": frozen_at_utc,
    }


def verify_existing(canonical_manifest_path: Path, expected: dict) -> None:
    """已有 canonical manifest 只能**只读验证**；内容不同即失败，不得覆盖。"""
    actual = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    if _canonical_json(actual) != _canonical_json(expected):
        differing = sorted(
            key for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            f"{canonical_manifest_path} 已存在且内容不同（差异字段：{differing}）；"
            "不得静默覆盖"
        )


def materialize(
    *,
    raw_dir: Path | str,
    source_manifest_path: Path | str,
    output_dir: Path | str,
    manifest_path: Path | str,
    frozen_at_utc: str,
) -> dict:
    """读出 canonical 表 → 写 Parquet → 写/校验 canonical manifest。"""
    source_manifest_path = Path(source_manifest_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    frozen = _parse_freeze_time_or_raise(frozen_at_utc)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    raw_paths = verify_raw_files(raw_dir, source_manifest)
    frame = load_singapore_2024_half_hour(raw_dir, source_manifest_path)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"canonical 表必须是 {EXPECTED_ROWS} 行，实际 {len(frame)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "half_hour.parquet"
    frame.to_parquet(parquet_path, index=False)
    parquet_sha = _sha256_file(parquet_path)

    manifest = build_manifest(
        frame=frame,
        parquet_sha256=parquet_sha,
        raw_paths=raw_paths,
        source_manifest_path=source_manifest_path,
        frozen_at_utc=frozen,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        verify_existing(manifest_path, manifest)
    else:
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")

    return {
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "output_parquet_sha256": parquet_sha,
        "row_count": int(len(frame)),
        "manifest": manifest,
    }


def _parse_freeze_time_or_raise(value: str) -> str:
    """本卡的 canonical 时间是**显式**输入，绝不取 `now()`。"""
    return _parse_freeze_timestamp(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="物化 Singapore-2024 半小时 canonical 表")
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data/raw/singapore_2024"))
    parser.add_argument("--source-manifest", default=str(
        REPO_ROOT / "data/manifest/singapore_2024.json"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--frozen-at-utc", required=True,
        help="显式冻结时刻（ISO-8601 带偏移）；**不得**使用运行时当前时间",
    )
    args = parser.parse_args(argv)

    try:
        result = materialize(
            raw_dir=args.raw_dir,
            source_manifest_path=args.source_manifest,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            frozen_at_utc=args.frozen_at_utc,
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"物化失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps({
        "parquet_path": result["parquet_path"],
        "manifest_path": result["manifest_path"],
        "row_count": result["row_count"],
        "output_parquet_sha256": result["output_parquet_sha256"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
