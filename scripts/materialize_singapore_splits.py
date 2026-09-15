#!/usr/bin/env python
"""M1.3d：冻结 Singapore-2024 的**连续 truth split** 并写可审计 split manifest。

**本脚本只冻结 truth 切分与 train-only 描述统计**；**不生成 forecast**、
**不创建 `train.json` / `validation.json` / `test.json`**、**不开始训练**。

那三个文件名留给「完整 forecast / `ScenarioBundle` readiness 成立之后」使用 ——
避免「文件存在」被误判为训练已就绪。

纪律与 M1.3b 一致：候选先写临时文件、先比较再 `os.replace`、失败回滚、
路径可移植、revision 由 Git 解析指向**本 split 生成实现**、generator dirty 时拒绝。
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scenario.splits import (
    EPISODE_ORIGIN_RULE,
    FORECAST_ORIGIN_RULE,
    FREQUENCY,
    SPLIT_NAMES,
    SPLIT_SCHEMA,
    SPLIT_SPECS,
    STATISTIC_COLUMNS,
    TIMEZONE,
    TOTAL_ROWS,
    UNAVAILABLE_COLUMNS,
    YEAR,
    train_only_statistics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
# 本 split 生成的**实现文件**：revision 取「HEAD 可达历史中最后修改它们的提交」
SPLIT_SOURCE_PATHS = (
    "scenario/splits.py",
    "scripts/materialize_singapore_splits.py",
)
DEFAULT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_splits.json"
DEFAULT_CANONICAL_PARQUET = REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet"
DEFAULT_CANONICAL_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"

# 本卡**绝不**创建这些名字（留给 forecast/ScenarioBundle readiness）
FORBIDDEN_MANIFEST_NAMES = ("train.json", "validation.json", "test.json")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout


def _generator_is_dirty() -> bool:
    """split 生成实现是否有**未提交**修改；dirty 时拒绝生成。"""
    return bool(_git("status", "--porcelain", "--", *SPLIT_SOURCE_PATHS).strip())


def resolve_split_materializer_revision() -> str:
    """由 Git 解析「最后修改 split 生成实现的提交」——**不得**用漂移的 HEAD。"""
    if _generator_is_dirty():
        raise ValueError(
            "split 生成实现文件存在未提交修改（未提交的代码不得为冻结 split 背书）；"
            f"请先提交：{list(SPLIT_SOURCE_PATHS)}"
        )
    try:
        revision = _git("log", "-1", "--format=%H", "--", *SPLIT_SOURCE_PATHS).strip()
    except (subprocess.CalledProcessError, OSError) as error:  # pragma: no cover
        raise ValueError(f"无法解析 split materializer revision：{error}") from error
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"split materializer revision 不是有效的 Git 提交：{revision!r}")
    return revision


def logical_repo_path(path: Path | str) -> str:
    """入库 manifest 只记**仓库相对逻辑路径**，绝不泄漏机器绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(manifest: dict) -> str:
    return json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)


def _parse_freeze_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"--frozen-at-utc 必须是 ISO-8601，实际 {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError("--frozen-at-utc 必须带显式时区偏移")
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _verify_canonical_pair(canonical_parquet_path: Path, canonical_manifest_path: Path) -> dict:
    """校验 canonical manifest 与其 parquet 自洽（不依赖 split manifest）。"""
    try:
        canonical = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"canonical manifest 不可读：{error}") from error
    if not isinstance(canonical, dict):
        raise ValueError("canonical manifest 顶层必须是 object")
    if canonical.get("row_count") != TOTAL_ROWS:
        raise ValueError(f"canonical manifest row_count 必须是 {TOTAL_ROWS}")
    if canonical.get("timezone") != TIMEZONE or canonical.get("frequency") != FREQUENCY:
        raise ValueError("canonical manifest 的 timezone/frequency 不符")
    actual = _sha256_file(canonical_parquet_path)
    if canonical.get("output_parquet_sha256") != actual:
        raise ValueError(
            "canonical parquet SHA-256 与 canonical manifest 不符："
            f"manifest={canonical.get('output_parquet_sha256')} 实际={actual}"
        )
    return canonical


def build_split_manifest(
    *,
    canonical_parquet_path: Path,
    canonical_manifest_path: Path,
    frozen_at_utc: str,
) -> dict:
    """构造 split manifest。**所有**可复现事实来自显式输入，不用运行时刻。

    注意：本函数**不**调用 `load_truth_split()`（后者需要已存在的 split manifest，
    会造成先有鸡还是先有蛋）；train-only 统计直接从 canonical 表的 train 行段切出。
    """
    _verify_canonical_pair(canonical_parquet_path, canonical_manifest_path)
    frame = pd.read_parquet(canonical_parquet_path)
    if len(frame) != TOTAL_ROWS:
        raise ValueError(f"canonical 行数必须是 {TOTAL_ROWS}，实际 {len(frame)}")
    if str(frame["timestamp"].dt.tz) != TIMEZONE:
        raise ValueError(f"canonical 时间戳时区必须是 {TIMEZONE}")

    train_spec = SPLIT_SPECS["train"]
    train_frame = frame.iloc[
        train_spec["row_start"]:train_spec["row_end_exclusive"]
    ]
    if len(train_frame) != train_spec["row_count"]:
        raise ValueError(
            f"train 行段 {len(train_frame)} != 冻结值 {train_spec['row_count']}"
        )
    statistics = train_only_statistics(train_frame)

    canonical_sha = _sha256_file(canonical_parquet_path)
    revision = resolve_split_materializer_revision()
    return {
        "schema": SPLIT_SCHEMA,
        "year": YEAR,
        "timezone": TIMEZONE,
        "frequency": FREQUENCY,
        "step_minutes": 30,
        "total_rows": TOTAL_ROWS,
        "canonical_parquet_path": logical_repo_path(canonical_parquet_path),
        "canonical_parquet_sha256": canonical_sha,
        "canonical_manifest_path": logical_repo_path(canonical_manifest_path),
        "canonical_manifest_sha256": _sha256_file(canonical_manifest_path),
        "materializer_revision": revision,
        "splits": {name: dict(SPLIT_SPECS[name]) for name in SPLIT_NAMES},
        "no_overlap": True,
        "no_gap": True,
        "randomized": False,
        "leap_day_split": "train",
        "episode_origin_rule": EPISODE_ORIGIN_RULE,
        "forecast_origin_rule": FORECAST_ORIGIN_RULE,
        "train_only_statistics": statistics,
        "train_only_statistics_source": {
            "split": "train",
            "row_start": train_spec["row_start"],
            "row_end_exclusive": train_spec["row_end_exclusive"],
            "start": train_spec["start"],
            "end_exclusive": train_spec["end_exclusive"],
            "columns": list(STATISTIC_COLUMNS),
            "canonical_parquet_sha256": canonical_sha,
            "statistics_implementation_revision": revision,
            "note": (
                "描述统计，**不得**自动冒充正式 normalization refs；"
                "正式 refs 最迟在 M1.3g 训练接线前冻结"
            ),
        },
        "unavailable_not_materialized": {
            name: "unavailable: 未冻结；不得用默认曲线/常数/全零/未来真值填充"
            for name in sorted(UNAVAILABLE_COLUMNS)
        },
        "readiness": {
            "truth_splits_ready": True,
            "forecast_ready": False,
            "formal_scenario_bundle_ready": False,
            "formal_training_ready": False,
        },
        "frozen_at_utc": _parse_freeze_timestamp(frozen_at_utc),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _forbid_readiness_names(manifest_path: Path) -> None:
    if manifest_path.name in FORBIDDEN_MANIFEST_NAMES:
        raise ValueError(
            f"{manifest_path.name} 是保留名（forecast/ScenarioBundle readiness 成立后才可用）；"
            "本卡不得创建它，以免被误判为训练已就绪"
        )


def materialize_splits(
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    manifest_path: Path | str = DEFAULT_MANIFEST,
    frozen_at_utc: str,
) -> dict:
    """物化 split manifest：候选先写临时文件、**先比较再落盘**、失败回滚。"""
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    manifest_path = Path(manifest_path)
    _forbid_readiness_names(manifest_path)

    # 先按 canonical 的 hash 与冻结边界算出「预期 split manifest 骨架」，
    # 再用它校验 canonical，最后才补 train-only 统计。
    candidate = build_split_manifest(
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        frozen_at_utc=frozen_at_utc,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=manifest_path.parent, prefix=f".{manifest_path.name}.", suffix=".tmp")
    os.close(handle)
    temp_manifest = Path(temp_name)
    installed = False
    try:
        temp_manifest.write_text(_canonical_json(candidate) + "\n", encoding="utf-8")
        if manifest_path.exists():
            frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(frozen, dict):
                raise ValueError("已存在的 split manifest 顶层必须是 object")
            if _canonical_json(frozen) != _canonical_json(candidate):
                differing = sorted(
                    key for key in set(frozen) | set(candidate)
                    if frozen.get(key) != candidate.get(key)
                )
                raise ValueError(
                    f"{manifest_path} 已存在且内容不同（差异字段：{differing}）；不得静默覆盖"
                )
            return {
                "manifest_path": str(manifest_path),
                "manifest": candidate,
                "written": False,
            }
        os.replace(temp_manifest, manifest_path)
        installed = True
    finally:
        if not installed:
            temp_manifest.unlink(missing_ok=True)

    return {"manifest_path": str(manifest_path), "manifest": candidate, "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结 Singapore-2024 连续 truth split")
    parser.add_argument("--canonical-parquet", default=str(DEFAULT_CANONICAL_PARQUET))
    parser.add_argument("--canonical-manifest", default=str(DEFAULT_CANONICAL_MANIFEST))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--frozen-at-utc", required=True)
    args = parser.parse_args(argv)

    try:
        result = materialize_splits(
            canonical_parquet_path=args.canonical_parquet,
            canonical_manifest_path=args.canonical_manifest,
            manifest_path=args.manifest,
            frozen_at_utc=args.frozen_at_utc,
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"split 物化失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps({
        "manifest_path": result["manifest_path"],
        "written": result["written"],
        "total_rows": result["manifest"]["total_rows"],
        "splits": {name: SPLIT_SPECS[name]["row_count"] for name in SPLIT_NAMES},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
