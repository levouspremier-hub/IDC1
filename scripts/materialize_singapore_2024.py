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
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

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
# 数据生成实现文件：`materializer_revision` 取「HEAD 可达历史中最后修改它们的提交」。
MATERIALIZER_SOURCE_PATHS = (
    "scenario/singapore_2024.py",
    "scripts/materialize_singapore_2024.py",
)
DEFAULT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_half_hour.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/singapore_2024"


def _git(*args: str) -> str:
    """运行只读 git 命令（便于测试 monkeypatch；不写任何仓库状态）。"""
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout


def _generator_is_dirty() -> bool:
    """生成实现文件是否有**未提交**修改。

    dirty 时拒绝生成：否则会用旧的 `materializer_revision` 为未提交代码
    产出的内容背书，provenance 就是假的。
    """
    status = _git("status", "--porcelain", "--", *MATERIALIZER_SOURCE_PATHS)
    return bool(status.strip())


def resolve_materializer_revision() -> str:
    """**数据生成实现**的 revision —— 由 Git 解析，指向最后修改生成实现的提交。

    刻意**不**用「当前 HEAD」：否则任何后续提交（哪怕只改文档）都会让已冻结的
    canonical manifest 失效，破坏幂等。**不得**由调用者传入未经验证的 revision。

    生成实现文件存在**未提交修改**时拒绝（`_generator_is_dirty`）。
    """
    if _generator_is_dirty():
        raise ValueError(
            "生成实现文件存在未提交修改（未提交的代码不得为冻结数据背书）；"
            f"请先提交：{list(MATERIALIZER_SOURCE_PATHS)}"
        )
    try:
        revision = _git("log", "-1", "--format=%H", "--", *MATERIALIZER_SOURCE_PATHS).strip()
    except (subprocess.CalledProcessError, OSError) as error:  # pragma: no cover
        raise ValueError(f"无法解析 materializer revision：{error}") from error
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"materializer revision 不是有效的 Git 提交：{revision!r}")
    return revision


_HEX64 = set("0123456789abcdef")


def _validated_frozen_manifest(manifest_path: Path) -> dict | None:
    """读取并**校验结构** frozen manifest；畸形一律 `ValueError`（不得泄漏 KeyError）。"""
    if not manifest_path.exists():
        return None
    try:
        frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"frozen canonical manifest 不可读：{error}") from error
    if not isinstance(frozen, dict):
        raise ValueError(
            f"frozen canonical manifest 顶层必须是 object，实际 {type(frozen).__name__}"
        )
    digest = frozen.get("output_parquet_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in _HEX64 for c in digest):
        raise ValueError(
            "frozen canonical manifest 的 output_parquet_sha256 必须是 64 位小写十六进制字符串，"
            f"实际 {digest!r}"
        )
    revision = frozen.get("materializer_revision")
    revision_ok = (
        isinstance(revision, str) and len(revision) == 40
        and all(c in _HEX64 for c in revision)
    )
    if not revision_ok:
        raise ValueError(
            "frozen canonical manifest 的 materializer_revision 必须是 40 位小写十六进制提交，"
            f"实际 {revision!r}"
        )
    return frozen


def logical_repo_path(path: Path | str) -> str:
    """入库 manifest 只记**仓库相对逻辑路径**，绝不泄漏机器绝对路径。

    仓库内文件 → `data/raw/...` 这样的相对路径；
    仓库外文件（如测试临时目录）→ 稳定的 `<external>/<文件名>`，
    保证不同 root 下生成的 provenance 一致。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


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
            "path": logical_repo_path(path),
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
        "source_manifest_path": logical_repo_path(source_manifest_path),
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
        "materializer_revision": resolve_materializer_revision(),
        "frozen_at_utc": frozen_at_utc,
    }


def _require_same_frozen_fields(frozen: dict, candidate: dict) -> None:
    """除 `output_parquet_sha256`（与 parquet 本体绑定）外，冻结字段必须完全一致。

    不一致即 **fail closed**：调用方不得靠重跑静默改写已冻结的 provenance。
    """
    differing = sorted(
        key for key in set(frozen) | set(candidate)
        if key != "output_parquet_sha256" and frozen.get(key) != candidate.get(key)
    )
    if differing:
        raise ValueError(f"已冻结的 canonical manifest 与本次生成不一致（差异字段：{differing}）")


def _atomic_write_text(path: Path, text: str) -> None:
    """同文件系统内写临时文件再 `os.replace`：要么全成，要么全不成。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def materialize(
    *,
    raw_dir: Path | str,
    source_manifest_path: Path | str,
    output_dir: Path | str,
    manifest_path: Path | str,
    frozen_at_utc: str,
) -> dict:
    """读出 canonical 表 → 候选写入临时文件 → **先比较再落盘**（失败原子性）。

    顺序固定为：验证 raw → 构建 frame → 写候选临时 parquet → 算候选 hash 与候选
    manifest → 与已有冻结 manifest 比较 → 只有确认安全才 `os.replace` 安装。

    任何失败路径都**不得**改动正式 parquet / manifest，也**不得**留下临时文件。
    """
    source_manifest_path = Path(source_manifest_path)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    frozen_time = _parse_freeze_time_or_raise(frozen_at_utc)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    raw_paths = verify_raw_files(raw_dir, source_manifest)
    frame = load_singapore_2024_half_hour(raw_dir, source_manifest_path)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"canonical 表必须是 {EXPECTED_ROWS} 行，实际 {len(frame)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "half_hour.parquet"

    # 候选 parquet 只写进**同文件系统**的临时文件；正式路径此刻不动。
    handle, temp_name = tempfile.mkstemp(dir=output_dir, prefix=".half_hour.", suffix=".tmp")
    os.close(handle)
    temp_parquet = Path(temp_name)
    installed_parquet = False
    try:
        frame.to_parquet(temp_parquet, index=False)
        candidate_sha = _sha256_file(temp_parquet)
        candidate_manifest = build_manifest(
            frame=frame,
            parquet_sha256=candidate_sha,
            raw_paths=raw_paths,
            source_manifest_path=source_manifest_path,
            frozen_at_utc=frozen_time,
        )
        frozen = _validated_frozen_manifest(manifest_path)

        if frozen is None and parquet_path.exists():
            # 首冻目标已存在来源不明的 parquet：拒绝把别人的产物当自己的覆盖目标。
            raise ValueError(
                f"canonical manifest 不存在，但 {parquet_path} 已存在；"
                "拒绝把来源不明的既有 parquet 当作首次冻结的可覆盖目标"
            )

        if frozen is not None:
            # 已有冻结 manifest：先把**全部**冻结字段比完，再决定是否落盘。
            _require_same_frozen_fields(frozen, candidate_manifest)
            frozen_sha = frozen["output_parquet_sha256"]
            if candidate_sha != frozen_sha:
                raise ValueError(
                    "候选 parquet 的 SHA-256 与冻结 manifest 不符："
                    f"candidate={candidate_sha} frozen={frozen_sha}"
                )
            if parquet_path.exists() and _sha256_file(parquet_path) == frozen_sha:
                pass  # 完全一致：正式 parquet 与 manifest 都不重写
            else:
                # 正式 parquet 缺失/损坏：仅在候选 hash 与冻结值相符时原子恢复
                os.replace(temp_parquet, parquet_path)
                installed_parquet = True
        else:
            # 首次冻结：两个产物都安装；**任一失败即回滚本次已安装的文件**。
            # 回滚只处理**本次创建**的路径（首冻时两者都不存在，故安全）。
            created: list[Path] = []
            try:
                os.replace(temp_parquet, parquet_path)
                installed_parquet = True
                created.append(parquet_path)
                _atomic_write_text(manifest_path, _canonical_json(candidate_manifest) + "\n")
                created.append(manifest_path)
            except BaseException:
                for path in created:
                    path.unlink(missing_ok=True)
                raise
    finally:
        if not installed_parquet:
            temp_parquet.unlink(missing_ok=True)

    return {
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "output_parquet_sha256": _sha256_file(parquet_path),
        "row_count": int(len(frame)),
        "manifest": candidate_manifest,
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
