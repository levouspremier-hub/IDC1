"""M1.3e：物化 Singapore-2024 **forecast policy manifest**。

冻结 `scenario/forecast.py` 的预测口径（trailing seasonal-naive，周期 48 个半小时步），
并如实登记四项**仍未 materialize** 的缺口。本 manifest **不**声明 forecast 已就绪，
更**不**声明正式训练可用。

要求（与 M1.3b/M1.3d 的物化器同构）：

- 精确冻结 schema：**未知字段拒绝**（未来扩展必须 bump `schema`）；
- 路径**可移植**：入库只记仓库相对逻辑路径，不泄漏机器绝对路径；
- `materializer_revision` 由 Git 解析「最后修改本 provider/materializer 实现的提交」，
  **不用**漂移的 HEAD；
- generator 有未提交修改时**拒绝生成**（不能用旧 revision 为新代码背书）；
- 上游三个 hash（canonical parquet / canonical manifest / split manifest）逐级校验，
  不符即 **fail closed**；
- **已存在且不同则拒绝覆盖**；
- **同输入重跑 bytes/hash/mtime_ns 不变**（幂等）；
- **首次写入失败不留半份 manifest 或临时文件**（原子安装）；
- **不修改** M1.3d split manifest 的 `forecast_ready=false`。

用法：

```bash
uv run python scripts/materialize_singapore_forecast_policy.py
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

from contracts import CONTRACT_VERSION_ID
from scenario.forecast import (
    AVAILABLE_DRIVERS,
    FORECAST_PERIOD_STEPS,
    FREQUENCY,
    METHOD,
    UNAVAILABLE_NOT_MATERIALIZED,
)
from scenario.splits import (
    SplitError,
    _require_canonical_utc,
    _verify_canonical,
    logical_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

POLICY_SCHEMA = "m1.3e-singapore-2024-forecast-policy-v1"

# 本 policy 的 revision 由「最后修改这些实现的提交」解析。
FORECAST_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/forecast.py",
    "scripts/materialize_singapore_forecast_policy.py",
)

INFORMATION_POLICY = "closed_open_[origin-48, origin)"
TARGET_POLICY = "half_open_[origin, origin+C)"
SEED_POLICY = None  # 冻结方法不使用随机数

READINESS: dict[str, bool] = {
    "available_driver_forecasts_ready": True,
    "complete_scenario_forecasts_ready": False,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}

# 冻结的键集合：**必须精确相等**；未知字段一律拒绝。
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
)


class ForecastPolicyError(ValueError):
    """forecast policy manifest 的**明确失败**。"""


# --- Git ---------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def resolve_forecast_materializer_revision() -> str:
    """本 policy 的 revision = 最后修改 provider/materializer 实现的提交。"""
    revision = _git(
        "log", "-1", "--format=%H", "--", *FORECAST_SOURCE_PATHS
    ).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ForecastPolicyError(
            f"materializer_revision 无法由 Git 解析（{revision!r}）："
            "provider/materializer 实现必须先提交"
        )
    return revision


def _generator_is_dirty() -> bool:
    """生成实现是否有未提交修改（含未跟踪的新文件）。"""
    status = _git("status", "--porcelain", "--", *FORECAST_SOURCE_PATHS)
    return bool(status.strip())


# --- 原子安装 ----------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    """同目录临时文件 + `os.replace` 原子安装；失败不留半成品。"""
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
    """规范 JSON：键排序、缩进 2、保留非 ASCII、末尾换行。"""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


# --- 校验 --------------------------------------------------------------------

def _require_exact_keys(mapping: object, *, field: str) -> dict:
    if not isinstance(mapping, dict):
        raise ForecastPolicyError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    expected = set(POLICY_MANIFEST_KEYS)
    actual = set(mapping)
    if actual != expected:
        raise ForecastPolicyError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - expected)} 缺少={sorted(expected - actual)}"
            "（未来扩展必须 bump schema）"
        )
    return mapping


def build_forecast_policy_manifest(
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    frozen_at_utc: str,
) -> dict:
    """构造候选 forecast policy manifest（**只读**上游，逐级校验 hash）。"""
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    _require_canonical_utc(frozen_at_utc, field="frozen_at_utc")

    try:
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ForecastPolicyError(f"split manifest 不可读：{error}") from error
    if not isinstance(split_manifest, dict):
        raise ForecastPolicyError("split manifest 必须是 object")

    # canonical manifest → canonical parquet 逐级校验（不符即 fail closed）
    parquet_sha = _verify_canonical(
        canonical_parquet_path, canonical_manifest_path, split_manifest
    )

    manifest = {
        "schema": POLICY_SCHEMA,
        "contract_version": CONTRACT_VERSION_ID,
        "canonical_parquet_path": logical_repo_path(canonical_parquet_path),
        "canonical_parquet_sha256": parquet_sha,
        "canonical_manifest_path": logical_repo_path(canonical_manifest_path),
        "canonical_manifest_sha256": hashlib.sha256(
            canonical_manifest_path.read_bytes()
        ).hexdigest(),
        "split_manifest_path": logical_repo_path(split_manifest_path),
        "split_manifest_sha256": hashlib.sha256(
            split_manifest_path.read_bytes()
        ).hexdigest(),
        "materializer_revision": resolve_forecast_materializer_revision(),
        "available_drivers": list(AVAILABLE_DRIVERS),
        "method": METHOD,
        "period_steps": FORECAST_PERIOD_STEPS,
        "frequency": FREQUENCY,
        "information_policy": INFORMATION_POLICY,
        "target_policy": TARGET_POLICY,
        "seed_policy": SEED_POLICY,
        "unavailable_not_materialized": list(UNAVAILABLE_NOT_MATERIALIZED),
        "readiness": dict(READINESS),
        "frozen_at_utc": frozen_at_utc,
    }
    return _require_exact_keys(manifest, field="forecast policy manifest")


# --- 物化 --------------------------------------------------------------------

def materialize_forecast_policy(
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    manifest_path: Path | str,
    frozen_at_utc: str,
) -> dict:
    """生成 / 校验 `data/manifest/singapore_2024_forecast_policy.json`。"""
    manifest_path = Path(manifest_path)
    # 目标目录**先**建立：这样「首次写入失败」的现场是「空目录」，而不是「目录不存在」。
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise ForecastPolicyError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书（未提交）"
        )

    candidate = build_forecast_policy_manifest(
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
        frozen_at_utc=frozen_at_utc,
    )
    text = _canonical_json(candidate)

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ForecastPolicyError(
                f"已存在的 forecast policy manifest 不是合法 JSON：{error}"
            ) from error
        _require_exact_keys(existing, field="已存在的 forecast policy manifest")
        if existing != candidate:
            raise ForecastPolicyError(
                f"已存在的 forecast policy manifest 与候选不同：拒绝覆盖 {manifest_path}"
            )
        # 完全相同：不重写，bytes/hash/mtime_ns 全部保持不变
        return {
            "manifest_path": manifest_path,
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "written": False,
        }

    _atomic_write_text(manifest_path, text)
    return {
        "manifest_path": manifest_path,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "written": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="物化 forecast policy manifest（M1.3e）")
    parser.add_argument("--canonical-parquet-path",
                        default="data/processed/singapore_2024/half_hour.parquet")
    parser.add_argument("--canonical-manifest-path",
                        default="data/manifest/singapore_2024_half_hour.json")
    parser.add_argument("--split-manifest-path",
                        default="data/manifest/singapore_2024_splits.json")
    parser.add_argument("--manifest-path",
                        default="data/manifest/singapore_2024_forecast_policy.json")
    parser.add_argument("--frozen-at-utc",
                        default=datetime.now(UTC).replace(microsecond=0).isoformat())
    args = parser.parse_args(argv)

    try:
        result = materialize_forecast_policy(
            canonical_parquet_path=REPO_ROOT / args.canonical_parquet_path,
            canonical_manifest_path=REPO_ROOT / args.canonical_manifest_path,
            split_manifest_path=REPO_ROOT / args.split_manifest_path,
            manifest_path=REPO_ROOT / args.manifest_path,
            frozen_at_utc=args.frozen_at_utc,
        )
    except (ForecastPolicyError, SplitError, OSError, ValueError) as error:
        print(f"materialize_singapore_forecast_policy: {error}", file=sys.stderr)
        return 1

    print(f"manifest_path={args.manifest_path}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    print(f"readiness={json.dumps(READINESS, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
