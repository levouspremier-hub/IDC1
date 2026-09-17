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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contracts import CONTRACT_VERSION_ID
from scenario.forecast import (
    AVAILABLE_DRIVERS,
    DEFAULT_POLICY_MANIFEST_PATH,
    FORECAST_PERIOD_STEPS,
    FORECAST_SOURCE_PATHS,
    FREQUENCY,
    INFORMATION_POLICY,
    METHOD,
    POLICY_MANIFEST_KEYS,
    POLICY_READINESS,
    POLICY_SCHEMA,
    SEED_POLICY,
    SUPERSEDES_POLICY_V1,
    TARGET_POLICY,
    UNAVAILABLE_NOT_MATERIALIZED,
    provider_code_revision,
)
from scenario.splits import (
    SplitError,
    _require_canonical_utc,
    load_truth_split,
    logical_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# 冻结的 policy schema 与键集合由 `scenario.forecast`（provider）**唯一**定义，
# 物化器只导入复用，避免两处漂移。`READINESS` 为旧名保留的别名。
READINESS = POLICY_READINESS
STATIC_POLICY_SOURCE_PATHS = FORECAST_SOURCE_PATHS


class ForecastPolicyError(ValueError):
    """forecast policy manifest 的**明确失败**。"""


# --- Git ---------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def resolve_forecast_materializer_revision() -> str:
    """本 policy 的 revision = 最后修改 provider/materializer 实现的提交。

    与 provider 使用**同一** `scenario.forecast.FORECAST_SOURCE_PATHS`，
    因此 artifact 的 `code_revision` 与本 manifest 的 `materializer_revision` 恒等。
    """
    try:
        return provider_code_revision()
    except ValueError as error:
        raise ForecastPolicyError(
            f"materializer_revision 无法由 Git 解析：{error}"
            "（provider/materializer 实现必须先提交）"
        ) from error


def _generator_is_dirty() -> bool:
    """生成实现是否有未提交修改（含未跟踪的新文件）。"""
    status = _git("status", "--porcelain", "--", *FORECAST_SOURCE_PATHS)
    return bool(status.strip())


def existing_frozen_at_utc(manifest_path: Path | str) -> str | None:
    """已存在且含规范 `frozen_at_utc` 的 manifest 的时间戳；否则 `None`。

    重复物化必须**复用**它，否则正式 manifest 会随墙钟漂移（M1.3g-0）。
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))["frozen_at_utc"]
        _require_canonical_utc(value, field="frozen_at_utc")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return str(value)


def verify_superseded_policy_v1(v1_path: Path | str = Path(
    SUPERSEDES_POLICY_V1["policy_manifest_path"]
)) -> str:
    """核验被取代的 v1 政策产物**确为冻结的那一份**（M1.3g-0）。

    取代登记里的 path 与 SHA-256 必须是**事实**：文件缺失、字节被改写、
    或自述的契约版本不是 `contract-v8`，一律 fail closed。
    """
    v1_path = Path(v1_path)
    if not v1_path.is_file():
        raise ForecastPolicyError(
            f"缺少被取代的 v1 政策产物 {v1_path}：取代登记不得凭空声明"
        )
    actual = hashlib.sha256(v1_path.read_bytes()).hexdigest()
    expected = SUPERSEDES_POLICY_V1["policy_manifest_sha256"]
    if actual != expected:
        raise ForecastPolicyError(
            f"v1 政策产物字节已改变：期望 {expected} 实际 {actual}"
            "（v1 是 contract-v8 的历史证据，不得改写）"
        )
    payload = json.loads(v1_path.read_text(encoding="utf-8"))
    declared = payload.get("contract_version")
    if declared != SUPERSEDES_POLICY_V1["policy_manifest_contract_version"]:
        raise ForecastPolicyError(
            f"v1 政策产物的 contract_version 必须是 "
            f"{SUPERSEDES_POLICY_V1['policy_manifest_contract_version']!r}，"
            f"实际 {declared!r}"
        )
    return actual


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
    """构造候选 forecast policy manifest（**只读**上游，逐级校验 hash）。

    M1.3e-R1：除 canonical manifest → canonical parquet 的 hash 链外，还会走
    **M1.3d 的完整严格校验**（`load_truth_split("train", ...)`：split manifest 的
    精确键集合/冻结声明/readiness/unavailable/train-only 统计重算，以及整条
    canonical 时间轴）。policy manifest 因此只会在「split 声明可被完整验证」时产出。
    """
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    _require_canonical_utc(frozen_at_utc, field="frozen_at_utc")

    # M1.3d 的**完整**严格校验（split manifest 精确键集合与冻结声明、canonical
    # manifest → canonical parquet 的 hash 链、整条 canonical 时间轴、
    # train-only 统计重算）。复用既有 reader，**不**复制宽松校验器，也**不**在
    # 严格校验之前先触碰 split manifest 的字段（否则畸形输入会泄漏 KeyError）。
    load_truth_split(
        "train",
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    parquet_sha = hashlib.sha256(canonical_parquet_path.read_bytes()).hexdigest()

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
        # M1.3g-0：v2 schema 的取代登记（逐字段恒等于冻结常量）
        "supersedes": dict(SUPERSEDES_POLICY_V1),
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
    """生成 / 校验 `data/manifest/singapore_2024_forecast_policy_v2.json`。"""
    manifest_path = Path(manifest_path)
    # 目标目录**先**建立：这样「首次写入失败」的现场是「空目录」，而不是「目录不存在」。
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise ForecastPolicyError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书（未提交）"
        )

    # M1.3g-0：取代登记必须是**事实**——v1 仍在、字节未变、自述 contract-v8
    if manifest_path.name == Path(DEFAULT_POLICY_MANIFEST_PATH).name:
        verify_superseded_policy_v1()

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
    parser.add_argument("--manifest-path", default=DEFAULT_POLICY_MANIFEST_PATH)
    parser.add_argument("--frozen-at-utc", default=None)
    args = parser.parse_args(argv)

    manifest_path = REPO_ROOT / args.manifest_path
    # 重复物化**复用**已冻结的 `frozen_at_utc`，避免正式 manifest 随墙钟漂移
    frozen_at_utc = (
        args.frozen_at_utc
        or existing_frozen_at_utc(manifest_path)
        or datetime.now(UTC).replace(microsecond=0).isoformat()
    )

    try:
        result = materialize_forecast_policy(
            canonical_parquet_path=REPO_ROOT / args.canonical_parquet_path,
            canonical_manifest_path=REPO_ROOT / args.canonical_manifest_path,
            split_manifest_path=REPO_ROOT / args.split_manifest_path,
            manifest_path=manifest_path,
            frozen_at_utc=frozen_at_utc,
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
