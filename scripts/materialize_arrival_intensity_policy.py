"""M1.3f-e-a：物化 B6 arrival-intensity policy manifest。

冻结 `scenario/arrival_intensity_policy.py` 的 B6-INTENSITY 口径。本物化器：

- **只输出 canonical 路径**（`data/manifest/m13f_arrival_intensity_policy_v1.json`），
  **不接受** `--manifest-path` / `--out-dir` 等任意目标；
- `materializer_revision` 由 Git 解析 B6 实现文件，不用漂移的 HEAD；
- generator 有未提交修改时拒绝生成；
- 上游 canonical 链逐级校验（`load_truth_split`），温度/容量重算一致才产出；
- 已存在且不同 → 拒绝覆盖；同输入重跑 bytes/hash/mtime_ns 不变（幂等）；
- 首次写入失败不留半份或临时文件（原子安装）。

用法：

```bash
uv run python scripts/materialize_arrival_intensity_policy.py           # 物化/复核
uv run python scripts/materialize_arrival_intensity_policy.py --verify  # 只读校验
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

from scenario.arrival_intensity_policy import (
    ArrivalIntensityPolicyError,
    B6_SOURCE_PATHS,
    CANONICAL_MANIFEST_LOGICAL,
    CANONICAL_PARQUET_LOGICAL,
    POLICY_MANIFEST_KEYS,
    SPLIT_MANIFEST_LOGICAL,
    b6_code_revision,
    build_b6_policy_manifest,
    canonical_policy_path,
    load_verified_b6_policy,
)
from scenario.splits import (
    SplitError,
    _require_canonical_utc,
    _require_exact_keys,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    """B6 实现文件是否有未提交修改（含未跟踪的新文件）。"""
    status = _git("status", "--porcelain", "--", *B6_SOURCE_PATHS)
    return bool(status.strip())


def existing_frozen_at_utc(manifest_path: Path | str) -> str | None:
    """已存在且含规范 `frozen_at_utc` 的 manifest 的时间戳；否则 `None`。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))["frozen_at_utc"]
        _require_canonical_utc(value, field="frozen_at_utc")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return str(value)


def _atomic_write_text(path: Path, text: str) -> None:
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


def materialize_b6_policy(*, frozen_at_utc: str) -> dict:
    """生成 / 复核 canonical B6 policy manifest。"""
    manifest_path = canonical_policy_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise ArrivalIntensityPolicyError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书"
        )

    candidate = build_b6_policy_manifest(
        canonical_parquet_path=REPO_ROOT / CANONICAL_PARQUET_LOGICAL,
        canonical_manifest_path=REPO_ROOT / CANONICAL_MANIFEST_LOGICAL,
        split_manifest_path=REPO_ROOT / SPLIT_MANIFEST_LOGICAL,
        frozen_at_utc=frozen_at_utc,
    )
    text = _canonical_json(candidate)

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ArrivalIntensityPolicyError(
                f"已存在的 policy 不是合法 JSON：{error}"
            ) from error
        _require_exact_keys(existing, field="已存在的 arrival intensity policy",
                            expected=POLICY_MANIFEST_KEYS)
        if existing != candidate:
            raise ArrivalIntensityPolicyError(
                f"已存在的 policy 与候选不同：拒绝覆盖 {manifest_path}"
            )
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
    parser = argparse.ArgumentParser(
        description="物化 B6 arrival-intensity policy manifest（M1.3f-e-a）"
    )
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    parser.add_argument("--frozen-at-utc", default=None)
    args = parser.parse_args(argv)

    # 只输出 canonical 路径：不接受 --manifest-path / --out-dir（argparse 会拒绝未知参数）

    if args.verify:
        try:
            load_verified_b6_policy()
        except (ArrivalIntensityPolicyError, SplitError, OSError, ValueError) as error:
            print(f"materialize_arrival_intensity_policy: {error}", file=sys.stderr)
            return 1
        print(f"verified canonical policy {canonical_policy_path()}")
        return 0

    manifest_path = canonical_policy_path()
    frozen_at_utc = (
        args.frozen_at_utc
        or existing_frozen_at_utc(manifest_path)
        or datetime.now(UTC).replace(microsecond=0).isoformat()
    )

    try:
        result = materialize_b6_policy(frozen_at_utc=frozen_at_utc)
    except (ArrivalIntensityPolicyError, SplitError, OSError, ValueError) as error:
        print(f"materialize_arrival_intensity_policy: {error}", file=sys.stderr)
        return 1

    print(f"manifest_path={manifest_path}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
