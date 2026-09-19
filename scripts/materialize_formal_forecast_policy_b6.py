"""M1.3f-e-b2-a：物化 **policy-v3**（绑定 B6 policy 与 exogenous v3 的 formal 候选）。

产出：`data/manifest/singapore_2024_forecast_policy_v3.json`（**新增**文件）。

**纪律**：

- **只输出 canonical 路径**：不接受 `--out-dir` / `--manifest-path` / `--revision`；
- `materializer_revision` 由 Git 解析**本卡实现文件**，不接受调用者覆盖；
- generator 有未提交修改时拒绝生成；
- 已存在且**语义不同** → 拒绝覆盖；相同则直接返回（bytes/hash/mtime 不变）；
- 首次写入失败不留半成品（原子安装）。

用法：

```bash
uv run python scripts/materialize_formal_forecast_policy_b6.py
uv run python scripts/materialize_formal_forecast_policy_b6.py --verify   # 只读
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.formal_scenario_b6 import (
    FormalB6Error,
    _generator_is_dirty,
    build_policy_v3_manifest,
    canonical_policy_v3_path,
    load_verified_policy_v3,
)
from scenario.splits import SplitError, _require_canonical_utc, _require_exact_keys

REPO_ROOT = Path(__file__).resolve().parent.parent


class FormalB6PolicyError(FormalB6Error):
    """policy-v3 物化的**明确失败**。"""


def existing_frozen_at_utc(manifest_path: Path) -> str | None:
    """已存在且含规范 `frozen_at_utc` 的 manifest 的时间戳；否则 None。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))["frozen_at_utc"]
        _require_canonical_utc(value, field="frozen_at_utc")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return str(value)


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def materialize_policy_v3(*, frozen_at_utc: str) -> dict:
    """生成 / 复核 canonical policy-v3。"""
    manifest_path = canonical_policy_v3_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise FormalB6PolicyError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书"
        )

    candidate = build_policy_v3_manifest(frozen_at_utc=frozen_at_utc)

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise FormalB6PolicyError(
                f"已存在的 policy-v3 不是合法 JSON：{error}"
            ) from error
        try:
            _require_exact_keys(existing, field="已存在的 policy-v3",
                                expected=tuple(candidate))
        except SplitError as error:
            raise FormalB6PolicyError(
                f"已存在的 policy-v3 键集合不符：{error}"
            ) from error
        if existing != candidate:
            raise FormalB6PolicyError(f"已存在的 policy-v3 与候选不同：拒绝覆盖 {manifest_path}")
        return {
            "manifest_path": manifest_path,
            "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "written": False,
        }

    _atomic_write_bytes(manifest_path, _canonical_json(candidate).encode("utf-8"))
    return {
        "manifest_path": manifest_path,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "written": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物化 policy-v3（M1.3f-e-b2-a，只写 canonical）"
    )
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    parser.add_argument("--frozen-at-utc", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify:
        try:
            load_verified_policy_v3()
        except (FormalB6Error, SplitError, OSError, ValueError) as error:
            print(f"materialize_formal_forecast_policy_b6: {error}", file=sys.stderr)
            return 1
        print(f"verified {canonical_policy_v3_path()}")
        return 0

    frozen_at_utc = (
        args.frozen_at_utc
        or existing_frozen_at_utc(canonical_policy_v3_path())
        or datetime.now(UTC).replace(microsecond=0).isoformat()
    )

    try:
        result = materialize_policy_v3(frozen_at_utc=frozen_at_utc)
    except (FormalB6Error, SplitError, OSError, ValueError) as error:
        print(f"materialize_formal_forecast_policy_b6: {error}", file=sys.stderr)
        return 1

    print(f"manifest_path={result['manifest_path']}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
