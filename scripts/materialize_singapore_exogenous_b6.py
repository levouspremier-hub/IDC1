"""M1.3f-e-b1：由 B6 policy 物化**版本化**外生驱动表 v3。

产出（全部为**新增**文件，v2 三项资产逐字节不动）：

- `data/processed/singapore_2024/exogenous_drivers_v3.parquet`（17,568 × 5，gitignore）
- `data/manifest/singapore_2024_exogenous_v3.json`（输出 manifest）
- `data/manifest/m13f_materialization_sources_v4.json`（来源 manifest v4）

**纪律**：

- **只输出 canonical 路径**：不接受 `--out-dir` / `--manifest-path`；
- 参数**不得**覆盖 scale / `rho` / seed / policy——它们全部来自已验证的 B6 policy；
- **v3 manifest 与 source-v4 共享一个 `frozen_at_utc`**；
- `materializer_revision` 由 Git 解析**本卡实现文件**（产物晚于最后一次代码修改）；
- 已存在且不同 → **拒绝覆盖**；相同则直接返回（bytes/hash/mtime 不变）；
- 首次写入失败不留半成品。

用法：

```bash
uv run python scripts/materialize_singapore_exogenous_b6.py
uv run python scripts/materialize_singapore_exogenous_b6.py --verify   # 只读
```
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.exogenous_drivers import (
    CANONICAL_MANIFEST,
    CANONICAL_PARQUET,
    SPLIT_MANIFEST,
    load_frozen_inputs,
)
from scenario.exogenous_drivers_b6 import (
    B6_EXOGENOUS_SOURCE_PATHS,
    B6ExogenousError,
    b6_exogenous_revision,
    build_v3_frame,
    build_v3_output_manifest,
    build_v3_source_manifest,
    canonical_v3_manifest_path,
    canonical_v3_parquet_path,
    canonical_v3_source_path,
    load_b6_policy,
    load_v2_arrival_template,
    load_verified_v3_manifest,
    load_verified_v3_source,
)
from scenario.splits import SplitError, _require_canonical_utc

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    """本卡实现文件是否有未提交修改（含未跟踪的新文件）。"""
    status = _git("status", "--porcelain", "--", *B6_EXOGENOUS_SOURCE_PATHS)
    return bool(status.strip())


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


def write_json_atomic(payload: dict, path: Path) -> None:
    """写 JSON：已存在且**语义不同** → fail closed；相同则不重写。"""
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise B6ExogenousError(
                f"已存在的 {path} 不是合法 JSON：{error}"
            ) from error
        if existing == payload:
            return
        raise B6ExogenousError(
            f"已存在的 {path} 与候选**语义不同**：拒绝覆盖冻结产物"
        )
    _atomic_write_bytes(path, text.encode("utf-8"))


def materialize_b6_exogenous(*, frozen_at_utc: str) -> dict:
    """计算并**原子**安装 v3 parquet / source-v4 / v3 manifest。"""
    out_parquet = canonical_v3_parquet_path()
    source_path = canonical_v3_source_path()
    out_manifest = canonical_v3_manifest_path()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise B6ExogenousError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书"
        )

    # 信任根：canonical B6 policy + v2 冻结 shape
    policy = load_b6_policy()
    template = load_v2_arrival_template()

    inputs = load_frozen_inputs(
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
    )
    frame = build_v3_frame(inputs, template=template, policy=policy)

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    body = buffer.getvalue()

    created: list[Path] = []
    try:
        if out_parquet.exists():
            if out_parquet.read_bytes() != body:
                raise B6ExogenousError(
                    f"{out_parquet} 已存在且内容不同：拒绝覆盖"
                )
        else:
            _atomic_write_bytes(out_parquet, body)
            created.append(out_parquet)

        if not source_path.exists():
            created.append(source_path)
        if not out_manifest.exists():
            created.append(out_manifest)

        # **先**写 source-v4：输出 manifest 会引用它的 SHA-256
        source_payload = build_v3_source_manifest(frozen_at_utc=frozen_at_utc)
        write_json_atomic(source_payload, source_path)

        payload = build_v3_output_manifest(
            inputs=inputs,
            frame=frame,
            output_path=out_parquet,
            source_manifest_sha256=hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
            template=template,
            policy=policy,
            materializer_revision=b6_exogenous_revision(),
            frozen_at_utc=frozen_at_utc,
        )
        write_json_atomic(payload, out_manifest)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {"output_path": out_parquet, "manifest_path": out_manifest,
            "source_path": source_path, "rows": len(frame)}


def materialize_exogenous_v3(*, frozen_at_utc: str) -> dict:
    """兼容别名（测试用）。"""
    return materialize_b6_exogenous(frozen_at_utc=frozen_at_utc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1.3f-e-b1 B6 exogenous v3 物化（默认不联网、只写 canonical）"
    )
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    parser.add_argument("--frozen-at-utc", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify:
        try:
            load_verified_v3_manifest()
            load_verified_v3_source()
        except (B6ExogenousError, SplitError, OSError, ValueError) as error:
            print(f"materialize_singapore_exogenous_b6: {error}", file=sys.stderr)
            return 1
        print(f"verified {canonical_v3_manifest_path()}")
        return 0

    frozen_at_utc = (
        args.frozen_at_utc
        or existing_frozen_at_utc(canonical_v3_manifest_path())
        or datetime.now(UTC).replace(microsecond=0).isoformat()
    )

    try:
        result = materialize_b6_exogenous(frozen_at_utc=frozen_at_utc)
    except (B6ExogenousError, SplitError, OSError, ValueError) as error:
        print(f"materialize_singapore_exogenous_b6: {error}", file=sys.stderr)
        return 1

    print(f"output={result['output_path']} rows={result['rows']}")
    print(f"manifest={result['manifest_path']}")
    print(f"source={result['source_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
