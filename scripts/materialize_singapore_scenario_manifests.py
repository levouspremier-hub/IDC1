#!/usr/bin/env python
"""M1.3g-c：物化**正式 split manifest triad**（`train` / `validation` / `test`）。

三份 manifest **只声明**冻结信任链与合法 origin 集合，**不保存** forecast 数值、
**不启动** env 或训练。严格 schema / 校验**唯一**由
`scenario.formal_split_manifests` 提供（物化器与未来 reader 共用）。

## 原子性（triad）

三份文件是**一个整体**：

- 首次物化：任一步写入失败 → **删除本次创建的全部文件**，目录保持为空；
- 三份**完全相同** → 不重写，`bytes` / `sha256` / `mtime_ns` **全不变**；
- 任一不同 / 仅部分存在 / 畸形既有文件 → **拒绝覆盖**。

## 信任根

公开签名**只接受路径**：不存在 `expected_*`、revision 覆盖、DataFrame 注入
或 `**kwargs`。唯一的 refs 是 `configs/frozen_refs/refs_v3.json`；
`refs.json`（v2）**明确拒绝**。

用法：

```bash
uv run python scripts/materialize_singapore_scenario_manifests.py
uv run python scripts/materialize_singapore_scenario_manifests.py --verify
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.formal_split_manifests import (
    FORMAL_SPLIT_DIR,
    SplitManifestError,
    build_split_manifest,
    load_verified_split_manifest,
    utc_now,
)
from scenario.splits import SPLIT_NAMES


class ScenarioManifestError(ValueError):
    """正式 split manifest triad 物化的**明确失败**。"""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _generator_is_dirty() -> bool:
    """与 schema 模块共用**同一**实现来源（避免两处漂移）。"""
    from scenario.formal_split_manifests import _generator_is_dirty as check

    return check()


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


def _existing_payloads(paths: dict[str, Path]) -> dict[str, dict] | None:
    """读取既有 triad；**部分存在 / 畸形 / schema 不符**一律拒绝。

    返回 `None` 表示三份都不存在（首次物化）。
    """
    present = {split: path.is_file() for split, path in paths.items()}
    if not any(present.values()):
        return None
    if not all(present.values()):
        raise ScenarioManifestError(
            f"正式 triad 只存在一部分：{[s for s, ok in present.items() if ok]}；"
            "拒绝在残缺状态上继续（既不补写也不覆盖）"
        )
    return {
        split: load_verified_split_manifest(path, expected_split=split)
        for split, path in paths.items()
    }


def materialize_split_manifest_triad(
    *,
    out_dir: Path | str = FORMAL_SPLIT_DIR,
    frozen_at_utc: str | None = None,
) -> dict:
    """物化 / 校验三份正式 split manifest（**原子 triad**）。

    **签名只有 `out_dir` 与冻结时间戳**：八个输入角色由
    `scenario.formal_split_manifests` **固定**，调用者**无法**传入
    `inputs` mapping、路径、`materializer_revision`、`expected_*` 信任根
    或 DataFrame 注入 —— 未知关键字一律 `TypeError`。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _generator_is_dirty():
        raise ScenarioManifestError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书（未提交）"
        )

    paths = {split: out_dir / f"{split}.json" for split in SPLIT_NAMES}
    existing = _existing_payloads(paths)

    # **R2 修正**：三个 split 必须共享**同一个** `frozen_at_utc`。
    # 若每个 `build_split_manifest` 各自采样墙钟，慢速首冻会写出**互不相同**
    # 的时间戳，使随后的幂等 `--verify` 以「语义不同」fail closed（实测 flake）。
    frozen_at = (
        existing["train"]["frozen_at_utc"] if existing is not None
        else (frozen_at_utc or utc_now())
    )
    candidates = {
        split: build_split_manifest(split, frozen_at_utc=frozen_at)
        for split in SPLIT_NAMES
    }

    if existing is not None:
        differing = [split for split in SPLIT_NAMES if existing[split] != candidates[split]]
        if differing:
            raise ScenarioManifestError(
                f"既有的正式 split manifest 与候选**语义不同**：{differing}；"
                "拒绝覆盖已冻结的 triad"
            )
        return {
            "paths": paths,
            "written": False,
            "sha256": {split: _sha256_file(paths[split]) for split in SPLIT_NAMES},
        }

    created: list[Path] = []
    try:
        for split in SPLIT_NAMES:
            _atomic_write_text(paths[split], _canonical_json(candidates[split]))
            created.append(paths[split])
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {
        "paths": paths,
        "written": True,
        "sha256": {split: _sha256_file(paths[split]) for split in SPLIT_NAMES},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物化正式 split manifest triad（M1.3g-c）")
    parser.add_argument("--verify", action="store_true",
                        help="只校验 / 幂等重跑（不改变既有文件）")
    parser.add_argument("--out-dir", default=str(FORMAL_SPLIT_DIR))
    args = parser.parse_args(argv)

    try:
        result = materialize_split_manifest_triad(out_dir=Path(args.out_dir))
    except (ScenarioManifestError, SplitManifestError, OSError, ValueError) as error:
        print(f"materialize_singapore_scenario_manifests: {error}", file=sys.stderr)
        return 1

    for split, path in result["paths"].items():
        print(f"{split}: {path} sha256={result['sha256'][split]}")
    print(f"written={result['written']}")
    if args.verify:
        for split, path in result["paths"].items():
            payload = json.loads(path.read_text(encoding="utf-8"))
            print(f"{split}: candidates={payload['candidate_origins']} "
                  f"readiness={json.dumps(payload['readiness'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
