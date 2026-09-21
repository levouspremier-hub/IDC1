"""M1.3g-f-b-b：物化 / 校验 canonical formal env 发布产物。

用法：

```bash
uv run python -m scripts.materialize_env_release --verify   # 只读校验（幂等）
uv run python -m scripts.materialize_env_release            # 物化（原子、拒绝覆盖）
```

**顺序约束**（卡 §bp.5 第 3 条）：所有受 `ENV_RELEASE_SOURCE_PATHS` 约束的代码
必须**先提交**，再物化——否则产物会记录一个「包含了未提交改动」的 revision。
物化器自身带 dirty 门，改完代码却忘记提交时会 fail closed。

**拒绝覆盖**：既存文件与候选**语义不同**时一律拒绝，绝不覆盖。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scenario.env_release as er


def _canonical_release_path() -> Path:
    """物化目标路径（单独一层，便于测试替换）。"""
    return er.canonical_release_path()


def verify() -> dict:
    """只读校验 canonical 产物；幂等，可重复调用。"""
    payload = er.load_verified_env_release()
    return {
        "path": str(_canonical_release_path()),
        "sha256": er._sha256_file(_canonical_release_path()),
        "readiness": dict(payload["readiness"]),
    }


def materialize() -> dict:
    """生成 canonical 产物；原子写入、拒绝覆盖不同的既存文件。"""
    out_path = _canonical_release_path()
    if er._generator_is_dirty():
        raise er.EnvReleaseError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书")

    candidate = er.build_env_release()
    if out_path.is_file():
        existing = er.load_verified_env_release(out_path)
        if existing != candidate:
            raise er.EnvReleaseError(
                f"{out_path} 与候选**语义不同**：拒绝覆盖已冻结的发布产物")
        return {"out_path": out_path, "sha256": er._sha256_file(out_path),
                "written": False}

    er.write_env_release(out_path, candidate)
    return {"out_path": out_path, "sha256": er._sha256_file(out_path), "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物化 / 校验 formal env 独立发布产物（M1.3g-f-b-b）")
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    args = parser.parse_args(argv)

    try:
        if args.verify:
            info = verify()
            print(f"verified {info['path']}")
            print(f"sha256={info['sha256']}")
            print(f"readiness={info['readiness']}")
            return 0
        result = materialize()
    except (er.EnvReleaseError, OSError, ValueError) as error:
        print(f"materialize_env_release: {error}", file=sys.stderr)
        return 1

    print(f"out_path={result['out_path']}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
