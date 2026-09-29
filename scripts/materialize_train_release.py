"""M1.3g-f-c-k：物化 / 校验 canonical **正式训练发布产物**。

```bash
uv run python -m scripts.materialize_train_release --verify   # 只读校验（幂等）
uv run python -m scripts.materialize_train_release            # 物化（原子、拒绝覆盖语义不同者）
```

**顺序约束**：所有受 `TRAIN_RELEASE_SOURCE_PATHS` 约束的代码必须**先提交**，再物化——
否则产物会记录一个「包含了未提交改动」的 revision。物化器自身带 dirty 门。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许以 `python scripts/materialize_train_release.py` 直接运行（补仓库根到 sys.path）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scenario.formal_train_release as tr  # noqa: E402


def _canonical_path() -> Path:
    return tr.canonical_train_release_path()


def verify() -> dict:
    payload = tr.load_verified_train_release()
    return {
        "path": str(_canonical_path()),
        "sha256": tr._sha256_file(_canonical_path()),
        "readiness": dict(payload["readiness"]),
    }


def materialize() -> dict:
    out_path = _canonical_path()
    if tr._generator_is_dirty():
        raise tr.TrainReleaseError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书")

    candidate = tr.build_train_release()
    if out_path.is_file():
        existing = tr.load_verified_train_release()
        if existing != candidate:
            raise tr.TrainReleaseError(
                f"{out_path} 与候选**语义不同**：拒绝覆盖已冻结的发布产物")
        return {"out_path": out_path, "sha256": tr._sha256_file(out_path),
                "written": False}

    tr.write_train_release(out_path, candidate)
    return {"out_path": out_path, "sha256": tr._sha256_file(out_path),
            "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物化 / 校验正式训练发布产物（M1.3g-f-c-k）")
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
    except (tr.TrainReleaseError, OSError, ValueError) as error:
        print(f"materialize_train_release: {error}", file=sys.stderr)
        return 1

    print(f"out_path={result['out_path']}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
