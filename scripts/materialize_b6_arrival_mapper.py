"""M1.3f-e-b / M1.3g-e-b：物化 **B6 arrival-to-Task mapper 参数 manifest**。

**唯一实现**在 `scenario.arrival_mapper`（mapper 与物化器共用同一严格入口）。
本脚本只是 CLI 壳：

- **仅**接受 `--help` 与 `--verify`；**不接受** out-dir / manifest-path / revision；
- canonical-only；原子写入；既存不同**拒绝覆盖**。

```bash
uv run python scripts/materialize_b6_arrival_mapper.py --verify
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.arrival_mapper import (
    ArrivalMapperError,
    _generator_is_dirty,
    atomic_write_bytes,
    build_mapper_manifest,
    canonical_mapper_manifest_path,
    load_verified_mapper_manifest,
    validate_mapper_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# 物化器自身的模块级别名：测试可通过 monkeypatch 注入失败
_atomic_write_bytes = atomic_write_bytes


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(Path(path), text.encode("utf-8"))


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _c_idc_base_work_per_hour() -> float:
    """B6 冻结硬件实现（`server_seed=0`）的 `C_IDC_base`。

    该解析放在**物化器**（`scripts/`，不在 `make check` 的 mypy 扫描范围内），
    结果写入 manifest，因此 mapper 模块**不**导入 `idc_model.task_model`。
    """
    from idc_model.task_model import IDCEnergyTaskModel

    return float(IDCEnergyTaskModel(task_seed=0, server_seed=0)
                 ._task_workload_capacity_ref())


def materialize_arrival_mapper_manifest(*, frozen_at_utc: str | None = None) -> dict:
    """生成 / 校验 canonical mapper manifest（原子、幂等、拒绝覆盖）。"""
    target = canonical_mapper_manifest_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if _generator_is_dirty():
        raise ArrivalMapperError(
            "生成器有未提交修改：拒绝用旧 revision 为未提交代码背书")

    stamp = frozen_at_utc
    if target.is_file():
        stamp = stamp or load_verified_mapper_manifest(target)["frozen_at_utc"]
    stamp = stamp or datetime.now(UTC).replace(microsecond=0).isoformat()

    candidate = validate_mapper_manifest(build_mapper_manifest(
        frozen_at_utc=stamp,
        c_idc_base_work_per_hour=_c_idc_base_work_per_hour()))

    if target.is_file():
        existing = load_verified_mapper_manifest(target)
        if existing != candidate:
            raise ArrivalMapperError(
                f"{target} 与候选**语义不同**：拒绝覆盖已冻结的 mapper manifest")
        return {"manifest_path": target,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "written": False}

    _atomic_write_text(target, _canonical_json(candidate))
    return {"manifest_path": target,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物化 B6 arrival-to-Task mapper 参数 manifest（M1.3g-e-b）")
    parser.add_argument("--verify", action="store_true", help="只读校验，不写")
    args = parser.parse_args(argv)

    try:
        if args.verify:
            load_verified_mapper_manifest()
            print(f"verified {canonical_mapper_manifest_path()}")
            return 0
        result = materialize_arrival_mapper_manifest()
    except (ArrivalMapperError, OSError, ValueError) as error:
        print(f"materialize_b6_arrival_mapper: {error}", file=sys.stderr)
        return 1

    print(f"manifest_path={result['manifest_path']}")
    print(f"sha256={result['sha256']}")
    print(f"written={result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
