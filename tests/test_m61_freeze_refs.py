"""M6.1 测试：冻结参考值存在、字段齐全、缺失即失败、不可重算。

**M1.3g-d 迁移**：refs 由声明的 `frozen-refs-v1` 升到 **train-only 的
`frozen-refs-v2`**；本文件只把 schema 与字段断言随新形状迁移
（逐值 provenance、顶层训练区间与来源绑定），
**原有意图一条都未弱化**：文件必须存在、字段必须齐全、缺失必须失败。
"""

import json
from pathlib import Path

import pytest

FROZEN_REFS_PATH = Path("configs/frozen_refs/refs.json")
SCHEMA_V2 = "frozen-refs-v2"


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"无冻结参考值文件 {path}（M6.1 需先冻结）")
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_refs_exist_and_complete():
    refs = _load(FROZEN_REFS_PATH)
    assert refs["schema_version"] == SCHEMA_V2
    for key in ["price_ref", "queue_ref", "cost_ref", "carbon_ref", "peak_power_ref_kW"]:
        assert key in refs["references"]
    # M1.3g-d：来源不再是自由字段，而是**逐值结构化 provenance**
    assert "training_range" in refs
    assert "sources" in refs
    assert "materializer_revision" in refs
    assert refs["materializer_revision"]


def test_every_reference_carries_structured_provenance():
    """v1 的自由字段（`source` / `data_hash` / `frozen`）已退役。"""
    refs = _load(FROZEN_REFS_PATH)
    for name, entry in refs["references"].items():
        assert set(entry) == {
            "value", "unit", "source_kind", "method", "derived_from",
            "training_range", "binding",
        }, name
        assert entry["source_kind"] in ("train_derived", "declared_physical_scale")
    for retired in ("source", "data_hash", "frozen"):
        assert retired not in refs, f"v2 不再有自由字段 {retired!r}"


def test_loader_fails_without_file(tmp_path):
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        _load(missing)
