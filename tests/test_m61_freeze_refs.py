"""M6.1 测试：冻结参考值存在、字段齐全、缺失即失败、不可重算。"""

import json
from pathlib import Path

import pytest

FROZEN_REFS_PATH = Path("configs/frozen_refs/refs.json")


def _load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"无冻结参考值文件 {path}（M6.1 需先冻结）")
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_refs_exist_and_complete():
    refs = _load(FROZEN_REFS_PATH)
    assert refs["schema_version"] == "frozen-refs-v1"
    for key in ["price_ref", "queue_ref", "cost_ref", "carbon_ref", "peak_power_ref_kW"]:
        assert key in refs["references"]
    assert "source" in refs
    assert "data_hash" in refs


def test_loader_fails_without_file(tmp_path):
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        _load(missing)
