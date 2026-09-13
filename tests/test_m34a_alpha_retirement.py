"""M3.4a 测试：旧 α 路径彻底退役（无残留、显式拒绝、逐组功耗差异保留）。"""

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from envs.idc_price_env import IDCPriceEnv20D

# 片段拼接：避免测试文件自身含待查字面量（否则会自命中）。
ALPHA_TOKENS = ("planned_load_" + "reserve_" + "alpha", "reserve_" + "alpha")


def test_env_signature_has_no_alpha():
    params = inspect.signature(IDCPriceEnv20D.__init__).parameters
    for token in ALPHA_TOKENS:
        assert token not in params, f"构造签名仍含 {token}"


def test_env_instance_has_no_alpha_attr():
    env = IDCPriceEnv20D()
    for token in ALPHA_TOKENS:
        assert not hasattr(env, token), f"实例仍有属性 {token}"


def test_env_has_no_legacy_alpha_method():
    assert not hasattr(IDCPriceEnv20D, "_actual_loads_from_completed_work")


def test_passing_alpha_is_explicitly_rejected():
    kwargs = {"planned_load_" + "reserve_" + "alpha": 0.4}
    with pytest.raises(TypeError):
        IDCPriceEnv20D(**kwargs)  # type: ignore[arg-type]


def test_run_config_has_no_alpha():
    import configs.config_ultimate as cfg

    flat = json.dumps(cfg.ENV_CONFIG, default=str)
    for token in ALPHA_TOKENS:
        assert token not in flat, f"ENV_CONFIG 仍含 {token}"


def test_frozen_refs_have_no_alpha():
    refs = json.loads(Path("configs/frozen_refs/refs.json").read_text(encoding="utf-8"))
    for token in ALPHA_TOKENS:
        assert token not in json.dumps(refs), f"frozen refs 仍含 {token}"


def test_group_distribution_still_changes_power():
    """同总完成量、不同组分布 → 逐组功耗差异仍存在（α 退役不削弱物理链）。"""
    env = IDCPriceEnv20D()
    a = np.full(env.action_dim, 0.5, dtype=np.float32)
    env.reset(seed=0)
    _, _, _, _, i1 = env.step(a)
    env.reset(seed=0)
    concentrated = a.copy()
    concentrated[:20] = 0.0
    concentrated[0] = 1.0
    _, _, _, _, i2 = env.step(concentrated)
    g1 = np.asarray(i1["completed_work_by_group"])
    g2 = np.asarray(i2["completed_work_by_group"])
    assert not np.allclose(g1, g2), "不同组分布应产生不同逐组完成量/功耗"


def test_no_alpha_residue_in_code_config_scripts_tests():
    """rg 等价检查：运行代码/配置/脚本/测试中不得有 alpha 残留。"""
    roots = ["envs", "configs", "scripts", "tests"]
    hits: list[str] = []
    for root in roots:
        for path in Path(root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in ALPHA_TOKENS:
                if token in text:
                    hits.append(f"{path}: {token}")
    assert hits == [], f"仍有 alpha 残留：{hits}"
