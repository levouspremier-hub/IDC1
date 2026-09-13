"""M4.4c 测试：收敛为唯一 H 步正确器（旧 safe_rl/corrector.py 已退役）。

唯一正确器实现为 `planning/corrector.py` + `safe_rl/corrector_wrapper.py`。
"""

import importlib
import inspect
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

import planning.corrector as planning_corrector
import safe_rl.corrector_wrapper as wrapper_mod
from envs.idc_price_env import IDCPriceEnv20D
from safe_rl.corrector_wrapper import CorrectorWrapper

N_GROUP = 20


# --- 1. 旧模块已退役 ---

def test_legacy_safe_rl_corrector_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("safe_rl.corrector")


def test_legacy_corrector_file_absent():
    assert not Path("safe_rl/corrector.py").exists()


def test_no_source_references_legacy_corrector():
    """现行 Python 源码（生产 + 脚本）不得引用 safe_rl.corrector。"""
    pattern = re.compile(r"(from|import)\s+safe_rl\.corrector(?!_wrapper)")
    offenders: list[str] = []
    for root in ("planning", "safe_rl", "scripts", "contracts"):
        for path in Path(root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path}:{i}")
    assert offenders == [], offenders


def test_makefile_does_not_scan_legacy_corrector():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "safe_rl/corrector.py" not in makefile
    assert "safe_rl/corrector_wrapper.py" in makefile


# --- 2. wrapper 只使用 planning.corrector.correct ---

def test_wrapper_uses_only_planning_corrector():
    src = inspect.getsource(wrapper_mod)
    assert "from planning.corrector import correct" in src
    assert "safe_rl.corrector" not in src.replace("safe_rl.corrector_wrapper", "")


def test_wrapper_calls_planning_corrector(monkeypatch):
    seen = {}
    real = planning_corrector.correct

    def _spy(snapshot, proposal, *, time_limit_s):
        seen["called"] = True
        return real(snapshot, proposal, time_limit_s=time_limit_s)

    monkeypatch.setattr(wrapper_mod, "correct", _spy)
    env = CorrectorWrapper(
        IDCPriceEnv20D(horizon=6, access_limit_kw=1000.0),
        corrector_time_limit_s=5.0,
    )
    env.reset(seed=0)
    a = np.concatenate([np.full(N_GROUP, 0.5, dtype=np.float32), np.array([0.0], dtype=np.float32)])
    env.step(a)
    assert seen.get("called") is True


# --- 3. 现行 wrapper 回归（raw/exec 分离、MIP backend、显式预算） ---

def test_wrapper_regression_raw_exec_and_backend():
    env = CorrectorWrapper(
        IDCPriceEnv20D(horizon=6, access_limit_kw=1000.0),
        corrector_time_limit_s=5.0,
    )
    env.reset(seed=0)
    raw = np.concatenate(
        [np.full(N_GROUP, 0.6, dtype=np.float32), np.array([0.1], dtype=np.float32)]
    )
    _, _, _, _, info = env.step(raw)
    np.testing.assert_allclose(np.asarray(info["raw_action"]), raw, atol=0.0)
    assert np.asarray(info["exec_action"]).shape == (N_GROUP + 1,)
    assert info["planner_backend"] == "mip"
    assert "stage_a_status" in info and "projection_offset" in info


def test_wrapper_requires_explicit_budget():
    with pytest.raises(TypeError):
        CorrectorWrapper(IDCPriceEnv20D(horizon=6))  # type: ignore[call-arg]


def test_no_legacy_module_importable_at_runtime():
    """子进程再确认一次（避免仅在进程内缓存影响）。"""
    out = subprocess.run(
        ["uv", "run", "python", "-c",
         "import importlib.util, sys;"
         "spec = importlib.util.find_spec('safe_rl.corrector');"
         "sys.exit(0 if spec is None else 1)"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
