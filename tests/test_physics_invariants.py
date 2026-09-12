"""M3.0 physics-invariant gates.

The three invariants that already hold in the current environment are asserted
green. The two xfail tests document the N1/N6 targets that M3.1-M3.6 will make
green; they are expected to fail until those cards land.
"""

from __future__ import annotations

import numpy as np
import pytest


def _action(compute: list[float], bess_action: float = 0.5) -> np.ndarray:
    vec = np.zeros(23, dtype=np.float32)
    vec[: len(compute)] = compute
    vec[22] = bess_action
    return vec


def test_soc_within_bounds(base_env) -> None:
    base_env.reset()
    socs = [base_env.bess_soc]
    for bess in (0.0, 1.0, 0.0, 1.0, 0.5):
        *_rest, info = base_env.step(_action([0.3] * base_env.model.N, bess_action=bess))
        socs.append(float(info["bess_soc"]))
    assert all(base_env.bess_soc_min - 1e-9 <= s <= base_env.bess_soc_max + 1e-9 for s in socs)


def test_charge_discharge_mutually_exclusive(base_env) -> None:
    base_env.reset()
    for bess in (0.0, 1.0, 0.0, 1.0, 0.5):
        *_rest, info = base_env.step(_action([0.3] * base_env.model.N, bess_action=bess))
        charge = float(info["bess_charge_power_kW"])
        discharge = float(info["bess_discharge_power_kW"])
        assert not (charge > 1e-9 and discharge > 1e-9)


def test_energy_balance(base_env) -> None:
    base_env.reset()
    *_rest, info = base_env.step(_action([0.5] * base_env.model.N, bess_action=0.0))  # charge
    grid = float(info["grid_energy_kWh"])
    idc = float(info["idc_energy_kWh"])
    charge = float(info["bess_charge_kWh"])
    discharge = float(info["bess_discharge_kWh"])
    pv_used = float(info["pv_used_kWh"])
    expected = max(idc + charge - discharge - pv_used, 0.0)
    assert abs(grid - expected) < 1e-6


@pytest.mark.xfail(reason="N6: action_dim is 23 until M3.6 collapses it to 20+1=21", strict=False)
def test_action_dim_is_21(base_env) -> None:
    assert base_env.action_dim == 21


@pytest.mark.xfail(reason="N1: per-group execution is not reported until M3.2", strict=False)
def test_per_group_execution_reported(base_env) -> None:
    base_env.reset()
    *_rest, info = base_env.step(_action([0.5] * base_env.model.N))
    assert "group_exec_work" in info
