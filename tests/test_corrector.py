"""M4.1 gates: the single-step corrector is deterministic and minimally modifies raw actions."""

from __future__ import annotations

import pytest

from contracts import DispatchProposal, SystemSnapshot
from safe_rl.corrector import SingleStepCorrector


def _ctx(**over: float) -> dict[str, float]:
    base: dict[str, float] = {
        "soc_min": 0.1,
        "soc_max": 0.9,
        "capacity_kwh": 10000.0,
        "charge_power_max_kw": 2000.0,
        "discharge_power_max_kw": 2000.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "delta_t_hours": 1.0,
    }
    base.update(over)
    return base


def _snap(soc: float, ctx: dict[str, float] | None = None) -> SystemSnapshot:
    return SystemSnapshot(t=0, task_states=[], soc=soc, corrector_context=ctx or _ctx())


def _prop(storage: float = 0.0) -> DispatchProposal:
    return DispatchProposal(compute=[0.5] * 20, storage=storage)


def test_identity_when_feasible() -> None:
    r = SingleStepCorrector().correct(_snap(0.5), _prop(storage=-0.2))
    assert r.exec_storage == pytest.approx(-0.2)
    assert r.correction_reason == "none"
    assert r.energy_flows["bess_discharge_kWh"] == 0.0


def test_charge_clipped_near_soc_max() -> None:
    r = SingleStepCorrector().correct(_snap(0.89), _prop(storage=-1.0))
    # headroom = (0.9 - 0.89) * 10000 = 100 kWh; max charge kW = 100 / (0.95 * 1) = 105.263
    assert r.exec_storage > -1.0
    assert r.correction_reason == "soc_charge_limit"
    assert r.exec_storage == pytest.approx(-105.263 / 2000.0, rel=1e-3)


def test_discharge_clipped_near_soc_min() -> None:
    r = SingleStepCorrector().correct(_snap(0.11), _prop(storage=1.0))
    # floor = (0.11 - 0.1) * 10000 = 100 kWh; max discharge kW = 100 * 0.95 = 95
    assert r.exec_storage < 1.0
    assert r.correction_reason == "soc_discharge_limit"


def test_no_sellback_caps_discharge() -> None:
    r = SingleStepCorrector().correct(_snap(0.5, _ctx(max_discharge_to_load_kw=500.0)), _prop(storage=1.0))
    assert r.exec_storage == pytest.approx(500.0 / 2000.0)
    assert r.correction_reason == "no_sellback"


def test_mutual_exclusion_and_determinism() -> None:
    c = SingleStepCorrector()
    r1 = c.correct(_snap(0.5), _prop(storage=-0.5))
    r2 = c.correct(_snap(0.5), _prop(storage=-0.5))
    assert r1 == r2
    ch = r1.energy_flows["bess_charge_kWh"]
    dis = r1.energy_flows["bess_discharge_kWh"]
    assert not (ch > 1e-9 and dis > 1e-9)


def test_missing_context_raises() -> None:
    snap = SystemSnapshot(t=0, task_states=[], soc=0.5, corrector_context={})
    with pytest.raises(ValueError):
        SingleStepCorrector().correct(snap, _prop(storage=0.0))
