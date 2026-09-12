"""M4.1 — action corrector interface and single-step corrector.

``a_exec = S(s, a_raw)``.

The corrector receives only a SystemSnapshot and a DispatchProposal. It must
NOT call any policy parameters or learned value functions. It minimally
adjusts the raw proposal to satisfy current-step physical constraints and
records why (``DispatchResult.correction_reason``).

Single-step scope (this module):
  * compute suggestions clipped to [0, 1];
  * storage suggestion corrected to respect SOC bounds, charge/discharge
    mutual exclusion, and no-sell-back (discharge capped by local load).

Rolling-horizon feasibility (M4.4) and the compute→power→access coupling build
on this interface; they do not belong to the single-step corrector.
"""

from __future__ import annotations

from typing import Protocol

from contracts import DispatchProposal, DispatchResult, SystemSnapshot


class Corrector(Protocol):
    def correct(self, snapshot: SystemSnapshot, proposal: DispatchProposal) -> DispatchResult: ...


class SingleStepCorrector:
    """Deterministic single-step physical corrector (M4.1)."""

    _REQUIRED_CTX = (
        "soc_min",
        "soc_max",
        "capacity_kwh",
        "charge_power_max_kw",
        "discharge_power_max_kw",
        "charge_efficiency",
        "discharge_efficiency",
        "delta_t_hours",
    )

    def correct(self, snapshot: SystemSnapshot, proposal: DispatchProposal) -> DispatchResult:
        compute_exec = [min(1.0, max(0.0, x)) for x in proposal.compute]
        exec_storage, charge_kw, discharge_kw, reason = self._correct_storage(
            snapshot, proposal.storage
        )
        dt = float(snapshot.corrector_context["delta_t_hours"])
        return DispatchResult(
            raw_compute=list(proposal.compute),
            raw_storage=proposal.storage,
            exec_compute=compute_exec,
            exec_storage=exec_storage,
            energy_flows={
                "bess_charge_kWh": charge_kw * dt,
                "bess_discharge_kWh": discharge_kw * dt,
            },
            correction_reason=reason,
            solve_status="feasible",
        )

    def _correct_storage(
        self, snapshot: SystemSnapshot, storage: float
    ) -> tuple[float, float, float, str]:
        ctx = snapshot.corrector_context
        missing = [k for k in self._REQUIRED_CTX if k not in ctx]
        if missing:
            raise ValueError(f"corrector_context missing required keys: {missing}")

        soc = float(snapshot.soc)
        soc_min = float(ctx["soc_min"])
        soc_max = float(ctx["soc_max"])
        cap = float(ctx["capacity_kwh"])
        cpmax = float(ctx["charge_power_max_kw"])
        dpmax = float(ctx["discharge_power_max_kw"])
        ce = float(ctx["charge_efficiency"])
        de = float(ctx["discharge_efficiency"])
        dt = float(ctx["delta_t_hours"])
        eps = 1e-9

        if storage < -eps:  # charge (negative)
            desired = abs(storage) * cpmax
            headroom_kw = max(soc_max - soc, 0.0) * cap / max(ce * dt, eps)
            charge_kw = min(desired, cpmax, headroom_kw)
            discharge_kw = 0.0
            reason = "none" if desired - charge_kw <= eps else "soc_charge_limit"
            exec_storage = -charge_kw / cpmax if cpmax > eps else 0.0
        elif storage > eps:  # discharge (positive)
            desired = storage * dpmax
            floor_kw = max(soc - soc_min, 0.0) * cap * de / max(dt, eps)
            load_kw = float(ctx.get("max_discharge_to_load_kw", float("inf")))
            discharge_kw = min(desired, dpmax, floor_kw, load_kw)
            charge_kw = 0.0
            if desired - discharge_kw <= eps:
                reason = "none"
            elif load_kw < discharge_kw + eps and load_kw < desired:
                reason = "no_sellback"
            else:
                reason = "soc_discharge_limit"
            exec_storage = discharge_kw / dpmax if dpmax > eps else 0.0
        else:
            charge_kw = discharge_kw = 0.0
            reason = "none"
            exec_storage = 0.0

        return exec_storage, charge_kw, discharge_kw, reason
