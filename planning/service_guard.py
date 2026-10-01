"""Reserve current service power without reading realized exogenous arrays."""

from __future__ import annotations

import numpy as np

from contracts.inventory import ArrivedServiceReserve
from idc_model.allocation import allocate_tasks

SERVICE_GUARD_VERSION = "arrived-service-reserve-v1"


def build_service_guard(env, tasks, capacity, *, temperature, arrival):
    version = getattr(env, "terminal_service_guard_version", None)
    if version is None:
        return None
    if version != SERVICE_GUARD_VERSION:
        raise ValueError("unknown terminal service guard version")
    if not bool(getattr(env, "terminal_inventory_enabled", False)):
        raise ValueError("service guard requires terminal inventory contract")
    margin = float(getattr(env, "terminal_service_temperature_margin_c", 0.))
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("invalid service temperature margin")
    now = int(env.current_step)
    live = {str(t.task_id): t for t in env.tasks if t.status != "not_arrived"}
    summaries = [dict(task_id=t.task_id, remaining_work=t.remaining_work,
                      max_rate=t.max_rate_work_per_step, priority=t.priority,
                      deadline=t.deadline, arrival=int(live[t.task_id].arrival_time))
                 for t in tasks]
    # Keep the executor's exact priority/deadline/arrival/id ordering. Serving a
    # critical task requires the preceding demands in that ordering as well.
    order = sorted(range(len(tasks)), key=lambda i: (
        -summaries[i]["priority"], summaries[i]["deadline"],
        summaries[i]["arrival"], summaries[i]["task_id"]))
    prefix = 0.
    required = 0.
    for i in order:
        task = tasks[i]
        demand = min(task.remaining_work, task.max_rate_work_per_step)
        runtime = live[task.task_id]
        deadline_need = max(task.remaining_work - task.max_rate_work_per_step
                            * max(min(task.deadline, int(env.horizon)) - now - 1, 0), 0.)
        continuity_need = (demand if not runtime.interruptible
                           and runtime.start_time is not None and demand > 0 else 0.)
        need = min(demand, max(deadline_need, continuity_need))
        if need > 1e-8:
            required = max(required, prefix + need)
        prefix += demand
    full = allocate_tasks(summaries, list(capacity))
    full_group = np.asarray(full.matrix, dtype=float).reshape(len(tasks), len(capacity)).sum(axis=0)
    floor_capacity = np.zeros(len(capacity))
    remaining = required
    for g, cap in enumerate(capacity):
        floor_capacity[g] = min(cap, remaining)
        remaining -= floor_capacity[g]
    floor = allocate_tasks(summaries, floor_capacity.tolist())
    actual_floor = np.asarray(floor.matrix, dtype=float).reshape(
        len(tasks), len(capacity)).sum(axis=0)
    # Current demand is exact arrived work. Future reserve assumes earliest
    # processing of that known work plus SAME-STEP processing of the causal B6
    # aggregate arrival forecast; no future task instances enter the model.
    # Reserve charging throughout the REAL remainder, so recovery cannot be
    # postponed into future headroom inconsistent with the execution reserve.
    rates = np.asarray(env.model.C_server, dtype=float) * float(env.delta_t_hours)
    remaining_known = [dict(t) for t in summaries]
    powers, charge_limits = [], []
    for k, temp in enumerate(temperature):
        if k == 0:
            known = full
            group = full_group.copy()
        else:
            known = allocate_tasks(remaining_known, list(capacity))
            group = np.asarray(known.matrix, dtype=float).reshape(
                len(tasks), len(capacity)).sum(axis=0)
            reserve_work = max(float(arrival[k]), 0.)
            for g, cap in enumerate(capacity):
                extra = min(cap - group[g], reserve_work)
                group[g] += extra
                reserve_work -= extra
        loads = float(env.base_load) + group / np.maximum(rates, 1e-6)
        power = float(env._idc_power_kw(np.clip(loads, 0., 1.), float(temp) + margin))
        powers.append(power)
        charge_limits.append(min(float(env.bess_charge_power_max_kW),
                                 max(float(env.access_limit_kw) - power, 0.)))
        for i, row in enumerate(known.matrix):
            remaining_known[i]["remaining_work"] = max(
                remaining_known[i]["remaining_work"] - sum(row), 0.)
    return ArrivedServiceReserve(
        temperature_margin_c=margin, charge_limit_kw=charge_limits[0],
        charge_limits_kw=charge_limits,
        reserved_service_power_kw=powers[0],
        group_work_floor=actual_floor.tolist(), current_allocation=floor.matrix,
        note="Current arrived deadlines and running noninterruptible work retain executor order; "
        "charging reserve assumes zero renewables at each remaining step, B6 temperature plus "
        "registered train margin, "
        "earliest known work plus same-step B6 aggregate arrivals; no unseen task instances. "
        "Reachability under these planning assumptions is not a physical impossibility proof.")
