"""Reserve current service power without reading realized exogenous arrays."""

from __future__ import annotations

import numpy as np

from contracts.inventory import ArrivedServiceReserve
from idc_model.allocation import allocate_tasks

SERVICE_GUARD_VERSION = "arrived-service-reserve-v1"


def build_service_guard(env, tasks, capacity, *, temperature):
    version = getattr(env, "terminal_service_guard_version", None)
    if version is None:
        return None
    if version != SERVICE_GUARD_VERSION:
        raise ValueError("unknown terminal service guard version")
    if not bool(getattr(env, "terminal_inventory_enabled", False)):
        raise ValueError("service guard requires terminal inventory contract")
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
    # Reserve ALL currently processable work against charging, using nonlinear
    # IDC physics at the visible B6 temperature. Renewable supply is assumed zero
    # for this current reserve; future supplies retain the declared B6 extension.
    rates = np.asarray(env.model.C_server, dtype=float) * float(env.delta_t_hours)
    loads = float(env.base_load) + full_group / np.maximum(rates, 1e-6)
    power = float(env._idc_power_kw(np.clip(loads, 0., 1.), float(temperature)))
    charge_limit = min(float(env.bess_charge_power_max_kW),
                       max(float(env.access_limit_kw) - power, 0.))
    return ArrivedServiceReserve(
        charge_limit_kw=charge_limit, reserved_service_power_kw=power,
        group_work_floor=actual_floor.tolist(), current_allocation=floor.matrix,
        note="Current arrived deadlines and running noninterruptible work retain executor order; "
        "charging reserve assumes zero current renewables and visible B6 temperature. "
        "Reachability under these planning assumptions is not a physical impossibility proof.")
