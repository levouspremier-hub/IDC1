"""Reserve current service power without reading realized exogenous arrays."""

from __future__ import annotations

import numpy as np

from contracts.inventory import ArrivedServiceReserve
from idc_model.allocation import allocate_tasks

SERVICE_GUARD_VERSION = "arrived-service-reserve-v2"


def power_upper_envelope(env, temperature):
    """Derivative bound for existing IT/COP model over loads [base, 1].

    k in [1,2] makes IT increasing and concave. Its maximum derivative is
    at base; COP's minimum and maximum IT bound the cooling derivative.
    This is a power majorant under the declared temperature assumption,
    not a bound on forecast error or realized temperature.
    """
    model = env.model
    base_load = float(env.base_load)
    if (not 0 <= base_load <= 1 or not 1 <= model.k <= 2
            or np.any(model.P_max < model.P_idle)):
        raise ValueError("unregistered nonlinear power envelope domain")
    cop_min = max(model.alpha * (model.T_target - float(temperature))
                  + model.beta * (1. if model.beta < 0 else base_load) + model.gamma, .1)
    it_max = float(model.calc_it_power(np.ones(model.N)))
    it_derivative = (model.P_max - model.P_idle) * (
        2. - model.k * base_load ** (model.k - 1.))
    derivative = it_derivative * (1. + 1. / cop_min)
    derivative += it_max * max(-model.beta, 0.) / (model.N * cop_min ** 2)
    rates = np.asarray(model.C_server) * float(env.delta_t_hours)
    coefficients = derivative / np.maximum(rates, 1e-6) / 1000.
    base = float(env._idc_power_kw(np.full(model.N, base_load), float(temperature)))
    return base, coefficients


def _coupled_reserve(env, summaries, capacity, floor, order, temperature, arrival, margin):
    remaining = [dict(task) for task in summaries]
    schedule, bases, coefficients, powers, limits = [], [], [], [], []
    arrivals, served, backlog = [], [], [0.]
    for k, temp in enumerate(temperature):
        allocation = allocate_tasks(remaining, list(capacity))
        matrix = np.asarray(allocation.matrix).reshape(len(summaries), len(capacity))
        group = matrix.sum(axis=0)
        amount = 0. if k == 0 else max(float(arrival[k]), 0.)
        demand = backlog[-1] + amount
        aggregate_work = min(demand, max(sum(capacity) - float(group.sum()), 0.))
        backlog.append(max(demand - aggregate_work, 0.))
        base, coefficient = power_upper_envelope(env, float(temp) + margin)
        power = base + float(coefficient @ group) + float(coefficient.max()) * aggregate_work
        schedule.append(allocation.matrix)
        bases.append(base)
        coefficients.append(coefficient.tolist())
        powers.append(power)
        limits.append(min(float(env.bess_charge_power_max_kW),
                          max(float(env.access_limit_kw) - power, 0.)))
        arrivals.append(amount)
        served.append(aggregate_work)
        for i in range(len(summaries)):
            remaining[i]["remaining_work"] = max(
                remaining[i]["remaining_work"] - float(matrix[i].sum()), 0.)
    per_task = np.asarray(schedule).reshape(len(temperature), len(summaries), len(capacity))
    end = per_task.sum(axis=(0, 2)).tolist()
    due = []
    shortfall = 0.
    for i, task in enumerate(summaries):
        cutoff = max(0, min(len(temperature), task["deadline"] - int(env.current_step)))
        due.append(float(per_task[:cutoff, i].sum()))
        shortfall += max(task["remaining_work"] - min(end[i], due[-1]), 0.)
    return ArrivedServiceReserve(
        version=SERVICE_GUARD_VERSION, temperature_margin_c=margin,
        charge_limit_kw=limits[0], charge_limits_kw=limits,
        reserved_service_power_kw=powers[0], group_work_floor=floor.sum(axis=0).tolist(),
        current_allocation=floor.tolist(), known_service_allocation=schedule,
        known_service_order=order, known_service_required_end_work=end,
        known_service_required_due_work=due, known_service_shortfall_work=shortfall,
        reserve_base_power_kw=bases, reserve_power_coefficients_kw_per_work=coefficients,
        aggregate_arrival_work=arrivals, aggregate_service_work=served,
        aggregate_backlog_work=backlog,
        note="V2 charging power is coupled to optimized known service and aggregate backlog; "
        "charge_limits_kw describe the complete service witness, not fixed planner bounds. "
        "Forecast arrivals are aggregate reservations with conserved carryover, not future tasks. "
        "Zero renewables and B6 temperature plus registered train margin are assumptions; "
        "reachability is not a realized physical impossibility proof.")


def build_service_guard(env, tasks, capacity, *, temperature, arrival):
    version = getattr(env, "terminal_service_guard_version", None)
    if version is None:
        return None
    if version not in ("arrived-service-reserve-v1", SERVICE_GUARD_VERSION):
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
    if version == SERVICE_GUARD_VERSION:
        return _coupled_reserve(
            env, summaries, capacity, np.asarray(floor.matrix).reshape(len(tasks), len(capacity)),
            order, temperature, arrival, margin)
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
