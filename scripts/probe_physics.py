#!/usr/bin/env python
"""M3.0 物理探针：记录改造前基线行为（固定种子）。

不改环境；只实例化现有 env 并以固定动作跑一个 episode，输出可复核的基线 JSON。
"""

from __future__ import annotations

import json
import time

import numpy as np

from envs.idc_price_env import IDCPriceEnv20D


def main() -> None:
    env = IDCPriceEnv20D()
    env.reset(seed=0)

    report: dict = {
        "action_dim": int(env.action_dim),
        "obs_dim": int(env.obs_dim),
        "num_server_groups": int(env.model.N),
        "horizon": int(env.horizon),
        "soc_init": float(env.bess_soc),
        "soc_min": float(env.bess_soc_min),
        "soc_max": float(env.bess_soc_max),
    }

    action = np.full(env.action_dim, 0.5, dtype=np.float32)
    socs = [float(env.bess_soc)]
    step_times: list[float] = []
    planned_capacities: list[float] = []
    deadline_misses: list[int] = []
    charge_kwh: list[float] = []
    discharge_kwh: list[float] = []
    grid_kw: list[float] = []
    pv_used_kw: list[float] = []

    for _ in range(env.horizon):
        t0 = time.perf_counter()
        _, _, terminated, truncated, info = env.step(action)
        step_times.append(time.perf_counter() - t0)
        socs.append(float(env.bess_soc))
        planned_capacities.append(float(info["planned_capacity"]))
        deadline_misses.append(int(info["new_deadline_miss_count"]))
        charge_kwh.append(float(info["bess_charge_kWh"]))
        discharge_kwh.append(float(info["bess_discharge_kWh"]))
        grid_kw.append(float(info["P_grid_kW"]))
        pv_used_kw.append(float(info["pv_used_kW"]))
        if terminated or truncated:
            break

    report["soc_range_observed"] = [min(socs), max(socs)]
    report["charge_discharge_simultaneous_steps"] = sum(
        1 for c, d in zip(charge_kwh, discharge_kwh) if c > 1e-6 and d > 1e-6
    )
    report["planned_capacity_is_scalar"] = True  # np.sum 塌缩（M3.1 目标改为向量）
    report["planned_capacity_sample"] = planned_capacities[:3]
    report["deadline_miss_total"] = sum(deadline_misses)
    report["grid_kw_mean"] = float(np.mean(grid_kw)) if grid_kw else None
    report["pv_used_kw_mean"] = float(np.mean(pv_used_kw)) if pv_used_kw else None
    report["wind_in_energy_balance"] = False  # wt_now 仅入 info（M3.5 目标接入）
    report["step_time_mean_s"] = float(np.mean(step_times)) if step_times else None
    report["step_time_p95_s"] = float(np.percentile(step_times, 95)) if step_times else None

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
