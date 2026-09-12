#!/usr/bin/env python
"""M8.2 代表性轨迹离线 AC/OPF：正常/临界/失败轨迹输出 AC 成功率/电压/负载/失败原因。

不改 grid_model/；不将离线指标用于选择测试日期或包装真实新加坡结果。
"""

from __future__ import annotations

import json

from grid_model.ieee14_loader import load_ieee14_case
from grid_model.opf_solver import solve_ac_opf


def validate_trajectories(idc_bus_id: int = 0) -> dict:
    case = load_ieee14_case()
    scenarios = {
        "normal": 25.0,
        "critical": 60.0,
        "failed": 500.0,
    }
    results: dict = {}
    for name, load_mw in scenarios.items():
        r = solve_ac_opf(case, idc_bus_id=idc_bus_id, idc_load_mw=load_mw)
        results[name] = {
            "success": bool(r.success),
            "message": str(r.message),
            "min_voltage_pu": min(r.bus_voltage_pu.values()) if r.success else None,
            "max_line_loading_percent": max(r.line_loading_percent.values()) if r.success else None,
            "max_trafo_loading_percent": (
                max(r.transformer_loading_percent.values()) if r.success else None
            ),
        }
    return results


if __name__ == "__main__":
    print(json.dumps(validate_trajectories(), indent=2, ensure_ascii=False))
