"""M4.2 单步物理修正器（纯函数，不接训练、不调用学习参数）。

只接受 `(SystemSnapshot, DispatchProposal)`，返回修正后的 `DispatchResult`。
确定性：同一输入产生位级相同结果（无随机性）。
"""

from __future__ import annotations

import numpy as np

from contracts.models import DispatchProposal, DispatchResult, SystemSnapshot


def correct(snapshot: SystemSnapshot, proposal: DispatchProposal) -> DispatchResult:
    raw_compute = [float(x) for x in proposal.compute_actions]
    raw_storage = float(proposal.storage_action)
    reasons: list[str] = []

    # 1. 计算范围 [0,1]
    exec_compute = [float(np.clip(x, 0.0, 1.0)) for x in raw_compute]
    if any(a != b for a, b in zip(exec_compute, raw_compute, strict=False)):
        reasons.append("compute_clip")

    # 2. 储能范围 [-1,1] + SOC 上下限（充放互斥由符号编码）
    exec_storage = float(np.clip(raw_storage, -1.0, 1.0))
    # 符号约定与环境一致：>0 放电、<0 充电
    if exec_storage > 0.0 and snapshot.soc_kwh <= snapshot.soc_min_kwh + 1e-6:
        exec_storage = 0.0
        reasons.append("soc_empty")
    elif exec_storage < 0.0 and snapshot.soc_kwh >= snapshot.soc_max_kwh - 1e-6:
        exec_storage = 0.0
        reasons.append("soc_full")
    if exec_storage != raw_storage:
        reasons.append("storage_clip")

    # 3. 接入上限：估算电网功率并等比缩减 compute，记录缺口
    business_gap = 0.0
    capacity = [max(float(c), 1e-6) for c in snapshot.group_capacity_kw]
    idc_power = sum(c * cap for c, cap in zip(exec_compute, capacity, strict=False))
    charge_power = max(exec_storage, 0.0) * snapshot.soc_max_kwh
    grid_power = idc_power + charge_power
    if grid_power > snapshot.access_limit_kw:
        scale = max(snapshot.access_limit_kw / grid_power, 0.0)
        exec_compute = [c * scale for c in exec_compute]
        business_gap = idc_power * (1.0 - scale)
        reasons.append("access_limit")

    return DispatchResult(
        raw_compute_actions=raw_compute,
        raw_storage_action=raw_storage,
        exec_compute_actions=exec_compute,
        exec_storage_action=exec_storage,
        correction_reason=",".join(reasons) if reasons else "none",
        business_gap=float(business_gap),
        solve_time_s=0.0,
        p_grid_kw=float(grid_power),
        soc_next_kwh=float(snapshot.soc_kwh),
        cost_sgd=0.0,
        carbon_kg=0.0,
    )
