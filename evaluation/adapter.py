"""M6-P1 统一评估适配器：五类方法输出**同一** `EvaluationRecord`。

与旧骨架（M6.2）的差别，逐条对应 `docs/M6_EVALUATION_PROTOCOL.md` §3：

- **`service_qualified` 不再默认 `true`**：必须显式传入服务标准；
  标准缺失或未冻结 ⇒ `None`（**未判定**），且未判定**不等于**达标；
- 成本**分列**购电费 / 电池退化费（SGD），混合目标量另名且**不标作 SGD**；
- 碳排与购电量分列；零完成量时单位碳排为 `None`（不可判定）；
- PV / 风电各自可用 / 使用 / 弃电 + `used/available` 利用率；
  可再生**占比**另名 `renewable_share`；
- 业务服务五分类计数与工作量、期末剩余、不可中断中断次数；
- 物理违规（接入 / SOC / 充放互斥 / 能量守恒）与电网峰值；
- raw→exec 修正、求解耗时中位数/P95、超时与回退；**无修正器 ⇒ 不适用（`None`）**；
- 缺失的原始量登记在 `not_computable`，**不以零代替**。

**本卡不宣称任何性能结论**：受控短跑 checkpoint 只用于验证契约与计量。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from contracts.models import (
    CorrectionMetrics,
    EvaluationRecord,
    PhysicalViolationMetrics,
    ServiceMetrics,
)
from evaluation.metrics import (
    aggregate_physical,
    check_physical_step,
    classify_tasks,
    correction_metrics,
    renewable_metrics,
    step_power_from_info,
)
from evaluation.service_standard import ServiceStandard, qualify

#: 协议 §1 的五类预定方法（顺序即报告顺序）。
PLANNED_METHODS: tuple[str, ...] = (
    "rule_baseline",
    "independent_rolling_optimization",
    "penalty_ppo",
    "safe_ppo_single_step_corrector",
    "safe_ppo_joint_rolling_corrector",
)

#: 动作模式：本卡只实现确定性均值（受控 short run 的固定动作）。
ACTION_MODES: tuple[str, ...] = ("deterministic_mean", "seeded_sample")

STATUS_EVALUATED = "evaluated"
STATUS_NOT_EVALUATED = "not_evaluated"
NOT_EVALUATED_NOTE = (
    "未评估：本卡未提供该方法的受控短跑输入（不得伪造比较结果或中性动作）"
)

CORRECTION_INFO_KEYS = (
    "raw_action", "exec_action", "correction_solve_time_s", "correction_reason",
)


def neutral_rule(obs: np.ndarray, action_dim: int = 21) -> np.ndarray:
    """规则基线：中性动作（compute 0.5、storage 0）。"""
    action = np.full(action_dim, 0.5, dtype=np.float32)
    action[20] = 0.0
    return action


def _episode_horizon(env) -> int:
    unwrapped = getattr(env, "unwrapped", env)
    return int(unwrapped.horizon)


def _observation(env) -> np.ndarray:
    if hasattr(env, "get_wrapper_attr"):
        try:
            provider = env.get_wrapper_attr("policy_observation")
        except AttributeError:
            provider = None
        if provider is not None:
            return np.asarray(provider(), dtype=np.float32)
    unwrapped = getattr(env, "unwrapped", env)
    return np.asarray(unwrapped._get_obs(), dtype=np.float32)


def evaluate(
    env,
    method: str,
    action_fn: Callable[[np.ndarray], np.ndarray],
    *,
    run_id: str,
    service_standard: ServiceStandard | None,
    seed: int = 0,
    action_mode: str = "deterministic_mean",
    checkpoint_id: str | None = None,
    checkpoint_role: str | None = None,
) -> EvaluationRecord:
    """跑一个 episode 并聚合 `EvaluationRecord`。

    `service_standard` **必须显式传入**（可以是 `None`，表示本次不判定资格）；
    它**没有默认值**，因此不存在「默认达标」的调用路径——项目标准
    `m6-service-standard-v1` 已冻结，但**不会**被隐式采用。
    """
    if action_mode not in ACTION_MODES:
        raise ValueError(f"未知 action_mode：{action_mode!r}，必须属于 {list(ACTION_MODES)}")
    if service_standard is not None and not isinstance(service_standard, ServiceStandard):
        raise TypeError(
            f"service_standard 必须是 ServiceStandard 或 None，"
            f"实际 {type(service_standard).__name__}"
        )

    unwrapped = getattr(env, "unwrapped", env)
    env.reset(seed=seed)
    horizon = _episode_horizon(env)
    soc_min_kwh = float(unwrapped.bess_soc_min) * float(unwrapped.bess_capacity_kWh)
    soc_max_kwh = float(unwrapped.bess_soc_max) * float(unwrapped.bess_capacity_kWh)

    physical_rows: list[dict[str, float]] = []
    correction_rows: list[dict[str, Any]] = []
    base_unserved_steps = 0
    steps = 0
    failure: str | None = None
    truncated_by_error = False

    try:
        for _ in range(horizon):
            obs = _observation(env)
            action = np.asarray(action_fn(obs), dtype=np.float32).reshape(-1)
            _obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            physical_rows.append(check_physical_step(
                step_power_from_info(info),
                access_limit_kw=float(unwrapped.access_limit_kw),
                soc_kwh=float(info["bess_energy_kWh"]),
                soc_min_kwh=soc_min_kwh,
                soc_max_kwh=soc_max_kwh,
            ))
            if float(info.get("unserved_base_load_kW", 0.0)) > 1e-6:
                base_unserved_steps += 1
            if all(key in info for key in CORRECTION_INFO_KEYS):
                correction_rows.append(dict(info))
            if terminated or truncated:
                break
    except Exception as error:  # noqa: BLE001 - 失败 run 必须**保留**并标注
        failure = type(error).__name__
        truncated_by_error = True

    not_computable: list[str] = []
    standard = service_standard
    if truncated_by_error:
        not_computable.append("episode.incomplete")

    service, service_missing = classify_tasks(
        unwrapped.tasks,
        horizon=horizon,
        non_interruptible_interruptions=int(
            unwrapped.total_non_interruptible_interruption_count),
    )
    not_computable.extend(service_missing)

    pv, pv_missing = renewable_metrics(
        key="pv",
        available_kwh=float(unwrapped.total_pv_available_kWh),
        used_kwh=float(unwrapped.total_pv_used_kWh),
        curtail_kwh=float(unwrapped.total_pv_curtail_kWh),
    )
    wind, wind_missing = renewable_metrics(
        key="wind",
        available_kwh=float(unwrapped.total_wind_available_kWh),
        used_kwh=float(unwrapped.total_wind_used_kWh),
        curtail_kwh=float(unwrapped.total_wind_curtail_kWh),
    )
    not_computable.extend(pv_missing)
    not_computable.extend(wind_missing)

    completed_work = float(unwrapped.total_completed_work)
    carbon_per_work: float | None = None
    if completed_work > 0.0:
        carbon_per_work = float(unwrapped.total_carbon_emission) / completed_work
    else:
        not_computable.append("carbon_per_completed_work")

    idc_energy = float(unwrapped.total_idc_energy_kWh)
    renewable_share: float | None = None
    if idc_energy > 0.0:
        renewable_share = (
            float(unwrapped.total_pv_used_kWh) + float(unwrapped.total_wind_used_kWh)
        ) / idc_energy
    else:
        not_computable.append("renewable_share")

    correction: CorrectionMetrics | None = None
    if correction_rows:
        correction = correction_metrics(correction_rows)

    physical = aggregate_physical(
        physical_rows, base_load_unserved_steps=base_unserved_steps)
    if failure is not None:
        qualified: bool | None = None
        note = f"未判定：episode 失败（{failure}），失败 run 不得判定达标"
    else:
        qualified, note = qualify_service(service, standard, physical)

    return EvaluationRecord(
        method=str(method),
        run_id=str(run_id),
        action_mode=str(action_mode),
        seed=int(seed),
        steps=int(steps),
        checkpoint_id=None if checkpoint_id is None else str(checkpoint_id),
        checkpoint_role=None if checkpoint_role is None else str(checkpoint_role),
        service=service,
        service_qualified=qualified,
        service_standard_id=None if standard is None else standard.standard_id,
        service_standard_frozen=bool(standard is not None and standard.frozen),
        service_qualification_note=note,
        purchase_cost_sgd=float(unwrapped.total_cost),
        bess_degradation_cost_sgd=float(unwrapped.total_bess_degradation_cost),
        mixed_objective_cost=float(unwrapped.total_objective_cost),
        grid_energy_kwh=float(unwrapped.total_grid_energy_kWh),
        carbon_kg_co2e=float(unwrapped.total_carbon_emission),
        carbon_per_completed_work=carbon_per_work,
        completed_work=completed_work,
        pv=pv,
        wind=wind,
        renewable_share=renewable_share,
        grid_peak_kw=float(unwrapped.episode_grid_peak_power_kW),
        physical=physical,
        correction=correction,
        not_computable=tuple(sorted(set(not_computable))),
        failure_classification=failure,
    )


#: 协议 §3 的**物理约束**门禁：任一违规即不得进入「同等服务成本/碳比较」。
VIOLATION_LABELS: dict[str, str] = {
    "access_limit": "接入上限违规（grid 购电超过 access_limit）",
    "soc": "SOC 越界",
    "charge_discharge_exclusion": "充放电互斥违规（同步充放）",
    "energy_conservation": "能量守恒违规",
}


def violations_in(physical: PhysicalViolationMetrics) -> tuple[str, ...]:
    """列出该 episode 命中的**物理约束违规**（协议 §3；无违规则为空元组）。

    `base_load_unserved_steps` 是**服务缺口**而非约束违规，故不计入本门禁，
    仅随 `PhysicalViolationMetrics` 报告。
    """
    found: list[str] = []
    if physical.access_limit_violation_steps > 0:
        found.append("access_limit")
    if physical.soc_violation_steps > 0:
        found.append("soc")
    if physical.charge_discharge_exclusion_violations > 0:
        found.append("charge_discharge_exclusion")
    if physical.energy_conservation_violations > 0:
        found.append("energy_conservation")
    return tuple(found)


def qualify_service(
    service: ServiceMetrics,
    standard: ServiceStandard | None,
    physical: PhysicalViolationMetrics,
) -> tuple[bool | None, str]:
    """服务资格 + **物理约束门禁**的唯一入口。

    物理违规（接入 / SOC / 充放互斥 / 守恒任一）⇒ **`False`**（可判定为不合格），
    即使服务指标满足标准也**不得**给出 `True` —— 该 run 不能进入
    「同等服务成本/碳比较」。无违规时才由 `service_standard.qualify` 判定。
    """
    violations = violations_in(physical)
    if violations:
        labels = "、".join(VIOLATION_LABELS[name] for name in violations)
        return False, (
            f"不合格：存在物理约束违规（{labels}）——不得进入同等服务成本/碳比较"
        )
    return qualify(service, standard)


def planned_method_rows(records: Sequence[EvaluationRecord]) -> list[dict[str, Any]]:
    """按协议 §1 的**五类预定方法**输出报告行；缺方法 ⇒ `not_evaluated`。

    缺方法时**不得**用中性动作或旧模型冒充，也**不得**省略该行。
    """
    by_method: dict[str, EvaluationRecord] = {}
    for record in records:
        by_method.setdefault(str(record.method), record)

    rows: list[dict[str, Any]] = []
    for method in PLANNED_METHODS:
        found = by_method.get(method)
        if found is None:
            rows.append({
                "method": method,
                "status": STATUS_NOT_EVALUATED,
                "run_id": None,
                "action_mode": None,
                "seed": None,
                "service_qualified": None,
                "purchase_cost_sgd": None,
                "bess_degradation_cost_sgd": None,
                "carbon_kg_co2e": None,
                "grid_energy_kwh": None,
                "failure_classification": None,
                "note": NOT_EVALUATED_NOTE,
            })
            continue
        rows.append({
            "method": method,
            "status": STATUS_EVALUATED,
            "run_id": found.run_id,
            "action_mode": found.action_mode,
            "seed": found.seed,
            "service_qualified": found.service_qualified,
            "purchase_cost_sgd": found.purchase_cost_sgd,
            "bess_degradation_cost_sgd": found.bess_degradation_cost_sgd,
            "carbon_kg_co2e": found.carbon_kg_co2e,
            "grid_energy_kwh": found.grid_energy_kwh,
            "failure_classification": found.failure_classification,
            "note": found.service_qualification_note,
        })
    unknown = sorted(set(by_method) - set(PLANNED_METHODS))
    if unknown:
        raise ValueError(f"未知方法（不在预定五类之内）：{unknown}")
    return rows


__all__ = [
    "ACTION_MODES",
    "NOT_EVALUATED_NOTE",
    "PLANNED_METHODS",
    "STATUS_EVALUATED",
    "STATUS_NOT_EVALUATED",
    "VIOLATION_LABELS",
    "evaluate",
    "neutral_rule",
    "qualify_service",
    "violations_in",
    "planned_method_rows",
]
