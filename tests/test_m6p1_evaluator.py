"""M6-P1：评估器契约（服务资格 / 指标口径 / 未判定与不可计算）。

**先红**：现有 `evaluation/adapter.py` 的 `evaluate(..., service_qualified: bool = True)`
**默认达标**，`renewable_utilization` 的分母是 IDC 用能（实为**占比**），
且 `EvaluationRecord` 没有成本/碳/购电量分列、风光可用/使用/弃电、物理违规、
raw→exec 修正、求解耗时、超时与回退等字段。

对应 `docs/M6_EVALUATION_PROTOCOL.md` §3 的指标表。
"""

import importlib
import inspect

import numpy as np
import pytest

from contracts.models import EvaluationRecord
from envs.idc_price_env import IDCPriceEnv20D

HORIZON = 8

# **明确标记的临时标准**：只用于测试评估器的计算路径，**不是**项目冻结阈值。
TEMPORARY_TEST_STANDARD = dict(
    standard_id="temporary-test-standard-not-a-project-freeze",
    frozen=True,
    on_time_task_rate_min=0.95,
    on_time_work_rate_min=0.95,
    end_leftover_work_fraction_max=0.01,
    non_interruptible_interruption_max=0,
)


def _standard(**overrides):
    mod = importlib.import_module("evaluation.service_standard")
    kwargs = dict(TEMPORARY_TEST_STANDARD)
    kwargs.update(overrides)
    return mod.ServiceStandard(**kwargs)


def _adapter():
    return importlib.import_module("evaluation.adapter")


def _env(**kwargs):
    env = IDCPriceEnv20D(horizon=HORIZON, task_seed=0, server_seed=0,
                         forecast_seed=300000, **kwargs)
    return env


def _neutral(obs):
    action = np.full(21, 0.5, dtype=np.float32)
    action[20] = 0.0
    return action


def _run(env, *, method="rule", action_fn=_neutral, standard=None, **kwargs):
    return _adapter().evaluate(
        env, method, action_fn, run_id="m6p1-test", service_standard=standard, **kwargs)


# =============================================================================
# 1. 服务资格：不再默认 true
# =============================================================================

def test_service_standard_has_no_default_value():
    """`service_standard` 必须**显式传入**；不得存在默认达标的路径。"""
    params = inspect.signature(_adapter().evaluate).parameters
    assert "service_standard" in params
    assert params["service_standard"].default is inspect.Parameter.empty, \
        "service_standard 不得有默认值（默认达标=违反 M6-P1 红线）"
    assert "service_qualified" not in params, \
        "调用方不得再直接传入 service_qualified（必须由标准计算）"


def test_omitting_the_standard_is_a_type_error():
    with pytest.raises(TypeError):
        _adapter().evaluate(_env(), "rule", _neutral, run_id="r0")


def test_undetermined_when_no_standard_is_declared():
    rec = _run(_env(), standard=None)
    assert rec.service_qualified is None, "未显式传入标准 ⇒ 未判定，绝不默认达标"
    assert rec.service_standard_id is None
    assert rec.service_standard_frozen is False
    assert "未判定" in rec.service_qualification_note


def test_undetermined_when_the_standard_is_not_frozen():
    rec = _run(_env(), standard=_standard(frozen=False))
    assert rec.service_qualified is None
    assert rec.service_standard_id == TEMPORARY_TEST_STANDARD["standard_id"]
    assert rec.service_standard_frozen is False
    assert "未判定" in rec.service_qualification_note


def test_undetermined_is_not_counted_as_qualified():
    """未判定**不得**被当作达标（消费方必须能区分 None 与 True）。"""
    rec = _run(_env(), standard=None)
    assert rec.service_qualified is not True


def test_qualification_is_computed_from_an_explicit_standard():
    rec = _run(_env(), standard=_standard())
    assert isinstance(rec.service_qualified, bool)
    assert rec.service_standard_id == TEMPORARY_TEST_STANDARD["standard_id"]
    assert rec.service_standard_frozen is True


def test_impossible_standard_fails_and_a_trivial_one_passes():
    strict = _run(_env(), standard=_standard(on_time_task_rate_min=1.0,
                                             on_time_work_rate_min=1.0,
                                             end_leftover_work_fraction_max=0.0))
    trivial = _run(_env(), standard=_standard(on_time_task_rate_min=0.0,
                                              on_time_work_rate_min=0.0,
                                              end_leftover_work_fraction_max=1.0,
                                              non_interruptible_interruption_max=10 ** 6))
    assert strict.service_qualified is False
    assert trivial.service_qualified is True


def test_undetermined_when_a_mandatory_component_has_a_zero_denominator():
    """没有任何任务到期（截止时点全在 episode 外）⇒ 按时率零分母 ⇒ 未判定。"""
    env = _env(num_tasks=0)          # 只剩初始积压任务，其截止时点 24 > horizon=8
    env.reset(seed=0)
    assert all(t.latest_finish_time >= HORIZON for t in env.tasks)
    rec = _run(env, standard=_standard())
    assert rec.service.due_in_episode_tasks == 0
    assert rec.service.on_time_task_rate is None
    assert rec.service_qualified is None, "零分母 ⇒ 未判定，不自动合格"


def test_the_project_standard_is_frozen_and_matches_the_protocol_record():
    """M6-P1-F5 冻结的唯一项目标准：ID 与四个阈值须等于协议 §2.1 的冻结记录。"""
    mod = importlib.import_module("evaluation.service_standard")
    standard = mod.FROZEN_PROJECT_SERVICE_STANDARD
    assert isinstance(standard, mod.ServiceStandard), \
        "冻结标准必须是实际的 ServiceStandard 实例，不是一个哨兵值"
    assert standard.standard_id == "m6-service-standard-v1"
    assert standard.frozen is True
    assert standard.on_time_task_rate_min == 0.95
    assert standard.on_time_work_rate_min == 0.95
    assert standard.end_leftover_work_fraction_max == 0.01
    assert standard.non_interruptible_interruption_max == 0


def test_frozen_standard_takes_effect_only_when_explicitly_passed():
    """冻结**不**等于隐式达标：显式传入才判定，不传仍是未判定。"""
    mod = importlib.import_module("evaluation.service_standard")

    implicit = _run(_env(), standard=None)
    assert implicit.service_qualified is None
    assert implicit.service_standard_id is None

    explicit = _run(_env(), standard=mod.FROZEN_PROJECT_SERVICE_STANDARD)
    assert isinstance(explicit.service_qualified, bool)
    assert explicit.service_standard_id == "m6-service-standard-v1"
    assert explicit.service_standard_frozen is True


# =============================================================================
# 2. 成本 / 碳 / 购电量分列
# =============================================================================

def test_cost_components_are_split_and_the_mixed_objective_is_not_sgd():
    env = _env()
    rec = _run(env, standard=None)

    assert rec.purchase_cost_sgd == pytest.approx(float(env.total_cost))
    assert rec.bess_degradation_cost_sgd == pytest.approx(
        float(env.total_bess_degradation_cost))
    assert rec.mixed_objective_cost == pytest.approx(float(env.total_objective_cost))
    assert rec.grid_energy_kwh == pytest.approx(float(env.total_grid_energy_kWh))
    assert rec.carbon_kg_co2e == pytest.approx(float(env.total_carbon_emission))

    units = EvaluationRecord.UNITS
    assert units["purchase_cost_sgd"] == "SGD"
    assert units["bess_degradation_cost_sgd"] == "SGD"
    assert units["carbon_kg_co2e"] == "kgCO2e"
    assert units["grid_energy_kwh"] == "kWh"
    assert "SGD" not in units["mixed_objective_cost"], \
        "混合目标（含期末罚项）不得标作 SGD"
    assert "not SGD" in units["mixed_objective_cost"] or \
        "mixed" in units["mixed_objective_cost"].lower()


def test_degradation_cost_is_not_folded_into_purchase_cost():
    env = _env()
    env.reset(seed=0)
    _run(env, standard=None)
    assert env.total_bess_degradation_cost >= 0.0
    assert env.total_objective_cost == pytest.approx(
        env.total_cost + env.total_bess_degradation_cost + env.terminal_settlement_penalty)


def test_carbon_per_completed_work_is_undetermined_without_completed_work():
    zero = _run(_env(), action_fn=lambda obs: np.zeros(21, dtype=np.float32), standard=None)
    assert zero.completed_work == 0.0, "零计算动作不得完成任何工作"
    assert zero.carbon_per_completed_work is None, "零完成量 ⇒ 不可判定，不填 0"
    assert "carbon_per_completed_work" in zero.not_computable

    # 非空洞性：正常的固定动作 episode 必须有已完成工作量与单位碳排
    normal = _run(_env(), standard=None)
    assert normal.completed_work > 0.0
    assert normal.carbon_per_completed_work is not None


# =============================================================================
# 3. 风光：分列可用 / 使用 / 弃电 + 利用率；占比另名
# =============================================================================

def test_renewables_are_reported_per_source_with_utilization():
    env = _env()
    rec = _run(env, standard=None)

    assert rec.pv.available_kwh == pytest.approx(float(env.total_pv_available_kWh))
    assert rec.pv.used_kwh == pytest.approx(float(env.total_pv_used_kWh))
    assert rec.pv.curtail_kwh == pytest.approx(float(env.total_pv_curtail_kWh))
    assert rec.wind.available_kwh == pytest.approx(float(env.total_wind_available_kWh))
    assert rec.wind.used_kwh == pytest.approx(float(env.total_wind_used_kWh))
    assert rec.wind.curtail_kwh == pytest.approx(float(env.total_wind_curtail_kWh))

    units = EvaluationRecord.UNITS
    for field in ("pv", "wind"):
        assert units[f"{field}.available_kwh"] == "kWh"
        assert units[f"{field}.used_kwh"] == "kWh"
        assert units[f"{field}.curtail_kwh"] == "kWh"
        assert "fraction" in units[f"{field}.utilization"]


def test_pv_utilization_is_used_over_available_not_a_share_of_demand():
    metrics = importlib.import_module("evaluation.metrics")
    pv, missing = metrics.renewable_metrics(
        key="pv", available_kwh=10.0, used_kwh=4.0, curtail_kwh=6.0)
    assert pv.utilization == pytest.approx(0.4), "利用率 = used / available"
    assert missing == ()

    # 端到端：与 env 的累计量一致，且与「可再生占比」是两个不同的量
    env = _env(pv_t=np.full(HORIZON, 1.0))
    rec = _run(env, standard=None)
    assert rec.pv.available_kwh > 0.0
    assert rec.pv.utilization == pytest.approx(
        env.total_pv_used_kWh / env.total_pv_available_kWh)
    if rec.renewable_share is not None:
        assert rec.renewable_share == pytest.approx(
            (env.total_pv_used_kWh + env.total_wind_used_kWh) / env.total_idc_energy_kWh)


def test_utilization_is_undetermined_when_nothing_is_available():
    # 无光照 ⇒ 可用量为 0 ⇒ 不可判定（不填 0）
    rec = _run(_env(pv_t=np.zeros(HORIZON)), standard=None)
    assert rec.pv.available_kwh == 0.0
    assert rec.pv.utilization is None, "可用量为 0 ⇒ 不可判定，不填 0"
    assert "pv.utilization" in rec.not_computable


def test_renewable_share_is_named_separately_from_utilization():
    env = _env()
    rec = _run(env, standard=None)
    assert hasattr(rec, "renewable_share")
    assert not hasattr(rec, "renewable_utilization"), \
        "占比不得再命名为 utilization"
    assert "fraction" in EvaluationRecord.UNITS["renewable_share"]
    if env.total_idc_energy_kWh > 0.0:
        assert rec.renewable_share == pytest.approx(
            (env.total_pv_used_kWh + env.total_wind_used_kWh) / env.total_idc_energy_kWh)


# =============================================================================
# 4. 业务服务计数：按时 / 逾期 / 期末剩余 / 不可中断
# =============================================================================

def test_service_counts_cover_the_episode():
    env = _env()
    rec = _run(env, standard=None)
    s = rec.service

    for name in ("due_in_episode_tasks", "due_in_episode_work",
                 "on_time_completed_tasks", "on_time_completed_work",
                 "overdue_completed_tasks", "overdue_completed_work",
                 "failed_tasks", "failed_work",
                 "overdue_backlog_tasks", "overdue_backlog_work",
                 "not_due_backlog_tasks", "not_due_backlog_work",
                 "end_leftover_work"):
        assert hasattr(s, name), f"缺少业务服务字段 {name}"

    assert s.non_interruptible_interruption_count == \
        int(env.total_non_interruptible_interruption_count)
    # 分类互斥完备：五个分类的**任务数**之和 == 已到达任务数
    arrived = sum(1 for task in env.tasks if task.status != "not_arrived")
    assert (s.on_time_completed_tasks + s.overdue_completed_tasks + s.failed_tasks
            + s.overdue_backlog_tasks + s.not_due_backlog_tasks) == arrived
    # 到期分母只覆盖截止时点在 episode 内的任务
    assert s.due_in_episode_tasks == sum(
        1 for task in env.tasks
        if task.status != "not_arrived" and task.latest_finish_time < HORIZON)


class _StubTask:
    """`classify_tasks` 只需要这几个字段（纯函数级验证，不跑 episode）。"""

    def __init__(self, *, status, workload, remaining, deadline, finish=None):
        self.status = status
        self.workload = workload
        self.remaining_work = remaining
        self.deadline = deadline
        self.arrival_time = 0
        self.finish_time = finish

    @property
    def latest_finish_time(self) -> int:
        return int(self.arrival_time + self.deadline)


def test_failed_tasks_stay_in_the_denominator():
    """已到达但失败的任务单列且**留在分母**里（`classify_tasks` 纯函数级）。"""
    metrics = importlib.import_module("evaluation.metrics")
    tasks = [
        _StubTask(status="finished", workload=10.0, remaining=0.0, deadline=3, finish=2),
        _StubTask(status="failed", workload=20.0, remaining=20.0, deadline=5),
        _StubTask(status="waiting", workload=30.0, remaining=30.0, deadline=6),
        _StubTask(status="waiting", workload=40.0, remaining=40.0, deadline=99),
        _StubTask(status="not_arrived", workload=50.0, remaining=50.0, deadline=2),
    ]
    service, _ = metrics.classify_tasks(
        tasks, horizon=8, non_interruptible_interruptions=2)

    assert service.failed_tasks == 1 and service.failed_work == pytest.approx(20.0)
    assert service.on_time_completed_tasks == 1
    assert service.overdue_backlog_tasks == 1        # deadline=6 已过、未完成
    assert service.not_due_backlog_tasks == 1        # deadline=99 在 episode 外
    # 互斥完备：五类计数之和 == 已到达任务数（失败任务**没有**消失）
    assert (service.on_time_completed_tasks + service.overdue_completed_tasks
            + service.failed_tasks + service.overdue_backlog_tasks
            + service.not_due_backlog_tasks) == 4
    # 分母覆盖已到期任务：按时 + 逾期积压 + 失败
    assert service.due_in_episode_tasks == 3
    assert service.on_time_task_rate == pytest.approx(1 / 3)
    assert service.end_leftover_work == pytest.approx(90.0)
    assert service.non_interruptible_interruption_count == 2


def test_service_rates_are_fractions_or_undetermined():
    rec = _run(_env(), standard=None)
    for name in ("on_time_task_rate", "on_time_work_rate", "end_leftover_work_fraction"):
        value = getattr(rec.service, name)
        assert value is None or 0.0 <= value <= 1.0 + 1e-9, name


# =============================================================================
# 5. 修正器：无修正器 ⇒ 不适用（不是 0 次修正）
# =============================================================================

def test_correction_metrics_are_not_applicable_without_a_corrector():
    rec = _run(_env(), standard=None)
    assert rec.correction is None, "无修正器时必须标不适用，不得填 0 次修正冒充测量"


def test_correction_metrics_capture_raw_to_exec_and_solve_time():
    from safe_rl.corrector_wrapper import CorrectorWrapper

    env = _env()
    wrapped = CorrectorWrapper(env, corrector_time_limit_s=0.25)
    rec = _run(wrapped, standard=None)
    c = rec.correction
    assert c is not None
    assert c.compute_abs_delta_mean >= 0.0
    assert c.storage_abs_delta_mean >= 0.0
    assert c.solve_time_median_s >= 0.0
    assert c.solve_time_p95_s >= c.solve_time_median_s
    assert c.timeout_count >= 0 and c.zero_action_fallback_count >= 0
    assert "s" in EvaluationRecord.UNITS["correction.solve_time_median_s"]


# =============================================================================
# 6. 物理违规
# =============================================================================

def test_physical_violations_are_reported_with_the_physics_tolerance():
    env = _env()
    rec = _run(env, standard=None)
    p = rec.physical
    assert p.energy_conservation_violations == 0, \
        "能量守恒按 tests/test_m33_group_power.py 的同一容差复核"
    assert p.charge_discharge_exclusion_violations == 0
    assert p.access_limit_violation_steps == 0
    assert p.soc_violation_steps == 0
    assert p.energy_conservation_max_gap_kw == pytest.approx(0.0, abs=1e-6)
    assert "kW" in EvaluationRecord.UNITS["physical.energy_conservation_max_gap_kw"]


def test_energy_conservation_check_actually_detects_a_violation():
    """非空洞性：注入一步非法功率必须被计为违规。"""
    env = _env()
    rec = _run(env, standard=None)
    metrics = importlib.import_module("evaluation.metrics")
    bad = dict(
        grid_power_kw=0.0, pv_available_kw=0.0, wind_available_kw=0.0,
        discharge_kw=0.0, idc_kw=1.0, charge_kw=0.0,
        pv_curtail_kw=0.0, wind_curtail_kw=0.0,
    )
    found = metrics.check_physical_step(bad, access_limit_kw=1e9,
                                        soc_kwh=0.0, soc_min_kwh=0.0, soc_max_kwh=1.0)
    assert found["energy_conservation_gap_kw"] == pytest.approx(1.0)
    assert rec.physical.energy_conservation_violations == 0


# =============================================================================
# 7. 失败 run 与「未评估」的方法矩阵
# =============================================================================

def test_failed_run_is_retained_with_a_classification():
    def _boom(obs):
        raise RuntimeError("action_fn exploded")

    rec = _run(_env(), action_fn=_boom, standard=None)
    assert rec.failure_classification == "RuntimeError", "失败 run 必须保留并标注原因"
    assert rec.service_qualified is None, "失败 run 绝不达标"
    assert rec.steps >= 0


def test_missing_methods_are_reported_as_not_evaluated():
    """五类方法矩阵：缺方法 ⇒ 「未评估」，不得伪造比较结果。"""
    adapter = _adapter()
    rec = _run(_env(), method="rule_baseline", standard=None)
    rows = adapter.planned_method_rows([rec])
    assert [row["method"] for row in rows] == list(adapter.PLANNED_METHODS)
    evaluated = [row for row in rows if row["status"] == "evaluated"]
    missing = [row for row in rows if row["status"] == "not_evaluated"]
    assert [row["method"] for row in evaluated] == ["rule_baseline"]
    assert len(missing) == len(adapter.PLANNED_METHODS) - 1
    for row in missing:
        assert row["service_qualified"] is None
        assert row["purchase_cost_sgd"] is None


def test_records_share_one_schema_and_declare_mode_and_seed():
    rec1 = _run(_env(), method="rule_baseline", standard=None,
                action_mode="deterministic_mean", seed=3)
    rec2 = _run(_env(), method="penalty_ppo", standard=None,
                action_mode="seeded_sample", seed=4)
    assert rec1.model_dump().keys() == rec2.model_dump().keys()
    assert rec1.action_mode == "deterministic_mean" and rec1.seed == 3
    assert rec2.action_mode == "seeded_sample" and rec2.seed == 4
    assert isinstance(rec1, EvaluationRecord)
    assert rec1.schema_version == "contract-v9"


def test_record_rejects_unknown_fields_and_requires_units():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EvaluationRecord(
            method="rule", run_id="r", action_mode="deterministic_mean", seed=0, steps=1,
            checkpoint_id=None, checkpoint_role=None, service=None,
            service_qualified=None, service_standard_id=None, service_standard_frozen=False,
            service_qualification_note="", purchase_cost_sgd=0.0,
            bess_degradation_cost_sgd=0.0, mixed_objective_cost=0.0, grid_energy_kwh=0.0,
            carbon_kg_co2e=0.0, carbon_per_completed_work=None, completed_work=0.0,
            pv=None, wind=None, renewable_share=None, grid_peak_kw=0.0, physical=None,
            correction=None, not_computable=(), surprise=1,
        )
    for name in ("purchase_cost_sgd", "carbon_kg_co2e", "grid_energy_kwh",
                 "mixed_objective_cost", "renewable_share", "grid_peak_kw"):
        assert name in EvaluationRecord.UNITS, f"{name} 未声明单位"


# =============================================================================
# 8. M6-P1-R1：资格口径（分子与到期分母同口径）与物理违规门禁
# =============================================================================

class _Task:
    """`classify_tasks` 需要的最小任务替身（纯函数级构造反例）。"""

    def __init__(self, *, deadline, status="finished", workload=1.0,
                 remaining=0.0, finish=None):
        self.deadline = deadline
        self.arrival_time = 0
        self.status = status
        self.workload = workload
        self.remaining_work = remaining
        self.finish_time = (
            finish if finish is not None
            else (deadline - 1 if status == "finished" else None)
        )

    @property
    def latest_finish_time(self) -> int:
        return int(self.arrival_time + self.deadline)


def test_on_time_rate_uses_a_same_caliber_numerator():
    """**反例**：1 个本 episode 到期任务 + 1 个提前完成但下个 episode 才到期的任务。

    分子（按时完成）**必须**只数「截止时点落在本 episode 内」的任务，
    与分母 `due_in_episode_*` 同口径 —— 否则比率会 **> 1**。
    """
    metrics = importlib.import_module("evaluation.metrics")
    tasks = [
        # 截止时点在 episode 内（deadline=3 < horizon=8），按时完成
        _Task(deadline=3, workload=10.0, finish=2),
        # **提前完成，但截止时点在下一个 episode**（deadline=12 ≥ horizon=8）
        _Task(deadline=12, workload=20.0, finish=1),
    ]
    service, _ = metrics.classify_tasks(
        tasks, horizon=8, non_interruptible_interruptions=0)

    assert service.due_in_episode_tasks == 1
    assert service.on_time_task_rate is not None and 0.0 <= service.on_time_task_rate <= 1.0, \
        f"按时任务率必须在 [0,1]，实际 {service.on_time_task_rate}"
    assert service.on_time_work_rate is not None and 0.0 <= service.on_time_work_rate <= 1.0, \
        f"按时工作量率必须在 [0,1]，实际 {service.on_time_work_rate}"
    # 分子与到期分母同口径：率 == 到期且按时的量 / 到期的量
    assert service.on_time_task_rate == pytest.approx(1.0)
    assert service.on_time_work_rate == pytest.approx(1.0)
    # 分类计数保留**全部**已到达任务（互斥完备）
    assert (service.on_time_completed_tasks + service.overdue_completed_tasks
            + service.failed_tasks + service.overdue_backlog_tasks
            + service.not_due_backlog_tasks) == 2
    assert service.on_time_completed_tasks == 2, "两个任务都按时完成（分类计数保留）"


def test_on_time_rate_is_bounded_for_episode_less_work():
    """到期分母为 0 ⇒ 不可判定（`None`），绝不返回 > 1 或自动合格。"""
    metrics = importlib.import_module("evaluation.metrics")
    tasks = [_Task(deadline=99, workload=5.0, finish=1)]
    service, missing = metrics.classify_tasks(
        tasks, horizon=8, non_interruptible_interruptions=0)
    assert service.due_in_episode_tasks == 0
    assert service.on_time_task_rate is None
    assert "service.on_time_task_rate" in missing


def _violating_step(**overrides):
    step = {
        "grid_power_kw": 1.0, "pv_available_kw": 0.0, "wind_available_kw": 0.0,
        "discharge_kw": 0.0, "idc_kw": 1.0, "charge_kw": 0.0,
        "pv_curtail_kw": 0.0, "wind_curtail_kw": 0.0,
    }
    step.update(overrides)
    return step


def test_no_physical_violation_allows_qualification():
    """对照：无物理违规时可产生 `True`。"""
    metrics = importlib.import_module("evaluation.metrics")
    rows = [metrics.check_physical_step(
        _violating_step(), access_limit_kw=1e9, soc_kwh=0.0,
        soc_min_kwh=0.0, soc_max_kwh=1.0)]
    physical = metrics.aggregate_physical(rows, base_load_unserved_steps=0)
    assert physical.energy_conservation_violations == 0
    assert physical.access_limit_violation_steps == 0
    assert _adapter().violations_in(physical) == ()


def test_each_single_physical_violation_blocks_comparison_eligibility():
    """**单项违规**各自必须阻止「同等服务成本/碳比较」的 `True`。"""
    metrics = importlib.import_module("evaluation.metrics")
    cases = {
        "access": dict(step=_violating_step(grid_power_kw=5.0), access_limit_kw=1.0),
        "soc": dict(step=_violating_step(), access_limit_kw=1e9, soc_kwh=2.0),
        "exclusion": dict(step=_violating_step(charge_kw=1.0), access_limit_kw=1e9),
        "conservation": dict(
            step=_violating_step(idc_kw=1.0, grid_power_kw=0.0), access_limit_kw=1e9),
    }
    for name, kwargs in cases.items():
        rows = [metrics.check_physical_step(
            kwargs.pop("step"), access_limit_kw=kwargs.pop("access_limit_kw"),
            soc_kwh=kwargs.pop("soc_kwh", 0.0), soc_min_kwh=0.0, soc_max_kwh=1.0)]
        physical = metrics.aggregate_physical(rows, base_load_unserved_steps=0)
        found = _adapter().violations_in(physical)
        assert found, f"{name} 违规未被识别"

        # 违规存在时，即使服务指标满足标准，也**不得**给出 True
        standard = _standard(on_time_task_rate_min=0.0, on_time_work_rate_min=0.0,
                             end_leftover_work_fraction_max=1.0,
                             non_interruptible_interruption_max=10 ** 6)
        qualified, note = _adapter().qualify_service(
            _passing_service(), standard, physical)
        assert qualified is not True, f"{name} 违规不得产生 True"
        assert qualified is False
        assert any(v in note for v in found) or "违规" in note


def _passing_service():
    from contracts.models import ServiceMetrics

    return ServiceMetrics(
        due_in_episode_tasks=1, due_in_episode_work=1.0,
        on_time_completed_tasks=1, on_time_completed_work=1.0,
        overdue_completed_tasks=0, overdue_completed_work=0.0,
        overdue_backlog_tasks=0, overdue_backlog_work=0.0,
        not_due_backlog_tasks=0, not_due_backlog_work=0.0,
        failed_tasks=0, failed_work=0.0,
        end_leftover_work=0.0, end_leftover_work_fraction=0.0,
        non_interruptible_interruption_count=0,
        on_time_task_rate=1.0, on_time_work_rate=1.0,
    )


def test_evaluate_marks_violating_episode_as_not_qualified():
    """端到端：物理违规的 episode 即使服务达标也不得 `service_qualified is True`。"""
    env = _env()
    rec = _run(env, standard=_standard(on_time_task_rate_min=0.0,
                                       on_time_work_rate_min=0.0,
                                       end_leftover_work_fraction_max=1.0,
                                       non_interruptible_interruption_max=10 ** 6))
    if _adapter().violations_in(rec.physical):
        assert rec.service_qualified is not True
    else:
        assert rec.service_qualified is True, "对照组：无违规且服务达标应为 True"
