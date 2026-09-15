"""M4.3 窗口工具测试（旧单步 MILP 用例已于 M4.4b 随该路径退役）。"""


from contracts.models import (
    ArtifactDigest,
    ForecastSeriesProvenance,
    PlanningExogenousForecast,
    ScenarioBundle,
    ScenarioForecastProvenance,
    SystemSnapshot,
    TaskState,
)
from planning.window import compute_window

# M1.3e：contract-v8 要求结构化 provenance（无 schema 的 `source_hashes` 已退役）。
_FIXTURE_GENERATED_AT = "2026-01-01T00:00:00+08:00"


def _fixture_series(field: str) -> ForecastSeriesProvenance:
    """单个序列的夹具 provenance（**非正式**，mode=synthetic）。"""
    return ForecastSeriesProvenance(
        series_name=field,
        source_kind="synthetic",
        method="test_fixture",
        generated_at=_FIXTURE_GENERATED_AT,
        information_cutoff_exclusive=_FIXTURE_GENERATED_AT,
        target_start=_FIXTURE_GENERATED_AT,
        target_end_exclusive="2026-01-01T04:00:00+08:00",
        lookback_start=None,
        lookback_end_exclusive=None,
        model_name="test_fixture",
        model_version="v1",
        code_revision="a" * 40,
        seed=None,
        sources=[
            ArtifactDigest(role="fixture", logical_path="tests/fixtures/none",
                           sha256="b" * 64)
        ],
    )


def _provenance() -> ScenarioForecastProvenance:
    """七个字段一一对应的测试夹具 provenance（**非正式**，mode=synthetic）。"""
    return ScenarioForecastProvenance(
        price_forecast=_fixture_series("price_forecast"),
        load_forecast=_fixture_series("load_forecast"),
        pv_forecast=_fixture_series("pv_forecast"),
        wind_forecast=_fixture_series("wind_forecast"),
        temperature_forecast=_fixture_series("temperature_forecast"),
        carbon_forecast=_fixture_series("carbon_forecast"),
        arrival_forecast=_fixture_series("arrival_forecast"),
    )


def _forecast() -> ScenarioBundle:
    return ScenarioBundle(
        split="train",
        start="0",
        horizon=24,
        forecast_cutoff=4,
        price_forecast=[0.2] * 24,
        load_forecast=[0.0] * 24,
        pv_forecast=[0.0] * 24,
        wind_forecast=[0.0] * 24,
        temperature_forecast=[28.0] * 24,
        arrival_forecast=[0.0] * 24,
        carbon_forecast=[0.0] * 24,
        mode="synthetic",
        generated_at=_FIXTURE_GENERATED_AT,
        forecast_provenance=_provenance(),
    )


def _snapshot() -> SystemSnapshot:
    tasks = [
        TaskState(
            task_id="t0",
            remaining_work=5.0,
            deadline=8,
            priority=1.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
        TaskState(
            task_id="t1",
            remaining_work=3.0,
            deadline=6,
            priority=2.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
    ]
    return SystemSnapshot(
        step=0,
        soc_kwh=50.0,
        soc_min_kwh=10.0,
        soc_max_kwh=90.0,
        group_work_capacity=[4.0, 6.0],
        delta_t_hours=1.0,
        planning_horizon_steps=24,
        soc_capacity_kwh=100.0,
        bess_charge_power_max_kw=20.0,
        bess_discharge_power_max_kw=20.0,
        bess_charge_efficiency=0.95,
        bess_discharge_efficiency=0.95,
        bess_degradation_cost_per_kwh=0.02,
        base_idc_power_forecast_kw=[14.0] * 24,
            planning_forecast=PlanningExogenousForecast(
            horizon_steps=24,
            price=[0.2] * 24, pv=[0.0] * 24, wind=[0.0] * 24,
            temperature=[28.0] * 24, carbon=[0.0] * 24, arrival=[0.0] * 24,
            base_idc_power=[14.0] * 24,
            visible_mask=[False] * 24, assumed_mask=[True] * 24,
            extension_policy="test fixture policy",
        ),

        group_power_coeff_kw_per_work=[0.01] * 2,
        group_power_upper_kw=[0.7] * 2,
        power_approximation_note="planning approximation; verify with env physics chain",
        access_limit_kw=12.0,
        budget_remaining_sgd=1000.0,
        tasks=tasks,
        forecast=_forecast(),
    )


def test_window_extends_to_latest_deadline():
    tasks = [
        TaskState(
            task_id="t",
            remaining_work=1.0,
            deadline=30,
            priority=1.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
    ]
    assert compute_window(tasks, default_horizon=24) == 30


def test_window_defaults_to_24():
    tasks = [
        TaskState(
            task_id="t",
            remaining_work=1.0,
            deadline=5,
            priority=1.0,
            status="waiting",
            max_rate_work_per_step=1.0,
        ),
    ]
    assert compute_window(tasks, default_horizon=24) == 24
