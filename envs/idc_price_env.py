import numpy as np
import gymnasium as gym
from gymnasium import spaces

from idc_model.task_model import IDCEnergyTaskModel
from idc_model.allocation import allocate_tasks
from idc_model.task_forecast import (
    SYNTHETIC_FORECAST_SOURCE,
    generate_task_arrival_forecast,
    task_forecast_metrics,
    validate_task_forecast_config,
)


def visible_window_slice(t: int, forecast_cutoff: int, horizon: int) -> tuple[int, int]:
    """统一定义（M3.10a）：时刻 t 的可见预测区间为 [t, t + forecast_cutoff)。

    `forecast_cutoff` = 当前时刻起、含当前时刻在内的可见预测点数；
    首个不可见真值下标 = t + forecast_cutoff。区间按 horizon 裁剪。
    """
    start = max(int(t), 0)
    end = min(int(t) + int(forecast_cutoff), int(horizon))
    return start, end


def decompose_supply(base_demand_kW: float, total_demand_kW: float, budget_kW: float) -> dict:
    """独立供应分解（M3.7c）：基础负载优先供给，剩余预算给任务，两断供独立计算。"""
    task_incremental = max(total_demand_kW - base_demand_kW, 0.0)
    base_served = min(base_demand_kW, budget_kW)
    task_budget = max(budget_kW - base_served, 0.0)
    task_served = min(task_incremental, task_budget)
    return {
        "task_incremental_demand_kW": task_incremental,
        "base_served_kW": base_served,
        "task_budget_kW": task_budget,
        "task_served_kW": task_served,
        "unserved_base_load_kW": base_demand_kW - base_served,
        "unserved_task_power_kW": task_incremental - task_served,
        "idc_served_kW": base_served + task_served,
    }


class IDCPriceEnv20D(gym.Env):
    """
    面向 PPO 的智算中心分时电价任务调度环境：ultimate 前瞻状态版。

    本版相对旧环境的核心变化：
    1. 接入 Task 对象，不再把任务本体压缩成聚合队列 Q；
    2. reset() 时重新生成任务，避免上一轮 episode 的任务状态污染下一轮；
    3. Q_t 只作为由 Task.remaining_work 统计得到的积压量；
    4. step() 内部按小时激活到达任务，并按优先级/期限约束执行任务；
    5. PPO 动作由 N 个 server-group 计算强度 + 1 个有符号储能动作组成（21 维，默认 N=20）；
    6. observation 维度按 6 + 10 + 6*N + 8*horizon 计算；默认 N=20、horizon=24 时为 328；
       （预测特征顺序：price, temperature, arrival, pv, wind, carbon, sin, cos）
    7. 功耗按逐组完成工作/组能力导出实际负载（不再有 α 预留损耗）；
    8. reward 扩展为任务类综合奖励：完成量、完整任务完成、高优先级任务完成、成本、积压、紧急积压、等待、超时、未使用能力、高电价高负载、暂停/恢复和不可暂停中断；
    9. 新增任务启停跟踪：记录任务启动、暂停、恢复与不可暂停任务中断。

    当前仍未做：
    - 服务器级任务绑定；
    - 真正的任务并行拆分；
    - PPO 直接选择具体 task_id。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 24,
        base_load: float = 0.05,
        max_task_load_per_server: float = 0.80,
        Q0: float = 50.0,
        num_tasks: int = 12,
        price_ref: float = 1.50,
        lambda_ref: float = 1000.0,
        queue_ref: float = 1500.0,
        queue_capacity_ref: float = 1500.0,
        cost_ref: float = 30.0,
        carbon_ref: float = 15.0,
        carbon_price: float = 0.0,
        delta_t_hours: float = 1.0,
        peak_power_threshold_kW: float = 18.0,
        peak_power_ref_kW: float = 10.0,
        grid_power_limit_kW: float = 18.0,
        access_limit_kw: float = 18.0,
        sla_ref: float = 50.0,
        sla_penalty_ref: float | None = None,
        bess_capacity_kWh: float = 100.0,
        bess_soc_init: float = 0.50,
        bess_soc_min: float = 0.10,
        bess_soc_max: float = 0.90,
        bess_soc_target: float = 0.50,
        bess_soc_final_tolerance: float = 0.05,
        bess_charge_power_max_kW: float = 20.0,
        bess_discharge_power_max_kW: float = 20.0,
        bess_charge_efficiency: float = 0.95,
        bess_discharge_efficiency: float = 0.95,
        bess_degradation_cost_per_kWh: float = 0.02,
        bess_degradation_cost_ref: float | None = None,
        reward_done_weight: float = 3.0,
        reward_cost_weight: float = 0.5,
        reward_carbon_weight: float = 0.3,
        reward_sla_weight: float = 0.8,
        reward_queue_weight: float = 0.8,
        reward_queue_overflow_weight: float = 1.0,
        reward_final_queue_weight: float = 2.0,
        reward_deadline_weight: float = 0.8,
        reward_unused_capacity_weight: float = 0.10,
        reward_finished_task_weight: float = 1.0,
        reward_priority_finish_weight: float = 0.8,
        reward_urgent_backlog_weight: float = 0.8,
        reward_waiting_weight: float = 0.3,
        reward_peak_load_weight: float = 0.2,
        reward_pause_weight: float = 0.2,
        reward_resume_weight: float = 0.05,
        reward_non_interruptible_weight: float = 1.0,
        reward_load_smooth_weight: float = 0.05,
        reward_action_smooth_weight: float = 0.03,
        reward_bess_degradation_weight: float = 1.0,
        reward_bess_invalid_action_weight: float = 0.2,
        reward_soc_final_weight: float = 2.0,
        reward_grid_peak_weight: float = 1.0,
        price_t=None,
        carbon_factor_t=None,
        T_amb=None,
        pv_t=None,
        pv_ref_kw: float = 1.0,
        wind_ref_kw: float = 1.0,
        carbon_factor_ref: float = 1.0,
        allow_pv_export: bool = False,
        wt_t=None,
        server_seed=None,
        task_seed=None,
        forecast_seed=None,
        task_forecast_mode: str = "noisy",
        forecast_error_level: float = 0.20,
        forecast_cutoff: int = 4,
        task_forecast_seed_offset: int = 300000,
        enable_server_group_model: bool = False,
        server_group_size: int = 1,
        num_server_groups: int = 20,
        task_workload_scale: float = 1.0,
        facility_rated_power_mw: float | None = None,
        bess_scale_factor: float = 1.0,
        scale_bess_with_idc: bool = False,
        formal_injection=None,
    ):
        super().__init__()

        # 1. 底层物理模型：建议使用最新 servercapacity 版本
        self.model = IDCEnergyTaskModel(
            N=int(num_server_groups),
            server_seed=server_seed,
            task_seed=task_seed,
            enable_server_group_model=enable_server_group_model,
            server_group_size=server_group_size,
            num_server_groups=num_server_groups,
            task_workload_scale=task_workload_scale,
        )

        self.task_forecast_mode, self.forecast_error_level = (
            validate_task_forecast_config(task_forecast_mode, forecast_error_level)
        )
        self.task_forecast_seed_offset = int(task_forecast_seed_offset)
        self.forecast_cutoff = int(forecast_cutoff)
        if forecast_seed is None:
            forecast_seed = (
                None
                if task_seed is None
                else int(task_seed) + self.task_forecast_seed_offset
            )
        self.forecast_seed = None if forecast_seed is None else int(forecast_seed)
        self.forecast_rng = np.random.default_rng(self.forecast_seed)

        # 2. 仿真参数
        self.horizon = int(horizon)
        self.base_load = float(base_load)
        self.max_task_load_per_server = float(max_task_load_per_server)
        self.server_group_model_enabled = bool(enable_server_group_model)
        self.server_group_size = max(int(server_group_size), 1) if self.server_group_model_enabled else 1
        self.num_server_groups = int(num_server_groups)
        self.effective_total_server_count = self.num_server_groups * self.server_group_size
        self.task_workload_scale = max(float(task_workload_scale), 0.0)
        self.facility_rated_power_mw = (
            None
            if facility_rated_power_mw is None
            else float(facility_rated_power_mw)
        )
        if self.facility_rated_power_mw is not None and self.facility_rated_power_mw <= 0.0:
            raise ValueError("facility_rated_power_mw must be positive when provided.")
        self.idc_power_scale_factor = float(self.server_group_size if self.server_group_model_enabled else 1)
        self.bess_scale_factor = max(float(bess_scale_factor), 0.0)
        self.scale_bess_with_idc = bool(scale_bess_with_idc)
        self.initial_Q = float(Q0) * max(self.task_workload_scale, 1e-9)
        self.num_tasks = int(num_tasks)

        # 3. 归一化参考值
        self.price_ref = float(price_ref)
        self.lambda_ref = float(lambda_ref) * max(self.task_workload_scale, 1e-9)
        self.queue_ref = float(queue_ref) * max(self.task_workload_scale, 1e-9)
        # queue_capacity_ref is the soft backlog capacity; Q above this is penalized, not terminated.
        self.queue_capacity_ref = float(queue_capacity_ref) * max(self.task_workload_scale, 1e-9)
        self.cost_ref = float(cost_ref) * max(self.idc_power_scale_factor, 1e-9)
        self.carbon_ref = float(carbon_ref) * max(self.idc_power_scale_factor, 1e-9)
        # carbon_price converts grid-purchased carbon emissions into a reporting-only carbon_cost.
        self.carbon_price = float(carbon_price)
        # delta_t_hours is the step length used to convert kW power into kWh energy.
        self.delta_t_hours = float(delta_t_hours)

        # --- M1.3g-e-c：**formal** 注入（显式入口；legacy 行为完全不变） ---------
        # 只有在显式传入已验证 injection 时才切换；缺失即 legacy。
        self.formal_injection = formal_injection
        self.formal = formal_injection is not None
        if self.formal:
            from scenario.env_injection import DELTA_T_HOURS as _FORMAL_DELTA

            if self.delta_t_hours != _FORMAL_DELTA:
                raise ValueError(
                    f"formal 注入要求 delta_t_hours == {_FORMAL_DELTA}（半小时语义）；"
                    f"实际 {self.delta_t_hours}")
            if int(formal_injection.horizon) != self.horizon:
                raise ValueError(
                    "formal 注入的 horizon 必须与环境的 horizon 一致："
                    f"{formal_injection.horizon} != {self.horizon}")
            if int(formal_injection.forecast_cutoff) != self.forecast_cutoff:
                raise ValueError(
                    "formal 注入的 forecast_cutoff 必须与环境的 forecast_cutoff 一致："
                    f"{formal_injection.forecast_cutoff} != {self.forecast_cutoff}")
            # 每步计划处理能力的账面记录（rate → work/step）
            self.last_planned_capacity_rate_work_per_hour = 0.0
            self.last_planned_capacity_per_step = 0.0
        self.peak_power_threshold_kW = float(peak_power_threshold_kW) * max(self.idc_power_scale_factor, 1e-9)
        self.peak_power_ref_kW = float(peak_power_ref_kW) * max(self.idc_power_scale_factor, 1e-9)
        self.grid_power_limit_kW = float(grid_power_limit_kW) * max(self.idc_power_scale_factor, 1e-9)
        self.access_limit_kw = float(access_limit_kw) * max(self.idc_power_scale_factor, 1e-9)
        # SLA pressure is task-count/priority/lateness based, not workload based.
        # Keep the legacy attribute as an alias for downstream reporting.
        self.sla_penalty_ref = float(
            sla_ref if sla_penalty_ref is None else sla_penalty_ref
        )
        if self.sla_penalty_ref <= 0.0:
            raise ValueError("sla_penalty_ref must be positive.")
        self.sla_ref = self.sla_penalty_ref
        bess_effective_scale = self.bess_scale_factor if self.scale_bess_with_idc else 1.0
        self.bess_capacity_kWh = float(bess_capacity_kWh) * bess_effective_scale
        self.bess_soc_init = float(bess_soc_init)
        self.bess_soc_min = float(bess_soc_min)
        self.bess_soc_max = float(bess_soc_max)
        self.bess_soc_target = float(bess_soc_target)
        self.bess_soc_final_tolerance = float(bess_soc_final_tolerance)
        self.bess_charge_power_max_kW = float(bess_charge_power_max_kW) * bess_effective_scale
        self.bess_discharge_power_max_kW = float(bess_discharge_power_max_kW) * bess_effective_scale
        self.bess_charge_efficiency = float(bess_charge_efficiency)
        self.bess_discharge_efficiency = float(bess_discharge_efficiency)
        self.bess_degradation_cost_per_kWh = float(bess_degradation_cost_per_kWh)
        derived_degradation_ref = (
            max(self.bess_charge_power_max_kW, self.bess_discharge_power_max_kW)
            * self.delta_t_hours
            * self.bess_degradation_cost_per_kWh
        )
        self.bess_degradation_cost_ref = float(
            derived_degradation_ref
            if bess_degradation_cost_ref is None
            else bess_degradation_cost_ref
        )
        if self.bess_degradation_cost_ref <= 0.0:
            raise ValueError("bess_degradation_cost_ref must be positive.")

        # 4. reward 权重
        self.reward_done_weight = float(reward_done_weight)
        self.reward_cost_weight = float(reward_cost_weight)
        self.reward_carbon_weight = float(reward_carbon_weight)
        self.reward_sla_weight = float(reward_sla_weight)
        self.reward_queue_weight = float(reward_queue_weight)
        self.reward_queue_overflow_weight = float(reward_queue_overflow_weight)
        self.reward_final_queue_weight = float(reward_final_queue_weight)
        self.reward_deadline_weight = float(reward_deadline_weight)
        self.reward_unused_capacity_weight = float(reward_unused_capacity_weight)
        self.reward_finished_task_weight = float(reward_finished_task_weight)
        self.reward_priority_finish_weight = float(reward_priority_finish_weight)
        self.reward_urgent_backlog_weight = float(reward_urgent_backlog_weight)
        self.reward_waiting_weight = float(reward_waiting_weight)
        self.reward_peak_load_weight = float(reward_peak_load_weight)
        self.reward_pause_weight = float(reward_pause_weight)
        self.reward_resume_weight = float(reward_resume_weight)
        self.reward_non_interruptible_weight = float(reward_non_interruptible_weight)
        self.reward_load_smooth_weight = float(reward_load_smooth_weight)
        self.reward_action_smooth_weight = float(reward_action_smooth_weight)
        self.reward_bess_degradation_weight = float(reward_bess_degradation_weight)
        self.reward_bess_invalid_action_weight = float(reward_bess_invalid_action_weight)
        self.reward_soc_final_weight = float(reward_soc_final_weight)
        self.reward_grid_peak_weight = float(reward_grid_peak_weight)

        # 5. 动作由 N 个 server-group 计算强度 + 1 个有符号储能动作组成（21 维）：
        #    action[0:N]：N 个 server-group 的计算强度（[0,1]）；
        #    action[N]：有符号储能动作（[-1,1]，负充电、正放电）。
        self.server_action_dim = self.model.N
        self.extra_action_dim = 1
        self.action_dim = self.server_action_dim + self.extra_action_dim
        # 21 维：20 组计算（[0,1]）+ 1 有符号储能（[-1,1]）
        low = np.concatenate([np.zeros(self.server_action_dim, dtype=np.float32), np.array([-1.0], dtype=np.float32)])
        high = np.ones(self.action_dim, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 6. 底层 observation 的维度随 N 和 horizon 变化：
        #    current_obs_dim = 6 个全局特征 + 10 个任务池特征 + 6 组 server-group 特征 × N；
        #    forecast_obs_dim = 8 组前瞻特征 × horizon
        #    （price, temperature, arrival, pv, wind, carbon, sin, cos）；
        #    obs_dim = 6 + 10 + 6*N + 8*horizon。
        #    默认 N=20、horizon=24 时，current=136、forecast=192、base obs=328。
        self.global_obs_dim = 6
        self.task_pool_obs_dim = 10
        self.server_feature_groups = 6
        self.current_obs_dim = (
            self.global_obs_dim
            + self.task_pool_obs_dim
            + self.server_feature_groups * self.model.N
        )
        # 预测特征组固定顺序（M3.10b）：price, temperature, arrival, pv, wind, carbon, sin, cos
        self.forecast_feature_groups = 8
        self.forecast_obs_dim = self.forecast_feature_groups * self.horizon
        self.obs_dim = self.current_obs_dim + self.forecast_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # 7. 固定 24 小时外部输入
        self.hours = np.arange(self.horizon)
        default_T_amb = 25 + 5 * np.sin(np.pi * (self.hours - 8) / 12)
        self.T_amb = self._validate_time_series("T_amb", T_amb) if T_amb is not None else default_T_amb

        if price_t is not None:
            self.price_t = self._validate_time_series("price_t", price_t)
        elif hasattr(self.model, "create_price_curve"):
            self.price_t = self.model.create_price_curve(horizon=self.horizon)
        else:
            self.price_t = self._create_price_curve(horizon=self.horizon)
        self.carbon_factor_t = (
            self._validate_time_series("carbon_factor_t", carbon_factor_t)
            if carbon_factor_t is not None
            else self._create_carbon_factor_curve(horizon=self.horizon)
        )
        self.pv_t = self._validate_time_series("pv_t", pv_t) if pv_t is not None else np.zeros(self.horizon)
        if np.any(self.pv_t < -1e-9):
            raise ValueError("pv_t must be non-negative in kW.")
        self.pv_t = np.maximum(self.pv_t, 0.0)
        # 归一化参考值只取**声明/冻结**尺度（M3.10c），不得与当前序列峰值比较，
        # 否则等于按 episode 动态重算。输入超出参考值时按裁剪处理，见 forecast_clipping()。
        self.pv_ref_kw = max(float(pv_ref_kw), 1e-6)
        self.carbon_factor_ref = max(float(carbon_factor_ref), 1e-6)
        self.allow_pv_export = bool(allow_pv_export)
        if self.allow_pv_export:
            raise ValueError("allow_pv_export=True is not supported in this first PV integration.")
        self.wt_t = self._validate_time_series("wt_t", wt_t) if wt_t is not None else np.zeros(self.horizon)
        self.wind_ref_kw = max(float(wind_ref_kw), 1e-6)

        # 7b. **M1.3g-e-c-R1**：formal 链在此统一覆盖调用者的默认 exogenous 与 refs
        #     （必须在上面所有默认赋值**之后**，否则会被覆盖回去）。
        if self.formal:
            self._apply_formal_injection()
        else:
            self.price_forecast_t = np.asarray(self.price_t, dtype=np.float64)
            self.temperature_forecast_t = np.asarray(self.T_amb, dtype=np.float64)
            self.pv_forecast_t = np.asarray(self.pv_t, dtype=np.float64)
            self.wind_forecast_t = np.asarray(self.wt_t, dtype=np.float64)
            self.carbon_forecast_t = np.asarray(self.carbon_factor_t, dtype=np.float64)
            self.arrival_forecast_t = np.zeros(self.horizon, dtype=np.float64)

        # 8. 运行状态变量会在 reset() 中初始化
        self.current_step = 0
        self.tasks = []
        self.true_task_arrival_profile = np.zeros(self.horizon, dtype=np.float64)
        self.task_arrival_forecast = np.zeros(self.horizon, dtype=np.float64)
        # Backward-compatible ground-truth alias. Observation builders must never
        # use this alias for future-looking features.
        self.lambda_t = self.true_task_arrival_profile
        self.Q_t = 0.0
        self.prev_loads = np.full(self.model.N, self.base_load, dtype=np.float32)
        self.prev_action = np.full(self.action_dim, 0.5, dtype=np.float32)
        self.bess_soc = float(np.clip(self.bess_soc_init, self.bess_soc_min, self.bess_soc_max))
        self.bess_energy_kWh = self.bess_soc * self.bess_capacity_kWh

        # 9. episode 累计指标
        self.total_energy_kWh = 0.0
        self.total_idc_energy_kWh = 0.0
        self.total_grid_energy_kWh = 0.0
        self.total_cost = 0.0
        self.total_carbon_emission = 0.0
        self.total_carbon_cost = 0.0
        self.episode_peak_power_kW = 0.0
        self.total_peak_excess_kW_hour = 0.0
        self.episode_grid_peak_power_kW = 0.0
        self.total_grid_peak_excess_kW_hour = 0.0
        self.total_completed_work = 0.0
        self.total_bess_charge_kWh = 0.0
        self.total_bess_discharge_kWh = 0.0
        self.total_bess_degradation_cost = 0.0
        self.total_pv_available_kWh = 0.0
        self.total_pv_used_kWh = 0.0
        self.total_pv_curtail_kWh = 0.0
        self.total_wind_available_kWh = 0.0
        self.total_wind_used_kWh = 0.0
        self.total_wind_curtail_kWh = 0.0
        self.deadline_miss_task_ids = set()
        # 终止结算账务字段（M3.6）
        self._settlement_done = False
        self.terminal_leftover_work = 0.0
        self.terminal_deadline_miss_count = 0
        self.terminal_soc_recovery_kwh = 0.0
        self.terminal_service_violation = 0
        self.terminal_settlement_penalty = 0.0
        self.total_objective_cost = 0.0

        # 10. 任务启停统计指标
        self.total_pause_count = 0
        self.total_resume_count = 0
        self.total_non_interruptible_interruption_count = 0

    def _validate_time_series(self, name: str, values) -> np.ndarray:
        """Validate externally supplied hourly data before the environment uses it."""
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.shape[0] != self.horizon:
            raise ValueError(
                f"{name} length must equal horizon={self.horizon}, got {arr.shape[0]}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values.")
        return arr

    def _create_price_curve(self, horizon: int = 24) -> np.ndarray:
        """备用分时电价曲线。"""
        price_t = np.zeros(horizon, dtype=np.float64)
        valley_price = 0.35
        flat_price = 0.65
        peak_price = 1.05

        for t in range(horizon):
            if 0 <= t < 7:
                price_t[t] = valley_price
            elif 10 <= t < 15 or 18 <= t < 21:
                price_t[t] = peak_price
            else:
                price_t[t] = flat_price

        return price_t

    def _create_carbon_factor_curve(self, horizon: int = 24) -> np.ndarray:
        """
        Create a default hourly grid carbon factor curve.

        Unit: kgCO2/kWh. The values are scenario assumptions for simulation:
        lower carbon intensity around midday, higher intensity during evening
        peak hours, and medium-high intensity overnight.
        """
        carbon_factor_t = np.zeros(horizon, dtype=np.float64)

        low_carbon = 0.45
        flat_carbon = 0.60
        night_carbon = 0.70
        peak_carbon = 0.80

        for t in range(horizon):
            if 10 <= t < 16:
                carbon_factor_t[t] = low_carbon
            elif 18 <= t < 22:
                carbon_factor_t[t] = peak_carbon
            elif 0 <= t < 7:
                carbon_factor_t[t] = night_carbon
            else:
                carbon_factor_t[t] = flat_carbon

        return carbon_factor_t

    def _server_group_info(self) -> dict:
        return {
            "facility_rated_power_mw": self.facility_rated_power_mw,
            "server_group_model_enabled": bool(self.server_group_model_enabled),
            "server_group_size": int(self.server_group_size),
            "num_server_groups": int(self.num_server_groups),
            "effective_total_server_count": int(self.effective_total_server_count),
            "total_group_capacity": float(getattr(self.model, "total_group_capacity", self.model.C_IDC)),
            "total_group_idle_power_kW": float(getattr(self.model, "total_group_idle_power_kW", np.sum(self.model.P_idle) / 1000.0)),
            "total_group_max_power_kW": float(getattr(self.model, "total_group_max_power_kW", np.sum(self.model.P_max) / 1000.0)),
        }

    def _task_scale_info(self) -> dict:
        total_workload = self._total_available_work() if self.tasks else 0.0
        task_count = len(self.tasks) if self.tasks else 0
        return {
            "task_workload_scale": float(self.task_workload_scale),
            "sla_penalty_ref": float(self.sla_penalty_ref),
            "effective_total_workload": float(total_workload),
            "average_task_workload": float(total_workload / task_count) if task_count > 0 else 0.0,
        }

    def _bess_static_info(self) -> dict:
        return {
            "bess_capacity_kWh": float(self.bess_capacity_kWh),
            "bess_charge_power_max_kW": float(self.bess_charge_power_max_kW),
            "bess_discharge_power_max_kW": float(self.bess_discharge_power_max_kW),
            "bess_charge_efficiency": float(self.bess_charge_efficiency),
            "bess_discharge_efficiency": float(self.bess_discharge_efficiency),
            "bess_degradation_cost_ref": float(self.bess_degradation_cost_ref),
            "bess_scale_factor": float(self.bess_scale_factor),
        }

    def _task_forecast_info(self, *, include_profiles: bool) -> dict:
        metrics = task_forecast_metrics(
            self.true_task_arrival_profile, self.task_arrival_forecast
        )
        if getattr(self, "formal", False):
            source = (
                "B6 causal expected arrival forecast "
                "(rate_template[slot] x 31.994 work/step; D3 expectation, not a draw)"
            )
        elif self.task_forecast_mode == "noisy":
            source = SYNTHETIC_FORECAST_SOURCE
        elif self.task_forecast_mode == "perfect":
            source = "oracle ground-truth copy for debug/upper-bound use only"
        else:
            source = "task-arrival forecast disabled (all-zero forecast)"
        info = {
            "task_forecast_mode": self.task_forecast_mode,
            "forecast_error_level": float(self.forecast_error_level),
            "task_forecast_seed": self.forecast_seed,
            "task_forecast_source": source,
            "task_forecast_mae": metrics.mae,
            "task_forecast_rmse": metrics.rmse,
            "task_forecast_mape_nonzero_percent": metrics.mape_nonzero_percent,
        }
        if include_profiles:
            info.update(
                {
                    "true_task_arrival_profile": self.true_task_arrival_profile.copy(),
                    "task_arrival_forecast": self.task_arrival_forecast.copy(),
                }
            )
        return info

    def _apply_formal_injection(self) -> None:
        """把**已验证**的 formal 注入统一落到环境上（realized / forecasts / refs）。

        - realized exogenous → `price_t` / `T_amb` / `carbon_factor_t` / `pv_t` / `wt_t`
          （**step 的物理计算**读它们）；
        - causal forecasts → `*_forecast_t`（**观测**读它们）；
        - refs_v4 的正式参考值逐项覆盖。
        两者**明确分离**，不得混用。
        """
        inj = self.formal_injection
        self.price_t = np.asarray(inj.price_sgd_per_kwh, dtype=np.float64)
        self.T_amb = np.asarray(inj.temperature_deg_c, dtype=np.float64)
        self.carbon_factor_t = np.asarray(inj.carbon_kg_per_kwh, dtype=np.float64)
        self.pv_t = np.asarray(inj.local_pv_kw, dtype=np.float64)
        self.wt_t = np.asarray(inj.wind_generation_kw, dtype=np.float64)
        for name, value in inj.formal_refs.items():
            setattr(self, name, float(value))
        causal = inj.causal_forecasts
        self.price_forecast_t = np.asarray(causal["price_forecast"], dtype=np.float64)
        self.temperature_forecast_t = np.asarray(
            causal["temperature_forecast"], dtype=np.float64)
        self.pv_forecast_t = np.asarray(causal["pv_forecast"], dtype=np.float64)
        self.wind_forecast_t = np.asarray(causal["wind_forecast"], dtype=np.float64)
        self.carbon_forecast_t = np.asarray(causal["carbon_forecast"], dtype=np.float64)
        self.arrival_forecast_t = np.asarray(causal["arrival_forecast"], dtype=np.float64)

    def _formal_provenance_info(self) -> dict:
        """formal 链的 provenance（**不含**任何未来 truth 数组）。"""
        if not self.formal:
            return {}
        inj = self.formal_injection
        return {
            "formal": True,
            "split": inj.split,
            "start": inj.start,
            "local_origin": int(inj.local_origin),
            "global_origin": int(inj.global_origin),
            "horizon": int(inj.horizon),
            "forecast_cutoff": int(inj.forecast_cutoff),
            "delta_t_hours": float(self.delta_t_hours),
            "lambda_ref_work_per_step": float(self.lambda_ref),
            "provenance_hash": inj.provenance_hash,
            "sources": [list(s) for s in inj.sources],
        }

    def reset(self, seed=None, options=None):
        """重置环境，开始新的 24 小时 episode。"""
        super().reset(seed=seed)

        self.current_step = 0
        self.prev_loads = np.full(self.model.N, self.base_load, dtype=np.float32)
        # prev_action anchors action smoothing; first step measures deviation from neutral dispatch.
        self.prev_action = np.full(self.action_dim, 0.5, dtype=np.float32)
        self.bess_soc = float(np.clip(self.bess_soc_init, self.bess_soc_min, self.bess_soc_max))
        self.bess_energy_kWh = self.bess_soc * self.bess_capacity_kWh

        self.total_energy_kWh = 0.0
        self.total_idc_energy_kWh = 0.0
        self.total_grid_energy_kWh = 0.0
        self.total_cost = 0.0
        self.total_carbon_emission = 0.0
        self.total_carbon_cost = 0.0
        self.episode_peak_power_kW = 0.0
        self.total_peak_excess_kW_hour = 0.0
        self.episode_grid_peak_power_kW = 0.0
        self.total_grid_peak_excess_kW_hour = 0.0
        self.total_completed_work = 0.0
        self.total_bess_charge_kWh = 0.0
        self.total_bess_discharge_kWh = 0.0
        self.total_bess_degradation_cost = 0.0
        self.total_pv_available_kWh = 0.0
        self.total_pv_used_kWh = 0.0
        self.total_pv_curtail_kWh = 0.0
        self.total_wind_available_kWh = 0.0
        self.total_wind_used_kWh = 0.0
        self.total_wind_curtail_kWh = 0.0
        self.deadline_miss_task_ids = set()
        # 终止结算账务字段（M3.6）
        self._settlement_done = False
        self.terminal_leftover_work = 0.0
        self.terminal_deadline_miss_count = 0
        self.terminal_soc_recovery_kwh = 0.0
        self.terminal_service_violation = 0
        self.terminal_settlement_penalty = 0.0
        self.total_objective_cost = 0.0
        self.total_pause_count = 0
        self.total_resume_count = 0
        self.total_non_interruptible_interruption_count = 0

        # 每个 episode 重新生成任务，防止 Task.status/remaining_work 等状态残留。
        # **M1.3g-e-c**：formal 路径**绝不**调用 demo/random 生成器。
        if self.formal:
            # **每次 reset 都新建 Task**：按不可变规格重建，绝不复用已执行实例。
            from scenario.env_injection import materialize_pristine_tasks

            self.tasks = list(materialize_pristine_tasks(self.formal_injection))
        else:
            self.tasks = self.model.create_demo_tasks(
                num_tasks=self.num_tasks,
                horizon=self.horizon,
            )

        # 将原来的 Q0 改造成初始积压任务，并纳入 Task 列表。
        initial_backlog_task = self.model.create_initial_backlog_task(self.initial_Q)
        self.tasks.insert(0, initial_backlog_task)
        self._initialize_task_runtime_state()

        if self.formal:
            # 真值 = mapper 的**整数账本**（每槽 aggregate，单位 work）；
            # 预测 = B6 causal **期望** forecast。两者**不得**互换。
            self.true_task_arrival_profile = np.asarray(
                [m / 1_000_000.0 for m in self.formal_injection.ledger_micro],
                dtype=np.float64,
            )
            self.lambda_t = self.true_task_arrival_profile
            self.task_arrival_forecast = np.asarray(
                self.formal_injection.arrival_forecast, dtype=np.float64)
        else:
            self.true_task_arrival_profile = self.model.build_task_arrival_curve(
                tasks=self.tasks,
                horizon=self.horizon,
            )
            self.lambda_t = self.true_task_arrival_profile
            self.task_arrival_forecast = generate_task_arrival_forecast(
                self.true_task_arrival_profile,
                mode=self.task_forecast_mode,
                error_level=self.forecast_error_level,
                rng=self.forecast_rng,
            )

        # reset 后先激活 t=0 已到达任务，让初始状态能看到初始积压。
        self._activate_arrivals(current_time=0)
        self.Q_t = self._compute_backlog_work()

        obs = self._get_obs()
        info = {
            "total_task_count": len(self.tasks),
            "initial_backlog_work": self.Q_t,
            "obs_dim": self.obs_dim,
            "pv_ref_kw": float(self.pv_ref_kw),
            "allow_pv_export": bool(self.allow_pv_export),
            **self._task_forecast_info(include_profiles=not self.formal),
            **(self._formal_provenance_info()),
            **self._server_group_info(),
            **self._task_scale_info(),
            **self._bess_static_info(),
        }

        return obs, info

    def step(self, action):
        """
        执行一步，也就是推进 1 小时。

        PPO 输入为 N+1 维 flat action（默认 N=20 时为 21 维）：
            action[0:N] 表示 N 个 server-group 的计算强度（[0, 1]）；
            action[N] 表示有符号储能动作（[-1, 1]，负充电、正放电）。

        本版处理逻辑：
            1. 动作先转换为计划任务负载和计划处理能力；
            2. 执行前按接入/可再生/储能算功率预算，限缩计划容量后走逐组 A[i,g] 分配；
            3. 每组实际负载由完成工作/组能力导出；
            4. 用实际负载计算功耗和成本；
            5. Q_t 由 Task.remaining_work 统计得到。
        """
        t = self.current_step

        # 1. 解析 N+1 维动作；默认 N=20 时为 21 维。
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_dim:
            raise ValueError(
                f"动作维度错误：期望 {self.action_dim} 维，实际 {action.shape[0]} 维。"
            )
        server_action = np.clip(action[:self.model.N], 0.0, 1.0)
        bess_raw_action = float(np.clip(action[self.model.N], -1.0, 1.0))
        # urgent/continuity 动作维度已移除（M3.9），由任务自身 priority/deadline 约束替代
        urgent_preference = 0.0
        continuity_preference = 0.0

        # 2. PPO 计划任务负载：仅用于计算计划处理能力，不直接用于功耗
        planned_task_loads = server_action * self.max_task_load_per_server
        planned_total_loads = np.clip(self.base_load + planned_task_loads, 0.0, 1.0)

        # 按每台服务器算力计算逐组计划处理能力（M3.1：不再 np.sum 成标量）
        planned_capacity_vec = planned_task_loads * np.asarray(self.model.C_server, dtype=np.float64)
        # M1.3g-e-c：`C_server` 是 **work/hour rate**；formal 链每一步的可执行量必须
        # 折算成 **work/step**（× delta_t_hours = 0.5）。legacy 路径 delta=1.0，
        # 因此**不**改变其既有语义。
        self.last_planned_capacity_rate_work_per_hour = float(planned_capacity_vec.sum())
        if self.formal:
            planned_capacity_vec = planned_capacity_vec * self.delta_t_hours
        planned_capacity = float(planned_capacity_vec.sum())
        self.last_planned_capacity_per_step = planned_capacity

        # 3. 当前小时外部输入
        T_amb_t = float(self.T_amb[t])
        price_now = float(self.price_t[t])
        carbon_factor_now = float(self.carbon_factor_t[t])
        pv_now = float(self.pv_t[t])
        wt_now = float(self.wt_t[t])
        lambda_now = float(self.true_task_arrival_profile[t])

        # 4. 当前小时新任务到达
        self._activate_arrivals(current_time=t)

        # 5. 执行前物理可行性投影（M3.7a）：先由可见可再生 + SOC/功率边界修正后的储能
        #    功率 + access_limit 算出计算负载的物理功率预算，据此限缩 planned_capacity_vec。
        # 5a. 执行储能功率（SOC/功率边界修正，不含 no-export 限制）
        if bess_raw_action < 0.0:
            desired_bess_charge_power_kW = abs(bess_raw_action) * self.bess_charge_power_max_kW
            desired_bess_discharge_power_kW = 0.0
        else:
            desired_bess_charge_power_kW = 0.0
            desired_bess_discharge_power_kW = bess_raw_action * self.bess_discharge_power_max_kW

        max_charge_energy_by_soc = max(
            (self.bess_soc_max - self.bess_soc) * self.bess_capacity_kWh, 0.0
        )
        max_discharge_energy_by_soc = max(
            (self.bess_soc - self.bess_soc_min) * self.bess_capacity_kWh, 0.0
        )
        charge_power_limit_by_soc = max_charge_energy_by_soc / max(
            self.delta_t_hours * self.bess_charge_efficiency, 1e-6
        )
        discharge_power_limit_by_soc = (
            max_discharge_energy_by_soc * self.bess_discharge_efficiency / max(self.delta_t_hours, 1e-6)
        )
        bess_charge_power_kW = min(
            desired_bess_charge_power_kW, self.bess_charge_power_max_kW, charge_power_limit_by_soc
        )
        bess_discharge_power_kW = min(
            desired_bess_discharge_power_kW, self.bess_discharge_power_max_kW, discharge_power_limit_by_soc
        )

        # 5b. 可再生能源可用量
        pv_available_kW = max(pv_now, 0.0)
        wind_available_kW = max(wt_now, 0.0)

        # 5c. 基础负载功率（必须先计入，不得挪给任务）
        P_base_kW = self._idc_power_kw(np.clip(self.base_load, 0.0, 1.0), T_amb_t)

        # 5d. 充电受接入上限物理投影：charge <= access + 可再生 + 放电 - 基础负载
        charge_headroom_kW = max(
            self.access_limit_kw
            + pv_available_kW
            + wind_available_kW
            + bess_discharge_power_kW
            - P_base_kW,
            0.0,
        )
        bess_charge_power_kW = min(bess_charge_power_kW, charge_headroom_kW)

        # 5e. IDC 功率预算 = 接入上限 + 放电 - 充电(已投影) + 可再生可用（无反送电）
        P_idc_budget_kW = max(
            self.access_limit_kw
            + bess_discharge_power_kW
            - bess_charge_power_kW
            + pv_available_kW
            + wind_available_kW,
            0.0,
        )

        # 5f. 任务功率预算 + 缩放（确定性可行性搜索；基础负载先计入）
        task_power_budget_kW = max(P_idc_budget_kW - P_base_kW, 0.0)
        if task_power_budget_kW <= 1e-9:
            scale = 0.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2.0
                task_power = self._idc_power_kw(
                    self.base_load + planned_task_loads * mid, T_amb_t
                ) - P_base_kW
                if task_power <= task_power_budget_kW:
                    lo = mid
                else:
                    hi = mid
            scale = lo

        # 5g. 限缩逐组计划容量
        scaled_capacity_vec = planned_capacity_vec * scale

        # 6. 执行任务（真实 A[i,g] 分配，用限缩后的容量）
        (
            completed_work,
            completed_work_by_group,
            w_unconstrained,
            w_feasible,
            newly_finished_count,
            newly_finished_priority_sum,
            new_deadline_miss_count,
            pause_count_this_step,
            resume_count_this_step,
            non_interruptible_interruption_this_step,
        ) = self._execute_tasks_action_guided(
            planned_capacity_vec=scaled_capacity_vec,
            unconstrained_capacity_vec=planned_capacity_vec,
            current_time=t,
            urgent_preference=urgent_preference,
            continuity_preference=continuity_preference,
        )

        access_curtailment_work = max(w_unconstrained - w_feasible, 0.0)
        unused_capacity = max(planned_capacity - completed_work, 0.0)

        # 7. 每组实际负载由完成工作/组能力导出（完整基础负载 + 任务负载）
        actual_task_loads = self._loads_from_group_completion(completed_work_by_group)
        actual_total_loads = np.clip(self.base_load + actual_task_loads, 0.0, 1.0)

        # 8. 用实际负载计算当前小时功耗（完整 IDC 需求），再按预算钳位为已服务部分
        L_matrix = actual_total_loads.reshape(1, -1)
        P_IDC_arr, P_IT_arr, PUE_arr, COP_arr, P_cooling_arr = self.model.calc_pue_and_total_power(
            L_matrix=L_matrix,
            T_amb=np.array([T_amb_t], dtype=np.float64),
        )
        P_IDC_t = float(P_IDC_arr[0])
        P_IT_t = float(P_IT_arr[0])
        PUE_t = float(PUE_arr[0])
        COP_t = float(COP_arr[0])
        P_cooling_t = float(P_cooling_arr[0])
        P_IDC_demand_kW = P_IDC_t / 1000.0
        P_base_demand_kW = P_base_kW
        dec = decompose_supply(P_base_demand_kW, P_IDC_demand_kW, P_idc_budget_kW)
        P_task_incremental_demand_kW = dec["task_incremental_demand_kW"]
        P_base_served_kW = dec["base_served_kW"]
        P_task_served_kW = dec["task_served_kW"]
        unserved_base_load_kW = dec["unserved_base_load_kW"]
        unserved_task_power_kW = dec["unserved_task_power_kW"]
        P_IDC_served_kW = dec["idc_served_kW"]
        P_IDC_kW = P_IDC_served_kW  # 旧字段，等价于 served

        # 9. 应用 no-export 放电限制（放电 <= IDC），再算可再生与购电
        bess_discharge_power_kW = min(bess_discharge_power_kW, P_IDC_kW)

        P_local_demand_kW = P_IDC_kW + bess_charge_power_kW
        P_local_net_before_pv_kW = P_local_demand_kW - bess_discharge_power_kW
        pv_used_kW = min(pv_available_kW, max(P_local_net_before_pv_kW, 0.0))
        pv_curtail_kW = max(pv_available_kW - pv_used_kW, 0.0)
        P_after_pv_kW = P_local_net_before_pv_kW - pv_used_kW
        wind_used_kW = min(wind_available_kW, max(P_after_pv_kW, 0.0))
        wind_curtail_kW = max(wind_available_kW - wind_used_kW, 0.0)
        P_bus_net_kW = P_after_pv_kW - wind_used_kW
        # 因任务容量已前置限缩，购电应 <= access_limit；无反送电则非负钳位。
        P_grid_kW = max(P_bus_net_kW, 0.0)

        # 10. SOC 更新（按实际充/放电功率）
        bess_charge_kWh = bess_charge_power_kW * self.delta_t_hours
        bess_discharge_kWh = bess_discharge_power_kW * self.delta_t_hours
        charged_energy_to_battery = bess_charge_kWh * self.bess_charge_efficiency
        discharged_energy_from_battery = bess_discharge_kWh / max(self.bess_discharge_efficiency, 1e-6)
        current_bess_energy_kWh = self.bess_energy_kWh
        bess_energy_next = current_bess_energy_kWh + charged_energy_to_battery - discharged_energy_from_battery
        bess_energy_next = float(np.clip(
            bess_energy_next,
            self.bess_soc_min * self.bess_capacity_kWh,
            self.bess_soc_max * self.bess_capacity_kWh,
        ))
        bess_soc_next = bess_energy_next / max(self.bess_capacity_kWh, 1e-6)

        bess_throughput_kWh = bess_charge_kWh + bess_discharge_kWh
        bess_degradation_cost = bess_throughput_kWh * self.bess_degradation_cost_per_kWh
        if bess_charge_power_kW > 1e-9:
            bess_mode = "charge"
        elif bess_discharge_power_kW > 1e-9:
            bess_mode = "discharge"
        else:
            bess_mode = "idle"
        invalid_bess_action = (
            abs(desired_bess_charge_power_kW - bess_charge_power_kW)
            + abs(desired_bess_discharge_power_kW - bess_discharge_power_kW)
        )

        # 11. 能量与成本
        idc_energy_kWh = P_IDC_kW * self.delta_t_hours
        grid_energy_kWh = P_grid_kW * self.delta_t_hours
        pv_available_kWh = pv_available_kW * self.delta_t_hours
        pv_used_kWh = pv_used_kW * self.delta_t_hours
        pv_curtail_kWh = pv_curtail_kW * self.delta_t_hours
        wind_available_kWh = wind_available_kW * self.delta_t_hours
        wind_used_kWh = wind_used_kW * self.delta_t_hours
        wind_curtail_kWh = wind_curtail_kW * self.delta_t_hours
        # Backward-compatible alias: energy_kWh now means grid-purchased energy for cost/carbon.
        energy_kWh = grid_energy_kWh
        cost_t = grid_energy_kWh * price_now
        carbon_emission_t = grid_energy_kWh * carbon_factor_now
        carbon_cost_t = carbon_emission_t * self.carbon_price

        # 9. 更新任务积压统计
        Q_next = self._compute_backlog_work()

        # 10. 累计统计
        self.total_idc_energy_kWh += idc_energy_kWh
        self.total_grid_energy_kWh += grid_energy_kWh
        # total_energy_kWh is kept for old scripts and follows total_grid_energy_kWh.
        self.total_energy_kWh = self.total_grid_energy_kWh
        self.total_cost += cost_t
        self.total_carbon_emission += carbon_emission_t
        self.total_carbon_cost += carbon_cost_t
        self.total_completed_work += completed_work
        self.total_bess_charge_kWh += bess_charge_kWh
        self.total_bess_discharge_kWh += bess_discharge_kWh
        self.total_bess_degradation_cost += bess_degradation_cost
        self.total_pv_available_kWh += pv_available_kWh
        self.total_pv_used_kWh += pv_used_kWh
        self.total_pv_curtail_kWh += pv_curtail_kWh
        self.total_wind_available_kWh += wind_available_kWh
        self.total_wind_used_kWh += wind_used_kWh
        self.total_wind_curtail_kWh += wind_curtail_kWh
        # 每步统一更新客观目标账务（M3.6a）：非终止步罚项为 0，终止步由结算写入。
        self.total_objective_cost = (
            self.total_cost + self.total_bess_degradation_cost + self.terminal_settlement_penalty
        )

        # 11. reward：任务类综合奖励
        # 奖励项：完成工作量、完整完成任务数、高优先级任务完成；
        # 惩罚项：成本、普通积压、紧急积压、等待压力、超时、未使用能力、高电价高负载、暂停/恢复和不可暂停任务中断。
        urgent_backlog_work, avg_waiting_pressure = self._compute_reward_task_pressure(current_time=t)
        sla_metrics = self._compute_sla_metrics(current_time=t + 1)

        completed_norm = completed_work / self.queue_ref
        finished_task_norm = newly_finished_count / max(len(self.tasks), 1)
        priority_finish_norm = newly_finished_priority_sum / max(5.0 * len(self.tasks), 1e-6)
        cost_norm = cost_t / self.cost_ref
        carbon_norm = carbon_emission_t / max(self.carbon_ref, 1e-6)
        queue_norm = Q_next / self.queue_ref
        # overflow_work is the portion of backlog above the soft queue capacity.
        overflow_work = max(Q_next - self.queue_capacity_ref, 0.0)
        overflow_norm = overflow_work / max(self.queue_ref, 1e-6)
        urgent_backlog_norm = urgent_backlog_work / max(self.queue_ref, 1e-6)
        waiting_norm = avg_waiting_pressure / max(self.horizon, 1)
        deadline_miss_norm = new_deadline_miss_count / max(len(self.tasks), 1)
        sla_penalty_norm = sla_metrics["sla_penalty"] / self.sla_penalty_ref
        unused_capacity_norm = unused_capacity / max(self.queue_ref, 1e-6)
        # Grid peak is based on P_grid; P_IDC_kW remains the physical IDC load metric.
        grid_power_kW = P_grid_kW
        grid_peak_power_kW = grid_power_kW
        grid_peak_excess_kW = max(grid_power_kW - self.grid_power_limit_kW, 0.0)
        self.episode_grid_peak_power_kW = max(self.episode_grid_peak_power_kW, grid_power_kW)
        self.total_grid_peak_excess_kW_hour += grid_peak_excess_kW * self.delta_t_hours
        # Backward-compatible aliases: peak_power_kW now means grid purchase peak power.
        peak_power_kW = grid_power_kW
        peak_excess_kW = grid_peak_excess_kW
        self.episode_peak_power_kW = self.episode_grid_peak_power_kW
        self.total_peak_excess_kW_hour = self.total_grid_peak_excess_kW_hour
        peak_load_norm = grid_peak_excess_kW / max(self.peak_power_ref_kW, 1e-6)
        pause_norm = pause_count_this_step / max(len(self.tasks), 1)
        resume_norm = resume_count_this_step / max(len(self.tasks), 1)
        non_interruptible_norm = non_interruptible_interruption_this_step / max(len(self.tasks), 1)
        # load_change penalizes rapid server utilization movement between adjacent hours.
        load_change = float(np.mean(np.abs(actual_total_loads - self.prev_loads)))
        # action_change penalizes policy jitter between adjacent continuous action vectors.
        action_change = float(np.mean(np.abs(action - self.prev_action)))
        bess_power_ref = max(self.bess_charge_power_max_kW, self.bess_discharge_power_max_kW, 1e-6)

        # Positive rewards: completed work and finished tasks.
        r_done = self.reward_done_weight * completed_norm
        r_finished_task = self.reward_finished_task_weight * finished_task_norm
        r_priority_finish = self.reward_priority_finish_weight * priority_finish_norm

        # Cost penalties: electricity cost and grid carbon emissions from IDC power.
        r_cost = -self.reward_cost_weight * cost_norm
        r_carbon = -self.reward_carbon_weight * carbon_norm

        # Queue and service-quality penalties: backlog, urgent backlog, deadline, SLA, and final queue.
        r_queue = -self.reward_queue_weight * queue_norm
        r_queue_overflow = -self.reward_queue_overflow_weight * overflow_norm
        r_urgent_backlog = -self.reward_urgent_backlog_weight * urgent_backlog_norm
        r_waiting = -self.reward_waiting_weight * waiting_norm
        r_deadline = -self.reward_deadline_weight * deadline_miss_norm
        r_sla = -self.reward_sla_weight * sla_penalty_norm

        # Resource and scheduling penalties: unused capacity, peak load, interruptions, and smoothing.
        r_unused = -self.reward_unused_capacity_weight * unused_capacity_norm
        r_grid_peak = -self.reward_grid_peak_weight * peak_load_norm
        r_peak_load = r_grid_peak
        r_pause = -self.reward_pause_weight * pause_norm
        r_resume = -self.reward_resume_weight * resume_norm
        r_non_interruptible = -self.reward_non_interruptible_weight * non_interruptible_norm
        r_load_smooth = -self.reward_load_smooth_weight * load_change
        r_action_smooth = -self.reward_action_smooth_weight * action_change

        # BESS penalties: degradation and invalid clipped actions are soft reward terms.
        r_bess_degradation = (
            -self.reward_bess_degradation_weight
            * bess_degradation_cost
            / self.bess_degradation_cost_ref
        )
        r_bess_invalid_action = (
            -self.reward_bess_invalid_action_weight
            * invalid_bess_action
            / bess_power_ref
        )
        r_final_queue = 0.0
        r_soc_final = 0.0

        reward = (
            r_done
            + r_finished_task
            + r_priority_finish
            + r_cost
            + r_carbon
            + r_queue
            + r_queue_overflow
            + r_urgent_backlog
            + r_waiting
            + r_deadline
            + r_sla
            + r_unused
            + r_peak_load
            + r_pause
            + r_resume
            + r_non_interruptible
            + r_load_smooth
            + r_action_smooth
            + r_bess_degradation
            + r_bess_invalid_action
        )

        terminated = (self.current_step + 1) >= self.horizon
        truncated = False

        # 最后一小时额外惩罚最终积压，避免 PPO 一直拖任务
        if terminated:
            final_queue_norm = Q_next / max(self.queue_ref, 1e-6)
            r_final_queue = -self.reward_final_queue_weight * final_queue_norm
            reward += r_final_queue
            soc_deviation = abs(bess_soc_next - self.bess_soc_target)
            soc_excess = max(soc_deviation - self.bess_soc_final_tolerance, 0.0)
            r_soc_final = -self.reward_soc_final_weight * soc_excess
            reward += r_soc_final
        else:
            soc_deviation = abs(bess_soc_next - self.bess_soc_target)
            soc_excess = 0.0

        # 12. 更新环境内部状态
        self.Q_t = Q_next
        self.prev_loads = actual_total_loads.astype(np.float32)
        self.prev_action = action.copy()
        self.bess_soc = float(bess_soc_next)
        self.bess_energy_kWh = float(bess_energy_next)
        self.current_step += 1

        # 12b. 终止结算账务（在 info 构建前应用，使扁平字段与累计一致）
        if terminated:
            settlement = self._apply_terminal_settlement()
        else:
            settlement = None

        # 13. 生成下一状态
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        # 14. 记录信息，方便训练后画图和计算指标
        task_metrics = self._compute_task_metrics()
        total_available_work = self._total_available_work()

        completion_rate = (
            self.total_completed_work / total_available_work
            if total_available_work > 0
            else 0.0
        )
        pv_utilization_rate = (
            self.total_pv_used_kWh / max(self.total_pv_available_kWh, 1e-9)
            if self.total_pv_available_kWh > 0.0
            else 0.0
        )
        wind_utilization_rate = (
            self.total_wind_used_kWh / max(self.total_wind_available_kWh, 1e-9)
            if self.total_wind_available_kWh > 0.0
            else 0.0
        )
        renewable_share = (
            (self.total_pv_used_kWh + self.total_wind_used_kWh) / max(self.total_idc_energy_kWh, 1e-9)
            if self.total_idc_energy_kWh > 0.0
            else 0.0
        )

        if self.total_completed_work > 0:
            unit_task_cost = self.total_cost / self.total_completed_work
            energy_per_task = self.total_grid_energy_kWh / self.total_completed_work
            idc_energy_per_task = self.total_idc_energy_kWh / self.total_completed_work
            carbon_per_task = self.total_carbon_emission / self.total_completed_work
        else:
            unit_task_cost = np.inf
            energy_per_task = np.inf
            idc_energy_per_task = np.inf
            carbon_per_task = np.inf

        info = {
            "hour": t,
            "price": price_now,
            "carbon_factor": carbon_factor_now,
            "PV": pv_now,
            "WT": wt_now,
            "pv_available_kW": float(pv_available_kW),
            "pv_used_kW": float(pv_used_kW),
            "pv_curtail_kW": float(pv_curtail_kW),
            "wind_available_kW": float(wind_available_kW),
            "wind_used_kW": float(wind_used_kW),
            "wind_curtail_kW": float(wind_curtail_kW),
            "allow_pv_export": bool(self.allow_pv_export),
            "lambda_t": lambda_now,
            **self._task_forecast_info(
                include_profiles=terminated and not self.formal),
            **(self._formal_provenance_info()),

            "action_mean": float(np.mean(server_action)),
            "action_min": float(np.min(server_action)),
            "action_max": float(np.max(server_action)),
            "urgent_preference": urgent_preference,
            "continuity_preference": continuity_preference,
            "bess_raw_action": float(bess_raw_action),

            "planned_task_load_mean": float(np.mean(planned_task_loads)),
            "planned_total_load_mean": float(np.mean(planned_total_loads)),
            "actual_task_load_mean": float(np.mean(actual_task_loads)),
            "actual_total_load_mean": float(np.mean(actual_total_loads)),

            "planned_capacity": planned_capacity,
            "raw_capacity": planned_capacity,  # 兼容旧字段名
            "planned_capacity_vec": planned_capacity_vec,
            "completed_work_by_group": completed_work_by_group,
            "completed_work": completed_work,
            "unused_capacity": unused_capacity,
            "Q": Q_next,
            "backlog_work": Q_next,
            "queue_capacity_ref": float(self.queue_capacity_ref),
            "overflow_work": float(overflow_work),

            "newly_finished_count": newly_finished_count,
            "newly_finished_priority_sum": float(newly_finished_priority_sum),
            "urgent_backlog_work": float(urgent_backlog_work),
            "avg_waiting_pressure": float(avg_waiting_pressure),
            "new_deadline_miss_count": new_deadline_miss_count,
            "deadline_miss_count": task_metrics["deadline_miss_count"],
            "sla_penalty": float(sla_metrics["sla_penalty"]),
            "sla_penalty_norm": float(sla_penalty_norm),
            "sla_violation_count": int(sla_metrics["sla_violation_count"]),
            "sla_violation_rate": float(sla_metrics["sla_violation_rate"]),
            "avg_task_delay": float(sla_metrics["avg_task_delay"]),
            "max_task_delay": float(sla_metrics["max_task_delay"]),
            "load_change": float(load_change),
            "action_change": float(action_change),

            "pause_count_this_step": pause_count_this_step,
            "resume_count_this_step": resume_count_this_step,
            "non_interruptible_interruption_this_step": non_interruptible_interruption_this_step,
            "total_pause_count": self.total_pause_count,
            "total_resume_count": self.total_resume_count,
            "total_non_interruptible_interruption_count": self.total_non_interruptible_interruption_count,

            # reward 分项，便于后续调参和定位问题
            "r_done": float(r_done),
            "r_finished_task": float(r_finished_task),
            "r_priority_finish": float(r_priority_finish),
            "r_cost": float(r_cost),
            "r_carbon": float(r_carbon),
            "r_queue": float(r_queue),
            "r_queue_overflow": float(r_queue_overflow),
            "r_urgent_backlog": float(r_urgent_backlog),
            "r_waiting": float(r_waiting),
            "r_deadline": float(r_deadline),
            "r_sla": float(r_sla),
            "r_unused": float(r_unused),
            "r_grid_peak": float(r_grid_peak),
            "r_peak_load": float(r_peak_load),
            "r_pause": float(r_pause),
            "r_resume": float(r_resume),
            "r_non_interruptible": float(r_non_interruptible),
            "r_load_smooth": float(r_load_smooth),
            "r_action_smooth": float(r_action_smooth),
            "r_bess_degradation": float(r_bess_degradation),
            "r_bess_invalid_action": float(r_bess_invalid_action),
            "r_final_queue": float(r_final_queue),
            "r_soc_final": float(r_soc_final),
            "reward_total": float(reward),

            "P_IDC": P_IDC_t,
            "P_IDC_kW": float(P_IDC_kW),  # 等价于 P_IDC_served_kW（旧字段）
            "P_IDC_demand_kW": float(P_IDC_demand_kW),
            "P_IDC_served_kW": float(P_IDC_served_kW),
            "P_base_demand_kW": float(P_base_demand_kW),
            "P_task_incremental_demand_kW": float(P_task_incremental_demand_kW),
            "P_base_served_kW": float(P_base_served_kW),
            "P_task_served_kW": float(P_task_served_kW),
            "P_local_demand_kW": float(P_local_demand_kW),
            "P_local_net_before_pv_kW": float(P_local_net_before_pv_kW),
            "P_bus_net_kW": float(P_bus_net_kW),
            "P_grid_kW": float(P_grid_kW),
            "access_limit_kw": float(self.access_limit_kw),
            "normalization_refs": self.normalization_refs(),
            "forecast_clipping": self.forecast_clipping(t),
            "unserved_base_load_kW": float(unserved_base_load_kW),
            "unserved_task_power_kW": float(unserved_task_power_kW),
            "access_curtailment_work": float(access_curtailment_work),
            "grid_power_kW": float(grid_power_kW),
            "grid_power_limit_kW": float(self.grid_power_limit_kW),
            **self._server_group_info(),
            **self._task_scale_info(),
            "bess_soc": float(self.bess_soc),
            "bess_energy_kWh": float(self.bess_energy_kWh),
            "bess_mode": bess_mode,
            "desired_bess_charge_power_kW": float(desired_bess_charge_power_kW),
            "desired_bess_discharge_power_kW": float(desired_bess_discharge_power_kW),
            "bess_charge_power_kW": float(bess_charge_power_kW),
            "bess_discharge_power_kW": float(bess_discharge_power_kW),
            "bess_available_charge_kWh": float(max_charge_energy_by_soc),
            "bess_available_discharge_kWh": float(max_discharge_energy_by_soc),
            "bess_charge_kWh": float(bess_charge_kWh),
            "bess_discharge_kWh": float(bess_discharge_kWh),
            "bess_cycle_throughput_kWh": float(bess_throughput_kWh),
            "bess_degradation_cost": float(bess_degradation_cost),
            **self._bess_static_info(),
            "invalid_bess_action": float(invalid_bess_action),
            "soc_deviation": float(soc_deviation),
            "soc_excess": float(soc_excess),
            "total_bess_charge_kWh": float(self.total_bess_charge_kWh),
            "total_bess_discharge_kWh": float(self.total_bess_discharge_kWh),
            "total_bess_degradation_cost": float(self.total_bess_degradation_cost),
            "P_IT": P_IT_t,
            "PUE": PUE_t,
            "COP": COP_t,
            "P_cooling": P_cooling_t,
            "grid_peak_power_kW": float(grid_peak_power_kW),
            "grid_peak_excess_kW": float(grid_peak_excess_kW),
            "episode_grid_peak_power_kW": float(self.episode_grid_peak_power_kW),
            "total_grid_peak_excess_kW_hour": float(self.total_grid_peak_excess_kW_hour),
            "idc_peak_power_kW": float(P_IDC_kW),
            # peak_* fields are legacy aliases for grid-side peak metrics after BESS.
            "peak_power_kW": float(peak_power_kW),
            "peak_power_threshold_kW": float(self.grid_power_limit_kW),
            "peak_excess_kW": float(peak_excess_kW),
            "episode_peak_power_kW": float(self.episode_peak_power_kW),
            "total_peak_excess_kW_hour": float(self.total_peak_excess_kW_hour),
            "energy_kWh": energy_kWh,
            "grid_energy_kWh": float(grid_energy_kWh),
            "idc_energy_kWh": float(idc_energy_kWh),
            "pv_available_kWh": float(pv_available_kWh),
            "pv_used_kWh": float(pv_used_kWh),
            "pv_curtail_kWh": float(pv_curtail_kWh),
            "cost": cost_t,
            "hourly_cost": cost_t,
            "carbon_emission": carbon_emission_t,
            "carbon_cost": float(carbon_cost_t),

            "total_energy_kWh": self.total_energy_kWh,
            "total_grid_energy_kWh": self.total_grid_energy_kWh,
            "total_idc_energy_kWh": self.total_idc_energy_kWh,
            "total_pv_available_kWh": float(self.total_pv_available_kWh),
            "total_pv_used_kWh": float(self.total_pv_used_kWh),
            "total_pv_curtail_kWh": float(self.total_pv_curtail_kWh),
            "total_wind_available_kWh": float(self.total_wind_available_kWh),
            "total_wind_used_kWh": float(self.total_wind_used_kWh),
            "total_wind_curtail_kWh": float(self.total_wind_curtail_kWh),
            "pv_utilization_rate": float(pv_utilization_rate),
            "renewable_share": float(renewable_share),
            "total_cost": self.total_cost,
            "total_carbon_emission": self.total_carbon_emission,
            "total_carbon_cost": self.total_carbon_cost,
            "total_completed_work": self.total_completed_work,
            "completion_rate": completion_rate,
            "unit_task_cost": unit_task_cost,
            "energy_per_task": energy_per_task,
            "idc_energy_per_task": idc_energy_per_task,
            "carbon_per_task": carbon_per_task,

            "electricity_cost": float(cost_t),
            "terminal_leftover_work": float(self.terminal_leftover_work),
            "terminal_deadline_miss_count": int(self.terminal_deadline_miss_count),
            "terminal_soc_recovery_kwh": float(self.terminal_soc_recovery_kwh),
            "terminal_service_violation": int(self.terminal_service_violation),
            "terminal_settlement_penalty": float(self.terminal_settlement_penalty),
            "total_objective_cost": float(self.total_objective_cost),
            "settlement": settlement,
            "task_classification": self._compute_task_classification(t),
            **task_metrics,
        }

        return obs, float(reward), terminated, truncated, info

    def _activate_arrivals(self, current_time: int) -> int:
        """激活当前小时到达的任务。"""
        count = 0
        for task in self.tasks:
            if task.arrival_time == current_time and task.status == "not_arrived":
                task.status = "waiting"
                count += 1
        return count

    def _initialize_task_runtime_state(self) -> None:
        """
        初始化每个 Task 的启停统计字段。

        这些字段不要求写入任务类定义中，环境层在每个 episode
        reset 后为任务对象动态添加，方便后续记录暂停、恢复和
        不可暂停任务中断。
        """
        for task in self.tasks:
            task.pause_count = 0
            task.resume_count = 0
            task.non_interruptible_interruption_count = 0
            task.last_executed_time = None
            task.is_paused = False

    def _task_selection_score(
        self,
        task,
        current_time: int,
        urgent_preference: float,
        continuity_preference: float,
    ) -> float:
        """
        根据 PPO 给出的任务偏好，为候选任务计算选择分数。

        urgent_preference 越高：
            越偏向 deadline 近、priority 高、已经超时的任务。
        continuity_preference 越高：
            越偏向继续执行已经启动但尚未完成的任务。

        当两个偏好都接近 0 时，所有任务得分接近 0，
        后续排序会退化为 FIFO。
        """
        urgent_preference = float(np.clip(urgent_preference, 0.0, 1.0))
        continuity_preference = float(np.clip(continuity_preference, 0.0, 1.0))

        deadline_left = float(task.latest_finish_time - current_time)
        # deadline 越近，deadline_score 越高；已超时任务给满分。
        if deadline_left <= 0:
            deadline_score = 1.0
            overdue_score = 1.0
        else:
            deadline_score = 1.0 - np.clip(deadline_left / max(self.horizon, 1), 0.0, 1.0)
            overdue_score = 0.0

        priority_score = np.clip(float(getattr(task, "priority", 0.0)) / 5.0, 0.0, 1.0)
        urgency_score = (
            0.55 * deadline_score
            + 0.35 * priority_score
            + 0.10 * overdue_score
        )

        # 已经启动但未完成的任务更需要连续执行。
        has_started = task.start_time is not None
        was_executed_before = getattr(task, "last_executed_time", None) is not None
        is_paused = bool(getattr(task, "is_paused", False))
        continuity_score = 1.0 if (has_started or was_executed_before or is_paused) else 0.0

        return float(
            urgent_preference * urgency_score
            + continuity_preference * continuity_score
        )

    def _execute_tasks_action_guided(
        self,
        planned_capacity_vec: np.ndarray,
        unconstrained_capacity_vec: np.ndarray,
        current_time: int,
        urgent_preference: float,
        continuity_preference: float,
    ):
        """
        根据动作中的任务偏好执行任务，并跟踪任务启停。

        启停机制：
        1. 任务第一次执行时，Task.execute 会自动记录 start_time；
        2. 已启动但未完成的任务，如果本小时没有继续执行，则记为 pause；
        3. 已暂停任务再次执行，则记为 resume；
        4. interruptible=False 的任务如果被暂停，记录不可暂停任务中断。

        任务选择：
        - 不再是纯 FIFO；
        - urgent_preference 控制对紧急/高优先级任务的偏向；
        - continuity_preference 控制对已启动未完成任务的连续执行偏向；
        - 当两个偏好都很低时，排序退化为 FIFO。
        """
        active_tasks = [
            task for task in self.tasks
            if task.status in ["waiting", "running", "paused"]
            and task.remaining_work > 1e-6
        ]

        # 显式 A[i,g] 分配（M3.2 接线）：任务摘要 + 每任务最大速率（workload/duration）
        summaries = [
            {
                "task_id": str(task.task_id),
                "remaining_work": float(task.remaining_work),
                "max_rate": float(task.workload / max(int(task.duration), 1)),
                "priority": float(task.priority),
                "deadline": int(task.latest_finish_time),
                "arrival": int(task.arrival_time),
            }
            for task in active_tasks
        ]
        feasible_allocation = allocate_tasks(
            summaries, [float(c) for c in np.asarray(planned_capacity_vec, dtype=np.float64)]
        )
        unconstrained_allocation = allocate_tasks(
            summaries, [float(c) for c in np.asarray(unconstrained_capacity_vec, dtype=np.float64)]
        )
        w_unconstrained = float(sum(sum(r) for r in unconstrained_allocation.matrix))
        w_feasible = float(sum(sum(r) for r in feasible_allocation.matrix))

        completed_by_group = np.zeros(self.model.N, dtype=np.float64)
        completed_this_hour = 0.0
        newly_finished_count = 0
        newly_finished_priority_sum = 0.0
        resume_count_this_step = 0
        executed_task_ids = set()

        for idx, task in enumerate(active_tasks):
            row = feasible_allocation.matrix[idx]
            task_work = float(sum(row))
            for g in range(self.model.N):
                completed_by_group[g] += row[g]
            if task_work <= 1e-9:
                continue

            before_status = task.status
            if bool(getattr(task, "is_paused", False)):
                task.resume_count = int(getattr(task, "resume_count", 0)) + 1
                self.total_resume_count += 1
                resume_count_this_step += 1
                task.is_paused = False

            actual_work = task.execute(work_amount=task_work, current_time=current_time)
            if actual_work > 1e-9:
                executed_task_ids.add(task.task_id)
                task.last_executed_time = int(current_time)
            completed_this_hour += actual_work

            if before_status != "finished" and task.status == "finished":
                newly_finished_count += 1
                newly_finished_priority_sum += float(getattr(task, "priority", 0.0))
                task.is_paused = False

        pause_count_this_step, non_interruptible_interruption_this_step = self._update_pause_events(
            current_time=current_time,
            executed_task_ids=executed_task_ids,
        )

        new_deadline_miss_count = self._update_deadline_miss(current_time=current_time)

        return (
            float(completed_this_hour),
            completed_by_group,
            float(w_unconstrained),
            float(w_feasible),
            int(newly_finished_count),
            float(newly_finished_priority_sum),
            int(new_deadline_miss_count),
            int(pause_count_this_step),
            int(resume_count_this_step),
            int(non_interruptible_interruption_this_step),
        )

    def _update_pause_events(self, current_time: int, executed_task_ids: set[int]):
        """
        根据本小时实际执行任务集合，更新暂停/不可暂停中断统计。

        判断逻辑：
        - 任务已经启动过 start_time is not None；
        - 任务尚未完成；
        - 本小时没有执行；
        - 任务当前还没有处于 paused 状态。

        满足以上条件时，认为该任务在本小时被暂停。若任务不可暂停
        interruptible=False，则额外记为不可暂停任务中断。
        """
        pause_count_this_step = 0
        non_interruptible_interruption_this_step = 0

        for task in self.tasks:
            if task.status in ["not_arrived", "finished", "failed"]:
                continue
            if task.remaining_work <= 1e-6:
                continue
            if task.start_time is None:
                continue
            if task.task_id in executed_task_ids:
                continue
            if bool(getattr(task, "is_paused", False)):
                continue

            task.pause_count = int(getattr(task, "pause_count", 0)) + 1
            task.is_paused = True
            task.status = "paused"
            self.total_pause_count += 1
            pause_count_this_step += 1

            if not bool(task.interruptible):
                task.non_interruptible_interruption_count = (
                    int(getattr(task, "non_interruptible_interruption_count", 0)) + 1
                )
                self.total_non_interruptible_interruption_count += 1
                non_interruptible_interruption_this_step += 1

        return int(pause_count_this_step), int(non_interruptible_interruption_this_step)

    def _update_deadline_miss(self, current_time: int) -> int:
        """
        记录新发生的 deadline miss（M3.5a：与分类器边界严格一致）。

        逾期判据（与 `_compute_task_classification` 相同）：
        - 未完成任务：`current_time + 1 > latest_finish_time`；
        - 已完成任务：`finish_time > latest_finish_time`（逾期完成）。

        本方法在任务执行**之后**调用，故必须显式覆盖「在期限跨越那一步正好完成」的任务，
        否则会漏记（该任务此刻已为 finished）。这里不会把任务标记为 failed；
        当前阶段只在 reward 和 info 中记录超时压力。
        """
        new_count = 0
        for task in self.tasks:
            if task.status in ["not_arrived", "failed"]:
                continue
            if task.task_id in self.deadline_miss_task_ids:
                continue
            if task.status == "finished":
                finish_time = (
                    int(task.finish_time) if task.finish_time is not None else int(current_time + 1)
                )
                is_miss = finish_time > int(task.latest_finish_time)
            else:
                is_miss = current_time + 1 > task.latest_finish_time
            if is_miss:
                self.deadline_miss_task_ids.add(task.task_id)
                new_count += 1
        return new_count

    def _loads_from_group_completion(self, completed_work_by_group: np.ndarray) -> np.ndarray:
        """每组实际负载 = 完成工作 / 组能力（M3.3 废除比例回分与 α）。"""
        c_server = np.asarray(self.model.C_server, dtype=np.float64)
        loads = np.asarray(completed_work_by_group, dtype=np.float64) / np.maximum(c_server, 1e-6)
        return np.clip(loads, 0.0, self.max_task_load_per_server)

    def _idc_power_kw(self, load_vector, T_amb) -> float:
        """把负载（标量广播到 N 组，或 N 组向量）映射为 IDC 总功率（kW）。"""
        load = np.asarray(load_vector, dtype=np.float64)
        if load.ndim == 0:
            load = np.full(self.model.N, float(load))
        L = load.reshape(1, -1)
        P, *_ = self.model.calc_pue_and_total_power(L, np.array([T_amb], dtype=np.float64))
        return float(P[0]) / 1000.0

    def _apply_terminal_settlement(self) -> dict:
        """终止结算账务（M3.6 重做）：计算并一次性写入结算字段，只执行一次。

        reward 结算项（r_final_queue / r_soc_final，用于 RL 奖励）、会计结算项
        （terminal_settlement_penalty 等，用于 episode 汇总与客观目标）、违规记录
        （terminal_service_violation，标志）三者关系：
        - reward 项 = -reward_*_weight × 归一化违规（无量纲，仅进 reward）；
        - 会计项 = 未加权的归一化违规之和（无量纲，进 total_objective_cost）；
        - 违规记录 = 是否发生任一违规的标志（供评估/报告）。
        三者互不混用；不把 reward 权重当货币系数。
        """
        if self._settlement_done:
            return {
                "leftover_work": self.terminal_leftover_work,
                "deadline_miss_total": self.terminal_deadline_miss_count,
                "soc_recovery_energy_kwh": self.terminal_soc_recovery_kwh,
                "service_violation": self.terminal_service_violation,
                "settlement_penalty": self.terminal_settlement_penalty,
            }

        leftover = float(self._compute_backlog_work())
        deadline_miss = int(len(self.deadline_miss_task_ids))
        soc_deviation = abs(self.bess_soc - self.bess_soc_target)
        soc_excess = max(soc_deviation - self.bess_soc_final_tolerance, 0.0)
        soc_recovery_kwh = soc_excess * self.bess_capacity_kWh

        leftover_norm = leftover / max(self.queue_ref, 1e-6)
        deadline_miss_norm = deadline_miss / max(len(self.tasks), 1)
        settlement_penalty = leftover_norm + deadline_miss_norm + soc_excess
        service_violation = 1 if (leftover > 1e-9 or deadline_miss > 0 or soc_recovery_kwh > 1e-9) else 0

        self.terminal_leftover_work = leftover
        self.terminal_deadline_miss_count = deadline_miss
        self.terminal_soc_recovery_kwh = soc_recovery_kwh
        self.terminal_service_violation = service_violation
        self.terminal_settlement_penalty = settlement_penalty
        self.total_objective_cost = (
            self.total_cost + self.total_bess_degradation_cost + settlement_penalty
        )
        self._settlement_done = True

        return {
            "leftover_work": leftover,
            "deadline_miss_total": deadline_miss,
            "soc_recovery_energy_kwh": soc_recovery_kwh,
            "service_violation": service_violation,
            "settlement_penalty": settlement_penalty,
        }

    def _compute_task_classification(self, current_time: int) -> dict:
        """M3.8 任务四分类：未到期积压 / 逾期积压 / 逾期完成 / 按时完成（互斥完备）。"""
        counts = {
            "not_due_backlog": 0,
            "overdue_backlog": 0,
            "overdue_completed": 0,
            "on_time_completed": 0,
        }
        for task in self.tasks:
            if task.status in ("not_arrived", "failed"):
                continue
            if task.status == "finished" or task.remaining_work <= 1e-6:
                finish = int(task.finish_time) if task.finish_time is not None else int(current_time)
                if finish > int(task.latest_finish_time):
                    counts["overdue_completed"] += 1
                else:
                    counts["on_time_completed"] += 1
            elif current_time + 1 > int(task.latest_finish_time):
                counts["overdue_backlog"] += 1
            else:
                counts["not_due_backlog"] += 1
        return counts

    def state_dict(self) -> dict:
        """M3.6 环境状态快照（任务、SOC、累计、RNG），用于中断恢复等价。"""
        import copy

        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state: dict) -> None:
        import copy

        self.__dict__.clear()
        self.__dict__.update(copy.deepcopy(state))

    def _compute_backlog_work(self) -> float:
        """由 Task 列表统计当前已到达但未完成任务的剩余工作量。"""
        return float(sum(
            task.remaining_work
            for task in self.tasks
            if task.status != "finished" and task.status != "not_arrived"
        ))

    def _total_available_work(self) -> float:
        """统计本 episode 内所有会到达任务的总工作量，包括初始积压任务。"""
        return float(sum(
            task.workload
            for task in self.tasks
            if task.arrival_time < self.horizon
        ))

    def _compute_reward_task_pressure(self, current_time: int) -> tuple[float, float]:
        """
        计算 reward 中使用的任务压力指标。

        urgent_backlog_work：快到 deadline 或已经超时的未完成任务量；
        avg_waiting_pressure：已到达未完成任务的平均等待/滞留时间。
        """
        urgent_window = 3
        unfinished_tasks = [
            task for task in self.tasks
            if task.status not in ["not_arrived", "finished", "failed"]
            and task.remaining_work > 1e-6
        ]

        urgent_backlog_work = 0.0
        waiting_pressure_list = []

        for task in unfinished_tasks:
            deadline_left = task.latest_finish_time - current_time
            priority = float(getattr(task, "priority", 1.0))
            # Urgent backlog means work whose latest finish time is close/overdue or priority is high.
            if deadline_left <= urgent_window or priority >= 4.0:
                urgent_backlog_work += float(task.remaining_work)

            # 对已经到达但尚未完成的任务，记录其滞留时间。
            # 尚未启动的任务等待时间 = 当前时间 - 到达时间；
            # 已启动但未完成的任务也计入滞留压力，避免任务长期半完成。
            waiting_pressure_list.append(max(0, current_time - task.arrival_time))

        avg_waiting_pressure = (
            float(np.mean(waiting_pressure_list))
            if len(waiting_pressure_list) > 0
            else 0.0
        )

        return float(urgent_backlog_work), float(avg_waiting_pressure)

    def _compute_sla_metrics(self, current_time: int) -> dict:
        """
        Compute simple SLA violation pressure for unfinished overdue tasks.

        Penalty unit is priority-weighted delay hours:
        sla_penalty = sum(priority_i * delay_hours_i).
        """
        arrived_tasks = [
            task for task in self.tasks
            if task.arrival_time < self.horizon
        ]
        active_tasks = [
            task for task in arrived_tasks
            if task.status not in ["not_arrived", "finished", "failed"]
            and task.remaining_work > 1e-6
        ]

        delay_list = []
        weighted_delay_list = []

        for task in active_tasks:
            delay_hours = max(0.0, float(current_time - task.latest_finish_time))
            if delay_hours <= 0.0:
                continue

            priority = max(float(getattr(task, "priority", 1.0)), 0.0)
            delay_list.append(delay_hours)
            weighted_delay_list.append(priority * delay_hours)

        total_task_count = max(len(arrived_tasks), 1)
        sla_violation_count = len(delay_list)
        sla_penalty = float(np.sum(weighted_delay_list)) if weighted_delay_list else 0.0

        return {
            "sla_penalty": sla_penalty,
            "sla_violation_count": int(sla_violation_count),
            "sla_violation_rate": float(sla_violation_count / total_task_count),
            "avg_task_delay": (
                float(np.mean(delay_list))
                if len(delay_list) > 0
                else 0.0
            ),
            "max_task_delay": (
                float(np.max(delay_list))
                if len(delay_list) > 0
                else 0.0
            ),
        }

    def _compute_task_metrics(self) -> dict:
        """计算任务级统计指标。"""
        arrived_tasks = [
            task for task in self.tasks
            if task.arrival_time < self.horizon
        ]
        finished_tasks = [
            task for task in arrived_tasks
            if task.status == "finished"
        ]
        unfinished_tasks = [
            task for task in arrived_tasks
            if task.status != "finished"
        ]

        total_task_count = len(arrived_tasks)
        finished_task_count = len(finished_tasks)
        unfinished_task_count = len(unfinished_tasks)

        task_completion_rate = (
            finished_task_count / total_task_count
            if total_task_count > 0
            else 0.0
        )

        waiting_time_list = [
            task.start_time - task.arrival_time
            for task in finished_tasks + unfinished_tasks
            if task.start_time is not None
        ]
        turnaround_time_list = [
            task.finish_time - task.arrival_time
            for task in finished_tasks
            if task.finish_time is not None
        ]

        paused_task_count = len([
            task for task in arrived_tasks
            if task.status == "paused"
        ])

        return {
            "total_task_count": total_task_count,
            "finished_task_count": finished_task_count,
            "unfinished_task_count": unfinished_task_count,
            "paused_task_count": int(paused_task_count),
            "task_completion_rate": float(task_completion_rate),
            "deadline_miss_count": int(len(self.deadline_miss_task_ids)),
            "deadline_miss_rate": (
                len(self.deadline_miss_task_ids) / total_task_count
                if total_task_count > 0
                else 0.0
            ),
            "avg_waiting_time": (
                float(np.mean(waiting_time_list))
                if len(waiting_time_list) > 0
                else 0.0
            ),
            "avg_turnaround_time": (
                float(np.mean(turnaround_time_list))
                if len(turnaround_time_list) > 0
                else 0.0
            ),
            "final_backlog_work": float(self.Q_t),
            "total_pause_count": int(self.total_pause_count),
            "total_resume_count": int(self.total_resume_count),
            "total_non_interruptible_interruption_count": int(
                self.total_non_interruptible_interruption_count
            ),
        }

    def _get_task_pool_features(self, current_time: int) -> np.ndarray:
        """
        构造任务池状态特征，共 10 维。

        这些特征来自 Task 对象，体现任务类改进的价值：
        PPO 不再只能看到总队列 Q，而是能看到任务数量、紧急任务量、
        超时任务量、平均 deadline、优先级、可暂停/可并行任务压力。
        """
        arrived_tasks = [
            task for task in self.tasks
            if task.arrival_time <= current_time
        ]
        waiting_tasks = [
            task for task in arrived_tasks
            if task.status in ["waiting", "paused"]
        ]
        running_tasks = [
            task for task in arrived_tasks
            if task.status == "running"
        ]
        finished_tasks = [
            task for task in arrived_tasks
            if task.status == "finished"
        ]
        unfinished_tasks = [
            task for task in arrived_tasks
            if task.status not in ["finished", "failed", "not_arrived"]
            and task.remaining_work > 1e-6
        ]

        total_task_ref = max(len(self.tasks), 1)
        work_ref = max(self.queue_ref, 1e-6)
        priority_ref = 5.0
        urgent_window = 3

        waiting_task_count_norm = len(waiting_tasks) / total_task_ref
        running_task_count_norm = len(running_tasks) / total_task_ref
        finished_task_count_norm = len(finished_tasks) / total_task_ref
        unfinished_task_count_norm = len(unfinished_tasks) / total_task_ref

        urgent_work = 0.0
        overdue_work = 0.0
        deadline_left_list = []
        priority_list = []
        parallelizable_work = 0.0
        interruptible_work = 0.0

        for task in unfinished_tasks:
            deadline_left = task.latest_finish_time - current_time
            deadline_left_list.append(deadline_left)
            priority_list.append(float(task.priority))

            if 0 < deadline_left <= urgent_window:
                urgent_work += float(task.remaining_work)
            if deadline_left <= 0:
                overdue_work += float(task.remaining_work)
            if bool(task.parallelizable):
                parallelizable_work += float(task.remaining_work)
            if bool(task.interruptible):
                interruptible_work += float(task.remaining_work)

        avg_deadline_left_norm = (
            np.mean(np.clip(deadline_left_list, 0, self.horizon)) / max(self.horizon, 1)
            if len(deadline_left_list) > 0
            else 0.0
        )
        avg_priority_norm = (
            np.mean(priority_list) / priority_ref
            if len(priority_list) > 0
            else 0.0
        )

        features = np.array([
            waiting_task_count_norm,
            running_task_count_norm,
            finished_task_count_norm,
            unfinished_task_count_norm,
            urgent_work / work_ref,
            overdue_work / work_ref,
            avg_deadline_left_norm,
            avg_priority_norm,
            parallelizable_work / work_ref,
            interruptible_work / work_ref,
        ], dtype=np.float32)

        return np.clip(features, 0.0, 1.5).astype(np.float32)

    def _get_server_features(self, current_time: int) -> np.ndarray:
        """
        构造服务器状态特征，共 6 * N 维。

        每台服务器包含 6 组信息：
        1. 上一小时实际总负载 prev_loads；
        2. 单台服务器算力容量 C_server；
        3. 服务器能效 C_server / P_max；
        4. 当前电价下的单位算力成本；
        5. 服务器剩余可用负载容量；
        6. 基于环境温度和负载估算的服务器温度。
        """
        eps = 1e-6
        price_now = float(self.price_t[current_time])
        prev_loads = np.clip(self.prev_loads.astype(np.float64), 0.0, 1.0)

        server_capacity = np.asarray(self.model.C_server, dtype=np.float64)
        server_capacity_norm = server_capacity / max(float(np.max(server_capacity)), eps)

        if hasattr(self.model, "server_compute_efficiency"):
            efficiency = np.asarray(self.model.server_compute_efficiency, dtype=np.float64)
        else:
            efficiency = server_capacity / np.maximum(np.asarray(self.model.P_max, dtype=np.float64), eps)
        server_efficiency_norm = efficiency / max(float(np.max(efficiency)), eps)

        # 单位算力成本：电价 × 任务相关功耗增量 / 服务器算力。
        # 这个特征能让 PPO 感知“哪台服务器在当前电价下更便宜”。
        incremental_power = np.maximum(
            np.asarray(self.model.P_max, dtype=np.float64) - np.asarray(self.model.P_idle, dtype=np.float64),
            eps,
        )
        server_unit_cost = price_now * incremental_power / np.maximum(server_capacity, eps)
        server_unit_cost_norm = server_unit_cost / max(float(np.max(server_unit_cost)), eps)

        server_available_norm = np.clip(1.0 - prev_loads, 0.0, 1.0)

        # 简化估算服务器温度：环境温度 + 负载造成的热量增量。
        # 当前不是热管理模型，只是为状态空间提供服务器局部状态特征。
        estimated_server_temp = self.T_amb[current_time] + 15.0 * prev_loads
        server_temp_norm = np.clip(estimated_server_temp / 80.0, 0.0, 1.0)

        features = np.concatenate([
            prev_loads.astype(np.float32),
            server_capacity_norm.astype(np.float32),
            server_efficiency_norm.astype(np.float32),
            server_unit_cost_norm.astype(np.float32),
            server_available_norm.astype(np.float32),
            server_temp_norm.astype(np.float32),
        ])

        return np.clip(features, 0.0, 1.5).astype(np.float32)

    def normalization_refs(self) -> dict:
        """当前运行环境实际使用的归一化参考值（可审计，M3.10c）。"""
        return {
            "price_ref": float(self.price_ref),
            "lambda_ref": float(self.lambda_ref),
            "pv_ref_kw": float(self.pv_ref_kw),
            "wind_ref_kw": float(self.wind_ref_kw),
            "carbon_factor_ref": float(self.carbon_factor_ref),
            "source": "declared_frozen",
        }

    def forecast_clipping(self, t: int | None = None) -> dict:
        """可见窗口内被裁剪的特征计数（M3.10c）：输入超参考值只裁剪，不改参考值。"""
        step = int(self.current_step if t is None else t)
        start, end = visible_window_slice(step, self.forecast_cutoff, self.horizon)
        eps = 1e-6

        def _count(series, ref) -> int:
            vals = np.asarray(series, dtype=np.float64)[start:end] / max(float(ref), eps)
            return int(np.sum(np.abs(vals) > 1.5))

        return {
            "price": _count(self.price_t, self.price_ref),
            "temperature": _count(self.T_amb, 40.0),
            "arrival": _count(self.task_arrival_forecast, self.lambda_ref),
            "pv": _count(self.pv_t, self.pv_ref_kw),
            "wind": _count(self.wt_t, self.wind_ref_kw),
            "carbon": _count(self.carbon_factor_t, self.carbon_factor_ref),
        }

    def _get_forecast_features(self) -> np.ndarray:
        """
        构造覆盖整个 horizon 的固定前瞻特征，共 8 * horizon 维。

        固定顺序（M3.10b/M3.10c）：
        1. 分时电价 price_t / price_ref；
        2. 环境温度 T_amb / 40；
        3. 任务到达量 forecast（与环境内部真实到达曲线隔离）/ lambda_ref；
        4. 光伏可用出力 pv_t / pv_ref_kw；
        5. 风电可用出力 wt_t / wind_ref_kw；
        6. 碳强度 carbon_factor_t / carbon_factor_ref；
        7. 小时时间编码 time_sin；
        8. 小时时间编码 time_cos。

        可见性：仅暴露 [t, t + forecast_cutoff)（M3.10a 统一定义），其余位置置零。
        归一化参考值只取声明/冻结尺度（M3.10c），超界只做裁剪，见 `forecast_clipping()`。

        这些变量属于已知/可预测的外部条件，不包含未来队列、未来任务完成状态、
        未来服务器真实负载等由 PPO 动作决定的结果，避免未来信息泄露。
        """
        eps = 1e-6
        hours = np.arange(self.horizon, dtype=np.float64)
        # 可见窗口掩码（M3.10a 统一定义）：仅暴露 [t, t + forecast_cutoff)。
        t = int(self.current_step)
        start, end = visible_window_slice(t, self.forecast_cutoff, self.horizon)
        visible = np.zeros(self.horizon, dtype=np.float64)
        visible[start:end] = 1.0

        # M1.3g-e-c-R1：formal 路径的观测通道只读 **causal forecast**；
        # legacy 路径行为不变（其 forecast 数组就是原数组的副本）。
        price_24h = np.asarray(self.price_forecast_t, dtype=np.float64) / max(self.price_ref, eps) * visible
        T_amb_24h = np.asarray(self.temperature_forecast_t, dtype=np.float64) / 40.0 * visible
        lambda_24h = np.asarray(
            self.task_arrival_forecast, dtype=np.float64
        ) / max(self.lambda_ref, eps) * visible
        pv_24h = np.asarray(self.pv_forecast_t, dtype=np.float64) / max(self.pv_ref_kw, eps) * visible
        wind_24h = np.asarray(self.wind_forecast_t, dtype=np.float64) / max(self.wind_ref_kw, eps) * visible
        carbon_24h = (
            np.asarray(self.carbon_factor_t, dtype=np.float64)
            / max(self.carbon_factor_ref, eps)
            * visible
        )
        time_sin_24h = np.sin(2 * np.pi * hours / max(self.horizon, 1))
        time_cos_24h = np.cos(2 * np.pi * hours / max(self.horizon, 1))

        # 固定顺序（M3.10b）：price, temperature, arrival, pv, wind, carbon, sin, cos
        forecast_features = np.concatenate([
            price_24h,
            T_amb_24h,
            lambda_24h,
            pv_24h,
            wind_24h,
            carbon_24h,
            time_sin_24h,
            time_cos_24h,
        ]).astype(np.float32)

        if forecast_features.shape[0] != self.forecast_obs_dim:
            raise RuntimeError(
                f"前瞻状态维度错误：期望 {self.forecast_obs_dim}，实际 {forecast_features.shape[0]}。"
            )

        # price/T/lambda/PV 归一化后理论上多为正值；sin/cos 在 [-1, 1]。
        # 这里做温和裁剪，避免偶发极端任务到达量造成输入过大。
        return np.clip(forecast_features, -1.5, 1.5).astype(np.float32)

    def _get_obs(self):
        """构造底层状态向量：6 + 10 + 6*N + 8*horizon 维。

        默认 N=20、horizon=24 时为 328 维：136 维当前特征和
        192 维前瞻特征（8 组 × horizon）。GridCoupledEnv 的 8 维 grid
        observation 不属于本方法，由外层 wrapper 在此向量末尾追加。
        """
        t = self.current_step

        T_norm = self.T_amb[t] / 40.0
        price_norm = self.price_t[t] / self.price_ref
        # M1.3g-e-c：formal 路径的 arrival 通道只用**该时点可见的 causal forecast**，
        # 不读取 realized truth（legacy 路径保持原语义）。
        lambda_source = (
            self.task_arrival_forecast[t] if self.formal
            else self.true_task_arrival_profile[t]
        )
        lambda_norm = lambda_source / self.lambda_ref
        Q_norm = self.Q_t / self.queue_ref

        time_sin = np.sin(2 * np.pi * t / self.horizon)
        time_cos = np.cos(2 * np.pi * t / self.horizon)

        global_features = np.array(
            [T_norm, price_norm, lambda_norm, Q_norm, time_sin, time_cos],
            dtype=np.float32,
        )
        task_pool_features = self._get_task_pool_features(current_time=t)
        server_features = self._get_server_features(current_time=t)

        current_features = np.concatenate([
            global_features,
            task_pool_features,
            server_features,
        ]).astype(np.float32)

        if current_features.shape[0] != self.current_obs_dim:
            raise RuntimeError(
                f"当前状态维度错误：期望 {self.current_obs_dim}，实际 {current_features.shape[0]}。"
            )

        forecast_features = self._get_forecast_features()

        obs = np.concatenate([
            current_features,
            forecast_features,
        ]).astype(np.float32)

        if obs.shape[0] != self.obs_dim:
            raise RuntimeError(
                f"状态维度错误：期望 {self.obs_dim}，实际 {obs.shape[0]}。"
            )

        return obs
