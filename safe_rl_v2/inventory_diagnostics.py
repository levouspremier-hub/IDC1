"""Train-only inventory/reward measurements shared by diagnostics and training."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from evaluation.adapter import violations_in
from evaluation.inventory import evaluate_with_inventory
from evaluation.service_standard import FROZEN_PROJECT_SERVICE_STANDARD
from safe_rl.corrector_wrapper import CorrectorWrapper
from safe_rl_v2.formal_train_loop import build_train_env


class RecordingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.records: list[dict[str, Any]] = []

    def step(self, action):
        result = self.env.step(action)
        _, reward, _, _, info = result
        self.records.append({**info, "measured_reward": float(reward)})
        return result


def inventory_episode_acceptance(episode):
    """Unknown/missing evidence and avoidable losses never bypass terminal review."""
    final = episode.get("final_planning_audit", {})
    assessment = episode.get("terminal_gap_assessment", final.get("terminal_gap_assessment", {}))
    terminal_ok = episode.get("target_qualified") is True
    if terminal_ok and assessment.get("classification") == "planner_consistency_loss":
        terminal_ok = False
    if not terminal_ok:
        terminal_ok = (assessment.get("classification") == "registered_initial_gap"
                       and assessment.get("explanation_complete") is True
                       and final.get("target_reachable") is False)
    return {
        "complete_episode": episode.get("episode_complete") is True,
        "frozen_service": episode.get("service_qualified") is True,
        "physical_constraints": episode.get("physical_violation_count") == 0,
        "inventory_band": episode.get("inventory_qualified") is True,
        "no_fallback": episode.get("fallbacks") == 0,
        "all_reachability_proven": episode.get("inventory_unproven_steps") == 0,
        "terminal_gap_explained": terminal_ok,
    }


def validation_ready(rows, *, expected_episodes, pairing_verified):
    return (pairing_verified is True and len(rows) == expected_episodes
            and all(all(inventory_episode_acceptance(row).values()) for row in rows))


def evaluate_origin(config, origin, seed, action_fn, *, terminal=True, run_id="diagnostic"):
    env, injection = build_train_env(origin, master_seed=seed, config=config)
    env.terminal_inventory_enabled = terminal
    wrapped = RecordingWrapper(CorrectorWrapper(env, corrector_time_limit_s=float(
        config["training"]["corrector"]["time_limit_s"])))
    record, inventory = evaluate_with_inventory(
        wrapped, "safe_ppo_joint_rolling_corrector", action_fn, run_id=run_id,
        service_standard=FROZEN_PROJECT_SERVICE_STANDARD, seed=seed)
    rows = wrapped.records
    gamma = float(config["training"]["ppo"]["gamma_per_step"])
    raw = np.array([r["raw_action"][-1] for r in rows])
    delta = np.array([abs(r["raw_action"][-1] - r["exec_action"][-1]) for r in rows])
    actual_storage = np.array([
        r["bess_discharge_power_kW"] / env.bess_discharge_power_max_kW
        - r["bess_charge_power_kW"] / env.bess_charge_power_max_kW for r in rows])
    audit = [r.get("inventory_audit", {}) for r in rows]
    result = {
        "origin": int(origin), "seed": int(seed), "steps": record.steps,
        "policy_observation_dimension": int(wrapped.observation_space.shape[0]),
        "policy_observation_version": wrapped.env.inventory_observation_version,
        "episode_complete": inventory.episode_complete,
        "service_qualified": record.service_qualified is True,
        "on_time_task_rate": record.service.on_time_task_rate,
        "on_time_work_rate": record.service.on_time_work_rate,
        "physical_violation_count": len(violations_in(record.physical)),
        "final_soc": inventory.final_soc,
        "target_gap_kwh": abs(inventory.final_energy_kwh - inventory.target_energy_kwh),
        "band_gap_kwh": max(abs(inventory.final_energy_kwh - inventory.target_energy_kwh)
                            - env.bess_soc_final_tolerance * env.bess_capacity_kWh, 0.0),
        "inventory_qualified": inventory.final_soc_within_env_tolerance,
        "target_qualified": abs(inventory.final_energy_kwh - inventory.target_energy_kwh) <= 1e-6,
        "purchase_cost_sgd": record.purchase_cost_sgd,
        "degradation_cost_sgd": record.bess_degradation_cost_sgd,
        "reward_units_per_purchase_sgd": float(env.reward_cost_weight / env.cost_ref),
        "reward_units_per_degradation_sgd": rows[0]["reward_semantics_audit"][
            "degradation_reward_per_sgd"],
        "reward_semantics": rows[0]["reward_semantics_audit"]["version"],
        "original_env_reward_sum": sum(r["reward_semantics_audit"]["original_env_reward"]
                                       for r in rows),
        "original_degradation_reward_sum": sum(
            r["reward_semantics_audit"]["original_degradation_reward"] for r in rows),
        "original_load_smooth_sum": sum(
            r["reward_semantics_audit"].get("original_load_smooth", r["r_load_smooth"])
            for r in rows),
        "original_action_smooth_sum": sum(
            r["reward_semantics_audit"].get("original_action_smooth", r["r_action_smooth"])
            for r in rows),
        "carbon_kg": record.carbon_kg_co2e,
        "charge_kwh": float(env.total_bess_charge_kWh),
        "discharge_kwh": float(env.total_bess_discharge_kWh),
        "storage_raw_mean": float(raw.mean()) if len(raw) else None,
        "storage_raw_std": float(raw.std()) if len(raw) else None,
        "storage_saturation_fraction": float((abs(raw) > 0.98).mean()) if len(raw) else None,
        "storage_abs_delta_mean": float(delta.mean()) if len(delta) else None,
        "actual_storage_abs_delta_mean": float(np.abs(raw - actual_storage).mean()),
        "requested_exec_to_actual_storage_abs_delta_mean": float(np.abs(
            np.array([r["exec_action"][-1] for r in rows]) - actual_storage).mean()),
        "fallbacks": sum(r["correction_reason"] in
                         ("timeout", "solver_failure", "proposal_invalid", "base_shortage")
                         for r in rows),
        "timeouts": sum(r.get("correction_solver_timeout", False) for r in rows),
        "stage_a_retained_steps": sum(r.get("correction_execution_source") == "stage_a"
                                      for r in rows),
        "inventory_unproven_steps": sum(a.get("target_reachable") is None for a in audit)
        if terminal else None,
        "inventory_unreachable_steps": sum(a.get("target_reachable") is False for a in audit)
        if terminal else None,
        "failure": record.failure_classification,
        "final_planning_audit": audit[-1] if terminal and audit else {},
        "terminal_gap_assessment": audit[-1].get("terminal_gap_assessment", {})
        if terminal and audit else {},
        "injection_provenance": injection.provenance_hash,
    }
    reward_keys = sorted({key for r in rows for key in r if key.startswith("r_")})
    for key in ["measured_reward", *reward_keys]:
        result[key + "_sum"] = sum(float(r[key]) for r in rows)
        result[key + "_discounted"] = sum(gamma ** i * float(r[key])
                                           for i, r in enumerate(rows))
    result["solve_time_p95_s"] = float(np.quantile(
        [r["correction_solve_time_s"] for r in rows], .95)) if rows else None
    return result, rows, record, inventory
