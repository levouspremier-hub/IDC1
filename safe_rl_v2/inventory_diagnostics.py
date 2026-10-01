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


def evaluate_origin(config, origin, seed, action_fn, *, terminal=True, run_id="diagnostic"):
    env, injection = build_train_env(origin, master_seed=seed, config=config)
    env.terminal_inventory_enabled = terminal
    wrapped = RecordingWrapper(CorrectorWrapper(env, corrector_time_limit_s=0.25))
    record, inventory = evaluate_with_inventory(
        wrapped, "safe_ppo_joint_rolling_corrector", action_fn, run_id=run_id,
        service_standard=FROZEN_PROJECT_SERVICE_STANDARD, seed=seed)
    rows = wrapped.records
    gamma = float(config["training"]["ppo"]["gamma_per_step"])
    raw = np.array([r["raw_action"][-1] for r in rows])
    delta = np.array([abs(r["raw_action"][-1] - r["exec_action"][-1]) for r in rows])
    audit = [r.get("inventory_audit", {}) for r in rows]
    result = {
        "origin": int(origin), "seed": int(seed), "steps": record.steps,
        "episode_complete": inventory.episode_complete,
        "service_qualified": record.service_qualified is True,
        "on_time_task_rate": record.service.on_time_task_rate,
        "on_time_work_rate": record.service.on_time_work_rate,
        "physical_violation_count": len(violations_in(record.physical)),
        "final_soc": inventory.final_soc,
        "target_gap_kwh": abs(inventory.final_energy_kwh - inventory.target_energy_kwh),
        "inventory_qualified": inventory.final_soc_within_env_tolerance,
        "target_qualified": abs(inventory.final_energy_kwh - inventory.target_energy_kwh) <= 1e-6,
        "purchase_cost_sgd": record.purchase_cost_sgd,
        "degradation_cost_sgd": record.bess_degradation_cost_sgd,
        "carbon_kg": record.carbon_kg_co2e,
        "charge_kwh": float(env.total_bess_charge_kWh),
        "discharge_kwh": float(env.total_bess_discharge_kWh),
        "storage_raw_mean": float(raw.mean()) if len(raw) else None,
        "storage_raw_std": float(raw.std()) if len(raw) else None,
        "storage_saturation_fraction": float((abs(raw) > 0.98).mean()) if len(raw) else None,
        "storage_abs_delta_mean": float(delta.mean()) if len(delta) else None,
        "fallbacks": sum(r["correction_reason"] in
                         ("timeout", "solver_failure", "proposal_invalid", "base_shortage")
                         for r in rows),
        "timeouts": sum(r["correction_reason"] == "timeout" for r in rows),
        "inventory_unproven_steps": sum(a.get("target_reachable") is None for a in audit)
        if terminal else None,
        "inventory_unreachable_steps": sum(a.get("target_reachable") is False for a in audit)
        if terminal else None,
        "failure": record.failure_classification,
        "injection_provenance": injection.provenance_hash,
    }
    for key in ("measured_reward", "r_cost", "r_soc_final", "r_bess_degradation", "r_carbon"):
        result[key + "_sum"] = sum(float(r[key]) for r in rows)
        result[key + "_discounted"] = sum(gamma ** i * float(r[key])
                                           for i, r in enumerate(rows))
    result["solve_time_p95_s"] = float(np.quantile(
        [r["correction_solve_time_s"] for r in rows], .95)) if rows else None
    return result, rows, record, inventory
