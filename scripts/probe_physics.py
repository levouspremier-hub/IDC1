"""M3.0 physics probe: control diagnostics + timing on the CURRENT execution chain.

Read-only baseline (does not modify any code). It demonstrates N1 — per-group
compute allocation collapses to a scalar — and records the sampled reachable
range of completed work and power.

Run:
    uv run python scripts/probe_physics.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "envs").exists() and (candidate / "grid_model").exists():
            root = candidate
            break
    else:
        raise RuntimeError("Cannot locate project root containing envs/ and grid_model/")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


PROJECT_ROOT = _ensure_project_root_on_path()

import numpy as np

from configs.config_ultimate import (
    DATA_CONFIG,
    ENV_CONFIG,
    IDC_SCALE_CONFIG,
    REWARD_CONFIG,
)
from data_io.data_loader import build_external_series_from_config
from envs.idc_price_env import IDCPriceEnv20D


def make_env(seed: int = 2026) -> IDCPriceEnv20D:
    return IDCPriceEnv20D(
        **ENV_CONFIG,
        **REWARD_CONFIG,
        **IDC_SCALE_CONFIG,
        **build_external_series_from_config(DATA_CONFIG, ENV_CONFIG["horizon"]),
        server_seed=seed,
        task_seed=seed,
        forecast_seed=seed + ENV_CONFIG["task_forecast_seed_offset"],
    )


def full_action(compute: list[float], bess_action: float = 0.5) -> np.ndarray:
    """Assemble a 23-dim raw action from a 20-dim compute vector (urgent/continuity = 0)."""
    vec = np.zeros(23, dtype=np.float32)
    vec[: len(compute)] = compute
    vec[20] = 0.0  # urgent preference
    vec[21] = 0.0  # continuity preference
    vec[22] = bess_action
    return vec


def one_step(env: IDCPriceEnv20D, action: np.ndarray) -> dict:
    env.reset()
    _obs, _reward, _terminated, _truncated, info = env.step(action)
    return info


def main() -> None:
    env = make_env()
    env.reset()
    n_groups = int(env.model.N)
    print(f"action_dim={env.action_dim}  obs_dim={env.obs_dim}  N={n_groups}")
    print(
        f"max_task_load_per_server={env.max_task_load_per_server}  "
        f"planned_load_reserve_alpha={env.planned_load_reserve_alpha}"
    )

    rows = []
    rows.append(("empty_queue", one_step(env, full_action([0.0] * n_groups))))
    rows.append(("all_max", one_step(env, full_action([1.0] * n_groups))))
    info_uniform = one_step(env, full_action([0.5] * n_groups))
    info_split = one_step(env, full_action([1.0] * (n_groups // 2) + [0.0] * (n_groups - n_groups // 2)))
    rows.append(("uniform_0.5", info_uniform))
    rows.append(("split_10x1.0", info_split))
    rows.append(("uniform_0.25", one_step(env, full_action([0.25] * n_groups))))

    print(f"\n{'case':16s} {'completed_work':>16s} {'P_IDC_kW':>12s} {'cost':>10s}")
    for name, info in rows:
        print(f"{name:16s} {info['completed_work']:16.4f} {info['P_IDC_kW']:12.4f} {info['total_cost']:10.4f}")

    same = abs(info_uniform["completed_work"] - info_split["completed_work"]) < 1e-9 and abs(
        info_uniform["P_IDC_kW"] - info_split["P_IDC_kW"]
    ) < 1e-9
    print(
        f"\n[N1 evidence] uniform_0.5 vs split_10x1.0 identical? {same} "
        f"(completed {info_uniform['completed_work']:.4f} vs {info_split['completed_work']:.4f}; "
        f"P_IDC {info_uniform['P_IDC_kW']:.4f} vs {info_split['P_IDC_kW']:.4f})"
    )
    print(
        "=> group distribution enters ONLY via the C_server-weighted scalar sum (envs/idc_price_env.py:570); "
        "C_server is heterogeneous (server_capacity_variation=0.25), so the weighted total differs, but there is "
        "NO per-group task allocation and actual power is reverse-scaled from planned loads (M3.1-M3.3 target)."
    )

    # Timing.
    env.reset()
    action = full_action([0.5] * n_groups)
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        _obs, _r, terminated, _trunc, _info = env.step(action)
        if terminated:
            env.reset()
    elapsed = time.perf_counter() - t0
    print(f"\ntiming: {n} env.step in {elapsed:.3f}s -> {elapsed / n * 1000:.3f} ms/step")

    # SOC bounds across charge/discharge extremes.
    socs: list[float] = []
    env.reset()
    for _ in range(5):
        *_rest, i = env.step(full_action([0.0] * n_groups, bess_action=0.0))  # charge
        socs.append(float(i["bess_soc"]))
    for _ in range(5):
        *_rest, i = env.step(full_action([0.0] * n_groups, bess_action=1.0))  # discharge
        socs.append(float(i["bess_soc"]))
    print(
        f"SOC range over charge/discharge extremes: min={min(socs):.3f} max={max(socs):.3f} "
        f"(bounds [{env.bess_soc_min}, {env.bess_soc_max}])"
    )


if __name__ == "__main__":
    main()
