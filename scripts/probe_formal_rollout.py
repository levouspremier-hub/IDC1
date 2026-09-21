"""M1.3g-f-c-a：formal 真实 rollout 闭环验证 probe（**只读**）。

用**已验证的正式链**跑一次短的真实数据 rollout，并输出**机器可读**证据：

```text
verified train v5 candidate origin → build_verified_formal_env_injection
  → IDCPriceEnv20D(formal_injection=…) → collect_rollout → RolloutBuffer
```

**不写任何 checkpoint、不更新任何参数、不宣称任何训练结果。**
本 probe 只证明「正式注入 → 真实采集」这条链可跑通且可重放。

用法：

```bash
uv run python -m scripts.probe_formal_rollout [--steps 6] [--horizon 8] [--seed 0]
```
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys

import numpy as np
import torch

SPLIT = "train"
# 已验证的 train v5 candidate origin（本地行 48）。
EPISODE_START = "2024-01-02T00:00:00+08:00"
DEFAULT_HORIZON = 8
FORECAST_CUTOFF = 4
DELTA_HOURS = 0.5
DEFAULT_STEPS = 6
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}


def _build_env(horizon: int):
    ei = importlib.import_module("scenario.env_injection")
    inj = ei.build_verified_formal_env_injection(
        SPLIT, start=EPISODE_START, horizon=horizon, forecast_cutoff=FORECAST_CUTOFF)
    env_cls = importlib.import_module("envs.idc_price_env").IDCPriceEnv20D
    env = env_cls(horizon=horizon, delta_t_hours=DELTA_HOURS,
                  formal_injection=inj, **ENV_SEED_KWARGS)
    return env, inj


def _policy(env, seed: int):
    from safe_rl_v2.policy import SafePPOPolicy

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return SafePPOPolicy(obs_dim=env.obs_dim)


def buffer_digest(buffer) -> str:
    """buffer 全部 transition 的**逐位**指纹（供重放比对）。"""
    h = hashlib.sha256()
    for t in buffer.transitions:
        for arr in (t.observation, t.next_observation, t.raw_action, t.exec_action):
            h.update(np.asarray(arr, dtype=np.float32).tobytes())
        for scalar in (t.old_raw_log_prob, t.reward, t.business_cost,
                       t.carbon_cost, t.electricity_cost_sgd):
            h.update(repr(float(scalar)).encode())
        h.update(b"1" if t.terminated else b"0")
        h.update(b"1" if t.truncated else b"0")
    return h.hexdigest()


def run_probe(*, steps: int = DEFAULT_STEPS, horizon: int = DEFAULT_HORIZON,
              seed: int = 0) -> dict:
    """跑一次 formal rollout 并返回**机器可读**证据（不产生任何训练结论）。"""
    from safe_rl_v2.buffer import RolloutBuffer
    from safe_rl_v2.rollout import collect_rollout

    env, inj = _build_env(horizon)
    policy = _policy(env, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    buffer = RolloutBuffer()
    stats = collect_rollout(env, policy, buffer, steps=steps, seed=seed,
                            generator=generator)

    rewards = [float(t.reward) for t in buffer.transitions]
    carbon = [float(t.carbon_cost) for t in buffer.transitions]
    now = int(env.current_step)

    return {
        "entry": "python -m scripts.probe_formal_rollout",
        "probe_only": True,
        "formal": bool(env.formal),
        "split": SPLIT,
        "episode_start": EPISODE_START,
        "horizon": int(horizon),
        "delta_t_hours": float(env.delta_t_hours),
        "steps_requested": int(steps),
        "steps_collected": int(stats["transitions"]),
        "buffer_len": len(buffer),
        "env_seed": int(stats["env_seed"]),
        "policy_rng_source": stats["policy_rng_source"],
        "contract_version": stats["contract_version"],
        "action_dim": int(stats["action_dim"]),
        "obs_dim": int(env.obs_dim),
        "ledger_micro_sum": int(sum(inj.ledger_micro)),
        "initial_backlog_work": float(env.initial_Q),
        "reward_sum": float(np.sum(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "carbon_emission_sum_kg": float(np.sum(carbon)),
        "old_raw_log_prob_first": float(buffer.transitions[0].old_raw_log_prob),
        "raw_equals_exec": all(
            np.array_equal(t.raw_action, t.exec_action) for t in buffer.transitions),
        "current_step_after": now,
        "buffer_digest": buffer_digest(buffer),
        "claims": {"trained": False, "performance_evaluated": False,
                   "convergence_claimed": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="formal 真实 rollout 闭环验证 probe（只读，不写 checkpoint）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        evidence = run_probe(steps=args.steps, horizon=args.horizon, seed=args.seed)
    except Exception as error:  # noqa: BLE001 - probe 失败必须原样透出
        print(f"probe_formal_rollout: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    print(json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
