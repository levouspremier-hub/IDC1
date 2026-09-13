#!/usr/bin/env python
"""M5.1b 确定性 rollout probe（**不训练**）。

用固定种子分别以 corrector 关闭 / 开启两种方式采集真实 rollout，报告：
transition 数、raw/exec 差异数、terminated/truncated 数、contract 版本。

**本 probe 不做 PPO 更新**：不实现也不调用 ratio/clip、GAE、advantage、actor loss、
熵项或拉格朗日乘子更新，不调用 `optimizer.step()`，不改变任何策略参数。
因此本 probe 的数值**只说明采集链可跑通**，
**不构成任何训练结果、收敛证据或性能结论**。

确定性说明：`IDCPriceEnv20D` 在 server/task/forecast 三类种子任一为 None 时使用
`default_rng(None)` 熵源，跨进程不可复现；本 probe 显式给出三者（**不修改 env**）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许以 `python scripts/probe_rollout_deterministic.py` 直接运行（补仓库根到 sys.path）
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from envs.idc_price_env import IDCPriceEnv20D
from runs.writer import write_run
from safe_rl_v2.buffer import UNIT_METADATA, RolloutBuffer
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

DEFAULT_SEED = 0
DEFAULT_STEPS = 8
DEFAULT_CORRECTOR_TIME_LIMIT_S = 0.05
# 环境三类种子必须同时显式给定，否则 default_rng(None) 使用熵源
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}


def build_env(**over) -> IDCPriceEnv20D:
    kwargs = dict(ENV_SEED_KWARGS)
    kwargs.update(over)
    return IDCPriceEnv20D(**kwargs)


def run_arm(*, corrector_on: bool, steps: int, seed: int, corrector_time_limit_s: float) -> dict:
    """单条采集臂；返回可 JSON 序列化的统计（不含任何训练量）。"""
    torch.manual_seed(seed)
    env = build_env()
    policy = SafePPOPolicy(obs_dim=env.obs_dim)
    policy.eval()  # 推理模式；本 probe 不训练

    before = [p.detach().clone() for p in policy.parameters()]
    buffer = RolloutBuffer()
    stats = collect_rollout(
        env,
        policy,
        buffer,
        steps=steps,
        seed=seed,
        corrector_on=corrector_on,
        corrector_time_limit_s=corrector_time_limit_s if corrector_on else None,
    )
    unchanged = all(
        torch.equal(old, new.detach()) for old, new in zip(before, policy.parameters(), strict=True)
    )

    payload = dict(stats)
    payload["arm"] = "corrector_on" if corrector_on else "corrector_off"
    payload["policy_parameters_unchanged"] = bool(unchanged)
    payload["old_raw_log_prob_finite"] = bool(
        np.all(np.isfinite([t.old_raw_log_prob for t in buffer.transitions]))
    )
    payload["buffer_to_dict_json_ok"] = bool(json.dumps(buffer.to_dict()))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性 rollout probe（不训练）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corrector-time-limit-s", type=float,
                        default=DEFAULT_CORRECTOR_TIME_LIMIT_S)
    parser.add_argument("--no-write-run", action="store_true", help="只打印，不写 runs/<id>/")
    args = parser.parse_args()

    arms = [
        run_arm(
            corrector_on=corrector_on,
            steps=args.steps,
            seed=args.seed,
            corrector_time_limit_s=args.corrector_time_limit_s,
        )
        for corrector_on in (False, True)
    ]

    # 确定性自检：同一配置重复运行必须逐元素一致
    repeat = run_arm(
        corrector_on=False,
        steps=args.steps,
        seed=args.seed,
        corrector_time_limit_s=args.corrector_time_limit_s,
    )
    deterministic = (
        repeat["transitions"] == arms[0]["transitions"]
        and repeat["raw_exec_difference_count"] == arms[0]["raw_exec_difference_count"]
    )

    report = {
        "probe": "m51b_deterministic_rollout",
        "trained": False,
        "note": (
            "本 probe 不训练：无 PPO ratio/clip、无 GAE/advantage、无 actor loss、"
            "无乘子更新、无 optimizer.step()；数值不代表训练或性能结论。"
        ),
        "steps_requested": args.steps,
        "seed": args.seed,
        "corrector_time_limit_s": args.corrector_time_limit_s,
        "deterministic_repeat_match": bool(deterministic),
        "arms": arms,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if not args.no_write_run:
        metrics = pd.DataFrame(
            [
                {
                    "arm": a["arm"],
                    "transitions": a["transitions"],
                    "raw_exec_difference_count": a["raw_exec_difference_count"],
                    "terminated_count": a["terminated_count"],
                    "truncated_count": a["truncated_count"],
                    "contract_version": a["contract_version"],
                    "corrector_on": a["corrector_on"],
                }
                for a in arms
            ]
        )
        run_id = f"m51b_rollout_probe_s{args.seed}_n{args.steps}"
        run_dir = write_run(
            run_id,
            config={
                "probe": "m51b_deterministic_rollout",
                "trained": False,
                "steps": args.steps,
                "seed": args.seed,
                "corrector_time_limit_s": args.corrector_time_limit_s,
                "env_seed_kwargs": ENV_SEED_KWARGS,
                "units": UNIT_METADATA,
            },
            metrics=metrics,
            report=report,
            seed=args.seed,
            command="uv run python scripts/probe_rollout_deterministic.py",
        )
        print(f"\nrun 产物：{run_dir}")


if __name__ == "__main__":
    main()
