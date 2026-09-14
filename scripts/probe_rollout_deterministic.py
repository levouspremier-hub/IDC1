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
import copy
import hashlib
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
from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S, resolve_corrector_budget
from runs.writer import write_run
from safe_rl_v2.buffer import UNIT_METADATA, RolloutBuffer
from safe_rl_v2.policy import SafePPOPolicy
from safe_rl_v2.rollout import collect_rollout

DEFAULT_SEED = 0
DEFAULT_STEPS = 8
# corrector 生产默认预算**不在本文件定义**：唯一来源是
# `planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S`（M5.4i）。
# 环境三类种子必须同时显式给定，否则 default_rng(None) 使用熵源
ENV_SEED_KWARGS = {"task_seed": 0, "server_seed": 0, "forecast_seed": 300000}

# 纯墙钟计时审计字段：不参与任何决策或损失，跨进程不可复现，比较确定性时剔除
WALL_CLOCK_ONLY_KEYS = (
    "correction_solve_time_s",
    "stage_a_solve_time_s",
    "stage_b_solve_time_s",
)


def build_env(**over) -> IDCPriceEnv20D:
    kwargs = dict(ENV_SEED_KWARGS)
    kwargs.update(over)
    return IDCPriceEnv20D(**kwargs)


def payload_fingerprint(buffer: RolloutBuffer) -> str:
    """完整 payload 的 sha256：跨进程确定性证据，而不只是计数相同。"""
    canonical = json.dumps(buffer.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def decision_payload_fingerprint(buffer: RolloutBuffer) -> str:
    """**决策相关** payload 的 sha256：剔除纯墙钟计时字段后的指纹。

    `corrector_solve_time_s` / `stage_a_solve_time_s` / `stage_b_solve_time_s` 是
    墙钟测量值，天然跨进程不可复现；它们不参与任何决策或损失。
    剔除后再比对，才能区分「MILP 求解路径真的不同」与「只差计时读数」。
    """
    payload = copy.deepcopy(buffer.to_dict())
    for entry in payload["transitions"]:
        for key in WALL_CLOCK_ONLY_KEYS:
            entry["correction_info"].pop(key, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_arm(*, corrector_on: bool, steps: int, seed: int, corrector_time_limit_s: float) -> dict:
    """单条采集臂；返回可 JSON 序列化的统计（不含任何训练量）。

    策略权重用一个**独立** generator 初始化，采样再换另一个显式 generator，
    全程不依赖 `torch.manual_seed`（全局 RNG 状态属于调用方）。
    """
    env = build_env()
    # 权重初始化借用全局 RNG，但用 fork_rng 还原，不改变调用方状态
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy = SafePPOPolicy(obs_dim=env.obs_dim)
    policy.eval()  # 推理模式；本 probe 不训练

    sampler = torch.Generator()
    sampler.manual_seed(seed)

    before = [p.detach().clone() for p in policy.parameters()]
    global_before = torch.get_rng_state().clone()
    buffer = RolloutBuffer()
    stats = collect_rollout(
        env,
        policy,
        buffer,
        steps=steps,
        seed=seed,
        corrector_on=corrector_on,
        corrector_time_limit_s=corrector_time_limit_s if corrector_on else None,
        generator=sampler,
    )
    unchanged = all(
        torch.equal(old, new.detach()) for old, new in zip(before, policy.parameters(), strict=True)
    )

    payload = dict(stats)
    payload["arm"] = "corrector_on" if corrector_on else "corrector_off"
    payload["policy_weights_seed"] = seed
    payload["policy_sampler_generator_seed"] = seed
    payload["global_torch_rng_untouched"] = bool(
        torch.equal(global_before, torch.get_rng_state())
    )
    payload["policy_parameters_unchanged"] = bool(unchanged)
    payload["old_raw_log_prob_finite"] = bool(
        np.all(np.isfinite([t.old_raw_log_prob for t in buffer.transitions]))
    )
    payload["buffer_to_dict_json_ok"] = bool(json.dumps(buffer.to_dict()))
    payload["payload_sha256"] = payload_fingerprint(buffer)
    payload["decision_payload_sha256"] = decision_payload_fingerprint(buffer)
    payload["wall_clock_only_keys"] = list(WALL_CLOCK_ONLY_KEYS)
    payload["payload_field_count"] = len(buffer.to_dict()["transitions"][0]) if len(buffer) else 0
    return payload


def _command_ledger(args, run_id: str) -> str:
    """命令账本：必须可重放。

    未显式给出预算时**不得**伪造 `--corrector-time-limit-s`；显式给出时必须记录
    （否则重放会静默退回生产默认，账本与实测不符）。
    """
    argv = ["uv run python scripts/probe_rollout_deterministic.py",
            "--steps", str(args.steps), "--seed", str(args.seed),
            "--base-dir", str(args.base_dir), "--run-id", run_id]
    if args.corrector_time_limit_s is not None:
        argv += ["--corrector-time-limit-s", str(args.corrector_time_limit_s)]
    return " ".join(argv)


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性 rollout probe（不训练）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corrector-time-limit-s", type=float, default=None,
                        help="显式 override；省略则使用生产默认"
                             " planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S")
    parser.add_argument("--no-write-run", action="store_true", help="只打印，不写 runs/<id>/")
    parser.add_argument("--base-dir", default="runs", help="run 产物根目录")
    parser.add_argument("--run-id", default=None, help="显式 run id（默认沿用 m51c_ 命名）")
    args = parser.parse_args()

    effective_budget, budget_source = resolve_corrector_budget(args.corrector_time_limit_s)

    arms = [
        run_arm(
            corrector_on=corrector_on,
            steps=args.steps,
            seed=args.seed,
            corrector_time_limit_s=effective_budget,
        )
        for corrector_on in (False, True)
    ]

    # 确定性自检：同一配置重复运行，**完整 payload 指纹**必须一致（不只是计数）
    repeat = run_arm(
        corrector_on=False,
        steps=args.steps,
        seed=args.seed,
        corrector_time_limit_s=effective_budget,
    )
    deterministic = repeat["decision_payload_sha256"] == arms[0]["decision_payload_sha256"]

    report = {
        "probe": "m51c_deterministic_rollout",
        "trained": False,
        "note": (
            "本 probe 不训练：无 PPO ratio/clip、无 GAE/advantage、无 actor loss、"
            "无乘子更新、无 optimizer.step()；数值不代表训练或性能结论。"
        ),
        "steps_requested": args.steps,
        "env_seed": args.seed,
        "env_seed_kwargs": ENV_SEED_KWARGS,
        "policy_weights_seed": args.seed,
        "policy_sampler_rng": "explicit torch.Generator（不使用 torch.manual_seed 采样）",
        "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
        "effective_corrector_time_limit_s": effective_budget,
        "corrector_time_limit_source": budget_source,
        "corrector_time_limit_s": effective_budget,
        "deterministic_repeat_decision_payload_match": bool(deterministic),
        "repeat_decision_payload_sha256": repeat["decision_payload_sha256"],
        "wall_clock_only_keys": list(WALL_CLOCK_ONLY_KEYS),
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
                    "env_seed": a["env_seed"],
                    "policy_rng_source": a["policy_rng_source"],
                    "payload_sha256": a["payload_sha256"],
                }
                for a in arms
            ]
        )
        run_id = args.run_id or f"m51c_rollout_probe_s{args.seed}_n{args.steps}"
        run_dir = write_run(
            run_id,
            config={
                "probe": "m51c_deterministic_rollout",
                "trained": False,
                "steps": args.steps,
                "env_seed": args.seed,
                "env_seed_kwargs": ENV_SEED_KWARGS,
                "policy_weights_seed": args.seed,
                "policy_sampler_rng": "explicit torch.Generator",
                "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
                "effective_corrector_time_limit_s": effective_budget,
                "corrector_time_limit_source": budget_source,
                "corrector_time_limit_s": effective_budget,
                "units": UNIT_METADATA,
            },
            metrics=metrics,
            report=report,
            base_dir=args.base_dir,
            seed=args.seed,
            command=_command_ledger(args, run_id),
            manifest_metadata={
                "production_corrector_time_limit_s": PRODUCTION_CORRECTOR_TIME_LIMIT_S,
                "effective_corrector_time_limit_s": effective_budget,
                "corrector_time_limit_source": budget_source,
            },
        )
        print(f"\nrun 产物：{run_dir}")


if __name__ == "__main__":
    main()
