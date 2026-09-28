"""M1.3g-f-c-j：**受控正式训练短跑**入口（独立、显式、非正式训练）。

```bash
uv run python -m safe_rl_v2.controlled_formal_train --run-id <id> --batches 3
uv run python -m safe_rl_v2.controlled_formal_train --run-id <id> --batches 2 \
    --checkpoint-out <CKPT>
uv run python -m safe_rl_v2.controlled_formal_train --run-id <id> --batches 3 \
    --resume-from <CKPT>
```

**这不是正式训练**：`training_scope=controlled_short_run`，`claims` 三项恒为 `False`；
不宣称性能、收敛或正式训练完成。正式训练的放行仍由 `formal_training_ready=false` 阻断，
512 批 × 3 seed 的规模**不在本入口内**。

超参数**全部**来自 `configs/training/idc_training_config_v1.json`；本入口**不提供**
任何覆盖训练超参数的开关（只有 origin 批次范围、RNG 种子与 checkpoint 路径）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from safe_rl_v2.formal_train_loop import (
    CONFIG_SCHEMA,
    CONTROLLED_BATCHES,
    FROZEN_CONFIG_LOGICAL_PATH,
    TRAINING_SCOPE,
    build_lagrangian,
    build_optimizer,
    build_seeded_policy,
    config_provenance_summary,
    controlled_origins,
    live_asset_hash_check,
    load_frozen_training_config,
    load_resume_checkpoint,
    peak_rss_bytes,
    run_training_batch,
    save_resume_checkpoint,
    train_env_seeds,
    training_source_ledger,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
STATEMENT = (
    "受控正式训练短跑：从冻结配置驱动的真实训练闭环跑若干批（train-only）。"
    "**不是**正式训练结果，**不是**性能或收敛结论；512 批 × 3 seed 的正式规模未运行。"
)
#: 两个 RNG 的种子偏移（**显式**，不用全局 RNG，也不重播种全局 RNG）。
SAMPLING_SEED_OFFSET = 0
SHUFFLE_SEED_OFFSET = 100_000


def machine_profile() -> dict[str, Any]:
    """目标机器信息（M9.1；用于预算报告的可核对上下文）。"""
    import platform

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": _total_memory_bytes(),
    }


def _total_memory_bytes() -> int | None:
    import subprocess

    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001 - 平台不支持时明确返回 None，不编造
        return None


def _generators(seed: int) -> tuple[torch.Generator, torch.Generator]:
    sampling = torch.Generator()
    sampling.manual_seed(int(seed) + SAMPLING_SEED_OFFSET)
    shuffle = torch.Generator()
    shuffle.manual_seed(int(seed) + SHUFFLE_SEED_OFFSET)
    return sampling, shuffle


def _obs_dim_from_first_origin(origin: int) -> int:
    from scripts.calibrate_training_config import build_env_for_origin

    env, _injection = build_env_for_origin(int(origin))
    return int(env.obs_dim)


def apply_frozen_thread_setting(config: dict) -> dict[str, Any]:
    """按冻结配置的 `backend.torch_num_threads` 设置 Torch 线程数（M9.1）。

    冻结配置 v1 声明 `torch_num_threads = 1`；若运行时不设置，进程会用 Torch 默认值
    （本机实测为 4），测得的就不是**冻结配置**下的吞吐。本函数**只**改线程数，
    不改任何训练数值与更新语义。
    """
    configured = int(config["training"]["backend"]["torch_num_threads"])
    effective_before = int(torch.get_num_threads())
    torch.set_num_threads(configured)
    return {
        "configured_torch_num_threads": configured,
        "effective_torch_num_threads": int(torch.get_num_threads()),
        "torch_num_threads_before_apply": effective_before,
    }


def _step_metrics(batch: dict[str, Any]) -> dict[str, Any]:
    """一批的**逐 transition 均值**指标（供 metrics.parquet 与报告）。"""
    return {
        "batch_index": int(batch["batch_index"]),
        "origins": ",".join(str(o) for o in batch["origins"]),
        "transitions": int(batch["transitions"]),
        "adam_steps_this_batch": int(batch["adam_steps_this_batch"]),
        "optimizer_steps_cumulative": int(batch["optimizer_steps_cumulative"]),
        "lagrangian_updates_this_batch": int(batch["lagrangian_updates_this_batch"]),
        "lagrangian_updates_cumulative": int(batch["lagrangian_updates_cumulative"]),
        "multiplier_business": float(batch["multipliers_post_update"]["business"]),
        "multiplier_carbon": float(batch["multipliers_post_update"]["carbon"]),
        "business_signal_mean": float(batch["constraint_signal_mean"]["business"]),
        "carbon_signal_mean": float(batch["constraint_signal_mean"]["carbon"]),
        "raw_exec_difference_count": int(batch["raw_exec_difference_count"]),
        "deadline_shortfall_steps": int(batch["deadline_shortfall_steps"]),
        "zero_action_fallback_steps": int(batch["zero_action_fallback_steps"]),
        "total_batch_s": float(batch["timing_s"]["total_batch_s"]),
        "env_build_s": float(batch["timing_s"]["env_build_s"]),
        "rollout_collect_s": float(batch["timing_s"]["rollout_collect_s"]),
        "ppo_update_s": float(batch["timing_s"]["ppo_update_s"]),
        "residual_unattributed_s": float(
            batch["timing_s"]["residual_unattributed_s"]),
        "corrector_solve_median_s": batch["corrector_solve_stats"]["median_s"],
        "corrector_solve_p95_s": batch["corrector_solve_stats"]["p95_s"],
        "peak_rss_bytes": int(batch["peak_rss_bytes"]),
    }


def _batch_record(batch: dict[str, Any], *, elapsed_s: float,
                  checkpoint: dict[str, Any] | None,
                  checkpoint_write_s: float) -> dict[str, Any]:
    """报告用的批次记录：去掉逐 step 明细（单独放），保留可对照的摘要。"""
    record = {k: v for k, v in batch.items() if k != "step_records"}
    record["elapsed_s"] = float(elapsed_s)
    record["checkpoint_write_s"] = float(checkpoint_write_s)
    record["checkpoint"] = checkpoint
    return record


def run(
    *,
    run_id: str,
    batches: int,
    seed: int,
    resume_from: str | None,
    checkpoint_out: str | None,
    base_dir: str,
) -> int:
    from runs.writer import git_revision, write_run

    config = load_frozen_training_config()
    thread_info = apply_frozen_thread_setting(config)
    summary = config_provenance_summary(config)
    training = config["training"]
    episodes_per_batch = int(training["sampling"]["episodes_per_batch"])
    corrector_time_limit_s = float(training["corrector"]["time_limit_s"])

    all_origins = controlled_origins()
    origins = all_origins[:CONTROLLED_BATCHES * episodes_per_batch]
    obs_dim = _obs_dim_from_first_origin(origins[0])

    # 策略初值必须**确定性**：否则两个进程的初始权重不同，闭环不可复现。
    policy = build_seeded_policy(config, obs_dim=obs_dim, seed=seed)
    optimizer = build_optimizer(config, policy)
    lagrangian = build_lagrangian(config)
    sampling_generator, shuffle_generator = _generators(seed)

    # 冻结配置记录的六项资产 hash 必须与 **live** 文件一致（实测比对，不抄字符串）
    asset_check = live_asset_hash_check(config)
    asset_mismatch = sorted(r for r, v in asset_check.items() if not v["match"])
    if asset_mismatch:
        raise ValueError(
            f"冻结配置记录的资产 hash 与 live 文件不一致：{asset_mismatch}")

    resume_info: dict[str, Any] | None = None
    next_batch_index = 0
    origin_provenance: dict[int, str] = {}
    if resume_from is not None:
        resume_info = load_resume_checkpoint(
            resume_from, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
            sampling_generator=sampling_generator, shuffle_generator=shuffle_generator,
            config=config, expected_obs_dim=obs_dim)
        next_batch_index = int(resume_info["next_batch_index"])
        origins = list(resume_info["origins"])
        origin_provenance = dict(resume_info["origin_provenance"])

    command = (
        "python -m safe_rl_v2.controlled_formal_train"
        f" --run-id {run_id} --batches {batches} --seed {seed}"
        + (f" --resume-from {resume_from}" if resume_from else "")
        + (f" --checkpoint-out {checkpoint_out}" if checkpoint_out else "")
    )

    batch_records: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for offset in range(int(batches)):
            index = next_batch_index + offset
            batch_origins = origins[
                index * episodes_per_batch:(index + 1) * episodes_per_batch]
            if len(batch_origins) != episodes_per_batch:
                raise ValueError(
                    f"批 {index} 需要 {episodes_per_batch} 个 origin，"
                    f"origin 列表只到 {len(origins)}")
            batch_started = time.perf_counter()
            batch = run_training_batch(
                policy, optimizer, lagrangian, sampling_generator, shuffle_generator,
                config=config, origins=batch_origins, batch_index=index, env_seed=seed,
                corrector_time_limit_s=corrector_time_limit_s, master_seed=seed)
            origin_provenance.update(
                {int(o): str(p) for o, p in batch["origin_provenance"].items()})
            checkpoint_info = None
            _ckpt_t0 = time.perf_counter()
            if checkpoint_out is not None:
                Path(checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
                checkpoint_info = save_resume_checkpoint(
                    checkpoint_out, policy=policy, optimizer=optimizer,
                    lagrangian=lagrangian, sampling_generator=sampling_generator,
                    shuffle_generator=shuffle_generator, config=config,
                    origins=origins, next_batch_index=index + 1, obs_dim=obs_dim,
                    origin_provenance=origin_provenance, master_seed=seed,
                    code_revision=git_revision())
            checkpoint_write_s = time.perf_counter() - _ckpt_t0
            batch_records.append(_batch_record(
                batch, elapsed_s=time.perf_counter() - batch_started,
                checkpoint=checkpoint_info,
                checkpoint_write_s=checkpoint_write_s))
            step_rows.extend(
                {"batch_index": index, **row} for row in batch["step_records"])
            print(f"  批 {index}: transitions={batch['transitions']} "
                  f"adam_steps_cum={batch['optimizer_steps_cumulative']} "
                  f"lagrangian_updates_cum={batch['lagrangian_updates_cumulative']} "
                  f"multipliers={batch['multipliers_post_update']}")
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录，不覆盖既有成功 run
        write_run(
            run_id,
            config={"entry": "python -m safe_rl_v2.controlled_formal_train",
                    "run_id": run_id, "status": "failed",
                    "training_scope": TRAINING_SCOPE,
                    "config_summary": summary},
            metrics=pd.DataFrame(), base_dir=str(REPO_ROOT / base_dir), seed=int(seed),
            command=command,
            report={"entry": "python -m safe_rl_v2.controlled_formal_train",
                    "statement": "本 run 失败，不代表任何训练结果。",
                    "claims": dict(CLAIMS), "training_scope": TRAINING_SCOPE,
                    "failure": f"{type(exc).__name__}: {exc}"},
            status="failed", failure_classification=type(exc).__name__)
        print(f"受控短跑失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    ledger = training_source_ledger(origin_provenance)
    report = {
        "entry": "python -m safe_rl_v2.controlled_formal_train",
        "statement": STATEMENT,
        "claims": dict(CLAIMS),
        "training_scope": TRAINING_SCOPE,
        "frozen_config": {
            "path": FROZEN_CONFIG_LOGICAL_PATH,
            "schema": CONFIG_SCHEMA,
            "summary": summary,
        },
        "origins": [int(o) for o in origins],
        "episodes_per_batch": episodes_per_batch,
        "steps_per_episode": int(training["sampling"]["horizon"]),
        "corrector": {
            "mode": "on",
            "time_limit_s": corrector_time_limit_s,
            "source": training["corrector"].get("source"),
        },
        "seed": int(seed),
        "machine": machine_profile(),
        "threads": thread_info,
        "env_seeds": train_env_seeds(config, master_seed=seed),
        "master_seed": int(seed),
        "source_ledger": dict(ledger),
        "asset_hash_check": asset_check,
        "origin_provenance": {str(o): p for o, p in sorted(origin_provenance.items())},
        "policy_init_seed": int(seed),
        "policy_init_note": ("策略初始权重在 fork_rng 区间内以 policy_init_seed 播种，"
                             "构造后还原全局 RNG；不重播种全局 RNG"),
        "seed_offsets": {"sampling": SAMPLING_SEED_OFFSET,
                         "shuffle": SHUFFLE_SEED_OFFSET,
                         "note": "只驱动策略采样与 minibatch 打乱；不重播种全局 RNG"},
        "batches_requested": int(batches),
        "batches_run": len(batch_records),
        "resume": resume_info,
        "batches": batch_records,
        "adam_steps_total": int(batch_records[-1]["optimizer_steps_cumulative"]),
        "lagrangian_updates_total": int(
            batch_records[-1]["lagrangian_updates_cumulative"]),
        "final_policy_state_digest": batch_records[-1]["policy_state_digest"],
        "final_optimizer_state_digest": batch_records[-1]["optimizer_state_digest"],
        "final_lagrangian": batch_records[-1]["lagrangian_state"],
        "final_generator_state_digests": {
            "sampling": batch_records[-1]["sampling_generator_state_digest"],
            "shuffle": batch_records[-1]["shuffle_generator_state_digest"],
        },
        "elapsed_s": float(elapsed),
        "peak_rss_bytes": peak_rss_bytes(),
        "checkpoint_size_bytes": (
            Path(checkpoint_out).stat().st_size
            if checkpoint_out is not None and Path(checkpoint_out).exists()
            else None),
        "code_revision": git_revision(),
        "checkpoint_out": checkpoint_out,
    }
    metrics = pd.DataFrame([_step_metrics(b) for b in batch_records])
    run_path = write_run(
        run_id,
        config={
            "entry": "python -m safe_rl_v2.controlled_formal_train",
            "run_id": run_id, "status": "success",
            "training_scope": TRAINING_SCOPE,
            "claims": dict(CLAIMS),
            "frozen_config_path": FROZEN_CONFIG_LOGICAL_PATH,
            "config_summary": summary,
            "origins": [int(o) for o in origins],
            "batches_run": len(batch_records),
            "seed": int(seed),
            "env_seeds": train_env_seeds(config, master_seed=seed),
            "source_ledger": dict(ledger),
        },
        metrics=metrics, base_dir=str(REPO_ROOT / base_dir), seed=int(seed),
        command=command, report=report, status="success",
        dependency_lock_hash=ledger["dependency_lock_hash"],
        data_hash=ledger["data_hash"], scenario_hash=ledger["scenario_hash"],
        manifest_metadata={"training_scope": TRAINING_SCOPE})
    print(f"run 产物：{run_path}")
    print(f"  scope={TRAINING_SCOPE}  batches={len(batch_records)}  seed={seed}")
    print(f"  adam_steps_total={report['adam_steps_total']}  "
          f"lagrangian_updates_total={report['lagrangian_updates_total']}")
    print(f"  env_seeds={report['env_seeds']}")
    print(f"  dependency_lock_hash={ledger['dependency_lock_hash'][:16]}…")
    print(f"  data_hash={ledger['data_hash'][:16]}…")
    print(f"  scenario_hash={ledger['scenario_hash'][:16]}…")
    print(f"  final_policy_state_digest={report['final_policy_state_digest'][:16]}…")
    print(f"  elapsed_s={elapsed:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m safe_rl_v2.controlled_formal_train",
        description="受控正式训练短跑（train-only；非正式训练）")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batches", type=int, default=CONTROLLED_BATCHES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-from", default=None,
                        help="批次边界训练恢复 checkpoint（下一批从该游标继续）")
    parser.add_argument("--checkpoint-out", default=None,
                        help="每批**结束后**写出批次边界训练恢复 checkpoint")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    if args.batches <= 0:
        parser.error("--batches 必须为正整数")
    return run(
        run_id=args.run_id, batches=int(args.batches), seed=int(args.seed),
        resume_from=args.resume_from, checkpoint_out=args.checkpoint_out,
        base_dir=args.base_dir)


if __name__ == "__main__":
    raise SystemExit(main())
