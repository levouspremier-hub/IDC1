"""M1.3g-f-c-k：**正式 train-only 训练入口**（安全 PPO ＋联合滚动修正）。

```bash
uv run python -m safe_rl_v2.formal_train --seed 0        # 512 批（冻结规模）
uv run python -m safe_rl_v2.formal_train --seed 1 --resume-from <ckpt>
```

数据来源**全部冻结**，入口**不提供**任何覆盖超参数 / 规模 / 日期 / origin 池的开关：

```text
超参数 / 种子偏移   configs/training/idc_training_config_v1.json（冻结配置 v1）
批次顺序 / origin 池  configs/experiments/m9_experiment_matrix_v3.json（预登记）
                    origin(b, s) = pool[(4*b + s) mod 212]，每批 4 个 48 步 episode
规模                512 批 / seed（矩阵 training_schedule.batches_per_seed）
seed                0 / 1 / 2（矩阵 training_schedule.seed_derivation.train_seeds）
```

放行合取门（全部通过才开跑）：

1. v5 train split 的 `readiness` **严格** both-false（v5 字节冻结）；
2. `env release v1` 验签通过且 `formal_env_ready=true`；
3. **`train release v1`** 验签通过且 `formal_training_ready=true`（本卡新增）；
4. 冻结训练配置 v1 与矩阵 v3 验签/校验通过，且入口实际读取的**就是**被发布产物绑定的那份。

**本入口只做训练**：不跑 validation/test、**不**声称收敛或性能；`claims` 三项恒为 `false`。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from safe_rl_v2.formal_train_loop import (
    FORMAL_RESUME_ARTIFACT_ROLE,
    FORMAL_RESUME_SCHEMA,
    FORMAL_TRAINING_SCOPE,
    build_lagrangian,
    build_optimizer,
    build_seeded_policy,
    build_train_env,
    config_provenance_summary,
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
    "正式 train-only 训练（安全 PPO ＋联合滚动修正，冻结配置 v1 + M9.2 矩阵 v3 预登记顺序）。"
    "本 run 只完成**训练**：**不**声称收敛、**不**声称性能、**未**运行 validation/test。"
)
MATRIX_LOGICAL = "configs/experiments/m9_experiment_matrix_v3.json"
TRAIN_SPLIT = "train"


class FormalTrainError(RuntimeError):
    """正式训练入口的前置条件或运行约定不符（一律 fail closed）。"""


def load_verified_matrix(path: Path | None = None) -> dict:
    """读取 M9.2 矩阵 v3（正式训练的**预登记批次顺序**来源）。"""
    resolved = path or (REPO_ROOT / MATRIX_LOGICAL)
    if not resolved.is_file():
        raise FormalTrainError(f"缺少实验矩阵：{resolved}")
    matrix = json.loads(resolved.read_text(encoding="utf-8"))
    if matrix.get("schema") != "m9-experiment-matrix-v3":
        raise FormalTrainError(
            f"实验矩阵 schema 必须为 m9-experiment-matrix-v3，实际 {matrix.get('schema')!r}")
    if matrix.get("status") != "frozen_matrix":
        raise FormalTrainError(f"实验矩阵未冻结：{matrix.get('status')!r}")
    if matrix.get("formal_training_ready") is not False:
        raise FormalTrainError("矩阵的 formal_training_ready 必须为 false（矩阵冻结 ≠ 训练放行）")
    return matrix


def require_release_chain() -> dict[str, Any]:
    """放行合取门：v5 both-false + env release + **train release** + 绑定一致。"""
    from scenario.b6_split_manifests import load_verified_split_manifest_v5
    from scenario.env_release import load_verified_env_release
    from scenario.formal_train_release import (
        TRAIN_RELEASE_BINDING_LOGICAL,
        load_verified_train_release,
    )

    payload = load_verified_split_manifest_v5(expected_split=TRAIN_SPLIT)
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict) or any(v is not False for v in readiness.values()):
        raise FormalTrainError(
            f"v5 train split 的 readiness 必须严格保持 both-false；实际 {readiness!r}")

    env_release = load_verified_env_release()
    if env_release["readiness"]["formal_env_ready"] is not True:
        raise FormalTrainError("env release v1 声明 formal_env_ready != true，拒绝开跑")

    train_release = load_verified_train_release()
    if train_release["readiness"]["formal_training_ready"] is not True:
        raise FormalTrainError(
            "train release v1 声明 formal_training_ready != true，拒绝开跑")

    # 入口**实际读取**的对象必须就是发布产物绑定的那一份（path + live sha256）
    import hashlib

    binds = train_release["binds"]
    live = {
        "training_config_v1": REPO_ROOT / TRAIN_RELEASE_BINDING_LOGICAL["training_config_v1"],
        "experiment_matrix_v3": REPO_ROOT / MATRIX_LOGICAL,
        "formal_train_entry": Path(__file__).resolve(),
        "formal_train_loop": REPO_ROOT / "safe_rl_v2/formal_train_loop.py",
    }
    for role, path in live.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != binds[role]["sha256"]:
            raise FormalTrainError(
                f"入口实际读取的 {role}（{path}）与 train release 绑定的字节不一致："
                f"live={actual[:16]}… bound={binds[role]['sha256'][:16]}…")
    return {"env_release": env_release, "train_release": train_release,
            "matrix_binds_verified": sorted(live)}


def preregistered_batches(matrix: dict, seed: int) -> list[list[int]]:
    """按**预登记规则**重建 512 批的逐批 origin 列表，并核对摘要。"""
    schedule = matrix["training_schedule"]
    pool = [int(o) for o in schedule["origin_pool"]]
    per_batch = int(schedule["episodes_per_batch"])
    batches = int(schedule["batches_per_seed"])
    seeds = [int(s) for s in schedule["seed_derivation"]["train_seeds"]]
    if seed not in seeds:
        raise FormalTrainError(f"seed {seed} 不在矩阵预登记的 {seeds} 中")
    rows = [[b, s, pool[(per_batch * b + s) % len(pool)]]
            for b in range(batches) for s in range(per_batch)]
    import hashlib

    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False)
                            .encode("utf-8")).hexdigest()
    if digest != schedule["batch_origin_list_digest"]:
        raise FormalTrainError(
            f"按矩阵规则重建的批次清单摘要 {digest[:16]}… 与矩阵登记值 "
            f"{schedule['batch_origin_list_digest'][:16]}… 不一致")
    return [[r[2] for r in rows[b * per_batch:(b + 1) * per_batch]]
            for b in range(batches)]


def run(*, run_id: str, seed: int, resume_from: str | None, base_dir: str,
        checkpoint_every: int) -> int:
    from runs.writer import git_revision, write_run

    release_chain = require_release_chain()
    matrix = load_verified_matrix()
    config = load_frozen_training_config()
    summary = config_provenance_summary(config)
    training = config["training"]
    per_batch = int(training["sampling"]["episodes_per_batch"])
    corrector_time_limit_s = float(training["corrector"]["time_limit_s"])
    batch_origins = preregistered_batches(matrix, int(seed))

    asset_check = live_asset_hash_check(config)
    mismatch = sorted(r for r, v in asset_check.items() if not v["match"])
    if mismatch:
        raise FormalTrainError(f"冻结配置记录的资产 hash 与 live 不一致：{mismatch}")

    obs_dim = int(build_train_env(batch_origins[0][0], master_seed=int(seed),
                                  config=config)[0].obs_dim)
    policy = build_seeded_policy(config, obs_dim=obs_dim, seed=int(seed))
    optimizer = build_optimizer(config, policy)
    lagrangian = build_lagrangian(config)
    from safe_rl_v2.controlled_formal_train import _generators

    sampling_generator, shuffle_generator = _generators(int(seed))

    run_dir = REPO_ROOT / base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rolling_ckpt = run_dir / "checkpoint_latest.pt"
    final_ckpt = run_dir / "checkpoint_final.pt"

    origin_provenance: dict[int, str] = {}
    next_index = 0
    if resume_from is not None:
        info = load_resume_checkpoint(
            resume_from, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
            sampling_generator=sampling_generator, shuffle_generator=shuffle_generator,
            config=config, expected_obs_dim=obs_dim,
            expected_schema=FORMAL_RESUME_SCHEMA,
            expected_scope=FORMAL_TRAINING_SCOPE,
            expected_role=FORMAL_RESUME_ARTIFACT_ROLE)
        next_index = int(info["next_batch_index"])
        origin_provenance = dict(info["origin_provenance"])

    command = (f"python -m safe_rl_v2.formal_train --run-id {run_id} --seed {seed}"
               + (f" --resume-from {resume_from}" if resume_from else ""))

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for index in range(next_index, len(batch_origins)):
            batch_started = time.perf_counter()
            batch = run_training_batch(
                policy, optimizer, lagrangian, sampling_generator, shuffle_generator,
                config=config, origins=batch_origins[index], batch_index=index,
                env_seed=int(seed), corrector_time_limit_s=corrector_time_limit_s,
                master_seed=int(seed))
            origin_provenance.update(
                {int(o): str(p) for o, p in batch["origin_provenance"].items()})
            checkpoint_write_s = 0.0
            if checkpoint_every > 0 and (
                    (index + 1) % checkpoint_every == 0 or index == len(batch_origins) - 1):
                _t0 = time.perf_counter()
                save_resume_checkpoint(
                    rolling_ckpt, policy=policy, optimizer=optimizer,
                    lagrangian=lagrangian, sampling_generator=sampling_generator,
                    shuffle_generator=shuffle_generator, config=config,
                    origins=[o for grp in batch_origins for o in grp],
                    next_batch_index=index + 1, obs_dim=obs_dim,
                    origin_provenance=origin_provenance, master_seed=int(seed),
                    code_revision=git_revision(),
                    training_scope=FORMAL_TRAINING_SCOPE,
                    artifact_role=FORMAL_RESUME_ARTIFACT_ROLE,
                    schema=FORMAL_RESUME_SCHEMA)
                checkpoint_write_s = time.perf_counter() - _t0
            record = {k: v for k, v in batch.items() if k != "step_records"}
            record["elapsed_s"] = time.perf_counter() - batch_started
            record["checkpoint_write_s"] = checkpoint_write_s
            records.append(record)
            if (index + 1) % 32 == 0 or index == 0:
                print(f"  seed {seed} 批 {index + 1}/{len(batch_origins)} "
                      f"adam_cum={batch['optimizer_steps_cumulative']} "
                      f"lag_cum={batch['lagrangian_updates_cumulative']} "
                      f"已用时 {time.perf_counter() - started:.0f}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录，不覆盖既有成功 run
        write_run(
            run_id,
            config={"entry": "python -m safe_rl_v2.formal_train", "run_id": run_id,
                    "status": "failed", "training_scope": FORMAL_TRAINING_SCOPE,
                    "seed": int(seed), "batches_completed": len(records)},
            metrics=pd.DataFrame(), base_dir=str(REPO_ROOT / base_dir), seed=int(seed),
            command=command,
            report={"entry": "python -m safe_rl_v2.formal_train", "claims": dict(CLAIMS),
                    "statement": STATEMENT,
                    "training_scope": FORMAL_TRAINING_SCOPE,
                    "batches_completed": len(records),
                    "failure": f"{type(exc).__name__}: {exc}"},
            status="failed", failure_classification=type(exc).__name__)
        print(f"正式训练失败（已完成 {len(records)} 批）：{type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    save_resume_checkpoint(
        final_ckpt, policy=policy, optimizer=optimizer, lagrangian=lagrangian,
        sampling_generator=sampling_generator, shuffle_generator=shuffle_generator,
        config=config, origins=[o for grp in batch_origins for o in grp],
        next_batch_index=len(batch_origins), obs_dim=obs_dim,
        origin_provenance=origin_provenance, master_seed=int(seed),
        code_revision=git_revision(), training_scope=FORMAL_TRAINING_SCOPE,
        artifact_role=FORMAL_RESUME_ARTIFACT_ROLE, schema=FORMAL_RESUME_SCHEMA)

    elapsed = time.perf_counter() - started
    ledger = training_source_ledger(origin_provenance)
    # 计数**从活对象取**，而不是从 `records[-1]` 取：从已完成的 checkpoint 恢复时
    # 本进程可能一批都不跑（records 为空），但累计计数仍然真实可读。
    from safe_rl_v2.formal_train_loop import _optimizer_steps

    adam_steps = int(_optimizer_steps(optimizer))
    lag_updates = int(getattr(lagrangian, "_updates", 0))
    transitions = adam_steps // 16 * 192
    report = {
        "entry": "python -m safe_rl_v2.formal_train",
        "statement": STATEMENT,
        "claims": dict(CLAIMS),
        "training_scope": FORMAL_TRAINING_SCOPE,
        "split": TRAIN_SPLIT,
        "seed": int(seed),
        "batches_run": len(records),
        "episodes_total": len(records) * per_batch,
        "transitions_total": transitions,
        "adam_steps_total": adam_steps,
        "lagrangian_updates_total": lag_updates,
        "batch_origin_list_digest": matrix["training_schedule"]["batch_origin_list_digest"],
        "batch_origin_rule": matrix["training_schedule"]["batch_origin_rule"],
        "origin_pool_size": len(matrix["training_schedule"]["origin_pool"]),
        "env_seeds": train_env_seeds(config, master_seed=int(seed)),
        "corrector": {"mode": "on", "time_limit_s": corrector_time_limit_s},
        "frozen_config": {"path": "configs/training/idc_training_config_v1.json",
                          "summary": summary},
        "matrix": {"path": MATRIX_LOGICAL, "version": matrix["version"],
                   "status": matrix["status"]},
        "release_chain": {"env_release_readiness":
                          release_chain["env_release"]["readiness"],
                          "train_release_readiness":
                          release_chain["train_release"]["readiness"],
                          "train_release_revision":
                          release_chain["train_release"]["release_revision"]},
        "asset_hash_check_all_match": True,
        "source_ledger": dict(ledger),
        "resumed_from": resume_from,
        "checkpoints": {"latest": str(rolling_ckpt.relative_to(REPO_ROOT)),
                        "final": str(final_ckpt.relative_to(REPO_ROOT)),
                        "final_sha256": _sha256(final_ckpt),
                        "latest_sha256": _sha256(rolling_ckpt)},
        "elapsed_s": float(elapsed),
        "peak_rss_bytes": peak_rss_bytes(),
        "zero_action_fallback_steps_total": int(sum(
            b["zero_action_fallback_steps"] for b in records)),
        "deadline_shortfall_steps_total": int(sum(
            b["deadline_shortfall_steps"] for b in records)),
        "failures": 0,
        "code_revision": git_revision(),
    }
    metrics = pd.DataFrame([{
        "batch_index": b["batch_index"], "origins": ",".join(str(o) for o in b["origins"]),
        "transitions": b["transitions"],
        "optimizer_steps_cumulative": b["optimizer_steps_cumulative"],
        "lagrangian_updates_cumulative": b["lagrangian_updates_cumulative"],
        "multiplier_business": b["multipliers_post_update"]["business"],
        "multiplier_carbon": b["multipliers_post_update"]["carbon"],
        "business_signal_mean": b["constraint_signal_mean"]["business"],
        "carbon_signal_mean": b["constraint_signal_mean"]["carbon"],
        "raw_exec_difference_count": b["raw_exec_difference_count"],
        "deadline_shortfall_steps": b["deadline_shortfall_steps"],
        "zero_action_fallback_steps": b["zero_action_fallback_steps"],
        "total_batch_s": b["timing_s"]["total_batch_s"],
        "elapsed_s": b["elapsed_s"],
        "checkpoint_write_s": b["checkpoint_write_s"],
        "peak_rss_bytes": b["peak_rss_bytes"],
    } for b in records])
    run_path = write_run(
        run_id,
        config={"entry": "python -m safe_rl_v2.formal_train", "run_id": run_id,
                "status": "success", "training_scope": FORMAL_TRAINING_SCOPE,
                "claims": dict(CLAIMS), "seed": int(seed),
                "batches_run": len(records), "split": TRAIN_SPLIT,
                "config_summary": summary, "source_ledger": dict(ledger),
                "code_revision": git_revision()},
        metrics=metrics, report=report, base_dir=str(REPO_ROOT / base_dir),
        seed=int(seed), command=command, status="success",
        dependency_lock_hash=ledger["dependency_lock_hash"],
        data_hash=ledger["data_hash"], scenario_hash=ledger["scenario_hash"],
        # `seed` 是 writer 的**保留字段**（由 `seed=` 参数写入），不得经
        # `manifest_metadata` 覆盖 —— 实测触发过 `ValueError` 并导致 512 批训练完成后
        # 无法落盘（M1.3g-f-c-k 本卡修正）。
        manifest_metadata={"training_scope": FORMAL_TRAINING_SCOPE,
                            "training_seed": int(seed),
                            "batches_run": len(records)})
    print(f"run 产物：{run_path}")
    print(f"  seed={seed}  批数={len(records)}  adam={report['adam_steps_total']}  "
          f"lag={report['lagrangian_updates_total']}")
    print(f"  checkpoint(final)={report['checkpoints']['final_sha256'][:16]}…  "
          f"耗时={elapsed:.0f}s")
    return 0


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m safe_rl_v2.formal_train",
        description="正式 train-only 训练（冻结配置 v1 + 矩阵 v3 预登记顺序；不跑评估）")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True, help="训练 seed（0/1/2）")
    parser.add_argument("--resume-from", default=None,
                        help="正式训练的批次边界恢复 checkpoint")
    parser.add_argument("--checkpoint-every", type=int, default=16,
                        help="每 N 批写一次滚动恢复 checkpoint（末批总是写）")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    return run(run_id=args.run_id, seed=int(args.seed),
               resume_from=args.resume_from, base_dir=args.base_dir,
               checkpoint_every=int(args.checkpoint_every))


if __name__ == "__main__":
    raise SystemExit(main())
