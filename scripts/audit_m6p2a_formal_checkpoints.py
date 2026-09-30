"""M6-P2a：**正式 checkpoint 审核**与 **train-only 服务／库存诊断**（只读）。

两件事：

```text
工作一  审核 runs/m13gfck_formal_train_seed{0,1,2}/ 三个正式训练产物
        （manifest / 512 批计数 / 批次顺序摘要 / checkpoint 契约 / 与 live 资产一致）
        并对每个 seed 各导出一次 `formal_training_policy` 评估输入，做源/导出对照。
工作二  在矩阵 v3 的 **212 个 train 日 origin × 3 seed = 636 个完整 episode** 上评估，
        显式传入冻结服务标准 `m6-service-standard-v1`，`scenario_seed = 0` **实际**驱动环境。
```

**边界**：不读/不运行 validation/test、不重训、不放宽任何服务/物理/终点 SOC 条件、
不做五方法公平收益比较；三个正式 checkpoint **只读**。

```bash
uv run python -m scripts.audit_m6p2a_formal_checkpoints --run-id <id> [--origins-limit N]
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

FORMAL_RUN_IDS = ("m13gfck_formal_train_seed0", "m13gfck_formal_train_seed1",
                  "m13gfck_formal_train_seed2")
RECOVERED_RUN_ID = "m13gfck_seed0_recovered_artifacts"
MATRIX_LOGICAL = "configs/experiments/m9_experiment_matrix_v3.json"
TRAINING_CONFIG_LOGICAL = "configs/training/idc_training_config_v1.json"
METHOD = "safe_ppo_joint_rolling_corrector"
SCENARIO_SEED = 0
EXPECTED_BATCHES = 512
EXPECTED_EPISODES = 2048
EXPECTED_TRANSITIONS = 98_304
EXPECTED_ADAM = 8192
EXPECTED_LAG = 512
HORIZON = 48

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
STATEMENT = (
    "M6-P2a：正式训练产物审核 + train-only（212 train 日 × 3 seed = 636 episode）"
    "服务与库存诊断。**不是** validation/test，**不是**五方法公平收益比较；"
    "**不**声称收敛或泛化性能。"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check(name: str, ok: bool, detail: str = "") -> dict[str, Any]:
    return {"check": name, "pass": bool(ok), "detail": detail}


# =============================== 工作一：审核 ===============================

def audit_formal_run(run_id: str) -> dict[str, Any]:
    """逐项核对一个正式 run 的产物（只读）。"""
    import torch

    from safe_rl_v2.formal_train import load_verified_matrix
    from safe_rl_v2.formal_train_loop import (
        FORMAL_RESUME_ARTIFACT_ROLE,
        FORMAL_RESUME_SCHEMA,
        FORMAL_TRAINING_SCOPE,
        _optimizer_steps,
        live_asset_hash_check,
        load_frozen_training_config,
    )

    run_dir = REPO_ROOT / "runs" / run_id
    checks: list[dict[str, Any]] = []
    if not run_dir.is_dir():
        return {"run_id": run_id, "exists": False, "checks": [
            _check("run 目录存在", False, str(run_dir))]}

    required = ("config.yaml", "metrics.parquet", "report.json", "figures", "manifest.json")
    checks.append(_check("五类产物齐备", all((run_dir / n).exists() for n in required),
                         ", ".join(sorted(p.name for p in run_dir.iterdir()))))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    metrics = pd.read_parquet(run_dir / "metrics.parquet")

    checks.append(_check("manifest.status == success", manifest["status"] == "success",
                         str(manifest["status"])))
    checks.append(_check("manifest 三个来源 hash 非空",
                         all(manifest[k] for k in ("dependency_lock_hash", "data_hash",
                                                   "scenario_hash")),
                         f"lock={str(manifest['dependency_lock_hash'])[:16]}…"))
    checks.append(_check("报告 claims 三项为 false",
                         report["claims"] == CLAIMS, str(report["claims"])))

    # 计数
    checks.append(_check(f"批数 == {EXPECTED_BATCHES}",
                         int(report["batches_run"]) == EXPECTED_BATCHES,
                         str(report["batches_run"])))
    checks.append(_check("metrics 行数 == 批数",
                         len(metrics) == int(report["batches_run"]) == len(metrics),
                         f"metrics={len(metrics)}"))
    checks.append(_check(f"episode == {EXPECTED_EPISODES}",
                         int(report["episodes_total"]) == EXPECTED_EPISODES,
                         str(report["episodes_total"])))
    checks.append(_check(f"transitions == {EXPECTED_TRANSITIONS}",
                         int(report["transitions_total"]) == EXPECTED_TRANSITIONS,
                         str(report["transitions_total"])))
    checks.append(_check(f"Adam step == {EXPECTED_ADAM}",
                         int(report["adam_steps_total"]) == EXPECTED_ADAM,
                         str(report["adam_steps_total"])))
    checks.append(_check(f"乘子更新 == {EXPECTED_LAG}",
                         int(report["lagrangian_updates_total"]) == EXPECTED_LAG,
                         str(report["lagrangian_updates_total"])))

    # 预登记 origin 顺序
    matrix = load_verified_matrix()
    digest = matrix["training_schedule"]["batch_origin_list_digest"]
    checks.append(_check("批次顺序摘要 == 矩阵 v3 预登记值",
                         report["batch_origin_list_digest"] == digest, digest[:16] + "…"))
    checks.append(_check("批次索引连续 0..511",
                         list(metrics["batch_index"]) == list(range(EXPECTED_BATCHES)), ""))

    # 环境种子与命令
    config = load_frozen_training_config()
    from safe_rl_v2.formal_train_loop import train_env_seeds
    seed = int(manifest["seed"])
    expect_seeds = train_env_seeds(config, master_seed=seed)
    checks.append(_check("报告 env_seeds == 由 seed 派生",
                         report["env_seeds"] == expect_seeds, str(expect_seeds)))
    checks.append(_check("manifest.command 含 run-id 与该 seed",
                         f"--run-id {run_id}" in manifest["command"]
                         and f"--seed {seed}" in manifest["command"], manifest["command"]))

    # checkpoint 契约
    ckpt = run_dir / "checkpoint_final.pt"
    checks.append(_check("checkpoint_final.pt 存在", ckpt.is_file(), str(ckpt)))
    if ckpt.is_file():
        checks.append(_check("checkpoint SHA == 报告记录值",
                             _sha256_file(ckpt) == report["checkpoints"]["final_sha256"],
                             report["checkpoints"]["final_sha256"][:16] + "…"))
        payload = torch.load(ckpt, weights_only=False)
        meta, state = payload["metadata"], payload["state"]
        checks.append(_check("checkpoint schema/role/scope 为正式训练契约",
                             meta["schema_hash"] == FORMAL_RESUME_SCHEMA
                             and state["artifact_role"] == FORMAL_RESUME_ARTIFACT_ROLE
                             and state["training_scope"] == FORMAL_TRAINING_SCOPE,
                             f"{meta['schema_hash']} / {state['artifact_role']}"))
        checks.append(_check("checkpoint 21 维动作 / obs 维度一致",
                             int(meta["action_dim"]) == 21 and int(meta["obs_dim"]) > 0,
                             f"action={meta['action_dim']} obs={meta['obs_dim']}"))
        checks.append(_check("checkpoint 冻结配置 == live 冻结配置 v1",
                             state["frozen_config"] == config, ""))
        checks.append(_check("checkpoint 优化器 / 两个 RNG / Lagrangian 状态齐备",
                             {"optimizer", "sampling_generator", "shuffle_generator",
                              "lagrangian"} <= set(state), ""))
        checks.append(_check("checkpoint next_batch_index == 512",
                             int(state["next_batch_index"]) == EXPECTED_BATCHES,
                             str(state["next_batch_index"])))
        # 用**正式 loader** 能读回，且优化器步数与报告一致
        from safe_rl_v2.formal_train_loop import (
            build_lagrangian,
            build_optimizer,
            build_seeded_policy,
            load_resume_checkpoint,
        )
        policy = build_seeded_policy(config, obs_dim=int(meta["obs_dim"]), seed=seed)
        opt = build_optimizer(config, policy)
        lag = build_lagrangian(config)
        import torch as _t
        sg, shg = _t.Generator(), _t.Generator()
        load_resume_checkpoint(
            ckpt, policy=policy, optimizer=opt, lagrangian=lag,
            sampling_generator=sg, shuffle_generator=shg, config=config,
            expected_obs_dim=int(meta["obs_dim"]),
            expected_schema=FORMAL_RESUME_SCHEMA,
            expected_scope=FORMAL_TRAINING_SCOPE,
            expected_role=FORMAL_RESUME_ARTIFACT_ROLE)
        checks.append(_check("正式 loader 读回后 Adam 步数 == 8192",
                             _optimizer_steps(opt) == EXPECTED_ADAM, str(_optimizer_steps(opt))))
        checks.append(_check("正式 loader 读回后乘子更新 == 512",
                             int(getattr(lag, "_updates", 0)) == EXPECTED_LAG,
                             str(getattr(lag, "_updates", 0))))
    # live 资产
    asset = live_asset_hash_check(config)
    checks.append(_check("冻结配置六项资产 hash 与 live 一致",
                         all(v["match"] for v in asset.values()),
                         str(sorted(r for r, v in asset.items() if not v["match"]))))

    return {
        "run_id": run_id,
        "exists": True,
        "manifest_revision": manifest["revision"],
        "manifest_seed": seed,
        "source_hashes": {k: manifest[k] for k in
                          ("dependency_lock_hash", "data_hash", "scenario_hash")},
        "checkpoint_final_sha256": report["checkpoints"]["final_sha256"],
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
    }


def audit_recovered_artifacts() -> dict[str, Any]:
    """`m13gfck_seed0_recovered_artifacts/` **单列**为历史诊断产物。"""
    run_dir = REPO_ROOT / "runs" / RECOVERED_RUN_ID
    if not run_dir.is_dir():
        return {"run_id": RECOVERED_RUN_ID, "exists": False}
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return {
        "run_id": RECOVERED_RUN_ID,
        "exists": True,
        "role": "historical_diagnostic_artifact",
        "note": ("M1.3g-f-c-k 缺陷①（manifest 保留字段 `seed`）发生后，从 "
                 "checkpoint_final.pt 恢复写出的一套产物：该次 invocation `batches_run=0`，"
                 "计数由活对象读出。**不计入**本卡三份正式 run 的审核对象。"),
        "batches_run_in_that_invocation": int(report["batches_run"]),
        "cumulative_adam_steps_read_from_checkpoint": int(report["adam_steps_total"]),
        "cumulative_lagrangian_updates": int(report["lagrangian_updates_total"]),
    }


# ============================ 工作一：导出与对照 ============================

def export_and_parity(seed: int, run_id: str) -> dict[str, Any]:
    """调用既有导出入口：formal_training_policy + 源/导出完整 episode 对照。"""
    from scripts import export_eval_input_from_training_checkpoint as exp

    run_dir = REPO_ROOT / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file() and json.loads(
            manifest_path.read_text(encoding="utf-8"))["status"] == "success":
        # 幂等：导出件已存在且成功 ⇒ 不重复导出（`write_run` 本就拒绝覆盖成功 run）
        rc = 0
    else:
        src = REPO_ROOT / f"runs/m13gfck_formal_train_seed{seed}/checkpoint_final.pt"
        rc = exp.main(["--source-checkpoint", str(src), "--run-id", run_id,
                       "--scenario-seed", str(SCENARIO_SEED),
                       "--training-seed", str(seed)])
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "export_run_id": run_id,
        "exit": int(rc),
        "status": json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["status"],
        "source_kind": report.get("source_kind"),
        "role": report["exported_checkpoint"]["artifact_role"],
        "source_sha256": report["source_checkpoint"]["sha256"],
        "exported_sha256": report["exported_checkpoint"]["sha256"],
        "source_unchanged": report["source_checkpoint"]["unchanged_during_run"],
        "parameter_updates": 0,
        "provenance_all_match": report["source_provenance_verification"]["all_match"],
        "actions_identical": report["deterministic_action_consistency"][
            "source_vs_exported_equal"],
        "episode_identical": report["source_vs_exported_episode"]["identical"],
        "episode_differing_fields": report["source_vs_exported_episode"][
            "evaluation_record_differing_fields"],
        "inventory_differing_fields": report["source_vs_exported_episode"][
            "inventory_differing_fields"],
    }


# =============================== 工作二：诊断 ===============================

def evaluate_one(seed: int, origin: int, loaded, deterministic_action,
                 config, standard) -> dict[str, Any]:
    """一个 (seed, origin) 的完整 episode 评估（显式传冻结服务标准）。"""
    from evaluation.adapter import evaluate
    from evaluation.inventory import PairingKey, capture_inventory
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
    from safe_rl.corrector_wrapper import CorrectorWrapper
    from safe_rl_v2.formal_train_loop import build_train_env

    # `build_train_env` 返回 `(env, injection)`；评估只需 env（injection 用于配对键）。
    raw_env, injection = build_train_env(int(origin), master_seed=SCENARIO_SEED,
                                         config=config)
    wrapped = CorrectorWrapper(
        raw_env, corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S)
    run_id = f"m6p2a_s{seed}_o{origin}"
    started = time.perf_counter()
    record = evaluate(
        wrapped, METHOD, lambda obs: deterministic_action(loaded.policy, obs),
        run_id=run_id, service_standard=standard, seed=SCENARIO_SEED,
        action_mode=loaded.action_mode,
        checkpoint_id=f"runs/m6p2a_export_seed{seed}/eval_input.pt",
        checkpoint_role=loaded.artifact_role)
    inventory = capture_inventory(wrapped, record)
    elapsed = time.perf_counter() - started
    provenance = str(getattr(injection, "provenance_hash", ""))

    correction = record.correction
    injection = getattr(wrapped.unwrapped, "formal_injection", None)
    key = PairingKey(split="train",
                     episode_start=str(getattr(injection, "start", "")),
                     scenario_seed=SCENARIO_SEED)
    return {
        "seed": int(seed),
        "origin": int(origin),
        "episode_start": key.episode_start,
        "scenario_seed": SCENARIO_SEED,
        "pairing_key": json.dumps(key.to_dict(), ensure_ascii=False),
        "steps": int(record.steps),
        "episode_complete": bool(record.steps == HORIZON
                                 and record.failure_classification is None),
        "failure_classification": record.failure_classification,
        # 服务两项 + 期末剩余
        "on_time_task_rate": record.service.on_time_task_rate,
        "on_time_work_rate": record.service.on_time_work_rate,
        "end_leftover_work_fraction": record.service.end_leftover_work_fraction,
        "end_leftover_work": record.service.end_leftover_work,
        "non_interruptible_interruption_count":
            record.service.non_interruptible_interruption_count,
        "due_in_episode_tasks": record.service.due_in_episode_tasks,
        "service_qualified": record.service_qualified,
        "service_standard_id": record.service_standard_id,
        "service_note": record.service_qualification_note,
        # 物理
        "access_limit_violation_steps": record.physical.access_limit_violation_steps,
        "soc_violation_steps": record.physical.soc_violation_steps,
        "charge_discharge_exclusion_violations":
            record.physical.charge_discharge_exclusion_violations,
        "energy_conservation_violations": record.physical.energy_conservation_violations,
        # 库存
        "initial_soc": inventory.initial_soc,
        "final_soc": inventory.final_soc,
        "initial_energy_kwh": inventory.initial_energy_kwh,
        "final_energy_kwh": inventory.final_energy_kwh,
        "soc_target": inventory.soc_target,
        "soc_final_tolerance": inventory.soc_final_tolerance,
        "final_soc_deviation": inventory.final_soc_deviation,
        "final_soc_within_env_tolerance": inventory.final_soc_within_env_tolerance,
        "terminal_soc_recovery_kwh": inventory.terminal_soc_recovery_kwh,
        "terminal_leftover_work": inventory.terminal_leftover_work,
        # 修正器
        "correction_timeout_count": None if correction is None else correction.timeout_count,
        "correction_zero_action_fallback_count":
            None if correction is None else correction.zero_action_fallback_count,
        "compute_abs_delta_mean": None if correction is None else correction.compute_abs_delta_mean,
        "storage_abs_delta_mean": None if correction is None else correction.storage_abs_delta_mean,
        "solve_time_median_s": None if correction is None else correction.solve_time_median_s,
        # 经济与碳
        "purchase_cost_sgd": float(record.purchase_cost_sgd),
        "carbon_kg_co2e": float(record.carbon_kg_co2e),
        "grid_energy_kwh": float(record.grid_energy_kwh),
        "completed_work": float(record.completed_work),
        "elapsed_s": float(elapsed),
        "origin_provenance": provenance,
    }


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    reasons = {
        "on_time_task_rate_below_min": 0,
        "on_time_work_rate_below_min": 0,
        "end_leftover_fraction_above_max": 0,
        "non_interruptible_interruption_above_max": 0,
        "physical_violation": 0,
        "episode_incomplete": 0,
        "final_soc_outside_env_tolerance": 0,
    }
    for r in rows:
        if not r["episode_complete"]:
            reasons["episode_incomplete"] += 1
        if r["on_time_task_rate"] is None or r["on_time_task_rate"] < 0.95:
            reasons["on_time_task_rate_below_min"] += 1
        if r["on_time_work_rate"] is None or r["on_time_work_rate"] < 0.95:
            reasons["on_time_work_rate_below_min"] += 1
        if (r["end_leftover_work_fraction"] is None
                or r["end_leftover_work_fraction"] > 0.01):
            reasons["end_leftover_fraction_above_max"] += 1
        if r["non_interruptible_interruption_count"] > 0:
            reasons["non_interruptible_interruption_above_max"] += 1
        if any(r[k] > 0 for k in ("access_limit_violation_steps", "soc_violation_steps",
                                  "charge_discharge_exclusion_violations",
                                  "energy_conservation_violations")):
            reasons["physical_violation"] += 1
        if not r["final_soc_within_env_tolerance"]:
            reasons["final_soc_outside_env_tolerance"] += 1
    return reasons


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    for seed in sorted({r["seed"] for r in rows}):
        sub = [r for r in rows if r["seed"] == seed]
        served = sum(1 for r in sub if r["service_qualified"] is True)
        inv_ok = sum(1 for r in sub if r["final_soc_within_env_tolerance"])
        both = sum(1 for r in sub if r["service_qualified"] is True
                   and r["final_soc_within_env_tolerance"])
        by_seed[str(seed)] = {
            "episodes": len(sub),
            "service_qualified_count": served,
            "service_qualified_rate": served / len(sub),
            "final_soc_within_tolerance_count": inv_ok,
            "final_soc_within_tolerance_rate": inv_ok / len(sub),
            "both_count": both,
            "both_rate": both / len(sub),
            "reason_counts": _reason_counts(sub),
            "on_time_task_rate": _stats([r["on_time_task_rate"] for r in sub]),
            "on_time_work_rate": _stats([r["on_time_work_rate"] for r in sub]),
            "end_leftover_work_fraction": _stats(
                [r["end_leftover_work_fraction"] for r in sub]),
            "final_soc": _stats([r["final_soc"] for r in sub]),
            "non_interruptible_interruption_count": _stats(
                [float(r["non_interruptible_interruption_count"]) for r in sub]),
            "purchase_cost_sgd": _stats([r["purchase_cost_sgd"] for r in sub]),
            "carbon_kg_co2e": _stats([r["carbon_kg_co2e"] for r in sub]),
        }
    return by_seed


def _stats(values: list[float | None]) -> dict[str, float | int | None]:
    present = [float(v) for v in values if v is not None]
    if not present:
        return {"n": 0, "min": None, "p50": None, "max": None}
    return {"n": len(present), "min": min(present), "p50": float(st.median(present)),
            "max": max(present)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.audit_m6p2a_formal_checkpoints",
        description="M6-P2a 正式 checkpoint 审核 + train-only 服务/库存诊断")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--origins-limit", type=int, default=0,
                        help="只跑前 N 个 train origin（0 = 全部 212）；用于单 origin 走通写盘")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from evaluation.service_standard import FROZEN_PROJECT_SERVICE_STANDARD
    from runs.writer import git_revision, write_run
    from safe_rl_v2.formal_train import load_verified_matrix
    from safe_rl_v2.formal_train_loop import load_frozen_training_config

    base = REPO_ROOT / args.base_dir
    command = (f"python -m scripts.audit_m6p2a_formal_checkpoints --run-id {args.run_id}"
               + (f" --origins-limit {args.origins_limit}" if args.origins_limit else ""))
    started = time.perf_counter()
    try:
        config = load_frozen_training_config()
        matrix = load_verified_matrix()
        pool = [int(o) for o in matrix["training_schedule"]["origin_pool"]]
        origins = pool[:args.origins_limit] if args.origins_limit > 0 else pool

        audit: dict[str, Any] | None = None
        exports: list[dict[str, Any]] = []
        if not args.skip_audit:
            audit = {
                "formal_runs": [audit_formal_run(r) for r in FORMAL_RUN_IDS],
                "historical_artifacts": audit_recovered_artifacts(),
                "contract": {
                    "expected_batches": EXPECTED_BATCHES,
                    "expected_episodes": EXPECTED_EPISODES,
                    "expected_transitions": EXPECTED_TRANSITIONS,
                    "expected_adam_steps": EXPECTED_ADAM,
                    "expected_lagrangian_updates": EXPECTED_LAG,
                },
            }
            audit["all_formal_runs_pass"] = all(r.get("all_pass") for r in audit["formal_runs"])
        for seed in (0, 1, 2):
            exports.append(export_and_parity(seed, f"m6p2a_export_seed{seed}"))

        rows: list[dict[str, Any]] = []
        from checkpointing.eval_input import load_evaluation_checkpoint
        from evaluation.controlled_run import deterministic_action as det_action
        from evaluation.sources import verify_evaluation_input_sources

        for seed in (0, 1, 2):
            # 策略**只**经正式 loader 从导出件加载（不直接读训练 checkpoint）
            loaded = load_evaluation_checkpoint(
                REPO_ROOT / f"runs/m6p2a_export_seed{seed}/eval_input.pt")
            verify_evaluation_input_sources(loaded)
            for origin in origins:
                rows.append(evaluate_one(seed, origin, loaded, det_action, config,
                                         FROZEN_PROJECT_SERVICE_STANDARD))
            print(f"  seed {seed} 完成 {len(origins)} 个 episode（累计 {len(rows)}）", flush=True)

        summary = summarise(rows)
        provenance = {int(r["origin"]): str(r["origin_provenance"])
                      for r in rows if r.get("origin_provenance")}
        for r in rows:
            # 账本已算完；metrics 里只留前缀便于人工核对，不留全文
            r["origin_provenance_prefix"] = str(r.pop("origin_provenance", ""))[:16]
        from safe_rl_v2.formal_train_loop import training_source_ledger
        ledger = training_source_ledger(provenance)
        report = {
            "entry": "python -m scripts.audit_m6p2a_formal_checkpoints",
            "statement": STATEMENT,
            "claims": dict(CLAIMS),
            "scope": "train_only", "split": "train",
            "scenario_seed": SCENARIO_SEED,
            "service_standard": {
                "standard_id": FROZEN_PROJECT_SERVICE_STANDARD.standard_id,
                "frozen": FROZEN_PROJECT_SERVICE_STANDARD.frozen,
                "passed_explicitly": True,
                "on_time_task_rate_min": FROZEN_PROJECT_SERVICE_STANDARD.on_time_task_rate_min,
                "on_time_work_rate_min": FROZEN_PROJECT_SERVICE_STANDARD.on_time_work_rate_min,
                "end_leftover_work_fraction_max":
                    FROZEN_PROJECT_SERVICE_STANDARD.end_leftover_work_fraction_max,
                "non_interruptible_interruption_max":
                    FROZEN_PROJECT_SERVICE_STANDARD.non_interruptible_interruption_max,
            },
            "origins": origins,
            "origins_count": len(origins),
            "seeds": [0, 1, 2],
            "episodes_total": len(rows),
            "audit": audit,
            "exports": exports,
            "per_seed_summary": summary,
            "method_label": METHOD,
            "fair_benefit_not_computed": (
                "本卡**不**计算五方法公平收益：同一方法的三个训练 seed **不得**当作"
                "五方法比较；配对需不同方法在同一配对键上的两侧结果。"),
            "source_ledger": dict(ledger),
            "origin_provenance_origins": len(provenance),
            "elapsed_s": float(time.perf_counter() - started),
            "code_revision": git_revision(),
        }
        metrics = pd.DataFrame(rows)
        run_path = write_run(
            args.run_id,
            config={"entry": "python -m scripts.audit_m6p2a_formal_checkpoints",
                    "run_id": args.run_id, "status": "success", "scope": "train_only",
                    "split": "train", "scenario_seed": SCENARIO_SEED,
                    "claims": dict(CLAIMS), "origins_count": len(origins),
                    "episodes_total": len(rows), "code_revision": git_revision()},
            metrics=metrics, report=report, base_dir=str(base), seed=SCENARIO_SEED,
            command=command, status="success",
            dependency_lock_hash=ledger["dependency_lock_hash"],
            data_hash=ledger["data_hash"], scenario_hash=ledger["scenario_hash"],
            manifest_metadata={"scope": "train_only", "episodes_total": len(rows),
                               "origins_count": len(origins)})
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(args.run_id, config={"run_id": args.run_id, "status": "failed"},
                  metrics=pd.DataFrame(), base_dir=str(base), seed=SCENARIO_SEED,
                  command=command,
                  report={"entry": "python -m scripts.audit_m6p2a_formal_checkpoints",
                          "claims": dict(CLAIMS),
                          "failure": f"{type(exc).__name__}: {exc}"},
                  status="failed", failure_classification=type(exc).__name__)
        print(f"M6-P2a 失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"run 产物：{run_path}")
    print(f"  episodes={len(rows)}  origins={len(origins)}  seeds=3")
    if audit:
        passed = sum(1 for r in audit["formal_runs"] if r.get("all_pass"))
        print(f"  正式 run 审核：{passed}/3 全通过")
    for seed_key, s in summary.items():
        print(f"  seed {seed_key}: 服务合格 {s['service_qualified_count']}/{s['episodes']} "
              f"({s['service_qualified_rate']:.3f})  "
              f"终点库存合格 {s['final_soc_within_tolerance_count']}/{s['episodes']} "
              f"({s['final_soc_within_tolerance_rate']:.3f})  "
              f"同时满足 {s['both_count']} ({s['both_rate']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
