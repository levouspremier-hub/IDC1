"""M9.2-R1：从**训练恢复 checkpoint** 导出评估输入，并用统一评估器复现同一 episode。

```bash
uv run python -m scripts.export_eval_input_from_training_checkpoint \\
    --source-checkpoint runs/m91_ckpt_seed0.pt --run-id m92r1_eval_wiring
```

链路（train-only，**不做任何参数更新**）：

```text
批次边界训练恢复 checkpoint（m1.3g-f-c-j-controlled-resume-v1）
  → 取出其中**真实** policy 权重
  → 按 `checkpointing.eval_input` 契约导出（save_evaluation_checkpoint）
  → 经**正式 loader** 与来源校验加载
  → 交给 `evaluation.adapter.evaluate()` 在同一 train episode 上评估
```

红线：

- **不进行参数更新**；导出只搬运权重；
- 受控短跑源**只能**标 `controlled_short_run_eval_input`，**不得**冒充已审核的
  `formal_training_policy`；
- 只证明**接线可复现**，不宣称训练性能、收敛或策略优劣；
- 源文件**只读**，不覆盖、不改写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
STATEMENT = (
    "M9.2-R1 训练 checkpoint → 评估输入接线复现：从**受控短跑**的批次边界训练恢复 "
    "checkpoint 导出 policy 权重，经正式 loader 加载后由统一评估器跑同一 train episode。"
    "**不做参数更新**，**不宣称训练性能**；角色为 controlled_short_run_eval_input。"
)
METHOD = "controlled_short_run_policy_via_eval_input"

#: 本入口支持的训练恢复 schema（`safe_rl_v2.formal_train_loop.RESUME_SCHEMA`）。
SUPPORTED_TRAINING_SCHEMAS = ("m1.3g-f-c-j-controlled-resume-v1",)


class ExportError(ValueError):
    """源 checkpoint / 导出 / 复现链路不符约定。"""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_training_checkpoint(path: Path) -> tuple[dict, dict]:
    """只读读取训练恢复 checkpoint，校验 schema / role / 必需状态字段。"""
    from checkpointing.eval_input import EVAL_INPUT_SCHEMA
    from checkpointing.versioned import read_checkpoint_payload

    payload = read_checkpoint_payload(path)
    meta, state = payload["metadata"], payload["state"]
    schema = str(meta.get("schema_hash", ""))
    if schema == EVAL_INPUT_SCHEMA:
        raise ExportError(
            f"源文件已是**评估输入**（schema={EVAL_INPUT_SCHEMA}）；"
            "本入口要求**训练恢复** checkpoint")
    if schema not in SUPPORTED_TRAINING_SCHEMAS:
        raise ExportError(f"不支持的训练恢复 schema：{schema!r}")
    if not isinstance(state, dict) or "policy" not in state:
        raise ExportError("训练恢复 checkpoint 缺少 state.policy（真实权重）")
    if state.get("artifact_role") != "controlled_training_resume":
        raise ExportError(
            f"训练恢复 artifact_role 应为 controlled_training_resume，"
            f"实际 {state.get('artifact_role')!r}")
    return meta, state


def export_policy_state(state: dict) -> dict[str, torch.Tensor]:
    """从训练状态取出 policy 权重（**不更新参数**，只搬运）。"""
    policy_state = state["policy"]
    if not isinstance(policy_state, dict) or not policy_state:
        raise ExportError("state.policy 必须是非空 state_dict")
    return {k: v.detach().clone() for k, v in policy_state.items()}


def deterministic_raw_action(policy, obs: np.ndarray) -> np.ndarray:
    """与受控链路**逐字相同**的确定性均值动作（有界 raw 动作）。"""
    from evaluation.controlled_run import deterministic_action

    return deterministic_action(policy, obs)


def _build_eval_env(origin: int, *, master_seed: int, config: dict):
    """用与训练**同一套**参数构造 train env（种子按冻结配置派生）。"""
    from safe_rl_v2.formal_train_loop import build_train_env

    return build_train_env(int(origin), master_seed=int(master_seed), config=config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_eval_input_from_training_checkpoint",
        description="训练恢复 checkpoint → 评估输入 → 统一评估（只读源，不更新参数）")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--exported-checkpoint", default=None,
                        help="导出件路径；默认 runs/<run-id>/eval_input.pt")
    parser.add_argument("--origin", type=int, default=None,
                        help="评估用的 train origin；默认取源 checkpoint 的第 0 个 origin")
    parser.add_argument("--seed", type=int, default=0, help="master seed（决定环境种子与 RNG）")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from checkpointing.eval_input import (
        CONTROLLED_ROLE,
        load_evaluation_checkpoint,
        save_evaluation_checkpoint,
    )
    from evaluation.adapter import evaluate
    from evaluation.controlled_run import source_ledger_hashes
    from evaluation.inventory import capture_inventory
    from evaluation.sources import (
        canonical_source_digests,
        verify_evaluation_input_sources,
    )
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S
    from runs.writer import git_revision, write_run
    from safe_rl.corrector_wrapper import CorrectorWrapper
    from safe_rl_v2.buffer import ACTION_DIM
    from safe_rl_v2.formal_train_loop import (
        build_seeded_policy,
        load_frozen_training_config,
        train_env_seeds,
    )

    base = REPO_ROOT / args.base_dir
    src = Path(args.source_checkpoint)
    if not src.is_absolute():
        src = REPO_ROOT / src
    exported = (Path(args.exported_checkpoint) if args.exported_checkpoint
                else base / args.run_id / "eval_input.pt")
    if not exported.is_absolute():
        exported = REPO_ROOT / exported
    command = ("python -m scripts.export_eval_input_from_training_checkpoint "
               f"--source-checkpoint {args.source_checkpoint} --run-id {args.run_id} "
               f"--seed {args.seed}")

    report: dict[str, Any] = {}
    if not src.is_file():
        print(f"源 checkpoint 不存在：{src}", file=sys.stderr)
        return 1
    source_sha_before = _sha256_file(src)

    try:
        config = load_frozen_training_config()
        meta, state = read_training_checkpoint(src)
        origins = [int(o) for o in state.get("origins") or []]
        if not origins:
            raise ExportError("训练恢复 checkpoint 缺少 origins（无法确定评估用 origin）")
        origin = int(args.origin) if args.origin is not None else origins[0]

        obs_dim = int(meta["obs_dim"])
        policy_state = export_policy_state(state)
        policy = build_seeded_policy(config, obs_dim=obs_dim, seed=int(args.seed))
        policy.load_state_dict(policy_state)
        policy.eval()

        # 确定性动作一致性：源 policy 对象 vs 导出后重新加载的 policy 对象
        torch.manual_seed(0)
        probe_obs = torch.randn(8, obs_dim, dtype=torch.float32)
        obs_np = probe_obs.numpy()
        src_actions = np.stack([deterministic_raw_action(policy, o) for o in obs_np])

        # 导出前先取 source ledger 与种子
        env_probe = _build_eval_env(origin, master_seed=int(args.seed), config=config)
        digests = canonical_source_digests()
        seed_offsets = train_env_seeds(config, master_seed=int(args.seed))
        env_seeds = {
            "task": int(seed_offsets["task_seed"]),
            "server": int(seed_offsets["server_seed"]),
            "forecast": int(seed_offsets["forecast_seed"]),
        }
        injection = env_probe.formal_injection
        start = str(injection.start)
        source_ledger = {
            "dependency_lock_hash": None, "data_hash": None,
            "scenario_hash": str(injection.provenance_hash),
        }
        lock = REPO_ROOT / "uv.lock"
        source_ledger["dependency_lock_hash"] = _sha256_file(lock)
        source_ledger["data_hash"] = hashlib.sha256(json.dumps(
            [[d.role, d.logical_path, d.sha256] for d in digests],
            sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        del env_probe

        checkpoint_info = save_evaluation_checkpoint(
            exported,
            policy=policy,
            obs_dim=obs_dim,
            code_revision=git_revision(),
            artifact_role=CONTROLLED_ROLE,
            action_mode="deterministic_mean",
            policy_config={
                "hidden": int(config["training"]["policy"]["hidden"]),
                "hidden_layers": int(config["training"]["policy"]["hidden_layers"]),
                "activation": str(config["training"]["policy"]["activation"]),
                "obs_dim": obs_dim, "action_dim": ACTION_DIM,
            },
            train_split="train",
            train_origin=int(origin),
            train_start=start,
            seeds=env_seeds,
            sources=[digest.model_dump() for digest in digests],
        )

        # 经**正式 loader** 加载并校验来源，再用**统一评估器**跑同一 episode
        loaded = load_evaluation_checkpoint(exported)
        verify_evaluation_input_sources(loaded)
        loaded_actions = np.stack(
            [deterministic_raw_action(loaded.policy, o) for o in obs_np])
        actions_match = bool(np.array_equal(src_actions, loaded_actions))

        eval_env = CorrectorWrapper(
            _build_eval_env(origin, master_seed=int(args.seed), config=config),
            corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S)
        record = evaluate(
            eval_env, METHOD, lambda obs: deterministic_raw_action(loaded.policy, obs),
            run_id=args.run_id, service_standard=None, seed=int(args.seed),
            action_mode=loaded.action_mode, checkpoint_id=str(exported),
            checkpoint_role=loaded.artifact_role)
        inventory = capture_inventory(eval_env, record)
        ledger_hashes = source_ledger_hashes(eval_env)
        source_sha_after = _sha256_file(src)
        if source_sha_after != source_sha_before:
            raise ExportError("源 checkpoint 在本次运行中被改写（必须只读）")

        report = {
            "entry": "python -m scripts.export_eval_input_from_training_checkpoint",
            "statement": STATEMENT,
            "claims": dict(CLAIMS),
            "training_scope": str(state.get("training_scope")),
            "source_checkpoint": {
                "path": str(src.relative_to(REPO_ROOT)),
                "sha256": source_sha_before,
                "schema_hash": str(meta["schema_hash"]),
                "code_revision": str(meta.get("code_revision", "")),
                "artifact_role": str(state.get("artifact_role")),
                "unchanged_during_run": True,
                "parameter_updates": 0,
            },
            "exported_checkpoint": dict(checkpoint_info),
            "role_rule": ("受控短跑源只标 controlled_short_run_eval_input；"
                          "**不**冒充已审核的 formal_training_policy"),
            "origin": int(origin),
            "start": start,
            "seeds": env_seeds,
            "master_seed": int(args.seed),
            "source_ledger_hashes": dict(ledger_hashes),
            "deterministic_action_consistency": {
                "probe_observations": int(obs_np.shape[0]),
                "source_vs_exported_equal": actions_match,
                "max_abs_diff": float(np.max(np.abs(src_actions - loaded_actions))),
            },
            "inventory": inventory.to_dict(),
            "evaluation": {
                "method": record.method,
                "steps": int(record.steps),
                "service_qualified": record.service_qualified,
                "purchase_cost_sgd": float(record.purchase_cost_sgd),
                "carbon_kg_co2e": float(record.carbon_kg_co2e),
                "on_time_task_rate": record.service.on_time_task_rate,
                "end_leftover_work": record.service.end_leftover_work,
                "non_interruptible_interruption_count":
                    record.service.non_interruptible_interruption_count,
            },
            "does_not_claim": "只证明接线可复现；不宣称训练性能、收敛或策略优劣。",
            "code_revision": git_revision(),
        }
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(args.run_id, config={"run_id": args.run_id, "status": "failed"},
                  metrics=pd.DataFrame(), base_dir=str(base), seed=int(args.seed),
                  command=command,
                  report={"entry": "python -m scripts.export_eval_input_from_training_checkpoint",
                          "claims": dict(CLAIMS),
                          "failure": f"{type(exc).__name__}: {exc}"},
                  status="failed", failure_classification=type(exc).__name__)
        print(f"导出/复现失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    metrics = pd.DataFrame([{
        "origin": int(origin), "steps": int(record.steps),
        "purchase_cost_sgd": float(record.purchase_cost_sgd),
        "carbon_kg_co2e": float(record.carbon_kg_co2e),
        "initial_soc": inventory.initial_soc, "final_soc": inventory.final_soc,
        "initial_energy_kwh": inventory.initial_energy_kwh,
        "final_energy_kwh": inventory.final_energy_kwh,
        "final_soc_deviation": inventory.final_soc_deviation,
        "terminal_soc_recovery_kwh": inventory.terminal_soc_recovery_kwh,
        "service_qualified": record.service_qualified,
        "actions_match": report["deterministic_action_consistency"][
            "source_vs_exported_equal"],
    }])
    run_path = write_run(
        args.run_id,
        config={"entry": "python -m scripts.export_eval_input_from_training_checkpoint",
                "run_id": args.run_id, "status": "success",
                "claims": dict(CLAIMS),
                "source_checkpoint_sha256": source_sha_before,
                "exported_checkpoint_sha256": checkpoint_info["sha256"],
                "artifact_role": checkpoint_info["artifact_role"],
                "parameter_updates": 0,
                "seeds": env_seeds, "code_revision": git_revision()},
        metrics=metrics, report=report, base_dir=str(base), seed=int(args.seed),
        command=command, status="success",
        dependency_lock_hash=source_ledger["dependency_lock_hash"],
        data_hash=source_ledger["data_hash"],
        scenario_hash=source_ledger["scenario_hash"],
        manifest_metadata={"artifact_role": checkpoint_info["artifact_role"]})

    print(f"run 产物：{run_path}")
    consistency = report["deterministic_action_consistency"]
    print("  源 checkpoint : " + report["source_checkpoint"]["sha256"][:16] + "…"
          "（本次运行只读未变：" + str(source_sha_before == source_sha_before) + "）")
    print("  导出件        : " + checkpoint_info["sha256"][:16] + "…"
          " role=" + str(checkpoint_info["artifact_role"]))
    print("  确定性动作一致: " + str(consistency["source_vs_exported_equal"])
          + "（max|Δ|=" + str(consistency["max_abs_diff"]) + "）")
    print(f"  库存          : initial_soc={inventory.initial_soc:.4f} "
          f"final_soc={inventory.final_soc:.4f} (目标 {inventory.soc_target:.4f} / "
          f"容差 {inventory.soc_final_tolerance:.4f})")
    print("  参数更新      : 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
