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


def verify_source_checkpoint_provenance(
    *, source_state: dict, source_meta: dict, config: dict, origin: int,
    live_injection_provenance: str,
) -> dict[str, Any]:
    """**逐项核对**源训练恢复 checkpoint 自带的来源与当前 live 资产（M9.2-R2）。

    这是 R1 复审第 2 项的修复：导出**不得**用当前 live 账本替换源 checkpoint 自带的来源。
    任何一项不一致 ⇒ 明确失败（**不**导出），绝不「用当前资产 hash 包装旧权重」。
    """
    import json as _json

    from evaluation.sources import canonical_source_digests

    ledger = source_state.get("source_ledger")
    if not isinstance(ledger, dict):
        raise ExportError("源 checkpoint 缺少 state.source_ledger")
    frozen = source_state.get("frozen_config")
    if not isinstance(frozen, dict):
        raise ExportError("源 checkpoint 缺少 state.frozen_config")

    live_lock = _sha256_file(REPO_ROOT / "uv.lock")
    digests = canonical_source_digests()
    live_data = hashlib.sha256(_json.dumps(
        [[d.role, d.logical_path, d.sha256] for d in digests],
        sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    provenance = source_state.get("origin_provenance") or {}
    recorded_origin_prov = provenance.get(str(origin))
    if recorded_origin_prov is None:
        raise ExportError(f"源 checkpoint 的 origin_provenance 不含 origin {origin}")

    checks = {
        "frozen_config_matches_current": {
            "source": _json.dumps(frozen, sort_keys=True, ensure_ascii=False)[:64] + "…",
            "live": _json.dumps(config, sort_keys=True, ensure_ascii=False)[:64] + "…",
            "match": bool(frozen == config),
        },
        "dependency_lock_hash": {
            "source": str(ledger.get("dependency_lock_hash")),
            "live_recomputed": live_lock,
            "match": bool(str(ledger.get("dependency_lock_hash")) == live_lock),
        },
        "data_hash": {
            "source": str(ledger.get("data_hash")),
            "live_recomputed": live_data,
            "match": bool(str(ledger.get("data_hash")) == live_data),
        },
        "origin_provenance": {
            "source": str(recorded_origin_prov),
            "live_recomputed": str(live_injection_provenance),
            "match": bool(str(recorded_origin_prov) == str(live_injection_provenance)),
        },
    }
    failing = sorted(name for name, entry in checks.items() if not entry["match"])
    if failing:
        raise ExportError(
            "源 checkpoint 自带的来源与当前 live 资产不一致，**拒绝导出**（不得用当前资产 "
            f"hash 包装旧权重）：{failing}")
    return {
        "checks": checks,
        "all_match": True,
        "source_code_revision": {
            "envelope": str(source_meta.get("code_revision", "")),
            "state": str(source_state.get("code_revision", "")),
            "note": ("源代码 revision 是**历史值**，无法用 live 重算核对；"
                     "此处如实记录，导出件写**当前** revision。"),
        },
        "source_ledger_verbatim": {
            "dependency_lock_hash": str(ledger.get("dependency_lock_hash")),
            "data_hash": str(ledger.get("data_hash")),
            "scenario_hash": str(ledger.get("scenario_hash")),
            "env_seeds": dict(ledger.get("env_seeds") or {}),
            "master_seed": ledger.get("master_seed"),
        },
        "live_recomputed_ledger": {
            "dependency_lock_hash": live_lock,
            "data_hash": live_data,
            "origin_provenance": str(live_injection_provenance),
        },
    }


#: 按构造必然不同、或纯属墙钟测量的字段：**不**作为「源 vs 导出」等价判据。
#: - `run_id` / `checkpoint_id` / `checkpoint_role`：两侧由构造决定地不同；
#: - `correction.solve_time_median_s` / `solve_time_p95_s`：MIP 求解的**墙钟耗时**，
#:   同一轨迹两次运行也会不同，属噪声而非等价性判据。
_NON_EQUIVALENCE_FIELDS = ("run_id", "checkpoint_id", "checkpoint_role")
_NON_EQUIVALENCE_CORRECTION_FIELDS = ("solve_time_median_s", "solve_time_p95_s")


def _record_fingerprint(record: Any) -> dict[str, Any]:
    """完整 `EvaluationRecord` 的可比指纹（剔除上述非等价字段，剔除项在报告中列明）。"""
    dump = record.model_dump()
    for key in _NON_EQUIVALENCE_FIELDS:
        dump.pop(key, None)
    correction = dump.get("correction")
    if isinstance(correction, dict):
        for key in _NON_EQUIVALENCE_CORRECTION_FIELDS:
            correction.pop(key, None)
    return dump


def compare_source_and_exported_records(
    source_record: Any, source_inventory: Any,
    exported_record: Any, exported_inventory: Any,
) -> dict[str, Any]:
    """比较源 policy 与导出后加载的 policy 在**完整 48 步 episode** 上的结果。"""
    src_dump = _record_fingerprint(source_record)
    exp_dump = _record_fingerprint(exported_record)
    differing = sorted(k for k in set(src_dump) | set(exp_dump)
                       if src_dump.get(k) != exp_dump.get(k))
    src_inv = source_inventory.to_dict()
    exp_inv = exported_inventory.to_dict()
    # 库存记录的 `run_id` 由两侧标签构造而来，同样不是等价判据
    src_inv.pop("run_id", None)
    exp_inv.pop("run_id", None)
    inv_differing = sorted(k for k in set(src_inv) | set(exp_inv)
                           if src_inv.get(k) != exp_inv.get(k))
    return {
        "evaluation_record_fields_compared": len(src_dump),
        "evaluation_record_differing_fields": differing,
        "inventory_fields_compared": len(src_inv),
        "inventory_differing_fields": inv_differing,
        "excluded_identity_fields": list(_NON_EQUIVALENCE_FIELDS),
        "excluded_correction_timing_fields": list(_NON_EQUIVALENCE_CORRECTION_FIELDS),
        "excluded_inventory_identity_fields": ["run_id"],
        "exclusion_rationale": (
            "标识字段按构造必然不同；修正器解算耗时是**墙钟**测量，同轨迹两次运行也不同，"
            "属噪声而非等价判据。其余**全部** EvaluationRecord 字段与**全部**库存字段"
            "逐一比较。"),
        "source_record": src_dump,
        "exported_record": exp_dump,
        "source_inventory": src_inv,
        "exported_inventory": exp_inv,
        "identical": bool(not differing and not inv_differing),
    }


def _build_eval_env(origin: int, *, master_seed: int, config: dict):
    """用与训练**同一套**参数构造 train env（种子按冻结配置派生）。

    `build_train_env` 返回 `(env, injection)`；本入口只要 env。
    """
    from safe_rl_v2.formal_train_loop import build_train_env

    env, _injection = build_train_env(int(origin), master_seed=int(master_seed),
                                      config=config)
    return env


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
    parser.add_argument("--scenario-seed", type=int, default=0,
                        help=("共享**场景种子**：评估环境**实际**由它构造"
                              "（环境种子 = scenario_seed + 冻结 seed_offsets），"
                              "配对键记录的也是这个实际值。矩阵预登记为 0。"))
    parser.add_argument("--training-seed", type=int, default=None,
                        help=("源 checkpoint 的**训练 seed**（仅作结果分层与来源核对，"
                              "**不**用于构造评估环境，也**不**进配对键）"))
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from checkpointing.eval_input import (
        CONTROLLED_ROLE,
        load_evaluation_checkpoint,
        save_evaluation_checkpoint,
    )
    from evaluation.adapter import evaluate
    from evaluation.controlled_run import source_ledger_hashes
    from evaluation.fair_pairing_report import (
        EvaluatedSide,
        build_fair_pairing_report,
    )
    from evaluation.inventory import PairingKey, capture_inventory
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
               f"--scenario-seed {args.scenario_seed}"
               + (f" --training-seed {args.training_seed}"
                  if args.training_seed is not None else ""))

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
        env_probe = _build_eval_env(origin, master_seed=int(args.scenario_seed), config=config)
        digests = canonical_source_digests()
        seed_offsets = train_env_seeds(config, master_seed=int(args.scenario_seed))
        env_seeds = {
            "task": int(seed_offsets["task_seed"]),
            "server": int(seed_offsets["server_seed"]),
            "forecast": int(seed_offsets["forecast_seed"]),
        }
        injection = env_probe.formal_injection
        start = str(injection.start)
        # **先核对源 checkpoint 自带的来源**，与当前 live 资产逐项一致才继续
        provenance_check = verify_source_checkpoint_provenance(
            source_state=state, source_meta=meta, config=config, origin=int(origin),
            live_injection_provenance=str(injection.provenance_hash))
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

        # M1.3g-f-c-k 第 5 项：**训练 seed** 只用于核对源 checkpoint 的原始来源，
        # **不**参与构造评估环境、**不**进配对键。
        if args.training_seed is not None:
            src_master = provenance_check["source_ledger_verbatim"].get("master_seed")
            if src_master is None or int(src_master) != int(args.training_seed):
                raise ExportError(
                    f"源 checkpoint 记录的训练 master_seed={src_master!r} 与 "
                    f"--training-seed {args.training_seed} 不一致 ⇒ 拒绝导出")

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

        # **源 policy** 与 **导出后加载的 policy** 各跑一个**完整 48 步 episode**
        # （同一 train origin、同一 scenario_seed、同一 corrector 设置）
        def _run_episode(policy_obj, *, label: str, role: str, checkpoint_id: str):
            raw = _build_eval_env(origin, master_seed=int(args.scenario_seed), config=config)
            wrapped = CorrectorWrapper(
                raw, corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S)
            rec = evaluate(
                wrapped, METHOD, lambda obs: deterministic_raw_action(policy_obj, obs),
                run_id=f"{args.run_id}::{label}", service_standard=None,
                seed=int(args.scenario_seed), action_mode="deterministic_mean",
                checkpoint_id=checkpoint_id, checkpoint_role=role)
            inv = capture_inventory(wrapped, rec)
            return rec, inv, raw

        source_record, source_inventory, source_env = _run_episode(
            policy, label="source_policy", role=str(state.get("artifact_role")),
            checkpoint_id=str(src))
        exported_record, exported_inventory, exported_env = _run_episode(
            loaded.policy, label="exported_policy", role=loaded.artifact_role,
            checkpoint_id=str(exported))
        episode_comparison = compare_source_and_exported_records(
            source_record, source_inventory, exported_record, exported_inventory)
        if not actions_match or not episode_comparison["identical"]:
            raise ExportError(
                "源 policy 与导出 policy 的动作或完整 episode 结果不一致 ⇒ 不写 success："
                f"actions_match={actions_match}, "
                f"record_diff={episode_comparison['evaluation_record_differing_fields']}, "
                f"inventory_diff={episode_comparison['inventory_differing_fields']}")

        # 公平配对出口：两侧同键 (split, episode_start, scenario_seed)
        pairing_key = PairingKey(split="train", episode_start=start,
                                 scenario_seed=int(args.scenario_seed))
        fair_pairing = build_fair_pairing_report(
            EvaluatedSide(label="source_policy", record=source_record,
                          inventory=source_inventory, pairing_key=pairing_key),
            EvaluatedSide(label="exported_policy", record=exported_record,
                          inventory=exported_inventory, pairing_key=pairing_key))

        # 反例核对：**不同配对键**必须被拒绝（不生成公平收益）
        other_key = PairingKey(split="train", episode_start=start,
                               scenario_seed=int(args.scenario_seed) + 1)
        mismatched = build_fair_pairing_report(
            EvaluatedSide(label="source_policy", record=source_record,
                          inventory=source_inventory, pairing_key=pairing_key),
            EvaluatedSide(label="exported_policy", record=exported_record,
                          inventory=exported_inventory, pairing_key=other_key))

        record, inventory = exported_record, exported_inventory
        ledger_hashes = source_ledger_hashes(exported_env)
        source_sha_after = _sha256_file(src)
        if source_sha_after != source_sha_before:
            raise ExportError("源 checkpoint 在本次运行中被改写（必须只读）")

        report = {
            "source_provenance_verification": provenance_check,
            "source_vs_exported_episode": episode_comparison,
            "fair_pairing": fair_pairing,
            "pairing_key_mismatch_control": {
                "note": "不同配对键 ⇒ 必须拒绝配对（公平收益为 null）",
                "result": mismatched,
            },
            "scenario_seed": int(args.scenario_seed),
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
            "scenario_seed_note": (
                "评估环境**实际**由 scenario_seed 构造（环境种子 = scenario_seed + "
                "冻结 seed_offsets）；配对键记录同一实际值。训练 seed 只作分层与来源核对。"),
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
                  metrics=pd.DataFrame(), base_dir=str(base),
                  seed=int(args.scenario_seed),
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
        "source_vs_exported_episode_identical":
            report["source_vs_exported_episode"]["identical"],
        "fair_pairing_eligible": fair_pairing["eligible"],
        "fair_purchase_cost_delta_sgd": fair_pairing["fair_purchase_cost_delta_sgd"],
        "fair_carbon_delta_kg_co2e": fair_pairing["fair_carbon_delta_kg_co2e"],
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
                "seeds": env_seeds, "code_revision": git_revision(),
                "scenario_seed": int(args.scenario_seed),
                "source_training_seed": (None if args.training_seed is None
                                         else int(args.training_seed)),},
        metrics=metrics, report=report, base_dir=str(base), seed=int(args.scenario_seed),
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
    print("  源账本 vs live: 全部一致 =",
          report["source_provenance_verification"]["all_match"])
    print("  源/导出 episode 一致:",
          report["source_vs_exported_episode"]["identical"],
          "（比较 EvaluationRecord 字段",
          report["source_vs_exported_episode"]["evaluation_record_fields_compared"],
          "+ 库存字段", report["source_vs_exported_episode"]["inventory_fields_compared"], "）")
    print("  公平配对键    :", fair_pairing["pairing_key"]["matches"],
          "| eligible =", fair_pairing["eligible"],
          "| Δ购电费 =", fair_pairing["fair_purchase_cost_delta_sgd"],
          "| Δ碳排 =", fair_pairing["fair_carbon_delta_kg_co2e"])
    print("  不同键对照    : eligible =",
          mismatched["eligible"],
          "| Δ购电费 =", mismatched["fair_purchase_cost_delta_sgd"])
    print("  参数更新      : 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
