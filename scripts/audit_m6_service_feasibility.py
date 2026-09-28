"""M6-P1-F1：train-only 服务达标线**可行性审计**（可重算，只读 train）。

复用既有链路，不新增任何环境 / 物理 / 评估语义：

```text
scripts.calibrate_training_config.select_origins()  → 24 个预先选定的 train origin
scripts.calibrate_training_config.build_env_for_origin() → 正式 train env（固定种子）
scripts.calibrate_training_config.reference_action()  → 固定提案 compute ∈ {1.0,0.5,0.25}，储能 0
planning.corrector.PRODUCTION_CORRECTOR_TIME_LIMIT_S      → corrector on，生产默认 0.25 s
evaluation.adapter.evaluate()            → M6-P1 评估器（service_standard=None）
```

**三种提案是可行性参考轨迹**，不是五方法的正式基线，也不是训练结果。
**本脚本不传入服务标准**：正式 `service_qualified` 恒为 `None`（未判定）——
项目标准 `m6-service-standard-v1` 已由 M6-P1-F5 冻结，但本脚本仍显式传
`service_standard=None`，以保持与历史 run 的可比性。
本脚本另出一列**「提案门槛诊断」**，按协议 §2 的**提案值**逐项判定，供人工裁决。

结论只依据这 24 个 train origin 与固定参考提案，**不外推**为全策略可达性证明。

```bash
uv run python -m scripts.audit_m6_service_feasibility --run-id <id>
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# 协议 §2 的**提案值**（未冻结；本卡只用于「提案门槛诊断」，不写入任何 frozen 契约）
from evaluation.service_standard import (  # noqa: E402
    PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX,
    PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX,
    PROPOSED_ON_TIME_TASK_RATE_MIN,
    PROPOSED_ON_TIME_WORK_RATE_MIN,
)

CLAIMS = {"trained": False, "performance_evaluated": False, "convergence_claimed": False}
METHOD = "reference_proposal_fixed_compute"
STATEMENT = (
    "train-only 服务达标线可行性审计：24 个预先选定的 train origin × 三种固定参考提案"
    "（corrector on，生产默认 0.25 s）。**不是**训练结果，**不是**五方法基线；"
    "本脚本不显式传入服务标准 ⇒ 正式 service_qualified 恒为未判定"
    "（项目标准 m6-service-standard-v1 已冻结，但不被隐式采用）。"
)

DIAGNOSIS_PASS = "pass"
DIAGNOSIS_FAIL = "fail"
DIAGNOSIS_UNDETERMINED = "undetermined"
CONCLUSION_OBSERVED = "样本中观察到达标轨迹"
CONCLUSION_NONE = "本次参考轨迹未展示达标"
CONCLUSION_INSUFFICIENT = "证据不足"

DIAGNOSIS_COLUMNS = (
    "on_time_task_rate",
    "on_time_work_rate",
    "end_leftover_work_fraction",
    "non_interruptible_interruption_count",
)


def _cmp(value: float | None, threshold: float, *, ge: bool) -> str:
    if value is None:
        return DIAGNOSIS_UNDETERMINED
    ok = (value >= threshold) if ge else (value <= threshold)
    return DIAGNOSIS_PASS if ok else DIAGNOSIS_FAIL


def proposal_threshold_diagnosis(record) -> dict[str, Any]:
    """**提案门槛诊断**：逐项按提案值判定（不是正式的 `service_qualified`）。"""
    service = record.service
    items = {
        "on_time_task_rate": _cmp(service.on_time_task_rate,
                                  PROPOSED_ON_TIME_TASK_RATE_MIN, ge=True),
        "on_time_work_rate": _cmp(service.on_time_work_rate,
                                  PROPOSED_ON_TIME_WORK_RATE_MIN, ge=True),
        "end_leftover_work_fraction": _cmp(service.end_leftover_work_fraction,
                                           PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX, ge=False),
        "non_interruptible_interruption_count": _cmp(
            float(service.non_interruptible_interruption_count),
            float(PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX), ge=False),
    }
    if DIAGNOSIS_FAIL in items.values():
        joint = DIAGNOSIS_FAIL
    elif DIAGNOSIS_UNDETERMINED in items.values():
        joint = DIAGNOSIS_UNDETERMINED
    else:
        joint = DIAGNOSIS_PASS
    return {"items": items, "joint": joint}


def _gaps(record) -> dict[str, float | None]:
    """未达标时逐项的**差距**（无差距 / 不可判定时为 None）。"""
    service = record.service
    out: dict[str, float | None] = {}
    if service.on_time_task_rate is not None:
        out["on_time_task_rate"] = PROPOSED_ON_TIME_TASK_RATE_MIN - service.on_time_task_rate
    else:
        out["on_time_task_rate"] = None
    if service.on_time_work_rate is not None:
        out["on_time_work_rate"] = PROPOSED_ON_TIME_WORK_RATE_MIN - service.on_time_work_rate
    else:
        out["on_time_work_rate"] = None
    if service.end_leftover_work_fraction is not None:
        out["end_leftover_work_fraction"] = (
            service.end_leftover_work_fraction - PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX)
    else:
        out["end_leftover_work_fraction"] = None
    out["non_interruptible_interruption_count"] = float(
        service.non_interruptible_interruption_count
        - PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX)
    return out


def audit_origin_proposal(origin: int, compute: float) -> dict[str, Any]:
    """一个 (origin, 提案) 的完整计量（走既有正式链与 M6-P1 评估器）。"""
    from evaluation.adapter import evaluate
    from evaluation.controlled_run import source_ledger_hashes
    from planning.corrector import PRODUCTION_CORRECTOR_TIME_LIMIT_S, resolve_corrector_budget
    from safe_rl.corrector_wrapper import CorrectorWrapper
    from scripts.calibrate_training_config import (
        build_env_for_origin,
        reference_action,
        start_for_origin,
    )

    env, _injection = build_env_for_origin(origin)
    horizon = int(env.horizon)
    wrapped = CorrectorWrapper(
        env, corrector_time_limit_s=PRODUCTION_CORRECTOR_TIME_LIMIT_S)
    action = reference_action(compute, env.action_dim, env.model.N)
    fixed = np.asarray(action, dtype=np.float32).copy()

    record = evaluate(
        wrapped, METHOD, lambda _obs: fixed,
        run_id=f"origin{origin}_compute{compute}", service_standard=None, seed=0,
        action_mode="deterministic_mean",
        checkpoint_id=None, checkpoint_role=None,
    )
    budget, source = resolve_corrector_budget(None, enabled=True)
    diagnosis = proposal_threshold_diagnosis(record)
    physical = record.physical
    row = {
        "origin": int(origin),
        "start": start_for_origin(origin),
        "compute": float(compute),
        "storage": 0.0,
        "horizon": horizon,
        "steps": int(record.steps),
        "episode_complete": bool(
            record.steps == horizon and record.failure_classification is None),
        "failure_classification": record.failure_classification,
        # 业务服务
        "due_in_episode_tasks": record.service.due_in_episode_tasks,
        "due_in_episode_work": record.service.due_in_episode_work,
        "on_time_task_rate": record.service.on_time_task_rate,
        "on_time_work_rate": record.service.on_time_work_rate,
        "end_leftover_work": record.service.end_leftover_work,
        "end_leftover_work_fraction": record.service.end_leftover_work_fraction,
        "non_interruptible_interruption_count":
            record.service.non_interruptible_interruption_count,
        "failed_tasks": record.service.failed_tasks,
        # 物理违规（四类）
        "access_limit_violation_steps": physical.access_limit_violation_steps,
        "soc_violation_steps": physical.soc_violation_steps,
        "charge_discharge_exclusion_violations":
            physical.charge_discharge_exclusion_violations,
        "energy_conservation_violations": physical.energy_conservation_violations,
        # 修正器失败面
        "timeout_count": None if record.correction is None else record.correction.timeout_count,
        "zero_action_fallback_count":
            None if record.correction is None else record.correction.zero_action_fallback_count,
        # **提案门槛诊断**（不是 service_qualified）
        **{f"diag_{k}": v for k, v in diagnosis["items"].items()},
        "diag_joint": diagnosis["joint"],
        **{f"gap_{k}": v for k, v in _gaps(record).items()},
        # 正式资格：本卡**不冻结**标准 ⇒ 恒为未判定
        "service_qualified": record.service_qualified,
        "service_standard_id": record.service_standard_id,
        "corrector_time_limit_s": float(budget),
        "corrector_time_limit_source": source,
        "source_ledger_hashes": source_ledger_hashes(env),
    }
    return row


def _conclusion(rows: list[dict[str, Any]]) -> tuple[str, str]:
    joints = [r["diag_joint"] for r in rows]
    if DIAGNOSIS_PASS in joints:
        return CONCLUSION_OBSERVED, "至少一条参考轨迹同时满足四项提案门槛。"
    if DIAGNOSIS_UNDETERMINED in joints:
        return CONCLUSION_INSUFFICIENT, (
            "存在不可判定的门槛分量（零分母），无法对全部参考轨迹作达标结论。")
    return CONCLUSION_NONE, (
        "本批 24 个 train origin × 三种固定参考提案均未同时满足四项提案门槛；"
        "**这不等于**任何策略都无法达标（参考提案不是策略可达性的上界）。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.audit_m6_service_feasibility",
        description="train-only 服务达标线可行性审计（M6-P1-F1）")
    parser.add_argument("--run-id", default="m6_service_feasibility_train_v1")
    parser.add_argument("--base-dir", default="runs")
    args = parser.parse_args(argv)

    from runs.writer import git_revision, write_run
    from scripts.calibrate_training_config import PROPOSALS, select_origins

    origins = select_origins()
    base = REPO_ROOT / args.base_dir
    command = f"python -m scripts.audit_m6_service_feasibility --run-id {args.run_id}"

    try:
        rows: list[dict[str, Any]] = []
        for origin in origins:
            for compute in PROPOSALS:
                rows.append(audit_origin_proposal(origin, compute))
        conclusion, conclusion_note = _conclusion(rows)
        joint_pass = [r for r in rows if r["diag_joint"] == DIAGNOSIS_PASS]
        ledger = rows[0]["source_ledger_hashes"]
        report = {
            "entry": "python -m scripts.audit_m6_service_feasibility",
            "statement": STATEMENT,
            "claims": dict(CLAIMS),
            "split": "train",
            "origins": [int(o) for o in origins],
            "proposals": [float(p) for p in PROPOSALS],
            "proposal_role": (
                "固定参考提案（compute=1.0/0.5/0.25，储能 0）——可行性参考轨迹，"
                "**不是**五方法基线，**不是**训练结果"),
            "service_standard": {
                "frozen": False,
                "service_qualified": "未判定（本卡不冻结阈值）",
                "proposed_on_time_task_rate_min": PROPOSED_ON_TIME_TASK_RATE_MIN,
                "proposed_on_time_work_rate_min": PROPOSED_ON_TIME_WORK_RATE_MIN,
                "proposed_end_leftover_work_fraction_max":
                    PROPOSED_END_LEFTOVER_WORK_FRACTION_MAX,
                "proposed_non_interruptible_interruption_max":
                    PROPOSED_NON_INTERRUPTIBLE_INTERRUPTION_MAX,
                "diagnosis_column": "提案门槛诊断（diag_* / diag_joint）",
            },
            "conclusion": conclusion,
            "conclusion_note": conclusion_note,
            "joint_pass_count": len(joint_pass),
            "joint_pass_dates": [r["start"] for r in joint_pass],
            "per_proposal_joint_pass": {
                str(p): sum(1 for r in rows
                            if r["compute"] == p and r["diag_joint"] == DIAGNOSIS_PASS)
                for p in PROPOSALS
            },
            "per_item_pass": {
                name: sum(1 for r in rows if r[f"diag_{name}"] == DIAGNOSIS_PASS)
                for name in DIAGNOSIS_COLUMNS
            },
            "episodes_incomplete": sum(1 for r in rows if not r["episode_complete"]),
            "timeouts": sum(int(r["timeout_count"] or 0) for r in rows),
            "zero_action_fallbacks":
                sum(int(r["zero_action_fallback_count"] or 0) for r in rows),
            "physical_violation_totals": {
                key: sum(int(r[key]) for r in rows) for key in (
                    "access_limit_violation_steps", "soc_violation_steps",
                    "charge_discharge_exclusion_violations",
                    "energy_conservation_violations")
            },
            "rows": rows,
            "code_revision": git_revision(),
        }
    except Exception as exc:  # noqa: BLE001 - 失败也如实记录
        write_run(args.run_id, config={"run_id": args.run_id, "status": "failed"},
                  metrics=pd.DataFrame(), base_dir=str(base), seed=0, command=command,
                  report={"entry": "python -m scripts.audit_m6_service_feasibility",
                          "claims": dict(CLAIMS),
                          "failure": f"{type(exc).__name__}: {exc}"},
                  status="failed", failure_classification=type(exc).__name__)
        print(f"审计失败：{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1

    config = {
        "entry": "python -m scripts.audit_m6_service_feasibility",
        "run_id": args.run_id,
        "status": "success",
        "split": "train",
        "origins": [int(o) for o in origins],
        "proposals": [float(p) for p in PROPOSALS],
        "corrector_mode": "on",
        "corrector_time_limit_s": rows[0]["corrector_time_limit_s"],
        "corrector_time_limit_source": rows[0]["corrector_time_limit_source"],
        "service_standard_frozen": False,
        "claims": dict(CLAIMS),
        "source_ledger_hashes": ledger,
        "code_revision": git_revision(),
    }
    metrics = pd.DataFrame([{k: v for k, v in r.items() if k != "source_ledger_hashes"}
                            for r in rows])
    run_path = write_run(
        args.run_id, config=config, metrics=metrics, report=report, base_dir=str(base),
        seed=0, command=command, status="success",
        dependency_lock_hash=ledger["dependency_lock_hash"],
        data_hash=ledger["data_hash"], scenario_hash=ledger["scenario_hash"])

    print(f"run 产物：{run_path}")
    print(f"结论：{conclusion}")
    print(f"  联合达标（提案门槛诊断）: {report['joint_pass_count']} / {len(rows)}"
          f"  逐提案 {report['per_proposal_joint_pass']}")
    print(f"  逐项 pass 数（共 {len(rows)}）: {report['per_item_pass']}")
    print(f"  未完成 episode {report['episodes_incomplete']} / 超时 {report['timeouts']}"
          f" / 回退 {report['zero_action_fallbacks']}")
    print(f"  物理违规合计: {report['physical_violation_totals']}")
    print(f"  dependency_lock_hash={ledger['dependency_lock_hash']}")
    print(f"  data_hash={ledger['data_hash']}")
    print(f"  scenario_hash={ledger['scenario_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
