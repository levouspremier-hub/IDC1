"""M9.2：实验矩阵的**只读**校验器。

从 **live** 已验签来源重算一切可重算的量，与矩阵逐项比对；**不写任何文件**。

```bash
uv run python -m scripts.verify_m92_matrix \
    --matrix configs/experiments/m9_experiment_matrix_v1.json
```

支持 **v1** 与 **v2** 两种矩阵：先跑对两者都成立的通用检查（§1–§9），
再按 `schema` 追加版本专属检查（v2 见 §10）。

校验内容（全部来自矩阵与 live 文件，不依赖矩阵自述）：

1. 六个来源文件 + 服务标准模块的 **live SHA-256** 与矩阵记录一致；
2. v5 三个 split 的 `split_rows` 区间**互不重叠**且首尾相接；
3. 评估清单：validation **61** 日、test **92** 日，逐条 origin 从 `candidate_origins.start`
   起每 48 步一个，时间轴 = `time_range.start + 30 min × local_origin`，**均落在本 split 内**；
   且清单摘要与矩阵记录一致；
4. test 不同日数 **≥ 30**；
5. 训练安排：origin 池 = train 内日对齐 origin（212 个），批次清单按规则逐项重建后
   摘要一致，且覆盖全部池内 origin；训练 seed **≥ 3**（本矩阵为 0/1/2）；
6. 五方法席位齐备、全部为 `planned/not_runnable`；
7. 服务标准 ID / 四项阈值、训练预算、refs 关键数值与 live 一致；
8. `formal_training_ready` 为 `false`；状态为 `frozen_matrix`；
9. **比例差异已如实登记**（不得声称已满足 60/20/20）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_METHODS = (
    "rule_baseline",
    "independent_rolling_optimization",
    "penalty_ppo",
    "safe_ppo_single_step_corrector",
    "safe_ppo_joint_rolling_corrector",
)
SPLITS = ("train", "validation", "test")


class MatrixVerificationError(ValueError):
    """矩阵与 live 来源不一致。"""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(rows: Any) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise MatrixVerificationError(f"缺少文件：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MatrixVerificationError(f"{path} 不是合法 JSON：{error}") from error
    if not isinstance(payload, dict):
        raise MatrixVerificationError(f"{path} 必须是 JSON object")
    return payload


def build_expected_episodes(split_manifest: dict) -> list[dict]:
    """**只**用 split manifest 的 time_range / candidate_origins 生成评估清单。"""
    start = datetime.fromisoformat(split_manifest["time_range"]["start"])
    lo = int(split_manifest["candidate_origins"]["start"])
    hi = int(split_manifest["candidate_origins"]["end_exclusive"])
    return [
        {"local_origin": local,
         "start": (start + timedelta(minutes=30 * local)).isoformat()}
        for local in range(lo, hi, 48)
    ]


def build_expected_training_rows(matrix: dict, pool: list[int]) -> list[list[int]]:
    rule = matrix["training_schedule"]
    batches = int(rule["batches_per_seed"])
    per_batch = int(rule["episodes_per_batch"])
    return [[b, s, pool[(per_batch * b + s) % len(pool)]]
            for b in range(batches) for s in range(per_batch)]


def verify(matrix: dict) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    # 1. 来源 hash -----------------------------------------------------------------
    for role, entry in sorted(matrix["sources"].items()):
        live = _sha256_file(REPO_ROOT / entry["logical_path"])
        check(f"source sha256 [{role}]", live == entry["sha256"],
              f"{entry['logical_path']} live={live[:16]}…")

    splits = {s: _load(REPO_ROOT / matrix["sources"][f"formal_split_v5_{s}"]["logical_path"])
              for s in SPLITS}
    refs = _load(REPO_ROOT / matrix["sources"]["refs_v4"]["logical_path"])
    mapper = _load(REPO_ROOT / matrix["sources"]["m13g_arrival_mapper_v1"]["logical_path"])
    train_cfg = _load(REPO_ROOT / matrix["sources"]["training_config_v1"]["logical_path"])
    release = _load(REPO_ROOT / matrix["sources"]["env_release_v1"]["logical_path"])

    # 2. split 不重叠 ---------------------------------------------------------------
    rows = {s: splits[s]["split_rows"] for s in SPLITS}
    contiguous = all(rows[a]["end_exclusive"] == rows[b]["start"]
                     for a, b in zip(SPLITS, SPLITS[1:], strict=False))
    check("三个 split 行区间首尾相接且不重叠", contiguous,
          f"train={rows['train']['start']}..{rows['train']['end_exclusive']} "
          f"val={rows['validation']['start']}..{rows['validation']['end_exclusive']} "
          f"test={rows['test']['start']}..{rows['test']['end_exclusive']}")

    # 3. 评估清单（只由 manifest 元数据重建）---------------------------------------
    for split, expected_count in (("validation", 61), ("test", 92)):
        expected = build_expected_episodes(splits[split])
        recorded = matrix["evaluation_schedule"][split]["episodes"]
        check(f"{split} 评估日数 == 记录值 ({expected_count})",
              len(recorded) == expected_count == len(expected),
              f"矩阵={len(recorded)} 重算={len(expected)} 期望={expected_count}")
        rebuilt = [{"split": split, "local_origin": e["local_origin"], "start": e["start"]}
                   for e in expected]
        check(f"{split} 清单逐条可重建（origin/时间轴）", recorded == rebuilt,
              "" if recorded == rebuilt else "矩阵与重算清单不一致")
        end_excl = datetime.fromisoformat(splits[split]["time_range"]["end_exclusive"])
        start_ts = [datetime.fromisoformat(e["start"]) for e in expected]
        check(f"{split} 全部 episode 落在本 split 时间范围内",
              all(datetime.fromisoformat(splits[split]["time_range"]["start"]) <= t
                  < end_excl for t in start_ts),
              f"首个={start_ts[0].isoformat()} 末个={start_ts[-1].isoformat()}")

    eval_rows = [[e["split"], e["local_origin"], e["start"]]
                 for e in matrix["evaluation_schedule"]["validation"]["episodes"]
                 + matrix["evaluation_schedule"]["test"]["episodes"]]
    check("评估清单摘要可重算",
          _canonical_digest(eval_rows) == matrix["evaluation_schedule"]["episode_list_digest"],
          matrix["evaluation_schedule"]["episode_list_digest"][:16] + "…")

    # 4. test 不同日数 ≥ 30 ---------------------------------------------------------
    distinct_test = len({e["start"] for e in
                         matrix["evaluation_schedule"]["test"]["episodes"]})
    check("test 不同评估日 ≥ 30", distinct_test >= 30,
          f"distinct={distinct_test} ≥ {matrix['evaluation_schedule']['min_required_test_days']}")

    # 5. 训练安排 ------------------------------------------------------------------
    pool = list(range(splits["train"]["candidate_origins"]["start"],
                      splits["train"]["candidate_origins"]["end_exclusive"], 48))
    check("训练 origin 池 == train 内日对齐 origin",
          pool == matrix["training_schedule"]["origin_pool"],
          f"重算 {len(pool)} 个，矩阵 {len(matrix['training_schedule']['origin_pool'])} 个")
    rebuilt_rows = build_expected_training_rows(matrix, pool)
    check("训练批次清单可逐项重建（摘要一致）",
          _canonical_digest(rebuilt_rows)
          == matrix["training_schedule"]["batch_origin_list_digest"],
          matrix["training_schedule"]["batch_origin_list_digest"][:16] + "…")
    drawn = {r[2] for r in rebuilt_rows}
    check("批次清单覆盖全部池内 origin（无遗漏）", drawn == set(pool),
          f"覆盖 {len(drawn)}/{len(pool)}")
    seeds = [int(s) for s in matrix["training_schedule"]["seed_derivation"]["train_seeds"]]
    check("训练 seed 不少于 3 个", len(seeds) >= 3, f"seeds={seeds}")
    check("训练 seed 与冻结训练配置一致",
          seeds == [int(x) for x in train_cfg["training"]["scale"]["train_seeds"]],
          f"矩阵={seeds}")
    check("批数 / 每批 episode 与冻结训练配置一致",
          int(matrix["training_schedule"]["batches_per_seed"])
          == int(train_cfg["training"]["scale"]["batches_per_seed"])
          and int(matrix["training_schedule"]["episodes_per_batch"])
          == int(train_cfg["training"]["sampling"]["episodes_per_batch"]),
          f"batches={matrix['training_schedule']['batches_per_seed']} "
          f"per_batch={matrix['training_schedule']['episodes_per_batch']}")
    check("受控短跑顺序未被冒充为正式长训顺序",
          "not_the_controlled_short_run" in matrix["training_schedule"], "")

    # 6. 五方法席位 ----------------------------------------------------------------
    ids = [m["method_id"] for m in matrix["methods"]]
    check("五方法席位齐备且顺序与评估器一致", ids == list(EXPECTED_METHODS), f"{ids}")
    check("全部方法为 planned/not_runnable 且不可运行",
          all(m["status"] == "planned/not_runnable" and m["runnable_now"] is False
              for m in matrix["methods"]), "")

    # 7. 服务标准 / 预算 / refs 绑定 ------------------------------------------------
    from evaluation.service_standard import FROZEN_PROJECT_SERVICE_STANDARD as STD

    std = matrix["scenario_freeze"]["service_standard"]
    check("服务标准 ID == m6-service-standard-v1", std["standard_id"] == "m6-service-standard-v1",
          std["standard_id"])
    check("服务标准四项阈值与 live 一致",
          (std["frozen"] is True and std["on_time_task_rate_min"] == STD.on_time_task_rate_min
           and std["on_time_work_rate_min"] == STD.on_time_work_rate_min
           and std["end_leftover_work_fraction_max"] == STD.end_leftover_work_fraction_max
           and std["non_interruptible_interruption_max"]
           == STD.non_interruptible_interruption_max), "")
    check("训练预算与冻结训练配置一致",
          matrix["scenario_freeze"]["training_config_v1"]["budgets"]
          == train_cfg["training"]["budgets"], "")
    live_refs = {k: v["value"] for k, v in refs["references"].items()}
    recorded_refs = {k: v["value"]
                     for k, v in matrix["scenario_freeze"]["capacity_and_normalization"].items()}
    check("归一化参考值（refs_v4）与 live 一致", recorded_refs == live_refs,
          f"{len(recorded_refs)} 项")
    check("任务/期限规则取自 arrival mapper（不手填）",
          matrix["scenario_freeze"]["task_and_deadline_rules"]["max_tasks_per_slot"]
          == mapper["max_tasks_per_slot"]
          and matrix["scenario_freeze"]["task_and_deadline_rules"]["e_work_bounds"]
          == mapper["e_work_bounds"], "")

    # 8. 冻结状态与放行 ------------------------------------------------------------
    check("矩阵状态 == frozen_matrix", matrix["status"] == "frozen_matrix", matrix["status"])
    check("formal_training_ready == false（矩阵冻结 ≠ 训练放行）",
          matrix["formal_training_ready"] is False
          and release["readiness"]["formal_training_ready"] is False, "")
    check("release readiness 与 live 一致",
          matrix["release_readiness"] == release["readiness"], "")

    # 9. 比例差异已如实登记 --------------------------------------------------------
    deviation = matrix["split_freeze"]["registered_deviation"]
    check("比例差异已登记且未声称满足 60/20/20",
          "60/20/20" in deviation["summary"]
          and "不声称已满足" in deviation["summary"]
          and deviation["decision"].startswith("保留"),
          deviation["summary"][:60] + "…")
    ratios = matrix["split_freeze"]["actual_origin_day_ratio"]
    # 10. v2 专属（M9.2-R1）------------------------------------------------------
    if str(matrix.get("schema", "")).endswith("-v2"):
        _verify_v2(matrix, checks, check)
    if str(matrix.get("schema", "")).endswith("-v3"):
        _verify_v2(matrix, checks, check)   # v3 沿用 v2 的全部约定
        _verify_v3(matrix, checks, check)

    check("实际比例可重算",
          all(abs(ratios[s] - (splits[s]["candidate_origins"]["end_exclusive"]
                               - splits[s]["candidate_origins"]["start"]) // 48
               / sum((splits[x]["candidate_origins"]["end_exclusive"]
                      - splits[x]["candidate_origins"]["start"]) // 48 for x in SPLITS))
              < 1e-12 for s in SPLITS),
          str({s: round(ratios[s], 4) for s in SPLITS}))
    return checks


NON_LEARNING_METHODS = ("rule_baseline", "independent_rolling_optimization")
PPO_METHODS = ("penalty_ppo", "safe_ppo_single_step_corrector",
               "safe_ppo_joint_rolling_corrector")


def _verify_v2(matrix: dict, checks: list, check) -> None:
    """M9.2-R1 的四处返修在 v2 中的落地校验。"""
    # ① 方法所需产物
    by_id = {m["method_id"]: m for m in matrix["methods"]}
    for mid in NON_LEARNING_METHODS:
        arts = " ".join(by_id[mid]["required_artifacts"])
        check(f"[v2] {mid} 不要求 PPO checkpoint / 训练批次",
              by_id[mid]["trains_ppo"] is False
              and by_id[mid]["has_training_seed"] is False
              and "PPO checkpoint" not in arts and "训练批次" not in arts,
              f"trains_ppo={by_id[mid]['trains_ppo']}")
        check(f"[v2] {mid} 显式登记「不需要」清单",
              by_id[mid]["explicitly_not_required"] == [
                  "PPO 训练批次（512 批 × seed）", "21 维 PPO checkpoint"], "")
    for mid in PPO_METHODS:
        arts = " ".join(by_id[mid]["required_artifacts"])
        check(f"[v2] {mid} 仍要求按新 21 维契约训练 + 已审核 checkpoint",
              by_id[mid]["trains_ppo"] is True
              and by_id[mid]["has_training_seed"] is True
              and "21 维" in arts and "checkpoint" in arts, "")

    # ② 配对键与分层
    pairing = matrix["pairing"]
    check("[v2] 配对键 == (split, episode_start, scenario_seed)",
          pairing["primary_pairing_key"] == ["split", "episode_start", "scenario_seed"],
          str(pairing["primary_pairing_key"]))
    check("[v2] 预登记主场景 scenario_seed == 0",
          pairing["preregistered_scenario_seed"] == 0, "")
    check("[v2] 训练 seed 只作分层、不进入配对键",
          "training_seed" not in pairing["primary_pairing_key"]
          and "分层" in pairing["training_seed_role"], "")
    check("[v2] 基线在同一场景只运行一次、不计作三个独立样本",
          pairing["baseline_runs_once_per_scenario"] is True
          and "不得计作三次独立基线运行" in pairing["baseline_pairing_rule"], "")
    check("[v2] 独立样本计数口径写明",
          "不**增加独立样本数" in pairing["sample_counting"]
          or "不增加独立样本数" in pairing["sample_counting"], "")

    # ③ 库存公平门禁
    fair = matrix["fair_cost_carbon_pairing"]
    conditions = " ".join(fair["conditions_all_required"])
    for token in ("完整运行 48 步", "service_qualified", "物理违规", "容量相同",
                  "bess_soc_final_tolerance", "PHYSICS_ABS_TOL"):
        check(f"[v2] 公平配对条件含 {token}", token in conditions, "")
    check("[v2] 条件不满足时不生成公平收益、原始结果仍保留",
          "不生成" in fair["on_failure"] and "完整保留" in fair["on_failure"], "")
    check("[v2] 库存字段清单含初末 SOC / 容量 / 容差 / 结算量",
          {"initial_soc", "final_soc", "capacity_kwh", "soc_final_tolerance",
           "terminal_soc_recovery_kwh"} <= set(fair["recorded_fields"]), "")
    check("[v2] terminal_soc_recovery_kwh 仅为诊断、不冒充购电或 SGD",
          "不**冒充" in fair["terminal_soc_recovery_is_diagnostic_only"]
          or "不冒充" in fair["terminal_soc_recovery_is_diagnostic_only"], "")
    check("[v2] 服务资格语义不变、库存是独立字段",
          "service_qualified` 原义保留" in fair["service_qualified_semantics_unchanged"], "")
    check("[v2] 日 episode 单位写明（非跨日连续库存实验）",
          "独立的 48 步日 episode" in fair["episode_unit"]
          and "不是**跨日连续库存实验" in fair["episode_unit"].replace("**", "**"), "")

    # ④ 训练 checkpoint → 评估输入接线
    wiring = matrix["eval_input_wiring"]
    check("[v2] 登记导出入口且参数更新为 0",
          wiring["entry"] ==
          "python -m scripts.export_eval_input_from_training_checkpoint"
          and int(wiring["parameter_updates"]) == 0, wiring["entry"])
    check("[v2] 受控短跑源不得冒充 formal_training_policy",
          "controlled_short_run_eval_input" in wiring["role_rule"]
          and "formal_training_policy" in wiring["role_rule"]
          and "不得" in wiring["role_rule"], "")
    check("[v2] 记录源/导出件 SHA、scope、来源、种子与 revision",
          {"源 checkpoint SHA-256", "导出件 SHA-256", "training_scope", "来源 hash",
           "种子", "代码 revision"} <= set(wiring["recorded"]), "")

    # v2 取代 v1：v1 文件必须仍在，且取代关系写明
    sup = matrix.get("supersedes") or {}
    v1_path = REPO_ROOT / str(sup.get("path", ""))
    check("[v2] 明确取代 v1 且 v1 文件保留可读",
          v1_path.is_file()
          and sup.get("sha256") == _sha256_file(v1_path)
          and "取代" in str(sup.get("relation", "")), str(sup.get("path")))
    unchanged = next((v for k, v in sup.items() if k.startswith("unchanged_from_")), [])
    check("[v2/v3] 逐字沿用上一版的冻结数值清单已登记",
          len(unchanged) >= 5, f"{len(unchanged)} 项")


PPO_TRAINING_TOKENS = ("PPO 训练配置", "512 批", "PPO checkpoint")


def _verify_v3(matrix: dict, checks: list, check) -> None:
    """M9.2-R2：非学习方法席位的 `blocking_reasons` 必须与其字段自洽。"""
    for mid in NON_LEARNING_METHODS:
        method = next(m for m in matrix["methods"] if m["method_id"] == mid)
        blockers = " ".join(method["blocking_reasons"])
        offending = [tok for tok in PPO_TRAINING_TOKENS if tok in blockers]
        check(f"[v3] {mid} 的 blocking_reasons 不提 PPO 训练口径",
              offending == [] and method["trains_ppo"] is False,
              f"命中={offending}" if offending else "只列自身尚缺的方法配置/实现/证据")
        check(f"[v3] {mid} 登记「不涉及 PPO 训练」的说明",
              "trains_ppo=false" in str(method.get("blocking_reasons_note", "")), "")

    for mid in PPO_METHODS:
        method = next(m for m in matrix["methods"] if m["method_id"] == mid)
        arts = " ".join(method["required_artifacts"])
        check(f"[v3] {mid} 仍要求按新 21 维契约训练 + 已审核 checkpoint",
              method["trains_ppo"] is True and "21 维" in arts and "checkpoint" in arts, "")

    history = (matrix.get("supersedes") or {}).get("history") or []
    versions = {str(entry.get("version")): str(entry.get("status")) for entry in history}
    check("[v3] v1/v2 均标为历史版本",
          versions.get("v1") == "historical" and versions.get("v2") == "historical",
          str(versions))
    for entry in history:
        path = REPO_ROOT / str(entry.get("path", ""))
        check(f"[v3] 历史版本 {entry.get('version')} 文件保留且 sha 一致",
              path.is_file() and entry.get("sha256") == _sha256_file(path),
              str(entry.get("path")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.verify_m92_matrix",
        description="M9.2 实验矩阵只读校验器（从 live 来源重算并逐项比对）")
    parser.add_argument("--matrix", default="configs/experiments/m9_experiment_matrix_v3.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.matrix)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        matrix = _load(path)
        checks = verify(matrix)
    except MatrixVerificationError as error:
        print(f"矩阵校验失败：{error}", file=sys.stderr)
        return 1

    failed = [name for name, ok, _ in checks if not ok]
    if not args.quiet:
        for name, ok, detail in checks:
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    print(f"\n矩阵：{path}")
    print(f"检查项 {len(checks)}；通过 {len(checks) - len(failed)}；失败 {len(failed)}")
    if failed:
        print(f"失败项：{failed}", file=sys.stderr)
        return 1
    print("结论：矩阵与 live 来源逐项一致 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
