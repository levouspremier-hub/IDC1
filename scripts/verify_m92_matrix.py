"""M9.2：实验矩阵的**只读**校验器。

从 **live** 已验签来源重算一切可重算的量，与矩阵逐项比对；**不写任何文件**。

```bash
uv run python -m scripts.verify_m92_matrix \
    --matrix configs/experiments/m9_experiment_matrix_v1.json
```

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
    check("实际比例可重算",
          all(abs(ratios[s] - (splits[s]["candidate_origins"]["end_exclusive"]
                               - splits[s]["candidate_origins"]["start"]) // 48
               / sum((splits[x]["candidate_origins"]["end_exclusive"]
                      - splits[x]["candidate_origins"]["start"]) // 48 for x in SPLITS))
              < 1e-12 for s in SPLITS),
          str({s: round(ratios[s], 4) for s in SPLITS}))
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.verify_m92_matrix",
        description="M9.2 实验矩阵只读校验器（从 live 来源重算并逐项比对）")
    parser.add_argument("--matrix", default="configs/experiments/m9_experiment_matrix_v1.json")
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
