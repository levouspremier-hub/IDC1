"""M1.3g-e-b：B6 arrival-to-Task 的**确定性 mapper**（纯函数）。

把**已验证** v3 驱动表的**逐槽 realized aggregate** 1:1 分割成
`idc_model.task.Task` stream。

## 不可变式

1. **1:1 守恒**：每槽与全 episode 的**固定点整数账本**之和**精确等于**该槽的
   原始 aggregate（`aggregate × work_unit_scale`）；float `Task.workload`
   只是**运行时**表示。
2. **不缩放、不合并、不移动、不丢弃** arrival。
3. **forecast 不得进入 Task truth**：每槽只读该槽的 **realized** aggregate；
   `expected_arrival_forecast()` 仅供对照，**不参与**任何 Task 构造。
4. **因果**：未来 slot 的 aggregate 改变**不得**影响此前 slot 的 Task。
5. **确定性**：相同 verified 输入 + manifest + origin + horizon + seed + revision
   ⇒ 同一 Task stream 与 canonical content hash。
6. `initial backlog` **不进入**本账本。

## 已批准参数（Route-A，**人工签核**，本模块只**读取**）

唯一新增 profile `E_micro_inference`：`duration=1 step`、`deadline=2 steps`、
`load_range=[0.04,0.25]`、`priority∈[2.6,3.4]`、`interruptible=parallelizable=false`。
本版本正式 B6 slots **优先** E。参数值一律来自
`data/manifest/m13g_arrival_mapper_v1.json`，**不在代码中另设默认**。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from idc_model.task import Task

REPO_ROOT = Path(__file__).resolve().parent.parent

MAPPER_MANIFEST_SCHEMA = "m1.3g-e-b-arrival-mapper-v1"
CANONICAL_MANIFEST_NAME = "m13g_arrival_mapper_v1.json"

DEFAULT_MANIFEST_LOGICAL = "data/manifest/m13g_arrival_mapper_v1.json"

# **本模块实现**（revision 与 dirty 检查使用**同一**集合）
ARRIVAL_MAPPER_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/arrival_mapper.py",
    "scripts/materialize_b6_arrival_mapper.py",
    "scenario/exogenous_drivers_b6.py",
    "scenario/b6_split_manifests.py",
    "scenario/b6_refs.py",
    "scenario/splits.py",
    "idc_model/task.py",
)


class ArrivalMapperError(ValueError):
    """mapper 的**明确失败**（无 fallback、不静默）。"""


# --- 数据类 ---------------------------------------------------------------------

@dataclass(frozen=True)
class MappedTask:
    """一个 Task 的**确定性**表示（含整数账本，供守恒对账）。"""

    task: Task
    slot_index: int
    ledger_work_micro: int


@dataclass(frozen=True)
class MappedSlot:
    slot_index: int
    aggregate_micro: int
    tasks: tuple[MappedTask, ...]


@dataclass(frozen=True)
class ArrivalTaskStream:
    split: str
    origin: int
    horizon: int
    seed: int
    slots: tuple[MappedSlot, ...]
    content_hash: str

    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(mt.task for slot in self.slots for mt in slot.tasks)


# --- Git / 路径 ----------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    return bool(_git("status", "--porcelain", "--", *ARRIVAL_MAPPER_SOURCE_PATHS).strip())


def mapper_code_revision() -> str:
    revision = _git("log", "-1", "--format=%H", "--",
                    *ARRIVAL_MAPPER_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ArrivalMapperError(f"mapper revision 无效：{revision!r}")
    return revision


def _canonical_manifest_dir() -> Path:
    """canonical manifest 目录（**私有**；测试只能 monkeypatch 它）。"""
    return REPO_ROOT / "data" / "manifest"


def canonical_mapper_manifest_path() -> Path:
    return _canonical_manifest_dir() / CANONICAL_MANIFEST_NAME


def _sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise ArrivalMapperError(f"冻结资产不可读：{path}：{error}") from error


# --- 已验证的正式链 -------------------------------------------------------------

def load_verified_mapper_chain(
    *,
    split_manifest_path: Path | str | None = None,
    refs_path: Path | str | None = None,
) -> dict:
    """**唯一**的信任链入口：B6 policy / v3 bundle / `refs_v4` / v5 split。

    全部走各自的**正式 loader**；副本、symlink、旧 v1–v4 split、
    `refs_v3`/`refs.json`、伪造 hash/revision、dirty source 一律 fail closed。
    """
    from scenario.arrival_intensity_policy import load_verified_b6_policy
    from scenario.b6_refs import load_verified_refs_v4
    from scenario.b6_split_manifests import load_verified_split_manifest_v5
    from scenario.exogenous_drivers_b6 import load_verified_v3_bundle

    if _generator_is_dirty():
        raise ArrivalMapperError(
            "mapper 实现有未提交修改：拒绝用旧 revision 为未提交代码背书")

    b6_policy = load_verified_b6_policy()
    bundle = load_verified_v3_bundle()
    refs = load_verified_refs_v4(refs_path) if refs_path is not None \
        else load_verified_refs_v4()
    splits = {
        name: load_verified_split_manifest_v5(
            split_manifest_path if split_manifest_path is not None else None,
            expected_split=name)
        for name in ("train", "validation", "test")
    } if split_manifest_path is None else {
        "train": load_verified_split_manifest_v5(
            split_manifest_path, expected_split="train")
    }
    return {"b6_policy": b6_policy, "bundle": bundle, "refs": refs, "splits": splits}


def verified_realized_aggregate() -> np.ndarray:
    """**realized** aggregate（唯一可进 Task truth 的量），来自已验证 v3 bundle。"""
    from scenario.exogenous_drivers_b6 import load_verified_v3_bundle

    frame = load_verified_v3_bundle()["frame"]
    return np.asarray(frame["arrival"], dtype=np.int64)


def expected_arrival_forecast(timestamps=None) -> np.ndarray:
    """**期望** forecast（仅供对照，**绝不**参与 Task 构造）。"""
    from scenario.formal_scenario_b6 import (
        expected_arrival_for_timestamps,
        verified_v3_template,
    )

    if timestamps is None:
        frame = pd.read_parquet(
            REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet")
        timestamps = pd.DatetimeIndex(frame["timestamp"])
    return expected_arrival_for_timestamps(pd.DatetimeIndex(timestamps),
                                           verified_v3_template())


def c_idc_base_work_per_hour() -> float:
    """B6 冻结硬件实现（`server_seed=0`）的 `C_IDC_base`。"""
    from idc_model.task_model import IDCEnergyTaskModel

    return float(IDCEnergyTaskModel(task_seed=0, server_seed=0)
                 ._task_workload_capacity_ref())


# --- manifest -----------------------------------------------------------------

def load_verified_mapper_manifest(path: Path | str | None = None) -> dict:
    """加载并**严格校验** canonical mapper manifest（无 fallback）。"""
    if path is None:
        path = canonical_mapper_manifest_path()
    else:
        given = Path(path).absolute()
        expected = canonical_mapper_manifest_path().absolute()
        if given != expected:
            raise ArrivalMapperError(
                f"mapper manifest 必须是**唯一 canonical** 文件 "
                f"{expected}；实际 {given}（不接受副本、别名、symlink）")
        path = expected
    if Path(path).is_symlink():
        raise ArrivalMapperError(f"mapper manifest 不得是 symlink：{path}")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArrivalMapperError(f"mapper manifest 不可读：{error}") from error
    return validate_mapper_manifest(payload)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArrivalMapperError(message)


def _require_exact_keys(mapping: object, *, field: str,
                        expected: tuple[str, ...]) -> dict:
    if not isinstance(mapping, dict):
        raise ArrivalMapperError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    if actual != set(expected):
        raise ArrivalMapperError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - set(expected))} 缺少={sorted(set(expected) - actual)}")
    return mapping


MANIFEST_TOP_KEYS: tuple[str, ...] = (
    "schema", "contract_version", "approved_decision_id", "approved_parameters",
    "profiles", "profile_order", "delta_t_hours", "work_unit_scale",
    "abs_tol_work", "rel_tol_work", "max_tasks_per_slot", "max_growth_rounds",
    "deterministic_algorithm", "seed_policy", "units", "sources",
    "source_revision", "note", "frozen_at_utc", "e_work_bounds",
)
PROFILE_KEYS: tuple[str, ...] = (
    "duration_steps", "deadline_steps", "load_range", "priority_range",
    "interruptible", "parallelizable", "source",
)


def validate_mapper_manifest(payload: object) -> dict:
    """manifest 的**唯一**严格校验入口。"""
    payload = _require_exact_keys(payload, field="mapper manifest",
                                  expected=MANIFEST_TOP_KEYS)
    _require(payload["schema"] == MAPPER_MANIFEST_SCHEMA,
             f"schema 必须是 {MAPPER_MANIFEST_SCHEMA!r}")
    prof = _require_exact_keys(payload["profiles"]["E_micro_inference"],
                               field="profiles.E_micro_inference",
                               expected=PROFILE_KEYS)
    _require(prof["duration_steps"] == 1 and prof["deadline_steps"] == 2,
             "E 的 duration/deadline 必须是 1 / 2 steps")
    _require(list(prof["load_range"]) == [0.04, 0.25], "E 的 load_range 必须是 [0.04, 0.25]")
    _require(list(prof["priority_range"]) == [2.6, 3.4], "E 的 priority_range 必须是 [2.6, 3.4]")
    _require(prof["interruptible"] is False and prof["parallelizable"] is False,
             "E 必须是 interruptible=false / parallelizable=false")
    _require(prof["source"] == "modeled_scenario", "E 的来源必须是 modeled_scenario")
    _require(payload["work_unit_scale"] == 1_000_000, "work_unit_scale 必须是 1e6")
    _require(payload["delta_t_hours"] == 0.5, "delta_t_hours 必须是 0.5")
    _require(payload["max_tasks_per_slot"] == 4, "max_tasks_per_slot 必须是 4")
    return payload


def _profile(payload: dict, key: str) -> dict:
    profiles = payload["profiles"]
    _require(key in profiles, f"manifest 缺少 profile {key!r}")
    return profiles[key]


# --- 整数账本与分割 -------------------------------------------------------------

def _bounds_micro(payload: dict, key: str = "E_micro_inference") -> tuple[int, int]:
    """把 profile 的 float 可行域换算成**整数 micro-work** 的**闭区间**。

    取 `ceil(w_min × scale)` 与 `floor(w_max × scale)`，因此任何落在整数区间内的
    账本值，其 float 表示**必然**落在原始 float 区间内。
    """
    scale = int(payload["work_unit_scale"])
    prof = _profile(payload, key)
    C = c_idc_base_work_per_hour()
    d = float(payload["delta_t_hours"])
    steps = int(prof["duration_steps"])
    lo, hi = float(prof["load_range"][0]), float(prof["load_range"][1])
    w_min = steps * lo * C * d
    w_max = steps * hi * C * d
    return math.ceil(w_min * scale), math.floor(w_max * scale)


def split_slot_aggregate_micro(aggregate_micro: int) -> tuple[int, ...]:
    """把**一个槽**的整数 aggregate 分成 1..N 个合法的整数 work 值。

    守恒精确；任一不可行即 **fail closed**（不 clip、不缩放、不丢弃）。
    """
    payload = load_verified_mapper_manifest()
    if isinstance(aggregate_micro, bool) or not isinstance(aggregate_micro, int):
        raise ArrivalMapperError(f"aggregate_micro 必须是整数，实际 {aggregate_micro!r}")
    if aggregate_micro <= 0:
        raise ArrivalMapperError(
            f"aggregate_micro 必须严格为正（不得创建零/负 workload Task），"
            f"实际 {aggregate_micro}")
    lo, hi = _bounds_micro(payload)
    max_tasks = int(payload["max_tasks_per_slot"])
    max_rounds = int(payload["max_growth_rounds"])

    n_min = math.ceil(aggregate_micro / hi)
    n_max = aggregate_micro // lo
    n = max(n_min, 1)
    rounds = 0
    while n <= min(n_max, max_tasks):
        if rounds > max_rounds:
            break
        base, rem = divmod(aggregate_micro, n)
        parts = tuple(base + 1 if i < rem else base for i in range(n))
        if all(lo <= p <= hi for p in parts):
            return parts
        n += 1
        rounds += 1
    raise ArrivalMapperError(
        f"槽 aggregate={aggregate_micro} micro-work 在 [{lo}, {hi}]、"
        f"最多 {max_tasks} 个任务 / {max_rounds} 轮下**无法覆盖**："
        "不得 clip、不得缩放、不得丢弃（fail closed）"
    )


# --- 确定性 id / priority -------------------------------------------------------

def _stable_key(*parts: Any) -> str:
    return hashlib.sha256(
        "|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def stable_task_id(split: str, origin: int, slot: int, ordinal: int, seed: int) -> int:
    """由 (split, origin, slot, ordinal, seed) 的**canonical 编码**导出。

    **禁止** Python `hash()`（进程间不稳定）。取 SHA-256 的前 8 字节 → 63 位正整数。
    """
    digest = _stable_key("task", split, origin, slot, ordinal, seed)
    return int(digest[:16], 16) >> 1


def deterministic_priority(split: str, origin: int, slot: int, ordinal: int,
                           seed: int, priority_range: tuple[float, float]) -> float:
    """由**显式 episode seed** 在批准区间内确定性生成（与调用顺序无关）。"""
    digest = _stable_key("priority", split, origin, slot, ordinal, seed)
    rng = np.random.default_rng(int(digest[:16], 16))
    lo, hi = float(priority_range[0]), float(priority_range[1])
    return float(rng.uniform(lo, hi))


# --- 构造 ---------------------------------------------------------------------

def build_arrival_task_stream(
    split: str,
    *,
    start: str,
    horizon: int,
    seed: int,
) -> ArrivalTaskStream:
    """把 `[origin, origin + horizon)` 的 realized aggregate 1:1 映射成 Task stream。

    `start` 必须是 canonical 30 分钟网格上的带时区 ISO 时间，精确映射到该 split
    的**本地 origin**（越界 / 非网格 / 非候选 origin 一律 fail closed）。
    """
    from scenario.b6_split_manifests import local_origin_from_start
    from scenario.splits import validate_episode_origin

    payload = load_verified_mapper_manifest()
    if _generator_is_dirty():
        raise ArrivalMapperError(
            "mapper 实现有未提交修改：拒绝用旧 revision 为未提交代码背书")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ArrivalMapperError(f"horizon 必须是正整数，实际 {horizon!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ArrivalMapperError(f"seed 必须是整数，实际 {seed!r}")

    origin = local_origin_from_start(split, start)
    global_origin = validate_episode_origin(split, origin, horizon)

    aggregate = verified_realized_aggregate()
    scale = int(payload["work_unit_scale"])
    prof = _profile(payload, "E_micro_inference")
    C = c_idc_base_work_per_hour()
    d = float(payload["delta_t_hours"])
    steps = int(prof["duration_steps"])
    deadline_steps = int(prof["deadline_steps"])
    priority_range = (float(prof["priority_range"][0]), float(prof["priority_range"][1]))
    seen_ids: set[int] = set()

    slots: list[MappedSlot] = []
    for offset in range(horizon):
        slot_index = global_origin + offset
        agg_units = int(aggregate[slot_index])
        if agg_units <= 0:
            raise ArrivalMapperError(
                f"槽 {slot_index} 的 realized aggregate = {agg_units}："
                "不得创建零/负 workload Task（fail closed）")
        agg_micro = agg_units * scale
        parts = split_slot_aggregate_micro(agg_micro)
        mapped: list[MappedTask] = []
        for ordinal, work_micro in enumerate(parts):
            task_id = stable_task_id(split, origin, slot_index, ordinal, seed)
            if task_id in seen_ids:
                raise ArrivalMapperError(f"task_id 碰撞：{task_id}")
            seen_ids.add(task_id)
            workload = work_micro / scale
            utilization = workload / (steps * C * d)
            load_profile = np.full(steps, utilization, dtype=np.float64)
            rebuilt = float(np.sum(load_profile) * C * d)
            diff = abs(rebuilt - workload)
            if diff > max(float(payload["abs_tol_work"]),
                          float(payload["rel_tol_work"]) * abs(workload)):
                raise ArrivalMapperError(
                    f"task {task_id} 的 workload 与载荷曲线不自洽："
                    f"差 {diff} 超容差")
            task = Task(
                task_id=task_id,
                profile_key="E_micro_inference",
                name="E_micro_inference",
                arrival_time=slot_index - global_origin,
                duration=steps,
                load_profile=load_profile,
                workload=workload,
                deadline=deadline_steps,
                priority=deterministic_priority(
                    split, origin, slot_index, ordinal, seed, priority_range),
                interruptible=bool(prof["interruptible"]),
                parallelizable=bool(prof["parallelizable"]),
            )
            mapped.append(MappedTask(task=task, slot_index=slot_index,
                                     ledger_work_micro=work_micro))
        if sum(m.ledger_work_micro for m in mapped) != agg_micro:
            raise ArrivalMapperError(f"槽 {slot_index} 的整数账本不守恒")
        slots.append(MappedSlot(slot_index=slot_index, aggregate_micro=agg_micro,
                                tasks=tuple(mapped)))

    stream = ArrivalTaskStream(
        split=split, origin=origin, horizon=horizon, seed=seed,
        slots=tuple(slots),
        content_hash="",  # 先占位，再按 canonical 表示计算
    )
    return ArrivalTaskStream(
        split=stream.split, origin=stream.origin, horizon=stream.horizon,
        seed=stream.seed, slots=stream.slots,
        content_hash=canonical_content_hash(stream),
    )


def canonical_content_hash(stream: ArrivalTaskStream) -> str:
    """基于**canonical 整数表示**的 content hash（不含平台相关浮点序列化）。"""
    canon = {
        "split": stream.split,
        "origin": stream.origin,
        "horizon": stream.horizon,
        "seed": stream.seed,
        "slots": [
            {
                "slot_index": slot.slot_index,
                "aggregate_micro": slot.aggregate_micro,
                "tasks": [
                    {"task_id": mt.task.task_id,
                     "ledger_work_micro": mt.ledger_work_micro}
                    for mt in slot.tasks
                ],
            }
            for slot in stream.slots
        ],
    }
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --- 物化（由 scripts/materialize_b6_arrival_mapper.py 调用） ------------------

def build_mapper_manifest(*, frozen_at_utc: str) -> dict:
    from scenario.arrival_intensity_policy import load_verified_b6_policy
    from scenario.b6_refs import load_verified_refs_v4
    from scenario.b6_split_manifests import load_verified_split_manifest_v5

    # 逐层调用正式 loader：链不通过即 fail closed（值本身由 mapper 读取时再取）
    load_verified_b6_policy()
    refs_path = REPO_ROOT / "configs/frozen_refs/refs_v4.json"
    load_verified_refs_v4(refs_path)
    for name in ("train", "validation", "test"):
        load_verified_split_manifest_v5(expected_split=name)
    C = c_idc_base_work_per_hour()
    d = 0.5
    lo, hi = 1 * 0.04 * C * d, 1 * 0.25 * C * d
    payload = {
        "schema": MAPPER_MANIFEST_SCHEMA,
        "contract_version": "contract-v9",
        "approved_decision_id": "M1.3f-e-b-d",
        "approved_parameters": {
            "profile_name": "E_micro_inference",
            "duration_hours": 0.5,
            "load_range": [0.04, 0.25],
            "deadline_hours": 1.0,
            "priority_range": [2.6, 3.4],
            "interruptible": False,
            "parallelizable": False,
            "expansion": "new_profile_do_not_modify_ABCD",
            "max_tasks_per_slot": 4,
            "max_growth_rounds": 4,
            "work_unit_scale": 1_000_000,
            "abs_tol_work": 1e-9,
            "rel_tol_work": 1e-12,
            "profile_order": [
                "E_micro_inference", "A_inference", "B_rl_training",
                "C_dl_training", "D_preprocess",
            ],
            "source": "modeled_scenario",
        },
        "profiles": {
            "E_micro_inference": {
                "duration_steps": 1, "deadline_steps": 2,
                "load_range": [0.04, 0.25], "priority_range": [2.6, 3.4],
                "interruptible": False, "parallelizable": False,
                "source": "modeled_scenario",
            },
        },
        "profile_order": [
            "E_micro_inference", "A_inference", "B_rl_training",
            "C_dl_training", "D_preprocess",
        ],
        "delta_t_hours": d,
        "work_unit_scale": 1_000_000,
        "abs_tol_work": 1e-9,
        "rel_tol_work": 1e-12,
        "max_tasks_per_slot": 4,
        "max_growth_rounds": 4,
        "deterministic_algorithm": (
            "n = smallest integer in [ceil(A/hi), floor(A/lo)] with "
            "ceil(A/n)..floor(A/n) within [lo,hi]; Hamilton split over n; "
            "fail closed beyond max_tasks_per_slot / max_growth_rounds"
        ),
        "seed_policy": "explicit per-episode seed; priority = f(split,origin,slot,ordinal,seed)",
        "units": {
            "work": "work-unit", "micro": "work-unit x 1e-6",
            "c_idc_base": "work-unit/hour", "delta_t_hours": "hour",
        },
        "sources": {
            "b6_intensity_policy": {
                "path": "data/manifest/m13f_arrival_intensity_policy_v1.json",
                "sha256": _sha256_file(
                    REPO_ROOT
                    / "data/manifest/m13f_arrival_intensity_policy_v1.json"),
            },
            "forecast_policy_v3": {
                "path": "data/manifest/singapore_2024_forecast_policy_v3.json",
                "sha256": _sha256_file(
                    REPO_ROOT
                    / "data/manifest/singapore_2024_forecast_policy_v3.json"),
            },
            "frozen_refs_v4": {
                "path": "configs/frozen_refs/refs_v4.json",
                "sha256": _sha256_file(refs_path),
            },
            "exogenous_v3_parquet": {
                "path": "data/processed/singapore_2024/exogenous_drivers_v3.parquet",
                "sha256": _sha256_file(
                    REPO_ROOT
                    / "data/processed/singapore_2024/exogenous_drivers_v3.parquet"),
            },
            "formal_split_v5_train": {
                "path": "data/manifest/formal_splits_v5/train.json",
                "sha256": _sha256_file(
                    REPO_ROOT / "data/manifest/formal_splits_v5/train.json"),
            },
        },
        "source_revision": mapper_code_revision(),
        "note": (
            "B6 arrival-to-Task 确定性 mapper 的冻结参数（Route-A 人工签核）。"
            "realized aggregate 是**唯一**任务账本输入；forecast 只作对照。"
            "参数仅供本卡读取，不在此处重新选择。"
        ),
        "frozen_at_utc": frozen_at_utc,
        "e_work_bounds": {"w_min": lo, "w_max": hi},
    }
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise ArrivalMapperError("mapper manifest 含绝对路径")
    return payload


# --- 原子写入 helper（物化器复用） ----------------------------------------------

def atomic_write_bytes(path: Path, body: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "ArrivalMapperError",
    "ArrivalTaskStream",
    "MAPPER_MANIFEST_SCHEMA",
    "MappedSlot",
    "MappedTask",
    "build_arrival_task_stream",
    "build_mapper_manifest",
    "c_idc_base_work_per_hour",
    "canonical_content_hash",
    "canonical_mapper_manifest_path",
    "deterministic_priority",
    "expected_arrival_forecast",
    "load_verified_mapper_chain",
    "load_verified_mapper_manifest",
    "mapper_code_revision",
    "split_slot_aggregate_micro",
    "stable_task_id",
    "validate_mapper_manifest",
    "verified_realized_aggregate",
]
