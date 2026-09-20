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

import functools
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
class MappedSlot:
    """一个槽的映射结果：`tasks` 与 `ledger` **同序等长**。

    `ledger[i]` 是 `tasks[i]` 的**整数**账本值（micro-work）；守恒对账以它为准，
    `Task.workload` 只是运行时 float 表示。
    """

    slot_index: int
    aggregate_micro: int
    tasks: tuple[Task, ...]
    ledger: tuple[int, ...]


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
        return tuple(task for slot in self.slots for task in slot.tasks)

    @property
    def ledger_micro(self) -> tuple[int, ...]:
        return tuple(v for slot in self.slots for v in slot.ledger)


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
    split: str = "train",
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
    # 只验证**本次实际使用**的 split；mapper manifest 另行按 hash 绑定三份 v5。
    splits = {
        split: load_verified_split_manifest_v5(
            split_manifest_path, expected_split=split)
    }
    return {"b6_policy": b6_policy, "bundle": bundle, "refs": refs, "splits": splits}


def _aggregate_from_verified_chain(chain: dict) -> np.ndarray:
    """从**已验证的 chain** 取 realized aggregate（唯一进入 Task truth 的量）。"""
    return np.asarray(chain["bundle"]["frame"]["arrival"], dtype=np.int64)


def verified_realized_aggregate() -> np.ndarray:
    """**realized** aggregate（唯一可进 Task truth 的量）。

    经**唯一**的 `load_verified_mapper_chain()` 取得——**不**另走未绑定的 loader。
    """
    return _aggregate_from_verified_chain(load_verified_mapper_chain())


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
    """B6 冻结硬件实现（`server_seed=0`）的 `C_IDC_base`。

    **只**从已冻结的 mapper manifest 读取——本模块**不**导入 `idc_model.task_model`
    （该文件含既有 mypy 错误且不在 `make check` 的扫描范围内；由**物化器**
    （`scripts/`，不在扫描范围）在冻结时解析一次并写入 manifest）。
    """
    return float(load_verified_mapper_manifest()["c_idc_base_work_per_hour"])


# --- manifest -----------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _live_c_idc_base() -> float:
    """**实测** B6 冻结硬件实现（`server_seed=0`）的 `C_IDC_base`。

    **动态 import**：mypy 不会跟进 `idc_model.task_model`（该文件含既有类型错误
    且不在 `make check` 的扫描范围内），因此不会把那些错误拉进门禁。
    **不修改** `task_model.py`；也**不**仅信任 manifest 自报常数。
    """
    import importlib

    tm = importlib.import_module("idc_model.task_model")
    return float(tm.IDCEnergyTaskModel(task_seed=0, server_seed=0)
                 ._task_workload_capacity_ref())


def load_verified_mapper_manifest(path: Path | str | None = None) -> dict:
    """加载并**严格校验** canonical mapper manifest（无 fallback）。

    **R1**：结构校验之后，由 **trusted live inputs + 当前 live revision + live
    `C_IDC_base` + 锚定冻结时刻**（已验证 `refs_v4` 的 `frozen_at_utc`）
    **重建**候选并**逐字段比较**。因此篡改、新字段、缺字段、旧 revision、
    dirty source、自报 `frozen_at_utc` **一律 fail closed**。
    """
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
    validated = validate_mapper_manifest(payload)

    if _generator_is_dirty():
        raise ArrivalMapperError(
            "mapper 实现有未提交修改：拒绝用旧 revision 为未提交代码背书")

    from scenario.b6_refs import load_verified_refs_v4

    anchor = str(load_verified_refs_v4()["frozen_at_utc"])
    if validated["frozen_at_utc"] != anchor:
        raise ArrivalMapperError(
            f"mapper manifest 的 frozen_at_utc 必须锚定已验证 refs_v4 的冻结时刻 "
            f"{anchor!r}；实际 {validated['frozen_at_utc']!r}")

    rebuilt = build_mapper_manifest(
        frozen_at_utc=anchor, c_idc_base_work_per_hour=_live_c_idc_base())
    if rebuilt != validated:
        differing = sorted(
            key for key in set(rebuilt) | set(validated)
            if rebuilt.get(key) != validated.get(key))
        raise ArrivalMapperError(
            f"mapper manifest 与由 trusted live inputs + live revision 重建的候选不符；"
            f"差异字段={differing}")
    return validated


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
    "c_idc_base_work_per_hour",
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
    _require(isinstance(payload["c_idc_base_work_per_hour"], float)
             and payload["c_idc_base_work_per_hour"] > 0.0,
             "c_idc_base_work_per_hour 必须是正 float")
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
    # 已验签的 payload 里就带着 live-verified 的 C_IDC_base，**不**重复加载 manifest
    C = float(payload["c_idc_base_work_per_hour"])
    d = float(payload["delta_t_hours"])
    steps = int(prof["duration_steps"])
    lo, hi = float(prof["load_range"][0]), float(prof["load_range"][1])
    w_min = steps * lo * C * d
    w_max = steps * hi * C * d
    return math.ceil(w_min * scale), math.floor(w_max * scale)


def split_slot_aggregate_micro(aggregate_micro: int, *,
                               payload: dict | None = None) -> tuple[int, ...]:
    """把**一个槽**的整数 aggregate 分成 1..N 个合法的整数 work 值。

    守恒精确；任一不可行即 **fail closed**（不 clip、不缩放、不丢弃）。

    `payload` 为**已验签**的 mapper manifest；批量调用（如逐槽构造 stream）
    必须传入它，避免对每个槽重复做整链重建比对。
    """
    payload = load_verified_mapper_manifest() if payload is None else payload
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

    chain = load_verified_mapper_chain(split)
    aggregate = _aggregate_from_verified_chain(chain)
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
        parts = split_slot_aggregate_micro(agg_micro, payload=payload)
        mapped: list[Task] = []
        ledger: list[int] = []
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
            mapped.append(task)
            ledger.append(work_micro)
        if sum(ledger) != agg_micro:
            raise ArrivalMapperError(f"槽 {slot_index} 的整数账本不守恒")
        slots.append(MappedSlot(slot_index=slot_index, aggregate_micro=agg_micro,
                                tasks=tuple(mapped), ledger=tuple(ledger)))

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


def _canonical_number(value: Any) -> Any:
    """跨平台稳定的数值 canonical 表示。

    float 用 `float.hex()`（IEEE-754 位级精确、与平台/序列化器无关）；
    bool 与 int 规范为 int（bool **不得**冒充数值）。
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value.hex()
    raise ArrivalMapperError(f"不支持的数值类型：{type(value).__name__}")


def canonical_stream_payload(stream: ArrivalTaskStream) -> dict:
    """完整 Task stream 的**确定性** canonical 表示（供 content hash 使用）。"""
    return {
        "split": stream.split,
        "origin": stream.origin,
        "horizon": stream.horizon,
        "seed": stream.seed,
        "slots": [
            {
                "slot_index": slot.slot_index,
                "aggregate_micro": slot.aggregate_micro,
                "ledger": [_canonical_number(v) for v in slot.ledger],
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "profile_key": task.profile_key,
                        "name": task.name,
                        "arrival_time": task.arrival_time,
                        "duration": task.duration,
                        "deadline": task.deadline,
                        "priority": _canonical_number(task.priority),
                        "interruptible": bool(task.interruptible),
                        "parallelizable": bool(task.parallelizable),
                        "workload": _canonical_number(float(task.workload)),
                        "load_profile": [
                            _canonical_number(float(v))
                            for v in np.asarray(task.load_profile, dtype=np.float64)
                        ],
                    }
                    for task in slot.tasks
                ],
            }
            for slot in stream.slots
        ],
    }


def canonical_content_hash(stream: ArrivalTaskStream) -> str:
    """完整 Task stream 的 content hash（**覆盖所有业务字段**）。

    R1：此前只覆盖 `(task_id, ledger)` 与 slot metadata，因此改
    `priority` / `deadline` / `load_profile` / `profile_key` **不会**改变 hash。
    现覆盖：`task_id` / `profile_key` / `name` / `arrival_time` / `duration` /
    `deadline` / `priority` / `interruptible` / `parallelizable` /
    **完整 `load_profile`** / **整数账本** / slot metadata / split·origin·horizon·seed。
    """
    canon = canonical_stream_payload(stream)
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --- 物化（由 scripts/materialize_b6_arrival_mapper.py 调用） ------------------

def build_mapper_manifest(*, frozen_at_utc: str,
                          c_idc_base_work_per_hour: float) -> dict:
    # **只**按字节 hash 绑定上游；**链的语义验证**由 `load_verified_mapper_chain()`
    # 与 `load_verified_mapper_manifest()` 各自完成（避免同一重验证反复嵌套）。
    refs_path = REPO_ROOT / "configs/frozen_refs/refs_v4.json"
    C = float(c_idc_base_work_per_hour)
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
        "c_idc_base_work_per_hour": C,
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
            "formal_split_v5_validation": {
                "path": "data/manifest/formal_splits_v5/validation.json",
                "sha256": _sha256_file(
                    REPO_ROOT / "data/manifest/formal_splits_v5/validation.json"),
            },
            "formal_split_v5_test": {
                "path": "data/manifest/formal_splits_v5/test.json",
                "sha256": _sha256_file(
                    REPO_ROOT / "data/manifest/formal_splits_v5/test.json"),
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
    "build_arrival_task_stream",
    "build_mapper_manifest",
    "c_idc_base_work_per_hour",
    "canonical_content_hash",
    "canonical_stream_payload",
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
