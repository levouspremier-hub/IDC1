"""M1.3g-e-c：**正式环境注入**构造层（纯构造，不接训练）。

把已经验签的 **B6 formal v5** 场景与 **M1.3g arrival mapper** 组装成一个
不可变的注入对象，供 `IDCPriceEnv20D(formal_injection=...)` 使用。

## 唯一信任链

`build_verified_formal_env_injection()` **只**通过既有正式入口取得数据：

```text
v5 split manifest (canonical-only)
  → B6 policy / exogenous v3 bundle / refs_v4
  → build_formal_scenario_b6(...)   （causal forecasts）
  → build_arrival_task_stream(...)  （mapper Tasks + 整数账本）
  → canonical parquet 的 episode 切片（**realized** exogenous）
```

**不接受**调用方传入任意 truth / forecast / refs / task stream；任一验证失败
**显式失败**，**不回退** demo / synthetic / 旧 v1–v4 链。

## 两个 arrival 口径**不得混用**

| 量 | 来源 | 进入哪里 |
|---|---|---|
| **realized** aggregate | v3 驱动表（mapper 账本） | `Task` 的 workload（**真值**） |
| **expected** forecast | `rate_template[slot] × 31.994` | 观测的 arrival 通道（**决策可见**） |

`Task.workload` 是**物理 work**；`lambda_ref_work_per_step = 31.994`
（= 63.988 work/hour × 0.5 h）用于**每步**归一化。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from idc_model.task import Task

REPO_ROOT = Path(__file__).resolve().parent.parent

DELTA_T_HOURS = 0.5
CANONICAL_PARQUET_LOGICAL = "data/processed/singapore_2024/half_hour.parquet"

# 环境每步归一化用的 arrival 尺度（= B6 冻结 rate × delta_t_hours）
B6_RATE_WORK_PER_HOUR = 63.988
B6_RATE_WORK_PER_STEP = 31.994

# 存量型参考值（**不**随步长缩放）
QUEUE_REFS_LOGICAL = "configs/frozen_refs/refs_v4.json"


class FormalInjectionError(ValueError):
    """正式环境注入的**明确失败**（无 fallback、不静默）。"""


@dataclass(frozen=True)
class FormalEnvInjection:
    """一个 episode 的**不可变**正式注入载荷。"""

    split: str
    start: str
    local_origin: int
    global_origin: int
    horizon: int
    forecast_cutoff: int
    delta_t_hours: float

    # canonical **realized** exogenous（episode 切片）
    price_sgd_per_kwh: tuple[float, ...]
    temperature_deg_c: tuple[float, ...]
    carbon_kg_per_kwh: tuple[float, ...]
    local_pv_kw: tuple[float, ...]
    wind_generation_kw: tuple[float, ...]

    # **causal** forecast（决策可见；来自 B6 ScenarioBundle）
    arrival_forecast: tuple[float, ...]
    lambda_ref_work_per_step: float

    # mapper 产物
    tasks: tuple[Task, ...]
    slots: tuple[Any, ...]
    ledger_micro: tuple[int, ...]

    # refs_v4 的存量型参考值
    queue_ref: float
    queue_capacity_ref: float

    # provenance
    provenance_hash: str
    sources: tuple[tuple[str, str], ...]


def _sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise FormalInjectionError(f"冻结资产不可读：{path}：{error}") from error


def build_verified_formal_env_injection(
    split: str,
    *,
    start: str,
    horizon: int,
    forecast_cutoff: int,
) -> FormalEnvInjection:
    """构造**唯一**的正式环境注入载荷（逐层走既有正式入口）。

    `start` 必须是 canonical 30 分钟网格上的带时区 ISO 时间，精确映射到该 split
    的**本地 origin**；越界 / 非网格 / 非候选 origin 一律 fail closed。
    """
    from scenario.arrival_mapper import (
        _aggregate_from_verified_chain,
        build_arrival_task_stream,
        load_verified_mapper_chain,
    )
    from scenario.b6_split_manifests import local_origin_from_start
    from scenario.formal_scenario_b6 import build_formal_scenario_b6
    from scenario.splits import SPLIT_SPECS, validate_episode_origin

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise FormalInjectionError(f"horizon 必须是正整数，实际 {horizon!r}")
    if isinstance(forecast_cutoff, bool) or not isinstance(forecast_cutoff, int) \
            or forecast_cutoff <= 0:
        raise FormalInjectionError(
            f"forecast_cutoff 必须是正整数，实际 {forecast_cutoff!r}")

    # 1) 唯一验证链（B6 policy / v3 bundle / refs_v4 / v5 split）
    chain = load_verified_mapper_chain(split)

    # 2) start → 本地 origin（既有严格映射）
    local_origin = local_origin_from_start(split, start)
    global_origin = validate_episode_origin(split, local_origin, horizon)

    # 3) mapper：逐槽 Tasks + 整数账本（**realized** aggregate）
    stream = build_arrival_task_stream(
        split, start=start, horizon=horizon, seed=0)
    aggregate = _aggregate_from_verified_chain(chain)
    slice_micro = tuple(int(aggregate[global_origin + k]) * 1_000_000
                        for k in range(horizon))
    if tuple(stream.slots[i].aggregate_micro for i in range(horizon)) != slice_micro:
        raise FormalInjectionError("mapper 账本与 v3 realized aggregate 不一致")

    # 4) B6 causal forecast（**期望**值，长度 == horizon）
    bundle = build_formal_scenario_b6(
        split,
        origin=local_origin,
        forecast_cutoff=horizon,
        canonical_parquet_path=REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
        canonical_manifest_path=REPO_ROOT / "data/manifest/singapore_2024_half_hour.json",
        split_manifest_path=REPO_ROOT / "data/manifest/singapore_2024_splits.json",
        policy_manifest_path=REPO_ROOT / "data/manifest/singapore_2024_forecast_policy_v3.json",
    )

    # 5) canonical realized exogenous 切片
    canonical = pd.read_parquet(REPO_ROOT / CANONICAL_PARQUET_LOGICAL)
    episode = canonical.iloc[global_origin:global_origin + horizon]
    if len(episode) != horizon:
        raise FormalInjectionError("canonical 切片长度与 horizon 不符")

    # 6) exogenous v3 切片（PV / 风电 / 碳）
    v3 = chain["bundle"]["frame"]
    v3_episode = v3.iloc[global_origin:global_origin + horizon]

    refs = chain["refs"]["references"]
    queue_ref = float(refs["queue_ref"]["value"])
    queue_capacity_ref = float(refs["queue_capacity_ref"]["value"])

    sources = tuple(sorted(
        (str(role), str(entry["sha256"]))
        for role, entry in chain["splits"][split]["inputs"].items()
    ))

    spec = SPLIT_SPECS[split]
    if not (spec["row_start"] <= global_origin < spec["row_end_exclusive"]):
        raise FormalInjectionError(f"origin 不属于 {split}")

    payload = FormalEnvInjection(
        split=split,
        start=start,
        local_origin=local_origin,
        global_origin=global_origin,
        horizon=horizon,
        forecast_cutoff=forecast_cutoff,
        delta_t_hours=DELTA_T_HOURS,
        price_sgd_per_kwh=tuple(float(v) for v in episode["price_sgd_per_kwh"]),
        temperature_deg_c=tuple(float(v) for v in episode["temperature_deg_c"]),
        carbon_kg_per_kwh=tuple(float(v) for v in v3_episode["carbon_intensity"]),
        local_pv_kw=tuple(float(v) for v in v3_episode["local_pv_kw"]),
        wind_generation_kw=tuple(float(v) for v in v3_episode["wind_generation_kw"]),
        arrival_forecast=tuple(float(v) for v in bundle.arrival_forecast),
        lambda_ref_work_per_step=B6_RATE_WORK_PER_STEP,
        tasks=tuple(stream.tasks),
        slots=tuple(stream.slots),
        ledger_micro=slice_micro,
        queue_ref=queue_ref,
        queue_capacity_ref=queue_capacity_ref,
        provenance_hash="",
        sources=sources,
    )
    return FormalEnvInjection(
        **{**payload.__dict__,
           "provenance_hash": _provenance_hash(payload)}
    )


def _provenance_hash(inj: FormalEnvInjection) -> str:
    """注入载荷的规范化内容 hash（整数账本 + 稳定浮点表示）。"""
    canon = {
        "split": inj.split,
        "start": inj.start,
        "local_origin": inj.local_origin,
        "global_origin": inj.global_origin,
        "horizon": inj.horizon,
        "forecast_cutoff": inj.forecast_cutoff,
        "delta_t_hours": inj.delta_t_hours.hex(),
        "ledger_micro": list(inj.ledger_micro),
        "task_ids": [t.task_id for t in inj.tasks],
        "arrival_forecast": [float(v).hex() for v in inj.arrival_forecast],
        "sources": [list(s) for s in inj.sources],
    }
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


__all__ = [
    "B6_RATE_WORK_PER_HOUR",
    "B6_RATE_WORK_PER_STEP",
    "DELTA_T_HOURS",
    "FormalEnvInjection",
    "FormalInjectionError",
    "build_verified_formal_env_injection",
]
