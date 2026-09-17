"""M1.3g-b：formal causal `ScenarioBundle` 的**构造内核**。

本模块把**已冻结**的四层资产变换成 `mode="formal"` 的七序列
`contracts.ScenarioBundle`：

```text
canonical parquet + canonical manifest + split manifest
        + forecast policy manifest（v2 / contract-v9）
        + exogenous v2 manifest + exogenous source v3 manifest
```

## 因果性（本模块的**唯一**合法性来源）

对全局 origin = `i`、cutoff = `C`，七条序列只用 `[i−48, i)` 的历史：

| 序列 | 构造 | `source_kind` |
|---|---|---|
| `price` / `load` / `temperature` | M1.3e `seasonal_naive_forecast`
  （只读 `[i−48, i)`） | `seasonal_naive` |
| `pv` | **forecast** 的 GHI / 温度 / 10 m 风速
  + **target 日历时刻**过 `local_pv_kw` | `modeled_scenario` |
| `wind` | **forecast** 的 10 m 风速过 `wind_generation_kw` | `modeled_scenario` |
| `carbon` | 经核验的 v2 **B1 常数** `0.402` | `human_approved_external_low_resolution` |
| `arrival` | **D3：期望值** `rate_template[hour*2 + minute//30] × 1000` | `modeled_scenario` |

**PV / 风电是五个 driver forecast 的逐点确定性变换**，因此因果性由构造继承：
`local_pv_kw` / `wind_generation_kw` 是逐点纯函数，其全部输入都来自 `[i−48, i)`。
`target` 的**日历时刻**是已知的未来日历，不是未来真值。

**arrival 明确不使用 Poisson 抽样**：formal forecast 是模板的**期望**
`λ(slot) = rate_template[slot] × 1000`；v2 驱动表里的 Poisson 列（`generate_arrival`）
只是**模拟场景的实际 arrival**，**不得**当作未来已知 forecast（D3）。

## 物理实现只有一份

PV / 风电**复用** `scenario/exogenous_drivers.py` 的既有函数
（`local_pv_kw` / `wind_generation_kw` / `load_wind_power_curve`），
本模块**不**复制任何物理规则，因此「真值物化」与「formal forecast」走的是
**同一份**代码。

## fail closed

七层来源逐层校验（policy → split → canonical manifest → parquet，
外加 exogenous 的两份 manifest 与 v2 output hash）；任一不符即拒绝。
**本模块不写任何文件、不构造 env、不接线训练。**
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from contracts.models import (
    ArtifactDigest,
    ForecastSeriesProvenance,
    ScenarioBundle,
    ScenarioForecastProvenance,
)
from scenario.exogenous_drivers import (
    ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR,
    ARRIVAL_TEMPLATE_SLOTS,
    B5_ARRIVAL_APPROVAL,
    B5_PV_APPROVAL,
    CARBON_KG_PER_KWH,
    arrival_template_slot,
    local_pv_kw,
    wind_generation_kw,
)
from scenario.forecast import (
    FORECAST_PERIOD_STEPS,
    FORECAST_SOURCE_PATHS,
    build_available_exogenous_forecast,
    # 刻意保留这个**未使用**的 re-export：M1.3g-b-R1 的回归用
    # `monkeypatch.setattr(module, "seasonal_naive_forecast", boom)` 证明
    # formal 内核**没有**第二次独立调用它（seasonal 三序列直接取自 artifact）。
    seasonal_naive_forecast,  # noqa: F401
)
from scenario.splits import (
    SplitName,
    logical_repo_path,
    validate_canonical_timeline,
    validate_forecast_origin,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

STEP_MINUTES = 30

# 七条序列的 `source_kind`（**精确**，不得漂移）
FORMAL_SOURCE_KINDS: dict[str, str] = {
    "price_forecast": "seasonal_naive",
    "load_forecast": "seasonal_naive",
    "temperature_forecast": "seasonal_naive",
    "pv_forecast": "modeled_scenario",
    "wind_forecast": "modeled_scenario",
    "carbon_forecast": "human_approved_external_low_resolution",
    "arrival_forecast": "modeled_scenario",
}
CARBON_SOURCE_KIND = FORMAL_SOURCE_KINDS["carbon_forecast"]

# canonical 列 ↔ bundle 字段（季节朴素 driver）
SEASONAL_DRIVER_COLUMNS: dict[str, str] = {
    "price_forecast": "price_sgd_per_kwh",
    "load_forecast": "system_load_mw",
    "temperature_forecast": "temperature_deg_c",
}
# PV / 风电的输入 driver 列
GHI_COLUMN = "ghi_w_per_m2"
WIND_SPEED_COLUMN = "wind_speed_10m_mps"
PV_TEMPERATURE_COLUMN = "temperature_deg_c"

# 正式 exogenous 资产的**冻结**身份（M1.3f-c-R1 的 v2/v3，逐字节绑定）
EXOGENOUS_SCHEMA = "m1.3c-singapore-2024-exogenous-v1"
EXOGENOUS_MANIFEST_SHA256 = (
    "640f26cda94b3479049fdbee56f05e1546a24fc3ecdf6286674c4fb415b484b9"
)
EXOGENOUS_SOURCE_MANIFEST_SHA256 = (
    "4203b4f399ee6433bfcdf63fa94ddd03a56a7bd1e6add45f697c3bec804da1b6"
)
EXOGENOUS_OUTPUT_SHA256 = (
    "11d322b2919e2180b596e6b02614acafdb3ee8d63682ae74e5a3ee1dbc8b92cf"
)
EXOGENOUS_OUTPUT_LOGICAL_PATH = (
    "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
)
# v1（M1.3f-c）产物：**不得**作为正式链证据
SUPERSEDED_EXOGENOUS_SHA256 = (
    "0c5e65d8fdc25ed8ced228e8087f13eb0605d0146d7e258caf54a552a246287c"
)

MODEL_NAME = "formal_scenario_kernel"
MODEL_VERSION = "v1"

# **formal 内核自身的**实现文件集合（M1.3g-b-R1）。
# 它**严格包含** provider 的 `FORECAST_SOURCE_PATHS`，外加真正参与 formal 语义的
# 两个实现：formal 内核本身与外生驱动物理实现。`code_revision` 由**这一组**路径
# 解析，因此任何一处改动都会改变 formal revision —— **不得**用旧提交为新实现背书。
FORMAL_SOURCE_PATHS: tuple[str, ...] = (
    *FORECAST_SOURCE_PATHS,
    "scenario/formal_scenario.py",
    "scenario/exogenous_drivers.py",
)


class FormalScenarioError(ValueError):
    """formal 场景构造的**明确失败**。"""


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def formal_code_revision() -> str:
    """**formal 内核自身的**冻结 revision（40 位小写 SHA）。

    由 `FORMAL_SOURCE_PATHS`（provider 五文件 + formal 内核 + 外生驱动实现）
    解析；**没有**调用者入口，因此 provenance 的 `code_revision` 不可能被伪造。
    """
    revision = _git("log", "-1", "--format=%H", "--", *FORMAL_SOURCE_PATHS).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise FormalScenarioError(f"formal code_revision 无效：{revision!r}")
    return revision


def formal_generator_is_dirty() -> bool:
    """formal 实现是否有未提交修改（含未跟踪的新文件）——**同一**路径集合。"""
    status = _git("status", "--porcelain", "--", *FORMAL_SOURCE_PATHS)
    return bool(status.strip())


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise FormalScenarioError(f"冻结资产不可读：{path}：{error}") from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalScenarioError(message)


def _require_dict(value: object, *, field: str) -> dict:
    """必须是 object；否则 fail closed（同时让类型检查器收窄）。"""
    if not isinstance(value, dict):
        raise FormalScenarioError(f"{field} 必须是 object，实际 {type(value).__name__}")
    return value


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormalScenarioError(f"{path} 不可读或不是合法 JSON：{error}") from error
    _require(isinstance(payload, dict), f"{path} 的顶层必须是 object")
    return payload


# --- exogenous v2 / source v3 的核验 -----------------------------------------

def _require_bound(
    declared_path: object, declared_sha: object, actual: Path, *, field: str
) -> None:
    """**交叉绑定**：manifest 的声明必须等于本次调用**实际使用**的对象。

    path 与 SHA-256 **逐项**校验——两边各自自洽但互不相符时**同样**拒绝。
    """
    expected_path = logical_repo_path(actual)
    _require(
        declared_path == expected_path,
        f"{field} 的声明路径与实际提供的对象不符："
        f"声明={declared_path!r} 实际={expected_path!r}",
    )
    measured = _sha256_file(actual)
    _require(
        declared_sha == measured,
        f"{field} 的声明 SHA-256 与实际提供的对象不符："
        f"声明={declared_sha!r} 实测={measured}",
    )


def load_verified_exogenous(
    exogenous_manifest_path: Path | str,
    exogenous_source_manifest_path: Path | str,
    *,
    exogenous_parquet_path: Path | str,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
) -> dict:
    """核验并返回 M1.3f-c-R1 的 v2 外生驱动声明。

    逐项校验：

    1. 两份 manifest 的**字节** SHA-256 等于**模块冻结常量**；
    2. v2 manifest 的 schema、B1（carbon）与 B5（PV / arrival）声明完整；
    3. 声明的 output path/hash 等于**实际提供**的 v2 驱动表（含冻结 output hash）；
    4. **交叉绑定**：声明的 canonical parquet / canonical manifest / split manifest
       的 path **与** SHA-256 逐项等于**本次 formal 调用实际使用**的对象；
    5. v1 产物**不得**被当作正式证据。

    **签名里没有任何 hash / revision / trust-root 覆盖参数**：信任根是模块
    级冻结常量，调用者无法替换。测试若要验证临时链，只能 monkeypatch
    **模块内部**的那几个常量（见 `tests/test_m13gb_formal_scenario.py`）。
    """
    exogenous_manifest_path = Path(exogenous_manifest_path)
    exogenous_source_manifest_path = Path(exogenous_source_manifest_path)
    exogenous_parquet_path = Path(exogenous_parquet_path)
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    actual = _sha256_file(exogenous_manifest_path)
    if actual == SUPERSEDED_EXOGENOUS_SHA256:
        raise FormalScenarioError(
            "传入的是 M1.3f-c 的 **v1** 外生驱动 manifest："
            "它标为 superseded_pre_approval_and_loss_fix，不得作为正式链证据"
        )
    _require(
        actual == EXOGENOUS_MANIFEST_SHA256,
        f"exogenous v2 manifest 的 SHA-256 与冻结登记不符："
        f"期望 {EXOGENOUS_MANIFEST_SHA256} 实际 {actual}",
    )
    _require(
        _sha256_file(exogenous_source_manifest_path) == EXOGENOUS_SOURCE_MANIFEST_SHA256,
        "exogenous source v3 manifest 的 SHA-256 与冻结登记不符",
    )

    payload = _load_json(exogenous_manifest_path)

    # 交叉绑定：v2 manifest 声明的上游对象 == 本次调用实际使用的对象
    _require_bound(
        payload.get("canonical_parquet_path"), payload.get("canonical_parquet_sha256"),
        canonical_parquet_path, field="exogenous.canonical_parquet",
    )
    _require_bound(
        payload.get("canonical_manifest_path"), payload.get("canonical_manifest_sha256"),
        canonical_manifest_path, field="exogenous.canonical_manifest",
    )
    _require_bound(
        payload.get("split_manifest_path"), payload.get("split_manifest_sha256"),
        split_manifest_path, field="exogenous.split_manifest",
    )
    _require(
        payload.get("schema") == EXOGENOUS_SCHEMA,
        f"exogenous v2 manifest 的 schema 必须是 {EXOGENOUS_SCHEMA!r}，"
        f"实际 {payload.get('schema')!r}",
    )

    columns = _require_dict(payload.get("columns"),
                            field="exogenous v2 manifest.columns")
    carbon = _require_dict(columns.get("carbon_intensity"),
                           field="columns.carbon_intensity")
    _require(
        carbon.get("classification") == CARBON_SOURCE_KIND,
        "carbon_intensity 的 classification 必须是 "
        f"{CARBON_SOURCE_KIND!r}，实际 {carbon.get('classification')!r}"
        "（不得写成 modeled_scenario）",
    )
    _require(
        carbon.get("value") == CARBON_KG_PER_KWH,
        f"carbon_intensity 的 B1 常数必须是 {CARBON_KG_PER_KWH}，"
        f"实际 {carbon.get('value')!r}",
    )
    decision = _require_dict(carbon.get("human_decision"),
                             field="columns.carbon_intensity.human_decision")
    _require(
        decision.get("decision_id") == "B1",
        f"carbon_intensity 缺少 B1 人工批准声明，实际 {decision!r}",
    )

    local_pv = _require_dict(columns.get("local_pv_kw"), field="columns.local_pv_kw")
    pv_approval = _require_dict(local_pv.get("b5_pv_approval"),
                                field="columns.local_pv_kw.b5_pv_approval")
    _require(
        pv_approval.get("decision_id") == B5_PV_APPROVAL["decision_id"],
        f"local_pv_kw 缺少 B5-PV 人工批准声明，实际 {pv_approval!r}",
    )
    for name, expected in B5_PV_APPROVAL.items():
        _require(
            pv_approval.get(name) == expected,
            f"local_pv_kw 的 B5-PV 声明 {name} 与冻结值不符："
            f"期望 {expected!r} 实际 {pv_approval.get(name)!r}",
        )

    arrival = _require_dict(columns.get("arrival"), field="columns.arrival")
    arrival_approval = _require_dict(arrival.get("b5_approval"),
                                     field="columns.arrival.b5_approval")
    _require(
        arrival_approval.get("decision_id") == B5_ARRIVAL_APPROVAL["decision_id"],
        f"arrival 缺少 B5-ARRIVAL 人工批准声明，实际 {arrival_approval!r}",
    )
    for name, expected in B5_ARRIVAL_APPROVAL.items():
        _require(
            arrival_approval.get(name) == expected,
            f"arrival 的 B5-ARRIVAL 声明 {name} 与冻结值不符："
            f"期望 {expected!r} 实际 {arrival_approval.get(name)!r}",
        )
    _require(
        arrival_approval.get("uses_archive_dates") is False,
        "arrival 的 B5-ARRIVAL 必须声明 uses_archive_dates=False",
    )
    _require(
        arrival.get("uses_archive_dates") is False,
        "arrival 不得声明使用 archive 的日期/星期/时区",
    )

    declared_output = _require_dict(payload.get("output"),
                                    field="exogenous v2 manifest.output")
    _require_bound(
        declared_output.get("path"), declared_output.get("sha256"),
        exogenous_parquet_path, field="exogenous.output",
    )
    _require(
        _sha256_file(exogenous_parquet_path) == EXOGENOUS_OUTPUT_SHA256,
        "v2 外生驱动表的 SHA-256 与冻结的 output hash 不符",
    )
    return payload


def exogenous_rate_template(verified_payload: dict) -> list[float]:
    """从**已通过 `load_verified_exogenous` 的 payload** 取 48 槽 rate template。

    刻意**不**接受 manifest 路径：formal 路径不得绕过核验单独读取 template。
    """
    columns = _require_dict(
        _require_dict(verified_payload, field="verified exogenous payload").get("columns"),
        field="exogenous manifest.columns",
    )
    arrival = _require_dict(columns.get("arrival"), field="columns.arrival")
    template = arrival.get("rate_template")
    if not isinstance(template, list) or len(template) != ARRIVAL_TEMPLATE_SLOTS:
        raise FormalScenarioError(
            f"arrival rate_template 必须是 {ARRIVAL_TEMPLATE_SLOTS} 项的 list，"
            f"实际 {type(template).__name__}"
        )
    return [float(value) for value in template]


def build_available_forecast_only(**kwargs):
    """暴露给审计/测试的**只读**入口：只构造 M1.3e 的 driver forecast artifact。

    生产链**不**使用它；它存在的目的是让「seasonal 序列确实取自 artifact」这一
    断言可以独立复算，而不必复制 provider 的内部调用。
    """
    return build_available_exogenous_forecast(**kwargs)


# --- PV / 风电：对 forecast 的逐点确定性变换 ---------------------------------

def pv_forecast(
    target_timestamps: pd.DatetimeIndex,
    ghi_w_per_m2,
    temp_air_deg_c,
    wind_speed_10m_mps,
) -> tuple[float, ...]:
    """由 **forecast** 的 GHI / 温度 / 10 m 风速构造 PV forecast（逐点）。

    **复用** `scenario.exogenous_drivers.local_pv_kw`——与真值物化是同一份物理实现。
    """
    _require(len(target_timestamps) > 0, "PV forecast 的 target 不得为空")
    values = local_pv_kw(
        pd.DatetimeIndex(target_timestamps), ghi_w_per_m2, temp_air_deg_c,
        wind_speed_10m_mps,
    )
    return tuple(float(v) for v in values)


def wind_forecast(wind_speed_10m_mps) -> tuple[float, ...]:
    """由 **forecast** 的 10 m 风速构造风电 forecast（逐点）。

    **复用** `scenario.exogenous_drivers.wind_generation_kw`。
    """
    values = wind_generation_kw(wind_speed_10m_mps)
    return tuple(float(v) for v in values)


def arrival_forecast(
    target_timestamps: pd.DatetimeIndex, rate_template
) -> tuple[float, ...]:
    """**D3**：arrival forecast 是模板的**期望** `λ(slot) × 1000`。

    **不**调用 `generate_arrival`、**不**使用 Poisson seed——
    v2 驱动表的 Poisson 列只是**模拟场景的实际 arrival**，不是未来已知 forecast。
    """
    template = tuple(float(v) for v in rate_template)
    _require(
        len(template) == ARRIVAL_TEMPLATE_SLOTS,
        f"rate_template 必须有 {ARRIVAL_TEMPLATE_SLOTS} 项，实际 {len(template)}",
    )
    slots = arrival_template_slot(pd.DatetimeIndex(target_timestamps))
    return tuple(
        template[int(slot)] * ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR for slot in slots
    )


# --- 组装 ---------------------------------------------------------------------

def build_formal_scenario(
    split: SplitName,
    *,
    origin: int,
    forecast_cutoff: int,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    policy_manifest_path: Path | str,
    exogenous_manifest_path: Path | str,
    exogenous_source_manifest_path: Path | str,
    horizon: int | None = None,
) -> ScenarioBundle:
    """构造 `mode="formal"` 的七序列 `ScenarioBundle`（**纯构造**，不写文件）。

    `origin` 是 split-**本地** half-hour step。历史窗口取**全局** `[i−48, i)`，
    因此 validation / test 的起点可以使用其**之前已经发生**的 canonical 历史。

    信任链：policy-v2 → split → canonical manifest → canonical parquet，
    外加 exogenous v2 manifest / source v3 manifest / v2 output hash，
    逐层校验，任一不符即 fail closed。
    """
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise FormalScenarioError(f"origin 必须是整数，实际 {origin!r}")
    if isinstance(forecast_cutoff, bool) or not isinstance(forecast_cutoff, int):
        raise FormalScenarioError(f"forecast_cutoff 必须是整数，实际 {forecast_cutoff!r}")

    # 0) formal 实现必须**已提交**：不得用旧 revision 为未提交的新实现背书
    if formal_generator_is_dirty():
        raise FormalScenarioError(
            "formal 实现有未提交修改：拒绝用旧 code_revision 为未提交代码背书（未提交）"
        )

    global_origin = validate_forecast_origin(split, origin, forecast_cutoff)
    history_start = global_origin - FORECAST_PERIOD_STEPS
    _require(
        history_start >= 0,
        f"{split} 内 origin={origin}（全局 {global_origin}）不足 "
        f"{FORECAST_PERIOD_STEPS} 步历史；fail closed",
    )

    # 1) 五类 driver forecast：**复用** M1.3e 的完整信任链与因果 provider
    artifact = build_available_exogenous_forecast(
        split,
        origin=origin,
        forecast_cutoff=forecast_cutoff,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
        policy_manifest_path=policy_manifest_path,
    )
    generated_at = artifact.generated_at
    target_timestamps = tuple(artifact.target_timestamps)

    # 2) canonical：再验一次时间轴（artifact 已验，此处供 PV 的日历时刻与历史窗口）
    canonical_parquet_path = Path(canonical_parquet_path)
    frame = pd.read_parquet(canonical_parquet_path)
    validate_canonical_timeline(frame, label="canonical")
    stamps = pd.DatetimeIndex(frame["timestamp"])

    # 3) exogenous v2 / source v3 的逐字节核验 + **与本次调用的交叉绑定**
    exogenous_parquet_path = REPO_ROOT / EXOGENOUS_OUTPUT_LOGICAL_PATH
    exogenous = load_verified_exogenous(
        exogenous_manifest_path,
        exogenous_source_manifest_path,
        exogenous_parquet_path=exogenous_parquet_path,
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )

    # 4) 七条序列：全部只用 `[i−48, i)`
    #    price / load / temperature **直接取自 artifact** 的对应 series ——
    #    **不**第二次独立调用 `seasonal_naive_forecast`（避免实现漂移）。
    drv = artifact.series.as_dict()
    series: dict[str, tuple[float, ...]] = {
        "price_forecast": drv["price_sgd_per_kwh"],
        "load_forecast": drv["system_load_mw"],
        "temperature_forecast": drv["temperature_deg_c"],
    }
    ghi_f = drv[GHI_COLUMN]
    temp_f = drv[PV_TEMPERATURE_COLUMN]
    v10_f = drv[WIND_SPEED_COLUMN]
    series["pv_forecast"] = pv_forecast(
        pd.DatetimeIndex(target_timestamps), ghi_f, temp_f, v10_f)
    series["wind_forecast"] = wind_forecast(v10_f)
    series["carbon_forecast"] = (CARBON_KG_PER_KWH,) * forecast_cutoff
    # arrival 的 template **只**从已验证 payload 取得（不再单独读 manifest）
    series["arrival_forecast"] = arrival_forecast(
        pd.DatetimeIndex(target_timestamps),
        exogenous_rate_template(exogenous),
    )

    # 5) provenance：七项 `generated_at` 恒等，时间顺序自洽；
    #    `code_revision` 是 **formal 内核自身**的冻结 revision（非 provider-only）
    code_revision = formal_code_revision()
    target_end_exclusive = (
        datetime.fromisoformat(target_timestamps[-1]) + timedelta(minutes=STEP_MINUTES)
    ).isoformat()
    lookback_start = pd.Timestamp(stamps[history_start]).isoformat()

    sources = _formal_sources(
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
        policy_manifest_path=policy_manifest_path,
        exogenous_manifest_path=exogenous_manifest_path,
        exogenous_source_manifest_path=exogenous_source_manifest_path,
        exogenous_parquet_path=exogenous_parquet_path,
        code_revision=code_revision,
    )

    provenance = {
        field: ForecastSeriesProvenance(
            series_name=field,
            source_kind=FORMAL_SOURCE_KINDS[field],
            method=_method_for(field),
            generated_at=generated_at,
            information_cutoff_exclusive=generated_at,
            target_start=generated_at,
            target_end_exclusive=target_end_exclusive,
            lookback_start=lookback_start,
            lookback_end_exclusive=generated_at,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            code_revision=code_revision,
            seed=None,
            sources=sources,
        )
        for field in FORMAL_SOURCE_KINDS
    }

    bundle = ScenarioBundle(
        split=split,
        start=str(origin),
        horizon=forecast_cutoff if horizon is None else horizon,
        forecast_cutoff=forecast_cutoff,
        price_forecast=series["price_forecast"],
        load_forecast=series["load_forecast"],
        pv_forecast=series["pv_forecast"],
        wind_forecast=series["wind_forecast"],
        temperature_forecast=series["temperature_forecast"],
        carbon_forecast=series["carbon_forecast"],
        arrival_forecast=series["arrival_forecast"],
        mode="formal",
        generated_at=generated_at,
        forecast_provenance=ScenarioForecastProvenance(**provenance),
    )
    # 自证：formal bundle 必须能通过训练 purpose gate
    from contracts.validators import validate_forecast_purpose

    validate_forecast_purpose(bundle, purpose="training")
    return bundle


def _method_for(field: str) -> str:
    if field in SEASONAL_DRIVER_COLUMNS:
        return "trailing_seasonal_naive"
    if field == "pv_forecast":
        return "pvlib_v0.15.2_chain_from_causal_driver_forecasts"
    if field == "wind_forecast":
        return "shear_law_and_frozen_power_curve_from_causal_wind_forecast"
    if field == "carbon_forecast":
        return "annual_constant_from_verified_v2_manifest"
    return "expected_rate_template_from_frozen_benchmark_calibration"


def _formal_sources(
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    policy_manifest_path: Path | str,
    exogenous_manifest_path: Path | str,
    exogenous_source_manifest_path: Path | str,
    exogenous_parquet_path: Path | str,
    code_revision: str,
) -> tuple[ArtifactDigest, ...]:
    """六条上游制品的 digest（角色、逻辑路径、实测 SHA-256）。"""
    entries = (
        ("canonical_parquet", canonical_parquet_path),
        ("canonical_manifest", canonical_manifest_path),
        ("split_manifest", split_manifest_path),
        ("forecast_policy_manifest", policy_manifest_path),
        ("exogenous_drivers_manifest", exogenous_manifest_path),
        ("exogenous_source_manifest", exogenous_source_manifest_path),
        # M1.3g-b-R1：**经验证**的 v2 驱动表本身也必须作为来源 digest
        ("exogenous_drivers_parquet", exogenous_parquet_path),
    )
    return tuple(
        ArtifactDigest(
            role=role,
            logical_path=logical_repo_path(Path(path)),
            sha256=_sha256_file(Path(path)),
        )
        for role, path in entries
    )

