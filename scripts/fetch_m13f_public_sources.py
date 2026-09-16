"""M1.3f-b：公开数据源获取、许可冻结与可复现实物化。

**本模块只做四件事**：登记公开来源 → 在显式 `--fetch` 下获取 → 校验（大小 / hash /
许可 / schema / URL）→ 原子安装并写可复现 manifest。它**不**生成正式
`local_pv_kw` / `wind_generation_kw` / `carbon_intensity` / `arrival`，
**不**构造 `ScenarioBundle`，**不**解除任何训练门禁。

## 网络纪律

- **默认运行不联网**：不带 `--fetch` 时等价于 `--verify`，只校验本地冻结文件；
- **只有显式 `--fetch`** 才会发起网络访问；
- 只允许 **HTTPS** 官方来源，不用第三方镜像；唯一例外是 loopback
  （`127.0.0.1` / `localhost`）的 `http://`，供本地 fixture 使用；
- hash / 许可 / HTTP / 大小 / schema 任一不符即 **fail closed**；
- 下载写临时文件，校验通过后**原子安装**；失败不留半文件；
- 已冻结文件内容不同则**拒绝覆盖**。

## 大小上限

已冻结的 M1.2 raw 资产实际为 0.27 / 0.34 / 0.85 / **14.47** MiB，
据此把单源上限固定为 **16 MiB**（既有最大 raw 资产向上取整）。
超过上限的来源默认**拒绝下载**，只登记元数据并升级人工；
**唯一例外**是经人工逐条批准、且带 `exception_cap_bytes` 的 pinned URL
（`effective_cap()` 只对这类来源放宽，通用上限不变）。

## 不可变性

- 正式 manifest **已存在且语义不同 → fail closed**，禁止覆盖；
- 完全相同 → 直接返回，**不改变 `mtime_ns`**；
- 重复 `--fetch` **复用**现存合法 `frozen_at_utc`，不得随墙钟漂移；
- 首次生成用临时文件 + 原子安装，任一步失败不破坏已有产物。

## 四条红线（固化为可测试守卫）

1. `assert_year_matches()`：2026 光伏 profile **不得**冒充 2024 真值；
2. `assert_resolution_matches()`：年度碳因子**不得**冒充半小时碳强度真值；
3. `assert_parameters_approved()`：只有**结构化人工批准**（带 `decision_id`
   与批准日期）的参数才可用；标为 `legacy_inherited` 的旧仿真假设
   **不得继承**为正式口径；
4. `assert_no_silent_replay()`：Azure 2019 trace **不得**被静默重放成 2024 arrival。

## 严格校验的单一入口

`validate_public_source_manifest()` 是**唯一**入口：顶层/entry/参数对象键集合精确、
`source_id` 集合与**顺序**精确、固定声明与 `SOURCE_SPECS` 逐字段恒等、
`blocked`/`refused` 来源**同样校验（不得跳过）**、`readiness`/红线/参数块完整恒等；
任何畸形输入**只抛** `SourcePolicyError`。`verify_local`、`--verify` 与 `--fetch`
完成后的复核**共用**它，不维护较弱的第二套规则。

## 用法

```bash
uv run python scripts/fetch_m13f_public_sources.py --verify   # 只校验本地（默认）
uv run python scripts/fetch_m13f_public_sources.py --fetch    # 显式联网获取并冻结
uv run python scripts/fetch_m13f_public_sources.py --report   # 只打印登记表
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data/manifest/m13f_public_sources.json"
RAW_DIR = REPO_ROOT / "data/raw/public_benchmarks"

MANIFEST_SCHEMA = "m1.3f-public-sources-v1"
CONTRACT_VERSION = "contract-v8"

# 先验锚点：既有最大 raw 资产 14.47 MiB 向上取整。
MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES_ANCHOR = (
    "既有 M1.2 raw 资产最大 14.47 MiB（sasea_demand_2024.zip）向上取整"
)

# --- 冻结的 manifest schema（**精确**键集合） --------------------------------

TOP_KEYS: tuple[str, ...] = (
    "schema", "contract_version", "max_source_bytes", "max_source_bytes_anchor",
    "frozen_at_utc", "sources", "human_approved_parameters",
    "unapproved_parameters", "red_lines", "readiness",
)
SOURCE_COMMON_KEYS: tuple[str, ...] = (
    "source_id", "route", "role", "classification", "status", "url", "license",
    "license_url", "pinned_ref", "data_year", "resolution",
)
STATUS_EXTRA_KEYS: dict[str, tuple[str, ...]] = {
    "frozen": ("logical_path", "bytes", "sha256"),
    "blocked": ("blocked_reason",),
    "refused_over_size_cap": (
        "refused_reason", "observed_content_length", "exception_cap_bytes",
        "decision_id",
    ),
}
PARAM_KEYS: tuple[str, ...] = ("value", "unit", "status", "decision_id", "approved_on")
PARAM_STATUSES: tuple[str, ...] = ("human_approved", "UNAPPROVED", "legacy_inherited")

RED_LINES: tuple[str, ...] = (
    "2026 Solar Generation Profile 不得冒充 2024 真值",
    "年度碳因子不得冒充半小时碳强度真值",
    "未批准参数不得生成正式 PV / 风电；pv_capacity_kw=500 不得继承",
    "Azure 2019 trace 不得被静默重放成 2024 arrival",
)

# 人工决定（**只登记，不实施**；§D.4 的 B1/B4）：
# 本轮把批准值写进 manifest，但**不**把任何 readiness 改为 true，
# 也**不**生成任何正式序列。
APPROVED_ON = "2026-09-16"


def _approved(value: Any, unit: str, decision_id: str) -> dict[str, Any]:
    return {
        "value": value, "unit": unit, "status": "human_approved",
        "decision_id": decision_id, "approved_on": APPROVED_ON,
    }


def _unapproved(unit: str) -> dict[str, Any]:
    return {
        "value": None, "unit": unit, "status": "UNAPPROVED",
        "decision_id": None, "approved_on": None,
    }


HUMAN_APPROVED_PARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "local_pv_kw": {
        "pv_capacity_kw": _approved(500.0, "kW", "B4"),
        "tilt_deg": _approved(10.0, "deg", "B4"),
        "azimuth_deg": _approved(180.0, "deg", "B4"),
        "array_type": _approved("fixed_open_rack", "text", "B4"),
        "losses_pct": _approved(14.0, "percent", "B4"),
        "gamma_pdc_per_deg_c": _approved(-0.004, "1/degC", "B4"),
        "dc_ac_ratio": _approved(1.2, "dimensionless", "B4"),
        "eta_inv_nom": _approved(0.96, "dimensionless", "B4"),
        "temperature_model": _approved("open_rack_glass_polymer", "text", "B4"),
    },
    "wind_generation_kw": {
        "turbine_model": _approved("E48/800", "text", "B4"),
        "hub_height_m": _approved(60.0, "m", "B4"),
        "shear_exponent": _approved(1.0 / 7.0, "dimensionless", "B4"),
        "rated_capacity_kw": _approved(800.0, "kW", "B4"),
    },
    "carbon_intensity": {
        "carbon_intensity_kg_per_kwh": _approved(0.402, "kgCO2/kWh", "B1"),
    },
}
# 仍然**未**获批的部分：arrival 的过程族/参数/seed 策略（B3 只批准下载 trace 包）。
UNAPPROVED_PARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    "arrival": {
        "process_family": _unapproved("text"),
        "parameters": _unapproved("n/a"),
        "seed_policy": _unapproved("n/a"),
    },
}

# 分类口径（卡面 §B.3）
CLASSIFICATIONS = (
    "observed", "external_low_resolution", "modeled_scenario", "benchmark_trace",
)
# 尚未被人工批准的物理参数：出现即拒绝生成正式 PV / 风电
UNAPPROVED = "UNAPPROVED"

_LOGICAL_PREFIX = "data/raw/public_benchmarks"

READINESS: dict[str, bool] = {
    "public_source_frozen": True,
    "local_pv_kw_ready": False,
    "wind_generation_kw_ready": False,
    "carbon_intensity_ready": False,
    "arrival_ready": False,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}


class SourcePolicyError(ValueError):
    """公开源获取/校验的**明确失败**（fail closed）。"""


# --- 来源登记 ----------------------------------------------------------------
#
# status 取值：
#   frozen                —— 已获取并通过全部校验（或被拒绝获取但仅登记元数据）
#   blocked               —— 官方机读来源在本环境不可达/不含目标年份，未获取
#   refused_over_size_cap —— 官方包超过大小上限且无官方单文件端点，拒绝下载
#
# classification 取值见 CLASSIFICATIONS。

SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    # --- B. local_pv_kw：PVWatts V8 模型来源（pvlib 实现）+ 许可 -------------
    {
        "source_id": "pvlib_pvwatts_license",
        "route": "B",
        "role": "model_license",
        "classification": "modeled_scenario",
        "url": "https://raw.githubusercontent.com/pvlib/pvlib-python/v0.15.2/LICENSE",
        "license": "BSD-3-Clause",
        "license_url": "https://raw.githubusercontent.com/pvlib/pvlib-python/v0.15.2/LICENSE",
        "pinned_ref": "v0.15.2",
        "local_name": "pvlib_v0.15.2_LICENSE.txt",
        "bytes": 1622,
        "sha256": "a02e12ddaada3cf0d5dbdd8affdd577c2eec640758a95842a304f7513b8c0be4",
        "data_year": None,
        "resolution": "n/a",
        "required_columns": (),
        "license_marker": "BSD 3-Clause",
        "status": "frozen",
    },
    {
        "source_id": "pvlib_pvwatts_model",
        "route": "B",
        "role": "model_source",
        "classification": "modeled_scenario",
        "url": "https://raw.githubusercontent.com/pvlib/pvlib-python/v0.15.2/pvlib/pvsystem.py",
        "license": "BSD-3-Clause",
        "license_url": "https://raw.githubusercontent.com/pvlib/pvlib-python/v0.15.2/LICENSE",
        "pinned_ref": "v0.15.2",
        "local_name": "pvlib_v0.15.2_pvsystem.py",
        "bytes": 117911,
        "sha256": "668afd274e69dd4741854f643640fc5d9052b86feba1b94da7ecebbaccfef8f3",
        "data_year": None,
        "resolution": "n/a",
        "required_columns": (),
        "license_marker": "def pvwatts_dc",
        "status": "frozen",
    },
    # --- C. wind_generation_kw：官方默认功率曲线与机组数据 + 许可 ------------
    {
        "source_id": "windpowerlib_license",
        "route": "C",
        "role": "model_license",
        "classification": "modeled_scenario",
        "url": "https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/LICENSE",
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/LICENSE",
        "pinned_ref": "v0.2.2",
        "local_name": "windpowerlib_v0.2.2_LICENSE.txt",
        "bytes": 1083,
        "sha256": "140f742e061d4c8e4c8a2bb45516d1d38a0e595a7d7ab6bad02d89dd9dd93f6a",
        "data_year": None,
        "resolution": "n/a",
        "required_columns": (),
        "license_marker": "MIT License",
        "status": "frozen",
    },
    {
        "source_id": "windpowerlib_power_curves",
        "route": "C",
        "role": "power_curve",
        "classification": "modeled_scenario",
        "url": (
            "https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/"
            "windpowerlib/data/default_turbine_data/power_curves.csv"
        ),
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/LICENSE",
        "pinned_ref": "v0.2.2",
        "local_name": "windpowerlib_v0.2.2_power_curves.csv",
        "bytes": 26042,
        "sha256": "7d91ddde701ce6d0ac4cacb31fac04b38f0664921b75ca44c174ca26cd394add",
        "data_year": None,
        "resolution": "per_turbine_wind_speed_curve",
        "required_columns": ("turbine_type", "0.0"),
        "license_marker": "turbine_type",
        "status": "frozen",
    },
    {
        "source_id": "windpowerlib_turbine_data",
        "route": "C",
        "role": "turbine_metadata",
        "classification": "modeled_scenario",
        "url": (
            "https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/"
            "windpowerlib/data/default_turbine_data/turbine_data.csv"
        ),
        "license": "MIT",
        "license_url": "https://raw.githubusercontent.com/wind-python/windpowerlib/v0.2.2/LICENSE",
        "pinned_ref": "v0.2.2",
        "local_name": "windpowerlib_v0.2.2_turbine_data.csv",
        "bytes": 22922,
        "sha256": "d379379ba20fe41ec151f74ad5ddf368f9e1cce596756ccea73888d0614c3840",
        "data_year": None,
        "resolution": "per_turbine_metadata",
        "required_columns": ("turbine_type", "nominal_power", "hub_height"),
        "license_marker": "turbine_type",
        "status": "frozen",
    },
    # --- A. carbon_intensity：官方机读来源不可达/不含 2024 -------------------
    {
        "source_id": "ema_grid_emission_factor_annual",
        "route": "A",
        "role": "carbon_intensity_candidate",
        "classification": "human_approved_external_low_resolution",
        "url": (
            "https://api-production.data.gov.sg/v2/public/api/datasets/"
            "d_3de362b580b2dd2fd50cc1006d4edd4f/metadata"
        ),
        "license": "Singapore Open Data Licence",
        "license_url": "https://data.gov.sg/open-data-licence",
        "pinned_ref": None,
        "local_name": None,
        "bytes": 0,
        "sha256": None,
        "data_year": None,
        "resolution": "annual_constant",
        "required_columns": (),
        "license_marker": None,
        "status": "blocked",
        "blocked_reason": (
            "ema.gov.sg 被 Incapsula 反爬拦截（返回挑战页，非真实内容）；"
            "data.gov.sg 的 EMA GEF 数据集 coverageEnd=2020-12-31，**不含 2024**，"
            "因此**无法**从本环境可达的官方机读来源核验 2024 值。"
            "人工决定 **B1**（2026-09-16）已批准 2024 年内常数 "
            "0.402 kgCO2/kWh（见 human_approved_parameters.carbon_intensity），"
            "classification = human_approved_external_low_resolution、"
            "resolution = annual_constant；**不得**描述为半小时实测或 "
            "half_hourly truth，且本轮**不**改变任何 readiness。"
        ),
        "coverage_start": "2005-01-01",
        "coverage_end": "2020-12-31",
        "target_year_unverifiable": 2024,
    },
    # --- D. arrival：官方 trace 包超上限且无单文件端点 ------------------------
    {
        "source_id": "azure_functions_2019_trace",
        "route": "D",
        "role": "arrival_candidate",
        "classification": "benchmark_trace",
        "url": (
            "https://github.com/Azure/AzurePublicDataset/releases/download/"
            "dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
        ),
        "license": "CC-BY-4.0",
        "license_url": "https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE",
        "pinned_ref": "dataset-functions-2019",
        "local_name": None,
        "bytes": 0,
        "sha256": None,
        "data_year": 2019,
        "resolution": "per_minute_invocations",
        "required_columns": (),
        "license_marker": None,
        "status": "refused_over_size_cap",
        "observed_content_length": 142968140,
        "refused_reason": (
            "官方发布包 Content-Length = 142,968,140 B ≈ 136.3 MiB，"
            "远超默认 16 MiB 上限；容器列举返回 PublicAccessNotPermitted，"
            "逐文件 blob 路径返回 HTTP 409（不可公开寻址）。"
            "按大小上限政策**拒绝盲目下载**，只登记元数据并升级人工。"
            "人工决定 **B3**（2026-09-16）已批准：**仅**该 pinned URL 适用 "
            "160 MiB 特例上限；通用 MAX_SOURCE_BYTES=16 MiB 保持不变；"
            "正式下载须**再次**核验 Content-Length、最终 URL、SHA-256、"
            "容器成员与许可。**本轮不下载该包**（status 仍为 "
            "refused_over_size_cap，未冻结）。"
        ),
        "exception_cap_bytes": 160 * 1024 * 1024,
        "decision_id": "B3",
        "published_revision": "revision 2, 20200618",
    },
)

FROZEN_SPECS = tuple(s for s in SOURCE_SPECS if s["status"] == "frozen")


# --- 严格类型工具（任何畸形输入只抛 SourcePolicyError） ----------------------

def _require_exact_keys(mapping: Any, *, field: str,
                        expected: Sequence[str]) -> dict:
    if not isinstance(mapping, dict):
        raise SourcePolicyError(f"{field} 必须是 object，实际 {type(mapping).__name__}")
    actual = set(mapping)
    wanted = set(expected)
    if actual != wanted:
        raise SourcePolicyError(
            f"{field} 键集合必须精确等于冻结 schema；"
            f"多出={sorted(actual - wanted)} 缺少={sorted(wanted - actual)}"
        )
    return mapping


def _require_plain_int(value: Any, *, field: str) -> int:
    """严格整数：拒绝 bool（`bool` 是 `int` 的子类）、浮点、字符串、None、容器。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourcePolicyError(f"{field} 必须是整数（bool 不算），实际 {value!r}")
    return value


def _require_str(value: Any, *, field: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise SourcePolicyError(f"{field} 必须是非空字符串，实际 {value!r}")
    return value


def _require_canonical_utc(value: Any, *, field: str) -> str:
    from datetime import datetime

    if not isinstance(value, str) or not value:
        raise SourcePolicyError(f"{field} 必须是非空字符串，实际 {value!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SourcePolicyError(f"{field} 不是合法 ISO-8601：{value!r}") from error
    if parsed.tzinfo is None:
        raise SourcePolicyError(f"{field} 必须带显式时区偏移：{value!r}")
    if parsed.isoformat() != value or not value.endswith("+00:00"):
        raise SourcePolicyError(f"{field} 必须是规范 UTC ISO-8601：{value!r}")
    return value


def _require_plain_date(value: Any, *, field: str) -> str:
    from datetime import date

    if not isinstance(value, str) or not value:
        raise SourcePolicyError(f"{field} 必须是 YYYY-MM-DD 字符串，实际 {value!r}")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise SourcePolicyError(f"{field} 不是合法日期：{value!r}") from error
    if parsed.isoformat() != value:
        raise SourcePolicyError(f"{field} 必须规范化为 YYYY-MM-DD：{value!r}")
    return value


# --- 校验守卫 ----------------------------------------------------------------

def effective_cap(spec: dict) -> int:
    """该来源适用的上限：默认 16 MiB；**仅**经人工特例批准的 pinned URL 可放宽。"""
    exception = spec.get("exception_cap_bytes")
    if exception is None:
        return MAX_SOURCE_BYTES
    return max(MAX_SOURCE_BYTES, _require_plain_int(
        exception, field=f"{spec.get('source_id')}.exception_cap_bytes"))


def assert_within_size_cap(spec: dict) -> None:
    """单源大小上限（发任何请求**之前**判定）。"""
    cap = effective_cap(spec)
    observed = spec.get("observed_content_length") or spec.get("bytes") or 0
    if observed > cap:
        raise SourcePolicyError(
            f"{spec['source_id']} 大小 {observed} B 超过上限 {cap} B"
            "（默认 16 MiB；锚点为既有最大 raw 资产 14.47 MiB）"
        )


def assert_year_matches(spec: dict, *, target_year: int) -> None:
    """年份守卫：来源的数据年份必须与目标年份一致（拒绝版本错配）。"""
    data_year = spec.get("data_year")
    if data_year is not None and data_year != target_year:
        raise SourcePolicyError(
            f"{spec['source_id']} 数据年份 {data_year} 与目标年份 {target_year} 不符："
            "不得把其它年份的 profile 冒充目标年份真值"
        )


def assert_resolution_matches(spec: dict, *, required: str) -> None:
    """分辨率守卫：低分辨率来源不得冒充高分辨率真值。"""
    resolution = spec.get("resolution")
    if resolution != required:
        raise SourcePolicyError(
            f"{spec['source_id']} 分辨率 {resolution!r} 与要求的 {required!r} 不符："
            "低时间分辨率来源不得冒充高分辨率真值"
        )


def assert_parameters_approved(params: dict, *, route: str) -> None:
    """参数守卫：只有**结构化人工批准**的参数才可用于正式 PV / 风电。

    `pv_capacity_kw` **可以**是 500 —— 只要它是一次**新的人工批准**
    （带 `decision_id` 与批准日期），而不是从旧仿真假设「legacy 继承」来的。
    这条区分是本轮修复的重点：原先「只要出现 `pv_capacity_kw` 就一律拒绝」
    的逻辑不可延续（合法的批准路径也被堵死）。
    """
    if route not in ("B", "C"):
        raise SourcePolicyError(f"route 必须是 'B' 或 'C'，实际 {route!r}")
    if not isinstance(params, dict):
        raise SourcePolicyError(f"参数块必须是 object，实际 {type(params).__name__}")
    for name, entry in params.items():
        if not isinstance(entry, dict):
            raise SourcePolicyError(
                f"{route} 路线参数 {name} 必须是结构化的批准记录（含 value/unit/"
                f"status/decision_id/approved_on），"
                f"实际 {type(entry).__name__}（{entry!r}）"
            )
        if set(entry) != set(PARAM_KEYS):
            raise SourcePolicyError(
                f"{route} 路线参数 {name} 键集合必须精确等于 {list(PARAM_KEYS)}，"
                f"实际 {sorted(entry)}"
            )
        status = entry["status"]
        if status == "legacy_inherited":
            raise SourcePolicyError(
                f"{route} 路线参数 {name} 标记为 legacy_inherited："
                "不得从旧仿真假设继承为正式口径"
            )
        if status != "human_approved":
            raise SourcePolicyError(
                f"{route} 路线参数 {name} 的状态是 {status!r}："
                "未经人工批准的参数不得生成正式输出"
            )
        decision_id = entry["decision_id"]
        if not isinstance(decision_id, str) or not decision_id:
            raise SourcePolicyError(
                f"{route} 路线参数 {name} 缺少 decision_id："
                "人工批准必须可追溯到具体决定"
            )
        _require_plain_date(entry["approved_on"], field=f"{name}.approved_on")


def assert_no_silent_replay(*, trace_year: int, target_year: int) -> None:
    """重放守卫：benchmark trace 不得被静默重放成目标年份的 arrival。"""
    if trace_year != target_year:
        raise SourcePolicyError(
            f"arrival trace 年份 {trace_year} 与目标年份 {target_year} 不符："
            "不得把该 trace 静默 replay 成目标年份的 arrival"
        )


# --- 网络 --------------------------------------------------------------------

def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host in ("127.0.0.1", "localhost", "::1")


def fetch_bytes(url: str, *, opener: Callable[[str], Any] | None = None) -> bytes:
    """获取 URL 的字节。**只允许 HTTPS**（loopback 的 http 仅供本地 fixture）。"""
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme != "https" and not (scheme == "http" and _is_loopback(url)):
        raise SourcePolicyError(
            f"只允许 HTTPS 官方来源（loopback http 除外），实际 {url!r}"
        )
    try:
        if opener is not None:
            response = opener(url)
            return response.read() if hasattr(response, "read") else bytes(response)
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            return response.read()
    except urllib.error.HTTPError as error:
        raise SourcePolicyError(f"{url} 返回 HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise SourcePolicyError(f"{url} 不可达：{error.reason}") from error


def _read_url(url: str) -> Any:
    return urllib.request.urlopen(url, timeout=60)  # noqa: S310


def _now_utc() -> str:
    """当前规范 UTC 时间戳（独立函数，便于测试注入）。"""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# --- 文件校验与原子安装 ------------------------------------------------------

def validate_frozen_file(spec: dict, path: Path) -> None:
    """大小 / hash / 许可标记 / schema 逐项校验。"""
    body = path.read_bytes()
    if len(body) != spec["bytes"]:
        raise SourcePolicyError(
            f"{spec['source_id']} 字节数 {len(body)} != 冻结值 {spec['bytes']}"
        )
    actual = hashlib.sha256(body).hexdigest()
    if actual != spec["sha256"]:
        raise SourcePolicyError(
            f"{spec['source_id']} SHA-256 不符：冻结={spec['sha256']} 实际={actual}"
        )
    marker = spec.get("license_marker")
    if marker is not None:
        text = body.decode("utf-8", errors="replace")
        if marker not in text:
            raise SourcePolicyError(
                f"{spec['source_id']} 缺少必需标记 {marker!r}（许可/schema 校验失败）"
            )
    columns = spec.get("required_columns") or ()
    if columns:
        header = body.decode("utf-8", errors="replace").splitlines()[0]
        present = header.split(",")
        missing = [c for c in columns if c not in present]
        if missing:
            raise SourcePolicyError(
                f"{spec['source_id']} schema 不符：缺少列 {missing}"
            )


def _atomic_install(path: Path, body: bytes) -> None:
    """同目录临时文件 + `os.replace` 原子安装；失败不留半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
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


def install_frozen(spec: dict, body: bytes, dest_dir: Path) -> Path:
    """安装到 `dest_dir/<local_name>`；**已存在且内容不同则拒绝覆盖**。"""
    local_name = spec["local_name"]
    if not local_name:
        raise SourcePolicyError(f"{spec['source_id']} 没有 local_name，不能安装")
    target = Path(dest_dir) / local_name
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() == spec["sha256"]:
            return target
        raise SourcePolicyError(
            f"{target} 已存在且内容不同：拒绝覆盖（先人工确认再处理）"
        )
    _atomic_install(target, body)
    validate_frozen_file(spec, target)
    return target


def download_to_temp(spec: dict, dest_dir: Path, *,
                     opener: Callable[[str], Any] | None = None) -> Path:
    """获取 → 校验 → 原子安装。任一步失败都不留下半文件。"""
    assert_within_size_cap(spec)
    body = fetch_bytes(spec["url"], opener=opener)
    if len(body) != spec["bytes"]:
        raise SourcePolicyError(
            f"{spec['source_id']} 远端字节数 {len(body)} != 登记值 {spec['bytes']}"
        )
    if hashlib.sha256(body).hexdigest() != spec["sha256"]:
        raise SourcePolicyError(f"{spec['source_id']} 远端 SHA-256 与登记值不符")
    return install_frozen(spec, body, dest_dir)


# --- manifest ----------------------------------------------------------------

def _require_spec_field(spec: dict, entry: dict, field: str) -> Any:
    expected = spec.get(field)
    actual = entry.get(field)
    if actual != expected:
        raise SourcePolicyError(
            f"{spec['source_id']}.{field} 必须与代码内 SOURCE_SPECS 恒等："
            f"期望 {expected!r}，实际 {actual!r}"
        )
    return actual


def validate_public_source_manifest(
    manifest_path: Path = MANIFEST_PATH, root: Path = REPO_ROOT
) -> dict:
    """**单一严格入口**：校验冻结 manifest 的每一层。

    `verify_local`、`--verify` 与 `--fetch` 完成后的复核都必须调用它，
    不得另维护一套更弱的规则。任何畸形输入只抛 `SourcePolicyError`。
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise SourcePolicyError(f"缺少冻结 manifest {manifest_path}：先运行 --fetch")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourcePolicyError(f"{manifest_path} 不是合法 JSON：{error}") from error

    payload = _require_exact_keys(payload, field="manifest", expected=TOP_KEYS)
    if payload["schema"] != MANIFEST_SCHEMA:
        raise SourcePolicyError(
            f"schema 必须是 {MANIFEST_SCHEMA!r}，实际 {payload['schema']!r}"
        )
    if payload["contract_version"] != CONTRACT_VERSION:
        raise SourcePolicyError(
            f"contract_version 必须是 {CONTRACT_VERSION!r}，"
            f"实际 {payload['contract_version']!r}"
        )
    if _require_plain_int(payload["max_source_bytes"],
                          field="max_source_bytes") != MAX_SOURCE_BYTES:
        raise SourcePolicyError(f"max_source_bytes 必须是 {MAX_SOURCE_BYTES}")
    if payload["max_source_bytes_anchor"] != MAX_SOURCE_BYTES_ANCHOR:
        raise SourcePolicyError("max_source_bytes_anchor 必须等于冻结的锚点说明")
    _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")

    sources = payload["sources"]
    if not isinstance(sources, list):
        raise SourcePolicyError(f"sources 必须是 list，实际 {type(sources).__name__}")
    expected_ids = [spec["source_id"] for spec in SOURCE_SPECS]
    actual_ids = []
    for entry in sources:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_id"), str):
            raise SourcePolicyError("sources[*] 必须是含字符串 source_id 的 object")
        actual_ids.append(entry["source_id"])
    if actual_ids != expected_ids:
        raise SourcePolicyError(
            "sources 的 source_id 集合与**顺序**必须精确等于 SOURCE_SPECS："
            f"期望 {expected_ids}，实际 {actual_ids}"
        )

    for spec, entry in zip(SOURCE_SPECS, sources, strict=True):
        status = entry.get("status")
        if status not in STATUS_EXTRA_KEYS:
            raise SourcePolicyError(
                f"{spec['source_id']}.status 未知：{status!r}"
            )
        _require_exact_keys(
            entry, field=f"sources[{spec['source_id']}]",
            expected=SOURCE_COMMON_KEYS + STATUS_EXTRA_KEYS[status],
        )
        # 固定声明必须与代码内 SOURCE_SPECS **逐字段恒等**
        for field in ("route", "role", "classification", "status", "url", "license",
                      "license_url", "pinned_ref", "data_year", "resolution"):
            _require_spec_field(spec, entry, field)
        _require_str(entry["url"], field=f"{spec['source_id']}.url")
        _require_str(entry["license"], field=f"{spec['source_id']}.license")

        if status == "frozen":
            _require_str(entry["logical_path"],
                         field=f"{spec['source_id']}.logical_path")
            if _require_plain_int(entry["bytes"],
                                  field=f"{spec['source_id']}.bytes") != spec["bytes"]:
                raise SourcePolicyError(f"{spec['source_id']}.bytes 与冻结值不符")
            if entry["sha256"] != spec["sha256"]:
                raise SourcePolicyError(f"{spec['source_id']}.sha256 与冻结值不符")
            target = Path(root) / entry["logical_path"]
            if not target.is_file():
                raise SourcePolicyError(
                    f"{spec['source_id']} 缺少本地冻结文件 {entry['logical_path']}"
                )
            validate_frozen_file(spec, target)
        else:
            # blocked / refused 来源**同样校验**，不得被跳过
            reason_field = ("blocked_reason" if status == "blocked"
                            else "refused_reason")
            _require_str(entry[reason_field],
                         field=f"{spec['source_id']}.{reason_field}")
            if status == "refused_over_size_cap":
                if _require_plain_int(
                    entry["observed_content_length"],
                    field=f"{spec['source_id']}.observed_content_length",
                ) != spec["observed_content_length"]:
                    raise SourcePolicyError(
                        f"{spec['source_id']}.observed_content_length 与实测值不符"
                    )
                if _require_plain_int(
                    entry["exception_cap_bytes"],
                    field=f"{spec['source_id']}.exception_cap_bytes",
                ) != spec["exception_cap_bytes"]:
                    raise SourcePolicyError(
                        f"{spec['source_id']}.exception_cap_bytes 与批准的例外上限不符"
                    )
                if entry["decision_id"] != spec["decision_id"]:
                    raise SourcePolicyError(
                        f"{spec['source_id']}.decision_id 与人工决定不符"
                    )

    # 参数块：已批准与未批准都必须**完整恒等**
    for key, expected in (("human_approved_parameters", HUMAN_APPROVED_PARAMETERS),
                          ("unapproved_parameters", UNAPPROVED_PARAMETERS)):
        block = payload[key]
        if not isinstance(block, dict) or set(block) != set(expected):
            raise SourcePolicyError(f"{key} 的顶层键必须精确等于冻结集合")
        for group, entries in expected.items():
            if not isinstance(block[group], dict) or set(block[group]) != set(entries):
                raise SourcePolicyError(f"{key}.{group} 的参数键必须精确等于冻结集合")
            for name, want in entries.items():
                got = block[group][name]
                _require_exact_keys(got, field=f"{key}.{group}.{name}",
                                    expected=PARAM_KEYS)
                if got != want:
                    raise SourcePolicyError(
                        f"{key}.{group}.{name} 与冻结的批准记录不符"
                    )

    if not isinstance(payload["red_lines"], list) or tuple(payload["red_lines"]) != RED_LINES:
        raise SourcePolicyError("red_lines 必须与冻结的四条红线逐项恒等")

    readiness = _require_exact_keys(payload["readiness"], field="readiness",
                                    expected=tuple(READINESS))
    for key, expected in READINESS.items():
        if readiness[key] is not expected:
            raise SourcePolicyError(
                f"readiness.{key} 必须严格为 {expected}，实际 {readiness[key]!r}"
            )

    return {"ok": True, "sources": len(sources),
            "frozen": sum(1 for e in sources if e["status"] == "frozen"),
            "not_frozen": sum(1 for e in sources if e["status"] != "frozen")}


def validated_source_ids(manifest_path: Path = MANIFEST_PATH,
                         root: Path = REPO_ROOT) -> tuple[str, ...]:
    """返回**全部**被校验过的 source_id（含 `blocked` / `refused`）。"""
    validate_public_source_manifest(manifest_path, root)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return tuple(entry["source_id"] for entry in payload["sources"])


def build_manifest(*, frozen_at_utc: str) -> dict:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "max_source_bytes_anchor": MAX_SOURCE_BYTES_ANCHOR,
        "frozen_at_utc": frozen_at_utc,
        "sources": [
            {
                "source_id": spec["source_id"],
                "route": spec["route"],
                "role": spec["role"],
                "classification": spec["classification"],
                "status": spec["status"],
                "url": spec["url"],
                "license": spec["license"],
                "license_url": spec["license_url"],
                "pinned_ref": spec["pinned_ref"],
                "data_year": spec["data_year"],
                "resolution": spec["resolution"],
                **(
                    {
                        "logical_path": f"{_LOGICAL_PREFIX}/{spec['local_name']}",
                        "bytes": spec["bytes"],
                        "sha256": spec["sha256"],
                    }
                    if spec["status"] == "frozen"
                    else {"blocked_reason": spec["blocked_reason"]}
                    if spec["status"] == "blocked"
                    else {
                        "refused_reason": spec["refused_reason"],
                        "observed_content_length": spec["observed_content_length"],
                        "exception_cap_bytes": spec["exception_cap_bytes"],
                        "decision_id": spec["decision_id"],
                    }
                ),
            }
            for spec in SOURCE_SPECS
        ],
        "human_approved_parameters": _copy_block(HUMAN_APPROVED_PARAMETERS),
        "unapproved_parameters": _copy_block(UNAPPROVED_PARAMETERS),
        "red_lines": list(RED_LINES),
        "readiness": dict(READINESS),
    }
    _assert_portable(payload)
    return payload


def _copy_block(block: dict) -> dict:
    return {group: {name: dict(entry) for name, entry in entries.items()}
            for group, entries in block.items()}


def _assert_portable(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise SourcePolicyError("manifest 含绝对路径，必须使用仓库相对 POSIX 路径")


def write_manifest_atomic(payload: dict, path: Path = MANIFEST_PATH) -> None:
    """写正式 manifest：**已存在且语义不同 → fail closed，禁止覆盖**。

    完全相同 → 直接返回（**不改变 `mtime_ns`**）；首次生成 → 临时文件 + 原子安装。
    """
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SourcePolicyError(
                f"已存在的 manifest {path} 不是合法 JSON：{error}"
            ) from error
        if existing == payload:
            return
        raise SourcePolicyError(
            f"已存在的 manifest {path} 与候选**语义不同**：拒绝覆盖冻结产物"
            "（如需变更请先人工确认并显式删除旧产物）"
        )
    _atomic_install(path, text.encode("utf-8"))


def verify_local(manifest_path: Path = MANIFEST_PATH,
                 root: Path = REPO_ROOT) -> dict:
    """只校验本地冻结文件；**不联网**；manifest 缺失即干净 fail closed。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise SourcePolicyError(
            f"缺少冻结 manifest {manifest_path}：先运行 --fetch 生成"
        )
    result = validate_public_source_manifest(manifest_path, root)
    return {"ok": True, "failures": [],
            "checked": result["sources"]}


def run_fetch(dest_dir: Path = RAW_DIR, *, manifest_path: Path = MANIFEST_PATH,
              opener: Callable[[str], Any] | None = None) -> dict:
    """显式联网：逐个获取 frozen 来源并写 manifest（原子）。

    **只**对 `status == "frozen"` 的来源做前置大小判定与下载；
    `blocked` / `refused_over_size_cap` 的来源**本就不下载**，
    它们的元数据与理由已登记在 manifest 里（超限来源不得阻断其余来源）。

    **不可变性**：若目标 manifest 已存在且**合法**，则**复用**它的
    `frozen_at_utc`，不得因当前墙钟而改变正式 manifest；
    已存在但**语义不同**时由 `write_manifest_atomic` fail closed。
    """
    for spec in FROZEN_SPECS:
        assert_within_size_cap(spec)
    for spec in FROZEN_SPECS:
        download_to_temp(spec, dest_dir, opener=opener)
    frozen_at = existing_frozen_at_utc(manifest_path) or _now_utc()
    write_manifest_atomic(build_manifest(frozen_at_utc=frozen_at), manifest_path)
    validate_public_source_manifest(manifest_path, REPO_ROOT)
    return {"fetched": [s["source_id"] for s in FROZEN_SPECS],
            "refused": [s["source_id"] for s in SOURCE_SPECS
                        if s["status"] != "frozen"],
            "frozen_at_utc": frozen_at}


def existing_frozen_at_utc(manifest_path: Path = MANIFEST_PATH) -> str | None:
    """已存在且**合法**的 manifest 的 `frozen_at_utc`；否则 None。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = payload["frozen_at_utc"]
        _require_canonical_utc(value, field="frozen_at_utc")
    except (json.JSONDecodeError, KeyError, TypeError, SourcePolicyError):
        return None
    return value


def _report() -> int:
    for spec in SOURCE_SPECS:
        print(f"  {spec['route']}  {spec['source_id']:<38} {spec['status']:<24} "
              f"{spec['classification']}")
    print(f"  frozen={len(FROZEN_SPECS)}  non_frozen="
          f"{len(SOURCE_SPECS) - len(FROZEN_SPECS)}  cap={MAX_SOURCE_BYTES} B")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1.3f-b 公开数据源获取 / 校验（默认不联网）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fetch", action="store_true",
                       help="显式联网获取并冻结（默认不联网）")
    group.add_argument("--verify", action="store_true",
                       help="只校验本地冻结文件（默认行为）")
    group.add_argument("--report", action="store_true", help="只打印来源登记表")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.report:
        return _report()
    if args.fetch:
        result = run_fetch()
        print(f"fetched={result['fetched']}")
        print(f"not_frozen={result['refused']}")
    result = verify_local()
    if not result["ok"]:
        for failure in result["failures"]:
            print(f"verify failed: {failure}", file=sys.stderr)
        return 1
    print(f"verify ok（已校验 {result['checked']} 个来源，含 blocked/refused）")
    _report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
