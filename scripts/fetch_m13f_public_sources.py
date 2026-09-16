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
**任何超过上限的来源一律拒绝下载**，只登记元数据并升级人工。

## 四条红线（固化为可测试守卫）

1. `assert_year_matches()`：2026 光伏 profile **不得**冒充 2024 真值；
2. `assert_resolution_matches()`：年度碳因子**不得**冒充半小时碳强度真值；
3. `assert_parameters_approved()`：未批准参数**不得**生成正式 PV / 风电
   （`pv_capacity_kw=500` 是旧仿真假设，**不得继承**）；
4. `assert_no_silent_replay()`：Azure 2019 trace **不得**被静默重放成 2024 arrival。

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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data/manifest/m13f_public_sources.json"
RAW_DIR = REPO_ROOT / "data/raw/public_benchmarks"

MANIFEST_SCHEMA = "m1.3f-public-sources-v1"
CONTRACT_VERSION = "contract-v8"

# 先验锚点：既有最大 raw 资产 14.47 MiB 向上取整。
MAX_SOURCE_BYTES = 16 * 1024 * 1024

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
        "classification": "external_low_resolution",
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
        "resolution": "annual",
        "required_columns": (),
        "license_marker": None,
        "status": "blocked",
        "blocked_reason": (
            "ema.gov.sg 被 Incapsula 反爬拦截（返回挑战页，非真实内容）；"
            "data.gov.sg 的 EMA GEF 数据集 coverageEnd=2020-12-31，"
            "**不含 2024**。因此卡片要求的「机器核验 2024 GEF = 0.402 kg CO2/kWh」"
            "在本环境无法完成，且不得写死该数值。"
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
            "远超 16 MiB 上限；容器列举返回 PublicAccessNotPermitted，"
            "逐文件 blob 路径返回 HTTP 409（不可公开寻址）。"
            "按大小上限政策**拒绝盲目下载**，只登记元数据并升级人工。"
        ),
        "published_revision": "revision 2, 20200618",
    },
)

FROZEN_SPECS = tuple(s for s in SOURCE_SPECS if s["status"] == "frozen")


# --- 校验守卫 ----------------------------------------------------------------

def assert_within_size_cap(spec: dict) -> None:
    """单源大小上限（发任何请求**之前**判定）。"""
    observed = spec.get("observed_content_length") or spec.get("bytes") or 0
    if observed > MAX_SOURCE_BYTES:
        raise SourcePolicyError(
            f"{spec['source_id']} 大小 {observed} B 超过上限 {MAX_SOURCE_BYTES} B"
            "（16 MiB；锚点为既有最大 raw 资产 14.47 MiB）"
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
    """参数守卫：未获人工批准的物理参数不得生成正式 PV / 风电。"""
    if route not in ("B", "C"):
        raise SourcePolicyError(f"route 必须是 'B' 或 'C'，实际 {route!r}")
    for name, value in params.items():
        if value == UNAPPROVED:
            raise SourcePolicyError(
                f"{route} 路线参数 {name} 的状态是 {UNAPPROVED}："
                "未经人工批准的参数不得生成正式输出"
            )
    if route == "B" and "pv_capacity_kw" in params:
        raise SourcePolicyError(
            "pv_capacity_kw 不得从旧仿真假设（500）继承为正式口径；"
            "必须由人工显式批准并提供来源"
        )


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


def download_to_temp(spec: dict, dest_dir: Path) -> Path:
    """获取 → 校验 → 原子安装。任一步失败都不留下半文件。"""
    assert_within_size_cap(spec)
    body = fetch_bytes(spec["url"])
    if len(body) != spec["bytes"]:
        raise SourcePolicyError(
            f"{spec['source_id']} 远端字节数 {len(body)} != 登记值 {spec['bytes']}"
        )
    if hashlib.sha256(body).hexdigest() != spec["sha256"]:
        raise SourcePolicyError(f"{spec['source_id']} 远端 SHA-256 与登记值不符")
    return install_frozen(spec, body, dest_dir)


# --- manifest ----------------------------------------------------------------

def build_manifest(*, frozen_at_utc: str) -> dict:
    payload = {
        "schema": MANIFEST_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "max_source_bytes_anchor": (
            "既有 M1.2 raw 资产最大 14.47 MiB（sasea_demand_2024.zip）向上取整"
        ),
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
                "logical_path": (
                    f"{_LOGICAL_PREFIX}/{spec['local_name']}"
                    if spec["status"] == "frozen" else None
                ),
                "bytes": spec["bytes"],
                "sha256": spec["sha256"],
                "data_year": spec["data_year"],
                "resolution": spec["resolution"],
                **({"blocked_reason": spec["blocked_reason"]}
                   if "blocked_reason" in spec else {}),
                **({"refused_reason": spec["refused_reason"],
                    "observed_content_length": spec["observed_content_length"]}
                   if "refused_reason" in spec else {}),
            }
            for spec in SOURCE_SPECS
        ],
        "unapproved_parameters": {
            "local_pv_kw": {
                "pv_capacity_kw": UNAPPROVED,
                "tilt_deg": UNAPPROVED,
                "azimuth_deg": UNAPPROVED,
                "array_type": UNAPPROVED,
                "losses_pct": UNAPPROVED,
            },
            "wind_generation_kw": {
                "hub_height_m": UNAPPROVED,
                "shear_exponent": UNAPPROVED,
                "turbine_model": UNAPPROVED,
                "rated_capacity_kw": UNAPPROVED,
            },
        },
        "red_lines": [
            "2026 Solar Generation Profile 不得冒充 2024 真值",
            "年度碳因子不得冒充半小时碳强度真值",
            "未批准参数不得生成正式 PV / 风电；pv_capacity_kw=500 不得继承",
            "Azure 2019 trace 不得被静默重放成 2024 arrival",
        ],
        "readiness": dict(READINESS),
    }
    _assert_portable(payload)
    return payload


def _assert_portable(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise SourcePolicyError("manifest 含绝对路径，必须使用仓库相对 POSIX 路径")


def write_manifest_atomic(payload: dict, path: Path = MANIFEST_PATH) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            return
    _atomic_install(path, text.encode("utf-8"))


def verify_local(manifest_path: Path = MANIFEST_PATH,
                 root: Path = REPO_ROOT) -> dict:
    """只校验本地冻结文件；**不联网**；manifest 缺失即干净 fail closed。"""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise SourcePolicyError(
            f"缺少冻结 manifest {manifest_path}：先运行 --fetch 生成"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SourcePolicyError(f"{manifest_path} 不是合法 JSON：{error}") from error
    failures: list[str] = []
    for entry in payload["sources"]:
        if entry["status"] != "frozen":
            continue
        spec = next(s for s in SOURCE_SPECS if s["source_id"] == entry["source_id"])
        target = Path(root) / entry["logical_path"]
        if not target.exists():
            failures.append(f"{entry['source_id']}: 缺少 {entry['logical_path']}")
            continue
        try:
            validate_frozen_file(spec, target)
        except SourcePolicyError as error:
            failures.append(str(error))
    return {"ok": not failures, "failures": failures,
            "checked": sum(1 for e in payload["sources"] if e["status"] == "frozen")}


def run_fetch(dest_dir: Path = RAW_DIR) -> dict:
    """显式联网：逐个获取 frozen 来源并写 manifest（原子）。

    **只**对 `status == "frozen"` 的来源做前置大小判定与下载；
    `blocked` / `refused_over_size_cap` 的来源**本就不下载**，
    它们的元数据与理由已登记在 manifest 里（超限来源不得阻断其余来源）。
    """
    for spec in FROZEN_SPECS:
        assert_within_size_cap(spec)
    for spec in FROZEN_SPECS:
        download_to_temp(spec, dest_dir)
    frozen_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    write_manifest_atomic(build_manifest(frozen_at_utc=frozen_at))
    return {"fetched": [s["source_id"] for s in FROZEN_SPECS],
            "refused": [s["source_id"] for s in SOURCE_SPECS
                        if s["status"] != "frozen"]}


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
    print(f"verify ok ({result['checked']} frozen sources)")
    _report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
