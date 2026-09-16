"""M1.3f-c：物化 Singapore-2024 四类外生驱动表。

产出：

- `data/processed/singapore_2024/exogenous_drivers.parquet`（17,568 行 × 5 列）
- `data/manifest/singapore_2024_exogenous.json`（输出 manifest）
- `data/manifest/m13f_materialization_sources.json`（**v2** 来源 manifest）

本脚本**不**改写 `half_hour.parquet`、**不**改写 M1.3f-b 的
`m13f_public_sources.json`（v1 保持字节不变）、**不**构造 `ScenarioBundle`、
**不**接入训练。

## 两个必须显式登记的口径（都不是「已批准参数」）

1. **PV 的两处 pvlib 默认选择**（辐照分解 `erbs`、透射 `isotropic`、
   反照率 `0.25`）——见 `scenario.exogenous_drivers`。
2. **Azure trace 的日期映射**：官方 archive 的**成员名里没有任何日期**，
   官方说明只写「collected in July of 2019」「14 files, one file per 24-h
   period」。日期映射取自**官方分析 notebook** 的绘图轴
   （`2019-07-15 14:00 UTC … 2019-07-28 23:59 UTC`）：
   **`d_k ↔ 2019-07-15 + (k-1) 天`（2019-07-15 是星期一，14 天正好两周）**，
   文件内第 `m` 分钟视为该日的第 `m` 分钟。
   **本条为可复现的显式假设，需人工确认。**

## 网络纪律

只有显式 `--fetch-arrival` 才联网；下载走临时文件 + 完整 SHA-256 校验 +
原子安装；已存在且不同则拒绝覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.exogenous_drivers import (
    ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR,
    ARRIVAL_SCALE_BASIS,
    ARRIVAL_SEED,
    ARRIVAL_UNIT,
    CANONICAL_MANIFEST,
    CANONICAL_PARQUET,
    CARBON_CLASSIFICATION,
    CARBON_KG_PER_KWH,
    CARBON_RESOLUTION,
    CARBON_SOURCE_URL,
    CARBON_UNIT,
    PVLIB_DEFAULTS_NOT_HUMAN_APPROVED,
    PV_PARAMS,
    PV_UNIT,
    SPLIT_MANIFEST,
    WIND_CURVE_FILE,
    WIND_CURVE_SHA256,
    WIND_PARAMS,
    WIND_UNIT,
    ExogenousDriverError,
    FrozenInputs,
    arrival_counts_from_trace,
    arrival_rate_template,
    build_frame,
    load_frozen_inputs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PARQUET = REPO_ROOT / "data/processed/singapore_2024/exogenous_drivers.parquet"
OUT_MANIFEST = REPO_ROOT / "data/manifest/singapore_2024_exogenous.json"
SOURCE_MANIFEST = REPO_ROOT / "data/manifest/m13f_materialization_sources.json"
V1_MANIFEST = REPO_ROOT / "data/manifest/m13f_public_sources.json"
PUBLIC_BENCHMARKS_DIR = REPO_ROOT / "data/raw/public_benchmarks"
ARRIVAL_ARCHIVE = PUBLIC_BENCHMARKS_DIR / "azurefunctions_dataset2019.tar.xz"

SOURCE_SCHEMA = "m1.3f-c-materialization-sources-v1"
OUTPUT_SCHEMA = "m1.3c-singapore-2024-exogenous-v1"
CONTRACT_VERSION = "contract-v8"

# B3 批准的精确资产
AZURE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
)
AZURE_CONTENT_LENGTH = 142968140
AZURE_LICENSE = "CC-BY-4.0"
AZURE_LICENSE_URL = "https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE"
AZURE_EXCEPTION_CAP_BYTES = 160 * 1024 * 1024
AZURE_DECISION_ID = "B3"

INVOCATION_MEMBER_PREFIX = "invocations_per_function_md.anon.d"
# 官方 notebook 的绘图轴 + 「14 个 24 小时文件」= 两周
TRACE_FIRST_DAY = "2019-07-15"
TRACE_DATE_MAPPING = (
    "官方 archive 的成员名**不含日期**；官方说明仅写 'collected in July of 2019' 与 "
    "'14 files, one file per 24-h period'。本卡据官方分析 notebook 的绘图轴"
    "（2019-07-15 14:00 UTC … 2019-07-28 23:59 UTC）取 "
    "d_k ↔ 2019-07-15 + (k-1) 天（2019-07-15 为星期一，14 天正好两周），"
    "文件内第 m 分钟视为该日第 m 分钟。**该映射为显式可复现假设，需人工确认。**"
)

READINESS: dict[str, bool] = {
    "local_pv_kw_ready": True,
    "wind_generation_kw_ready": True,
    "carbon_intensity_ready": True,
    "arrival_ready": True,
    "exogenous_drivers_ready": True,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}


class MaterializationError(ValueError):
    """物化的**明确失败**（fail closed）。"""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def resolve_materializer_revision() -> str:
    """本物化实现的 revision（由 Git 解析，不用漂移的 HEAD）。"""
    paths = ("scenario/exogenous_drivers.py",
             "scripts/materialize_singapore_exogenous.py")
    revision = _git("log", "-1", "--format=%H", "--", *paths).strip()
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise MaterializationError(f"materializer revision 无效：{revision!r}")
    return revision


# --- 原子安装 ----------------------------------------------------------------

def _atomic_write_bytes(path: Path, body: bytes) -> None:
    path = Path(path)
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


def write_json_atomic(payload: dict, path: Path) -> None:
    """写 JSON：**已存在且不同 → fail closed，禁止覆盖**；相同则不重写。"""
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise MaterializationError(
                f"已存在的 {path} 不是合法 JSON：{error}"
            ) from error
        if existing == payload:
            return
        raise MaterializationError(
            f"已存在的 {path} 与候选**语义不同**：拒绝覆盖冻结产物"
        )
    _atomic_write_bytes(path, text.encode("utf-8"))


# --- Azure archive -----------------------------------------------------------

def fetch_bytes(url: str, *, opener: Callable[[str], Any] | None = None) -> bytes:
    """HTTPS-only 读取（loopback http 仅供本地 fixture）。"""
    scheme = urllib.parse.urlsplit(url).scheme
    host = urllib.parse.urlsplit(url).hostname or ""
    if scheme != "https" and not (scheme == "http" and host in ("127.0.0.1", "localhost")):
        raise MaterializationError(f"只允许 HTTPS 官方来源，实际 {url!r}")
    try:
        if opener is not None:
            response = opener(url)
            return response.read() if hasattr(response, "read") else bytes(response)
        with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
            verify_final_url(url, response.geturl())
            return response.read()
    except urllib.error.HTTPError as error:
        raise MaterializationError(f"{url} 返回 HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise MaterializationError(f"{url} 不可达：{error.reason}") from error


# GitHub release asset 会 302 到它自己的签名 CDN；这不是第三方镜像
_ALLOWED_ASSET_HOSTS = (
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
)


def verify_final_url(pinned_url: str, final_url: str) -> str:
    """核验重定向：最终 host 必须仍在 **GitHub 资产域**内，且文件名精确一致。

    **不**要求「零重定向」（GitHub release asset 必然 302 到其自有 CDN），
    但**必须**挡住跳到任意第三方主机或改名的响应。
    """
    pinned = urllib.parse.urlsplit(pinned_url)
    final = urllib.parse.urlsplit(final_url)
    if final.scheme != "https":
        raise MaterializationError(f"最终 URL 必须是 HTTPS：{final_url!r}")
    if final.hostname not in _ALLOWED_ASSET_HOSTS:
        raise MaterializationError(
            f"最终 URL 跳到非 GitHub 资产域：{final.hostname!r}"
        )
    pinned_name = Path(pinned.path).name
    if pinned_name not in final.path and not (
        urllib.parse.parse_qs(final.query).get(
            "response-content-disposition", [""])[0].find(pinned_name) >= 0
    ):
        raise MaterializationError(
            f"最终 URL 未指向 pinned 资产 {pinned_name!r}：{final.path!r}"
        )
    return final.hostname


def content_length_of(url: str) -> int:
    """只取 `Content-Length`（**不下载正文**），用于下载前的上限判定。"""
    request = urllib.request.Request(url, method="HEAD")  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            final_url = response.geturl()
            length = response.headers.get("Content-Length")
    except urllib.error.HTTPError as error:
        raise MaterializationError(f"HEAD {url} 返回 HTTP {error.code}") from error
    if length is None:
        raise MaterializationError("响应缺少 Content-Length：拒绝盲目下载")
    verify_final_url(url, final_url)
    return int(length)


def fetch_arrival_archive(
    *, dest: Path = ARRIVAL_ARCHIVE, opener: Callable[[str], Any] | None = None,
    verify_remote: bool = True,
) -> dict:
    """下载并**原子安装**获批的 Azure archive；已存在且不同则拒绝覆盖。"""
    dest = Path(dest)
    if verify_remote and opener is None:
        length = content_length_of(AZURE_URL)
        if length != AZURE_CONTENT_LENGTH:
            raise MaterializationError(
                f"远端 Content-Length {length} != 批准的 {AZURE_CONTENT_LENGTH}"
            )
        if length > AZURE_EXCEPTION_CAP_BYTES:
            raise MaterializationError(
                f"远端大小 {length} 超过 B3 特例上限 {AZURE_EXCEPTION_CAP_BYTES}"
            )
    body = fetch_bytes(AZURE_URL, opener=opener)
    if len(body) != AZURE_CONTENT_LENGTH:
        raise MaterializationError(
            f"下载字节数 {len(body)} != 批准的 {AZURE_CONTENT_LENGTH}"
        )
    digest = hashlib.sha256(body).hexdigest()
    if dest.exists():
        if _sha256_file(dest) != digest:
            raise MaterializationError(
                f"{dest} 已存在且内容不同：拒绝覆盖（先人工确认）"
            )
        return {"path": dest, "sha256": digest, "bytes": len(body), "written": False}
    _atomic_write_bytes(dest, body)
    if _sha256_file(dest) != digest:
        dest.unlink(missing_ok=True)
        raise MaterializationError("安装后 SHA-256 不一致：已回滚")
    return {"path": dest, "sha256": digest, "bytes": len(body), "written": True}


def _assert_safe_member(name: str) -> None:
    """tar 成员路径安全检查（拒绝绝对路径、`..`、出界符号链接）。"""
    if name.startswith("/") or "\\" in name:
        raise MaterializationError(f"tar 成员路径不安全：{name!r}")
    for segment in name.split("/"):
        if segment in ("", ".", ".."):
            raise MaterializationError(f"tar 成员路径含非法片段：{name!r}")


def scan_arrival_archive(path: Path = ARRIVAL_ARCHIVE) -> dict:
    """流式扫描成员（**不解压到磁盘**）：登记使用与未使用的成员。"""
    path = Path(path)
    if not path.is_file():
        raise MaterializationError(f"缺少 Azure archive {path}")
    used: list[str] = []
    unused: list[str] = []
    with tarfile.open(path, mode="r|xz") as archive:
        for member in archive:
            _assert_safe_member(member.name)
            if member.issym() or member.islnk():
                raise MaterializationError(f"tar 含链接成员（拒绝）：{member.name!r}")
            if not member.isfile():
                continue
            if Path(member.name).name.startswith(INVOCATION_MEMBER_PREFIX):
                used.append(member.name)
            else:
                unused.append(member.name)
    if not used:
        raise MaterializationError(
            f"archive 中找不到 {INVOCATION_MEMBER_PREFIX}* 成员：schema 不符"
        )
    return {"used_members": sorted(used), "unused_members": sorted(unused),
            "used_count": len(used), "unused_count": len(unused)}


def trace_minute_totals(path: Path = ARRIVAL_ARCHIVE) -> pd.Series:
    """流式读取全部 invocation 成员 → `(date, minute)` 的调用总数。

    每个成员是**一个 24 小时周期**：每分钟一列（1..1440），逐函数一行。
    日期映射见模块 docstring 的 `TRACE_DATE_MAPPING`。
    """
    path = Path(path)
    frames: list[pd.Series] = []
    with tarfile.open(path, mode="r|xz") as archive:
        for member in archive:
            _assert_safe_member(member.name)
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.startswith(INVOCATION_MEMBER_PREFIX):
                continue
            suffix = name[len(INVOCATION_MEMBER_PREFIX):]
            day_index = int(suffix.split(".")[0].lstrip("d"))
            extracted = archive.extractfile(member)
            if extracted is None:
                raise MaterializationError(f"无法读取成员 {member.name}")
            frame = pd.read_csv(io.BytesIO(extracted.read()))
            minute_columns = [c for c in frame.columns
                              if str(c).isdigit()]
            if len(minute_columns) != 1440:
                raise MaterializationError(
                    f"{name} 的分钟列数 {len(minute_columns)} != 1440：schema 不符"
                )
            totals = frame[minute_columns].sum(axis=0)
            date = pd.Timestamp(TRACE_FIRST_DAY) + pd.Timedelta(days=day_index - 1)
            index = pd.MultiIndex.from_tuples(
                [(date.date(), int(c)) for c in minute_columns],
                names=["date", "minute"],
            )
            frames.append(pd.Series(totals.to_numpy(dtype=float), index=index))
    if not frames:
        raise MaterializationError("没有读到任何 invocation 成员")
    combined = pd.concat(frames)
    return combined.groupby(level=[0, 1]).sum().sort_index()


# --- 物化 ---------------------------------------------------------------------

def _column_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "nonzero_count": int((values != 0).sum()),
        "count": int(values.size),
    }


def build_manifest(
    *,
    inputs: FrozenInputs,
    frame: pd.DataFrame,
    output_path: Path,
    template: np.ndarray,
    archive: dict,
    members: dict,
    materializer_revision: str,
    frozen_at_utc: str,
) -> dict:
    payload = {
        "schema": OUTPUT_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "materializer_revision": materializer_revision,
        "frozen_at_utc": frozen_at_utc,
        "canonical_parquet_path": "data/processed/singapore_2024/half_hour.parquet",
        "canonical_parquet_sha256": inputs.canonical_parquet_sha256,
        "canonical_manifest_path": "data/manifest/singapore_2024_half_hour.json",
        "canonical_manifest_sha256": _sha256_file(CANONICAL_MANIFEST),
        "split_manifest_path": "data/manifest/singapore_2024_splits.json",
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "public_source_manifest_path": "data/manifest/m13f_public_sources.json",
        "public_source_manifest_sha256": _sha256_file(V1_MANIFEST),
        "materialization_sources_path": "data/manifest/m13f_materialization_sources.json",
        "materialization_sources_sha256": _sha256_file(SOURCE_MANIFEST),
        "pyproject_sha256": _sha256_file(REPO_ROOT / "pyproject.toml"),
        "uv_lock_sha256": _sha256_file(REPO_ROOT / "uv.lock"),
        "columns": {
            "local_pv_kw": {
                "classification": "modeled_scenario",
                "unit": PV_UNIT,
                "method": "pvlib_v0.15.2_chain",
                "parameters": dict(PV_PARAMS),
                "pvlib_defaults_not_human_approved":
                    dict(PVLIB_DEFAULTS_NOT_HUMAN_APPROVED),
                "note": "**modeled**，不是 IDC 本地 PV 现场实测；输入只来自同 timestamp 的 ERA5 GHI/温度/风速",
            },
            "wind_generation_kw": {
                "classification": "modeled_scenario",
                "unit": WIND_UNIT,
                "method": "shear_law_then_frozen_power_curve",
                "parameters": dict(WIND_PARAMS),
                "power_curve": {
                    "member": WIND_CURVE_FILE, "sha256": WIND_CURVE_SHA256,
                    "unit": "W→kW", "interpolation": "linear",
                    "below_cut_in": "0.0",
                    "above_last_defined_speed": "0.0（保守停机；曲线未定义切出）",
                    "cap": WIND_PARAMS["rated_capacity_kw"],
                },
                "note": "**modeled**，不是本地风机实测；**不得**把风速当成发电量",
            },
            "carbon_intensity": {
                "classification": CARBON_CLASSIFICATION,
                "resolution": CARBON_RESOLUTION,
                "unit": CARBON_UNIT,
                "method": "annual_constant",
                "value": CARBON_KG_PER_KWH,
                "source_url": CARBON_SOURCE_URL,
                "human_decision": {"decision_id": "B1", "approved_on": "2026-09-16"},
                "note": "**不是**半小时观测，**不是**实时边际排放因子；env 默认日曲线与 carbon_factor_ref 均未使用",
            },
            "arrival": {
                "classification": "modeled_scenario_calibrated_from_benchmark_trace",
                "unit": ARRIVAL_UNIT,
                "method": "poisson_forward_generation_from_frozen_template",
                "seed": ARRIVAL_SEED,
                "mean_arrival_work_units_per_half_hour":
                    ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR,
                "scale_basis": ARRIVAL_SCALE_BASIS,
                "rate_template": [[float(v) for v in row] for row in template],
                "date_mapping": TRACE_DATE_MAPPING,
                "note": "2019 trace 只校准**分布形状**，不重放、不改称 2024 真实到达；生成只依赖冻结 template、星期/时刻与固定 seed",
            },
        },
        "azure": {
            "url": AZURE_URL,
            "content_length": AZURE_CONTENT_LENGTH,
            "sha256": archive["sha256"],
            "license": AZURE_LICENSE,
            "license_url": AZURE_LICENSE_URL,
            "decision_id": AZURE_DECISION_ID,
            "exception_cap_bytes": AZURE_EXCEPTION_CAP_BYTES,
            "used_members": members["used_members"],
            "unused_members": members["unused_members"],
        },
        "output": {
            "path": "data/processed/singapore_2024/exogenous_drivers.parquet",
            "sha256": _sha256_file(output_path),
            "rows": len(frame),
            "columns": list(frame.columns),
            "timezone": "Asia/Singapore",
            "start": str(frame["timestamp"].iloc[0]),
            "end_exclusive": str(
                frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=30)
            ),
            "statistics": {
                name: _column_stats(frame[name].to_numpy())
                for name in frame.columns if name != "timestamp"
            },
        },
        "readiness": dict(READINESS),
    }
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise MaterializationError("manifest 含绝对路径")
    return payload


def build_source_manifest(
    *, archive: dict, members: dict, frozen_at_utc: str
) -> dict:
    """v2 来源 manifest：引用 v1 并登记依赖锁与 Azure archive。"""
    import pvlib

    payload = {
        "schema": SOURCE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "frozen_at_utc": frozen_at_utc,
        "public_source_manifest_path": "data/manifest/m13f_public_sources.json",
        "public_source_manifest_sha256": _sha256_file(V1_MANIFEST),
        "pyproject_path": "pyproject.toml",
        "pyproject_sha256": _sha256_file(REPO_ROOT / "pyproject.toml"),
        "uv_lock_path": "uv.lock",
        "uv_lock_sha256": _sha256_file(REPO_ROOT / "uv.lock"),
        "pvlib": {
            "version": pvlib.__version__,
            "pin": "pvlib==0.15.2",
            "license": "BSD-3-Clause",
            "url": "https://pypi.org/project/pvlib/0.15.2/",
        },
        "azure": {
            "source_id": "azure_functions_2019_trace",
            "url": AZURE_URL,
            "content_length": AZURE_CONTENT_LENGTH,
            "sha256": archive["sha256"],
            "license": AZURE_LICENSE,
            "license_url": AZURE_LICENSE_URL,
            "decision_id": AZURE_DECISION_ID,
            "exception_cap_bytes": AZURE_EXCEPTION_CAP_BYTES,
            "downloaded_bytes": archive["bytes"],
            "final_url_verified": archive.get("final_url_verified"),
            "allowed_final_hosts": archive.get("allowed_final_hosts"),
            "used_members": members["used_members"],
            "unused_members": members["unused_members"],
            "date_mapping": TRACE_DATE_MAPPING,
        },
        "wind_power_curve": {
            "source_id": "windpowerlib_power_curves",
            "member": WIND_CURVE_FILE,
            "sha256": WIND_CURVE_SHA256,
            "pinned_ref": "v0.2.2",
        },
    }
    return payload


def materialize_exogenous_drivers(
    *,
    out_parquet: Path = OUT_PARQUET,
    out_manifest: Path = OUT_MANIFEST,
    archive_path: Path = ARRIVAL_ARCHIVE,
    frozen_at_utc: str | None = None,
) -> dict:
    """计算并**原子**安装输出表与 manifest。任一步失败不留半成品。"""
    out_parquet = Path(out_parquet)
    out_manifest = Path(out_manifest)
    frozen_at_utc = frozen_at_utc or _now_utc()

    inputs = load_frozen_inputs(
        canonical_parquet_path=CANONICAL_PARQUET,
        canonical_manifest_path=CANONICAL_MANIFEST,
        split_manifest_path=SPLIT_MANIFEST,
    )
    members = scan_arrival_archive(archive_path)
    minute_totals = trace_minute_totals(archive_path)
    template = arrival_rate_template(arrival_counts_from_trace(minute_totals))
    frame = build_frame(inputs, template=template)

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    body = buffer.getvalue()

    created: list[Path] = []
    try:
        if out_parquet.exists():
            if out_parquet.read_bytes() != body:
                raise MaterializationError(
                    f"{out_parquet} 已存在且内容不同：拒绝覆盖"
                )
        else:
            _atomic_write_bytes(out_parquet, body)
            created.append(out_parquet)

        archive_info = {"sha256": _sha256_file(archive_path),
                        "bytes": archive_path.stat().st_size,
                        "final_url_verified": True,
                        "allowed_final_hosts": list(_ALLOWED_ASSET_HOSTS)}
        payload = build_manifest(
            inputs=inputs, frame=frame, output_path=out_parquet, template=template,
            archive=archive_info, members=members,
            materializer_revision=resolve_materializer_revision(),
            frozen_at_utc=frozen_at_utc,
        )
        source_payload = build_source_manifest(
            archive=archive_info, members=members, frozen_at_utc=frozen_at_utc)
        write_json_atomic(source_payload, SOURCE_MANIFEST)
        write_json_atomic(payload, out_manifest)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {"output_path": out_parquet, "manifest_path": out_manifest,
            "rows": len(frame), "template": template}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="M1.3f-c 四类外生驱动物化（默认不联网）")
    parser.add_argument("--fetch-arrival", action="store_true",
                        help="显式联网下载获批的 Azure Functions 2019 archive")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.fetch_arrival:
        result = fetch_arrival_archive()
        print(f"arrival archive: bytes={result['bytes']} sha256={result['sha256']} "
              f"written={result['written']}")
        members = scan_arrival_archive()
        print(f"members: used={members['used_count']} unused={members['unused_count']}")

    result = materialize_exogenous_drivers()
    print(f"output={result['output_path']} rows={result['rows']}")
    print(f"manifest={result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
