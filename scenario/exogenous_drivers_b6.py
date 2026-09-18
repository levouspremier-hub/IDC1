"""M1.3f-e-b1：由 **B6 policy** 物化版本化外生驱动表 **v3**。

与 v2（`scenario/exogenous_drivers.py` + `scripts/materialize_singapore_exogenous.py`）
的关系：

- **只读复用** v2 的四类驱动**实现**（PV pvlib 链、风电切变律+冻结功率曲线、
  carbon 年内常数、arrival 的 48 槽 slot 选择与 Poisson 实现）。
  **不修改**它们——否则会改变 v2 产物的 `materializer_revision` 与其证据链。
- `local_pv_kw` / `wind_generation_kw` / `carbon_intensity` **与 v2 逐行相同**。
- `arrival` **改用 B6 尺度**：`lambda_t = rate_template[slot] × 31.994 work/half-hour`，
  **不继承** B5 的 `1000` scale。

## arrival 的 shape 与 scale 分离

- **shape**：复用 v2 manifest 里已冻结的 **48-slot day-of-benchmark-period**
  rate template（均值精确为 1）。读取前**先按冻结 SHA-256 校验 v2 manifest**；
  Azure invocation count **只**贡献形状，**不得**被当作 work-unit。
- **scale**：来自**已验证的 canonical B6 policy** 的
  `main_expected_amount_work_per_half_hour = 31.994`（`lambda_t` 的尺度）。

## realization 与诊断

`realization` 用固定 seed（取自 policy 的 `realization_seed = 20240916`）做 Poisson
前向生成。`realized_annual_mean` / `rho_realized` **只是诊断登记**：
`forbid_realization_feedback = true`，**禁止**用它们反向修改 `31.994`、`79.985`
或 `rho_target = 0.80`。

本模块**不**切换 formal loader、**不**生成 refs 新版本、**不**建 `formal_splits_v5`、
**不**接 mapper、**不**训练。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scenario.arrival_intensity_policy import (
    load_verified_b6_policy,
)
from scenario.exogenous_drivers import (
    ARRIVAL_TEMPLATE_SHAPE,
    CANONICAL_MANIFEST,
    CANONICAL_PARQUET,
    SPLIT_MANIFEST,
    FrozenInputs,
    arrival_template_slot,
    carbon_intensity,
    generate_arrival,
    local_pv_kw,
    wind_generation_kw,
)
from scenario.splits import (
    SplitError,
    _require_canonical_utc,
    _require_dict,
    _require_exact_keys,
    _require_git_sha40,
    _require_hex64,
    logical_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- 冻结常量 -----------------------------------------------------------------

B6_OUTPUT_SCHEMA = "m1.3feb-b6-exogenous-v1"
B6_SOURCE_SCHEMA = "m1.3feb-materialization-sources-v1"

V3_MANIFEST_NAME = "singapore_2024_exogenous_v3.json"
V3_SOURCE_NAME = "m13f_materialization_sources_v4.json"

V3_PARQUET_LOGICAL = "data/processed/singapore_2024/exogenous_drivers_v3.parquet"
V3_MANIFEST_LOGICAL = f"data/manifest/{V3_MANIFEST_NAME}"
V3_SOURCE_LOGICAL = f"data/manifest/{V3_SOURCE_NAME}"

V2_EXOGENOUS_MANIFEST_LOGICAL = "data/manifest/singapore_2024_exogenous_v2.json"
# v2 输出 manifest 的冻结 SHA-256（**本卡唯一**允许的 arrival shape 来源）
V2_EXOGENOUS_MANIFEST_SHA256 = (
    "640f26cda94b3479049fdbee56f05e1546a24fc3ecdf6286674c4fb415b484b9"
)
V2_EXOGENOUS_PARQUET_LOGICAL = (
    "data/processed/singapore_2024/exogenous_drivers_v2.parquet"
)
V2_EXOGENOUS_PARQUET_SHA256 = (
    "11d322b2919e2180b596e6b02614acafdb3ee8d63682ae74e5a3ee1dbc8b92cf"
)
V2_SOURCE_MANIFEST_LOGICAL = "data/manifest/m13f_materialization_sources_v3.json"
V2_SOURCE_MANIFEST_SHA256 = (
    "4203b4f399ee6433bfcdf63fa94ddd03a56a7bd1e6add45f697c3bec804da1b6"
)

B6_POLICY_LOGICAL = "data/manifest/m13f_arrival_intensity_policy_v1.json"

# v2 的 superseded 前代（v1）：登记为历史，**不改写**
V1_SUPERSEDED = {
    "status": "superseded_pre_approval_and_loss_fix",
    "parquet_sha256": "0c5e65d8fdc25ed8ced228e8087f13eb0605d0146d7e258caf54a552a246287c",
    "manifest_sha256": "46d88c38247bf1eb1568e84abc48647f1f4aeb58a5c0b525a30b8bfb0b2ac224",
}

# B6 尺度**不继承** B5 的 1000（本卡自己的冻结声明，不修改 v2 模块）
B5_ARRIVAL_SCALE_INHERITED = False

# 本卡实现文件的**唯一** revision 源
B6_EXOGENOUS_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/exogenous_drivers_b6.py",
    "scripts/materialize_singapore_exogenous_b6.py",
)

_COLUMNS: tuple[str, ...] = (
    "timestamp", "local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival",
)

OUTPUT_MANIFEST_KEYS: tuple[str, ...] = (
    "schema",
    "materializer_revision",
    "frozen_at_utc",
    "canonical_parquet_path",
    "canonical_parquet_sha256",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "b6_policy_path",
    "b6_policy_sha256",
    "arrival_shape_source_manifest_path",
    "arrival_shape_source_manifest_sha256",
    "materialization_sources_path",
    "materialization_sources_sha256",
    "columns",
    "output",
    "predecessor",
    "readiness",
)

SOURCE_MANIFEST_KEYS: tuple[str, ...] = (
    "schema",
    "frozen_at_utc",
    "b6_policy_path",
    "b6_policy_sha256",
    "v2_output_manifest_path",
    "v2_output_manifest_sha256",
    "v2_output_parquet_path",
    "v2_output_parquet_sha256",
    "v2_source_manifest_path",
    "v2_source_manifest_sha256",
    "canonical_parquet_path",
    "canonical_parquet_sha256",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "pyproject_sha256",
    "uv_lock_sha256",
    "arrival_shape_rule",
    "arrival_scale_rule",
    "shared_columns_rule",
)

# 与 v2 **相同**的 readiness（本卡**不修改** readiness）
READINESS: dict[str, bool] = {
    "local_pv_kw_ready": True,
    "wind_generation_kw_ready": True,
    "carbon_intensity_ready": True,
    "arrival_ready": True,
    "exogenous_drivers_ready": True,
    "formal_scenario_bundle_ready": False,
    "formal_training_ready": False,
}

ARRIVAL_SHAPE_RULE = (
    "复用 v2 manifest 冻结的 48-slot day-of-benchmark-period rate template"
    "（均值精确为 1）；Azure invocation count **只**贡献形状，不是 work-unit"
)
ARRIVAL_SCALE_RULE = (
    "lambda_t = rate_template[slot] × policy.main_expected_amount_work_per_half_hour"
    "（31.994 work/half-hour）；**不继承** B5 的 1000 scale"
)
SHARED_COLUMNS_RULE = (
    "local_pv_kw / wind_generation_kw / carbon_intensity 复用 v2 的同一实现与同一输入，"
    "因此与 v2 **逐行相同**"
)


class B6ExogenousError(ValueError):
    """B6 exogenous v3 的**明确失败**（不 fallback、不静默）。"""


def _sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- 代码 provenance ----------------------------------------------------------

def b6_exogenous_revision() -> str:
    """本物化实现的 revision（由 Git 解析 B6_SOURCE_PATHS，不用漂移的 HEAD）。"""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *B6_EXOGENOUS_SOURCE_PATHS],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as error:
        raise B6ExogenousError(
            f"materializer_revision 无法由 Git 解析：{error}"
        ) from error
    sha = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        _require_git_sha40(sha, field="materializer_revision")
    except SplitError as error:
        raise B6ExogenousError(
            f"materializer_revision 无法由 Git 解析：{error}"
            "（本卡实现文件必须先提交）"
        ) from error
    return sha


# --- B6 policy（canonical-only） ----------------------------------------------

def load_b6_policy() -> dict:
    """通过 **canonical loader** 读取并验证 B6 policy（无 fallback）。"""
    return load_verified_b6_policy()


# --- arrival shape（v2 manifest 逐字节绑定） -----------------------------------

def load_v2_arrival_template() -> np.ndarray:
    """读取 v2 manifest 的冻结 rate template；**先**校验其 SHA-256。

    Azure trace **只**贡献形状：返回的 template 均值必须精确为 1。
    """
    path = REPO_ROOT / V2_EXOGENOUS_MANIFEST_LOGICAL
    if not path.is_file():
        raise B6ExogenousError(f"缺少 v2 exogenous manifest {path}")
    actual = _sha256_file(path)
    if actual != V2_EXOGENOUS_MANIFEST_SHA256:
        raise B6ExogenousError(
            f"v2 exogenous manifest SHA-256 不符：期望 "
            f"{V2_EXOGENOUS_MANIFEST_SHA256} 实际 {actual}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    template = np.asarray(
        payload["columns"]["arrival"]["rate_template"], dtype=float
    )
    if template.shape != ARRIVAL_TEMPLATE_SHAPE:
        raise B6ExogenousError(
            f"冻结 template 形状必须是 {ARRIVAL_TEMPLATE_SHAPE}，实际 {template.shape}"
        )
    if not np.isclose(template.mean(), 1.0, rtol=1e-12):
        raise B6ExogenousError("冻结 template 均值必须精确为 1（只贡献 shape）")
    return template


# --- arrival expected / realized ----------------------------------------------

def _b6_amount(policy: dict) -> float:
    amount = policy["main_expected_amount_work_per_half_hour"]
    if float(amount) != 31.994:
        raise B6ExogenousError(
            f"policy 的 main_expected_amount_work_per_half_hour 必须是 31.994，"
            f"实际 {amount!r}"
        )
    return float(amount)


def expected_arrival(
    timestamps: pd.DatetimeIndex,
    template: np.ndarray,
    policy: dict,
) -> np.ndarray:
    """**期望** arrival（`lambda_t`，不是抽样值）：`template[slot] × 31.994`。"""
    template = np.asarray(template, dtype=float)
    if template.shape != ARRIVAL_TEMPLATE_SHAPE:
        raise B6ExogenousError(f"template 形状必须是 {ARRIVAL_TEMPLATE_SHAPE}")
    stamps = pd.DatetimeIndex(timestamps)
    slots = arrival_template_slot(stamps)
    return template[slots] * _b6_amount(policy)


def realize_arrival(
    timestamps: pd.DatetimeIndex,
    template: np.ndarray,
    policy: dict,
) -> np.ndarray:
    """**realization**：`Poisson(lambda_t)`，seed 取自 policy（固定）。

    复用 v2 的冻结 Poisson 实现，但 **显式传入 B6 尺度**——
    **不继承** B5 的 `1000`，也不使用 `ARRIVAL_MEAN_WORK_UNITS_PER_HALF_HOUR`。
    """
    seed = policy["realization_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise B6ExogenousError(f"realization_seed 必须是整数，实际 {seed!r}")
    arrival = generate_arrival(
        pd.DatetimeIndex(timestamps),
        np.asarray(template, dtype=float),
        seed=seed,
        mean_per_half_hour=_b6_amount(policy),
    )
    return np.asarray(arrival, dtype=np.int64)


# --- 组装 ---------------------------------------------------------------------

def build_v3_frame(
    inputs: FrozenInputs,
    *,
    template: np.ndarray,
    policy: dict,
) -> pd.DataFrame:
    """组装 v3 输出表：三列与 v2 逐行相同，`arrival` 用 B6 尺度。"""
    frame = inputs.frame
    stamps = pd.DatetimeIndex(frame["timestamp"])
    shared = {
        "local_pv_kw": local_pv_kw(
            stamps,
            frame["ghi_w_per_m2"].to_numpy(dtype=float),
            frame["temperature_deg_c"].to_numpy(dtype=float),
            frame["wind_speed_10m_mps"].to_numpy(dtype=float),
        ),
        "wind_generation_kw": wind_generation_kw(
            frame["wind_speed_10m_mps"].to_numpy(dtype=float)
        ),
        "carbon_intensity": carbon_intensity(len(frame)),
        "arrival": realize_arrival(stamps, template, policy),
    }
    out = pd.DataFrame({"timestamp": stamps, **shared})
    if tuple(out.columns) != _COLUMNS:
        raise B6ExogenousError(f"v3 列不符：{tuple(out.columns)}")
    if len(out) != len(frame):
        raise B6ExogenousError("v3 行数与 canonical 不一致")
    return out


def rho_realized(realized: np.ndarray, policy: dict) -> float:
    """**诊断**量：realized 均值 / 声明持续容量。**不得**用于生成或反调。"""
    declared = float(policy["declared_capacity_work_per_hour"])
    if declared <= 0:
        raise B6ExogenousError("declared capacity 必须为正")
    # realized 是 work/half-hour；换算到 per-hour 后除以 capacity
    per_hour = float(np.asarray(realized, dtype=float).mean()) / 0.5
    return per_hour / declared


# --- canonical 路径门禁 -------------------------------------------------------

def _canonical_v3_dir() -> Path:
    """canonical v3 目录的**私有**解析器（测试只能 monkeypatch 它）。"""
    return REPO_ROOT / "data" / "manifest"


def canonical_v3_manifest_path() -> Path:
    return _canonical_v3_dir() / V3_MANIFEST_NAME


def canonical_v3_source_path() -> Path:
    return _canonical_v3_dir() / V3_SOURCE_NAME


def canonical_v3_parquet_path() -> Path:
    return REPO_ROOT / V3_PARQUET_LOGICAL


def _require_canonical_location(path: Path | str | None) -> Path:
    """路径必须**精确等于** canonical v3 manifest（在读取 JSON 之前）。

    比较**词法绝对路径**（`absolute()` 不做符号链接解析），因此副本、别名、
    symlink 一律拒绝；**不得** fallback 到 v2。
    """
    given = Path(path).absolute() if path is not None else \
        canonical_v3_manifest_path().absolute()
    expected = canonical_v3_manifest_path().absolute()
    if given == expected:
        return expected
    raise B6ExogenousError(
        f"v3 manifest 路径必须是**唯一 canonical** 文件 "
        f"{logical_repo_path(expected)}；实际 {logical_repo_path(given)}"
        "（不接受副本、别名、symlink 或其它生成版本；无 fallback）"
    )


# --- 严格加载 + 校验 ----------------------------------------------------------

def _binding_pairs(payload: dict) -> tuple[tuple[str, str, str], ...]:
    return (
        ("canonical_parquet_path", "canonical_parquet_sha256",
         "data/processed/singapore_2024/half_hour.parquet"),
        ("canonical_manifest_path", "canonical_manifest_sha256",
         "data/manifest/singapore_2024_half_hour.json"),
        ("split_manifest_path", "split_manifest_sha256",
         "data/manifest/singapore_2024_splits.json"),
        ("b6_policy_path", "b6_policy_sha256", B6_POLICY_LOGICAL),
        ("arrival_shape_source_manifest_path", "arrival_shape_source_manifest_sha256",
         V2_EXOGENOUS_MANIFEST_LOGICAL),
        ("materialization_sources_path", "materialization_sources_sha256",
         V3_SOURCE_LOGICAL),
    )


def _verify_live_bindings(payload: dict) -> None:
    for path_field, sha_field, logical in _binding_pairs(payload):
        if payload[path_field] != logical:
            raise B6ExogenousError(
                f"{path_field} 必须精确等于 {logical!r}，实际 {payload[path_field]!r}"
            )
        expected = _require_hex64(payload[sha_field], field=sha_field)
        actual = _sha256_file(REPO_ROOT / logical)
        if actual != expected:
            raise B6ExogenousError(
                f"{sha_field} 与实际文件不符：声明 {expected} 实际 {actual}"
            )


def load_verified_v3_manifest(path: Path | str | None = None) -> dict:
    """加载并**严格校验** canonical v3 输出 manifest（路径先于 JSON 读取）。

    **无 fallback**：不接受副本 / symlink / 旧 v2。
    """
    path = _require_canonical_location(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B6ExogenousError(f"v3 manifest 不可读：{error}") from error
    payload = _require_dict(payload, field="v3 exogenous manifest")
    _require_exact_keys(payload, field="v3 exogenous manifest",
                        expected=OUTPUT_MANIFEST_KEYS)
    if payload["schema"] != B6_OUTPUT_SCHEMA:
        raise B6ExogenousError(
            f"schema 必须是 {B6_OUTPUT_SCHEMA!r}，实际 {payload['schema']!r}"
        )
    _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")

    revision = _require_git_sha40(payload["materializer_revision"],
                                  field="materializer_revision")
    if revision != b6_exogenous_revision():
        raise B6ExogenousError(
            f"materializer_revision 与 live 解析不符：声明 {revision} "
            f"live {b6_exogenous_revision()}"
        )

    _verify_live_bindings(payload)

    output = _require_dict(payload["output"], field="output")
    out_sha = _require_hex64(output["sha256"], field="output.sha256")
    actual_sha = _sha256_file(canonical_v3_parquet_path())
    if actual_sha != out_sha:
        raise B6ExogenousError(
            f"output.sha256 与实际 v3 parquet 不符：声明 {out_sha} 实际 {actual_sha}"
        )
    if output["path"] != V3_PARQUET_LOGICAL:
        raise B6ExogenousError(
            f"output.path 必须精确等于 {V3_PARQUET_LOGICAL!r}"
        )
    return payload


def load_verified_v3_source() -> dict:
    """加载并校验 canonical source-v4 manifest。"""
    path = canonical_v3_source_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B6ExogenousError(f"source-v4 manifest 不可读：{error}") from error
    payload = _require_dict(payload, field="source manifest")
    _require_exact_keys(payload, field="source manifest",
                        expected=SOURCE_MANIFEST_KEYS)
    if payload["schema"] != B6_SOURCE_SCHEMA:
        raise B6ExogenousError(f"source schema 必须是 {B6_SOURCE_SCHEMA!r}")
    _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")
    for path_field, sha_field, logical in (
        ("v2_output_manifest_path", "v2_output_manifest_sha256",
         V2_EXOGENOUS_MANIFEST_LOGICAL),
        ("v2_output_parquet_path", "v2_output_parquet_sha256",
         V2_EXOGENOUS_PARQUET_LOGICAL),
        ("v2_source_manifest_path", "v2_source_manifest_sha256",
         V2_SOURCE_MANIFEST_LOGICAL),
        ("b6_policy_path", "b6_policy_sha256", B6_POLICY_LOGICAL),
    ):
        if payload[path_field] != logical:
            raise B6ExogenousError(
                f"{path_field} 必须精确等于 {logical!r}"
            )
        expected = _require_hex64(payload[sha_field], field=sha_field)
        if _sha256_file(REPO_ROOT / logical) != expected:
            raise B6ExogenousError(f"{sha_field} 与实际文件不符")
    return payload


def build_v3_output_manifest(
    *,
    inputs: FrozenInputs,
    frame: pd.DataFrame,
    output_path: Path,
    source_manifest_sha256: str,
    template: np.ndarray,
    policy: dict,
    materializer_revision: str,
    frozen_at_utc: str,
) -> dict:
    """构造 v3 输出 manifest（含 diagnostic 的 realized mean 与 `rho_realized`）。"""
    arrival = np.asarray(frame["arrival"], dtype=np.int64)
    realized_mean = float(arrival.mean())
    payload: dict[str, Any] = {
        "schema": B6_OUTPUT_SCHEMA,
        "materializer_revision": materializer_revision,
        "frozen_at_utc": frozen_at_utc,
        "canonical_parquet_path": "data/processed/singapore_2024/half_hour.parquet",
        "canonical_parquet_sha256": inputs.canonical_parquet_sha256,
        "canonical_manifest_path": "data/manifest/singapore_2024_half_hour.json",
        "canonical_manifest_sha256": _sha256_file(CANONICAL_MANIFEST),
        "split_manifest_path": "data/manifest/singapore_2024_splits.json",
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "b6_policy_path": B6_POLICY_LOGICAL,
        "b6_policy_sha256": _sha256_file(REPO_ROOT / B6_POLICY_LOGICAL),
        "arrival_shape_source_manifest_path": V2_EXOGENOUS_MANIFEST_LOGICAL,
        "arrival_shape_source_manifest_sha256": V2_EXOGENOUS_MANIFEST_SHA256,
        "materialization_sources_path": V3_SOURCE_LOGICAL,
        "materialization_sources_sha256": source_manifest_sha256,
        "columns": {
            "local_pv_kw": {
                "classification": "modeled_scenario",
                "unit": "kW",
                "rule": SHARED_COLUMNS_RULE,
            },
            "wind_generation_kw": {
                "classification": "modeled_scenario",
                "unit": "kW",
                "rule": SHARED_COLUMNS_RULE,
            },
            "carbon_intensity": {
                "classification": "human_approved_external_low_resolution",
                "unit": "kgCO2/kWh",
                "rule": SHARED_COLUMNS_RULE,
            },
            "arrival": {
                "classification": "modeled_scenario",
                "unit": "work-units/step",
                "shape_rule": ARRIVAL_SHAPE_RULE,
                "scale_rule": ARRIVAL_SCALE_RULE,
                "shape_source_manifest_path": V2_EXOGENOUS_MANIFEST_LOGICAL,
                "shape_source_manifest_sha256": V2_EXOGENOUS_MANIFEST_SHA256,
                "template_slots": int(np.asarray(template).size),
                "template_mean": float(np.asarray(template, dtype=float).mean()),
                "b5_scale_inherited": B5_ARRIVAL_SCALE_INHERITED,
                "mean_arrival_work_units_per_half_hour_scale": _b6_amount(policy),
                "expected_annual_mean": float(
                    expected_arrival(
                        pd.DatetimeIndex(frame["timestamp"]), template, policy
                    ).mean()
                ),
                "realized_annual_mean": realized_mean,
                "rho_realized": rho_realized(arrival, policy),
                "rho_realized_purpose": policy["rho_realized_purpose"],
                "forbid_realization_feedback": policy["forbid_realization_feedback"],
                "realization_seed": policy["realization_seed"],
                "diagnostic_note": (
                    "realized mean 与 rho_realized **只是诊断登记**；"
                    "禁止反向修改 31.994 / 79.985 / rho_target=0.80"
                ),
            },
        },
        "output": {
            "path": V3_PARQUET_LOGICAL,
            "sha256": _sha256_file(output_path),
            "rows": len(frame),
            "columns": list(frame.columns),
            "timezone": "Asia/Singapore",
            "start": str(frame["timestamp"].iloc[0]),
            "end_exclusive": str(
                frame["timestamp"].iloc[-1] + pd.Timedelta(minutes=30)
            ),
        },
        "predecessor": {
            "status": "predecessor_formal_chain_still_bound_to_v2",
            "output_manifest_path": V2_EXOGENOUS_MANIFEST_LOGICAL,
            "output_manifest_sha256": V2_EXOGENOUS_MANIFEST_SHA256,
            "output_parquet_path": V2_EXOGENOUS_PARQUET_LOGICAL,
            "output_parquet_sha256": V2_EXOGENOUS_PARQUET_SHA256,
            "source_manifest_path": V2_SOURCE_MANIFEST_LOGICAL,
            "source_manifest_sha256": V2_SOURCE_MANIFEST_SHA256,
            "note": (
                "v2 **逐字节保留**；formal 链在 M1.3f-e-b2 之前**仍**绑定 v2；"
                "本卡不做 loader 切换、不迁移、无 fallback"
            ),
            "v1": dict(V1_SUPERSEDED),
        },
        "readiness": dict(READINESS),
    }
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise B6ExogenousError("manifest 含绝对路径")
    _require_exact_keys(payload, field="v3 exogenous manifest",
                        expected=OUTPUT_MANIFEST_KEYS)
    return payload


def build_v3_source_manifest(*, frozen_at_utc: str) -> dict:
    """构造 source-v4 来源 manifest（引用 v2 三项资产与 B6 policy）。"""
    payload: dict[str, Any] = {
        "schema": B6_SOURCE_SCHEMA,
        "frozen_at_utc": frozen_at_utc,
        "b6_policy_path": B6_POLICY_LOGICAL,
        "b6_policy_sha256": _sha256_file(REPO_ROOT / B6_POLICY_LOGICAL),
        "v2_output_manifest_path": V2_EXOGENOUS_MANIFEST_LOGICAL,
        "v2_output_manifest_sha256": V2_EXOGENOUS_MANIFEST_SHA256,
        "v2_output_parquet_path": V2_EXOGENOUS_PARQUET_LOGICAL,
        "v2_output_parquet_sha256": V2_EXOGENOUS_PARQUET_SHA256,
        "v2_source_manifest_path": V2_SOURCE_MANIFEST_LOGICAL,
        "v2_source_manifest_sha256": V2_SOURCE_MANIFEST_SHA256,
        "canonical_parquet_path": "data/processed/singapore_2024/half_hour.parquet",
        "canonical_parquet_sha256": _sha256_file(CANONICAL_PARQUET),
        "canonical_manifest_path": "data/manifest/singapore_2024_half_hour.json",
        "canonical_manifest_sha256": _sha256_file(CANONICAL_MANIFEST),
        "split_manifest_path": "data/manifest/singapore_2024_splits.json",
        "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
        "pyproject_sha256": _sha256_file(REPO_ROOT / "pyproject.toml"),
        "uv_lock_sha256": _sha256_file(REPO_ROOT / "uv.lock"),
        "arrival_shape_rule": ARRIVAL_SHAPE_RULE,
        "arrival_scale_rule": ARRIVAL_SCALE_RULE,
        "shared_columns_rule": SHARED_COLUMNS_RULE,
    }
    text = json.dumps(payload, ensure_ascii=False)
    if "/Users/" in text or str(REPO_ROOT) in text:
        raise B6ExogenousError("source manifest 含绝对路径")
    _require_exact_keys(payload, field="source manifest",
                        expected=SOURCE_MANIFEST_KEYS)
    return payload


__all__ = [
    "B6ExogenousError",
    "B6_OUTPUT_SCHEMA",
    "B6_SOURCE_SCHEMA",
    "V2_EXOGENOUS_MANIFEST_SHA256",
    "b6_exogenous_revision",
    "load_b6_policy",
    "load_v2_arrival_template",
    "expected_arrival",
    "realize_arrival",
    "rho_realized",
    "build_v3_frame",
    "canonical_v3_manifest_path",
    "canonical_v3_source_path",
    "canonical_v3_parquet_path",
    "load_verified_v3_manifest",
    "load_verified_v3_source",
    "build_v3_output_manifest",
    "build_v3_source_manifest",
]
