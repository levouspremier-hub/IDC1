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
    load_frozen_inputs,
    local_pv_kw,
    wind_generation_kw,
)
from scenario.splits import (
    SplitError,
    _require_canonical_utc,
    _require_dict,
    _require_exact_keys,
    _require_git_sha40,
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

# --- 嵌套对象的精确键集合（R1：语义信任边界） ---------------------------------

SHARED_COLUMN_NAMES: tuple[str, ...] = (
    "local_pv_kw", "wind_generation_kw", "carbon_intensity",
)
COLUMN_OBJECT_KEYS: tuple[str, ...] = (
    "local_pv_kw", "wind_generation_kw", "carbon_intensity", "arrival",
)
SHARED_COLUMN_KEYS: tuple[str, ...] = ("classification", "unit", "rule")
ARRIVAL_OBJECT_KEYS: tuple[str, ...] = (
    "classification",
    "unit",
    "shape_rule",
    "scale_rule",
    "shape_source_manifest_path",
    "shape_source_manifest_sha256",
    "template_slots",
    "template_mean",
    "b5_scale_inherited",
    "mean_arrival_work_units_per_half_hour_scale",
    "expected_annual_mean",
    "realized_annual_mean",
    "rho_realized",
    "rho_realized_purpose",
    "forbid_realization_feedback",
    "realization_seed",
    "diagnostic_note",
)
OUTPUT_OBJECT_KEYS: tuple[str, ...] = (
    "path", "sha256", "rows", "columns", "timezone", "start", "end_exclusive",
)
V1_KEYS: tuple[str, ...] = ("status", "parquet_sha256", "manifest_sha256")
PREDECESSOR_KEYS: tuple[str, ...] = (
    "status",
    "output_manifest_path",
    "output_manifest_sha256",
    "output_parquet_path",
    "output_parquet_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "note",
    "v1",
)
READINESS_KEYS: tuple[str, ...] = (
    "local_pv_kw_ready",
    "wind_generation_kw_ready",
    "carbon_intensity_ready",
    "arrival_ready",
    "exogenous_drivers_ready",
    "formal_scenario_bundle_ready",
    "formal_training_ready",
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


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def _generator_is_dirty() -> bool:
    """本卡实现文件是否有未提交修改（含未跟踪的新文件）。

    dirty 时**不得**生产 verified 结果：revision 无法为未提交代码背书。
    """
    status = _git("status", "--porcelain", "--", *B6_EXOGENOUS_SOURCE_PATHS)
    return bool(status.strip())


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


def _canonical_v3_parquet_dir() -> Path:
    """canonical v3 parquet 目录的**私有**解析器（测试只能 monkeypatch 它）。"""
    return REPO_ROOT / "data" / "processed" / "singapore_2024"


def canonical_v3_parquet_path() -> Path:
    return _canonical_v3_parquet_dir() / Path(V3_PARQUET_LOGICAL).name


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

def _require_plain_canonical_file(path: Path | str, *, field: str) -> Path:
    """必须是**普通文件**且**不是 symlink**（在读取之前）。"""
    p = Path(path)
    if p.is_symlink():
        raise B6ExogenousError(f"{field} 不得是 symlink：{p}")
    if not p.is_file():
        raise B6ExogenousError(f"{field} 必须是普通文件（且存在）：{p}")
    return p


def _read_json(path: Path, *, field: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B6ExogenousError(f"{field} 不可读：{error}") from error
    return _require_dict(payload, field=field)


def _require_nested_keys(payload: dict) -> None:
    """output manifest 的**所有嵌套对象**必须使用精确键集合。"""
    columns = _require_dict(payload["columns"], field="columns")
    _require_exact_keys(columns, field="columns", expected=COLUMN_OBJECT_KEYS)
    for name in SHARED_COLUMN_NAMES:
        _require_exact_keys(
            _require_dict(columns[name], field=f"columns.{name}"),
            field=f"columns.{name}", expected=SHARED_COLUMN_KEYS)
    _require_exact_keys(
        _require_dict(columns["arrival"], field="columns.arrival"),
        field="columns.arrival", expected=ARRIVAL_OBJECT_KEYS)
    _require_exact_keys(
        _require_dict(payload["output"], field="output"),
        field="output", expected=OUTPUT_OBJECT_KEYS)
    predecessor = _require_dict(payload["predecessor"], field="predecessor")
    _require_exact_keys(predecessor, field="predecessor",
                        expected=PREDECESSOR_KEYS)
    _require_exact_keys(
        _require_dict(predecessor["v1"], field="predecessor.v1"),
        field="predecessor.v1", expected=V1_KEYS)
    _require_exact_keys(
        _require_dict(payload["readiness"], field="readiness"),
        field="readiness", expected=READINESS_KEYS)


def _assert_frames_identical(recomputed: pd.DataFrame, on_disk: pd.DataFrame) -> None:
    """结构和离散值精确一致；功率仅允许预定物理尺度的 float64 舍入误差。"""
    from scenario.portable_numeric import assert_power_roundoff
    if list(recomputed.columns) != list(on_disk.columns):
        raise B6ExogenousError(
            f"v3 列名不符：重算 {list(recomputed.columns)} "
            f"磁盘 {list(on_disk.columns)}"
        )
    for column in recomputed.columns:
        if recomputed[column].dtype != on_disk[column].dtype:
            raise B6ExogenousError(
                f"{column} 的 dtype 不符：重算 {recomputed[column].dtype} "
                f"磁盘 {on_disk[column].dtype}"
            )
    if len(recomputed) != len(on_disk):
        raise B6ExogenousError(
            f"v3 行数不符：重算 {len(recomputed)} 磁盘 {len(on_disk)}"
        )
    if not (recomputed["timestamp"].to_numpy() == on_disk["timestamp"].to_numpy()).all():
        raise B6ExogenousError("v3 timestamp 与重算结果不符")
    for column in recomputed.columns:
        if column == "timestamp":
            continue
        left = recomputed[column].to_numpy()
        right = on_disk[column].to_numpy()
        if column in ("local_pv_kw", "wind_generation_kw"):
            try:
                assert_power_roundoff(left, right, 500. if column == "local_pv_kw" else 800.)
            except ValueError as error:
                raise B6ExogenousError(f"{column} 与重算结果不符：{error}") from error
            continue
        if not (left == right).all():
            mismatch = int((left != right).sum())
            raise B6ExogenousError(
                f"{column} 与重算结果不符（{mismatch} 行不同）："
                "parquet 不是由可信输入确定性重算得到的同一结果"
            )


def load_verified_v3_bundle() -> dict:
    """**统一生产入口**：一次性完成 v3 bundle 的**完整语义**验证。

    依次验证：三份产物均为预期路径下的普通文件（非 symlink）；manifest 与 source
    的**顶层及所有嵌套对象**精确键集合；B6 policy 走其正式 loader；v2 shape
    manifest 按冻结 hash；source-v4 由 trusted constants + live hashes
    **重新构造**并逐字段相等；两份 manifest 共享 `frozen_at_utc`；由 canonical 输入
    + policy + 冻结 template **重算**完整 DataFrame 并与磁盘 parquet 逐列逐值一致；
    再用重算结果**重新构造** output manifest 并**逐字段等于**文件。

    因此 arrival 的嵌套语义、`rho_realized`、predecessor、readiness 等
    **全部由 policy 与重算结果导出**，**不信任 JSON 自报**。任一不符 fail closed。
    """
    if _generator_is_dirty():
        raise B6ExogenousError(
            "B6 实现文件有未提交修改：verified 入口拒绝用旧 revision 为未提交代码背书"
        )

    manifest_path = canonical_v3_manifest_path()
    source_path = canonical_v3_source_path()
    parquet_path = canonical_v3_parquet_path()
    for field, path in (("v3 manifest", manifest_path),
                        ("source-v4 manifest", source_path),
                        ("v3 parquet", parquet_path),
                        ("B6 policy", REPO_ROOT / B6_POLICY_LOGICAL),
                        ("v2 shape manifest", REPO_ROOT / V2_EXOGENOUS_MANIFEST_LOGICAL)):
        _require_plain_canonical_file(path, field=field)

    payload = _read_json(manifest_path, field="v3 exogenous manifest")
    _require_exact_keys(payload, field="v3 exogenous manifest",
                        expected=OUTPUT_MANIFEST_KEYS)
    source_payload = _read_json(source_path, field="source-v4 manifest")
    _require_exact_keys(source_payload, field="source-v4 manifest",
                        expected=SOURCE_MANIFEST_KEYS)
    _require_nested_keys(payload)

    if payload["schema"] != B6_OUTPUT_SCHEMA:
        raise B6ExogenousError(
            f"schema 必须是 {B6_OUTPUT_SCHEMA!r}，实际 {payload['schema']!r}"
        )
    if source_payload["schema"] != B6_SOURCE_SCHEMA:
        raise B6ExogenousError(
            f"source schema 必须是 {B6_SOURCE_SCHEMA!r}"
        )
    _require_canonical_utc(payload["frozen_at_utc"], field="frozen_at_utc")
    _require_canonical_utc(source_payload["frozen_at_utc"],
                           field="source.frozen_at_utc")

    # 信任根：canonical policy（正式 loader）与 v2 冻结 shape
    policy = load_b6_policy()
    template = load_v2_arrival_template()

    # source-v4 必须能由 trusted constants + live hashes 重新构造
    rebuilt_source = build_v3_source_manifest(
        frozen_at_utc=source_payload["frozen_at_utc"])
    if rebuilt_source != source_payload:
        raise B6ExogenousError(
            "source-v4 manifest 与由 trusted constants + live hashes 重建的结果不符"
        )

    # 两份 manifest 必须共享同一冻结时刻
    if payload["frozen_at_utc"] != source_payload["frozen_at_utc"]:
        raise B6ExogenousError(
            f"output 与 source 的 frozen_at_utc 不一致："
            f"{payload['frozen_at_utc']!r} vs {source_payload['frozen_at_utc']!r}"
        )

    # 重算完整 DataFrame 并与磁盘 parquet 逐列逐值比对
    inputs = load_frozen_inputs(
        canonical_parquet_path=REPO_ROOT / "data/processed/singapore_2024/half_hour.parquet",
        canonical_manifest_path=REPO_ROOT / "data/manifest/singapore_2024_half_hour.json",
        split_manifest_path=REPO_ROOT / "data/manifest/singapore_2024_splits.json",
    )
    recomputed = build_v3_frame(inputs, template=template, policy=policy)
    on_disk = pd.read_parquet(parquet_path)
    _assert_frames_identical(recomputed, on_disk)

    # 用重算结果 + live source hash + live revision 重建 output manifest
    live_revision = b6_exogenous_revision()
    declared_revision = _require_git_sha40(payload["materializer_revision"],
                                           field="materializer_revision")
    from scenario.portable_numeric import assert_frozen_parquet, verify_frozen_recipe

    try:
        if not recomputed.equals(on_disk):
            assert_frozen_parquet(parquet_path)
        if declared_revision != live_revision:
            verify_frozen_recipe(REPO_ROOT, declared_revision, parquet_path)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise B6ExogenousError(f"materializer_revision/冻结资产验证失败：{error}") from error
    if declared_revision != live_revision:
        live_revision = declared_revision  # Registered, byte-anchored original recipe above.
    rebuilt = build_v3_output_manifest(
        inputs=inputs,
        frame=on_disk,
        output_path=parquet_path,
        source_manifest_sha256=_sha256_file(source_path),
        template=template,
        policy=policy,
        materializer_revision=live_revision,
        frozen_at_utc=payload["frozen_at_utc"],
    )
    if rebuilt != payload:
        differing = sorted(
            key for key in set(rebuilt) | set(payload)
            if rebuilt.get(key) != payload.get(key)
        )
        raise B6ExogenousError(
            f"v3 manifest 与由 policy + 重算结果重建的语义不符；差异字段={differing}"
        )
    return {"manifest": payload, "source": source_payload,
            "frame": on_disk, "policy": policy,
            "verification": {"schema": "portable-frozen-verifier-v1",
                             "revision": b6_exogenous_revision(),
                             "materializer_revision": declared_revision}}


def load_verified_v3_manifest(path: Path | str | None = None) -> dict:
    """输出 manifest 的严格加载：**唯一实现**是 `load_verified_v3_bundle()`。

    传入路径时先做 canonical 位置检查（副本 / 别名 / symlink 一律拒绝），
    随后委托统一入口完成**完整语义**验证。
    """
    if path is not None:
        _require_canonical_location(path)
    return load_verified_v3_bundle()["manifest"]


def load_verified_v3_source() -> dict:
    """source-v4 manifest 的严格加载（同样委托统一入口，无第二套规则）。"""
    return load_verified_v3_bundle()["source"]


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
    "load_verified_v3_bundle",
    "load_verified_v3_manifest",
    "load_verified_v3_source",
    "build_v3_output_manifest",
    "build_v3_source_manifest",
]
