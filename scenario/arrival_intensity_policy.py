"""M1.3f-e-a：**B6 arrival-intensity policy** 的冻结语义与 canonical 校验。

本模块把 B6-INTENSITY 人工裁决（D3/D4）机器可读化，并冻结为唯一 canonical
policy manifest `data/manifest/m13f_arrival_intensity_policy_v1.json`：

- 硬件 realization：`server_seed=0`；
- 温度口径：canonical train `[0,10224)` 的 **max** `temperature_deg_c` = **33.2°C**；
- 固定参数：`access_limit_kw=18.0`、`base_load=0.05`、`max_task_load_per_server=0.80`；
- 排除 PV / 风电 / BESS；均匀逐服务器负载 + 现有 `calc_pue_and_total_power` 接入二分；
- 原始持续容量 ≈ **79.985193** work/hour，**向下截断**到 3 位小数 → **79.985** work/hour；
- `rho_target=0.80` → main expected **63.988 work/hour** = **31.994 work/半小时**
  （`delta_t_hours=0.5`）；
- `source_kind=modeled_scenario`（human-approved；不声称实测 workload）；
- `rho_realized` 仅 `diagnostic_only`、禁止反调；`1000 work/半小时` 保持 stress 候选。

本模块**只冻结 policy，不物化新版 arrival**、不生成新 exogenous、不接线 env/train。
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from scenario.splits import (
    SplitError,
    _require_canonical_utc,
    _require_dict,
    _require_exact_keys,
    _require_git_sha40,
    _require_hex64,
    load_truth_split,
    logical_repo_path,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- 冻结的 B6 决策 -----------------------------------------------------------

POLICY_SCHEMA = "m1.3fea-b6-arrival-intensity-policy-v1"
CANONICAL_POLICY_NAME = "m13f_arrival_intensity_policy_v1.json"

DECISION_ID = "B6-INTENSITY"
APPROVED_ON = "2026-09-18"
SOURCE_KIND = "modeled_scenario"
EMPIRICAL_WORKLOAD_CLAIM = False

TEMPERATURE_FIELD = "temperature_deg_c"
TRAIN_RANGE: dict[str, Any] = {"split": "train", "row_start": 0, "row_end_exclusive": 10224}
TRAIN_MAX_TEMPERATURE_DEG_C = 33.2

SERVER_SEED = 0
ACCESS_LIMIT_KW = 18.0
BASE_LOAD = 0.05
MAX_TASK_LOAD_PER_SERVER = 0.80
RENEWABLE_STORAGE_CONTRIBUTION = "excluded"

CAPACITY_FORMULA = (
    "uniform per-server load; binary-search the load L where "
    "calc_pue_and_total_power(L, T_amb) / 1000 == access_limit_kw; "
    "sustainable_capacity = (L - base_load) * sum(C_server)"
)
CAPACITY_ROUNDING_RULE = "truncate_down_to_3_decimal_places"
UNROUNDED_CAPACITY_WORK_PER_HOUR = 79.985193
DECLARED_CAPACITY_WORK_PER_HOUR = 79.985

RHO_TARGET = 0.80
MAIN_EXPECTED_RATE_WORK_PER_HOUR = 63.988
MAIN_EXPECTED_AMOUNT_WORK_PER_HALF_HOUR = 31.994
DELTA_T_HOURS = 0.5

RHO_REALIZED_PURPOSE = "diagnostic_only"
REALIZATION_SEED = 20240916
FORBID_REALIZATION_FEEDBACK = True

AZURE_CONTRIBUTION = "shape_only"
STRESS_CANDIDATE = "1000 work-units/half-hour"
SENSITIVITY_LEVELS = "none"

# --- 代码 provenance ----------------------------------------------------------
#
# revision 由这一组实现文件解析（provider/物化器语义的**唯一**版本源）。
B6_SOURCE_PATHS: tuple[str, ...] = (
    "scenario/arrival_intensity_policy.py",
    "scripts/materialize_arrival_intensity_policy.py",
)

# 生产链的**固定**逻辑路径（loader 的 hash/path 绑定对象）。
CANONICAL_PARQUET_LOGICAL = "data/processed/singapore_2024/half_hour.parquet"
CANONICAL_MANIFEST_LOGICAL = "data/manifest/singapore_2024_half_hour.json"
SPLIT_MANIFEST_LOGICAL = "data/manifest/singapore_2024_splits.json"

POLICY_MANIFEST_KEYS: tuple[str, ...] = (
    "schema",
    "decision_id",
    "approved_on",
    "source_kind",
    "empirical_workload_claim",
    "canonical_parquet_path",
    "canonical_parquet_sha256",
    "canonical_manifest_path",
    "canonical_manifest_sha256",
    "split_manifest_path",
    "split_manifest_sha256",
    "temperature_field",
    "train_range",
    "train_max_temperature_deg_c",
    "server_seed",
    "access_limit_kw",
    "base_load",
    "max_task_load_per_server",
    "renewable_storage_contribution",
    "capacity_formula",
    "capacity_rounding_rule",
    "unrounded_capacity_work_per_hour",
    "declared_capacity_work_per_hour",
    "rho_target",
    "main_expected_rate_work_per_hour",
    "main_expected_amount_work_per_half_hour",
    "delta_t_hours",
    "rho_realized_purpose",
    "realization_seed",
    "forbid_realization_feedback",
    "azure_contribution",
    "stress_candidate",
    "sensitivity_levels",
    "materializer_revision",
    "frozen_at_utc",
)


class ArrivalIntensityPolicyError(ValueError):
    """B6 arrival-intensity policy 的**明确失败**（不 fallback、不静默）。"""


# --- Git / 数值 helper --------------------------------------------------------

def b6_code_revision() -> str:
    """本 policy 的 revision = 最后修改 B6_SOURCE_PATHS 的提交。"""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", *B6_SOURCE_PATHS],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as error:
        raise ArrivalIntensityPolicyError(
            f"materializer_revision 无法由 Git 解析：{error}"
            "（policy/materializer 实现必须先提交）"
        ) from error
    sha = out.strip().splitlines()[0].strip() if out.strip() else ""
    _require_git_sha40(sha, field="materializer_revision")
    return sha


def recompute_train_max_temperature(frame: Any) -> float:
    """train 切片的最大温度（frame 必须已是 train 切片，由调用方切片）。"""
    if TEMPERATURE_FIELD not in frame.columns:
        raise ArrivalIntensityPolicyError(
            f"train 切片缺少 {TEMPERATURE_FIELD!r} 列"
        )
    return float(frame[TEMPERATURE_FIELD].max())


def recompute_sustainable_capacity(server_seed: int, T_amb: float) -> float:
    """按冻结口径重算持续任务服务能力（work/hour）。

    均匀逐服务器负载，二分解出 `calc_pue_and_total_power == access_limit_kw` 的负载
    `lo`，返回 `(lo - base_load) * sum(C_server)`。**不读 validation/test。**
    """
    import numpy as np
    from envs.idc_price_env import IDCPriceEnv20D

    env = IDCPriceEnv20D(horizon=24, task_seed=0, server_seed=server_seed,
                         forecast_seed=300000)
    model = env.model
    total_capacity = float(np.asarray(model.C_server).sum())
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        loads = np.full((1, model.N), mid, dtype=float)
        power_kw = float(
            model.calc_pue_and_total_power(
                L_matrix=loads, T_amb=np.array([T_amb])
            )[0][0]
        ) / 1000.0
        if power_kw <= ACCESS_LIMIT_KW:
            lo = mid
        else:
            hi = mid
    return (lo - BASE_LOAD) * total_capacity


def truncate_down(value: float, decimals: int) -> float:
    """向下截断到 `decimals` 位小数（对正数即 floor）。"""
    factor = 10 ** decimals
    return float(math.floor(value * factor) / factor)


def declared_capacity_from(unrounded: float) -> float:
    """由未截断容量得到**声明容量**（向下截断到 3 位小数）。"""
    return truncate_down(unrounded, 3)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- canonical 路径门禁 -------------------------------------------------------

def _canonical_policy_dir() -> Path:
    """canonical policy 目录的**私有**解析器（测试只能 monkeypatch 它）。"""
    return REPO_ROOT / "data" / "manifest"


def canonical_policy_path() -> Path:
    """B6 policy 的**唯一 canonical** 绝对路径。"""
    return _canonical_policy_dir() / CANONICAL_POLICY_NAME


def _require_canonical_location(path: Path | str) -> Path:
    """**硬门禁**：路径必须**精确等于** canonical policy 文件（在读取 JSON 之前）。

    比较的是**词法绝对路径**（`absolute()` 不做符号链接解析），因此副本、别名、
    symlink 一律拒绝；**不得** fallback 到任何旧 policy。
    """
    given = Path(path).absolute()
    expected = canonical_policy_path().absolute()
    if given == expected:
        return expected
    raise ArrivalIntensityPolicyError(
        f"policy 路径必须是**唯一 canonical** 文件 "
        f"{logical_repo_path(expected)}；实际 {logical_repo_path(given)}"
        "（不接受副本、别名、symlink 或其它生成版本；无 fallback）"
    )


# --- 构建 candidate manifest --------------------------------------------------

def build_b6_policy_manifest(
    *,
    canonical_parquet_path: Path | str,
    canonical_manifest_path: Path | str,
    split_manifest_path: Path | str,
    frozen_at_utc: str,
) -> dict:
    """构造 candidate policy manifest（只读上游，重算温度与容量）。"""
    canonical_parquet_path = Path(canonical_parquet_path)
    canonical_manifest_path = Path(canonical_manifest_path)
    split_manifest_path = Path(split_manifest_path)

    _require_canonical_utc(frozen_at_utc, field="frozen_at_utc")

    # M1.3d 完整严格链（canonical manifest → parquet hash + split 边界）
    train = load_truth_split(
        "train",
        canonical_parquet_path=canonical_parquet_path,
        canonical_manifest_path=canonical_manifest_path,
        split_manifest_path=split_manifest_path,
    )
    train_max_temp = recompute_train_max_temperature(train)
    if train_max_temp != TRAIN_MAX_TEMPERATURE_DEG_C:
        raise ArrivalIntensityPolicyError(
            f"train max temperature 重算值 {train_max_temp!r} != 冻结值 "
            f"{TRAIN_MAX_TEMPERATURE_DEG_C!r}"
        )

    unrounded = recompute_sustainable_capacity(SERVER_SEED, TRAIN_MAX_TEMPERATURE_DEG_C)
    declared = declared_capacity_from(unrounded)
    if declared != DECLARED_CAPACITY_WORK_PER_HOUR:
        raise ArrivalIntensityPolicyError(
            f"声明容量重算值 {declared!r} != 冻结值 {DECLARED_CAPACITY_WORK_PER_HOUR!r}"
        )

    main_rate = round(declared * RHO_TARGET, 3)
    main_amount = round(main_rate * DELTA_T_HOURS, 3)

    manifest: dict[str, Any] = {
        "schema": POLICY_SCHEMA,
        "decision_id": DECISION_ID,
        "approved_on": APPROVED_ON,
        "source_kind": SOURCE_KIND,
        "empirical_workload_claim": EMPIRICAL_WORKLOAD_CLAIM,
        "canonical_parquet_path": logical_repo_path(canonical_parquet_path),
        "canonical_parquet_sha256": _sha256_file(canonical_parquet_path),
        "canonical_manifest_path": logical_repo_path(canonical_manifest_path),
        "canonical_manifest_sha256": _sha256_file(canonical_manifest_path),
        "split_manifest_path": logical_repo_path(split_manifest_path),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "temperature_field": TEMPERATURE_FIELD,
        "train_range": dict(TRAIN_RANGE),
        "train_max_temperature_deg_c": train_max_temp,
        "server_seed": SERVER_SEED,
        "access_limit_kw": ACCESS_LIMIT_KW,
        "base_load": BASE_LOAD,
        "max_task_load_per_server": MAX_TASK_LOAD_PER_SERVER,
        "renewable_storage_contribution": RENEWABLE_STORAGE_CONTRIBUTION,
        "capacity_formula": CAPACITY_FORMULA,
        "capacity_rounding_rule": CAPACITY_ROUNDING_RULE,
        "unrounded_capacity_work_per_hour": unrounded,
        "declared_capacity_work_per_hour": declared,
        "rho_target": RHO_TARGET,
        "main_expected_rate_work_per_hour": main_rate,
        "main_expected_amount_work_per_half_hour": main_amount,
        "delta_t_hours": DELTA_T_HOURS,
        "rho_realized_purpose": RHO_REALIZED_PURPOSE,
        "realization_seed": REALIZATION_SEED,
        "forbid_realization_feedback": FORBID_REALIZATION_FEEDBACK,
        "azure_contribution": AZURE_CONTRIBUTION,
        "stress_candidate": STRESS_CANDIDATE,
        "sensitivity_levels": SENSITIVITY_LEVELS,
        "materializer_revision": b6_code_revision(),
        "frozen_at_utc": frozen_at_utc,
    }
    _require_exact_keys(manifest, field="arrival intensity policy",
                        expected=POLICY_MANIFEST_KEYS)
    return manifest


# --- 严格加载 + 校验 ----------------------------------------------------------

_FROZEN_SCALARS: dict[str, Any] = {
    "decision_id": DECISION_ID,
    "approved_on": APPROVED_ON,
    "source_kind": SOURCE_KIND,
    "temperature_field": TEMPERATURE_FIELD,
    "train_range": dict(TRAIN_RANGE),
    "train_max_temperature_deg_c": TRAIN_MAX_TEMPERATURE_DEG_C,
    "server_seed": SERVER_SEED,
    "access_limit_kw": ACCESS_LIMIT_KW,
    "base_load": BASE_LOAD,
    "max_task_load_per_server": MAX_TASK_LOAD_PER_SERVER,
    "renewable_storage_contribution": RENEWABLE_STORAGE_CONTRIBUTION,
    "capacity_formula": CAPACITY_FORMULA,
    "capacity_rounding_rule": CAPACITY_ROUNDING_RULE,
    "declared_capacity_work_per_hour": DECLARED_CAPACITY_WORK_PER_HOUR,
    "rho_target": RHO_TARGET,
    "main_expected_rate_work_per_hour": MAIN_EXPECTED_RATE_WORK_PER_HOUR,
    "main_expected_amount_work_per_half_hour": MAIN_EXPECTED_AMOUNT_WORK_PER_HALF_HOUR,
    "delta_t_hours": DELTA_T_HOURS,
    "rho_realized_purpose": RHO_REALIZED_PURPOSE,
    "realization_seed": REALIZATION_SEED,
    "forbid_realization_feedback": FORBID_REALIZATION_FEEDBACK,
    "azure_contribution": AZURE_CONTRIBUTION,
    "stress_candidate": STRESS_CANDIDATE,
    "sensitivity_levels": SENSITIVITY_LEVELS,
}


def _verify_source_binding(payload: dict) -> None:
    """声明的 source path/hash 必须与**生产实际文件**逐项相符（无 fallback）。"""
    for logical, field_path, field_sha in (
        (CANONICAL_PARQUET_LOGICAL, "canonical_parquet_path", "canonical_parquet_sha256"),
        (CANONICAL_MANIFEST_LOGICAL, "canonical_manifest_path", "canonical_manifest_sha256"),
        (SPLIT_MANIFEST_LOGICAL, "split_manifest_path", "split_manifest_sha256"),
    ):
        if payload[field_path] != logical:
            raise ArrivalIntensityPolicyError(
                f"{field_path} 必须精确等于 {logical!r}，实际 {payload[field_path]!r}"
            )
        expected_sha = _require_hex64(payload[field_sha], field=field_sha)
        actual_sha = _sha256_file(REPO_ROOT / logical)
        if actual_sha != expected_sha:
            raise ArrivalIntensityPolicyError(
                f"{field_sha} 与实际文件不符：声明 {expected_sha} 实际 {actual_sha}"
            )


def _verify_recomputed_values(payload: dict) -> None:
    """重算温度 / 容量 / 派生量，必须与 manifest 记录一致。"""
    train = load_truth_split(
        "train",
        canonical_parquet_path=REPO_ROOT / CANONICAL_PARQUET_LOGICAL,
        canonical_manifest_path=REPO_ROOT / CANONICAL_MANIFEST_LOGICAL,
        split_manifest_path=REPO_ROOT / SPLIT_MANIFEST_LOGICAL,
    )
    recomputed_temp = recompute_train_max_temperature(train)
    if recomputed_temp != payload["train_max_temperature_deg_c"]:
        raise ArrivalIntensityPolicyError(
            f"train_max_temperature_deg_c 重算值 {recomputed_temp!r} != "
            f"manifest {payload['train_max_temperature_deg_c']!r}"
        )

    unrounded = recompute_sustainable_capacity(SERVER_SEED, TRAIN_MAX_TEMPERATURE_DEG_C)
    declared = declared_capacity_from(unrounded)
    if abs(unrounded - payload["unrounded_capacity_work_per_hour"]) > 1e-4:
        raise ArrivalIntensityPolicyError(
            f"unrounded_capacity 重算值 {unrounded!r} != "
            f"manifest {payload['unrounded_capacity_work_per_hour']!r}"
        )
    if declared != payload["declared_capacity_work_per_hour"]:
        raise ArrivalIntensityPolicyError(
            f"declared_capacity 重算值 {declared!r} != "
            f"manifest {payload['declared_capacity_work_per_hour']!r}"
        )

    main_rate = round(declared * payload["rho_target"], 3)
    main_amount = round(main_rate * payload["delta_t_hours"], 3)
    if main_rate != payload["main_expected_rate_work_per_hour"]:
        raise ArrivalIntensityPolicyError(
            f"main_expected_rate 重算值 {main_rate!r} != "
            f"manifest {payload['main_expected_rate_work_per_hour']!r}"
        )
    if main_amount != payload["main_expected_amount_work_per_half_hour"]:
        raise ArrivalIntensityPolicyError(
            f"main_expected_amount 重算值 {main_amount!r} != "
            f"manifest {payload['main_expected_amount_work_per_half_hour']!r}"
        )


def load_verified_b6_policy(path: Path | str | None = None) -> dict:
    """加载并**严格校验** canonical B6 policy（路径先于 JSON 读取）。

    依次校验：canonical 路径、精确键集合 + schema、冻结标量、source hash/path 绑定、
    materializer_revision、重算值。任一不符 fail closed，**无 fallback**。
    """
    if path is None:
        path = canonical_policy_path()
    path = _require_canonical_location(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArrivalIntensityPolicyError(f"policy 不可读：{error}") from error
    payload = _require_dict(payload, field="arrival intensity policy")
    _require_exact_keys(payload, field="arrival intensity policy",
                        expected=POLICY_MANIFEST_KEYS)

    if payload["schema"] != POLICY_SCHEMA:
        raise ArrivalIntensityPolicyError(
            f"schema 必须是 {POLICY_SCHEMA!r}，实际 {payload['schema']!r}"
        )
    if payload["empirical_workload_claim"] is not EMPIRICAL_WORKLOAD_CLAIM:
        raise ArrivalIntensityPolicyError("empirical_workload_claim 必须严格为 false")

    for field, expected in _FROZEN_SCALARS.items():
        if payload[field] != expected:
            raise ArrivalIntensityPolicyError(
                f"{field} 必须严格等于 {expected!r}，实际 {payload[field]!r}"
            )

    _verify_source_binding(payload)

    declared_revision = _require_git_sha40(payload["materializer_revision"],
                                           field="materializer_revision")
    live_revision = b6_code_revision()
    if declared_revision != live_revision:
        raise ArrivalIntensityPolicyError(
            f"materializer_revision 与 live 解析不符：声明 {declared_revision} "
            f"live {live_revision}"
        )

    _verify_recomputed_values(payload)
    return payload


__all__ = [
    "ArrivalIntensityPolicyError",
    "POLICY_SCHEMA",
    "POLICY_MANIFEST_KEYS",
    "CANONICAL_POLICY_NAME",
    "b6_code_revision",
    "recompute_train_max_temperature",
    "recompute_sustainable_capacity",
    "truncate_down",
    "declared_capacity_from",
    "canonical_policy_path",
    "build_b6_policy_manifest",
    "load_verified_b6_policy",
]
