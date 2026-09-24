"""M6-P1：评估输入的**配置与资产来源**绑定与实时校验。

`checkpointing.eval_input` 只校验来源的**结构**（角色集合 / 规范逻辑路径 / SHA 形状）。
本模块负责**语义**绑定：每个角色对应**唯一**规范逻辑路径，且记录值必须等于
**当前磁盘上**该文件的实测 SHA-256。任何篡改（改 hash、把角色指向副本/别名）
一律 fail closed，**无 fallback**。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from checkpointing.eval_input import (
    EVAL_INPUT_REQUIRED_SOURCE_ROLES,
    EvaluationInputCheckpoint,
)
from contracts.models import ArtifactDigest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 角色 → 规范逻辑路径（仓库相对）。**唯一**绑定，不做别名或副本解析。
ROLE_LOGICAL_PATHS: dict[str, str] = {
    "canonical_parquet": "data/processed/singapore_2024/half_hour.parquet",
    "canonical_manifest": "data/manifest/singapore_2024_half_hour.json",
    "truth_split_manifest": "data/manifest/singapore_2024_splits.json",
    "forecast_policy_v3": "data/manifest/singapore_2024_forecast_policy_v3.json",
    "b6_arrival_policy": "data/manifest/m13f_arrival_intensity_policy_v1.json",
    "exogenous_v3_manifest": "data/manifest/singapore_2024_exogenous_v3.json",
    "exogenous_v3_source_manifest": "data/manifest/m13f_materialization_sources_v4.json",
    "exogenous_v3_parquet": "data/processed/singapore_2024/exogenous_drivers_v3.parquet",
    "refs_v4": "configs/frozen_refs/refs_v4.json",
    "formal_split_manifest_train": "data/manifest/formal_splits_v5/train.json",
    "arrival_mapper_policy": "data/manifest/m13g_arrival_mapper_v1.json",
    "env_release": "configs/release/idc_formal_env_release_v1.json",
}


class EvaluationSourceError(ValueError):
    """来源绑定的明确失败。"""


def sha256_file(path: Path | str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as error:
        raise EvaluationSourceError(f"来源资产不可读：{path}：{error}") from error


def canonical_source_paths() -> dict[str, Path]:
    """每个角色的规范绝对路径（键集合必须精确等于必需角色集合）。"""
    if set(ROLE_LOGICAL_PATHS) != set(EVAL_INPUT_REQUIRED_SOURCE_ROLES):
        raise EvaluationSourceError(
            "ROLE_LOGICAL_PATHS 与评估输入的必需来源角色不一致："
            f"缺少={sorted(set(EVAL_INPUT_REQUIRED_SOURCE_ROLES) - set(ROLE_LOGICAL_PATHS))}，"
            f"多余={sorted(set(ROLE_LOGICAL_PATHS) - set(EVAL_INPUT_REQUIRED_SOURCE_ROLES))}"
        )
    return {role: REPO_ROOT / logical for role, logical in ROLE_LOGICAL_PATHS.items()}


def canonical_source_digests() -> tuple[ArtifactDigest, ...]:
    """**当前**上游资产的实测 digest（按必需角色的固定顺序）。"""
    paths = canonical_source_paths()
    return tuple(
        ArtifactDigest(
            role=role,
            logical_path=ROLE_LOGICAL_PATHS[role],
            sha256=sha256_file(paths[role]),
        )
        for role in EVAL_INPUT_REQUIRED_SOURCE_ROLES
    )


def verify_evaluation_input_sources(checkpoint: EvaluationInputCheckpoint) -> None:
    """记录来源必须**逐项**等于规范路径 + 当前实测 SHA-256。"""
    expected = {digest.role: digest for digest in canonical_source_digests()}
    recorded = {digest.role: digest for digest in checkpoint.sources}
    if set(recorded) != set(expected):
        raise EvaluationSourceError(
            f"来源角色集合不符：缺少={sorted(set(expected) - set(recorded))}，"
            f"多余={sorted(set(recorded) - set(expected))}"
        )
    for role in EVAL_INPUT_REQUIRED_SOURCE_ROLES:
        got, want = recorded[role], expected[role]
        if got.logical_path != want.logical_path:
            raise EvaluationSourceError(
                f"来源 {role!r} 的逻辑路径必须精确等于 {want.logical_path!r}，"
                f"实际 {got.logical_path!r}（不接受副本、别名或 symlink）"
            )
        if got.sha256 != want.sha256:
            raise EvaluationSourceError(
                f"来源 {role!r} 的 SHA-256 与当前资产不符："
                f"记录 {got.sha256}，实测 {want.sha256}"
            )


__all__ = [
    "ROLE_LOGICAL_PATHS",
    "EvaluationSourceError",
    "canonical_source_digests",
    "canonical_source_paths",
    "sha256_file",
    "verify_evaluation_input_sources",
]
