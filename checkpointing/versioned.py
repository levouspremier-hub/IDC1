"""M2.3 版本化新主链 checkpoint 保存/加载壳。

红线：不修改 `marl/checkpointing/`；旧 23 维或无版本/schema 不符的件必须明确失败，
绝不填零、截断或 best-effort 加载。旧 MARL 模型只在旧基线进程中使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from contracts import CONTRACT_VERSION_ID

CURRENT_CONTRACT_VERSION = CONTRACT_VERSION_ID  # 唯一版本源（M3.10c）
CURRENT_ACTION_DIM = 21  # 20 compute + 1 signed storage


class CheckpointVersionError(ValueError):
    """版本/维度/schema 不符时抛出。"""


# 信封**自身**的五个元数据键。扩展字段（M6-P1 的评估输入契约等）必须走
# `extras`，**不**与这五个混在一起：`load()` 的「无版本 / 维度 / schema」拒绝
# 语义必须保持逐字不变。
ENVELOPE_METADATA_KEYS: tuple[str, ...] = (
    "contract_version_id", "action_dim", "obs_dim", "schema_hash", "code_revision",
)


@dataclass
class VersionedCheckpoint:
    contract_version_id: str
    action_dim: int
    obs_dim: int
    schema_hash: str
    code_revision: str
    state: dict[str, Any]
    # M6-P1：**可选**扩展元数据。默认空 ⇒ `save()` 写出的 metadata 与既有格式
    # 逐字节相同；`load()` 也**不**解析它（严格的键集合校验由上层契约负责）。
    extras: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        payload = {
            "metadata": {
                "contract_version_id": self.contract_version_id,
                "action_dim": self.action_dim,
                "obs_dim": self.obs_dim,
                "schema_hash": self.schema_hash,
                "code_revision": self.code_revision,
                **dict(self.extras),
            },
            "state": self.state,
        }
        torch.save(payload, str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_action_dim: int,
        expected_obs_dim: int,
        expected_schema_hash: str,
    ) -> VersionedCheckpoint:
        payload = read_checkpoint_payload(path)
        meta = validate_envelope_metadata(
            payload["metadata"],
            expected_action_dim=expected_action_dim,
            expected_obs_dim=expected_obs_dim,
            expected_schema_hash=expected_schema_hash,
        )
        return cls(
            contract_version_id=meta["contract_version_id"],
            action_dim=meta["action_dim"],
            obs_dim=meta["obs_dim"],
            schema_hash=meta["schema_hash"],
            code_revision=meta["code_revision"],
            state=payload["state"],
            extras={k: v for k, v in meta.items() if k not in ENVELOPE_METADATA_KEYS},
        )


def read_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    """读取 checkpoint 原始载荷（`{"metadata", "state"}`），**每个文件只读一次**。

    上层契约（如 M6-P1 的评估输入）先用本函数拿到载荷，再调用
    `validate_envelope_metadata` 复用同一份信封语义，避免二次读取造成的
    「读到的不是同一个文件」。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    # weights_only=False：加载本仓库自产 checkpoint（含元数据与任意状态），非不可信输入。
    payload = torch.load(str(path), weights_only=False)
    if not isinstance(payload, dict) or payload.get("metadata") is None:
        raise CheckpointVersionError("checkpoint 缺少 metadata（无版本）")
    if payload.get("state") is None:
        raise CheckpointVersionError("checkpoint 缺少 state")
    return payload


def validate_envelope_metadata(
    meta: object,
    *,
    expected_action_dim: int,
    expected_obs_dim: int,
    expected_schema_hash: str,
) -> dict[str, Any]:
    """信封校验：版本 / 动作维度 / obs 维度 / schema。**无版本即拒绝。**"""
    if not isinstance(meta, dict):
        raise CheckpointVersionError("checkpoint 缺少 metadata（无版本）")
    if meta.get("contract_version_id") != CURRENT_CONTRACT_VERSION:
        raise CheckpointVersionError(
            f"contract_version_id 不匹配: {meta.get('contract_version_id')} "
            f"!= {CURRENT_CONTRACT_VERSION}"
        )
    if meta.get("action_dim") != expected_action_dim:
        raise CheckpointVersionError(
            f"action_dim {meta.get('action_dim')} != 期望 {expected_action_dim}"
        )
    if meta.get("obs_dim") != expected_obs_dim:
        raise CheckpointVersionError(
            f"obs_dim {meta.get('obs_dim')} != 期望 {expected_obs_dim}"
        )
    if meta.get("schema_hash") != expected_schema_hash:
        raise CheckpointVersionError("schema_hash 不匹配")
    return dict(meta)
