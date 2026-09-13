"""M2.3 版本化新主链 checkpoint 保存/加载壳。

红线：不修改 `marl/checkpointing/`；旧 23 维或无版本/schema 不符的件必须明确失败，
绝不填零、截断或 best-effort 加载。旧 MARL 模型只在旧基线进程中使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CURRENT_CONTRACT_VERSION = "contract-v2"
CURRENT_ACTION_DIM = 21  # 20 compute + 1 signed storage


class CheckpointVersionError(ValueError):
    """版本/维度/schema 不符时抛出。"""


@dataclass
class VersionedCheckpoint:
    contract_version_id: str
    action_dim: int
    obs_dim: int
    schema_hash: str
    code_revision: str
    state: dict[str, Any]

    def save(self, path: str | Path) -> None:
        payload = {
            "metadata": {
                "contract_version_id": self.contract_version_id,
                "action_dim": self.action_dim,
                "obs_dim": self.obs_dim,
                "schema_hash": self.schema_hash,
                "code_revision": self.code_revision,
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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        # weights_only=False：加载本仓库自产 checkpoint（含元数据与任意状态），非不可信输入。
        payload = torch.load(str(path), weights_only=False)
        meta = payload.get("metadata")
        if meta is None:
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
        return cls(
            contract_version_id=meta["contract_version_id"],
            action_dim=meta["action_dim"],
            obs_dim=meta["obs_dim"],
            schema_hash=meta["schema_hash"],
            code_revision=meta["code_revision"],
            state=payload["state"],
        )
