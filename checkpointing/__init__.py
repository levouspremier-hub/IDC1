"""M2.3 版本化新主链 checkpoint（不修改 marl/checkpointing/）。"""

from checkpointing.versioned import (
    CURRENT_ACTION_DIM,
    CURRENT_CONTRACT_VERSION,
    CheckpointVersionError,
    VersionedCheckpoint,
)

__all__ = [
    "CURRENT_ACTION_DIM",
    "CURRENT_CONTRACT_VERSION",
    "CheckpointVersionError",
    "VersionedCheckpoint",
]
