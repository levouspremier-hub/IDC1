"""M2.3 版本化新主链 checkpoint（不修改 marl/checkpointing/）。

M6-P1 起另加**评估输入契约**（`checkpointing.eval_input`）：复用本模块的
版本化信封，并补齐评估必须知道的角色 / 动作模式 / 策略结构 / train origin /
种子 / 配置与资产来源。
"""

from checkpointing.eval_input import (
    CONTROLLED_ROLE,
    EVAL_INPUT_REQUIRED_SOURCE_ROLES,
    EVAL_INPUT_SCHEMA,
    FORMAL_ROLE,
    EvaluationInputCheckpoint,
    EvaluationInputError,
    load_evaluation_checkpoint,
    save_evaluation_checkpoint,
)
from checkpointing.versioned import (
    CURRENT_ACTION_DIM,
    CURRENT_CONTRACT_VERSION,
    ENVELOPE_METADATA_KEYS,
    CheckpointVersionError,
    VersionedCheckpoint,
    read_checkpoint_payload,
    validate_envelope_metadata,
)

__all__ = [
    "CONTROLLED_ROLE",
    "CURRENT_ACTION_DIM",
    "CURRENT_CONTRACT_VERSION",
    "ENVELOPE_METADATA_KEYS",
    "EVAL_INPUT_REQUIRED_SOURCE_ROLES",
    "EVAL_INPUT_SCHEMA",
    "FORMAL_ROLE",
    "CheckpointVersionError",
    "EvaluationInputCheckpoint",
    "EvaluationInputError",
    "VersionedCheckpoint",
    "load_evaluation_checkpoint",
    "read_checkpoint_payload",
    "save_evaluation_checkpoint",
    "validate_envelope_metadata",
]
