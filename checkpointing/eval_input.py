"""M6-P1：**评估输入契约** —— 评估可读 checkpoint 的版本、维度、权重与来源。

复用 `checkpointing.VersionedCheckpoint` 的版本化信封（版本 / 21 维动作 /
观测维度 / schema），在其上补齐评估**必须**知道的信息：

```text
artifact_role   本件的角色（受控短跑评估输入 / 正式训练策略）
action_mode     评估时的动作模式（deterministic_mean）
policy_config   策略结构（hidden / hidden_layers / activation / obs_dim / action_dim）
train_split     只能来自 train（validation / test 一律拒绝）
train_origin    受控短跑的 train origin（split 内局部序号）与其规范 start
seeds           task / server / forecast 三个环境种子
sources         配置与资产来源（role + 规范逻辑路径 + SHA-256），**逐项**绑定
```

**红线**：

- 缺字段、多字段、维度 / 版本 / schema / 来源不符、非 `train` origin、
  权重形状与声明不符 —— **一律 fail closed**，绝不填零、截断或 best-effort 加载；
- 本格式是**受控短跑评估输入**，**不是**正式训练产物；
  `safe_rl_v2/ppo_two_batch.py` 的 two-batch 恢复格式（schema
  `m1.3g-f-c-e-two-batch-resume-v1`）**不得**冒充本契约；
- 正式训练 checkpoint 后续必须满足**同一**评估输入契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from checkpointing.versioned import (
    ENVELOPE_METADATA_KEYS,
    VersionedCheckpoint,
    read_checkpoint_payload,
    validate_envelope_metadata,
)
from contracts.models import ArtifactDigest

# 本契约的 schema 标识（**不得**与 two-batch 恢复格式共用）。
EVAL_INPUT_SCHEMA = "m6p1-eval-input-v1"

CONTROLLED_ROLE = "controlled_short_run_eval_input"
FORMAL_ROLE = "formal_training_policy"
EVAL_INPUT_ARTIFACT_ROLES: tuple[str, ...] = (CONTROLLED_ROLE, FORMAL_ROLE)

EVAL_INPUT_ACTION_MODES: tuple[str, ...] = ("deterministic_mean",)

EVAL_INPUT_POLICY_CONFIG_KEYS: tuple[str, ...] = (
    "hidden", "hidden_layers", "activation", "obs_dim", "action_dim",
)
EVAL_INPUT_SEED_KEYS: tuple[str, ...] = ("task", "server", "forecast")
# `state` 必须**至少**含 policy；允许携带训练状态，但不得出现未知键。
EVAL_INPUT_REQUIRED_STATE_KEYS: tuple[str, ...] = ("policy",)
EVAL_INPUT_ALLOWED_STATE_KEYS: tuple[str, ...] = ("policy", "optimizer", "lagrangian")

# 评估**必须**能追溯到的配置 / 资产来源角色。逐项绑定规范逻辑路径 + SHA-256。
EVAL_INPUT_REQUIRED_SOURCE_ROLES: tuple[str, ...] = (
    "canonical_parquet",
    "canonical_manifest",
    "truth_split_manifest",
    "forecast_policy_v3",
    "b6_arrival_policy",
    "exogenous_v3_manifest",
    "exogenous_v3_source_manifest",
    "exogenous_v3_parquet",
    "refs_v4",
    "formal_split_manifest_train",
    "arrival_mapper_policy",
    "env_release",
)

EVAL_INPUT_METADATA_KEYS: tuple[str, ...] = (
    *ENVELOPE_METADATA_KEYS,
    "artifact_role",
    "action_mode",
    "policy_config",
    "train_split",
    "train_origin",
    "train_start",
    "seeds",
    "sources",
)

EVAL_INPUT_TRAIN_SPLIT = "train"


class EvaluationInputError(ValueError):
    """评估输入契约的**明确失败**（无 fallback、不静默）。"""


@dataclass(frozen=True)
class EvaluationInputCheckpoint:
    """**已验签**的评估输入：信封 + 角色 + 动作模式 + 结构 + train origin + 权重。"""

    envelope: VersionedCheckpoint
    artifact_role: str
    action_mode: str
    policy_config: dict[str, Any]
    train_split: str
    train_origin: int
    train_start: str
    seeds: dict[str, int]
    sources: tuple[ArtifactDigest, ...]
    policy: Any  # safe_rl_v2.policy.SafePPOPolicy（延迟导入，避免模块级依赖）

    @property
    def is_controlled_short_run(self) -> bool:
        return self.artifact_role == CONTROLLED_ROLE


def _policy_class():
    """延迟导入：`safe_rl_v2` 是上层模块，`checkpointing` 不在模块级依赖它。"""
    from safe_rl_v2.policy import SafePPOPolicy

    return SafePPOPolicy


def _require_plain_int(value: object, *, field: str) -> int:
    """**coercion 之前**的严格整数（bool 不是整数）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationInputError(f"{field} 必须是整数（bool 不算），实际 {value!r}")
    return value


def _require_policy_config(config: object, *, obs_dim: int, action_dim: int) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise EvaluationInputError(f"policy_config 必须是对象，实际 {type(config).__name__}")
    if set(config) != set(EVAL_INPUT_POLICY_CONFIG_KEYS):
        missing = sorted(set(EVAL_INPUT_POLICY_CONFIG_KEYS) - set(config))
        extra = sorted(set(config) - set(EVAL_INPUT_POLICY_CONFIG_KEYS))
        raise EvaluationInputError(
            f"policy_config 键集合必须精确等于 {list(EVAL_INPUT_POLICY_CONFIG_KEYS)}；"
            f"缺少={missing}，多余={extra}"
        )
    hidden = _require_plain_int(config["hidden"], field="policy_config.hidden")
    layers = _require_plain_int(config["hidden_layers"], field="policy_config.hidden_layers")
    cfg_obs = _require_plain_int(config["obs_dim"], field="policy_config.obs_dim")
    cfg_act = _require_plain_int(config["action_dim"], field="policy_config.action_dim")
    if layers != 1:
        raise EvaluationInputError(
            f"policy_config.hidden_layers 必须为 1（单隐层策略），实际 {layers}"
        )
    if hidden <= 0:
        raise EvaluationInputError(f"policy_config.hidden 必须为正，实际 {hidden}")
    if cfg_obs != obs_dim:
        raise EvaluationInputError(
            f"policy_config.obs_dim {cfg_obs} != 信封 obs_dim {obs_dim}"
        )
    if cfg_act != action_dim:
        raise EvaluationInputError(
            f"policy_config.action_dim {cfg_act} != 信封 action_dim {action_dim}"
        )
    if not isinstance(config["activation"], str) or not config["activation"]:
        raise EvaluationInputError("policy_config.activation 必须是非空字符串")
    return dict(config)


def _require_seeds(seeds: object) -> dict[str, int]:
    if not isinstance(seeds, dict):
        raise EvaluationInputError(f"seeds 必须是对象，实际 {type(seeds).__name__}")
    if set(seeds) != set(EVAL_INPUT_SEED_KEYS):
        missing = sorted(set(EVAL_INPUT_SEED_KEYS) - set(seeds))
        extra = sorted(set(seeds) - set(EVAL_INPUT_SEED_KEYS))
        raise EvaluationInputError(
            f"seeds 键集合必须精确等于 {list(EVAL_INPUT_SEED_KEYS)}；"
            f"缺少={missing}，多余={extra}"
        )
    return {
        key: _require_plain_int(seeds[key], field=f"seed[{key}]")
        for key in EVAL_INPUT_SEED_KEYS
    }


def _require_sources(sources: object) -> tuple[ArtifactDigest, ...]:
    if isinstance(sources, (str, bytes)) or not isinstance(sources, (list, tuple)):
        raise EvaluationInputError(
            f"sources 必须是 list/tuple，实际 {type(sources).__name__}"
        )
    if not sources:
        raise EvaluationInputError("sources 不得为空（评估输入必须绑定配置与资产来源）")
    digests: list[ArtifactDigest] = []
    for index, entry in enumerate(sources):
        if not isinstance(entry, dict):
            raise EvaluationInputError(f"sources[{index}] 必须是对象")
        try:
            digests.append(ArtifactDigest(**entry))
        except Exception as error:  # pydantic ValidationError → 统一为本契约错误
            raise EvaluationInputError(f"sources[{index}] 非法：{error}") from error
    roles = [digest.role for digest in digests]
    if len(set(roles)) != len(roles):
        duplicated = sorted({role for role in roles if roles.count(role) > 1})
        raise EvaluationInputError(f"sources.role 必须唯一，重复={duplicated}")
    missing = sorted(set(EVAL_INPUT_REQUIRED_SOURCE_ROLES) - set(roles))
    if missing:
        raise EvaluationInputError(
            f"sources 缺少必需来源角色：{missing}"
            f"（必需集合 {list(EVAL_INPUT_REQUIRED_SOURCE_ROLES)}）"
        )
    unknown = sorted(set(roles) - set(EVAL_INPUT_REQUIRED_SOURCE_ROLES))
    if unknown:
        raise EvaluationInputError(f"sources 含未知来源角色：{unknown}")
    return tuple(digests)


def _require_state(state: object, *, expected_keys: set[str]) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise EvaluationInputError(f"state 必须是对象，实际 {type(state).__name__}")
    unknown = sorted(set(state) - set(EVAL_INPUT_ALLOWED_STATE_KEYS))
    if unknown:
        raise EvaluationInputError(
            f"state 含未知键：{unknown}（允许 {list(EVAL_INPUT_ALLOWED_STATE_KEYS)}）"
        )
    missing = sorted(set(EVAL_INPUT_REQUIRED_STATE_KEYS) - set(state))
    if missing:
        raise EvaluationInputError(f"state 缺少必需键：{missing}")
    if set(state["policy"]) != expected_keys:
        missing_p = sorted(expected_keys - set(state["policy"]))
        extra_p = sorted(set(state["policy"]) - expected_keys)
        raise EvaluationInputError(
            f"state['policy'] 的键集合必须与策略结构一致；缺少={missing_p}，多余={extra_p}"
        )
    for key, value in state["policy"].items():
        if not torch.is_tensor(value):
            raise EvaluationInputError(f"state['policy'][{key!r}] 必须是张量")
    return state


def _build_policy(policy_config: dict[str, Any], state: dict[str, Any]):
    policy = _policy_class()(
        obs_dim=int(policy_config["obs_dim"]),
        action_dim=int(policy_config["action_dim"]),
        hidden=int(policy_config["hidden"]),
    )
    try:
        policy.load_state_dict(state["policy"])
    except RuntimeError as error:
        raise EvaluationInputError(
            "策略权重与声明的 policy_config 不符（obs_dim="
            f"{policy_config['obs_dim']}, hidden={policy_config['hidden']}, "
            f"action_dim={policy_config['action_dim']}）：{error}"
        ) from error
    policy.eval()
    return policy


def save_evaluation_checkpoint(
    path: str | Path,
    *,
    policy: Any,
    obs_dim: int,
    code_revision: str,
    artifact_role: str,
    action_mode: str,
    policy_config: dict[str, Any],
    train_split: str,
    train_origin: int,
    train_start: str,
    seeds: dict[str, int],
    sources: object,
    action_dim: int = 21,
) -> dict[str, Any]:
    """按评估输入契约保存 checkpoint（**写之前**先完整校验，失败则不落盘）。"""
    obs_dim = _require_plain_int(obs_dim, field="obs_dim")
    action_dim = _require_plain_int(action_dim, field="action_dim")
    if action_dim != 21:
        raise EvaluationInputError(
            f"评估输入的 action_dim 只接受 21（20 计算 + 1 储能），实际 {action_dim}"
        )
    if artifact_role not in EVAL_INPUT_ARTIFACT_ROLES:
        raise EvaluationInputError(
            f"artifact_role 必须是 {list(EVAL_INPUT_ARTIFACT_ROLES)} 之一，"
            f"实际 {artifact_role!r}"
        )
    if action_mode not in EVAL_INPUT_ACTION_MODES:
        raise EvaluationInputError(
            f"action_mode 必须是 {list(EVAL_INPUT_ACTION_MODES)} 之一，实际 {action_mode!r}"
        )
    if train_split != EVAL_INPUT_TRAIN_SPLIT:
        raise EvaluationInputError(
            f"train_split 只能是 {EVAL_INPUT_TRAIN_SPLIT!r}（评估输入不得来自 "
            f"validation/test），实际 {train_split!r}"
        )
    train_origin = _require_plain_int(train_origin, field="train_origin")
    if train_origin < 0:
        raise EvaluationInputError(f"train_origin 不得为负，实际 {train_origin}")
    if not isinstance(train_start, str) or not train_start:
        raise EvaluationInputError("train_start 必须是非空字符串")

    config = _require_policy_config(policy_config, obs_dim=obs_dim, action_dim=action_dim)
    seed_map = _require_seeds(seeds)
    digests = _require_sources(sources)

    if policy is None:
        raise EvaluationInputError("policy 不得为空（缺少策略权重）")
    state: dict[str, Any] = {
        "policy": {k: v.detach().clone() for k, v in policy.state_dict().items()},
    }
    expected_keys = set(_policy_class()(
        obs_dim=obs_dim, action_dim=action_dim, hidden=int(config["hidden"])).state_dict())
    _require_state(state, expected_keys=expected_keys)

    checkpoint = VersionedCheckpoint(
        contract_version_id=__import__("contracts").CONTRACT_VERSION_ID,
        action_dim=action_dim,
        obs_dim=obs_dim,
        schema_hash=EVAL_INPUT_SCHEMA,
        code_revision=str(code_revision),
        state=state,
        extras={
            "artifact_role": artifact_role,
            "action_mode": action_mode,
            "policy_config": config,
            "train_split": train_split,
            "train_origin": train_origin,
            "train_start": train_start,
            "seeds": seed_map,
            "sources": [digest.model_dump() for digest in digests],
        },
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.save(path)
    # 落盘后**再读一次**并用正式 loader 复核：写出的字节必须真的可被评估读取。
    loaded = load_evaluation_checkpoint(path)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_hash": EVAL_INPUT_SCHEMA,
        "artifact_role": artifact_role,
        "action_mode": action_mode,
        "train_split": train_split,
        "train_origin": train_origin,
        "train_start": train_start,
        "seeds": dict(seed_map),
        "is_controlled_short_run": loaded.is_controlled_short_run,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluation_checkpoint(
    path: str | Path, *, expected_action_dim: int = 21
) -> EvaluationInputCheckpoint:
    """读取并**严格校验**评估输入 checkpoint（每个文件只读一次）。"""
    payload = read_checkpoint_payload(path)
    meta = payload["metadata"]
    if not isinstance(meta, dict):
        raise EvaluationInputError("checkpoint 缺少 metadata（无版本）")
    # 顺序有意如此：**先**验信封（版本 / 动作维度 / schema / obs 维度是否为整数），
    # **再**验评估输入契约的精确键集合。这样「版本不符」与「缺或多字段」不会互相掩盖。
    obs_dim = _require_plain_int(meta.get("obs_dim"), field="obs_dim")
    envelope = validate_envelope_metadata(
        meta,
        expected_action_dim=_require_plain_int(expected_action_dim, field="expected_action_dim"),
        expected_obs_dim=obs_dim,
        expected_schema_hash=EVAL_INPUT_SCHEMA,
    )
    if set(meta) != set(EVAL_INPUT_METADATA_KEYS):
        missing = sorted(set(EVAL_INPUT_METADATA_KEYS) - set(meta))
        extra = sorted(set(meta) - set(EVAL_INPUT_METADATA_KEYS))
        raise EvaluationInputError(
            f"metadata 键集合必须精确等于评估输入契约；缺少={missing}，多余={extra}"
        )

    artifact_role = meta["artifact_role"]
    if artifact_role not in EVAL_INPUT_ARTIFACT_ROLES:
        raise EvaluationInputError(
            f"artifact_role 必须是 {list(EVAL_INPUT_ARTIFACT_ROLES)} 之一，"
            f"实际 {artifact_role!r}"
        )
    action_mode = meta["action_mode"]
    if action_mode not in EVAL_INPUT_ACTION_MODES:
        raise EvaluationInputError(
            f"action_mode 必须是 {list(EVAL_INPUT_ACTION_MODES)} 之一，实际 {action_mode!r}"
        )
    if meta["train_split"] != EVAL_INPUT_TRAIN_SPLIT:
        raise EvaluationInputError(
            f"train_split 只能是 {EVAL_INPUT_TRAIN_SPLIT!r}，实际 {meta['train_split']!r}"
        )
    train_origin = _require_plain_int(meta["train_origin"], field="train_origin")
    train_start = meta["train_start"]
    if not isinstance(train_start, str) or not train_start:
        raise EvaluationInputError("train_start 必须是非空字符串")
    config = _require_policy_config(
        meta["policy_config"], obs_dim=obs_dim, action_dim=envelope["action_dim"])
    seed_map = _require_seeds(meta["seeds"])
    digests = _require_sources(meta["sources"])

    state = _require_state(
        payload["state"],
        expected_keys=set(_policy_class()(
            obs_dim=obs_dim, action_dim=int(envelope["action_dim"]),
            hidden=int(config["hidden"])).state_dict()),
    )
    policy = _build_policy(config, state)

    return EvaluationInputCheckpoint(
        envelope=VersionedCheckpoint(
            contract_version_id=envelope["contract_version_id"],
            action_dim=envelope["action_dim"],
            obs_dim=envelope["obs_dim"],
            schema_hash=envelope["schema_hash"],
            code_revision=envelope["code_revision"],
            state=state,
        ),
        artifact_role=artifact_role,
        action_mode=action_mode,
        policy_config=config,
        train_split=meta["train_split"],
        train_origin=train_origin,
        train_start=train_start,
        seeds=seed_map,
        sources=digests,
        policy=policy,
    )


__all__ = [
    "CONTROLLED_ROLE",
    "EVAL_INPUT_ACTION_MODES",
    "EVAL_INPUT_ARTIFACT_ROLES",
    "EVAL_INPUT_METADATA_KEYS",
    "EVAL_INPUT_REQUIRED_SOURCE_ROLES",
    "EVAL_INPUT_SCHEMA",
    "EVAL_INPUT_TRAIN_SPLIT",
    "FORMAL_ROLE",
    "EvaluationInputCheckpoint",
    "EvaluationInputError",
    "load_evaluation_checkpoint",
    "save_evaluation_checkpoint",
]
