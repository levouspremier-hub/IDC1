"""M5.1 / M5.1a 版本化 rollout 记录契约（唯一训练样本格式）。

红线：
- `old_raw_log_prob` **只绑定 `raw_action`**；执行动作侧不存在任何 log-prob 字段
  （见 `docs/task_cards/M5.1a.md` §5）；
- `add()` 对数组与 `correction_info` 做**防御性复制**，调用方后续修改不得污染已存 transition；
- 严格拒绝：动作维度错误、观测非一维/空/维度前后不一致、非有限数值、缺失版本、
  版本不一致、未知旧版本、`terminated`/`truncated` 非布尔、`correction_info` 不可序列化；
- `from_dict()` **不补默认字段、不接受旧 payload**。

单位纪律：业务违规量（计数）、碳排放量（质量）、电费（货币）三类量纲互不相同，
字段名带单位后缀，且 payload 携带 `units` 元数据，禁止互相覆盖或混算。

本卡只做数据契约与持久化校验：**不实现** PPO ratio/clip、GAE、actor loss、乘子更新或训练循环接线
（见 `docs/task_cards/M5.1a.md` §6，属 M5.1b）。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from contracts import CONTRACT_VERSION_ID

CONTRACT_VERSION = CONTRACT_VERSION_ID  # 唯一版本源（M3.10c）
ACTION_DIM = 21  # 20 compute + 1 signed storage

# --- 单位元数据：三类量纲互不相同，禁止混合 ---------------------------------
UNIT_BUSINESS_COST = "violation_count"  # business_cost：业务违规量（计数，非货币）
UNIT_CARBON_COST = "kgCO2e"  # carbon_cost：碳排放量（质量，非货币）
UNIT_ELECTRICITY_COST = "SGD"  # electricity_cost_sgd：电费（货币）
UNIT_REWARD = "dimensionless"

UNIT_METADATA: dict[str, str] = {
    "reward": UNIT_REWARD,
    "business_cost": UNIT_BUSINESS_COST,
    "carbon_cost": UNIT_CARBON_COST,
    "electricity_cost_sgd": UNIT_ELECTRICITY_COST,
}

# 持久化 payload 的必填字段（缺任一即拒绝，不补默认值）
_REQUIRED_TRANSITION_FIELDS = (
    "observation",
    "next_observation",
    "raw_action",
    "old_raw_log_prob",
    "exec_action",
    "reward",
    "business_cost",
    "carbon_cost",
    "electricity_cost_sgd",
    "terminated",
    "truncated",
    "correction_info",
    "contract_version",
)
_REQUIRED_TOP_FIELDS = ("contract_version", "action_dim", "units", "transitions")

_REAL_SCALAR_TYPES = (int, float, np.integer, np.floating)


@dataclass
class Transition:
    """单步 rollout 记录（raw 与 exec 严格分离）。

    三类量纲字段互相独立、不得混用：`business_cost`（违规计数）、
    `carbon_cost`（kgCO2e 排放质量）、`electricity_cost_sgd`（SGD 电费）。
    """

    observation: np.ndarray
    next_observation: np.ndarray
    raw_action: np.ndarray
    old_raw_log_prob: float
    exec_action: np.ndarray
    reward: float
    business_cost: float
    carbon_cost: float
    electricity_cost_sgd: float
    terminated: bool
    truncated: bool
    correction_info: dict = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION


def _as_float_array(value: Any, name: str, expected_dim: int | None = None) -> np.ndarray:
    """转换为一维 float64 数组并做防御性复制；形状/有限性不合法即拒绝。"""
    try:
        arr = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须为数值数组: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须为一维数组，got ndim={arr.ndim} shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} 不得为空数组")
    if expected_dim is not None and arr.shape != (expected_dim,):
        raise ValueError(f"{name} 必须 {expected_dim} 维，got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限数值 (non-finite: NaN/Inf)")
    return arr


def _finite_scalar(value: Any, name: str) -> float:
    """只接受实数标量：字符串/布尔/容器一律拒绝，不静默转换。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, _REAL_SCALAR_TYPES):
        raise TypeError(f"{name} 必须为实数标量，got {type(value).__name__}={value!r}")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} 为非有限数值 (non-finite: NaN/Inf)")
    return scalar


def _as_bool(value: Any, name: str) -> bool:
    """只接受 bool / numpy.bool_：禁止 bool("false")、bool(1) 之类的静默转换。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(
        f"{name} 必须为 bool 或 numpy.bool_，got {type(value).__name__}={value!r}"
        "（禁止静默转换）"
    )


def _sanitize_correction_info(info: Any) -> dict:
    """correction_info 必须可深拷贝且可 JSON 序列化（禁止 torch tensor / numpy scalar 等）。"""
    if not isinstance(info, dict):
        raise TypeError(f"correction_info 必须为 dict，got {type(info).__name__}")
    try:
        cloned = copy.deepcopy(info)
    except Exception as exc:  # pragma: no cover - 防御
        raise TypeError(f"correction_info 无法深拷贝: {exc}") from exc
    try:
        json.dumps(cloned)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"correction_info 必须可 JSON 序列化: {exc}") from exc
    return cloned


def _validate_units(units: Any) -> dict:
    if not isinstance(units, dict):
        raise TypeError(f"units 必须为 dict，got {type(units).__name__}")
    for key, expected in UNIT_METADATA.items():
        if key not in units:
            raise ValueError(f"units 缺少字段 {key!r}（不补默认值）")
        if units[key] != expected:
            raise ValueError(
                f"units[{key!r}] 与契约不一致：{units[key]!r} != {expected!r}（单位不得混用）"
            )
    if len(set(UNIT_METADATA.values())) != len(UNIT_METADATA):
        raise ValueError("units 中存在重复量纲：违规计数/排放质量/货币必须互不相同")
    return dict(units)


class RolloutBuffer:
    """内存 rollout 容器；观测维度由 buffer 内既有记录锁定。"""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []
        self._obs_dim: int | None = None

    def add(self, transition: Transition) -> None:
        """校验并**防御性复制**后追加。"""
        if transition.contract_version != CONTRACT_VERSION:
            raise ValueError(
                f"contract_version 不匹配：{transition.contract_version} != {CONTRACT_VERSION}"
            )
        obs = _as_float_array(transition.observation, "observation")
        nobs = _as_float_array(transition.next_observation, "next_observation")
        if obs.shape != nobs.shape:
            raise ValueError(
                f"observation/next_observation 维度不一致：{obs.shape} != {nobs.shape}"
            )
        if self._obs_dim is None:
            self._obs_dim = int(obs.shape[0])
        elif obs.shape[0] != self._obs_dim:
            raise ValueError(
                "observation 维度与 buffer 内既有记录不一致："
                f"{obs.shape[0]} != {self._obs_dim}"
            )
        raw = _as_float_array(transition.raw_action, "raw_action", ACTION_DIM)
        exec_a = _as_float_array(transition.exec_action, "exec_action", ACTION_DIM)
        logp = _finite_scalar(transition.old_raw_log_prob, "old_raw_log_prob")
        reward = _finite_scalar(transition.reward, "reward")
        business = _finite_scalar(transition.business_cost, "business_cost")
        carbon = _finite_scalar(transition.carbon_cost, "carbon_cost")
        electricity = _finite_scalar(transition.electricity_cost_sgd, "electricity_cost_sgd")
        terminated = _as_bool(transition.terminated, "terminated")
        truncated = _as_bool(transition.truncated, "truncated")
        info = _sanitize_correction_info(transition.correction_info)

        self.transitions.append(
            Transition(
                observation=obs,
                next_observation=nobs,
                raw_action=raw,
                old_raw_log_prob=logp,
                exec_action=exec_a,
                reward=reward,
                business_cost=business,
                carbon_cost=carbon,
                electricity_cost_sgd=electricity,
                terminated=terminated,
                truncated=truncated,
                correction_info=info,
                contract_version=CONTRACT_VERSION,
            )
        )

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()
        self._obs_dim = None

    def to_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "action_dim": ACTION_DIM,
            "units": dict(UNIT_METADATA),
            "transitions": [
                {
                    "observation": t.observation.tolist(),
                    "next_observation": t.next_observation.tolist(),
                    "raw_action": t.raw_action.tolist(),
                    "old_raw_log_prob": t.old_raw_log_prob,
                    "exec_action": t.exec_action.tolist(),
                    "reward": t.reward,
                    "business_cost": t.business_cost,
                    "carbon_cost": t.carbon_cost,
                    "electricity_cost_sgd": t.electricity_cost_sgd,
                    "terminated": bool(t.terminated),
                    "truncated": bool(t.truncated),
                    "correction_info": copy.deepcopy(t.correction_info),
                    "contract_version": t.contract_version,
                }
                for t in self.transitions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RolloutBuffer:
        """严格反序列化：缺失字段、版本不一致或旧 payload 一律拒绝（不补默认值）。"""
        if not isinstance(data, dict):
            raise TypeError(f"payload 必须为 dict，got {type(data).__name__}")
        if "contract_version" not in data:
            raise ValueError("payload 缺少顶层字段 'contract_version'（不接受无版本 payload）")
        if data["contract_version"] != CONTRACT_VERSION:
            raise ValueError(
                "contract_version 与当前契约不匹配（旧版本 rollout 不得静默读取）："
                f"{data['contract_version']!r} != {CONTRACT_VERSION!r}"
            )
        for key in _REQUIRED_TOP_FIELDS:
            if key not in data:
                raise ValueError(f"payload 缺少顶层字段 {key!r}")
        declared_dim = data["action_dim"]
        if isinstance(declared_dim, bool) or not isinstance(declared_dim, int):
            raise TypeError(
                f"action_dim 必须为 int，got {type(declared_dim).__name__}={declared_dim!r}"
            )
        if declared_dim != ACTION_DIM:
            raise ValueError(f"action_dim 必须为 {ACTION_DIM}，got {declared_dim}")
        _validate_units(data["units"])

        entries = data["transitions"]
        if not isinstance(entries, list):
            raise TypeError(f"transitions 必须为 list，got {type(entries).__name__}")

        buffer = cls()
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError(f"transition 必须为 dict，got {type(entry).__name__}")
            for key in _REQUIRED_TRANSITION_FIELDS:
                if key not in entry:
                    raise ValueError(f"transition 缺少字段 {key!r}（不补默认值）")
            if entry["contract_version"] != CONTRACT_VERSION:
                raise ValueError(
                    "transition contract_version 与顶层不一致："
                    f"{entry['contract_version']} != {CONTRACT_VERSION}"
                )
            buffer.add(
                Transition(
                    observation=np.asarray(entry["observation"], dtype=np.float64),
                    next_observation=np.asarray(entry["next_observation"], dtype=np.float64),
                    raw_action=np.asarray(entry["raw_action"], dtype=np.float64),
                    old_raw_log_prob=entry["old_raw_log_prob"],
                    exec_action=np.asarray(entry["exec_action"], dtype=np.float64),
                    reward=entry["reward"],
                    business_cost=entry["business_cost"],
                    carbon_cost=entry["carbon_cost"],
                    electricity_cost_sgd=entry["electricity_cost_sgd"],
                    terminated=entry["terminated"],
                    truncated=entry["truncated"],
                    correction_info=entry["correction_info"],
                    contract_version=entry["contract_version"],
                )
            )
        return buffer
