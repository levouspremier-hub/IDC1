"""M5.2a / M5.2c 三套独立价值：收益 / 业务违规量 / 碳排放量，各有独立 value 与**终端感知** GAE。

不把期望约束学习表述为逐步硬安全保证（这是约束 RL 的估计，非硬保证）。

**逐 transition 的 next value（M5.2c 勘误）**：每条 transition 自带自己的
`observation` 与 `next_observation`，因此 bootstrap 必须使用**该条 transition 自己的**
`next_values[t]`。早期 API 用长度 `T+1` 的 `values` 并以 `values[t+1]` 作 bootstrap，
仅在「单 episode、`obs[t+1] == next_obs[t]`」时巧合成立；一旦出现中间 `truncated`
或多 episode buffer，`obs[t+1]` 属于**下一条 episode**，语义即错。该 API 已**退役**，
传入长度 `T+1` 的数组会明确报错，不做静默兼容。

递推（反向，t = T-1 … 0）：

    bootstrap[t] = not terminated[t]
    carry[t]     = not terminated[t] and not truncated[t]
    delta[t]     = signal[t] + gamma * bootstrap[t] * next_values[t] - current_values[t]
    gae[t]       = delta[t] + gamma * lam * carry[t] * gae[t+1]
    advantage[t] = gae[t]
    target[t]    = gae[t] + current_values[t]

| 情形 | bootstrap | 递推 |
|---|---|---|
| `terminated` | 0（不使用 `next_values[t]`） | 中断 |
| `truncated` | 用**本条** `next_values[t]` | 中断 |
| 非边界 | 用**本条** `next_values[t]` | 正常 |

`terminated` 与 `truncated` 是**必填关键字参数**：不提供任何「无掩码即默认终止语义」的
静默路径。

业务违规量的单位约定：
    信号来自 `envs/idc_price_env.py::_compute_sla_metrics` 的 `sla_violation_count`，
    语义为**每步活跃逾期 SLA 违规计数**：对每个 `arrival_time < horizon`、状态不在
    `not_arrived/finished/failed`、且 `remaining_work > 1e-6` 的任务，若
    `current_time > latest_finish_time` 则计 1。
    **不是唯一违约任务数**——同一任务在其持续逾期的每一步都会被重新计 1。
    故 rollout 内逐步求和 = 「违规任务·步」(task-steps overdue)，
    与 episode 级「唯一违约任务数」**不等价**，两者不得互相换算或顶替。
    M5.3 的拉格朗日预算必须按**同一单位**设定；碳排放（kgCO2e）与电费（SGD）
    是另外两套独立量纲，电费**不参与**本模块任何 target。
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_signal_gae", "compute_three_value_targets"]

SIGNAL_HEADS = ("reward", "business", "carbon")

# 旧 T+1 `values` API 的报错提示（退役，不静默兼容）
_RETIRED_VALUES_HINT = (
    "（旧 T+1 `values` API 已退役：必须传逐 transition 的 current_values / next_values）"
)


def _as_1d_finite(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须为一维数组，got ndim={arr.ndim} shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} 不得为空数组")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限数值 (non-finite: NaN/Inf)")
    return arr


def _as_length(values, name: str, expected: int) -> np.ndarray:
    """按**长度恰好为 expected** 校验；旧的 T+1 形态给出专门的报错提示。"""
    arr = _as_1d_finite(values, name)
    if arr.shape[0] != expected:
        hint = _RETIRED_VALUES_HINT if arr.shape[0] == expected + 1 else ""
        raise ValueError(f"{name} 长度必须为 {expected}，got {arr.shape[0]}{hint}")
    return arr


def _as_bool_mask(mask, name: str, expected_len: int) -> np.ndarray:
    """只接受真正的布尔数组：整数/浮点 0-1 数组一律拒绝（不得静默转换）。"""
    arr = np.asarray(mask)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须为一维数组，got ndim={arr.ndim} shape={arr.shape}")
    if arr.dtype != np.bool_:
        raise TypeError(f"{name} 的 dtype 必须为 bool，got {arr.dtype}（禁止 0/1 静默转换）")
    if arr.shape[0] != expected_len:
        raise ValueError(f"{name} 长度必须为 {expected_len}，got {arr.shape[0]}")
    return arr


def _check_rate(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} 必须为实数，got {type(value).__name__}={value!r}")
    rate = float(value)
    if not np.isfinite(rate):
        raise ValueError(f"{name} 必须有限，got {rate!r}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} 必须落在 [0, 1]，got {rate!r}")
    return rate


def compute_signal_gae(
    signal,
    current_values,
    next_values,
    *,
    terminated,
    truncated,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """单头 GAE。返回 `(advantage, target)`，长度均为 `T`。

    - `current_values[t]`：第 `t` 条 transition **自身** `observation` 的 critic 估计；
    - `next_values[t]`：第 `t` 条 transition **自身** `next_observation` 的 critic 估计。
    """
    signal_arr = _as_1d_finite(signal, "signal")
    n_steps = signal_arr.shape[0]
    current_arr = _as_length(current_values, "current_values", n_steps)
    next_arr = _as_length(next_values, "next_values", n_steps)
    terminated_arr = _as_bool_mask(terminated, "terminated", n_steps)
    truncated_arr = _as_bool_mask(truncated, "truncated", n_steps)
    if np.any(terminated_arr & truncated_arr):
        raise ValueError(
            "terminated 与 truncated 不得同时为真（两者语义互斥）："
            f"第 {np.flatnonzero(terminated_arr & truncated_arr).tolist()} 步"
        )
    gamma_f = _check_rate(gamma, "gamma")
    lam_f = _check_rate(lam, "lam")

    # 终止步不 bootstrap；终止或截断都中断反向递推
    bootstrap_mask = ~terminated_arr
    carry_mask = ~(terminated_arr | truncated_arr)

    advantages = np.zeros(n_steps, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        delta = (
            signal_arr[t]
            + gamma_f * float(next_arr[t]) * bool(bootstrap_mask[t])
            - current_arr[t]
        )
        last_gae = delta + gamma_f * lam_f * bool(carry_mask[t]) * last_gae
        advantages[t] = last_gae

    return advantages, advantages + current_arr


def compute_three_value_targets(
    rewards,
    business_violations,
    carbon_emissions,
    current_values: dict[str, np.ndarray],
    next_values: dict[str, np.ndarray],
    *,
    terminated,
    truncated,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """三套**完全独立**的 target：收益、业务违规量、碳排放量各跑一次 GAE。

    每头各有自己的 `current_values` 与 `next_values`；改变任意一头，不影响另外两头。
    电费（SGD）不是本函数输入，不会进入任何一套 target。
    """
    signals = {
        "reward": _as_1d_finite(rewards, "rewards"),
        "business": _as_1d_finite(business_violations, "business_violations"),
        "carbon": _as_1d_finite(carbon_emissions, "carbon_emissions"),
    }
    n_steps = signals["reward"].shape[0]
    for head, arr in signals.items():
        if arr.shape[0] != n_steps:
            raise ValueError(
                f"三套 signal 长度必须一致：reward={n_steps} vs {head}={arr.shape[0]}"
            )
    for name, mapping in (("current_values", current_values), ("next_values", next_values)):
        if not isinstance(mapping, dict):
            raise TypeError(f"{name} 必须为 dict，got {type(mapping).__name__}")

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for head in SIGNAL_HEADS:
        if head not in current_values:
            raise ValueError(f"current_values 缺少 {head!r} 头")
        if head not in next_values:
            raise ValueError(f"next_values 缺少 {head!r} 头")
        # 每头独立调用：不共享中间量，任一头的变化不会传播到其他头
        result[head] = compute_signal_gae(
            signals[head],
            _as_length(current_values[head], f"current_values[{head!r}]", n_steps),
            _as_length(next_values[head], f"next_values[{head!r}]", n_steps),
            terminated=terminated,
            truncated=truncated,
            gamma=gamma,
            lam=lam,
        )
    return result
