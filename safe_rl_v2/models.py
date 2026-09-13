"""M5.2a 三套独立价值：收益 / 业务违规量 / 碳排放量各有独立 value 与**终端感知** GAE。

不把期望约束学习表述为逐步硬安全保证（这是约束 RL 的估计，非硬保证）。

终端语义（三种情形必须区分）：

| 情形 | bootstrap | 递推 |
|---|---|---|
| `terminated` | 0（`values[T]` 不参与） | 中断 |
| `truncated` | 允许用该步 `next_observation` 的 value | 中断 |
| 非终止/截断（rollout 因 steps 上限结束） | 允许 | 正常 |

`terminated` 与 `truncated` 是**必填关键字参数**：本模块不提供任何「无掩码即默认
终止语义」的静默路径（旧的 `compute_gae(costs, values, gamma, lam)` 已移除）。

业务违规量的单位约定（M5.2a 起）：
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


def _as_1d_finite(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} 必须为一维数组，got ndim={arr.ndim} shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} 不得为空数组")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限数值 (non-finite: NaN/Inf)")
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
    values,
    *,
    terminated,
    truncated,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """单头 GAE。返回 `(advantage, target)`，长度均为 `T`。

    `values` 长度为 `T+1`：`values[t]` 是 `obs[t]` 的 critic 估计，
    `values[T]` 是最后一步 `next_observation` 的 bootstrap 估计。
    """
    signal_arr = _as_1d_finite(signal, "signal")
    n_steps = signal_arr.shape[0]
    values_arr = _as_1d_finite(values, "values")
    if values_arr.shape[0] != n_steps + 1:
        raise ValueError(
            f"values 长度必须为 len(signal)+1 = {n_steps + 1}，got {values_arr.shape[0]}"
        )
    terminated_arr = _as_bool_mask(terminated, "terminated", n_steps)
    truncated_arr = _as_bool_mask(truncated, "truncated", n_steps)
    if np.any(terminated_arr & truncated_arr):
        raise ValueError(
            "terminated 与 truncated 不得同时为真（两者语义互斥）："
            f"第 {np.flatnonzero(terminated_arr & truncated_arr).tolist()} 步"
        )
    gamma_f = _check_rate(gamma, "gamma")
    lam_f = _check_rate(lam, "lam")

    # 截断仍允许 bootstrap；终止或截断都中断反向递推
    bootstrap_mask = ~terminated_arr
    carry_mask = ~(terminated_arr | truncated_arr)

    advantages = np.zeros(n_steps, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        delta = (
            signal_arr[t]
            + gamma_f * values_arr[t + 1] * bool(bootstrap_mask[t])
            - values_arr[t]
        )
        last_gae = delta + gamma_f * lam_f * bool(carry_mask[t]) * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values_arr[:n_steps]


def compute_three_value_targets(
    rewards,
    business_violations,
    carbon_emissions,
    values: dict[str, np.ndarray],
    *,
    terminated,
    truncated,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """三套**完全独立**的 target：收益、业务违规量、碳排放量各跑一次 GAE。

    改变任意一条 signal 或其对应 value，不影响另外两条。
    电费（SGD）不是本函数输入，不会进入任何一套 target。
    """
    signals = {
        "reward": _as_1d_finite(rewards, "rewards"),
        "business": _as_1d_finite(business_violations, "business_violations"),
        "carbon": _as_1d_finite(carbon_emissions, "carbon_emissions"),
    }
    if not isinstance(values, dict):
        raise TypeError(f"values 必须为 dict，got {type(values).__name__}")

    sizes = {head: arr.shape[0] for head, arr in signals.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"三套 signal 长度必须一致，got {sizes}")

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for head in SIGNAL_HEADS:
        if head not in values:
            raise ValueError(f"values 缺少 {head!r} 头的 critic 估计")
        # 每头独立调用：不共享中间量，任一头的变化不会传播到其他头
        result[head] = compute_signal_gae(
            signals[head],
            _as_1d_finite(values[head], f"values[{head!r}]"),
            terminated=terminated,
            truncated=truncated,
            gamma=gamma,
            lam=lam,
        )
    return result
