"""M5.3a–e 多约束乘子状态：每约束独立、带单位、可验证、可恢复。

本模块是**纯状态机**：`update()` 只消费每 transition 的量并维护乘子。
乘子由 `safe_rl_v2.train` 在 actor objective 中以
`A_reward − λ_business·A_business − λ_carbon·A_carbon` 消费（M5.3b）。

单位（每 transition）：
- `business` → `violation_task_steps`（每步活跃逾期 SLA 违规计数，见
  `envs/idc_price_env.py::_compute_sla_metrics`；**不是唯一违约任务数**，
  同一任务在持续逾期的每一步都计 1，故 rollout 内累计为「违规任务·步」）；
- `carbon`   → `kgCO2e`（每步电网购电的排放质量）。
**电费（SGD）不是约束**，既不能作为新约束加入，也不能顶替上述任一单位。
两者在物理上**非负**：负信号必须被明确拒绝，**不得**裁剪为 0。

聚合口径固定为**每 transition mean**，由本模块内部计算：
调用方只能传入逐 transition 的序列，**不能**绕过聚合直接塞标量。

持久化 schema（`state_dict`）包含版本、聚合口径、全局更新序号与每约束的
完整定义与历史，`load_state_dict` 对**全部字段**做严格校验，且拒绝是原子的：
缺字段、单位不符、约束集合不符、版本不符（含旧 `contract-v6`）、
状态内部不自洽（M5.3c）、`estimate` 为负（M5.3e）一律显式拒绝。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from contracts import CONTRACT_VERSION_ID

CONTRACT_VERSION = CONTRACT_VERSION_ID  # 唯一版本源（M3.10c）

UNIT_VIOLATION_TASK_STEPS = "violation_task_steps"
UNIT_KG_CO2E = "kgCO2e"

AGGREGATION_PER_TRANSITION_MEAN = "per_transition_mean"

REQUIRED_CONSTRAINTS = ("business", "carbon")
REQUIRED_UNITS: dict[str, str] = {
    "business": UNIT_VIOLATION_TASK_STEPS,
    "carbon": UNIT_KG_CO2E,
}

# state_dict 中每个约束条目必须出现的字段（缺任一即拒绝）
_CONSTRAINT_FIELDS = (
    "name",
    "budget",
    "unit",
    "learning_rate",
    "max_multiplier",
    "estimate",
    "multiplier",
    "updates",
    "log",
)
_REQUIRED_TOP_FIELDS = ("contract_version", "aggregation", "updates", "constraints")


def _as_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} 必须为实数，got {type(value).__name__}={value!r}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} 必须有限，got {out!r}")
    return out


def _as_positive_float(value: Any, name: str) -> float:
    out = _as_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} 必须为正数，got {out!r}")
    return out


def _as_non_negative_float(value: Any, name: str) -> float:
    out = _as_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} 必须非负，got {out!r}")
    return out


def _as_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须为整数，got {type(value).__name__}={value!r}")
    if value < 0:
        raise ValueError(f"{name} 必须非负，got {value!r}")
    return int(value)


def validate_constraint_signals(
    batch_signals: Mapping[str, ArrayLike], expected: Iterable[str]
) -> dict[str, float]:
    """**纯**预检：校验每约束的逐 transition 信号，并返回 per-transition mean。

    本函数不接触任何实例状态，因此可安全地用作训练轮次的前置检查
    （见 M5.3f）：`update()` 与训练预检**共用同一份规则**。

    对每个约束要求：序列（非标量）、一维、非空、全有限、**全非负**。
    负值必须被明确拒绝而**不得**裁剪为 0：`business` 的单位是
    `violation_task_steps`（违规任务·步计数）、`carbon` 的单位是 `kgCO2e`
    （排放质量），两者物理上不可能为负；裁剪会把上游缺陷掩盖成被改写的数据，
    并让乘子按错误的量更新。
    """
    expected_names = sorted(expected)
    if not isinstance(batch_signals, Mapping):
        raise TypeError(
            f"batch_signals 必须为 Mapping（每约束一个逐 transition 序列），"
            f"got {type(batch_signals).__name__}"
        )
    if set(batch_signals) != set(expected_names):
        raise ValueError(
            f"batch_signals 的 constraints 集合必须恰为 {expected_names}，"
            f"got {sorted(batch_signals)}"
        )

    estimates: dict[str, float] = {}
    for name in expected_names:
        raw = batch_signals[name]
        if isinstance(raw, (str, bytes)) or np.isscalar(raw):
            raise TypeError(
                f"batch_signals[{name!r}] 必须是逐 transition 的序列，"
                "不得直接传标量（聚合口径固定为 per-transition mean）"
            )
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"batch_signals[{name!r}] 必须为一维，got ndim={values.ndim}")
        if values.size == 0:
            raise ValueError(f"batch_signals[{name!r}] 不得为空")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"batch_signals[{name!r}] 含非有限数值")
        # 逐 transition 校验非负（不能只看均值：[-1, 1] 的均值恰为 0）
        negative = np.flatnonzero(values < 0.0)
        if negative.size:
            index = int(negative[0])
            raise ValueError(
                f"batch_signals[{name!r}] 含负值："
                f"第 {index} 个 transition 为 {float(values[index])!r}，"
                f"但约束 {name!r} 的物理域非负（不得裁剪为 0）"
            )
        estimates[name] = float(np.mean(values))
    return estimates


@dataclass(frozen=True)
class ConstraintSpec:
    """单个约束的显式定义：五个字段缺一不可。"""

    name: str
    budget: float
    unit: str
    learning_rate: float
    max_multiplier: float

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_CONSTRAINTS:
            raise ValueError(
                f"未知 constraint {self.name!r}；必须恰为 {list(REQUIRED_CONSTRAINTS)}"
                "（电费不是约束）"
            )
        expected_unit = REQUIRED_UNITS[self.name]
        if self.unit != expected_unit:
            raise ValueError(
                f"constraint {self.name!r} 的 unit 必须为 {expected_unit!r}，got {self.unit!r}"
            )
        object.__setattr__(self, "budget", _as_non_negative_float(self.budget, "budget"))
        object.__setattr__(
            self, "learning_rate", _as_positive_float(self.learning_rate, "learning_rate")
        )
        object.__setattr__(
            self, "max_multiplier", _as_positive_float(self.max_multiplier, "max_multiplier")
        )


@dataclass
class ConstraintState:
    """单约束的运行状态（含定义副本与历史，供持久化与审阅）。"""

    name: str
    budget: float
    unit: str
    learning_rate: float
    max_multiplier: float
    estimate: float = 0.0
    multiplier: float = 0.0
    updates: int = 0
    log: list[float] = field(default_factory=list)


class Lagrangian:
    """多约束乘子状态机；约束之间**完全独立**。"""

    def __init__(
        self,
        specs: Sequence[ConstraintSpec],
        *,
        aggregation: str = AGGREGATION_PER_TRANSITION_MEAN,
    ) -> None:
        if aggregation != AGGREGATION_PER_TRANSITION_MEAN:
            raise ValueError(
                f"aggregation 必须为 {AGGREGATION_PER_TRANSITION_MEAN!r}，got {aggregation!r}"
            )
        if isinstance(specs, (str, bytes)) or not isinstance(specs, Sequence):
            raise TypeError(f"specs 必须为 ConstraintSpec 序列，got {type(specs).__name__}")
        if not specs:
            raise ValueError("specs 不得为空")

        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"constraint 名不得重复：{names}")
        if set(names) != set(REQUIRED_CONSTRAINTS):
            raise ValueError(
                f"constraints 集合必须恰为 {list(REQUIRED_CONSTRAINTS)}，got {sorted(names)}"
            )

        self.aggregation = aggregation
        self._updates = 0
        self.constraints: dict[str, ConstraintState] = {
            spec.name: ConstraintState(
                name=spec.name,
                budget=spec.budget,
                unit=spec.unit,
                learning_rate=spec.learning_rate,
                max_multiplier=spec.max_multiplier,
            )
            for spec in specs
        }

    # --- 更新 ---------------------------------------------------------------

    def update(self, batch_signals: Mapping[str, ArrayLike]) -> dict[str, float]:
        """按**每 transition mean** 聚合 `batch_signals` 并更新各约束乘子。

        返回更新后的乘子。传入标量、空序列、非有限值或键集不符一律报错。
        """
        # 校验全部完成后才进入下面的写入循环，故失败是原子的
        estimates = validate_constraint_signals(batch_signals, self.constraints)

        updated: dict[str, float] = {}
        for name in sorted(self.constraints):
            state = self.constraints[name]
            candidate = state.multiplier + state.learning_rate * (
                estimates[name] - state.budget
            )
            state.estimate = estimates[name]
            state.multiplier = float(min(max(candidate, 0.0), state.max_multiplier))
            state.updates += 1
            state.log.append(state.multiplier)
            updated[name] = state.multiplier

        self._updates += 1
        return updated

    def multipliers(self) -> dict[str, float]:
        return {name: state.multiplier for name, state in self.constraints.items()}

    # --- 持久化 -------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "aggregation": self.aggregation,
            "updates": self._updates,
            "constraints": {
                name: {
                    "name": state.name,
                    "budget": state.budget,
                    "unit": state.unit,
                    "learning_rate": state.learning_rate,
                    "max_multiplier": state.max_multiplier,
                    "estimate": state.estimate,
                    "multiplier": state.multiplier,
                    "updates": state.updates,
                    "log": list(state.log),
                }
                for name, state in self.constraints.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """严格反序列化；任何不符都在**写入前**拒绝（拒绝是原子的）。"""
        if not isinstance(state, dict):
            raise TypeError(f"state 必须为 dict，got {type(state).__name__}")
        if "contract_version" not in state:
            raise ValueError("state 缺少顶层字段 'contract_version'（不接受无版本状态）")
        if state["contract_version"] != CONTRACT_VERSION:
            raise ValueError(
                "contract_version 与当前契约不匹配（旧版本乘子状态不得静默读取）："
                f"{state['contract_version']!r} != {CONTRACT_VERSION!r}"
            )
        for key in _REQUIRED_TOP_FIELDS:
            if key not in state:
                raise ValueError(f"state 缺少顶层字段 {key!r}")
        if state["aggregation"] != self.aggregation:
            raise ValueError(
                f"state aggregation 与当前实例不一致："
                f"{state['aggregation']!r} != {self.aggregation!r}"
            )
        outer_updates = _as_non_negative_int(state["updates"], "updates")

        entries = state["constraints"]
        if not isinstance(entries, dict):
            raise TypeError(f"state['constraints'] 必须为 dict，got {type(entries).__name__}")
        if set(entries) != set(self.constraints):
            raise ValueError(
                "state constraints 集合与当前实例不一致："
                f"{sorted(entries)} != {sorted(self.constraints)}"
            )

        # 先在临时结构里完成全部校验，再原子写入
        rebuilt: dict[str, ConstraintState] = {}
        for name in sorted(self.constraints):
            entry = entries[name]
            if not isinstance(entry, dict):
                raise TypeError(f"constraints[{name!r}] 必须为 dict")
            for key in _CONSTRAINT_FIELDS:
                if key not in entry:
                    raise ValueError(f"constraints[{name!r}] 缺少字段 {key!r}")

            current = self.constraints[name]
            if entry["name"] != name:
                raise ValueError(f"constraints[{name!r}].name 不符：{entry['name']!r}")
            if entry["unit"] != current.unit:
                raise ValueError(
                    f"constraints[{name!r}] 的 unit 不符：{entry['unit']!r} != {current.unit!r}"
                )
            budget = _as_finite_float(entry["budget"], f"constraints[{name!r}].budget")
            if budget != current.budget:
                raise ValueError(
                    f"constraints[{name!r}] budget 与当前 spec 不符："
                    f"{budget!r} != {current.budget!r}"
                )
            learning_rate = _as_finite_float(
                entry["learning_rate"], f"constraints[{name!r}].learning_rate"
            )
            if learning_rate != current.learning_rate:
                raise ValueError(
                    f"constraints[{name!r}] learning_rate 与当前 spec 不符："
                    f"{learning_rate!r} != {current.learning_rate!r}"
                )
            max_multiplier = _as_finite_float(
                entry["max_multiplier"], f"constraints[{name!r}].max_multiplier"
            )
            if max_multiplier != current.max_multiplier:
                raise ValueError(
                    f"constraints[{name!r}] max_multiplier 与当前 spec 不符："
                    f"{max_multiplier!r} != {current.max_multiplier!r}"
                )

            estimate = _as_finite_float(entry["estimate"], f"constraints[{name!r}].estimate")
            # 物理域：estimate 是每 transition mean，单位决定其不可能为负
            if estimate < 0.0:
                raise ValueError(
                    f"constraints[{name!r}].estimate 必须非负"
                    f"（{current.unit} 的物理域非负），got {estimate!r}"
                )
            multiplier = _as_finite_float(
                entry["multiplier"], f"constraints[{name!r}].multiplier"
            )
            if multiplier < 0.0 or multiplier > max_multiplier:
                raise ValueError(
                    f"constraints[{name!r}] multiplier 越界："
                    f"{multiplier!r} 不在 [0, {max_multiplier!r}]"
                )
            entry_updates = _as_non_negative_int(
                entry["updates"], f"constraints[{name!r}].updates"
            )
            log = entry["log"]
            if not isinstance(log, list):
                raise TypeError(f"constraints[{name!r}].log 必须为 list")
            values = [_as_finite_float(v, f"constraints[{name!r}].log") for v in log]
            if len(values) != entry_updates:
                raise ValueError(
                    f"constraints[{name!r}].log 长度必须等于 updates "
                    f"({entry_updates})，got {len(values)}"
                )

            # --- M5.3c：状态内部自洽性（update() 恒等式；拒绝不可能的历史）---
            # R3：逐约束 updates 必须与顶层严格同步（两者在同一次 update() 中各加 1）
            if entry_updates != outer_updates:
                raise ValueError(
                    f"constraints[{name!r}].updates 必须等于顶层 updates "
                    f"({outer_updates})，got {entry_updates}"
                )
            # R1：log 的每个值都必须有限且落在 [0, max_multiplier]
            #     （update() 写入 log 的就是被 clamp 过的 multiplier）
            for index, value in enumerate(values):
                if value < 0.0 or value > max_multiplier:
                    raise ValueError(
                        f"constraints[{name!r}].log[{index}] 越界："
                        f"{value!r} 不在 [0, {max_multiplier!r}]"
                        "（update() 只会写入被截断到该区间的乘子）"
                    )
            # R2：有更新时 log 最后一项必须严格等于当前 multiplier
            if entry_updates > 0 and values[-1] != multiplier:
                raise ValueError(
                    f"constraints[{name!r}].log 最后一项必须等于当前 multiplier："
                    f"{values[-1]!r} != {multiplier!r}"
                )
            # R4：零更新状态必须与构造后的零状态逐位一致
            if outer_updates == 0:
                if multiplier != 0.0:
                    raise ValueError(
                        f"constraints[{name!r}].multiplier 在零更新状态下必须为 0，"
                        f"got {multiplier!r}"
                    )
                if estimate != 0.0:
                    raise ValueError(
                        f"constraints[{name!r}].estimate 在零更新状态下必须为 0，"
                        f"got {estimate!r}"
                    )
                if values:
                    raise ValueError(
                        f"constraints[{name!r}].log 在零更新状态下必须为空，"
                        f"got {len(values)} 项"
                    )

            rebuilt[name] = ConstraintState(
                name=name,
                budget=budget,
                unit=entry["unit"],
                learning_rate=learning_rate,
                max_multiplier=max_multiplier,
                estimate=estimate,
                multiplier=multiplier,
                updates=entry_updates,
                log=values,
            )

        self.constraints = rebuilt
        self._updates = outer_updates
