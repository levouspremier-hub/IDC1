"""M5.3 多约束乘子：每种约束有独立预算/估计/乘子/更新日志。

不共享一个无标签乘子；状态往返保留全部约束状态（可嵌入 VersionedCheckpoint.state）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConstraintState:
    budget: float
    estimate: float = 0.0
    multiplier: float = 0.0
    log: list[float] = field(default_factory=list)


class Lagrangian:
    def __init__(self, budgets: dict[str, float], lr: float = 0.01) -> None:
        self.lr = float(lr)
        self.constraints: dict[str, ConstraintState] = {
            name: ConstraintState(budget=float(b)) for name, b in budgets.items()
        }

    def update(self, costs: dict[str, float]) -> None:
        """用每约束成本更新估计与乘子（乘子非负）。"""
        for name, cost in costs.items():
            c = self.constraints[name]
            c.estimate = float(cost)
            c.multiplier = max(0.0, c.multiplier + self.lr * (c.estimate - c.budget))
            c.log.append(float(c.multiplier))

    def multipliers(self) -> dict[str, float]:
        return {name: c.multiplier for name, c in self.constraints.items()}

    def state_dict(self) -> dict:
        return {
            name: {
                "budget": c.budget,
                "estimate": c.estimate,
                "multiplier": c.multiplier,
                "log": list(c.log),
            }
            for name, c in self.constraints.items()
        }

    def load_state_dict(self, state: dict) -> None:
        for name, c in state.items():
            self.constraints[name] = ConstraintState(
                budget=c["budget"],
                estimate=c["estimate"],
                multiplier=c["multiplier"],
                log=list(c["log"]),
            )
