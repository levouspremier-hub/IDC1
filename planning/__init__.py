"""Rolling-window joint feasibility planner (M4).

Model (per-group capacity, task execution-rate + deadline completion, storage
with charge/discharge mutual exclusion, renewable + no-sell-back + access
limit), solver (HiGHS via scipy.optimize.milp), and the M4.3 scale/timing probe.
"""

from planning.corrector import RollingCorrector
from planning.model import PlanningProblem, TaskPlan, build_milp
from planning.solver import SolveOutcome, solve

__all__ = [
    "PlanningProblem",
    "RollingCorrector",
    "SolveOutcome",
    "TaskPlan",
    "build_milp",
    "solve",
]
