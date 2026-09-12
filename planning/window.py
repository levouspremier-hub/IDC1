"""M4.3 滚动窗口：默认 24h，任务期限超窗时扩到最晚期限。"""

from __future__ import annotations

from contracts.models import TaskState


def compute_window(tasks: list[TaskState], default_horizon: int = 24) -> int:
    """返回窗口长度 = max(default_horizon, 最晚 deadline)。"""
    latest = max((int(task.deadline) for task in tasks), default=0)
    return max(default_horizon, latest)
