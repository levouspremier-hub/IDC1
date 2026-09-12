"""M3.2 任务×组实际分配器：输出矩形非负 A[i,g]（TaskAllocation 契约）。

纯函数，不依赖旧 Task 对象；输入为任务摘要 dict 列表与组容量，输出 `contracts.TaskAllocation`。

调度口径（本卡，待人工确认）：
- 任务按 (priority 降序, deadline 升序, arrival 升序, task_id) 排序后依次分配；
- 每任务最多分配 `min(remaining_work, max_rate)`，按组序 0→n_group-1 贪心填组。
"""

from __future__ import annotations

from contracts.models import TaskAllocation


def allocate_tasks(
    tasks: list[dict],
    group_capacity: list[float],
) -> TaskAllocation:
    """分配任务工作到组，返回矩形非负 A[i,g]。

    约束（M3.2 验收）：
    - A[i,g] >= 0
    - 每组 sum_i A[i,g] <= group_capacity[g]
    - 每任务 sum_g A[i,g] <= min(remaining_work, max_rate)
    - 总完成量 == sum(A)
    """
    n_group = len(group_capacity)
    if n_group == 0:
        raise ValueError("group_capacity 不能为空")

    task_ids = [str(t["task_id"]) for t in tasks]
    group_ids = list(range(n_group))
    matrix = [[0.0] * n_group for _ in range(len(tasks))]
    remaining_capacity = [max(float(c), 0.0) for c in group_capacity]

    order = sorted(
        range(len(tasks)),
        key=lambda i: (
            -float(tasks[i].get("priority", 0.0)),
            int(tasks[i].get("deadline", 0)),
            int(tasks[i].get("arrival", 0)),
            str(tasks[i]["task_id"]),
        ),
    )

    for i in order:
        remaining_work = max(float(tasks[i].get("remaining_work", 0.0)), 0.0)
        max_rate = float(tasks[i].get("max_rate", remaining_work))
        demand = min(remaining_work, max_rate)
        for g in range(n_group):
            if demand <= 1e-9:
                break
            if remaining_capacity[g] <= 1e-6:
                continue
            take = min(demand, remaining_capacity[g])
            matrix[i][g] += take
            remaining_capacity[g] -= take
            demand -= take

    return TaskAllocation(task_ids=task_ids, group_ids=group_ids, matrix=matrix)
