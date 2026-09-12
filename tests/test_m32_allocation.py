"""M3.2 分配器属性测试：无任务/任务不足/任务充足/异构容量/非负/max_rate。"""

import pytest

from idc_model.allocation import allocate_tasks


def _task(tid: str, remaining: float, **kw) -> dict:
    d = {
        "task_id": tid,
        "remaining_work": remaining,
        "max_rate": 1e9,
        "priority": 1.0,
        "deadline": 100,
        "arrival": 0,
    }
    d.update(kw)
    return d


def _col_sum(matrix, g: int) -> float:
    return sum(row[g] for row in matrix)


def test_no_tasks():
    a = allocate_tasks([], [10.0, 10.0])
    assert a.task_ids == []
    assert a.matrix == []
    assert sum(sum(r) for r in a.matrix) == 0.0


def test_insufficient_tasks():
    a = allocate_tasks([_task("t1", 5.0)], [10.0, 10.0])
    assert a.matrix[0][0] == pytest.approx(5.0)
    assert sum(a.matrix[0]) == pytest.approx(5.0)
    assert _col_sum(a.matrix, 0) <= 10.0
    assert _col_sum(a.matrix, 1) <= 10.0


def test_sufficient_tasks():
    tasks = [_task(f"t{i}", 10.0, arrival=i) for i in range(3)]
    a = allocate_tasks(tasks, [10.0, 10.0])
    total = sum(sum(r) for r in a.matrix)
    assert total == pytest.approx(20.0)  # 容量饱和
    for r in a.matrix:
        assert sum(r) <= 10.0 + 1e-9


def test_heterogeneous_capacity():
    tasks = [_task(f"t{i}", 10.0, arrival=i) for i in range(3)]
    a = allocate_tasks(tasks, [5.0, 15.0])
    assert _col_sum(a.matrix, 0) == pytest.approx(5.0)
    assert _col_sum(a.matrix, 1) == pytest.approx(15.0)


def test_max_rate_limits_task():
    a = allocate_tasks([_task("t1", 10.0, max_rate=3.0)], [10.0, 10.0])
    assert sum(a.matrix[0]) == pytest.approx(3.0)


def test_matrix_nonnegative():
    a = allocate_tasks([_task("t1", 2.0)], [1.0, 1.0])
    for r in a.matrix:
        for x in r:
            assert x >= 0.0
