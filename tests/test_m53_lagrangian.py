"""M5.3 测试：独立乘子、状态往返、乘子非负。

M5.3a 迁移：`Lagrangian` 改为显式 `ConstraintSpec` 序列，`update()` 只接受
**每 transition 的序列**（聚合口径由模块内部固定为 mean）。本文件断言未改动、
未删除、未弱化，只把构造点与输入形态迁移到新 API；
`test_multiplier_never_negative` 因新契约要求约束集合恰为 business+carbon，
由单约束改为双约束，但**断言本身不变**。
"""

from safe_rl_v2.lagrangian import (
    UNIT_KG_CO2E,
    UNIT_VIOLATION_TASK_STEPS,
    ConstraintSpec,
    Lagrangian,
)

SPECS = (
    ConstraintSpec(
        name="business", budget=5.0, unit=UNIT_VIOLATION_TASK_STEPS,
        learning_rate=0.1, max_multiplier=100.0,
    ),
    ConstraintSpec(
        name="carbon", budget=10.0, unit=UNIT_KG_CO2E,
        learning_rate=0.1, max_multiplier=100.0,
    ),
)


def test_independent_multipliers():
    lag = Lagrangian(SPECS)
    lag.update({"business": [7.0], "carbon": [3.0]})
    m = lag.multipliers()
    assert m["business"] == 0.2  # (7-5)*0.1
    assert m["carbon"] == 0.0  # (3-10)*0.1 < 0 → 截断为 0


def test_state_roundtrip():
    lag = Lagrangian(SPECS)
    lag.update({"business": [7.0], "carbon": [12.0]})
    lag.update({"business": [6.0], "carbon": [11.0]})

    restored = Lagrangian(SPECS)
    restored.load_state_dict(lag.state_dict())
    assert restored.multipliers() == lag.multipliers()
    assert restored.constraints["business"].log == lag.constraints["business"].log


def test_multiplier_never_negative():
    lag = Lagrangian(SPECS)
    for _ in range(5):
        lag.update({"business": [0.0], "carbon": [0.0]})  # 持续低于预算
    assert lag.multipliers()["business"] == 0.0
