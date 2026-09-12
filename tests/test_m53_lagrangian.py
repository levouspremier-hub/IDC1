"""M5.3 测试：独立乘子、状态往返、乘子非负。"""

from safe_rl_v2.lagrangian import Lagrangian


def test_independent_multipliers():
    lag = Lagrangian({"business": 5.0, "carbon": 10.0}, lr=0.1)
    lag.update({"business": 7.0, "carbon": 3.0})
    m = lag.multipliers()
    assert m["business"] == 0.2  # (7-5)*0.1
    assert m["carbon"] == 0.0  # (3-10)*0.1 < 0 → 截断为 0


def test_state_roundtrip():
    lag = Lagrangian({"business": 5.0, "carbon": 10.0}, lr=0.1)
    lag.update({"business": 7.0, "carbon": 12.0})
    lag.update({"business": 6.0, "carbon": 11.0})

    restored = Lagrangian({"business": 5.0, "carbon": 10.0}, lr=0.1)
    restored.load_state_dict(lag.state_dict())
    assert restored.multipliers() == lag.multipliers()
    assert restored.constraints["business"].log == lag.constraints["business"].log


def test_multiplier_never_negative():
    lag = Lagrangian({"business": 5.0}, lr=0.5)
    for _ in range(5):
        lag.update({"business": 0.0})  # 持续低于预算
    assert lag.multipliers()["business"] == 0.0
