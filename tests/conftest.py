"""全局测试夹具与测试标记注册（M0.4）。

标记定义：
- leakage：未来信息泄漏回归
- resume：中断恢复一致性
- slow：慢速测试（`make check` 用 `-m 'not slow'` 排除）
"""


def pytest_configure(config):
    config.addinivalue_line("markers", "leakage: future-information leakage regression")
    config.addinivalue_line("markers", "resume: interruption-recovery consistency")
    config.addinivalue_line("markers", "slow: slow tests excluded by make check")
