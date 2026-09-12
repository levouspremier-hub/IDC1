"""M0.4 测试骨架：验证 pytest 可发现并运行测试、必需标记已注册。"""


def test_skeleton_discovered():
    """骨架自检：本文件能被 pytest 发现并运行。"""
    assert True


def test_required_markers_registered(pytestconfig):
    """验收：marker leakage / resume / slow 已定义。"""
    registered = {m.strip().split(":")[0] for m in pytestconfig.getini("markers")}
    assert {"leakage", "resume", "slow"} <= registered
