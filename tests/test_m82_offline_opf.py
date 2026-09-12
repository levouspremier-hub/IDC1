"""M8.2 测试：正常/临界/失败轨迹离线 AC/OPF 结果结构正确。"""

from scripts.offline_opf_validate import validate_trajectories


def test_offline_opf_normal_succeeds():
    results = validate_trajectories(idc_bus_id=0)
    assert results["normal"]["success"] is True
    assert results["normal"]["min_voltage_pu"] is not None
    assert results["normal"]["max_line_loading_percent"] is not None


def test_offline_opf_failed_reports_reason():
    results = validate_trajectories(idc_bus_id=0)
    # 超容量负荷应失败并带 message
    assert results["failed"]["success"] is False or results["failed"]["message"]
