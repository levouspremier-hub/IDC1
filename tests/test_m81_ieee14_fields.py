"""M8.1 测试：IEEE-14 字段/单位/容量核查（只读）。"""

import pandapower.networks as pn


def test_ieee14_structure():
    net = pn.case14()
    assert net.sn_mva == 100.0
    assert len(net.bus) == 14
    assert len(net.line) == 15
    assert len(net.trafo) == 5
    assert net.load["p_mw"].sum() == 259.0
    assert net.gen["p_mw"].sum() == 40.0


def test_ieee14_fields_present():
    net = pn.case14()
    assert "vn_kv" in net.bus.columns
    assert "max_vm_pu" in net.bus.columns
    assert "p_mw" in net.load.columns
    assert "q_mvar" in net.load.columns
    assert "p_mw" in net.gen.columns
    assert "vm_pu" in net.gen.columns
    assert "r_ohm_per_km" in net.line.columns
    assert "x_ohm_per_km" in net.line.columns
    assert "max_i_ka" in net.line.columns
    assert "sn_mva" in net.trafo.columns
    assert "vn_hv_kv" in net.trafo.columns
