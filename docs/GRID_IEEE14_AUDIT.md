# IEEE-14 电网字段与容量核查（M8.1）

> 角色：Agent，只读审计。不改 `grid_model/`。生成日期：2026-09-13。

## 1. 结论声明

本仓库 `grid_model/` 使用的网络是 **pandapower 标准测试网络 `case14`（IEEE 14-bus 测试系统）**，**不是新加坡真实电网**。它仅用于 M8.2 的离线 AC/OPF 验证，绝不把 OPF 指标塞回主训练 reward。

## 2. 网络字段与单位

- 基准功率 `sn_mva = 100` MVA。
- 母线 `net.bus`（14 条）：`vn_kv`（标称电压 kV）、`max_vm_pu`/`min_vm_pu`（电压上下限 p.u.）、`type`、`zone`、`in_service`。
- 负荷 `net.load`（总 259.0 MW）：`p_mw`（有功 MW）、`q_mvar`（无功 MVAr）、`const_z_p_percent` 等（ZIP 比例 %）、`sn_mva`、`scaling`、`in_service`。
- 发电 `net.gen`（总 40.0 MW）：`p_mw`（有功 MW）、`vm_pu`（电压设定 p.u.）、`min_q_mvar`/`max_q_mvar`（无功限 MVAr）、`max_p_mw`/`min_p_mw`。
- 外部电网 `net.ext_grid`（slack）：`vm_pu`（1.0）、`va_degree`（相角）、`max_p_mw`/`min_p_mw`。
- 线路 `net.line`（15 条）：`r_ohm_per_km`（Ω/km）、`x_ohm_per_km`（Ω/km）、`c_nf_per_km`（nF/km）、`g_us_per_km`（µS/km）、`max_i_ka`（载流 kA）、`length_km`（km）。
- 变压器 `net.trafo`（5 台）：`sn_mva`（容量 MVA）、`vn_hv_kv`/`vn_lv_kv`（高/低压 kV）、`vk_percent`/`vkr_percent`（短路电压 %）、`pfe_kw`（铁损 kW）、`i0_percent`（空载电流 %）。

## 3. IDC 等效接入点与容量

- IDC 作为等效负荷，通过 `idc_bus_id`（pandapower bus index，0–13 对应 IEEE 母线 1–14）接入任一 IEEE-14 母线（`grid_model/opf_solver.py::_apply_idc_load`）。
- 仓库敏感性扫描（`data/grid_node_sensitivity/ieee14_node_sensitivity_static.csv`）以 `idc_load_mw = 1.0`、`delta_p_mw = 0.1` 的扰动扫过全部 14 条母线，记录 DC/AC LMP、MEF、电压/线路越限与诊断。
- IDC 设施本身为单数据中心 20 服务器组模型（名义 25 MW 量级），主仿真中通过 `access_limit_kw`/`grid_power_limit_kW` 与电网解耦；离线验证时才把 IDC 负荷接入 IEEE-14。

## 4. 禁止项

- 不把 OPF/LMP/MEF 指标塞回主训练 reward；不把 IEEE-14 表述为新加坡真实网络。
