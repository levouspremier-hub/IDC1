# 执行链与数据审计（M1.1）

> 角色：Agent，只读审计。逐项以文件/行号标注现有执行链，每项含「现状 / 与首版差距 / 下游任务卡」。
> 生成日期：2026-09-12。基线：`e8c6087`。分支：`p1-data-contracts`。
> 依据：`docs/IMPLEMENTATION_PLAN.md` §2、§4。

## 0. 总览

| 链路元素 | 现状（一句话） | 下游任务卡 |
|---|---|---|
| 动作 | 23 维（20 强度 + urgent + continuity + BESS） | M3.9（收敛 21 维） |
| 任务执行 | 逐组计划负载被 `np.sum` 压成标量容量 | M3.1/M3.2（A 矩阵分配） |
| 功耗 | 按完成量比例回分 + α 预留损耗 | M3.3/M3.4 |
| 风电/光伏 | 仅 PV 抵消净负荷；风电只写入 info 不入平衡 | M3.5 |
| SOC | 每日 reset，充放按 SOC 硬限幅 | M3.6（跨日）、M3.7 |
| 接入 | `max(P_net, 0)` 硬钳位无反送电 | M3.7（接入上限与计费分离） |
| 预测 | 仅任务到达有预测；价格/PV/风用真值 | M1.3（可见预测）、M3.10 |
| 奖励 | 20 项和，成本/碳/队列/削峰等 | M3 不改 reward 掩盖物理 |
| 归一化 | `config_ultimate.py` + env 硬编码 ref | M6.1（冻结 refs） |
| 数据 | CSV 或合成兜底；风默认全零 | M1.2（下载核验） |
| checkpoint | MARL 正式管理器；单智能体用 SB3 | M2.3（版本化新主链） |

## 1. 动作空间

- **现状**：`envs/idc_price_env.py:243-245` `action_dim = server_action_dim(N) + extra_action_dim(3) = 23`；`:246-250` `Box(0,1)`。`:559-562` 解析 `action[0:20]` 服务器组强度、`[20]` urgent、`[21]` continuity、`[22]` BESS；`:563` `bess_raw_action = 2*bess_action-1`。`:552-557` 维度不符抛 `ValueError`。
- **差距**：首版要求 21 维（20 compute + 1 signed storage），urgent/continuity 由任务约束替代。
- **下游**：M3.9。

## 2. 任务执行与分配

- **现状**：`:566` `planned_task_loads = server_action * max_task_load_per_server`；**:570 标量塌缩** `planned_capacity = float(np.sum(planned_task_loads * C_server))`；`:592-597` 以标量 `available_capacity` 传入 `_execute_tasks_action_guided`（`:1124-1130` 签名）；`:1146` `remaining_capacity = max(available_capacity, 0)` 逐任务扣减（`:1179-1201`，`idc_model/task.py:59-102` `Task.execute`）。
- **差距**：首版要求任务×组矩阵 `A[i,g]`，逐组不超容量、逐任务不超 max rate/remaining。
- **下游**：M3.1（传逐组 `planned_capacity_vec`）、M3.2（`idc_model/allocation.py`）。

## 3. 功耗链

- **现状**：`idc_model/power_model.py:106-131` `calc_it_power`（`P_idle + (P_max-P_idle)*(2L-L**k)`）；`:151-167` `calc_pue_and_total_power`（COP/PUE/P_IDC）。env `:610-620` 用 `actual_total_loads` 调功耗；`:1286-1320` `_actual_loads_from_completed_work` 按 `usage_ratio` 比例回分，加 `planned_load_reserve_alpha`（α）预留损耗（`:1316-1318`）。
- **差距**：首版要求实际功耗由 A 矩阵导出，废除比例回分；α 仅作隔离诊断不入正式链。
- **下游**：M3.3、M3.4。

## 4. 风电与光伏能量流

- **现状**：`:576-577` 读 `pv_now`/`wt_now`；`:687-693` 仅 PV 抵消本地净负荷（`P_grid_kW = max(P_bus_net_kW, 0)`）。**风电未接入能量平衡**：`wt_now` 仅 `:896` 写入 `info["WT"]`，未参与任何 `P_*`/能量方程；`:300` `wt_t` 默认 `np.zeros`。
- **差距**：首版要求风/PV/储能各工况守恒，风进入无反送电平衡并记录 available/used/curtailed。
- **下游**：M3.5。

## 5. BESS / SOC

- **现状**：`:465-466` reset 将 `bess_soc` 回 `bess_soc_init`；`:626-659` 充放按 SOC 上限硬限幅（`:642-647` SOC 限功率）；`:850-851` 步末更新 `bess_soc`/`bess_energy_kWh`（`:671` `bess_soc_next`，`:665-670` 能量上下限 clip）。
- **差距**：首版要求跨日不重置 SOC、统一尾段结算。
- **下游**：M3.6、M3.7。

## 6. 电网接入与无反送电

- **现状**：`:693` `P_grid_kW = max(P_bus_net_kW, 0.0)` 硬钳位无反送电；`:685-686` 注释明确首版不反送；`:297-299` `allow_pv_export=True` 直接 `ValueError`。
- **差距**：首版要求 `access_limit_kw` 为不可违反物理上限，与峰值计费阈值分离，执行前检查。
- **下游**：M3.7。

## 7. 预测

- **现状**：仅任务到达有预测（`idc_model/task_forecast.py:41-65` `generate_task_arrival_forecast`，perfect/noisy/none）；env `:506-511` 生成 `task_arrival_forecast`。**价格/PV/风无独立预测**：`_get_forecast_features`（`:1634-1677`）直接使用 `price_t`/`pv_t` 真值数组，风无预测。
- **差距**：首版要求 `build_scenario` 只返回可见预测，改变未来真值不改变当前输入。
- **下游**：M1.3、M3.10。

## 8. 奖励

- **现状**：`:731-763` 归一化量；`:767-803` 逐项 reward（`r_done/r_finished_task/r_priority_finish` +、`r_cost/r_carbon/r_queue/r_overflow/r_urgent/r_waiting/r_deadline/r_sla/r_unused/r_peak/r_pause/r_resume/r_non_interruptible/r_load_smooth/r_action_smooth/r_bess_degradation/r_bess_invalid` −）；`:807-828` 求和；`:834-844` 终止项 `r_final_queue`/`r_soc_final`。权重 `:213-235`，默认 `configs/config_ultimate.py:98-123`。
- **差距**：M3 物理改造不得以 reward 变化掩盖；M5 三套独立 value/GAE（收益/业务/碳）。
- **下游**：M3（禁改 reward）、M5.2。

## 9. 归一化参考值

- **现状**：`configs/config_ultimate.py:24-65` 硬编码 ref（`:35 price_ref=1.50`、`:37 queue_ref=6000`、`:39 cost_ref=60`、`:40 carbon_ref=15`、`:44-46 peak/grid_limit`、`:64 planned_load_reserve_alpha=0.40`）；env 构造 `:164-210` 缩放。
- **差距**：首版要求 refs 只来自训练集或预定物理尺度，冻结后共享，不得按测试日重算。
- **下游**：M6.1（`freeze_refs.py`）。

## 10. 数据源

- **现状**：`data_io/data_loader.py` CSV 加载器（`:10-51`）+ 合成兜底；env `:280-291` 价格/碳/温度默认合成曲线，`:291` `pv_t` 默认 `np.zeros`，`:300` `wt_t` 默认 `np.zeros`；`configs/config_ultimate.py:80-96` `DATA_CONFIG` 各 `*_csv_path=None`。仓库仅含 `USEP_May-2026.csv`（1 个月）+ NEMS 负荷。
- **差距**：首版要求连续一年价格/负荷对齐 + 可再生数据，无默认曲线回退（正式模式）。
- **下游**：M1.2（下载核验冻结）。

## 11. checkpoint

- **现状**：正式 MARL 管理器 `marl/checkpointing/training_checkpoint.py:227 TrainingCheckpointManager`（`save/load`，schema version `:22`），仅由 `train/train_harl_mappo_short.py:995-1005` 调用；单智能体 `train/train_ppo_ultimate.py:817-821,851` 用 SB3 `CheckpointCallback`/`model.save`（无版本契约）。
- **差距**：首版要求新主链 checkpoint 含 `contract_version_id`、动作/观测 schema hash、代码 revision；无版本/23 维/schema 不符均明确失败。
- **下游**：M2.3（新 `checkpointing/` 或 `safe_rl_v2` loader，不改 `marl/checkpointing/`）。

## 12. 审计复核项（M1.1 验收）

- ✅ `action_dim=23`：`envs/idc_price_env.py:243-245`
- ✅ 风电未接入能量平衡：`wt_now` 仅 `:896` 入 info，`:300` 默认全零
- ✅ 每日 reset：`:465-466`（SOC）、`:490-494`（任务重建）
- ✅ 默认合成数据：`:280-291`、`:300`，`data_io/data_loader.py`
- ✅ 硬编码参考值：`configs/config_ultimate.py:24-65`、env `:164-210`
