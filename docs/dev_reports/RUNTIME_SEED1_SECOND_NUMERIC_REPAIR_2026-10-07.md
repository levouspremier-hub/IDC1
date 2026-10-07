# v11 seed1第二次数值失败与v12必要修复

runtime-formal-three-seed-v3-v11已failed/exit1，66文件370832206字节全回传验签。seed0完整512批/98304 transitions/8192 Adam/512乘子/2048回合成功且全冻结质量通过、A降级0/fallback0，fresh无resume；训练17121.277秒（约4h45m）。41文件354851755字节独立完整回传验签、core5/21动作523观察/schema/原v3 binding及final真实next512/全Adam param step8192/乘子512/finite/212来源重验通过，标准runs/m6p2c_formal_seed0_terminal_audit_v11_v1/。报告SHA73af26d5f892459a98a07c33ff8f628bad0425c0cbadd22892646adb43c8f4ec，final CP SHA565b7acf9674d53ac1a78f61f827432ba28fc8e1dcd72e2baf900aa70b8dd1d6；不混入下一版本科学结果。

同一parent自动fresh进入seed1，完成58批/11136 transitions/928 Adam/58乘子/232合格回合后，第59批origin1008/step18 Aoptimal/Binfeasible，调用.203974秒，非超时；18partial/raw/oldlogprob1.4657745361328125、失败批0更新/失败步未执行env、批前latest真实next58/Adam928/乘子58/原v3绑定全保留。failed_batch SHA b0be5998f5340df53a1b238d08265866da8772e28d5b71e4453ffc18a26fcaf4。seed2被阻断未创建，当前无训练进程；标准runs/m6p2c_v11_seed1_numeric_failure_audit_v1/。已有v10/先前v11失败及其各seed0成功全部保留。

N卡61d1e91先登记，两历史失败×两精度唯一runtime-seed1-two-failure-precision-probe-v1（原5403f21执行源）四次原.50固定诊断已全回传验签。旧9120在1e-9/1e-10均A/Boptimal；新1008在1e-9仍Aoptimal/Binfeasible，唯一非整数为首步充电mode z0（变量12281）8.131518354187604e-10，A offset1.1811164527906033；1e-10时A/Boptimal，整数残差0，A offset1.1916523914909947、最终row残差7e-15，.170288秒。两组A矩阵/边界/目标/整数声明逐字节相同；实际solver options/原始矩阵、所有观察均保留，0环境/PPO。近整数充电mode见证虽能通过原1e-6复核，却不能证明精确整数条件下该offset可行；只在B收紧精度无法修正A的offset。此前1e-9对原9120及已登记资格成立，不代表所有未知正式输入已覆盖。

red ed8948b及b0bf1e1先提交；最小修复a587764仅内部inventory求解精度1e-9收紧1e-10与准确说明，不放宽物理/服务/终点/offset1e-6/验收1e-6，不提高.50、不把B失败改成可执行降级，PPO/奖励/网络/队列/SOC/日期及r5配置不改。98对应回归/Ruff/mypy通过。新v12候选冻结revision fbfcea79303cf6103d7ff7bda1bfcf93fcabca08。只做更严格精度修复，不声称任意未知输入永不失败。

已唯一提交runtime-seed1-two-failure-repair-acceptance-v1（两个原失败各固定五次.50，0环境PPO、原A数学输入字节相同）和runtime-campaign-qualification-v12（全门禁/真实resume/48诊断/三seed各8批/4h前后共享，全工作量不省略），固定fbfcea79303cf6103d7ff7bda1bfcf93fcabca08、资格sealed bd21c3a8d561ea52b0c7f66b9f9ccee3f43e60224397f7bbd8649542a43abbaa，主机global单槽串行。全部成功及全回传验签前，不冻结release v4、不启动正式512。

直接交付控制器runs/m6p2c_v12_release_delivery_v1/controller.py等待新补验及完整资格，之后严格freeze v4并以最小必要数据/canonical资格/evidence交付，真实C/D预留及rw通过后唯一runtime-formal-three-seed-v4-v12，parent m6p2c_formal_three_seed_v4_v12，child m6p2c_formal_train_seedN_v4_v12均fresh。同一launcher每前seed完整512/98304/8192/512/2048和质量/产物验签才下seed，不独投或并发。旧L输入loader/idle既有receipt仅作不变输入/资源操作历史；新版完整门禁/诊断直接核验新版source和候选，不用旧receipt替代资格。源码变化不跨绑定恢复/旧CP换签，旧seed0成功独立保留；三seed主任务未完成，hourlyACTIVE辅助直接流程，validation/test封存。
