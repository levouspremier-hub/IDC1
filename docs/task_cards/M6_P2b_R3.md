# M6-P2b r3：规划一致性、聚合积压与验收闸门返修

起点c3e37fb626003364530acf393459ccc0cf820a31，分支p5-eval-viz-m6-p2b-runtime，
开卡status为空、diff --check通过。用户授权先修复复审发现的问题，后续训练仅seed0。

## 边界

允许contracts/inventory.py、planning/service_guard.py/model.py/corrector.py、
safe_rl/corrector_wrapper.py、safe_rl_v2/inventory_diagnostics.py/formal_train_loop.py/
inventory_train.py、scripts/m6p2b_inventory_repair.py/defect_audit.py及必要的本卡
safe_rl_v2/rollout.py（仅汇总已记录inventory审计，不改变raw/logp/buffer语义）、
受控诊断入口、scenario/inventory_release.py、checkpointing/inventory_eval_input.py、
相关tests、docs及全新v2_r3/r4配置/发布、v4_r3/r4矩阵和runs/m6p2b_*_v2_r3/r4产物。
保留旧文件。新service reserve版本与实际规划代码、wrapper、奖励及配置共同绑定。

采用调度耦合的保守功率上界：基于现有非线性IT/COP公式及预测温度+冻结4.4°C，
对任务分配量建立功率增量上界；充电时零可再生预留。已到达任务进度用规划分配
计算；未来任务仅有聚合工作量及积压平衡，不实例化未来任务。保证R的库存证明
包含已到达任务的可行服务要求，证书不得用任意丢弃服务的slack冒充共同可达。

## 禁止项

不改env.step、受保护目录、冻结refs、数据/批次/服务统计规则、物理约束或raw/logp。
不重置SOC、不清空实际队列、不增加逐步50%罚项或改动奖励系数。不放宽1e-6配对。
R/A/B共享0.25秒；超时/求解失败不作为不可达证明。仅train，不跑validation/test、
seed1/2或正式512批长训。若短跑暴露问题，保留失败资产并返修，不自动启动长训。

## 改前失败证据与验收命令

r2复审runs/m6p2b_defect_audit_v2_r2/report.json：准确预测、同初始状态，低计算
提案终点差0.0584192919 kWh，提前执行提案差2.8312e-8；聚合到达[0,3,0]容量2
遗漏最后1 work；短跑94/96目标未达且末步不可达仍被放行。

先新增并提交tests/test_m6p2b_schedule_consistency.py的失败回归，覆盖上述算例、
聚合积压、服务兼容证书、功率上界、0.25预算、输入泄漏及末步绕过闸门。
验收：uv run pytest -q tests/test_m6p2b_schedule_consistency.py及相关库存/奖励/
恢复/发布测试；uv run ruff check；uv run mypy --explicit-package-bases；make check；
git diff --check。实现后重新跑固定种子受控复审、24-origin train-only标定/收益
诊断、版本验签和seed0×8批短跑及单seed闸门。新版本恢复和旧checkpoint拒绝测试。

## 证据与回滚

每个可运行诊断保存五类产物、代码/数据/锁/场景来源hash、命令、失败状态与读回；
记录每步规划条件/预测进度/实际进度及缺口变化。输入变化不能单凭SOC推断预测
误差；同输入与同执行进度下的缺口扩大必须阻塞验收。短跑报告服务/库存/共同目标、
充放电/收益、raw→exec及有效优势，不能把修正器成功当作PPO学会。
回滚点c3e37fb；本卡失败测试、实现、冻结资产、发布与验收分别独立提交，逆序
revert。最终报告说明残余风险、单seed结论与validation readiness，工作树干净。

## 补充反例与候选版本（2026-10-02）

r3的72日标定通过并生成冻结候选配置/矩阵，但尚未发布、没有checkpoint。
补充准确预测反例证明，开始不可中断任务后，旧模型仍允许未来暂停该任务而
错算回充余量。失败证据runs/m6p2b_r3_noninterruptible_redtests/及新增回归已提交。
后续增加已到达任务的全时域执行优先级与不可中断连续性约束（包括可部分执行
的首次启动），与现有环境分配器一致；不读取未来任务、不增加0.25预算。
保留r3候选及全部失败/成功诊断，最终使用全新r4配置、矩阵和发布重新标定。

## r4 实现进展（2026-10-02，未验收）

6d22ab6 将当前已知任务的执行优先级、首次部分执行和启动后连续性接入真实
终点前全部步数。相关测试转绿；4416/6624 两日完整48步，服务、共同目标50%、
物理均通过，无回退、超时或未证明可达性。证据：
runs/m6p2b_service_trace_v2_r4_retry1/。首次 trace 的完成任务索引异常保留于
runs/m6p2b_service_trace_v2_r4/；报告明确两例 IndexError，但旧诊断入口仅以
未捕获异常判断 manifest 成功，这是诊断状态缺陷。本轮修正入口，后续同时检查
完整 episode 和实质资格，不改动已保存历史产物。

runs/m6p2b_calibration_v2_r4/ 的72例服务/目标均通过，但7例有一次超时，
因此报告 passed=false，未冻结配置、未发布、未训练。继续添加由既有每步速率、
任务量、截止时间推导的变量界限和累计进度界；不改变可行域、不增加预算。
之后以新 run-id 重跑标定。奖励保持 common-sgd-degradation-v1；未新增奖励修复。
