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

## 学习信号补充范围

r4 最终72日收益对照全部完整、服务/目标/物理通过、无回退，48个公平配对中
9个净获利，但仅1个实际训练奖励增益为正；平滑项是主要抵消项。24个日初
储能探针全部同执行结果，不以日初坍缩断言全程信号消失，也不直接修改奖励。
允许新增 scripts/m6p2b_storage_state_probe.py：固定24-origin、固定时刻
0/12/24/36，在因果价格提案 amplitude=.1 的实际当前状态复制环境，保持计算
提案相同，分别试储能 -.1/0/.1。记录执行、收益/退化、原始奖励分项、任务进度
变化和回退；实际当前外生值仅用于环境执行，规划输入仍沿用已验签预测。
不选择有利时刻、不读未来任务或真值、不更新权重；每个求解仍0.25秒。
先给出平滑项是否属于目标的具体分析及同状态证据，再决定是否另开奖励返修卡；
本卡不自动改奖励，不启动长训。证据 runs/m6p2b_storage_state_probe_v2_r4/。

## 诊断证实后的奖励返修边界与公式

旧禁止“改动奖励系数”继续生效：不调原权重或refs；本节只授权在wrapper替换
两个路径依赖平滑惩罚的语义。允许safe_rl/corrector_wrapper.py、
safe_rl_v2/formal_train_loop.py、scenario/inventory_release.py及诊断入口、奖励测试，
新增全新v2_r5训练配置/发布及v4_r5矩阵；r3/r4和诊断保留。依据已有用户计划
“诊断证实需要奖励修复，先形成公式、系数依据和train-only对照，落实再短跑”。

完整证据：runs/m6p2b_reward_counterfactual_v2_r4_final/的48个公平配对有9个
净获利，8个实测奖励为负，两个平滑项抵消经济/服务收益；只在算术中去掉这
两项则9个均为正。runs/m6p2b_storage_state_probe_v2_r4/：96个固定状态中
64个执行非坍缩，288次探针无回退、24基础episode合格，排除全程映射信号消失。
本结论不把未计入协议的负载/动作摆动当作货币成本，不直接按获利计数选系数。

新语义common-sgd-potential-smooth-v1：
Phi(s) = -w_load * mean(abs(prev_loads-base_load))
         -w_action * mean(abs(prev_exec_action))；终止状态Phi=0。
r_new = r_common_sgd - r_load_smooth - r_action_smooth
        + gamma*Phi(next_state) - Phi(state)。
两系数沿用现有环境0.05/0.03，不调参；gamma必须显式等于冻结PPO gamma。
完整episode的折扣塑形和=-Phi(initial)，同初始状态下与提案无关；对初始Phi=0
的正式episode恰为0。保留成本/碳/服务/退化/终端分项，新增塑形分项并保留原
平滑值审计。绝不新增逐步SOC惩罚，不改env.step或物理/任务约束。

先提交失败测试，覆盖逐项账目、完整折扣望远镜恒等式、动作/状态不变、缺失
折扣率报错；然后修改wrapper。重跑24-origin标定、收益对照、恢复/发布验签、
make check，再仅seed0×8批。证据使用全新r5 run-id。独立回滚点989970b，
返修测试/实现/冻结资产/发布/验收分别提交，不覆盖r4。长训仍禁止。

初态常数核对：现有环境reset的prev_action全为0.5，故正式初态Phi=-0.015；
完整episode折扣塑形和为共同常数+0.015，策略间差为零。上文“初态Phi=0时为零”
是数学条件而非本环境的初始值；不修改reset或增加势函数偏移来追求数值零。
两组完整序列单测直接按实际初态验证-Phi(initial)。r5指标保留每个episode的
r_potential_smooth累计/折扣值，最终报告核对此共同常数，不能当作额外调度收益。

## 最后接口红线返修（r6发布，标定系数沿用r5）

门禁复审实际复现：当前wrapper接收23维raw，生成21维exec并推进step=1。
这是AGENTS红线6的截断兼容缺陷；checkpoint/buffer虽已校验21维，公开wrapper
入口同样必须拒绝。先新增失败测试覆盖20/22/23维，断言在规划与环境推进前
明确ValueError；合法21维仍逐元素保留raw。允许wrapper的入口维度校验、
新tests/test_m6p2b_wrapper_action_contract.py、scenario/inventory_release.py，
新增release v2_r6及r6受控runs。不会改env.step，不影响合法动作的规划/奖励。

r6仅发布入口校验修复；沿用已验签且行为不变的r5冻结配置/矩阵及24-origin
标定，不重算系数/预算/refs，不重做有效动作的奖励选择。新release绑定实际
wrapper代码并注明资产继承依据，旧r5发布保留为历史，不能混用checkpoint绑定。
回滚点3e5e767；测试/实现/发布/验收分别提交。

完整make check r5约55%时主动SIGINT中止，原生console/JUnit及五类产物已
保留并写失败状态，不能声称完整门禁通过。补充回归转绿后对最终代码重新
make check，并使用全新r6 run-id运行仅seed0×8批及单seed验收。验收命令仍为
相关测试、完整make check、发布/矩阵验签、git diff --check及产物读回。
