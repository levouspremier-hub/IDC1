# 本地修复：探索性验收、服务守恒与B变换预算

## 范围与状态

用户要求修复代码审查发现的问题，随后确认主机暂时离线、先完成本地修复。本轮没有训练、远端投递、候选冻结或正式放行。起始HEAD e2a4426；分支p5-eval-viz-m6-p2c-root-numerics。Y的四个已知未提交文件保留，AD/AE只显式提交各自修改；model.py的AE进度界hunk通过git add -p单独暂存并核对，Y变换不混入AE。

## 已修复的根因

1. **探索性测试验收混淆**：原run要求所有case.passed，无视expectation。现在生产可执行passed与诊断acceptance分离：must_pass只能接受原证书通过的解；exploratory仅额外接受绑定原模型且经过精确有理数复核的区间矛盾。必须超过两侧1e-10证书容差之和；ULP级空区间、仅infeasible标签、缺源、超时、未知和目标证书失败仍不能通过。原失败和passed=false完整保留，不改变正式准入判据。
2. **验收输入绑定缺失**：scientific_binding增加tests/fixtures、测试源码和configs文件hash。诊断结束、preflight结束和冻结前同时检查源码/验收输入/锁，不允许注册表或测试变化后复用旧通过记录。
3. **预留服务过计**：9744/44 task45仅剩27，预留需求为27.000000000000004；math.fsum正确舍入为27，而组片段的精确和仍比27多1/562949953421312。因此仅换求和函数还不足以保证守恒。第一版账本修复后继续复核，发现小数速率连续浮点扣减仍可把1.7×48累计成81.60000000000011。因此最终改为binary64输入的有理数账本，按真实剩余与单步速率计服务，保留非零未服务尾量；required_end/due向下编码，不凭求和尾数多要求工作。allocator排序/规则、物理任务和环境执行均不改。
4. **派生进度界的二次舍入**：即使服务累计正确，1.2-.3×3也会编码为.30000000000000004，大于.3。新service_progress_bounds对原输入做有理数运算，仅对数学蕴含的冗余下界向下、上界向上编码；真实任务/速率行保持。输入本身要求27+ULP而任务只有27时仍保留矛盾，不夹紧、不改历史fixture。
5. **B变换耗尽预算后仍启动求解**：新增红测试证明旧代码会以time_limit=0再次调用milp。Y候选增加变换后预算检查，耗尽时走原有B未启动/原证书认证A路径；B真的infeasible仍失败，不增加生产重试或时间。

## 代码复核与限制

- 任务累计的每步服务不超过精确remaining和rate；end/due来自同一账本，不能大于原任务，也不能跨步累积超过速率总量；截止期前缀和未完成尾量有独立回归。
- 派生界通过H=1..48所有前缀、四种整数/小数速率与精确有理数比较；不是通过放宽原业务/容量约束制造可行。
- 二次幂行缩放精确往返只覆盖缩放，不能证明连续原点平移、求解器presolve或目标计算对所有未来输入无误。原矩阵/整数/目标证书、唯一LP、共享预算、失败即停均保留。
- 主机离线；六锚点/78个源输入仍缺失。AB恢复必须使用原源环境生成历史快照，不能使用新服务会计生成的状态冒充旧轨迹一致。新科学源不得跨源继承旧short/资格。
- “不可能再中断”不是修复验收条件。未经来源闭合和新源完整资格，不启动正式训练。

## 证据

- AD红：runs/m6p2c_ad_red_v1，12项预期失败；绿：runs/m6p2c_ad_green_v1，30通过。静态：m6p2c_ad_static_v1。
- AE原案例红：m6p2c_ae_red_v1，1失败6通过；原案例/服务绿：m6p2c_ae_green_v1，16通过；关联：m6p2c_ae_related_v1，56通过。
- AE同类速率红：m6p2c_ae_rate_red_v1，6失败；精确进度界红：m6p2c_ae_progress_red_v1，5失败；最终针对性绿：m6p2c_ae_progress_green_v1，exit0；静态m6p2c_ae_exact_static_v1，exit0。
- Y预算红：m6p2c_y_deadline_red_v1；首轮绿尝试m6p2c_y_deadline_green_v1被既有确定性选项构造结构测试拦下，保留。恢复原共享_options直接调用接线后m6p2c_y_deadline_green_v2全部58通过，未削弱旧测试。
- 最终联合回归m6p2c_local_repair_regression_v2：277 passed、0 failure/error，exit0，78.233s，源/验收输入/锁结束复核一致。包括全部m6p2c数值、诊断/前置、service/schedule/inventory/wrapper/sparse及原投影/确定性选项测试。早期v1不代表后来精确账本/进度界的最终验收。
- 最终m6p2c_local_repair_preflight_v1：static exit0；latest 1/1、0.269s；history 143/143、32.596s；boundary 234中156 passed/accepted、78 input_source_contract_failure、46.027s，无其他失败类别。前置链真实failed/exit1/89.252s并停在boundary，没有启动后续make check、冻结或任何训练。旧9084376的完整门禁通过不挪用于本次源码；最终完整门禁待六源锚点恢复、边界通过后执行。

本轮实现提交：AD a14b9d8；AE a837ba8（首轮账本），67f8e91（最终精确累计与派生界，生产修复采用后者）。红测试d7f41d4、5a6862c、f8cbf60、cfc86af；Y预算红测试7f393a7。Y模型变换/预算候选仍未满足既定科学提交条件，四文件不打包提交；完整diff保存在runs/m6p2c_local_repair_regression_v2/y_candidate.patch并有SHA收据。

各运行含config.yaml、metrics.parquet、report.json、figures/、manifest.json；测试运行另有JUnit、command.log及源码/验收输入hash。撤销已提交修改只用git revert，Y未验收修改仍按原任务卡保全。
