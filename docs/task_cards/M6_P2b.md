# M6-P2b：终点库存修复、train-only 验收与三种子重训

## 开卡 / 回滚点

- 起点：`55be5e53330636d9f7d21cf26631d47b220dbd91`；开卡工作树为空，diff --check exit 0。
- 独立分支：`p5-eval-viz-m6-p2b-runtime`（从失败测试提交 d7ccf56 分出；历史诊断分支 p5-eval-viz-m6-p2b/c936f53 保留，未改写历史）。公共 forecast 契约不修改，避免冻结 B6 revision 失效。
- 回滚：在独立 worktree 从最终 HEAD newest-first `git revert` 本卡提交，核对起点 tree；不合并 paper-baseline。
- 授权：用户明确要求实现本卡，包括短跑、版本发布、验收通过后的 3×512 批重训及 636 条 train 诊断。

## 边界

允许：新增 contracts/inventory.py（公共 forecast 契约保持原字节）；planning/model.py、planning/snapshot_adapter.py、planning/corrector.py；safe_rl/corrector_wrapper.py；safe_rl_v2/rollout.py、formal_train_loop.py，以及新增 inventory 训练/诊断模块；新增 scenario/inventory_release.py 与 checkpointing/inventory_eval_input.py；scripts/m6p2b_inventory_repair.py；Makefile；本卡相关测试、docs 文档、新 v2 配置/发布与 v4 矩阵；全新 runs/m6p2b_* 产物。

旧配置/发布/矩阵/运行产物保留。新正式入口须绑定训练语义，不允许旧 checkpoint 跨版本恢复。历史重放只读权重，不进行更新。

本卡暴露失败后的诊断入口还包括 scripts/m6p2b_short_trace.py、
scripts/m6p2b_budget_probe.py、scripts/m6p2b_reward_counterfactual.py；允许补充
planning/service_guard.py 与 tests/test_m6p2b_service_guard.py，研究在既有 B6
信息边界内为当前已到达任务保留服务功率。不得借此改变任务分配优先级、读取未来
具体任务或把保守规划假设当作实际物理不可达证明。服务保护尚未实现；若采用新
语义，先建立失败测试，重新标定并写全新 v2_r2 / v4_r2 资产及 run-id，保留首轮资产。

学习信号复查发现基础 520 维观测没有 SOC。允许在 safe_rl/corrector_wrapper.py
新增 `terminal-state-observation-v1`：在原始观测末尾追加当前 SOC、既定目标 SOC、
真实剩余步数/episode 步数三个无量纲量；原始观测及环境 step 不改。正式维度为
523，新配置/checkpoint/导出绑定观测版本，旧权重只进入显式历史诊断。
evaluation/adapter.py 的观测取值接口需保留 wrapper 的版本化观测，使训练、诊断
和正式导出一致；该变更不运行 validation/test。后续五方法共享该可见状态定义。

## 禁止项

不读或运行 validation/test；不改冻结服务标准、refs_v4、日期、批次顺序或种子；不放宽物理/任务约束，不重置 SOC、清空队列或跳过任务。不可达、超时、预测误差分别记录，不伪造成功。不改受保护目录，不把 exec 写入 raw 或 log-prob。原奖励优先；禁止逐步保持 50% SOC 的惩罚。若涉及环境 step，先提交失败测试。

## 改前证据与验收命令

改前：`uv run python -m scripts.m6p2b_inventory_repair --help` → No module named scripts.m6p2b_inventory_repair，exit 1。现有 snapshot 缺 episode 终点，滚动 cap=24，投影无终点库存约束。

```sh
uv run pytest tests/test_m6p2b_terminal_inventory.py
uv run python -m scripts.m6p2b_inventory_repair --help
uv run python -m scripts.m6p2b_inventory_repair --phase replay --run-id m6p2b_historical_replay_v2
uv run python -m scripts.m6p2b_inventory_repair --phase calibrate --run-id m6p2b_calibration_v2
uv run python -m scripts.m6p2b_inventory_repair --phase release --run-id m6p2b_release_v2
uv run python -m safe_rl_v2.inventory_train --short --seed 0 --run-id m6p2b_short_seed0_v2
uv run python -m safe_rl_v2.inventory_train --short --seed 1 --run-id m6p2b_short_seed1_v2
uv run python -m safe_rl_v2.inventory_train --short --seed 2 --run-id m6p2b_short_seed2_v2
uv run python -m scripts.m6p2b_inventory_repair --phase gate --run-id m6p2b_short_gate_v2
# gate 通过后逐 seed 从头完整训练；不得用短跑 checkpoint 冒充正式恢复
uv run python -m safe_rl_v2.inventory_train --seed 0 --run-id m6p2b_formal_seed0_v2
uv run python -m safe_rl_v2.inventory_train --seed 1 --run-id m6p2b_formal_seed1_v2
uv run python -m safe_rl_v2.inventory_train --seed 2 --run-id m6p2b_formal_seed2_v2
uv run python -m scripts.m6p2b_inventory_repair --phase audit --run-id m6p2b_trainonly_212x3_v2
make check
git diff --check
git status --short
```

失败定位及奖励依据补充命令（train-only，权重更新为零）：

```sh
uv run python -m scripts.m6p2b_short_trace --seed 0 --batch-index 0 --run-id m6p2b_short_trace_seed0_batch0_v2_r1
uv run python -m scripts.m6p2b_budget_probe --trace-run m6p2b_short_trace_seed0_batch2_v2 --baseline-revision ed184bf --run-id m6p2b_runtime_budget_verified_v2_r8
uv run python -m scripts.m6p2b_reward_counterfactual --run-id m6p2b_reward_counterfactual_v2
uv run pytest tests/test_m6p2b_sparse_assembly.py
```

## 证据产物与验收标准

每个 run 必须 config.yaml、metrics.parquet、report.json、figures/、manifest.json，包含 revision、lock/data/scenario hash、种子、真实命令、失败状态。失败产物不得删除或覆盖。

测试覆盖真实终点、末步、上下偏差、不可达安全执行、共享阶段约束、因果来源、raw/log-prob、新版恢复与旧 checkpoint 拒绝。24 个既有 train origin 的历史三 seed 重放、三种参考提案标定；三 seed 各 8 批真实短跑，共 4608 transitions，最终写盘读回；gate 检查完整服务、物理、库存以及学习信号与求解预算。任何未解决失败不得启动长训。

通过后按冻结 512 批 × seed 0/1/2 顺序从新初始化训练，审核每 seed 98304 transitions、8192 Adam step、512 乘子更新，再完成 636 条 train-only 诊断。真实终点对准现有 50% 目标；45–55% 资格及方法终点电量差 ≤1e-6 kWh 的配对规则不变。报告 validation readiness 与缺失四方法；本卡不运行 validation/test。

## 状态

执行中。首轮三 seed ×8 批已完成最终 checkpoint / 五类产物与读回验签，
但服务 35/96、库存区间 66/96、共同目标 57/96、物理违规 0、回退 135 次；
短跑闸门失败，未启动长训、636 条新训练诊断或 validation/test。原奖励未改。
已修复部分求解预算与 Stage B 原生数值错误；该候选代码尚未重新发布，首轮
release/checkpoint 绑定保留为历史证据，不能在代码 hash 不一致时用于新正式训练。

### 原奖励短跑失败后的 r2 候选

已提交观测接线与当前服务预留。`arrived-service-reserve-v1` 使用当前已到达任务的
期限、已启动不可中断状态及既有分配顺序，设置当前计算下界；为所有当前可处理
工作以 B6 可见温度的非线性 IDC 功率预留充电余量，可再生按明确的零供给假设。
这是收紧的规划假设，不能写成实际物理不可达证明；预算、真实终点、Stage R→A→B、
原奖励及物理校验不变。环境 step 未修改。

当前预留的一日诊断保留了服务，但暴露日终回充余量过于乐观的问题；已提交全
剩余时域充电预留的失败测试。修复应只使用已到达任务与 B6 的聚合到达预测，
不实例化或读取未到达任务的内容。所有预留假设进入快照及审计。

r2 资产尚未冻结。后续正式验收使用全新路径：

```sh
uv run pytest tests/test_m6p2b_service_guard.py tests/test_m6p2b_inventory_observation.py tests/test_m6p2b_release_checkpoint.py
uv run python -m scripts.m6p2b_reward_counterfactual --candidate --run-id m6p2b_reward_counterfactual_v2_r3
uv run python -m scripts.m6p2b_inventory_repair --phase calibrate --run-id m6p2b_calibration_v2_r2
uv run python -m scripts.m6p2b_inventory_repair --phase release --run-id m6p2b_release_v2_r2
uv run python -m safe_rl_v2.inventory_train --short --seed 0 --run-id m6p2b_short_seed0_v2_r2
uv run python -m safe_rl_v2.inventory_train --short --seed 1 --run-id m6p2b_short_seed1_v2_r2
uv run python -m safe_rl_v2.inventory_train --short --seed 2 --run-id m6p2b_short_seed2_v2_r2
uv run python -m scripts.m6p2b_inventory_repair --phase gate --run-id m6p2b_short_gate_v2_r2
```

上述 r3 counterfactual 为已完成的全 24-origin、零参数更新原奖励候选诊断；
r4 是全未来预留及新奖励的实测对照，残余两个服务失败已逐步定位并返修。
初版 r2 因零购电费计算奖励斜率时报错，失败产物完整保留，原始逐步数据未成功
落盘，报告明确标注仅有控制台舍入摘要，不能当作正式测量。斜率现从实际环境
系数与冻结参考值计算，后续异常会保存已完成 episode 和完整失败状态。

具体奖励返修依据与公式已写入 `docs/audits/M6_P2b_INVENTORY_REPAIR.md` 的 r2 节：
完整 24-origin 原奖励对照出现 4 个公平且有净现金收益、原奖励却下降的配对，
退化费单位奖励尺度比购电费高 600 倍。允许在既有 wrapper 中实施
`common-sgd-degradation-v1`，只替换退化奖励为
`-(reward_cost_weight/cost_ref)*实际退化SGD`，其他分项、refs_v4 与环境 step 不变。
先提交 `tests/test_m6p2b_reward_semantics.py` 的失败测试；落实后重新短跑，不能
把算术正向信号称为 PPO 已学会，也不能凭此放行未解决的服务/库存问题。

长训准备允许在 formal_train_loop.py 复用已验签 train 注入，并扩大 snapshot_adapter
的同源 bundle 缓存。依据首轮真实分项计时，前 3 批每批重复环境构造约 19–20 秒，
不是 PPO 更新瓶颈。该优化不修改 B6 生成代码、日期、输入或 RNG：缓存键须包括
实际输入/代码完整 hash 与生成 revision/dirty 状态，资产变化必须重新验签而不能
返回旧场景；每个环境得到独立 deepcopy 注入、数组、任务及 SOC，新旧环境状态
不能共享。先提交 `tests/test_m6p2b_verified_input_reuse.py` 的失败测试，完成后重新
核对恢复、泄漏与逐位输入一致性。当前 reward/service 诊断结束前不修改其运行代码。

服务失败定位允许新增 `scripts/m6p2b_service_trace.py`：仅 train origin 4416/6624，
同一既定因果提案、零参数更新，记录每步已到达任务、计划/执行动作、服务预留及
物理削减。当前步真实物理量仅作执行后诊断，绝不进入规划输入或改变求解调用。
保存五类产物、来源 hash、失败日期与完整事件，重新读取核对；该入口不是正式验收。
验收命令：`uv run python -m scripts.m6p2b_service_trace --run-id m6p2b_service_trace_v2_r1`。
改前证据：`runs/m6p2b_reward_counterfactual_v2_r4/metrics.parquet` 的两条 amplitude=.1
服务失败，72 日中服务 70、库存区间 72、物理违规及回退 0。输入复用提交 43fac23
为本定位批次的独立回滚点，定位后必须先提交对应失败测试再修改执行语义。

温度误差返修允许在同一 trace 入口增加 `--temperature-calibration`，只取固定24
train origin 的已验签温度预测与执行诊断真值，冻结 `ceil(max(真值-预测,0)*10)/10`
摄氏度作为服务功率预留的温度上偏假设。物理执行仍用原真值链；预测数组不变，
假设不构成物理不可达证明，超出该训练误差范围仍须记录，不保证 heldout 数据。
允许在 inventory 契约、service_guard、训练配置/发布接线中显式绑定该常数、公式与
证据 hash。验收新增 `tests/test_m6p2b_service_guard.py` 的温度低估失败用例，先提交
失败测试；范围仍不含环境 step。回滚点 d03e327，误差证据见 service_trace_v2_r1。
