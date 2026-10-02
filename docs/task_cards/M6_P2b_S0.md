# M6-P2b S0：显式单seed正式长训授权

日期2026-10-03，起点d69c91a，分支p5-eval-viz-m6-p2b-runtime，开卡工作树为空。
用户明确要求开始正式长训、只启动seed0；此授权取代前卡“不启动长训”的范围。

## 边界

仅修改safe_rl_v2/inventory_train.py的启动门禁/审计字段/CLI，
scenario/inventory_release.py的r7发布与seed0授权绑定，新release v2_r7、
专属测试、文档和全新runs/m6p2b_*_v2_r7产物。继承已冻结r5配置/矩阵/标定。
单seed授权只验证r6的seed0短跑，不恢复该短跑权重，正式512批从新初始化开始。
规划/环境/奖励/优化器/buffer/正式批次顺序均不修改。

## 禁止项

不得伪造formal_three_seed_gate=true；默认三seed门禁保持。显式seed0授权不得
用于seed1/2、短跑或旧checkpoint恢复。物理、服务、SOC目标/区间、0.25预算、
refs_v4、raw/log-prob和受保护目录不变。只train，不运行validation/test。
不覆盖旧run。验证的旧短跑只能作为启动证据；新正式checkpoint绑定r7实际入口。

## 改前失败与验收

旧require_short_gate(binding)查找runs/m6p2b_short_gate_v2_r6，要求三seed，
现有通过产物却为m6p2b_short_gate_seed0_v2_r6且formal_three_seed_gate=false。
新单seed接口尚不存在，先提交失败测试。
命令：uv run pytest -q tests/test_m6p2b_seed0_launch.py及全部test_m6p2b_*；
uv run ruff check；uv run mypy --explicit-package-bases；发布/矩阵/旧短跑验签；
git diff --check。既有完整make check 3127通过的物理/训练实现保持不变，
本卡追加启动入口针对性回归和静态检查，不把旧完整结果改写为r7完整检查。

## 证据

新r7发布保存继承r6单seed门禁路径/hash与允许变化的两份入口源码；验证其余
语义绑定、配置、矩阵和原生短跑/checkpoint hash不变，旧权重不得用于初始化。
正式run保存五类产物、来源hash、512批顺序、命令、seed0限定授权和运行状态；
每批写最新checkpoint/journal，最终写盘/验签交给正式入口。后台进程与console
日志保留，起跑后核实第一批checkpoint和存活，不宣称训练已完成。

## 回滚

独立回滚点d69c91a；任务卡、失败测试、入口、发布与启动记录分别提交。
逆序revert，保留所有run，不使用破坏性Git。收尾工作树为空。


## 启动验收（2026-10-03）

失败测试02bda65：4失败、1通过，缺少单seed入口；原生证据
runs/m6p2b_seed0_launch_red_v2_r7/。实现73fcc09、发布b4f49a5。
专属12项及原发布/恢复测试合计26项通过；ruff与两份入口mypy通过。
完整相关测试原生runs/m6p2b_related_check_v2_r7/：93通过、0失败/错误/跳过，
实现来源未变、读回验签通过。本卡未重新运行完整make check，不将旧r6结果
冒充r7全仓检查；已验收的规划、奖励、环境与优化器文件均保持字节不变。

新r7继承r6单seed闸门，现场校验原生报告/manifest/checkpoint hash、32条资格，
配置/矩阵/奖励/guard/523观测及除两个入口外全部源码绑定一致。默认三seed
门禁保持，formal_three_seed_gate始终false，显式入口拒绝seed1/2及short混用。
正式训练从新初始化开始，CLI没有resume参数，短跑checkpoint只作验收证据。

运行：runs/m6p2b_formal_train_seed0_v2_r7/，512批，98,304 transitions，
命令python -m safe_rl_v2.inventory_train --seed0-formal --seed 0
--run-id m6p2b_formal_train_seed0_v2_r7。后台uv PID56898，训练Python PID56900，
独立会话，不依赖本聊天持续打开；caffeinate -i仅在进程存续期间防止空闲休眠。
console.log、launch.json、五类原生产物已保存；状态running而非success。
每批保存journal/latest checkpoint，首批及每16批更新运行中报告/来源账。
首批4/4完整、服务/物理/库存/目标合格，无回退，与既有短跑正式前缀相同。
launch_verification.json观察到第3批checkpoint，schema/角色/seed/实际r7绑定
已验签；仅证明起跑，未宣称512批完成。seed1/2、validation/test均未启动。

后续不要在该进程运行期间修改绑定的实现/配置/数据；遇到失败保留run，通过
本版正式checkpoint按同绑定恢复，不用旧短跑或历史正式权重恢复。当前仅为
单seed正式运行，不具备新版三seed复审、636诊断或五方法validation readiness。
回滚先有序停止本run、保留产物，再逆序revert本卡独立实现/发布提交；d69c91a
为开卡回滚点。本卡结束提交启动记录、diff --check和工作树为空。
