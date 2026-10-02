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
