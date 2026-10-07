# 上一轮失败原因与修复复核（2026-10-08）

用户本轮要求找出上一轮失败原因、修复并汇报。读取交接后核对实际状态，确认最近正式训练失败为 v4/v12 seed1：完成 39 批后，origin7584 step1 的 A optimal、B infeasible，耗时 0.141172 秒。失败步未执行环境、失败批 PPO 更新为零；49 条 partial、原始动作概率、checkpoint 和失败历史均完整保留。

## 原因与现有修复

固定输入对照中，A 的整数残差为零，其见证满足 B 的完整原矩阵；在数学模型、目标、边界、整数声明逐字节相同的条件下，B 开启 HiGHS presolve 返回 infeasible，关闭后 optimal。这将问题定位至 B 的 presolve 行为；尚未定位 HiGHS 内部具体错误行。

O 卡已在本轮开始前完成最小修复：仅 inventory 阶段 B 的一次求解设置 `presolve=False`，通过原共享 options helper 接线。A、reachability、非 inventory 选项保持；不改变矩阵、目标、物理/服务/库存终点约束、1e-10 solver 精度、1e-6 正式验收容差或 0.50 秒共享预算，没有重试。红回归 `f610554`、修复 `fd13936`、helper 接线 `e4ed897`，最终冻结候选 `3d892a4f82b92808e7138fdf953af634e73cd1b8`。本轮核实修复有效，没有追加科学源码改动。

## 本轮验收

- `uv run pytest -q tests/test_m6p2c_seed1_stage_b_numerics.py tests/test_m54g_raw_projection_deterministic_options.py`：退出码 0，17 项通过。数值回归使用 2 秒以独立检验正确性；正式预算表现来自下面已有主机 0.50 秒补验，未改变正式预算。SciPy 提示附加选项原样传入 HiGHS，测试无失败。
- `uv run python runs/m6p2c_previous_failure_verification_v1/verify.py`：退出码 0；原失败 parent 65 文件 / 376266452 字节和 v14 补验 13 文件 / 55458 字节全部 SHA 通过，native receipt 为 all_files_verified。
- 三份历史原失败各 5 次，15/15 A/B optimal、可执行、原检查通过、整数残差零；耗时 0.278915—0.388127 秒。实际 B presolve=False、A 默认 True；原输入前后 SHA 一致，fixture 来源一致，环境/PPO 操作均零。
- 只读主机资格 `runtime-campaign-qualification-v14`：固定最终候选 revision、succeeded、exit0；本地已回传顶层资格报告的 SHA 与主机 receipt 对应条目一致，报告 passed、qualification_complete、failure=null。

完整资格本地回传仍进行中。标准证据保存时有 2847/3619 文件大小匹配、3860531417/4864627699 字节，native 全量凭据尚未生成；大小匹配不是全 SHA 通过。此前只读快照 2451 文件 / 3326870635 字节，证明传输继续增长。控制器和 watcher 存活、控制器无 error，等待全量验签；未竞争 collect、重启进程、另投任务或重跑已通过资格。本报告确认修复及补验通过，不宣称新三种子 512 批训练完成。

## 证据与工作树

独立标准产物：`runs/m6p2c_previous_failure_verification_v1/`，含 config.yaml、metrics.parquet、report.json、figures/、manifest.json，另存 pytest.log 与只读复验脚本。manifest 绑定代码 revision、uv.lock hash、补验 receipt hash、候选 hash 和命令。run 的 success 仅表示本轮独立复核通过。

任务卡 `docs/task_cards/M6.P2C-Q.md`；活动分支 `p5-eval-viz-m6-p2c-runtime-repair`，起始 `1bda30140fc6e1acb668d233572ae7b72299e748`。仅新增核查文档和忽略区独立证据，不修改执行闭包，不合并 paper-baseline。文档提交可独立 git revert；既有 O 卡修复不撤销。
