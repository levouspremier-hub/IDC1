# AGENTS.md —— 本仓库协作代理规则书

> 依据：`docs/IMPLEMENTATION_PLAN.md` §1、`docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md`。
> 本文件为规则汇总；规则文字源自上述两份文件，修改规则前需人工确认。

## 1. 不可违反的红线（IMPLEMENTATION_PLAN §1.1）

1. 不放松接入容量、SOC、充放电互斥或任务约束来制造「可行」。
2. 不清空队列、不重置 SOC、不跳过任务来通过测试或评估。
3. 测试和评估不得读取未来真值；预测数组必须带来源、生成时刻和可见窗口。
4. 归一化参考值只来自训练集或预定物理尺度，冻结后所有方法共享；不得按测试日重算。
5. `marl/`、`grid_model/` 主逻辑、`legacy/` 和顶层兼容 shim 保持不改；旧 MARL checkpoint 不进入新主链。
6. 新主链禁止接受旧 23 维动作或无版本 checkpoint；应明确报错，不能填零或截断兼容。
7. 修改 `envs/idc_price_env.py::step()` 前，先提交该卡的失败测试；实现后该测试必须转绿。
8. PPO buffer 永远保存 `a_raw` 与其 log-prob；`a_exec` 只记录在 transition/info，绝不覆盖原始概率记录。

## 2. 分支规则（IMPLEMENTATION_PLAN §1.4）

分支依次使用：`p0-bootstrap`（M0）→ `p1-data-contracts`（M1/M2）→ `p2-physics/<card>`（M3）→ `p3-corrector/<card>`（M4）→ `p4-safeppo/<card>`（M5）→ `p5-eval-viz`（M6/M7/M8）→ `p6-experiments`（M9）。

`main`（本仓库为 `paper-baseline`）永远不改，只合并人工确认过的卡。

## 3. 产物规范（IMPLEMENTATION_PLAN §1.4）

每次可运行任务写入 `runs/<run_id>/`：`config.yaml`、`metrics.parquet`、`report.json`、`figures/`、`manifest.json`。`manifest.json` 写代码 revision、依赖锁 hash、数据 hash、场景 hash、种子、命令和失败状态。

## 4. 任务卡格式（IMPLEMENTATION_PLAN §1.3）

每张卡开工前在 `docs/task_cards/Mx.y.md` 写入五项，缺一不得开工：

- 边界（允许修改的文件和接口）
- 禁止项（不可改变的语义与红线）
- 验收命令（实现前存在、失败原因明确）
- 证据产物（测试名、机器可读结果、命令与路径）
- 回滚点（独立分支上的一个提交）

## 5. 测试标记（markers）

- `@pytest.mark.leakage`：未来信息泄漏回归
- `@pytest.mark.resume`：中断恢复一致性
- `@pytest.mark.slow`：慢速测试（`make check` 用 `-m 'not slow'` 排除）

## 6. Git 纪律（VSCODE_CLAUDE_EXECUTION_PROTOCOL §2）

- 每张卡：开卡检查（`git status`/分支/HEAD/`diff --check`）→ 任务卡提交 → 可验证编辑批次（先测试后提交，`git add` 显式列文件）→ 验收记录提交。
- 禁止：`git add -A` / `git add .` / `commit --amend` / `rebase` / `reset --hard` / `checkout --` / `git clean` / 强制推送。
- 每张卡结束 `git status --short` 必须为空。
