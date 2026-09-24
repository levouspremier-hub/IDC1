# M6-P0：提前确定正式评估协议

## 开卡状态

- 起点：`4f97cf9e90f2456ff8914c73d05cbbf9e8c188f4`；开工前 `git status --short` 为空，当前分支为 `p4-safeppo-m51a-rollout-contract-m12-integration`，`git diff --check` 无输出。
- 本卡只确定协议及后续实施顺序。它不运行 validation/test，不宣称已有正式评估结论。

## 边界

- 新增 `docs/M6_EVALUATION_PROTOCOL.md`；更新 `docs/IMPLEMENTATION_PLAN.md` 的 M6 与放行顺序，并在 `docs/CHATGPT_HANDOFF_CURRENT.md` 指明新顺序；`.gitignore` 只加该协议文件的最窄放行规则；本卡记录在此文件。
- 将既有 `evaluation/adapter.py` 与 `contracts.models.EvaluationRecord` 的口径差距登记为后续 M6-P1 的实施输入，不在本卡改代码。

## 禁止项

- 不读取 validation/test 结果来选业务阈值、调方法或挑日期。
- 不修改 `evaluation/`、`contracts/`、`envs/`、checkpoint、训练配置、冻结资产、readiness 或旧 23 维模型。
- 不把受控短跑 checkpoint 当正式训练产物，不把 `total_objective_cost` 标作 SGD。

## 验收命令

开工前已有的只读基线：`git status --short`、`git branch --show-current`、`git rev-parse HEAD`、`git diff --check`；现有 `uv run pytest tests/test_m62_eval.py -q` 作为骨架回归基线。本卡为文档协议卡，无需人为制造失败测试。完成后执行 `git diff --check <起点>..HEAD` 与 `git status --short`。

## 证据产物

- `docs/M6_EVALUATION_PROTOCOL.md`：研究主问题、服务资格定义与阈值决策门、指标口径/来源、五类方法、配对与不确定性、checkpoint/数据使用顺序、失败处理。
- `docs/IMPLEMENTATION_PLAN.md`：M6-P0 → checkpoint 契约稳定 → M6-P1 评估器与受控短跑验证 → 正式训练产物审核 → validation → 最终 test。
- 本卡验收记录：改动路径、命令结果、未决定事项、最终 SHA、从最终 HEAD 在独立 worktree 验证的 newest-first 回滚。

## 回滚点

本卡与协议正文分开提交；在独立 worktree 从最终 HEAD 以 newest-first `git revert` 验证回到本卡起点的 tree。完成时工作树必须干净。

## 验收记录（协议准备稿）

- 提交：`a79dc7d`（开卡）、`33eff26`（协议/实施顺序/交接入口）。本节为证据收口提交。
- `uv run pytest tests/test_m62_eval.py -q`：3 passed，确认旧骨架仍可运行；该测试不证明新协议已实现。
- 审计发现：`evaluation/adapter.py` 的服务资格默认 true、可再生占比分母与 `EvaluationRecord` 字段不足，均已写入 M6-P1 实施边界。
- **未冻结** 95%／1% 提案；待方向确认。未改评估器、环境、checkpoint、训练配置或 readiness；未运行正式训练、validation、test。
- 完整范围回滚从最终 HEAD 依次 revert 本节提交、`33eff26`、`a79dc7d`；起点 `4f97cf9^{tree}` 为比较基准。实际验证结果由最终提交后的独立 worktree 命令核对。
