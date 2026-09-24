# M6-P0：提前确定正式评估协议

## 开卡状态

- 起点：`4f97cf9e90f2456ff8914c73d05cbbf9e8c188f4`；开工前 `git status --short` 为空，当前分支为 `p4-safeppo-m51a-rollout-contract-m12-integration`，`git diff --check` 无输出。
- 本卡只确定协议及后续实施顺序。它不运行 validation/test，不宣称已有正式评估结论。

## 边界

- 新增 `docs/M6_EVALUATION_PROTOCOL.md`；更新 `docs/IMPLEMENTATION_PLAN.md` 的 M6 与放行顺序；本卡记录在此文件。
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
