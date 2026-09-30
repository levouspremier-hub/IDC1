# M6-P2a：正式 checkpoint 审核与 train-only 服务／库存诊断

## 0. 开卡状态

| 字段 | 值 |
|---|---|
| 分支 | `p4-safeppo-m51a-rollout-contract-m12-integration` |
| 强制起点 SHA | `646f8cc5c3a0fc797473ef4c767360f7ea8980bb`（tree `7488437b6bcad36acaffac948aa949b695bc6072`，实测一致） |
| 开工前 `git status --short` | **空**；`git diff --check` **exit 0** |
| 状态 | 执行中 |

### 0.1 改前失败原因（实测，未改既有测试）

```text
uv run python -m scripts.audit_m6p2a_formal_checkpoints --help
  → No module named scripts.audit_m6p2a_formal_checkpoints（exit≠0）
ls scripts/ | grep -i m6p2 → 无
```

## 1. 目标

**只做两件事**：①审核 `runs/m13gfck_formal_train_seed{0,1,2}/` 三个正式训练产物；
②在**矩阵 v3 的全部 212 个 train 日 origin × 3 个 seed = 636 个完整 episode** 上做
train-only 服务／库存诊断。**不做** validation/test、**不**重训、**不**做五方法公平收益比较。

## 2. 边界（允许修改 / 新增）

```text
scripts/audit_m6p2a_formal_checkpoints.py（**新增**：只读审计 + 诊断入口）
runs/m6p2a_formal_trainonly_212x3_v1/（**新增** run 产物）
runs/m6p2a_export_seed{0,1,2}/（三个 seed 各一次导出与源/导出对照）
任务卡、handoff
```

复用既有 checkpoint loader、统一评估器、冻结服务标准 `m6-service-standard-v1` 与库存记录。
三个正式训练 checkpoint **只读**。必要时修正诊断脚本本身。

### 2.1 固定输入

```text
policy        runs/m13gfck_formal_train_seed{0,1,2}/checkpoint_final.pt（只读）
role          formal_training_policy（经评估输入契约导出后加载）
origin 池     矩阵 v3 training_schedule.origin_pool（212 个 train 日 origin）
scenario_seed 0（**实际驱动环境**：env 种子 = 0 + 冻结 seed_offsets）
service       FROZEN_PROJECT_SERVICE_STANDARD（**显式传入** ⇒ 资格为真值，不再是「未判定」）
corrector     on，生产默认 0.25 s
method 标签   safe_ppo_joint_rolling_corrector（训练 seed 只作策略标识/分层）
```

## 3. 禁止项

- 不读、不运行 validation/test；不重训；不放宽服务、物理或终点 SOC 条件。
- 不把同一方法的三个训练 seed 当作**五方法公平收益**比较；本卡不计算公平收益。
- 不因单条异常轨迹推断整体表现。
- 不为假设性边界编写大量回归测试。

## 4. 验收命令

```bash
uv run python -m scripts.audit_m6p2a_formal_checkpoints --help
# 先单 origin 走通「评估 → 写盘 → 重新读取」
uv run python -m scripts.audit_m6p2a_formal_checkpoints --run-id <smoke-id> --origins-limit 1
# 全量 636
uv run python -m scripts.audit_m6p2a_formal_checkpoints \
    --run-id m6p2a_formal_trainonly_212x3_v1
uv run python -m scripts.materialize_train_release --verify
uv run python -m scripts.verify_m92_matrix --matrix configs/experiments/m9_experiment_matrix_v3.json
make check && git diff --check && git status --short
```

## 5. 证据产物

1. 三个正式 run 的逐项审核结果（manifest / revision / 三 hash / 种子 / 命令 /
   report 与 metrics 的 512 批、episode 与 step 计数、批次顺序摘要、checkpoint SHA 与可加载性、
   scope/role/21 维/冻结配置/优化器与 RNG 状态）。
2. `m13gfck_seed0_recovered_artifacts/` **单列**为历史诊断产物，不混入三份正式 run。
3. 三个 seed 各一次 `formal_training_policy` 导出 + **同一 train origin** 的
   源/导出确定性动作与完整 episode 对照（结果一致 ⇒ 通过；**不更新参数**）。
4. run `m6p2a_formal_trainonly_212x3_v1`：`config.yaml`、**636 行**逐 episode `metrics.parquet`、
   `report.json`、`figures/`、`manifest.json`（三 hash / revision / 种子 / 命令 / 状态可核对）。
5. 每 seed 的服务合格率、终点库存合格率、两者同时满足率与原因分布。
6. 代码优化建议（含实测依据、预期影响、是否改训练语义、是否必须重训）；
   **明确给出「现有 checkpoint 是否具备进入 validation 的条件」及理由**，不宣称泛化性能。

## 6. 回滚点

逐文件 `git add`、分批提交；从**最终 HEAD** 在独立 worktree newest-first 回滚本卡全部提交，
核对 tree 与起点 `7488437b6bcad36acaffac948aa949b695bc6072` 一致；工作树干净。

---

## 7. 验收记录

（完成后写入）
