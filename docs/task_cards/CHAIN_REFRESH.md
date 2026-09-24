# CHAIN-REFRESH：B6 正式资产链的前向刷新（8 个文件）

## 0. 开卡状态

| 字段 | 值 |
|---|---|
| 分支 | `p4-safeppo-m51a-rollout-contract-m12-integration` |
| 起点 SHA | `626f97f`（M6-P1 实现提交） |
| 开工前 `git status --short` | **空**；`git diff --check` **exit 0** |
| 触发 | M6-P1 修改 `contracts/models.py` + `contracts/__init__.py`（评估记录契约） |
| 状态 | 执行中 |

## 1. 只读审计（实测，未预设文件数）

`b6_formal_code_revision()` / `provider_code_revision()` 的语义是
**「最后触碰该路径集的提交」**（不是内容哈希）。`FORECAST_SOURCE_PATHS` 含
`contracts/{__init__,models}.py`，故 M6-P1 的实现提交把两个解析器同时从
`cda31ab1…` / `080483b7…` 推进到 `626f97f…`，冻结资产因此校验失败。

### 1.1 哪些 revision 解析器受影响

| 解析器 | 路径数 | 含 `contracts/` 改动 | 记录的 revision | live |
|---|---|---|---|---|
| `provider_code_revision`（policy v1/v2） | 5 | **YES** | `080483b7…` | `626f97f…` |
| `b6_formal_code_revision`（policy v3） | 12 | **YES** | `cda31ab1…` | `626f97f…` |
| `refs_code_revision`（refs_v4） | 9 | no | `577f1db4…` | 同 ✓ |
| `b6_split_manifests.resolve_materializer_revision`（v5 triad） | 9 | no | `577f1db4…` | 同 ✓ |
| `formal_split_manifests.resolve_materializer_revision`（v1–v4） | 4 | no | `df61a3ea…` | 同 ✓ |
| `mapper_code_revision` | 8 | no | `a8065990…` | 同 ✓ |
| `b6_code_revision`（b6 arrival policy） | 2 | no | `3456f964…` | 同 ✓ |
| `b6_exogenous_revision`（exogenous v3） | 2 | no | `34f2de03…` | 同 ✓ |

### 1.2 SHA 传播闭包（自底向上）

```text
policy-v2  ──(自身 revision)─────────────────────────────► 失效
   │ 被 policy-v3 以 seasonal_rule_source_policy_sha256 绑定
   ▼
policy-v3  ──(自身 revision)─────────────────────────────► 失效
   │ 被 refs_v4 / v5 triad / mapper 以 sha256 绑定
   ▼
refs_v4  ──(被 v5 triad / mapper / env_release 绑定)──────► 因 v3 变化而失效
   ▼
formal_splits_v5 {train,validation,test} ──(被 mapper / env_release 绑定)─► 失效
   ▼
m13g_arrival_mapper_v1 ──(被 env_release 绑定)───────────► 失效
   ▼
idc_formal_env_release_v1 ───────────────────────────────► 失效（闭包顶点）
```

> ⚠️ **§1.2 第一版审计漏了 v4 triad —— 如实登记**。首轮手工列举只走了 10 个资产，
> 漏掉 `data/manifest/formal_splits_v4/{train,validation,test}.json`（它同样以
> `inputs.forecast_policy_manifest` 绑定 policy-v2）。**由 `make check` 抓出**
> （`tests/test_m13gc_scenario_manifests.py` 13 项失败），随后改用
> **穷举扫描**（遍历 `data/manifest/**` 与 `configs/**` 的**全部** SHA 指针并与磁盘实测比对）
> 复核，得到下方的完整闭包。**教训：闭包必须穷举扫描，不得手工列举。**

**完整闭包 = 11 个文件**（**不是** 5 个）：

1. `data/manifest/singapore_2024_forecast_policy_v2.json`
2. `data/manifest/singapore_2024_forecast_policy_v3.json`
3. `configs/frozen_refs/refs_v4.json`
4. `data/manifest/formal_splits_v5/train.json`
5. `data/manifest/formal_splits_v5/validation.json`
6. `data/manifest/formal_splits_v5/test.json`
7. `data/manifest/m13g_arrival_mapper_v1.json`
8. `configs/release/idc_formal_env_release_v1.json`
9. `data/manifest/formal_splits_v4/train.json`
10. `data/manifest/formal_splits_v4/validation.json`
11. `data/manifest/formal_splits_v4/test.json`

**不在闭包内**（已实测其 revision 与 SHA 指针均未变）：`singapore_2024_splits.json`、
`singapore_2024_half_hour.json`、`singapore_2024_exogenous_v3.json` +
`m13f_materialization_sources_v4.json`、`m13f_arrival_intensity_policy_v1.json`、
`formal_splits_v2/v3/v4`（已被取代）、`refs.json`/`refs_v2`/`refs_v3`（历史证据）。
`policy v1`（`0bf31f80…`）由 policy-v2 以 `supersedes` 登记，**字节不变**。

### 1.3 失效实测（起点状态）

```text
load_verified_policy_v3()      -> FormalB6Error: 差异字段=['materializer_revision']
load_verified_refs_v4()        -> 内部严格链先失败（同一条 policy-v3 错误）
load_verified_mapper_chain()   -> 同上
load_verified_split_manifest_v5('train') -> 同上
load_verified_b6_policy()      -> OK
load_verified_v3_bundle()      -> OK
m13gch2* / m13ge* 系列测试      -> 63 failed（全部 formal env 构造）
```

## 2. 边界（允许修改的文件）

```text
上列 §1.2 的 11 个资产文件（**只**允许 revision 与相互引用的 SHA 指针变化）
docs/task_cards/CHAIN_REFRESH.md、docs/task_cards/M6_P1.md
docs/WORK_HANDOFF.md、docs/NEW_CONVERSATION_HANDOFF.md
evaluation/controlled_run.py（M6-P1 的评估 episode 改用与受控短跑一致的修正器配置）
tests/test_m13feb2b_formal_cutover.py、tests/test_m13geb_arrival_mapper.py、
tests/test_m13gc_scenario_manifests.py（**登记**的枚举外迁移：三处钉住旧冻结值/
旧跨代不变式的断言，见 §8.4）
```

### 2.1 登记：枚举外的测试迁移（三处，逐条说明是否放宽）

| 文件 | 变化 | 是否放宽 |
|---|---|---|
| `test_m13feb2b_formal_cutover.py` | `PROTECTED`/`V4_TRIAD` 的 SHA 换成刷新后的新冻结值 | **否**（仍是逐字节相等） |
| `test_m13geb_arrival_mapper.py` | `PROTECTED` 的 policy-v2/v3/refs_v4 SHA 换成新值 | **否**（仍是逐字节相等） |
| `test_m13gc_scenario_manifests.py` | `test_v4_business_semantics_match_v3` 的 `inputs` 比较由「整张 SHA 表」收窄为「角色集合 + 逐角色路径」 | **是（已披露）**：`inputs` 的 SHA 属于该测试 docstring 明示豁免的「路径 / revision」范畴；v3 是**已取代的历史证据**，其绑定描述生成时的上游状态。九个业务语义键（origin 集合、切分、时间轴、frequency、readiness…）的比较**逐字段未变**。 |

## 3. 禁止项

- **禁止** `amend` / `rebase` / `reset --hard` / 移动或改写历史分支（用户明令）。
- **禁止**改动 8 个文件里的**任何数值、日期、切分边界、origin 集合、readiness 标志**；
  唯一允许的变化是 `materializer_revision` / `source_revision` 与指向被刷新文件的
  `sha256`（**逐字段实测证明**）。
- **禁止**改写 `policy v1`（`0bf31f80…`）、已被取代的 v2/v3/v4 triad 与 refs v1/v2/v3。
- **禁止**改 `envs/`、物理 / 任务约束、readiness 取值、`contracts/`。
- 不得用「重新物化」掩盖任何语义变化：每层刷新后必须与刷新前逐字段 diff。

## 4. 验收命令

```bash
# 逐层验签
uv run python scripts/materialize_b6_arrival_mapper.py --verify
uv run python scripts/materialize_env_release.py --verify
uv run python scripts/materialize_b6_refs.py --verify
uv run python scripts/materialize_b6_split_manifests.py --verify
uv run python scripts/materialize_formal_forecast_policy_b6.py --verify
# 正式链 + 回归
uv run pytest tests/test_m13feb2a_formal_b6.py tests/test_m13feb_b6_exogenous.py \
  tests/test_m13fe_arrival_intensity_policy.py tests/test_m13geb_arrival_mapper.py \
  tests/test_m13gch2r1_formal_causal_snapshot.py tests/test_m13ge_env_injection.py -q
make check && make smoke
git diff --check && git status --short
```

## 5. 证据产物

- §1 的只读审计表（解析器 × 路径集 × revision）。
- **旧/新 SHA 账本**（8 个文件 + 每层字段 diff 的差异字段清单）。
- 每层验签退出码；正式链与相关回归的实测结果；`make check` / `make smoke` 退出码。
- 刷新后 M6-P1 受控短跑可运行的证明（M6-P1 卡内验收）。

## 6. 回滚点

见 §8（实测，无占位符）。本卡不做历史改写，回滚按 newest-first `git revert` 验证。

## 7. 与 M6-P1 的关系

M6-P1 的实现（`626f97f`）**不撤回**：它是本卡的触发原因，也是「评估输入契约」
的正式交付。**M6-P1 的受控短跑验收须待本卡完成、正式链恢复后再跑**；
在此之前 M6-P1 处于「实现已提交、链路验收待续」状态。

---

## 8. 验收记录

### 8.1 起止与提交

（完成后填写）

### 8.2 旧/新 SHA 账本与逐层字段 diff

（完成后填写）

### 8.3 正式链验签与回归

（完成后填写）

### 8.4 回滚（实测）

（完成后填写）
