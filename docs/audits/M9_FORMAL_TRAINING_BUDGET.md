# M9.1 当前正式 PPO 训练链的资源预算

> **范围**：只覆盖**当前已实现的 PPO 主链**——冻结配置 v1、formal env、corrector on
> 生产默认 0.25 s、4 episode × 48 步采集、4 epoch × 4 minibatch 更新、批次边界 checkpoint。
> 规则 / 独立滚动优化 / 惩罚 PPO / 安全 PPO 四类方法**未实现、未测**，
> 其时间与五方法实验总预算**不能**由本报告的数字化出。
>
> **不是**训练结果，**不是**性能或收敛结论。缓解措施**未**采用：未减少批数 / 种子 / 日期，
> 未关闭 corrector。

## 1. 机器与测量样本

```text
machine      Apple M5（arm64），macOS 26.6.2，逻辑 CPU 10，内存 16 GiB (17179869184 B)
torch        2.14.0
torch 线程数  配置值 1（冻结配置 backend.torch_num_threads）；实测生效值 1
             （本次修正前进程用 Torch 默认值 4 —— 见 §5）
样本         3 个 seed（0/1/2）× 3 批连续短跑 = 9 批
             seed 0/1/2 的环境种子分别为 0/1/300000、1/2/300001、2/3/300002
来源         三条短跑 run 与预算 run 的 revision 均指向含测量代码的提交；
             dependency_lock_hash / data_hash / scenario_hash 三个 hash **非空**
```

## 2. 每批耗时（实测，9 批）

| 分项 | 均值 | 说明 |
|---|---|---|
| `env_build_s` | **20.31 s** | 4 个 origin 的 formal env 构造 + verified injection |
| `rollout_collect_s` | **15.34 s** | 策略采样 + 环境步进 + corrector 求解（**不含**构造） |
| `ppo_update_s` | **0.016 s** | **16 次 Adam step** 的计时；计时器在 advantage / critic target 计算**之后**启动，故该值**不含**该次前向计算（见下） |
| `checkpoint_write_s` | **0.023 s** | 批次边界训练恢复 checkpoint 写入 |
| `residual_unattributed_s` | 0.004 s | **差值**（整批 − 已测分项），**不是**独立测量值 |
| **端到端每批** | **p50 35.63 s / p95 36.78 s**（min 34.93、max 36.76） | 批内墙钟 + checkpoint 写入 |

- **修正器解算耗时**：中位数 **0.0728–0.0789 s**、P95 **0.1055–0.1471 s**（逐批、共 9 批）。
- **超时 / 回退**：9 批共 **12 步**零动作回退（合计 9 × 192 = 1728 步，占 0.69%）。
- **峰值常驻内存**：**0.50 GiB**（运行期最大 `ru_maxrss`），占目标机器内存 **3.1%**。
- **首次初始化 vs 稳定批次**：批 0 的端到端成本 p50 **35.75 s**，与其他批次（p50 35.63 s）
  无显著差异；一次性初始化（进程启动、模块导入、verified loader 首次读取）**未单独隔离测量**，
  故不在外推中单列——**未用差值冒充**该成本。
- **`ppo_update_s` 的覆盖范围（M9.2 更正）**：初版把该值写成「含 advantage/target 计算」，
  但计时器实际在 `compute_advantage_oriented_arrays()` **之后**才启动。因此 0.016 s
  **只覆盖 16 次 Adam step 本身**；该批的 advantage / critic target 前向计算落在
  `rollout_collect_s` 与 `ppo_update_s` 之间的未计时区间，被归入
  `residual_unattributed_s`（实测 0.004 s，本可忽略，故外推数值不受影响）。
  **计时数据与外推数值均未改动**，仅更正描述。

## 3. 512 批 / seed × 3 seed 外推

```text
公式：per_seed = per_batch × 512 ；all_seeds = per_seed × 3
      per_batch = 批内墙钟(实测) + 批次边界 checkpoint 写入(实测)
典型值 = 稳定批次 p50 = 35.63 s ；保守值 = 稳定批次 p95 = 36.78 s
```

| 量 | 典型值 | 保守值 |
|---|---|---|
| 每批 | 35.63 s | 36.78 s |
| **每 seed（512 批）** | **5.07 h** | **5.23 h** |
| **全部 3 seed** | **15.20 h** | **15.69 h** |
| transitions / seed | 98 304 | 同 |
| transitions 合计 | 294 912 | 同 |

**磁盘**：单个批次边界 checkpoint 实测 **850 533 B**；按**每批都保存**这一上界，
每 seed 512 个 ≈ 0.41 GiB，**3 seed 合计 ≈ 1.22 GiB**。
体积与批数成线性，若改为每 N 批保存则按 1/N 缩减。run 目录本身另计（数量级更小）。

**内存**：峰值 0.50 GiB vs 机器 16 GiB（3.1%）——内存**不是**约束。

**未包含的时间**：首次初始化的一次性成本（未单独隔离）；最终 policy 导出与评估；
validation/test；**其他四类方法**（未实现）；数据下载 / 环境安装 / 依赖解析。

## 4. 建议

**本机运行（无需外移）**。依据：三个 seed 共 9 批实测，稳定批成本 p50/p95 为 35.63 / 36.78 s；
外推 512 批 × 3 seed 约需 **15.2–15.7 小时连续运行**；峰值内存 0.50 GiB（占机器 3%）；
磁盘上界 1.22 GiB。

**必须一并知悉的限制**：

- 本卡**未**获得可用工时上限，因此**不**判定「是否能在某截止期内完成」——
  只报告需要的连续运行时间与资源，**不自造通过门槛**；
- 上述为**连续**运行时间；中途重启、机器休眠、并发负载都会延长；
- 其他四类方法未测，五方法实验总预算需在各自实现后另行实测。

## 5. 测量期间发现并修正的接线问题（登记）

1. **Torch 线程数**：冻结配置 v1 声明 `backend.torch_num_threads = 1`，但受控入口此前
   未设置，进程实际使用 Torch 默认值 **4** ⇒ 测得的不是**冻结配置**下的吞吐。
   本卡新增 `apply_frozen_thread_setting()`，并在报告中同时记录配置值与生效值。
   **只改线程数**：同 seed 的最终 policy 摘要与设置前逐位相同。
2. **计时分项重叠**：初版 `rollout_collect_s` 把 `env_build_s` 一并计入，
   导致 `residual_unattributed_s` 出现 −20.2 s 的负数；已改为分别独立计时。
3. **误导字段**：初版报告含 `run_dir_size_bytes`，但 run 目录在报告**之后**才由
   `write_run` 创建，该字段恒为 0；已删除，run 目录大小改由预算脚本从磁盘实测。

## 6. 可重算性

```text
预算 run      runs/m91_formal_training_budget_v1/（manifest.status=success）
入口          python -m scripts.m91_formal_training_budget \
                --run-id m91_formal_training_budget_v1 \
                --short-run m91_short_seed0 --short-run m91_short_seed1 \
                --short-run m91_short_seed2
短跑 run      runs/m91_short_seed{0,1,2}/（各 3 批，seed 0/1/2）
checkpoint    runs/m91_ckpt_seed{0,1,2}.pt（各 850 533 B）
计时          独立计时器读数；区间内**未**启用 tracemalloc；峰值内存取 ru_maxrss
```
