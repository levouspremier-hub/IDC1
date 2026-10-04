# WSL 平台修复与联机验收

工作分支：p5-eval-viz-m6-p2b-s0-diagnosis；未合并paper-baseline。
实际完整验收SHA：03117453b317b29e96889bbd096e65bb99d3aec6。
主机：idc-y9000p，w1877，/home/w1877/idc-host。

修复CSV哈希列被pandas数值推断引发的崩溃；跨平台浮点重算校验仅在原注册parquet SHA和完整生成配方一致时接受，加载仍返回原冻结值。refs、forecast、split及arrival mapper的原生成stamp通过完整原SOURCE_PATHS字节/AST核对保留，生成逻辑变化仍拒绝旧资产。dirty、篡改、来源、checkpoint校验保留。

M54g原随机实例的0.05秒超时在Mac与WSL复现；仅将选项接线断言的输入换为已有两任务合成夹具，继续真实求解A/B并要求至少两次调用，保留0.05秒预算和所有生产约束。其余复杂及跨进程slow用例原样保留；本次未执行全部slow。

| 验收 | 结果 |
|---|---|
| WSL focused-v2 | 76 passed，0失败/错误 |
| WSL gate-v5 Ruff/mypy | 通过，mypy覆盖194源 |
| WSL gate-v5 make check | 3190 passed，47 deselected，0失败/错误；2472.19s |
| 额外slow真实数据重建 | 1 passed |
| 原始CSV arrival模板与冻结cache | 逐位一致 |
| Mac mapper/portable与选项测试 | 64与12项通过 |
| 25项正式checkpoint绑定及uv.lock | 与9b7b59a逐字节一致 |
| gate-v4、focused-v1/v2、gate-v5全部回传 | 全部receipt验签通过，失败保留 |

证据：runs/remote_host-portability-gate-v5/{make-check.xml,focused.xml,slow.xml,trace_equivalence.json,receipt.json,local_transfer_receipt.json}；标准五类汇总与hash在runs/m6_host_portability_v2/。大产物和checkpoint不上传GitHub。

Mac代码提交推送后，工具以固定SHA在主机独立worktree执行；自动watch全部增量回传并验签。操作见docs/REMOTE_EXECUTION.md。当前单槽，不在运行中改版本或覆盖已有job/run-id。

未启动正式训练、seed1/2或held-out评估；本验收不证明seed0历史根因或长期故障已解决。Windows重启恢复及并发负载仍未实测，应独立验收。
