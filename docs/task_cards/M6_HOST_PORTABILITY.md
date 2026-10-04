# M6 HOST PORTABILITY — WSL 工程门禁修复

## 边界
允许 scripts/materialize_singapore_exogenous.py、tests/test_m13f_exogenous_drivers.py、新增独立回归 tests/test_host_portability.py 及远端验收文档。先以原锁、原资产在独立进程复现并量化，再选择保留数据含义的最小修复；必要时扩大边界须先更新此卡。

## 禁止项
不修改冻结数据、hash、依赖锁、规划/奖励/更新语义及求解预算；不移除失败测试或宽泛容差掩盖实质差异；不启动训练、held-out 或 seed1/2；保留失败和原模型。CSV 聚合必须保持分钟列排序、成员排序、非负调用数量和因果来源。

## 验收命令
原失败：host-check-v4 make check；host-check-data-v1 pytest tests/test_m13f_exogenous_drivers.py -x -vv；host-check-data-probe-v1 独立 first_freeze_failure 测试 exit -11。新增失败回归先提交，再本地针对性 pytest/ruff；WSL 完整对应测试与 make check，新 job-id、全部回传验签。

## 证据产物
runs/m6_host_portability_v1/ 五类汇总及 hash；独立远端 job 的 console、JSON 数值比较、XML（崩溃时明确缺失）、receipt。记录精确浮点误差与同机确定性，CSV 最小输入/阶段及聚合等价性；不将主机门禁问题归因为 seed0。

## 回滚点
工作分支 p5-eval-viz-m6-p2b-s0-diagnosis，开卡前 9b7b59a2be5fa462942a27cc19745d476ade2e46；使用独立提交保留回滚点，不改 paper-baseline。

## 正式加载链扩展（完整门禁捕获）
host-portability-gate-v1：2882 passed、275 failed、20 errors；集中于 B6 重算逐位比较。允许新增 scenario/portable_numeric.py、scenario/exogenous_drivers_b6.py 的验证函数和加载函数、tests/test_host_portability.py；不改生成公式。历史 materializer_revision 只可在注册原 revision、原 parquet SHA、原脚本字节以及排除上述两个验证函数后的完整 AST 一致时接受；新验证版本单独登记。保留 dirty、路径、来源、checkpoint 校验。加载返回原冻结 parquet 数值。已注册原资产不变，协调改写 parquet/hash 即使仅改变1 ULP也必须拒绝。先失败回归再实现，回滚 ada06c0。
