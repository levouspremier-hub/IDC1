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
host-portability-gate-v1：2882 passed、275 failed、20 errors；集中于 B6 重算逐位比较。允许新增 scenario/portable_numeric.py、scenario/exogenous_drivers_b6.py 的验证函数和加载函数、tests/test_host_portability.py 及 tests/test_m13feb_b6_exogenous.py 的协调篡改回归；不改生成公式。历史 materializer_revision 只可在注册原 revision、原 parquet SHA、原脚本字节以及排除上述两个验证函数后的完整 AST 一致时接受；新验证版本单独登记。保留 dirty、路径、来源、checkpoint 校验。加载返回原冻结 parquet 数值。已注册原资产不变，协调改写 parquet/hash 即使仅改变1 ULP也必须拒绝。先失败回归再实现，回滚 ada06c0。

## 下游 revision 级联修复
v2 门禁已消除 B6 数值重算失败，但 refs_v4 / policy-v3 / v5 的源码集合覆盖 B6 验证器，导致 revision 漂移。允许 scenario/b6_refs.py、scenario/formal_scenario_b6.py、scenario/b6_split_manifests.py 的 revision 查询函数接入注册生成配方核验。必须仍使用完整原 SOURCE_PATHS 进行 dirty 与 Git revision 查询；只在完整源代码（仅排除三个 revision 查询函数和两个已登记 B6 消费验证函数）与原注册 commit 一致时返回原生成 revision。任何生成代码变化返回新 live revision，旧资产必须拒绝；不改变 SOURCE_PATHS，不重写资产。新增失败回归先提交再实现；回滚 cb76ecc。

## 归因、实现与当前验收
- 原锁 pandas2.3.3 的哈希自动推断缺陷：最小公开输入 `81e3104049863b72` 在 WSL inferred exit -11、显式 str exit0；上游 https://github.com/pandas-dev/pandas/issues/62617。ea81708 失败规格，1ee90e5 只固定四个标识列字符串类型，分钟数值列与排序/求和不变。
- PV max abs 3.765876499528531e-13 kW（520行、最多424 ULP）；wind max abs 5.684341886080802e-14 kW（366行）。carbon/arrival 逐位一致。以冻结500/800kW物理尺度的8×float64 epsilon绝对界限验证，保留零位置、finite、shape/dtype/timestamp及离散量精确性；1e-9kW和零位置变更回归必须拒绝。
- d4361a9/cb76ecc：新 portable-frozen-verifier-v1 仅接受原注册 parquet SHA 与原配方commit；校验原源码SHA、完整AST（仅排除两个已登记消费验证函数）和原materializer脚本字节。返回原磁盘frame；验证版本/revision/source hash另记，未重发任何冻结资产。
- v1完整门禁2882 passed/275 failed/20 errors，均受B6逐位比较级联；v2为2894 passed/270 failed/20 errors，主要为下游 live revision 漂移；v3在Ruff因同时期capsule测试import排序失败，未进入pytest。所有失败完整保留、回传验签；capsule lint已由其任务修复，当前全链Ruff通过。首次host-portability-tests-v1误以文件作为命令，uv仅运行定义，虽exit0但**无pytest**，不计为通过；正确v2为56通过、1个slow浮点比较失败，原始失败保留。
- 975cb2c：refs_v4/v5/policy-v3只在完整原SOURCE_PATHS代码与原登记配方一致时保留原生成stamp；保留原dirty集合和原Git查询。生成代码变化仍给新live revision，旧资产拒绝。缓存仅用于不可变Git blob与按源码文本键的AST，未缓存可变资产校验。
- 当前本地120项 host portability / formal B6 / cutover 回归全绿，包括伪造revision、生成源码变更、dirty helper和协调1ULP改写hash拒绝；全链Ruff与mypy（194源）通过。25个正式checkpoint绑定源和uv.lock与9b7b59a一致。runs/m6_host_portability_v1/保存五类审计与receipt。
- 主机Tailscale Online=false，SSH重复连接超时；原v3失败产物已完整回传。已准备新的固定SHA v4请求（本机忽略日志目录），**尚未启动**，待主机恢复后完整make check、slow重建与真实CSV模板精确等价性复核。当前不声称WSL验收全绿、不宣称seed0根因解决，不启动正式训练/新seed/held-out。

## 到达映射器级联修复
主机恢复后 v4 完整门禁：2963 passed、206 failed、20 errors，主要因 arrival_mapper 的完整 SOURCE_PATHS 覆盖前述验证器，导致 source_revision 漂移。扩大允许范围至 scenario/arrival_mapper.py::mapper_code_revision 和 portable_numeric 的注册消费者表；保留完整路径、dirty 校验和全部任务映射逻辑，仅在完整配方与原注册 a806599080bfd43796ba3c2ad696a19e2ba28f42 等价时保留原 stamp。先提交失败回归，针对性测试后再完整主机验收。另有 M54g 0.05 秒真实求解夹具 timeout，独立按原预算复测，不扩大预算。回滚点 b502fe2744a21350a78a210f555f7b923e45c689。

## M54g 选项接线夹具稳定性
原测试 test_both_raw_projection_stages_use_deterministic_mip_options 在 v4 主机和本机针对性复测均真实 timeout；这是要求较大随机实例必须在0.05s完成的夹具脆弱性，不据此认定正式0.25s预算失效。允许仅修改该测试，复用 tests/test_m44_corrector.py 的两个任务、两个组、24步合成 snapshot，保留真实 correct()/milp、A/B至少两次、确定性选项断言与原0.05s预算；其余复杂实例及跨进程slow保持原样，不改生产代码/物理约束/任务状态。先留存原失败，新的针对性和完整门禁验证。回滚986d19a。

## 主机恢复后的针对性证据
- v4 完整失败产物已全部回传验签：2963 passed、206 failed、20 errors；映射器修复先87359bc失败规格、再986d19a实现。mapper 原完整 SOURCE_PATHS 与容量源 task_model 保持不变。
- host-portability-focused-v1：64 passed、1 failed，唯一失败为原M54g大随机夹具timeout，未执行后续项；原XML和全部产物保留。
- 0311745只替换真实A/B选项接线测试的输入夹具为已有两任务合成snapshot，0.05s预算、真实milp调用和>=2阶段断言保留，生产代码及其余复杂/slow测试不变。
- host-portability-focused-v2（03117453b317b29e96889bbd096e65bb99d3aec6）：76 tests、0 failures/errors；全部回传验签。Mac mapper/portable为64项通过，选项测试12项通过。25个正式checkpoint绑定源与uv.lock对9b7b59a逐字节不变。
- host-portability-gate-v5固定同一0311745，单槽运行完整make check、slow真实数据重建及CSV冻结模板逐位复核，Ruff/mypy已通过；其终态以新审计runs/m6_host_portability_v2为准，不把针对性通过当作完整门禁完成。
