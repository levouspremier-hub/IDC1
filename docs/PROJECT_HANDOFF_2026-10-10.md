# IDC 项目转移文档：连续训练失败、数值修复与当前现场

> 核对日期：2026-10-10，Asia/Shanghai。接手者必须重新查询真实任务状态。
> 本文是交接快照，不是启动脚本。执行授权来自用户：根因修复、新源三个短跑及完整准入全部通过并验签后，自动启动 fresh 三 seed 串行正式训练。
> **首页结论：v26 三 seed 短跑已通过，但完整准入失败；新版正式训练没有启动。行归一化修复仍处于未提交、未完成全验收的状态。**

## 1. 接手时必须知道的当前状态

| 项目 | 实际状态 |
|---|---|
| 仓库 | `/Users/levous/Desktop/IDC` |
| 分支 | `p5-eval-viz-m6-p2c-root-numerics`；不自动合并 `paper-baseline` |
| 文档开工前 HEAD | `c4f570f`；后续文档提交不等于科学源码提交 |
| 已冻结的 v26 科学源 | `d0973828b30c0cb763a3876128f53970b8eec2d5` |
| v26 候选冻结提交 | `041d180` |
| v26 candidate050 SHA256 | `bec45f02b5b86f552112a6eecf3c1ffbc2fb9937875a7360250d73ec0eabb2d2` |
| v26 主机投递 PIN | `66eb909fd9c1297fe75c89b87a27d6368dc5039c` |
| v26 短跑 | `runtime-fixed-short-v26-v1`，succeeded / exit0 |
| v26 完整准入 | `runtime-campaign-qualification-v26`，failed / exit1 |
| v26 直控 | `runs/m6p2c_v26_direct_launch_v1/controller.py`，终态 `stopped_on_failed_gate` |
| 原 PID | Mac23606 / watcher25293 / host3182155 均属历史标识，不用于证明当前存活 |
| 最新正式发布/任务 | v7 未生成；`runtime-formal-three-seed-v7-v26` 未启动 |
| 新版 v27 | 尚无冻结 candidate；尚无新主机短跑或完整准入 |
| 当前工作树 | 有四个此前遗留的未提交科学文件，见第 5 节；不丢弃、不擅自重置 |

**过期记录警告：**[2026-10-08 交接](PROJECT_HANDOFF_2026-10-08.md)首页的 v14 当前状态已过期；[详细运行报告](dev_reports/RUNTIME_FIXED_SHORT_LAUNCH_2026-10-08.md)末尾和 `runs/m6p2c_fixed_short_formal_launch_v1/report.json` 仍停留在 v26 qualification running。最新终态应以 [v26 直控报告](../runs/m6p2c_v26_direct_launch_v1/report.json)、[主机终态状态](../runs/remote_runtime-campaign-qualification-v26/remote_status.json)及 [v26 独立失败审计](../runs/m6p2c_v26_coverage_failure_audit_v1/report.json)为准。

## 2. 项目目标与不可变边界

研究新主链安全 PPO 与两阶段滚动修正对业务服务、成本和碳排的影响。当前契约是 **21 维动作、523 维观察、版本/role/source 明确绑定的 checkpoint**。阶段 A 最小化对原始动作的偏移，阶段 B 在 A 偏移上界内优化经济目标。

- `planning/`：MIP/LP 数值契约、原矩阵证书、共享 deadline。
- `safe_rl_v2/` 与 `checkpointing/`：新训练入口、真实优化器/乘子/RNG、恢复绑定。
- `scenario/runtime_release.py`、`scripts/runtime_qualification.py`：冻结候选、完整准入和严格发布。
- `scripts/idc_remote.py`：主机单槽、传输与逐文件 SHA 验签。

必须读 [AGENTS.md](../AGENTS.md)。不能放宽容量、SOC、充放电互斥或任务约束；不能清队列、重置 SOC、跳任务或读取未来真值；归一化冻结且共享。`marl/`、`grid_model/` 主逻辑、`legacy/` 和顶层 shim 不改。PPO 保存 `a_raw` 与原 log-prob，执行动作不得覆盖它们。原物理模型/阶段目标、严格原矩阵 `1e-10` 证书、offset `1e-6`、A/B 共享 `0.50s` 预算保持。validation/test 封存。

成功训练仅证明执行与训练状态符合协议，不能直接推出收敛或性能优势。

## 3. 连续失败经过：修复了什么，又在哪里再次失败

下面的“原点/步”指 origin / episode step；覆盖批数指失败前完整提交的学习批次。每次新源都重新做对应短跑，不跨版本继承成功。

| 版本/阶段 | 失败与根因证据 | 对应修复与下一步结果 |
|---|---|---|
| v10/v11/v12 历史正式 | seed0 曾分别完整跑完，seed1 在新状态出现近整数储能模式、A 偏移证书或 B presolve 问题。旧 seed0 成果真实保留，不能拼成新版三 seed 成功。 | 收紧内部 inventory 数值契约并逐步加入整数/原矩阵/目标见证；旧资格没有充分覆盖真实学习后的状态。见旧交接历史章节。 |
| v14 正式 seed0 | 完成32批后，6192/47 极接近 SOC 终点时失败。此前短训与固定策略4h soak通过仍未覆盖该状态。 | v15–v17 本机修复继续被终点扰动、原整数见证或 B 不可行拦下，未把本机失败版冒充主机通过。 |
| v18 短跑 | 加入 H=1 储能域、网络充电界及以 A 为连续原点的 B 表述；44固定观察通过，但短跑在384/36出现非规范整数见证。 | v19仅在既定前置条件下，选择整数候选并调用一次原完整连续 LP，返回仍需原矩阵/目标证书。 |
| v19 短跑 | 55观察通过；短跑480/45再次出现储能模式/净能量支持问题。 | v20加入原 SOC 平衡蕴含的整数能量 cover，保持原整数可行域。 |
| v20-r2 短跑 | seed0完整8批；seed1仅完成2批，576/44，A ghost约 `4.96e-11`、正charge约 `1.45e-10`。nearest z0=0 的原LP确实不可行。 | v21按必要净充/放方向与已有正流选择支持模式；只在唯一LP前选候选，失败不换模式再试。 |
| v21 门禁 | 三 seed 短跑全过，但完整门禁 Ruff 因新 tests 的7处 import 分组空行失败。 | 仅格式修复且 AST 等价，同源成功 short 经重新验签复用；新 r2 从完整门禁实际执行。属于检查范围遗漏，不是科学失败。 |
| v21-r2 完整准入 | seed0覆盖106批；seed1完成10批，1968/18，B ghost约 `8.21e-11`、nearest原行残差 `4.34e-9`；唯一LP真不可行。 | v22修复“两流严格0、模式仍为分数”的必要净能量支持。原/平移/关闭presolve的LP对照都不可行，证明不能简单归因于坐标。 |
| v22 完整准入 | 三 seed short全过；seed0覆盖106批，seed1完成32批，6336/2失败。未来必须服务的功耗没有被充电网络上界充分扣除；未来净能量差仅约 `1.83e-10` 却数学上必须当前charge mode1。raw ghost绕过reserve，给虚假A下界。 | v23加入原 backlog 平衡和 upper 蕴含的服务下界、保守充电上界与必要整数模式。原flow LP可行但主目标恶化约 `.00593`，旧证书正确拒绝，不改证书制造成功。 |
| v23 完整准入 | 三 seed short全过；seed0完成68批，3072/18，SOC约 `50.00000000049`，H30。A原整数/原行/主目标全过，B presolve=False报告不可行；完整B、固定A整数LP及连续LP均可行。 | v24对全部输入统一冻结B presolve=True。诊断中的冗余行/列消去等没有进入生产；不按seed切换、不增加求解重试。 |
| v24 完整准入 | 三 seed short全过；seed0覆盖106批，seed1完成13批，2544/23。B optimal带ghost；v3按总净充强制当前mode1，唯一LP真不可行。未来已选充电容量其实足够，nearest原LP可行。 | v25仅当nearest模式的向外总功率上界仍不足净需求时启用既有模式支持规则。容量足够只允许保留候选，不能当可行证明；仍唯一原LP及严格证书。 |
| v25 完整准入 | 三 seed short全过；seed0/1均覆盖106批，seed2完成44批，8496/6。nearest LP可行且原行通过，但经济incumbent差 `3.509e-10 > 1e-10`，正确拒绝。 | v26由原business_balance、投影L1、inverse-capacity及B offset行推导储能功率域；精确binary64有理数、向外界和有效cut，保留tiny正值。原8496返回exact mode0/charge0；132观察通过。 |
| v26 完整准入 | 三 seed short全过；seed0完成103批，9744/44，H4/SOC约 `50.000000000116`。A证书全过，B presolve=True报告不可行。 | 当前Y卡：原固定A整数LP可行，极小功率列缩放仍失败；完整B行归一化候选在原失败案例通过，但全版本验收未完成。不能宣布问题已彻底解决。 |

对应任务卡：[S](task_cards/M6.P2C-S.md)（v18–v21）、[T](task_cards/M6.P2C-T.md)（v22）、[U](task_cards/M6.P2C-U.md)（v23）、[V](task_cards/M6.P2C-V.md)（v24）、[W](task_cards/M6.P2C-W.md)（v25）、[X](task_cards/M6.P2C-X.md)（v26）、[Y](task_cards/M6.P2C-Y.md)（当前未完成）。逐轮原失败、数学模型和对照均保留在相应 `runs/m6p2c_vN_*`，不要重跑失败实验挑成功覆盖历史。

## 4. 这些失败意味着什么

这不是可以忽略的“强化学习随机性”。已证实的问题主要集中在两阶段混合整数修正器：极小储能流、接近 SOC 终点、整数模式与条件 reserve 的联动，以及 solver 的有限精度和 presolve 对数值表述的敏感性。`optimal` 标签不能替代严格整数、原矩阵、A 主目标下界和 B incumbent 证书。

连续几轮出现的问题不是完全相同：有些是取整后真正不可行，有些是候选模式选择过强，有些是有效域推导遗漏已有功耗，有些是可行模型被 solver 误判，另一些是严格经济证书正确阻止虚假收益。**必须先分别判别，再最小修复；不能统一归结为提高容差或增加时间。**

短跑每seed只有8批/32回合，覆盖有限；真实学习改变动作和状态分布，后面会走到固定策略诊断或短跑没有遇到的边界。补齐13历史失败×11扰动能防回归，但仍不等于任意新状态可行。完整准入因此包含每seed106批、两轮注册origin学习覆盖及4h稳定性；512批正式仍可能暴露新的状态问题。

工程上的教训也明确：早期“本机检查通过”漏了 tests 的 Ruff 范围；多处运行报告和巡检提示滞后，把旧running快照连续传递。接手必须依据终态文件/真实进程，不依据标题、历史PID或 submitted 状态。单个旧seed0完整成功也不能证明新版本三seed合格。

## 5. 最新v26现场与未提交修复

### 5.1 已独立核实的原失败

[v26失败审计](../runs/m6p2c_v26_coverage_failure_audit_v1/report.json)：213 native文件全部SHA/size通过。seed0 before/latest 均 next103，9个优化器参数全部有state且Adam step1648，乘子103，412合格回合，0fallback、0正常A保留；实际恢复的来源、角色、排序、finite、sampling/shuffle RNG通过。失败批140partial，失败步0环境执行、失败批0参数更新。seed1/2覆盖尚未开始。

原 failed_batch SHA256：`f71cacfc7e4ed8ff4ce4e4b84dfb7dcd848c857107797a9e2c59064e14d2a07e`。原fixture：[9744/44](../tests/fixtures/m6p2c_seed0_origin9744_step44.json)。原A objective/reported/dual均 `1.038575459050627`、gap0。新域charge上界约 `2e-5`，冗余cut系数约50000；tiny净discharge约 `2.20e-10`。

### 5.2 当前修复只是候选，以下边界不可混淆

Y卡先提交 `6964ac3`，原失败红回归先提交 `c4f570f`。当前未提交文件为：

1. `planning/numeric_contract.py`：新增B正比例行归一化 `positive-max-norm-row-equilibration-v1`。
2. `planning/model.py`：B求解副本使用归一化行，原矩阵仍用于严格 witness 校验；共享deadline和求解次数不变。
3. `scenario/runtime_release.py`：拟绑定新行尺度版本并登记9744 fixture；execution version暂改v3。
4. `tests/test_m6p2c_root_numeric_contract.py`：候选绑定与行变换回归。

这些改动**尚未提交为科学源、尚未冻结新候选、尚未投递主机**。不要使用旧v26 candidate为它们签名；不能直接提交v26失败任务。

已完成的数学诊断和本机检查：

- 原B/功率列缩放：均infeasible；固定A整数LP为optimal。
- 取消连续原点平移的诊断：143例中141通过；9744 recorded与soc_plus_ulp仍失败。不能以简单取消平移作为完成修复。
- 完整B各行按正比例max-norm归一后的原9744案例：A/B optimal，恢复原坐标后原矩阵残差 `1.776e-14`、整数0、严格 `1e-10` witness通过。
- `test_m6p2c_power_coordinates.py`、`test_m6p2c_root_numeric_contract.py`、`test_m6p2c_terminal_storage_bounds.py`三文件相关测试执行至100%，exit0。
- `make check`的Ruff通过、mypy207sourcefiles通过；pytest执行被用户新消息打断，没有可引用的完整最终结果。当前无该检查进程，不得补写“make check全部通过”。

尚待完成：生产行归一化的全部143观察、完整门禁结果、变换与原精度契约的充分审查、Y卡边界/报告更新、科学源提交、新候选冻结、新源主机三个短跑与完整准入。此前“143回放正在执行”的口头进度不是通过证据；已有143结果是取消平移的诊断，不能挪作行归一化验收。

诊断路径：[原模型及功率坐标](../runs/m6p2c_v26_power_coordinate_probe_v1/report.json)、[取消平移回放](../runs/m6p2c_v26_power_coordinate_probe_v1/unshifted_replay.json)、[行归一化单例](../runs/m6p2c_v26_power_coordinate_probe_v1/scaled_real_case.json)。最后一个文件含RuntimeWarning前缀，整个文件不能直接作为纯JSON加载；应保留原日志，另导出结构化结果后才作标准验收证据。

数学正比例行变换不改变精确可行域，但浮点实现与solver绝对容差相互作用仍要审查。最终接受必须在未缩放的原矩阵、原界和原目标上通过原严格证书，不能用缩放后的残差替代。上述单例改善支持数值表述/presolve敏感性的诊断，尚不能证明所有未来状态稳定。

## 6. 流量、传输与交付问题

用户报告本月流量100GB；已做本机文件审计，**尚未取得运营商/代理计费记录，不能把逻辑字节直接等同网络账单**。

- 88个 `runs/remote_*/receipt.json` 所列终态产物总计40,763,401,310字节（十进制40.76GB）；其中 `.jsonl` 33.87GB、`.json` 6.38GB、checkpoint `.pt` 0.44GB。
- 早期v8/v10-r4/v11-r3/v12完整准入每轮回传约6.49–7.99GB；主要是大量逐步/回合日志，并非模型权重。
- 早期[原资产receipt](../runs/m6p2c_delivery_capacity_repair_v1/original_asset_receipt.json)有27,229文件、27.95GB；按SHA去重仅17.46GB，约10.49GB为同内容不同路径。里面包含17.90GB的历史remote回传目录。
- `scripts/idc_remote.py::asset_records()`收集所有终态run；旧`seed()`向每个新digest目录投递全包，没有跨digest文件复用，因此可能重复上传历史证据。
- watcher每60秒`collect`；rsync采用`-rtc --partial`，未启用rsync自身压缩。rsync会比较并传变化，不能说每分钟把全部40GB重传，但运行中持续新增大日志/变动报告仍有成本。
- 后期controller改为最小sealed资产与主机同SHA文件链接复用。v26资格逻辑输入约532.5MB，143已有文件链接、5文件新增传输；逻辑输入量也不是实际新增上传量。

40.76GB终态回传加早期大包/重复上传足以构成重要流量来源，但若100GB来自代理套餐，还须检查SSH/Tailscale/Git/依赖下载实际是否经过计费代理。当前没有线路计费归因结论。Mac本地 `runs` 占用约67GB，含本地canonical副本/诊断，磁盘占用不能直接换算网络流量。

后续传输优化应另开工程卡：记录每次rsync实际sent/received统计；进度只回传小摘要，大日志终态完整回传并压缩；输入按SHA跨集合复用，杜绝重新上传历史回传目录。保留完整原文件与验签语义；不通过删除失败/数据/CP/history工作树来省流量或腾空间。

## 7. 接手执行顺序与正式放行条件

1. 先核对分支、HEAD、dirty四文件、Y卡、v26终态状态和receipt；阅读未提交diff，保留现场。更新过期总record/详细报告/巡检上下文，明确v26已failed。
2. 审查行归一化修复，记录完整原模型与变换模型绑定、positive factors、原坐标/原目标证书和共享deadline；完成未完成的143观察与原Makefile范围门禁。失败就保存并继续根因判断，不挑成功覆盖。
3. 最小源通过后正常显式commit，再冻结新版本candidate；原v26 source/candidate/short保持历史，不跨source复用。普通push后固定科学PIN。
4. 确认唯一主机全局`execution.lock`单槽；新源主机143观察及fresh seed0→1→2各8批short全部实际执行。各seed1536trans/9参数Adam128/8乘子/32合格/0fallback，完整native/canonical SHA及真实CP/RNG/source/role/order通过。
5. 同源完整准入重新打开上述short CP验签后可复用；其余全部gate、真实resume、143观察、48诊断、每seed106批两origin周期学习覆盖、4h共享soak全部实际通过，所有文件完整回传验签。
6. 才能生成严格`configs/release/idc_runtime_formal_release_v7.json`，重新检查Windows C>=30GiB、D>=30GiB+输入copy+8GiB、root rw和最小不可变资产receipt。
7. fresh三seed串行正式：每seed512批/98,304trans/9参数Adam8,192/512乘子/2,048合格/0fallback。前seed真实final CP、所有质量与文件SHA通过后才能下一seed。禁止独立seed1/2与跨source恢复。

用户后续正式授权覆盖了早期“仅分析”的暂停要求；条件满足后无需再次请求启动许可。直接控制器仅在正式首批真实CP通过后退出，其success不等于三seed完成。正常合格A超时保留且0fallback不触发重训。完整失败必须完整保全；任何科学源/候选改变需要新版本同源准入。

原relay69815也只作历史标识；每次submit前实际检查WSL `127.0.0.1:19065`。缺失只恢复本任务已有SSH reverse，不改Windows全局proxy/firewall/电源/CPU/用户游戏服务。短暂SSH失联先查唯一job，不重复投递科学任务。当前旧v26流程已终态，不重启它。

## 8. 可直接复制给接手者的提示

> 接手 `/Users/levous/Desktop/IDC` 的数值根因修复。先读本文件、AGENTS、Y卡、v26原失败审计和未提交diff。v26三seed short已过，完整资格在seed0第103批后9744/44失败，正式未启动。当前B行归一化只在原案例和相关测试通过，143生产观察/完整门禁没有完成，也没有v27冻结候选。不要相信旧running巡检提示，不重启终态controller，不跨source继承short。先补齐诊断/严格原矩阵与目标证书/本机验收，再提交新源、冻结新候选、主机fresh三个短跑及全部同源资格，全回传验签后按用户原授权自动fresh三seed串行正式。重点同时处理数值域/presolve敏感性、真实学习覆盖不足、报告滞后和历史产物重复传输；任何失败完整保留。

## 9. 若同时转移到另一台机器

Git只保存已提交的代码/配置/部分标准证据，**仅clone不能恢复当前现场**。须另行保全 `data/`、完整 `runs/`（含失败、checkpoint、原模型诊断）、当前四文件未提交diff，以及这些文件所对应的基准HEAD。10月3日旧换机包不包含本轮故障与修复；本文未新建大体积换机包。

转移前记录每文件大小/SHA256和Git HEAD，传后核验；保留项目内相对路径。按当前工作分支恢复，使用Python3.12和 `uv.lock` 重建环境，不复制 `.venv` 作为跨平台运行时。SSH alias、远端根路径、loopback转发和本机连接配置须在新机单独核验，不把口令/私钥写进交接文档或Git。原主机历史job仍保留，新机接手不意味着重启它们。
