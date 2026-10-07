# IDC 项目交接：修复、资格与三种子正式训练

> 快照时间：2026-10-08 07:24:32 CST（Asia/Shanghai）。状态会继续变化；接手时先查实际job、进程和receipt。
> 当前主任务未完成。用户已授权必要修复、对应验收后自动推进fresh三种子串行正式训练，无需再次请求启动许可。

## 1. 接手先看这一页

项目正在M6/P2C阶段处理正式训练可靠性、运行效率和主机交付。**当前最终候选是v14，主机完整资格已经成功；全量回传仍在进行。正式release v6与新512批任务尚未生成。**

唯一直接控制器负责“补验与资格验签 → 严格发布 → 最小资产交付 → 容量复验 → 正式启动”；小时巡检只辅助它。控制器存活时只读检查，不另起修复/发布/训练或竞争collect。

2026-10-05的[旧交接](CHATGPT_HANDOFF_RUNTIME_REPAIR_2026-10-05.md)与[旧执行计划](RUNTIME_REPAIR_EXECUTION_PLAN_2026-10-05.md)是历史来源。其中“暂停/暂不自动长训”已被用户后续明确授权三种子长训覆盖。用户讨论过并发，但最后明确继续原计划：**SSH全局单槽、seed0→seed1→seed2串行；Mac仅轻量开发、审计、传输。**

| 项目 | 当前值 |
|---|---|
| 仓库 | `/Users/levous/Desktop/IDC` |
| 活动分支 | `p5-eval-viz-m6-p2c-runtime-repair`；不改/自动合并`paper-baseline` |
| 写交接前HEAD | `f436a93fdd36433421e401ef5e9a8164d3334599`；本交接会有新的纯文档提交 |
| 最终候选固定revision | `3d892a4f82b92808e7138fdf953af634e73cd1b8`（v14） |
| 当前资格job | `runtime-campaign-qualification-v14`：succeeded/exit0 |
| 主机资格controller | 历史PID1415545；任务已终态，勿重启 |
| Mac全回传watcher | PID12265，仍存活；实际PID以ps核验 |
| 直接控制器 | `runs/m6p2c_v14_release_delivery_v1/controller.py`；uv12284/Python12285存活 |
| 控制器阶段 | 已验签十五次原失败补验，等待完整资格回传 |
| 资格完整receipt | 3619文件，4864627699字节；主机receipt已生成 |
| 此刻传输快照 | 1380/3619文件、1883400221/4864627699字节大小匹配；**不是全SHA通过** |
| native完整回传凭据 | `local_transfer_receipt.json`尚未生成 |
| 未来严格发布 | `configs/release/idc_runtime_formal_release_v6.json`，待控制器生成 |
| 未来唯一正式job | `runtime-formal-three-seed-v6-v14`，目前不存在 |
| 未来parent/children | `m6p2c_formal_three_seed_v6_v14`；`m6p2c_formal_train_seedN_v6_v14`，N=0/1/2 |

不要把最新仓库HEAD、候选固定revision和未来发布revision混为一谈。纯文档提交不改变已经冻结的执行闭包；source改变必须重新绑定。

## 2. 项目目的、科学边界与主链

研究目标是在相同业务、物理和信息条件下，检验安全PPO加联合滚动修正对成本、购电归属碳排和后续可执行性的作用。单数据中心、20服务器组、20计算动作加1储能动作；当前formal checkpoint契约为**21动作、523观察、明确schema/role和来源绑定**。旧MARL/23维或无版本checkpoint不能进入新主链。

主链与职责：

- `scenario/`、`contracts/`：冻结场景、数据来源和接口。
- `envs/`、`idc_model/`：真实任务执行、功耗和物理状态。不要重置SOC/队列或跳任务制造成功。
- `planning/`：有共享deadline的修正器；A主偏移阶段、B经济阶段及inventory/service约束。
- `safe_rl_v2/`：新训练入口、PPO原始动作概率记录、库存终点诊断、失败保存和更新计数。
- `checkpointing/`：版本/schema、真实训练状态、RNG和执行绑定。
- `scripts/runtime_qualification.py`、`scenario/runtime_release.py`：资格与严格发布。
- `scripts/idc_remote.py`、`runs/writer.py`：单槽主机任务、传输验签与标准产物。

完整训练不是性能/收敛结论。validation/test仍封存；正式训练验收后，再按[评估协议](M6_EVALUATION_PROTOCOL.md)与冻结实验矩阵推进，不自行开展最终评估。

## 3. 已完成修复：不要重写

| 阶段 | 已确认成果与证据 |
|---|---|
| 晚批超时/构模 | 等价稀疏装配、共享deadline审计及晚批212-origin缓存验收；原矩阵/目标/排序/容差保留。见[G卡](task_cards/M6.P2C-G.md)和[报告](dev_reports/RUNTIME_TIMEOUT_REPAIR_2026-10-05.md)。 |
| 主机恢复/网络 | WSL已迁到D盘，根rw，C/D真实余量恢复；任务裸仓库使用Mac SSH reverse SOCKS relay。见[H卡](task_cards/M6.P2C-H.md)。历史C2MiB不是现状。 |
| 交付容量 | 退役输入同SHA/size共享inode保留全部路径与历史，逻辑释放37.33GB；只含必要资产/canonical资格的最小集合及投递预留。见[J卡](task_cards/M6.P2C-J.md)和[报告](dev_reports/RUNTIME_DELIVERY_CAPACITY_2026-10-06.md)。用户清理解除容量阻碍，没有执行管理员compact。 |
| 第一、第二次数值问题 | 内部inventory A/B精度1e-8→1e-9→1e-10；不改变物理/服务/终点/offset1e-6/验收1e-6、预算或PPO。各版本原失败、对应资格与旧seed0成功保留。见[K卡](task_cards/M6.P2C-K.md)、[N卡](task_cards/M6.P2C-N.md)。 |
| 资格输入交付 | 资格专用104文件/468217305字节sealed集合补齐固定r7诊断policy和受控fixture，严格loader预检通过。旧r7只作固定诊断，不给新正式初始化。见[L卡](task_cards/M6.P2C-L.md)。 |
| 旧确定性测试 | 原legacy .05测试受时间分支/主机负载影响；用户退出Overwatch后一次性原测试验收通过，没有加预算或改比较。历史唯一因果未完全确认。见[M卡](task_cards/M6.P2C-M.md)。不再等待用户关闭游戏。 |
| 最终B presolve修复 | 原7584失败的A已完全整数，且满足B原矩阵，B presolve仍误判不可行。仅关闭inventory阶段B presolve，A/reachability/noninventory、1e-10和原deadline保持，一次求解无重试。见[O卡](task_cards/M6.P2C-O.md)和[最新报告](dev_reports/RUNTIME_SEED1_PRESOLVE_REPAIR_2026-10-08.md)。 |

O卡先红`f610554`，最小修复`fd13936`，共享options直接helper接线`e4ed897`，最终v14冻结`3d892a4`。92对应回归/Ruff/mypy通过；binary测试采用8个IEEE-754 ulp以容纳约4e-16表示差异，正式物理容差没有放宽。新旧三个输入六组固定对照的A/B矩阵、边界、目标和整数声明逐字节相同。

关闭B presolve增加了经济阶段计算量，可能提高允许的A保留比例。v14三个8批短训的A保留次数为4/11/4，fallback均0，全部质量通过。正常合格A保留不阻断，不以它重新训练；后续仍应统计来源比例与经济效果，不能把允许降级等同于B总是求得经济最优。v14正式512批的总耗时尚无最终实测，不用旧v12约4h51m承诺新版速度。

## 4. 当前v14资格与十五次补验

### 原失败补验：已完整通过并验签

- job：`runtime-seed1-three-failure-presolve-acceptance-v1-r3`，固定最终v14 revision。
- 三份原失败各5次，共15次；A/B均optimal、可执行、整数残差0。
- .278915—.388127秒，仍为0.50共享预算；原A数学输入相同，输入前后SHA相同，CPU1、0环境/PPO。
- 13文件/55458字节全回传验签；在`runs/remote_runtime-seed1-three-failure-presolve-acceptance-v1-r3/`。勿重跑。

### 完整资格：主机已通过，Mac完整回传待完成

- job/run：`runtime-campaign-qualification-v14` / `runtime_campaign_qualification_v14`。
- 门禁3249 passed、47 deselected；三边界真实resume通过；48诊断通过。
- seed0/1/2各8批、128Adam、8乘子、32合格回合。
- shared soak14409.392730389秒、1082共享回合；前后各24，共1130回合全部通过，无新失败。
- 资格sealed asset：`bd21c3a8d561ea52b0c7f66b9f9ccee3f43e60224397f7bbd8649542a43abbaa`。
- Mac原始回传：`runs/remote_runtime-campaign-qualification-v14/`，日志`.idc_remote_logs/runtime-campaign-qualification-v14.log`。
- 大小匹配仅代表传输进度。必须native完整凭据`all_files_verified`、receipt逐SHA和资格/补验绑定都通过，控制器才会发布。

控制器与host launcher已compile验收，当前存活、没有error。正常传输在增长，不重启、不竞争collect。若控制器失败，先读具体error/阶段，保留现场；仅做必要交付或接口修复，不以重跑通过的科学资格代替处理。

## 5. 必须保留的历史：这些不算新版三seed完成

| 版本/任务 | 真实终态 |
|---|---|
| v8原seed0 | 371完整批后超时，20partial，失败批0更新；原失败/CP/RNG/来源保留。 |
| v10/v2-r2 | seed0完整512成功；seed1完成100批后9120 step18 Aoptimal/Binfeasible、66partial，seed2未启动；69文件完整回传。 |
| v11/v3 | seed0完整512成功；seed1完成58批后1008 step18 Aoptimal/Binfeasible、18partial，seed2未启动；66文件完整回传。 |
| v12/v4 | seed0完整512成功；seed1完成39批后7584 step1 Aoptimal/Binfeasible、49partial，seed2未启动；65文件376266452字节全回传验签。 |
| v13资格 | 因已知共享options AST接线问题受控SIGINT/exit130，未完成。原finalization receipt在残留pytest停止后有单log尾变化；旧receipt/日志不改，完整post-stop独立归档验签，不能宣称资格通过。 |

最近v12 seed1失败步未执行env、失败批0更新；before/latest实际next39、全Adam624、乘子39、finite及原v4binding匹配，21维raw和oldlogprob4.46904182434082保留。failed SHA：`a2e73b204e4bf0e4c2f61392c3d0e014adbabb5306d1fb37e647c3e5ea919aa8`。

关键证据目录：

- `runs/remote_runtime-formal-three-seed-v4-v12/`：完整失败parent。
- `runs/m6p2c_formal_seed0_terminal_audit_v12_v1/`：独立41文件354584286字节及seed0真实final CP/更新/全部质量。
- `runs/m6p2c_v12_seed1_numeric_failure_triage_v1/`：失败保全审核、固定探针、红绿日志、`numeric_diagnosis_supplement.json`；core审核success只表示保全，不表示训练成功。
- `runs/m6p2c_v13_qualification_interruption_audit_v1/`：完整18文件181855字节post-stop快照及原receipt/最终log。

原v12 seed0 report SHA：`a6db8f4e8144256ad5aa16402de074693dfa9915fc3b3d8a7fd1809e70d6364a`；final CP：`f4b945cb1d0e1902500bc4028256a71270df0b64568e2f3586b78848d5f159bd`。
原v4 release SHA：`38706e3bafe052d77f1c7bd5abaa5ac846d08c23cbbfda90a5c6f0cb8437f8bc`。这些都保持原绑定，不改签、不用于新初始化、不混成v14三种子结果。

## 6. 主机、容量、网络与数据保护

- SSH：`idc-y9000p`，当前Tailscale IP`100.73.26.18`，用户`w1877`。主机根`/home/w1877/idc-host`，job在`jobs/<job>/`。
- 活动科学重型任务只有一个，使用全局`execution.lock`；不要另起主机重型回归或Mac全套重型资格。
- 任务活动产物在`workspace/runs/`；终态移到`output/runs/`，旧workspace缺文件不等于产物丢失。
- Windows WSL发行版在`D:\WSL\Ubuntu-IDC`。测实际`Get-Volume C,D`，不能用guest虚拟df证明宿主余量。
- 最近C136260374528/D173389398016字节，根rw；新投递必须C≥30GiB、D≥30GiB+实际输入copy+8GiB依赖/输出预留，准备后30GiB守卫仍生效。
- `.idc_remote.json`当前选择资格专用bd21集合；未来formal最小asset由控制器生成，不手改为含raw/canonical重复的大d314集合。
- 退役输入hardlink只允许同SHA/size、原路径和历史保留、终态且不再写入；活动worker实际copy隔离。不要删失败、sealed资产、worktree、checkpoint或历史数据腾空间。
- Git裸仓库仅loopback SOCKS`127.0.0.1:19065`，依赖Mac SSH reverse relay，历史relay PID34510（实际核验）。按H卡恢复并验证Git fetch；不改Windows/global proxy/firewall/电源CPU/用户游戏或服务。
- SSH可能间歇超时。先查同一唯一job/现有controller和传输增长，不重复投递。存活且正在增长的传输不重启、不抢collect。
- 不再等待管理员compact/备份或再报告历史C2MiB/D14.6GB；当前没有这项外部阻碍。

## 7. 接手动作与只读命令

从仓库目录执行，先读[AGENTS.md](../AGENTS.md)、[实施计划](IMPLEMENTATION_PLAN.md)、[协作协议](VSCODE_CLAUDE_EXECUTION_PROTOCOL.md)、[O卡](task_cards/M6.P2C-O.md)和[最新修复报告](dev_reports/RUNTIME_SEED1_PRESOLVE_REPAIR_2026-10-08.md)。

```bash
cd /Users/levous/Desktop/IDC
git status --short
git branch --show-current
git log -3 --oneline
git diff --check
uv run python -m scripts.idc_remote status runtime-campaign-qualification-v14
tail -60 runs/m6p2c_v14_release_delivery_v1/controller.log
ps -p 12284,12285,12265 -o pid,ppid,etime,state,command
```

只读核对native凭据与完整SHA：

```bash
uv run python - <<'PY'
from pathlib import Path
from scripts import idc_remote as r
p = Path('runs/remote_runtime-campaign-qualification-v14')
f = p / 'local_transfer_receipt.json'
print('完整回传凭据存在:', f.exists())
if f.exists():
    t = r.read(f)
    assert t['all_files_verified']
    assert t['remote_status']['state'] == 'succeeded'
    r.verify_receipt(p, r.read(p / 'receipt.json'))
    print('全部源产物SHA通过')
PY
```

按实际状态推进：

1. **控制器存活、回传增长**：只读等待，不另投任务、改执行闭包或重复补验。
2. **控制器失败**：读取report/error/log、完整保留；必要修复与对应验收后续已授权流程，不用无限重跑挑成功。
3. **release生成/新formal出现**：先验release严格binding、最小输入SHA、真实容量和fresh启动；查实际Python/journal/批前latest CP/质量，不能仅报submitted。
4. **每seed终态**：独立完整回传receipt，全2048回合门禁和真实final next512、全部Adam8192、乘子512、finite/RNG/来源绑定、core artifact重验。parent完整回传继续原watcher。
5. **三seed都完整通过且parent完整回传验签**：再给最终完成报告；validation/test仍依冻结评估协议另行推进。

每seed必须512批、98304 transitions、8192 Adam、512乘子、2048合格回合，前seed完整质量和产物验签后才下seed。主任务尚未完成，小时automation `idc`保持ACTIVE：正常进展/已知不变阻碍安静，只在完整放行、新seed启动/完成、新异常、无法修复/需外部动作或三seed最终验收时通知。

## 8. 故障分级、修复纪律与接手提示

普通警告、指标波动及允许的合格A保留：保留记录并继续，结束后统一审计；不要一有异常就重新跑全部资格。网络/传输/容量修复未改执行源码：只验受影响操作，继承有效科学证据。

无可执行修正、物理/非有限/服务/终点失败属于阻断：失败前停止环境/PPO、完整保留并定位。新的执行源码修复先五项卡/提交红回归/最小修复/新候选与对应资格，禁止旧CP换签恢复。若再次数值失败，审视数值方案与覆盖，不能循环收紧一级精度加重复四小时。减少验收范围需要明确证据和兼容既有发布门禁，不能暗中放宽或跳阶段。

共同红线：不删失败/补采挑成功、不清队列重置SOC跳任务、不放宽接入容量/SOC/充放电互斥/服务/终点/offset/容差、不读未来真值、不改奖励网络PPO制造通过；buffer始终保留a_raw及其logprob，a_exec只记录执行侧。`marl/`、`grid_model/`主逻辑、`legacy/`和顶层shim保持不改；改env.step须先提交失败测试。

Git：每卡先status/branch/HEAD/diff、五项卡提交、先测试后可验证批次与显式add、验收记录提交；禁止add-A/add-dot/amend/rebase/reset-hard/checkout--/clean/force push，结束status clean。不自动合并paper-baseline。

可直接发送给接手代理：

> 读取`/Users/levous/Desktop/IDC/docs/PROJECT_HANDOFF_2026-10-08.md`，核对最新实际job/控制器/完整receipt后继续已授权流程。不要重写已修复项、复投已有任务、跨版本恢复或混旧seed0。主机单槽、三个fresh种子串行；当前v14主机资格已通过，正在全量回传，存活直控负责验签发布交付启动。小时巡检仅辅助；新阻碍先必要修复与对应验收，无法修复/需外部用户动作再报告，正常持续进展安静。
