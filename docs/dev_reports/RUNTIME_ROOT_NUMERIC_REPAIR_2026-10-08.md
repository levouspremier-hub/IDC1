# 两阶段数值契约与资格覆盖的系统修复

本轮用户要求从根源修复反复正式训练失败。代码修复和本地验收已完成，独立分支 `p5-eval-viz-m6-p2c-root-numerics`；新 v15 候选明确 `formal_training_ready=false`，尚未完成同源主机完整资格。本报告不宣称所有未知输入无失败，也不宣称正式三种子完成。

## 根因与修复

原失败暴露了两层缺口：A 的近整数/微小原矩阵违规见证能通过 1e-6 物理检查，却被直接用于 B 的严格目标上界；固定诊断策略重复四小时与三个 seed 各8批，覆盖不到正式学习后续的状态与动作。另发现 A 原来沿用相对 MIP gap，其“optimal”标签不足以证明下一阶段绝对 1e-6 的主目标要求。

新 `planning/numeric_contract.py` 独立验证整数见证：不改连续变量，将可认证的整数表示规范为精确整数，再以已有 solver 1e-10 检查全部原矩阵及边界。8ulp只是机器表示审计界；超出该界但严格小于冻结 solver 精度的误差，必须由完整原矩阵的整数见证证明，>=1e-10 的记录 ghost 拒绝。A 实际目标必须与返回 fun 一致，lower bound 有限、方向正确且绝对 gap 不超过原 offset 容差；inventory 主目标和 reachability 设置零相对/绝对 MIP gap。A、B 和 reachability 的候选检查均有独立审计，不合格 A 不启动 B，失败停止环境/PPO的既有规则保持。

B presolve=False 的原 O 卡修复保留。九次固定对照证明关闭 A presolve 不能消除旧 ghost，还增加耗时，故没有扩展关闭 A。新机制没有额外求解或重试，没有放宽物理/业务/终点/offset，原 0.50 秒共享 deadline、PPO/奖励/采样/网络和原模型系数保持。

新资格仍保留 gate、真实 resume、48诊断、三seed各8批、共享4小时，并新增：

- `numerics`：3份原失败×7个登记动作变体=21观察，来源及fixture SHA绑定，读取实际 solver options、矩阵/目标/边界/整数声明 hash、A目标及整数/执行见证，0环境/PPO。
- `--coverage`：三个seed分别按原预登记顺序真实学习两个完整origin周期。当前212起点、每批4回合，共106批/424回合/20352 transitions/1696 Adam/106乘子每seed；沿用原算法与预算，独立 `runtime_learning_qualification` scope/role，不能冒充正式训练或评估权重。
- 严格 release 重新读上述证据及真实 final checkpoint：确认每origin两次、精确原顺序、实际游标/全部Adam/乘子/RNG/finite/来源绑定和全部质量；只有旧固定策略资格、缺少真实CP或错角色都拒绝。原训练工作量仍512批，没有用资格替代正式训练。

## 证据

- 红提交 `3087678`：原代码不能拒绝 A ghost 且缺数值/学习覆盖门禁，注册失败日志保留。
- 核心修复 `d7f5fe03d1174c9a0330be001ad12284140e95fd`；候选冻结 `76c1319`，v15三预算均独立校验绑定通过；实际资格仍使用 .50，无放宽预算。真实资格CP/RNG/错角色验收提交 `7d70123`。
- 最终相关回归131项通过；另有122项（含较慢两批恢复一致性）和最终26项新契约测试通过。按JUnit classname/name去重共148项，不把重复运行累加。Ruff/mypy通过。源码完成后的重点回归覆盖终点、服务、泄漏、稀疏矩阵装配、原失败、deadline/选项和发布门禁；未在Mac运行重型全资格。
- 本机新候选数值验收21/21 A/B optimal、可执行、整数残差0、见证检查通过，.166549—.447753秒，全部使用原 .50 预算且0环境/PPO。三个 recorded 输入的 A 全部数学数组（CSR data/indices/indptr、目标、上下界、行界和整数声明）与改前逐字节一致；不是通过改变模型通过验收。
- 新标准证据 `runs/m6p2c_root_numerics_v1/` 与 `runs/m6p2c_root_numerics_acceptance_v15_local_v1/`，含五类标准产物、原红绿/阶段错误日志、XML、输入/矩阵证据、固定21观察和 receipt hash。success仅表示本地工程验收；`host_complete_qualification_passed=false`。

中间发现并保留：观察脚本首次误传未支持参数（0完整观察/环境/PPO）；固定8ulp拒绝了原不可达终点服务测试的1.754e-14正常求解表示，改为严格整数见证证明后原断言转绿；一次验收命令写了不存在的测试文件（0测试），修正命令后执行真实验收。未删除/修改原测试断言或通过反复采样覆盖失败。

## 边界与后续

现有 v14 正式 job `runtime-formal-three-seed-v6-v14` 固定 `5d10cb0bffe81cbec530cbe13375f46e59d3fbca` 已由原控制器启动，独立继续，不受本地分支编辑影响。其科学资格与训练产物属于 v14，不能继承为 v15或换签恢复。不要在新源码 checkout 用旧 candidate/release 绑定失败推断远端 v14 训练失败；历史审计使用其固定源码。

本卡止于可审阅源码修复和本地证据，依据 `docs/VSCODE_CLAUDE_EXECUTION_PROTOCOL.md` §2.3/§6：M6语义卡测试通过后审阅停点；提交不代表下一卡放行。审阅后，v15需在主机全局单槽完成同源全门禁/真实恢复/21数值/48诊断/3×8/3×106/4h及完整SHA验签，才可严格发布新release并fresh正式训练。新源码没有 v15 正式release，不自动混用旧成功，也不把本地数值验收当完整运行可靠性证明。

受保护目录、顶层shim、env.step均未修改；旧失败/SOC/队列/CP全部保留，paper-baseline未改/未合并。源码与候选及本卡文档可按本卡提交逆序 `git revert`；先撤新候选再撤核心源码，旧 O 卡修复不撤销。卡结束 status 应为空。
