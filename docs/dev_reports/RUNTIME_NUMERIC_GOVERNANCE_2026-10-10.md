# 2026-10-10 数值诊断治理与放行阻断记录

本轮实际实现已经提交独立诊断Z卡和开发流程AA卡；Y科学候选尚未提交/冻结。旧v26证据保持原样，正式512训练未启动。目标主机由用户确认离线。本文不会将交接历史授权视作本次正式训练授权。

## 实现与源码处置

现场分支p5-eval-viz-m6-p2c-root-numerics，起点65043c6。开工四个未提交文件已逐项审阅并归Y：planning/model.py、planning/numeric_contract.py、scenario/runtime_release.py、tests/test_m6p2c_root_numeric_contract.py。它们目前继续留在工作树，未为“干净”打包提交。AGENTS、paper-baseline和保护目录不变；显式add，先任务卡、红测试、实现和证据记录。

Z共享runtime_qualification.replay_numeric_case，诊断调用生产模型/修正器/原证书，独立模式可收集全部失败并非零退出，正式numerics继续失败即停。来源/完整字段严格验证，实际文件与diff SHA绑定未提交科学源码。固定登记历史143、边界234、全量377，逐例重建输入、矩阵SHA、求解版本/配置/结果/证书及阶段耗时完整保存。诊断不PPO更新、不环境推进。成功轨迹抽取单独入口不求解、不PPO更新，只执行已有历史动作且要求精确来源与逐步观察一致。

AA执行static→latest→history→boundary→make check，前项失败阻断后续；冻结仍用既有候选生成和验签接口，并要求科学修改已提交且六成功锚点齐备。记录真实终态、退出码、阶段耗时，SIGTERM/中断仅清理自己创建的进程组。训练/传输未执行记not_started与null耗时。传输优化没有实施，未编造实际字节数。

Y候选改为可精确往返的二次幂行缩放，以及A连续解四分之一平移原点；不改变整数坐标。正有限缩放、零行/无穷界校验，若丢失系数或有限界位模式则整行保持1。原空区间明确保留并审计，绝不夹紧。模型原矩阵/界/目标及1e-10原证书仍唯一接受依据；.50共享deadline包含变换时间，offset1e-6、生产求解次数与原物理/科学语义不变。该候选解决已有回放，不能据此宣称整个参数域稳定。

## 可复制命令

每次用新run-id；同一成功run不可覆盖。

```sh
uv run python -m scripts.runtime_numeric_diagnostic --suite latest --run-id "numeric_latest_$(date +%Y%m%d_%H%M%S)"
uv run python -m scripts.runtime_numeric_diagnostic --suite history --run-id "numeric_history_$(date +%Y%m%d_%H%M%S)"
uv run python -m scripts.runtime_numeric_diagnostic --suite boundary --run-id "numeric_boundary_$(date +%Y%m%d_%H%M%S)"
uv run python -m scripts.runtime_numeric_diagnostic --suite all --run-id "numeric_all_$(date +%Y%m%d_%H%M%S)"
uv run python -m scripts.runtime_numeric_anchor_extract --run-id "numeric_anchors_$(date +%Y%m%d_%H%M%S)"
uv run python -m scripts.runtime_development --run-id "numeric_preflight_$(date +%Y%m%d_%H%M%S)"
make check
```

暂不使用--freeze-version；当前输入和科学提交条件均不满足。所有可运行任务用runs/<id>五类产物；report中的case_record指向完整逐例证据。

## 当前证据

- m6p2c_aa_current_preflight_v1：真实failed/exit1/102.716949s。static exit0/0.458604s；latest子进程exit0/3.490478s；history子进程exit0/39.188934s；boundary子进程exit1/59.403839s；gate未启动、冻结未执行。
- m6p2c_aa_current_preflight_v1_latest：1/1 passed，诊断内部0.270084s/exit0，原9744/44。
- m6p2c_aa_current_preflight_v1_history：143/143 passed，35.912337s/exit0。
- m6p2c_aa_current_preflight_v1_boundary：234全部保留，156 passed、78 input_source_contract_failure，55.923333s/exit1。历史快照上的12固定扰动没有删减；6成功状态各13观察仍待合格输入。
- m6p2c_y_z_aa_related_v2：95 passed，pytest40.94s，控制器41.462341s/exit0；95覆盖行变换、原证书、坐标、终端储能、诊断来源和前置流程。v1测试文件名错误exit4已保留，不作通过证据。
- m6p2c_z_current_all_v4：377例，299 passed、78 input_source_contract_failure，121.064797s/exit1；无其他失败分类。实际科学文件树SHA b50ab1ba1b713a0957199c5f05379ebeba56162b7de8e7f1f147e5fcceb3e812，diff SHA 6610bb19cfe618a8e300e5a1d48b6900a5e98c1a26b5b55736af18d1a2e7762d。
- 当前完整门禁m6p2c_y_z_local_gate_v2真实succeeded/exit0：3364 passed、47按既有规则deselected、500 warnings；pytest 4229.80s，make check 4230.881837s，完整控制器4231.124372s。额外控制Ruff/mypy exit0；全过程科学文件树/diff/lock SHA与结束现场一致。pytest.xml、make_check.log及五类产物完整保存。
- make static强制范围含全部本次控制脚本，Ruff通过、mypy214sourcefiles通过。另查全scripts目录185条既有Ruff问题，位于旧工具等范围外文件，不混称全目录通过。
- 旧m6p2c_y_z_local_gate_v1因执行期间源码变更明确中断，1522 passed/47deselected后KeyboardInterrupt，make exit2/控制器exit1；完整日志和原report/manifest备份保留，不作为本次门禁证据。
- m6p2c_z_all_v3原型299通过/82.129487s/exit0，但使用较早边界规则，不作当前377例验收。不得利用原型结果掩盖待输入案例。

## 具体阻塞与下一步

m6p2c_z_success_anchors_v1真实failed/exit1/5.982850s，在推进原动作之前，重建训练injection与旧原主机SHA不一致；六锚点未生成。m6p2c_z_sealed_forecast_identity_v1只读对照仍不匹配，禁止人工贴预测数组绕过来源。数据资产字节相同不足以证明注入同源，原因尚未确定；不把它直接归因为CPU或求解器。

9744原模型存在有限界lower>upper约3.55e-15的区间。诊断保留逐行证据，同时原1e-10证书可接受A锚点，不能仅凭另一次求解成功判定原失败一定是求解器误判。离线对照产物与生产路径隔离，不作为失败补救结果。

下一卡M6.P2C-AB已先登记来源精确恢复和原区间只读审计，主机恢复后先检查真实唯一任务/execution.lock，再恢复原源输入。六锚点验签和全部本地验收通过后才能科学提交并冻结新候选，随后主机快速诊断→fresh seed0→1→2各8批→既有全部完整准入项（gate/resume/数值回放/48诊断/每seed106批/4h）→完整回传SHA/真实checkpoint验签。新边界集只增补诊断，不替换既有准入。

m6p2c_current_remote_block_v1记录主机离线，未提交远端新任务，未核实当前锁和唯一活跃任务，新短跑/完整准入均not_started，正式512未启动。旧v26最终failed，不给当前科学源码沿用旧签名。本轮新冻结候选数0、新长准入数0；以后若连续两个冻结候选仍出现快测未捕获的同类数值故障，停止第三轮长准入并保全案例、提交专项数值表述报告与重构卡，不放宽精度或预算。

全量v4已执行299例的阶段计时合计（其余78未进入模型）：model_build_s=3.976985s；b_transform_s=1.369993s；a_certificate_s=0.084529s；b_certificate_s=0.080616s；solver_s=15.479954s。证书计时包含其内部既有polish，solver_s包含求解调用；这些项存在包含关系，不能相加冒充总墙钟。版本{'scipy': '1.18.1', 'backend': 'scipy-highs', 'highs': '1.12.0'}；lock SHA 8e6bd4a67311b32b2369b2f2b9c1fc8780214b59c452982ecddd2cf9cbe51e0a。

最终放行阻断收据：runs/m6p2c_current_release_block_receipt_v1，汇集上述终态报告SHA、门禁日志/完整JUnit SHA和四个Y未提交文件SHA。完整门禁通过不能覆盖边界78缺源失败；Y和Z成功锚点扩展仍未闭卡，不以干净工作树为由提交Y科学文件。独立工程实现提交868705c、02d191a、87572b4，证据记录bf97e4b，后续卡a4bcd58；全轮没有新冻结候选、新远端长准入或正式512训练。
