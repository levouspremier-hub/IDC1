# 正式训练超时修复工作账

用户要求当前互动任务直接完成超时定位、必要修复/验收并恢复已授权训练；小时巡检仅辅助。本任务修复负责人正在工作，定时任务在此期间只读检查，不修改源码/文档或提交新job。

原失败：runtime-formal-seed0-v1，371完成批/5936 Adam/371乘子；失败批20 partial、0更新，未执行失败步环境；原39文件receipt全部验签。audit整个调用.566569742秒，A/B/reachability实际计时均0，原stage_a=time_limit仅为empty-result默认标签，不能独断实际A求解超时。主机固定原快照诊断与长进程heap/GC构模诊断待执行，当前不宣称根因和修复完成。

为优先定位导致正式训练停止的缺陷，将中断尚未完成的v9资格soak，保留原始产物及明确人工中断原因；不将中断认定为自然solver失败或4h通过。原排队probe继续使用既有单槽。若执行代码修复，旧v9资格不能沿用，必须新版本和对应完整主机资格/晚批证据后才恢复新发布的正式三种子训练。不移走预算起点、不关闭GC、不放宽任何约束或容差，不换签恢复原checkpoint。

交互修复诊断结果：原源码的固定5次快照冷重放及保留371批journal的5次heap诊断均通过，heap调用.3324—.3695秒；124412 nnz稀疏构模.0488—.0603秒。所有观察均未出现generation2 GC，所谓due_major诊断的阈值设置并未真正触发major，因此不能据此宣称复现或排除major GC。确认的是原现场deadline在任何实际求解之前耗尽；历史缺少阶段/GC计时，无法倒推出唯一耗时来源。安全停止行为正确，保留原失败。

红回归52e2589：稀疏装配25000 nnz导致36次GC、求解前超时错误标A=time_limit，两项明确失败；旧COO到CSR字节等价参考通过。最小修复使用NumPy CSR缓冲避免每非零元tuple，保留规范列排序、float64、全部系数；原共享预算起点/限额、solver选项、可行域/证书、raw/Adam均不改。分别标注model_assembly、inventory_certificate、inventory_reachability、before_stage_a，实际A/B结果语义不改。24项轻量回归转绿，Ruff及model mypy通过；主机比较/完整验收尚待，不宣称正式ready。
