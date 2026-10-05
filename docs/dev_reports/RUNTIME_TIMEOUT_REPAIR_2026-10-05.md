# 正式训练超时修复工作账

用户要求当前互动任务直接完成超时定位、必要修复/验收并恢复已授权训练；小时巡检仅辅助。本任务修复负责人正在工作，定时任务在此期间只读检查，不修改源码/文档或提交新job。

原失败：runtime-formal-seed0-v1，371完成批/5936 Adam/371乘子；失败批20 partial、0更新，未执行失败步环境；原39文件receipt全部验签。audit整个调用.566569742秒，A/B/reachability实际计时均0，原stage_a=time_limit仅为empty-result默认标签，不能独断实际A求解超时。主机固定原快照诊断与长进程heap/GC构模诊断待执行，当前不宣称根因和修复完成。

为优先定位导致正式训练停止的缺陷，将中断尚未完成的v9资格soak，保留原始产物及明确人工中断原因；不将中断认定为自然solver失败或4h通过。原排队probe继续使用既有单槽。若执行代码修复，旧v9资格不能沿用，必须新版本和对应完整主机资格/晚批证据后才恢复新发布的正式三种子训练。不移走预算起点、不关闭GC、不放宽任何约束或容差，不换签恢复原checkpoint。

交互修复诊断结果：原源码的固定5次快照冷重放及保留371批journal的5次heap诊断均通过，heap调用.3324—.3695秒；124412 nnz稀疏构模.0488—.0603秒。所有观察均未出现generation2 GC，所谓due_major诊断的阈值设置并未真正触发major，因此不能据此宣称复现或排除major GC。确认的是原现场deadline在任何实际求解之前耗尽；历史缺少阶段/GC计时，无法倒推出唯一耗时来源。安全停止行为正确，保留原失败。

红回归52e2589：稀疏装配25000 nnz导致36次GC、求解前超时错误标A=time_limit，两项明确失败；旧COO到CSR字节等价参考通过。最小修复使用NumPy CSR缓冲避免每非零元tuple，保留规范列排序、float64、全部系数；原共享预算起点/限额、solver选项、可行域/证书、raw/Adam均不改。分别标注model_assembly、inventory_certificate、inventory_reachability、before_stage_a，实际A/B结果语义不改。24项轻量回归转绿，Ruff及model mypy通过；主机比较/完整验收尚待，不宣称正式ready。

修复候选v10冻结提交8389f89bfaced4c04c2c9846520e5a679ac72bfe，执行源码04a4007；三档候选只供注册，实际资格仍固定.50。原冷probe、heap baseline、人工中断v9三个job的终态receipt全部逐文件验签。Mac标准证据runs/m6p2c_timeout_csr_local_v1/含red/green原日志、精确主机probe代码及receipt。主机新比较job runtime-timeout-heap-repair-v1/controller1178269/watcher25717，真实快照124412 nnz逐行比较旧COO的data/indices/indptr，比较在测量correct调用之后执行，不偷预算。新完整资格runtime-campaign-qualification-v10/controller1178390/watcher25756，源8389f89，run runtime_campaign_qualification_v10，依序gate/真实resume/48/三种子8批/完整4h；各只提交一次，串行同一锁。

22:36起新增外部阻碍：两个新job均成功提交并返回queued后，Tailscale把100.73.26.18/localhost-0标记offline；本地tailnet在线，两个5秒peer ping无回复，多次SSH连接超时。因此无法确认新job是否取得锁、开始或终止，更不能说主机已验收或正式训练已恢复。未重复投递、不修改原job/worktree、不追加正式任务。已请用户确认主机开机、WSL/Tailscale及睡眠状态；机器端离线无可用远程修复入口。小时heartbeat保留ACTIVE辅助只读，互动任务仍负责修复/验收/发布/恢复。

用户确认主机恢复并继续后：22:43 SSH恢复，boot_id 5d51db43-fdeb-4ced-811e-c51a8d637d7a、uptime4min，两个旧job已worker消失interrupted，终态receipt均验签；受控新IDheap-repair-v2/qualification-v10-r2各重投一次。随后发现/usr/bin/lscpu、dmesg、sudo I/O error；/dev/sdd根emergency_ro，ext4错误7次/ext4_journal_check_start；新worker均消失，尚未执行验收。Windows宿主Get-Volume确认C仅2097152字节、D233123258368字节；Ubuntu VHD位于C的AppData/wsl，145788764160字节，解释为何Linux虚拟823GB剩余并不保障宿主存储。已开H卡7715079，停止新增任务；Windows自带WSL2.7.12可用manage --move，D余量满足整VHD+30GiB。

Windows独立控制PID2364迁移进行中，脚本和标准运行状态runs/m6p2c_host_storage_repair_v1/；Windows日志D:\IDC-host-repair-20261005\。按先shutdown→全VHD SHA→官方move D:\WSL\Ubuntu-IDC→启动前SHA/长度完全一致→C/D30GiB余量→Ubuntu服务/二进制可读/写fsync验收执行；不对挂载的ext4强行remount/fsck，不注销发行版、不删任何失败证据。SSH在停机阶段不可用，未据此宣布迁移成功。若恢复后仍需离线fsck，要先备份与具体方案，不能跳过；原正式训练仍未恢复。

23:04—23:05重复SSH限时检查仍连接超时。Windows控制进程确实曾成功创建，但停机后没有独立Windows远程通道，因此不能确认其当前存活/phase，更不能把已发起迁移说成迁移完成。已请求用户在Windows读取D:\IDC-host-repair-20261005\status.json中的phase/error；若脚本停于hash_original/moving/hash_moved可据真实状态继续等待，若blocked须读日志定位；不重复启动迁移、不重复投递任务。当前主机修复验收仍待结果，正式ready=false、正式三种子尚未恢复，小时巡检继续只读辅助。

2026-10-06重验：主机当前C约144GB/D97GB、D上Ubuntu rw/错误0，迁移全SHA相同，旧length条件是停机前后量测混用误报；存储验收见RUNTIME_HOST_REACCEPTANCE_2026-10-06.md。随后Git URL-specific global proxy旧12451端口不可达；当前网关未变，已仅本任务裸仓库覆盖为经现有SSH的127.0.0.1:19065 SOCKS，实测精确URL ls-remote和固定8389f89 fetch通过，未改训练源码/Windows全局设置。v3/r3只Git setup失败全部保留验签。

v4/run m6p2c_timeout_heap_repair_v4已终态succeeded/exit0且全部回传验签，固定五次原failed snapshot/371 journal观察全部executable、A/B optimal，10个实际CSR矩阵（含B附加行）data/indices/indptr与旧COO完全相同；wall .162652/.153210/.165832/.153181/.209204秒，124412nnz主矩阵装配约.0097—.0103秒。最后一次带profile，不把不同时刻/重启后的旧baseline .33—.37秒与此次值当作因果速度实验；并未复现原major GC或唯一历史延迟原因。无env.step/PPO更新，原失败输入哈希不变。

完整runtime-campaign-qualification-v10-r4/controller3856同源码8389f89/.50运行，gate至46%，后续真实resume/48/3x8/4h仍待。另按G扩展提交runtime-formal-late-cache-probe-v1/controller42692/watcher35272，排在完整资格之后：保留旧371 checkpoint/journal对象，按212原train-origin逐项重建当前正式verified缓存并核对provenance，保持默认GC、固定五次原失败快照correct、无环境执行/PPO或旧checkpoint重绑；新release同时需要该晚批缓存验收与完整资格。纯诊断warm_elapsed_s记录从缓存预热开始直到最后输入复核（包含后续五次观察），不作为独立工厂性能指标。互动任务继续负责全部闭环，小时任务仅辅助只读。Mac本任务网络relay PID34510，临时idle-sleep assertion PID35335随relay结束退出（显示器不保持唤醒），重型任务均主机单槽。
