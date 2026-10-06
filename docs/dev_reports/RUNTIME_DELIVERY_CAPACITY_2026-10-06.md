# 正式训练交付容量修复

原正式启动在复制准备后因D盘14604066816字节低于30GiB守卫停止，seed run一个也未创建，参数更新0；失败全回传receipt验签。C盘约143GB、根rw；不是C盘耗尽或科学资格失败。原发布edc30e5b...666e20c及全部资格原样保留。

原sealed资产27950623466字节，资格raw/canonical各约8GB重复并全部copy到工作树。新的必要完整集合6013文件/8158465336字节，候选95资产与发布5895 evidence均保留、源/目的全SHA验签，digest 6fa5ab27527c24a27e3d40d4b7a4809f37e203d492f024d5fb613355a57c50ca，未投r2。

在全局锁内只处理已终态自身formal/qualification输入副本：37270文件、37330714625字节经前后全SHA/size核对与原子inode共享，所有历史文件路径/内容保留，未删除job/worktree/数据或触碰输出/执行源码。原raw操作manifest Git cwd缺失而revision=unknown保留；执行源SHA、逐文件JSONL及原工作树revision另记录，Mac标准验收manifest记录当前仓库revision。

首次trim受root PATH无/usr/sbin影响未执行；唯一绝对路径/usr/sbin/fstrim exit0，Windows D仍14603354112字节。WSL VHDX为非稀疏动态盘228549722112字节，逻辑空闲未回收为NTFS实际余量。新投递前D必须至少48960654648字节（30GiB底线+实际输入copy+8GiB依赖/输出保留），C仍至少30GiB；守卫不降低。

当前远程Windows token IsAdministrator=False，Optimize-VHD不可用，安全在线方法未能归还物理空间。已准备并通过PowerShell原生Parser零错误校验的管理员离线compact脚本，Mac runs/m6p2c_delivery_capacity_repair_v1/compact-wsl-reviewed.ps1 与 Windows D:\IDC-host-repair-20261006\compact-wsl-reviewed.ps1 全SHA相同。须备份目录可用空间≥完整VHD长度+30GiB，脚本先停WSL、完整备份SHA验签，再只读attach/compact/detach、恢复服务、全源/资产/旧失败输入验签及实际容量；它自身不会启动训练。未执行该管理员脚本、未启用--allow-unsafe、未停WSL或复投正式任务。

外部管理员动作成功（或D实际余量达到上述预留）后，复验原release SHA、新最小资产和网络/根rw，以唯一runtime-formal-three-seed-v2-v10-r2、parent m6p2c_formal_three_seed_v2_v10_r2继续。三个seed原v2_v10名称均未创建，可首次fresh初始化；科学入口与512/更新/质量仍原样串行，不补采、不跨绑定恢复。

2026-10-06 12:20重新核对：D盘实际可用147318947840字节（约147GB），C142723215360字节，根rw。外部清理已解除预留阻碍，代理未执行离线compact。完整源/release、旧sealed f20/d314及原371批失败SHA已复验，最小asset6013文件再次全SHA验签，旧三seed目录均未创建且r2尚不存在；全部条件通过后使用新r2提交，启动进度以后续真实journal为准。标准新验收runs/m6p2c_delivery_capacity_acceptance_v2/。
