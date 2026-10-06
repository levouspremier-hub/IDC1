# v10 主机资格与正式训练放行

完整主机资格及独立晚批缓存验收全部通过，回传逐文件验签。v10固定0.50秒，执行闭包8389f89，正式新release v2允许种子0/1/2全部新初始化，按同一SSH全局锁依序执行。原371批失败与所有中断/诊断保留，没有跨版本换签恢复。

补验重建212-origin缓存、默认GC、Torch1、五次原失败快照全部可执行；无环境步或PPO更新。独立补验作为规范资格gate目录后收集附录绑定，原gate核心五文件与原远端证据不改。

release SHA256：edc30e5b9f7175754072ae7a7ac15618617a34f17b2a087f7aacf19cc666e20c。正式资格目录：runs/runtime_campaign_qualification_v10_r4。全部阶段与附录文件哈希在release evidence中。主机单seed预计耗时以本次短训及实际长训另报，快照耗时不能当吞吐。

正式任务将以唯一ID runtime-formal-three-seed-v2-v10 提交；只有各seed完整512批/98304 transition/8192 Adam/512乘子/2048回合资格及产物通过后才进入下一seed。

2026-10-06 12:27恢复实训：新job runtime-formal-three-seed-v2-v10-r2/controller308291，固定revision15d65b505f0881203240be79b7b43a1f5d4d1dfe，最小asset6fa5ab...7c50ca，原release edc30e5b...666e20c不变，watcher59855自动完整回传。主机实际seed0 Python309920，已完成1/512批、192transitions、16Adam/1乘子、4/4回合通过服务/物理/库存/统一终点全部既定标准，A降级0/fallback0，批前/latest存在、无failed_batch。旧failed parent/原v8失败保留且不重用；当前三seed都从新发布fresh初始化，seed1/2由同一parent在前seed完整512/更新/质量与产物验签通过后串行启动，不独立重投。当前仅启动已核实，不宣称三seed训练完成。标准新恢复验收runs/m6p2c_delivery_capacity_acceptance_v2/及receipt。未执行管理员compact或更改科学执行源码/预算/工作量。
