# v14 主机资格及正式训练放行

原seed1数值失败最小修复：只关闭inventory阶段B presolve；阶段A/reachability/noninventory选项及1e-10精度保持，矩阵/目标/物理/服务/终点及0.50共享预算不变。三份历史失败各五次原快照主机验收与新版全门禁/真实恢复/48诊断/三seed8批/4h共享soak及前后对照均通过、全部回传验签。补验附录绑定原输入SHA，原gate及旧失败内容保留。

新版release v6 SHA：ccc89197a97448a1dd9753c9c79567765452c640f48faa8a6f216babdec6bd4d；资格固定revision 3d892a4f82b92808e7138fdf953af634e73cd1b8。原v10/v11/v12 seed0成功及三次seed1失败独立保留，不跨版本换签恢复。按已有用户授权新fresh seed0→1→2主机单槽512批/98304 transitions/8192 Adam/512乘子/2048回合，完整质量及产物验签后才下seed。此报告是运行放行，非三seed完成或收敛结论，validation/test封存。
