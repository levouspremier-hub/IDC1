# seed1 第二阶段不可行修复

原v10-r2 parent failed，seed0完整512批成功保留，seed1完成100批/19200 transitions/1600 Adam/100乘子/400合格回合后在origin9120 step18停止，seed2未启动。失败批66条partial/raw/logprob、批前/latest及RNG/来源全部保留，失败步未执行环境、失败批0更新。完整parent69文件/391488203字节回传receipt已验签；两个CP实际next100/Adam1600/乘子100/finite/原v10绑定符合。证据runs/m6p2c_seed1_failure_triage_v1/，failure SHA1d7292ea79821cb4efe2d82dd63bc1e01c1d7f55bcf26d6d9d728441ea5fd88a。

A optimal/B infeasible，调用约0.189秒，非超时；磁盘/内存正常。原快照在Mac和主机可复现。原A整数残差5.156e-9，完整原B复核通过，但原B矩阵在默认/关闭presolve/仅B精度1e-9/1e-10全部返回不可行。因此不能归因于presolve，也不能只改B。Mac固定全A/B精度1e-9与1e-10均最优、整数残差0、全矩阵残差约7e-15/9e-16。此证据定位到A近整数数值见证与后续offset限值的敏感性，不声称已定位HiGHS内部具体错误行。

红回归658d589/bda171c/9c64776已提交，最小修复e42c7f6仅将inventory两阶段内部求解精度1e-8收紧为1e-9；候选构造准确记录同值与运行说明。数学目标/矩阵/服务/物理/终点/验收1e-6/offset1e-6/0.50共享预算均不放宽，无重试或新A降级语义。97项对应回归通过，Ruff/mypy通过。v11候选冻结提交3b1033fa5ecfdc634ea2063f27adc655a957657a，原r5科学配置不改。

已唯一提交runtime-seed1-numeric-repair-v1（五次原失败快照.50，全A矩阵/目标/边界逐字节比原captured A，0环境/PPO）；runtime-campaign-qualification-v11固定.50由同一主机全局锁串行执行，完整门禁/真实resume/48/三seed各8批/4h前后共享验收尚待。两者成功且完整回传验签前不发布/启动正式512。源码新绑定不允许旧CP换签恢复；新版正式campaign必须同源三seed fresh，原v10 seed0成功及seed1失败独立保留，不混成新版三seed结果。validation/test仍封存。
