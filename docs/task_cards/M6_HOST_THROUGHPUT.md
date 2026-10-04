# M6 HOST THROUGHPUT — Y9000P 当前冻结链实测

## 边界
仅新增本任务卡、docs/dev_reports/精简验收记录和新runs产物。使用已实现 safe_rl_v2.inventory_train --short --seed 0 原入口，8个预登记train批次（4×48步/批），独立初始化策略并真实PPO更新；不得加载或替换原正式final权重。不修改训练入口、配置、发布、规划、奖励、绑定或依赖。

## 禁止项
不启动512批完整训练、seed1/2、validation/test；不扩大0.25s预算、改线程/GPU配置或约束，不改变日期顺序，不掩盖回退与不合格记录。保持CPU/torch=1冻结口径，不用旧v1或合成求解benchmark替代。8批测量仅用于吞吐外推，不构成正式训练放行或长期故障排除。

## 验收命令
python3 -m scripts.idc_remote submit --job host-throughput-seed0-v1 --revision <本卡已推送完整SHA> -- python -m safe_rl_v2.inventory_train --short --seed 0 --run-id host_throughput_seed0_v1
python3 -m scripts.idc_remote collect host-throughput-seed0-v1
逐文件校验远端receipt和终态local_transfer_receipt；严格复核训练artifact_verification、8批/1536步/128 Adam/8 Lagrangian、仅train来源、绑定和冻结配置。保留非零回退或任何失败，吞吐与资格单独报告。

## 证据产物
runs/remote_host-throughput-seed0-v1/全部输出（包含短程checkpoint及五类产物）；runs/host_throughput_estimate_v1/五类汇总与hash。记录外层命令墙钟、内层elapsed、逐批env_build/rollout/PPO/残余耗时和回退；首批与后7批统计分开。checkpoint/journal等仅能在总体未拆分耗时中标识，不伪造逐次写盘耗时。512批估算使用稳定批均值加实测未拆分成本，给观察到的min/max场景范围（不是置信区间），注明日期/策略演变/进程长期状态可能改变吞吐。

## 回滚点
工作分支p5-eval-viz-m6-p2b-s0-diagnosis，开卡前3d649caa4fa10ca6a809247e8c925b6cc3b4b394；无生产代码修改。全部文档分批提交推送，远端SHA核对，收尾工作树干净。

## 实测验收
已使用固定6d3afa2，以观测wrapper计时上述原入口子进程（命令和wrapper源码保存在远端manifest/request）；exit0，8批/1536步/128 Adam/8 Lagrangian及train预登记顺序、冻结配置、严格绑定均核对。入口墙钟436.657s，后7批均值53.031s，512批工时初估7.549h；7.443–7.643h为实测min/max情景，不是保证范围。
18个远端文件及训练artifact_verification全部回传验签；估算五类产物与receipt在runs/host_throughput_estimate_v1/。12步回退、库存32/32和服务31/32合格完整保留，**测量完成不代表短程资格通过或正式训练放行**。未修改任何生产源码、预算/线程/GPU配置、旧final或冻结资产；未启动完整训练/seed1/2/held-out。精简审计见docs/dev_reports/HOST_THROUGHPUT_2026-10-04.md。
