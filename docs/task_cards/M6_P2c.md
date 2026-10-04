# M6-P2c：可行 A 解保留、训练停止与主机预算验收

用户于本轮明确授权实施完整 Mac 开发／SSH 主机执行方案。起点 a6086fb；
独立分支 p5-eval-viz-m6-p2c-runtime-repair；开工 status/diff --check 均为空。

## 边界

planning/model.py、planning/corrector.py、safe_rl/corrector_wrapper.py、
safe_rl_v2/{rollout,formal_train_loop,inventory_train,ppo_update}.py，必要的新版本发布／
配置／实验绑定、独立远程验收脚本、针对性测试、本卡及验收记录。
Mac 只运行轻量检查；主机固定 SHA 单槽执行完整检查、预算比较、短训、4h
稳定性与门禁通过后的 seed0 长训。随后审核再安排 seed1/2。

## 禁止项

不改奖励、优化器超参数、数据切分、任务与物理约束、SOC 目标、raw/log-prob
和受保护目录。不跑 validation/test。不覆盖历史产物、不补时求解、不把 B
超时改称最优。仅 A 最优且完整检查通过可以保留；其他无合格方案停止训练。

## 验收命令（先登记）

`.venv/bin/python -m pytest tests/test_m6p2c_runtime_repair.py -q`
（改前此模块不存在；新增真实 A/B 加受控超时测试后先实测失败）。
`.venv/bin/ruff check planning/model.py planning/corrector.py safe_rl/corrector_wrapper.py safe_rl_v2/rollout.py safe_rl_v2/formal_train_loop.py safe_rl_v2/inventory_train.py tests/test_m6p2c_runtime_repair.py`
相应主链 mypy；主机 `make check` 后再启动计时实验，不能并行。
预算 0.25/0.5/1.0s，六 train origins、均值及采样0/1/2、fresh/shared各24回合；
从最小通过预算依次做3x8批短训和4h验收，失败保留，不无限重试。

## 证据产物

`runs/m6p2c_*/`、`runs/remote_m6p2c_*/` 保存五类产物、测试XML、命令、
失败快照、独立计时、来源和传输receipt。最终预算/训练资格只采用主机结果。
新发布绑定旧模型仅为诊断输入，新训练从头初始化。短跑/稳定性不通过则
停止该档位；三档都不通过停止，不放松标准。

## 回滚点

独立分支起点 a6086fb；按提交逆序 revert，保留所有 run 与历史资产。
每批实现先 focused 验收再显式列文件提交；卡结尾工作树干净。

## 本轮停止边界（用户后续指令）

用户要求“修复完当前发现的问题后先停止当前任务”。本轮仅完成并提交
修正器保留合格 A、无可执行方案停止、非有限梯度阻断、批前保存与失败证据、
预算读取和分项计时。针对性本地验收后停止。
不提交远程任务、不推送或启动训练，不生成新的正式发布或冻结预算。
后续主机完整检查、三档预算实验、3×8 批、4h、正式 seed0 全部未执行。
旧 r7 发布与源码绑定不再一致，保持拒绝新训练；不能沿用旧资格启动。
本轮未完成的远程编排草稿未纳入提交，恢复任务时另行完成验收接线。
