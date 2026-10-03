# M6 REMOTE EXECUTION — Mac 开发 / WSL2 长任务

## 边界
新增 scripts/idc_remote.py、tests/test_idc_remote.py、docs/REMOTE_EXECUTION.md、README 入口与 .gitignore 文档白名单。仅执行层、文件传输与证据记录，不改正式源码/配置绑定。Mac SSH 连接参数存在被忽略本地配置；主机每任务固定完整 commit，隔离工作目录，单并发默认排队，失败产物保留。全部任务文件由 rsync 增量拉回，终态按 receipt 校验；Mac 可断开后主动 watch 恢复拉取。

## 禁止项
不启动正式训练或 validation/test；不在运行目录 git pull；不双向同步 .git/.venv；不扩大求解预算；不删除远端或本地产物；不上传凭据；未配置远端不能宣称联机验收完成。现有长时诊断不触碰/中断。

## 验收命令
`.venv/bin/python -m pytest tests/test_idc_remote.py -q`（实现前模块不存在，收集失败）；`uv run ruff check scripts/idc_remote.py tests/test_idc_remote.py`；mypy 新脚本；隔离临时 Git 仓库运行假任务验证 detached 子进程、排队、完整/失败 hash、代码固定与接收防覆盖。联机后执行 doctor→seed assets→submit check→watch。

## 证据产物
单测/本地集成运行 runs/m6_remote_execution_local_v1/ 的五类产物与测试 XML；任务 remote_status.json/receipt.json/console.log；文档注明待完成 WSL/SSH 安装与远端验收。

## 回滚点
工作分支 p5-eval-viz-m6-p2b-s0-diagnosis，开卡前 HEAD 9f127ab5d900fa0a66df8c012d440ae70253c219；开卡提交作回滚点，后续显式 revert；保留任务与传输文件。
