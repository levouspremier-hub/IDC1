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

## 本地验收 / 待主机接入

- 开卡 `7bc95e0`；失败规格 `386738a` 在模块未实现时收集失败；实现 `b644513` 已普通推送。
- 新增 11 项测试全部通过，包含真实 rsync 的同大小/同 mtime 内容损坏修复，临时 Git 仓库的 detached 子进程、三任务单槽排队、成功/失败/源码突变分类、checkpoint 保留、孤儿任务回收和 receipt 验签。
- Ruff、mypy 通过；25 个发布源码与开卡前 9f127ab hash 完全一致；未改配置、预算或训练语义。
- runs/m6_remote_execution_local_v1/ 保存五类产物、测试 XML、完整命令与源 hash、receipt。仅假 uv/临时仓库，不代表真实 WSL/依赖/网络验收。
- 当前 Mac 四小时零更新 soak 仍在运行；本次未中断它，也未并行启动完整 make check 或实际训练。局部门禁针对新增执行工具，不能冒称全仓完整门禁已重跑。
- 用户主机已有 WSL2 和 GPT；主机准备指令已写入 docs/REMOTE_EXECUTION.md，待返回 SSH 用户名、Tailscale 地址、端口、远端根路径和 host key 指纹后，才在 Mac 配置并完成首次联机 check/回传验收。
- 默认单槽执行，全部新 runs/checkpoint/失败证据与显式输出自动回传；Mac 重启后需 watch 恢复；Windows 重启会中断计算，保留证据，不自动重跑。
- 所有实现与文档提交普通推送至当前工作分支；不合并 paper-baseline。最终远端 SHA 在交付消息中核对。

## 联机验收返修边界

真实 WSL host-check-v1 退出127，console 明确 make 未安装；SSH 非交互 PATH 还需覆盖 uv 位于 ~/.local/bin 的情况。允许 scripts/idc_remote.py 新增 task_environment 将已解析 uv 的父目录加入子任务 PATH，并新增失败规格；只改执行层，不改正式源/锁/预算。主机安装 make 属已授权环境准备，失败任务不覆盖，重测使用 host-check-v2。验收仍为本地 runner 测试/静态检查及真实主机 make check。回滚点 a5d5322。

## 发布预检查返修

host-check-v2 在取 Git SHA 时失败：本机新任务卡 c20e8f5 尚未推送，远端 not our ref；未进入测试。保留失败产物。允许提交前 fetch/核对 requested revision 是否在 origin 实际已发布 heads 的祖先中，拒绝未发布 SHA，禁止创建远端任务；不自动提交或推送用户未提交内容。新增失败回归后实现，原层边界不变，回滚点 c20e8f5。
