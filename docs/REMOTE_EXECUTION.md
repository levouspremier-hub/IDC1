# Mac 开发 / Y9000P WSL2 执行：方案 1

本工具仅负责固定代码版本、任务隔离、SSH 执行和全部产物回传。
当前正式训练故障仍未闭合；配置完成不代表授权运行正式训练、seed 1/2 或 held-out 评估。
Mac 现有长时零更新诊断继续运行，本次不操作其进程或绑定源码。

## 发给主机 GPT 的准备指令

下面整段可直接交给 Y9000P 上的 GPT：

> 请准备 IDC 的 WSL2 计算主机，供 Mac 通过 SSH 提交固定 Git SHA 的长任务。
> 已有 WSL2；先确认 Ubuntu 版本、Linux 用户名、systemd 状态、磁盘空间。
> 仓库 https://github.com/levouspremier-hub/IDC1 ，工作分支
> p5-eval-viz-m6-p2b-s0-diagnosis 。在 WSL Linux 文件系统内准备目录，不要放 /mnt/c。
> 阅读该分支 docs/REMOTE_EXECUTION.md；安装 make、git、rsync、openssh-server、curl、
> uv 和 Python 3.12，以及 WSL 内的 Tailscale。需要管理员密码或网页登录时由我操作，
> 不读取/展示密码、token、私钥。启用普通 SSH 服务，通过 Tailscale 地址连接，
> 不对公网路由器开放 22 端口。保留现有 SSH 设置和 authorized_keys。
> Mac 公钥尚未提供，先完成主机准备，再等待公钥，追加后验证权限。
> 克隆工作分支到 ~/IDC-control，创建 ~/idc-host，运行 uv sync --frozen；
> 未收到数据前不要启动完整检查、诊断或训练。不要修改发布、锁文件、预算、
> CUDA 配置或 checkpoint 绑定，不启动 seed1/2、正式训练或 validation/test。
> 返回：WSL 用户名、Tailscale IPv4、SSH 端口、~/idc-host 的绝对路径，
> SSH host ED25519 公钥指纹，以及 git/rsync/uv/Python 版本与配置结果。
> 同时说明 WSL/SSH/Tailscale 能否在 Windows 重启后恢复；不要宣称尚未测试的
> Mac SSH 连接或训练已通过。

主机 GPT 可按以下顺序实施，已安装项先检查后跳过。

Windows PowerShell：`wsl --list --verbose` 确認目标发行版为 WSL2。
已有可用 Ubuntu 不重新安装。[Microsoft WSL 安装文档](https://learn.microsoft.com/en-us/windows/wsl/install)
建议项目放 WSL 的 `/home/<user>/`，由 Linux 工具访问。[微软环境说明](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)

Ubuntu：

```sh
sudo apt-get update
sudo apt-get install -y make git rsync openssh-server curl ca-certificates
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.12
sudo systemctl enable --now ssh
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# 等待 Mac 公钥后追加到 authorized_keys；保留已有公钥。
# chmod 600 ~/.ssh/authorized_keys
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
mkdir -p ~/idc-host
git clone --branch p5-eval-viz-m6-p2b-s0-diagnosis https://github.com/levouspremier-hub/IDC1.git ~/IDC-control
cd ~/IDC-control
~/.local/bin/uv sync --frozen
```

uv 使用[官方安装流程](https://docs.astral.sh/uv/getting-started/installation/)。
如果 systemd 未启用，按 WSL 实际配置处理；不要中断任何现有长任务。
Tailscale 安装/启动/login 按[官方 WSL2 指南](https://tailscale.com/docs/install/windows/wsl2)，
放在 WSL 内以便直接访问 Linux SSH，Mac 加入同一 tailnet。此方案使用普通
OpenSSH 经 Tailscale 网络连接，不需要启用 Tailscale SSH 或改变公网端口映射。
Windows 设置插电不睡眠；WSL shutdown/Windows 重启会终止任务，不能把 SSH 断开
与主机关机混为一谈。第一次先手动启动 WSL；系统开机自动恢复应单独验证。

## Mac 接入

主机准备完成后生成专用 SSH key（由用户设置口令），将 **.pub 公钥**发送给主机
GPT 追加到 authorized_keys；私钥只留在 Mac。加载到 ssh-agent 后 BatchMode 可用。

```sh
ssh-keygen -t ed25519 -f ~/.ssh/idc_y9000p -C idc-mac-to-y9000p
ssh-add --apple-use-keychain ~/.ssh/idc_y9000p
```

在 `~/.ssh/config` 添加，按主机返回值替换：

```sshconfig
Host idc-y9000p
  HostName <WSL 的 Tailscale IPv4>
  User <WSL 用户名>
  Port 22
  IdentityFile ~/.ssh/idc_y9000p
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 3
```

首次 `ssh idc-y9000p true` 时核对主机给出的 host key 指纹，再完成正常确认。
工具不关闭 host key 检查、不复制私钥、不在仓库保存密码。
之后运行：

```sh
cd /Users/levous/Desktop/IDC
python3 -m scripts.idc_remote configure --host idc-y9000p --remote-root /home/<user>/idc-host
python3 -m scripts.idc_remote doctor
python3 -m scripts.idc_remote seed
```

`doctor` 部署控制脚本并读取主机信息；它不安装依赖或运行任务。
`seed` 首次将 data 与全部已终止 runs（包含模型、失败证据及资格证据）传过去，
按文件 hash 检查并封存资产集合。运行中的 Mac 诊断不会打包。
代码通过 GitHub 获取，输入资产按 hash 保留；普通代码更新不必重传所有数据。
数据/证据更新时建立新的资产集合，旧集合保留。已封存集合再次 seed 会拒绝修改，
可使用 configure 的 `--asset-set <hash>` 选择已有集合。
输入每任务独立复制，避免一个任务改坏其他任务；需给足磁盘空间，工具不自动清理。

## 日常启动、查看和全部回传

先 commit/push；每个 job 名字唯一，禁止复用，HEAD 会解析为完整 SHA。
提交前会 fetch origin，并核对指定 SHA 是否属于 GitHub 当前已发布分支的历史。
尚未推送的版本会在创建主机任务之前被拒绝；历史已发布版本仍可指定。
先做联机检查，正式训练仍待单独授权：

```sh
python3 -m scripts.idc_remote submit --job host-check-001 -- make check
python3 -m scripts.idc_remote status host-check-001
python3 -m scripts.idc_remote watch host-check-001
```

submit 自动启动一个脱离终端的 Mac watch 进程，其日志在
`.idc_remote_logs/<job>.log`；正常情况下无需再手动 watch。
全部产物落在 Mac `runs/remote_<job>/`，不会改写原有本地 run：

- console.log、固定 revision/命令/状态与五类外层运行产物；
- 该任务新建的所有 runs 子目录，包含 checkpoint、逐步日志、图表及失败产物；
- 命令直接写入 `$IDC_JOB_OUTPUT` 的其他文件；
- 远端逐文件 receipt.json 和本机验签 local_transfer_receipt.json。

自定义命令通过 `submit --job <唯一名> -- <可执行文件> <参数>` 传 argv，不经过 shell。
需要环境变量的命令可通过 `env`，复杂组合写成已提交脚本；不要传 shell 字符串。
例如测试 XML 写入 `$IDC_JOB_OUTPUT` 时在提交的 Python 脚本内读取该变量。
输出应写 runs/ 或该输出目录；源码、输入副本、缓存和 .venv 不属于回传产物。
输入 runs 若需要修改，先以新 run-id 创建输出，不覆写继承的证据。

主机使用独立 Git worktree 检出固定 SHA，依赖 `uv sync --frozen`，随后执行
`uv run --frozen <argv>`。默认单槽 flock 排队包含环境准备阶段，避免多个任务争用
CPU 触发当前 0.25 秒求解预算问题。该版本不支持扩大并发或改变 GPU/训练配置。
提交后 GitHub 新提交不改变已启动或排队任务。

Mac sleep/断网期间主机继续计算，回传进程恢复网络后重试。
Mac 重启或手动终止 watcher 后重新 `watch <job>` 即可恢复；没有注册系统常驻服务。
也可以 `collect <job>` 单次增量拉取。每次只传不同内容，终态必须通过 SHA 校验。
完成所有文件前不会写本机成功传输 receipt。文件不自动删除，旧结果完整保留。

环境准备失败会保留日志并标记失败；环境不可用时不能生成 Parquet，明确写
metrics_unavailable.json，不能伪造五类产物齐全。主机进程消失标记 interrupted，
保留可恢复产物，不自动重跑。输出 symlink 不回传目标内容，记录传输失败。

## 首次联机验收

1. Mac→WSL SSH、doctor、资产全量验签。
2. 提交 check；确认 Mac SSH 断开后主机继续，排队不重叠。
3. 核对失败和成功任务全部产物回到 Mac、receipt 验签，确认代码/配置不变。
4. Mac 断网后恢复、watch 重启恢复；单独确认 Windows 重启后的服务恢复。
5. 新主机单任务耗时与求解状态验收后，再决定是否允许长诊断或正式训练。

本地已用临时 Git 仓库/假 uv 验证 detached 子进程、单槽队列、固定 SHA、失败
checkpoint 保留及五类外层产物。它不等于真实 Windows/WSL/网络/依赖验收。

2026-10-04 实测：SSH、doctor、3686 个资产文件验签、成功/失败任务自动全部回传
均通过，make 4.4.1 已安装。host-check-v1 缺 make、v2 未发布 SHA、v3 主机
GitHub HTTPS TLS 中断均保留失败产物；三次都未进入 pytest，完整门禁仍待通过。
WSL IPv4 访问 GitHub 独立检查连接超时。需先恢复主机 GitHub HTTPS，再使用
新 job-id 运行 make check；不关闭 TLS 校验、不修改依赖锁或求解预算。
Mac 断网重试与 Windows 重启恢复仍未实测，不据此宣称正式训练准备完成。

后续实测：WSL GitHub HTTPS 已恢复，host-check-v4 按 456c26a 启动真实门禁。
Ruff/mypy 通过，pytest 因冻结 PV 逐位比较失败及 pandas C CSV parser 段错误未完成。
独立复测见 docs/task_cards/M6_REMOTE_CONNECT.md 与 runs/m6_remote_host_gate_v4/；
三项新任务均已全部回传验签。当前待解决跨平台测试问题，不能据此启动正式训练。

跨平台修复已见 docs/task_cards/M6_HOST_PORTABILITY.md：保留原冻结文件与checkpoint绑定，注册生成配方与新验证器版本分别核验。本地120项相关回归和全链静态检查通过；当前主机离线，修复后的WSL完整门禁仍待恢复连接后以新任务验收。
