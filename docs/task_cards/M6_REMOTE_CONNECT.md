# M6 REMOTE CONNECT — 主机首次接入

## 边界
配置本机专用 SSH key、公钥、host key 钉扎和 idc-y9000p alias；官方 Mac Tailscale 应用安装与用户登录；被忽略的 .idc_remote.json。只更新 docs/task_cards/ 的验收记录，不改业务代码、锁、配置或模型绑定。实际主机地址、用户名、根目录只写本机被忽略配置，不入 Git。

## 禁止项
私钥、登录 token 不进仓库/日志；保留现有 SSH 配置；不跳过 host key 校验；不启动正式训练/held-out；远端仅已授权执行层验收。不得把用户报告的主机准备当作 Mac 联机通过。

## 验收命令
SSH host fingerprint 必须匹配用户提供 SHA256:zOc842xLD/okEhgIpKHBdBaMB2a7Bk81O0/dAwgaVRY；`ssh -o BatchMode=yes idc-y9000p true`；doctor→seed→submit check→watch，产物完整 hash 回传。未授权 Mac tailnet 登录、远端未装公钥时应明确标待完成。

## 证据产物
SSH 公钥本机文件及指纹（允许发送主机追加 authorized_keys），本机运行 runs/m6_remote_connect_v1/ 五类产物与状态；远端任务独立 job-id 和回传目录。只保存非敏感状态，不保存密钥内容于 Git。

## 回滚点
独立工作分支 p5-eval-viz-m6-p2b-s0-diagnosis，开卡前 b8e78339bf240df5ea0c6381db484eba55528d4f；开卡提交为回滚点。配置用事先备份恢复，专用 key 保留至用户明确撤销，不删除已有凭据。

## 接入准备进度（待用户完成授权）

- 开卡 ea72731；Mac 专用 Ed25519 key 已创建，私钥仅本机 0600，公钥已交给用户转主机追加；不记录 key 内容于 Git。
- ~/.ssh/config 事先备份后追加 alias；StrictHostKeyChecking=yes，被忽略 .idc_remote.json 已配置。既有 OrbStack Include 保留。
- 官方 Tailscale 1.102.4 从 vendor 包站下载，codesign --verify --deep --strict 通过，安装至 /Applications；客户端已打开，等待用户完成系统扩展授权与同一 tailnet 登录。
- 初次 host key 扫描未得到公钥（连接被关闭）；没有写入未经核验的 known_hosts，没有关闭安全校验。
- runs/m6_remote_connect_v1/ 五类产物状态 waiting_user_setup；不宣称 SSH、主机执行或回传验收已通过。待用户确认主机已装公钥及 Mac 网络授权后继续。
- 未启动任何远端训练、诊断或完整检查；未改源、锁文件、模型及预算。

- 更新：用户已完成 Mac Tailscale 权限与同一网络登录；客户端 Connected、主机在线。主机 ED25519 实测指纹与用户给定值完全一致，已钉扎 known_hosts。SSH BatchMode 到达认证阶段，但返回 Permission denied (publickey,password)，仍待主机追加/核对 Mac 公钥及 authorized_keys 权限。运行状态更新为 waiting_remote_public_key；没有启动远端任务。

## 首次真实联机结果 / 待补 make

- SSH BatchMode 实际通过，doctor 通过；项目 Python3.12 可用，主机32逻辑CPU与充足空间。
- 资产集 5005eb32bc2b9971c3153d3d6be0999039572bfa25694fcadffef4af9fce7eed：3686文件已双端验签。
- host-transfer-ok-v1 exit0 与 host-transfer-fail-v1 预定 exit7 完整自动回传，分别13个文件验签，均保留约1.25MB payload与内外层运行证据，SSH会话结束后工作进程仍能完成。
- host-check-v1 exit127：make未安装，未进入测试；8个文件完整自动回传验签。原失败保留，不覆盖。
- 实测非交互 SSH make/uv 均不在 PATH：uv有 ~/.local/bin/uv 实体，make缺包。返修范围已在 M6_REMOTE_EXECUTION 登记，失败规格 1ac1bbd、修复 c12aa45；12项本地runner测试+Ruff+mypy通过，证据 runs/m6_remote_path_repair_v1/；25个正式源绑定未变。
- 修复已普通推送并经 doctor 部署；待主机交互认证安装 make，随后新 job host-check-v2 完整复测。sudo -n 拒绝交互认证，未尝试取得密码或放松sudo安全设置。
- runs/m6_remote_connect_v1/ 状态 waiting_host_make_install；链路验收部分通过，完整工程检查未通过，不宣称训练准备完成。未启动正式训练、held-out或诊断。

## 安装 make 后的复测 / GitHub 网络阻塞

- GNU Make 4.4.1 实测可用。host-check-v2 请求未发布 c20e8f5，GitHub not our ref，未进入测试；完整回传验签。提交入口已补已发布历史预检查（584c4d6 失败规格、c1f14e5 修复），14 项 runner 测试、Ruff/mypy 通过；证据 runs/m6_remote_publication_preflight_v1/，25 个正式绑定源与 c12aa45 一致。
- c1f14e5 完整 SHA 已普通推送并与 origin 分支 SHA 核对一致；doctor 已部署。host-check-v3 请求该 SHA，在主机 fetch 阶段遇 GnuTLS recv error (-110)，未进入测试；失败产物及 receipt 已完整回传验签。独立 WSL IPv4 curl GitHub 连接 8 秒超时，SSH/回传正常。
- runs/m6_remote_connect_v1/ 更新为 waiting_host_github_network。需修复 WSL GitHub HTTPS，再用新 job-id 完整门禁；保留三次失败。未改算法、发布配置、预算或 checkpoint；未启动正式训练、held-out 或诊断。
