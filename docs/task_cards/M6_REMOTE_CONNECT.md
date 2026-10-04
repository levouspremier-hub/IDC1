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
