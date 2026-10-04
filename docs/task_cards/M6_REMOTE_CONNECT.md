# M6 REMOTE CONNECT — 主机首次接入

## 边界
配置本机专用 SSH key、公钥、host key 钉扎和 idc-y9000p alias；官方 Mac Tailscale 应用安装与用户登录；被忽略的 .idc_remote.json。只更新 docs/task_cards/ 的验收记录，不改业务代码、锁、配置或模型绑定。WSL w1877@100.73.26.18:22，根 /home/w1877/idc-host。

## 禁止项
私钥、登录 token 不进仓库/日志；保留现有 SSH 配置；不跳过 host key 校验；不启动正式训练/held-out；远端仅已授权执行层验收。不得把用户报告的主机准备当作 Mac 联机通过。

## 验收命令
SSH host fingerprint 必须匹配用户提供 SHA256:zOc842xLD/okEhgIpKHBdBaMB2a7Bk81O0/dAwgaVRY；`ssh -o BatchMode=yes idc-y9000p true`；doctor→seed→submit check→watch，产物完整 hash 回传。未授权 Mac tailnet 登录、远端未装公钥时应明确标待完成。

## 证据产物
SSH 公钥本机文件及指纹（允许发送主机追加 authorized_keys），本机运行 runs/m6_remote_connect_v1/ 五类产物与状态；远端任务独立 job-id 和回传目录。只保存非敏感状态，不保存密钥内容于 Git。

## 回滚点
独立工作分支 p5-eval-viz-m6-p2b-s0-diagnosis，开卡前 b8e78339bf240df5ea0c6381db484eba55528d4f；开卡提交为回滚点。配置用事先备份恢复，专用 key 保留至用户明确撤销，不删除已有凭据。
