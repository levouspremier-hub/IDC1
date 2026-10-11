# 校园内网传输：日常默认路径

自 2026-10-11 起，Mac 与训练主机之间的项目 SSH、模型/数据传输及结果回传
默认使用 `idc-y9000p-campus`。校内路径失败就报错，不自动回退公网或 Tailscale。
原 `idc-y9000p` 保留为管理/恢复通道。该约定不增加训练、全量同步或评估授权。

## 已验收状态

| 项目 | 2026-10-11 现场结果 |
|---|---|
| 校内连接 | 双向对端测试标识通过；正式 SSH 公钥认证登录目标 WSL |
| 文件传输 | WSL→Mac 33,554,432 字节，8.25 秒，两端 SHA-256 一致 |
| 物理路径 | Mac nettop 为 en0；Windows 同一 TCP 四元组及返回路由为 WLAN |
| 计费验收 | **待确认**；不是“实测免流通过” |

文件 SHA-256：`0917b1ed07ded6aac39a90fe023b5283427a845e4e91c1d8ca602822680e7784`。
关闭传输压缩，测试暂限速 4096 KiB/s，因此 8.25 秒不代表链路最大吞吐。
两端随机文件和临时监听器已清理，受限 SSH 入口保留。

学校[官方说明](https://xwb.neu.edu.cn/_s242/10187/mainm.psp) FAQ Q5 正文规定
校内资源和 IPv6 资源不计账户流量，经过校园网 IPv4 出口的互联网访问才统计。
本路径符合校内连接条件，按政策应不扣套餐；计费实验仍未闭合。

Mac 浏览器查询页连接失败，其 DNS 将门户解析为代理虚拟地址。通过目标 WSL
读取门户自身使用的 HTTPS 只读接口取得新读数：12:30:40 测试前和 12:38:16
延后查询均为 **112.98 GB**，原始 `sum_bytes` 为 **121309429192**。
门户按 1024 进位、两位小数显示；原始字节及已用时长均未观察到刷新。
同账号有 4 台设备在线，用户确认手机持续使用流量；不能以读数不变证明免流，
也不能把后台增长全部算到此次测试。未扩大到 GB 级测试。

机器摘要见 [验收 JSON](dev_reports/CAMPUS_TRANSFER_ACCEPTANCE_2026-10-11.json)。
详细原始记录留在 Mac 仓库外 `/Users/levous/Desktop/campus-transfer-20261011/`。

## 当前连接配置

以下是 2026-10-11 现场快照，地址变化后应重新核验。

`Mac 172.29.51.199/en0 → 校园网关172.29.255.254 → Windows 172.26.61.42/WLAN:2222 → WSL 172.19.32.63:22`

Mac `~/.ssh/config` 的独立别名关键项：

```sshconfig
Host idc-y9000p-campus
  HostName 172.26.61.42
  Port 2222
  User w1877
  IdentityFile ~/.ssh/idc_y9000p
  IdentitiesOnly yes
  StrictHostKeyChecking yes
  UserKnownHostsFile ~/.ssh/idc-campus.known_hosts
  BindInterface en0
  AddressFamily inet
  ProxyCommand /usr/bin/nc -4 -b en0 -s 172.29.51.199 -G 5 %h %p
  ProxyJump none
  Compression no
  BatchMode yes
  ConnectTimeout 5
  ConnectionAttempts 1
  ServerAliveInterval 15
  ServerAliveCountMax 2
```

`nc -b en0` 显式绑定物理接口；仅设置 SSH 源地址不足以替代该约束。
WSL ED25519 主机密钥已通过可信 Tailscale 管理连接核对并钉扎：
`SHA256:zOc842xLD/okEhgIpKHBdBaMB2a7Bk81O0/dAwgaVRY`。
主机密钥变化必须重新经可信通道核对，不关闭校验。

Windows 转发只监听校园 IP 的 2222 端口。防火墙规则
`IDC-Campus-Mac-2222-20261011` 只允许来源 `172.29.51.199`，本地地址
`172.26.61.42`，TCP2222，接口 WLAN；原防火墙保持开启。
返回 Mac 的 `/32` 路由经 `172.26.255.254`、接口20/WLAN、metric1，
仅写 ActiveStore。未改变全局代理或覆盖整个校园网的路由。

## 日常连接及传输命令

```bash
ssh -v idc-y9000p-campus 'hostname; uname -s; printf "%s\n" "$SSH_CONNECTION"'
# 上传明确选择的文件；远端目录需存在
rsync -rt --partial --progress -e 'ssh -o Compression=no' /absolute/path/model.pt idc-y9000p-campus:/home/w1877/chosen-directory/
# 下载明确选择的结果目录，末尾 / 表示同步目录内容
rsync -rt --partial --progress -e 'ssh -o Compression=no' idc-y9000p-campus:/home/w1877/chosen-result/ /absolute/path/local-result/
# 已有任务状态及增量回传读取本机 .idc_remote.json 中的 campus host
python3 -m scripts.idc_remote status <已有job-id>
python3 -m scripts.idc_remote collect <已有job-id>
```

`SSH_CONNECTION` 在 WSL 中看到 Windows 转发源 `172.19.32.1`；Mac 源 IP
应从 Windows 2222 的 TCP 连接表核对。不要据此误判走了 Tailscale。

Mac openrsync 兼容 rsync2.6.9/协议29；WSL 为 rsync3.4.1/协议32，以上基础
选项已实际传输验证。失败后重跑相同命令可利用部分文件及增量算法。
不默认 `--delete`、压缩、整个项目、全部 checkpoint 或历史运行目录。
已运行的 watcher 可能仍持有启动时的旧配置；本次不重启这些进程，
新启动的命令使用当前 campus 配置。

Windows 只读检查：

```powershell
Find-NetRoute -RemoteIPAddress 172.29.51.199
Get-NetTCPConnection -LocalPort 2222
netsh interface portproxy show all
Get-NetFirewallRule -Name IDC-Campus-Mac-2222-20261011 | Get-NetFirewallAddressFilter
```

## 地址变化与 Windows 重启

- Mac IP 改变：更新别名 nc 的 `-s`、Windows 防火墙 RemoteAddress、返回 `/32`
  路由；删除旧 `/32`。网卡名改变时也更新 nc `-b` 和 BindInterface。
- Windows IP 改变：更新 HostName、known_hosts 的 `[新IP]:2222`、转发
  listenaddress 及防火墙 LocalAddress；保留并核对原 WSL 密钥。
- WSL 重启后：经可信管理通道 `ip -4 -br addr show eth0` 复核，再由管理员
  使用 `netsh interface portproxy set v4tov4 listenaddress=<当前Windows校园IP> listenport=2222 connectaddress=<当前WSL IP> connectport=22 protocol=tcp`。
- Windows 重启会丢失 ActiveStore 路由。现场地址一致时，由管理员单独重建：

```powershell
New-NetRoute -DestinationPrefix '172.29.51.199/32' -InterfaceIndex 20 -NextHop '172.26.255.254' -RouteMetric 1 -PolicyStore ActiveStore
```

不要覆盖已有路由或重复整套初始化。更新后重新手动验证 SSH 和物理路径；
尚未部署自动化，也未宣称 Windows 重启恢复已验收。

## 原值及回滚

修改前 Mac 无 campus 别名及独立主机密钥文件；Windows 无本次 `/32`、
2222 转发和同名防火墙规则；本机项目配置 host 为 `idc-y9000p`。
原配置备份及完整命令保存在仓库外验收目录，不提交凭据。

本机项目默认配置回滚只恢复 host，保留 asset_set/remote_root：

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path('.idc_remote.json')
d = json.loads(p.read_text())
d['host'] = 'idc-y9000p'
p.write_text(json.dumps(d, indent=2) + '\n')
PY
```

恢复后，该项目默认传输不再属于本次已验证的校内路径。
Mac SSH 回滚脚本：`sh /Users/levous/Desktop/campus-transfer-20261011/rollback-mac.sh`。
Windows 管理员回滚：

```powershell
netsh interface portproxy delete v4tov4 listenaddress=172.26.61.42 listenport=2222 protocol=tcp
Remove-NetFirewallRule -Name 'IDC-Campus-Mac-2222-20261011'
Remove-NetRoute -DestinationPrefix '172.29.51.199/32' -InterfaceIndex 20 -NextHop '172.26.255.254' -PolicyStore ActiveStore -Confirm:$false
```

保留原 Tailscale 管理入口。未关闭防火墙、修改训练代码或启动训练。
