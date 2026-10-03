# M6 PROJECT TRANSFER — 整理与换机同步

## 边界
允许整理四份根目录历史审计至 docs/archive/，本地文献侧项与临时 manifests 至 local_archive/；新增 README.md、docs/PROJECT_TRANSFER.md、迁移清单与 .gitignore 白名单。源代码、正式配置与绑定文件字节保持不变。创建仓库外换机备份（代码历史 bundle、数据与 runs 及本地资料、hash 清单），推送当前工作分支至现有 origin。

## 禁止项
不删除训练证据、数据或 checkpoint；不移动业务包、根兼容 shim；不改变规划/奖励/训练语义；不启动训练或 validation/test；不合并 paper-baseline，不重写历史、不强推。大产物及凭据不入 Git。换机不代表批准正式训练。

## 验收命令
`git diff --check`；核对发布 SOURCE_PATHS 与 checkpoint SHA；`git bundle verify <备份>/IDC.bundle`；读取归档核验逐文件 SHA-256 与大小；`git ls-remote origin refs/heads/p5-eval-viz-m6-p2b-s0-diagnosis` 与本机 HEAD 相等。当前远端验收缺少 Git 凭据而失败；浏览器授权后重试。

## 证据产物
README 和迁移说明，仓库外 transfer_manifest.json/备份包，五类运行产物 runs/m6_project_transfer_20261003_v1/，远端 SHA 及本地干净状态。只作文件整理，沿用已通过的完整 3146 项门禁，不重复训练测试。

## 回滚点
独立工作分支 p5-eval-viz-m6-p2b-s0-diagnosis，开卡前 58683e76a4a98cdd81442befe17211347e223ecf；开卡提交作为回滚点，后续显式 revert；本地移动按映射搬回，归档保留。

## 验收与交付

- 开卡提交 `33eeb55`，整理提交 `f6ea7bf`；四份审计 100% 内容不变地移动；本地文献侧项、临时 manifest 保留于 local_archive/。
- 初始换机包 revision `f6ea7bf4bc9237b73ec9ffaddb74f26cb61f47a3`，3324 个文件、930134949 bytes 未压缩、251673856 bytes 压缩；bundle verify 和逐文件 SHA 校验通过。交付记录提交后重建同一路径换机包，再核验最终 manifest 登记的 HEAD。
- 25 个发布绑定源码与开卡前字节完全一致，登记 final checkpoint hash 不变。受保护代码、正式配置/数据/runs 路径不变；没有训练/评估运行。
- 五类产物位于 `runs/m6_project_transfer_20261003_v1/`；仓库外备份位于 `/Users/levous/Desktop/IDC-transfer-2026-10-03/`，GitHub 不上传 checkpoint 等大产物。
- 已使用用户在浏览器确认的 GitHub CLI 授权恢复本机 Git 登录，凭据保存在本机 keyring，未进入代码或备份。
- `git push -u origin p5-eval-viz-m6-p2b-s0-diagnosis` 成功；远端 SHA `f6ea7bf4bc9237b73ec9ffaddb74f26cb61f47a3` 与整理提交相同。交付记录提交后继续普通推送并核对最终远端 SHA，结果保存在包外 delivery_receipt.json。
- 原完整门禁 3146 项已通过；本次仅文档/文件归档，核验 diff、源绑定、模型与迁移完整性，不重复训练测试。换机本身不构成正式训练授权。
- 原临时文件映射：carbon-lit-search/、litsearch/ → local_archive/research/；somewhere/*.json → local_archive/generated_manifests/。回滚时按映射搬回，tracked 文档用显式 revert。
