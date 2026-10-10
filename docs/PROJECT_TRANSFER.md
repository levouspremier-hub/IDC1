# IDC 换机与 Git 同步（2026-10-03）

> 最新项目状态与连续故障交接见 [2026-10-10 项目转移文档](PROJECT_HANDOFF_2026-10-10.md)。下面是历史换机包说明；旧分支、状态及10月3日备份不能作为当前训练现场。

## 当前交付

仓库：<https://github.com/levouspremier-hub/IDC1>。
工作分支：`p5-eval-viz-m6-p2b-s0-diagnosis`；`paper-baseline` 保留不合并。
实际远端同步状态与 SHA 以本卡收尾记录为准，不能把本地提交当作已上传。

代码、冻结配置、数据 manifest 和文档进 Git；原始/处理数据、checkpoint、完整
runs 证据保留在独立换机包。只克隆 GitHub 不足以恢复现有项目。

换机包位于本机 `/Users/levous/Desktop/IDC-transfer-2026-10-03/`：

- `IDC.bundle`：所有本地 Git refs 的自包含历史备份。
- `workspace.tar.gz`：工作目录中代码、data、runs、docs、本地归档；排除 `.git`、
  `.venv`、Python/测试缓存、`.DS_Store`、本地 `.claude` 配置等机器状态。
- `transfer_manifest.json`：每个归档文件的路径、大小、SHA-256，以及 bundle/压缩包
  SHA、打包时 HEAD、分支、发布源码绑定与 final checkpoint hash。
- `RESTORE.md`、`verify_transfer.py`：复制后的校验和恢复说明。

包保留全部 runs（包括失败、旧版本、校准与短跑资格证据），避免只复制 final
checkpoint 后缺少发布依赖。文献侧项和临时 manifest 归入 `local_archive/`；不删资料。
四份 2026-08-14 根目录审计移至 `docs/archive/2026-08-14/`。
业务包、受保护目录、根兼容 shim、数据和 runs 路径保持原样。

## 新机器恢复

先将整个换机包复制到新电脑；文件路径允许改变，项目内相对路径必须保持。

```sh
python3 verify_transfer.py
# 在空的目标父目录中，从已核对的远端工作分支克隆：
git clone --branch p5-eval-viz-m6-p2b-s0-diagnosis https://github.com/levouspremier-hub/IDC1.git IDC
# 如远端仍不可用，可从自包含 bundle 恢复同一分支：
# git clone --branch p5-eval-viz-m6-p2b-s0-diagnosis /path/to/IDC.bundle IDC
# 从换机包解压到新项目根目录，不能混入另一版本的现有 checkout：
tar -xzf /path/to/workspace.tar.gz -C IDC
cd IDC
git status --short
uv sync --frozen
make check
```

`verify_transfer.py` 先核验压缩包/bundle hash，再逐文件流式核验内容；不要跳过。
bundle 恢复后将 origin URL 设置为上面的 GitHub 地址。远端后续新增提交时应先
恢复包登记的版本，再按 Git 正常更新，避免覆盖新版本文件。

Python 必须为 3.12；重建 `.venv`。保持 `uv.lock`、发布/配置/矩阵及 25 个源码
绑定字节一致，不为新机器硬件改写已登记 checkpoint 元数据。锁文件跨平台实际
依赖与求解器运行时需在新机核对；过去 Mac 测试通过不能替代新机验收。

## 必须保留的模型与证据

正式失败运行：`runs/m6p2b_formal_train_seed0_v2_r7/`。
登记 final：`checkpoint_final.pt`，SHA-256：
`472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935`。
诊断：`runs/m6p2b_seed0_diagnosis_v2/`；失败 v1、重放、主机记录与完整门禁也在包内。

正式长训出现大量回退及服务失败，历史逐步 trace 未保存，根因尚未闭合。
最终策略有限诊断 48/48 合格、72 次同输入重放一致，完整门禁 3146 项通过。
下一步应在新机先登记并做 train-only 长时零更新诊断/资源探针；迁移任务没有
授权启动正式训练、seed 1/2、validation/test，也没有证明故障已修复。
若最终修复改变规划、奖励或训练更新语义，必须新版本/新资产重新训练。
