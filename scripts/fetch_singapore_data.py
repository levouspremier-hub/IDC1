#!/usr/bin/env python
"""新加坡 USEP/需求/可再生数据下载脚本（M1.2）。

状态：阻塞。官方源（EMC NEMS 门户）需注册/登录，且无无需登录的全年替代源。
本脚本仅记录数据源目录，并在无凭证时明确拒绝下载、绝不伪造数据。

解除阻塞后：提供 EMC 账户凭证（环境变量 EMC_USERNAME/EMC_PASSWORD 或直接下载的 CSV），
再实现具体下载与 sha256 核验逻辑。
"""

from __future__ import annotations

import os
import sys

DATA_SOURCES = {
    "usep_demand": {
        "url": "https://www.nems.emcsg.com/en/nems-prices",
        "fields": ["USEP ($/MWh)", "DEMAND (MW)", "SOLAR(MW)", "LCP ($/MWh)"],
        "frequency": "half-hourly",
        "max_days_per_file": 31,
        "access": "login required (EMC NEMS portal)",
        "units": "USEP SGD/MWh -> convert to SGD/kWh; DEMAND MW",
    },
    "weather": {
        "url": "MSS / NEA / ERA5-CDS",
        "fields": ["temperature (C)", "solar irradiance", "wind speed"],
        "access": "registration required",
    },
}

BLOCKED_MESSAGE = (
    "M1.2 数据下载被阻塞：官方源需注册/登录，且无无需登录的连续全年替代源。\n"
    "未下载任何数据；不伪造、不重复日填全年。\n"
    "解除条件：提供无需登录的公开数据源，或另行授权账户凭证下载。"
)


def main() -> int:
    if "--sources" in sys.argv:
        for name, meta in DATA_SOURCES.items():
            print(f"[{name}] {meta['url']}  ({meta['access']})")
        return 0

    if not (os.environ.get("EMC_USERNAME") and os.environ.get("EMC_PASSWORD")):
        print(BLOCKED_MESSAGE, file=sys.stderr)
        return 2

    # 此处为解除阻塞后待实现：带凭证下载 + sha256 核验 + 写 manifest。
    print("凭证已提供，但下载逻辑待实现（需先解除 M1.2 阻塞并确认访问权限）。", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
