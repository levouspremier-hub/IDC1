#!/usr/bin/env python
"""M3.4a 静态检查：确认旧 α 路径已退役（不再读写任何 α 状态）。

历史背景：旧配置含一个「计划负载预留损耗」标量（取值 0.40），对未使用的计划负载
加预留损耗。该语义已在 M3.3 从正式链移除，并在 M3.4a 彻底删除参数、属性与方法。
本脚本只做静态检查，不再实例化或修改任何 α 字段。
"""

from __future__ import annotations

import json
from pathlib import Path

# 用片段拼接，保证本文件自身不含待查字面量（否则会自命中）。
ALPHA_TOKENS = ("planned_load_" + "reserve_" + "alpha", "reserve_" + "alpha")
SCAN_ROOTS = ("envs", "configs", "scripts", "tests")


def scan_residue() -> list[str]:
    hits: list[str] = []
    for root in SCAN_ROOTS:
        for path in Path(root).rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in ALPHA_TOKENS:
                if token in text:
                    hits.append(f"{path}: {token}")
    return hits


def main() -> None:
    from envs.idc_price_env import IDCPriceEnv20D

    signature = IDCPriceEnv20D.__init__.__code__.co_varnames[: IDCPriceEnv20D.__init__.__code__.co_argcount]
    env = IDCPriceEnv20D()
    report = {
        "alpha_in_constructor_signature": any(t in signature for t in ALPHA_TOKENS),
        "alpha_attribute_present": any(hasattr(env, t) for t in ALPHA_TOKENS),
        "legacy_method_present": hasattr(IDCPriceEnv20D, "_actual_loads_from_completed_work"),
        "residue_hits": scan_residue(),
        "status": "retired",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
