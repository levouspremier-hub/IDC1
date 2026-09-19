"""M1.3f-e-b2-b：物化 B6 归一化参考值 `configs/frozen_refs/refs_v4.json`。

**唯一实现**在 `scenario.b6_refs`（loader 与物化器共用同一严格入口）；
本脚本只是 CLI 壳，**不**接受任何输出路径 / 信任根覆盖参数。

```bash
uv run python scripts/materialize_b6_refs.py            # 物化 / 复核
uv run python scripts/materialize_b6_refs.py --verify   # 只读校验
```
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenario.b6_refs import main

if __name__ == "__main__":
    raise SystemExit(main())
