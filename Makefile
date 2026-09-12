# 数据中心能源协同调度 —— 工程门禁（M0.3）
# 受保护目录（marl/ grid_model/ legacy/ 顶层 shim）不纳入任何目标扫描。
# 新主链扫描目录随 M1–M5 逐步加入；M0.3 时均为空/未创建。

SHELL := /bin/bash

MAIN_CHAIN_DIRS := scenario contracts checkpointing planning safe_rl_v2 tests
MAIN_PY := $(shell find $(MAIN_CHAIN_DIRS) -name '*.py' -not -path '*/__pycache__/*' 2>/dev/null)

.PHONY: check test contract probe smoke train eval figures report

# 主门禁：依次运行 ruff -> mypy -> pytest -m 'not slow'
check:
	@echo "== ruff check =="
	@if [ -z "$(MAIN_PY)" ]; then echo "ruff: 无新主链文件（M1-M5 将加入）"; else uv run ruff check $(MAIN_PY); fi
	@echo "== mypy =="
	@if [ -z "$(MAIN_PY)" ]; then echo "mypy: 无新主链文件（M1-M5 将加入）"; else uv run mypy $(MAIN_PY); fi
	@echo "== pytest -m 'not slow' =="
	@uv run pytest -m 'not slow'; code=$$?; if [ $$code -eq 5 ]; then echo "pytest: 0 个测试被收集（M0.4 增加测试骨架）"; exit 0; else exit $$code; fi

test:
	@uv run pytest; code=$$?; if [ $$code -eq 5 ]; then echo "pytest: 0 个测试被收集（M0.4 增加测试骨架）"; exit 0; else exit $$code; fi

contract:
	@test -f contracts/__init__.py || { echo "模块未完成：contracts/ 无契约模块（M2.1 交付）"; exit 1; }
	@uv run pytest tests/test_contracts*.py

probe:
	@test -f scripts/probe_physics.py || { echo "模块未完成：scripts/probe_physics.py 不存在（M3.0 交付）"; exit 1; }
	@uv run python -m scripts.probe_physics

smoke:
	@test -f scripts/smoke_main_chain.py || { echo "模块未完成：scripts/smoke_main_chain.py 不存在（M4 交付）"; exit 1; }
	@uv run python scripts/smoke_main_chain.py

train:
	@test -f safe_rl_v2/train.py || { echo "模块未完成：safe_rl_v2/train.py 不存在（M5.4 交付）"; exit 1; }
	@uv run python safe_rl_v2/train.py

eval:
	@test -d evaluation || { echo "模块未完成：evaluation/ 不存在（M6.2 交付）"; exit 1; }
	@uv run python -m evaluation

figures:
	@test -f viz/figures.py || { echo "模块未完成：viz/figures.py 不存在（M7.2 交付）"; exit 1; }
	@uv run python viz/figures.py

report:
	@test -f scripts/build_report.py || { echo "模块未完成：scripts/build_report.py 不存在（M7.2 交付）"; exit 1; }
	@uv run python scripts/build_report.py
