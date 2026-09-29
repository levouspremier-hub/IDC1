# 数据中心能源协同调度 —— 工程门禁（M0.3）
# 受保护目录（marl/ grid_model/ legacy/ 顶层 shim）不纳入任何目标扫描。
# 新主链扫描目录随 M1–M5 逐步加入；M0.3 时均为空/未创建。

SHELL := /bin/bash

MAIN_CHAIN_DIRS := scenario contracts checkpointing planning safe_rl_v2 evaluation viz tests
MAIN_CHAIN_FILES := idc_model/allocation.py safe_rl/corrector_wrapper.py runs/writer.py
MAIN_PY := $(shell find $(MAIN_CHAIN_DIRS) -name '*.py' -not -path '*/__pycache__/*' 2>/dev/null) $(foreach f,$(MAIN_CHAIN_FILES),$(wildcard $(f)))

# 训练入口参数透传：make train TRAIN_ARGS='--synthetic-smoke --steps 8 --seed 0 --corrector off'
TRAIN_ARGS ?=

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

# M1.3g-f-c-k：`make train` 进入**正式 train-only 训练循环**（不再落入 synthetic dry run）。
# 入口为 `safe_rl_v2/formal_train.py`：冻结配置 v1 + M9.2 矩阵 v3 预登记顺序，
# 经 env release v1 与 train release v1 双重验签后放行。`safe_rl_v2/train.py` 的
# synthetic 路径与既有守卫**保持不变**（本卡的正式入口是新模块）。
train:
	@test -f safe_rl_v2/formal_train.py || { echo "模块未完成：safe_rl_v2/formal_train.py 不存在（M1.3g-f-c-k 交付）"; exit 1; }
	@uv run python -m safe_rl_v2.formal_train $(TRAIN_ARGS)

eval:
	@test -d evaluation || { echo "模块未完成：evaluation/ 不存在（M6.2 交付）"; exit 1; }
	@uv run python -m evaluation

figures:
	@test -f viz/figures.py || { echo "模块未完成：viz/figures.py 不存在（M7.2 交付）"; exit 1; }
	@uv run python viz/figures.py

report:
	@test -f scripts/build_report.py || { echo "模块未完成：scripts/build_report.py 不存在（M7.2 交付）"; exit 1; }
	@uv run python scripts/build_report.py
