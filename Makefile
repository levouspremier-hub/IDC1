# IDC energy scheduling — command gates (docs/EXECUTION_PLAN.md §5.2)
# Recipes MUST be indented with a single TAB character.

PY      := uv run python
PYTEST  := uv run pytest
RUFF    := uv run ruff

.PHONY: setup check test probe smoke train eval figures report contract

setup:  ## Create/lock the environment
	uv sync

check:  ## Static + fast unit gate (new tests/ only; excludes slow & marl)
	$(RUFF) check contracts tests
	$(PYTEST) tests/ -q -m "not slow"

test:   ## Full new unit test suite (tests/)
	$(PYTEST) tests/ -q

probe:  ## Solver scale/timing/memory probe (planning/probe.py)
	$(PY) planning/probe.py

smoke:  ## End-to-end minimal vertical slice (scripts/vertical_slice.py)
	$(PY) scripts/vertical_slice.py

train:  ## Run a training run -> runs/<id>
	$(PY) train/train_ppo_ultimate.py

eval:   ## Unified evaluation
	$(PY) eval/eval_base.py

figures:  ## Render figures from runs/
	$(PY) viz/figures.py

report:  ## Machine-generated report from runs/*/metrics.parquet
	$(PY) scripts/build_report.py

contract:  ## Contract version consistency check
	$(PY) -c "from contracts import CONTRACT_VERSION, CONTRACT_VERSION_ID; print(f'contract_version={CONTRACT_VERSION} id={CONTRACT_VERSION_ID}')"
