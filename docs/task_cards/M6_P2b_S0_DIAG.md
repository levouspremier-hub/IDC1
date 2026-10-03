# M6-P2b-S0-DIAG — seed 0 late fallback diagnosis

## Authorization and baseline

User approved the diagnosis plan on 2026-10-03: final-checkpoint train-only
fixed-policy diagnosis, no parameter updates, no seed 1/2 or full retraining.
Baseline `23cf26c3c82cb55150a97d7108b886def617cb8c`; clean worktree and
`git diff --check` passed. Branch `p5-eval-viz-m6-p2b-s0-diagnosis`.

## Boundary

New `scripts/m6p2b_seed0_diagnosis.py`, `tests/test_m6p2b_seed0_diagnosis.py`,
this card and `docs/audits/M6_P2b_S0_DIAGNOSIS.md`. Existing release-bound
sources remain byte-identical. Diagnostic-only runtime observers may wrap
snapshot/solver calls without changing arguments/results; record their scope
and overhead limitations. New runs only; original run/checkpoint read-only.

## Prohibitions

No held-out dates, policy/optimizer/multiplier updates, budget increase,
constraint relaxation, history replacement, checkpoint relabeling or protected
directory edits. No exact historical failure claim from missing transitions.
No full retrain, seed 1/2 training or formal evaluation. A root-specific semantic
repair requires a separately specified regression and versioned change.

## Acceptance commands (declared before implementation)

- `.venv/bin/python -m pytest tests/test_m6p2b_seed0_diagnosis.py -q`
  Currently absent: diagnostic runner/tests do not exist.
- `.venv/bin/python -m scripts.m6p2b_seed0_diagnosis --run-id m6p2b_seed0_diagnosis_v1`
  Currently fails with missing module. Must run the preregistered 48 episodes.
- `make check` after diagnostic implementation; explicit lint/type checks for
  the new script, which is outside the Makefile directory list.
- `git diff --check`; release source hash equality; source checkpoint and
  parameter hash equality; final run artifact receipt validation.

## Evidence

Origins `[5040,5088,5136,5184,5568,6576]`, environment seed 0; deterministic
actions plus sampled actions with generator seeds 0/1/2. Each case in a fresh
process, then all cases in one shared process: 48 episodes, 2304 steps.
New run contains five standard artifacts, per-step actions/status/timings/task
state, failure/predecessor replay snapshots, historical batch comparisons and
source hashes. No retries replace failures. Same-input solver replay, if
needed, uses separate run IDs with zero policy updates.

## Rollback / Git

Start commit is this card's independent rollback point; revert subsequent
commits explicitly (no reset/amend/rebase). All code/test/docs commits pushed
to origin on this branch; verify remote HEAD, leave clean worktree. Large run
artifacts/checkpoints remain ignored; concise evidence and hashes enter Git.

## Findings before implementation

Zero-based index 450 and 503 share origins 5040/5088/5136/5184. Fallbacks
0/0/0/0 versus 0/32/48/48; index 502 has one fallback on origin 4992.
Historical step_records are optimizer updates, not environment transitions.
No raw action/failure-stage trace is persisted. Model deadline starts after
inventory validation; correction_solve_time_s includes that validation and
model/postprocessing, but excludes snapshot/service-guard construction.
