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

## Implementation and diagnostic evidence

- `920648c`: card rollback point; `c544c83`: pre-implementation specification
  test failed at collection because the diagnostic module did not exist.
- `525f2c2`: standalone runtime observer and fixed-policy runner, six tests pass;
  `ddfea2c`: signed forecast provenance and exact failure-phase recording,
  seven tests pass; `eca5885`: same-input observer off/on replay.
- `a5f1a33`: detailed audit at `docs/audits/M6_P2b_S0_DIAGNOSIS.md`.
- Unit command `.venv/bin/python -m pytest tests/test_m6p2b_seed0_diagnosis.py -q`
  passed 7 tests. Explicit Ruff on script/tests and mypy on the new script passed.
- Original command produced `runs/m6p2b_seed0_diagnosis_v1/`, intentionally
  interrupted after seven episodes to correct empty per-step provenance. Its
  failed manifest, partial evidence and receipt remain; no overwrite.
- `.venv/bin/python -m scripts.m6p2b_seed0_diagnosis --run-id m6p2b_seed0_diagnosis_v2`
  completed 48/48 episodes and 2304/2304 steps. All strict episode acceptance
  fields true, no fallback/physical violation, unchanged checkpoint and sources.
  All 24 paired raw and executed trajectories were exactly identical.
- `.venv/bin/python -m scripts.m6p2b_seed0_diagnosis --run-id m6p2b_seed0_replay_v1 --replay-source-run m6p2b_seed0_diagnosis_v2`
  completed 36 observer off/on comparisons (72 solves), all equivalent, no
  fallback, no parameter update. Six deterministic initial snapshots, three
  repeats each in fresh and shared process arms; no historical replay claim.
- `runs/m6p2b_seed0_analysis_v1/`: machine-readable timing and paired differences.
- Source final checkpoint remains
  `472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935`.
- Full `make check` passed: 3146 tests, 47 deselected, 429 warnings, zero
  failures/errors, unchanged bound source hashes; command elapsed 2304.59 s.
  Five standard artifacts, XML and receipt are retained in
  `runs/m6p2b_seed0_diagnosis_check_v1/`.
- `runs/m6p2b_seed0_host_context_v1/` retains scoped read-only OS CPU/battery
  queries and extracted rows. Encoded residency changes near onset are not
  decoded frequency measurements; nearby CPMS reduction samples remain 0%.
  Host correlation is not a proven failure mechanism.
- All six run receipts were reverified against files at closure, including v1.
  No timed diagnostic ran concurrently with the full check.

## Root-cause decision / remaining work

No root-specific semantic repair or new training is justified by these finite
results. Original index-503 failure classification/stage/raw/task trace is absent;
the new final-policy run does not reproduce it. Low battery and brief thermal
pressure are recorded host context, not proven CPU throttling. Historical failure
is retained, final policy is not replaced, and validation readiness stays false.

All implementation edits stayed inside the declared boundary. No protected or
release-bound source changed. Remote upload remains blocked: HTTPS push has no
credential, SSH port 22 closes; user was informed and asked to restore normal Git
authentication without sharing a token in chat. No force-push/history rewrite.
Rollback implementation with `git revert eca5885 ddfea2c 525f2c2 c544c83` in that
order; revert documentation commits separately if desired. Root investigation
remains open; this card's completed diagnostic is not a declaration of repair.
