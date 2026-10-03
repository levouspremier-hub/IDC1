# M6-P2b-S0-SOAK — preregistered four-hour frozen-policy diagnosis

## Boundary and authorization

User authorized execution on 2026-10-03. Start: clean
`fdd6dd4`, branch `p5-eval-viz-m6-p2b-s0-diagnosis`.
Allowed: new `scripts/m6p2b_seed0_soak.py`, new
`tests/test_m6p2b_seed0_soak.py`, optional persistent-policy argument in
`scripts/m6p2b_seed0_diagnosis.py`, this card and a concise soak audit.
No release-bound source edits. Preserve strict original checkpoint export/load.

## Prohibitions

Train-only origins 5040,5088,5136,5184,5568,6576; deterministic and sample_0/1/2.
No parameter updates, optimizer, changed constraints/budget/reward, validation/test,
formal training or training seeds 1/2. No artificial battery depletion/thermal
stress. Preserve every failed run; never overwrite or resume into an existing ID.
No claims about historical failure stages from this new final-policy experiment.

## Preregistered protocol and stop rules

Run ID `m6p2b_seed0_soak_v1`. Before and after the shared phase, all 24 cases
run in separate fresh processes. The shared phase retains one policy and process
for >=14400 monotonic seconds of repeated complete episodes, in the fixed
origin-major/mode-minor order. Finish the current episode at the time boundary.
Rebuild environment and sampling RNG per episode; never reset within a trajectory.
Record cycle/ordinal/PID, start/end time and parameters/source hashes.
Host sample every 60 seconds and at episode boundaries: AC power, thermal state,
process RSS/CPU, load and disk. Require AC at start; abort after the current episode
on detected loss of AC, telemetry failure, changed power settings, <5 GiB free disk, or source/hash drift.
An episode failure is retained and ends the shared loop after that full episode;
replay its first failure snapshot in a fresh process, then perform post controls
if host conditions remain valid. Any failed acceptance makes the run failed.
Interrupted/exception runs write failed artifacts. Duration incomplete is never
reported as success. SIGTERM/SIGINT are recorded; SIGKILL cannot run cleanup.
No other heavy diagnostic/test workload runs during the timed phase.

## Acceptance commands (before implementation)

- `.venv/bin/python -m pytest tests/test_m6p2b_seed0_soak.py tests/test_m6p2b_seed0_diagnosis.py -q`
  New soak module/tests absent at registration; first specification must fail.
- `.venv/bin/ruff check scripts/m6p2b_seed0_soak.py scripts/m6p2b_seed0_diagnosis.py tests/test_m6p2b_seed0_soak.py`
- `.venv/bin/mypy --follow-imports=silent scripts/m6p2b_seed0_soak.py scripts/m6p2b_seed0_diagnosis.py`
- Launch `.venv/bin/python -m scripts.m6p2b_seed0_soak --run-id m6p2b_seed0_soak_v1`
  under detached `caffeinate -i -s`; close only after checking outcome, counts,
  hashes, host records, before/after controls and monotonic shared duration.
- `git diff --check`; focused regression suffices for this standalone orchestration
  change (previous full gate 3146 passed); do not run full gate during timed soak.

## Evidence and rollback

Five standard artifacts under `runs/<run_id>/`, host JSONL, per-episode step JSONL,
initial/first-failure/predecessor captures, policy/source hashes and final recursive
receipt. Progress report is updated after each episode. Logs/large assets ignored.
Task-card commit is rollback point; use explicit revert for subsequent commits.
Commit and push explicit files to existing origin branch; verify remote SHA.
Remote baseline now synchronized by the machine-transfer task. No baseline merge.

## Implementation checks and launch

- `dd743bb`: registered card; `798a53f`: initial failing specifications
  (collection failed because soak module was absent).
- `0562eef`: implementation; 12 focused tests passed, Ruff and mypy passed.
  Includes a mocked end-to-end orchestration check for retained policy identity,
  duration boundary and both 24-case controls. No training semantics changed.
- Remote origin branch SHA verified as `0562eef78eaea3b1264cfe08b842f016bb5f4961`
  before launch. Baseline synchronization/authentication is now working.
- Started 2026-10-03 23:38:53 Asia/Shanghai, detached Python PID 23796,
  under `caffeinate -i -s`. Log: `runs/m6p2b_seed0_soak_v1.console.log`.
  AC attached at launch; first fresh-before deterministic origin 5040 completed
  with zero fallback and qualified service. Four-hour clock starts only after
  all 24 pre-controls, not at launcher time. This is a launch record, not acceptance.
- Thread heartbeat automation `seed-0` checks every 30 minutes, quiet while
  normal; on completion it verifies artifacts/comparisons and records/pushes audit.
  No timed source changes or heavy tests permitted while running.
