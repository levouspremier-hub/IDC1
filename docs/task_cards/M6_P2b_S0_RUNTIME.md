# M6-P2b-S0-RUNTIME — explain captured runtime slowdown before repair

## Boundary

User requested root investigation and repair on 2026-10-04. Start clean
0b3b2f9, existing work branch. Allowed: scripts/m6p2b_seed0_runtime.py,
tests/test_m6p2b_seed0_runtime.py, standalone diagnostic observers, this card
and concise audit. No bound source changes until specific failure regression
is committed and the minimal repair boundary is explicitly appended here.

## Prohibitions

No budget enlargement, constraint relaxation, rewritten failure class, policy
update, validation/test, formal training or seeds 1/2. Strict final checkpoint
and original release checks remain intact. Preserve all failed runs and originals.
Runtime hypothesis is not historical batch-503 proof. No unregistered host stress.

## Protocol / evidence

Use captured first-failure snapshot/proposal from soak ordinal 440, origin 5136.
Two isolated process arms: unchanged published solver options, and diagnostic-only
HiGHS threads=1 (if supported by installed SciPy); neither edits release source.
512 identical sequential solves per arm; no policy updates, .25 s unchanged.
Record call wall/process/thread CPU time, rusage context switches/faults, GC,
thread count, stage timings/options/status and action equivalence to clean replay.
Do not run both arms concurrently. Each arm stops after 512 solves, or exception;
no more than 10 minutes total before reporting. On any timeout, immediately replay
the same input in a new process with original options and record CPU/wall metrics.
This probes solver-pool oversubscription and runtime scheduling, not all env states.
A code repair requires an identified mechanism, failing test and equivalence checks.

## Acceptance commands

`.venv/bin/python -m pytest tests/test_m6p2b_seed0_runtime.py -q` (module absent
at registration; first failing specification precedes implementation).
Ruff/mypy on new scripts; `.venv/bin/python -m scripts.m6p2b_seed0_runtime
--run-id m6p2b_seed0_runtime_v1`; verify standard five artifacts, receipts,
checkpoint/source hashes, no-update scope and equivalent optimal action outputs.

## Rollback / delivery

Task-card commit is rollback point. Explicit staged commits and ordinary push to
origin work branch; verify remote SHA. No large artifacts or checkpoint upload;
new run directories retained with code/dependency/data/scenario hashes and command.
