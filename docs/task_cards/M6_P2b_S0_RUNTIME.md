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

## Evidence-led extension (before next run)

Initial probe finished 512 solves per arm, all optimal and exec-identical.
Single-thread reduced CPU work/context switches but not wall time; this does
not justify changing production solver options. No proposed algorithm repair.
The initial probe's outer milp CPU wrapper hides the immediate model caller
from the Observer stage spy; those stage/size metadata are not reliable.
CPU/wall/options/output measurements remain valid. Fix diagnostic placement
by timing inside Observer's existing solver wrapper, preserving its caller.

Allowed additional standalone script: scripts/m6p2b_seed0_scheduling.py.
New run ID m6p2b_seed0_scheduling_v1: original 24 fresh controls, then shared
train-only fixed-policy episodes for up to 4 h using original solver options,
then 24 post controls, same host guards and early stop as the soak card.
Record per-call and model process/thread CPU times and involuntary switches.
At the first timeout persist original step before any additional solve; temporarily
suspend parent observer and replay identical snapshot/proposal in the old process
then a new process immediately (3 pairs off/on in each). These are separate
probes, not additional budget for the original action, which remains unchanged.
Post-failure timings/trajectory are explicitly affected by the probes. Capture
top CPU process names/PIDs locally; do not publish unrelated app activity in Git.
No changed planning semantics, no training. Long observation auto-follow-up
continues root investigation; no semantic fix is claimed from a normal short probe.

The soak script may also receive optional diagnostic metadata and include the
scheduling observer source hash in its existing drift check; default protocol
is unchanged. On-timeout old/fresh probes run only once globally and are never
used to replace the failed action or relabel its failure.
