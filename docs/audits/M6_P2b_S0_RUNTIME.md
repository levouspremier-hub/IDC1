# Seed 0 runtime investigation: thread hypothesis not sufficient

The captured soak failure remains registered. No production planning, reward,
training update or release-bound source changed. No semantic repair, formal
training or held-out run is claimed. Root investigation remains active.

## Captured-input comparison

`runs/m6p2b_seed0_runtime_v1/`: 512 sequential solves with original published
options in one fresh process, then 512 with diagnostic-only HiGHS threads=1 in
another process. Exact original snapshot/proposal SHA is
`f80d0ce944a333319ddbab4e574149048312e16e9fd0755c6264388f4a106586`.
Every solve used the shared .25 s budget; source/checkpoint checks remained strict.
All 1024 solves were optimal, with exactly identical executed actions.

| measurement | published options | diagnostic threads=1 |
| --- | ---: | ---: |
| median wall, s | .083268 | .083585 |
| P95 wall, s | .087160 | .086708 |
| max wall, s | .128098 | .116894 |
| median process CPU, s | .099503 | .083498 |
| sampled native threads | 17 | 13 |
| median involuntary context switches | 34 | 11 |
| max aggregate GC pause per call, s | .033628 | .033156 |

The installed SciPy supports threads=1, contrary to a historical code comment.
It reduces CPU work and switches but has no meaningful wall-time benefit in
this probe. Do not promote it to a cure for the captured slowdown. A normal
bounded probe does not prove old-process stability or exclude external contention.
Scoped read-only OS queries did not recover a target-process AppNap/thermal
transition or task-specific CPU history establishing the cause. Nearby unrelated
application activity is local-only evidence, not proof of contention or a Git asset.

## Diagnostic correction and acceptance

The initial probe's extra outer milp wrapper concealed the immediate planning
caller from the stage observer. Its stage/size labels are unreliable; call
CPU/wall/options/output measurements remain useful. The source run is immutable.
CPU timing was moved into the existing observer, preserving objective identity
and model caller. Process/thread CPU and involuntary switches are recorded at
the actual solver call; model CPU and generation-wise GC wall totals are also
recorded. No solver arguments or returned actions are changed.

A failing regression (missing CPU field before implementation) was committed
before the correction. Fifteen focused tests, Ruff and mypy passed. Four real
captured-input solves in `runs/m6p2b_seed0_cpu_observer_acceptance_v1/` confirm
A/B labels, 11079 model variables and CPU fields; all optimal. Receipts of both
runs were reverified against files. This repairs an observation defect, not the
seed-0 runtime failure.

## Next registered discriminating run

`m6p2b_seed0_scheduling_v1` retains original solver options, strict final policy
binding and .25 s budget. Same 24 fresh-before controls, up to four hours of
shared fixed-policy train-only episodes, and 24 fresh-after controls, with the
same AC/disk/source-drift guards. First timeout is written before any probe.
Then the exact input is replayed immediately in the old process and a fresh
process, three observer off/on pairs each. Original failed action is retained;
extra probes do not expand its budget. Later trajectory/timing is explicitly
affected by these separate probes and must not be treated as an unobserved run.

Compare model/solver wall versus CPU time, GC and context switches, and the
near-contemporaneous old/fresh controls. No sufficient mechanism is identified
yet to justify changing constraints, reward or optimizer. Root-specific repair
must start with a failing regression, maintain the physical/service contract,
and preserve failed evidence; any semantic adoption requires new version/assets
and fresh training, with no early-checkpoint substitution.

## Scheduling probe result (2026-10-04)

`m6p2b_seed0_scheduling_v1` failed as registered, without four-hour acceptance.
Shared phase lasted 478.395 s / 82 episodes; episode ordinal 81, origin 5136,
sample_0 failed. Both 24-case fresh control arms passed. Total 130 episodes /
6240 steps, parameter and source hashes unchanged; 403 receipt files rehashed.
All 129 other complete raw/exec trajectories exactly match fresh-before.
Read-only full comparison and replay details: `m6p2b_seed0_scheduling_audit_v1`.

First failure was step 1 (human step 2), B time limit. Raw action, SOC, task
state, continuity arrays and signed forecast provenance match healthy control.
The failed model has 3649 variables, 2259 constraints passed to solvers and
329 integers. Build including validation was .025034 s versus .023134 s in
control; model GC total .002855 s, which does not explain the first timeout.

| timing / counter | fresh-before | failed shared |
| --- | ---: | ---: |
| model wall, s | .136778 | .294922 |
| model process CPU, s | .189397 | .257379 |
| A wall / process CPU, s | .064165 / .097237 | .087558 / .107233 |
| B passed remaining budget, s | .161280 | .135213 |
| B wall / process CPU, s | .043815 / .059145 | .178471 / .121035 |
| B involuntary switches | 8 | 1326 |

Immediate same-input replay: old process one timeout in six calls, fresh process
one timeout in six calls. Both retain optimal successful calls as well. Fresh
last replay also recorded .095571 s generation-2 GC and .015494 s generation-0
GC within a .268777 s failed call. Thus long-lived process state is not necessary
for this failure, and garbage collection can consume budget in a separate replay;
it was not the primary pause at the original first timeout. These probes cannot
be added to the original action's budget, and later episode timing/state is affected
by them. All original failed executed actions remain unchanged and retained.

Scoped read-only macOS logs record XProtect scanning activation at 13:12:47.154;
first-failure persistence/probe start was 13:12:47.455. Local host sampling at
13:13:02.741 recorded XprotectService at 546.4% CPU, and at 13:13:22.911 at
244.6%. Parent sampled RSS fell from 389024 to 110576 KiB during this interval.
No target AV process appeared in the preceding boundary sample. Together with
high context-switch counts and failures in fresh and old processes, this strongly
implicates host-wide resource variability rather than a necessary process leak.
CPU frequency, instruction counts and exclusive AV causation are not established;
no security service was disabled or altered.

The earlier midnight soak has a nearby XProtect event at 00:25:06.036, about two
seconds after its first failure, so that event alone cannot establish its trigger.
The historical training 04:24:20–04:27:00 query returned no XProtect entries.
Do not claim the exact batch-503 cause is proved. The evidence identifies a
reproducible failure class: host-sensitive wall-budget exhaustion can occur with
unchanged valid policy input and cascade into backlog/service failure.

Registered mitigation experiment: execute the same strict 48-case train-only
acceptance in the existing WSL single-slot queue, keeping code, assets, solver
budget and constraints fixed. No reward/optimizer/planner change is justified
by this host evidence. Dedicated-host qualification is pending; cross-platform
engineering gates are tracked separately and must pass before formal readiness.
