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
