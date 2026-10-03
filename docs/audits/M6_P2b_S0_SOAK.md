# M6-P2b-S0-SOAK: runtime fallback captured; duration acceptance failed

## Result

`runs/m6p2b_seed0_soak_v1/` stopped as preregistered after a failed full
shared-process episode. It is **failed**, not a four-hour pass. No parameters,
planning/reward semantics or solver budgets changed; no training or held-out
run was started. Original seed-0 final checkpoint remains registered.

Times are Asia/Shanghai: launch 2026-10-03 23:38:54, shared phase 23:46:04
through 2026-10-04 00:25:13 (2349.214 s / 39 min 9 s), post-controls and
final failed artifact write completed 00:32:34. All 489 episodes / 23472 steps
are retained: 24 fresh-before, 441 shared, 24 fresh-after. Both fresh arms
passed all 24 strict episode acceptances; the first 440 shared episodes passed.
The shared episode with zero-based ordinal 440, origin 5136, deterministic
mode, had 24 timeouts, two later deadline_shortfall results, failed service,
no physical violations, and a 3.7143 kWh terminal target gap. It passed the
inventory band, which does not satisfy the stricter episode acceptance.

## First divergence and timing evidence

First timeout is zero-based step 13 (human step 14). The preceding step and
first-failure snapshot/proposal are preserved under
`shared/000440_5136_deterministic/`. Against the fresh-before control, raw
actions and executed actions are exactly equal through step 12. At step 13,
raw action, task list, SOC, forecast provenance and continuity arrays match;
executed actions first differ. Raw actions and task state first diverge at
step 14, after the initial fallback. This is evidence for this new run only;
it does not reconstruct historical batch index 503.

At step 13 SOC is 50 kWh; 15 arrived task records include 14 finished records
and one waiting task, ID 4718932309969739063, remaining work 31, deadline 15,
maximum rate 31/step, not started and non-interruptible. Raw storage action is
0.0619665086; fallback exec is all zeros and actual charge/discharge both 0 kW.
There are 11079 model variables, 1965 constraints passed to each solver and
175 integer variables. Full 21-dimensional actions and task records are in
both the source trace and the machine-readable audit report.

| interval / outcome | fresh-before step 13 | failed shared step 13 |
| --- | ---: | ---: |
| snapshot including guard (outside deadline), s | .002999 | .016942 |
| validation before deadline, s | .000405 | .002546 |
| pre-R build including validation, s | .025126 | .096547 |
| R certificate phase, s | .003569 | .015462 |
| A solver passed remainder, s | .221525 | .139734 |
| A solver wall time, s | .023999 | .127646 |
| A status | optimal | optimal |
| B solver passed remainder, s | .197201 | .010782 |
| B solver wall time, s | .025303 | .047335 |
| B status | optimal | time limit, no primal result |
| correction call total, s | .083866 | .291195 |

R used the certificate path; no R MILP ran at this first failure. The shared
budget was passed down as a decreasing remainder. Broad slowdown in build,
certificate and A left little budget for B. B's observed wall time exceeded
its supplied remainder by about 36.6 ms; this is an observed call overrun,
not evidence that the code passed a fresh .25 s budget to each stage. Solver
setup/termination granularity, scheduling and runtime state are not separated
by these timers. Overall correction time includes work outside the timed
solver and is not a hard .25 s wall-time guarantee. No budget was enlarged.

The predecessor step 12 already had .27824 s model wall time but optimal
A/B results and no fallback. The failed episode's timeout stages are B:3,
A:11, after_A_before_B:2, R:8; later deadline_shortfall steps are separate.
This stage classification uses observed calls, not aliased stage_a_status.

## Replay, trajectory comparisons and host evidence

The exact captured step-13 snapshot/proposal was replayed in one new process,
three repetitions each with observers off/on (six calls). All returned no
failure with optimal A/B and zero target gap; paired exec differences were
exactly zero. Wall time ranged .081997–.108312 s. This fresh replay occurred
after the failed episode; there was no simultaneous old-process replay, so
recovery cannot prove a state leak rather than a transient host event.

All 488 other complete raw/exec trajectories exactly match their corresponding
fresh-before case. Only shared ordinal 440 differs (maximum raw difference
.0184198; exec difference 1.0). Every episode used the same parameter hash;
24 distinct pre-control PIDs, one shared PID and 24 distinct post-control PIDs
were verified. Checkpoint SHA remains
`472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935`.
Bound source hashes and both diagnostic source hashes match before/after and
the files inspected at closure. Git HEAD advanced during unrelated remote-job
work; the source-hash check confirms no bound/diagnostic source drift.

All 545 host samples report AC attached, unchanged power settings, no telemetry
errors and no thermal/performance warning recorded by pmset. Sampled parent
RSS peaks at 496608 KiB; minimum free disk is 341830172672 bytes. These coarse
samples cannot exclude brief pressure, CPU-frequency changes or competing
process scheduling; no per-call CPU-time or competing-process trace was captured.
The new failure shows low battery is not necessary for this failure mode.

## Evidence closure and decision

All 1476 source receipt entries were rehashed successfully; the five standard
artifacts and first-failure/predecessor captures are present. Read-only analysis
is retained separately at `runs/m6p2b_seed0_soak_audit_v1/`, including all 489
comparisons, first-failure and predecessor detail, replay results and host summary.
Its success status means the audit completed, not that the source soak passed.
No source artifacts were overwritten. Implementation checks were 12 focused
tests plus Ruff/mypy; no redundant full test workload ran during this audit.

The immediate mechanism is now observed: reduced remaining B budget after
runtime slowdown caused a timeout and zero fallback, followed by task-state
and policy-input divergence. The reason for the slowdown and its relationship
to historical batch 503 are still unresolved. Do not start formal training or
seed 1/2 on this evidence. A separately registered same-input repeated probe
with old/fresh-process controls and CPU-time/scheduling measurements can separate
the remaining hypotheses before any semantic repair is selected. Do not hide
this failure with budget inflation, constraint relaxation or checkpoint selection.
