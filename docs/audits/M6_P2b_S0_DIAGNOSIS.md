# M6-P2b-S0-DIAG — final seed 0 fixed-policy diagnosis

## Outcome and limits

The preregistered 48 episodes / 2304 transitions completed with **zero fallbacks**,
zero physical violations, and 48/48 strict inventory-episode acceptances. Final
policy weights were not updated. Fresh and shared processes produced identical
raw and executed actions for all 24 paired trajectories. This does **not** repair
or invalidate the historical training failure, prove three-hour process stability,
or establish convergence / held-out performance. The historical root cause remains
unproven. No planning, reward, optimizer, protected source, release or frozen
configuration was changed; no formal training, seed 1/2 or validation/test ran.

## Historical transition and missing evidence

Use zero-based `batch_index`; index 503/450 are human-numbered batches 504/451.
Both use origins 5040/5088/5136/5184, the train dates 2024-04-15 through 04-18.

| index | origin | fallbacks | service qualified | unproven steps |
| --- | --- | ---: | --- | ---: |
| 450 | all four | 0 each | all true | 0 each |
| 502 | 4992 | 1 | true | 0 |
| 503 | 5040 | 0 | true | 0 |
| 503 | 5088 | 32 | false | 9 |
| 503 | 5136 | 48 | false | 45 |
| 503 | 5184 | 48 | false | 46 |
| 511 | 6576/6624/6672/6720 | 48 each | all false | 45/45/43/44 |

The 32 fallback count does not identify the first failing step or establish that
those steps were contiguous. Saved `step_records` are 16 optimizer minibatches,
not 192 environment transitions. Original raw actions, per-step failure classes,
R/A/B status and pre-failure task states were not persisted. Only latest/final
checkpoints exist. Neither policy/transition digests nor final weights reconstruct
the historical index-503 inputs. Missing information remains **unknown**.

`run_training_batch` collects all 192 transitions before its PPO updates. Thus
the index-503 first optimizer loss (117149.8359375) and mean storage-head gradient
(56.5461) follow the affected rollout; they cannot establish that this same batch's
update caused its first fallback. An earlier policy update is not excluded.

Index 450 correction-call median/P95 were 0.08903/0.11412 s; index 503 was
0.31954/0.49838 s. Peak RSS was unchanged at 1141.4 MiB across indices 498–503,
then 1161.0 MiB from 504 onward. This does not prove a leak or rule out host-wide
memory pressure. The native `HighsMipSolverData::transformNewIntegerFeasibleSolution`
message also appears in successful new diagnostics; it alone is not a failure class.

## Timing contract and observation

`correction_solve_time_s` measures the call to the projection model, including
inventory validation, model construction, R/A/B work and result processing. It
excludes snapshot and service-guard construction. The inventory deadline is
created **after** snapshot validation, before model construction. Each actual
R/A/B MILP receives the current remainder of that shared 0.25 s deadline.
End-to-end wall time is not identical to a solver's configured remaining budget.

The diagnostic wraps existing functions without editing their source bytes or
changing arguments/results. It observes actual MILP objective identity, status,
message, passed budget, wall time, and caller phase. In particular, the original
empty-result helper defaults `stage_a_status` to the overall failure status even
when A never started; the diagnostic records a separate observed phase rather
than accepting that label as proof of an A-stage failure.

| measurement | fresh median / P95 / max (s) | shared median / P95 / max (s) |
| --- | --- | --- |
| correction call | .079763 / .114712 / .190798 | .080779 / .114088 / .172395 |
| snapshot including guard | .002708 / .028940 / .334375 | .002734 / .028571 / .238615 |
| validation before deadline | .000389 / .000453 / .001692 | .000395 / .000448 / .000500 |
| pre-R build including validation | .022647 / .028852 / .060540 | .023097 / .030078 / .052860 |
| R certificate / solve | .002950 / .003865 / .025730 | .002981 / .003873 / .027394 |
| A call | .024202 / .035095 / .103634 | .024364 / .034179 / .078300 |
| B call | .024389 / .034431 / .062415 | .024662 / .034158 / .061710 |
| post last solver | .004218 / .005139 / .008704 | .004275 / .005182 / .006690 |

These nested intervals are not additive independent quantities. Timers include
observer overhead; no profiler/line tracer ran. Each arm had 1152 A and 1152 B
solver calls; R required MILP on 4 steps and otherwise used the complete-service
zero-gap certificate. All actual solver calls returned optimal. Passed budgets
were .106224–.246953 s fresh and .122871–.246906 s shared; no observed MILP wall
time exceeded its passed remainder. These measurements apply to the new run,
not the missing historical trace. A snapshot taking .334 s was outside the
planner deadline and did not cause fallback.

## Fixed-policy experiment and reproducibility

Run `runs/m6p2b_seed0_diagnosis_v2/` used origins
`[5040,5088,5136,5184,5568,6576]`, environment seed 0, deterministic actions plus
sampling-generator seeds 0/1/2. Training seed remains 0; sampling seeds are not
additional training seeds. Each fresh case ran in its own process; the shared
arm ran all 24 cases in order in one process with a new environment and RNG per
episode. CPU thread settings came from the frozen training configuration.

All 48 episodes passed completion, frozen service, physical, inventory band,
no-fallback, reachability and terminal-gap checks. All raw and exec paired
differences were exactly zero. Strict export/load checks preserved the original
523-observation/21-action binding; no optimizer or multiplier was created.
Policy state hashes and all 25 release-source hashes remained unchanged.
Source final checkpoint SHA-256 remains:
`472b42467448a8c3ebe00a57a4f44000aea7a98574b43ba3316d14eded5ff935`.

`runs/m6p2b_seed0_replay_v1/` repeats the six recorded deterministic initial
snapshot/proposal pairs three times in fresh and shared processes, with observer
off/on: 36 paired comparisons, 72 solver invocations. All are equivalent under
the declared 1e-6 action tolerance with identical failure/stage/gap results.
These are causal inputs from the **new final-policy run**, not historical states.
The replay probe's limited initial-state coverage does not prove equivalence
for every near-deadline state or reproduce a three-hour workload.

Per-step JSONL retains raw/exec/actual storage powers, task IDs/status/work/rate/
deadlines/continuity, SOC, signed forecast provenance and visible/assumed masks,
model size, timing and process ID. Initial and first-failure/predecessor snapshot
capture is implemented; no failure captures were needed in the completed run.
Five standard artifacts and a recursive hash receipt are present for each run.

## Host evidence: correlation, not proven cause

Read-only macOS power logs are saved in the v2 run's `system_power_evidence.json`,
with exact commands, output and timezone (Asia/Shanghai). They show battery
10% at 04:13:12, 5% at 04:23:22, thermal pressure 1 at 04:25:16, returning to 0
at 04:26:06, then battery 4% at 04:30:12 and 3% at 04:35:42. A separate `pmset`
inspection showed low-power sleep at 04:52:14, after training had finished.
The new diagnostic ran on AC power. No CPU-frequency history was recovered.

Using the original latest-checkpoint mtime and subtracting recorded tail-batch
durations yields an **upper estimate** of 04:24:57.948 for index 503's start;
unmeasured checkpoint/log-writing intervals make this an estimate, not an exact
wall-clock timestamp. Thermal pressure is temporally nearby but returns to 0
while subsequent batches keep failing. Battery/thermal evidence therefore must
not be promoted to a proven CPU-throttling cause. Current results weaken the
hypothesis of a final policy that persistently fails on these dates, while
leaving long-process and historical host-state mechanisms unresolved.

## Checks, retained failures and next decision

- New diagnostic unit tests: 7 passed; explicit script Ruff and mypy passed.
- Full `make check`: running; final count and artifact receipt recorded at closure.
- v2 artifact receipt: 150 files verified immediately after completion.
- `runs/m6p2b_seed0_analysis_v1/` contains machine-readable timing and all 24
  paired raw/exec differences.
- `runs/m6p2b_seed0_diagnosis_v1/` is retained with failed status: 7 completed
  episodes, then intentionally interrupted before the next rollout because the
  recorder read forecast provenance from the wrong layer. Initial snapshots
  retained provenance, but per-step provenance was empty. The diagnostic-only
  fix and leakage-marked regression preceded the new v2 run; v1 was not overwritten.

No root-specific algorithm repair or short training acceptance is claimed.
Proceeding to a semantic repair or formal retraining requires a reproducible
failure mechanism (or a specifically registered runtime mitigation experiment),
with failing regression first. Original r7 failed training and final checkpoint
remain registered; no early checkpoint was selected. Seed 1/2 and formal
validation/test remain on hold. A finite, no-update process-duration/resource
probe with persisted per-step failures is the next discriminating experiment;
its runtime and host conditions must be specified before launching it.

## Git delivery

Branch `p5-eval-viz-m6-p2b-s0-diagnosis`, baseline `23cf26c`.
Commits so far: `920648c` card, `c544c83` failing specifications, `525f2c2` runner,
`ddfea2c` provenance/phase correction, `eca5885` same-input replay.
HTTPS push failed because no GitHub credential was available; SSH port 22 was
closed. The GitHub connector also reported that baseline `23cf26c` was absent
remotely. Local commits are retained; remote synchronization is **not complete**.
No merge, history rewrite or force push was attempted.
