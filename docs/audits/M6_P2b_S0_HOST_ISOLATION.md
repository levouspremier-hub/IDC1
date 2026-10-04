# Seed0 isolated-host diagnosis — 2026-10-04

## Evidence and current conclusion

Scheduling-v1 failed after 478.395 s shared operation (82 shared episodes,
24 fresh before / 24 fresh after, 6240 total steps). First B timeout coincided
with unusually high involuntary switches and nearby XProtect activity; same-input
immediate old and fresh process replays both timed out. See RUNTIME audit for
phase budgets/CPU, controls and probe interference. Host scheduling variability is
a concrete explanation for this observed failure. It does not establish the
exclusive trigger of historical index503, whose step-level evidence is absent.
No algorithm change or formal-training readiness is claimed.

## Strict full-policy host preflight failure

Job seed0-host-isolation-v1 pinned 8a3f1f2 failed before policy load/inference or
environment steps. B6 verifier rejected 520 local_pv_kw values differing during
Linux reconstruction. All failed inner/outer artifacts remain under
runs/remote_seed0-host-isolation-v1; local_transfer_receipt reports all files
verified. No checkpoint validation was bypassed, and no old manifest/asset
was rewritten. The separate portability task is repairing authentic frozen
recipe verification. As of this audit, current Mac strict verify_release still
rejects refs_v4 materializer_revision after those source changes; full loading
chain qualification remains pending. This preflight failure cannot explain the
original training anomaly.

## Bounded physical input comparison

Registered contract-only capsule: original origin5136/step13 serialized input
SHA f80d0ce944a333319ddbab4e574149048312e16e9fd0755c6264388f4a106586,
original release ledger SHA 412057ca837cbea5b4e08017317eae6c1d797897f32f7389fa38553354ddf22f,
original successful replay SHA 6397c5623eb752dbf3fc1ff0429dcd0429c18a86e03ac7db3984f88a793b9de9.
Every recorded asset/source byte hash and original checkpoint file hash is
checked; typed train snapshot/proposal only. No policy is loaded and this does
not replace strict checkpoint/full-environment qualification. Published solver
options and original .25 shared deadline remain unchanged.

Failing input-tamper specification committed 9ccbb53 before implementation
59abbeb. Eleven focused tests pass; Ruff and mypy pass. Mac run
runs/m6p2b_seed0_capsule_mac_v1 completes 128/128 without failure, zero executed
action difference from successful original input replay. All five artifact
hashes reverified. Wall median/P95/max .083577/.109042/.118946 s;
process CPU .099871/.124786/.158370 s; thread CPU .083090/.108333/.113017 s.
Timing includes observation overhead and does not establish four-hour stability.

Remote seed0-capsule-isolation-v1 pinned full
59abbeb0e855b0598801560261b607a8c05b75aa is queued behind the existing single-slot
portability gate; no competing remote heavy job is started. It uses existing
sealed input assets and a new output run-id m6p2b_seed0_capsule_wsl_v1.
WSL outcome and receipts remain pending. Automation has been updated to this
job and the strict-preflight limitations. After the authentic loading chain
passes, a newly registered full-policy probe is required before any longer
host qualification or short training. No formal training or held-out run starts.

## Completed WSL capsule / 2026-10-04 follow-up

Original remote job completed all 128 solves, but inner diagnosis FAILED:
index79 is B timeout; other127 optimal, maximum successful exec difference
3.327511877149192e-13. Reverified all14 outer receipt files and five inner
artifact hashes; local transfer receipt all_files_verified=true. Original
outer state succeeded/exit0 is a diagnostic CLI defect, not solver acceptance.
No historical evidence is overwritten or relabeled.

WSL wall median/P95/max .241579/.294944/.319371 s, process CPU
6.927133/7.222455/7.363023 s, thread CPU .238399/.294913/.319325 s.
The large process/thread CPU gap indicates substantial additional process CPU,
but no recorded pool inventory identifies which library/threads caused it.
The capsule did not apply original policy's frozen Torch setting, limiting host
comparison; this run does not qualify the original policy's execution environment.

First failed input is unchanged from adjacent successes. GC generation2 consumed
.116919 s (plus generation0 .002685 and generation1 .003006).
Pre-R build/validation .125005 s; R .004550; A optimal .034741 s with
.074687 budget; B timeout .069335 s with actual passed remainder .030667 s.
Adjacent index78 and80 are successful. Total correction wall >.25 alone is not
evidence that budget propagation is broken; native per-stage options show shared
remainder. This GC pause is an additional budget-consumption mechanism,
not a proof of historical503 cause or a production fix recommendation.

CLI result propagation: failing specification05b6a1f committed first; repair
17d3c11 returns exit1 after retaining all failed data/receipt. Restoration
specification1f5600f precedes a7180b7 diagnostic runtime modes; 13 focused tests,
Ruff and mypy pass. No solver, reward, physical constraints or update logic edited.
Full make gate/strict release remain blocked by separately owned provenance repair.

Two registered new128-solve counterfactual arms use original frozen Torch config,
GC-on/off in separate processes, same authenticated capsule/options/.25 budget.
GC-off is diagnostic only, never a production change. Submission attempts for
seed0-capsule-frozen-gc-on-v1 and seed0-capsule-frozen-gc-off-v1 at a7180b7 fail
SSH connection before reaching the remote submit command (exit255/port22 timeout).
No remote acknowledgement or watcher exists; these runs are NOT claimed started.
Preserve ambiguity if later recovery reveals requests; query status before any
submission and do not overwrite evidence. Bounded independent BatchMode SSH check
also times out. User must restore host/tailnet/SSH reachability; automated root
follow-up is paused for this external blocker. No training has started.
