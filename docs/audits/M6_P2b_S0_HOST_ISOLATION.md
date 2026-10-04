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
