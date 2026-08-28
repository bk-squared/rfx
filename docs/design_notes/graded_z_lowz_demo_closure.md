# graded-z efficiency demo — CLOSURE (falsifiers fired; demo STOPPED)

Date: 2026-08-29. The ONE R2-tight attempt pre-declared in
`graded_z_lowz_demo_predeclaration.md` (committed a43aa35, BEFORE any run)
has been executed and is SPENT. Per the STOP fence: no tuning, no second
attempt. This note records the outcome.

## Run record

- Script: `scripts/diagnostics/graded_z_lowz_demo.py` (production APIs,
  preflight ON and fully surfaced, raw extraction per issue #470).
- Full per-frequency traces + preflight text:
  `docs/design_notes/graded_z_lowz_demo_results.json`.
- Timing-only pilot: 24 s at num_periods=2 (arm B). Full attempt: arm A
  44.7 s, arm B 85.9 s — both drives per arm, single CPU session, well
  under the 20-min budget. No VESSL escalation needed.
- Validity preconditions ALL met: settling_db −103.5/−112.9 dB (arm A)
  and −98.4/−105.2 dB (arm B), far below the −40 dB rule;
  `multi_drive_solve` assembly both arms; 8 of 9 gate-band bins pass the
  `np.all(reliable)` screen (one 4.48 GHz bin flagged by the documented
  standing-wave-null screen, excluded as pre-declared).
- Boundedness protocol: no |S11| > 1 anywhere in band; min Re(V0/I0) over
  the gate band = +22.1 Ω (arm A) / +21.6 Ω (arm B) — both positive, so
  the |S11|≤1 ⇔ Re(V/I)≥0 theorem was never even stressed.

## Falsifier verdicts (thresholds as pre-declared; none moved)

| Falsifier | Arm A (uniform h/4 z) | Arm B (production graded) |
|---|---|---|
| F1 max\|S11\| ≥ 0.10 | held (0.0116) | held (0.0496) |
| F2 mean\|S21\| ≤ 0.95 | held (1.0005) | held (0.9991) |
| F3 med Z0 err ≥ 5% | held (25.15 Ω, 1.14%) | **FIRED (27.11 Ω, +6.58%)** |
| F4 eps_eff ∉ (1, 3.38) | held (3.193) | held (3.173) |
| F5 z-cell ratio < 2.0 | — | **FIRED (37/24 = 1.54)** |
| F6 z-cost ratio ≥ 1.0 | — | **FIRED (1.78)** |

Three falsifiers fired → the graded-z efficiency demo FAILS on both the
efficiency half (F5, F6) and the accuracy half (F3). The measured
wallclock corroborates F6 independently: the graded arm took 1.92× LONGER
than the uniform-fine-z arm on the same machine.

## Mechanism (both halves have a concrete, already-documented cause)

1. **Accuracy (F3):** `smooth_grading` is called by `_make_dz_profile`
   WITHOUT `preserve_regions`, so it inserts transition cells that inflate
   the profile (nominal 1.754 mm column → realized 2.380 mm) and destroy
   the exact substrate-top snap the function itself constructs: the
   254 µm interface lands mid-cell at fraction 0.346 — inside preflight's
   own [0.10, 0.40] mixed-cell danger zone, which preflight duly warned
   about pre-run. This is precisely the issue-#48 failure mode pinned by
   `tests/test_smooth_grading_preserve.py`; the production profile builder
   simply does not use the fix that test pins. The +6.6% Z0 bias against
   the 5% committed envelope follows.
2. **Efficiency (F5, F6):** the thirds rule splits the top substrate cell
   into 2/3 + 1/3, so dz_min = (1/3)·(h_sub/4) = 21.2 µm, which drags the
   NU Courant dt down 2.75× below the uniform-h/4 arm's dt. The z-cell
   saving that survives smoothing overhead (1.54×, not the ledger's ~3×)
   is smaller than the dt penalty, so the graded mesh COSTS ~1.8× more
   cell-steps per simulated second. On fixtures with a taller air column
   the cell ratio improves toward 3×, but the dt penalty is
   column-height-independent — the trade only breaks even when the air
   column is several times taller than this validated fixture class uses.

## What the attempt DID establish (positive, durable)

The escape-hatch geometry itself is sound: the wide low-Z line
(W = 6·h_sub = 1524 µm, HJ Z0 = 25.44 Ω) at xy dx = W/8 = 190.5 µm over
an ALIGNED uniform h_sub/4 dz-profile (arm A — the NU code path with an
anisotropic dz ≠ dx mesh) passes the full committed thru envelope with
big margins: |S11| ≤ 0.012, |S21| ≈ 1.000, Z0 within 1.14% of
Hammerstad-Jensen — far better than any committed 50 Ω thru fixture at
comparable cost (mean|S11| 0.116 at dx = 80 µm). The structural tension
(savings ⊥ resolution) is genuinely decoupled by the wide line; what
fails is the PRODUCTION PROFILE GENERATOR, not the physics or the
extractor.

Sub-item (b) witness (S11 re-referenced to the extracted Z0 via
V·I split): max over gate band 0.155 (arm A) / 0.174 (arm B) — an order
of magnitude larger than the wave-split S11 on the same record, dominated
by the probe-0 V/I standing-wave content; recorded as a witness only, as
pre-declared (no committed threshold exists).

## Disposition

- Demo: STOPPED by pre-declared falsifiers. Do not re-attempt the demo.
- The R2 reopen condition permitted by fence rules is an IMPLEMENTATION
  DEFECT finding, and one is recorded here: `_make_dz_profile` ignores
  `smooth_grading(preserve_regions=...)` — the exact machinery the repo
  already ships and pins for this failure mode — and its thirds-rule
  split imposes an unconditional 3× dt penalty. Fixing the profile
  builder (preserve the feature block; reconsider the thirds split's
  dz_min cost) is a separate correctness/feature work item on the
  builder, NOT a rerun of this demo. Any future demo re-attempt requires
  that architecture change first, per the fence.
- Preflight advisory gap noticed while running (not fixed here): the MSL
  substrate-resolution and mixed-cell advisories compute h_sub/dx from
  the scalar nominal dx even when a dz_profile resolves the substrate
  finely (they reported "1 substrate cell, h_sub/dx = 1.333" for BOTH
  arms, including the aligned 4-cell arm A). Harmless-loud here, but an
  NU-aware version would have discriminated the two arms.
