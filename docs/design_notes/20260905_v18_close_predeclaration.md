# v1.8 waveguide chain closure — pre-declaration (2026-09-05)

**Status: PRE-DECLARATION.** Written before the closing measurement runs. Base: `origin/main`
at the SHA this note is committed on. Contract: `docs/design_notes/chain_closure_contract.md`.
Predecessors: `waveguide_chain_battery_predeclaration.md` (run 1, #867),
`waveguide_chain_battery_remeasure_predeclaration.md` (run 2, #893).

## 0. What is red, and why neither red is physics

After #893, 9 of 185 verdicts are red in two families. Both are float32 effects on the
differentiation side; the forward physics is green at the claims rung (column power
1.000975, complex reciprocity 6.98e-3, referees green).

**(a) Forward identity, flux lane, 8 legs.** `max|S_traced − S_untraced|` scaled to
rtol 1e-5 / atol 1e-7 reads 1.08–1.76 (> 1). Mechanism, from run 2's own x64 witness:
float32 reassociation of the 2849-step Poynting DFT under the reverse-mode tape. The same
identity in x64 reads 1.74e-10 / 2.83e-10 / 1.05e-8 on the three legs that carry the witness
— ten orders inside the bar. The other five legs lack the witness only because
`waveguide_chain_battery_measure.py` computes it for `objectives[0]` of each (dut, lane, θ)
group; the x64 primal identity does not depend on the objective, so the same `S64` serves
all four.

**(b) AD-vs-FD, `pec_short | flux | eps | s11_mag2`, 1 leg.** A physically zero derivative
(|S11| = 1 in front of a PEC for any lossless window). Run 1: float32 AD +2.683e-5 (its noise
floor, wrong sign), FD −7.245e-7, x64 AD −9.821e-7, ratio 1.36. **Run 2, on the corrected
port: FD −5.154e-8, x64 AD −2.943e-7, ratio 5.71, same sign.** Both fell toward zero with the
port fix (FD 14×, x64 AD 3.3×) — what a physically-zero derivative should do — so the ratio
of two shrinking O(1e-7) residuals is not a stable statistic. The FD is genuine: the float64
pair resolves 2.25e7 ULP against a 1e4 floor (the span helper is dtype-aware; not the #527
class). No precision defect to fix, and no pre-declared branch fits cleanly: run 2's x64 AD
is 3.34× off run 1's, just outside the remeasure note's "factor 3, same sign" branch.

## 1. The declaration (PI decision 2026-09-05)

> **Contract criterion 1 (forward identity) and criterion 3(a) (AD-vs-FD) are evaluated
> under x64 on the differentiable lanes.** The forward default stays `precision="float32"`.

Evidence the decision rests on, recorded so it is not re-argued: float32 is the forward use
case (default; every claims-bearing artifact; 15 of 21 crossval scripts) and has never been
short there; it has failed four times on the differentiation side (#527, #477 FD comparators;
#630 Yee arithmetic dtype; run 2's tape reassociation and zero-derivative leg); x64 costs
1.25–1.49× wall and 2× memory on the RTX 4090 with forward physics identical (+0.073 % band
mean, asy equal to four digits) — `docs/research_notes/20260904_vi_envelope_predeclaration.md`.

What this is NOT: a tolerance change. rtol 1e-5 / atol 1e-7 and rel 0.05 are untouched. The
float32 readings stay in the artifact as the recorded envelope of the float32 tape.

## 2. Changes, and the one new gate

1. **Script**: `forward_identity_x64` attached to every leg in a group (one `S64` per group,
   already computed); `g_ad_x64` computed for every flux-lane leg, not only `objectives[0]`.
   Both float32 and x64 readings stored; `primary_precision` field says which the gate reads.
2. **Gate module**: `forward_identity_pass` reads the x64 metric on lanes declared under x64.
   A `zero_derivative_entry` (sign and factor-3 on the x64 gradient against FD — the remeasure
   note's own §(ii) branch) is **computed and stored for the pre-declared zero-derivative
   leg, and that leg's verdict is `report_only`**, on the note's exit (c): the derivative is
   physically zero, both AD and FD are O(1e-7) discretization residuals that fell with the
   port correction, and a convergence verdict on their ratio would be a claim about noise.
   On run 2's stored numbers that ratio is 5.71 — it FAILS the factor-3 branch — written here
   so the report-only status is not mistaken for a pass. **Rejected alternative, on the
   record**: widening the factor to 6 so the leg passes. That is the silent gate loosening the
   repo forbids; a leg that needed the bar moved is a report, not a pass. Every other leg keeps
   rel ≤ 0.05; the weight-bearing PEC-short magnitude leg (`sigma`, rel 4.9e-4) carries that
   DUT's AD-vs-FD claim.
3. **Test**: the 8 + 1 `xfail(strict=True)` marks are removed AFTER the run shows green — strict
   xfail forbids removing them first.

## 3. Predicted outcome, written before the run

| family | legs | predicted | how it could instead fail |
|---|---|---|---|
| forward identity, flux, x64 | 8 | all ≤ 1e-7 scaled (three measured 1.7e-10…1.05e-8) | any leg > 1e-3 scaled: reassociation is NOT confined to the DFT; report next to the flux numbers, do not close |
| forward identity, `normalize=False` | 4 | bit-identical (0) on GPU at the claims rung, as run 2 stored; the CPU coarse-rung smoke read 0.23 and is not the claim | > 0 on GPU at the claims rung: the x64 context leaked into the `normalize=False` path |
| AD-vs-FD zero-derivative, x64 | 1 | **report_only** (declared in §2); stored ratio expected ~3–8, same sign, both |g| ≤ 1e-6 | sign flip, or either |g| above 1e-5: float64 tape and FD disagree on a NON-zero derivative — defect on both precisions, root-cause, do not close |
| AD-vs-FD, all other legs, x64 | 15 | rel ≤ 0.05, and ≤ run 2's float32 rel (1.0e-4…1.1e-2) | any leg's rel RISES under x64: the float32 pass was noise agreeing with noise — report as a finding |
| everything else | 152 | unchanged from run 2 to replay tolerance | any drift: the x64 context leaked into a non-AD path |

**Expected census: 184 pass / report_only + 1 report_only (the zero-derivative leg), 0 fail, 0 not_interpretable.** Zero-cost replay of run 2's artifact through the new gate: float32 primary → exactly 9 red (§4 falsifier holds); x64 primary → the 3 forward-identity legs carrying a witness go green at 1.7e-10 / 2.8e-10 / 1.05e-8 scaled; the 5 without one are what the run measures. If that lands, the
waveguide family is declared chain-closed (v1.8) through the ledger row and support matrix,
per the contract's "How a family is declared chain-closed".

## 3.1 CPU smoke of the closing script, before the run (coarse rung, local)

All eight flux legs read `primary_precision = x64` with a witness each; the zero-derivative
leg lands `report_only`, the other seven `pass`. x64 forward identity 0.08–0.14 scaled
against float32's 0.16–4.10 on the same legs — the declared shape. One thing to state ahead
of the run: **the float32 AD was NaN on all four `pec_short | flux` legs at the coarse rung
on CPU** (eps s11_mag2 / re_s11 / im_s11 and sigma s11_mag2; e.g. sigma: x64 −6.894 against
FD −6.898, rel 7e-4). The reader script counted four; my first read of the log tail saw one. Both GPU runs at the claims rung
read it finite at −6.4214, identical to six digits, so the closing run is not expected to
see it; if it does, the script's non-finite path (already designed for this) carries the
verdict on the x64 reading, the float32 NaN is stored beside it, and the row is reported as
a finding — a float32 tape producing a NaN the x64 tape does not is the declaration's
premise, not a surprise to it. It is not read as a pass by the x64 lane alone.

## 4. Falsifier for the declaration itself

Run the SAME script with the x64 primary disabled (float32 primary, one flag). It must
reproduce run 2's 9 red to the stored digits. If it does not, the script change altered
something other than which reading the gate reads, and the closing is not attributable to
the declaration.

## 5. Scope fences, unchanged

Uniform single-mode WR-90 only; junctions, multimode, nonuniform excluded; phase referee =
Airy only; `pec_short` / `slab` / `thru` at a/9, a/18, a/36. #854's four deferrals stay
deferred. The record-window finding of the VI-envelope campaign (#894) does not touch this
battery's committed numbers: its band sits at f/f_c ≥ 1.28 and 40→120-period twins moved
nothing above 3.3e-5 (post-merge review, `20260905_post_merge_review_20_prs.md`).

## R3

- Memory: `project_v18_kickoff_state` (PI decisions recorded), `project_precision_carry_dtype_family`
  (f64 cost), consistent with `feedback_root_cause_before_gate_change` — no tolerance moves; the
  one new gate is the pre-declaration's own branch language promoted, and only for the leg it
  was written for.
- R2 attempts: 3rd measurement of this battery; each previous one closed families with a named
  mechanism (#868 aperture, #869 witness) and this one names its mechanism (float32 tape) with
  an x64 witness already in hand. Not a repeat.
- Falsifier: §4 — the float32-primary rerun must reproduce the 9 red.

## 6. Outcome (written after the run; nothing above this line was edited)

Run: VESSL 369367258638, commit `f914a7ca`, gpu-rtx4090, solve wall 1350.5 s, 14:39–15:17 UTC
2026-09-05. Artifact `tests/fixtures/waveguide_chain_battery/fixture_v18_close.json`
(schema_version 3, supersedes run 2's `fixture_guide_cell_aperture.json`), replay
`tests/oracle/test_waveguide_chain_battery_v18_close.py`, log
`docs/vessl-logs/waveguide_chain_battery_v18_close_369367258638_completed.log` (primary
checkout). Reader: `scripts/diagnostics/waveguide_chain_battery_v18_close_read.py`.

§3, row by row, against the prediction:

| row | measured | branch |
|---|---|---|
| flux forward identity (8 legs, x64 primary) | worst scaled 1.051e-8 (`slab\|flux`), 1.738e-10 / 2.835e-10 (`pec_short\|flux` eps / sigma); abs 1.76e-15 / 2.31e-15 / 2.24e-14 | as predicted (≤ 1e-7); the float32 reading beside it is run 2's to the last digit (1.083215 / 1.514875 / 1.760720) |
| `normalize=False` identity (8 legs) | 0 on all, GPU, claims rung | as predicted |
| zero-derivative leg | `report_only`; x64 g_ad −2.943e-7, FD −5.154e-8, same sign, ratio 5.709 (outside the factor-3 band, written so report_only is not read as pass); float32 g_ad +2.786e-5, rel 541.5 | as predicted |
| other 15 AD-vs-FD legs | all pass, rel 1.22e-4 … 1.074e-2; FD spans 8.7e13 … 4.8e15 ULP on fourteen legs and 3.44e11 on the zero-derivative leg's `false`-lane sibling (its own number, as in run 2); no leg's rel rose above the noise floor under x64 | as predicted |
| everything else (152) | 18 cells bit-identical to run 2 (max\|ΔS\| = 0 on every cell, settling and column power identical); plane-shift rotations and the wrong-sign witness identical; all 10 ladders identical | as predicted, with one exception below |
| §3.1 NaN watch | no non-finite float32 gradient at the claims rung on the GPU (the four NaN legs were the coarse rung on CPU) | as predicted |

§4 falsifier: the same ad_fd stage with `RFX_CHAIN_PRIMARY=float32` gives **exactly 9 red**
(8 forward-identity flux legs + the zero-derivative leg), run 2's set. The declaration's own
refutation held.

Closing census after the pin step: **134 pass / 51 report_only / 0 fail / 0 not_interpretable**
(run 2: 126 / 50 / 9 / 0). The +1 report_only is the zero-derivative leg; the 8 fail → pass are
the flux forward-identity legs. No gate, tolerance or pin moved: the gradient-invariance pin
is 0.001 from the same envelope 2.3242906e-07 as run 2, and every ladder pin is run 2's.

Three things found after the run, recorded rather than smoothed:

1. **`recompute_verdicts` lacked the report_only branch** the measurement driver applied, so
   the pod-assembled `verdicts` dict read `fail` on the zero-derivative leg while the leg's own
   entry read `report_only` (the reader counted legs, which is why it printed 0). The branch
   was added to the gate module (schema_version ≥ 3 only; runs 1 and 2 replay unchanged) and
   the dict regenerated by the pin step. A declared lane whose leg was read at float32 now
   recomputes as `not_interpretable`, so the declaration cannot be satisfied vacuously.
2. **The plane-shift stage sourced its base gradient from `g_ad`**, which the declaration had
   made the x64 primary on the flux lane, against a float32 shifted gradient — a mixed reading
   the diff against run 2 exposed (six flux-lane `rel_change` values moved from ~1e-7 to
   5.9e-7 … 4.7e-6; the excluded zero-derivative leg to 96.3, which is
   |+2.804e-5 − (−2.943e-7)| / 2.943e-7 with +2.804e-5 the float32 shifted-plane gradient).
   Criterion 3(b) is not under the declaration, so the
   pin step rebuilt the six entries from the stored float32 base and shifted gradients:
   bit-identical to run 2 on eleven of twelve entries and 2.2e-16 apart on the twelfth (a
   degrees→radians round trip; inputs identical). The mixed reading is kept under
   `gradient_invariance_x64_base` as what it is — the float32 gradient's distance from x64 on
   that lane. The stage now writes the float32 base directly.
3. **§2 item 3 named marks that cannot be removed.** The 8 + 1 `xfail(strict=True)` marks
   live in run 1's replay (`KNOWN_RED` in `tests/oracle/test_waveguide_chain_battery.py`) and
   read the frozen run-1 artifact, whose numbers do not change; they stay, as run 2 left run 1's
   rotation marks. What closes is the run-3 replay: every verdict adjudicated at its measured
   value, zero red, zero xfail.

Also witnessed, free: every float32 AD gradient, every float32 forward-identity reading and
every x64 witness present in both runs is bit-identical between run 2 and run 3 (rel 0 on all
16 legs), so the flux lane's reverse-mode pass is reproducible on this GPU; the difference the
declaration reads is precision, not run-to-run noise.

Provenance strings written after the pod's assemble step, listed in the artifact's
`provenance.post_run_edits`: `run_id` (from the log filename; `VESSL_RUN_ID` was unset, as in
run 2), `supersedes` / `supersedes_reason` (the pod ran with run 2's constants). No number.

**Declared: the rectangular-waveguide family (uniform, single-mode, `normalize=False` and
`normalize="flux"`) is chain-closed (v1.8)** under the contract's four criteria, with criterion
1 and 3(a) read under x64 on the flux lane and the scope fences of §5 unchanged.
