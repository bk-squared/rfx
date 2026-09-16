# Graded-mesh S-parameter accuracy — the WR-90 control: results

Companion to `graded_mesh_sparameter_accuracy_predeclaration.md`, which was
committed as `f5712d6b` before any arm ran. No window, profile, decision rule
or expectation in that note is changed here. Corrections to the note's own
reasoning are appended as C1/C2 at the end and named where they apply.

**Artifact:** `tests/fixtures/graded_mesh_sparameter_accuracy/wr90_control.json`
— per-bin dumps for all 14 cells, the realized meshes, the rasterization
records, the allowance table, the Yee predictions and the recomputed verdicts.
**Driver:** `scripts/diagnostics/graded_mesh_sparameter_accuracy_measure.py`.
**Run:** local CPU, `jax 0.6.2`, backend `cpu`, `jax_enable_x64 = False`,
float32, rfx 1.8.0, worktree of `origin/main` `6d721a56`. 14 cells, 250 s of
solve time in total. No VESSL, no GPU.

---

## 1. Headline

| rule (note §4) | result |
|---|---|
| **R-1** `dev_B ≤ dev_C + A_B`, 13 gates × 17 bins | **holds everywhere.** Worst margin +1.2827e-3 on the magnitude gates, +0.0735° on the phase gates — the allowance is essentially untouched. |
| **R-2** `dev_B ≤ dev_A + A_B + A_dt` | **fires on one gate, one bin**: slab‖S11‖ vs Airy at 8.4 GHz, margin −1.271e-4. Traced in §4: arm B equals arm C to 1.05e-6, so the cause is the z refinement, not the grading. |
| **W-BC** `max\|S_B − S_C\| ≤ 1e-4` | **holds**: worst 1.047e-6 (declared expectation "≈ 1e-6"). |
| **F-Z** ratio-2.0 on z expected NOT to exceed the allowance | **holds**: `max\|S_D1 − S_C\| = 1.696e-6`. The observables cannot see the z axis at all. |
| **F-A** arm D2 must exceed the allowance | **fires as declared**: the thru's S21 phase residual exceeds by up to +2.205°, at every one of the 17 bins; no flux-lane magnitude exceeds, also as declared. |

**What that adds up to.** The multi-band z profile inside the #785 envelope
costs this fixture nothing measurable on any of the chain battery's
analytic-referenced gates — and the same fixture proves it cannot test the
envelope, because a deliberately out-of-envelope ratio on the same axis is
equally invisible (F-Z). The WR-90 control's real yield is elsewhere: it
measures the NU flux S-lane end to end against analytic oracles, it shows the
port extractor's z weighting is exact to float32 for a z-invariant mode, and
it measures — for the first time on an S-parameter — what grading the
**propagation** axis costs. That cost is a phase term the #785 reflection
budget does not bound, and it is predictable from the Yee dispersion relation
to better than 1 %.

---

## 2. Instrument provenance

Arm A (the battery's own uniform build, re-measured on this CPU at this
commit) against the two committed battery artifacts, per S entry:

| comparison | max abs diff | verdict |
|---|---|---|
| schema 4 (`fixture_931_realized_pec_forward2_run369367259427.json`) — thru | 5.96e-7 | pass (gate 1e-4) |
| schema 4 — pec_short | 5.09e-7 | pass |
| schema 4 — slab | 9.20e-7 | pass |
| schema 3 (`fixture_v18_close.json`) — thru | 1.20e-6 | pass |
| schema 3 — slab | 1.79e-6 | pass |
| schema 3 — pec_short | **5.02e-1** | **fail** — see C1 |

The schema-3 pec_short miss is entirely in **S22**; |S22| is 1.0000 in both,
so 0.502 is a phase rotation of 28.9°, and S11/S21/S12 agree to 2e-6 or
better. Schema 3 was measured under the pre-#931 σ = 1e10 device cell-fill
operator; today's main realizes PEC edges, which is exactly why schema 4
exists (`tests/fixtures/waveguide_chain_battery/README.md`: "preserved
unchanged and superseded because the device lane now applies realized PEC
edges"). Against the operator main actually ships, arm A reproduces the
committed run to 5e-7. No artifact was edited.

---

## 3. Realized meshes, rasterization and preflight

Every realized profile equals its declared vector cell for cell:

* arm B z (mm): `0.635 ×3 | 0.889 ×2 | 0.635 ×3 | 0.889 ×3 | 0.635 ×3`
  — 14 cells, every adjacent ratio exactly 1.4, 4 transitions, 3 fine bands;
* arm C z: `0.635 ×16`;
* arm D1 z (mm): `0.635 ×2 | 1.27 ×2 | 0.635 ×2 | 1.27 ×3 | 0.635 ×2`
  — 11 cells, ratio 2.0, 4 transitions;
* arm D2 x: 86 cells, `{1.27, 2.54}` mm, the 2.54 mm band 10 cells long;
* arm E x: 92 cells, `{1.27, 1.778}` mm, the 1.778 mm band 10 cells long.

Realized guide extents, every arm: Σ(y) − a = 3.5e-18 m, Σ(z) − b ≤ 1.7e-18 m,
Σ(x) − 121.92 mm ≤ 2.8e-17 m — float64 accumulation, nothing more.

**#325-class rasterization check — passes on every arm.** Read off the
assembled material arrays the solver uses (`pec_mask` for the short, εr > 2
for the slab):

| arm | PEC short cells | x run | y cells | z cells | slab cells | x run |
|---|---|---|---|---|---|---|
| A | 576 | [80, 83] | 18 | 8 | 1152 | [78, 85] |
| B | 1008 | [80, 83] | 18 | 14 | 2016 | [78, 85] |
| C | 1152 | [80, 83] | 18 | 16 | 2304 | [78, 85] |
| D1 | 792 | [80, 83] | 18 | 11 | 1584 | [78, 85] |

Both DUTs fill the whole cross-section on every arm and their x runs are
identical across arms: 4 cells for the short, 8 for the slab, as drawn. The
graded z mesh moved nothing onto the wrong cells, and #369's failure mode (a
conductor silently dropped on the NU lane) did not occur — the short is
present with the right cell count on all four arms.

**Settling witness, per drive, per arm:** −94.0 dB (worst, slab) to −98.3 dB,
against the −40 dB rule. No arm needed the record-length rerun.

**Preflight, verbatim** (the declared expectations in §2 of the
pre-declaration all held; the full strings are stored per arm in the
artifact):

* arms A, B, C: no ratio finding. Arms D1, D2, E: the ratio finding fires, and
  says what the pre-declaration said it would —
  D1: *"dz_profile max adjacent cell ratio 2.000 exceeds the validated
  multi-band grading cap 1.4 … the -54 dB reflection class at 30
  cells/wavelength (it scales as (dz/lambda)^2), against -44 dB measured at
  ratio 2.0. COST: an unquantified reflection and grading-dispersion error at
  that transition."*
  D2 and E: *"dx_profile max adjacent cell ratio 2.000 / 1.400 exceeds the
  in-plane grading threshold (the validated 1.4 multi-band cap is a z-axis
  envelope: no witness in it grades an in-plane axis) 1.3"*.
* every arm, both ports: *"the record is shorter than 3 x the far-boundary
  round trip — waveguide_port[0] (+x): T/tau_far = 2.125"* (2.124 on the
  `dt`-refined arms), plus the port index mirror audit note. Properties of the
  committed fixture at `num_periods = 40`, identical across arms.
* `pec_short`, every arm: *"PEC 'pec_like' x-extent 5.08mm = 4.0 cells —
  volume under-resolved"*.
* `slab`, every arm: the `10.2 cells per λ_eff` notice on x and y, plus the
  lossless-dielectric-in-CPML notice. The **z** notice is the one preflight
  string that differs between arms, and it differs correctly: it reads the
  arm's **coarsest realized z cell**. Arm A `10.2 cells per λ_eff …
  dx=1.27mm`; arm B `14.5 … dx=889µm`; arm D1 `10.2 … dx=1.27mm`; **arm C
  emits no z notice at all**, because its uniform 0.635 mm z cell reaches
  20.3 cells per λ_eff and clears the advisory's own ≥ 20 threshold.

No new finding appeared on any graded arm. Worth recording for its own sake:
preflight reads the graded axis correctly and reports the coarse band, so the
input-fidelity layer sees a difference between arms B, C and D1 that none of
the measured observables can see.

---

## 4. Per-gate, per-arm

Deviations are from the **analytic** reference, so all arms are judged against
the same numbers.

| arm \| dut | column power max | \|S11\| PEC short (min…max) | slab vs Airy, mag | slab vs Airy, phase | thru S21 residual |
|---|---|---|---|---|---|
| A \| thru | 1.0000032 | — | — | — | 0.03497° |
| A \| pec_short | 1.0000215 | 0.9999973 … 1.0000108 | — | — | — |
| A \| slab | 1.0000364 | — | 0.036165 | 10.2260° | — |
| B \| thru | 1.0000033 | — | — | — | 0.03376° |
| B \| pec_short | 1.0000169 | 0.9999966 … 1.0000085 | — | — | — |
| B \| slab | 1.0000340 | — | 0.037575 | 9.4280° | — |
| C \| thru | 1.0000033 | — | — | — | 0.03377° |
| C \| pec_short | 1.0000167 | 0.9999966 … 1.0000083 | — | — | — |
| C \| slab | 1.0000343 | — | 0.037576 | 9.4280° | — |
| D1 \| thru | 1.0000033 | — | — | — | 0.03377° |
| D1 \| pec_short | 1.0000172 | 0.9999966 … 1.0000086 | — | — | — |
| D1 \| slab | 1.0000338 | — | 0.037576 | 9.4280° | — |
| D2 \| thru | 1.0000035 | — | — | — | **1.45786°** |
| E \| thru | 1.0000033 | — | — | — | **0.34872°** |

Gates: column power ≤ 1.02, |S11| ∈ [0.99, 1.03], slab magnitude ≤ 0.05,
slab phase ≤ 15°. **Every arm passes every committed gate**, including the two
deliberately out-of-envelope ones.

### The one firing rule, traced (R5)

`R-2 | slab | slab_airy_s11_mag` fires at the 8.4 GHz bin only:

| f / GHz | dev_A | dev_B | dev_C | allowance | R-2 margin | R-1 margin |
|---|---|---|---|---|---|---|
| 8.4 | 0.036165 | 0.037575 | 0.037576 | 0.001283 | **−0.000127** | +0.001284 |
| 8.6 | 0.030484 | 0.031765 | 0.031766 | 0.001429 | +0.000148 | +0.001430 |
| 9.0 | 0.028209 | 0.029309 | 0.029310 | 0.001731 | +0.000630 | +0.001731 |
| 10.0 | 0.024855 | 0.025293 | 0.025293 | 0.002546 | +0.002108 | +0.002546 |
| 11.0 | 0.022514 | 0.022617 | 0.022617 | 0.003447 | +0.003344 | +0.003448 |
| 11.6 | 0.017159 | 0.016909 | 0.016909 | 0.004029 | +0.004279 | +0.004029 |

(the full 17-bin trace is in the artifact). Two readings, both independent of
the metric that fired:

1. **dev_B = dev_C at every bin**, to 1e-6. The graded profile contributes
   nothing; the shift from arm A is the z refinement (0.635 mm instead of
   1.27 mm, and with it a 29 % smaller `dt`). R-1, the equal-`dt` comparison
   the pre-declaration named as primary, holds with margin +1.28e-3.
2. The shift has the **sign and shape of a resolution change, not of a
   reflection**: it is largest at the band bottom (+1.41e-3 at 8.4 GHz) and
   changes sign near the band top (−2.5e-4 at 11.6 GHz), while the phase
   deviation improves at every bin (10.226° → 9.428°). A spurious reflection
   would add, not trade magnitude for phase.

So the firing is a statement about refining z at fixed dx, not about grading.
It is reported, not explained away: refining one transverse axis of this
fixture moves the slab magnitude comparison by about 1.4e-3 at the band
bottom, which is 3.5 % of the gate.

### Pairwise deltas

| pair | max \|ΔS\| | S21 phase rms |
|---|---|---|
| B − C, thru / pec_short / slab | 5.62e-7 / 4.81e-7 / 1.05e-6 | 0.0000° |
| D1 − C, thru / pec_short / slab | 7.55e-7 / 4.30e-7 / 1.70e-6 | 0.0000° |
| B − A, thru / pec_short / slab | 1.564e-2 / 1.463e-2 / 1.150e-2 | 0.6567° / 0.0000° / 0.6871° |
| C − A, thru / pec_short / slab | 1.564e-2 / 1.463e-2 / 1.150e-2 | 0.6567° / 0.0000° / 0.6871° |
| D2 − A, thru | 4.277e-2 | 1.4233° |
| E − A, thru | 9.439e-3 | 0.3141° |

B − A and C − A are equal to the digits shown on every DUT. That is the
cleanest statement the lane produces: **at equal minimum cell, the multi-band
z profile and the uniform-fine z mesh give the same S-matrix.**

---

## 5. The falsifier, and the term the allowance misses

### F-A: arm D2 (ratio 2.0 on the propagation axis), thru

| f / GHz | dev_A | dev_D2 | allowance A_B | excess |
|---|---|---|---|---|
| 8.4 | 0.01011 | 0.40294 | 0.07350 | +0.31933 |
| 10.0 | 0.03475 | 1.22116 | 0.14589 | +1.04052 |
| 11.6 | 0.04971 | 2.48577 | 0.23087 | **+2.20519** |

Degrees, all 17 bins over. The complex delta against arm A is 4.277e-2, ten
times the allowance's own maximum (4.029e-3). The flux-lane magnitudes do not
exceed (`column_power` max excess −1.283e-3, `reciprocity_complex`
−1.291e-3) — which the pre-declaration §1.3 said before the run, from the
structure of the two-run flux extraction. **The falsifier behaves, and it
behaves on exactly the observable the pre-declaration named.**

### The Yee-dispersion term, predicted before the run

| term | predicted (deg rms) | measured (deg rms) | rel. error |
|---|---|---|---|
| `dt` change alone (B / C vs A, thru) | 0.6586 | 0.6567 | **0.28 %** |
| arm D2's 25.4 mm band of 2.54 mm cells | 1.4186 | 1.4233 | **0.33 %** |
| arm E's 17.78 mm band of 1.778 mm cells | 0.3138 | 0.3141 | **0.09 %** |

All three from `Δφ = Σ_cells [β_Yee(f; d_cell, dt) − β_Yee(f; dx, dt_ref)]·d_cell`,
computed with no FDTD from the realized profiles. The phase cost of a graded
propagation path is not a reflection and is not bounded by the F-S2 budget;
it is the coarse cells' own numerical dispersion, and it is predictable to
under half a percent on this fixture.

### Arm E: the in-envelope ratio on the propagation axis

Ratio 1.4, two transitions, 17.78 mm of 1.778 mm cells in an 81.28 mm
reference-plane separation. Measured S21 phase residual **0.34872° rms**
against arm A's 0.03497°, an excess of 0.3141° rms — **over arm B's
reflection-derived allowance at all 17 of 17 bins**, max excess +0.309° at
11.6 GHz. Every committed gate still passes (the residual is a witness, not a
gate), and the excess is the Yee term above, predicted to 0.09 %.

This is the substantive answer this lane has for #810 beyond the control's own
scope: **inside the validated ratio, on the axis a wave actually traverses,
the reflection budget under-predicts the S-parameter cost — the binding term
is dispersive, not reflective.** Arm E grades an axis the support matrix
explicitly does not cover, so nothing is promoted on it; it is one measured
point, on one fixture, at one rung.

---

## 6. Cost

| arm | cells | steps | `dt` (s) | solve wall (s, per DUT) |
|---|---|---|---|---|
| A | 28 215 | 1 425 | 2.421350e-12 | 11.5 – 15.3 |
| B | 47 025 | 2 014 | 1.712153e-12 | 19.7 – 22.0 |
| C | 53 295 | 2 014 | 1.712153e-12 | 21.1 – 22.6 |
| D1 | 37 620 | 2 014 | 1.712153e-12 | 17.9 – 18.6 |
| D2 | 26 505 | 1 425 | 2.421350e-12 | 12.5 |
| E | 27 531 | 1 425 | 2.421350e-12 | 12.6 |

At equal finest cell, the multi-band profile buys **11.8 % fewer cells and
about 5 % less wall time than uniform-fine (B vs C), for a zero change in
every measured S-parameter** — the row #785 could not state before, now
stated for an observable. Against the case's own uniform mesh the graded arm
is 1.67× the cells and 1.41× the steps, because its finest cell is finer:
grading buys back part of a refinement's cost, it does not make refinement
free. On the propagation axis the arms coarsen instead of refine, so they are
cheaper than arm A and the phase cost is the number above: arm E is 2.4 %
fewer cells than arm A (27 531 vs 28 215) at 0.3141° rms of phase, arm D2 is
6.1 % fewer (26 505) at 1.4233°. Both keep arm A's `dt` and step count, since
their minimum cell is still 1.27 mm. Those savings are small because only a
10-cell band was coarsened; the point of the table is the rate, not the total
— coarsening 25.4 mm of an 81.28 mm path by 2× cost 1.42° rms, and the Yee
term above says what any other band would cost without running it.

---

## 7. What is NOT established

* **No promotion.** The evidence rule's fence ("Nonuniform S-parameter paths —
  Shadow unless they pass a co-refined analytic/external field or S-parameter
  oracle") is not lifted by this lane and this note proposes no support-matrix
  edit. Whether analytic oracles at one rung on one geometry class satisfy
  "co-refined analytic … oracle" is the PI's call.
* **No external solver.** cv11's Meep/openEMS legs are #854-deferred and were
  not used; every reference here is analytic.
* **One rung** (`mid`, dx = 1.27 mm), **one geometry class** (single-mode
  rectangular guide), **one lane** (`flux`), **one precision** (float32).
  No ladder, no AD leg, no plane-shift leg.
* **MSL is untouched.** #810's load-bearing Tier-1 case is cv06b, where a
  substrate/trace/air stack means the mode does traverse the graded axis.
  Nothing here transfers to it.
* **The z-axis envelope is still untested by an observable.** F-Z is the proof
  that this fixture cannot test it: a ratio-2.0 z profile, four transitions,
  changes the S-matrix by 1.7e-6. A fixture whose mode varies along the graded
  axis is required, and the WR-90 is not one.
* The `slab` arms sit at 10.2 cells per λ_eff inside εr = 4, with the
  preflight's own "~5 % |S21| deficit expected" notice standing. The
  arm-to-arm comparison is unaffected (all arms share it), but the absolute
  slab-vs-Airy numbers are that fixture's, not a resolution claim.

## 8. What the next attempt would need (R2)

R2 attempts on this mechanism hypothesis: **1**. A second attempt on the
WR-90 needs a named new falsifier, in writing, and "run it at another rung"
is not one — F-Z says the axis is invisible at every rung. The two
architectures that would actually advance #810 are: (a) the cv06b MSL case,
where the graded axis carries the mode; (b) a WR-90 variant with the graded
axis along propagation, judged against an external solver rather than against
the analytic oracles, since arm E already shows the analytic-referenced gates
pass while the phase moves by ten times the reflection budget.

---

## Corrections (append-only; no window moved)

**C1 — the pre-declaration picked the wrong reference artifact for one DUT.**
§1.4 chose `fixture_v18_close.json` (schema 3) as arm A's provenance
comparison, reasoning that it is "the run whose forward cells were measured
under the same operator the v1.8 chain closure quotes". That is true of the
closure, and false of today's main: the device lane's PEC operator changed
under #931, and the battery's README says so. The consequence is a 28.9°
S22 phase difference on `pec_short` and nothing else (§2). Repair: the driver
now compares arm A against **both** artifacts and stores both readings; the
schema-4 comparison is the one that speaks to this tree. Nothing about the
arms, the windows or the analytic references depends on the choice — every
decision rule in §4 of the pre-declaration is referenced to the analytic
oracle, not to either artifact.

**C2 — two reader defects in the driver, found by their own assertions.**
(i) `realized_mesh` sliced the padded cell array without dropping the NU
builder's trailing bounding node (#562), so every graded arm's first
rasterization record carried one extra cell and a sum error of exactly one
cell width. (ii) `dut_rasterization` read the PEC short off the `sigma`
array, but σ = 1e10 is above `Simulation._PEC_SIGMA_THRESHOLD` = 1e6, so the
block lives in `pec_mask` and the reader returned an empty conductor. Both
were repaired before the measured run, both now raise rather than report a
wrong number (the realized extent is checked against the declared extent to
1e-12 m, and a `pec_short` assembled with no `pec_mask` raises "#369 class"),
and all arms were re-run with the repaired readers. Neither defect touched a
solve, an S-matrix or a window; the S-parameters before and after the repair
are the same numbers (the driver is deterministic and the readers are
read-only).

**R3:** memory=`physics_validation_evidence_rule.md` line 129 + known-issues
chain-status row ("#810 dz-graded evidence absent") + the battery README's
schema-3/4 operator note | R2-attempts=1 | falsifier=F-A fired as declared
(arm D2, +2.205° at 11.6 GHz against a 0.231° allowance)
