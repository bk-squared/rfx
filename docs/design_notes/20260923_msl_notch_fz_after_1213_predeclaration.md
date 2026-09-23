# The MSL notch's substrate-cell ladder after #1213 — pre-declaration (R3)

**Status:** pre-declaration, committed before any arm runs. The windows in §4 are frozen from
this commit, and each arm gets one attempt. Lane: nu-mesh. Tracker: #810. It follows
`20260922_msl_notch_fz_ladder_predeclaration.md` (record
`validation/research/multiband_nu/results/msl_notch_graded_fz.json`, "the R2 record"), whose
arms are reused here and not re-run.
**Written by** the NU lane leader (bk-workspace).

## 0. The question, physically

In R2, the in-plane cell stays at 24.29 µm and only the substrate cell is cut (42.3 → 31.75 →
21.2 → 15.9 µm). On that ladder the stub notch converged with order 0.71–0.80 (|S21|² vertex,
R2 §F.6), to 0.06–0.26 % below openEMS. Every rung was solved before main's #1213. Before
#1213, an E component tangential to a material interface multiplied by the ε of the one cell
that owns its edge. After #1213 it multiplies by the mean of the four cells sharing the edge.
#1213's own record says the old rule makes a dielectric interface first order, and that it
solved a three-cell substrate half a cell thin.

The substrate–air plane under the strip is such an interface. Its tangential E edges (Ex, Ey
on that plane) took the air cell's ε. The normal Ez edges, which lie inside the substrate
layer, took the substrate's. So for the tangential field, the top half-cell of the substrate
behaved as air. That half-cell is the layer right under the strip, beside it, where the
fringing field is strongest, and in effect it is an air gap of about FZ/2 under the strip.
Such a gap lowers ε_eff and raises the notch, more on a coarser substrate cell. That is the
direction R2 measured. This is the leader's hypothesis, and testing it is what this note is
for.

This note asks one question. **Was R2's substrate term the old interface rule?** If it was,
then on a tree containing #1213:
- each rung reads lower than it did,
- the ladder's span shrinks,
- its order rises toward two,
- and the substrate cell the mesher needs drops from R2's 17–21 cells to far fewer.

If it was not, the ladder barely moves and R2's rule stands for the current solver.

Both ladders solve the same board, and the interface rules differ only by a term that vanishes
with the cell. So as the substrate cell goes to zero both must reach the same limit. That
check needs no reference.

## 1. Facts reused (from the R2 record, not re-run)

|S21|² vertex (`refined_extremum(transform="power")`), R2 §F.6:

| arm | FZ (µm) | notch (GHz) |
|---|---|---|
| Z6 | 42.3333 | 3.752466 |
| Z8 | 31.7500 | 3.736288 |
| C_off_re | 21.1667 | 3.718408 |
| Z16 | 15.8750 | 3.708800 |

Span Z6 → Z16: 43.666 MHz. Declared-rule order 0.7518, limit 3.66901 GHz. The five readings
put the limit at 3.66450–3.67183 GHz. Reference: openEMS `stage_b_fine`, read the same way,
3.673868 GHz. Solver: `rfx/` tree of d558382f (= c4d6aee8).

## 2. Arms (five new solves)

| arm | tree | F (µm) | n | FZ (µm) | n_z | why |
|---|---|---|---|---|---|---|
| Z6m | main (after this note) | 24.292 | 24 | 42.333 | 6 | ladder point |
| Z8m | main | 24.292 | 24 | 31.750 | 8 | ladder point |
| Cm | main | 24.292 | 24 | 21.167 | 12 | ladder point |
| Z16m | main | 24.292 | 24 | 15.875 | 16 | ladder point |
| Z6r | main with #1213's `rfx/` change reverted | 24.292 | 24 | 42.333 | 6 | attribution: which change moved the board |

Every build parameter is R2's rung spec, unchanged: the instrument is `build_graded` with the
same rung entries. The build test must show each new arm's mesh bit-identical to its R2
namesake's stored mesh block before anything is submitted. The only intended difference is the
`rfx/` tree.

**About Z6r.** It is a branch off the same main commit with `git revert` of 76f68f9f applied to
`rfx/` only. If the revert does not apply cleanly, Z6r is dropped. In that case the report
says "the `rfx/` trees differ by #1213 and by N other commits", and it attributes nothing to
#1213 alone. The branch is never merged.

Cost: R2's walls were 357 / 716 / 1008 / 1082 s, plus Z6r ≈ 360 s, so about 1 GPU-hour on
RTX 4090s. VESSL: one job per arm through `scripts/vessl_submit.sh`, following the preset's
queue rule (all five at once if the preset has free nodes, otherwise at most two at a time).
Back up to `rfx-nu/runs/` first, and delete runs only after the record is committed.

## 3. Estimator

The primary is the |S21|² vertex (`refined_extremum(transform="power")`). R2 §F.6 showed it is
exact at a lossy zero on these bins (≤ 143 Hz on the recorded grid, against ±2.51 MHz for the
log parabola). The log vertex is printed beside it and judges nothing.

## 4. Frozen windows

- **W9: did the board move, and was it #1213?** For each rung, report Δ = f(new) − f(R2).
  Verdict "moved" if |Δ(Z6)| ≥ 1.0 MHz. That is seven times the estimator's worst error on
  the recorded grid and 1/16 of R2's first step. Attribution: if Z6r exists and
  |f(Z6r) − f_R2(Z6)| ≤ 0.05 MHz, then the rest of the `rfx/` diff moved Z6 by nothing, and
  #1213 alone moved it from R2's Z6 to Z6m. Otherwise the report names what the rest of the diff moved. If Z6r does not
  exist, no attribution.
- **W10: the new ladder.** Z6m → Z8m → Cm → Z16m. Report monotonicity, the span, the order by
  R2's declared rule, the three-parameter fit and the two step-ratio triples, each with its
  limit, and the limit's distance from 3.673868 GHz. Verdict "the interface was the substrate
  term" if the span is at most half of R2's (≤ 21.8 MHz) AND the declared-rule order is ≥ 1.5.
  "Not the interface" if the span is ≥ 0.8 of R2's (≥ 34.9 MHz). Between the two, "partly",
  reported with the numbers.
- **W11: one limit.** The two ladders, R2's and the new one, must converge to the same notch,
  because the rules differ by a term that vanishes with the cell. Take |L_new − L_R2| with both
  limits by the declared rule. HELD if it is ≤ 0.20 % of 3.673868 GHz, which is the spread of
  R2's five readings (0.1996 %). FIRED otherwise. A FIRED W11 means at least one of the two
  ladders is not in its asymptotic range, and the note says so rather than choosing between
  them.
- **W12: the substrate rule for the current solver.** W8's derivation (R2 §F.4) applied to
  the new ladder under |S21|²: FZ for 1 % of its own limit and n_z = ceil(h / FZ), per reading.
  Reported, no verdict. The finest rung's distance from the reference is reported beside it.

## 5. Expectations, written before the run

The expectation is that the interface was most of R2's substrate term. Each new rung reads
lower, Z6m most (by several MHz). The span falls well below half of R2's, the order comes out
near two, and W11 holds. The substrate rule falls to single-digit n_z. If W10 says "not the
interface", R2's rule stands for the current solver, and the next suspect is the field
singularity at the strip edge in the plane normal to the sheet (R2 §0's original hypothesis).

## 6. Instrument and regression

- No `rfx/` change on the branch that is merged. Z6r's branch is not merged.
- The R2 record and its replay are untouched. New record:
  `results/msl_notch_graded_fz_after_1213.json`. The new replay test re-derives W9–W12 from the
  two records with the instrument's own functions and pins the verdicts, fired ones included.
- The Results section is emitted by `--tables`, and the note's Results equal it byte for byte.
- Provenance for each arm: the commit the job read, the `rfx/` tree hash, run id, GPU device
  kind, and dirty state.
- New gates follow the two-mutation rule: turning the check off goes red, and so does reviving
  the defect with the helper call kept.

## 7. Who does what

One Opus instance implements from this note and submits the jobs. The leader reads the records
and writes the Conclusions. A separate Opus instance reviews.
