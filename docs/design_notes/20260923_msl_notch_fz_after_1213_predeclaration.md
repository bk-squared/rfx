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

## Results (facts)

Appended after the runs; sections 0-7 above are unchanged.  Every number below is read
from `validation/research/multiband_nu/results/msl_notch_graded_fz_after_1213.json` and, for R2's four
rungs, from `msl_notch_graded_fz.json`, by `after_1213_markdown_tables()` in the instrument.
`tests/unit/nonuniform/test_msl_notch_fz_after_1213_replay.py` re-derives the same numbers from
the same two files.  Every notch is the |S21|^2 vertex (section 3) unless a column says `log`;
the log vertex judges nothing.  The reference is the openEMS tutorial's `stage_b_fine` read the
same way: 3.673868 GHz (|S21|^2), 3.674356 GHz (log).

### G.1 The mesh each arm solved, against its R2 namesake

Both sides of every comparison are the profiles and fields the GPU job's own build wrote into
its record; nothing is rebuilt here.  "Declared mesh fields" is every field of
`MESH_BLOCK_FIELDS`: the rung, the board drawn on it, the probe and feed placement, the grid
shape and the time step.  "Realized block" is R3's reading of the lattice (sheet plane, node
rows, permittivity either side of the sheet), which each `rfx/` tree takes for itself.

| arm | R2 namesake | F (um) | FZ (um) | substrate cells | z tail cell (um) | grid nodes | grid cells | dt (fs) | three profiles equal, bit for bit | declared mesh fields equal | realized block equal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Z6m | Z6 | 24.2915 | 42.3333 | 6 | 114.7303 | 267x270x33 | 2,289,728 | 52.5605 | True | True | True |
| Z8m | Z8 | 24.2915 | 31.7500 | 8 | 111.0135 | 267x270x38 | 2,647,498 | 49.8894 | True | True | True |
| Cm | C_off_re | 24.2915 | 21.1667 | 12 | 109.6286 | 267x270x47 | 3,291,484 | 44.0446 | True | True | True |
| Z16m | Z16 | 24.2915 | 15.8750 | 16 | 107.7639 | 267x270x56 | 3,935,470 | 38.4993 | True | True | True |
| Z6r | Z6 | 24.2915 | 42.3333 | 6 | 114.7303 | 267x270x33 | 2,289,728 | 52.5605 | True | True | True |

### G.2 What each arm measured

| arm | record | notch (GHz) | log (GHz) | above the reference (%) | depth (dB) | worst settling (dB) | worst passivity excess | grid cells | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| Z6m | new | 3.674809 | 3.672677 | +0.0256 | -44.65 | -91.45 | 0.00447 | 2,289,728 | 372.2 |
| Z8m | new | 3.676144 | 3.675747 | +0.0620 | -43.17 | -92.74 | 0.00450 | 2,647,498 | 739.1 |
| Cm | new | 3.677308 | 3.678996 | +0.0936 | -44.09 | -94.01 | 0.00450 | 3,291,484 | 1047.7 |
| Z16m | new | 3.677785 | 3.679892 | +0.1066 | -44.66 | -94.63 | 0.00452 | 3,935,470 | 1079.7 |
| Z6r | new | 3.752466 | 3.749938 | +2.1394 | -46.65 | -95.19 | 0.00434 | 2,289,728 | 357.4 |
| Z6 | R2 | 3.752466 | 3.749938 | +2.1394 | -46.65 | -95.19 | 0.00434 | 2,289,728 | 357.0 |
| Z8 | R2 | 3.736288 | 3.733780 | +1.6990 | -47.18 | -94.97 | 0.00441 | 2,647,498 | 715.9 |
| C_off_re | R2 | 3.718408 | 3.716635 | +1.2123 | -50.81 | -94.89 | 0.00443 | 3,291,484 | 1008.1 |
| Z16 | R2 | 3.708800 | 3.710389 | +0.9508 | -44.11 | -95.07 | 0.00447 | 3,935,470 | 1081.9 |

Bars, for the two witness columns: ring-down -40 dB, passivity excess 0.01.

### G.3 Provenance

`rfx/ tree` is `git rev-parse <commit>:rfx`, read by the job in the source repository.
`exported tree matches` compares it with the same hash taken of the `rfx/` the job exported and
imported, before anything imported it.  R2's arms carry neither field: they were recorded before
the instrument read them, and their blocks are left as measured.

| arm | VESSL run | commit | rfx/ tree | exported tree matches | GPU | jax | started (UTC) |
|---|---|---|---|---|---|---|---|
| Z6m | 369367263949 | 17971050aa0d | b91a81bf4fe2 | True | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-23T10:29:11+00:00 |
| Z8m | 369367263956 | 17971050aa0d | b91a81bf4fe2 | True | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-23T10:40:42+00:00 |
| Cm | 369367263959 | 17971050aa0d | b91a81bf4fe2 | True | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-23T10:45:07+00:00 |
| Z16m | 369367263960 | 17971050aa0d | b91a81bf4fe2 | True | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-23T10:48:42+00:00 |
| Z6r | 369367263950 | 9c8fd9b60520 | 2af37eafe9e1 | True | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-23T10:37:08+00:00 |
| Z6 | 369367263365 | d558382f3f5f | n/a | n/a | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-22T09:17:12+00:00 |
| Z8 | 369367263367 | d558382f3f5f | n/a | n/a | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-22T09:25:28+00:00 |
| C_off_re | 369367263518 | c4d6aee830e2 | n/a | n/a | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-22T16:38:45+00:00 |
| Z16 | 369367263368 | d558382f3f5f | n/a | n/a | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | 2026-09-22T09:28:13+00:00 |

The branch each new arm's commit lives on, both on origin:

| commit | branch | what it is |
|---|---|---|
| 17971050aa0d | `nu/notch-fz-after-1213` | the note's own branch: main cca8ee5b plus the note and this instrument; its rfx/ is main's |
| 9c8fd9b60520 | `nu/notch-fz-after-1213-revert1213` | a measurement branch, never merged: the commit above with #1213's rfx/ change reverted and nothing else, for Z6r |

### G.4 The frozen windows

**W9 -- did the board move, and was it #1213?**  Each rung against its R2 namesake: the same
mesh (G.1), a different `rfx/` tree (G.3).

| rung | R2 namesake | FZ (um) | R2 (GHz) | new (GHz) | new minus R2 (MHz) | R2, log (GHz) | new, log (GHz) | new minus R2, log (MHz) |
|---|---|---|---|---|---|---|---|---|
| Z6m | Z6 | 42.3333 | 3.752466 | 3.674809 | -77.6571 | 3.749938 | 3.672677 | -77.2613 |
| Z8m | Z8 | 31.7500 | 3.736288 | 3.676144 | -60.1437 | 3.733780 | 3.675747 | -58.0334 |
| Cm | C_off_re | 21.1667 | 3.718408 | 3.677308 | -41.0999 | 3.716635 | 3.678996 | -37.6389 |
| Z16m | Z16 | 15.8750 | 3.708800 | 3.677785 | -31.0151 | 3.710389 | 3.679892 | -30.4970 |

| Z6m minus Z6 (MHz) | bar (MHz) | verdict |
|---|---|---|
| -77.6571 | 1.00 | moved |

Attribution.  Z6r is Z6's mesh on main with #1213's `rfx/` change reverted, so on one
mesh the step from R2's Z6 to Z6m splits into the rest of the `rfx/` diff (R2's Z6 to Z6r)
and #1213's own change (Z6r to Z6m).

| estimator | R2's Z6 (GHz) | Z6r (GHz) | Z6m (GHz) | Z6r minus R2's Z6 (MHz) | Z6m minus Z6r (MHz) |
|---|---|---|---|---|---|
| power | 3.752466 | 3.752466 | 3.674809 | +0.0000 | -77.6571 |
| log | 3.749938 | 3.749938 | 3.672677 | +0.0000 | -77.2613 |

| abs(Z6r minus R2's Z6) (MHz) | bar (MHz) | attribution |
|---|---|---|
| 0.0000 | 0.05 | #1213 alone |

**W10 -- the new ladder.**  Z6m -> Z8m -> Cm -> Z16m against R2's Z6 -> Z8 -> C_off_re -> Z16, at
F = 24.2915 um.  Each step is from the rung above.

| FZ (um) | substrate cells | R2 rung | R2 (GHz) | step (MHz) | new rung | new (GHz) | step (MHz) | new, log (GHz) | step, log (MHz) |
|---|---|---|---|---|---|---|---|---|---|
| 42.3333 | 6 | Z6 | 3.752466 | n/a | Z6m | 3.674809 | n/a | 3.672677 | n/a |
| 31.7500 | 8 | Z8 | 3.736288 | -16.1780 | Z8m | 3.676144 | +1.3354 | 3.675747 | +3.0700 |
| 21.1667 | 12 | C_off_re | 3.718408 | -17.8804 | Cm | 3.677308 | +1.1634 | 3.678996 | +3.2492 |
| 15.8750 | 16 | Z16 | 3.708800 | -9.6076 | Z16m | 3.677785 | +0.4772 | 3.679892 | +0.8962 |

Every reading of the order, both ladders, each with its limit and the limit's distance from
the reference.  The first four take the limit from the finest two rungs the reading covers, at
its order; the three-parameter fit carries its own.  The declared rule is the first row.

| how the order is read | ladder | order | limit (GHz) | limit minus the reference (MHz) | same (%) |
|---|---|---|---|---|---|
| declared: least squares of the log step against the midpoint of each pair's log FZ | R2 | 0.7518 | 3.669008 | -4.8606 | -0.1323 |
| declared: least squares of the log step against the midpoint of each pair's log FZ | new | 1.4847 | 3.678681 | +4.8123 | +0.1310 |
| the same, each step at the coarser rung of its pair | R2 | 0.8030 | 3.671829 | -2.0396 | -0.0555 |
| the same, each step at the coarser rung of its pair | new | 1.5315 | 3.678647 | +4.7786 | +0.1301 |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | R2 | 0.7064 | 3.664495 | -9.3731 | -0.2551 |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | new | 1.4074 | 3.678820 | +4.9515 | +0.1348 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | R2 | 0.7958 | 3.671457 | -2.4113 | -0.0656 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | new | 1.5576 | 3.678629 | +4.7608 | +0.1296 |
| f_inf + A FZ^p fitted to all four notches at once | R2 | 0.7429 | 3.667992 | -5.8769 | -0.1600 |
| f_inf + A FZ^p fitted to all four notches at once | new | 1.4562 | 3.678729 | +4.8610 | +0.1323 |

The three-parameter fit leaves 0.0514 MHz rms on R2's ladder and 0.0052 MHz rms on the new
one, four points and three unknowns each.

| ladder | estimator | monotone | span, coarsest minus finest (MHz) | declared-rule order | declared-rule limit (GHz) |
|---|---|---|---|---|---|
| R2 | power | True | +43.6659 | 0.7518 | 3.669008 |
| R2 | log | True | +39.5489 | 1.3713 | 3.697475 |
| new | power | True | -2.9760 | 1.4847 | 3.678681 |
| new | log | True | -7.2153 | 1.7763 | 3.681236 |

| new span over R2's | half of R2's span (MHz) | order bar | 0.8 of R2's span (MHz) | verdict |
|---|---|---|---|---|
| -0.0682 | 21.8330 | 1.5 | 34.9328 | partly |

**W11 -- one limit.**  Both limits by the declared rule, under |S21|^2.

| new ladder's limit (GHz) | R2's limit (GHz) | abs difference (MHz) | same, % of the reference | bar (%) | bar (MHz) | R2's five readings' spread (%) | verdict |
|---|---|---|---|---|---|---|---|
| 3.678681 | 3.669008 | 9.6729 | 0.2633 | 0.20 | 7.3477 | 0.1996 | FIRED |

**W12 -- the substrate rule for the current solver.**  Reported, no verdict.  W8's derivation
on the new ladder, per reading: FZ = (bar x limit / abs(A))^(1/p) and n_z = ceil(h / FZ),
the bar 1 % of that reading's OWN limit.

| how the order is read | estimator | order | limit (GHz) | abs(A) | FZ for the bar (um) | n_z for the bar |
|---|---|---|---|---|---|---|
| declared: least squares of the log step against the midpoint of each pair's log FZ | power | 1.4847 | 3.678681 | 1.1951e+13 | 193.8880 | 2 |
| declared: least squares of the log step against the midpoint of each pair's log FZ | log | 1.7763 | 3.681236 | 4.5025e+14 | 102.3486 | 3 |
| the same, each step at the coarser rung of its pair | power | 1.5315 | 3.678647 | 1.9296e+13 | 184.1512 | 2 |
| the same, each step at the coarser rung of its pair | log | 1.8682 | 3.681151 | 1.1652e+15 | 96.6780 | 3 |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | power | 1.4074 | 3.678820 | 5.7318e+12 | 204.4266 | 2 |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | log | 0.8333 | 3.687079 | 6.3495e+10 | 130.7974 | 2 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | power | 1.5576 | 3.678629 | 2.5212e+13 | 179.1327 | 2 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | log | 2.6578 | 3.680672 | 4.4476e+18 | 67.6662 | 4 |
| f_inf + A FZ^p fitted to all four notches at once | power | 1.4562 | 3.678729 | 9.1633e+12 | 196.9212 | 2 |
| f_inf + A FZ^p fitted to all four notches at once | log | 1.3950 | 3.682534 | 1.2497e+13 | 108.5079 | 3 |

The finest rung, Z16m, from the reference: +0.1066 % (|S21|^2), +0.1507 % (log).

Conclusions: leader fills.
