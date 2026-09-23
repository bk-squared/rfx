# The MSL notch filter's remaining first-order term — FZ ladder pre-declaration (R2)

**Status:** pre-declaration, committed before any arm runs. Windows in §4 are
frozen from this commit. One attempt per arm. Lane: nu-mesh. Trackers: #810;
follows `20260922_msl_notch_graded_mesh_predeclaration.md` (the record
`validation/research/multiband_nu/results/msl_notch_graded.json`, whose arms
are reused here and not re-run). Independent of the grid-core step 0
(`20260922_nu_grid_core_predeclaration.md`); it runs in parallel.
**Written by** the NU lane leader (bk-workspace, Fable) after the
independent design review of 2026-09-22 asked for exactly this ladder.

## 0. The question, physically

On the band-graded mesh the stub notch converges to the openEMS tutorial's
board at FIRST order in the cell (fitted 0.92 along a ladder that shrank the
in-plane fine cell F and the substrate cell FZ together: 47.2/42.3 →
35.9/31.75 → 24.3/21.2 µm), and the finest rung still reads 1.15 % high
(3.7166 vs 3.6744 GHz). Moving the in-plane nodes 0.35 cell inside the metal
RAISED the notch by 0.40 %, away from the reference, so the solved strip width
is not what keeps it high. A microstrip's field is singular at the strip's
edge in the plane normal to the sheet: the cell that resolves that
singularity is FZ (the sheet sits on a z node plane; the edge's normal-plane
field lives in the FZ × F cells around it), and the previous ladder could not
separate FZ from F because it cut both. This note asks: **which cell carries
the remaining first-order error — FZ (substrate-normal resolution at the
sheet), F (in-plane, the T-junction and the open end), or the edge offset
itself?** The answer is the "substrate rule" the mesher (step 1) needs as a
default, and it decides whether the notch's 1 % bar is reachable by refining
z alone (cheap: the z column is 24–47 cells) or needs F (expensive: F enters
squared).

## 1. Facts reused (from the committed record; not re-run)

| arm | F (µm) | FZ (µm) | placement | notch (GHz) | grid cells | wall (s) |
|---|---|---|---|---|---|---|
| A_on | 50.000 | 42.333 | on-node, n = 12 | 3.73232 | 1,502,976 | 101.4 |
| A_off | 47.244 | 42.333 | offset, n = 12 | 3.74739 | 1,545,600 | 106.1 |
| B_off | 35.928 | 31.750 | offset, n = 16 | 3.73221 | 2,106,340 | 333.1 |
| C_off | 24.292 | 21.167 | offset, n = 24 | 3.71663 | 3,291,484 | 1007.7 |

(`msl_notch_graded.json`, tables R.1–R.3 of the previous note; VESSL
369367263264/265/266/283 on one RTX 4090.) Reference: openEMS `stage_b_fine`
3.67436 GHz. Board and instrument: the previous note's, unchanged (line
2.159 mm from y_lo, 10 mm arms, `mode="laplace"`, probes 14/6/5 coarse cells,
C = 127 µm runways, ratio caps 1.3 in plane and 1.4 in z).

## 2. Arms (five new solves; offset placement unless stated)

| arm | F (µm) | n | FZ (µm) | n_z | what it isolates |
|---|---|---|---|---|---|
| Z6 | 24.292 | 24 | 42.333 | 6 | FZ ladder point at C's F |
| Z8 | 24.292 | 24 | 31.750 | 8 | FZ ladder point |
| (C_off, reused) | 24.292 | 24 | 21.167 | 12 | FZ ladder point |
| Z16 | 24.292 | 24 | 15.875 | 16 | the finest FZ; the next rung down |
| F16Z6 | 35.928 | 16 | 42.333 | 8 → **6** | F ladder point at A's FZ (with A_off and Z6 it gives three F values at FZ = 42.333) |
| ON13 | 46.154 | 13, on-node | 42.333 | 6 | on-node at nearly A_off's F (46.15 vs 47.24 µm, 2.3 % apart): the offset's own effect at fixed FZ and near-fixed F |

Everything else per arm is the previous note's: band margins (smallest whole
number of fine cells covering 500 µm), z band 0 → 2h at FZ then a ratio-1.4
ramp and a uniform tail solved per rung, nine cells of exactly C before each
in-plane absorber face, the same refusals R1–R4, the same estimator and
comparison functions, the same `n_freqs = 400`, `num_periods = 20`. The
instrument gains a general rung spec (n_z, n, placement) and writes a
SEPARATE record `results/msl_notch_graded_fz.json`; the first record is not
edited. Provenance per arm as before (commit from the submitter, GPU device
kind, run id).

## 3. Cost

Z6 ≈ 2.5 M cells at the A time step (≈ 300 s); Z8 ≈ 2.8 M at the B step
(≈ 500 s); Z16 ≈ 3.9 M at 3/4 of C's step (≈ 1800 s); F16Z6 ≈ 1.6 M
(≈ 150 s); ON13 ≈ 1.6 M (≈ 110 s). About one GPU-hour on one RTX 4090.

## 4. Frozen windows

- **W5 — the FZ ladder is a ladder.** Notches of Z6 → Z8 → C_off → Z16
  monotone; fitted order p_z from the four points (least squares of
  log|Δf| against log FZ on successive differences, the same rule as the
  previous note's W3, which is defined because these four are one board);
  the limit from the two finest at p_z. Reported: p_z, the limit, and the
  distance of Z16 and of the limit from 3.67436 GHz. Verdict rule: HELD if
  monotone and p_z ∈ [0.5, 2.5]; otherwise FIRED (the ladder is not a
  power law in FZ alone, which is itself the finding).
- **W6 — attribution at the A→C step.** With ΔZ = f(Z6) − f(C_off) (FZ
  42.333 → 21.167 at F = 24.292) and ΔF = f(A_off) − f(Z6) (F 47.244 →
  24.292 at FZ = 42.333), and the total T = f(A_off) − f(C_off) = 30.76 MHz:
  "FZ dominant" if |ΔZ| ≥ 2T/3, "F dominant" if |ΔF| ≥ 2T/3, "shared"
  otherwise. This is a pre-declared classification, not a pass/fail; the
  three numbers are reported whatever the class. The cross term
  T − ΔZ − ΔF is reported (it is zero only if the two effects add).
- **W7 — the offset's own sign.** If f(ON13) < f(A_off) — the on-node arm
  at nearly the same F reads lower than the offset arm — the offset raises
  the notch on its own (sign confirmed, magnitude f(A_off) − f(ON13)
  reported beside the 2.3 % F difference). If f(ON13) ≥ f(A_off), the
  previous note's 0.40 % was the F difference and the offset's own effect
  is below the F slope; recorded as such. The F slope at fixed FZ, from
  {A_on (n = 12, F = 50.0), ON13 (n = 13, F = 46.15)}, both on-node, is
  reported so the reader can correct A_on → A_off for its F change.
- **W8 — is z enough?** If W5 holds and the FZ-ladder limit is within 1 % of
  3.67436 GHz, refining z alone reaches the case's bar at F = 24.292 µm;
  the mesher's substrate rule is then "n_z such that the FZ term is below
  the bar", read from the ladder. If the limit stays above 1 %, the
  remaining term is in F (T-junction, open end) and the substrate rule
  alone does not close the bar. Reported either way with the number.

## 5. Expectations, written before the run

p_z near 1 (a first-order edge singularity in the normal plane); ΔZ
dominant (the previous ladder's 0.92 with both cells cut, and the offset
result, both point at z); Z16 within about 0.8 % of the reference; f(ON13)
below f(A_off) by a few tenths of a percent (the offset's own sign
confirmed). If instead ΔF dominates, the T-junction or the open end carries
the error and the next declaration is an F-only ladder at fixed FZ with the
stub length and width held exact.

## 6. Instrument requirements and regression

- No `rfx/` change. The first record and its replay test are untouched
  (`git diff` on `results/msl_notch_graded.json` empty).
- The rung spec generalizes without changing any existing arm's build:
  `build_graded("A", "offset")` etc. must produce bit-identical profiles
  (test: the five recorded arms' `profiles()` reproduce the stored
  `mesh` block bit-for-bit).
- Replay test `tests/unit/nonuniform/test_msl_notch_fz_replay.py`: the
  W5–W8 arithmetic re-derived from the two records with the instrument's
  own functions; recorded verdicts pinned, fired ones included.
- Build test extended for the new rungs (R1–R3 refusals and the three
  mutations of the previous note apply unchanged).
- VESSL: one job per arm, the previous job file with the arm parameter;
  backups under `rfx-nu/runs/` before anything else; runs deleted only
  after the record is committed.

## 7. Who does what

Implementation: one Opus instance from this note. The leader reads the
record and writes the conclusions. Review: a separate Opus instance, one
round; P1/P2 fixes back to the same reviewer. Documentation to #1171.

## A.2 C_off re-measured at the ladder's commit (declared after review, before running)

**Status:** declared 2026-09-22 after the independent review of this branch and
before the arm ran. It adds one solve and changes no window in §4: W5–W8 keep
their arms, their rules and their verdicts exactly as frozen. What it adds is a
second reading of each of them on a ladder that does not span two solver builds.

### Why

The ladder's third rung, C_off, is reused from the first record and was solved
at commit `827d5ecf`. Z6, Z8 and Z16 were solved at `d558382f`. Between those
two commits `rfx/` changed by 640 insertions and 102 deletions across 11 files
— the non-uniform runner, the CPML setup and the Yee E update among them (the
#1183 and #1190 work). The four notches therefore come from two solver builds,
and the order and limit W5 fits are read off the differences between them. The
review's arithmetic: a 1.41 MHz shift on C_off alone, 0.038 % of its notch,
moves the fitted order from 1.37 to 1.08 and the limit from 0.63 % above the
reference to 0.41 %. A ladder whose fitted order is that sensitive to one rung
must not span two builds.

### The arm

One new solve, `C_off_re`: the SAME rung entry as C_off — n_z = 12, n = 24,
offset placement, the same board, the same 10 mm arms — at this branch's
instrument commit. One attempt. Same refusals R1–R4, same estimator, same
`n_freqs` and `num_periods`. It is recorded in `msl_notch_graded_fz.json` under
its own key; C_off is not edited and not removed.

The two meshes must be identical for the comparison to be about the solver, so
that is checked rather than assumed: every declared mesh field equal, and all
three cell profiles compared bit-for-bit.

### What is reported

1. **The re-measurement itself.** f(C_off_re), f(C_off), their difference in MHz
   and in per cent, each arm's commit, VESSL run, grid cells, time step and wall
   time, and the two mesh checks above.
2. **W5, W6 and W8 twice.** Once on the arms §4 names (Z6, Z8, C_off, Z16 for
   W5 and W8; A_off, Z6, C_off for W6) and once with the re-measured rung in
   place of C_off. Both rows stay in the tables; neither replaces the other.
   Each block states how many commits of `rfx/` its rungs were solved at. W6's
   in-plane leg still spans two builds in both rows, because A_off comes from
   the first record either way, and the tables say so.
3. **The order read more than one way.** Four rungs support several readings of
   a single power and they need not agree, so all of them go in the record
   rather than in prose: the declared least-squares slope of log|step| against
   the midpoint of each pair's log FZ; the same slope with each step placed at
   the coarser rung of its pair; the order that exactly reproduces the ratio of
   two successive steps, one per consecutive triple (a coarse one and a fine
   one); and a plain three-parameter fit of f = f_inf + A·FZ^p to all four
   notches at once, with the rms it leaves. The §4 verdict is judged on the
   declared reading and on nothing else.

No conclusion is drawn here. The leader writes those.

## Results (facts)

Appended after the runs; sections 0-7 above are unchanged.  Every number below is read
from `validation/research/multiband_nu/results/msl_notch_graded_fz.json` and, for the three
reused arms, from `msl_notch_graded.json`, by `fz_markdown_tables()` in the instrument.
`tests/unit/nonuniform/test_msl_notch_fz_replay.py` re-derives the same numbers from the
same two files.  The reference notch, 3.67436 GHz, is the openEMS tutorial's `stage_b_fine`
read by the case's own estimator, not a number typed from section 1.

### F.0 What the instrument had to change, and why

One change, and it moves no window.  The rung table now holds a (substrate cells, cells
across the metal) pair per rung with the two independent, which is what an FZ ladder at a
fixed in-plane cell needs.  The arms already recorded are rebuilt from the first record and
compared against it (`tests/unit/nonuniform/test_msl_notch_graded_build.py`): every cell of every fine
band, every cell of every coarse run, every profile sum, the fine cell, the substrate cell,
the solved z tail cell, every declared coordinate and every segment length are
bit-identical.  What is not is the last bit of the cells inside a solved geometric ramp:
those arms were solved in the GPU job's container (numpy on python 3.10) and the gate runs
in the project venv (numpy 2.4 on python 3.11), and the two disagree in the last bit or two
of `np.sum` and of the power ufunc, which is what the ramp's cell ratio is bisected on.

Measured by rebuilding all 5 of them against `msl_notch_graded.json`, on the machine that printed this
table rather than from a sentence typed once:

| arm | ramp cells in its three profiles | cells whose last bit moved | worst distance (ulp) | every profile sums to the same domain |
|---|---|---|---|---|
| A_off | 10 / 17 / 12 | 7 | 6 | True |
| A_off_longarms | 10 / 17 / 12 | 1 | 2 | True |
| A_on | 10 / 17 / 12 | 2 | 2 | True |
| B_off | 8 / 20 / 13 | 3 | 1 | True |
| C_off | 16 / 27 / 14 | 1 | 1 | True |

So at most 4 cells of one profile and 7 of one arm, 14 across all 5,
out of profiles carrying 8 to 27 ramp cells each; worst distance 6 ulp, which is
about 1e-15 of a cell's own size, on cells in the coarse transition away from the metal.
No cell touching the line, the stub or the substrate is among them.  The gate refuses beyond
8 ulp; a rung changed by one cell moves these numbers by about 1e13 ulp.

The second change is forced by one arm.  A node sits on the metal's centre line only when
a whole number of fine cells reaches it, so only when the cell count across the 600 um
metal is even.  The arms below lay an odd number, their centre falls half a fine cell
between two node rows, and the feed is placed on the metal's own centre rather than on a
node: the MSL port lays its width span at the coordinate it is given plus and minus half
the trace width, so a node half a cell off would drive the strip off-centre.  R1 asks
instead that the two rows straddling the centre be one fine cell apart with the centre
exactly halfway.

| arm | cells across the metal | fine cell (um) | node rows either side of the centre (um) | centre declared at (um) | midpoint minus centre (m) |
|---|---|---|---|---|---|
| ON13 | 13 | 46.1538 | 2435.9231 / 2482.0769 | 2459.0000 | 2.17e-18 |

### F.1 The mesh each arm solved

The five new arms first, then the three arms reused from the first record.  Cell counts are
the cells the solver steps, interior plus the absorber pad.

| arm | new or reused | in-plane fine cell F (um) | cells across the metal | substrate cell FZ (um) | substrate cells | placement | z tail cell (um) | band margin (fine cells) | grid nodes | grid cells | dt (fs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C_off_re | new | 24.2915 | 24 | 21.1667 | 12 | offset | 109.6286 | 21 | 267x270x47 | 3,291,484 | 44.0446 |
| F16Z6 | new | 35.9281 | 16 | 42.3333 | 6 | offset | 114.7303 | 14 | 241x230x33 | 1,758,720 | 71.9353 |
| ON13 | new | 46.1538 | 13 | 42.3333 | 6 | on-node | 114.7303 | 11 | 232x212x33 | 1,559,712 | 85.3531 |
| Z16 | new | 24.2915 | 24 | 15.8750 | 16 | offset | 107.7639 | 21 | 267x270x56 | 3,935,470 | 38.4993 |
| Z6 | new | 24.2915 | 24 | 42.3333 | 6 | offset | 114.7303 | 21 | 267x270x33 | 2,289,728 | 52.5605 |
| Z8 | new | 24.2915 | 24 | 31.7500 | 8 | offset | 111.0135 | 21 | 267x270x38 | 2,647,498 | 49.8894 |
| A_on | reused | 50.0000 | 12 | 42.3333 | 6 | on-node | 114.7303 | 10 | 229x207x33 | 1,502,976 | 89.6116 |
| A_off | reused | 47.2441 | 12 | 42.3333 | 6 | offset | 114.7303 | 11 | 231x211x33 | 1,545,600 | 86.6012 |
| C_off | reused | 24.2915 | 24 | 21.1667 | 12 | offset | 109.6286 | 21 | 267x270x47 | 3,291,484 | 44.0446 |

### F.2 What the lattice realized

| arm | sheet plane z (um) | PEC volume cells | line node rows | stub node cols | metal node span (um) | stub length (um) | declared offset (um) | worst residual (m) |
|---|---|---|---|---|---|---|---|---|
| C_off_re | 254.0000 | 0 | 25 | 25 | 582.9960 | 12000.0000 | 8.5020 | 3.64e-18 |
| F16Z6 | 254.0000 | 0 | 17 | 17 | 574.8503 | 12000.0000 | 12.5749 | 6.49e-18 |
| ON13 | 254.0000 | 0 | 14 | 14 | 600.0000 | 12000.0000 | 0.0000 | 6.94e-18 |
| Z16 | 254.0000 | 0 | 25 | 25 | 582.9960 | 12000.0000 | 8.5020 | 3.64e-18 |
| Z6 | 254.0000 | 0 | 25 | 25 | 582.9960 | 12000.0000 | 8.5020 | 3.64e-18 |
| Z8 | 254.0000 | 0 | 25 | 25 | 582.9960 | 12000.0000 | 8.5020 | 3.64e-18 |
| A_on | 254.0000 | 0 | 13 | 13 | 600.0000 | 12000.0000 | 0.0000 | n/a |
| A_off | 254.0000 | 0 | 13 | 13 | 566.9291 | 12000.0000 | 16.5354 | n/a |
| C_off | 254.0000 | 0 | 25 | 25 | 582.9960 | 12000.0000 | 8.5020 | n/a |

### F.3 What each arm measured

| arm | notch (GHz) | above the reference (%) | depth (dB) | -10 dB BW (MHz) | fitted Z0 (ohm) | worst settling (dB) | worst passivity excess | grid cells | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| C_off_re | 3.71663 | +1.1507 | -50.81 | 777.8 | 48.38 | -94.89 | 0.00443 | 3,291,484 | 1008.1 |
| F16Z6 | 3.74828 | +2.0119 | -50.64 | 786.0 | 48.60 | -95.41 | 0.00432 | 1,758,720 | 160.8 |
| ON13 | 3.73338 | +1.6065 | -47.76 | 771.5 | 47.09 | -93.14 | 0.00453 | 1,559,712 | 91.7 |
| Z16 | 3.71039 | +0.9807 | -44.11 | 775.3 | 48.33 | -95.07 | 0.00447 | 3,935,470 | 1081.9 |
| Z6 | 3.74994 | +2.0570 | -46.65 | 785.0 | 48.41 | -95.19 | 0.00434 | 2,289,728 | 357.0 |
| Z8 | 3.73378 | +1.6173 | -47.18 | 781.7 | 48.41 | -94.97 | 0.00441 | 2,647,498 | 715.9 |
| A_on | 3.73232 | +1.5775 | -51.07 | 770.4 | 47.00 | -92.89 | 0.00455 | 1,502,976 | 101.4 |
| A_off | 3.74739 | +1.9876 | -54.88 | 786.6 | 48.74 | -95.48 | 0.00430 | 1,545,600 | 106.1 |
| C_off | 3.71663 | +1.1507 | -50.81 | 777.8 | 48.38 | -94.89 | 0.00443 | 3,291,484 | 1007.7 |

Bars, for the two witness columns: ring-down -40 dB, passivity excess 0.01.

### F.4 The frozen windows

**W5 -- the FZ ladder is a ladder (as section 4 declares it).**  Four arms at one in-plane cell,
F = 24.2915 um; only the substrate cell moves.  The four rungs were solved at 2 commits
(d558382f3f5f, 827d5ecf6021), and the simulator itself -- the `rfx/` tree at each of them -- is NOT the same at all of them.
A commit that touches only a note, a test or this instrument leaves the simulator alone, so
what a ladder needs is one solver and not one label.

| arm | substrate cells | FZ (um) | z tail cell (um) | dt (fs) | notch (GHz) | step from the rung above (MHz) |
|---|---|---|---|---|---|---|
| Z6 | 6 | 42.3333 | 114.7303 | 52.5605 | 3.74994 | n/a |
| Z8 | 8 | 31.7500 | 111.0135 | 49.8894 | 3.73378 | -16.158 |
| C_off | 12 | 21.1667 | 109.6286 | 44.0446 | 3.71663 | -17.145 |
| Z16 | 16 | 15.8750 | 107.7639 | 38.4993 | 3.71039 | -6.246 |

Two things move down the ladder besides the substrate cell, and both are consequences of
it rather than free choices.  The air column above the band is a fixed 1.246 mm and its
nine-cell uniform tail is solved to fill it, so the tail cell shrinks as FZ does; and the
time step is the Courant limit of the smallest cell, so it falls with FZ too.  Neither
changes the board: the metal, the substrate and the in-plane cell are the same on all
four rungs, which F.1 and F.2 show.

| monotone | fitted order p_z | order window | limit (GHz) | reference (GHz) | finest rung from the reference (%) | limit from the reference (%) | verdict |
|---|---|---|---|---|---|---|---|
| True | 1.3713 | 0.5 to 2.5 | 3.69748 | 3.67436 | 0.9807 | 0.6292 | HELD |

Four rungs carry more than one way to read an order, and they do not have to agree.  The
window is judged on the first row, which is what section 4 asks for; the rest are reported
so the spread is in the record.

| how the order is read | order | limit (GHz) |
|---|---|---|
| declared: least squares of the log step against the midpoint of each pair's log FZ | 1.3713 | 3.69748 |
| the same, each step at the coarser rung of its pair | 1.4448 | n/a |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | 0.8258 | 3.67352 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | 1.8888 | 3.70174 |
| f_inf + A FZ^p fitted to all four notches at once | 1.1908 | 3.69189 |

The three-parameter fit leaves 0.517 MHz rms on four points with three unknowns.  The
declared fit's three residuals in the log step are -1.78e-01, +3.56e-01, -1.78e-01.

**W5 -- the FZ ladder is a ladder (addendum A.2, one solver build).**  Four arms at one in-plane cell,
F = 24.2915 um; only the substrate cell moves.  The four rungs were solved at 2 commits
(d558382f3f5f, c4d6aee830e2), and the simulator itself -- the `rfx/` tree at each of them -- is the SAME at all of them.
A commit that touches only a note, a test or this instrument leaves the simulator alone, so
what a ladder needs is one solver and not one label.

| arm | substrate cells | FZ (um) | z tail cell (um) | dt (fs) | notch (GHz) | step from the rung above (MHz) |
|---|---|---|---|---|---|---|
| Z6 | 6 | 42.3333 | 114.7303 | 52.5605 | 3.74994 | n/a |
| Z8 | 8 | 31.7500 | 111.0135 | 49.8894 | 3.73378 | -16.158 |
| C_off_re | 12 | 21.1667 | 109.6286 | 44.0446 | 3.71663 | -17.145 |
| Z16 | 16 | 15.8750 | 107.7639 | 38.4993 | 3.71039 | -6.246 |

Two things move down the ladder besides the substrate cell, and both are consequences of
it rather than free choices.  The air column above the band is a fixed 1.246 mm and its
nine-cell uniform tail is solved to fill it, so the tail cell shrinks as FZ does; and the
time step is the Courant limit of the smallest cell, so it falls with FZ too.  Neither
changes the board: the metal, the substrate and the in-plane cell are the same on all
four rungs, which F.1 and F.2 show.

| monotone | fitted order p_z | order window | limit (GHz) | reference (GHz) | finest rung from the reference (%) | limit from the reference (%) | verdict |
|---|---|---|---|---|---|---|---|
| True | 1.3713 | 0.5 to 2.5 | 3.69748 | 3.67436 | 0.9807 | 0.6292 | HELD |

Four rungs carry more than one way to read an order, and they do not have to agree.  The
window is judged on the first row, which is what section 4 asks for; the rest are reported
so the spread is in the record.

| how the order is read | order | limit (GHz) |
|---|---|---|
| declared: least squares of the log step against the midpoint of each pair's log FZ | 1.3713 | 3.69748 |
| the same, each step at the coarser rung of its pair | 1.4448 | n/a |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | 0.8258 | 3.67352 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | 1.8888 | 3.70174 |
| f_inf + A FZ^p fitted to all four notches at once | 1.1908 | 3.69189 |

The three-parameter fit leaves 0.517 MHz rms on four points with three unknowns.  The
declared fit's three residuals in the log step are -1.78e-01, +3.56e-01, -1.78e-01.

**W6 -- attribution at the A_off to C_off step (section 4's rung).**  A_off and Z6 share the
substrate cell, so what separates them is the in-plane cell; Z6 and C_off share the in-plane
cell, so what separates them is the substrate cell.

| leg | commits | the `rfx/` tree at them |
|---|---|---|
| substrate, Z6 -> C_off | d558382f3f5f, 827d5ecf6021 | two different simulators |
| in-plane, A_off -> Z6 | 827d5ecf6021, d558382f3f5f | two different simulators |

A_off comes from the first record in both rows, so the in-plane leg is read across two
simulators whichever rung the substrate leg uses.

| step | what moves | from (um) | to (um) | notch shift (MHz) | share of the total |
|---|---|---|---|---|---|
| A_off -> Z6 | in-plane cell F | 47.2441 | 24.2915 | -2.549 | -0.0829 |
| Z6 -> C_off | substrate cell FZ | 42.3333 | 21.1667 | +33.303 | +1.0829 |
| A_off -> C_off | both | | | +30.754 | 1.0000 |

| abs(dZ) (MHz) | abs(dF) (MHz) | bar, two thirds of abs(T) (MHz) | classification |
|---|---|---|---|
| 33.303 | 2.549 | 20.503 | FZ dominant |

The cross term T - dZ - dF is 0.000e+00 Hz and is zero by construction, not by
measurement: A_off to Z6 to C_off is one path from A_off's corner of the (F, FZ) plane to
C_off's, so its two legs telescope.  A cross term that could be non-zero needs the fourth
corner, F = 47.244 um at FZ = 21.167 um, which is not an arm of this note.  The printed
value is the float64 residue of the cancellation.

**W6 -- attribution at the A_off to C_off_re step (addendum A.2's rung).**  A_off and Z6 share the
substrate cell, so what separates them is the in-plane cell; Z6 and C_off_re share the in-plane
cell, so what separates them is the substrate cell.

| leg | commits | the `rfx/` tree at them |
|---|---|---|
| substrate, Z6 -> C_off_re | d558382f3f5f, c4d6aee830e2 | one simulator |
| in-plane, A_off -> Z6 | 827d5ecf6021, d558382f3f5f | two different simulators |

A_off comes from the first record in both rows, so the in-plane leg is read across two
simulators whichever rung the substrate leg uses.

| step | what moves | from (um) | to (um) | notch shift (MHz) | share of the total |
|---|---|---|---|---|---|
| A_off -> Z6 | in-plane cell F | 47.2441 | 24.2915 | -2.549 | -0.0829 |
| Z6 -> C_off_re | substrate cell FZ | 42.3333 | 21.1667 | +33.303 | +1.0829 |
| A_off -> C_off_re | both | | | +30.754 | 1.0000 |

| abs(dZ) (MHz) | abs(dF) (MHz) | bar, two thirds of abs(T) (MHz) | classification |
|---|---|---|---|
| 33.303 | 2.549 | 20.503 | FZ dominant |

The cross term T - dZ - dF is 0.000e+00 Hz and is zero by construction, not by
measurement: A_off to Z6 to C_off_re is one path from A_off's corner of the (F, FZ) plane to
C_off_re's, so its two legs telescope.  A cross term that could be non-zero needs the fourth
corner, F = 47.244 um at FZ = 21.167 um, which is not an arm of this note.  The printed
value is the float64 residue of the cancellation.

**Reported, no window: three in-plane cells at one substrate cell** (FZ = 42.3333 um).  W6's dF is the first and last of these.

| arm | F (um) | notch (GHz) |
|---|---|---|
| A_off | 47.2441 | 3.74739 |
| F16Z6 | 35.9281 | 3.74828 |
| Z6 | 24.2915 | 3.74994 |

Monotone in F: True.

**W7 -- the offset's own sign.**  ON13 puts its nodes ON the metal edges at nearly A_off's
in-plane cell and the same substrate cell (42.3333 um).

| arm | placement | F (um) | notch (GHz) |
|---|---|---|---|
| A_off | offset | 47.2441 | 3.74739 |
| ON13 | on-node | 46.1538 | 3.73338 |
| A_on | on-node | 50.0000 | 3.73232 |

| A_off minus ON13 (MHz) | same (%) | their F gap (%) | F slope on the two on-node arms (MHz/um) | what that slope makes of the F gap (MHz) | on-node reads lower |
|---|---|---|---|---|---|
| +14.006 | +0.3752 | +2.3622 | -0.2767 | -0.302 | True |

Reading, per the pre-declaration: the offset raises the notch on its own: the on-node arm at nearly the same in-plane cell reads lower.

**W8 -- is z enough? (section 4's ladder)**  Where the notch goes as the substrate cell goes to
zero at F = 24.2915 um, against the reference.  Two different simulators under the four rungs.

| arm | substrate cells | FZ (um) | notch (GHz) | from the reference (%) |
|---|---|---|---|---|
| Z6 | 6 | 42.3333 | 3.74994 | 2.0570 |
| Z8 | 8 | 31.7500 | 3.73378 | 1.6173 |
| C_off | 12 | 21.1667 | 3.71663 | 1.1507 |
| Z16 | 16 | 15.8750 | 3.71039 | 0.9807 |

| ladder limit (GHz) | reference (GHz) | limit from the reference (%) | bar (%) | W5 | verdict |
|---|---|---|---|---|---|
| 3.69748 | 3.67436 | 0.6292 | 1 | HELD | HELD |

Derived, with its derivation: the fit puts the notch abs(A) FZ^p from its OWN limit, so
the cell that brings that term inside the bar is
FZ = (bar x limit / abs(A))^(1/p) and the substrate rule is n_z = ceil(h / FZ).  The bar
is a fraction of the limit, not of the external reference: what is being bounded is how
far the notch still has to fall on this ladder, and the ladder converges to its own limit.

| abs(A) | p | bar, 1 % of the limit (MHz) | FZ for the bar (um) | n_z for the bar |
|---|---|---|---|---|
| 4.9246e+13 | 1.3713 | 36.975 | 34.1871 | 8 |

**W8 -- is z enough? (addendum A.2's ladder)**  Where the notch goes as the substrate cell goes to
zero at F = 24.2915 um, against the reference.  One simulator under the four rungs.

| arm | substrate cells | FZ (um) | notch (GHz) | from the reference (%) |
|---|---|---|---|---|
| Z6 | 6 | 42.3333 | 3.74994 | 2.0570 |
| Z8 | 8 | 31.7500 | 3.73378 | 1.6173 |
| C_off_re | 12 | 21.1667 | 3.71663 | 1.1507 |
| Z16 | 16 | 15.8750 | 3.71039 | 0.9807 |

| ladder limit (GHz) | reference (GHz) | limit from the reference (%) | bar (%) | W5 | verdict |
|---|---|---|---|---|---|
| 3.69748 | 3.67436 | 0.6292 | 1 | HELD | HELD |

Derived, with its derivation: the fit puts the notch abs(A) FZ^p from its OWN limit, so
the cell that brings that term inside the bar is
FZ = (bar x limit / abs(A))^(1/p) and the substrate rule is n_z = ceil(h / FZ).  The bar
is a fraction of the limit, not of the external reference: what is being bounded is how
far the notch still has to fall on this ladder, and the ladder converges to its own limit.

| abs(A) | p | bar, 1 % of the limit (MHz) | FZ for the bar (um) | n_z for the bar |
|---|---|---|---|---|
| 4.9246e+13 | 1.3713 | 36.975 | 34.1871 | 8 |

### F.4a The same rung solved twice, at two solver builds (addendum A.2)

`C_off_re` and `C_off` declare the same mesh -- the same rung entry, so the same 12 substrate cells and
24 cells across the metal, the same board and the same stub.  What differs is the commit of `rfx/`
that solved them.  Whether their two meshes really were identical is checked rather than
assumed, because a mesh that moved would make this comparison say nothing about the solver.

| arm | commit | VESSL run | notch (GHz) | grid cells | dt (fs) | wall (s) |
|---|---|---|---|---|---|---|
| C_off_re | c4d6aee830e2 | 369367263518 | 3.716635 | 3,291,484 | 44.0446 | 1008.1 |
| C_off | 827d5ecf6021 | 369367263283 | 3.716635 | 3,291,484 | 44.0446 | 1007.7 |

| C_off_re minus C_off (MHz) | same (%) | notch equal to the last bit | whole S curve equal to the last bit | every declared mesh field equal | all three cell profiles bit-identical |
|---|---|---|---|---|---|
| +0.0000 | +0.00000 | True | True | True | True |

The two commits differ in `rfx/`.  What the table above reports is what
that difference did to this board, measured rather than argued.

### F.4b Meshes costed but not solved

Built through the same `build_graded` every arm uses and put through the same R1 and R3,
so a mesh that could not be solved cannot be costed here either.  The time step is the
grid's own Courant number read off the lattice, and the step count is the one a run would
derive from it for the case's 20 periods.  No field is stepped and no S parameter exists for
these rows.

| cells across the metal | F (um) | substrate cells | FZ (um) | z tail cell (um) | grid nodes | grid cells | dt (fs) | time steps | what it is |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 47.2441 | 16 | 15.8750 | 107.7639 | 231x211x56 | 2,656,500 | 47.3494 | 60,342 | the coarse in-plane band with a fine z band: A_off's 12 cells across the metal at Z16's 16 substrate cells |

### F.5 Provenance

| arm | VESSL run | commit | dirty | GPU | jax | backend | started (UTC) |
|---|---|---|---|---|---|---|---|
| C_off_re | 369367263518 | c4d6aee830e2 | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T16:38:45+00:00 |
| F16Z6 | 369367263366 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:22:04+00:00 |
| ON13 | 369367263364 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:17:14+00:00 |
| Z16 | 369367263368 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:28:13+00:00 |
| Z6 | 369367263365 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:17:12+00:00 |
| Z8 | 369367263367 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:25:28+00:00 |
| A_on | 369367263265 | 827d5ecf6021 | None | n/a | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T04:59:25+00:00 |
| A_off | 369367263264 | 827d5ecf6021 | None | n/a | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T04:59:23+00:00 |
| C_off | 369367263283 | 827d5ecf6021 | None | n/a | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T05:15:49+00:00 |

The GPU model is not recorded for A_on, A_off, C_off: those arms ran before the instrument read `device_kind`, and their blocks are left as they were measured rather than back-filled.

### F.6 The same arms read with an estimator that is exact at a transmission zero

Reported, no window.  W5 to W8 above are frozen on the case's estimator and are judged on it;
nothing below changes a verdict or is one.  Every notch here is read again from the same
recorded curves by the same shared `refined_extremum` (`validation/crossval/comparators/spectral_features.py`):
the deepest bin in the reference band and the vertex of the parabola through it and its two
neighbours.  Only the domain of that parabola differs between the two columns.  `log` fits it
in the log of the magnitude of S21; that is the case's estimator, the one F.3 and F.4 print.
`power` fits it in the squared magnitude of S21.  Near a simple zero moved off the real
frequency axis by loss, the squared magnitude is ((f - f0) / B)^2 + floor, a parabola in f,
so that vertex is exact there wherever the zero falls inside its bin.  Every distance below
is from the reference read with the same estimator.

**Every arm's notch, both ways.**  The last two columns are where each vertex falls from the
deepest bin, in bins.

| arm | log (GHz) | power (GHz) | power minus log (MHz) | log vertex from the deepest bin (bins) | power vertex from the deepest bin (bins) |
|---|---|---|---|---|---|
| C_off_re | 3.716635 | 3.718408 | +1.773 | +0.0535 | +0.1658 |
| F16Z6 | 3.748281 | 3.750134 | +1.852 | +0.0578 | +0.1751 |
| ON13 | 3.733383 | 3.735796 | +2.414 | +0.1142 | +0.2671 |
| Z16 | 3.710389 | 3.708800 | -1.589 | -0.3420 | -0.4427 |
| Z6 | 3.749938 | 3.752466 | +2.528 | +0.1627 | +0.3229 |
| Z8 | 3.733780 | 3.736288 | +2.508 | +0.1394 | +0.2982 |
| A_on | 3.732318 | 3.733944 | +1.626 | +0.0468 | +0.1498 |
| A_off | 3.747389 | 3.747451 | +0.062 | +0.0013 | +0.0052 |
| C_off | 3.716635 | 3.718408 | +1.773 | +0.0535 | +0.1658 |
| openEMS `stage_b_fine`, the reference | 3.674356 | 3.673868 | -0.487 | -0.2559 | -0.3673 |

**The FZ ladder, both ways.**  Addendum A.2's one-build ladder at F = 24.2915 um; only the
substrate cell moves.  Each step is from the rung above.

| arm | FZ (um) | log (GHz) | step (MHz) | power (GHz) | step (MHz) |
|---|---|---|---|---|---|
| Z6 | 42.3333 | 3.749938 | n/a | 3.752466 | n/a |
| Z8 | 31.7500 | 3.733780 | -16.158 | 3.736288 | -16.178 |
| C_off_re | 21.1667 | 3.716635 | -17.145 | 3.718408 | -17.880 |
| Z16 | 15.8750 | 3.710389 | -6.246 | 3.708800 | -9.608 |

The declared ladder, with C_off in C_off_re's place, reads the same to the last bit under
both estimators: True.  Monotone: log True, power True.  The finest rung from the reference: log
+0.9807 %, power +0.9508 %.

**Every reading of the order, both ways,** each with its limit and the substrate rule it
implies.  The first four readings take the limit the way W5 does, from the finest two rungs
the reading covers, at its order; F.4 prints no limit for the coarse-endpoint reading, and
this is the one it would have.  The substrate rule is W8's derivation applied to each reading:
FZ = (bar x limit / abs(A))^(1/p) and n_z = ceil(h / FZ), the bar 1 % of that reading's OWN limit.

| how the order is read | estimator | order | limit (GHz) | limit from the reference (%) | abs(A) | FZ for the bar (um) | n_z for the bar |
|---|---|---|---|---|---|---|---|
| declared: least squares of the log step against the midpoint of each pair's log FZ | log | 1.3713 | 3.69748 | +0.6292 | 4.9246e+13 | 34.1871 | 8 |
| declared: least squares of the log step against the midpoint of each pair's log FZ | power | 0.7518 | 3.66901 | -0.1323 | 1.6137e+11 | 14.2503 | 18 |
| the same, each step at the coarser rung of its pair | log | 1.4448 | 3.69827 | +0.6508 | 1.0408e+14 | 34.3609 | 8 |
| the same, each step at the coarser rung of its pair | power | 0.8030 | 3.67183 | -0.0555 | 2.6402e+11 | 15.7398 | 17 |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | log | 0.8258 | 3.67352 | -0.0227 | 3.1224e+11 | 17.4364 | 15 |
| the ratio of two successive steps, rungs 42.333 / 31.750 / 21.167 um | power | 0.7064 | 3.66450 | -0.2551 | 1.0806e+11 | 12.2543 | 21 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | log | 1.8888 | 3.70174 | +0.7452 | 1.0046e+16 | 34.2703 | 8 |
| the ratio of two successive steps, rungs 31.750 / 21.167 / 15.875 um | power | 0.7958 | 3.67146 | -0.0656 | 2.4643e+11 | 15.5400 | 17 |
| f_inf + A FZ^p fitted to all four notches at once | log | 1.1908 | 3.69189 | +0.4772 | 9.3933e+12 | 28.8747 | 9 |
| f_inf + A FZ^p fitted to all four notches at once | power | 0.7429 | 3.66799 | -0.1600 | 1.4989e+11 | 13.7675 | 19 |

The three-parameter fit leaves 0.517 MHz rms (log) and 0.051 MHz rms (power) on four points with
three unknowns.

**W6's three numbers, both ways** (A_off -> Z6 -> C_off_re).

| estimator | T, A_off -> C_off_re (MHz) | dZ, Z6 -> C_off_re (MHz) | dF, A_off -> Z6 (MHz) |
|---|---|---|---|
| log | +30.754 | +33.303 | -2.549 |
| power | +29.043 | +34.058 | -5.015 |

**The three in-plane points, both ways,** at FZ = 42.3333 um.  Each step is from the row above.

| arm | F (um) | log (GHz) | step (MHz) | power (GHz) | step (MHz) |
|---|---|---|---|---|---|
| A_off | 47.2441 | 3.747389 | n/a | 3.747451 | n/a |
| F16Z6 | 35.9281 | 3.748281 | +0.893 | 3.750134 | +2.682 |
| Z6 | 24.2915 | 3.749938 | +1.657 | 3.752466 | +2.333 |

**W7's row, both ways.**

| estimator | A_off minus ON13 (MHz) | F slope on the two on-node arms (MHz/um) | what that slope makes of the F gap (MHz) | A_off minus ON13 without that part (MHz) |
|---|---|---|---|---|
| log | +14.006 | -0.2767 | -0.302 | +14.308 |
| power | +11.655 | -0.4815 | -0.525 | +12.180 |

**Instrument evidence (a), a diagnostic: the vertex from wider stencils.**  Least-squares
parabolas over 3, 5 and 7 samples centred on the deepest bin, on the bin-index abscissa the shared
estimator assumes; at 3 samples this is the shared estimator's own vertex.  No window and no
reading above uses the wider stencils.

| arm | estimator | 3 samples (GHz) | 5 samples (GHz) | 7 samples (GHz) | largest minus smallest (kHz) |
|---|---|---|---|---|---|
| Z6 | log | 3.749938 | 3.750301 | 3.750370 | 432.4 |
| Z6 | power | 3.752466 | 3.752466 | 3.752466 | 0.2 |
| Z8 | log | 3.733780 | 3.734214 | 3.734307 | 526.9 |
| Z8 | power | 3.736288 | 3.736289 | 3.736290 | 2.0 |
| C_off_re | log | 3.716635 | 3.717035 | 3.717158 | 523.2 |
| C_off_re | power | 3.718408 | 3.718409 | 3.718410 | 2.0 |
| Z16 | log | 3.710389 | 3.711219 | 3.711366 | 977.3 |
| Z16 | power | 3.708800 | 3.708802 | 3.708804 | 4.0 |

**Instrument evidence (b), a diagnostic: a synthetic lossy zero at each ladder arm's own
bins.**  The squared magnitude is ((f - f0) / B)^2 + floor, B and the floor read off the arm by
the 3-sample power parabola at its notch, sampled on the arm's own frequency grid.  f0 is put at
tenths of a bin above the arm's deepest bin, and each error is the estimate minus f0.  The
power column is a parabola in f by construction.  The recorded grid is not exactly even and
the shared estimator takes it as even, so the power error is also given on an evenly spaced
grid of the same bin width, as the control.

| arm | bin width (MHz) | spacing spread over the sweep (Hz) | B (GHz) | floor (dB) | the synthetic's deepest bin across the sweep (dB) | the arm's own deepest bin (dB) |
|---|---|---|---|---|---|---|
| Z6 | 15.789312 | 1024 | 1.1888 | -54.88 | -43.25 to -54.88 | -46.65 |
| Z8 | 15.789568 | 1024 | 1.1839 | -54.77 | -43.21 to -54.77 | -47.18 |
| C_off_re | 15.789312 | 1024 | 1.1778 | -54.74 | -43.16 to -54.74 | -50.81 |
| Z16 | 15.789312 | 1024 | 1.1744 | -54.66 | -43.13 to -54.66 | -44.11 |

| f0 above the deepest bin (bins) | Z6 log error (MHz) | Z8 log error (MHz) | C_off_re log error (MHz) | Z16 log error (MHz) | largest abs power error of the four, recorded grid (Hz) | the same, evenly spaced grid (Hz) |
|---|---|---|---|---|---|---|
| 0.0 | +0.000 | -0.000 | +0.000 | +0.000 | 1.3e+02 | 0.0e+00 |
| 0.1 | -1.143 | -1.142 | -1.142 | -1.141 | 1.1e+02 | 0.0e+00 |
| 0.2 | -2.044 | -2.043 | -2.043 | -2.042 | 9.2e+01 | 0.0e+00 |
| 0.3 | -2.513 | -2.511 | -2.511 | -2.510 | 6.7e+01 | 0.0e+00 |
| 0.4 | -2.180 | -2.179 | -2.179 | -2.178 | 3.6e+01 | 0.0e+00 |
| 0.5 | +0.000 | +0.000 | +0.000 | +0.000 | 0.0e+00 | 0.0e+00 |
| 0.6 | +2.180 | +2.179 | +2.179 | +2.178 | 1.4e+02 | 0.0e+00 |
| 0.7 | +2.513 | +2.511 | +2.511 | +2.510 | 1.4e+02 | 0.0e+00 |
| 0.8 | +2.044 | +2.043 | +2.043 | +2.042 | 1.4e+02 | 0.0e+00 |
| 0.9 | +1.143 | +1.142 | +1.142 | +1.141 | 1.4e+02 | 0.0e+00 |

Across the sweep and the four arms the log error runs from -2.513 to +2.513 MHz; the largest power error is 1.4e+02 Hz on the recorded grid and
0.0e+00 Hz on the evenly spaced one.

**The first record's graded ladder, both ways.**  A_off, B_off and C_off cut both cells together.  Read
the way W3 reads it: the order from all three rungs, the limit from the finest two at that order.

| arm | F (um) | FZ (um) | log (GHz) | power (GHz) |
|---|---|---|---|---|
| A_off | 47.2441 | 42.3333 | 3.747389 | 3.747451 |
| B_off | 35.9281 | 31.7500 | 3.732214 | 3.733704 |
| C_off | 24.2915 | 21.1667 | 3.716635 | 3.718408 |

| estimator | ratio of the two steps | the ratio an order near zero gives | order | limit (GHz) | limit from the reference (%) |
|---|---|---|---|---|---|
| log | 0.9740 | 0.7095 | 0.9225 | 3.68229 | +0.2159 |
| power | 0.8987 | 0.7095 | 0.6867 | 3.67076 | -0.0846 |

### Conclusions (leader, rewritten after the review and the A.2 re-measurement; read against F.0–F.5, A.2, F.4b and the two records)

**The remaining first-order term is the substrate-normal cell at the sheet, and that is the
finding this record carries with room to spare.** With the in-plane cell held at 24.29 µm,
cutting only the substrate cell 42.3 → 31.75 → 21.2 → 15.9 µm walks the notch 3.7499 →
3.7338 → 3.7166 → 3.7104 GHz, monotone (W5 held). With the substrate cell held at 42.3 µm,
cutting the in-plane cell 47.2 → 35.9 → 24.3 µm moves it 2.5 MHz, and upward. At the step
the first note could not attribute (A_off → C_off, 30.8 MHz), the substrate leg carries
33.3 MHz and the in-plane leg −2.5 MHz (W6: FZ dominant by a factor of thirteen; the cross
term is zero by construction, since the two legs telescope, and a real cross term would
need the fourth corner, which is not an arm here). The reused rung was re-solved at the
ladder's own commit and reproduced bit for bit — |S21|, |S11| and the frequency grid as
arrays, 400 bins — so the eleven-file `rfx/` change between the two commits moved this
board by nothing, and the ladder stands on one solver build (A.2).

**What the record does NOT settle: a single order, a single limit, a single substrate
rule.** The four notches are not one power law. The order read from the two coarse steps is
0.83, from the two fine steps 1.89, from the declared rule 1.37, from a three-parameter
fit to all four 1.19 (rms 0.5 MHz); the limits those readings give span 3.6735 to
3.7017 GHz, that is −0.02 % to +0.75 % from the reference, the declared rule's +0.63 %
among them. The pre-declared verdicts survive every reading (W5, W8 held), but no sentence
below quotes one of these numbers without its spread. The apparent order RISES down the
ladder (0.83 → 1.89), which a first-order term mixed with a second-order one would not do
(that mixture's apparent order falls toward 1 as the cell shrinks); the earlier version of
these conclusions said the opposite and is withdrawn. What a rising apparent order says is
only that the two coarse rungs are not in the asymptotic regime of whatever law the fine
rungs follow; four points cannot say more.

**Refining z alone reaches the case's 1 % bar; whether it reaches the reference depends on
which of the four readings one trusts.** The finest rung (FZ = 15.9 µm, 3.94 M cells,
1082 s on one RTX 4090) reads +0.98 %. The declared rule's limit is +0.63 % and the
coarse-pair reading's is −0.02 %; the residual above the reference at F = 24.29 µm is
therefore somewhere between "gone" and "0.75 %", and this record cannot narrow it. Two
candidates for a residual, if there is one, and what this record says about each: the
edge-offset rule — the on-node arm reads 0.383 % lower than the offset arm once the 2.4 %
in-plane-cell difference is removed (W7; the correction has the sign that ENLARGES the
offset's own effect, 14.3 MHz, not the 14 − 0.3 first written here), measured at
F = 47 µm where 0.35 cell is 16.5 µm; at the ladder's F = 24.29 µm the same rule moves
8.5 µm, so the offset's own effect there is of order half that, ~0.2 %; and the graded
board's line sits 609 µm further from the y_lo absorber than the case's, unmeasured. The
reference's own convergence (its last two rungs 0.008 % apart) is not a candidate. The two
arms that would place a residual — an offset at 0.30 and 0.25 cell at fixed F and FZ, and
the case's own board with a fine y_lo runway once the CPML slots read per face (step 0b)
— are new declarations, not run here.

**What this gives the mesher, stated with the spread.** The substrate rule: the notch's
substrate term drops inside 1 % of the ladder's own limit at n_z = 8 by the declared rule
(FZ ≤ 34.2 µm; F.4), and 8 to 10 across the four readings; inside 1 % of the external
reference at n_z = 16 (the finest rung, +0.98 %). The in-plane rule: at FZ = 42.3 µm the
three in-plane points 47.2 / 35.9 / 24.3 µm span 2.5 MHz (0.07 % of the notch) and
accelerate (0.08 then 0.14 MHz/µm), so the fine cell across a 600 µm strip need not go
below about 47 µm (n = 12) for the notch at that substrate cell — measured at one FZ only,
not a convergence statement. The cheap path the two rules suggest, n = 12 with n_z = 16, is
costed by the instrument and not solved: 2,656,500 grid cells, dt 47.35 fs, 60,342 steps
(F.4b) — about two thirds of Z16's cells at 1.2× its time step. The edge rule stays 0.35
cell by default with a per-sheet override, as the design review asked; its own effect on
this structure is 0.38 % of the notch per 0.35 cell at F = 47 µm, which is what a mesher's
override would be tuning.

**The record's honesty items.** The five arms reused from the first record rebuild from it
with every declared quantity bit-identical; the last bit of four ramp cells per profile,
seven per arm, differs between the GPU container's numpy and the project venv's (≤ 6 ulp,
8e-16 of a cell, none touching the metal or the substrate; F.0 prints the table from the
record). The 13-cell on-node arm has no node on the strip's centre line, so its feed is
placed on the metal's own centre with the two straddling rows one cell apart (F.0); the
check that enforces it can no longer be switched off by dropping a key (review P3). Two
things move down the ladder besides the substrate cell — the solved z tail cell and the
time step — and both follow from it (W5 block).

**Not re-run.** The finest rung missed the reference by 0.98 %. The arms that would place
the residual are new declarations with their own purpose, not a re-run of this one.
