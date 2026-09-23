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

### Conclusions (leader; rewritten 2026-09-23 after the verification review showed that the notch estimator bent the ladder; read against F.0–F.6 and the two records)

**The substrate-normal cell at the sheet carries the notch's remaining discretization error.
With the in-plane cell held at 24.29 µm, refining it alone converges the notch to 0.06–0.26 %
below openEMS.** Cutting only the substrate cell 42.3 → 31.75 → 21.2 → 15.9 µm lowers the
notch monotonically (W5 held). Read with the |S21|² vertex (F.6), the four rungs follow one
power law in the substrate cell. The five readings of the order all give 0.71 to 0.80, and a
three-parameter fit to the four notches leaves 0.051 MHz rms. The limits the five readings
give are 3.6645 to 3.6718 GHz. openEMS `stage_b_fine`, read the same way, is 3.67387 GHz, so
the limits sit 0.06 to 0.26 % below it. The step the first note could not attribute,
A_off → C_off, splits into a substrate leg of 34.1 MHz and an in-plane leg of −5.0 MHz, out
of a 29.0 MHz total. Read with the case's log vertex, the split is 33.3 / −2.5 / 30.8 MHz.
W6 classifies FZ as dominant under both estimators. The reused rung was re-solved at the
ladder's commit and reproduced bit for bit (A.2), so the ladder stands on one solver build.

**What changed from the previous version: the instrument, not the solver.** The case reads a
notch as the vertex of a parabola through three bins of log|S21|. Near a transmission zero
that loss has moved off the real axis, |S21|² is ((f − f0)/B)² + floor, a parabola in f. Its
logarithm is not a parabola, so the log vertex misses the zero, by an amount that depends on
where the zero falls inside its 15.8 MHz bin. F.6(b) puts a synthetic zero with each ladder
arm's own B (1.17–1.19 GHz), floor (−54.7 to −54.9 dB) and frequency grid at tenths of a bin. The log
vertex errs from −2.51 to +2.51 MHz. The |S21|² vertex errs by at most 140 Hz, and only
because the recorded grid's spacing alternates by 1024 Hz while the estimator assumes it
even; on an even grid the error is zero. The record shows the same thing (F.6(a)): the
|S21|² vertex moves at most 4 kHz between 3-, 5- and 7-sample fits, and the log vertex moves
0.4 to 1.0 MHz. The rung-to-rung steps are 6 to 18 MHz, so an error of up to ±2.5 MHz that
differs from rung to rung was enough to bend the ladder. The previous version of these
conclusions read the bent ladder, and four of its readings are withdrawn: "not one power
law", an order rising from 0.83 to 1.89, a residual "between gone and 0.75 %", and in-plane
steps that accelerate. The frozen windows are judged on the estimator §4 declared, as frozen
windows must be. Under both estimators the verdicts come out the same: W5, W6 and W8 held,
and on W7 the on-node arm reads lower.

**The first note's ladder was read with the same bias.** In that ladder A_off → B_off → C_off
cut both cells together. Read with the |S21|² vertex, its order is 0.69 and its limit 3.67076
GHz, 0.08 % below the reference. The log reading merged with #1191 gave 0.92 and 0.22 % above
(F.6, last table). "First order" was the estimator's number. Other notch or resonance
readings that use the log parabola on bins this coarse relative to the feature's depth carry
the same kind of error.

**What this gives the mesher: the substrate rule is stricter than the log reading said.** An
order below one means that halving the cell removes less than half of the error. W8's
derivation applied to each |S21|² reading puts the substrate term inside 1 % of that
reading's own limit at n_z = 18 by the declared rule, and 17 to 21 across the five readings
(F.6). That is FZ ≈ 12–16 µm on this 254 µm substrate. These n_z values are derived, not
solved. The n_z = 8 of the log reading is withdrawn. The finest solved rung (n_z = 16, 3.94 M
cells, 1082 s on one RTX 4090) is already within 1 % of the external reference, at +0.95 %,
because the ladder converges to slightly below it. In the plane, at FZ = 42.3 µm, cutting F
from 47.2 to 35.9 to 24.3 µm raises the notch by +2.7 then +2.3 MHz: 5.0 MHz in all, 0.13 %
of 3.75 GHz, with the steps shrinking. That was measured at one substrate cell only. There is
no in-plane ladder at a fine substrate cell, so this record gives no in-plane rule. The
pairing the two ladders suggest, n = 12 across the strip with n_z = 16, has been costed by
the instrument but not solved: 2,656,500 cells, dt 47.35 fs, 60,342 steps (F.4b).

**Where the limit sits, and what this record cannot place.** The limit is the substrate
cell → 0 limit at F = 24.29 µm with the 0.35-cell edge rule, not the continuum answer. Two
in-plane terms remain in it, and this record has measured each only at F ≈ 47 µm and
FZ = 42.3 µm. First, the in-plane cell: refining it from 47.2 to 24.3 µm raised the notch
there by 5.0 MHz. Its size and sign at a fine substrate cell are unmeasured. Second, the edge
offset: at nearly the same in-plane cell (47.2 vs 46.2 µm) the on-node arm reads
11.7 MHz below the offset arm. Taking out the 2.4 % in-plane-cell difference with the
on-node arms' own slope makes it 12.2 MHz, 0.33 % of 3.74 GHz (F.6, W7 row; the log vertex
gave 14.3 MHz). That is the 0.35-cell rule's own effect where 0.35 cell is 16.5 µm. At the
ladder's F = 24.29 µm the rule moves the edge 8.5 µm, and no arm measures its effect there.
The line on the graded board also sits 609 µm further from the y_lo absorber than on the
case's own board, which is unmeasured. Any of these could account for the 0.06–0.26 %.
This record does not say which.

**The record's honesty items.** The five arms reused from the first record rebuild from it
with every declared quantity bit-identical. In the ramp cells, at most four per profile and
at most seven per arm differ in the last bit between the GPU container's numpy and the
project venv's. The difference is at most 6 ulp, 8e-16 of a cell, and none of those cells
touches the metal or the substrate (F.0). The 13-cell on-node arm has no node on the strip's
centre line, so its feed sits on the metal's own centre, between two rows one cell apart
(F.0). Besides the substrate cell, two other quantities change down the ladder: the solved z
tail cell and the time step. Both follow from the substrate cell (W5 block). In a checkout
where git cannot see the solving commits, the solver-tree tests skip and name the missing
commit. The record's own evidence, C_off and C_off_re bit-identical in all four S curves and
all three profiles, stays asserted unconditionally.

**Not re-run.** The windows stand as declared. Placing the remaining 0.06–0.26 % needs new
declarations with their own purpose, none of them re-runs of this one:
- an in-plane ladder at FZ = 15.9 µm;
- an edge offset of 0.30 and of 0.25 cell at fixed F and FZ;
- the case's own board with a fine y_lo runway, once the CPML slots read per face (step 0b).
