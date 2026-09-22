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
fixed in-plane cell needs.  The five arms already recorded are rebuilt from the first
record and compared against it (`tests/unit/nonuniform/test_msl_notch_graded_build.py`): every cell of every
fine band, every cell of every coarse run, every profile sum, the fine cell, the substrate
cell, the solved z tail cell, every declared coordinate and every segment length are
bit-identical.  What is not is the last bit of the cells inside a solved geometric ramp:
those arms were solved in the GPU job's container (numpy on python 3.10) and the gate runs
in the project venv (numpy 2.4 on python 3.11), and the two disagree in the last bit or two
of `np.sum` and of the power ufunc, which is what the ramp's cell ratio is bisected on.
Measured across the five arms: at most 6 of a profile's 8 to 27 ramp cells differ, by at
most 6 ulp, which is 8e-16 of a cell's own size, on cells in the coarse transition away
from the metal.  No cell touching the line, the stub or the substrate is among them, and
every profile sums to the same domain to the last bit.  The gate refuses beyond 8 ulp; a
rung changed by one cell moves these numbers by about 1e13 ulp.

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

**W5 -- the FZ ladder is a ladder.**  Four arms at one in-plane cell, F = 24.2915 um; only the substrate cell moves.

| arm | substrate cells | FZ (um) | notch (GHz) | step from the rung above (MHz) |
|---|---|---|---|---|
| Z6 | 6 | 42.3333 | 3.74994 | n/a |
| Z8 | 8 | 31.7500 | 3.73378 | -16.158 |
| C_off | 12 | 21.1667 | 3.71663 | -17.145 |
| Z16 | 16 | 15.8750 | 3.71039 | -6.246 |

| monotone | fitted order p_z | order window | limit (GHz) | reference (GHz) | Z16 from the reference (%) | limit from the reference (%) | verdict |
|---|---|---|---|---|---|---|---|
| True | 1.3713 | 0.5 to 2.5 | 3.69748 | 3.67436 | 0.9807 | 0.6292 | HELD |

The order is a least-squares slope of log|step| against the midpoint of the two rungs' log FZ.
Placing each step at the coarser rung of its pair instead gives 1.4448; the fit's three residuals are
-1.78e-01, +3.56e-01, -1.78e-01.

**W6 -- attribution at the A_off to C_off step.**  A_off and Z6 share the substrate cell,
so what separates them is the in-plane cell; Z6 and C_off share the in-plane cell, so what
separates them is the substrate cell.

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

**W8 -- is z enough?**  Where the notch goes as the substrate cell goes to zero at
F = 24.2915 um, against the reference.

| arm | substrate cells | FZ (um) | notch (GHz) | from the reference (%) |
|---|---|---|---|---|
| Z6 | 6 | 42.3333 | 3.74994 | 2.0570 |
| Z8 | 8 | 31.7500 | 3.73378 | 1.6173 |
| C_off | 12 | 21.1667 | 3.71663 | 1.1507 |
| Z16 | 16 | 15.8750 | 3.71039 | 0.9807 |

| ladder limit (GHz) | reference (GHz) | limit from the reference (%) | bar (%) | W5 | verdict |
|---|---|---|---|---|---|
| 3.69748 | 3.67436 | 0.6292 | 1 | HELD | HELD |

Derived, with its derivation: the fit puts the notch abs(A) FZ^p from its limit, so the
cell that brings that inside the bar is FZ = (bar / abs(A))^(1/p) and the substrate rule is
n_z = ceil(h / FZ).

| abs(A) | p | FZ for the bar (um) | n_z for the bar |
|---|---|---|---|
| 4.9246e+13 | 1.3713 | 34.0311 | 8 |

### F.5 Provenance

| arm | VESSL run | commit | dirty | GPU | jax | backend | started (UTC) |
|---|---|---|---|---|---|---|---|
| F16Z6 | 369367263366 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:22:04+00:00 |
| ON13 | 369367263364 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:17:14+00:00 |
| Z16 | 369367263368 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:28:13+00:00 |
| Z6 | 369367263365 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:17:12+00:00 |
| Z8 | 369367263367 | d558382f3f5f | None | NVIDIA GeForce RTX 4090 | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T09:25:28+00:00 |
| A_on | 369367263265 | 827d5ecf6021 | None | n/a | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T04:59:25+00:00 |
| A_off | 369367263264 | 827d5ecf6021 | None | n/a | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T04:59:23+00:00 |
| C_off | 369367263283 | 827d5ecf6021 | None | n/a | 0.4.33.dev20241023+e3c6d6430 | gpu | 2026-09-22T05:15:49+00:00 |

The GPU model is not recorded for A_on, A_off, C_off: those arms ran before the instrument read `device_kind`, and their blocks are left as they were measured rather than back-filled.

Conclusions: leader fills.

