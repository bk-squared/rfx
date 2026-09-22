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
