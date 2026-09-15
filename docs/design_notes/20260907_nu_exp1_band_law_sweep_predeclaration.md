# E1 — does the narrow-band transition law generalize? Resolution x ratio sweep of the W6 witness — pre-declaration

**Status:** pre-declaration. This commit precedes every FDTD measurement.
Every numeric window below is frozen from this commit onward; results are
appended under "Results", never edited into the windows. One attempt per
arm; a fired window is a result, not a bug to tune away.
**Class:** validity-domain measurement of a law already witnessed at one
fixture (family: W2 `w2_w3_reflection` F-S2, W6 `w6_band_builder` F8,
note `20260907_nu_band_profile_predeclaration.md` section 3 and its
reviewer/re-execution rows; open decision 4 of that lane).
**Tree:** worktree `rfx-nu-exp1`, branch `exp/nu-experiments-e1`, based on
`feat/nu-band-accuracy-ad @ fea9f078` (carries `feat/nu-band-profile` and
`origin/main d990e18c`). Instrument commit (this tree, before any run):
the commit that carries this note; the instrument is
`validation/research/multiband_nu/w6_band_builder.py` with the `--sweep`
mode added (the lane-A F7/F8 code paths are untouched; the lane-A replay
test `tests/unit/nonuniform/test_band_builder_chain_model.py` passes on the
changed file, 15 passed).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-exp1/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `argv` and `started_utc`. The chain-model predictions in
this note are the file `results/e1_band_law_sweep_model.json`, written
by `--model-only` on this tree at HEAD `fea9f078` with the not-yet-committed instrument in the worktree (`git_dirty` true in that JSON; the instrument, this note, the test and that JSON are one commit) —
the tables below are generated from that JSON, not retyped.
Date: 2026-09-07 (KST). CPU only (a GPU lane runs elsewhere on the shared
machine).

## 0. Scope — what lane A established and what this lane measures

Lane A (W6 F8, `results/w6_band_builder.json`, re-executed on `5bf9d16b`)
witnessed, at ONE resolution pair (fine 1.0 mm = 29.98 cells per free-space
wavelength at 10 GHz, coarse 1.96 mm = 15.30) and ONE ratio (1.4), for
fine bands of 2 / 4 / 8 / 16 (+32, 64) cells between two cap ramps:

- the reflection of a band of n_b fine cells follows the Fabry-Perot form
  of the two opposite-sign ramp reflections,
  `R(n_b) = 2 R_single |sin(k_g (L + c))|`, L = n_b x fine cell,
  `R_single = 5.7907e-3` (one 1.96 -> 1.4 -> 1.0 mm ramp, chain model),
  discrete `k_g(1.0 mm) = 0.18163 /mm`, `c = 1.87 mm`;
- FDTD within 0.2-5.7 % of the chain model on the six rows, every row
  inside the window `|R_meas - R_model| <= 0.20 R_model + 3e-5`;
- bounded by `2 R_single` (chain maximum `2 R_single x 0.9999`).

Lane A's own closing sentence: "All of this is one resolution ... quote
the law, not the dB." Open decision 4 of that lane asks for a second
resolution pair / ratio sweep before the row may be quoted as a law. This
lane is that sweep. PI rule applied: measure the validity domain (the law
AND its range); no claim is locked to one fixture.

Measured here: 3 fine resolutions x 3 ratios x 5 band widths = 45 band
arms, plus 9 single-ramp arms (one per (resolution, ratio) cell, so that
law check (i) is a measurement and not a fit), plus 9 B reference runs.
Every arm is seconds on CPU.

## 1. Findings made while writing this note (zero FDTD)

### 1a. The resolution knob: fine cell scaled, frequency and transverse fixture fixed

Two ways to change "cells per wavelength" were open: scale F0 or scale the
fine cell. F0 is not available: the fixture's TE10 cutoff is 5.0 GHz
(b = 30 mm), so 5 GHz is evanescent and 20 GHz puts the 1.5 mm transverse
cell at 10 cells per wavelength and moves the guide dispersion. The fine
cell is scaled instead: `dz_fine = 1.0 mm x 30 / N` for N in {15, 30, 60}
= 2.0 / 1.0 / 0.5 mm (14.99 / 29.98 / 59.96 cells per free-space
wavelength at 10 GHz; the lane-A value was 29.98, not 30.00, and is kept
exactly). F0 = 10 GHz, sigma_t = 64 ps, dxy = 1.5 mm, b = 30 mm,
a = 4.5 mm, PEC closed, `cpml_layers = 0` are the lane-A values unchanged,
so `k_g` of the fine band is 0.18224 / 0.18163 / 0.18158 /mm (lambda_g
34.48 / 34.59 / 34.60 mm) — the guide wavelength barely moves, the cell
count per guide wavelength is what changes.

### 1b. The ratio knob: coarse = r^2 x fine, one ramp cell r x fine, builder cap = r

Lane A's transition is a cap ramp: coarse 1.4^2 = 1.96 mm -> ramp 1.4 mm ->
fine 1.0 mm, both steps exactly at the cap, one ramp cell, emitted by
`make_band_profile(..., max_ratio = 1.4)`. The generalization keeps that
structure: coarse = r^2 x fine, ramp = r x fine, `max_ratio = r`, r in
{1.2, 1.4, 2.0}. So a "ratio 2.0" cell has a coarse cell 4 x the fine one
(two steps of 2.0), not a single 2.0 step. **Builder exactness, checked
before any run:** `make_band_profile` emits the declared vector
`[coarse] x n_lead | ramp | [fine] x n_b | ramp | [coarse] x n_tail` to
1e-12 m per cell on all 45 band profiles and the declared
`[coarse] x n_lead | ramp | [fine] x n_tail_fine` on all 9 single-ramp
profiles (`builder_matches_declared_vector` true, 54 of 54, Table R). If
a run-time rebuild ever differs, the chain model is re-run on the builder's
actual output and that is the model (the lane-A rule).

### 1c. The runways are the lane-A PHYSICAL layout, re-cut in the setting's cells

The lane-A gate geometry is a set of physical lengths that happen to be
multiples of 1.96 mm: source 166.6 mm, probe 196.0 mm, transition
274.4 mm, tail 294.0 mm, B reference 784.0 mm, run 2.883 ns. Each cell
count of a setting is that length divided by the setting's coarse cell,
rounded half-up (Table S). The 30 / 1.4 cell therefore reproduces lane A
bit for bit — K_SRC 85, K_PRB 100, 140 / 150 / 400 cells, dt
2.402765e-12 s, 1200 steps — and its five rows are lane A's frozen
predictions digit for digit (7.4916e-3, 1.0141e-2, 1.1296e-2, 1.2101e-3,
1.5111e-3). That cell is the sweep's control: it must return lane A's
measured values (7.4364e-3, 1.0063e-2, 1.1164e-2, 1.2559e-3, 1.4248e-3)
to float32 reproducibility (declared: <= 1e-6 relative; the lane-A
re-execution reproduced its first attempt to 3e-10).

Run length: `n_steps = ceil(max(2.883 ns, t_s + 0.1 ns) / dt)`; the
second term matters only for the 15 / 2.0 cell (vg 0.60 c, t_s 3.169 ns
-> 1057 steps). dt is set by the grid from the fine cell and dxy
(3.094 / 2.403 / 1.494 ps); the B reference `[coarse] x n_B + [fine] x 4`
pins dt to the A run's (checked per arm, `dt_matches_b`).

### 1d. Gates hold geometrically at every setting (Table G)

Five margins per arm, all from the chain-model group velocities of the
setting (nothing asserted at run time — a violated gate would be
reported as such): reflection inside the gate
(`gate_end - (t_r + 4 sigma)`), last inner return inside it (band: to
the far ramp and back; single ramp: the ramp cell), run covers the gate,
B-run pin/far-wall return after the gate, incident pulse inside the
incident gate. Every one of the 54 arms holds; the smallest margin is
0.256 ns = 4 sigma_t on the incident gate (its construction
`t_inc_end = arrival + 8 sigma`), the next 0.278 ns (inner return, N15
r1.2 n_b = 32), everything else >= 0.36 ns.

### 1e. Where the coarse lattice stops propagating (a gating risk, quantified)

The pulse is 10 GHz +/- 2.5 GHz (1 sigma in frequency). The coarse
lattice's TE10 Bloch wave stops propagating (`bloch_kz` argument = 1) at
34.1 / 25.1 / **12.97** GHz for the three 15-cell settings, 50.1 / 24.5 GHz
for 30 / 1.4 and 30 / 2.0, 48.4 GHz for 60 / 2.0, above 60 GHz elsewhere
(Table S). Only the **15 / 2.0** cell (coarse cell 8.0 mm = 3.75 cells per
free-space wavelength, vg 0.599 c at 10 GHz) has the stopband edge inside
the pulse's 1.2-sigma band. Components near that edge are slow and arrive
late; the rectangular gate then leaks them into the 10 GHz bin. Expected
before the run: this cell is outside the -54 dB accuracy class by both
knobs and may also be outside the instrument's own gating validity; the
per-arm windows there may fire for the instrument's reason, and that is
recorded as fired.

### 1f. The absolute floor of the instrument vs the smallest rows

Lane A's absolute deviations were 4.6e-5 to 1.3e-4 (0.2-1.2 % relative on
the peak rows, 3.8-5.7 % on the null rows). Three rows of this sweep have
`R_model` under 2.5e-4: 30 / 1.2 n_b = 16 (1.0041e-4, window
[5.03e-5, 1.50e-4]), 60 / 1.2 n_b = 32 (1.1832e-4, [6.47e-5, 1.72e-4]),
60 / 1.4 n_b = 32 (1.9694e-4, [1.28e-4, 2.66e-4]). Their half-windows
(5.0e-5, 5.4e-5, 6.9e-5) are at the size of lane A's absolute deviations.
Declared reading rule: a fired row with `R_model < 2.5e-4` and
`|R_meas - R_model| <= 1.5e-4` is recorded as **fired — at the
instrument's absolute floor** (the floor being a measured number, not a
window change); it counts as fired for the cell's law-domain verdict and
the floor is reported as the smallest `R` the method resolves.

### 1g. c derived from the chain model before the run — it scales with the ramp cell

Lane A fitted `c = 1.87 mm` at ramp cell DR = 1.4 mm. Derivation: a cap
ramp is two cell-size steps, S1 (coarse -> ramp) at the ramp's outer node
and S2 (ramp -> fine) at the band edge, both of ratio r. In the weak,
long-wavelength limit a step's reflection scales with the square of the
local cell (the (dz/lambda)^2 law) at fixed ratio, so
`|S1| / |S2| = ((r^2 + r) / (r + 1))^2 = r^2`. The ramp's effective
reflection plane is the amplitude-weighted centroid of the two step
planes, `w DR` outside the band edge with `w = r^2 / (1 + r^2)`, and the
Fabry-Perot round trip picks up that offset on both ramps:

    c = 2 w DR = 2 DR r^2 / (1 + r^2),   DR = r x dz_fine.

Predictions: `c / DR` = 1.180 (r = 1.2), 1.324 (1.4), 1.600 (2.0),
i.e. c **scales linearly with the ramp cell (with resolution at fixed
ratio) and grows with the ratio**. The exact chain model fitted over
n_b = 0..80 (Table L, `c_model`) gives c / DR = 1.182-1.205 / 1.328-1.383
/ 1.611-1.791 — the closed form is within 0.1-2 % at 30 and 60 cells for
r <= 1.4, within 4 % at 15 cells for r <= 1.4, and 3-12 % low at r = 2.0
(k d is not small there). The frozen prediction per cell is `c_model`
(the chain fit); the closed form is the explanation and the scaling
statement. At 30 / 1.4 the chain fit is 1.873 mm (lane A's 1.87).

The FP form with `c_model` reproduces the chain model to rms
1.6e-10 .. 1.2e-5 over n_b = 0..80 for r <= 1.4 and 30 / 2.0, 4.6e-7 at
60 / 2.0, 3.7e-3 at 15 / 2.0 (R_single 0.155 there: the weak-reflection
form `2 R sin` is off by the `1 - R^2 e^{2 i phi}` denominator, chain
maximum 0.977 x 2 R_single).

### 1h. The (dz/lambda)^2 law on the model itself

Chain-model `R_single(N) / R_single(30)` (Table I): 3.863 / 0.2524
(r = 1.2), 3.889 / 0.2521 (1.4), **4.852** / 0.2441 (2.0) against the pure
(dz/lambda)^2 ratios 4 / 0.25. Exponent 1.95-1.99 for r <= 1.4 and for
60 / 2.0; 2.28 for 15 / 2.0. The support-matrix sentence ("the N^-2 law
is asymptotic and under-predicts the reflection below ~30 cells per
wavelength") is reproduced on the ramp: at 15 cells the model sits 3 %
below 4x for r <= 1.4 and 21 % above it for r = 2.0. Law check (i)
compares the MEASURED ratio to the model's own ratio, so it can hold at
15 / 2.0 even though the model there is not (dz/lambda)^2.

## 2. Fixture (all lengths from `fixtures.py` and lane A; per-setting numbers in Table S)

### Table S — settings (cells, dt, run length, class)

| cell | fine (mm) | ramp (mm) | coarse (mm) | fine / coarse cells per lambda0 | K_SRC / K_PRB | lead / tail / B / single tail (cells) | dt (ps) | n_steps (run ns) | vg_c / c | coarse stopband (GHz) | -54 dB class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N15_r1.2 | 2.000 | 2.400 | 2.880 | 14.99 / 10.41 | 58 / 68 | 95 / 102 / 272 / 147 | 3.094381 | 932 (2.884) | 0.840 | 34.14 | outside |
| N15_r1.4 | 2.000 | 2.800 | 3.920 | 14.99 / 7.65 | 43 / 50 | 70 / 75 / 200 / 147 | 3.094381 | 932 (2.884) | 0.813 | 25.1 | outside |
| N15_r2 | 2.000 | 4.000 | 8.000 | 14.99 / 3.75 | 21 / 25 | 34 / 37 / 98 / 147 | 3.094381 | 1057 (3.271) | 0.599 | 12.97 | outside |
| N30_r1.2 | 1.000 | 1.200 | 1.440 | 29.98 / 20.82 | 116 / 136 | 191 / 204 / 544 / 294 | 2.402765 | 1200 (2.883) | 0.861 | None | inside |
| N30_r1.4 | 1.000 | 1.400 | 1.960 | 29.98 / 15.30 | 85 / 100 | 140 / 150 / 400 / 294 | 2.402765 | 1200 (2.883) | 0.855 | 50.14 | inside |
| N30_r2 | 1.000 | 2.000 | 4.000 | 29.98 / 7.49 | 42 / 49 | 69 / 74 / 196 / 294 | 2.402765 | 1200 (2.883) | 0.810 | 24.52 | outside |
| N60_r1.2 | 0.500 | 0.600 | 0.720 | 59.96 / 41.64 | 231 / 272 | 381 / 408 / 1089 / 588 | 1.493514 | 1931 (2.884) | 0.866 | None | inside |
| N60_r1.4 | 0.500 | 0.700 | 0.980 | 59.96 / 30.59 | 170 / 200 | 280 / 300 / 800 / 588 | 1.493514 | 1931 (2.884) | 0.864 | None | inside |
| N60_r2 | 0.500 | 1.000 | 2.000 | 59.96 / 14.99 | 83 / 98 | 137 / 147 / 392 / 588 | 1.493514 | 1931 (2.884) | 0.853 | 48.39 | outside |
### Table G — gate geometry per cell (ns; single ramp, n_b = 2 and n_b = 32 shown; all 54 arms hold, per-arm values are in the JSON)

| cell | arm | t_r | t_inner_last | t_s | t_f | gate_end (steps) | t_inc_end | B pin return | run end | margins: reflection / inner / run / B pin / incident | hold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N15_r1.2 | single | 1.052 | 1.071 | 2.379 | 3.363 | 2.123 (685) | 0.946 | 6.427 | 2.884 | 0.815 / 0.796 / 0.761 / 4.304 / 0.256 | True |
| N15_r1.2 | n_b=2 | 1.052 | 1.121 | 2.379 | 3.454 | 2.123 (685) | 0.946 | 6.427 | 2.884 | 0.815 / 0.746 / 0.761 / 4.304 / 0.256 | True |
| N15_r1.2 | n_b=32 | 1.052 | 1.589 | 2.379 | 3.922 | 2.123 (685) | 0.946 | 6.427 | 2.884 | 0.815 / 0.278 / 0.761 / 4.304 / 0.256 | True |
| N15_r1.4 | single | 1.076 | 1.098 | 2.458 | 3.390 | 2.202 (711) | 0.945 | 6.637 | 2.884 | 0.870 / 0.848 / 0.682 / 4.435 / 0.256 | True |
| N15_r1.4 | n_b=2 | 1.076 | 1.151 | 2.458 | 3.562 | 2.202 (711) | 0.945 | 6.637 | 2.884 | 0.870 / 0.795 / 0.682 / 4.435 / 0.256 | True |
| N15_r1.4 | n_b=32 | 1.076 | 1.619 | 2.458 | 4.030 | 2.202 (711) | 0.945 | 6.637 | 2.884 | 0.870 / 0.327 / 0.682 / 4.435 / 0.256 | True |
| N15_r2 | single | 1.299 | 1.332 | 3.169 | 3.624 | 2.913 (941) | 1.010 | 8.868 | 3.271 | 1.358 / 1.325 / 0.358 / 5.954 / 0.256 | True |
| N15_r2 | n_b=2 | 1.299 | 1.396 | 3.169 | 4.691 | 2.913 (941) | 1.010 | 8.868 | 3.271 | 1.358 / 1.261 / 0.358 / 5.954 / 0.256 | True |
| N15_r2 | n_b=32 | 1.299 | 1.864 | 3.169 | 5.159 | 2.913 (941) | 1.010 | 8.868 | 3.271 | 1.358 / 0.793 / 0.358 / 5.954 / 0.256 | True |
| N30_r1.2 | single | 1.045 | 1.054 | 2.339 | 3.321 | 2.083 (866) | 0.944 | 6.276 | 2.883 | 0.782 / 0.773 / 0.800 / 4.193 / 0.256 | True |
| N30_r1.2 | n_b=2 | 1.045 | 1.079 | 2.339 | 3.354 | 2.083 (866) | 0.944 | 6.276 | 2.883 | 0.782 / 0.748 / 0.800 / 4.193 / 0.256 | True |
| N30_r1.2 | n_b=32 | 1.045 | 1.310 | 2.339 | 3.586 | 2.083 (866) | 0.944 | 6.276 | 2.883 | 0.782 / 0.517 / 0.800 / 4.193 / 0.256 | True |
| N30_r1.4 | single | 1.047 | 1.057 | 2.347 | 3.324 | 2.091 (870) | 0.947 | 6.323 | 2.883 | 0.788 / 0.777 / 0.793 / 4.233 / 0.256 | True |
| N30_r1.4 | n_b=2 | 1.047 | 1.084 | 2.347 | 3.378 | 2.091 (870) | 0.947 | 6.323 | 2.883 | 0.788 / 0.751 / 0.793 / 4.233 / 0.256 | True |
| N30_r1.4 | n_b=32 | 1.047 | 1.315 | 2.347 | 3.609 | 2.091 (870) | 0.947 | 6.323 | 2.883 | 0.788 / 0.520 / 0.793 / 4.233 / 0.256 | True |
| N30_r2 | single | 1.095 | 1.110 | 2.479 | 3.377 | 2.223 (925) | 0.947 | 6.665 | 2.883 | 0.872 / 0.857 / 0.660 / 4.442 / 0.256 | True |
| N30_r2 | n_b=2 | 1.095 | 1.141 | 2.479 | 3.581 | 2.223 (925) | 0.947 | 6.665 | 2.883 | 0.872 / 0.826 / 0.660 / 4.442 / 0.256 | True |
| N30_r2 | n_b=32 | 1.095 | 1.373 | 2.479 | 3.812 | 2.223 (925) | 0.947 | 6.665 | 2.883 | 0.872 / 0.594 / 0.660 / 4.442 / 0.256 | True |
| N60_r1.2 | single | 1.039 | 1.043 | 2.321 | 3.307 | 2.065 (1382) | 0.946 | 6.250 | 2.884 | 0.770 / 0.765 / 0.819 / 4.185 / 0.256 | True |
| N60_r1.2 | n_b=2 | 1.039 | 1.056 | 2.321 | 3.320 | 2.065 (1382) | 0.946 | 6.250 | 2.884 | 0.770 / 0.753 / 0.819 / 4.185 / 0.256 | True |
| N60_r1.2 | n_b=32 | 1.039 | 1.171 | 2.321 | 3.435 | 2.065 (1382) | 0.946 | 6.250 | 2.884 | 0.770 / 0.638 / 0.819 / 4.185 / 0.256 | True |
| N60_r1.4 | single | 1.039 | 1.044 | 2.325 | 3.308 | 2.069 (1385) | 0.946 | 6.261 | 2.884 | 0.774 / 0.769 / 0.815 / 4.191 / 0.256 | True |
| N60_r1.4 | n_b=2 | 1.039 | 1.057 | 2.325 | 3.328 | 2.069 (1385) | 0.946 | 6.261 | 2.884 | 0.774 / 0.756 / 0.815 / 4.191 / 0.256 | True |
| N60_r1.4 | n_b=32 | 1.039 | 1.173 | 2.325 | 3.443 | 2.069 (1385) | 0.946 | 6.261 | 2.884 | 0.774 / 0.641 / 0.815 / 4.191 / 0.256 | True |
| N60_r2 | single | 1.047 | 1.055 | 2.346 | 3.319 | 2.090 (1399) | 0.949 | 6.335 | 2.884 | 0.786 / 0.779 / 0.794 / 4.245 / 0.256 | True |
| N60_r2 | n_b=2 | 1.047 | 1.071 | 2.346 | 3.370 | 2.090 (1399) | 0.949 | 6.335 | 2.884 | 0.786 / 0.763 / 0.794 / 4.245 / 0.256 | True |
| N60_r2 | n_b=32 | 1.047 | 1.186 | 2.346 | 3.485 | 2.090 (1399) | 0.949 | 6.335 | 2.884 | 0.786 / 0.648 / 0.794 / 4.245 / 0.256 | True |

## 3. Falsifiers and windows (frozen)

Per arm (45 band arms + 9 single-ramp arms), the lane-A window unchanged:

    |R_meas - R_model| <= 0.20 R_model + 3e-5        (amplitude)

`R_meas = |DFT_F0(A - B, [0, gate_end])| / |DFT_F0(B, [0, t_inc_end])|`,
`R_model` the exact discrete chain model on the builder's vector at the
run's dt. Per-arm reporting: `R_meas`, dB, deviation absolute and
relative, the five gate margins, `gates_hold`, `dt_matches_b`,
`builder_matches_declared_vector`.

Law checks:

- **(i) resolution scaling of the single ramp.** For each ratio and
  N in {15, 60}: `(R_single_meas(N) / R_single_meas(30)) /
  (R_single_model(N) / R_single_model(30))` in **[0.75, 1.25]**
  (Table I carries the model ratios). Reported alongside: the model
  exponent vs 2 and the pure (dz/lambda)^2 ratio.
- **(ii) the bound.** Every band arm: `R_meas <= 2 R_single_model x 1.05`
  (Table L, last column; `R_single_model` is the chain model's single
  ramp of that cell).
- **(iii) null / peak positions.** Per cell, `c_meas` = least-squares c
  in `2 R_single_model |sin(k_g (n_b dz_fine + c))|` over the five
  measured rows (R_single and k_g fixed to the model; search
  [0, 3 DR], scan + two refinements, resolution 3 DR / 4000 / 4000^2);
  window **`|c_meas - c_model| <= 0.10 DR`** (Table L, per cell:
  0.060-0.400 mm). Calibration for the 0.10 DR: the same fit on lane
  A's five measured rows gives 1.8866 mm against the chain's 1.8731 mm,
  0.0135 mm = 0.0096 DR; the window is ten times that residue.
- **Control.** The 30 / 1.4 cell's five `R_meas` equal lane A's to
  1e-6 relative.

Cell verdicts, defined before the run:

- **inside the law domain**: every one of its five width windows held,
  its single-ramp window held, (ii) held at every width, (iii) held;
- **inside the -54 dB accuracy class**: adjacent-cell ratio <= 1.4 and
  fine band >= 30 cells per free-space wavelength — the support-matrix
  envelope whose per-step reflection is -54 dB at 30 cells (F-S2). By
  construction 30 / 1.2, 30 / 1.4, 60 / 1.2, 60 / 1.4 are inside; every
  15-cell cell is outside by resolution, every 2.0 cell by the cap.
  The deliverable is the 3 x 3 table of (law domain, accuracy class)
  per cell: a cell may follow the law and still be outside the class.

Expectations written before the run (so a surprise is recognizable):
r <= 1.4 at 30 and 60 cells inside the law domain, with the three
sub-2.5e-4 rows of 1f the likeliest fires (instrument floor); 15-cell
r <= 1.4 inside the law domain (R_single 8.4e-3 / 2.3e-2, well above the
floor, gates hold); 30 / 2.0 and 60 / 2.0 following the law (chain
maximum 0.999 x 2 R_single; R_single 3.2e-2 / 7.8e-3) while outside the
class; 15 / 2.0 the one cell that may fail the law for the instrument's
reason (1e), R_single 0.155.

## 4. Run commands (declared) and output

Three calls (one per resolution, each well under the 600 s Bash cap;
`--resume` continues the same JSON), from the worktree:

    cd /Users/byungkwankim/Documents/rfx-nu-exp1 && PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp1 \
      /Users/byungkwankim/Documents/rfx/.venv/bin/python -m validation.research.multiband_nu.w6_band_builder \
      --sweep --fine-cells-per-lambda 30 --ratio 1.2,1.4,2.0 --widths 2,4,8,16,32 \
      --out validation/research/multiband_nu/results/e1_band_law_sweep.json
    ... --fine-cells-per-lambda 15 ... --resume
    ... --fine-cells-per-lambda 60 ... --resume

Output: `results/e1_band_law_sweep.json` (per cell: setting, model,
single arm, five band arms, `c_meas`, verdict flags; top level: law (i)
records, windows, provenance). Replay test (chain model only, no FDTD):
`tests/unit/nonuniform/test_e1_band_law_replay.py`, tolerances declared
in its docstring before its first run — it replays every `R_model`,
`R_single_model`, `c_model`, the window arithmetic, the gate geometry,
the law checks (i)-(iii) and the recorded verdicts from the JSON.

One attempt per arm. No re-rolls. A fired window is reported as fired.

## 5. Chain-model predictions per arm (frozen; from `results/e1_band_law_sweep_model.json`)

### Table L — law side per cell (chain model, frozen)

| cell | R_single model | dB | k_g fine (/mm) | lambda_g (mm) | c closed form 2 DR r^2/(1+r^2) (mm) | c_model chain fit n_b 0..80 (mm) | c_model / DR | fit rms (0..80) | chain max / 2 R_single | (iii) window on c (mm) | (ii) bound 2 R_single x 1.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N15_r1.2 | 8.4395e-03 | -41.5 | 0.18224 | 34.478 | 2.833 | 2.892 | 1.205 | 6.05e-07 | 0.9998 | +/- 0.240 | 1.7723e-02 |
| N15_r1.4 | 2.2518e-02 | -32.9 | 0.18224 | 34.478 | 3.708 | 3.873 | 1.383 | 1.16e-05 | 0.9995 | +/- 0.280 | 4.7287e-02 |
| N15_r2 | 1.5464e-01 | -16.2 | 0.18224 | 34.478 | 6.400 | 7.166 | 1.791 | 3.74e-03 | 0.9766 | +/- 0.400 | 3.2473e-01 |
| N30_r1.2 | 2.1846e-03 | -53.2 | 0.18163 | 34.593 | 1.416 | 1.423 | 1.186 | 1.07e-08 | 0.9998 | +/- 0.120 | 4.5876e-03 |
| N30_r1.4 | 5.7907e-03 | -44.7 | 0.18163 | 34.593 | 1.854 | 1.873 | 1.338 | 2.00e-07 | 0.9999 | +/- 0.140 | 1.2160e-02 |
| N30_r2 | 3.1873e-02 | -29.9 | 0.18163 | 34.593 | 3.200 | 3.289 | 1.644 | 3.33e-05 | 0.9989 | +/- 0.200 | 6.6934e-02 |
| N60_r1.2 | 5.5131e-04 | -65.2 | 0.18158 | 34.602 | 0.708 | 0.709 | 1.182 | 1.59e-10 | 0.9999 | +/- 0.060 | 1.1577e-03 |
| N60_r1.4 | 1.4597e-03 | -56.7 | 0.18158 | 34.602 | 0.927 | 0.929 | 1.328 | 2.96e-09 | 1.0000 | +/- 0.070 | 3.0654e-03 |
| N60_r2 | 7.7788e-03 | -42.2 | 0.18158 | 34.602 | 1.600 | 1.611 | 1.611 | 4.57e-07 | 0.9999 | +/- 0.100 | 1.6336e-02 |

### Table R — per-arm chain-model predictions and frozen windows (45 band arms)

| cell | n_b | band (mm) | nz | builder exact | R_model | dB | window on R_meas (frozen) | FP form 2 R_single sin(k_g (L + c_model)) |
|---|---|---|---|---|---|---|---|---|
| N15_r1.2 | 2 | 4.0 | 201 | True | 1.6048e-02 | -35.9 | [1.2809e-02, 1.9288e-02] | 1.6049e-02 |
| N15_r1.2 | 4 | 8.0 | 203 | True | 1.5452e-02 | -36.2 | [1.2332e-02, 1.8572e-02] | 1.5453e-02 |
| N15_r1.2 | 8 | 16.0 | 207 | True | 5.0067e-03 | -46.0 | [3.9754e-03, 6.0380e-03] | 5.0064e-03 |
| N15_r1.2 | 16 | 32.0 | 215 | True | 1.2700e-03 | -57.9 | [9.8604e-04, 1.5541e-03] | 1.2700e-03 |
| N15_r1.2 | 32 | 64.0 | 231 | True | 6.2037e-03 | -44.1 | [4.9330e-03, 7.4744e-03] | 6.2034e-03 |
| N15_r1.4 | 2 | 4.0 | 149 | True | 4.4597e-02 | -27.0 | [3.5648e-02, 5.3547e-02] | 4.4619e-02 |
| N15_r1.4 | 4 | 8.0 | 151 | True | 3.7341e-02 | -28.6 | [2.9843e-02, 4.4839e-02] | 3.7348e-02 |
| N15_r1.4 | 8 | 16.0 | 155 | True | 2.0803e-02 | -33.6 | [1.6612e-02, 2.4993e-02] | 2.0797e-02 |
| N15_r1.4 | 16 | 32.0 | 163 | True | 1.1330e-02 | -38.9 | [9.0336e-03, 1.3625e-02] | 1.1324e-02 |
| N15_r1.4 | 32 | 64.0 | 179 | True | 8.8391e-03 | -41.1 | [7.0413e-03, 1.0637e-02] | 8.8351e-03 |
| N15_r2 | 2 | 4.0 | 75 | True | 2.7263e-01 | -11.3 | [2.1807e-01, 3.2718e-01] | 2.7657e-01 |
| N15_r2 | 4 | 8.0 | 77 | True | 1.1613e-01 | -18.7 | [9.2871e-02, 1.3938e-01] | 1.1409e-01 |
| N15_r2 | 8 | 16.0 | 81 | True | 2.6913e-01 | -11.4 | [2.1527e-01, 3.2298e-01] | 2.7277e-01 |
| N15_r2 | 16 | 32.0 | 89 | True | 2.3236e-01 | -12.7 | [1.8586e-01, 2.7887e-01] | 2.3321e-01 |
| N15_r2 | 32 | 64.0 | 105 | True | 1.2316e-01 | -18.2 | [9.8501e-02, 1.4783e-01] | 1.2117e-01 |
| N30_r1.2 | 2 | 2.0 | 399 | True | 2.5449e-03 | -51.9 | [2.0059e-03, 3.0839e-03] | 2.5449e-03 |
| N30_r1.2 | 4 | 4.0 | 401 | True | 3.6408e-03 | -48.8 | [2.8826e-03, 4.3989e-03] | 3.6408e-03 |
| N30_r1.2 | 8 | 8.0 | 405 | True | 4.3260e-03 | -47.3 | [3.4308e-03, 5.2212e-03] | 4.3260e-03 |
| N30_r1.2 | 16 | 16.0 | 413 | True | 1.0041e-04 | -80.0 | [5.0326e-05, 1.5049e-04] | 1.0041e-04 |
| N30_r1.2 | 32 | 32.0 | 429 | True | 9.2166e-04 | -60.7 | [7.0733e-04, 1.1360e-03] | 9.2166e-04 |
| N30_r1.4 | 2 | 2.0 | 294 | True | 7.4916e-03 | -42.5 | [5.9633e-03, 9.0199e-03] | 7.4915e-03 |
| N30_r1.4 | 4 | 4.0 | 296 | True | 1.0141e-02 | -39.9 | [8.0826e-03, 1.2199e-02] | 1.0141e-02 |
| N30_r1.4 | 8 | 8.0 | 300 | True | 1.1296e-02 | -38.9 | [9.0066e-03, 1.3585e-02] | 1.1296e-02 |
| N30_r1.4 | 16 | 16.0 | 308 | True | 1.2101e-03 | -58.3 | [9.3810e-04, 1.4822e-03] | 1.2101e-03 |
| N30_r1.4 | 32 | 32.0 | 324 | True | 1.5111e-03 | -56.4 | [1.1789e-03, 1.8434e-03] | 1.5111e-03 |
| N30_r2 | 2 | 2.0 | 147 | True | 5.2225e-02 | -25.6 | [4.1750e-02, 6.2701e-02] | 5.2244e-02 |
| N30_r2 | 4 | 4.0 | 149 | True | 6.1758e-02 | -24.2 | [4.9377e-02, 7.4140e-02] | 6.1813e-02 |
| N30_r2 | 8 | 8.0 | 153 | True | 5.6522e-02 | -25.0 | [4.5187e-02, 6.7856e-02] | 5.6555e-02 |
| N30_r2 | 16 | 16.0 | 161 | True | 2.2583e-02 | -32.9 | [1.8037e-02, 2.7130e-02] | 2.2566e-02 |
| N30_r2 | 32 | 32.0 | 177 | True | 8.0389e-03 | -41.9 | [6.4011e-03, 9.6767e-03] | 8.0310e-03 |
| N60_r1.2 | 2 | 1.0 | 793 | True | 3.3671e-04 | -69.5 | [2.3937e-04, 4.3405e-04] | 3.3671e-04 |
| N60_r1.2 | 4 | 2.0 | 795 | True | 5.2078e-04 | -65.7 | [3.8662e-04, 6.5494e-04] | 5.2078e-04 |
| N60_r1.2 | 8 | 4.0 | 799 | True | 8.3206e-04 | -61.6 | [6.3565e-04, 1.0285e-03] | 8.3206e-04 |
| N60_r1.2 | 16 | 8.0 | 807 | True | 1.1025e-03 | -59.2 | [8.5204e-04, 1.3531e-03] | 1.1026e-03 |
| N60_r1.2 | 32 | 16.0 | 823 | True | 1.1832e-04 | -78.5 | [6.4655e-05, 1.7198e-04] | 1.1832e-04 |
| N60_r1.4 | 2 | 1.0 | 584 | True | 1.0020e-03 | -60.0 | [7.7159e-04, 1.2324e-03] | 1.0020e-03 |
| N60_r1.4 | 4 | 2.0 | 586 | True | 1.4807e-03 | -56.6 | [1.1546e-03, 1.8068e-03] | 1.4807e-03 |
| N60_r1.4 | 8 | 4.0 | 590 | True | 2.2779e-03 | -52.8 | [1.7923e-03, 2.7635e-03] | 2.2779e-03 |
| N60_r1.4 | 16 | 8.0 | 598 | True | 2.9156e-03 | -50.7 | [2.3025e-03, 3.5288e-03] | 2.9157e-03 |
| N60_r1.4 | 32 | 16.0 | 614 | True | 1.9694e-04 | -74.1 | [1.2755e-04, 2.6632e-04] | 1.9694e-04 |
| N60_r2 | 2 | 1.0 | 288 | True | 7.1028e-03 | -43.0 | [5.6522e-03, 8.5534e-03] | 7.1026e-03 |
| N60_r2 | 4 | 2.0 | 290 | True | 9.4856e-03 | -40.5 | [7.5585e-03, 1.1413e-02] | 9.4854e-03 |
| N60_r2 | 8 | 4.0 | 294 | True | 1.3247e-02 | -37.6 | [1.0568e-02, 1.5926e-02] | 1.3247e-02 |
| N60_r2 | 16 | 8.0 | 302 | True | 1.5321e-02 | -36.3 | [1.2227e-02, 1.8415e-02] | 1.5322e-02 |
| N60_r2 | 32 | 16.0 | 318 | True | 8.7464e-04 | -61.2 | [6.6971e-04, 1.0796e-03] | 8.7459e-04 |

### Table I — law check (i), model side (frozen)

| ratio | N / 30 | (dz/lambda)^2 ratio | model ratio R_single(N) / R_single(30) | model exponent | window on measured ratio / model ratio |
|---|---|---|---|---|---|
| 1.2 | 15 / 30 | 4.0000 | 3.8632 | 1.950 | [0.75, 1.25] |
| 1.2 | 60 / 30 | 0.2500 | 0.2524 | 1.986 | [0.75, 1.25] |
| 1.4 | 15 / 30 | 4.0000 | 3.8886 | 1.959 | [0.75, 1.25] |
| 1.4 | 60 / 30 | 0.2500 | 0.2521 | 1.988 | [0.75, 1.25] |
| 2.0 | 15 / 30 | 4.0000 | 4.8516 | 2.278 | [0.75, 1.25] |
| 2.0 | 60 / 30 | 0.2500 | 0.2441 | 2.035 | [0.75, 1.25] |

## Results (appended after measurement; no window above changed)

### First attempt (a2cb6cf3, resolution 30 only) — instrument defect found, not a physics result

The declared N = 30 call was run once on `a2cb6cf3`. The 30 / 1.4 control
cell returned lane A's five values digit for digit and its single ramp
5.7302e-3 (model 5.7907e-3, 1.0 %); 30 / 1.2 returned rows 1.2-2.1 % off
the model (24.8 % on the 1.0e-4 null row, inside its window); **30 / 2.0
returned R_meas = 1.13 on the single ramp and 0.17-1.13 on the band rows
— an amplitude ratio above 1, which no reflection can produce.**

Diagnosis (scratch diagnostics, no arm re-run): the A and B traces of
that cell differ before any reflection can reach the probe (A peaks at
0.83 at 0.387 ns, B at 3.40 at 0.569 ns; t_r = 1.09 ns), and the
difference tracks the number of fine cells in the profile, not the
geometry. Cause, read in the instrument: `_run_probe`, the lane-A helper
the E1 arm reused, injects the TE10 source at the module constant
`K_SRC = 85` and probes at `K_PRB = 100` — lane A's cell indices — not at
the setting's `k_src` / `k_prb`. Table S declares the source and probe
planes at 166.6 / 196.0 mm re-cut in the setting's coarse cell; what ran
was cell 85 / cell 100 in every profile. For 30 / 1.4 the two coincide
(the control was valid). For 30 / 1.2 (coarse 1.44 mm) the planes sat at
122.4 / 144.0 mm, still in the 275 mm coarse lead of both A and B, so
those rows are reflection measurements at an undeclared geometry (the
gate margins computed for 166.6 / 196.0 were conservative there: the
actual reflection arrives 0.375 ns later than computed and still 0.4 ns
inside the gate). For 30 / 2.0 (coarse 4.0 mm, 69-cell lead) cell 85 is
inside the A profiles' fine tail (single ramp), coarse tail (n_b <= 8) or
fine band (n_b = 16, 32) while B's cell 85 is a coarse lead cell: A and B
were different excitations, and their difference is not a reflection.

The first-attempt JSON is kept as
`results/e1_band_law_sweep_first_attempt_a2cb6cf3.json`; every row of it
is in the table below with its status. None of these numbers is quoted
against a window as a law result.

Provenance: git_sha `a2cb6cf3`, git_dirty False, started 2026-09-07T08:11:59Z, wallclock 11.2 s, argv `--sweep --fine-cells-per-lambda 30 --ratio 1.2,1.4,2.0 --widths 2,4,8,16,32 --out validation/research/multiband_nu/results/e1_band_law_sweep.json`.

| cell | arm | cell 85 / cell 100 as coarse-cell multiples (mm; the physical z in the A runs is smaller where fine cells precede the cell) | declared (mm) | R_meas | R_model | window (frozen) | recorded verdict | status of the number |
|---|---|---|---|---|---|---|---|---|
| N30_r1.2 | single | 122.4 / 144.0 | 167.0 / 195.8 | 2.1535e-03 | 2.1846e-03 | [1.7177e-03, 2.6515e-03] | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.2 | n_b = 2 | 122.4 / 144.0 | 167.0 / 195.8 | 2.5042e-03 | 2.5449e-03 | [2.0059e-03, 3.0839e-03] | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.2 | n_b = 4 | 122.4 / 144.0 | 167.0 / 195.8 | 3.5861e-03 | 3.6408e-03 | [2.8826e-03, 4.3989e-03] | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.2 | n_b = 8 | 122.4 / 144.0 | 167.0 / 195.8 | 4.2722e-03 | 4.3260e-03 | [3.4308e-03, 5.2212e-03] | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.2 | n_b = 16 | 122.4 / 144.0 | 167.0 / 195.8 | 7.5533e-05 | 1.0041e-04 | [5.0326e-05, 1.5049e-04] | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.2 | n_b = 32 | 122.4 / 144.0 | 167.0 / 195.8 | 9.0240e-04 | 9.2166e-04 | [7.0733e-04, 1.1360e-03] | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.2 | c fit | | | c_meas 1.397 mm | c_model 1.423 mm | +/- 0.120 mm | inside | reflection at an undeclared geometry; reported, not gated |
| N30_r1.4 | single | 166.6 / 196.0 | 166.6 / 196.0 | 5.7302e-03 | 5.7907e-03 | [4.6026e-03, 6.9788e-03] | inside | valid (planes coincide with lane A) |
| N30_r1.4 | n_b = 2 | 166.6 / 196.0 | 166.6 / 196.0 | 7.4364e-03 | 7.4916e-03 | [5.9633e-03, 9.0199e-03] | inside | valid (planes coincide with lane A) |
| N30_r1.4 | n_b = 4 | 166.6 / 196.0 | 166.6 / 196.0 | 1.0063e-02 | 1.0141e-02 | [8.0826e-03, 1.2199e-02] | inside | valid (planes coincide with lane A) |
| N30_r1.4 | n_b = 8 | 166.6 / 196.0 | 166.6 / 196.0 | 1.1164e-02 | 1.1296e-02 | [9.0066e-03, 1.3585e-02] | inside | valid (planes coincide with lane A) |
| N30_r1.4 | n_b = 16 | 166.6 / 196.0 | 166.6 / 196.0 | 1.2559e-03 | 1.2101e-03 | [9.3810e-04, 1.4822e-03] | inside | valid (planes coincide with lane A) |
| N30_r1.4 | n_b = 32 | 166.6 / 196.0 | 166.6 / 196.0 | 1.4248e-03 | 1.5111e-03 | [1.1789e-03, 1.8434e-03] | inside | valid (planes coincide with lane A) |
| N30_r1.4 | c fit | | | c_meas 1.887 mm | c_model 1.873 mm | +/- 0.140 mm | inside | valid (planes coincide with lane A) |
| N30_r2 | single | 340.0 / 400.0 | 168.0 / 196.0 | 1.1329e+00 | 3.1873e-02 | [2.5469e-02, 3.8278e-02] | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |
| N30_r2 | n_b = 2 | 340.0 / 400.0 | 168.0 / 196.0 | 8.5823e-01 | 5.2225e-02 | [4.1750e-02, 6.2701e-02] | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |
| N30_r2 | n_b = 4 | 340.0 / 400.0 | 168.0 / 196.0 | 6.1866e-01 | 6.1758e-02 | [4.9377e-02, 7.4140e-02] | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |
| N30_r2 | n_b = 8 | 340.0 / 400.0 | 168.0 / 196.0 | 1.6610e-01 | 5.6522e-02 | [4.5187e-02, 6.7856e-02] | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |
| N30_r2 | n_b = 16 | 340.0 / 400.0 | 168.0 / 196.0 | 8.6470e-01 | 2.2583e-02 | [1.8037e-02, 2.7130e-02] | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |
| N30_r2 | n_b = 32 | 340.0 / 400.0 | 168.0 / 196.0 | 1.1270e+00 | 8.0389e-03 | [6.4011e-03, 9.6767e-03] | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |
| N30_r2 | c fit | | | c_meas 6.000 mm | c_model 3.289 mm | +/- 0.200 mm | fired | not a reflection measurement (source inside A's fine tail / band / coarse tail; in B's coarse lead) |

Instrument fix (committed with this section, before the second attempt):
`_run_probe(profile, n_steps, k_src=K_SRC, k_prb=K_PRB)` — defaults keep
the lane-A F8 path byte-identical (its replay test passes) and E1 passes
the setting's cells for both the A arms and the B reference. Each arm now
records `k_src_used`, `k_prb_used`, `source_probe_planes_mm` and
`source_probe_in_coarse_lead` (the probe cell before the transition and
both cells equal to the coarse cell of THIS profile to 1e-12 m), and
`gates_hold` includes that check; the replay test pins all of it per
arm. Had that check existed, the first attempt would have reported
`gates_hold = False` on every 30 / 2.0 arm and on none of the others.

### Second attempt declared (all nine cells)

Why: the instrument did not realize the declared fixture on eight of the
nine cells (the control cell being the exception), so the first attempt
is not a measurement of the declared arms. This is a second execution of
the declared commands on the corrected instrument, not a re-roll of a
fired window: **no window, no law check, no expectation in sections 3
and 5 changes.** The control cell 30 / 1.4 is re-run with the others (its
source and probe cells are unchanged, so it must return its first-attempt
values to float32 reproducibility, and lane A's to 1e-6 relative — that
is a check on the fix, not a new measurement). Both attempts stay in the
tree; the second writes `results/e1_band_law_sweep.json`. Same three
calls as section 4, resolution 30 first.

### Second attempt — measured (b1cd9e63 / ec008ead; no window above changed)

Provenance: git_sha `ec008ead`, git_dirty True, argv of the finalizing call `--sweep --fine-cells-per-lambda 15,30,60 --ratio 1.2,1.4,2.0 --widths 2,4,8,16,32 --out validation/research/multiband_nu/results/e1_band_law_sweep.json --resume`, rfx_file `/Users/byungkwankim/Documents/rfx-nu-exp1/rfx/__init__.py`; the three measuring calls: N = 30 on `b1cd9e63` (fresh JSON), N = 15 and N = 60 on `b1cd9e63` with `--resume`; the finalizing `--resume` call on `ec008ead` re-ran no arm (all nine cells present) and recomputed the law-(i) records over all cells. Per-cell wallclock 2.8-4.6 s (B run + 6 arms).

### Table A — per-arm results (54 arms; window `|R_meas - R_model| <= 0.20 R_model + 3e-5`, frozen)

| cell | arm | nz | R_meas | dB | R_model | dev abs | dev rel (%) | window (frozen) | (ii) bound 2 R_1 x 1.05 | gates hold | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N15_r1.2 | single ramp | 243 | 8.3382e-03 | -41.6 | 8.4395e-03 | 1.01e-04 | 1.20 | [6.7216e-03, 1.0157e-02] | — | True | HELD |
| N15_r1.2 | n_b = 2 | 201 | 1.5884e-02 | -36.0 | 1.6048e-02 | 1.64e-04 | 1.02 | [1.2809e-02, 1.9288e-02] | 1.7723e-02 held | True | HELD |
| N15_r1.2 | n_b = 4 | 203 | 1.5246e-02 | -36.3 | 1.5452e-02 | 2.06e-04 | 1.34 | [1.2332e-02, 1.8572e-02] | 1.7723e-02 held | True | HELD |
| N15_r1.2 | n_b = 8 | 207 | 5.0114e-03 | -46.0 | 5.0067e-03 | 4.74e-06 | 0.09 | [3.9754e-03, 6.0380e-03] | 1.7723e-02 held | True | HELD |
| N15_r1.2 | n_b = 16 | 215 | 1.3329e-03 | -57.5 | 1.2700e-03 | 6.28e-05 | 4.94 | [9.8604e-04, 1.5541e-03] | 1.7723e-02 held | True | HELD |
| N15_r1.2 | n_b = 32 | 231 | 6.0761e-03 | -44.3 | 6.2037e-03 | 1.28e-04 | 2.06 | [4.9330e-03, 7.4744e-03] | 1.7723e-02 held | True | HELD |
| N15_r1.4 | single ramp | 218 | 2.2269e-02 | -33.0 | 2.2518e-02 | 2.49e-04 | 1.10 | [1.7984e-02, 2.7051e-02] | — | True | HELD |
| N15_r1.4 | n_b = 2 | 149 | 4.4091e-02 | -27.1 | 4.4597e-02 | 5.06e-04 | 1.13 | [3.5648e-02, 5.3547e-02] | 4.7287e-02 held | True | HELD |
| N15_r1.4 | n_b = 4 | 151 | 3.6858e-02 | -28.7 | 3.7341e-02 | 4.82e-04 | 1.29 | [2.9843e-02, 4.4839e-02] | 4.7287e-02 held | True | HELD |
| N15_r1.4 | n_b = 8 | 155 | 2.0654e-02 | -33.7 | 2.0803e-02 | 1.49e-04 | 0.71 | [1.6612e-02, 2.4993e-02] | 4.7287e-02 held | True | HELD |
| N15_r1.4 | n_b = 16 | 163 | 1.1287e-02 | -38.9 | 1.1330e-02 | 4.26e-05 | 0.38 | [9.0336e-03, 1.3625e-02] | 4.7287e-02 held | True | HELD |
| N15_r1.4 | n_b = 32 | 179 | 8.5780e-03 | -41.3 | 8.8391e-03 | 2.61e-04 | 2.95 | [7.0413e-03, 1.0637e-02] | 4.7287e-02 held | True | HELD |
| N15_r2 | single ramp | 182 | 1.7725e-01 | -15.0 | 1.5464e-01 | 2.26e-02 | 14.62 | [1.2368e-01, 1.8559e-01] | — | True | HELD |
| N15_r2 | n_b = 2 | 75 | 2.8346e-01 | -11.0 | 2.7263e-01 | 1.08e-02 | 3.98 | [2.1807e-01, 3.2718e-01] | 3.2473e-01 held | True | HELD |
| N15_r2 | n_b = 4 | 77 | 1.5513e-01 | -16.2 | 1.1613e-01 | 3.90e-02 | 33.59 | [9.2871e-02, 1.3938e-01] | 3.2473e-01 held | True | **FIRED** |
| N15_r2 | n_b = 8 | 81 | 2.4680e-01 | -12.2 | 2.6913e-01 | 2.23e-02 | 8.30 | [2.1527e-01, 3.2298e-01] | 3.2473e-01 held | True | HELD |
| N15_r2 | n_b = 16 | 89 | 2.3596e-01 | -12.5 | 2.3236e-01 | 3.59e-03 | 1.55 | [1.8586e-01, 2.7887e-01] | 3.2473e-01 held | True | HELD |
| N15_r2 | n_b = 32 | 105 | 9.6782e-02 | -20.3 | 1.2316e-01 | 2.64e-02 | 21.42 | [9.8501e-02, 1.4783e-01] | 3.2473e-01 held | True | **FIRED** |
| N30_r1.2 | single ramp | 486 | 2.1745e-03 | -53.3 | 2.1846e-03 | 1.01e-05 | 0.46 | [1.7177e-03, 2.6515e-03] | — | True | HELD |
| N30_r1.2 | n_b = 2 | 399 | 2.5541e-03 | -51.9 | 2.5449e-03 | 9.19e-06 | 0.36 | [2.0059e-03, 3.0839e-03] | 4.5876e-03 held | True | HELD |
| N30_r1.2 | n_b = 4 | 401 | 3.6439e-03 | -48.8 | 3.6408e-03 | 3.11e-06 | 0.09 | [2.8826e-03, 4.3989e-03] | 4.5876e-03 held | True | HELD |
| N30_r1.2 | n_b = 8 | 405 | 4.2903e-03 | -47.4 | 4.3260e-03 | 3.56e-05 | 0.82 | [3.4308e-03, 5.2212e-03] | 4.5876e-03 held | True | HELD |
| N30_r1.2 | n_b = 16 | 413 | 1.4175e-04 | -77.0 | 1.0041e-04 | 4.13e-05 | 41.18 | [5.0326e-05, 1.5049e-04] | 4.5876e-03 held | True | HELD |
| N30_r1.2 | n_b = 32 | 429 | 8.7097e-04 | -61.2 | 9.2166e-04 | 5.07e-05 | 5.50 | [7.0733e-04, 1.1360e-03] | 4.5876e-03 held | True | HELD |
| N30_r1.4 | single ramp | 435 | 5.7302e-03 | -44.8 | 5.7907e-03 | 6.05e-05 | 1.04 | [4.6026e-03, 6.9788e-03] | — | True | HELD |
| N30_r1.4 | n_b = 2 | 294 | 7.4364e-03 | -42.6 | 7.4916e-03 | 5.51e-05 | 0.74 | [5.9633e-03, 9.0199e-03] | 1.2160e-02 held | True | HELD |
| N30_r1.4 | n_b = 4 | 296 | 1.0063e-02 | -39.9 | 1.0141e-02 | 7.80e-05 | 0.77 | [8.0826e-03, 1.2199e-02] | 1.2160e-02 held | True | HELD |
| N30_r1.4 | n_b = 8 | 300 | 1.1164e-02 | -39.0 | 1.1296e-02 | 1.32e-04 | 1.16 | [9.0066e-03, 1.3585e-02] | 1.2160e-02 held | True | HELD |
| N30_r1.4 | n_b = 16 | 308 | 1.2559e-03 | -58.0 | 1.2101e-03 | 4.58e-05 | 3.78 | [9.3810e-04, 1.4822e-03] | 1.2160e-02 held | True | HELD |
| N30_r1.4 | n_b = 32 | 324 | 1.4248e-03 | -56.9 | 1.5111e-03 | 8.63e-05 | 5.71 | [1.1789e-03, 1.8434e-03] | 1.2160e-02 held | True | HELD |
| N30_r2 | single ramp | 364 | 3.1445e-02 | -30.0 | 3.1873e-02 | 4.29e-04 | 1.34 | [2.5469e-02, 3.8278e-02] | — | True | HELD |
| N30_r2 | n_b = 2 | 147 | 5.1606e-02 | -25.7 | 5.2225e-02 | 6.20e-04 | 1.19 | [4.1750e-02, 6.2701e-02] | 6.6934e-02 held | True | HELD |
| N30_r2 | n_b = 4 | 149 | 6.0981e-02 | -24.3 | 6.1758e-02 | 7.77e-04 | 1.26 | [4.9377e-02, 7.4140e-02] | 6.6934e-02 held | True | HELD |
| N30_r2 | n_b = 8 | 153 | 5.5724e-02 | -25.1 | 5.6522e-02 | 7.98e-04 | 1.41 | [4.5187e-02, 6.7856e-02] | 6.6934e-02 held | True | HELD |
| N30_r2 | n_b = 16 | 161 | 2.2454e-02 | -33.0 | 2.2583e-02 | 1.30e-04 | 0.57 | [1.8037e-02, 2.7130e-02] | 6.6934e-02 held | True | HELD |
| N30_r2 | n_b = 32 | 177 | 8.0931e-03 | -41.8 | 8.0389e-03 | 5.42e-05 | 0.67 | [6.4011e-03, 9.6767e-03] | 6.6934e-02 held | True | HELD |
| N60_r1.2 | single ramp | 970 | 5.4227e-04 | -65.3 | 5.5131e-04 | 9.04e-06 | 1.64 | [4.1104e-04, 6.9157e-04] | — | True | HELD |
| N60_r1.2 | n_b = 2 | 793 | 3.3807e-04 | -69.4 | 3.3671e-04 | 1.36e-06 | 0.40 | [2.3937e-04, 4.3405e-04] | 1.1577e-03 held | True | HELD |
| N60_r1.2 | n_b = 4 | 795 | 5.2290e-04 | -65.6 | 5.2078e-04 | 2.12e-06 | 0.41 | [3.8662e-04, 6.5494e-04] | 1.1577e-03 held | True | HELD |
| N60_r1.2 | n_b = 8 | 799 | 8.3168e-04 | -61.6 | 8.3206e-04 | 3.76e-07 | 0.05 | [6.3565e-04, 1.0285e-03] | 1.1577e-03 held | True | HELD |
| N60_r1.2 | n_b = 16 | 807 | 1.0886e-03 | -59.3 | 1.1025e-03 | 1.39e-05 | 1.26 | [8.5204e-04, 1.3531e-03] | 1.1577e-03 held | True | HELD |
| N60_r1.2 | n_b = 32 | 823 | 1.0093e-04 | -79.9 | 1.1832e-04 | 1.74e-05 | 14.69 | [6.4655e-05, 1.7198e-04] | 1.1577e-03 held | True | HELD |
| N60_r1.4 | single ramp | 869 | 1.4436e-03 | -56.8 | 1.4597e-03 | 1.61e-05 | 1.10 | [1.1378e-03, 1.7816e-03] | — | True | HELD |
| N60_r1.4 | n_b = 2 | 584 | 1.0066e-03 | -59.9 | 1.0020e-03 | 4.62e-06 | 0.46 | [7.7159e-04, 1.2324e-03] | 3.0654e-03 held | True | HELD |
| N60_r1.4 | n_b = 4 | 586 | 1.4880e-03 | -56.5 | 1.4807e-03 | 7.30e-06 | 0.49 | [1.1546e-03, 1.8068e-03] | 3.0654e-03 held | True | HELD |
| N60_r1.4 | n_b = 8 | 590 | 2.2805e-03 | -52.8 | 2.2779e-03 | 2.60e-06 | 0.11 | [1.7923e-03, 2.7635e-03] | 3.0654e-03 held | True | HELD |
| N60_r1.4 | n_b = 16 | 598 | 2.8845e-03 | -50.8 | 2.9156e-03 | 3.12e-05 | 1.07 | [2.3025e-03, 3.5288e-03] | 3.0654e-03 held | True | HELD |
| N60_r1.4 | n_b = 32 | 614 | 1.5633e-04 | -76.1 | 1.9694e-04 | 4.06e-05 | 20.62 | [1.2755e-04, 2.6632e-04] | 3.0654e-03 held | True | HELD |
| N60_r2 | single ramp | 726 | 7.6709e-03 | -42.3 | 7.7788e-03 | 1.08e-04 | 1.39 | [6.1931e-03, 9.3646e-03] | — | True | HELD |
| N60_r2 | n_b = 2 | 288 | 7.0269e-03 | -43.1 | 7.1028e-03 | 7.59e-05 | 1.07 | [5.6522e-03, 8.5534e-03] | 1.6336e-02 held | True | HELD |
| N60_r2 | n_b = 4 | 290 | 9.3869e-03 | -40.5 | 9.4856e-03 | 9.86e-05 | 1.04 | [7.5585e-03, 1.1413e-02] | 1.6336e-02 held | True | HELD |
| N60_r2 | n_b = 8 | 294 | 1.3112e-02 | -37.6 | 1.3247e-02 | 1.35e-04 | 1.02 | [1.0568e-02, 1.5926e-02] | 1.6336e-02 held | True | HELD |
| N60_r2 | n_b = 16 | 302 | 1.5123e-02 | -36.4 | 1.5321e-02 | 1.97e-04 | 1.29 | [1.2227e-02, 1.8415e-02] | 1.6336e-02 held | True | HELD |
| N60_r2 | n_b = 32 | 318 | 9.3702e-04 | -60.6 | 8.7464e-04 | 6.24e-05 | 7.13 | [6.6971e-04, 1.0796e-03] | 1.6336e-02 held | True | HELD |

### Table B — law checks per cell and the validity-domain verdicts

| cell | R_single meas / model (dev %) | c_meas (mm) | c_model (mm) | dev (mm) | window +/- (mm) | (iii) | widths fired | (ii) fired | gates hold (all 6) | law domain | -54 dB class |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N15_r1.2 | 8.3382e-03 / 8.4395e-03 (1.20) | 2.915 | 2.892 | 0.024 | 0.240 | HELD | none | no | True | **inside** | outside |
| N15_r1.4 | 2.2269e-02 / 2.2518e-02 (1.10) | 3.885 | 3.873 | 0.012 | 0.280 | HELD | none | no | True | **inside** | outside |
| N15_r2 | 1.7725e-01 / 1.5464e-01 (14.62) | 6.663 | 7.166 | 0.503 | 0.400 | **FIRED** | [4, 32] | no | True | **OUTSIDE** | outside |
| N30_r1.2 | 2.1745e-03 / 2.1846e-03 (0.46) | 1.469 | 1.423 | 0.045 | 0.120 | HELD | none | no | True | **inside** | inside |
| N30_r1.4 | 5.7302e-03 / 5.7907e-03 (1.04) | 1.887 | 1.873 | 0.014 | 0.140 | HELD | none | no | True | **inside** | inside |
| N30_r2 | 3.1445e-02 / 3.1873e-02 (1.34) | 3.281 | 3.289 | 0.008 | 0.200 | HELD | none | no | True | **inside** | outside |
| N60_r1.2 | 5.4227e-04 / 5.5131e-04 (1.64) | 0.742 | 0.709 | 0.033 | 0.060 | HELD | none | no | True | **inside** | inside |
| N60_r1.4 | 1.4436e-03 / 1.4597e-03 (1.10) | 0.964 | 0.929 | 0.034 | 0.070 | HELD | none | no | True | **inside** | inside |
| N60_r2 | 7.6709e-03 / 7.7788e-03 (1.39) | 1.595 | 1.611 | 0.016 | 0.100 | HELD | none | no | True | **inside** | outside |

### Table C — law check (i): single-ramp resolution scaling, measured vs model (window [0.75, 1.25] on measured / model)

| ratio | N / 30 | (dz/lambda)^2 | model ratio (exponent) | measured ratio | measured / model | verdict |
|---|---|---|---|---|---|---|
| 1.2 | 15 / 30 | 4.0000 | 3.8632 (1.950) | 3.8345 | 0.9926 | HELD |
| 1.2 | 60 / 30 | 0.2500 | 0.2524 (1.986) | 0.2494 | 0.9882 | HELD |
| 1.4 | 15 / 30 | 4.0000 | 3.8886 (1.959) | 3.8862 | 0.9994 | HELD |
| 1.4 | 60 / 30 | 0.2500 | 0.2521 (1.988) | 0.2519 | 0.9994 | HELD |
| 2.0 | 15 / 30 | 4.0000 | 4.8516 (2.278) | 5.6368 | 1.1619 | HELD |
| 2.0 | 60 / 30 | 0.2500 | 0.2441 (2.035) | 0.2439 | 0.9996 | HELD |

### Table D — the 3 x 3 validity map (rows: fine cells per free-space wavelength; columns: ratio)

| N \ r | 1.2 | 1.4 | 2.0 |
|---|---|---|---|
| 15 | law inside / class outside; R_1 -41.5 dB | law inside / class outside; R_1 -32.9 dB | law OUTSIDE / class outside; R_1 -16.2 dB |
| 30 | law inside / class inside; R_1 -53.2 dB | law inside / class inside; R_1 -44.7 dB | law inside / class outside; R_1 -29.9 dB |
| 60 | law inside / class inside; R_1 -65.2 dB | law inside / class inside; R_1 -56.7 dB | law inside / class outside; R_1 -42.2 dB |

### Reading the tables

**Control.** The 30 / 1.4 cell returns its first-attempt values and lane
A's six measured values bit for bit (relative difference 0.0 on the single
ramp and all five widths) — the fix moved nothing where the planes
already coincided. The 30 / 1.2 rows moved 0.4-3.6 % (47 % on the 1.0e-4
null row) between the undeclared planes of the first attempt and the
declared ones, which is the size of the effect of a 45 mm plane shift
through the gate's leakage — reported for the record, the first-attempt
rows are not law results.

**Per-arm windows (54 arms).** 52 HELD, **2 FIRED**, both in the 15 / 2.0
cell: n_b = 4 (R_meas 1.5513e-1 vs model 1.1613e-1, +33.6 %) and
n_b = 32 (9.678e-2 vs 1.2316e-1, -21.4 %); that cell's single ramp
(+14.6 %), n_b = 2 (+4.0 %), 8 (-8.3 %) and 16 (+1.6 %) are inside. Every
other arm is within 0.05-5.7 % of the chain model on rows above 2.5e-4
and within 1.7e-5 .. 4.1e-5 ABSOLUTE on the three rows below it (1f:
14.7 %, 20.6 % and 41.2 % relative on 1.18e-4, 1.97e-4 and 1.00e-4 —
all inside their windows, so the declared floor reading rule was never
invoked; the instrument's absolute floor on this fixture measures
<= 4.1e-5 at 60 cells and <= 8.6e-5 over every row under 3e-3).
Gates held on all 54 arms (the per-arm plane check included).

**Law (i), resolution scaling of the single ramp.** HELD on all six
pairs, measured / model 0.9926, 0.9882 (r = 1.2), 0.9994, 0.9994 (1.4),
1.1619, 0.9996 (2.0). Measured ratios 3.83 / 3.89 (15 vs 30) and
0.2494 / 0.2519 (60 vs 30) at r <= 1.4 against the pure (dz/lambda)^2
values 4 / 0.25: the single-ramp reflection scales as (dz/lambda)^2 to
within 4 % over 15-60 cells per wavelength at r <= 1.4, and the chain
model's own 3 % shortfall at 15 cells is what the FDTD returns. At
r = 2.0 the 15-cell ramp reflects 5.64 x the 30-cell one (model 4.85,
exponent 2.28): still inside the 25 % window on measured / model, but the
(dz/lambda)^2 statement itself is not a fit there.

**Law (ii), the bound.** HELD at all 45 widths, including the fired cell
(its largest row, 0.2835, sits under 2 x 0.1546 x 1.05 = 0.3247).

**Law (iii), c.** HELD on eight cells, **FIRED on 15 / 2.0** (c_meas
6.663 mm vs c_model 7.166 mm, deviation 0.503 mm against +/- 0.400 mm).
On the eight held cells c_meas / DR = 1.215 / 1.224 / 1.237 (r = 1.2 at
15 / 30 / 60 cells), 1.388 / 1.348 / 1.377 (1.4), 1.641 / 1.595 (2.0 at
30 / 60) against the closed form 2 r^2 / (1 + r^2) = 1.180 / 1.324 /
1.600: c is linear in the ramp cell at fixed ratio (the three
resolutions agree within 2 % of each other at every ratio) and the
closed form predicts it to +3-5 % (r = 1.2), +2-5 % (1.4), 0-3 % (2.0).
The prediction of 1g stands as measured: c scales with DR = r x dz_fine.

**The one cell outside the law domain, 15 / 2.0, is the one 1e named
before the run** (coarse cell 8.0 mm = 3.75 cells per free-space
wavelength, vg 0.60 c, coarse-lattice stopband edge 12.97 GHz inside the
pulse's band). The arms fail in the pattern of gate leakage, not of a
different law: the errors alternate in sign along the width ladder
(+4.0, +33.6, -8.3, +1.6, -21.4 %), the bound holds, and the two fired
rows are the two that sit nearest a null of the Fabry-Perot form
(n_b = 4 and 32 are the two smallest R_model of the cell, 0.116 and
0.123). Whether the chain model or the instrument owns those two numbers
is not decided by this lane: the chain model is exact for the discrete
scheme at a single frequency, the instrument's gate is a rectangular
window over a pulse of which 12 % (the part above 12.97 GHz) cannot
propagate in the coarse cells at all. Recorded as OUTSIDE; not tuned.

### Validity domain, as measured

Law (the deliverable, quoted with its range): for a fine band of n_b
cells between two cap ramps (coarse r^2 d -> r d -> d and back), the
reflection at 10 GHz on the PEC-closed TE10 fixture is

    R(n_b) = 2 R_single |sin(k_g (n_b d + c))|,   c = 2 (r d) r^2 / (1 + r^2)  (+3-5 %),
    R_single ∝ (d / lambda)^2 at fixed r (within 4 % over 15-60 cells for r <= 1.4),
    R(n_b) <= 2 R_single at every width,

with R_single the chain model's single ramp (measured within 0.5-1.6 %
on eight cells, 14.6 % on 15 / 2.0). Inside the law domain: **eight of
nine cells** — fine bands of 2-32 cells at 15, 30 and 60 fine cells per
free-space wavelength for r = 1.2 and 1.4, and at 30 and 60 cells for
r = 2.0 (FDTD within 0.05-5.7 % of the chain model on every row above
2.5e-4). Outside: 15 / 2.0 (coarse cell 3.75 cells per wavelength; two
of five width rows and the c window fired). Inside the -54 dB accuracy
class (r <= 1.4, >= 30 cells): four cells, all inside the law domain.
The law domain is larger than the accuracy class: the 15-cell cells at
r <= 1.4 and the r = 2.0 cells at 30 and 60 cells follow the law with
R_single of -41.5 / -32.9 / -29.9 / -42.2 dB — outside the class by the
reflection they carry, not by the law they follow.

Not witnessed here: bands under 2 cells, in-plane grading, an absorber
present, frequencies other than 10 GHz, ratios above 2.0, coarse cells
under 3.75 cells per free-space wavelength.

`stopped = false`. Windows unchanged. Three commits carry this lane: the
pre-declaration (`a2cb6cf3`), the first attempt with its defect record
and the fix (`b1cd9e63`), the aggregation fix (`ec008ead`); the results
JSON and this section follow in one commit.

### Replay tolerance re-declared (2026-09-11)

`tests/unit/nonuniform/test_e1_band_law_replay.py` replayed chain-model
floats to 1e-9 relative, on the strength of lane A reproducing its own rows
to 3e-10 on one machine. A second machine falsified that: the dense solve
of the N60_r1.2 profile (n_b = 32, 1100 cells) read 1.1831862931777621e-4
on one GitHub runner (PR #963, fast-suite (4)) and 1.1831862916448015e-4 on
another (main run 34572755776 at b7ad4e93) and on the lab pod -- 1.3e-9
apart, from the LAPACK build, not the physics. The replay bound for
chain-model floats is now 1e-8 (8x the measured spread); no window, gate
or committed number moved. Same class as the W6 F7 finding (PR #956),
whose remedy there was a closed form for the single junction; the full
band profiles here have no closed form, so the bound carries the
measurement instead.
