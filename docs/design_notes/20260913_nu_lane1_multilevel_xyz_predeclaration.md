# Lane 1 — multi-level unequal bands on x, y, z (E5) — pre-declaration

**Status:** pre-declaration. This commit precedes every FDTD measurement of
the lane. Every numeric window below is copied from the program document
(`20260913_nu_full_functionality_program.md` section 4.4, frozen there at
`d6bc5dde`) and is frozen here again; results are appended under
"Results", never edited into the windows. One attempt per arm; a fired
window is a result, not a bug to tune away.
**Class:** validity-domain measurement of a law already witnessed on one
axis (E1, `20260907_nu_exp1_band_law_sweep_predeclaration.md`: z, one fine
+ one coarse size, symmetric cap ramps, 52 of 54 arms held) — extended to
unequal coarse sides, asymmetric traversals, two fine sizes in one column,
and the x and y axes.
**Tree:** worktree `rfx-nu-full`, branch `feat/nu-full-functionality`,
based on `feat/nu-ad-directional @ 88460d24` (on `origin/main fa392913`);
program document at `d6bc5dde`. Instrument: the new module
`validation/research/multiband_nu/e5_multilevel_axes.py` (this commit).
The five files `w6_band_builder.py`, `chain_model.py`,
`w2_w3_reflection.py`, `harness.py`, `fixtures.py` are not edited by this
lane (`git diff --stat` on them is empty at every commit). No `rfx/`
change.
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-full/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary
checkout). Every JSON the instrument writes records `rfx.__file__`,
`git_sha`, `git_dirty`, `argv`, `started_utc`. The chain-model
predictions in this note are the file `results/e5_model.json`, written by
`--model-only` on this tree (HEAD `d6bc5dde`, `git_dirty` true because
the instrument, this note and that JSON are one commit); the tables in
section 5 are generated from it by `--note-tables`, not retyped. CPU
only; baseline before any change: `test_e1_band_law_replay.py` +
`test_band_builder_chain_model.py` = 56 passed.
Date: 2026-09-14 (KST).

## 0. Question (program 4.1) and what this note adds to it

Does the z transition law measured by E1 extend to (a) a fine band between
two DIFFERENT coarse sizes, traversed in both directions, (b) two fine
bands of different sizes in one column, and (c) the same structures on
the x and y axes — and what is its range there? Deliverable: a table of
(level pattern, axis) cells with inside/outside verdicts, the law
generalized to unequal ramp amplitudes (the two-amplitude Fabry-Perot
form), and the first FDTD-measured in-plane reflection numbers.

Everything the program document specifies for Lane 1 (4.2 instrument,
4.3 arms, 4.4 windows, 4.5 reading rule, 4.6 regression, 4.7 cost, 4.8
PI confirmation) is taken as written. Section 1 records what was found
while turning it into code, with zero FDTD; where the program's wording
could not be built as written, the declared alternative is stated there
BEFORE any run.

## 1. Findings made while writing this note (zero FDTD)

### 1a. Every declared vector is the builder's output (G5 rule applied)

Each arm is declared as a list of segments `(role, target, N, protected)`
with N the TOTAL count of target-valued cells; the ramp between two
segments is the builder's own geometric ramp (`m = ceil(ln(ratio)/ln(cap)
- 1e-10)` steps, `rho = ratio^(1/m)`, placed inside the coarser segment,
its last cell being the first plateau cell and counted in N), and the
span of every segment is `pin + ramp cells + N x target`, accumulated left
to right with `N x target` as one product — so the S pattern on z hands
`make_band_profile` the E1 edges bit for bit (`z1 = 140 dc + dr`, `z3 =
(z2 + dr) + 150 dc`, checked: identical edges, identical expected
vectors, identical builder output for n_b = 2..32 and the single). For an
exact power of the cap the ramp cell is `v x cap^i` (E1's `d_f x r`); for
the 0.5 -> 1.96 mm ramp `rho = 3.92^(1/5) = 1.3141871`, cells 0.65709 /
0.86354 / 1.13486 / 1.49142 mm (the program's "1.4888 / 1.1334 / 0.8628 /
0.6568" were the G5 review's mis-declared-span plateau, not this rule).
Measured on this tree, every declared vector of every arm on z and on the
pinned axes matches the builder to <= 2.6e-17 m (the E1 flag is 1e-12 m);
the builder's realized band cell is 1.0000000000000009e-3 m (S, n_b <=
16), 9.9999999999999915e-4 m (n_b = 32) and 5.0000000000000044e-4 m
(T) — the same ulp dust E1 carried.

One builder behaviour had to be declared around: an end segment whose
target EQUALS the pin (the 1.0 mm free end after a protected 0.5 mm band,
pattern T on x/y) cannot be realized with one target cell — the plateau
solve holds the pin-side ramp classification over a closed piece and
counts a one-cell "down ramp" as a plateau cell, so with N = 1 it lands
on `[0.6634, 0.8801, 0.8801, 1.0]` instead. With N = 2 the output is the
declared `[0.62996, 0.79370, 1.0, 1.0] + pin`, exact. The end segment is
declared with N = 2 (three 1.0 mm cells at the end, all instrumentation).

### 1b. The program's B profile for T on x/y is not buildable as written

Program 4.2: `make_band_profile([0, L_B, L_B + 4 x 0.5 mm], [1.96, 0.5],
protected=[False, True], max_ratio=1.4, boundary_cell=1.0e-3)`. Measured:
`ValueError: boundary_cell needs a FREE last segment (a protected end
segment cannot host the pinned cell)`. Declared alternative (same
purpose — a realized 0.5 mm band beyond the gate that pins dt, ends at
the 1.0 mm pin): three segments `[lead 1.96 x 400 (free), 0.5 x 4
(protected), end 1.0 x 2 (free)]` with `boundary_cell = 1.0 mm`, which the
builder realizes exactly as `[1.0, 1.4, 1.96 x 400, 1.4914, 1.1349,
0.8635, 0.6571, 0.5 x 4, 0.63, 0.7937, 1.0, 1.0, 1.0]` (415 cells). Its
far structure begins after 400 lead cells, as E1's B does (margin
`b_far_return_direct_after_gate` 2.92 ns, Table G).

### 1c. Pins, the chain slice, and the source/probe indices on x and y

`boundary_cell = 1.0 mm` puts `[1.0, 1.4]` before a 1.96 mm lead (m_lo =
2), `[1.0, 1.4, 1.96]` before a 2.744 mm lead (m_lo = 3), `[1.0]` before
a 1.4 mm lead (m_lo = 1), and NOTHING distinguishable before a 1.0 mm
lead (the pin IS a lead cell; m_lo = 1 by bookkeeping, the wave sees a
uniform 275-cell fine lead). Mirror images at the hi end (m_hi = 2 / 3 /
1; after a protected fine tail the end segment of 1a: m_hi = 3, after
T's 0.5 mm tail m_hi = 5, after T's L1 dt-pin m_hi = 11). The chain model
is solved on the structure slice `prof[m_lo : len - m_hi]` — the pins,
their ramps, dt-pin bands and end segments are instrumentation shared by
A and B. Consequence checked on this tree: `R_model` of every arm is the
SAME number on z, x and y (Table R), so the x/y predictions ARE the z
predictions; only the gate times move (pin delay 0.019 ns per pass).
`k_src_used = m_lo + k_src`, `k_prb_used = m_lo + k_prb` (S: 87/102, P2:
64/74, P4: 120/141, fine-lead singles 168/197).

### 1d. B references: five per axis, and the dt pin is the DECLARED minimum

Per axis the arms need five B runs, one per lead cell: `B_1.96` (S, P1,
P3 bands and their coarse-lead singles), `B_2.744` (P2), `B_1.4` (P4),
`B_1.0` (the four fine-lead singles P1.R, P2.R, P3.R, P4.R), `B_T`
(1.96 mm lead with the 0.5 mm pin). On z the B is E1's explicit vector
`[lead] x n_B + [d_min] x 4` with `d_min` the DECLARED family minimum
(1.0 mm, or 0.5 mm for T) — not the realized minimum the program's 4.2
wording names. Reason: W4 asks the L1-0 control to reproduce E1 to 1e-6
and reports bit-identity, and E1's B is exactly this vector; the ulp
between the declared and realized minimum (5.0000000000000044e-4 vs
5e-4 m on T, 4.4e-19 m) moves dt by 1e-15 relative, 1.5e-27 s, against
the 1e-20 s `dt_matches_b` criterion — reported per arm as `dt_diff_s`,
never asserted. On x/y the B is built by `make_band_profile` with the
same `boundary_cell` (1b for T); `B_1.0` there is `[1.0] x 786` (784 lead
cells + 2 pins) against z's `[1.0] x 788`, both far ends after the gate.
n_B = 400 / 286 / 560 / 784 cells (784.0 mm re-cut in the lead cell).

### 1e. The relabel identity needs its own B: E1's B fails the x/y guard

E1's B `[1.96] x 400 + [1.0] x 4` ends on 1.0 mm: on x
`dx_profile[-1] != dx` raises, on y the two ends differ and raise (G1).
L1-R therefore uses `B_sym = [1.96] x 400 | 1.4 | [1.0] x 4 | 1.4 |
[1.96] x 4` (builder-made from `[lead 400 (free), 1.0 x 4 (protected),
end 1.96 x 4 (free)]`, 410 cells, both ends 1.96 mm, passes G1 on x and
y without pins), run on z, x AND y, so the relabel comparison is
B-for-B. Declared and reported (not a window): on z, `R_meas(B_sym)` vs
the L1-0 `R_meas(E1 B)` for S n_b = 4 — the two B traces are identical
before the gate up to the numerical precursor of the far structure, so
the expectation is <= 1e-6 relative; the re-run of the z A arm in the
relabel call must be bit-identical to the L1-0 trace (same tree, same
process class), reported as a boolean.

### 1f. A sixth gate margin: the DIRECT far-end return of B

E1's `b_pin_return_after_gate` is `T0 + (z_src + 2 z_B - z_prb)/vg -
gate_end`, the path source -> lo wall -> far end -> probe; the earliest
far-end arrival is the direct path `T0 + (2 z_B - z_src - z_prb)/vg`,
2 z_src / vg = 1.30 ns earlier. Both are computed; `gates_hold` requires
all six margins > 0. The direct margin is 2.89-2.98 ns on every arm and
axis (Table G) — the E1 conclusion stands, the number E1 printed for it
did not describe the earliest return.

### 1g. The frozen S model against E1's committed model

`R_model` of the six S arms differs from E1's `results/e1_band_law_sweep.
json` N30_r1.4 values by 4.4e-13 to 7.1e-13 relative (same builder
input, same profile, same dt; the dense-solve noise class of G18,
against the 1e-8 W4 model window); `c_model` 1.8730539100875 mm vs E1's
1.8730539103500 mm, 2.6e-13 m (window 1e-9 m in the replay). Gate
margins: this instrument evaluates gates on the BUILDER output (as
E1's measured arms did), E1's model rows used the declared vector — the
`inner_return_inside` margin therefore differs from E1's MODEL rows by
1.3e-12 ns (the finite-difference `vg_of` on a 1.0000000000000009e-3 m
cell) and is expected to equal E1's measured-ARM rows exactly. The other
four margins are bit-identical to E1's model rows.

### 1h. What the chain model says the new patterns do (the numbers a surprise is measured against)

- P1/P2 (band between 1.96 and 2.744 mm): the three-step side reflects
  `R = 1.2460e-2` (steps 1.998e-3, 3.963e-3, 7.955e-3 — the 1.96 -> 2.744
  step alone is 37 % above the whole S ramp), 2.15 x the one-ramp side
  `5.7907e-3`. Fabry-Perot minimum `|R_L - R_R| = 6.669e-3`, chain
  minimum over n_b = 0..80 `6.6695e-3`; the five declared rows sit at
  7.0e-3 .. 1.78e-2 (-43 .. -35 dB). `c_model = 3.294 mm` on BOTH
  traversals (P1 and P2 give the same R_model to 1e-15: reciprocity of a
  lossless two-port, |S11| = |S22|), closed-form centroid 0.927 + 2.301
  = 3.228 mm (-0.066 mm, inside the 0.140 mm window by half). So the
  program's likeliest-fire "W3 on P2" has no model support: the chain
  says P1 and P2 are the same cell.
- P3/P4 (one side is the 1.4 mm coarse itself): the stepped side is one
  cap step, `1.9982e-3` (0.345 x the ramp side); minimum `3.792e-3`,
  rows 3.8e-3 .. 7.8e-3; `c_model = 0.937 mm`, closed form 0.927 mm
  (-0.010 mm). Again P3 = P4 to 1e-15.
- T: `R_1L = R_1R = 5.7996e-3` (the S ramp at dt_T), `R_2L = R_2R =
  6.6366e-3` (the five-step 1.96 -> 0.5 mm ramp; steps 3.416e-3,
  1.957e-3, 1.126e-3, 6.50e-4, 3.75e-4); sum 2.4872e-2, bound 2.6116e-2.
  Band rows 2.2989e-2 / 2.3139e-2 (n_c = 4) and 1.2933e-2 / 9.2903e-3
  (n_c = 16); the largest is 0.88 of the bound.
- Smallest modelled row anywhere: 1.2101e-3 (S n_b = 16, E1's) — no row
  is under the 2.5e-4 floor rule; it is carried, not expected to fire.
- Runways: 2.744 mm = 10.93 cells per lambda0, Bloch argument 0.249,
  vg/c 0.841, stopband edge 35.56 GHz (1.96 mm: 15.30 / 0.178 / 0.855 /
  50.14 GHz). dt_T = 1.493514e-12 s, n_steps 1931; the 1.0 mm family
  dt = 2.402765e-12 s, n_steps 1200 on every axis.

### 1i. Axis identities measured while writing (no FDTD)

For the S n_b = 4 profile: grids (4, 21, 297) on z, (297, 4, 21) on x,
(21, 297, 4) on y; `dt_x - dt_z = 0.0`, `dt_y - dt_z = 0.0`
(2.402764936661261e-12 s); `np.array_equal` of the float32 `inv_dz` /
`inv_dz_h` (z) with `inv_dx` / `inv_dx_h` (x) and `inv_dy` (y): True.
The scalar `grid.dy` on the x fixture and `grid.dx` on the y fixture are
1.5 mm (the transverse cell), `grid.dx` on x and `grid.dy` on y are the
profile's end cell (1.96 mm unpinned, 1.0 mm pinned) — the closed-box
path must not read them; L1-R is the measurement of that.

### 1j. Run accounting (declared): identical runs execute once

A run is keyed by (axis, profile bytes, k_src, k_prb, n_steps); within
one call an identical key is executed once and every arm that uses it
records `run_id` and `fdtd_run_cached`. P1.L and P3.L are S.L; P4.R is
P2.R; the 1.0-family singles share `B_1.0`. Distinct FDTD runs per axis:
5 B + 8 singles + 29 band arms = 42; three axes 126; L1-R 6 (z A re-run,
B_sym on z/x/y, A on x/y); pin bridge 7 (pinned S: B, single, five
widths) — 139 runs, each 0.4-0.8 s plus a JIT per distinct grid shape.

## 2. Fixture (lane A / E1 values, unchanged; program 4.2)

F0 = 10 GHz, sigma_t = 64 ps, t0 = 5 sigma_t, d_t = 1.5 mm, b = 30 mm (20
cells on the sine axis), a = 4.5 mm (3 cells on the invariant axis), PEC
closed, `cpml_layers = 0`, cap 1.4, fine d_f = 1.0 mm (29.98 cells per
lambda0), pattern T's second fine size 0.5 mm. Axis map (cyclic): z =
(`ex`, invariant x, sine y) — the E1 fixture through `harness.
build_pec_fixture` verbatim; x = (`ey`, invariant y, sine z) built as
`make_nonuniform_grid((0, 0), dz_profile=full(20, d_t), dx=profile[0],
cpml_layers=0, dx_profile=profile, dy_profile=full(3, d_t))`; y = (`ez`,
invariant z, sine x) with `dx_profile=full(20, d_t), dx=d_t,
dy_profile=profile, dz_profile=full(3, d_t)`. Sources: the E1 list in the
E1 order (invariant outer, sine inner, `sin(pi s d_t / b)`), component
relabeled, `k_src` in the graded slot; probe at (1, n_sin // 2, k_prb)
in (invariant, sine, graded) order. Physical planes re-cut in the lead
cell (E1 1c): source 166.6 mm, probe 196.0 mm, lead 274.4 mm, tail
294.0 mm, B 784.0 mm from the end of the lo pin, rounded half-up.

### Table S — settings per pattern (z; x/y differ by the pin cells only)

| pattern | c_L (mm) | c_R (mm) | d_min (mm) | dt (s) | n_steps | k_src/k_prb | n_lead/n_tail/n_B | coarsest cells/lambda0 | stopband (GHz) | -54 dB class |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 1.960 | 1.960 | 1.00 | 2.402765e-12 | 1200 | 85/100 | 140/150/400 | 15.30 | 50.14 | inside |
| P1 | 1.960 | 2.744 | 1.00 | 2.402765e-12 | 1200 | 85/100 | 140/107/400 | 10.93 | 35.56 | inside |
| P2 | 2.744 | 1.960 | 1.00 | 2.402765e-12 | 1200 | 61/71 | 100/150/286 | 10.93 | 35.56 | inside |
| P3 | 1.960 | 1.400 | 1.00 | 2.402765e-12 | 1200 | 85/100 | 140/210/400 | 15.30 | 50.14 | inside |
| P4 | 1.400 | 1.960 | 1.00 | 2.402765e-12 | 1200 | 119/140 | 196/150/560 | 15.30 | 50.14 | inside |
| T | 1.960 | 1.960 | 0.50 | 1.493514e-12 | 1931 | 85/100 | 140/150/400 | 15.30 | 49.39 | inside |


Per pattern and axis the FDTD arms are: B references (1d); singles — S:
L (1.96 | 1.4 | [1.0] x 294); P1: L, R ([1.0] x 274 | 1.4, 1.96 | [2.744]
x 107); P2: L ([2.744] x 100 | 1.96, 1.4 | [1.0] x 294), R ([1.0] x 274 |
1.4 | [1.96] x 150); P3: L, R ([1.0] x 274 | [1.4] x 210); P4: L ([1.4] x
196 | [1.0] x 294), R (= P2.R); T: L1 (S.L with the dt pin `1.0 x 293, 1.0,
0.7937, 0.63, 0.5 x 4` on z), L2 (1.96 x 140 | 1.4914, 1.1349, 0.8635,
0.6571 | [0.5] x 588); band arms — S, P1-P4 at n_b in {2, 4, 8, 16, 32};
T at (n_c, n_b2) in {4, 16} x {4, 8} with band 1 fixed at 4 cells
(`1.96 x 140 | 1.4 | 1.0 x 4 | 1.4 | 1.96 x n_c | 1.4914, 1.1349, 0.8635,
0.6571 | 0.5 x n_b2 | 0.6571, 0.8635, 1.1349, 1.4914 | 1.96 x 150`). The
chain-only mirrors (never run): S.R, T.R1 ([1.0] x 274 | 1.4 | [1.96] x
150 at dt_T), T.R2 ([0.5] x 549 | 0.6571 .. 1.4914 | [1.96] x 150),
evaluated unpinned (a 0.5 mm lead cannot host the coarser 1.0 mm pin).

## 3. Falsifiers and windows (frozen; program 4.4 verbatim)

- **W1 — per-arm amplitude, every band and single arm, every axis:**
  `|R_meas - R_model| <= 0.20 R_model + 3e-5`, `R_meas = |DFT_F0(A - B,
  [0, gate_end])| / |DFT_F0(B, [0, t_inc_end])|` (E1 form), `R_model` the
  chain model on the structure slice of the builder's vector at the run's
  dt. Floor reading rule (E1 1f): a fired row with `R_model < 2.5e-4` and
  `|R_meas - R_model| <= 1.5e-4` is recorded "fired — at the instrument's
  absolute floor". No modelled row is under 2.5e-4 (1h).
  *Reviewer note (second pass, 2026-09-14; window unchanged):* the
  rectangular DFT windows make `R_meas` depend on where the two window
  ends fall. The incident pulse (sigma_t = 64 ps, sigma_f = 2.5 GHz)
  reaches down to the discrete TE10 cutoff of this grid (5.0 GHz, 2
  sigma_f below F0), whose near-cutoff energy arrives late and rings;
  the 8-sigma incident end truncates that ringing and leaks it into the
  10 GHz bin. Measured by the review's scan on five recorded arms and
  reproduced on this tree before this edit (five CPU reruns, each
  reproducing the stored `R_meas` to the float64 bit); re-recorded
  through the instrument's own `--window-scan` as the diagnostic
  declared in the "Second pass" section (`results/e5_window_scan.json`):
  over every window end the declared rules would allow, `|DFT_F0(B)|`
  moves 3.3 % peak to peak (beat = the cutoff) and `R_meas` moves -2.0 ..
  +2.2 % (S n_b = 4) to -3.3 .. +4.5 % (x P1 n_b = 32) of the chain. The 20 % window absorbs this by 5-10 x
  (the closest arm sits at 0.26 of its half-window), the 8-sigma rule is
  E1's declared form, not tuned; but the instrument RESOLVES about 3 %,
  so sub-3 % differences between arms in the Results section (pinned vs
  unpinned, P1 vs P2, the 0.00-1.09 % single-transition agreement) are
  readings of the window rule as much as of the physics. Verdicts
  unaffected.
- **W2 — the bound:** every band arm `R_meas <= 1.05 x (R_L + R_R)` for
  S, P1-P4 (S: `2 R_single x 1.05`, E1 (ii)), and `R_meas <= 1.05 x (R_1L
  + R_1R + R_2L + R_2R)` for T. Numbers: 1.2160e-2 (S), 1.9163e-2
  (P1/P2), 8.1784e-3 (P3/P4), 2.6116e-2 (T).
- **W3 — phase:** per (pattern, axis) cell for S, P1-P4, `|c_meas -
  c_model| <= 0.10 x DR = 0.140 mm`, `c_meas` from `fit_c_two_amp` over
  the five measured rows with `R_L`, `R_R`, `k_g` fixed to the model
  (search [0, 3 DR], scan + two refinements at 3 DR / 4000 / 4000^2).
  `c_model` = 1.873 (S), 3.294 (P1, P2), 0.937 mm (P3, P4). Hypothesis
  reported, not gated: the `(local cell)^2` centroid sum 1.854 / 3.228 /
  0.927 mm. T has no c window (W1 + W2 per arm).
- **W4 — control:** L1-0 pattern S on z: the six `R_meas` equal E1's
  N30_r1.4 values (single 5.730199918373753e-3; widths 7.436432071888371e-3
  / 1.006271249900852e-2 / 1.1164199503515876e-2 / 1.2559000028761838e-3
  / 1.4248078745109632e-3) to <= 1e-6 relative, each `R_model` to <= 1e-8
  relative; bit-identity reported as a number.
- **W5 — relabel identity and pin bridge:** pattern S, n_b = 4, x and y
  vs z: `|R_meas_axis - R_meas_z| / R_meas_z <= 1e-6` and `max_t
  |trace_axis - trace_z| / max_t |trace_z| <= 1e-6` (A traces; the B_sym
  traces reported the same way). Pin bridge on z: pinned S vs unpinned
  L1-0 at every width, `|R_meas_pinned - R_meas_unpinned| <= 1.5e-4`
  absolute. Reading rule: if W5 fires on an axis, that axis's arms still
  run and are gated by W1-W3 on their own model; the cell verdict carries
  "relabel identity fired: instrument class"; the fire is not tuned.
  *Reviewer note (second pass, 2026-09-14; window unchanged):* the
  `R_meas` sub-criterion at 1e-6 is a bit-identity test in disguise.
  `R_meas` is a ~1 % differenced quantity (`A - B` over `B`), so a
  trace-level difference of 7e-7 maps to up to 7e-5 on `R_meas`; the
  1e-6 `R_meas` window can only be met when the traces are bit-identical,
  which is why it fired on y while both trace criteria (also 1e-6) held.
  The stored y-vs-z difference is deterministic (y re-run twice is
  bit-identical to itself and to the stored trace), 1.8 float32 eps RMS
  against the running maximum of the trace, non-growing over 1200 steps
  (7.7e-7 in the first 200-step block, 1.1e-6 in the last). The fire is
  a defect of the sub-criterion's design, not of the kernel; it stays a
  fire (declared window), and the reading rule handled it.
- **W6 — gates:** the six margins (1f) of every arm are positive in the
  frozen model JSON (Table G: smallest 0.2560 ns, the incident-gate
  construction, on every arm; inner-return margin on T's longest
  structure n_c = 16 / n_b2 = 8: 0.397 ns z, 0.416 ns x/y — the program
  expected ~0.42 ns). `gates_hold` is measured per arm (geometry AND
  `dt_matches_b` AND `source_probe_in_lead` AND `lead_f32_identical`).
  *Reviewer note (second pass, 2026-09-14; window unchanged):* the
  0.2560 ns "smallest margin" is 4 sigma_t by construction, not a
  measurement: `incident_inside = t_inc_end - (t_inc_arr + 4 sigma)`
  with `t_inc_end = min(t_inc_arr + 8 sigma, t_echo - 4 sigma)`, and the
  8-sigma term binds on every arm here (0.947 ns against 1.478 ns), so
  the margin is exactly 4 x 64 ps on all 120 arms; it could only fail if
  the source-wall echo term bound, which the geometry (z_src >= 67 mm)
  rules out. The other five margins are real: reflection_inside
  0.776-0.815 ns, inner_return_inside 0.397 ns (T n_c = 16 / n_b2 = 8,
  z) at the tightest, direct B far-end return 2.89-2.98 ns. The lane's
  tightest measured gate is the 0.397 ns inner return, not 0.256 ns.
- **Cell verdicts (before the run):** "inside the law domain" = every W1
  held (singles and widths), W2 held at every width, and (S, P1-P4) W3
  held; "inside the -54 dB class" by E1's rule verbatim (every adjacent
  ratio <= 1.4 on the builder output, fine band at the nominal 30 cells
  per lambda0) — every pattern is inside by construction, the coarsest
  runway's cells per lambda0 reported beside it.

Expectations (program 4.4, kept; 1h adjusts one): S inside on z (control),
inside on x/y if W5 holds; P1, P2 inside on all axes (the chain gives
them the same numbers, so the program's "W3 on P2" fire is not expected
from the model — if it fires it is an FDTD/direction effect the chain
does not contain); P3, P4 inside; T inside, the n_c = 16 / n_b2 = 8 arm
the likeliest W1 fire (tightest inner-return margin). Out of scope
(G19): coarse runways under 3.75 cells per lambda0, ratios above 2.0,
frequencies other than 10 GHz, bands under 2 cells, dielectric or lossy
interiors, any absorber.

## 4. Run commands (declared) and output

From the worktree, `PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-full
/Users/byungkwankim/Documents/rfx/.venv/bin/python -m validation.research.
multiband_nu.e5_multilevel_axes ...`, in this order, each one Bash call:

1. `--model-only --out results/e5_model.json` (this commit; zero FDTD).
2. `--axis z --patterns S --out results/e5_z.json` (L1-0; writes the W4
   control against `results/e1_band_law_sweep.json`).
3. `--relabel --z-json results/e5_z.json --out results/e5_relabel.json`
   (L1-R).
4. `--axis z --patterns P1,P2,P3,P4,T --resume --out results/e5_z.json`
   (L1-Z).
5. `--axis x --out results/e5_x.json` (L1-X).
6. `--pin-bridge --z-json results/e5_z.json --out results/e5_pinbridge.json`.
7. `--axis y --out results/e5_y.json` (L1-Y).

Replay: `tests/unit/nonuniform/test_e5_multilevel_replay.py` (chain model
only) re-derives every `R_model`, window, gate, W1/W2/W3 verdict, the W4
and W5 comparisons and the cell verdicts from the JSONs at 1e-8 (chain
floats) / 1e-9 m (c re-fits) / 1e-9 relative (gate times) and pins the
recorded verdicts, fired ones included. Regression: `test_e1_band_law_
replay.py` and `test_band_builder_chain_model.py` pass before (56) and
after.

## 5. Chain-model predictions per arm (frozen; from `results/e5_model.json`)

### Table L — law side per pattern (chain model, frozen; identical on x, y, z)

| pattern | R_L | R_R | R_L + R_R | bound 1.05 (R_L + R_R) | k_g (1/mm) | c_model (mm) | c closed form (mm) | closed - model (mm) | c window +/- (mm) | chain max / (R_L+R_R) | chain min (0..80) | |R_L - R_R| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | 5.7907e-03 | 5.7907e-03 | 1.1581e-02 | 1.2160e-02 | 0.18163 | 1.873 | 1.854 | -0.019 | 0.140 | 0.9999 | 3.6061e-05 | 9.8446e-16 |
| P1 | 5.7907e-03 | 1.2460e-02 | 1.8250e-02 | 1.9163e-02 | 0.18163 | 3.294 | 3.228 | -0.066 | 0.140 | 0.9999 | 6.6695e-03 | 6.6690e-03 |
| P2 | 1.2460e-02 | 5.7907e-03 | 1.8250e-02 | 1.9163e-02 | 0.18163 | 3.294 | 3.228 | -0.066 | 0.140 | 0.9999 | 6.6695e-03 | 6.6690e-03 |
| P3 | 5.7907e-03 | 1.9982e-03 | 7.7889e-03 | 8.1784e-03 | 0.18163 | 0.937 | 0.927 | -0.010 | 0.140 | 1.0000 | 3.7929e-03 | 3.7924e-03 |
| P4 | 1.9982e-03 | 5.7907e-03 | 7.7889e-03 | 8.1784e-03 | 0.18163 | 0.937 | 0.927 | -0.010 | 0.140 | 1.0000 | 3.7929e-03 | 3.7924e-03 |

Pattern T amplitudes (chain, dt = 1.493514e-12 s): R_1L = 5.7996e-03, R_1R = 5.7996e-03, R_2L = 6.6366e-03, R_2R = 6.6366e-03; sum = 2.4872e-02; bound 1.05 x sum = 2.6116e-02.

### Table R — per-arm chain-model predictions and frozen W1 windows (z; the same R_model on x and y)

| pattern | arm | cells (z) | builder exact | R_model | dB | window (frozen) | W2 bound | fp form (two-amp) | min gate margin z / x / y (ns) |
|---|---|---|---|---|---|---|---|---|---|
| S | single L | 435 | True | 5.7907e-03 | -44.7 | [4.6026e-03, 6.9788e-03] | — | — | 0.256 / 0.256 / 0.256 |
| S | S_nb2 | 294 | True | 7.4916e-03 | -42.5 | [5.9633e-03, 9.0199e-03] | 1.2160e-02 | 7.4915e-03 | 0.256 / 0.256 / 0.256 |
| S | S_nb4 | 296 | True | 1.0141e-02 | -39.9 | [8.0826e-03, 1.2199e-02] | 1.2160e-02 | 1.0141e-02 | 0.256 / 0.256 / 0.256 |
| S | S_nb8 | 300 | True | 1.1296e-02 | -38.9 | [9.0066e-03, 1.3585e-02] | 1.2160e-02 | 1.1296e-02 | 0.256 / 0.256 / 0.256 |
| S | S_nb16 | 308 | True | 1.2101e-03 | -58.3 | [9.3810e-04, 1.4822e-03] | 1.2160e-02 | 1.2101e-03 | 0.256 / 0.256 / 0.256 |
| S | S_nb32 | 324 | True | 1.5111e-03 | -56.4 | [1.1789e-03, 1.8434e-03] | 1.2160e-02 | 1.5111e-03 | 0.256 / 0.256 / 0.256 |
| P1 | single L | 435 | True | 5.7907e-03 | -44.7 | [4.6026e-03, 6.9788e-03] | — | — | 0.256 / 0.256 / 0.256 |
| P1 | single R | 383 | True | 1.2460e-02 | -38.1 | [9.9378e-03, 1.4982e-02] | — | — | 0.256 / 0.256 / 0.256 |
| P1 | P1_nb2 | 252 | True | 1.5446e-02 | -36.2 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 | 1.5446e-02 | 0.256 / 0.256 / 0.256 |
| P1 | P1_nb4 | 254 | True | 1.7774e-02 | -35.0 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 | 1.7775e-02 | 0.256 / 0.256 / 0.256 |
| P1 | P1_nb8 | 258 | True | 1.6474e-02 | -35.7 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 | 1.6474e-02 | 0.256 / 0.256 / 0.256 |
| P1 | P1_nb16 | 266 | True | 8.9908e-03 | -40.9 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 | 8.9903e-03 | 0.256 / 0.256 / 0.256 |
| P1 | P1_nb32 | 282 | True | 7.0095e-03 | -43.1 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 | 7.0090e-03 | 0.256 / 0.256 / 0.256 |
| P2 | single L | 396 | True | 1.2460e-02 | -38.1 | [9.9378e-03, 1.4982e-02] | — | — | 0.256 / 0.256 / 0.256 |
| P2 | single R | 425 | True | 5.7907e-03 | -44.7 | [4.6026e-03, 6.9788e-03] | — | — | 0.256 / 0.256 / 0.256 |
| P2 | P2_nb2 | 255 | True | 1.5446e-02 | -36.2 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 | 1.5446e-02 | 0.256 / 0.256 / 0.256 |
| P2 | P2_nb4 | 257 | True | 1.7774e-02 | -35.0 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 | 1.7775e-02 | 0.256 / 0.256 / 0.256 |
| P2 | P2_nb8 | 261 | True | 1.6474e-02 | -35.7 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 | 1.6474e-02 | 0.256 / 0.256 / 0.256 |
| P2 | P2_nb16 | 269 | True | 8.9908e-03 | -40.9 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 | 8.9903e-03 | 0.256 / 0.256 / 0.256 |
| P2 | P2_nb32 | 285 | True | 7.0095e-03 | -43.1 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 | 7.0090e-03 | 0.256 / 0.256 / 0.256 |
| P3 | single L | 435 | True | 5.7907e-03 | -44.7 | [4.6026e-03, 6.9788e-03] | — | — | 0.256 / 0.256 / 0.256 |
| P3 | single R | 484 | True | 1.9982e-03 | -54.0 | [1.5686e-03, 2.4279e-03] | — | — | 0.256 / 0.256 / 0.256 |
| P3 | P3_nb2 | 353 | True | 5.1330e-03 | -45.8 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 | 5.1330e-03 | 0.256 / 0.256 / 0.256 |
| P3 | P3_nb4 | 355 | True | 6.5292e-03 | -43.7 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 | 6.5292e-03 | 0.256 / 0.256 / 0.256 |
| P3 | P3_nb8 | 359 | True | 7.7807e-03 | -42.2 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 | 7.7808e-03 | 0.256 / 0.256 / 0.256 |
| P3 | P3_nb16 | 367 | True | 3.8185e-03 | -48.4 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 | 3.8184e-03 | 0.256 / 0.256 / 0.256 |
| P3 | P3_nb32 | 383 | True | 4.2953e-03 | -47.3 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 | 4.2953e-03 | 0.256 / 0.256 / 0.256 |
| P4 | single L | 490 | True | 1.9982e-03 | -54.0 | [1.5686e-03, 2.4279e-03] | — | — | 0.256 / 0.256 / 0.256 |
| P4 | single R | 425 | True | 5.7907e-03 | -44.7 | [4.6026e-03, 6.9788e-03] | — | — | 0.256 / 0.256 / 0.256 |
| P4 | P4_nb2 | 349 | True | 5.1330e-03 | -45.8 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 | 5.1330e-03 | 0.256 / 0.256 / 0.256 |
| P4 | P4_nb4 | 351 | True | 6.5292e-03 | -43.7 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 | 6.5292e-03 | 0.256 / 0.256 / 0.256 |
| P4 | P4_nb8 | 355 | True | 7.7807e-03 | -42.2 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 | 7.7808e-03 | 0.256 / 0.256 / 0.256 |
| P4 | P4_nb16 | 363 | True | 3.8185e-03 | -48.4 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 | 3.8184e-03 | 0.256 / 0.256 / 0.256 |
| P4 | P4_nb32 | 379 | True | 4.2953e-03 | -47.3 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 | 4.2953e-03 | 0.256 / 0.256 / 0.256 |
| T | single L1 | 441 | True | 5.7996e-03 | -44.7 | [4.6097e-03, 6.9895e-03] | — | — | 0.256 / 0.256 / 0.256 |
| T | single L2 | 732 | True | 6.6366e-03 | -43.6 | [5.2793e-03, 7.9939e-03] | — | — | 0.256 / 0.256 / 0.256 |
| T | T_c4_b4 | 312 | True | 2.2989e-02 | -32.8 | [1.8361e-02, 2.7617e-02] | 2.6116e-02 | — | 0.256 / 0.256 / 0.256 |
| T | T_c4_b8 | 316 | True | 2.3139e-02 | -32.7 | [1.8482e-02, 2.7797e-02] | 2.6116e-02 | — | 0.256 / 0.256 / 0.256 |
| T | T_c16_b4 | 324 | True | 1.2933e-02 | -37.8 | [1.0317e-02, 1.5550e-02] | 2.6116e-02 | — | 0.256 / 0.256 / 0.256 |
| T | T_c16_b8 | 328 | True | 9.2903e-03 | -40.6 | [7.4022e-03, 1.1178e-02] | 2.6116e-02 | — | 0.256 / 0.256 / 0.256 |

### Table G — gate margins (ns) per arm and axis, the five E1 margins plus the direct B far-end return (all must be > 0 before the run)

| axis | pattern | arm | reflection_inside | inner_return_inside | run_covers_gate | b_pin_return_after_gate | incident_inside | b_far_return_direct_after_gate | gates_hold |
|---|---|---|---|---|---|---|---|---|---|
| z | S | L | 0.788 | 0.777 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | S | S_nb2 | 0.788 | 0.751 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | S | S_nb4 | 0.788 | 0.736 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | S | S_nb8 | 0.788 | 0.705 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | S | S_nb16 | 0.788 | 0.643 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | S | S_nb32 | 0.788 | 0.520 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P1 | L | 0.788 | 0.777 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P1 | R | 0.776 | 0.750 | 0.818 | 4.189 | 0.256 | 2.901 | True |
| z | P1 | P1_nb2 | 0.788 | 0.736 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P1 | P1_nb4 | 0.788 | 0.720 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P1 | P1_nb8 | 0.788 | 0.689 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P1 | P1_nb16 | 0.788 | 0.628 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P1 | P1_nb32 | 0.788 | 0.504 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P2 | L | 0.815 | 0.789 | 0.752 | 4.303 | 0.256 | 2.976 | True |
| z | P2 | R | 0.776 | 0.765 | 0.818 | 4.189 | 0.256 | 2.901 | True |
| z | P2 | P2_nb2 | 0.815 | 0.763 | 0.752 | 4.303 | 0.256 | 2.976 | True |
| z | P2 | P2_nb4 | 0.815 | 0.747 | 0.752 | 4.303 | 0.256 | 2.976 | True |
| z | P2 | P2_nb8 | 0.815 | 0.717 | 0.752 | 4.303 | 0.256 | 2.976 | True |
| z | P2 | P2_nb16 | 0.815 | 0.655 | 0.752 | 4.303 | 0.256 | 2.976 | True |
| z | P2 | P2_nb32 | 0.815 | 0.531 | 0.752 | 4.303 | 0.256 | 2.976 | True |
| z | P3 | L | 0.788 | 0.777 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P3 | R | 0.776 | 0.776 | 0.818 | 4.189 | 0.256 | 2.901 | True |
| z | P3 | P3_nb2 | 0.788 | 0.762 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P3 | P3_nb4 | 0.788 | 0.746 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P3 | P3_nb8 | 0.788 | 0.716 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P3 | P3_nb16 | 0.788 | 0.654 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P3 | P3_nb32 | 0.788 | 0.530 | 0.793 | 4.233 | 0.256 | 2.933 | True |
| z | P4 | L | 0.778 | 0.778 | 0.809 | 4.201 | 0.256 | 2.912 | True |
| z | P4 | R | 0.776 | 0.765 | 0.818 | 4.189 | 0.256 | 2.901 | True |
| z | P4 | P4_nb2 | 0.778 | 0.752 | 0.809 | 4.201 | 0.256 | 2.912 | True |
| z | P4 | P4_nb4 | 0.778 | 0.736 | 0.809 | 4.201 | 0.256 | 2.912 | True |
| z | P4 | P4_nb8 | 0.778 | 0.705 | 0.809 | 4.201 | 0.256 | 2.912 | True |
| z | P4 | P4_nb16 | 0.778 | 0.644 | 0.809 | 4.201 | 0.256 | 2.912 | True |
| z | P4 | P4_nb32 | 0.778 | 0.520 | 0.809 | 4.201 | 0.256 | 2.912 | True |
| z | T | L1 | 0.790 | 0.779 | 0.790 | 4.239 | 0.256 | 2.937 | True |
| z | T | L2 | 0.790 | 0.758 | 0.790 | 4.239 | 0.256 | 2.937 | True |
| z | T | T_c4_b4 | 0.790 | 0.597 | 0.790 | 4.239 | 0.256 | 2.937 | True |
| z | T | T_c4_b8 | 0.790 | 0.581 | 0.790 | 4.239 | 0.256 | 2.937 | True |
| z | T | T_c16_b4 | 0.790 | 0.413 | 0.790 | 4.239 | 0.256 | 2.937 | True |
| z | T | T_c16_b8 | 0.790 | 0.397 | 0.790 | 4.239 | 0.256 | 2.937 | True |
| x | S | L | 0.807 | 0.796 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | S | S_nb2 | 0.807 | 0.770 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | S | S_nb4 | 0.807 | 0.754 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | S | S_nb8 | 0.807 | 0.723 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | S | S_nb16 | 0.807 | 0.662 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | S | S_nb32 | 0.807 | 0.538 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P1 | L | 0.807 | 0.796 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P1 | R | 0.783 | 0.757 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| x | P1 | P1_nb2 | 0.807 | 0.754 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P1 | P1_nb4 | 0.807 | 0.739 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P1 | P1_nb8 | 0.807 | 0.708 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P1 | P1_nb16 | 0.807 | 0.646 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P1 | P1_nb32 | 0.807 | 0.523 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P2 | L | 0.849 | 0.823 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| x | P2 | R | 0.783 | 0.773 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| x | P2 | P2_nb2 | 0.849 | 0.797 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| x | P2 | P2_nb4 | 0.849 | 0.781 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| x | P2 | P2_nb8 | 0.849 | 0.750 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| x | P2 | P2_nb16 | 0.849 | 0.689 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| x | P2 | P2_nb32 | 0.849 | 0.565 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| x | P3 | L | 0.807 | 0.796 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P3 | R | 0.783 | 0.783 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| x | P3 | P3_nb2 | 0.807 | 0.780 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P3 | P3_nb4 | 0.807 | 0.765 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P3 | P3_nb8 | 0.807 | 0.734 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P3 | P3_nb16 | 0.807 | 0.672 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P3 | P3_nb32 | 0.807 | 0.549 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| x | P4 | L | 0.786 | 0.786 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| x | P4 | R | 0.783 | 0.773 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| x | P4 | P4_nb2 | 0.786 | 0.759 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| x | P4 | P4_nb4 | 0.786 | 0.744 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| x | P4 | P4_nb8 | 0.786 | 0.713 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| x | P4 | P4_nb16 | 0.786 | 0.651 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| x | P4 | P4_nb32 | 0.786 | 0.528 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| x | T | L1 | 0.809 | 0.798 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| x | T | L2 | 0.809 | 0.777 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| x | T | T_c4_b4 | 0.809 | 0.615 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| x | T | T_c4_b8 | 0.809 | 0.600 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| x | T | T_c16_b4 | 0.809 | 0.431 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| x | T | T_c16_b8 | 0.809 | 0.416 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| y | S | L | 0.807 | 0.796 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | S | S_nb2 | 0.807 | 0.770 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | S | S_nb4 | 0.807 | 0.754 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | S | S_nb8 | 0.807 | 0.723 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | S | S_nb16 | 0.807 | 0.662 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | S | S_nb32 | 0.807 | 0.538 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P1 | L | 0.807 | 0.796 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P1 | R | 0.783 | 0.757 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| y | P1 | P1_nb2 | 0.807 | 0.754 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P1 | P1_nb4 | 0.807 | 0.739 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P1 | P1_nb8 | 0.807 | 0.708 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P1 | P1_nb16 | 0.807 | 0.646 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P1 | P1_nb32 | 0.807 | 0.523 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P2 | L | 0.849 | 0.823 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| y | P2 | R | 0.783 | 0.773 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| y | P2 | P2_nb2 | 0.849 | 0.797 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| y | P2 | P2_nb4 | 0.849 | 0.781 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| y | P2 | P2_nb8 | 0.849 | 0.750 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| y | P2 | P2_nb16 | 0.849 | 0.689 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| y | P2 | P2_nb32 | 0.849 | 0.565 | 0.719 | 4.269 | 0.256 | 2.942 | True |
| y | P3 | L | 0.807 | 0.796 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P3 | R | 0.783 | 0.783 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| y | P3 | P3_nb2 | 0.807 | 0.780 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P3 | P3_nb4 | 0.807 | 0.765 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P3 | P3_nb8 | 0.807 | 0.734 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P3 | P3_nb16 | 0.807 | 0.672 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P3 | P3_nb32 | 0.807 | 0.549 | 0.774 | 4.214 | 0.256 | 2.914 | True |
| y | P4 | L | 0.786 | 0.786 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| y | P4 | R | 0.783 | 0.773 | 0.811 | 4.181 | 0.256 | 2.893 | True |
| y | P4 | P4_nb2 | 0.786 | 0.759 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| y | P4 | P4_nb4 | 0.786 | 0.744 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| y | P4 | P4_nb8 | 0.786 | 0.713 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| y | P4 | P4_nb16 | 0.786 | 0.651 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| y | P4 | P4_nb32 | 0.786 | 0.528 | 0.801 | 4.194 | 0.256 | 2.904 | True |
| y | T | L1 | 0.809 | 0.798 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| y | T | L2 | 0.809 | 0.777 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| y | T | T_c4_b4 | 0.809 | 0.615 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| y | T | T_c4_b8 | 0.809 | 0.600 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| y | T | T_c16_b4 | 0.809 | 0.431 | 0.772 | 4.220 | 0.256 | 2.918 | True |
| y | T | T_c16_b8 | 0.809 | 0.416 | 0.772 | 4.220 | 0.256 | 2.918 | True |

Smallest margin over every arm and axis: 0.2560 ns (expectation >= 0.256 ns).


## Results (appended after measurement; no window above changed)

### Measured (instrument `be606f45`, tests `ea208024`, measurements `b4646528`; no window above changed)

Every JSON records `rfx_file` under `/Users/byungkwankim/Documents/rfx-nu-full/rfx/__init__.py`
and `git_sha ea208024` (the tests commit, the last before any FDTD).
`git_dirty` is true in every measurement JSON: the flag is `git status
--porcelain` at the START of the call, and the results JSON the previous
sub-lane had just written was still untracked at that moment. The L1-0
call's own provenance block was overwritten by the resumed L1-Z call
(only its `git_sha`, `started_utc 15:55:38Z` and `argv` survive in
`resumed_from`; its `git_dirty` and `wallclock_s` are not stored
anywhere). What the record does support: `ea208024` was committed at
15:55:28Z and the first FDTD started at 15:55:38Z; the instrument file
is byte-identical between `be606f45` and `b4646528`; nothing under
`rfx/`, `validation/` code or `tests/` was modified between `ea208024`
and the measurements — `git diff --stat` on those paths across the
lane's commits is empty for `rfx/` and the five reused files. (The
first version of this paragraph also said the tree was clean at
`ea208024` and nothing ran in between; that was read from the terminal
and a `git status` that are not stored — second-pass correction.)
Sub-lanes ran in the declared order (L1-0, L1-R, L1-Z, L1-X, pin bridge,
L1-Y), one attempt each. FDTD runs executed: z 7 (L1-0) + 37 (L1-Z, a new
process, so the S single and `B_1.96` ran again) = 44, x 42, y 42, L1-R 6,
pin bridge 7 — 141 runs (the `b4646528` commit message says 139, the
declared count of 1j; the two extra are the L1-Z re-runs). Stored
wallclock: z 21.1 s (resumed call only), x 30.6 s, y 43.0 s, relabel
3.5 s, pin bridge 4.6 s = 102.8 s; the L1-0 call's wallclock was
overwritten (the "107 s" first written here was the terminal total,
second-pass correction). The L1-Z re-run of the S single reproduced the
L1-0 `R_meas` 5.730199918373753e-3 to the float64 bit in a new process
(`P1.L`, `fdtd_run_cached` false) — a determinism datum.

One instrument defect, found by the replay test after the runs and
recorded: the L1-Z `--resume` call rebuilt `e5_z.json` from the previous
file's `cells` only and dropped the top-level `w4_control` block that
L1-0 had written. The block is a pure function of the stored S cell and
E1's JSON (`w4_control()`, no FDTD); it was regenerated by `--refresh-w4`
and the JSON carries `w4_control_recomputed_from_stored` with the
reason. Provenance of that step (second-pass correction): the block is
stamped `git_sha b4646528`, but the `--refresh-w4` path it ran through
was uncommitted instrument code at that moment (16:10:11Z; it was
committed in `3622cf01` at 16:13:31Z), i.e. a clean sha for a dirty
tree — the stamp had no dirty flag. `w4_control()` itself is unchanged
since `be606f45` (the post-measurement diff of the instrument is
additive: a pathlib import, the resume carry-forward, the two printers
and the argparse branches), and `test_results_w4_control_against_e1`
re-derives the block from the stored S cell and E1's JSON (six rows,
`meas_rel` 0.0, `model_rel` 1.7e-13 .. 6.3e-12). The block now carries
`git_dirty_at_refresh: true` (annotation added in the second pass) and
the `--refresh-w4` stamp records `git_dirty` from now on. The regenerated
rows are the ones the replay test re-derives; the sentence first written
here, that they "equal the L1-0 printout", refers to terminal output
that is not stored. The resume path now carries the block forward.

### Regression (W4) — HELD, bit-identical

L1-0 control against E1 (`results/e1_band_law_sweep.json` N30_r1.4):

| arm | R_meas (this lane) | R_meas (E1) | relative difference | bit-identical | R_model relative difference | verdict |
|---|---|---|---|---|---|---|
| single | 5.7301999183737530e-03 | 5.7301999183737530e-03 | 0.0e+00 | True | 4.42e-13 | HELD |
| n_b=2 | 7.4364320718883709e-03 | 7.4364320718883709e-03 | 0.0e+00 | True | 6.81e-13 | HELD |
| n_b=4 | 1.0062712499008520e-02 | 1.0062712499008520e-02 | 0.0e+00 | True | 1.72e-13 | HELD |
| n_b=8 | 1.1164199503515876e-02 | 1.1164199503515876e-02 | 0.0e+00 | True | 7.09e-13 | HELD |
| n_b=16 | 1.2559000028761838e-03 | 1.2559000028761838e-03 | 0.0e+00 | True | 6.32e-12 | HELD |
| n_b=32 | 1.4248078745109632e-03 | 1.4248078745109632e-03 | 0.0e+00 | True | 2.26e-12 | HELD |

`c_model` difference 2.62e-13 m, `c_meas` difference 3.41e-12 m (E1 JSON git_sha ec008ead). Windows 1e-6 (R_meas) / 1e-8 (R_model) relative.

The L1-0 z control through the new module reproduces E1's six `R_meas`
with relative difference 0.0 (float64 bit-identity), `R_model` within
1.7e-13 .. 6.3e-12 (dense-solve noise, window 1e-8), the c fits within
3.4e-12 m. The five E1 gate margins of the measured arms equal E1's
measured-arm rows; the sixth (direct far-end return) is 2.933 ns.
`test_e1_band_law_replay.py` + `test_band_builder_chain_model.py`: 56
passed before and after.

### W5 — relabel identity: x HELD at bit identity, y FIRED (instrument class)

Pattern S, n_b = 4, unpinned, B_sym on every axis:

| axis | comp | grid | dt (s) | R_meas | R_model | A trace rel max diff vs z | A bit-identical | B_sym trace rel max diff | B bit-identical | R_meas rel diff vs z | W5 (1e-6) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| z | ex | [4, 21, 297] | 2.402765e-12 | 1.006271e-02 | 1.014071e-02 | — | — | — | — | — | reference |
| x | ey | [297, 4, 21] | 2.402765e-12 | 1.006271e-02 | 1.014071e-02 | 0.000e+00 | True | 0.000e+00 | True | 0.000e+00 | HELD |
| y | ez | [21, 297, 4] | 2.402765e-12 | 1.006265e-02 | 1.014071e-02 | 6.710e-07 | False | 6.852e-07 | False | 6.018e-06 | **FIRED** |

z bridge: `R_meas` with E1's B 1.0062712499e-02 vs with B_sym 1.0062712499e-02 (relative 0.0e+00); the z A arm re-run in the relabel call bit-identical to L1-0: True; E1-B vs B_sym traces before the gate, relative max difference 0.0e+00.

Reading. z -> x is an exact relabeling of the closed-box kernel: the A
and B_sym traces on x are bit-identical to z's over all 1200 steps and
`R_meas` is the same float64. z -> y is NOT bit-identical: the traces
differ by 6.71e-7 (A) and 6.85e-7 (B) of their maximum — inside the
1e-6 trace window — and `R_meas`, a 1 % differenced quantity, differs by
6.018e-6 relative, above the 1e-6 window: **W5 FIRED on y by the
`R_meas` criterion**, with both trace criteria held. The number is the
float32 summation-order class (1200 steps of float32 leapfrog; a single
ulp is 6e-8), not a physics difference: the y arms below agree with the
x arms to <= 4.1e-5 relative on every one of the 40 arm pairs (4.08e-5
largest, median 3.4e-6), and their W1-W3 verdicts are identical. Per the declared
reading rule the y arms ran and are gated by W1-W3 on their own model;
every y cell carries the flag "relabel identity FIRED: instrument
class". The window is not widened; the program's section-2 statement
that the kernels are exact cyclic relabelings is measured true for
z -> x and false at the 7e-7 level for z -> y (an asymmetry in
summation order somewhere on the two-step relabeling — not located in
this lane, no `rfx/` change; a candidate item for the program's gap map).
z bridge: `B_sym` is an exact substitute for E1's B (`R_meas` identical
to the float64 bit; B traces identical before the gate), and the z A arm
re-run in the relabel process is bit-identical to L1-0.

### W5 — pin bridge on z: HELD at every width

Pattern S pinned as on x/y vs the unpinned L1-0 arm (window 1.5e-4 absolute):

| n_b | R_meas pinned | R_meas unpinned | abs diff | rel diff | R_model rel diff | verdict |
|---|---|---|---|---|---|---|
| 2 | 7.474019e-03 | 7.436432e-03 | 3.759e-05 | 5.054e-03 | 3.5e-13 | HELD |
| 4 | 1.009260e-02 | 1.006271e-02 | 2.989e-05 | 2.970e-03 | 5.3e-15 | HELD |
| 8 | 1.120660e-02 | 1.116420e-02 | 4.240e-05 | 3.798e-03 | 1.2e-14 | HELD |
| 16 | 1.233882e-03 | 1.255900e-03 | 2.202e-05 | 1.753e-02 | 2.2e-12 | HELD |
| 32 | 1.460402e-03 | 1.424808e-03 | 3.559e-05 | 2.498e-02 | 2.8e-12 | HELD |

Pinned single on z: R_meas 5.770686e-03 (model 5.790690e-03, dev 0.35 %); pinned c_meas 1.883 mm.

The pins move `R_meas` by 2.2e-5 .. 4.2e-5 absolute (0.3-2.5 %
relative), one seventh of the window at most, and — reported, not
gated — TOWARD the chain model in 36 of the 40 (pinned x vs unpinned z)
arm pairs: the pinned arms (x, y, pinned z) deviate 0.00-3.36 % from the
chain (median 0.49 %), the unpinned z arms 0.19-5.71 % (median 1.02 %);
the single ramp reads 5.7707e-3 pinned against 5.7302e-3 unpinned for a
chain value of 5.7907e-3. The pins are an instrumentation of the lo
wall (a 1.0 | 1.4 | 1.96 mm step pair the source-wall echo crosses
twice); the chain model has no wall, and both readings are inside the
W1 windows and inside E1's measured floor class. Recorded as a number;
not explained here. *Second-pass reading:* the pinned-vs-unpinned
difference (0.3-2.5 %) is under the instrument's window-end sensitivity
(about 3 %, W1 reviewer note and the "Second pass" section), so "36 of
40 closer" is a count, not evidence that the pins improve the physics.

### W1, W2, W3 on every (pattern, axis) cell — all HELD; 18 of 18 cells inside the law domain

Table A — per-arm results, every axis (window `|R_meas - R_model| <= 0.20 R_model + 3e-5`, frozen):

| axis | pattern | arm | cells | R_meas | dB | R_model | dev abs | dev rel (%) | window (frozen) | W2 bound | gates hold | dt_A - dt_B (s) | lead f32 identical | run reused | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | S | single L | 435 | 5.7302e-03 | -44.8 | 5.7907e-03 | 6.05e-05 | 1.04 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | False | HELD |
| z | S | S_nb2 | 294 | 7.4364e-03 | -42.6 | 7.4916e-03 | 5.51e-05 | 0.74 | [5.9633e-03, 9.0199e-03] | 1.2160e-02 held | True | 1.2e-27 | True | False | HELD |
| z | S | S_nb4 | 296 | 1.0063e-02 | -39.9 | 1.0141e-02 | 7.80e-05 | 0.77 | [8.0826e-03, 1.2199e-02] | 1.2160e-02 held | True | 1.2e-27 | True | False | HELD |
| z | S | S_nb8 | 300 | 1.1164e-02 | -39.0 | 1.1296e-02 | 1.32e-04 | 1.16 | [9.0066e-03, 1.3585e-02] | 1.2160e-02 held | True | 1.2e-27 | True | False | HELD |
| z | S | S_nb16 | 308 | 1.2559e-03 | -58.0 | 1.2101e-03 | 4.58e-05 | 3.78 | [9.3810e-04, 1.4822e-03] | 1.2160e-02 held | True | 1.2e-27 | True | False | HELD |
| z | S | S_nb32 | 324 | 1.4248e-03 | -56.9 | 1.5111e-03 | 8.63e-05 | 5.71 | [1.1789e-03, 1.8434e-03] | 1.2160e-02 held | True | 1.6e-27 | True | False | HELD |
| z | P1 | single L | 435 | 5.7302e-03 | -44.8 | 5.7907e-03 | 6.05e-05 | 1.04 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | False | HELD |
| z | P1 | single R | 383 | 1.2329e-02 | -38.2 | 1.2460e-02 | 1.31e-04 | 1.05 | [9.9378e-03, 1.4982e-02] | — | True | 0.0e+00 | True | False | HELD |
| z | P1 | P1_nb2 | 252 | 1.5313e-02 | -36.3 | 1.5446e-02 | 1.33e-04 | 0.86 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P1 | P1_nb4 | 254 | 1.7613e-02 | -35.1 | 1.7774e-02 | 1.62e-04 | 0.91 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P1 | P1_nb8 | 258 | 1.6257e-02 | -35.8 | 1.6474e-02 | 2.16e-04 | 1.31 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P1 | P1_nb16 | 266 | 8.9278e-03 | -41.0 | 8.9908e-03 | 6.30e-05 | 0.70 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P1 | P1_nb32 | 282 | 6.9614e-03 | -43.1 | 7.0095e-03 | 4.81e-05 | 0.69 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 held | True | 1.6e-27 | True | False | HELD |
| z | P2 | single L | 396 | 1.2324e-02 | -38.2 | 1.2460e-02 | 1.36e-04 | 1.09 | [9.9378e-03, 1.4982e-02] | — | True | 4.0e-28 | True | False | HELD |
| z | P2 | single R | 425 | 5.7352e-03 | -44.8 | 5.7907e-03 | 5.55e-05 | 0.96 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | False | HELD |
| z | P2 | P2_nb2 | 255 | 1.5315e-02 | -36.3 | 1.5446e-02 | 1.31e-04 | 0.85 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P2 | P2_nb4 | 257 | 1.7598e-02 | -35.1 | 1.7774e-02 | 1.76e-04 | 0.99 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P2 | P2_nb8 | 261 | 1.6261e-02 | -35.8 | 1.6474e-02 | 2.12e-04 | 1.29 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P2 | P2_nb16 | 269 | 8.9378e-03 | -41.0 | 8.9908e-03 | 5.30e-05 | 0.59 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P2 | P2_nb32 | 285 | 6.9570e-03 | -43.2 | 7.0095e-03 | 5.25e-05 | 0.75 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 held | True | 1.2e-27 | True | False | HELD |
| z | P3 | single L | 435 | 5.7302e-03 | -44.8 | 5.7907e-03 | 6.05e-05 | 1.04 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | True | HELD |
| z | P3 | single R | 484 | 1.9832e-03 | -54.1 | 1.9982e-03 | 1.51e-05 | 0.75 | [1.5686e-03, 2.4279e-03] | — | True | 0.0e+00 | True | False | HELD |
| z | P3 | P3_nb2 | 353 | 5.0848e-03 | -45.9 | 5.1330e-03 | 4.82e-05 | 0.94 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P3 | P3_nb4 | 355 | 6.4750e-03 | -43.8 | 6.5292e-03 | 5.42e-05 | 0.83 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P3 | P3_nb8 | 359 | 7.6989e-03 | -42.3 | 7.7807e-03 | 8.19e-05 | 1.05 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P3 | P3_nb16 | 367 | 3.7771e-03 | -48.5 | 3.8185e-03 | 4.14e-05 | 1.08 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P3 | P3_nb32 | 383 | 4.2336e-03 | -47.5 | 4.2953e-03 | 6.18e-05 | 1.44 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 held | True | 1.6e-27 | True | False | HELD |
| z | P4 | single L | 490 | 1.9820e-03 | -54.1 | 1.9982e-03 | 1.62e-05 | 0.81 | [1.5686e-03, 2.4279e-03] | — | True | 4.0e-28 | True | False | HELD |
| z | P4 | single R | 425 | 5.7352e-03 | -44.8 | 5.7907e-03 | 5.55e-05 | 0.96 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | True | HELD |
| z | P4 | P4_nb2 | 349 | 5.1038e-03 | -45.8 | 5.1330e-03 | 2.92e-05 | 0.57 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P4 | P4_nb4 | 351 | 6.5166e-03 | -43.7 | 6.5292e-03 | 1.26e-05 | 0.19 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P4 | P4_nb8 | 355 | 7.7116e-03 | -42.3 | 7.7807e-03 | 6.91e-05 | 0.89 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P4 | P4_nb16 | 363 | 3.7595e-03 | -48.5 | 3.8185e-03 | 5.90e-05 | 1.54 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | P4 | P4_nb32 | 379 | 4.2105e-03 | -47.5 | 4.2953e-03 | 8.48e-05 | 1.97 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 held | True | 1.2e-27 | True | False | HELD |
| z | T | single L1 | 441 | 5.7385e-03 | -44.8 | 5.7996e-03 | 6.11e-05 | 1.05 | [4.6097e-03, 6.9895e-03] | — | True | 1.4e-27 | True | False | HELD |
| z | T | single L2 | 732 | 6.5667e-03 | -43.7 | 6.6366e-03 | 6.99e-05 | 1.05 | [5.2793e-03, 7.9939e-03] | — | True | 0.0e+00 | True | False | HELD |
| z | T | T_c4_b4 | 312 | 2.2759e-02 | -32.9 | 2.2989e-02 | 2.30e-04 | 1.00 | [1.8361e-02, 2.7617e-02] | 2.6116e-02 held | True | 1.4e-27 | True | False | HELD |
| z | T | T_c4_b8 | 316 | 2.2896e-02 | -32.8 | 2.3139e-02 | 2.44e-04 | 1.05 | [1.8482e-02, 2.7797e-02] | 2.6116e-02 held | True | 1.4e-27 | True | False | HELD |
| z | T | T_c16_b4 | 324 | 1.2741e-02 | -37.9 | 1.2933e-02 | 1.92e-04 | 1.49 | [1.0317e-02, 1.5550e-02] | 2.6116e-02 held | True | 1.4e-27 | True | False | HELD |
| z | T | T_c16_b8 | 328 | 9.1272e-03 | -40.8 | 9.2903e-03 | 1.63e-04 | 1.76 | [7.4022e-03, 1.1178e-02] | 2.6116e-02 held | True | 1.4e-27 | True | False | HELD |
| x | S | single L | 440 | 5.7707e-03 | -44.8 | 5.7907e-03 | 2.00e-05 | 0.35 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | False | HELD |
| x | S | S_nb2 | 298 | 7.4740e-03 | -42.5 | 7.4916e-03 | 1.76e-05 | 0.23 | [5.9633e-03, 9.0199e-03] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| x | S | S_nb4 | 300 | 1.0093e-02 | -39.9 | 1.0141e-02 | 4.81e-05 | 0.47 | [8.0826e-03, 1.2199e-02] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| x | S | S_nb8 | 304 | 1.1207e-02 | -39.0 | 1.1296e-02 | 8.91e-05 | 0.79 | [9.0066e-03, 1.3585e-02] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| x | S | S_nb16 | 312 | 1.2339e-03 | -58.2 | 1.2101e-03 | 2.38e-05 | 1.96 | [9.3810e-04, 1.4822e-03] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| x | S | S_nb32 | 328 | 1.4604e-03 | -56.7 | 1.5111e-03 | 5.07e-05 | 3.36 | [1.1789e-03, 1.8434e-03] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P1 | single L | 440 | 5.7707e-03 | -44.8 | 5.7907e-03 | 2.00e-05 | 0.35 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | True | HELD |
| x | P1 | single R | 387 | 1.2387e-02 | -38.1 | 1.2460e-02 | 7.28e-05 | 0.58 | [9.9378e-03, 1.4982e-02] | — | True | 4.0e-28 | True | False | HELD |
| x | P1 | P1_nb2 | 257 | 1.5349e-02 | -36.3 | 1.5446e-02 | 9.70e-05 | 0.63 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P1 | P1_nb4 | 259 | 1.7642e-02 | -35.1 | 1.7774e-02 | 1.33e-04 | 0.75 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P1 | P1_nb8 | 263 | 1.6300e-02 | -35.8 | 1.6474e-02 | 1.73e-04 | 1.05 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P1 | P1_nb16 | 271 | 8.8929e-03 | -41.0 | 8.9908e-03 | 9.79e-05 | 1.09 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P1 | P1_nb32 | 287 | 6.9078e-03 | -43.2 | 7.0095e-03 | 1.02e-04 | 1.45 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P2 | single L | 402 | 1.2426e-02 | -38.1 | 1.2460e-02 | 3.38e-05 | 0.27 | [9.9378e-03, 1.4982e-02] | — | True | 0.0e+00 | True | False | HELD |
| x | P2 | single R | 428 | 5.7882e-03 | -44.7 | 5.7907e-03 | 2.50e-06 | 0.04 | [4.6026e-03, 6.9788e-03] | — | True | 4.0e-28 | True | False | HELD |
| x | P2 | P2_nb2 | 260 | 1.5369e-02 | -36.3 | 1.5446e-02 | 7.64e-05 | 0.49 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P2 | P2_nb4 | 262 | 1.7679e-02 | -35.1 | 1.7774e-02 | 9.57e-05 | 0.54 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P2 | P2_nb8 | 266 | 1.6393e-02 | -35.7 | 1.6474e-02 | 8.10e-05 | 0.49 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P2 | P2_nb16 | 274 | 8.9724e-03 | -40.9 | 8.9908e-03 | 1.84e-05 | 0.20 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| x | P2 | P2_nb32 | 290 | 7.0377e-03 | -43.1 | 7.0095e-03 | 2.82e-05 | 0.40 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 held | True | 1.6e-27 | True | False | HELD |
| x | P3 | single L | 440 | 5.7707e-03 | -44.8 | 5.7907e-03 | 2.00e-05 | 0.35 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | True | HELD |
| x | P3 | single R | 486 | 2.0051e-03 | -54.0 | 1.9982e-03 | 6.89e-06 | 0.34 | [1.5686e-03, 2.4279e-03] | — | True | 4.0e-28 | True | False | HELD |
| x | P3 | P3_nb2 | 356 | 5.1229e-03 | -45.8 | 5.1330e-03 | 1.01e-05 | 0.20 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P3 | P3_nb4 | 358 | 6.5112e-03 | -43.7 | 6.5292e-03 | 1.80e-05 | 0.28 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P3 | P3_nb8 | 362 | 7.7388e-03 | -42.2 | 7.7807e-03 | 4.19e-05 | 0.54 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P3 | P3_nb16 | 370 | 3.8185e-03 | -48.4 | 3.8185e-03 | 2.95e-08 | 0.00 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P3 | P3_nb32 | 386 | 4.2804e-03 | -47.4 | 4.2953e-03 | 1.49e-05 | 0.35 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P4 | single L | 494 | 1.9982e-03 | -54.0 | 1.9982e-03 | 4.05e-08 | 0.00 | [1.5686e-03, 2.4279e-03] | — | True | 4.0e-28 | True | False | HELD |
| x | P4 | single R | 428 | 5.7882e-03 | -44.7 | 5.7907e-03 | 2.50e-06 | 0.04 | [4.6026e-03, 6.9788e-03] | — | True | 4.0e-28 | True | True | HELD |
| x | P4 | P4_nb2 | 352 | 5.1176e-03 | -45.8 | 5.1330e-03 | 1.54e-05 | 0.30 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P4 | P4_nb4 | 354 | 6.5256e-03 | -43.7 | 6.5292e-03 | 3.61e-06 | 0.06 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P4 | P4_nb8 | 358 | 7.7297e-03 | -42.2 | 7.7807e-03 | 5.10e-05 | 0.66 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P4 | P4_nb16 | 366 | 3.7459e-03 | -48.5 | 3.8185e-03 | 7.26e-05 | 1.90 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | P4 | P4_nb32 | 382 | 4.2034e-03 | -47.5 | 4.2953e-03 | 9.20e-05 | 2.14 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| x | T | single L1 | 448 | 5.7730e-03 | -44.8 | 5.7996e-03 | 2.66e-05 | 0.46 | [4.6097e-03, 6.9895e-03] | — | True | 0.0e+00 | True | False | HELD |
| x | T | single L2 | 739 | 6.6002e-03 | -43.6 | 6.6366e-03 | 3.64e-05 | 0.55 | [5.2793e-03, 7.9939e-03] | — | True | 1.4e-27 | True | False | HELD |
| x | T | T_c4_b4 | 316 | 2.2784e-02 | -32.8 | 2.2989e-02 | 2.06e-04 | 0.89 | [1.8361e-02, 2.7617e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| x | T | T_c4_b8 | 320 | 2.2928e-02 | -32.8 | 2.3139e-02 | 2.12e-04 | 0.92 | [1.8482e-02, 2.7797e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| x | T | T_c16_b4 | 328 | 1.2792e-02 | -37.9 | 1.2933e-02 | 1.41e-04 | 1.09 | [1.0317e-02, 1.5550e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| x | T | T_c16_b8 | 332 | 9.1738e-03 | -40.7 | 9.2903e-03 | 1.16e-04 | 1.25 | [7.4022e-03, 1.1178e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| y | S | single L | 440 | 5.7707e-03 | -44.8 | 5.7907e-03 | 2.00e-05 | 0.34 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | False | HELD |
| y | S | S_nb2 | 298 | 7.4740e-03 | -42.5 | 7.4916e-03 | 1.76e-05 | 0.24 | [5.9633e-03, 9.0199e-03] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| y | S | S_nb4 | 300 | 1.0093e-02 | -39.9 | 1.0141e-02 | 4.81e-05 | 0.47 | [8.0826e-03, 1.2199e-02] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| y | S | S_nb8 | 304 | 1.1207e-02 | -39.0 | 1.1296e-02 | 8.91e-05 | 0.79 | [9.0066e-03, 1.3585e-02] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| y | S | S_nb16 | 312 | 1.2339e-03 | -58.2 | 1.2101e-03 | 2.38e-05 | 1.97 | [9.3810e-04, 1.4822e-03] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| y | S | S_nb32 | 328 | 1.4604e-03 | -56.7 | 1.5111e-03 | 5.07e-05 | 3.36 | [1.1789e-03, 1.8434e-03] | 1.2160e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P1 | single L | 440 | 5.7707e-03 | -44.8 | 5.7907e-03 | 2.00e-05 | 0.34 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | True | HELD |
| y | P1 | single R | 387 | 1.2387e-02 | -38.1 | 1.2460e-02 | 7.28e-05 | 0.58 | [9.9378e-03, 1.4982e-02] | — | True | 4.0e-28 | True | False | HELD |
| y | P1 | P1_nb2 | 257 | 1.5349e-02 | -36.3 | 1.5446e-02 | 9.70e-05 | 0.63 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P1 | P1_nb4 | 259 | 1.7642e-02 | -35.1 | 1.7774e-02 | 1.33e-04 | 0.75 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P1 | P1_nb8 | 263 | 1.6300e-02 | -35.8 | 1.6474e-02 | 1.73e-04 | 1.05 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P1 | P1_nb16 | 271 | 8.8929e-03 | -41.0 | 8.9908e-03 | 9.79e-05 | 1.09 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P1 | P1_nb32 | 287 | 6.9078e-03 | -43.2 | 7.0095e-03 | 1.02e-04 | 1.45 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P2 | single L | 402 | 1.2426e-02 | -38.1 | 1.2460e-02 | 3.39e-05 | 0.27 | [9.9378e-03, 1.4982e-02] | — | True | 0.0e+00 | True | False | HELD |
| y | P2 | single R | 428 | 5.7882e-03 | -44.7 | 5.7907e-03 | 2.51e-06 | 0.04 | [4.6026e-03, 6.9788e-03] | — | True | 4.0e-28 | True | False | HELD |
| y | P2 | P2_nb2 | 260 | 1.5369e-02 | -36.3 | 1.5446e-02 | 7.64e-05 | 0.49 | [1.2326e-02, 1.8565e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P2 | P2_nb4 | 262 | 1.7679e-02 | -35.1 | 1.7774e-02 | 9.58e-05 | 0.54 | [1.4189e-02, 2.1359e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P2 | P2_nb8 | 266 | 1.6392e-02 | -35.7 | 1.6474e-02 | 8.11e-05 | 0.49 | [1.3149e-02, 1.9798e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P2 | P2_nb16 | 274 | 8.9724e-03 | -40.9 | 8.9908e-03 | 1.84e-05 | 0.21 | [7.1626e-03, 1.0819e-02] | 1.9163e-02 held | True | 0.0e+00 | True | False | HELD |
| y | P2 | P2_nb32 | 290 | 7.0376e-03 | -43.1 | 7.0095e-03 | 2.81e-05 | 0.40 | [5.5776e-03, 8.4414e-03] | 1.9163e-02 held | True | 1.6e-27 | True | False | HELD |
| y | P3 | single L | 440 | 5.7707e-03 | -44.8 | 5.7907e-03 | 2.00e-05 | 0.34 | [4.6026e-03, 6.9788e-03] | — | True | 0.0e+00 | True | True | HELD |
| y | P3 | single R | 486 | 2.0052e-03 | -54.0 | 1.9982e-03 | 6.93e-06 | 0.35 | [1.5686e-03, 2.4279e-03] | — | True | 4.0e-28 | True | False | HELD |
| y | P3 | P3_nb2 | 356 | 5.1228e-03 | -45.8 | 5.1330e-03 | 1.01e-05 | 0.20 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P3 | P3_nb4 | 358 | 6.5111e-03 | -43.7 | 6.5292e-03 | 1.81e-05 | 0.28 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P3 | P3_nb8 | 362 | 7.7388e-03 | -42.2 | 7.7807e-03 | 4.19e-05 | 0.54 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P3 | P3_nb16 | 370 | 3.8185e-03 | -48.4 | 3.8185e-03 | 2.05e-08 | 0.00 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P3 | P3_nb32 | 386 | 4.2804e-03 | -47.4 | 4.2953e-03 | 1.49e-05 | 0.35 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P4 | single L | 494 | 1.9982e-03 | -54.0 | 1.9982e-03 | 6.71e-08 | 0.00 | [1.5686e-03, 2.4279e-03] | — | True | 4.0e-28 | True | False | HELD |
| y | P4 | single R | 428 | 5.7882e-03 | -44.7 | 5.7907e-03 | 2.51e-06 | 0.04 | [4.6026e-03, 6.9788e-03] | — | True | 4.0e-28 | True | True | HELD |
| y | P4 | P4_nb2 | 352 | 5.1175e-03 | -45.8 | 5.1330e-03 | 1.54e-05 | 0.30 | [4.0764e-03, 6.1896e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P4 | P4_nb4 | 354 | 6.5255e-03 | -43.7 | 6.5292e-03 | 3.70e-06 | 0.06 | [5.1934e-03, 7.8650e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P4 | P4_nb8 | 358 | 7.7298e-03 | -42.2 | 7.7807e-03 | 5.10e-05 | 0.65 | [6.1946e-03, 9.3669e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P4 | P4_nb16 | 366 | 3.7458e-03 | -48.5 | 3.8185e-03 | 7.26e-05 | 1.90 | [3.0248e-03, 4.6122e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | P4 | P4_nb32 | 382 | 4.2034e-03 | -47.5 | 4.2953e-03 | 9.19e-05 | 2.14 | [3.4063e-03, 5.1844e-03] | 8.1784e-03 held | True | 0.0e+00 | True | False | HELD |
| y | T | single L1 | 448 | 5.7729e-03 | -44.8 | 5.7996e-03 | 2.67e-05 | 0.46 | [4.6097e-03, 6.9895e-03] | — | True | 0.0e+00 | True | False | HELD |
| y | T | single L2 | 739 | 6.6001e-03 | -43.6 | 6.6366e-03 | 3.65e-05 | 0.55 | [5.2793e-03, 7.9939e-03] | — | True | 1.4e-27 | True | False | HELD |
| y | T | T_c4_b4 | 316 | 2.2784e-02 | -32.8 | 2.2989e-02 | 2.06e-04 | 0.89 | [1.8361e-02, 2.7617e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| y | T | T_c4_b8 | 320 | 2.2928e-02 | -32.8 | 2.3139e-02 | 2.12e-04 | 0.92 | [1.8482e-02, 2.7797e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| y | T | T_c16_b4 | 328 | 1.2793e-02 | -37.9 | 1.2933e-02 | 1.41e-04 | 1.09 | [1.0317e-02, 1.5550e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |
| y | T | T_c16_b8 | 332 | 9.1739e-03 | -40.7 | 9.2903e-03 | 1.16e-04 | 1.25 | [7.4022e-03, 1.1178e-02] | 2.6116e-02 held | True | 0.0e+00 | True | False | HELD |

Arms: 120 HELD, 0 FIRED of 120. Largest relative deviation per axis: z 5.71 %, x 3.36 %, y 3.36 %.

Table B — law checks per (pattern, axis) cell and the validity-domain verdicts

| axis | pattern | R_L meas / model (dev %) | R_R meas / model (dev %) | c_meas (mm) | c_model (mm) | dev (mm) | window +/- (mm) | W3 | W1 fired arms | W2 fired | gates hold (all) | law domain | -54 dB class | coarsest cells/lambda0 | relabel flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z | S | 5.7302e-03 / 5.7907e-03 (1.04) | — (chain mirror 5.7907e-03) | 1.887 | 1.873 | 0.014 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | — |
| z | P1 | 5.7302e-03 / 5.7907e-03 (1.04) | 1.2329e-02 / 1.2460e-02 (1.05) | 3.271 | 3.294 | 0.023 | 0.140 | HELD | none | no | True | **inside** | inside | 10.93 | — |
| z | P2 | 1.2324e-02 / 1.2460e-02 (1.09) | 5.7352e-03 / 5.7907e-03 (0.96) | 3.272 | 3.294 | 0.022 | 0.140 | HELD | none | no | True | **inside** | inside | 10.93 | — |
| z | P3 | 5.7302e-03 / 5.7907e-03 (1.04) | 1.9832e-03 / 1.9982e-03 (0.75) | 0.918 | 0.937 | 0.019 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | — |
| z | P4 | 1.9820e-03 / 1.9982e-03 (0.81) | 5.7352e-03 / 5.7907e-03 (0.96) | 0.961 | 0.937 | 0.024 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | — |
| z | T | 5.7385e-03 / 5.7996e-03 (1.05) | 6.5667e-03 / 6.6366e-03 (1.05) | — | — | — | — | n/a | none | no | True | **inside** | inside | 15.30 | — |
| x | S | 5.7707e-03 / 5.7907e-03 (0.35) | — (chain mirror 5.7907e-03) | 1.883 | 1.873 | 0.010 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | held |
| x | P1 | 5.7707e-03 / 5.7907e-03 (0.35) | 1.2387e-02 / 1.2460e-02 (0.58) | 3.261 | 3.294 | 0.033 | 0.140 | HELD | none | no | True | **inside** | inside | 10.93 | held |
| x | P2 | 1.2426e-02 / 1.2460e-02 (0.27) | 5.7882e-03 / 5.7907e-03 (0.04) | 3.284 | 3.294 | 0.010 | 0.140 | HELD | none | no | True | **inside** | inside | 10.93 | held |
| x | P3 | 5.7707e-03 / 5.7907e-03 (0.35) | 2.0051e-03 / 1.9982e-03 (0.34) | 0.930 | 0.937 | 0.006 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | held |
| x | P4 | 1.9982e-03 / 1.9982e-03 (0.00) | 5.7882e-03 / 5.7907e-03 (0.04) | 0.977 | 0.937 | 0.040 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | held |
| x | T | 5.7730e-03 / 5.7996e-03 (0.46) | 6.6002e-03 / 6.6366e-03 (0.55) | — | — | — | — | n/a | none | no | True | **inside** | inside | 15.30 | held |
| y | S | 5.7707e-03 / 5.7907e-03 (0.34) | — (chain mirror 5.7907e-03) | 1.883 | 1.873 | 0.010 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | identity FIRED: instrument class |
| y | P1 | 5.7707e-03 / 5.7907e-03 (0.34) | 1.2387e-02 / 1.2460e-02 (0.58) | 3.261 | 3.294 | 0.033 | 0.140 | HELD | none | no | True | **inside** | inside | 10.93 | identity FIRED: instrument class |
| y | P2 | 1.2426e-02 / 1.2460e-02 (0.27) | 5.7882e-03 / 5.7907e-03 (0.04) | 3.284 | 3.294 | 0.010 | 0.140 | HELD | none | no | True | **inside** | inside | 10.93 | identity FIRED: instrument class |
| y | P3 | 5.7707e-03 / 5.7907e-03 (0.34) | 2.0052e-03 / 1.9982e-03 (0.35) | 0.930 | 0.937 | 0.006 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | identity FIRED: instrument class |
| y | P4 | 1.9982e-03 / 1.9982e-03 (0.00) | 5.7882e-03 / 5.7907e-03 (0.04) | 0.977 | 0.937 | 0.040 | 0.140 | HELD | none | no | True | **inside** | inside | 15.30 | identity FIRED: instrument class |
| y | T | 5.7729e-03 / 5.7996e-03 (0.46) | 6.6001e-03 / 6.6366e-03 (0.55) | — | — | — | — | n/a | none | no | True | **inside** | inside | 15.30 | identity FIRED: instrument class |

### Reading the tables

- **Unequal coarse sides (P1, P2).** The three-step side (1.0 -> 1.4 ->
  1.96 -> 2.744 mm) reflects 1.2329e-2 / 1.2387e-2 (z / in-plane) against
  the chain's 1.2460e-2 (1.05 % / 0.58 %), 2.15 x the one-ramp side; the
  band rows follow the two-amplitude Fabry-Perot form with `c_meas`
  3.271 / 3.272 mm on z and 3.261 / 3.284 mm in-plane against `c_model`
  3.294 mm (window 0.140 mm; the centroid closed form 3.228 mm sits 0.066
  mm from the chain and 0.02-0.06 mm from the measurements). **Traversal
  direction does not matter at the gate level:** P1 and P2 are the same
  cell in the chain (reciprocity, |S11| = |S22|, equal to 1e-15) and both
  are inside W1 and W3 on every axis; the program's likeliest fire "W3
  on P2" did not occur (dev 0.022 mm z, 0.010 mm x/y). The measured
  |R_meas(P1) - R_meas(P2)| / R_meas(P2) per width (n_b 2/4/8/16/32) is
  z 0.02 / 0.08 / 0.02 / 0.11 / 0.06 %, x 0.13 / 0.21 / 0.56 / 0.89 /
  1.84 %, y the same as x (the sentence first written here, "within
  0.1-0.5 % on every axis", mis-stated both ends — second-pass
  correction). The in-plane spread grows with n_b to 1.84 % (P1 n_b = 32
  -1.45 %, P2 n_b = 32 +0.40 % from the chain; P3 vs P4 in-plane up to
  1.94 %), about 17 x the z spread: the pinned lo wall differs between
  the two traversals (`[1.0, 1.4]` before a 1.96 mm lead vs `[1.0, 1.4,
  1.96]` before a 2.744 mm lead), so P1 and P2 have different wall-echo
  instrumentation in-plane and none on z. All of it is under the ~3 %
  window-end sensitivity of the instrument (W1 reviewer note).
- **A coarse side that is the ramp size itself (P3, P4).** The single
  cap step 1.0 -> 1.4 mm reflects 1.9832e-3 (z) / 2.0051e-3 (x, y) against
  1.9982e-3 (0.75 % / 0.34 %), 0.345 x the ramp side; the band nulls sit
  at |R_L - R_R| = 3.79e-3 as the chain says (rows 3.76e-3 .. 7.78e-3
  measured, none below 3.7e-3); `c_meas` 0.918 / 0.961 mm (z) and 0.930 /
  0.977 mm (x, y) against 0.937 mm, the largest c deviation of the lane
  (0.040 mm, P4 in-plane, 0.29 of the window). The closed form 0.927 mm
  is within 0.010 mm of the chain.
- **Two fine sizes in one column (T).** The five-step ramp into the
  0.5 mm band (60 cells per lambda0) reflects 6.5667e-3 (z) / 6.6002e-3
  (x, y) against 6.6366e-3 (1.05 % / 0.55 %) — about the same as the one
  cell ramp into the 1.0 mm band (5.74e-3 / 5.77e-3 vs 5.7996e-3),
  although it spans 3.92 x in cell size: the per-step amplitudes fall as
  the cells shrink (3.42e-3, 1.96e-3, 1.13e-3, 6.5e-4, 3.8e-4). The four
  two-band arms are within 0.89-1.76 % of the chain, all under the
  four-amplitude bound (largest measured 0.878 of 2.6116e-2 — 0.8767 z,
  0.8779 x/y; the bound held with 12 % to spare, the model said 0.886). The likeliest-fire arm
  (n_c = 16, n_b2 = 8, inner-return margin 0.397 ns) deviates 1.76 % on z
  and 1.25 % in-plane.
- **In-plane.** The first FDTD-measured in-plane reflection numbers: 40
  arms on x and 40 on y, every one inside its window, x and y equal to
  <= 4.1e-5 relative, deviations 0.00-3.36 % (median 0.49 %; the
  instrument resolves about 3 %, so the median is a floor reading). The
  -54 dB class is met by construction on every cell (every adjacent
  ratio 1.4000 on the builder output, 1.3142 inside the 0.5 mm ramp);
  the coarsest runway is 10.9 cells per lambda0 (2.744 mm) on P1/P2.
- **Largest deviations.** z: 5.71 % (S n_b = 32, E1's own row); x and y:
  3.36 % (S n_b = 32). Every other row is under 2.2 %. The floor rule
  (2.5e-4) never applied — the smallest row measured is 1.2339e-3.

### Validity domain, as measured

Law (the deliverable, quoted with its range): for a fine band of n_b
cells of size d between two transitions whose single-transition
amplitudes are `R_L` (seen from the lead runway) and `R_R` (seen from
inside the band), the reflection at 10 GHz on the PEC-closed TE10 fixture
is

    R(n_b) = sqrt(R_L^2 + R_R^2 - 2 R_L R_R cos(2 k_g (n_b d + c))),
    R(n_b) <= R_L + R_R at every width (1.05 slack held on 87 of 87 band arms),
    c = c_model of the chain (measured within 0.006-0.040 mm on 15 cells, window 0.140 mm),
    c ~ sum over the two sides of the (local cell)^2 centroid of that side's step planes
        (0.927 mm per one-ramp side, 2.301 mm per two-ramp side, 0 for a bare cap step;
        within 0.010-0.066 mm of the chain, an explanation, not the gate),

with `R_L`, `R_R` the chain model's single transitions, each measured
within 0.00-1.09 % of the chain on every axis (eleven single-arm entries
per axis of eight distinct FDTD runs — P1.L = P3.L = S.L and P4.R = P2.R
— 33 verdicts, 24 distinct runs over three axes; the first version of
this sentence said "fifteen single arms", which is the number of
(pattern, axis) cells with a c fit, a different count). Two fine bands
of different sizes in one column follow the chain within 1.8 % and the
coherent four-amplitude bound. Precision of those agreement figures
(second pass): the instrument's window-end sensitivity is about 3 %
(-2.0 .. +4.5 % over the legal window ends on five arms, W1 reviewer
note), so "within 1.09 %" and "within 1.8 %" are the nominal-window
readings and the law is witnessed at the 3 % level, not the 1 % level.

Inside the law domain, measured here: **18 of 18 (pattern, axis)
cells** — the E1 control (S) and the four unequal-side patterns P1-P4
(coarse sides 1.4 / 1.96 / 2.744 mm, i.e. 21.4 / 15.3 / 10.9 cells per
lambda0, ramps of 0, 1, 2 intermediate cells, both traversal
directions) at band widths 2-32 cells of 1.0 mm, and the two-size pattern
T (1.0 mm x 4 and 0.5 mm x 4 / 8 with 4 / 16 coarse cells between, ramps
of 1 and 4 intermediate cells) — on z, x and y, at cap 1.4, PEC closed,
vacuum, 10 GHz, no absorber; in-plane with 1.0 mm boundary pins (G1) on
every profile. Instrument class on the axes: z -> x bit-identical,
z -> y float32-class (7e-7 on traces, 6e-6 on R_meas — W5 fired there and
the y verdicts carry the flag). Outside / not witnessed here: coarse
runways under 10.9 cells per lambda0 (E1's 3.75 was outside on z),
ratios other than 1.4 (E1 measured 1.2 and 2.0 on z only), bands under 2
cells, three or more fine sizes in one column, dielectric or lossy
interiors, an absorber, frequencies other than 10 GHz, in-plane profiles
whose end cells differ from each other (G1 — Lane 3).

`stopped = false`. Windows unchanged. Commits of this lane: the
pre-declaration with the instrument and the frozen model (`be606f45`),
the replay test and classification entries (`ea208024`), the raw
measurements (`b4646528`), this results section with the W4-block
regeneration and the results-table printer (the commit that carries it).

## Second pass (review of the record, 2026-09-14)

Ten review findings, each reproduced from the stored JSONs or by reruns
on this tree before anything was edited (`rfx.__file__` under this
worktree; scratch scripts `window_scan.py`, `analyze_stored.py`,
`fdtd_checks.py` in the session scratchpad, not part of the record).
No window was edited; no verdict changed; no arm was re-attempted (no
finding showed a measurement to be invalid: the closest arm to a W1 edge
is at 0.26 of its half-window, x S n_b = 32). `stopped` stays false.

### What was corrected in the text above (each marked "second-pass correction" in place)

| # | finding (review lens) | what the record said | what the stored numbers say | where |
|---|---|---|---|---|
| 1 | window-end systematic ~3 % (physics, major) | sub-1 % agreement quoted as resolved | instrument resolves about 3 %; verdicts unaffected | W1 reviewer note; "Reading the tables"; "Validity domain" |
| 2, 6 | P1 = P2 "within 0.1-0.5 % on every axis" (physics + record, major) | 0.1-0.5 % | z 0.02-0.11 %, x/y 0.13-1.84 %, growing with n_b; P3/P4 in-plane up to 1.94 % | "Reading the tables" |
| 3 | W5 `R_meas` 1e-6 is bit-identity in disguise (minor) | fire read as float32 class | same reading, the sub-criterion's design named as the defect; deterministic, 1.8 eps RMS | W5 reviewer note |
| 4 | W6 0.256 ns is tautological (minor) | "smallest margin 0.2560 ns" | 4 sigma by construction; the tightest measured gate is 0.397 ns (inner return) | W6 reviewer note |
| 5 | "fifteen single arms of eight distinct kinds" (minor) | 15 | 11 entries x 3 axes = 33 verdicts, 8 distinct runs per axis, 24 over three axes | "Validity domain" |
| 7 | L1-0 provenance overwritten; 107 s (record, minor) | 107 s; "tree clean, nothing ran in between" | stored wallclock sums to 102.8 s; the clean-tree sentence is not re-derivable from the record | "Measured" |
| 8 | `--refresh-w4` from uncommitted code stamped as a clean sha (record, minor) | `git_sha b4646528`, no dirty flag | `git_dirty_at_refresh: true` annotation added to the block; the CLI stamp records `git_dirty` from now on | "Measured"; `e5_z.json` |
| 9 | rounding (record, minor) | 4.0e-5; 0.877; commit message 139 runs | 4.08e-5 (quoted <= 4.1e-5); 0.8779 (quoted 0.878); 141 runs, 139 was the declared count | W5, "Reading the tables", "Measured" |
| 10 | program document section 2 "exact cyclic relabelings" (record, minor) | claim stands in the frozen plan | z -> x measured exact, z -> y float32 class; addendum appended to the program document (gap-map candidate), section 2 not edited | program document |

Finding 1 also renames one reading: "pinned arms deviate less (36 of
40)" is a count under the instrument's resolution, not evidence about
the pins (pin-bridge paragraph). Finding 2's physical content is new to
the record: the in-plane P1/P2 spread is ~17 x the z spread because the
pinned lo wall differs between the two traversals — an instrumentation
asymmetry, unrecorded before.

### Declared diagnostic: window-end scan (declared here BEFORE it ran through the instrument)

Purpose: put the review's window-sensitivity numbers into the record
through the instrument, not a scratch script. Not an arm; no verdict is
computed from it. Five recorded arms are re-run once each through the
same `run_probe_axis` / `make_b_profile` / `dft_at` path (S single L and
S n_b = 4 on z, T n_c = 16 / n_b2 = 8 on z, P1 n_b = 32 and P2 n_b = 32
on x, pinned as recorded), 10 CPU FDTD runs incl. the B references.
For each arm: (a) the rerun `R_meas` must equal the stored `R_meas` to
the float64 bit (precondition; if it does not, the scan is reported and
the discrepancy becomes a finding); (b) the incident window end is moved
from arrival + 4 sigma to arrival + 12 sigma (the declared rule is
arrival + 8 sigma; the source-wall echo bound, 1.478 ns, is beyond 12
sigma on every arm) and `|DFT_F0(B)|` is recorded relative to the
nominal, with the 4-8 sigma sub-band, the peak-to-peak and the dominant
beat of the scan (FFT of the scan, resolution ~1.3 GHz) beside the
discrete TE10 cutoff of the grid (`s0(f_c) = sy`, chain-model
dispersion); (c) the reflection window end is moved from `t_inner_last +
4 sigma` to `gate_end` and `|DFT_F0(A - B)|` recorded the same way; (d)
`R_meas` over every (reflection end, incident end) pair, as a range
relative to the chain. Expectation (from the reproduction on this tree):
(a) true on all five; (b) 0.9727 .. 1.0053, p-p 3.26 %, beat 5.28 GHz
against a 4.99 GHz cutoff; (c) 0.993 .. 1.002 (S n_b = 4), 0.993 ..
1.007 (S single), 0.993 .. 1.012 (T), 0.987 .. 1.031 (x P1), 0.975 ..
1.000 (x P2); (d) -1.99 .. +2.19 %, -2.29 .. +2.48 %, -2.98 .. +2.20 %,
-3.25 .. +4.50 %, -2.67 .. +3.15 % of the chain. Output
`results/e5_window_scan.json`; replayed by
`test_e5_multilevel_replay.py::test_results_window_scan_replays` (the
stored ranges re-derived from the stored numbers, the bit-identity
flags, and the statement that no recorded arm is within 0.30 of its
half-window of a W1 edge). Command: `--window-scan
validation/research/multiband_nu/results --out
validation/research/multiband_nu/results/e5_window_scan.json`.
