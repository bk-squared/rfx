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
- **W6 — gates:** the six margins (1f) of every arm are positive in the
  frozen model JSON (Table G: smallest 0.2560 ns, the incident-gate
  construction, on every arm; inner-return margin on T's longest
  structure n_c = 16 / n_b2 = 8: 0.397 ns z, 0.416 ns x/y — the program
  expected ~0.42 ns). `gates_hold` is measured per arm (geometry AND
  `dt_matches_b` AND `source_probe_in_lead` AND `lead_f32_identical`).
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
