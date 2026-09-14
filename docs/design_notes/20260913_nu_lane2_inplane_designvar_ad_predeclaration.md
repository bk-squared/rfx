# Lane 2 — in-plane design-variable autodiff (E6) — pre-declaration

**Status:** pre-declaration. This commit precedes every FDTD measurement of
the lane. Every numeric window below is copied from the program document
(`20260913_nu_full_functionality_program.md` section 5.4, frozen there at
`d6bc5dde`) and is frozen here again; results are appended under
"Results", never edited into the windows. One attempt per arm (the
instrument's `.started` claim file refuses a rerun); a fired window is a
result, not a bug to tune away.
**Class:** gradient-correctness measurement by the AD-Q judges (Taylor
order with the 32-quantum eligibility rule and the longest-monotone-run
selector; FD agreement against the FD's own Richardson + roundoff bar with
the rho-in-[2, 8] validity check; the synthetic self-check before every
arm; the stop_gradient(dt) revert-proof) — never a flat percentage —
along two PHYSICAL in-plane design variables on a 3-D NU fixture. The
judges are imported verbatim from `adq_designvar.py`; that file and
`e4_diff_stackup.py` are not edited by this lane.
**Tree:** worktree `rfx-nu-full`, branch `feat/nu-full-functionality`,
based on `feat/nu-ad-directional @ 88460d24` (on `origin/main fa392913`);
program document at `d6bc5dde`, Lane 1 closed at `847269a5`. Instrument:
the new module `validation/research/multiband_nu/e6_inplane_designvar.py`
(this commit). No `rfx/` change (`git diff --stat rfx/` empty at every
commit of the lane).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-full/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `argv`, `started_utc`, the judge self-check and the fixture
constants. The model numbers in this note are the file
`results/e6_model.json`, written by `--arm model` on this tree (HEAD
`847269a5`, `git_dirty` true because the instrument, this note and that
JSON are one commit; zero FDTD, 1.1 s). CPU only. Baseline before any
change: `test_adq_designvar_replay.py` + `test_band_accuracy_ad_replay.py`
= 31 passed, 1 xfailed.
Date: 2026-09-14 (KST).

## 0. Question (program 5.1)

Is the reverse-mode gradient of a 3-D NU simulation correct along two
PHYSICAL in-plane design variables — the width `w` of a fine x band (a
trace-width surrogate: a PEC strip whose x edges are the band edges,
attached by node index) and its position `x_c` — including the CFL `dt`
path through the x tied set, for (a) a transient loss (L1 class) and (b)
a smooth spectral observable that does not carry the L2 ripple? Judged by
the AD-Q judges verbatim; coverage of the x cell-space gradient reported.

Everything the program document specifies for Lane 2 (5.2 instrument, 5.3
arms, 5.4 windows, 5.5 regression, 5.6 cost, 5.7 claims, 5.8 PI
confirmations — the index-attached PEC strip and the Hann + comb
observable, both taken as confirmed by the executor's brief) is taken as
written. Section 1 records what was found while turning it into code, with
zero FDTD; where the program's wording could not be built as written, the
declared alternative is stated there BEFORE any run.

## 1. Findings made while writing this note (zero FDTD)

### 1a. The PEC strip kwarg: `run_nonuniform(pec_edge_masks=...)`

Program 5.2 names `pec_mask_override` / `pec_occupancy_override`; those
are the `rfx/api/_execute.py` layer's names. The functional entry point
this lane drives, `run_nonuniform` (`rfx/nonuniform.py`, signature at line
2402), accepts `pec_edge_masks=(Mx, My, Mz)` directly — the SAME realized
edge masks that `pec_mask_override` feeds through
`realized_pec_edge_masks`, applied per step by `apply_pec_edges` (E times
`1 - mask`, tracer-safe). Declared: the strip is the constant boolean
triple `Mx[20:40, :, 12]`, `My[20:41, :, 12]`, `Mz` empty — 20 `ex` edges
between the 21 band-edge nodes `i = 20 .. 40` and 21 `ey` node columns,
all 13 y nodes, on the z-node plane `k_s = 12`. The program's "thin
in-instrument step wrapper" alternative is not needed and not used; m7
records the kwarg as accepted.

### 1b. The loss is parametrized by the displacement `delta = p - p0`

`losses_along` (AD-Q, verbatim) casts `x + h v` to float32. Parametrized
by `p` itself, a displacement `h w0` would be quantized to the float32 ulp
of `p0` (2.3e-10 m at `w0 = 2 mm`, 9.3e-10 m at `x_c0 = 9 mm`), 1.6 % /
6 % of the smallest ladder step `2^-17 x 2 mm = 1.5e-8 m`. The map here
takes `delta` (float32) and forms `seg_len = seg_len0 + A delta` inside
the trace, so the displacement is realized to float32 precision of the
displacement itself; `losses_along(loss, zeros(2), v, HS)` is then used
verbatim with `x = 0`. At `delta = 0` the map returns `cells0 x 1.0`
exactly, so the traced cells at the nominal are bit-identical to the
float32 cast of the builder's float64 output (checked, no FDTD).

### 1c. The pinned map's free length is 7.5 mm, not 8 mm

The lead and tail segments hold one pinned 0.5 mm cell each, so a band
change of `h w0 / 2` is absorbed by 7.5 mm of free cells, not 8 mm. The
program's seam-ratio formula wrote `1.287 (1 + h x 1 mm / 8 mm) / (1 - h)`
(1.494 at h = 2^-3); with the pinned map it is `1.2871 (1 + h x 1 mm /
7.5 mm) / (1 - h)` = **1.4955** at h = 2^-3 (Table L below, computed from
the map). Not a window (program 5.4: recorded per ladder point). Along
`x_c` the lead-side seam is `1.2871 (1 + h x 2 mm / 7.5 mm)` = 1.3300 at
h = 2^-3 and above the 1.3 advisory from h = 2^-6 (1.3085), the tail-side
seam falling by the same factor.

### 1d. The builder's end-segment plateau is 0.4546 mm

`make_band_profile` realizes the 8 mm end segments as the 0.5 mm pin, 14
plateau cells of 0.45462563366059 mm and a 5-cell geometric ramp
0.35322 / 0.27443 / 0.21322 / 0.16566 / 0.12871 mm down to the band (the
exact-fit plateau below the 0.5 mm target, G5 rule; six steps of ratio
`(0.4546/0.1)^(1/6) = 1.2871` from the plateau to the band, every adjacent
ratio under the 1.3 cap, the seam being one of them).
The declared vector IS the builder's output (60 cells, counts 20/20/20,
interfaces at nodes 20 / 40 / 60 to 1.4e-17 m); the map scales it, it
never re-solves it.

### 1e. Which `dt` uses the revert-proof cuts

`dt` enters the forward model through the update coefficients, the source
waveform (physical time `t = n dt`, traced), the L2s Hann taper and the
DFT phase. The revert-proof replaces `grid.dt` by `stop_gradient(grid.dt)`
right after `make_nonuniform_grid`, before any of those consumers read it,
so "the dt path" means every use of `dt` at once; the inverse-spacing
arrays keep their cell gradient. Forward values are bit-identical with and
without the cut (recorded per arm as `forward_values_identical`).

### 1f. Floor and the primary verdict

The floor is measured for BOTH arms from the start (program 5.2): 64
evaluations along `+v` at relative h uniform in [1e-5, 1e-4], quadratic
fit, `sigma = RMS residual x sqrt(64/61)` (the AD-Q second-attempt
arithmetic), the cubic must reduce the RMS by < 10 % or `sigma` is flagged
unreliable. Declared reading rule: the PRIMARY verdict of every direction
uses `sigma_eff = max(ulp(loss0), sigma)` in both judges whether or not
`sigma` is flagged (the flag is a recorded diagnostic); the 1-ulp verdicts
are stored alongside (`order_1ulp`, `fd_1ulp`) for comparison with
`stack_l1`. The revert arm re-uses E6-L1's `sigma_eff` per control (same
loss function, same nominal) and re-runs the true ladder; it records
whether that ladder is bit-identical to E6-L1's.

### 1g. The source is a soft `ez` addition, by index

`run_nonuniform`'s `sources` contract adds the waveform value to the named
field at the node (`field.at[i, j, k].add`), so the program's "Gaussian-
sine `jz`" is realized as a soft `ez` source at index (8, 6, 6); the
waveform `exp(-((t - t0)/sigma_t)^2) sin(2 pi F0 t)` with `t = n dt`
(traced). Probe `ez` at (43, 6, 14) by index. Vacuum, `cpml_layers = 0`,
PEC on all six faces (the closed-box path reads no scalar `grid.dx`).

### 1h. What the map's float32 view carries

m1 and m2 are float64 statements about the map (thresholds 1e-9 m). The
float32 cells as the solver sees them sum to 18 mm within 1.47e-9 m over
the whole ladder (0.8 float32 ulp of the 18 mm column) — recorded, not a
window. The float32 AD Jacobian of the map agrees with the exact float64
Jacobian to 3.0e-9 (max absolute entry difference; entries are of order
0.01-0.06).

## 2. Fixture (program 5.2, unchanged)

| item | value |
|---|---|
| x profile | `make_band_profile([0, 8, 10, 18] mm, [0.5, 0.1, 0.5] mm, protected=[F, T, F], max_ratio=1.3, boundary_cell=0.5 mm)` -> 60 cells, 20/20/20 |
| tied set (x) | cells 20..39 (the band), all 1.0e-4 m; first ramp cell 1.2871e-4 m; y and z minima 0.5 mm not tied |
| y, z | 12 x 0.5 mm, 24 x 0.5 mm; grid shape (61, 13, 25) nodes |
| dt(p0) | 3.1776247825516075e-13 s (float64 concrete and the concrete grid agree bit for bit) |
| PEC strip | z-node plane 12, x nodes 20..40 (21), all y; `pec_edge_masks` (1a) |
| source / probe | soft `ez` at (8, 6, 6); `ez` at (43, 6, 14); F0 = 20 GHz, sigma_t = 20 ps, t0 = 100 ps |
| L1 | `sum_n ez(n)^2`, N1 = 600 steps (190.7 ps at dt(p0)) |
| L2s | N2 = 2400 steps (762.6 ps); Hann `h(t) = 0.5 (1 - cos(2 pi t / T_w))` for `t < T_w`, else 0; `T_w = 0.8 N2 dt(p0)` = 610.1 ps FIXED; `1/T_w` = 1.64 GHz; comb `f_k = 20 GHz + k x 0.5 GHz`, k = -10..10 (15-25 GHz, 10 GHz wide); `sum_k |dt sum_n h(n dt) ez(n) e^{-2 pi i f_k n dt}|^2`; `checkpoint_every = 100` |
| controls | `w` (band width, column (-1/2, 1, -1/2) on (lead, band, tail)), `x_c` (column (1, 0, -1)); `p0 = (2 mm, 9 mm)`; ladder displacement `h x 2 mm` for BOTH |
| ladder | `HS = 2^-k`, k = 17..3 (AD-Q); floor `FLOOR_HS` = 64 points in [1e-5, 1e-4] |
| map Jacobian (d cells / d p, exact) | lead free cells: (-0.030308, +0.060617) plateau, (-0.008581, +0.017161) last ramp cell; band cells (0.05, 0); tail mirrored with the `x_c` sign flipped; pinned cells (0, 0) |

Why the L2s observable is smooth in the design variable (declared): the
taper reaches zero with zero slope at `T_w` and the record is longer than
`T_w` on every ladder point (`N2 dt / T_w` >= 1.103, minimum at the h =
2^-3 shrink of `w`, Table L), so no sample at the record end is truncated
with finite weight — the rectangular-window truncation phase that rippled
the AD-Q L2 power (~1 % in thickness) does not exist here. The comb (10
GHz) is wider than the window's main lobe (1.64 GHz), so the band integral
is a smooth functional of the windowed spectrum. What moves with the
design is the sample grid `t = n dt` inside a smooth integrand — a
Riemann-sum drift, second order in `h`. If the observable still FIRES with
rho-valid steps, that is the result of this lane (program 5.4: no second
observable is tried inside it).

## 3. Falsifiers and windows (frozen; program 5.4 verbatim)

- **Order:** R0 slope in **[0.9, 1.1]**, R1 slope in **[1.8, 2.2]**
  (AD-Q, frozen; the upper edge 2.2 carries no gradient-correctness
  meaning and stays; the lower-side criterion R1 >= 1.8 is reported
  alongside as the labelled diagnostic). No eligible run of >= 4 points =>
  INCONCLUSIVE; a fitted miss => FIRED. Judge: `fit_order`, 32-quantum
  eligibility with `sigma_eff` (1f), curvature guard, longest monotone run.
- **FD:** `|g - D(h)| <= 3 B(h)` at every informative step (`rho` in
  [2, 8]); a direction is FIRED if ANY informative step fires, HELD if at
  least one is informative and none fires, else INCONCLUSIVE;
  INCONCLUSIVE if `3B/|g| > 0.15`. Judge: `fd_budget` with `scale = 2 mm`
  and `sigma_eff`.
- **Revert-proof (E6-R):** `w` FIRED on order (R1 slope below 1.8, or no
  fit) with the dt path removed AND `x_c` verdicts (order and FD)
  unchanged from E6-L1 — both required for E6-L1's `w` verdict to be read
  as covering the dt path. If `w` HOLDS here the gate has no power against
  the dt path on this fixture at this ladder, and E6-L1's `w` verdict may
  not be read as verifying the dt path; the note will say so.
- **Map checks m1-m8 (instrument checks, not windows; a failure is fixed
  BEFORE the arms and recorded):** m1 `|sum cells(p) - 18 mm| <= 1e-9 m`
  at every ladder point of both controls, both signs; m2 every interface
  on its node to `<= 1e-9 m` at every ladder point; m3 tied-set spread of
  each Jacobian column EXACTLY 0.0 and the float32 tied set equal to the
  20 band cells at every ladder point; m4 boundary-cell Jacobian exactly
  0.0 (float64 and float32 AD); m5 `J^T g` vs the direct parameter
  gradient `<= 1e-5` relative per component (L1); m6 `d(dt)/dw` AD vs
  analytic `<= 1e-6` relative and `d(dt)/dx_c` exactly 0; m7 strip
  occupancy exactly 21 nodes and the kwarg accepted (1a); m8 traced L1 at
  p0 vs the concrete float64-map L1 at p0 `<= 1e-6` relative.
- **Stretch record (G20, declared, not a window):** `seam_ratio(h)` per
  ladder point, both controls, both signs (Table L), so the support matrix
  can state the range of the claim.
- **Self-check before every arm:** `selfcheck()` must reproduce R1 2.003
  (HELD) and 0.909 (FIRED); recorded in each JSON (model arm: 2.00301 /
  0.90943, `all_pass` true).
- **Regression (program 5.5):** no `rfx/` change; `adq_designvar.py` and
  `e4_diff_stackup.py` unedited; the two existing replay tests pass before
  (31 passed, 1 xfailed) and after; the new replay test re-derives every
  verdict, fit, bar, coverage and map check from the JSONs without FDTD.
- **Coverage (E6-C, reported, no minimum):** `coverage(g_cell, J)` with
  `J` the 60 x 2 exact map Jacobian and `g_cell` the AD gradient over the
  60 x cells (strip fixed by index), for all controls, for the controls
  passing BOTH gates, per control, and the dimensionless variant (nominal
  cell widths as scales); plus the norm split lead / band / tail.

Reading rules for INCONCLUSIVE (AD-Q verbatim): a direction whose order
fit has no eligible run of four points, or whose FD budget has no
informative step, is INCONCLUSIVE — recorded as such, never widened into
HELD; the `3B/|g|` at the narrowest informative step and the count of
eligible points are reported so the reader sees why.

Expectations before the run (program 5.4, copied): E6-L1 `w` HELD on both
gates (AD-Q h_thin R1 2.047, 3B 2.8e-4; the E6 fixture's dt share is
larger since the tied set is 20 of 60 cells); `x_c` HELD (a pure shift
changes the lead/tail free cells by 3.3 % at the ladder top; small
gradient, so INCONCLUSIVE on order is the likeliest non-HELD outcome, as
eps_air was); E6-R `w` FIRED, `x_c` unchanged; E6-L2s `w` HELD on order
with the physical-time Hann window.

*Reviewer note (second pass, 2026-09-14; paragraph unchanged):* this
paragraph is labelled "program 5.4, copied" but is not verbatim: program
5.4 says "0.125 mm / 8 mm = 1.6 % at the ladder top"; the 3.3 % above was
written into this note at `6fec85f5`, before any run, from section 1c's
7.5 mm free length (0.25 mm / 7.5 mm = 3.3 %). A pre-run rewording, not a
post-measurement edit; the windows in this section match program 5.4
verbatim. Recorded so that "copied" is not read as "identical".

## 4. Arms and run commands (declared; each arm one Bash call, one attempt)

From the worktree, `PYTHONPATH` pinned to it, `python -m
validation.research.multiband_nu.e6_inplane_designvar --arm ARM`, writing
`results/e6_ARM.json` (a `.started` claim next to it refuses a rerun):

1. `model` — zero FDTD; this note's numbers (done in this commit).
2. `map_checks` (E6-M) — m1-m8; two L1 solves (m8) and two L1 gradients
   (m5); `results/e6_map_checks.json`.
3. `l1` (E6-L1) — floor (64 runs) + ladder (30 runs) per control, one
   cell-space and one parameter gradient; `results/e6_l1.json` (the floor
   is stored inside it under each direction, not in a separate file).
4. `revert_l1_nodt` (E6-R) — true L1 ladder (30 runs per control) judged
   against the no-dt gradient; `results/e6_revert_l1_nodt.json`.
5. `l2s` (E6-L2s) — as 3 on the spectral observable, checkpointed;
   `results/e6_l2s.json` (floor inside).
6. `coverage` (E6-C) — zero FDTD, from the two JSONs above;
   `results/e6_coverage.json`.

Order: 2, 3, 4, 5, 6. Declared run counts: E6-M 2, E6-L1 188, E6-R 60,
E6-L2s 188 -> 438 FDTD runs plus 6 gradients; program estimate under 5
CPU minutes.

## 5. Model predictions (frozen; from `results/e6_model.json`, zero FDTD)

- `d(dt)/dw` analytic = **1.4711225845146326e-10 s/m** (`d(dt)/d(dx_min)`
  x `d(dx_min)/dw`, `d(dx_min)/dw = 0.1 mm / 2 mm = 0.05`); the review's
  1.42632e-10 was computed with a different transverse term. m6 compares
  the float32 AD value to this number.
- `dt / dt(p0)` along `w`: 0.8827 at h = 2^-3 shrink (program: "at most
  12.5 %"; measured from the map 11.7 %), 1.1141 at h = 2^-3 growth;
  exactly 1.0 along `x_c` at every point.
- Tied set: the 20 band cells at every one of the 60 ladder points of both
  controls and both signs (growth side at h = 2^-3: band 1.125e-4 m vs
  first ramp cell 1.2656e-4 m).
- Column length error (float64 map): <= 6.9e-18 m over the ladder.

### Table L — seam ratio and dt per ladder point (float64 map; not a window)

| h | `w` shrink: seam lo = hi | `w` growth: seam | `w` shrink dt/dt0 | `w` shrink N2 dt / T_w | `x_c` +: seam lo / hi |
|---|---|---|---|---|---|
| 2^-3 | 1.4955 | 1.1250 | 0.8827 | 1.1034 | 1.3300 / 1.2442 |
| 2^-4 | 1.3843 | 1.2013 | 0.9417 | 1.1772 | 1.3085 / 1.2656 |
| 2^-5 | 1.3341 | 1.2429 | 0.9710 | 1.2137 | 1.2978 / 1.2764 |
| 2^-6 | 1.3102 | 1.2646 | 0.9855 | 1.2319 | 1.2925 / 1.2817 |
| 2^-7 | 1.2986 | 1.2758 | 0.9928 | 1.2409 | 1.2898 / 1.2844 |
| 2^-17 | 1.2871 | 1.2871 | 1.0000 | 1.2500 | 1.2871 / 1.2871 |

Above the 1.3 in-plane advisory cap: `w` shrink from h = 2^-6 upward,
`x_c` (lead side) from h = 2^-6 upward. The traced path does not evaluate
the cap (G20); the numbers bound the range of any claim.

## 6. What the lane can and cannot claim (program 5.7, written before the run)

Can: "AD gradient along a fine-band width and position on x, with the
band edges carrying a PEC strip attached by node index, verified by Taylor
order and FD-inside-3B on a transient and on a smooth spectral observable,
dt path included, on the closed vacuum fixture of section 2, over relative
displacements [2^-17, 2^-3] of a 2 mm band" (or whichever sub-range the
windows hold on). Cannot: anything about coordinate-rasterized geometry
(`Box` gradient 0), about ports (host-coerced on traced x/y), about an
absorber (cpml 0), about y (the same map applies by relabeling but is not
measured here), or about `interface_eps='dual_average'` under trace.

## Results (appended after measurement; no window above changed)

### Measured (instrument `6fec85f5`, tests `bd1ea262`, measurements `60781ccc` .. `4f2d22af`; no window above changed)

Every measured arm ran once, on this tree, CPU, `rfx.__file__` under
`rfx-nu-full`, `git_dirty` false (E6-M at `bd1ea262`, E6-L1 at `60781ccc`,
E6-R at `fa2eae4d`, E6-L2s at `17ece761`, E6-C at `dd24ec3a`). The judge
self-check passed before every arm and is bit-identical to the self-check
stored in all five AD-Q JSONs (R1 2.0030105652046206 HELD /
0.9094336472889952 FIRED). 442 forward solves and 8 reverse-mode gradients
in total (second-pass correction: the count is not stored in any JSON; it
is the instrument's call structure read against the stored arrays — E6-M
2 solves (m8 traced + concrete) + 2 gradients (m5); E6-L1 `loss0` + 2 x
(64 floor + 30 ladder) = 189 solves + 2 gradients; E6-R `loss0` + `nodt0`
+ 2 x 30 = 62 solves + 2 gradients; E6-L2s 189 + 2; total 2 + 189 + 62 +
189 = 442 and 8; the declared 438 + 6 in section 4 omitted the four
nominal evaluations and E6-M's two gradients. The post-hoc diagnostic's
two further gradients are outside this tally); wallclock E6-M 6.4 s,
E6-L1 13.5 s, E6-R 5.8 s, E6-L2s 63.9 s, E6-C zero FDTD (each
`elapsed_seconds`). Raw JSONs: `results/e6_{model,map_checks,l1,
revert_l1_nodt,l2s,coverage}.json` plus the post-hoc diagnostic
`results/e6_diag_revert_cellwise.json` (declared below, zero ladder
runs; see the "Second pass" section for its producer — the first pass was
a scratch script and is kept as
`e6_diag_revert_cellwise_firstpass_4f2d22af.json`).
Replay: `tests/unit/nonuniform/test_e6_inplane_designvar_replay.py`.

### Regression (program 5.5) — HELD

`git diff --stat rfx/` empty at every commit; `adq_designvar.py` and
`e4_diff_stackup.py` unedited; the two existing replay tests 31 passed /
1 xfailed before and after; self-check bit-identical (above).

### E6-M — all eight map checks pass

| check | measured | threshold |
|---|---|---|
| m1 column length | 6.9e-18 m (max over 60 ladder points) | 1e-9 m |
| m2 interfaces on nodes | 2.4e-17 m | 1e-9 m |
| m3 tied-set spread | [0.0, 0.0]; float32 tied set = cells 20..39 at all 60 points | exactly 0.0 |
| m4 boundary Jacobian | [[0, 0], [0, 0]] float64 and float32 AD | exactly 0.0 |
| m5 `J^T g` vs direct (L1) | w 1.53e-7, x_c 7.27e-7 relative | 1e-5 |
| m6 `d(dt)/dw` AD vs analytic | 1.4711222518e-10 vs 1.4711225845e-10 s/m: 2.26e-7; `d(dt)/dx_c` exactly 0.0 | 1e-6 |
| m7 strip | 21 `ey` node columns, 20 `ex` edges; `pec_edge_masks` accepted | 21 |
| m8 traced vs concrete L1 at p0 | 4.584793467e-3 vs 4.584793933e-3: 1.02e-7 (dt 3.17762472e-13 float32 vs 3.17762478e-13 float64) | 1e-6 |

### E6-L1 (transient probe energy, 600 steps)

`loss0` = 4.584793467074633e-3, ulp 4.66e-10. `g_param` = (-6.031999588,
+0.0273133516) per metre; chain rule `J^T g_cell` = (-6.0320005,
+0.0273133318). Floors (sigma in ulp of loss0, both cubic-reliable): w
2.84, x_c 1.86 — so `sigma_eff` is the measured sigma for both.

| control | R0 | R1 | pts | window (rel. h) | order | FD (informative steps) | narrowest 3B/\|g\| (h, rho) | max err/3B |
|---|---|---|---|---|---|---|---|---|
| **w** | 1.038 +- 0.007 | **1.945** +- 0.012 | 5 | 4.9e-4 .. 7.8e-3 | **HELD** | **HELD** (4: 9.8e-4 .. 7.8e-3, rho 2.33 / 4.31 / 3.88 / 3.79) | **2.0e-3** (9.8e-4, 2.33) | 0.335 |
| x_c | — | — | 1 | — | INCONCLUSIVE | HELD (2: 7.8e-3, 1.6e-2; rho 3.69 / 3.63) | 2.4e-2 (7.8e-3, 3.69) | 0.312 |

1-ulp verdicts (reported alongside): identical for w (same 5-point
window); x_c order INCONCLUSIVE with 2 eligible points. x_c: its gradient
(0.0273 /m) is 220x smaller than w's and is the difference of a lead
contribution +1.1678 and a tail contribution -1.1405 (diagnostic JSON);
the order gate is INCONCLUSIVE — the outcome section 3 named as the
likeliest non-HELD one. The FD gate holds at two rho-valid steps.
Second-pass correction of the reason: the first-pass text said "R1
clears 32 sigma_eff at one ladder point only (h = 2^-7)". The stored
ladder says otherwise: `r1 / (32 sigma_eff)` = 2.86, 11.8, 49.7, 218,
1037 at h = 2^-7 .. 2^-3 — FIVE points clear the 32-quantum line; four
are excluded by the judge's curvature guard `|L+ + L- - 2 L0| <= 0.25
|L+ - L-|` (ratios 0.18 at 2^-7, then 0.36, 0.68, 1.11, 1.31). So x_c on
L1 is not noise-limited: the loss along x_c is quadratic-dominated
(the direct consequence of the 43x cancellation — the linear term is
small, the curvature is not), and an order test along x_c is
structurally unavailable on this fixture at this ladder; a lower floor
would add no eligible point. For L2s x_c the first-pass reason IS the
binding one (the 32-quantum line: `r1 / (32 sigma_eff)` 0.96 at 2^-6,
curvature ratio <= 0.004 everywhere). Verdicts unchanged.

### E6-R — the gate has power against the dt path

Forward values with and without `stop_gradient(dt)` identical; the true
ladder bit-identical to E6-L1's; `g_true` bit-identical to E6-L1's.
dt-path share: **w 0.5884** (AD-Q h_thin: 0.1852), x_c -1.48e-4 (see the
diagnostic). w judged against the no-dt gradient (-2.4831 vs true
-6.0320): order **FIRED**, R1 slope **1.022** (11 points, 7.6e-6 ..
7.8e-3), R0 1.013; FD **FIRED** at all three informative steps (err/3B
294 / 145 / 35). x_c: INCONCLUSIVE / HELD — the same verdicts as E6-L1.
Declared requirement met (`requirement_met` true): E6-L1's w verdict
covers the dt path.

**Diagnostic (post-hoc, labelled, zero ladder runs;
`results/e6_diag_revert_cellwise.json`, two backward passes at
`4f2d22af`; second-pass correction: that first pass was produced by a
scratch script with no producer on the tree and no `.started` claim — it
is kept as `e6_diag_revert_cellwise_firstpass_4f2d22af.json`, and the
file named here is now written by the committed arm `--arm
diag_revert_cellwise`, declared and measured in the "Second pass"
section):** the program's 5.3 text expected x_c's dt share to be
"exactly 0". Measured -1.48e-4. Cell-wise: the tied cells carry the dt
path (mean difference -3.549 per cell, spread 2.5e-6); the NON-tied cell
gradients differ between the two compiled programs by up to 7.4e-6
absolute / 2.0e-5 relative (not zero), and the chain-rule x_c share
-1.47e-4 is those differences amplified by the 43x lead/tail
cancellation. Cause class: `stop_gradient(dt)` changes the compiled
reverse program (dt enters the source waveform, the update coefficients
and the CFL here; AD-Q's L1 waveform was step-indexed), so float32
accumulation order differs at the 1e-5 level. AD-Q's "exactly 0" for the
core controls was a property of that fixture's compilation, not a
guarantee. The frozen requirement is on the verdicts (unchanged); the
expectation "exactly 0" is recorded as not met and re-read as "zero within
float32 recompilation". No window changed.

### E6-L2s (smooth spectral observable, 2400 steps, checkpoint 100)

`loss0` = 3.031254431866461e-24 (V^2 s^2 per node, comb-summed), ulp
1.97e-31. `g_param` = (-3.666778564e-21, +7.333234819e-22) per metre;
chain rule (-3.666778828e-21, +7.333234332e-22): 7.2e-8 / 6.6e-8
relative. Floors: w **14.08 ulp** (cubic/quadratic RMS 0.988, reliable),
x_c 1.95 ulp (reliable).

| control | R0 | R1 | pts | window (rel. h) | order | FD (informative steps) | narrowest 3B/\|g\| (h, rho) | max err/3B |
|---|---|---|---|---|---|---|---|---|
| **w** | 0.950 +- 0.011 | **1.962** +- 0.011 | 6 | 3.9e-3 .. 1.25e-1 | **HELD** | **HELD** (4: 7.8e-3 .. 6.25e-2; rho 2.91 / 4.07 / 4.00 / 4.08) | **7.2e-4** (7.8e-3, 2.91) | 0.332 |
| x_c | — | — | 3 | 3.1e-2 .. 1.25e-1 | INCONCLUSIVE | HELD (3: 1.6e-2 .. 6.25e-2; rho 4.62 / 3.47 / 4.04) | 1.2e-4 (1.6e-2, 4.62) | 0.332 |

1-ulp verdicts (reported alongside, NOT the primary verdict): w HELD with
8 points (9.8e-4 .. 1.25e-1), R0 0.966, R1 2.007; x_c order **HELD** with
4 points (1.6e-2 .. 1.25e-1), R0 0.998, R1 2.134, FD HELD (3). With the
measured floor (1.95 ulp) the fourth point of x_c drops below the
32-quantum line and the primary verdict is INCONCLUSIVE by the declared
rule (1f); it is recorded as such.

Smoothness, as measured: the Richardson ratio sits in [2, 8] at four
consecutive steps for w and three for x_c (7.8e-3 .. 6.25e-2), and the FD
bar shrinks to 7.2e-4 / 1.2e-4 of the gradient — the AD-Q L2 thickness
arm (`adq_stack_l2_floor.json`, h_thin), by contrast, had a 43.6-ulp
floor, an INCONCLUSIVE order fit, and an FD gate that FIRED at h 3.1e-2
(err/3B 3.96) after two rho-valid HELD steps at 1.95e-3 and 3.9e-3 (rho
3.23 / 2.63) with no rho-valid step between (second-pass correction: the
first-pass text said "no rho-valid window", which is true of that arm's
order fit only, not of its FD gate). The physical-time Hann taper with
a fixed `T_w` removes the truncation ripple, as declared in section 2.

### E6-C — coverage of the x cell-space gradient (no minimum assumed)

| arm | all controls (rank 2) | verified (w) | w only | x_c only | dimensionless | norm total / band / lead / tail |
|---|---|---|---|---|---|---|
| L1 | **0.8023** | **0.8023** | 0.8023 | 0.0030 | 0.3726 | 26.87 / 21.81 / 14.22 / 6.64 |
| L2s | **0.6803** | **0.6711** | 0.6711 | 0.1116 | 0.3835 | 1.953e-20 / 1.632e-20 / 1.006e-20 / 3.68e-21 |

The band cells hold 81 % (L1) / 84 % (L2s) of the gradient norm; the
width direction captures 80 % / 67 % of the whole; the position direction
adds 0.3 % / 1.1 % (AD-Q stack: 0.8875 / 0.8830 with seven controls).

### Falsifier ledger

| falsifier | w, L1 | x_c, L1 | w, L2s | x_c, L2s |
|---|---|---|---|---|
| Order R0 in [0.9, 1.1], R1 in [1.8, 2.2] | HELD (1.038, 1.945) | INCONCLUSIVE (1 pt) | HELD (0.950, 1.962) | INCONCLUSIVE (3 pts) |
| FD inside 3B at every rho-valid step | HELD (4 steps, 3B/\|g\| 2.0e-3) | HELD (2 steps, 2.4e-2) | HELD (4 steps, 7.2e-4) | HELD (3 steps, 1.2e-4) |
| Revert-proof (L1): w FIRED, x_c unchanged | FIRED as required (R1 1.022) | unchanged | — | — |
| Map checks m1-m8 | all pass | | | |

### Law statement and validity domain, as measured

On the closed vacuum NU fixture of section 2 (x graded 0.5 / 0.1 / 0.5 mm
at cap 1.3 with the end cells pinned, a PEC strip attached by node index
on the 21 band-edge nodes, source and probe by index), the reverse-mode
gradient of both a transient loss and the physical-time-Hann comb-
integrated spectral power along the fine-band WIDTH `w` — the CFL dt path
included, 58.8 % of that gradient — is second-order correct (R1 1.945 /
1.962) and agrees with central FD inside the FD's own 3B bar at every
Richardson-valid step, 3B/|g| down to 2.0e-3 (transient) and 7.2e-4
(spectral); the same judges FIRE (R1 1.022, err/3B 294) when the dt path
is removed. Validity domain: relative width displacements 4.9e-4 ..
7.8e-3 (L1 order), 9.8e-4 .. 7.8e-3 (L1 FD), 3.9e-3 .. 1.25e-1 (L2s
order), 7.8e-3 .. 6.25e-2 (L2s FD) of a 2 mm band — on x only. Seam
ratio reached, PER GATE AND OBSERVABLE (second-pass correction; Table L,
shrink side): L1 order and L1 FD both end at h = 2^-7, seam **1.2986** —
BELOW the 1.3 in-plane advisory, so the transient arm verified nothing
above the cap (from 2^-6 the transient loss is quadratic-dominated along
w: the curvature guard fails at 0.31 / 0.73 / 1.28 / 0.63 for 2^-6 ..
2^-3 and 3B/|g| is 0.245 at 2^-6 — not eligible, not informative); L2s
FD ends at h = 2^-4, seam **1.3843**; L2s
order alone reaches h = 2^-3, seam **1.4955**, as the top point of a
6-point slope fit. "Seam ratio to 1.50 with both gates holding" is
therefore not a statement any single gate/observable pair supports; the
widest BOTH-gates range above the cap is L2s over 2^-6 .. 2^-4 (seam
1.3102 .. 1.3843). Along the band POSITION `x_c`, the FD gate holds at
Richardson-valid steps (3B/|g| 2.4e-2 / 1.2e-4) on both observables, but
the order gate is INCONCLUSIVE on both (1 and 3 eligible points; the
gradient is a 43x cancellation of lead and tail contributions): `x_c` is
FD-supported and NOT order-verified by this lane. The design directions
span 0.80 (L1) / 0.68 (L2s) of the cell-space gradient norm; the verified
direction alone 0.80 / 0.67.

Not covered (program 5.7, unchanged): coordinate-rasterized geometry,
ports, an absorber, the y axis, `interface_eps='dual_average'` under
trace, seam ratios beyond 1.4955, and any x_c order claim.

### What the support matrix may say after this lane

(Second-pass correction: the first-pass sentence read "over relative
displacements up to 2^-3 (seam ratio to 1.50) ... crossed it from h =
2^-6 with both gates holding on w"; that collapsed four per-gate windows
into a union no single gate supports — see the law statement above.)
"In-plane design-variable AD (x): band width verified by AD-Q judges on a
transient (both gates, relative displacements 9.8e-4 .. 7.8e-3, seam
ratio to 1.30) and a smooth spectral observable (both gates 7.8e-3 ..
6.25e-2, seam ratio to 1.38; order alone to 1.25e-1, seam 1.50), dt path
included; band position FD-supported, order INCONCLUSIVE; PEC strip by
node index; no absorber; no coordinate attachment." The 1.3 in-plane cap
remains an advisory the traced path does not evaluate (G20); only the
spectral observable's gates hold above it (L2s both gates over 2^-6 ..
2^-4, seam 1.31 .. 1.38); the transient arm's gates end at 2^-7 (seam
1.2986), below the cap.

## Second pass (review of the record, 2026-09-14)

Seven review findings, each reproduced from the stored JSONs or from
`git show` on this tree before anything was edited (`rfx.__file__` under
this worktree; scratch scripts `rv_inspect.py` and the review's
`rv_model.py` / `rv_ladder.log` in the session scratchpad, not part of
the record). No window was edited; no verdict changed; no ladder arm was
re-attempted (no finding showed a ladder measurement to be invalid). One
post-hoc diagnostic is re-produced through a committed arm (declared
below) because its first pass had no producer on the tree. `stopped`
stays false.

### What was corrected in the text above (each marked "second-pass correction" or "Reviewer note" in place)

| # | finding (review lens) | what the record said | what the stored numbers say | where |
|---|---|---|---|---|
| 1 | validity-domain compression (physics, major) | "up to 2^-3 (seam ratio to 1.50) ... crossed the cap from h = 2^-6 with both gates holding on w" | L1 both gates end at 2^-7 (seam 1.2986, below the 1.3 advisory; from 2^-6 the curvature guard fails 0.31 .. 1.28 and 3B/\|g\| 0.245); L2s FD ends at 2^-4 (1.3843); only the L2s order fit reaches 2^-3 (1.4955), as the top of a 6-point slope | "Law statement and validity domain"; "What the support matrix may say" |
| 2 | x_c L1 INCONCLUSIVE mis-attributed to the 32-sigma rule (physics, minor) | "R1 clears 32 sigma_eff at one ladder point only" | five points clear 32 sigma_eff (2.86 .. 1037); the curvature guard admits only 2^-7 (0.18; then 0.36 / 0.68 / 1.11 / 1.31) — quadratic-dominated direction, not noise-limited; L2s x_c IS 32-quantum-limited (0.96 at 2^-6) | E6-L1 paragraph |
| 3 | pre-declared replay assertion `share["x_c"] == 0.0` rewritten after measurement (record, major) | the declared line was deleted at `064d6035` and replaced by the measured value; between `17ece761` and `4f2d22af` the committed test failed on the committed JSON | the declared line is restored verbatim under `xfail(strict=True)` (it fails, as measured, and the failure is the record); the measured share is pinned in a separate record test | `test_e6_inplane_designvar_replay.py` |
| 4 | cell-wise diagnostic JSON has no producer on the tree (record, major) | "two backward passes at 4f2d22af" | produced by a scratch script; no arm, no `.started`, no `argv` | committed arm `--arm diag_revert_cellwise`, declared below; first pass kept as `e6_diag_revert_cellwise_firstpass_4f2d22af.json` |
| 5 | "program 5.4, copied" is not verbatim (record, minor) | 3.3 % | program 5.4 says 1.6 %; the 3.3 % is a pre-run rewording from section 1c | Reviewer note under section 3's expectations |
| 6 | AD-Q L2 thickness arm "no rho-valid window" (record, minor) | no rho-valid window | order INCONCLUSIVE, but the FD gate had two rho-valid HELD steps (1.95e-3, 3.9e-3) before the FIRE at 3.1e-2 | E6-L2s "Smoothness" paragraph |
| 7 | "442 forward solves and 8 gradients" has no stored source (record, minor) | 442 / 8 | re-derived from the instrument's call structure: 2 + 189 + 62 + 189 = 442, 2 + 2 + 2 + 2 = 8; the declared 438 + 6 omitted the nominal evaluations and E6-M's gradients | "Measured" |

Finding 1's physical content: the transient arm's verification of `w`
never crossed the 1.3 in-plane advisory; what crossed it is the smooth
spectral observable (both gates over 2^-6 .. 2^-4, seam 1.31 .. 1.38).
Finding 2's: an order test along `x_c` is structurally unavailable on
this fixture — the direction is quadratic-dominated because of the
lead/tail cancellation, so a lower floor would not add eligible points.

### Declared diagnostic arm: `diag_revert_cellwise` (declared here BEFORE it ran through the instrument)

Purpose: give `results/e6_diag_revert_cellwise.json` a committed
producer. Not a window; not a ladder arm; no verdict is computed from it.
The arm (`e6_inplane_designvar.py::diag_revert_cellwise`) takes two
cell-space L1 gradients at the nominal (`jax.grad(l1_cells)` with and
without `stop_gradient(dt)`) and two parameter-space gradients (the same
two calls as E6-R), zero forward ladder runs, and derives the same
fields as the first pass: non-tied max |diff| and relative diff, tied
mean and spread, chain-rule shares, lead/tail contributions. It runs
under the instrument's `.started` claim and stamps `argv`, `git_sha`,
`git_dirty`, `rfx.__file__`. The first pass (scratch script,
`4f2d22af`) is renamed `e6_diag_revert_cellwise_firstpass_4f2d22af.json`
(content untouched) and the arm records whether it reproduces it.
Expectation (one attempt): `g_cell_true` bit-identical to `e6_l1.json`;
`param_ad_true` / `param_ad_nodt` bit-identical to `e6_revert_l1_nodt.json`;
`g_cell_nodt` and every derived scalar bit-identical to the first pass
(the code that computes them is unchanged since `4f2d22af`: `git diff
4f2d22af -- rfx/ validation/research/multiband_nu/e6_inplane_designvar.py`
touches only the new arm and the docstring). If any of these is false
the mismatch is reported as a finding, not tuned. Replay:
`test_revert_cellwise_diagnostic_replay` (both files, bit-identity
between them, the shares re-derived from the arrays). Command: `python
-m validation.research.multiband_nu.e6_inplane_designvar --arm
diag_revert_cellwise`.

### `diag_revert_cellwise` — measured (one attempt, `99b510b3`, `git_dirty` false, 07:22:21Z, two cell-space + two parameter-space gradients, zero ladder runs, 2.35 s CPU)

| expectation | measured |
|---|---|
| `g_cell_true` bit-identical to `e6_l1.json` | True |
| `param_ad_true` / `param_ad_nodt` bit-identical to `e6_revert_l1_nodt.json` | True / True |
| `g_cell_nodt` bit-identical to the first pass | True (all 60 cells) |
| derived scalars bit-identical to the first pass | 7 of 9: `nontied_max_abs_diff` 7.3909759521484375e-06, `nontied_max_rel_diff` 1.967644921421007e-05, `tied_diff_mean` -3.548943030834198, `tied_diff_spread` 2.5033950805664062e-06, `xc_lead_contrib_true` 1.1678033525465108, `xc_tail_contrib_true` -1.1404900207734423, `param_dt_share` [0.5883520737516621, -0.0001483932070827205] all equal; **`chain_w_share` 0.5883521473691763 vs first pass ...762 (1 ulp); `chain_xc_share` -1.4699955535986395e-4 vs -1.469995553598647e-4 (28 ulp)** |

The expectation "every derived scalar bit-identical" did NOT hold for
the two chain-rule shares; recorded, the arm not re-run. Cause,
identified after the run by evaluating both forms on the stored arrays
(scratch, no FDTD): the first pass divided `J[:, i] . diff` by the column
dot `J[:, i] . g_true`; the committed arm divides by `chain_true[i]`
taken from the matmul `J^T g_true` (the field it also stores) — the same
quantity, reduced in a different order, so the denominators differ in
their last float64 bits. The `w` share moves 1 ulp; the `x_c` share moves
28 ulp because its denominator 0.0273 is the residue of the 1.1678 -
1.1405 cancellation (relative amplification 43x on a 1-ulp denominator
change). Both files re-derive from their own arrays inside 1e-12
(`test_revert_cellwise_diagnostic_replay`, both files); the second-pass
test pins the arrays' bit-identity, the seven equal scalars, and the two
share differences as measured (1 and 28 ulp). Nothing in any verdict or
in the note's text depends on the digits beyond -1.47e-4 / 0.5884.
