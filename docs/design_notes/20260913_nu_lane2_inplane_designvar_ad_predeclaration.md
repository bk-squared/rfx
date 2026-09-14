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
