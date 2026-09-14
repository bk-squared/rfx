# NU full functionality — gap map and ordered lane program

**Status:** program document (planning). No code, no measurement in this
commit. Lanes 1 and 2 are fully specified here: their instruments, arms,
frozen numeric windows, regression requirements and costs are declared in
this document, and the executors act on it alone. Every window in sections
4 and 5 is frozen from this commit onward; results are appended to each
lane's own pre-declaration note, never edited into these windows. One
attempt per declared arm; a fired window is a result. Lanes 3+ are planned,
not executed, and each states the PI decision it waits on.
**Tree:** worktree `rfx-nu-full`, branch `feat/nu-full-functionality`, based
on `feat/nu-ad-directional @ 88460d24` (on `origin/main fa392913`).
**Import provenance for every number cited:** the three reviews that feed
the gap map ran on this tree with `rfx.__file__` under
`/Users/byungkwankim/Documents/rfx-nu-full/rfx/__init__.py`, CPU only.
Every JSON a lane writes records `rfx.__file__`, `git_sha`, `git_dirty`,
`argv`, `started_utc` (the `run_e1` provenance block,
`w6_band_builder.py:693-703`).
**PI rules applied throughout:** physical fidelity first; measure the
validity domain (a law with its range), never a single fixture; one lane at
a time; experimental features stay opt-in (no E2/E3-style default moves;
`_MULTIBAND_RATIO_CAP = 1.4`, `_INPLANE_RATIO_CAP = 1.3` unchanged); a
gradient claim goes through the AD-Q judges, never a flat percentage.
Date: 2026-09-14 (KST).

## 0. Target, stated so it can be falsified

"NU full functionality" means, on every one of the three axes:

- (T1) arbitrary per-axis band patterns: several fine sizes, several coarse
  sizes, asymmetric ramps (the two sides of a band at different coarse
  sizes, or a column that ends on a size it did not start on), in one
  column, with every adjacent ratio at or under the cap;
- (T2) the transition law with its range on such patterns — reflection per
  transition, bound on the sum, phase of the Fabry-Perot form — measured
  on x, y and z, not argued from z;
- (T3) autodiff along PHYSICAL design variables (a band's width and
  position, a layer's thickness, a permittivity) on every axis, judged by
  Taylor order and FD-inside-its-own-error-bar, including the CFL `dt`
  path through the tied minimum cells of that axis;
- (T4) the consumers that read a scalar `grid.dx`/`grid.dy` (CPML x/y
  faces, waveguide-port injection, TFSF, preflight tolerances) reading the
  per-cell size instead, or refusing loudly;
- (T5) an in-plane builder entry point (fine band around a feature, not
  edge fitting) and a multi-level policy, opt-in;
- (T6) a differentiable builder in `rfx/` (opt-in) so that (T3) does not
  require a hand-written map per fixture.

What is ALREADY on this tree is cited in section 2 and is not re-derived.

## 1. Gap map (deduplicated from the evidence-map, code-assumptions,
## autodiff-surface and harness-reuse reviews; each with its pointer)

Severity is the reviews' own. "Blocks" names the lane the gap blocks.

| # | gap | evidence pointer (this tree, 88460d24) | severity | blocks |
|---|---|---|---|---|
| G1 | An in-plane profile whose two end cells differ cannot be constructed at all: `make_nonuniform_grid` requires `dx_profile[0] == dx_profile[-1] == dx` and `dy_profile[0] == dy_profile[-1]` unconditionally, even PEC-closed with `cpml_layers=0`; z has no such contract. Live: `make_band_profile([0,6,9,12] mm, [1,0.5,0.5] mm, 1.4)` builds on z (nz 20), raises as `dx_profile` ("must equal boundary dx") and as `dy_profile` ("boundary cells must match each other"). The tracer path skips the check and the CPML still reads the scalar. | `rfx/nonuniform.py:354-378` (x), `:381-401` (y), `:357-362, 386-391` (tracer); `rfx/boundaries/cpml.py:174-197` (`_get_axis_cell_sizes` returns `float(grid.dx)`, `float(grid.dy)`, but `dz_lo`, `dz_hi`) | blocking (T1 on x/y) | Lane 1 (worked around by end pins, section 4.2), Lane 3 (fixed) |
| G2 | CPML x/y face cell size is one scalar per axis on both faces; `CPMLAxisParams` already carries `dx_x_lo/dx_x_hi/dx_y_lo/dx_y_hi` slots; z reads `dz[0]`/`dz[-1]`. On an asymmetric x profile the x_hi sigma/kappa scaling would use the x_lo cell. | `rfx/boundaries/cpml.py:174-197, 462, 488-502, 641-643, 926-928`; repro R4 (init_cpml on a z-asymmetric grid: `dx_x_lo = dx_x_hi = 0.001`, `dz_lo = 0.000981`, `dz_hi = 0.000458`) | blocking (T4) | Lane 3 |
| G3 | No witness harness can build or excite an in-plane-graded fixture for the F-S1/F-S2/F-S3 class: `build_pec_fixture(dz_profile, domain_xy, dxy: float)`; `te10_sources` loops `i` over `nx-1`, sine over `j` with `float(grid.dy)`, emits `'ex'` at slot `k`; `_run_probe`/`run_probe` probe `(1, ny//2, k, 'ex')`; `e1_arm` reads `float(grid.dy)` and `planes_ok` vs `coarse_cell_m`; `e1_gates` places planes at `k * coarse_cell_m`. The chain model is 1-D in the graded axis with a scalar sine-axis spacing. | `harness.py:24-37, 40-62`; `w2_w3_reflection.py:40-51, 66-74`; `w6_band_builder.py:226-238, 260, 509-518, 555-559`; `chain_model.py:1-25, 34-39, 122-135` | blocking (T2 on x/y) | Lane 1 |
| G4 | No accuracy witness on ANY axis for multi-level fine or coarse sizes or asymmetric traversals: every F-S1..F-S4/W6/E1/A1-A3 fixture is one fine + one coarse with symmetric ramps. The only multi-level FDTD arm is W7 A1 "AZ" (reported only, no order). F7 (PCB, 46 steps) is chain-model only. The E1 gate slice hardcodes one ramp cell per side (`prof[n_lead : n_lead+2+n_b]`); expected vectors hardcode `[c]*n + [r d] + [d]*n_b + [r d] + [c]*n`; the B reference pins dt with the DECLARED fine cell while a protected band realizes `span/n` (measured 0.5000000000000004 mm for a declared 0.5 mm band); `e1_fit_c` assumes two EQUAL ramp reflections. | `fixtures.py:54-68, 84-89`; `w4r3_zdominant_cavity.py:134-141`; `w6_band_builder.py:217-222, 270-271, 388-412, 415-433, 515-518`; A1 AZ rows in `20260907_nu_band_accuracy_ad_predeclaration.md`; live check: a 5-segment asymmetric z profile with 14 distinct sizes builds (max ratio 1.357, nz 42) and `chain_model.scattering` evaluates it (|R| 1.10e-3, |T| 0.998) — nothing has measured it | major (T1, T2) | Lane 1 |
| G5 | The builder's geometric ramp is `m = ceil(ln(ratio)/ln(cap))` cells at `rho = ratio^(1/m)` (NOT cap per step), and a free segment whose span is not exactly `ramp_sum + n * target` gets a SOLVED plateau below target (measured: mis-declared middle span, 1.96 -> 0.5 mm, plateau 1.9558 mm, ramp 1.4888/1.1334/0.8628/0.6568 mm, rho 1.314). A multi-level arm must declare spans by the same rule or the model is evaluated on a different structure. | `rfx/nonuniform.py:612-625` (`_ramp_steps`), `:1165-1170`; harness-reuse smoke | major (instrument) | Lane 1 (rule in 4.2) |
| G6 | The only in-plane accuracy evidence is one witness class: W7 A2 (x and y, two 0.25 mm fine bands in ~1 mm coarse, caps 1.3 AND 1.4, vacuum TM110, |err| <= 0.0275 %) and A3 (all three axes, TM111, err -0.0350 %) — resonance vs analytic at one scale; not energy, not per-transition, not round-trip, not an order fit. `support_matrix.md` still says in-plane is "UNCOVERED" and "no in-plane provenance for the 1.3 lock" and cites neither A2 nor A3 (no occurrence of TM110/TM111/w7). | `results/w7_accuracy_ad.json` a2.judge_ext / a3.judge_ext; `docs/guides/support_matrix.md:100, 136-150, 208-224` | major (docs stale, understating) | standing item S1 (section 6) |
| G7 | No design-variable gradient has been judged by the AD-Q judges on an in-plane axis. E4/AD-Q are z-only (20/4/20/20 stack, transverse uniform 0.5 mm). The only AD-Q-grade in-plane result is the resolution arm's whole-axis dilation (y R1 1.990 / 7 pts, z R1 1.882 / 6 pts, HELD) — a box-width control, not a band width or position; x has no Taylor arm. L2 thickness is NOT verified (resonant DFT power rippled ~1 % in thickness). | `adq_designvar.py:261`; `e4_diff_stackup.py:59-62, 174`; `20260913_nu_ad_designvar_predeclaration.md` "Attempt-1 resolution arm", "What the NU autodiff claim is" | major (T3) | Lane 2 |
| G8 | No general differentiable map from edges/widths to a per-axis cell vector: `make_band_profile` is host numpy; coordinate `Box` rasterization is argmin (`mask_on_coords` gradient exactly 0 on a traced axis, measured on z by E4 and on x by the review); the only design-variable path is the hand-written E4 map. `interface_eps='dual_average'` refuses traced profiles, so second-order interfaces and mesh-AD are exclusive on the production path. | `20260907_nu_band_accuracy_ad_predeclaration.md` section 0; `e4_diff_stackup.py:106-128`; `adq_designvar.py:36-45`; `rfx/geometry/csg.py:289-356`; `rfx/runners/nonuniform.py:33-47` | major (T3, T6) | Lane 2 (map in validation/research), Lane 9 (in rfx/) |
| G9 | On a traced in-plane profile, `position_to_index` falls back to a UNIFORM nominal mesh at the boundary cell (measured: probe at x = 9.0 mm resolves to index 20 traced vs 32 concrete; node 20 sits at 7.706 mm, outside the band; nominal span 30 mm vs true 18 mm) — wrong cell, silently. Wire/waveguide ports `np.asarray`/`float()` the x/y arrays and `dt` (loud). | `rfx/nonuniform.py:474-507, 536, 1484-1513`; `rfx/runners/nonuniform.py:532, 542, 969-975` | blocking for coordinate attachment on the traced path | Lane 2 (attach by index, as E4), Lane 9 |
| G10 | Tied-set direction constancy is a MAP invariant, not a solver property: JAX's `min` JVP is the equal split; it equals the directional derivative only if every tied cell moves at the same rate. Measured: half-band stretch AD d(min)/dp = 5.0e-5 vs FD+ 0 / FD- 1.0e-4. Holds for `make_band_profile` bands (band = n equal cells; first ramp cell strictly above the band, 1.287x at cap 1.3) and breaks for `make_z_profile`'s `emax` mode (first ramp cell equals the fine value). | `rfx/nonuniform.py:681-699, 826-872`; autodiff-surface scratch `inplane_tied_set.json` | major (T3) | Lane 2 (asserted at build, 5.3 m3) |
| G11 | The CPML x/y contract on the traced path: a map that lets the end segments absorb a band change moves `cells[0]`/`cells[-1]` (naive stretch Jacobian -0.03125 on both) while the absorber reads the scalar `dx` — miscalibrated, gradient through the CPML silently zero. The concrete path's 1e-12 guard rejects float32 map output (`0.0005000000237` vs `0.0005`). | `rfx/nonuniform.py:141-145, 357-376`; `rfx/boundaries/cpml.py:184-185, 483-498` | major (T3 with absorber) | Lane 2 (pinned map, cpml 0, float64 concrete calls), Lane 3 |
| G12 | The L2 ripple: a rectangular DFT at fixed `n_steps` has duration `n_steps * dt` that moves with any control on the tied set; a Hann/Tukey weight exists (`dft_window_weight`) but is indexed by STEP; `harminv` is host numpy. A smooth spectral observable must be declared before the next L2-class arm. | `rfx/core/dft_utils.py:8-51`; `rfx/nonuniform.py ~2298-2325`; `rfx/harminv.py:22-263`; `20260913_nu_ad_designvar_predeclaration.md:403-423` | major (T3, spectral) | Lane 2 (declared in 5.4) |
| G13 | Waveguide-port one-sided TFSF injection scales by `1/grid.dx` (boundary scalar), not the local x cell: a port inside a 0.75 mm band on a 1.5 mm boundary mesh injects both corrections at 0.500x of the local metric (repro R2, `coeff_H_used/local = coeff_E_used/local = 0.5000`); existing NU port tests keep ports in the boundary-sized block, so nothing catches it. | `rfx/nonuniform.py:2070, 2113, 2180`; `rfx/sources/waveguide_port.py:1061, 1214, 1255` | major (T4) | Lane 4 |
| G14 | TFSF plane wave on an x-graded mesh: 1-D auxiliary grid uniform at `grid.dx`, preflight keyed on `dz_profile` only, so a dx-only graded TFSF sim gets "All checks passed" while the aux span is 0.1365 m against a 0.0950 m 3-D box (repro R3). | `rfx/runners/nonuniform.py:1266-1308`; `rfx/nonuniform.py:2037-2055`; `rfx/api/_preflight.py:6533` | major (T4) | Lane 5 |
| G15 | No in-plane counterpart of the auto z builder: `SimConfig` has `dz_profile` only, `needs_nonuniform_z` from `z_features` only, `_make_dz_profile` is the only path to the thirds rule / fine-cell-at-feature-plane / protected-seam policy, `mesh_planner.profiles_present` hard-codes x/y False. `make_band_profile(boundary_cell=)` takes ONE float for both ends (engine already takes `pin_lo`/`pin_hi`); `make_z_profile` is one fine + one coarse. | `rfx/auto_config.py:231-246, 468-481, 573-585, 692-698, 830-831`; `rfx/api/_mesh.py:97-101`; `rfx/mesh_planner.py:228-254`; `rfx/nonuniform.py:1129-1133, 1219-1220, 1223-1314` | major (T5) | Lane 6, Lane 7 |
| G16 | Absorber + in-plane grading: `boundary_cell` pins ONE cell per end; `nu_grading_reaches_absorber` asks for `cpml_layers` uniform cells at ratio deviation <= 1e-6; the exact-fit plateau deviates 3.9e-2 (12 mm end segment) to 4.3e-3 (100 mm), so every builder-made in-plane profile with CPML draws the advisory on all four faces; no accuracy witness on ANY axis has an absorber present. | `20260907_nu_band_profile_predeclaration.md` R7 / F5; `rfx/api/_preflight.py:5987, 6103`; `support_matrix.md:173-196` | major (T4) | Lane 3 (with G1/G2) |
| G17 | Scalar-dx tolerances: preflight port-on-face half-cell tolerance (`_dx_axis = [dx, dx, dx]`), MSL absorber-proximity margin (documented false positive on `dx_profile`), z CPML thickness read from LEADING entries on both faces; NU-limitation preflight family (`nonuniform_tfsf`, `nonuniform_cpml_thin`) keyed on `dz_profile` only; MSL eigenmode launch uses `float(grid.dx)` (xfail lane). | `rfx/api/_preflight.py:4628-4643, 5076-5079, 6529-6582, 8109-8124`; `rfx/sources/msl_port.py:1268-1283` | minor | Lane 5 |
| G18 | Chain-model conditioning: dense `(n+2)x(n+2)` solve ~1e-8 relative noise on `R_total`, LAPACK-build dependent (replay bound re-declared 1e-8); Riccati cascade prototyped, missed by ~20 %, unbuilt. Limits how tight a multi-level window can be, not whether the arm can be modelled. | `chain_model.py:52-110, 165-171`; E1 note "Replay tolerance re-declared (2026-09-11)" | minor | Lane 1 (window unchanged; replay 1e-8) |
| G19 | Residual unwitnessed ranges inside the z law: bands under 2 cells (n_b = 1 modelled 5.77e-3, n_b = 0 3.87e-3), frequencies other than 10 GHz for the reflection class, ratios above 2.0, coarse cells under 3.75 cells per lambda0 (E1 N15/r2.0 OUTSIDE), dielectric contrast above 4.3, lossy/dispersive media. Multi-level patterns land in these more often. | E1 note Tables A/B/D and "Not witnessed here"; band-profile note "W6 row: L_eff dropped" | minor | inherited by every lane; declared out of scope per lane |
| G20 | Preflight and the 1.3 in-plane cap are skipped on the traced path (`is_tracer` gates), so an optimizer can walk a seam ratio out of the law unnoticed (nominal seam 1.287 against 1.3: a 1 % band shrink with the neighbours absorbing exceeds it). | `tests/unit/autodiff/test_nonuniform_forward_grad.py:202-211`; `rfx/api/_preflight.py` `_INPLANE_RATIO_CAP` | minor | Lane 2 (declared stretch bound), Lane 9 |

## 2. Already on this tree — cite, do not re-derive

- Builder: `rfx.make_band_profile` is axis-agnostic and multi-level (I1
  interfaces <= 1e-12 m, I2 every adjacent ratio <= cap incl. seams, I3
  exact sum to 1e-12 m, I4 `boundary_cell` pin, I5 axis-agnostic; F1-F5
  HELD, 3000-stack fuzz 0 failures). Pinned B on x measured on this tree:
  `[1.0, 1.4, 1.96 x 400, 1.4, 1.0]` mm, pinned A equals its declared
  vector to 8.7e-19 m; float32 `inv_dx`/`inv_dx_h` of the A and B leads
  bit-equal; `dt_A == dt_B` exactly.
- Kernel symmetry: `update_h_nu` / `curl_h_nu` (`rfx/core/yee.py:411-453,
  323-343`), `_shift_*`, `_profile_to_inv_arrays`, `_append_bounding_node`,
  `_pad_profile`, and the per-axis-minimum dt are exact cyclic relabelings
  x -> y -> z -> x. The closed-box path (`cpml_layers=0`, no ports, no
  TFSF) reads no scalar `grid.dx`/`grid.dy`. The one asymmetry is
  `apply_pec`'s z-only `ez[:,:,-1] = 0` (`rfx/boundaries/pec.py:52-53`),
  argued inert on the bounding plane (a closed subsystem since
  `inv_d_h[N-1] = 0`) — MEASURED in Lane 1 arm L1-R, not argued into the
  record. A grade-x fixture builds and runs (finite, dt_x == dt_z =
  2.402765e-12 s, 0.27-0.41 s per run vs 0.40-0.44 s on z).
- Chain model: `chain_model.scattering(profile, n_lead, n_tail, f, dt, dy,
  b)` solves any explicit interior between two uniform runways (multi-level,
  `d1 != d2`, any junction count); `step_reflection` is a platform-exact
  closed form for one junction; `bloch_kz` raises when a runway is
  evanescent. `(dt, dy, b)` mean (time step, sine-axis spacing, sine-axis
  extent) on any axis.
- z law with its range (E1, `results/e1_band_law_sweep.json`): 52 of 54
  arms HELD; `R(n_b) = 2 R_single |sin(k_g (n_b d + c))|`, `c = 2 (r d)
  r^2/(1+r^2)` within +3-5 %, `R_single ∝ (d/lambda)^2` within 4 % over
  15-60 cells for r <= 1.4; inside the law: 8 of 9 (N, r) cells; outside:
  N15/r2.0 (coarse 3.75 cells per lambda0; n_b = 4 +33.6 %, n_b = 32
  -21.4 %, c window 0.503 vs +/- 0.400 mm). Lane-A control cell 30/1.4:
  K_SRC 85 / K_PRB 100 / lead 140 / tail 150 / B 400 / 1200 steps, dt
  2.4027649366612596e-12 s, n_b = 4 row `R_meas 1.006271249900852e-2`,
  `R_model 1.0140707009494808e-2`.
- Accuracy (lane F, merged): A1 z-stratified dielectric order p_mb 2.010 /
  p_uc 2.007, G3 residual <= 0.059 MHz; A2 in-plane TM110 |err| <= 0.0275 %
  at caps 1.3 and 1.4; A3 all three axes TM111 err -0.0350 % (model
  -0.0349 %). E3 `interface_eps='dual_average'` opt-in, measured on A1 only.
- AD-Q (this branch): L1 h_core_left / h_thin / h_core_right /
  eps_core_right HELD on order (R1 in [1.8, 2.2]) AND FD (`|g - D| <= 3B`,
  h_thin 3B 2.8e-4 relative), coverage 0.8830 of the cell-space gradient
  norm; dt-path revert-proof FIRES (h_thin R1 0.947 without dt, dt share
  0.1852); L2 eps HELD, L2 thickness NOT verified (h_thin FD fires at
  relative h 3.1e-2, measured floor 43.6 ulp); judge self-test on this
  tree: true slope R1 2.003 HELD, 10 % wrong slope R1 0.909 FIRED.
- Autodiff surface measured by the review on an x fixture (edges 0/8/10/18
  mm, targets 0.5/0.1/0.5 mm, cap 1.3, `boundary_cell` 0.5 mm -> 60 cells
  20/20/20): tied set = exactly the 20 band cells (first ramp cell / band
  1.287); segment-wise stretch Jacobian spread 0.0 on the tied set (value
  1/20), 0.0 along a band shift; d(dt)/dw AD vs analytic 1.42632e-10 both
  (rel 1.0e-7 float32), grid-level 1.9e-7; interfaces on nodes after a 1 %
  stretch 2.6e-10 m (pinned map), column length 2e-9 m; pinned map boundary
  Jacobian exactly 0.0; `Box.mask_on_coords` gradient exactly 0.0 on traced
  x (21 occupied nodes); solver smoke through `run_nonuniform` with
  index-attached source/probe: grad -43.946 vs central FD -43.952.
- Wallclock (CPU): 0.4-0.8 s per FDTD run of the W6/E1 class (E1 per-cell
  2.8-4.6 s for 7 runs; W6 F8 4.68 s for 7 runs; W2/W3 14.7 s for 19 runs).

## 3. Lane order and the rule for moving between lanes

Lanes run one at a time (PI). A lane opens with its own pre-declaration
note (windows copied from here verbatim, plus the chain-model or map
numbers that this document says "the instrument freezes before any run"),
runs its arms once, appends results, and closes with a replay test. The
next lane does not open until the previous lane's results section is
committed. Order and what each lane needs from the PI:

| lane | name | executes | needs a PI decision? |
|---|---|---|---|
| 1 | Multi-level unequal bands on x, y, z | now (this program) | none blocking; one confirmation (4.8) |
| 2 | In-plane design-variable autodiff | after Lane 1 closes | two confirmations (5.8), none blocking |
| S1 | Support-matrix correction for merged W7 A2/A3 (docs only) | any time, folded into the next closing lane | none |
| 3 | In-plane consumers: CPML x/y per-face endpoint pin, absorber runway | planned | yes — production boundary contract (6.1) |
| 4 | Waveguide-port local-dx injection | planned | yes — production source change (6.2) |
| 5 | TFSF / preflight refusals keyed on dx/dy | planned | no (raise-tier contract), lead approves (6.3) |
| 6 | In-plane auto builder: fine band around features, not edge fitting | planned | yes — policy (6.4) |
| 7 | Multi-level policy (which sizes the builder emits, thirds rule in-plane) | planned | yes — policy (6.5) |
| 8 | Unequal-band accuracy at observable level (order fit on multi-level dielectric) | planned | no; window copied from A1 (6.6) |
| 9 | Differentiable builder in `rfx/` (opt-in), dual_average on traced profiles | planned | yes — API surface (6.7) |

## 4. Lane 1 — multi-level unequal bands on x, y, z (fully specified)

### 4.1 Question

Does the transition law measured on z for ONE fine + ONE coarse size with
symmetric cap ramps (E1) extend to (a) a fine band between two DIFFERENT
coarse sizes, traversed in both directions, (b) two fine bands of different
sizes in one column, and (c) the same structures on the x and y axes — and
what is its range there? Deliverable: a table of (level pattern, axis)
cells with inside/outside verdicts, the law generalized to unequal ramp
amplitudes, and the first FDTD-measured in-plane reflection numbers.

### 4.2 Instrument (reuse named; one new module; zero edits to five files)

New module `validation/research/multiband_nu/e5_multilevel_axes.py`.
Imports and reuses verbatim: `chain_model.scattering`, `step_reflection`,
`bloch_kz`, `s0_sy`; `w2_w3_reflection.gaussian_sine`, `dft_at`, `vg_of`;
`w6_band_builder._cell_delay`, `_git_sha`, `_git_dirty`, `e1_setting`,
`e1_stopband_edge_hz`, `_round_half_up`; `rfx.make_band_profile`,
`rfx.make_nonuniform_grid`, `run_nonuniform`. The five files
`w6_band_builder.py`, `chain_model.py`, `w2_w3_reflection.py`,
`harness.py`, `fixtures.py` are NOT edited (regression, 4.6).

Added in the new module (names fixed so the pre-declaration and replay
test can refer to them):

- `AxisSpec(axis, comp, inv_axis, sin_axis, n_inv=3, n_sin=20, d_t=DXY)`
  with the cyclic map z -> (comp `ex`; invariant x; sine y) [the existing
  fixture], x -> (`ey`; invariant y; sine z), y -> (`ez`; invariant z;
  sine x). Transposition relabelings are NOT used (they flip the curl
  sign; |R| would still be invariant, but bit-identity would not be).
- `build_pec_fixture_axis(profile, spec)`: z: the identical
  `make_nonuniform_grid((A_X, B_Y), dz_profile=profile, dx=DXY,
  cpml_layers=0)` call; x: `make_nonuniform_grid((0, 0),
  dz_profile=full(n_sin, d_t), dx=float(profile[0]), cpml_layers=0,
  dx_profile=profile, dy_profile=full(n_inv, d_t))`; y: `dx_profile=
  full(n_sin, d_t), dx=d_t, dy_profile=profile, dz_profile=full(n_inv,
  d_t)`. `dy_profile` is passed explicitly on the x fixture because
  `dy_profile=None` makes the y spacing equal the scalar `dx` (G1 review).
- `te10_sources_axis(grid, spec, k_src, waveform)`: same list, same order
  (invariant outer, sine inner), `sin(pi j d_t / b)` with `b = (n_sin - 1)
  d_t`, `k_src` in the graded slot, `comp = spec.comp`.
- `run_probe_axis(profile, spec, k_src, k_prb, n_steps)` -> `(grid, trace)`,
  probe at (1 on the invariant axis, `n_sin // 2` on the sine axis,
  `k_prb` on the graded axis, `spec.comp`).
- `expected_vector(segments, cap, pin)`: the builder's own rule, `m =
  ceil(ln(ratio)/ln(cap))`, `rho = ratio^(1/m)`, ramps INSIDE the coarser
  segment; every arm's span is declared as `pin + ramp cells + n * target`
  so the builder emits the declared vector exactly. `builder_matches_
  declared_vector` (1e-12 m per cell, the E1 flag) is reported per arm; if
  false, the chain model is re-run on the builder's ACTUAL output and that
  is `R_model` (the lane-A rule, E1 1b).
- `b_profile_general(lead_cell, n_B, pin_cell)`: the B reference, `[lead]
  x n_B + [pin_cell] x 4`, with `pin_cell = float(prof_A.min())` — the
  REALIZED minimum (G4). On x/y the B profile is built by the same
  `make_band_profile` call with the same `boundary_cell`, so the dt pin
  comes from the realized protected band on both A and B.
- `gates_general(prof, n_lead, n_tail, k_src, k_prb, dt, d_t, b, n_steps,
  kind)`: the five E1 margins, with the band-internal last return taken
  over `prof[n_lead : len(prof) - n_tail]` (bit-identical to E1's slice
  for the existing arms, where `len - n_tail == n_lead + 2 + n_b`), and
  `vg_c`, source/probe positions from node positions (cumsum), so the pin
  cells are counted. `gates_hold` is a RESULT, nothing asserted.
- `arm_general(...)`: `R_meas = |DFT_F0(A - B, [0, gate_end])| /
  |DFT_F0(B, [0, t_inc_end])|`, the E1 form unchanged, with `R_model =
  scattering(prof[m_pin_lo : len - m_pin_hi], n_lead, n_tail, F0, dt,
  d_t, b)` — the pins are instrumentation shared by A and B and are
  stripped before the chain solve (G3 review). Reported per arm: `R_meas`,
  dB, deviation absolute and relative, the five margins, `gates_hold`,
  `dt_matches_b` (|dt_A - dt_B| < 1e-20 s, reported not asserted),
  `builder_matches_declared_vector`, `k_src_used`, `k_prb_used`,
  `source_probe_in_lead` (both cells equal the LEAD cell of THIS profile
  to 1e-12 m and the probe precedes the first transition),
  `lead_f32_identical` (`np.array_equal` of `grid_A.inv_d[:n_lead+1]` and
  `inv_d_h` with `grid_B`'s — the differencing's precondition).
- `fit_c_two_amp(n_bs, r_vals, d_f, R_L, R_R, k_g, dr)`: least-squares c
  in `R(n_b) = sqrt(R_L^2 + R_R^2 - 2 R_L R_R cos(2 k_g (n_b d_f + c)))`
  with `R_L`, `R_R`, `k_g` fixed to the chain model; search `[0, 3 dr]`,
  scan + two refinements at resolution `3 dr / 4000 / 4000^2` (E1 (iii)
  verbatim). For `R_L = R_R` this reduces algebraically to `2 R |sin(k_g
  (L + c))|`; `e1_fit_c` is left untouched (its replay test re-fits at
  1e-9 m).
- Result keys carry axis and level: `f'{axis}_{pattern}'` with `pattern`
  in {`S`, `P1`, `P2`, `P3`, `P4`, `T`} (4.3), so `--resume` never
  collides with an E1 cell. Provenance block as `run_e1`.

Fixed physical fixture (lane A / E1 values, unchanged): F0 = 10 GHz,
sigma_t = 64 ps, t0 = 5 sigma_t, d_t = DXY = 1.5 mm, b = 30 mm (20 cells),
a = 4.5 mm (3 cells), PEC closed, `cpml_layers = 0`, fine d_f = 1.0 mm
(29.98 cells per lambda0), cap 1.4. Physical planes re-cut in the LEAD
cell of each profile (E1 1c rule): source 166.6 mm, probe 196.0 mm, lead
274.4 mm, tail 294.0 mm, B 784.0 mm, all measured from the end of the lo
pin; cell counts = length / lead cell, rounded half-up. Run length
`n_steps = ceil(max(2.883 ns, t_s + 0.1 ns) / dt)` (E1 1c). For the z
family (patterns S, P1-P4) `dt = 2.402765e-12 s` and `n_steps = 1200`
whenever the minimum cell is 1.0 mm; for pattern T the minimum is 0.5 mm,
`dt = 1.493514e-12 s` (E1 N60 Table S) and `n_steps = 1931`.

End pins: on z, none (the z family reproduces E1's unpinned vectors). On x
and y, EVERY profile (A, B, single) is built with `boundary_cell = d_lo`
where `d_lo` is the fine cell that satisfies G1 (`dx_profile[0] ==
dx_profile[-1] == dx`): `d_lo = 1.0 mm` for S, P1-P4, T. This puts `[1.0,
1.4]` before the lead on the lo side and `[1.4, 1.0]` after the tail on
the hi side (measured on this tree for the S pattern: `[1.0, 1.4, 1.96 x
n, ..., 1.4, 1.0]`, k_src/k_prb 85/100 -> 87/102). For pattern T on x/y
the B profile is `make_band_profile([0, L_B, L_B + 4 x 0.5 mm],
[1.96, 0.5], protected=[False, True], max_ratio=1.4, boundary_cell=1.0e-3)`
so that B carries a realized 0.5 mm protected band (dt pin) beyond the
gate; its far-end structure returns after the gate (E1 gate "B pin / far
wall return after the gate", margin >= 4.19 ns on every E1 arm).

### 4.3 Arms (the (level pattern, axis) cells)

Patterns, all at cap 1.4, fine d_f = 1.0 mm unless stated, widths
n_b in {2, 4, 8, 16, 32} (E1):

| pattern | structure (lead -> band -> tail), mm | ramp cells (builder rule) | what it tests |
|---|---|---|---|
| S | 1.96 | 1.4 | [1.0] x n_b | 1.4 | 1.96 | one per side, rho 1.4 | the E1 30/1.4 control (regression + relabel identity) |
| P1 | 1.96 | 1.4 | [1.0] x n_b | 1.4, 1.96 | 2.744 | one left (rho 1.4), two right (rho 1.4; 2.744 = 1.4^3 d_f, m = 3 measured on this tree) | c_L != c_R, entered from the finer coarse side |
| P2 | 2.744 | 1.96, 1.4 | [1.0] x n_b | 1.4 | 1.96 | two left, one right | same structure traversed from the coarser side (asymmetric traversal) |
| P3 | 1.96 | 1.4 | [1.0] x n_b | 1.4 (= c_R, no ramp cell) | one left, zero right (the band edge is the single step at cap) | a coarse side that is itself the ramp size |
| P4 | 1.4 | [1.0] x n_b | 1.4 | 1.96 | zero left, one right | mirror of P3 |
| T | 1.96 | 1.4 | [1.0] x 4 | 1.4 | [1.96] x n_c | 4 ramp cells | [0.5] x n_b2 | 4 ramp cells | 1.96 | band 1: one per side (rho 1.4); band 2: m = ceil(ln 3.92 / ln 1.4) = 5, rho = 3.92^(1/5) = 1.3143, cells 1.4888 / 1.1334 / 0.8628 / 0.6568 (measured on this tree) | two fine bands of different sizes (30 and 60 cells per lambda0) and ramps of different length in one column; n_c in {4, 16}, n_b2 in {4, 8} |

Per pattern and axis the runs are: B reference; single-transition arms
(one per distinct side: S 1; P1-P4 2 each — the left ramp with a fine
tail, and the right ramp as fine lead -> ramp -> coarse tail, both built
as `[lead] x n_lead | ramp | [fine or coarse] x n_tail`); band arms (S,
P1-P4: five widths; T: the four (n_c, n_b2) combinations, with band 1
fixed at n_b = 4). Counts per axis: S 7, P1-P4 8 each = 32, T 1 + 2 + 4 =
7; 46 FDTD runs per axis, 138 for z, x, y, plus the two relabel controls.

Sub-lanes, run in this order, each its own JSON (`results/e5_z.json`,
`e5_x.json`, `e5_y.json`), each preceded by the frozen model JSON:

1. **L1-M (chain model, frozen first, zero FDTD):** `--model-only` writes
   `results/e5_model.json` with, per (pattern, axis) cell and arm:
   `R_model`, the per-side single-transition amplitudes `R_L`, `R_R`
   (`scattering` on each side's own single profile), `k_g` of the fine
   band, `c_model` (chain fit over n_b = 0..80 with `fit_c_two_amp`),
   `vg` per runway, stopband edge per runway (`e1_stopband_edge_hz` on
   the COARSER runway), the five gate margins, `n_steps`, `dt`, and the
   per-arm window. This JSON is committed with the lane's pre-declaration
   note BEFORE any FDTD run; the tables in that note are generated from
   it, not retyped.
2. **L1-0 (z control):** pattern S on z through the NEW code, all seven
   runs.
3. **L1-R (relabel identity):** pattern S, n_b = 4, run on x and on y
   with NO pins (the symmetric S profile passes G1), compared with the
   L1-0 z trace and R_meas.
4. **L1-Z:** patterns P1-P4 and T on z (39 runs).
5. **L1-X:** patterns S, P1-P4, T on x with pins (46 runs), including the
   pinned-vs-unpinned bridge (4.4, W5) on pattern S.
6. **L1-Y:** the same on y (46 runs).

### 4.4 Frozen falsifiers (numeric windows and the precedent each is derived from)

- **W1 — per-arm amplitude, every band and single arm, every axis:**
  `|R_meas - R_model| <= 0.20 R_model + 3e-5`. Precedent: lane A / E1
  section 3, unchanged (E1: 52 of 54 HELD; the 3e-5 is `w2_w3_reflection.
  FS2_FLOOR`). Floor reading rule as E1 1f: a fired row with `R_model <
  2.5e-4` and `|R_meas - R_model| <= 1.5e-4` is recorded "fired — at the
  instrument's absolute floor" (E1 measured floor <= 4.1e-5 at 60 cells,
  <= 8.6e-5 over rows under 3e-3). No pattern here has a modelled row
  under 2.5e-4 unless the model JSON says so (P1/P2 nulls sit at
  `|R_L - R_R| > 0`); the rule is carried, not expected to fire.
- **W2 — the bound (law (ii) generalized):** every band arm, `R_meas <=
  1.05 x (R_L + R_R)` for S, P1-P4 (S: `R_L = R_R = R_single`, so this is
  E1's `2 R_single x 1.05` exactly), and `R_meas <= 1.05 x (R_1L + R_1R +
  R_2L + R_2R)` for T (four single-transition amplitudes; the coherent
  triangle inequality of F7a). Precedent: E1 (ii) slack 0.05, HELD on all
  45 widths including the fired cell.
- **W3 — phase (law (iii) generalized):** per (pattern, axis) cell for S,
  P1-P4, `|c_meas - c_model| <= 0.10 x DR`, `DR = r d_f = 1.4 mm`, i.e.
  **+/- 0.140 mm** — the ramp cell adjacent to the band, which every side
  of every pattern here shares (for P3/P4's stepped side the "ramp cell"
  is the 1.4 mm coarse itself). Precedent: E1 (iii), calibrated as ten
  times lane A's 0.0096 DR residue; E1 30/1.4 window +/- 0.140 mm, held
  at 0.014 mm. `c_meas` from `fit_c_two_amp` over the five measured rows
  with `R_L`, `R_R`, `k_g` fixed to the model. Hypothesis reported
  alongside, NOT gated: `c = sum over sides of w_s`, with `w_s` the
  amplitude-weighted centroid of that side's step planes outside the band
  edge using the `(local cell)^2` weights of E1 1g — one ramp cell: `DR
  r^2/(1+r^2) = 0.927 mm`; two ramp cells (P1 right, P2 left): `(r^2 x
  1.4 + r^4 x 3.36)/(1 + r^2 + r^4) = 2.301 mm`; no ramp cell (P3 right,
  P4 left): 0. The frozen prediction is `c_model` (the chain fit), as in
  E1; the closed form is the explanation and its deviation is reported.
  T has no closed form and no c window: its deliverable is W1 + W2 per
  arm.
- **W4 — control (regression through the new code):** L1-0 pattern S on
  z: each of the six `R_meas` (single + five widths) equals E1's
  `results/e1_band_law_sweep.json` N30_r1.4 values (single 5.7302e-3,
  widths 7.4364e-3 / 1.0063e-2 / 1.1164e-2 / 1.2559e-3 / 1.4248e-3) to
  **<= 1e-6 relative** (E1 1c control window; E1 reproduced lane A at
  0.0), and each `R_model` equals the E1 JSON's to **<= 1e-8 relative**
  (the replay bound re-declared 2026-09-11 from the 1.3e-9 cross-platform
  spread). The bit-identity (relative difference 0.0 or not) is reported
  as a number.
- **W5 — relabel identity (L1-R) and the pin bridge:** for pattern S,
  n_b = 4, x and y vs z: `|R_meas_axis - R_meas_z| / R_meas_z <= 1e-6`
  and `max_t |trace_axis - trace_z| / max_t |trace_z| <= 1e-6` (the
  float32 reproducibility class of W4; dt_x == dt_z measured). Pin bridge
  on z: the pinned S profile (built as on x/y, run on z) vs the unpinned
  L1-0 arm at every width: `|R_meas_pinned - R_meas_unpinned| <= 1.5e-4`
  absolute (E1 1f's declared absolute-floor number; lane A's absolute
  deviations were 4.6e-5 .. 1.3e-4). Reading rule, declared: if W5 fires
  on an axis, that axis's arms still run and are gated by W1-W3 on their
  own model (the physics gate), and the cell verdict carries the flag
  "relabel identity fired: instrument class"; the fire is not tuned.
- **W6 — gates:** the five margins of every arm from the frozen model
  JSON must be positive before the run (a non-positive margin means the
  arm is redesigned BEFORE the run, in the pre-declaration, never after);
  expectation written here: smallest margin >= 0.256 ns (E1's smallest,
  the incident-gate construction `arrival + 8 sigma`), inner-return
  margin on T's longest structure (n_c = 16, n_b2 = 8: 4 + 1.4 x 2 +
  31.36 + 4.14 x 2 + 4 = 49 mm of interior) expected ~0.42 ns (E1 n_b =
  32 at 30/1.4: 0.520 ns for 32 mm). `gates_hold` measured per arm.
- **Cell verdicts (before the run):** a (pattern, axis) cell is "inside
  the law domain" if every one of its W1 windows held, W2 held at every
  width, and (S, P1-P4) W3 held; "inside the -54 dB class" by E1's rule
  verbatim (every adjacent ratio <= 1.4 and fine band >= 30 cells per
  lambda0), with the coarsest runway's cells per lambda0 reported beside
  it (P1/P2: 2.744 mm = 10.9 cells per lambda0; the class rule does not
  bound the coarse cell and this document does not change the rule).

Expectations written before the run, so a surprise is recognizable:

| cell | z | x | y | -54 dB class | likeliest fire |
|---|---|---|---|---|---|
| S | inside (E1 control) | inside if W5 holds | inside if W5 holds | inside | none |
| P1, P2 | inside | inside | inside | inside by E1's rule; coarsest runway 10.9 cells per lambda0 (reported) | W3 on P2 (three-step side at the lead; the centroid form is off by the phase across 3.36 mm = 0.61 rad at k_g) |
| P3, P4 | inside | inside | inside | inside | none (R_L != R_R by ~r^2, nulls at |R_L - R_R| well above 3e-5) |
| T | inside | inside | inside | inside (every step <= 1.4; 30 and 60 cells per lambda0) | W1 on the n_c = 16 / n_b2 = 8 arm (longest interior, tightest inner-return margin) |

Declared out of scope (inherited, G19): coarse runways under 3.75 cells
per lambda0, ratios above 2.0, frequencies other than 10 GHz, bands of
fewer than 2 cells, dielectric or lossy interiors, any absorber.

### 4.5 Reading rule for x/y if G1 bites in a way not covered by pins

If `make_nonuniform_grid` refuses any pinned x/y profile (it must not —
pins make both ends `1.0 mm == dx`; measured on this tree for S and B),
the arm is recorded "not buildable on this tree" with the exception text,
and no production change is made inside this lane (that is Lane 3).

### 4.6 Regression requirement (bit-identical existing arms)

- `git diff --stat` on `w6_band_builder.py`, `chain_model.py`,
  `w2_w3_reflection.py`, `harness.py`, `fixtures.py` is EMPTY at every
  commit of the lane.
- `tests/unit/nonuniform/test_e1_band_law_replay.py` (chain floats 1e-8,
  c 1e-9 m) and `test_band_builder_chain_model.py` (1e-9) pass before and
  after, run with the pinned PYTHONPATH; the pass counts are recorded in
  the lane note.
- W4: the L1-0 z control through the new module reproduces E1's N30_r1.4
  rows to 1e-6 relative (R_meas) and 1e-8 (R_model); the lane-A F8 n_b = 4
  row (`R_meas 1.006271249900852e-2`, `R_model 1.0140707009494808e-2`)
  is quoted against the same numbers.
- New replay test `tests/unit/nonuniform/test_e5_multilevel_replay.py`:
  chain model only (no FDTD), replays every `R_model`, `R_L`, `R_R`,
  `c_model`, the window arithmetic, the gate geometry, W2/W3 and the
  recorded verdicts from the three JSONs at 1e-8; asserts recorded
  verdicts, including fired ones.

### 4.7 Cost

Per FDTD run 0.4-0.8 s on CPU (section 2); 138 runs + 2 relabel + 5 pin
bridge = 145 runs, about 2 CPU minutes; chain solves in milliseconds (T
profiles nz ~ 450-700). Three `--model-only` + three measuring calls, each
far under the 600 s Bash cap. The cost is the pre-declaration and the
model JSON, not compute.

### 4.8 PI decision needed

None blocking. One confirmation, stated so silence is not consent: the
in-plane instrument uses `boundary_cell` end pins (two extra cells per
end, stripped before the chain solve) instead of relaxing the
`make_nonuniform_grid` x/y end-cell guard. The alternative is a production
change and belongs to Lane 3. If the PI prefers the guard relaxed first,
Lane 3's item 1 runs before Lane 1's L1-X.

## 5. Lane 2 — in-plane design-variable autodiff (fully specified)

### 5.1 Question

Is the reverse-mode gradient of a 3-D NU simulation correct along two
PHYSICAL in-plane design variables — the width `w` of a fine x band (a
trace-width surrogate: a PEC strip whose x edges are the band edges) and
its position `x_c` — including the CFL `dt` path through the x tied set,
for (a) a smooth transient loss (L1 class) and (b) a smooth spectral
observable that does not carry the L2 ripple? Judged by the AD-Q judges
verbatim; coverage of the x cell-space gradient reported.

### 5.2 Instrument

New module `validation/research/multiband_nu/e6_inplane_designvar.py`.
Imports verbatim, no copies: `adq_designvar.fit_order`, `fd_budget`,
`selfcheck`, `coverage`, `ulp`, `losses_along`, `HS`, `N_QUANTA`,
`RHO_BAND`; `e4_diff_stackup.asym`; `rfx.make_band_profile`,
`rfx.make_nonuniform_grid`, `run_nonuniform`. `adq_designvar.py` and
`e4_diff_stackup.py` are NOT edited.

**MeshMap (host-built, pure-jnp forward; lives in validation/research,
not rfx/):** `MeshMap.from_builder(profile0_f64, seg_id, pinned_mask,
seg_len0)` stores the CONCRETE builder output at the nominal design
`p0`, the segment id of every cell, the pinned-cell mask (the two
`boundary_cell` ends), the nominal free length of every segment and the
node index of every segment edge. `cells(seg_len)` = `where(pinned,
cells0, cells0 * (seg_len - pinned_len)[seg_id] / free_len0[seg_id])`
— fixed per-segment cell counts, every interface stays on a node by
construction, boundary cells exactly fixed (pinned variant; measured
boundary Jacobian 0.0). Design variables: `w` (band segment length; the
two end segments each absorb half, `A` column (-1/2, 1, -1/2) — the
h_thin form) and `x_c` (lead +delta, tail -delta, band unchanged,
column (1, 0, -1)). Total length exact by construction of `A`. Concrete
calls use float64 (the 1e-12 guard, G11); traced calls are float32.

**Fixture (PEC-closed, 3-D, vacuum, cpml 0):** x: `make_band_profile(
[0, 8e-3, 10e-3, 18e-3], [0.5e-3, 0.1e-3, 0.5e-3], protected=[False,
True, False], max_ratio=1.3, boundary_cell=0.5e-3)` — the review's
measured fixture: 60 cells, segments 20/20/20, tied set = the 20 band
cells, first ramp cell / band = 1.287. y: 12 cells of 0.5 mm (6 mm); z:
24 cells of 0.5 mm (12 mm). `dt = 0.99 / (c0 sqrt(1/dx_min^2 + 1/dy_min^2
+ 1/dz_min^2))` with `dx_min = 0.1 mm` -> 3.17e-13 s at p0 (the
instrument records the exact value). PEC strip: tangential E (`ex`, `ey`)
zeroed on the z-node plane `k_s = 12` for x nodes `i_L .. i_R` (the band
segment's edge nodes from the map, 21 nodes) and all y — attached BY NODE
INDEX through the NU forward path's PEC mask override
(`pec_mask_override` / `pec_occupancy_override`, `rfx/api/_execute.py:
2206-2260`); the executor confirms the kwarg in the map-check self-test
(5.3, m7) and, if `run_nonuniform` does not accept it, applies the mask
in a thin in-instrument step wrapper (declared alternative, recorded
which). No coordinate-based `Box` (gradient exactly 0, G8) and no
coordinate attachment anywhere (G9). Source: Gaussian-sine `jz` at index
(8, 6, 6) in the lead segment, F0 = 20 GHz, sigma_t = 20 ps, t0 = 5
sigma_t. Probe: `ez` at (i_R + 3, 6, k_s + 2), i.e. three cells into the
tail segment, two planes above the strip, by index.

**Observables (declared here, before any run):**
- **L1 (transient, E4 class):** `sum_n ez(n)^2` over `N1 = 600` steps
  (190 ps at dt(p0): the pulse completes inside the window; E4's 120
  steps at 0.95 ps was 114 ps).
- **L2s (smooth spectral, replaces the rippled fixed-frequency DFT
  power; G12):** `N2 = 2400` steps; Hann taper in PHYSICAL time, `h(t) =
  0.5 (1 - cos(2 pi t / T_w))` for `t = n dt < T_w`, zero after, with
  **`T_w = 0.8 x N2 x dt(p0)` fixed at the nominal design** (so the
  window duration does not move with the design; the ladder's largest
  step, relative h = 2^-3, shrinks dt by at most 12.5 %, and `N2 dt >=
  0.875 N2 dt(p0) > T_w` on every ladder point); band-integrated power
  `sum_k |sum_n h(n dt) ez(n) exp(-2 pi i f_k n dt) dt|^2` over the comb
  `f_k = F0 + k x 0.5 GHz`, k = -10 .. 10 (10 GHz wide, wider than
  `1 / T_w` = 1.6 GHz). In-trace, in-script (as E4's L2 DFT).

**Ladders and judges (AD-Q verbatim):** relative steps `HS = 2^-k`, k =
3 .. 17, absolute displacement `h x w0` for BOTH controls (`w0 = 2 mm`;
the shift uses the band width as its scale so the two ladders are
commensurate); R0 / R1 with `fit_order` (32-quantum eligibility,
curvature guard, longest monotone run >= 4 points, OLS in log-log);
`fd_budget` (Richardson `T`, roundoff `Q`, bar `B = Q + T`, `rho` in
[2, 8], INCONCLUSIVE if `3B/|g| > 0.15`); `selfcheck()` before EVERY arm
(must reproduce R1 2.003 HELD / 0.909 FIRED, the tree's values). Noise
floor: MEASURED per control and arm from the start (the L2 second-attempt
method: 64 evaluations uniform in relative h [1e-5, 1e-4], quadratic fit,
`sigma` = RMS residual x sqrt(64/61), cubic must reduce RMS by < 10 % or
`sigma` is marked unreliable), used as `max(ulp, sigma)` in both judges;
the 1-ulp verdict is reported alongside for comparison with stack_l1.

### 5.3 Arms (one attempt each, separate JSONs)

1. **E6-M (map checks, zero FDTD):** `results/e6_map_checks.json`.
   m1 column length `|sum cells(p) - 18 mm| <= 1e-9 m` at every ladder
   point of both controls (E4 `MAP_LEN_TOL` 1e-9); m2 every interface on
   a node to `<= 1e-9 m` after each ladder displacement (measured
   2.6e-10 m pinned at 1 %); m3 tied-set spread along each control
   EXACTLY 0.0 (E4's `assert spread == 0.`), and the tied set of the
   whole grid equals the 20 band cells (y and z minima 0.5 mm not tied
   to x's 0.1 mm); m4 boundary-cell Jacobian exactly 0.0 for both
   controls; m5 `J^T g` vs the direct parameter gradient `<= 1e-5`
   relative (E4 `MAP_JAC_REL` 1e-6 declared, stack_l1 measured 2.2e-6
   absolute; 1e-5 relative is the declared bound here); m6 `d(dt)/dw`
   AD vs analytic `<= 1e-6` relative (measured 1.0e-7 float32,
   grid-level 1.9e-7); m7 strip occupancy exactly 21 nodes at every
   ladder point (fixed topology) and the override kwarg accepted (or the
   wrapper alternative recorded); m8 forward identity: traced loss at p0
   equals the concrete float64-map loss at p0 to float32 (`<= 1e-6`
   relative). Any m-check failing means the map is fixed BEFORE the
   arms and the fix recorded; these are instrument checks, not windows.
2. **E6-L1 (`results/e6_l1.json`):** controls `w`, `x_c` on the L1 loss.
3. **E6-R (`results/e6_revert_l1_nodt.json`):** the TRUE L1 losses along
   both controls judged against the gradient with `dt` through
   `stop_gradient` (the `measure_revert` pattern). Declared: **`w` must
   FIRE** (it moves the x tied set; dt-path share reported); **`x_c`
   must return the SAME verdicts as E6-L1** (its dt-path share is
   exactly 0: the band cells do not change). If `w` HOLDS here, the
   Taylor gate has no power against the dt path on this fixture at this
   ladder, and E6-L1's `w` verdict may not be read as verifying the dt
   path; the note says so.
4. **E6-L2s (`results/e6_l2s.json`, `e6_l2s_floor.json`):** `w`, `x_c`
   on the smooth spectral observable, measured floor first.
5. **E6-C (coverage, from the E6-L1 and E6-L2s gradients, no new runs):**
   `coverage(g_x, J)` with `J` the 60 x 2 map Jacobian and `g_x` the AD
   gradient over the 60 x cells; reported for all controls and for the
   controls passing BOTH gates (E4: 0.8875 / 0.8830); dimensionless
   variant with nominal cell widths as scales. No minimum assumed.

### 5.4 Frozen falsifiers

- **Order:** R0 slope in **[0.9, 1.1]**, R1 slope in **[1.8, 2.2]**
  (AD-Q, frozen; the upper edge 2.2 carries no gradient-correctness
  meaning — recorded in the AD-Q note — and stays; the lower-side
  criterion R1 >= 1.8 is reported alongside as the labelled diagnostic,
  as in the L2 second attempt). No eligible run of >= 4 points =>
  INCONCLUSIVE; a fitted miss => FIRED.
- **FD:** `|g - D(h)| <= 3 B(h)` at every informative step (`rho` in
  [2, 8]); a direction is FIRED if ANY informative step fires, HELD if at
  least one is informative and none fires, else INCONCLUSIVE;
  INCONCLUSIVE if `3B/|g| > 0.15` (AD-Q verbatim).
- **Revert-proof:** `w` FIRED on order (R1 slope below 1.8 or no fit)
  with the dt path removed AND `x_c` verdicts unchanged — both required
  for E6-L1's `w` verdict to be read as covering the dt path (AD-Q C).
- **Map checks m1-m8:** thresholds as listed (1e-9 m, 1e-9 m, 0.0, 0.0,
  1e-5, 1e-6, 21, 1e-6).
- **Stretch bound (G20, declared):** along `w` the seam ratio is
  `1.287 x (1 + h x 1 mm / 8 mm) / (1 - h)` on the shrink side (band
  cells scale by `1 - h`, each 8 mm neighbour grows by `h x w0 / 2`):
  1.494 at the ladder top h = 2^-3, and above the 1.3 in-plane advisory
  cap from h = 2^-6 (1.309) upward; on the growth side the ratio falls
  (1.127 at h = 2^-3). The tied set stays the 20 band cells over the
  whole ladder: at h = 2^-3 growth the band cell is 0.1125 mm against a
  first ramp cell of 0.1287 x (1 - 0.0156) = 0.1267 mm (checked in m3 at
  every ladder point). Recorded per ladder point as `seam_ratio(h)`; NOT
  a window (the cap is an advisory and the traced path does not evaluate
  it), reported so the support matrix can state the range of the claim.

Expectations before the run: E6-L1 `w` HELD on both gates (E4/AD-Q
h_thin: R1 2.047, 3B 2.8e-4; the E6 fixture's `dt` share is larger since
x's tied set is 20 of 60 cells); `x_c` HELD (a pure shift changes the
lead/tail cells by 0.125 mm / 8 mm = 1.6 % at the ladder top; small
gradient, so INCONCLUSIVE on order is the likeliest non-HELD outcome, as
eps_air was); E6-R `w` FIRED, `x_c` unchanged; E6-L2s `w` HELD on order
with the physical-time Hann window — if it FIRES with `rho`-valid steps
the observable choice is recorded as the result and no second observable
is tried inside this lane.

### 5.5 Regression requirement

- No `rfx/` change. `git diff --stat rfx/` empty at every commit.
- `tests/unit/nonuniform/test_adq_designvar_replay.py` (12 passed on this
  tree) and `test_band_accuracy_ad_replay.py` pass before and after;
  `adq_designvar.py`, `e4_diff_stackup.py` unedited.
- `selfcheck()` reproduces R1 2.003 (HELD) and 0.909 (FIRED) before every
  arm; recorded in each JSON.
- New replay test `tests/unit/nonuniform/test_e6_inplane_designvar_
  replay.py`: reconstructs the fits, error bars, verdicts, coverage and
  the map checks from the JSONs without FDTD; pins fired and inconclusive
  outcomes as recorded.

### 5.6 Cost

Grid 60 x 12 x 24 = 17,280 cells. L1: 600 steps, ~0.1 s per run after
JIT; ladder 15 points x 2 sides x 2 controls = 60 runs + L0 + gradient,
plus revert (60), plus floor (64 x 2) -> ~250 runs, under 1 minute. L2s:
2400 steps, ~0.4 s per run; 60 ladder runs + 128 floor runs -> ~1.5
minutes. Map checks: seconds. Whole lane under 5 CPU minutes; each arm
one Bash call.

### 5.7 What the lane can and cannot claim (written before the run)

Can: "AD gradient along a fine-band width and position on x, with the
band edges carrying a PEC strip attached by node index, verified by
Taylor order and FD-inside-3B on a transient and on a smooth spectral
observable, dt path included, on the closed vacuum fixture of 5.2, over
relative displacements [2^-17, 2^-3] of a 2 mm band" (or whichever
sub-range the windows hold on). Cannot: anything about coordinate-
rasterized geometry (`Box` gradient 0), about ports (host-coerced on
traced x/y, G9), about an absorber (cpml 0, G11), about y (the same map
applies by relabeling but is not measured here; a y arm is a Lane 2b
sub-lane if the PI asks), or about `interface_eps='dual_average'` under
trace (G8).

### 5.8 PI decisions needed

Two confirmations, none blocking:

1. The "trace-width surrogate" is a PEC strip whose edges are the band
   edges, attached by node index — the lane verifies AD along the design
   variable on the index-attached path and says nothing about
   `Box`/coordinate rasterization. If the PI wants the coordinate path
   covered, that is Lane 9 (differentiable builder in `rfx/`), not this
   lane.
2. The smooth spectral observable is a physical-time Hann taper plus a
   0.5 GHz comb integral (5.2). If the PI prefers a differentiable
   resonance-frequency estimator instead, it does not exist on the tree
   (`harminv` is host numpy) and would be new instrument code, declared
   in a separate pre-declaration before Lane 2 runs.

## 6. Lanes 3+ (planned, not executed) and the standing docs item

Each lane opens with its own pre-declaration; the windows below are the
FORM and the precedent to copy from, not yet frozen numbers where a
number needs a measurement this program does not have.

**S1 — support-matrix correction (docs only, no measurement).** Question:
none; the matrix understates. Change: the 'Multi-band graded mesh' row
cites W7 A2 (x and y, two 0.25 mm fine bands in ~1 mm coarse, caps 1.3
and 1.4, TM110, |err| <= 0.0275 %) and A3 (all three axes, TM111, err
-0.0350 %), states the witness class (resonance vs analytic at one
scale; not energy / per-transition / round-trip / order), and replaces
"no in-plane provenance for the 1.3 lock" with "in-plane provenance
exists in the resonance class only, at caps 1.3 and 1.4 on vacuum
fixtures; the reflection class is Lane 1". `_INPLANE_RATIO_CAP` unchanged;
`test_inplane_grading_lock_stays_at_1_3` unchanged. Regression: no code.
Cost: minutes. Decision: none.

**Lane 3 — in-plane consumers: CPML x/y per-face endpoint, absorber
runway.** Question: with the x/y end-cell guard relaxed to the z contract
(per-face cells `dx_arr[0]`, `dx_arr[-2]`; `_get_axis_cell_sizes` per
face; `CPMLAxisParams` per-face slots populated), does an asymmetric
x-graded absorber reflect as its z-graded mirror does? Instrument: the
Lane 1 relabel fixture with `cpml_layers > 0` on the graded axis's two
faces, absorber reflection measured by the E1 differencing (an A run with
absorber vs a B run with a longer runway), on z first then x. Falsifier
form: (i) existing tests bit-identical for symmetric-ended profiles
(`tests/unit/nonuniform/test_nonuniform_xy.py`, `test_farfield_inplane_
nonuniform.py`, waveguide NU tests); (ii) x-vs-z relabel identity of the
absorber reflection within the W5 float32 class (1e-6 relative) on the
same profile; (iii) an absorber-reflection window derived from the z
measurement of (i) BEFORE x runs. Plus the runway question (G16): a
declared `boundary_cells` runway argument or an advisory keyed on the
realized runway ratio; window: the advisory's 1e-6 deviation is replaced
by a measured number from Lane 3's own absorber arm, not guessed. Cost:
~1 day instrument + minutes of runs. Decision: YES — a production
boundary-contract change in `rfx/nonuniform.py` and `cpml.py`; opt-in is
not meaningful here (the guard either exists or not), so the PI decides
whether relaxing it (with the per-face CPML fix) is authorized.

**Lane 4 — waveguide-port injection at the local cell.** Question: with
`coeff = dt / (MU_0 dx_local)` and `dt / (EPS_0 dx_dual)` at the port's
own cells, does a port inside a fine x band return the S-parameters of
the same physical guide with the port in the coarse block? Instrument:
`tests/unit/sparams/test_waveguide_nu_sparam.py`'s WR-90 fixture
(coarse 1.5 mm | fine 0.75 mm | coarse) with the port moved to 45 mm
inside the fine block (repro R2: `coeff_used / local = 0.5000` today).
Falsifier form: |S11|, |S21| of the fine-block port vs the coarse-block
port on the same guide, window copied from the matrix's own NU waveguide
number (`max_mag_abs_diff = 0.008529` for the graded-dy Palace
comparison, `support_matrix.md:95`) — declared as the window on the
fine-vs-coarse port difference; existing NU port tests bit-identical
(ports in boundary-sized cells: `dx_local == grid.dx`, so the fix is a
no-op there and the test suite must show it). Cost: hours. Decision:
YES — production change in `rfx/sources/waveguide_port.py`
(`apply_waveguide_port_h/e`) and the `grid.dx` call sites in
`rfx/nonuniform.py:2070, 2113, 2180`.

**Lane 5 — TFSF and preflight families keyed on dx/dy.** Question: none
(contract). Change: `_validate_cfg_nonuniform_limitations` keys
`nonuniform_tfsf` on any of the three profiles, refusing TFSF on a
dx/dy-graded mesh (raise tier) until a graded 1-D auxiliary exists; the
`nonuniform_cpml_thin` advisory gets x/y versions; port-on-face
tolerance and MSL absorber margin read `_local_cell`. Falsifier: repro
R3 raises instead of "All checks passed"; existing preflight tests
unchanged. Cost: hours. Decision: no (raise-tier, no default behaviour
of a valid simulation changes); lead approves.

**Lane 6 — in-plane auto builder: fine band around features, NOT edge
fitting.** Question: does an auto-emitted `dx_profile`/`dy_profile`
(feature scan on x/y, `SimConfig` slots, `_mesh.py` resolution, mesh
planner `profiles_present`) realize a declared stack with the builder's
invariants, and does an A2-class witness on such a mesh hold? Falsifier
form: F1-F5 of the band-profile note (interfaces 1e-12 m, ratio <= cap +
1e-9, sum 1e-12 m, pin, round trip) on the emitted profiles; A2's
TM110 window (|err| <= 0.0275 % class, copied from `w7_accuracy_ad.json`
a2.judge_ext at lane pre-declaration) on a fixture whose fine bands are
auto-placed around a declared feature. Regression: `auto_configure` z
output bit-identical on every existing fixture (no dz change). Cost: 1-2
days. Decision: YES — policy: "fine band around features" (a protected
band of the feature's extent plus a declared margin, cap ramps outside)
versus edge fitting is the PI's call; this program recommends the former
(the builder already realizes it; edge fitting is the argmin path with
zero gradient, G8).

**Lane 7 — multi-level policy.** Question: which sizes an auto builder
emits when features of different scale share an axis (two fine sizes vs
one minimum), and whether the thirds rule applies in-plane. Instrument:
Lane 1's T pattern (two sizes) evidence and the F7 PCB chain profile;
falsifier: per-transition reflections of the emitted multi-level profile
sum under the W2 bound with 1.05 slack, and the emitted profile's dt is
not more than a declared factor below the single-minimum alternative.
Decision: YES — policy; the numbers come from Lane 1.

**Lane 8 — unequal-band accuracy at observable level.** Question: is the
convergence order on a multi-level DIELECTRIC profile (A1 stack with the
thin band at a different fine size than the cores, and the AZ-class
thirds sub-cells) second order, i.e. does the reported-only A1 AZ arm get
an order? Instrument: `w7_accuracy_ad.py` A1 path with a multi-level MB
profile whose thin cell scales with s (the reason AZ had no order).
Falsifier: A1-F1 form verbatim — `p_mb >= 1.8` (anomaly flag > 2.4),
`p_uc` in [1.8, 2.2], G3 residual `<= 0.15 MHz` window (A1: 0.059 MHz
measured), dual-cell eps assembled by the instrument. Regression: the
A1 UC/MB rows reproduce (`results/w7_accuracy_ad.json` a1) to 1e-6.
Cost: hours. Decision: no.

**Lane 9 — differentiable builder in `rfx/` (opt-in).** Question: can
`MeshMap` (Lane 2) be productized beside `make_band_profile` — host-built
map, pure-jnp `cells(seg_len)`, published interface node indices — with
index-based material/PEC attachment and a traced-profile path for
`interface_eps='dual_average'`, without changing any default? Falsifier:
Lane 2's m1-m8 on the production map; E3-B delta gate (default path
byte-identical); E6-L1 reproduced through the production map to 1e-6.
Decision: YES — API surface and the ownership of `position_to_index`'s
traced fallback (G9: nominal CONCRETE profile instead of uniform).

## 7. What this program does not do

No window in sections 4-5 is edited after this commit. No E2/E3-style
default changes. No lane runs before the previous closes. No claim is
locked to a fixture: every lane's deliverable is a table of cells with
verdicts and the law's range. The three reviews' scratch files
(`nu_full_assumptions_repro.{py,json}`, `inplane_tied_set.json`) are
evidence for the gap map only; none of their numbers is a lane result.

## Addendum (Lane 1 result, 2026-09-14; sections 1-7 above not edited)

Section 2's claim that the kernel path is an exact cyclic relabeling
x -> y -> z -> x was measured in Lane 1 arm L1-R (pattern S, n_b = 4,
B_sym, 1200 steps; stored in `results/e5_relabel.json`, lane note
`20260913_nu_lane1_multilevel_xyz_predeclaration.md`, W5):

- z -> x: exact. A and B traces bit-identical (relative max difference
  0.0), `R_meas` identical in float64, `dt_x - dt_z = 0`.
- z -> y: not exact. Trace relative max difference 6.71e-7 (A) and
  6.85e-7 (B_sym), first differing step 16, 1.8 float32 eps RMS against
  the running maximum and non-growing over 1200 steps; `R_meas` relative
  difference 6.02e-6. Deterministic (y re-run twice is bit-identical to
  itself). The W5 `R_meas` sub-criterion (1e-6) fired on y; both trace
  criteria held; the window was not widened and the y cells carry the
  declared "relabel identity fired: instrument class" flag. The y arms
  then agree with the x arms to <= 4.1e-5 relative on all 40 arm pairs
  with identical W1-W3 verdicts.

Gap-map candidate (not located, not fixed in Lane 1; no `rfx/` change):
a float32 summation-order difference on the z -> y leg of the
relabeling, somewhere in the kernel path listed in section 2. Its size
(7e-7 on traces) is far below every Lane 1 window, so it changes no
verdict; the claim "exact cyclic relabelings" should be read as
"exact for z -> x, float32-roundoff class for z -> y" until the leg is
located. The scan of the rectangular DFT window ends recorded in the
lane note's second pass (about 3 % of `R_meas` on this instrument, TE10
cutoff ringing) is the other Lane 1 finding that later lanes reusing
the E1 observable should read before quoting sub-3 % agreement.
