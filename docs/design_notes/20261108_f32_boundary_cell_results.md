# F32 boundary-cell amendment results — 2026-11-08

Base: `e59f4586cf4291bff7fd0136a20e390624be7e6c`.
Design: rfx-archive `968f6a941`, `rfx/records/20261108-f32-boundary-cell/design.md`,
including the final decision record, “nominal profile; tolerance 4 eps”.
Branch: `sw-structure/f32-boundary-cell`.
Implementation commit: the single commit containing this report, directly above the base;
its full hash is supplied in the delivery message (a committed file cannot contain its own commit hash).

Decision 1a uses `max(DECLARED_SPAN_TOL_M, 4 * eps(dtype) * abs(boundary))`.
Decision 3a checks traced x/y boundary cells only when that axis has no nominal entry.
Each branch explains the existing nominal/window and structure-resolution contract in three lines.
No existing test expectations, other tolerances, pad/bounding-node functions, or z handling changed.

## Diff of graded_grid.py against e59f4586c

```diff
diff --git a/rfx/model/graded_grid.py b/rfx/model/graded_grid.py
index c240ea742..15c935f37 100644
--- a/rfx/model/graded_grid.py
+++ b/rfx/model/graded_grid.py
@@ -776,7 +776,7 @@ def _assert_boundary_cells(profile, boundary, axis, *, match_each_other=False):
     dtype = getattr(profile, "dtype", np.dtype(np.float64))
     eps = np.finfo(dtype if np.issubdtype(dtype, np.floating) else np.float64).eps
     boundary = float(boundary)
-    tolerance = max(DECLARED_SPAN_TOL_M, float(eps) * abs(boundary))
+    tolerance = max(DECLARED_SPAN_TOL_M, 4 * float(eps) * abs(boundary))

     def _check(lo, hi):
         ends = ((-1, hi),) if match_each_other else ((0, lo), (-1, hi))
@@ -897,10 +897,12 @@ def make_nonuniform_grid(
         nx_interior = realized_cells(domain_xy[0], dx, kinds["x"])
         dx_prof_phys = np.full(nx_interior, float(dx))
     elif is_tracer(dx_profile):
-        # CPML reads the traced end cells; validate against the stored scalar
-        # when their values exist, using the realized float32 precision.
+        # With a nominal profile, ends follow the fixed-cell check outside
+        # declared windows and are free inside; agreement with dx is judged
+        # where a structure is resolved (rfx#2372).
         dx_prof_phys = jnp.asarray(dx_profile, dtype=jnp.float32)
-        _assert_boundary_cells(dx_prof_phys, dx, "x")
+        if nominal_profiles is None or "x" not in nominal_profiles:
+            _assert_boundary_cells(dx_prof_phys, dx, "x")
     else:
         _assert_boundary_cells(dx_profile, dx, "x")
         dx_prof_phys = np.asarray(dx_profile, dtype=np.float64)
@@ -917,11 +919,13 @@ def make_nonuniform_grid(
         dy_prof_phys = np.full(ny_interior, float(dx))
         dy_boundary = float(dx)
     elif is_tracer(dy_profile):
-        # Traced y stores the concrete dx as its boundary width, so both
-        # realized ends must match it at float32 precision.
+        # With a nominal profile, ends follow the fixed-cell check outside
+        # declared windows and are free inside; agreement with dx is judged
+        # where a structure is resolved (rfx#2372).
         dy_prof_phys = jnp.asarray(dy_profile, dtype=jnp.float32)
         dy_boundary = float(dx)
-        _assert_boundary_cells(dy_prof_phys, dy_boundary, "y")
+        if nominal_profiles is None or "y" not in nominal_profiles:
+            _assert_boundary_cells(dy_prof_phys, dy_boundary, "y")
     else:
         dy_boundary = float(dy_profile[0])
         _assert_boundary_cells(dy_profile, dy_boundary, "y", match_each_other=True)
```

## T7

The initial 20-cell float32 profile used 2e-3 for the first cell and 1e-3 for the other
19 cells, widened unchanged to numpy float64 as the nominal. A JVP through Simulation
with the original 20 mm x domain was refused before grid construction:

```text
ValueError: nominal_profiles['x'] sums to 0.021000000997446477 m but the declared domain x extent is 0.02 m; the nominal profile must describe that same domain.
```

The closest declaration changes only the x domain to the nominal sum
(`0.021000000997446477` m). It retains dx=1e-3, the same 20 float32 cells, the same
profile widened to numpy float64, and no window. The Simulation reaches
`make_nonuniform_grid` under `jax.jvp`; the first interior width is compared with
the literal `np.float32(2e-3)`, not a value derived from the nominal.

```python
def test_t7_traced_nominal_x_boundary_can_differ_from_dx():
    profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[0].set(2e-3)
    nominal = np.asarray(profile, dtype=np.float64)

    def build(cells):
        # The nominal sum must match the declared extent, including f32 rounding.
        grid = Simulation(
            freq_max=20e9, domain=(float(nominal.sum()), 20e-3, 10e-3), dx=1e-3,
            boundary="cpml", cpml_layers=4, dx_profile=cells,
            nominal_profiles={"x": nominal},
        )._build_nonuniform_grid()
        assert grid.dx == 0.001
        assert grid.nominal_cells["x"].dtype == np.dtype("float64")
        assert grid.traced_cells is None
        return grid.dx_arr[grid.pad_x_lo]

    first = np.asarray(_jvp_cells(build, profile))
    assert first.dtype == np.dtype("float32")
    assert first == np.float32(2e-3)
```

## Mutation results

Each mutation started from the passing implementation, ran only
`tests/contracts/test_graded_boundary_cell_rule.py` in its own mac-serial call,
and restored the source in a `finally` block before the next mutation.
The final source was compared byte-for-byte with the pre-mutation copy.

| Mutation | Result | Failing test and test-file line | Failure |
| --- | --- | --- | --- |
| M1: replace both traced-axis guards with `if False:` | Red: 3 failed, 6 passed | T3[jvp]:53; T4[jvp]:61; T6:79 | `Failed: DID NOT RAISE Exception` |
| M2: replace boundary tolerance with `DECLARED_SPAN_TOL_M` | Red: 5 failed, 4 passed | T1:32; T2:42; T4[concrete]:61; T4[jvp]:61; T6:76 | T1/T2/T6 refuse valid float32 widths at the 1e-12 floor; both T4 cases fail their last-cell message assertion because the first cell is already refused. |
| M3: replace both traced-axis guards with `if True:` | Red: 1 failed, 8 passed | T7:99 (JVP build) | `ValueError: dx_profile[0]=0.0020000000949949026 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=4.76837158203125e-10 m.` |

M2's T1 diagnostic is
`ValueError: dx_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.`
T1's success contract fails at its build call; M3 likewise fails T7's build-success
contract before the final literal assertion at line 101. Both errors originate
at `rfx/model/graded_grid.py:792`. Full assertion and traceback output follows below.

## Regression results

The specified 17-file pytest invocation completed with exit 0:

```text
1637 passed, 4 deselected, 1 xfailed, 454 warnings in 335.36s (0:05:35)
```

T1–T7 comprise nine parametrized cases; all nine passed. The four deselected
cases are the existing slow width-16/24 uniform/graded cases of
`test_resistive_strip_is_solved_at_its_drawn_width` in `test_sheet_footprint_width.py`.

All seven previously red cases pass unchanged:

- `tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_nominal_edge_mismatch_refuses_candidate_under_jit[clear]`
- `tests/unit/nonuniform/test_traced_cell_window.py::test_e4_fixed_widths[outside-grad]`
- `tests/unit/nonuniform/test_traced_cell_window.py::test_e4_fixed_widths[outside-jit_grad]`
- `tests/unit/nonuniform/test_traced_cell_window.py::test_e4_fixed_widths[two_ulp-grad]`
- `tests/unit/nonuniform/test_traced_cell_window.py::test_e4_fixed_widths[two_ulp-jit_grad]`
- `tests/unit/nonuniform/test_traced_cell_window.py::test_two_windows_feed_cells_stay_fixed[below]`
- `tests/unit/nonuniform/test_traced_cell_window.py::test_two_windows_feed_cells_stay_fixed[above]`

The requested regression measurement includes `test_nonuniform_forward_grad.py`,
`test_pinned_sheet_on_traced_mesh.py`, `test_wire_port_on_traced_mesh.py`,
`test_cpml_boundary_cell_gradient.py`, and `test_waveguide_traced_window_gradient.py`;
all selected tests in those files passed. No claim is made for tests outside the
specified invocation or its four deselected cases. The existing strict xfail is
`test_b_the_traced_record_at_the_primal_is_the_concrete_forward`
in `test_traced_graded_axis_nominal_profile.py` (#1373).

Ruff completed with exit 0: `All checks passed!`.
Real-target smoke: none needed; this change is a CPU-only grid build admission rule.
All started commands exited normally; no signal was sent to a mac-serial call.
No push, PR, remote write, or cluster job was performed.

## Commands and complete outputs

Every invocation used `REMI_AGENT_SESSION=sw-structure REMI_AGENT_GROUP=rfx/core`.

Pytest command:

```sh
REMI_AGENT_SESSION=sw-structure REMI_AGENT_GROUP=rfx/core ~/bin/mac-serial ~/Documents/rfx/.venv/bin/python -m pytest -q tests/contracts/test_graded_boundary_cell_rule.py tests/contracts/test_traced_mesh_conductor_matches_concrete.py tests/unit/nonuniform/test_traced_cell_window.py tests/unit/autodiff/test_ad_memory_grid_and_sheet.py tests/unit/autodiff/test_nonuniform_forward_grad.py tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py tests/unit/autodiff/test_wire_port_on_traced_mesh.py tests/unit/nonuniform/test_traced_graded_axis_nominal_profile.py tests/unit/sources/test_wire_radius_traced_axis.py tests/unit/sparams/test_waveguide_traced_window_gradient.py tests/unit/boundaries/test_cpml_boundary_cell_gradient.py tests/unit/nonuniform/test_waveguide_port_local_metric.py tests/unit/materials/test_sheet_footprint_width.py tests/contracts/test_film_drawn_size.py tests/contracts/test_sheet_drawn_gap.py tests/contracts/test_traced_mesh_conductor_reaching_absorber_refused.py tests/contracts/test_waveguide_port_traced_mesh_refused.py
```

<details>
<summary>pytest complete output</summary>

```text
mac-serial: waiting for a free slot (holders: 99876 2026-11-08 06:36:12 ?@showcase--marker /Users/byungkwankim/Documents/rfx/.;4110 2026-11-08 06:37:28 ?@rfx-adv--f-distance /Users/byungkwankim/rfx-adv/.venv;)
........................................................................ [  4%]
........................................................................ [  8%]
...................................x.................................... [ 13%]
........................................................................ [ 17%]
........................................................................ [ 21%]
........................................................................ [ 26%]
........................................................................ [ 30%]
........................................................................ [ 35%]
........................................................................ [ 39%]
........................................................................ [ 43%]
........................................................................ [ 48%]
........................................................................ [ 52%]
........................................................................ [ 57%]
........................................................................ [ 61%]
........................................................................ [ 65%]
........................................................................ [ 70%]
........................................................................ [ 74%]
........................................................................ [ 79%]
........................................................................ [ 83%]
........................................................................ [ 87%]
........................................................................ [ 92%]
........................................................................ [ 96%]
......................................................                   [100%]
=============================== warnings summary ===============================
tests/contracts/test_traced_mesh_conductor_matches_concrete.py: 13 warnings
tests/unit/nonuniform/test_traced_graded_axis_nominal_profile.py: 2 warnings
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/nonuniform/test_traced_graded_axis_nominal_profile.py:77: AccuracyWarning: dx_profile has max adjacent cell ratio 2.000 (> 1.4, the validated in-plane grading cap — docs/guides/support_matrix.md, 'Multi-band graded mesh'). This may cause numerical reflections. Use rfx.smooth_grading(dx_profile) to fix.
    sim = Simulation(freq_max=40e9, domain=(LX, N_Y * DX, N_Z * DX), dx=DX,

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_a1_board_eager
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_a1_ring_jit
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_a4_clear_shape_is_identity_with_check
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_a6_paths[nonuniform]
tests/contracts/test_traced_mesh_conductor_reaching_absorber_refused.py::test_reaching_box_continues
tests/contracts/test_traced_mesh_conductor_reaching_absorber_refused.py::test_box_ending_one_tenth_cell_short_on_traced_axis_continues
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:40: AccuracyWarning: Soft-source PEC admission check cannot run on a traced mesh or traced conductor edges/source positions; no concrete realized edge classification is available.
    product = solve_conductors(sim, grid, nonuniform=True)

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_a8_shape_refusals[sheet]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:204: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(Cylinder((.002, .005, board.Z_G), .002, 0., axis="z"))

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_unpadded_traced_axis_needs_nominal_for_candidate[True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:217: AccuracyWarning: dy_profile has max adjacent cell ratio 4.000 (> 1.4, the validated in-plane grading cap — docs/guides/support_matrix.md, 'Multi-band graded mesh'). This may cause numerical reflections. Use rfx.smooth_grading(dy_profile) to fix.
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_lattice_wrapper_keeps_refusal
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:265: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(shape, sigma_bulk=5.8e7)

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_holed_impedance_sheet_refused[eager]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_holed_impedance_sheet_refused[jit]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:279: AccuracyWarning: dy_profile has max adjacent cell ratio 4.000 (> 1.4, the validated in-plane grading cap — docs/guides/support_matrix.md, 'Multi-band graded mesh'). This may cause numerical reflections. Use rfx.smooth_grading(dy_profile) to fix.
    sim = Simulation(domain=(8., 8., 8.), dx=1., freq_max=1e6,

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_holed_impedance_sheet_refused[eager]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_holed_impedance_sheet_refused[jit]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:283: AccuracyWarning: add_thin_conductor: surface_impedance_f0=1000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 0.0002609 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(

tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_impedance_box_sheet_matches_concrete_and_checks_drift[both_x_faces]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_impedance_box_sheet_matches_concrete_and_checks_drift[both_x_faces]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_impedance_box_sheet_matches_concrete_and_checks_drift[both_x_faces]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_impedance_box_sheet_matches_concrete_and_checks_drift[inset_1p1]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_impedance_box_sheet_matches_concrete_and_checks_drift[inset_1p1]
tests/contracts/test_traced_mesh_conductor_matches_concrete.py::test_impedance_box_sheet_matches_concrete_and_checks_drift[inset_1p1]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_traced_mesh_conductor_matches_concrete.py:316: AccuracyWarning: add_thin_conductor: surface_impedance_f0=1000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 0.0002609 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(Box((lo, 2 * board.DX, board.Z_G),

tests/unit/autodiff/test_ad_memory_grid_and_sheet.py::test_pec_sheet_is_not_charged_for_the_f0_operator
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_ad_memory_grid_and_sheet.py:185: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(Box((0.002, 0.002, 0.002), (0.018, 0.018, 0.002)),

tests/unit/autodiff/test_ad_memory_grid_and_sheet.py::test_unbuildable_profile_refuses_ad_tape_estimation
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_ad_memory_grid_and_sheet.py:236: AccuracyWarning: dx_profile has max adjacent cell ratio 3.000 (> 1.4, the validated in-plane grading cap — docs/guides/support_matrix.md, 'Multi-band graded mesh'). This may cause numerical reflections. Use rfx.smooth_grading(dx_profile) to fix.
    sim = Simulation(freq_max=20e9, domain=(4e-3, 4e-3, 3e-3), dx=DX,

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_smoke
tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_grad_eps_matches_fd
tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_accepted
tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_grad
tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_interpolates_hard_pec
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:40: DeprecationWarning: add_source(..., amplitude_kind=None) now means 'current' (E += Cb*I/dV, I is a current moment in A·m) on every path. Pass amplitude_kind explicitly to silence this warning.
    sim.add_source((0.005, 0.005, 0.001), "ez")

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_smoke
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:48: AccuracyWarning: [forward] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - face x_lo is declared cpml with 4 layers. face x_hi is declared cpml with 4 layers. face y_lo is declared cpml with 4 layers. face y_hi is declared cpml with 4 layers. face z_lo is declared cpml with 4 layers. face z_hi is declared cpml with 4 layers. CPML reference reflections measured at 1 mm cells with the shipped CFS alpha: 4 layers: -20 dB at 2 GHz, -23 dB at 10 GHz, -17 dB at 30 GHz, -6 dB below 1 GHz. These reference values do not predict reflection on a different mesh or for UPML. Use at least 8 absorbing layers; 16 layers is the default recommendation.
    fr = sim.forward(n_steps=100)

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_grad_eps_matches_fd
tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_grad_eps_matches_fd
tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_grad_eps_matches_fd
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:74: AccuracyWarning: [forward] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - face x_lo is declared cpml with 4 layers. face x_hi is declared cpml with 4 layers. face y_lo is declared cpml with 4 layers. face y_hi is declared cpml with 4 layers. face z_lo is declared cpml with 4 layers. face z_hi is declared cpml with 4 layers. CPML reference reflections measured at 1 mm cells with the shipped CFS alpha: 4 layers: -20 dB at 2 GHz, -23 dB at 10 GHz, -17 dB at 30 GHz, -6 dB below 1 GHz. These reference values do not predict reflection on a different mesh or for UPML. Use at least 8 absorbing layers; 16 layers is the default recommendation.
    fr = sim.forward(eps_override=eps, n_steps=n_steps)

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_accepted
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:114: AccuracyWarning: [forward] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - face x_lo is declared cpml with 4 layers. face x_hi is declared cpml with 4 layers. face y_lo is declared cpml with 4 layers. face y_hi is declared cpml with 4 layers. face z_lo is declared cpml with 4 layers. face z_hi is declared cpml with 4 layers. CPML reference reflections measured at 1 mm cells with the shipped CFS alpha: 4 layers: -20 dB at 2 GHz, -23 dB at 10 GHz, -17 dB at 30 GHz, -6 dB below 1 GHz. These reference values do not predict reflection on a different mesh or for UPML. Use at least 8 absorbing layers; 16 layers is the default recommendation.
    fr_base = sim.forward(n_steps=50)

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_accepted
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:119: AccuracyWarning: [forward] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - face x_lo is declared cpml with 4 layers. face x_hi is declared cpml with 4 layers. face y_lo is declared cpml with 4 layers. face y_hi is declared cpml with 4 layers. face z_lo is declared cpml with 4 layers. face z_hi is declared cpml with 4 layers. CPML reference reflections measured at 1 mm cells with the shipped CFS alpha: 4 layers: -20 dB at 2 GHz, -23 dB at 10 GHz, -17 dB at 30 GHz, -6 dB below 1 GHz. These reference values do not predict reflection on a different mesh or for UPML. Use at least 8 absorbing layers; 16 layers is the default recommendation.
    fr_zero = sim.forward(pec_occupancy_override=occ_zero, n_steps=50)

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_accepted
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:128: AccuracyWarning: [forward] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - face x_lo is declared cpml with 4 layers. face x_hi is declared cpml with 4 layers. face y_lo is declared cpml with 4 layers. face y_hi is declared cpml with 4 layers. face z_lo is declared cpml with 4 layers. face z_hi is declared cpml with 4 layers. CPML reference reflections measured at 1 mm cells with the shipped CFS alpha: 4 layers: -20 dB at 2 GHz, -23 dB at 10 GHz, -17 dB at 30 GHz, -6 dB below 1 GHz. These reference values do not predict reflection on a different mesh or for UPML. Use at least 8 absorbing layers; 16 layers is the default recommendation.
    fr_shield = sim.forward(pec_occupancy_override=occ_shield, n_steps=50)

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_forward_nonuniform_pec_occupancy_grad
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:143: AccuracyWarning: [forward] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - face x_lo is declared cpml with 4 layers. face x_hi is declared cpml with 4 layers. face y_lo is declared cpml with 4 layers. face y_hi is declared cpml with 4 layers. face z_lo is declared cpml with 4 layers. face z_hi is declared cpml with 4 layers. CPML reference reflections measured at 1 mm cells with the shipped CFS alpha: 4 layers: -20 dB at 2 GHz, -23 dB at 10 GHz, -17 dB at 30 GHz, -6 dB below 1 GHz. These reference values do not predict reflection on a different mesh or for UPML. Use at least 8 absorbing layers; 16 layers is the default recommendation.
    fr = sim.forward(pec_occupancy_override=occ, n_steps=40)

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_sim_forward_grad_wrt_dz_profile_through_init
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:225: DeprecationWarning: add_source(..., amplitude_kind=None) now means 'current' (E += Cb*I/dV, I is a current moment in A·m) on every path. Pass amplitude_kind explicitly to silence this warning.
    sim.add_source((0.005, 0.005, 0.001), "ez")

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_sim_forward_grad_wrt_dx_profile_through_init
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:306: DeprecationWarning: add_source(..., amplitude_kind=None) now means 'current' (E += Cb*I/dV, I is a current moment in A·m) on every path. Pass amplitude_kind explicitly to silence this warning.
    sim.add_source((0.005, 0.005, 0.001), "ez")

tests/unit/autodiff/test_nonuniform_forward_grad.py::test_sim_forward_grad_wrt_dy_profile_through_init
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_nonuniform_forward_grad.py:330: DeprecationWarning: add_source(..., amplitude_kind=None) now means 'current' (E += Cb*I/dV, I is a current moment in A·m) on every path. Pass amplitude_kind explicitly to silence this warning.
    sim.add_source((0.005, 0.005, 0.001), "ez")

tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py::test_a_metric_sheet_on_a_traced_mesh_is_still_refused
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py:129: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(Box((I_LO * DX, J_LO * DX, Z_PATCH),

tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py::test_patch_edge_gradient_matches_a_central_difference[A_plus_x_edge]
tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py::test_patch_edge_gradient_matches_a_central_difference[B_probe_past_the_edge]
tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py::test_patch_edge_gradient_matches_a_central_difference[C_joint_xy]
tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py::test_patch_edge_gradient_matches_a_central_difference[D_low_side_edge]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py:360: AccuracyWarning: Soft-source PEC admission check cannot run on a traced mesh or traced conductor edges/source positions; no concrete realized edge classification is available.
    return jnp.sum(sim.forward(n_steps=n_steps, skip_preflight=True)

tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py::test_fidelity_report_cross_checks_a_pinned_and_a_metric_sheet_together
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_pinned_sheet_on_traced_mesh.py:491: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(Box((2 * DX, 2 * DX, Z_GND),

tests/unit/autodiff/test_wire_port_on_traced_mesh.py: 10 warnings
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_wire_port_on_traced_mesh.py:148: AccuracyWarning: Soft-source PEC admission check cannot run on a traced mesh or traced conductor edges/source positions; no concrete realized edge classification is available.
    r = run_nonuniform_path(_board(delta, pin_dt=pin_dt, axis=axis),

tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_a_passive_wire_port_also_sizes_itself_on_the_traced_mesh
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_wire_port_on_traced_mesh.py:389: AccuracyWarning: Soft-source PEC admission check cannot run on a traced mesh or traced conductor edges/source positions; no concrete realized edge classification is available.
    r = run_nonuniform_path(_driven_board_with_probe(d), n_steps=60,

tests/unit/autodiff/test_wire_port_on_traced_mesh.py::test_the_pinned_step_reaches_the_waveguide_s_matrix_lane
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/autodiff/test_wire_port_on_traced_mesh.py:631: AccuracyWarning: dx_profile has max adjacent cell ratio 1.667 (> 1.4, the validated in-plane grading cap — docs/guides/support_matrix.md, 'Multi-band graded mesh'). This may cause numerical reflections. Use rfx.smooth_grading(dx_profile) to fix.
    sim = Simulation(

tests/unit/sources/test_wire_radius_traced_axis.py::test_forward_width_gradient_is_finite
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sources/test_wire_radius_traced_axis.py:215: ConfigurationWarning: S-parameters from one record require an isolated drive; excited ports: port 0 at (0.007, 0.007, 0.007); port 1 at (0.007, 0.01, 0.007). set excite=False on every port but the driven one and run once per driven port, or use run(compute_s_params=True) on one device, which drives one port per run; returning no S-parameters.
    result = sim.forward(n_steps=24, skip_preflight=True)

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_window_flux_jacfwd_reaches_extraction
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: [compute_waveguide_s_matrix] preflight found 1 advisory issue(s) - pass skip_preflight=True to suppress:
    - all dielectric(s) ['slab'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_window_flux_jacfwd_reaches_extraction
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightErrorWarning: the record ends BEFORE the far-boundary round trip arrives — waveguide_port[0] (+x): T/tau_far = 0.005419. T = 0.008746 ns (2 steps x dt = 4.373 ps; num_periods = 20 at freq_max = 11.6 GHz), tau_far = 2 x far_path / v_g = 1.614 ns. far_path = 152.4 mm (port plane 12.7 mm to the x-hi outer wall on a 121.9 mm domain, absorber pad 43.18 mm included); v_g(f_min)/c = 0.62993 at f_min = 8.4 GHz with the port's discrete cutoff f_c = 6.5239 GHz. num_periods >= 57 makes T/tau_far >= 3. Finer grids need more — measured 3, 5 and 8 at a/18, a/36 and a/72 on the WR-90 battery. A multimode port is audited on its lowest-cutoff mode, so this is a lower bound.
    self._preflight_waveguide_setup(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_window_flux_jacfwd_reaches_extraction
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightErrorWarning: the record ends BEFORE the far-boundary round trip arrives — waveguide_port[1] (-x): T/tau_far = 0.005419. T = 0.008746 ns (2 steps x dt = 4.373 ps; num_periods = 20 at freq_max = 11.6 GHz), tau_far = 2 x far_path / v_g = 1.614 ns. far_path = 152.4 mm (port plane 109.2 mm to the x-lo outer wall on a 121.9 mm domain, absorber pad 43.18 mm included); v_g(f_min)/c = 0.62993 at f_min = 8.4 GHz with the port's discrete cutoff f_c = 6.5239 GHz. num_periods >= 57 makes T/tau_far >= 3. Finer grids need more — measured 3, 5 and 8 at a/18, a/36 and a/72 on the WR-90 battery. A multimode port is audited on its lowest-cutoff mode, so this is a lower bound.
    self._preflight_waveguide_setup(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_window_flux_jacfwd_reaches_extraction
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightWarning: waveguide port index mirror audit, source E plane (x-axis): i(+) = 22, i(-) = 61, sum = 83 on n_axis = 83; the covariant sum for a primal plane is 82. The +1 difference is the KNOWN shipped offset — apply_waveguide_port_e places the '-' port's E correction at x_index + 1 — not an asymmetry in this setup. Reported as information.
    self._preflight_waveguide_setup(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_window_flux_jacfwd_reaches_extraction
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: ring-down settling witness NOT PASSED (pole-tail amplitude threshold -40 dB): port left driven: undetermined (source end is unavailable; source end is unavailable), port right driven: undetermined (source end is unavailable; source end is unavailable). The num_periods=20 record has no passing value witness. Increase num_periods and check identification before quoting any S value (see the result's settling_db and settling_witness fields).
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightWarning: the record is shorter than 3 x the far-boundary round trip — waveguide_port[0] (+x): T/tau_far = 2.138. T = 3.4503 ns (789 steps x dt = 4.373 ps; num_periods = 20 at freq_max = 11.6 GHz), tau_far = 2 x far_path / v_g = 1.614 ns. far_path = 152.4 mm (port plane 12.7 mm to the x-hi outer wall on a 121.9 mm domain, absorber pad 43.18 mm included); v_g(f_min)/c = 0.62993 at f_min = 8.4 GHz with the port's discrete cutoff f_c = 6.5239 GHz. num_periods >= 57 makes T/tau_far >= 3. Finer grids need more — measured 3, 5 and 8 at a/18, a/36 and a/72 on the WR-90 battery. A multimode port is audited on its lowest-cutoff mode, so this is a lower bound.
    self._preflight_waveguide_setup(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightWarning: the record is shorter than 3 x the far-boundary round trip — waveguide_port[1] (-x): T/tau_far = 2.138. T = 3.4503 ns (789 steps x dt = 4.373 ps; num_periods = 20 at freq_max = 11.6 GHz), tau_far = 2 x far_path / v_g = 1.614 ns. far_path = 152.4 mm (port plane 109.2 mm to the x-lo outer wall on a 121.9 mm domain, absorber pad 43.18 mm included); v_g(f_min)/c = 0.62993 at f_min = 8.4 GHz with the port's discrete cutoff f_c = 6.5239 GHz. num_periods >= 57 makes T/tau_far >= 3. Finer grids need more — measured 3, 5 and 8 at a/18, a/36 and a/72 on the WR-90 battery. A multimode port is audited on its lowest-cutoff mode, so this is a lower bound.
    self._preflight_waveguide_setup(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: [compute_waveguide_s_matrix] preflight found 4 advisory issue(s) - pass skip_preflight=True to suppress:
    - dielectric 'slab' on x: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - dielectric 'slab' on y: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - dielectric 'slab' on z: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - all dielectric(s) ['slab'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: compute_waveguide_s_matrix: the returned S is not passive at 11 of 17 frequency bins, worst sigma_max = 1.01849 at 11.4 GHz. S is returned as extracted. A passive structure cannot scatter more power than it receives, so these bins are an extraction or record artifact.
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: [compute_waveguide_s_matrix] preflight found 4 advisory issue(s) - pass skip_preflight=True to suppress:
    - dielectric 'slab' on x: 5.2 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.49mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - dielectric 'slab' on y: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - dielectric 'slab' on z: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - all dielectric(s) ['slab'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: compute_waveguide_s_matrix: the returned S is not passive at 11 of 17 frequency bins, worst sigma_max = 1.01828 at 11.4 GHz. S is returned as extracted. A passive structure cannot scatter more power than it receives, so these bins are an extraction or record artifact.
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: [compute_waveguide_s_matrix] preflight found 4 advisory issue(s) - pass skip_preflight=True to suppress:
    - dielectric 'slab' on x: 5.0 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.59mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - dielectric 'slab' on y: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - dielectric 'slab' on z: 5.1 cells per λ_eff (eps_r=4.00, freq_max=11.6GHz, dx=2.54mm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
    - all dielectric(s) ['slab'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/sparams/test_waveguide_traced_window_gradient.py::test_b1_b2_window_gradient_all_17_bins
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/sparams/test_waveguide_traced_window_gradient.py:101: AccuracyWarning: compute_waveguide_s_matrix: the returned S is not passive at 10 of 17 frequency bins, worst sigma_max = 1.01868 at 11.4 GHz. S is returned as extracted. A passive structure cannot scatter more power than it receives, so these bins are an extraction or record artifact.
    return jnp.asarray(section(delta).compute_waveguide_s_matrix(

tests/unit/materials/test_sheet_footprint_width.py: 14 warnings
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/materials/test_sheet_footprint_width.py:89: AccuracyWarning: no ring-down settling witness on this run: this result recorded no probe time series: add a point probe (sim.add_probe(position, component)) away from every source/port drive cell — a record on the drive cell measures source turn-off, not ring-down, and is not an independent witness — and retain its time series, or use run(until_decay=...) to bound the ring-down through its stop criterion. This run requests NTFF and/or field-DFT output, so its truncation is unguarded; settling_db is None, which is not a pass.
    result = sim.run(n_steps=N_STEPS, compute_s_params=False, skip_preflight=True)

tests/unit/materials/test_sheet_footprint_width.py: 20 warnings
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/materials/test_sheet_footprint_width.py:71: AccuracyWarning: add_thin_conductor: surface_impedance_f0=9000000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 376.7 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(

tests/unit/materials/test_sheet_footprint_width.py::test_end_row_marking_cases
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/materials/test_sheet_footprint_width.py:166: AccuracyWarning: add_thin_conductor: surface_impedance_f0=9000000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 1.885 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(Box(tuple(v * DX for v in lo), tuple(v * DX for v in hi)),

tests/unit/materials/test_sheet_footprint_width.py::test_end_row_marking_cases
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/materials/test_sheet_footprint_width.py:191: AccuracyWarning: add_thin_conductor: surface_impedance_f0=9000000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 1.885 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(Cylinder((5 * DX, 5 * DX, 5 * DX), 3 * DX, 0.0, axis="z"),

tests/unit/materials/test_sheet_footprint_width.py::test_traced_and_eager_graded_meshes_solve_the_same_strip
tests/unit/materials/test_sheet_footprint_width.py::test_traced_and_eager_graded_meshes_solve_the_same_strip
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/unit/materials/test_sheet_footprint_width.py:287: AccuracyWarning: add_thin_conductor: surface_impedance_f0=9000000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 376.7 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(Box((0.0, 1 * DX, z), (NX * DX, 9 * DX, z)),

tests/contracts/test_film_drawn_size.py::test_t4_whole_plane_on_cpml_grid[True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_film_drawn_size.py:101: AccuracyWarning: dy_profile has max adjacent cell ratio 1.500 (> 1.4, the validated in-plane grading cap — docs/guides/support_matrix.md, 'Multi-band graded mesh'). This may cause numerical reflections. Use rfx.smooth_grading(dy_profile) to fix.
    sim = Simulation(10e9, (.024, .016, .012), dx=.001, boundary='cpml', cpml_layers=8, **kw)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[1-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.8000000000000004e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0571909584179, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[1-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.8200000000000002e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.062385538123, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[2-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.9600000000000002e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0101525445522, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[2-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.96e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0964920970947, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[2-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.996e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.104677037928, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[3-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_mirrored[1-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.8000000000000007e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0571909584179, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[3-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=2.1999999999999993e-05 m^2, A_r=2.4e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0444659357342, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[3-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_mirrored[1-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.78e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.05190907372, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film[3-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=2.2200000000000004e-05 m^2, A_r=2.4e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0397504898201, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_extra_gaps[intervals0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.88e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0314212462588, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_extra_gaps[intervals0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.8799999999999996e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0774687919711, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_extra_gaps[intervals1]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:219: PreflightWarning: Box DC film: A_d=1.9199999999999996e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0871290708247, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0005-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.1599999999999996e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0377495513506, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.2400000000000002e-05 m^2, A_r=2.4e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0350983390135, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0005-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.1599999999999996e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0377495513506, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.2400000000000002e-05 m^2, A_r=2.4e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0350983390135, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0005-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.5680000000000005e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0413534982775, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.6719999999999995e-05 m^2, A_r=2.88e-05 m^2; e=sqrt(A_r/A_d)-1=+0.038192810309, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0005-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.5680000000000005e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0413534982775, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.6719999999999995e-05 m^2, A_r=2.88e-05 m^2; e=sqrt(A_r/A_d)-1=+0.038192810309, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.2399996000000005e-05 m^2, A_r=2.4e-05 m^2; e=sqrt(A_r/A_d)-1=+0.035098431433, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.2399996000000005e-05 m^2, A_r=2.4e-05 m^2; e=sqrt(A_r/A_d)-1=+0.035098431433, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.6719995999999997e-05 m^2, A_r=2.88e-05 m^2; e=sqrt(A_r/A_d)-1=+0.038192888018, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[1e-09-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.6719995999999997e-05 m^2, A_r=2.88e-05 m^2; e=sqrt(A_r/A_d)-1=+0.038192888018, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.2200000000000004e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0508420042475, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.2200000000000004e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0508420042475, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.6519999999999997e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0566578461098, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[5e-05-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.6519999999999997e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0566578461098, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.1200000000000004e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0287141376427, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.1200000000000004e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0287141376427, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.5519999999999996e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0383530324063, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0003-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.5519999999999996e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0383530324063, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0005-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.4719999999999994e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0229163000877, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0005-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.4719999999999994e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0229163000877, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.8600000000000005e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0369516947304, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=1.8600000000000005e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0369516947304, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.2919999999999997e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0147257816116, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.00095-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.2919999999999997e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0147257816116, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.84e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0425720702854, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=1.84e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0425720702854, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.271999999999999e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0191822184311, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.001-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.271999999999999e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0191822184311, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.7200000000000008e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0355143556592, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-False]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=1.7200000000000008e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0355143556592, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2.1519999999999997e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0472126217216, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_gap_sweep[0.0013-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:312: PreflightWarning: Box DC film: A_d=2.1519999999999997e-05 m^2, A_r=2.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0472126217216, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    record = sim.realized_geometry()

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_coverage_owner[False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_coverage_owner[True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_vmap_uses_separated_masks
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_preflight_reports_solved_area
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.9600000000000002e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0101525445522, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_coverage_owner[False]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_coverage_owner[True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_vmap_uses_separated_masks
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_preflight_reports_solved_area
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.9199999999999996e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0871290708247, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_coverage_owner_fraction
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.92e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0206207261597, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_coverage_owner_fraction
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.6799999999999995e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0240999270515, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_periodic_images[0.0002]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.9200000000000003e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0871290708247, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_vmap_uses_separated_masks
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:365: PreflightWarning: Box DC film: A_d=1.9600000000000002e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0101525445522, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    _, _, _, film = _apply_batched_thin_conductors(sim, grid, *batch)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_vmap_uses_separated_masks
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:365: PreflightWarning: Box DC film: A_d=1.9199999999999996e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0871290708247, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    _, _, _, film = _apply_batched_thin_conductors(sim, grid, *batch)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals0-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=3.9999999999999996e-05 m^2, A_r=4.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0440306508911, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals0-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:412: PreflightWarning: Box DC film: A_d=3.9999999999999996e-05 m^2, A_r=4.36e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0440306508911, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    old, _ = apply_thin_conductor(grid, tc, old, snap=sim._snap)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals1-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=2e-05 m^2, A_r=2.44e-05 m^2; e=sqrt(A_r/A_d)-1=+0.104536101719, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals1-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals2-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.9999999999999998e-05 m^2, A_r=1.92e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0202041028867, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals1-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:412: PreflightWarning: Box DC film: A_d=2e-05 m^2, A_r=2.44e-05 m^2; e=sqrt(A_r/A_d)-1=+0.104536101719, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    old, _ = apply_thin_conductor(grid, tc, old, snap=sim._snap)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals1-True]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals2-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:412: PreflightWarning: Box DC film: A_d=1.9999999999999998e-05 m^2, A_r=1.92e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0202041028867, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    old, _ = apply_thin_conductor(grid, tc, old, snap=sim._snap)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals3-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.2e-05 m^2, A_r=9.6e-06 m^2; e=sqrt(A_r/A_d)-1=-0.105572809, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals3-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:275: PreflightWarning: Box DC film: A_d=1.1999999999999997e-05 m^2, A_r=1.48e-05 m^2; e=sqrt(A_r/A_d)-1=+0.110555416597, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, grid, nonuniform=graded)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals3-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:412: PreflightWarning: Box DC film: A_d=1.2e-05 m^2, A_r=9.6e-06 m^2; e=sqrt(A_r/A_d)-1=-0.105572809, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    old, _ = apply_thin_conductor(grid, tc, old, snap=sim._snap)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_unchanged_folds[intervals3-True]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:412: PreflightWarning: Box DC film: A_d=1.1999999999999997e-05 m^2, A_r=1.48e-05 m^2; e=sqrt(A_r/A_d)-1=+0.110555416597, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    old, _ = apply_thin_conductor(grid, tc, old, snap=sim._snap)

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[False-0-1]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[False-1-2]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[False-2-0]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[True-0-1]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[True-1-2]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[True-2-0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:427: PreflightWarning: Box DC film: A_d=1.9600000000000002e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0101525445522, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, sim._build_realized_grid())

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[False-0-1]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[False-1-2]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[False-2-0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:427: PreflightWarning: Box DC film: A_d=1.9199999999999996e-05 m^2, A_r=1.6e-05 m^2; e=sqrt(A_r/A_d)-1=-0.0871290708247, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, sim._build_realized_grid())

tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[True-0-1]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[True-1-2]
tests/contracts/test_sheet_drawn_gap.py::test_k7_dc_film_faces_and_planes[True-2-0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:427: PreflightWarning: Box DC film: A_d=1.9199999999999996e-05 m^2, A_r=2e-05 m^2; e=sqrt(A_r/A_d)-1=+0.0206207261597, bar=0.01. Put the in-plane edges on mesh nodes or refine the in-plane mesh, or use snap="declared" to accept the declared geometry.
    root = C.realized_conductors(sim, sim._build_realized_grid())

tests/contracts/test_sheet_drawn_gap.py: 168 warnings
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:464: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. You passed thickness=1e-05 m, which is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(Box((lo, .004, .004), (hi, .015, .004)),

tests/contracts/test_sheet_drawn_gap.py: 24 warnings
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:517: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(box, sigma_bulk=5.8e7, thickness=35e-6)

tests/contracts/test_sheet_drawn_gap.py::test_realization_former_seam_case
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:588: ConfigurationWarning: add_thin_conductor: sigma_bulk=5.80e+07 S/m is at or above the 1e+06 S/m PEC threshold, so this is a LOSSLESS PEC sheet. thickness is not used here, and neither is eps_r. Every metal is above this threshold — a lower sigma_bulk would be a different material, not thinner copper. For conductor loss pass surface_impedance_f0=<band centre Hz> (Leontovich sheet).
    sim.add_thin_conductor(Box((.010, lo, .005), (.010, hi, .015)),

tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[gap-pec]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[gap-f0]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[touch-pec]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[touch-f0]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[different_planes-pec]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[different_planes-f0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:642: AccuracyWarning: no field energy was recorded — the source may not have coupled (position off-grid, wrong component, or zero amplitude)
    sim.run(n_steps=3, skip_preflight=True)

tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[gap-f0]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[touch-f0]
tests/contracts/test_sheet_drawn_gap.py::test_f1_2d_run[different_planes-f0]
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_sheet_drawn_gap.py:631: AccuracyWarning: add_thin_conductor: surface_impedance_f0=15000000000.0 Hz — Leontovich surface-resistance sheet with Rs0 = 0.03195 ohm/sq. Rs is FREQUENCY-FLAT within the run, frozen at f0, with relative band error |sqrt(f/f0)-1|; the reactive part Xs = Rs (internal inductance) is NOT modeled, so resonances read slightly high.
    sim.add_thin_conductor(shape, sigma_bulk=5.8e7, surface_impedance_f0=15e9)

tests/contracts/test_waveguide_port_traced_mesh_refused.py::test_concrete_primal_still_runs
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:385: AccuracyWarning: compute_waveguide_s_matrix: absorber on the x propagation axis is thinner than the documented 0.5 guide-wavelength far-port discipline at the lowest measured frequency 8.500 GHz (mode cutoff 6.524 GHz, lambda_g = 55.0 mm): x-lo 8 cells = 20.3 mm (0.37 lambda_g), x-hi 8 cells = 20.3 mm (0.37 lambda_g), against a required 27.5 mm. A thin absorber reflects guided energy and can set the accuracy envelope instead of discretization: in the WR-90 iris lane residual |S11| ripple was 0.0706 at 0.30 lambda_g, 0.0366 at 0.50, and 0.0093 at 0.75, so 0.5 lambda_g is a floor and not a target. Raise cpml_layers to at least 11 (0.75 lambda_g needs 17).
    _warn_thin_absorber_vs_guide_wavelength(

tests/contracts/test_waveguide_port_traced_mesh_refused.py::test_concrete_primal_still_runs
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightErrorWarning: the record ends BEFORE the far-boundary round trip arrives — waveguide_port[0] (+x): T/tau_far = 0.007379. T = 0.0083874 ns (2 steps x dt = 4.1937 ps; num_periods = 20 at freq_max = 13 GHz), tau_far = 2 x far_path / v_g = 1.1367 ns. far_path = 109.2 mm (port plane 12.7 mm to the x-hi outer wall on a 101.6 mm domain, absorber pad 20.32 mm included); v_g(f_min)/c = 0.64103 at f_min = 8.5 GHz with the port's discrete cutoff f_c = 6.5239 GHz. num_periods >= 45 makes T/tau_far >= 3. Finer grids need more — measured 3, 5 and 8 at a/18, a/36 and a/72 on the WR-90 battery. A multimode port is audited on its lowest-cutoff mode, so this is a lower bound.
    self._preflight_waveguide_setup(

tests/contracts/test_waveguide_port_traced_mesh_refused.py::test_concrete_primal_still_runs
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightErrorWarning: the record ends BEFORE the far-boundary round trip arrives — waveguide_port[1] (-x): T/tau_far = 0.007379. T = 0.0083874 ns (2 steps x dt = 4.1937 ps; num_periods = 20 at freq_max = 13 GHz), tau_far = 2 x far_path / v_g = 1.1367 ns. far_path = 109.2 mm (port plane 88.9 mm to the x-lo outer wall on a 101.6 mm domain, absorber pad 20.32 mm included); v_g(f_min)/c = 0.64103 at f_min = 8.5 GHz with the port's discrete cutoff f_c = 6.5239 GHz. num_periods >= 45 makes T/tau_far >= 3. Finer grids need more — measured 3, 5 and 8 at a/18, a/36 and a/72 on the WR-90 battery. A multimode port is audited on its lowest-cutoff mode, so this is a lower bound.
    self._preflight_waveguide_setup(

tests/contracts/test_waveguide_port_traced_mesh_refused.py::test_concrete_primal_still_runs
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/sparams/waveguide.py:395: PreflightWarning: waveguide port index mirror audit, source E plane (x-axis): i(+) = 13, i(-) = 44, sum = 57 on n_axis = 57; the covariant sum for a primal plane is 56. The +1 difference is the KNOWN shipped offset — apply_waveguide_port_e places the '-' port's E correction at x_index + 1 — not an asymmetry in this setup. Reported as information.
    self._preflight_waveguide_setup(

tests/contracts/test_waveguide_port_traced_mesh_refused.py::test_concrete_primal_still_runs
  /Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/tests/contracts/test_waveguide_port_traced_mesh_refused.py:69: AccuracyWarning: ring-down settling witness NOT PASSED (pole-tail amplitude threshold -40 dB): port left driven: undetermined (source end is unavailable; source end is unavailable), port right driven: undetermined (source end is unavailable; source end is unavailable). The num_periods=20 record has no passing value witness. Up to 1.49e+07 of that share is the port's known drive beyond the record (a waveguide port's H near cutoff decays as t^-3/2). Run a longer fixed record: until_decay stops on field energy and does not judge this per-bin budget, and until_identified does not yet admit waveguide ports. Increase num_periods and check identification before quoting any S value (see the result's settling_db and settling_witness fields).
    r = sim.compute_waveguide_s_matrix(n_steps=2, normalize="flux", skip_preflight=True)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1637 passed, 4 deselected, 1 xfailed, 454 warnings in 335.36s (0:05:35)
```

</details>

Mutation M1 command (one call):

```sh
~/bin/mac-serial ~/Documents/rfx/.venv/bin/python -m pytest -q tests/contracts/test_graded_boundary_cell_rule.py
```

<details>
<summary>M1 complete output</summary>

```text
mac-serial: waiting for a free slot (holders: 45656 2026-11-08 06:43:25 ?@absorber--rf-census-rows /Users/byungkwankim/Documen;36775 2026-11-08 06:43:40 ?@G /Users/byungkwankim/Documents/rfx/.venv/bin/python;)
...F.F.F.                                                                [100%]
=================================== FAILURES ===================================
__________________ test_t3_first_x_boundary_cell_refused[jvp] __________________

traced = True

    @pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jvp"])
    def test_t3_first_x_boundary_cell_refused(traced):
        profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[0].set(1.001e-3)
>       with pytest.raises(Exception if traced else ValueError,
                           match=r"dx_profile\[0\].*must equal boundary dx"):
E                          Failed: DID NOT RAISE Exception

tests/contracts/test_graded_boundary_cell_rule.py:53: Failed
__________________ test_t4_last_x_boundary_cell_refused[jvp] ___________________

traced = True

    @pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jvp"])
    def test_t4_last_x_boundary_cell_refused(traced):
        profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[-1].set(1.001e-3)
>       with pytest.raises(Exception if traced else ValueError,
                           match=r"dx_profile\[-1\].*must equal boundary dx"):
E                          Failed: DID NOT RAISE Exception

tests/contracts/test_graded_boundary_cell_rule.py:61: Failed
___________________ test_t6_traced_y_boundary_cells_checked ____________________

    def test_t6_traced_y_boundary_cells_checked():
        profile = jnp.full(20, 1e-3, dtype=jnp.float32)
        np.testing.assert_array_equal(
            _jvp_cells(_y_cells, profile), np.full(29, 0.001, dtype=np.float32),
        )
        bad = profile.at[-1].set(1.001e-3)
>       with pytest.raises(Exception, match=r"dy_profile\[-1\].*must equal boundary dx"):
E       Failed: DID NOT RAISE Exception

tests/contracts/test_graded_boundary_cell_rule.py:79: Failed
=========================== short test summary info ============================
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t3_first_x_boundary_cell_refused[jvp]
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t4_last_x_boundary_cell_refused[jvp]
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t6_traced_y_boundary_cells_checked
3 failed, 6 passed in 1.13s
```

</details>

Mutation M2 command (one call):

```sh
~/bin/mac-serial ~/Documents/rfx/.venv/bin/python -m pytest -q tests/contracts/test_graded_boundary_cell_rule.py
```

<details>
<summary>M2 complete output</summary>

```text
mac-serial: waiting for a free slot (holders: 48643 2026-11-08 06:44:06 ?@rfx-adv--f-distance /Users/byungkwankim/rfx-adv/.ven;)
FF..FF.F.                                                                [100%]
=================================== FAILURES ===================================
________________ test_t1_concrete_float32_boundary_cells_build _________________

    def test_t1_concrete_float32_boundary_cells_build():
>       grid = _grid(dx_profile=jnp.full(20, 1e-3, dtype=jnp.float32))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/contracts/test_graded_boundary_cell_rule.py:32:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/contracts/test_graded_boundary_cell_rule.py:15: in _grid
    )._build_nonuniform_grid()
      ^^^^^^^^^^^^^^^^^^^^^^^^
rfx/api/_compile.py:490: in _build_nonuniform_grid
    grid = build_nonuniform_grid(
rfx/runners/nonuniform.py:178: in build_nonuniform_grid
    return make_nonuniform_grid(
rfx/model/graded_grid.py:907: in make_nonuniform_grid
    _assert_boundary_cells(dx_profile, dx, "x")
rfx/model/graded_grid.py:798: in _assert_boundary_cells
    traced_check("graded mesh", "boundary cells", _check, profile[0], profile[-1])
rfx/_traced_check.py:42: in traced_check
    return check(*operands)
           ^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

lo = Array(0.001, dtype=float32), hi = Array(0.001, dtype=float32)

    def _check(lo, hi):
        ends = ((-1, hi),) if match_each_other else ((0, lo), (-1, hi))
        for index, cell in ends:
            if not abs(float(cell) - boundary) <= tolerance:
                if match_each_other:
                    raise ValueError(
                        f"dy_profile boundary cells must match each other "
                        f"(got lo={boundary}, hi={float(cell)}); "
                        f"tolerance={tolerance} m."
                    )
                detail = " (CPML cells use the boundary spacing)" if index == 0 else ""
>               raise ValueError(
                    f"d{axis}_profile[{index}]={float(cell)} must equal boundary "
                    f"dx={boundary}{detail}; tolerance={tolerance} m."
                )
E               ValueError: dx_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.

rfx/model/graded_grid.py:792: ValueError
___________ test_t2_jvp_float32_cells_are_bit_identical_to_concrete ____________

    def test_t2_jvp_float32_cells_are_bit_identical_to_concrete():
        profile = jnp.full(20, 1e-3, dtype=jnp.float32)
>       concrete = np.asarray(_x_cells(profile))
                              ^^^^^^^^^^^^^^^^^

tests/contracts/test_graded_boundary_cell_rule.py:42:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/contracts/test_graded_boundary_cell_rule.py:19: in _x_cells
    return _grid(dx_profile=profile).dx_arr
           ^^^^^^^^^^^^^^^^^^^^^^^^^
tests/contracts/test_graded_boundary_cell_rule.py:15: in _grid
    )._build_nonuniform_grid()
      ^^^^^^^^^^^^^^^^^^^^^^^^
rfx/api/_compile.py:490: in _build_nonuniform_grid
    grid = build_nonuniform_grid(
rfx/runners/nonuniform.py:178: in build_nonuniform_grid
    return make_nonuniform_grid(
rfx/model/graded_grid.py:907: in make_nonuniform_grid
    _assert_boundary_cells(dx_profile, dx, "x")
rfx/model/graded_grid.py:798: in _assert_boundary_cells
    traced_check("graded mesh", "boundary cells", _check, profile[0], profile[-1])
rfx/_traced_check.py:42: in traced_check
    return check(*operands)
           ^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

lo = Array(0.001, dtype=float32), hi = Array(0.001, dtype=float32)

    def _check(lo, hi):
        ends = ((-1, hi),) if match_each_other else ((0, lo), (-1, hi))
        for index, cell in ends:
            if not abs(float(cell) - boundary) <= tolerance:
                if match_each_other:
                    raise ValueError(
                        f"dy_profile boundary cells must match each other "
                        f"(got lo={boundary}, hi={float(cell)}); "
                        f"tolerance={tolerance} m."
                    )
                detail = " (CPML cells use the boundary spacing)" if index == 0 else ""
>               raise ValueError(
                    f"d{axis}_profile[{index}]={float(cell)} must equal boundary "
                    f"dx={boundary}{detail}; tolerance={tolerance} m."
                )
E               ValueError: dx_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.

rfx/model/graded_grid.py:792: ValueError
________________ test_t4_last_x_boundary_cell_refused[concrete] ________________

traced = False

    @pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jvp"])
    def test_t4_last_x_boundary_cell_refused(traced):
        profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[-1].set(1.001e-3)
>       with pytest.raises(Exception if traced else ValueError,
                           match=r"dx_profile\[-1\].*must equal boundary dx"):
E                          AssertionError: Regex pattern did not match.
E                            Expected regex: 'dx_profile\\[-1\\].*must equal boundary dx'
E                            Actual message: 'dx_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.'

tests/contracts/test_graded_boundary_cell_rule.py:61: AssertionError
__________________ test_t4_last_x_boundary_cell_refused[jvp] ___________________

traced = True

    @pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jvp"])
    def test_t4_last_x_boundary_cell_refused(traced):
        profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[-1].set(1.001e-3)
>       with pytest.raises(Exception if traced else ValueError,
                           match=r"dx_profile\[-1\].*must equal boundary dx"):
E                          AssertionError: Regex pattern did not match.
E                            Expected regex: 'dx_profile\\[-1\\].*must equal boundary dx'
E                            Actual message: 'dx_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.\n--------------------\nFor simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.'

tests/contracts/test_graded_boundary_cell_rule.py:61: AssertionError
------------------------------ Captured log call -------------------------------
ERROR    jax._src.debugging:debugging.py:103 jax.debug.callback failed
Traceback (most recent call last):
  File "/Users/byungkwankim/Documents/rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py", line 101, in debug_callback_impl
    callback(*args)
  File "/Users/byungkwankim/Documents/rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py", line 505, in _flat_callback
    callback(*args_, **kwargs_)
  File "/Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/model/graded_grid.py", line 792, in _check
    raise ValueError(
ValueError: dx_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.
___________________ test_t6_traced_y_boundary_cells_checked ____________________

    def test_t6_traced_y_boundary_cells_checked():
        profile = jnp.full(20, 1e-3, dtype=jnp.float32)
        np.testing.assert_array_equal(
>           _jvp_cells(_y_cells, profile), np.full(29, 0.001, dtype=np.float32),
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        )

tests/contracts/test_graded_boundary_cell_rule.py:76:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/contracts/test_graded_boundary_cell_rule.py:27: in _jvp_cells
    primal, _ = jax.jvp(build, (profile,), (jnp.zeros_like(profile),))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/contracts/test_graded_boundary_cell_rule.py:23: in _y_cells
    return _grid(dy_profile=profile).dy_arr
           ^^^^^^^^^^^^^^^^^^^^^^^^^
tests/contracts/test_graded_boundary_cell_rule.py:15: in _grid
    )._build_nonuniform_grid()
      ^^^^^^^^^^^^^^^^^^^^^^^^
rfx/api/_compile.py:490: in _build_nonuniform_grid
    grid = build_nonuniform_grid(
rfx/runners/nonuniform.py:178: in build_nonuniform_grid
    return make_nonuniform_grid(
rfx/model/graded_grid.py:928: in make_nonuniform_grid
    _assert_boundary_cells(dy_prof_phys, dy_boundary, "y")
rfx/model/graded_grid.py:798: in _assert_boundary_cells
    traced_check("graded mesh", "boundary cells", _check, profile[0], profile[-1])
rfx/_traced_check.py:47: in traced_check
    return jax.debug.callback(check, *operands, ordered=ordered)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:517: in debug_callback
    _debug_callback(callback, *args, **kwargs)
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:509: in _debug_callback
    debug_callback_p.bind(
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:137: in debug_callback_jvp_rule
    return debug_callback_p.bind(*primals, **params), []
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:101: in debug_callback_impl
    callback(*args)
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:505: in _flat_callback
    callback(*args_, **kwargs_)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

lo = Array(0.001, dtype=float32), hi = Array(0.001, dtype=float32)

    def _check(lo, hi):
        ends = ((-1, hi),) if match_each_other else ((0, lo), (-1, hi))
        for index, cell in ends:
            if not abs(float(cell) - boundary) <= tolerance:
                if match_each_other:
                    raise ValueError(
                        f"dy_profile boundary cells must match each other "
                        f"(got lo={boundary}, hi={float(cell)}); "
                        f"tolerance={tolerance} m."
                    )
                detail = " (CPML cells use the boundary spacing)" if index == 0 else ""
>               raise ValueError(
                    f"d{axis}_profile[{index}]={float(cell)} must equal boundary "
                    f"dx={boundary}{detail}; tolerance={tolerance} m."
                )
E               ValueError: dy_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.
E               --------------------
E               For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

rfx/model/graded_grid.py:792: ValueError
------------------------------ Captured log call -------------------------------
ERROR    jax._src.debugging:debugging.py:103 jax.debug.callback failed
Traceback (most recent call last):
  File "/Users/byungkwankim/Documents/rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py", line 101, in debug_callback_impl
    callback(*args)
  File "/Users/byungkwankim/Documents/rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py", line 505, in _flat_callback
    callback(*args_, **kwargs_)
  File "/Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/model/graded_grid.py", line 792, in _check
    raise ValueError(
ValueError: dy_profile[0]=0.0010000000474974513 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=1e-12 m.
=========================== short test summary info ============================
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t1_concrete_float32_boundary_cells_build
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t2_jvp_float32_cells_are_bit_identical_to_concrete
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t4_last_x_boundary_cell_refused[concrete]
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t4_last_x_boundary_cell_refused[jvp]
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t6_traced_y_boundary_cells_checked
5 failed, 4 passed in 0.96s
```

</details>

Mutation M3 command (one call):

```sh
~/bin/mac-serial ~/Documents/rfx/.venv/bin/python -m pytest -q tests/contracts/test_graded_boundary_cell_rule.py
```

<details>
<summary>M3 complete output</summary>

```text
mac-serial: waiting for a free slot (holders: )
........F                                                                [100%]
=================================== FAILURES ===================================
_____________ test_t7_traced_nominal_x_boundary_can_differ_from_dx _____________

    def test_t7_traced_nominal_x_boundary_can_differ_from_dx():
        profile = jnp.full(20, 1e-3, dtype=jnp.float32).at[0].set(2e-3)
        nominal = np.asarray(profile, dtype=np.float64)

        def build(cells):
            # The nominal sum must match the declared extent, including f32 rounding.
            grid = Simulation(
                freq_max=20e9, domain=(float(nominal.sum()), 20e-3, 10e-3), dx=1e-3,
                boundary="cpml", cpml_layers=4, dx_profile=cells,
                nominal_profiles={"x": nominal},
            )._build_nonuniform_grid()
            assert grid.dx == 0.001
            assert grid.nominal_cells["x"].dtype == np.dtype("float64")
            assert grid.traced_cells is None
            return grid.dx_arr[grid.pad_x_lo]

>       first = np.asarray(_jvp_cells(build, profile))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/contracts/test_graded_boundary_cell_rule.py:99:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/contracts/test_graded_boundary_cell_rule.py:27: in _jvp_cells
    primal, _ = jax.jvp(build, (profile,), (jnp.zeros_like(profile),))
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/contracts/test_graded_boundary_cell_rule.py:93: in build
    )._build_nonuniform_grid()
      ^^^^^^^^^^^^^^^^^^^^^^^^
rfx/api/_compile.py:490: in _build_nonuniform_grid
    grid = build_nonuniform_grid(
rfx/runners/nonuniform.py:178: in build_nonuniform_grid
    return make_nonuniform_grid(
rfx/model/graded_grid.py:905: in make_nonuniform_grid
    _assert_boundary_cells(dx_prof_phys, dx, "x")
rfx/model/graded_grid.py:798: in _assert_boundary_cells
    traced_check("graded mesh", "boundary cells", _check, profile[0], profile[-1])
rfx/_traced_check.py:47: in traced_check
    return jax.debug.callback(check, *operands, ordered=ordered)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:517: in debug_callback
    _debug_callback(callback, *args, **kwargs)
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:509: in _debug_callback
    debug_callback_p.bind(
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:137: in debug_callback_jvp_rule
    return debug_callback_p.bind(*primals, **params), []
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:101: in debug_callback_impl
    callback(*args)
../../rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py:505: in _flat_callback
    callback(*args_, **kwargs_)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

lo = Array(0.002, dtype=float32), hi = Array(0.001, dtype=float32)

    def _check(lo, hi):
        ends = ((-1, hi),) if match_each_other else ((0, lo), (-1, hi))
        for index, cell in ends:
            if not abs(float(cell) - boundary) <= tolerance:
                if match_each_other:
                    raise ValueError(
                        f"dy_profile boundary cells must match each other "
                        f"(got lo={boundary}, hi={float(cell)}); "
                        f"tolerance={tolerance} m."
                    )
                detail = " (CPML cells use the boundary spacing)" if index == 0 else ""
>               raise ValueError(
                    f"d{axis}_profile[{index}]={float(cell)} must equal boundary "
                    f"dx={boundary}{detail}; tolerance={tolerance} m."
                )
E               ValueError: dx_profile[0]=0.0020000000949949026 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=4.76837158203125e-10 m.
E               --------------------
E               For simplicity, JAX has removed its internal frames from the traceback of the following exception. Set JAX_TRACEBACK_FILTERING=off to include these.

rfx/model/graded_grid.py:792: ValueError
------------------------------ Captured log call -------------------------------
ERROR    jax._src.debugging:debugging.py:103 jax.debug.callback failed
Traceback (most recent call last):
  File "/Users/byungkwankim/Documents/rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py", line 101, in debug_callback_impl
    callback(*args)
  File "/Users/byungkwankim/Documents/rfx/.venv/lib/python3.11/site-packages/jax/_src/debugging.py", line 505, in _flat_callback
    callback(*args_, **kwargs_)
  File "/Users/byungkwankim/Documents/rfx-worktrees/sw-structure--f32-boundary-cell/rfx/model/graded_grid.py", line 792, in _check
    raise ValueError(
ValueError: dx_profile[0]=0.0020000000949949026 must equal boundary dx=0.001 (CPML cells use the boundary spacing); tolerance=4.76837158203125e-10 m.
=========================== short test summary info ============================
FAILED tests/contracts/test_graded_boundary_cell_rule.py::test_t7_traced_nominal_x_boundary_can_differ_from_dx
1 failed, 8 passed in 1.34s
```

</details>

Ruff command:

```sh
~/bin/mac-serial ~/Documents/rfx/.venv/bin/python -m ruff check rfx/model/graded_grid.py tests/contracts/test_graded_boundary_cell_rule.py --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402
```

<details>
<summary>ruff complete output</summary>

```text
mac-serial: waiting for a free slot (holders: 54062 2026-11-08 06:44:29 ?@rfx-adv--f-distance /Users/byungkwankim/rfx-adv/.ven;)
All checks passed!
```

</details>

Conclusion: 리더가 채움.
