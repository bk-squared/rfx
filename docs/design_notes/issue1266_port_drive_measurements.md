# #1266 / #1373 step c: port drive measurements

Base: `f43641a1`. Branch: `nu/1266-port-drive-unit`.
Resource choice: shared Mac CPU, one Python/pytest process started by this task at a time; no VESSL submission.
Python environment: `/Users/byungkwankim/Documents/rfx/.venv`, Python 3.11.2,
NumPy 2.4.6, JAX 0.10.2; Mac13,1 arm64, 64 GiB RAM, Darwin 25.5.0.
Scratch, logs and uncommitted raw arrays: `_scratch_1266/`.
No existing lock/unit/crossval test is edited or re-pinned. No changelog fragment is edited.
Conclusions: 리더가 채움.

## D1: drive-site census

All listed admitted paths reach `rfx/sources/port_drive.py:10 (port_drive_waveform)`.
The function reads `cell_component_e_coeffs(materials, cell, component, grid.dt)[1]`
and `port_d_parallel`, then builds `(Cb/d_parallel)*w/n_live`.
The supplied material arrays carry the edge-owned load and applicable override.

| Path | Site | Disposition |
|---|---|---|
| Public low-level lumped builder | `rfx/simulation.py:458` | delegates to shared helper |
| Public low-level wire builder | `rfx/simulation.py:474` | delegates for each live edge, with `n_live` |
| Uniform run lumped / wire | `rfx/runners/uniform.py:530`, `:511` | existing wrappers reach shared helper |
| Uniform forward lumped / wire, including override | `rfx/api/_execute.py:2026`, `:1880` | existing wrappers reach shared helper |
| Graded run / forward wire, including override | `rfx/runners/nonuniform.py:1276` | removed current-source route; shared helper consumes `materials_drive` |
| Graded run / forward lumped, including override | `rfx/runners/nonuniform.py:1343` | removed current-source route; shared helper consumes `materials_drive` |
| Subgrid fine region wire / lumped | `rfx/runners/subgridded.py:460`, `:475` | removed inline drive expression; shared helper consumes `mats_f` |
| Distributed uniform run lumped | `rfx/runners/distributed_v2.py:961` | existing wrapper reaches shared helper |
| Differentiable material-fit lumped | `rfx/differentiable_material_fit.py:556` | existing wrapper reaches shared helper |
| ADI run / forward | `rfx/runners/_admission.py:451` | driven lumped and wire refused; unchanged |
| Distributed uniform run wire | `rfx/runners/_admission.py:453` | refused; unchanged |
| Distributed graded forward ports | `rfx/runners/_admission.py:451`, `:453` | refused; unchanged |
| Distributed graded override drive | `rfx/api/_execute.py:3101` | plain sources only; `material_drive` retains current-source volumes |
| Disjoint research subgrid | `rfx/runners/disjoint.py:42` | impedance ports refused; unchanged |

## D2: other-port read census

* MSL: both lanes use `rfx/sources/msl_port.py:1019` (`make_msl_port_sources`).
  The unshaped feed uses `Cb*w/(d_normal*n_z)` at line 1068; the Laplace feed
  uses `Cb*ez_profile*w` at line 1099. Graded setup calls it at
  `rfx/runners/nonuniform.py:698`. Its docstring defines the shaped force and
  unit centre-line mode integral; this is not a prescribed terminal voltage.
  Graded eigenmode J+M launch is refused (`nonuniform.py:668`). These paths
  were not changed. A shared builder establishes the formula, not full-field
  equality for arbitrarily different meshes.
* Coax: `Simulation.run` and `forward` reject `add_coaxial_port` at
  `rfx/api/_execute.py:4884` and `:4290`. They direct callers to the separate
  coaxial reflection/two-port solvers. There is no admitted pair of these
  uniform/graded run/forward paths to compare. No coax implementation changed.
* Waveguide: both lanes use `init_waveguide_port` and
  `apply_waveguide_port_h/e` from `rfx/sources/waveguide_port.py`.
  Graded config is built at `rfx/runners/nonuniform.py:527`; injection calls
  are `rfx/nonuniform.py:2879` and `:2940`. Graded injection reads the local
  primal H-plane and dual E-plane metrics (`rfx/nonuniform.py:686`);
  uniform injection passes `grid.dx` (`rfx/simulation.py:2197`, `:2320`).
  The metrics coincide on equal cells. No waveguide implementation changed.

## D3: absolute cross-lane contract

`tests/contracts/test_port_drive_unit_1266.py` uses a declared domain of
(17.3, 14.7, 16.2) mm, port (6.3, 5.2, 4.4) mm, probe (7.1, 5.3, 4.2) mm,
wire extent 2.3 mm, an x-low PEC wall and five CPML walls. All positions
are away from the domain midplanes; opposite-wall distances differ. The
flux plane is x=10.1 mm. Waveform: GaussianPulse(f0=8 GHz, bandwidth=0.9),
50 ohms; bins 4/6/8 GHz; record approximately 0.8 ns. Arrays and steps are
float32. The flux integral is evaluated from complex64 DFT accumulators
in NumPy float64 (`exact_f64=True`). The reported power in W is
`2*flux_spectrum/T**2`, using the rectangular record length T. It is the
power of that record's Fourier component, not pulse-integrated energy.

For equal cells, all three graded profiles explicitly have the same cell
counts as the uniform grid. This matters for the non-integer domain:
`make_nonuniform_grid`'s omitted x/y profiles use round, whereas Grid uses
ceil. The test does not alter either meshing rule.

| Port | Uniform probe peak V/m | Max trace discrepancy / float32 ULP of peak | Max power discrepancy / peak |
|---|---:|---:|---:|
| Lumped | 7.9013791084 | 6 | 2.77933148e-7 |
| Wire | 1.1577416658 | 8 | 1.32501145e-7 |

Equal-cell bars: 9 ULP and 1e-4, respectively. Both tests passed on this Mac.
An earlier, more distant probe at (9.1, 7.3, 6.2) mm measured 154 ULP
(lumped) and 22.5 ULP (wire), outside the trace bar. These observations
remain in `_scratch_1266/equal.log` and `trend.log`. The passing contract
covers the declared nearby probe; it does not establish that bar at every
probe location.

The final genuinely graded profiles replace groups of four fine z cells
in the fixed 10--14 mm band with widths (1.25h, 1.5h, 1.25h), keeping the
band endpoints, total realized length, and CPML interface fixed. The maximum
adjacent ratio is 1.25, inside the repository's 1.4 grading cap. The uniform
comparator uses h everywhere. CPML physical thickness stays 3 mm.
Two refinements: h=1 mm to 0.5 mm to 0.25 mm.

| Port | h mm | Trace error / peak | Power error / peak |
|---|---:|---:|---:|
| Lumped | 1 | 1.08706263e-4 | 1.10713370e-4 |
| Lumped | 0.5 | 1.06377797e-2 | 1.40359449e-5 |
| Lumped | 0.25 | 2.45062140e-4 | 8.77608376e-6 |
| Wire | 1 | 8.72131277e-5 | 9.29103800e-5 |
| Wire | 0.5 | 5.53286445e-5 | 1.23146579e-5 |
| Wire | 0.25 | 1.91641975e-5 | 8.02874818e-6 |

The lumped trace trend is nonmonotone. No convergence order is inferred.
The new test's tolerances are twice the largest observed discrepancy over
these three meshes, rounded upward: lumped trace 0.022, wire trace 0.00018,
power 0.00023. These are measured envelopes for this fixture.

The earlier graded profile used pairs (1.25h, 1.75h) above 10 mm and
triggered a maximum-adjacent-ratio warning (1.75 > 1.4). Its preliminary
trend and contract outputs remain in `trend_near.log` and `contract.log`.
The table above is the replacement profile, from `contract_final_profile.log`.

## J1: same-machine before/after

The baseline is the actual `f43641a1` package extracted with `git archive`
inside `_scratch_1266/baseline`, selected via PYTHONPATH. Both versions use
one 1 mm equal-cell z profile (graded execution lane), domain/port/extent/
pulse as above, but the probe is (9.1, 7.3, 6.2) mm and x/y profiles are
omitted. Both versions therefore have the same realized grid. Three CPML
layers per absorbing wall. Run records: 420 steps, T=8.00761445203e-10 s.
Lumped S11 is from forward with explicit bins and 420 steps (graded run
refuses lumped S extraction); wire S11 is from run with explicit bins.
The raw script and arrays are `_scratch_1266/measure.py`, `before_*.npz`,
`after_*.npz`. Wall times before: 11.95/2.54 s; after: 7.21/0.65 s for
lumped/wire, including builds and compilation in each process.

| Quantity | Port | Before | After | After / before |
|---|---|---:|---:|---:|
| Probe peak V/m | Lumped | 44609.26171875 | 0.0446090586483 | 9.99995448e-7 |
| Probe peak V/m | Wire | 101686.03125 | 0.101686052978 | 1.00000021e-6 |
| Power W, 4 GHz | Lumped | -0.774534763 | -7.74534166e-13 | 9.99999230e-13 |
| Power W, 6 GHz | Lumped | 0.315543755 | 3.15543948e-13 | 1.00000061e-12 |
| Power W, 8 GHz | Lumped | 8.875023977 | 8.87502855e-12 | 1.00000052e-12 |
| Power W, 4 GHz | Wire | -0.342582908 | -3.42583335e-13 | 1.00000125e-12 |
| Power W, 6 GHz | Wire | 0.352808639 | 3.52808808e-13 | 1.00000048e-12 |
| Power W, 8 GHz | Wire | 5.749973646 | 5.74997701e-12 | 1.00000059e-12 |

These J1 power numbers convert the recorded default float32 flux sums to W;
the new D3 contract uses float64 summation of the same accumulator type.

| Port | GHz | S11 before | S11 after |
|---|---:|---|---|
| Lumped | 4 | 0.999014497 - 0.044723779j | 0.999014437 - 0.044723742j |
| Lumped | 6 | 0.997736514 - 0.067508630j | 0.997736573 - 0.067508593j |
| Lumped | 8 | 0.995820642 - 0.090836264j | 0.995820761 - 0.090836123j |
| Wire | 4 | 0.998723269 - 0.050946917j | 0.998723269 - 0.050946943j |
| Wire | 6 | 0.996902108 - 0.078225285j | 0.996902049 - 0.078225315j |
| Wire | 8 | 0.993612230 - 0.108007133j | 0.993612230 - 0.108007073j |

Max |delta S11|: lumped 1.85068583e-7; wire 6.66400187e-8.

The requested prediction `n*dV/d_parallel` is 1e-6 for lumped (n=1)
and 3e-6 for this wire (n=3). Measured field ratio / that prediction:
0.999995448 and 0.333333405, respectively. The old source is
`(Cb/dV)*w/n` and the new source is `(Cb/d_parallel)*w/n`, so their algebraic
ratio at fixed Cb is `dV/d_parallel=1e-6` for both. Their power ratio is the
square, 1e-12. The shared helper also changes the old graded host-float Cb
arithmetic to the E-update helper's array arithmetic. The measurements
above include that rounding difference. Conclusions: 리더가 채움.

D1 additional eager sites: `rfx/sources/sources.py:415` (`apply_lumped_port`)
and `:644` (`apply_wire_port`) now call the same helper with one time sample.
The eager S-matrix loops call these applicators at `rfx/probes/probes.py:1744`
and `:2102`. Their former multiply/divide order is replaced by the uniform
table builder's order. This includes the public low-level applicators, not
only Simulation paths. The helper's scalar mode preserves the caller's
sample time; table mode retains float32 `arange(n_steps)*dt`.
The wire rasterizer was checked directly: padded endpoints (6,8,7) to
(6,8,10), three live edges. `current_source_volume/port_d_parallel` on the
stored metric arrays is 1.0000000949949049e-6; the table uses the nominal
1 mm metric for its prediction.

D2 equal-cell field measurements (`_scratch_1266/other_ports.py`):

| Kind | Uniform probe peak V/m | Graded probe peak V/m | Max absolute difference V/m | Difference / peak |
|---|---:|---:|---:|---:|
| MSL Laplace feed | 10.520322 | 10.520323 | 3.8146973e-6 | 3.6260272e-7 |
| Waveguide TE10 | 23.135862 | 23.135860 | 5.7220460e-6 | 2.4732364e-7 |

MSL uses `_model()` in `tests/unit/ports/test_msl_realized_port_contract.py`,
a 24 mm cube, 1 mm cells, ground at 6 mm, trace at 10 mm, substrate eps=3,
feed x=8 mm, probe (10.3,12.2,8.2) mm; 300 steps. The graded version adds
24 equal z cells to the same declaration. Waveguide: 22.5 x 60 x 10.5 mm,
1.5 mm cells, +y TE10 launch at y=15 mm, PEC x/z walls, eight y CPML layers,
f0=10 GHz, bandwidth=0.5, probe (9,24,4.5) mm; 260 steps. The graded version
supplies equal profiles on all axes. Times including compilation were
3.07 s and 3.22 s. The first attempted waveguide aperture used 10.16 mm
against a 10.5 mm snapped span and was rejected; the measured declaration
above uses the explicit 10.5 mm aperture on both lanes. These measurements
cover these equal-cell declarations only.

The refinement declarations hold the physical coordinates and wire extent
fixed. A lumped port still spans one mesh edge at each resolution; its
realized gap therefore shrinks with h. No claim of a fixed-width lumped gap
across the refinement ladder is made.

## J2 and J3

The final full contract passed: **8 passed**, 52.71 s, exit 0
(`contract_final.log`, with the final profile and final tolerances).
Mutation uses process-local monkeypatches in `_scratch_1266/mutation.py`;
no mutation remains in the worktree.

| Mutation | Implementation | Result |
|---|---|---|
| (a), no shared graded call | Graded builder alias replaced by old `make_current_source(...)[4]/n_live` | exit **1**, both equal-cell tests failed, 4.74 s |
| (b), old unit with shared call retained | Invoke shared helper, multiply its result by `d_parallel/dV` | exit **1**, all six genuine-grading tests failed, 52.80 s |

Mutation (a) trace errors: 1.6570377e13 / 9.711836e12 ULP for lumped/wire.
Mutation (b) at h=0.25 mm: lumped trace error 1.6000138e7, power error
2.559976809e14 (peak-relative). The helper call is executed in (b).
These final reruns are in `mutation_a_final.log` and `mutation_b_final.log`.

Initial required-suite results:

| File | Result | pytest seconds |
|---|---|---:|
| `tests/contracts/test_path_disposition.py` | 9 passed | 6.65 |
| `tests/unit/runners/test_path_disposition_cells.py` | 1311 passed, 3 xfailed, 4 strict XPASS failures | 340.14 |
| `tests/unit/runners/test_realized_model_cells.py` | 264 passed, 17 xfailed | 343.71 |
| `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py` | 24 passed | 29.47 |
| `tests/unit/nonuniform/test_nu_drive_sees_override_1267.py` | 11 passed, 9 failed | 81.82 |
| `tests/unit/nonuniform/test_wire_port_drive_cb_1256.py` | 4 passed, 11 failed | 25.34 |

The four path-disposition XPASS cells are lumped/wire × run_nonuniform /
fwd_nonuniform. `tests/contracts/path_disposition.py` still marks those
comparisons strict xfail #1266. Only the explicitly requested P0 table,
`tests/contracts/realized_model.py`, was changed. Refusal and unreachable
cells were left untouched.

Eight failures in each of #1256 and #1267 assert that their
`make_current_source` recorder saw a wire, lumped, wire+C, or lumped+C drive
(two checks each). It now sees zero calls for those ports. The unchanged
MSL and plain-source recorder checks continue to pass. The old mutation
hooks also intercept `make_current_source`, so they no longer mutate a
lumped/wire port: this accounts for #1267's remaining mutation failure and
#1256's load/capacitor mutation failures. The new J2 mutations intercept
the new graded builder alias and do fail the new field/power contract.

### Moved checks, measured against f43641a1

The two unchanged #1256/#1267 files were also run against the extracted
baseline package on this Mac: **35 passed**, exit 0, 144.12 s. The package
path is printed in `_scratch_1266/baseline_tests.log`; the after logs name
all 20 failing test IDs. Values below are printed precision unless more
digits were present in the assertion output. No pin was updated.

| Existing check | Before | After | Ratio / prediction |
|---|---|---|---|
| #1256 `graded*dx²/uniform` at 4 GHz (pin 1±0.001) | 0.9999997 - 0.0000000j | 2.4999997233e-7 - 1.8932680e-14j | about 2.5e-7; dV/d_parallel=2.5e-7; requested n*dV/d_parallel=7.5e-7 (n=3) |
| #1256 load-mutation lane ratio | 2.911215 - 0.000000j | 0.00000025 - 0.00000000j | about 8.59e-8; old mutation no longer reaches port builder, not a unit-only comparison |
| #1256 load-mutation dt ratio | 0.850197 - 0.002139j | 0.9998583174 - 0.0025155263j | about 1.176; old predicted 0.850318 no longer applies to the unmutated drive |
| #1256 capacitor-mutation max delta abs(S11) | 4.85e-3 | 1.7881393e-7 | about 3.69e-5; old >1e-3 mutation-effect pin no longer met |
| #1267 override-mutation abs(delta S11), 4/6/8 GHz | [0.00192360, 0.00482701, 0.00693468] | [0,0,0] | 0; mutation hook no longer reaches port builder |
| #1267 override-mutation probe discrepancy | 0.234 | 0.000 | 0 at printed precision; same hook issue |
| #1267 override-mutation raised-epsilon field ratio | 1.036126 - 0.000000j | 1.000000 - 0.000000j | about 0.965133; old kappa prediction 1.03612604 was for the removed mutation |
| #1267 override-mutation upper AD / FD | -0.343701 / -0.341822 | -0.341801 / -0.341797 | AD after/before about 0.994472; not a unit-scale pin |
| #1267 override-mutation whole AD / FD | 1.671176 / 0.983906 | 0.984178 / 0.984033 | AD after/before about 0.588914; not a unit-scale pin |
| #1267 override-mutation gradient excess | 0.687270 | 0.000145 | about 2.11e-4; old predicted excess 0.68699828 was for the removed mutation |

The recorder failures do not produce a new Cb value: `handed == []`.
Their before counts were three edges for wire/wire+C, one for lumped/
lumped+C; all four after counts are zero. This is recorder visibility,
not zero physical drive. Each case appears once in the equality check and
once in the mutation check in each file (16 failed tests):

| Recorder | Case | Before max abs(Cb_drive/Cb_step - 1), unmutated / mutated | After | Ratio / unit prediction |
|---|---|---|---|---|
| #1256 | wire | 0 / 1.911 | no captured call | not defined |
| #1256 | lumped | 0 / 0.6371 | no captured call | not defined |
| #1256 | wire+C | 0 / 22.96 | no captured call | not defined |
| #1256 | lumped+C | 0 / 40.82 | no captured call | not defined |
| #1267 | wire | 0 / 0.6931 | no captured call | not defined |
| #1267 | lumped | 0 / 1.233 | no captured call | not defined |
| #1267 | wire+C | 0 / 0.02893 | no captured call | not defined |
| #1267 | lumped+C | 0 / 0.02947 | no captured call | not defined |

The old hooks, expectations, and source-unit compensation in these tests
remain byte-for-byte unchanged. Conclusions: 리더가 채움.

#1257 adds two numerical failures (11 passed, 2 failed, 16.99 s):
`test_graded_lane_on_uniform_cells_matches_the_uniform_lane[lumped-debye]`
and `[lumped-lorentz]`. `_TO_UNIFORM_UNITS['lumped']` is still `DX*DX=1e-6`.
The unchanged relative-trace-error pin is <=1e-4. Direct before/after
replays of these fixtures (`dispersive_before.log`, `dispersive_after.log`)
measured:

| Material | Graded peak before V/m | Graded peak after V/m | After / before | Predicted n*dV/d_parallel (n=1) |
|---|---:|---:|---:|---:|
| Debye | 498246.6875 | 0.498246759176 | 1.000000144e-6 | 1e-6 |
| Lorentz | 498360.46875 | 0.498360306025 | 9.999996735e-7 | 1e-6 |

| Old compensated error pin | Before | After | After / before | Predicted after error for the retained dx² compensation |
|---|---:|---:|---:|---:|
| Debye | 4.073273304e-6 | 0.999999000001 | 245502.554 | 1 - 1e-6 |
| Lorentz | 3.110843772e-6 | 0.9999989999995 | 321455.873 | 1 - 1e-6 |

The passing 50/5000-ohm late-field ratio checks also moved numerically:
Debye 0.000360919277902 -> 0.000361143239635 (ratio 1.000620531);
Lorentz 0.000374190078425 -> 0.000374363728414 (ratio 1.000464069).
An amplitude-only rescaling predicts ratio 1; these figures include the
float32 arithmetic changes. Their unchanged comparison against the uniform
lane stayed inside the existing tolerance. No #1257 test was edited.

During the extended census, the first traced-mesh run exposed a new
failure: `port_d_parallel` delegated to `_axis_cell_sizes`, whose all-axis
NumPy conversion refused a traced mesh. The old graded current-source
builder had not used that helper. `port_d_parallel` now reads only the
component's parallel-axis entry and returns a tracer unchanged; concrete
values still read the same stored-array entry. The transverse-metric and
`port_sigma` host-only guard remains. The failed intermediate run was
17 passed / 10 failed; it was followed by a full traced-mesh rerun after
this change. No traced-mesh test was changed.
The traced-mesh rerun passed **27/27**, 64.08 s; the concrete/guard port
metric file passed **77/77**, 7.91 s. The additional forward-JIT contract
passed **34/34**, 54.78 s. The final `port_d_parallel` change therefore
retains the mesh derivatives exercised by that existing file, including
forward/reverse mode and finite-difference comparisons. The earlier traced
failure is not a remaining failed pin.

## Final verification and J4 census

The [file/test census](issue1266_port_test_census.md) lists all 69 static
candidate files, their archived runtime sums, current Mac wall times,
remaining expensive/unexecuted files, and 230 instrumented graded-port
model declarations in executed tests. The 57 executed files include the
required judges and additional eager/metric tests beyond that candidate
list. Latest per-file results total **2891 passed, 26 failed, 28 xfailed,
2 skipped, 114 deselected**. This excludes the new contract's additional
**8 passed** and avoids counting repeated runs twice. The final port-cell
recheck is another execution of **18 passed**, 263 deselected, 18.02 s;
those cells are already included in the full P0 total.

Remaining failures are confined to four unchanged test files: 4 strict
XPASS cells, 19 checks tied to the former make_current_source port call, and 3 retained
unit-compensation checks. Every failed ID is listed in the census, and
all moved numerical checks and missing recorder values are tabulated above.
No result is asserted for an unexecuted expensive test or its pins.

Additional output-path coverage: the current-moment file's port-driven
plane-route and ringdown-completion cases passed **4/4**, 13.67 s (113 other
cases deselected). The graded RLC-under-override file passed **2/2**, 32.12 s.
Eager wire application passed **5/5**; the known-load lumped file had
**8 passed, 6 existing xfails**. The recorded durations in the separate
census explicitly distinguish historical `.test_durations` values from
measurements on this Mac.

Crossval read census: `rt5880_patch/test_rt5880_patch.py:399` builds a
uniform mesh for its wire port; it declares no graded profile. The
`msl_notch_filter` model uses MSL feeds, and its grading mention at line
1004 is prose. No graded lumped/wire model was found in those crossval
solvers. The conservative `test_reciprocity_adjoint.py` candidate has
separate `graded` and `ports` refusal cases (lines 202--219), not a combined
graded-port solve; its full 2400-step gradient suite was not executed.

Lint: the new helper and new contract pass Ruff. Across touched Python
files, Ruff reports the same three pre-existing F401 imports found in
`f43641a1` (`edge_averaged_materials`, `e_component_coeffs` in simulation;
`EPS_0` in sources). `git diff --check` passes.

Conclusions: 리더가 채움.
