# Auto-mesh regression audit (2026-09-08)

Base: `65b70a77f305a53af19d6c7b2d78a40ac84348d5`, branch
`feat/931-automesh-regressions`. The supplied bisect ran these seven cases
at `7988d4b9` (7 passed) and `65b70a77` (7 failed). This audit accepts that
bisect and distinguishes a broken numerical/validation behavior from tests
that depended on the uniform surrogate the parity repair intentionally removed.

## Mechanisms and verdicts

| Reported case | Verdict | Mechanism and repair |
| --- | --- | --- |
| `test_floquet_auto_mesh_rejects_nonuniform_fallback` | CODE | Registration read the resolved `_dz_profile`, so an inferred profile caused a declaration-time exception before the model reached `run()`. Registration now reads the explicit declaration; preflight still refuses the resolved NU model in both `run()` and `forward()`, including after a preview has populated the resolution cache. |
| `test_accessor_uses_the_nonuniform_grid_on_a_graded_mesh` | TEST | The accessor already selects NU correctly; the test failed earlier while asking that profiled simulation's uniform-only builder for a comparator. The comparator now comes from an independent uniform simulation, retaining the unequal-shapes and graded-profile assertions. |
| `test_registered_geometry_rejected[geometry]` | CODE | The complete-line coax driver planned a mesh from forbidden registered dielectric geometry, then reported the resulting inferred profile instead of the unsupported declaration. Its input validation now reads declared profiles, allowing the existing geometry refusal to execute without auto-meshing an input it does not consume. |
| `test_registered_geometry_rejected[thin_conductor]` | CODE | The same ordering fed a forbidden thin-conductor registration into mesh planning and reported its inferred profile. Both complete-line drivers now validate declaration inputs before planning, with a test that fails if auto-configuration is called. |
| `test_flush_box_lossy_and_magnetic_materials_reach_the_face` | TEST | The helper checked the empty model's default dx before adding geometry, but later sampled by that old lattice's index after auto-refinement. The fixture now explicitly pins its intended coarse dx; the eps, sigma and mu assertions are unchanged. |
| `test_flush_box_reads_material_through_the_face_nonuniform` | TEST | The shared helper called `_build_grid()` before assembly despite its explicit `dz_profile`. It now inspects `_build_realized_grid()` and retains the coarse dx and all three material-cut assertions. |
| `test_thin_conductor_api_integration` | TEST | The auto-meshed board legitimately selects NU, while the test directly requested uniform construction/assembly. It now assembles and indexes on the selected lattice, with an additional explicit-uniform control and the same sheet ownership assertions. |

The material-value case is a separate mechanism, not a missing face fill.
Measured in this worktree with CPU JAX and `PYTHONPATH=$PWD`:

| Fixture / sample | dx (mm) | Sample z (mm) | Face `(eps_r, sigma, mu_r)` |
| --- | ---: | ---: | --- |
| Original fixture, `k=pad+10` after adding geometry | 0.5 | 5.0 | `(1, 0, 1)` |
| Same resolved mesh, physical sample inside the slab | 0.5 | 10.0 | `(4, 0.050000000745, 2)` |
| Coarse fixture pinned to `C0 / freq_max / 20`, original indices | 1.49896229 | 14.9896229 | `(4, 0.050000000745, 2)` |

The slab starts at z = 5.99584916 mm. The first sample is therefore correctly
vacuum. Moving the material boundary or extending the fill would introduce a
numerical defect. Pinning the coarse fixture also preserves the #655 stress
case: refinement would reduce the notch's reflection and weaken that test.

## Principle and scope

`_declared_mesh` is a snapshot of caller inputs and never plans a mesh. The
existing `_dx`, `_domain`, and profile fields remain cached resolved views.
Declaration validation uses the former; execution, capability checks,
inspection, export and optimization use the latter. Reading the declaration
does not mutate it or invalidate the resolved cache.

The sibling `compute_coaxial_line_reflection()` has the same complete-geometry
ownership rule and receives the same repair. `compute_coax_msl_transition()`
consumes caller geometry and keeps its resolved-mesh capability guard. No
numerical rasterization, ownership, CPML extension, tolerance or refusal is
relaxed. The documentation-model run/forward parity tests remain in the suite.

## Blast-radius selection

Run the complete default `tests/` lane. The only marker filtering is the
repository's existing `not gpu and not slow and not slow_physics`; there are
no additional exclusions, including for the four failures separately identified
by the user. This selects consumer behavior across the whole package rather
than files adjacent to `_mesh.py`.

| Changed production surface in the parity merge | Selected coverage |
| --- | --- |
| `api/{__init__,_compile,_execute,_mesh,_preflight,_sparams}.py` | contracts; unit API, grid, preflight, materials, boundaries, ports, S-parameters, nonuniform; crossval and locks |
| `geometry/rasterize_grid.py`, `runners/nonuniform.py` | contracts; unit geometry, materials, boundaries, nonuniform, runners, ports |
| `amr.py` | unit subgrid, including AMR surrogate |
| `interop/_design.py` | studio interop document, schema, codecs, value validation, export |
| `optimize.py`, `topology.py` | unit autodiff and grid auto-mesh consumer tests |
| `probes/sparam_driver.py` | unit S-parameters, ports, auto-mesh driver purity |
| `profiling.py`, `vmap_sweep.py` | unit runners profiling, sweep, eligibility, DFT planes |
| `visualize.py`, `visualize3d.py`, `dashboard/app.py` | unit API visualization, realized grid, 3-D viewers, dashboard |

Command (one pytest controller, at most four workers, 1,200-second wall limit):

```sh
PYTHONPATH=$PWD JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 \
  timeout 1200 python -m pytest tests -q -n 4 -p no:cacheprovider \
  --no-header -rfE --junitxml=.validation-931-regressions/default.xml \
  > .validation-931-regressions/default.log 2>&1
```

The focused selection contains the seven reported cases expanded with preview
order, run/forward, sibling-driver and explicit-uniform controls, plus a direct
declaration snapshot test: **14 passed, 0 failed** in 7.47 seconds.

The full default selection completed in three **non-overlapping** invocations.
The first two were gracefully interrupted before the 1,200-second wall limit
to preserve their JUnit results. Continuations selected only node IDs absent
from those results; no failing test was excluded. The final invocation used
`--dist=worksteal`, still with `-n 4` and `timeout 1200`.

| Invocation | Passed | Failed | Skipped | Existing xfails | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial default selection | 4159 | 1 | 13 | 2 | 1137.15 |
| Remaining 2951 node IDs | 2190 | 0 | 13 | 25 | 1138.37 |
| Final 723 node IDs | 719 | 0 | 0 | 4 | 1020.97 |
| **Combined** | **7068** | **1** | **26** | **31** | |

Collection selected **7125 of 7526 node IDs** (401 deselected by the default
markers). The reports account for every selected node ID exactly once, plus
one module-level collection skip: **zero missing, zero duplicates**. All 14
focused cases also passed in the full selection. Nameless JUnit placeholders
for interrupted in-flight tests are not treated as completed results.

The sole failure is
`tests/oracle/test_waveguide_port_validation_battery.py::test_pec_short_s11_magnitude`,
the separately owned fixture assertion identified in the task. Its declared
x-plane, 0.085654988 m, is 0.345 of a local 0.001 m cell from the nearest
node, 0.086 m; the unchanged assertion permits 0.25. Neither that fixture nor
its gate is changed. The three environment-related failures reported from the
VESSL run did not appear in this local CPU run. No VESSL job was launched.

Existing skips include unavailable Plotly (the `test_visualize_3d` module and
two realized-grid far-field checks), so those optional visualization checks
are not validated here. GPU, slow and slow-physics lanes are outside the default
selection. Existing skip/xfail markers are unchanged.

Local evidence is retained under `.validation-931-regressions/`: the four
test logs/XML reports (focused plus three full-lane parts), collected node IDs,
continuation manifests/scripts, and `aggregate.json` containing the coverage
accounting assertions. Pre-existing `.validation-931/` is untouched. The
continuation commands were:

```sh
PYTHONPATH=$PWD JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 \
  timeout 1200 python .validation-931-regressions/run_remaining.py
PYTHONPATH=$PWD JAX_PLATFORMS=cpu MPLBACKEND=Agg OMP_NUM_THREADS=1 \
  timeout 1200 python .validation-931-regressions/resume_default.py \
  --prior default.xml --prior remaining.xml --output final
```

`git diff --check` passes. Ruff on changed Python files reports one existing
F401 (`jax.numpy` imported but unused) in `test_thin_conductor.py`; running
Ruff on that file from `65b70a77` reports the same finding at its original
line 511. This repair adds no lint findings and leaves that unrelated import
unchanged.
