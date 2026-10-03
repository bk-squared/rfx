# #1465 — subgridded admission implementation and measurements

Base: `origin/main = 292e3903`; branch `nu/1465-subgrid-refuse`.
Commands ran in this worktree, serially, with the existing rfx Python 3.11
venv, CPU JAX, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and bytecode
writes disabled. Scratch and measurement output are in ignored
`_scratch_1465/`. No physics-stability claim is made from the short tests.

## Implemented behavior and locations

| File:line | Change |
|---|---|
| `rfx/subgridding/_notice.py:3` | Shared facts/remedy text and `ExperimentalSubgridWarning(UserWarning)`. Existing warning subclasses concern preflight, auto-meshing, or lumped diagonals; none is a generic experimental-lane notice. |
| `rfx/runners/_admission.py:462` | Refinement is no longer unconditionally admitted by subgridding. Research/off opt in through the lane gate at line 806; production uses the existing `NotImplementedError` admission refusal. |
| `rfx/api/_execute.py:529` | Admission before the private API wrapper enters the runner module. |
| `rfx/api/_execute.py:4971` | Admission immediately after run dispatch selects subgridding. |
| `rfx/runners/subgridded.py:54` | Direct one-shot entry refuses production before reading the grid or stepping. |
| `rfx/runners/subgridded.py:757` | Direct path entry refuses production; research/off emit one notice. Internal main/replay calls suppress duplicate notices. |
| `rfx/subgridding/validation.py:591` | Production issue `subgrid_unstable_unverified`; `supported=False`, level `unsupported-unstable-unverified`. Other envelope findings are retained. |
| `rfx/api/__init__.py:799` | Public refinement and validation documentation updated. |
| `tests/contracts/test_subgrid_requires_experimental.py:1` | One new contract file: default refusal before patched runner entry, sequential sweep, direct entries, real research/off runs with one notice, both validation APIs. |
| `tests/contracts/path_disposition.py:180` | Earlier production-envelope cells now name #1465; slab row refuses by default, with research/off gate. Other carried subgrid cells explicitly describe opt-in behavior. |
| `tests/contracts/test_path_disposition.py:324` | Table/admission contract also pins the production versus opt-in slab disposition. |
| `tests/contracts/test_realized_boundary.py:43` | Ten historical B1 subgrid refusal reasons superseded by #1465; B1 artifact unchanged; two earlier errors remain. |
| `validation/research/subgrid/13_subgrid_material_validation.py:1` | Example now expects both production subgrid arms unsupported. Uniform reference retained. |

The exact notice states: “The subgridded lane is unstable and unverified: an
empty lossless PEC cavity's first resonance is 8 % high at a 1.5 mm coarse
mesh and its field grows exponentially (#1465). Use a graded mesh
(dx/dy/dz profiles) for local resolution, or pass validation="research" to
run it anyway.” Existing earlier configuration/preflight errors are retained.

## Entries reaching the runner

* `Simulation.run()` → `_ExecuteMixin._run_subgridded()` →
  `run_subgridded_path()` → `_run_subgridded_once()`.
* `vmap_material_sweep()` (`rfx/vmap_sweep.py:901`) → its
  `_sequential_fallback()` (`:1143`, calls `sim_copy.run` at `:1188`) →
  the same run path. The default refusal is tested before runner entry.
* Direct internal `_ExecuteMixin._run_subgridded`, the exported
  `rfx.runners.run_subgridded_path` (also available in its module), and
  private `_run_subgridded_once` are guarded independently.
* `run_subgridded_path` performs the initial solve and optional per-driven-port
  S-matrix replays through `_run_subgridded_once` (lines 762 and 818).
* `forward`, optimization, batched vmap, distributed, and calculator paths
  already refuse refinement rather than entering this module. Graded dispatch
  retains its existing refinement refusal. The grid-level
  `rfx.subgridding.runner.run_subgridded` is a separate low-level runner taking
  grids; it does not route through `rfx/runners/subgridded.py`.

## Census and tests

[Call-by-call census and initial failing test IDs](1465-subgrid-census.md).
All 28 existing omitted-validation calls and the two production-default
helpers were exercised through their owning tests. Twelve omitted-validation
call sites were changed to research with a #1465 comment. The lid helper's
production default was also changed to research for experimental boundary
checks; explicit production cases assert refusal. No test was deleted.

| Execution group | Before caller adaptation | Final result |
|---|---:|---:|
| Eleven direct census test selections, parametrized | 34 passed, 2 failed | 36 passed |
| `test_sheet_impedance.py -k subgrid` | 1 failed | 1 passed |
| Full `test_refinement_refused_on_graded_mesh.py` | 25 passed, 2 failed | 27 passed |
| `test_removed_coaxial_s_matrix_lane.py -k sbp` | 1 passed | 1 passed |
| `test_calculator_admission.py -k refinement` | 1 passed | 1 passed |
| `test_distributed_admission_refusals.py -k refinement` | 2 passed | 2 passed |
| `test_path_disposition_cells.py -k subgrid` | 66 passed, 43 failed | 109 passed |
| Same file, `-k 'guarded_lid or slab_reaching'` | — | 6 passed |
| `test_realized_model_cells.py -k subgrid` | 11 passed, 19 failed | 28 passed, 2 existing xfails |
| `test_silent_drop_warnings.py -k subgrid` | 7 failed | 7 passed |
| Preflight helper's two owning tests | — | 2 passed |
| `test_realized_boundary.py -k subgridded` | 2 passed, 10 failed (changed refusal reason) | 12 passed, included in final group below |
| Final contracts/entry/lock selection | — | 34 passed, 2 skipped, 490 deselected |

The final selection was:

```text
python -m pytest -q \
  tests/contracts/test_subgrid_requires_experimental.py \
  tests/contracts/test_path_disposition.py \
  tests/contracts/test_realized_boundary.py \
  tests/contracts/test_one_node_tie_rule.py \
  tests/locks/test_runner_split_bit_identity.py \
  -k 'subgrid or path_disposition'
```

The two skips are the pre-split baseline lock without an external baseline
and the one-node-tie subgrid/non-uniform combination. Counts above overlap
(e.g. boundary checks and three direct census tests in the nonuniform file);
they are execution-group counts, not an aggregate unique-test count. No full
repository suite was run. An initial `-k subgrid` selection of preflight
checks selected zero tests; the two owning tests were subsequently selected
by name and passed. `ruff check --no-cache` on changed implementation/contract
files and `git diff --check` passed.

Mutation: changing the subgrid refinement admission gate to unconditional
acceptance made both default-refusal tests fail with
`Failed: subgridded runner entered` (2 failed, 6 deselected; exit 1).
The gate was restored before final checks. This mutation was run before the
additional sequential-sweep contract was added.

## Graded/non-uniform controls

Original `292e3903` rfx sources were extracted with `git archive` into
`_scratch_1465/base`; no new git repository was made. The same control script
ran against the original and changed sources, sequentially. `cmp` of their
JSON records returned 0.

| Control, 8 steps | Original versus changed |
|---|---|
| Explicit `dx_profile`, no refinement, 12 mm PEC box | Probe `(8,1)`, float32, byte-identical; SHA256 `3fa6267e7d43c614a9d82e5ac35629b58a7ae5528e056d72d83024aac49f007f`; dt `1.9065748695309645e-12` s. |
| Same explicit graded mesh plus default refinement | Same `NotImplementedError` and byte-identical full message (`subgridding on a non-uniform mesh`, #1240). |
| Auto-meshed thin-dielectric geometry, no refinement | Probe `(8,1)`, float32, byte-identical; SHA256 `66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925`; dt `5.494277160899145e-14` s. This short record is all zero. |
| Same automatic non-uniform mesh plus default refinement | Same `NotImplementedError` and byte-identical full message. |

The existing 27-test graded/refinement file also passed, including auto-mesh,
forward, distributed, calculator, and explicit research subgrid cases.

## Source physics facts and example measurement

Read-only source: `~/Documents/rfx-archive/rfx/records/20261003-1373-adi-subgrid-interface-eps/controls_report.md`, controls 3 and 4.
Control 3's empty 1.5 mm case reports first resonance 8.13942 GHz against
7.50637 GHz (+8.43%) and fitted decay −0.0500 ns⁻¹. Control 4 is the original
slab case extended to 32 ns: at 1.5 mm its 28–32 ns / 4–8 ns RMS ratio is
11.69, with first-mode fitted decay −0.1172 ns⁻¹ in all three fit windows.
The long instability reproduction was not rerun here; the implementation
uses the PI's supplied facts and decision.

The updated material-validation research example's `run_example()` completed:
`status=passed`, vacuum support False, dielectric support False; rejection
codes include `subgrid_unstable_unverified`. Its retained uniform dielectric
reference gave analytic 3.533088000006387 GHz, FFT peak
3.5327847558337674 GHz, absolute relative error 0.00858297819412569%.
Output was written only under scratch, not to the example's usual artifact.

Conclusions: 리더가 채움.
