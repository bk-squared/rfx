# Remove `solver="adi"` (#1487): predeclared plan

Status: predeclared 2026-10-29, before any code. Owner: layers. Base: main after
the ADI layer move (#2209) lands post-cut; this plan is written against that
branch's layout (`rfx/adi.py` shim, `rfx/stepping/adi.py`, `rfx/model/adi_admission.py`).
No code commit lands on this branch before #2209 is on main.

Decision being implemented: the PI decided on 2026-10-05 to remove ADI in 2.1
rather than fix it (#1487 body). Measured causes: interior PEC forced by
overwrite (spectral radius 1.001 inside CFL), an unmatched sigma sponge
(-10.4 dB at 10 GHz against -93 dB for CPML), dielectric faces half a cell off
(+2.2 % on a slab-loaded cavity at 1.5 mm); median time-step gain 1.22x on 41
public grids. This PR changes no Yee result.

## 1. What goes

Modules deleted:

| module | role today |
|---|---|
| `rfx/stepping/adi.py` | ADI kernels (2-D TMz, 3-D Zheng-Chen-Zhang), Thomas solve, sponge sigma |
| `rfx/adi.py` | compatibility shim left by #2209 |
| `rfx/runners/adi.py` | `_run_adi_from_materials` lane runner |
| `rfx/model/adi_admission.py` | interior-PEC and interface refusals |
| `rfx/_adi_notice.py` | `ADI_WARNING`, `ADI_SPONGE_REFUSAL`, `ExperimentalADIWarning` |
| `scripts/diagnostics/adi_pec_stability.py`, `adi_pec_report.py` | research diagnostics; their records stay in rfx-archive |

Code removed from modules that stay:

| file | removed |
|---|---|
| `rfx/__init__.py` | the `from rfx.adi import ...` line (7 names) and the comment row naming `ADIState2D/3D` |
| `rfx/api/__init__.py` (five file) | the `solver == "adi"` construction block, the per-face absorber guard, the ADI docstring text, `self._adi_cfl_factor` |
| `rfx/api/_execute.py` (five file) | the ADI dispatch (`_run_adi_from_materials` import and both call sites, `dt_adi`) |
| `rfx/api/_preflight.py` | `_validate_adi_interior_pec`, `_validate_cfg_adi_interior_pec`, `_validate_cfg_adi_boundary`, `_validate_cfg_adi_3d_accuracy`, `_validate_adi_configuration` |
| `rfx/preflight/_registry.py` | the three `_validate_cfg_adi_*` ConfigCheck rows |
| `rfx/runners/_admission.py` | lanes `run_adi`/`fwd_adi`, `_ADI`, `_adi_homogeneous`, `_ADI_MATERIAL_ROWS`, the `_adi_cfl_factor` rows and the `_solver` row's ADI lane set |
| `rfx/runners/uniform_forward.py` | the ADI branch |
| `rfx/model/conductors.py`, `rfx/model/__init__.py` | `admit_adi_sources` |
| `rfx/boundaries/depths.py`, `rfx/boundaries/__init__.py` | `adi_uniform_faces` |
| `rfx/artifacts.py`, `rfx/interop/_design.py` | the `adi_cfl_factor` field (see section 3) |

The five files' line counts only fall; size numbers are set to what
`check_file_size_ratchet.py --base` prints for the files the PR touches;
`layer_baseline.json` is regenerated and only falls.

Census before the first code commit (the table above is from the #2209 branch;
the implementation re-runs it on the new main and adds any row it finds):

```
git grep -nE '(\badi\b|\badi_|_adi\b|_adi_|\bADI\b|Adi[A-Z])' -- rfx tests scripts examples
```

## 2. What refuses

One refusal, at declaration, on every path: `Simulation(solver="adi")` raises
`ValueError` before any grid is built. Its message says ADI was removed in 2.1,
states the reason in one sentence (unmatched absorber, half-cell interface
error, unstable interior PEC; median 1.22x step gain) and names the
alternatives: `solver="yee"` on a graded mesh (`dz_profile`, `dx_profile`,
`dy_profile`) for thin layers, and local refinement where it is supported.

The issue text names `run()` and `forward()`; refusing at the constructor
refuses both and also `preflight()`, the S-matrix calculators and interop
rebuilds, with one message. The refusal sits in a model-layer admission
function (`rfx/model/solver_admission.py`, one constant and one function), and
the constructor calls it, so the five file gains no message text.

| path | row |
|---|---|
| uniform | refuses (constructor) |
| graded | refuses (constructor) |
| distributed (uniform and graded) | refuses (constructor) |
| subgridded | refuses (constructor) |
| ADI lane | removed |

`solver` keeps its keyword and default `"yee"`; any value other than `"yee"`
raises, the removal message for `"adi"` and the existing message otherwise.

## 3. Public names that change

| name | 2.0 | 2.1 |
|---|---|---|
| `rfx.ADIState2D`, `rfx.ADIState3D`, `rfx.thomas_solve`, `rfx.adi_step_2d`, `rfx.run_adi_2d`, `rfx.adi_step_3d`, `rfx.run_adi_3d` | top-level imports (not in `__all__`; listed in `docs/guides/api_symbol_inventory.json`, which the PR regenerates) | gone; attribute access raises `AttributeError` naming the removal (module `__getattr__` in `rfx/__init__.py`, one table of the 7 names) |
| `import rfx.adi` | module | `ModuleNotFoundError` (module deleted; the top-level message covers the names users imported) |
| `Simulation(solver="adi")` | experimental lane | `ValueError` (section 2) |
| `Simulation(adi_cfl_factor=...)` | ADI step factor, default 2.0, no effect on Yee | keyword kept for 2.1 with default `None`; any value raises the removal message, so a script that assumed ADI fails instead of running Yee at another step. Keyword removed in 2.2 |
| `ExperimentalADIWarning` (`rfx._adi_notice`) | private module, named in the support matrix | gone |
| `rfx.boundaries.adi_uniform_faces`, `rfx.model.admit_adi_sources` | layer-package exports | gone (internal callers only) |
| design document `solver.adi_cfl_factor` | written by `to_document` | not written; the reader accepts an old document's field when `solver.solver == "yee"` (the value never reached a Yee run) and refuses a document with `solver.solver == "adi"` through the constructor |
| artifact record `adi_cfl_factor` | written | not written |

## 4. Deprecation path

- 2.0 (already shipped as rc): `ExperimentalADIWarning` on every run, support
  matrix row "experimental". There is no "removed in 2.1" text in 2.0 today.
  Question for the desk (section 7): whether the 2.0 final carries one line in
  `ADI_WARNING` and the upgrading page saying ADI is removed in 2.1.
- 2.1 (this PR): removal and refusals as above; changelog fragment
  `changelog.d/<PR>.removed.md`; `docs/guides/known_limitations.md` one line
  (what was removed, why, what to use); the support-matrix ADI rows (the "ADI experimental" row and the
  refusal rows `admission:adi-absorber`, `admission:adi-interface`) replaced
  by one refusal row `admission:adi-removed` naming the contract test.
- 2.2: the `adi_cfl_factor` keyword and the top-level `__getattr__` table go.

Public docs pages that describe ADI (`docs/public/guide/adi-solver.mdx`,
`examples/tutorials/adi_solver_demo.py`, mentions in about 25 pages) are
removed or reworded by docs-2.0 in the docs batch, except what a required
check demands in this PR (site map, symbol inventory, a link checker): those
the PR changes itself. The list goes to the docs-batch issue.

## 5. Tests that move

Deleted (ADI-only; the behaviour they pin no longer exists):
`tests/unit/runners/test_adi.py`, `test_adi_gradient.py`,
`test_adi_experimental_and_sponge_refusal.py`,
`test_adi_material_interface_refusal.py`, `tests/unit/preflight/test_adi_preflight.py`,
`tests/unit/boundaries/test_adi_cpml_reflection.py`,
`tests/unit/autodiff/test_adi_2d_material_gradient.py`,
`tests/data/preflight_split_snapshot/adi_conductor_sheet.json`.

Edited (shared tables and parametrisations lose their ADI rows; no Yee number
or tolerance changes): `tests/contracts/path_disposition.py`,
`test_path_disposition.py`, `tests/unit/runners/test_path_disposition_cells.py`,
`boundary_registry.json` / `.py`, `test_boundary_baseline.py`, `boundary_fields.py`,
`test_pec_lane_parity_numeric.py`, `test_pec_single_owner_lock.py`,
`test_eps_r_unit_coefficient_builders.py`, `test_film_one_plane.py`,
`test_material_averaging_calls.py`, `test_execute_split_qualnames.py`,
`tests/locks/test_preflight_registry_sequence.py`, `test_preflight_split_snapshot.py`,
`tests/unit/autodiff/test_gradient_scale_invariance.py`, `test_calculator_admission.py`,
`test_review_tier1_validation_battery.py`, `test_none_means_current.py`,
`tests/interop/test_interop_design_document.py`, and the rest of the census
rows with one to three ADI references each. Any edit that changes a non-ADI
expected value stops the PR: that would mean ADI removal moved a Yee result.

New, one contract test file `tests/contracts/test_adi_removed.py`:
1. `Simulation(solver="adi")` raises the removal message on a uniform and a graded declaration.
2. `Simulation(adi_cfl_factor=2.0)` raises it.
3. `rfx.run_adi_2d` (each of the 7 names) raises `AttributeError` with the message.
4. A 2.0 design document with `solver.solver == "yee"` and `adi_cfl_factor` rebuilds to the same Simulation; one with `"adi"` raises.
5. No module under `rfx/` imports a deleted module (AST scan of every import).

Mutations (both kinds, reported in the PR body): (a) the constructor refusal
switched off goes red in 1; the `__getattr__` table emptied goes red in 3;
(b) the shim `rfx/adi.py` restored with an import from a live module goes red
in 5; `"adi"` re-added to the admitted solver set goes red in 1.

## 6. Done when

- the census grep returns only the refusal, the `__getattr__` table, the reader
  compatibility branch, the new test, the changelog fragment and docs left to the batch;
- `check_layer_structure.py`: the five files 0, rule R 0, rule L 0, ADI rows gone from the S count;
- the local gate passes; fresh `import rfx` and each layer package load in a new interpreter;
- one review by a separate instance (pure deletion plus one refusal: one review).

## 7. Open for the desk

1. A "removed in 2.1" line in the 2.0 final's ADI warning and upgrading page: yes or no (it would land in the release, not here).
2. Keeping `adi_cfl_factor` for one release with a refusal, against removing it in 2.1 (then a 2.0 script gets `TypeError` without the reason).

## 8. Decision record (rfx-core-desk, 2026-10-29)

1. Yes, a "removed in 2.1" line, docs only: the upgrading page and the ADI row
   of the support matrix, in docs-2.0's final 2.0 PR. No change to
   `ADI_WARNING` or other code before the cut.
2. `adi_cfl_factor` stays one release as a refusal that names the removal; it
   is removed in 2.2 (section 3 as written).

The rest of the design is accepted; code starts after #2209 lands.
