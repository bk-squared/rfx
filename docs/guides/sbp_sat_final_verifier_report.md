# SBP-SAT final verifier report

## Purpose

Tie every current public/documented SBP-SAT claim to concrete tests,
benchmarks, and unsupported-case documentation.

## Final verdict

The current SBP-SAT lane is **documented correctly** for public consumption as
an **experimental, proxy-only, axis-aligned arbitrary-box** lane with selected
reflector/periodic and bounded CPML boundary subsets.

The final verifier recommendation is:

- keep the lane visible;
- keep the lane experimental;
- do not promote beyond the current claim set.

## Claim-to-evidence map

| Public/documented claim | Evidence |
|---|---|
| SBP-SAT is experimental, not claims-bearing | `docs/guides/support_matrix.md`, `docs/guides/support_matrix.json`, `tests/test_support_matrix_sbp_sat.py` |
| Public wording stays narrow and exact | `docs/public/guide/subgridding.mdx`, `docs/public/api/support-boundaries.mdx`, `README.md`, `tests/test_public_subgridding_docs_contract.py` |
| Current runtime surface is arbitrary-box / selected reflector-periodic subset / bounded CPML subset / soft point source / point probe only | `docs/guides/support_matrix.json`, `tests/test_support_matrix_sbp_sat.py`, `tests/test_sbp_sat_api_guards.py`, `rfx/runners/subgridded.py` |
| Public benchmark evidence remains proxy-only | `tests/test_subgrid_crossval.py`, `tests/test_sbp_sat_box_crossval.py`, `tests/test_sbp_sat_boundary_crossval.py`, `tests/test_sbp_sat_absorbing_crossval.py`, `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md`, `tests/test_support_matrix_sbp_sat.py` |
| Bounded-CPML point-probe R/T feasibility is internal and inconclusive, not public support | `tests/test_sbp_sat_true_rt_feasibility.py`, `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md`, `docs/guides/support_matrix.json`, `tests/test_support_matrix_sbp_sat.py` |
| Future widening is blocked until explicit gates are met | Milestone 5-8 RFC docs + contract tests |

## Unsupported-case traceability

The following unsupported-case groups are all explicitly documented and tied to
hard-fail or blocked-gate evidence:

- UPML, per-face CPML thickness overrides, calibrated open-boundary claims,
  or broader mixed boundary coexistence beyond the currently implemented
  reflector/periodic/CPML subsets
- refined-region advanced ports and observables
- material/time widening beyond the current proxy baseline

Artifacts:

- `docs/guides/sbp_sat_boundary_coexistence_rfc.md`
- `docs/guides/sbp_sat_all_pec_box_refinement_spec.md`
- `docs/guides/sbp_sat_ports_observables_rfc.md`
- `docs/guides/sbp_sat_materials_time_integration_rfc.md`
- corresponding contract tests under `tests/test_sbp_sat_*_contract.py`

## Evidence floor satisfied today

### Public/docs alignment

- support matrix and public docs are aligned
- public subgridding guide stays inside the exact retained scope
- migration/changelog caveats keep wording narrow

### Executable checks

- support-matrix metadata locks
- public-doc wording locks
- proxy benchmark tests
- bounded-CPML point-probe feasibility probe, currently inconclusive and
  therefore non-promotional
- API hard-fail tests
- RFC contract tests for Milestones 5-8

## Promotion decision

Based on the evidence above, the verifier does **not** recommend promotion to a
broader support status.

The current verified statement is:

> SBP-SAT subgridding is available as an experimental axis-aligned arbitrary-box
> lane with selected reflector/periodic and bounded CPML boundary subsets, soft
> point sources, point probes, proxy benchmark evidence, and internal
> bounded-CPML point-probe feasibility evidence that is inconclusive for public
> true R/T.

Anything broader would exceed the presently verified evidence.
