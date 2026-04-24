# SBP-SAT final verifier report

## Purpose

Tie every current public/documented SBP-SAT claim to concrete tests,
benchmarks, and unsupported-case documentation.

## Final verdict

The current SBP-SAT lane is **documented correctly** for public consumption as
an **experimental, proxy-only, all-PEC z-slab** lane.

The final verifier recommendation is:

- keep the lane visible;
- keep the lane experimental;
- do not promote beyond the current claim set.

## Claim-to-evidence map

| Public/documented claim | Evidence |
|---|---|
| SBP-SAT is experimental, not claims-bearing | `docs/guides/support_matrix.md`, `docs/guides/support_matrix.json`, `tests/test_support_matrix_sbp_sat.py` |
| Public wording stays narrow and exact | `docs/public/guide/subgridding.mdx`, `docs/public/api/support-boundaries.mdx`, `README.md`, `tests/test_public_subgridding_docs_contract.py` |
| Current runtime surface is all-PEC / z-slab / soft point source / point probe only | `docs/guides/support_matrix.json`, `tests/test_support_matrix_sbp_sat.py`, `tests/test_sbp_sat_api_guards.py`, `rfx/runners/subgridded.py` |
| Current benchmark evidence is proxy-only | `tests/test_subgrid_crossval.py`, `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md`, `tests/test_support_matrix_sbp_sat.py` |
| Future widening is blocked until explicit gates are met | Milestone 5-8 RFC docs + contract tests |

## Unsupported-case traceability

The following unsupported-case groups are all explicitly documented and tied to
hard-fail or blocked-gate evidence:

- non-PEC boundary coexistence
- arbitrary 6-face box refinement
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
- API hard-fail tests
- RFC contract tests for Milestones 5-8

## Promotion decision

Based on the evidence above, the verifier does **not** recommend promotion to a
broader support status.

The current verified statement is:

> SBP-SAT subgridding is available as an experimental all-PEC z-slab lane with
> soft point sources, point probes, and proxy benchmark evidence only.

Anything broader would exceed the presently verified evidence.
