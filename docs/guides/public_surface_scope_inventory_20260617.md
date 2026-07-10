# Public Surface Scope Inventory — 2026-06-17

## Purpose

This maintainer inventory lists repository surfaces that require explicit support-matrix promotion before they appear in public user guides. It is not part of the deployed public site route map.

## Public guide inclusion criteria

Public user guides should currently include:

- uniform Cartesian Yee RF/FDTD workflows;
- current patch and rectangular-waveguide cross-check examples;
- port-family S-parameter calculators inside their documented envelopes;
- conservative differentiable design workflows with proxy objectives and explicit validation caveats.

A repository surface should stay out of public user guides until it has a public workflow, support-matrix entry, and validation evidence appropriate to its claims.

## Repository surfaces requiring explicit promotion

| Surface | Representative locations | Current public-guide status | Inclusion requirement |
|---|---|---|---|
| SBP-SAT / subgridding | `rfx/subgridding/**`, `rfx/runners/subgridded.py`, `rfx/runners/disjoint.py`, `examples/research/subgrid/**`, `tests/test_subgrid*`, `scripts/subgrid_*` | outside public support scope | add a support-matrix entry, public workflow, and validation envelope before publication |
| Repo-local operating notes | `docs/agent/**` | outside public site scope | publish only after an explicit user-facing documentation plan exists |
| Incomplete guide stubs | `docs/public/guide/antenna-metrics.mdx`, `comparison.mdx`, `conformal-pec.mdx`, `material-fitting.mdx`, `topology-optimisation.mdx` | outside route inventory | publish only after the pages are complete and support-matrix aligned |
| Floquet/RIS workflows | `rfx/floquet.py`, `rfx/ris.py`, `Simulation.add_floquet_port(...)`, `compute_floquet_s_params`, `examples/tap_paper/**` | outside public support scope | add a promoted API contract and external periodic-cell validation envelope |
| Conformal PEC | `rfx/geometry/conformal.py`, `conformal=True` / conformal boundary support | outside public support scope | demonstrate a stable accuracy envelope before publication |
| ADI solver | `rfx/adi.py`, top-level ADI imports | outside public support scope | add a user-facing workflow and accuracy/stability envelope |
| Distributed / multi-GPU execution | `devices=...` run paths, distributed/nonuniform code paths, `jax.pmap` usage | outside public support scope | add maintained examples and support-combination checks |
| Deprecated coaxial S-matrix path | `Simulation.compute_coaxial_s_matrix(...)`, `CoaxialSMatrixResult`, low-level coaxial plane helpers | outside coaxial claim surface | keep public coaxial claims on `compute_coaxial_line_reflection(...)` unless a new envelope is validated |
| Generalized planar ports | stripline/CPW/microstrip-to-coax planning diagnostics and support-matrix future-family entries | outside public support scope | implement public APIs and external validation per family |
| AMR and surrogate export | `rfx/amr.py`, `rfx/surrogate.py` | outside public guide scope | add a supported user workflow and examples |
| Streamlit dashboard | `rfx/dashboard/**` | outside public guide scope | add an install, test, and release path before publication |
| Diagnostic/archive scripts and local validation artifacts | `scripts/diagnostics/**`, `scripts/archive/**`, local validation artifact paths | maintainer evidence only | summarize outcomes in support docs without exposing run-log detail |

## Admission guardrails

- Public guide routes should map to maintained user workflows, not to exploratory or compatibility code.
- Generated API pages are lookup aids and do not promote a symbol by themselves.
- Support claims should cite the support matrix or a public validation page rather than run-log or local artifact paths.
