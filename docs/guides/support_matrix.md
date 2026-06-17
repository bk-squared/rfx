# rfx Support Matrix

Status legend:
- **supported** — part of the current claims-bearing surface
- **shadow** — retained and tested, but not claims-bearing yet
- **experimental** — research-grade / partial / promotion pending
- **unsupported** — should fail clearly rather than degrade silently

## Current claims-bearing reference lane

**Lane:** uniform Cartesian Yee RF workflows

### Scope
- boundaries: `pec`, `cpml`, `upml`
- sources: point/current sources, lumped/wire ports, specialized
  microstrip-line ports, waveguide ports, and coaxial line-reflection setups
  only inside the documented coax transmission-line envelope
- observables: time-series probes, flux monitors, Harminv resonances,
  NTFF/far-field where benchmarked, and S-parameters only through the
  port-family envelopes in `docs/guides/sparameter_support_matrix.md`
- materials: isotropic linear, conductive, and validated dispersive subsets
- workflows: cavity, waveguide, patch antenna, simple scattering, resonance, de-embedding, selected differentiable proxy objectives

## Lane summary

| Lane | Status | Current role | Notes |
|---|---|---|---|
| Uniform Yee RF lane | **supported** | claims-bearing reference lane | primary correctness surface |
| Nonuniform graded-z | **shadow** | preserved thin-substrate lane | no silent fallback; no new promotions until re-qualified |
| SBP-SAT / subgridding | experimental | research lane | not part of the claims-bearing surface |
| ADI | experimental | research lane | separate accuracy/stability envelope |
| Distributed | experimental | scaling lane | not part of correctness-bearing baseline |
| Floquet/Bloch | experimental | periodic/phased-array lane | M18 synthetic modal oracle and M20 real-FDTD DFT-plane replay exist; promotion still pending analytic/RCWA/external benchmark ladder |
| Coaxial line reflection | broad-E5 physics demonstrated — audit BLOCKED | one-port transmission-line reflection lane | M74 broad-E5 physics demonstrated (analytic Γ envelope + MEEP broad-E4 cross-check), but evidence artifacts live in gitignored `.omx/`; clean-checkout audit reports `coaxial_port` BLOCKED; re-validation pending validation-framework rework; older `compute_coaxial_s_matrix(...)` remains deprecated / experimental |

## Reference-lane support table

| Dimension | Supported subset |
|---|---|
| grid / runner | uniform Cartesian Yee |
| materials | isotropic linear, conductive, validated Debye/Lorentz/Drude subsets |
| absorbers | PEC, CPML, bounded UPML |
| sources | point/current, lumped port, wire port, specialized microstrip-line port, waveguide port, coaxial line-reflection setups inside the documented envelope |
| observables | probes, flux, Harminv resonance, benchmarked NTFF, and port-family S-parameters with the E-levels / artifacts in `sparameter_support_matrix.md` |
| optimization-facing observables | validated proxy objectives only until explicitly promoted |

## S-parameter calculation contract

The port-family-specific S-parameter contract lives in
`docs/guides/sparameter_support_matrix.md` with a machine-readable companion at
`docs/guides/sparameter_support_matrix.json`.
Evidence terminology is defined in
`docs/guides/physics_validation_evidence_rule.md`; pytest success alone is
not physics validation.

Canonical API rule:
- `add_port(..., extent=None)` and `add_port(..., extent=...)` use
  `run(compute_s_params=True)` for full `Result.s_params` matrices.
- The same lumped/wire `add_port(...)` family uses
  `forward(port_s11_freqs=...)` only for uniform single-device S11 vectors.
- `add_msl_port(...)` uses `compute_msl_s_matrix()`.
- `add_waveguide_port(...)` uses `compute_waveguide_s_matrix()` for full
  multi-port matrices; `run()` exposes only per-port `waveguide_sparams`.
- `add_coaxial_port(...)` uses `compute_coaxial_line_reflection(...)` for the
  promoted one-port coaxial transmission-line reflection envelope; the older
  `compute_coaxial_s_matrix(...)` path is deprecated / experimental.
- `add_floquet_port(...)`, sources, TFSF, probes, and flux monitors do not have
  a promoted high-level S-parameter calculator.

## Nonuniform graded-z shadow lane

Current retained subset:
- graded-z thin-substrate workflows with probes and current-style excitation
- smoke/convergence coverage in `tests/test_nonuniform_api.py` and `tests/test_nonuniform_convergence.py`
- selected dispersive-material smoke coverage

Current policy:
- preserved for continuity and qualification work
- **not** part of the claims-bearing reference lane
- no silent fallback / no silent feature dropping
- no new promotions until contract + benchmark ladder exist

### Explicit unsupported combinations in the nonuniform lane

| Combination | Status | Expected behavior |
|---|---|---|
| Floquet + nonuniform | unsupported | hard-fail |
| NTFF + nonuniform | unsupported | hard-fail |
| DFT planes + nonuniform | unsupported | hard-fail |
| TFSF + nonuniform | unsupported | hard-fail |
| Waveguide ports + nonuniform | shadow / restricted | only documented `compute_waveguide_s_matrix(normalize=True)` single-mode path; otherwise hard-fail |
| Lumped-port S-parameters + nonuniform | unsupported | hard-fail |
| `compute_msl_s_matrix()` + nonuniform | unsupported | hard-fail |
| Coaxial port + nonuniform | unsupported | hard-fail |
| Lumped RLC + nonuniform | unsupported | hard-fail |


## Reporting and export artifact rule

Export helpers such as Touchstone writers, plots, HDF5 snapshots, CSV sweep
summaries, and JSON manifests are reproducibility aids. They preserve data and
make reviews easier, but they do **not** change the support status of the
underlying feature. A public report should include the originating workflow,
source/port family, support status, command, git SHA, and pass/fail metric, and
then cite the applicable support/evidence matrix before making a physics claim.


## Promotion rule

A lane or feature can be promoted to **supported** only when all of the following exist:
1. support-matrix entry
2. explicit source/observable contract where relevant
3. unit + integration tests
4. analytic, dump-replay, external-solver, benchmark, or convergence evidence
5. docs/examples/API wording aligned to the promoted scope
