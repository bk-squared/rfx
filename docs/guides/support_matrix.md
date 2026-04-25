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
- sources: point/current sources, lumped/wire ports, waveguide ports
- observables: time-series probes, flux monitors, S-parameters, Harminv resonances, NTFF/far-field where benchmarked
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
| Floquet/Bloch | experimental | periodic/phased-array lane | promotion pending explicit benchmark ladder |

## SBP-SAT / subgridding experimental lane

Current retained subset:
- all-PEC outer boundaries
- mixed PEC/PMC reflector-face `BoundarySpec` subsets
- periodic axes when the refinement box is either interior to that axis or spans it end-to-end
- bounded CPML absorbing faces when the refinement box stays outside every active absorber pad plus a one-coarse-cell guard
- do not mix PMC faces with periodic axes, or CPML faces with reflector/periodic faces, in the same supported subset
- one axis-aligned refinement box only; CPML cases require the explicit absorber guard
- soft point sources and point probes only
- proxy numerical-equivalence comparison against a uniform-fine reference

Current policy:
- preserved as a research lane, not a claims-bearing public support surface
- unsupported combinations hard-fail instead of warning or silently dropping
- public support promotion is blocked until true R/T evidence exists
- a bounded-CPML point-probe feasibility probe now exists, but it is
  inconclusive and remains internal evidence only

### Benchmark evidence

| Evidence layer | Status | Artifact | Claim level |
|---|---|---|---|
| unit / integration | implemented | `tests/test_sbp_sat_api_guards.py`, `tests/test_sbp_sat_face_ops.py`, `tests/test_sbp_sat_3d.py`, `tests/test_sbp_sat_alpha.py`, `tests/test_sbp_sat_jit.py` | API guards, operator behavior, smoke stability, JIT seam |
| proxy crossval | implemented | `tests/test_subgrid_crossval.py` | single-probe DFT amplitude/phase vs uniform-fine reference; **not** physical R/T |
| box proxy crossval | implemented | `tests/test_sbp_sat_box_crossval.py` | internal arbitrary-box x/y-face plus edge/corner proxy fixtures; **not** public R/T |
| boundary proxy crossval | implemented | `tests/test_sbp_sat_boundary_crossval.py` | internal PMC reflector plus periodic full-axis/interior proxy fixtures; mixed PMC+periodic remains blocked; **not** public R/T |
| absorbing proxy crossval | implemented | `tests/test_sbp_sat_absorbing_crossval.py` | internal CPML interior-box decay and late-tail proxy fixtures; **not** public R/T or S-parameters |
| bounded-CPML point-probe R/T feasibility | inconclusive | `tests/test_sbp_sat_true_rt_feasibility.py` | internal measurement-contract probe only; **not** public R/T, S-parameters, flux, or port evidence |
| true reflection/transmission | deferred | `docs/guides/sbp_sat_zslab_true_rt_benchmark_spec.md` | no public R/T, S-parameter, or calibrated open-boundary claim yet |

Proxy tolerance is intentionally narrow and local: relative amplitude error
`<= 5%` and phase error `<= 5°` against a uniform-fine reference for the
current PEC-cavity proxy fixture. That tolerance checks numerical closeness to
the reference run; it does not validate incident/reflected/transmitted field
separation, energy balance, or calibrated S-parameters.

### Explicit unsupported combinations in the SBP-SAT lane

| Combination | Status | Expected behavior |
|---|---|---|
| UPML boundary | unsupported | hard-fail |
| CPML refinement box inside absorber guard | unsupported | hard-fail |
| per-face CPML thickness override with subgrid | unsupported | hard-fail |
| mixed reflector + CPML `BoundarySpec` | unsupported | hard-fail |
| mixed periodic + CPML `BoundarySpec` | unsupported | hard-fail |
| mixed PMC + periodic `BoundarySpec` | unsupported | hard-fail |
| periodic axis touched on only one side | unsupported | hard-fail |
| geometry-driven `xy_margin` auto-box refinement | unsupported | hard-fail |
| NTFF | unsupported | hard-fail |
| DFT planes | unsupported | hard-fail |
| flux monitors | unsupported | hard-fail |
| TFSF | unsupported | hard-fail |
| waveguide or Floquet ports | unsupported | hard-fail |
| lumped RLC | unsupported | hard-fail |
| coaxial ports | unsupported | hard-fail |
| impedance point ports or wire/extent ports | unsupported | hard-fail |

## Reference-lane support table

| Dimension | Supported subset |
|---|---|
| grid / runner | uniform Cartesian Yee |
| materials | isotropic linear, conductive, validated Debye/Lorentz/Drude subsets |
| absorbers | PEC, CPML, bounded UPML |
| sources | point/current, lumped port, wire port, waveguide port |
| observables | probes, flux, calibrated S-parameters, Harminv resonance, benchmarked NTFF |
| optimization-facing observables | validated proxy objectives only until explicitly promoted |

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
| Waveguide ports + nonuniform | unsupported | hard-fail |
| Lumped RLC + nonuniform | unsupported | hard-fail |

## Promotion rule

A lane or feature can be promoted to **supported** only when all of the following exist:
1. support-matrix entry
2. explicit source/observable contract where relevant
3. unit + integration tests
4. benchmark / convergence evidence
5. docs/examples/API wording aligned to the promoted scope
