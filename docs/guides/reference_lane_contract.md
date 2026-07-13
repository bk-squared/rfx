# rfx Reference-Lane Contract

## Scope
This document defines the current **claims-bearing reference lane** for rfx:

- uniform Cartesian Yee
- RF / microwave workflows
- `pec`, `cpml`, or bounded `upml` boundaries
- point/current sources, lumped/wire ports, specialized microstrip-line ports,
  waveguide ports
- probes, flux monitors, Harminv resonance, benchmarked NTFF/far-field, and
  S-parameters only through the port-family envelopes in
  `docs/guides/sparameter_support_matrix.md`

Everything outside this scope must be treated by the support matrix, not by implication.

## Field model
The reference lane solves Maxwell curl updates on a staggered Cartesian Yee grid.

- `∂t B = -curl(E)`
- `∂t D = curl(H) - J`
- `D = ε E + P_disp + P_nl`
- `B = μ H`

The reference lane is currently defined for the implementation subsets in `docs/guides/support_matrix.md`.

## Contract surfaces

### Physics evidence contract

The reference lane is not defined by pytest success alone. The durable evidence
taxonomy lives in `docs/guides/physics_validation_evidence_rule.md`.

In short:
- API/shape/no-crash tests are contract tests, not physics validation.
- Physics validation requires an analytic oracle, independent field-dump
  replay, or an external full-wave cross-solver comparison.
- Claims-bearing status additionally requires the valid mesh/frequency/geometry
  envelope to be stated next to the claim.

### Sources
Supported source families in the reference lane:
- point/current-style sources
- lumped and wire ports
- specialized microstrip-line ports
- waveguide ports

Each promoted source must have:
- normalization convention
- allowed boundary/material combinations
- observable compatibility

### Observables
Claims-bearing observable families:
- time-series probes
- flux monitors
- S-parameters with the E-level, artifact, metric, and envelope stated in
  `docs/guides/sparameter_support_matrix.md`
- Harminv resonance extraction
- benchmarked NTFF/far-field workflows

Each promoted observable must have:
- field sampling convention
- timing convention
- normalization convention
- benchmark / convergence evidence

### S-parameter calculators

The detailed port-family contract is versioned in
`docs/guides/sparameter_support_matrix.md`. The reference-lane rule is:

- `run(compute_s_params=True)` means `add_port(...)` lumped/wire ports only.
- `forward(port_s11_freqs=...)` means lumped/wire S11 vectors on the uniform
  single-device differentiable path only.
- `compute_msl_s_matrix()` owns specialized microstrip-line ports.
- `compute_waveguide_s_matrix()` owns rectangular waveguide full S-matrices.
- Coaxial line reflection is promoted only through
  `compute_coaxial_line_reflection(...)` and only inside its documented
  one-port transmission-line envelope; the older `compute_coaxial_s_matrix(...)`
  single-plane path remains deprecated / experimental.
- Floquet has modal helper/replay and analytic slab diagnostics only; source,
  TFSF, probe, and flux-monitor surfaces are not implied S-parameter
  calculators.

## Support-boundary rule
Unsupported combinations must **hard-fail** instead of silently degrading, mutating, or dropping features.

## Immediate evidence floor
The current reference lane should remain tied to concrete reproducible evidence, including:
- `python -m pytest tests/test_crossval_manifest_contract.py -q`
- `JAX_ENABLE_X64=1 python validation/crossval/01_waveguide_bend.py`
- `PYTHONPATH=. python scripts/run_crossval_cpu.py`
- `JAX_PLATFORM_NAME=cpu python -m pytest tests/test_api.py tests/test_nonuniform_api.py tests/test_nonuniform_convergence.py -q`
- `vessl run create -f scripts/vessl_crossval_external.yaml`
- `vessl run create -f scripts/vessl_gpu_suite.yaml`

The scheduled external-reference membership is also read from
`validation/crossval/manifest.json` by `.github/workflows/validation.yml`; do not
maintain a second script list here.

## Shadow-lane relationship
The nonuniform graded-z lane is preserved, but it is not claims-bearing until its own contract and benchmark ladder exist.
