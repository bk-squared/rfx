# rfx Reference-Lane Contract

## Scope
This document defines the current **claims-bearing reference lane** for rfx:

- uniform Cartesian Yee
- RF / microwave workflows
- `pec`, `cpml`, or bounded `upml` boundaries
- point/current sources, lumped/wire ports, waveguide ports
- probes, flux monitors, S-parameters, Harminv resonance, benchmarked NTFF/far-field

Everything outside this scope must be treated by the support matrix, not by implication.

## Field model
The reference lane solves Maxwell curl updates on a staggered Cartesian Yee grid.

- `∂t B = -curl(E)`
- `∂t D = curl(H) - J`
- `D = ε E + P_disp + P_nl`
- `B = μ H`

The reference lane is currently defined for the implementation subsets in `docs/guides/support_matrix.md`.

## Contract surfaces

### Sources
Supported source families in the reference lane:
- point/current-style sources
- lumped and wire ports
- waveguide ports

Each promoted source must have:
- normalization convention
- allowed boundary/material combinations
- observable compatibility

### Observables
Claims-bearing observable families:
- time-series probes
- flux monitors
- calibrated S-parameters
- Harminv resonance extraction
- benchmarked NTFF/far-field workflows

Each promoted observable must have:
- field sampling convention
- timing convention
- normalization convention
- benchmark / convergence evidence

## Support-boundary rule
Unsupported combinations must **hard-fail** instead of silently degrading, mutating, or dropping features.

## Immediate evidence floor
The current reference lane should remain tied to concrete reproducible evidence, including:
- `JAX_ENABLE_X64=1 python examples/crossval/01_meep_waveguide_bend.py`
- `JAX_PLATFORM_NAME=cpu python -m pytest tests/test_api.py tests/test_nonuniform_api.py tests/test_nonuniform_convergence.py -q`
- `vessl run create -f scripts/vessl_v140_validation_v3.yaml`

## Shadow-lane relationship
The nonuniform graded-z lane is preserved, but it is not claims-bearing until its own contract and benchmark ladder exist.
