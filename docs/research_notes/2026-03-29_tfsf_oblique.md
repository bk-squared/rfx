# 2026-03-29: TFSF broadened to true oblique normal-incidence-plane support

## What changed

- `rfx.sources.tfsf.init_tfsf(...)` now supports `angle_deg`
- The oblique path uses analytic incident-field evaluation on the x TFSF
  boundaries while preserving the existing boundary-correction structure
- Supported scope:
  - propagation along `+x` or `-x`
  - oblique tilt in the single transverse plane implied by polarization
    - `polarization='ez'` → x-y plane
    - `polarization='ey'` → x-z plane
- Existing normal-incidence 1D auxiliary path remains in place for `angle_deg=0`
- `Simulation.add_tfsf_source(...)` now accepts `angle_deg`

## Verification

- `pytest -q tests/test_api.py tests/test_simulation.py` → 33 passed
- `pytest -q tests/test_physics.py::test_fresnel_normal_incidence tests/test_physics.py::test_fresnel_oblique_te tests/test_waveguide_port.py tests/test_dft_probes.py tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs` → 16 passed
- `python -m py_compile rfx/sources/tfsf.py rfx/api.py tests/test_api.py tests/test_simulation.py` → passed
- LSP diagnostics on affected files → 0 errors

## Functionality impact

This closes the core oblique-incidence TFSF functionality gap for the current
single-transverse-plane formulation.

## Remaining limitations

- the oblique path is still tied to plain periodic boundaries rather than full
  Bloch-periodic boundary conditions
- the incidence plane is limited by polarization (`ez`→x-y, `ey`→x-z)
- there is no arbitrary 3D azimuth / full-vector wave specification yet

## Updated next priorities

1. broader waveguide API beyond the current empty rectangular-waveguide scope
2. more general Bloch/azimuthal plane-wave specification if needed
