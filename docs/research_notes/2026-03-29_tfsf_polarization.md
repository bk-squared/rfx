# 2026-03-29: TFSF broadened to Ey/Ez normal-incidence polarization support

## What changed

- `rfx.sources.tfsf.init_tfsf(...)` now supports:
  - `polarization='ez'` (existing path)
  - `polarization='ey'` (new path)
- The 1D auxiliary FDTD now carries the correct sign conventions for:
  - `Ez/Hy`
  - `Ey/Hz`
- TFSF boundary corrections now update the matching field pair based on the
  selected polarization.
- `Simulation.add_tfsf_source(...)` now accepts `polarization='ez' | 'ey'`.

## Verification

- `pytest -q tests/test_api.py tests/test_simulation.py` → 29 passed
- `pytest -q tests/test_physics.py::test_fresnel_normal_incidence tests/test_physics.py::test_fresnel_oblique_te tests/test_waveguide_port.py tests/test_dft_probes.py tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs` → 16 passed
- `python -m py_compile rfx/sources/tfsf.py rfx/api.py tests/test_api.py tests/test_simulation.py` → passed
- LSP diagnostics on affected files → 0 errors

## Functionality impact

This broadens the TFSF API within the existing normal-incidence formulation.
It does **not** add oblique incidence or reverse-direction excitation.

## Updated next priorities

1. reverse-direction / oblique-incidence TFSF generalization
2. broader waveguide API beyond the current empty rectangular-waveguide scope
