# 2026-03-29: TFSF broadened to reverse-direction normal-incidence support

## What changed

- `rfx.sources.tfsf.init_tfsf(...)` now supports:
  - `direction='+x'` (existing forward path)
  - `direction='-x'` (new reverse path)
- Reverse propagation is implemented by launching the 1D auxiliary source from
  the opposite margin while reusing the same TFSF boundary-correction logic.
- `Simulation.add_tfsf_source(...)` now accepts `direction='+x' | '-x'`.

## Verification

- `pytest -q tests/test_api.py tests/test_simulation.py` → 31 passed
- `pytest -q tests/test_physics.py::test_fresnel_normal_incidence tests/test_physics.py::test_fresnel_oblique_te tests/test_waveguide_port.py tests/test_dft_probes.py tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs` → 16 passed
- `python -m py_compile rfx/sources/tfsf.py rfx/api.py tests/test_api.py tests/test_simulation.py` → passed
- LSP diagnostics on affected files → 0 errors

## Functionality impact

This closes the reverse-direction slice of the remaining TFSF work. The
remaining TFSF generalization gap is true oblique-incidence support beyond the
current 1D auxiliary formulation.

## Updated next priorities

1. true oblique-incidence TFSF generalization
2. broader waveguide API beyond the current empty rectangular-waveguide scope
