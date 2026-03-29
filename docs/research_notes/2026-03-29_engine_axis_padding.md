# 2026-03-29: Engine-level per-axis CPML padding enabled broader waveguide support

## What changed

### Core engine
- `rfx.grid.Grid` now supports `cpml_axes`, so CPML padding can be applied on a
  subset of axes instead of always all three
- `Grid` now tracks per-axis pads (`pad_x/pad_y/pad_z`, `axis_pads`)
- `position_to_index(...)` now respects per-axis padding
- `rfx.geometry.csg` now uses per-axis pad offsets instead of assuming symmetric
  padding on every axis

### Functional impact
- This removes the core engine assumption that blocked non-empty x-CPML / y-z
  PEC waveguide simulations
- The waveguide API can now support geometry-loaded guides instead of only the
  empty-guide case

## Verification

- `pytest -q tests/test_grid.py tests/test_api.py tests/test_waveguide_port.py` → 35 passed
- `pytest -q tests/test_simulation.py tests/test_dft_probes.py tests/test_physics.py::test_fresnel_normal_incidence tests/test_physics.py::test_fresnel_oblique_te tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_farfield.py::test_ntff_accumulation_runs` → 19 passed
- `python -m py_compile rfx/grid.py rfx/geometry/csg.py rfx/api.py tests/test_grid.py tests/test_api.py` → passed
- LSP diagnostics on affected files → 0 errors

## Updated next priorities

1. more general Bloch/azimuthal plane-wave specification if needed
2. further waveguide modeling breadth (multiple ports, non-rectangular cross-sections, etc.) if needed
