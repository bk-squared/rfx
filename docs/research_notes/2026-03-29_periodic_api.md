# 2026-03-29: High-level API now exposes periodic-boundary controls

## What changed

- `rfx.api.Simulation` now exposes `set_periodic_axes(...)`
- The high-level API routes periodic-axis choices into the existing low-level
  `periodic=(x, y, z)` support in `rfx.simulation.run()`
- Empty string disables manual periodic overrides; any combination of `x`, `y`,
  `z` is accepted

## Guardrails

- invalid axis names fail clearly
- manual periodic-axis overrides are rejected when using specialized boundary
  configurations that already own the boundary policy:
  - TFSF
  - waveguide-port API

## Verification

- `pytest -q tests/test_api.py` → 19 passed
- `pytest -q tests/test_simulation.py tests/test_waveguide_port.py tests/test_dft_probes.py tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs` → 22 passed
- `python -m py_compile rfx/api.py tests/test_api.py` → passed
- LSP diagnostics on affected files → 0 errors

## Functionality impact

This closes the next priority from
`docs/research_notes/2026-03-29_functionality_gap_analysis.md`.

## Updated next priorities

1. broaden TFSF incidence / polarization support
2. broaden waveguide API beyond the current empty rectangular-waveguide scope
