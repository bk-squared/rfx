# 2026-03-29: TFSF integration hardened across runner and high-level API

## Why this step

The Stage 2 roadmap left one clear integration gap after the standalone TFSF
implementation: plane-wave workflows still depended on manual Python loops.
That gap is now closed at both the low-level runner and high-level API, and the
follow-up hardening work now fails known-bad TFSF setups loudly instead of
letting them produce misleading results.

## What changed

### Low-level runner
- `rfx.simulation.run()` accepts:
  - `periodic=(x, y, z)`
  - `tfsf=(cfg, state)`
- Periodic axes automatically skip CPML and PEC handling.
- The compiled loop preserves the manual leapfrog ordering:
  1. `update_h`
  2. `apply_tfsf_h`
  3. `update_tfsf_1d_h`
  4. `update_e`
  5. `apply_tfsf_e`
  6. `update_tfsf_1d_e`

### High-level API
- `rfx.api.Simulation` exposes `add_tfsf_source(...)` for the currently
  supported plane-wave mode:
  - normal incidence along `+x`
  - `Ez` polarization
  - `boundary='cpml'`
  - `cpml_layers > 0`
  - `mode='3d'`
- The API automatically routes TFSF runs through the compiled runner with:
  - `periodic=(False, True, True)`
  - `cpml_axes='x'`

### Hardening / issue resolution
- Unsupported combinations now fail clearly:
  - TFSF with PEC boundaries
  - TFSF with `cpml_layers=0`
  - TFSF with non-3D mode
  - TFSF together with lumped ports
- `rfx.sources.tfsf.init_tfsf()` now rejects impossible geometry when
  `cpml_layers + tfsf_margin` leaves no valid TFSF box.
- The high-level API now validates that the TFSF x-boundary planes remain
  vacuum, matching the documented correction assumption.

## Verification

- `pytest -q tests/test_api.py tests/test_simulation.py` → 20 passed
- `pytest -q tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs` → 5 passed
- `python -m py_compile rfx/api.py rfx/simulation.py rfx/sources/tfsf.py tests/test_api.py tests/test_simulation.py` → passed
- `ruff check ...` unavailable here (`ruff: command not found`)

## Remaining constraint

TFSF is now guarded against the known bad configurations above, but the feature
scope is still intentionally narrow: normal-incidence `+x`, `Ez`, `3d`, using
the existing TFSF formulation.
