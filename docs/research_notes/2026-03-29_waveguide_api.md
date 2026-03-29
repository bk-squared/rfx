# 2026-03-29: Waveguide ports integrated into the compiled runner and high-level API

## What changed

### Compiled runner
- `rfx.simulation.run()` now accepts `waveguide_ports=[...]`
- Waveguide-port injection and modal DFT accumulation run inside the compiled scan loop
- `SimResult.waveguide_ports` returns the final accumulated configs

### High-level API
- `rfx.api.Simulation` now exposes `add_waveguide_port(...)`
- `Result.waveguide_ports` returns named waveguide-port configs
- The initial API scope is intentionally narrow:
  - rectangular waveguide uses the full y-z domain
  - `boundary='cpml'`
  - `cpml_layers > 0`
  - `mode='3d'`
  - no lumped ports or TFSF in the same run
  - no geometry/thin-conductor modifications in this first API version
- The API uses an x-CPML / y-z PEC grid layout that matches the existing
  low-level waveguide tests

### Package exports
- waveguide helpers are now exported from:
  - `rfx.sources.__init__`
  - `rfx.__init__`

## Verification

- `pytest -q tests/test_api.py tests/test_simulation.py tests/test_waveguide_port.py` → 31 passed
- `pytest -q tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs tests/test_dft_probes.py` → 8 passed
- `python -m py_compile rfx/api.py rfx/simulation.py rfx/__init__.py rfx/sources/__init__.py tests/test_api.py tests/test_simulation.py tests/test_waveguide_port.py` → passed
- LSP diagnostics on affected files → 0 errors

## Functionality impact

This closes the second priority from
`docs/research_notes/2026-03-29_functionality_gap_analysis.md`.

## Updated next priorities

1. expose periodic-boundary controls in the high-level API
2. broaden TFSF incidence / polarization support
3. broaden waveguide API beyond the current empty rectangular-waveguide scope
