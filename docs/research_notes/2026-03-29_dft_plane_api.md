# 2026-03-29: High-level API now supports DFT plane probes

## What changed

### Compiled runner
- `rfx.simulation.run()` now accepts `dft_planes=[...]`
- DFT plane accumulators are updated inside the compiled scan loop and returned
  in `SimResult.dft_planes`

### High-level API
- `rfx.api.Simulation` now exposes `add_dft_plane_probe(...)`
- `Result` now carries `dft_planes`, keyed by probe name
- Supported configuration:
  - `axis='x' | 'y' | 'z'`
  - physical `coordinate` in metres
  - any field component in `ex/ey/ez/hx/hy/hz`
  - explicit `freqs` or default frequency grid

## Verification

- `pytest -q tests/test_api.py tests/test_simulation.py tests/test_dft_probes.py` → 25 passed
- `pytest -q tests/test_lorentz.py::test_lorentz_simulation_runner tests/test_lorentz.py::test_mixed_debye_and_lorentz_runner tests/test_api.py::test_checkpoint_through_api tests/test_api.py::test_gradient_through_api tests/test_farfield.py::test_ntff_accumulation_runs` → 5 passed
- `python -m py_compile rfx/api.py rfx/simulation.py tests/test_api.py tests/test_simulation.py` → passed
- LSP diagnostics on affected files → 0 errors

## Functionality impact

This closes the highest-priority functionality gap identified in
`docs/research_notes/2026-03-29_functionality_gap_analysis.md`.

## Updated next priorities

1. integrate waveguide ports into the compiled runner / high-level API
2. expose periodic-boundary controls in the high-level API
3. broaden TFSF incidence / polarization support after the above
