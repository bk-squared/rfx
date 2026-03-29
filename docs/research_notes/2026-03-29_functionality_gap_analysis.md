# 2026-03-29: Further functionality analysis after TFSF hardening

## Current state summary

The branch now supports:
- low-level compiled TFSF + periodic-axis execution
- high-level `Simulation.add_tfsf_source(...)` for the current normal-incidence plane-wave path
- guarded failure for impossible TFSF geometry and non-vacuum TFSF boundary planes

## Highest-value next implementations (functionality view)

### 1. Expose DFT plane probes through the high-level API

**Why this matters**
The codebase already has working plane-frequency monitors, but they are not
reachable from `Simulation`.

**Evidence**
- DFT plane probes exist in `rfx/probes/probes.py:315-393`
- They are exported in `rfx/probes/__init__.py:3-6`
- The high-level API currently exposes ports, point probes, TFSF, NTFF, and snapshots, but no DFT plane monitor builder in `rfx/api.py:285-390`
- The functionality is already validated by `tests/test_dft_probes.py:77-178`

**Recommended implementation**
Add a declarative API surface like `add_dft_plane_probe(...)` and return the
accumulated spectra in `Result`.

### 2. Integrate waveguide ports into the main execution surfaces

**Why this matters**
Waveguide functionality exists and is tested, but it is still isolated from the
compiled runner/API workflows that users are most likely to use.

**Evidence**
- Waveguide source/measurement logic exists in `rfx/sources/waveguide_port.py:1-220`
- Physics-style tests exist in `tests/test_waveguide_port.py:195-279`
- `rfx/sources/__init__.py:1-3` does not expose waveguide-port helpers
- `rfx/__init__.py:5-27` does not export waveguide-port functionality
- `Simulation` has no waveguide-port builder in `rfx/api.py:285-390`

**Recommended implementation**
First integrate waveguide ports into the compiled runner with a narrow supported
configuration, then expose them through `Simulation`.

### 3. Expose periodic-boundary controls in the high-level API

**Why this matters**
The low-level runner now supports explicit periodic axes, but the high-level API
cannot express that capability directly except through the hardcoded TFSF path.

**Evidence**
- Low-level support exists in `rfx/simulation.py` (periodic argument already wired)
- High-level `Simulation.run(...)` does not expose periodic-axis or CPML-axis overrides in `rfx/api.py:522-532`

**Recommended implementation**
Add a small, explicit high-level boundary-control surface rather than forcing
advanced users to drop down to the functional API.

### 4. Broaden TFSF scope only after monitor/port integration

**Why this matters**
Oblique incidence and richer polarization are useful, but they are less urgent
than exposing already-implemented monitor and port functionality.

**Evidence**
- Current high-level TFSF scope is intentionally narrow and documented in `rfx/api.py:328-332`
- Current research note also records the narrow intentional scope in `docs/research_notes/2026-03-29_compiled_tfsf_runner.md:56-60`

**Recommended implementation**
Defer oblique/polarization generalization until the API exposes the existing DFT
plane and waveguide workflows cleanly.

## Priority recommendation

1. **More general Bloch/azimuthal plane-wave specification if needed**
2. **Further waveguide modeling breadth if needed**
