---
title: "Migration Guide"
sidebar:
  order: 90
---

This guide helps users coming from **Meep** or **OpenEMS** translate their
workflows into rfx. rfx follows the same Yee-cell FDTD physics, so the core
concepts are familiar -- the API surface is different.

---

## Concept Mapping

| Concept | Meep | OpenEMS | rfx |
|---------|------|---------|-----|
| Grid setup | `Simulation(resolution=N)` | `InitCSX()` + `InitFDTD()` | `Simulation(freq_max=...)` or `Simulation.auto(...)` |
| Cell size | `resolution` (cells/unit) | `SetDeltaUnit(1e-3)` | `dx=` in metres (auto-calculated from `freq_max`) |
| Source | `EigenModeSource`, `Source` | `AddExcitation` | `add_port()`, `add_source()` |
| S-parameters | `add_flux()` + post-processing | `CalcPort` | `compute_s_params=True` in `run()` |
| Resonance finding | `harminv(...)` | Manual FFT | `result.find_resonances()` |
| Auto-stop | `stop_when_fields_decayed` | `EndCriteria` | `run(until_decay=1e-3)` |
| Materials | `Medium(epsilon=...)` | `AddMaterial` | `sim.add(shape, material="fr4")` or `sim.add_material(...)` |
| PEC | `PerfectElectricalConductor` | `AddMetal` | `material="pec"` |
| PML / ABC | `PML(thickness)` | `AddPML` | `boundary="cpml"` |
| Dispersive media | `LorentzianSusceptibility` | `AddLorentzMaterial` | `DebyePole`, `LorentzPole`, `drude_pole()` |
| Differentiable | Not available | Not available | `jax.grad(loss_fn)(params)` |
| Inverse design | Not native (adjoint plugin) | Not native | `rfx.optimize(sim, design_region, objective)` |
| Subgridding | Not available | Not available | experimental / specialized |
| Non-uniform mesh | Not native | `SmoothMeshLines` | `dz_profile` or `auto_configure()` |

---

## Quick Translation

### Meep: Rectangular Cavity Resonance

```python
# Meep
import meep as mp

sim = mp.Simulation(
    cell_size=mp.Vector3(0.1, 0.1, 0.05),
    resolution=50,
    boundary_layers=[],
)
sim.sources = [mp.Source(
    mp.GaussianSource(frequency=2.0, fwidth=0.5),
    component=mp.Ez,
    center=mp.Vector3(0.03, 0.03, 0.02),
)]
sim.run(mp.at_beginning(mp.output_epsilon),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(), 1e-6))
```

```python
# rfx equivalent
from rfx import Simulation

sim = Simulation(freq_max=5e9, domain=(0.1, 0.1, 0.05), boundary="pec")
sim.add_source(position=(0.03, 0.03, 0.02), component="ez")
sim.add_probe(position=(0.06, 0.06, 0.02), component="ez")
result = sim.run(until_decay=1e-3)
freqs = result.find_resonances()
```

### OpenEMS: Waveguide S-Parameters

```matlab
% OpenEMS (MATLAB)
CSX = InitCSX();
FDTD = InitFDTD('EndCriteria', 1e-5);
CSX = AddMetal(CSX, 'PEC');
CSX = AddBox(CSX, 'PEC', 10, [0 0 0], [40 20 10]);
[CSX, port{1}] = AddRectWaveGuidePort(CSX, 0, 1, ...);
RunOpenEMS(Sim_Path, Sim_CSX, '--engine=multithreaded');
port = calcPort(port, Sim_Path, freq);
s11 = port{1}.uf.ref ./ port{1}.uf.inc;
```

```python
# rfx equivalent
from rfx import Simulation, Box

sim = Simulation(freq_max=15e9, domain=(0.04, 0.02, 0.01), boundary="cpml")
sim.add(Box((0, 0, 0), (0.04, 0.02, 0.01)), material="pec")
sim.add_port(
    position=(0.005, 0.01, 0.005),
    component="ez",
    waveform=GaussianPulse(f0=10e9),
)
result = sim.run(until_decay=1e-5, compute_s_params=True)
s11 = result.s_params[0, 0, :]
```

### Meep Adjoint -> rfx Inverse Design

```python
# Meep (requires meep.adjoint plugin)
opt = mpa.OptimizationProblem(...)
opt.update_design([design_params])
f, g = opt()  # forward + adjoint
```

```python
# rfx (native JAX autodiff)
import jax

def loss(eps_r):
    sim = Simulation(freq_max=8e9, domain=(0.04, 0.01, 0.01), boundary="cpml")
    sim.add_port(position=(0.008, 0.005, 0.005), component="ez")
    result = sim.run(n_steps=150, compute_s_params=True, eps_r_override=eps_r)
    return -jnp.abs(result.s_params[0, 0, 50])

grad_fn = jax.grad(loss)
gradient = grad_fn(initial_eps_r)
```

---

## What rfx Does Differently

### JAX-Native, GPU-Accelerated

rfx runs on GPU out of the box via JAX. No separate engine binary, no
file-based I/O between the Python front-end and a C++ solver. The full
time-stepping loop is JIT-compiled and executes on GPU (~3,000 Mcells/s on
RTX 4090).

### Differentiable by Default

Every rfx simulation is differentiable with `jax.grad`. Gradients propagate
through the entire FDTD time-stepping, sources, probes, and post-processing.
This enables gradient-based inverse design without adjoint plugins or
finite-difference approximations.

### Declarative Builder API

rfx uses a single `Simulation` object that accumulates geometry, materials,
sources, and probes. Call `run()` once -- S-parameters, resonances, and
field snapshots are computed automatically based on what was requested.

### Built-in Material Library

Eleven common RF materials (`fr4`, `rogers4003c`, `alumina`, `silicon`,
`copper`, `pec`, etc.) are available by name. No need to look up
permittivity tables for standard substrates and conductors.

### Auto-Configuration

`Simulation(freq_max=N)` automatically sets cell size, time step, CPML
thickness, and source waveform. For most antenna and waveguide problems you
only need to specify `freq_max` and `domain`.

### Non-Uniform Z for Thin Substrates

For PCB and patch-style problems, the most practical recent `rfx` workflow is
graded z meshing via `dz_profile` / `auto_configure()`, rather than assuming
one globally fine uniform mesh.

---

## Common Gotchas

| Issue | Solution |
|-------|----------|
| Coordinates are in metres, not mm or cell units | All positions use SI metres |
| `freq_max` is in Hz, not normalized frequency | Use `freq_max=5e9` for 5 GHz |
| PML is outside the domain you specify | `domain=` is the physical region; CPML pads are added automatically |
| `run()` returns a `Result` object | Access fields via `result.s_params`, `result.time_series`, `result.find_resonances()` |
| No mesh file export | rfx solves in-process; use `sim.grid` to inspect the mesh |

---

## Further Reading

- [Quick Start](/rfx/guide/quickstart/) -- first simulation in 15 minutes
- [Simulation API](/rfx/guide/api-reference/) -- current builder reference
- [Sources & Ports](/rfx/guide/sources-ports/) -- source vs. port workflows
- [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) -- thin-substrate workflow
- [Inverse Design](/rfx/guide/inverse-design/) -- gradient-based optimization
- [Advanced Features](/rfx/guide/advanced/) -- dispersive materials, CFS-CPML
- [Geometry & Limitations](/rfx/guide/geometry-and-limitations/) -- tool comparison
