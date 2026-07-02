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
| Cell size | `resolution` (cells/unit) | `SetDeltaUnit(1e-3)` | `dx=` in meters (auto-calculated from `freq_max`) |
| Source | `EigenModeSource`, `Source` | `AddExcitation` | `add_port()`, `add_source()` |
| S-parameters | `add_flux()` + post-processing | `CalcPort` | port-family-specific: lumped/wire `run(compute_s_params=True)`, MSL `compute_msl_s_matrix()`, waveguide `compute_waveguide_s_matrix()` |
| Resonance finding | `harminv(...)` | Manual FFT | `result.find_resonances()` |
| Auto-stop | `stop_when_fields_decayed` | `EndCriteria` | `run(until_decay=1e-3)` |
| Materials | `Medium(epsilon=...)` | `AddMaterial` | `sim.add(shape, material="fr4")` or `sim.add_material(...)` |
| PEC | `PerfectElectricalConductor` | `AddMetal` | `material="pec"` |
| PML / ABC | `PML(thickness)` | `AddPML` | `boundary="cpml"` |
| Dispersive media | `LorentzianSusceptibility` | `AddLorentzMaterial` | `DebyePole`, `LorentzPole`, `drude_pole()` |
| Differentiable | adjoint-solver workflows for selected design-region objectives | not native | `jax.grad(loss_fn)(params)` on supported JAX-traced workflows |
| Inverse design | `meep.adjoint` / `OptimizationProblem` | not native | `rfx.optimize(sim, design_region, objective)` |
| Non-uniform mesh | Not native | `SmoothMeshLines` | bounded `dz_profile` / `auto_configure()` workflows |

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
modes = result.find_resonances()   # list of HarminvMode (fields: freq, decay, Q, ...)
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
# rfx equivalent (waveguide modal-port family)
import jax.numpy as jnp
from rfx import Simulation

# WR-90 rectangular guide. The guide walls are implied by the modal port
# and the transverse (y, z) domain extents -- no explicit PEC fill is needed.
freqs = jnp.linspace(8e9, 11.5e9, 8)   # above the ~6.56 GHz TE10 cutoff
sim = Simulation(freq_max=12e9, domain=(0.10, 0.02286, 0.01016),
                 dx=2e-3, boundary="cpml", cpml_layers=8)
sim.add_waveguide_port(0.024, direction="+x", freqs=freqs, f0=9.75e9, name="in")
sim.add_waveguide_port(0.076, direction="-x", freqs=freqs, f0=9.75e9, name="out")

# Modal V/I decomposition (the default, normalize=False). For dispersion-
# corrected transmission magnitude, pass normalize="flux".
result = sim.compute_waveguide_s_matrix(num_periods=30, normalize=False)
s11 = result.s_params[0, 0, :]   # s_params shape: (n_ports, n_ports, n_freqs)
```

The waveguide path is recommended only within the documented
rectangular-guide gates. Do not treat `run(compute_s_params=True)` as a
universal OpenEMS `CalcPort` equivalent; it is the lumped/wire `add_port(...)`
calculator only.

### Meep Adjoint Solver -> rfx Inverse Design

```python
# Meep (uses the meep.adjoint module)
opt = mpa.OptimizationProblem(...)
opt.update_design([design_params])
f, g = opt()  # forward + adjoint
```

```python
# rfx: gradient-based inverse design (optimize() runs jax.grad through the
# differentiable sim.forward() internally -- no adjoint code to hand-write)
import jax.numpy as jnp
from rfx import Simulation, Box, DesignRegion, GaussianPulse
from rfx.optimize import optimize
from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp

sim = Simulation(freq_max=5e9, domain=(0.05, 0.05, 0.025), dx=2.5e-3, boundary="pec")
sim.add_material("slab_init", eps_r=4.0)
sim.add(Box((0.015, 0.015, 0.005), (0.035, 0.035, 0.020)), material="slab_init")
sim.add_port((0.025, 0.025, 0.0125), "ez",
             waveform=GaussianPulse(f0=3e9, bandwidth=0.8))

region = DesignRegion(
    corner_lo=(0.015, 0.015, 0.005),
    corner_hi=(0.035, 0.035, 0.020),
    eps_range=(1.0, 12.0),
)
objective = minimize_s11_at_freq_wave_decomp(target_freq=3e9, port_idx=0)

# port_s11_freqs is required by this objective (it accumulates per-port V/I
# DFTs at those frequencies inside the JIT scan body).
result = optimize(sim, region, objective,
                  n_iters=50, lr=0.01,
                  port_s11_freqs=jnp.asarray([3e9]))
# result.eps_design (optimized permittivity), result.loss_history (per-iter)
```

---

## What rfx Does Differently

### JAX-Native, GPU-Accelerated

rfx runs on GPU out of the box via JAX. No separate engine binary, no
file-based I/O between the Python front-end and a C++ solver. The full
time-stepping loop is a single JIT-compiled `jax.lax.scan` that executes on
GPU (the project README reports ~7,300 Mcells/s on an RTX 4090).

### JAX-Native Differentiation

Supported rfx optimization workflows are written so `jax.grad` can propagate
through the implemented discrete time-stepping, sources, probes, and objective
post-processing. This enables gradient-based inverse design from ordinary Python
loss functions, while final RF claims still need the relevant validation path.

### Declarative Builder API

rfx uses a single `Simulation` object that accumulates geometry, materials,
sources, and probes. `run()` computes S-parameters when ports exist
(`compute_s_params` defaults to True in that case) and field snapshots when a
`SnapshotSpec` is passed; resonances are extracted on demand from the returned
`Result` via `result.find_resonances()`.

### Built-in Material Library

A built-in material library covers common RF substrates and conductors
(`fr4`, `rogers4003c`, `rogers4350b`, `rt_duroid_5880`, `alumina`, `ptfe`,
`silicon`, `copper`, `aluminum`, `pec`, `air`, `vacuum`, `water_20c`) by name,
so you do not have to look up permittivity tables for standard materials.

### Auto-Configuration

When you do not pass `dx`, rfx derives the cell size from `freq_max` (and the
time step from the CFL limit) at run time, and point sources default to a
`GaussianPulse` centered at `freq_max/2`. CPML uses a 16-layer pad by
default (`cpml_layers=16`). For most antenna and waveguide problems you only
need `freq_max` and `domain`; `Simulation.auto(freq_range=(f_min, f_max))`
additionally proposes a domain and mesh from a target band.

### Bounded Non-Uniform Z for Thin Substrates

For PCB and patch-style problems, graded z meshing via the `dz_profile=`
constructor argument or `auto_configure()` can reduce cell count when the thin
feature is primarily along z. Treat it as a bounded workflow: use the
documented support checks and validate the final observable through the
relevant uniform or external reference lane. See
[Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) for the support envelope.

---

## Common Gotchas

| Issue | Solution |
|-------|----------|
| Coordinates are in meters, not mm or cell units | All positions use SI meters |
| `freq_max` is in Hz, not normalized frequency | Use `freq_max=5e9` for 5 GHz |
| PML is outside the domain you specify | `domain=` is the physical region; CPML pads are added automatically |
| `run()` returns a `Result` object | Access fields via `result.s_params`, `result.time_series`, `result.find_resonances()` |
| No mesh file export | rfx solves in-process; the solved `Grid` is on `result.grid`, and `rfx.plan_simulation_mesh(sim)` audits the mesh before a run |

---

## Further Reading

- [Quick Start](/rfx/guide/quickstart/) -- first simulation in 15 minutes
- [Simulation API](/rfx/guide/api-reference/) -- current builder reference
- [Sources & Ports](/rfx/guide/sources-ports/) -- source vs. port workflows
- [Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/) -- Meep-informed gradient concepts
- [Inverse Design](/rfx/guide/inverse-design/) -- gradient-based optimization
