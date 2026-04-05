---
title: "Advanced Features"
sidebar:
  order: 12
---

This page collects features beyond the minimal builder workflow. The current
advanced surface spans distributed execution, radiation-aware optimisation,
material fitting, nonlinear materials, and a research-grade example bundle.

## Multi-GPU Distributed FDTD

rfx can distribute the Yee update across multiple GPUs on a single host via
`jax.pmap`.

```python
import jax
from rfx import Simulation

sim = Simulation(freq_max=10e9, domain=(0.2, 0.04, 0.02), boundary="cpml")
result = sim.run(n_steps=2000, devices=jax.devices()[:2])
```

Current distributed support is strongest for slab-decomposed 3-D problems with
PEC/CPML boundaries, soft sources, probes, dispersive media, and lumped-port
workflows. Unsupported configurations are designed to fail clearly or fall back
conservatively.

## NTFF-Aware Far-Field Optimisation

The optimiser now supports objectives that consume raw NTFF accumulation data.
If your objective accepts an `ntff_box` keyword argument, `optimize()` will
build the far-field box and pass it into the loss.

```python
import jax.numpy as jnp
from rfx import Simulation, compute_far_field_jax

sim = Simulation(freq_max=10e9, domain=(0.12, 0.06, 0.04), boundary="cpml")
sim.add_ntff_box(
    corner_lo=(0.02, 0.01, 0.01),
    corner_hi=(0.10, 0.05, 0.03),
    freqs=[10e9],
)
grid = sim._build_grid()
theta = jnp.linspace(0.0, jnp.pi, 181)
phi = jnp.array([0.0])

def objective(result, ntff_box=None):
    ff = compute_far_field_jax(result.ntff_data, ntff_box, grid, theta, phi)
    broadside_power = jnp.abs(ff.E_theta[0, 90, 0]) ** 2 + jnp.abs(ff.E_phi[0, 90, 0]) ** 2
    return -broadside_power
```

Use this pattern for beam steering, broadside-gain maximisation, or null
placement objectives. In practice you usually capture the design grid in the
objective closure before calling `optimize()`.

## Time-Domain Proxy Objectives for Differentiable Loops

Inside `optimize()` and `topology_optimize()`, the most robust built-in losses
are the **time-domain proxy objectives**:

```python
from rfx import minimize_reflected_energy, maximize_transmitted_energy

obj_reflect = minimize_reflected_energy(port_probe_idx=0)
obj_transmit = maximize_transmitted_energy(output_probe_idx=-1)
```

These avoid assuming that a full post-processed S-parameter matrix is available
inside the JIT-traced forward pass.

## Dispersive Materials and Material Fitting

### Debye / Lorentz models

```python
from rfx import DebyePole, LorentzPole

sim.add_material("water", eps_r=5.0, debye_poles=[
    DebyePole(eps_s=80.0, tau=9.4e-12)
])

sim.add_material("resonant", lorentz_poles=[
    LorentzPole(eps_s=2.0, omega_0=2e10, delta=1e8, omega_p=3e10)
])
```

### Differentiable fitting from S-parameters

```python
from rfx import differentiable_material_fit

fit = differentiable_material_fit(...)
```

This is one of the most distinctive rfx workflows: use FDTD itself as the
forward model while fitting Debye/Lorentz poles to measured or synthetic
fixtures.

## Nonlinear Materials and RIS Workflows

rfx includes a **Kerr nonlinear material** path via ADE and a dedicated
**RIS unit-cell workflow** for programmable-surface experiments. These are
advanced research features rather than beginner-path tutorials, but they are
part of the current public capability surface.

## Non-Uniform Mesh, Lumped RLC, and Geometry Helpers

### Non-uniform z mesh

For thin substrates and layered RF structures, `rfx` supports graded z spacing:

```python
sim = Simulation(
    freq_max=4e9,
    domain=(0.08, 0.06, 0.0),
    dx=5e-4,
    dz_profile=dz_profile,
)
```

See [Non-Uniform Mesh](/rfx/guide/nonuniform-mesh/) for the recommended workflow.

### Lumped RLC elements

```python
sim.add_lumped_rlc(
    position=(0.02, 0.02, 0.01),
    component="ez",
    R=50.0,
    L=1e-9,
    C=1e-12,
    topology="series",
)
```

### Via and CurvedPatch helpers

```python
from rfx import Via, CurvedPatch
```

These expose convenient PCB / conformal-antenna primitives at the top level.

## Mixed Precision and Field Animation

```python
from rfx import save_field_animation

save_field_animation(result, "field_evolution.gif")
```

For memory-constrained runs, mixed-precision field storage is also available in
current stable releases.

## Advanced Example Bundle

The repo ships a research-style example bundle under `examples/50_advanced/`:

- patch bandwidth optimisation
- waveguide filter inverse design
- broadband matching
- array mutual coupling
- dielectric-lens beam shaping
- S-parameter-based material characterisation
- visualization showcase

The matching `examples/50_advanced/gpu_validation/` scripts provide a compact
regression layer for these workflows.
