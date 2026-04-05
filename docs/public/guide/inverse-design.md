---
title: "Inverse Design"
sidebar:
  order: 16
---

rfx is fully differentiable — `jax.grad` computes gradients through the entire
FDTD simulation, enabling gradient-based inverse design of RF structures.

## How it works

JAX traces the computation graph through all FDTD time steps.
`jax.checkpoint` reduces memory from `O(n_steps)` to `O(sqrt(n_steps))` by
recomputing forward states during backpropagation.

```text
Forward:  eps_r → FDTD steps → probes / NTFF / loss
Backward: jax.grad(loss)(eps_r) → gradient of eps_r
```

## Manual gradient loop

The most flexible approach is still a custom objective written directly against
`run()`:

```python
import jax
import jax.numpy as jnp
from rfx.grid import Grid
from rfx.core.yee import MaterialArrays
from rfx.simulation import run, make_source, make_probe, ProbeSpec
from rfx.sources.sources import GaussianPulse

grid = Grid(freq_max=8e9, domain=(0.04, 0.01, 0.01), dx=0.001, cpml_layers=6)
src = make_source(grid, (0.008, 0.005, 0.005), "ez", GaussianPulse(f0=4e9), n_steps=150)
probe = ProbeSpec(i=30, j=5, k=5, component="ez")

sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
mu_r = jnp.ones(grid.shape, dtype=jnp.float32)

def objective(eps_r):
    mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
    result = run(grid, mats, 150, sources=[src], probes=[probe],
                 boundary="pec", checkpoint=True)
    return -jnp.sum(result.time_series ** 2)  # maximize transmission
```

## Built-in objectives: choose the right family

### 1) Post-processed S-parameter objectives

These are convenient when you already have a completed simulation result with
S-parameters:

```python
from rfx import minimize_s11, maximize_s21, target_impedance, maximize_bandwidth

obj_s11 = minimize_s11(freqs=jnp.array([5e9]), target_db=-10)
obj_s21 = maximize_s21(freqs=jnp.linspace(4e9, 6e9, 20))
obj_z = target_impedance(freq=5e9, z_target=50.0)
obj_bw = maximize_bandwidth(f_center=5e9, f_bw=2e9, s11_threshold=-10)
```

### 2) Differentiable loop objectives for `optimize()` / `topology_optimize()`

Inside the traced forward pass, rfx does **not** build a full post-processed
S-parameter matrix. For gradient-based optimisation loops, prefer the proxy
losses below:

```python
from rfx import minimize_reflected_energy, maximize_transmitted_energy

obj_reflect = minimize_reflected_energy(port_probe_idx=0)
obj_transmit = maximize_transmitted_energy(output_probe_idx=-1)
```

These are the recommended defaults for reflection-minimisation and
throughput-maximisation tasks.

## Design-region API

```python
from rfx import Simulation, DesignRegion, optimize

sim = Simulation(freq_max=10e9, domain=(0.1, 0.04, 0.02), boundary="cpml")
sim.add_port(...)

region = DesignRegion(
    corner_lo=(0.03, 0.0, 0.0),
    corner_hi=(0.07, 0.04, 0.02),
    eps_range=(1.0, 6.0),
)

result = optimize(
    sim,
    region,
    objective=minimize_reflected_energy(port_probe_idx=0),
    n_iters=50,
    lr=0.01,
)
```

## Far-field objectives with NTFF data

A recent improvement makes `optimize()` NTFF-aware. If your objective accepts
`ntff_box=...`, the optimiser will build the far-field box and pass it in.

```python
import jax.numpy as jnp
from rfx import compute_far_field_jax

grid = sim._build_grid()  # advanced usage: capture once outside the objective
theta = jnp.linspace(0.0, jnp.pi, 181)
phi = jnp.array([0.0])

def objective(result, ntff_box=None):
    ff = compute_far_field_jax(result.ntff_data, ntff_box, grid, theta, phi)
    broadside = jnp.abs(ff.E_theta[0, 90, 0]) ** 2 + jnp.abs(ff.E_phi[0, 90, 0]) ** 2
    return -broadside
```

This enables beam shaping, broadside maximisation, and other radiation-aware
advanced objectives.

## Tips

- **Always use `checkpoint=True`** in custom loops — it saves large amounts of memory.
- **Start with small grids** for design iteration, then scale up for the final verification run.
- **Learning rate**: `0.01–0.1` is a good first range for permittivity optimisation.
- **Proxy objectives first**: when in doubt, start with `minimize_reflected_energy()` or `maximize_transmitted_energy()`.
- **Use NTFF objectives selectively** — they are powerful, but more expensive than probe-only losses.
- **GPU acceleration** is automatic when JAX sees CUDA devices.
