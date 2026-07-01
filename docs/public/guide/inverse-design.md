---
title: "Inverse Design"
sidebar:
  order: 16
---

rfx exposes JAX-differentiable FDTD workflows: `jax.grad` computes reverse-mode
gradients of a scalar loss through the implemented discrete time-domain
calculation. If you are coming from Meep's adjoint terminology, start with
[Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/).

## How it works

JAX traces the supported solver and objective path as a computation graph.
`jax.checkpoint` reduces reverse-mode memory by recomputing forward states
during backpropagation.

```text
Forward:  eps_r → FDTD steps → probes / NTFF / loss
Backward: jax.grad(loss)(eps_r) → gradient of eps_r
```

## Manual gradient loop

For custom objectives, keep the loop on the high-level differentiable API. Build
the `Simulation` once, create an `eps_override` array with the configured grid
shape, and differentiate a scalar loss from `Simulation.forward(...)`:

```python
import jax
import jax.numpy as jnp

# sim is an already configured Simulation with sources/probes/ports.
# eps0 has the same shape as the simulation grid's permittivity array.
def objective(eps_r):
    result = sim.forward(eps_override=eps_r, n_steps=150, checkpoint=True)
    return -jnp.sum(result.time_series ** 2)  # example proxy loss

grad = jax.grad(objective)(eps0)
```

This computes the gradient of the implemented discrete proxy objective. It does
not by itself validate the final RF observable; run the relevant port, resonance,
far-field, or convergence check after optimization.

## Built-in objectives: choose the right family

### 1) Post-processed S-parameter objectives

These are convenient when you already have a completed simulation result with
S-parameters. They do not upgrade the physics claim of that result; use the
port-family evidence envelope before treating the objective as validated.

```python
from rfx import minimize_s11, maximize_s21, target_impedance, maximize_bandwidth

obj_s11 = minimize_s11(freqs=jnp.array([5e9]), target_db=-10)
obj_s21 = maximize_s21(freqs=jnp.linspace(4e9, 6e9, 20))
obj_z = target_impedance(freq=5e9, z_target=50.0)
obj_bw = maximize_bandwidth(f_center=5e9, f_bw=2e9, s11_threshold=-10)
```

### 2) Differentiable loop objectives for `optimize()`

Inside the traced forward pass, rfx does **not** build a full post-processed
S-parameter matrix for every port family. For gradient-based optimization loops,
prefer the proxy losses below unless the selected calculator explicitly documents
its differentiable S-parameter path:

```python
from rfx import minimize_reflected_energy, maximize_transmitted_energy

obj_reflect = minimize_reflected_energy(port_probe_idx=0)
obj_transmit = maximize_transmitted_energy(output_probe_idx=-1)
```

These are the recommended defaults for reflection-minimization and
throughput-maximization tasks.

For NTFF/directivity optimization, prefer
`maximize_directivity(..., log_ratio=True)` when the design variable can change
total radiated power; this keeps the directivity-gradient sign consistent with
the full ratio objective.

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

`optimize()` can pass NTFF data to objectives that explicitly accept
`ntff_box=...`. Use this only when the NTFF setup and final far-field validation
path are documented for the workflow.

```python
from rfx import maximize_directivity

objective = maximize_directivity(
    theta_target=0.0,
    phi_target=0.0,
    log_ratio=True,
)
```

This supports target-direction directivity and other radiation-aware objectives
when the simulation has a documented NTFF setup and the final design is re-run
through the far-field validation path.

## Tips

- **Use `checkpoint=True` or the documented segmented checkpoint knob** in custom loops when reverse-mode memory is the limiting factor.
- **Start with small grids** for design iteration, then scale up for the final verification run.
- **Learning rate**: `0.01–0.1` is a good first range for permittivity optimization.
- **Proxy objectives first**: when in doubt, start with `minimize_reflected_energy()` or `maximize_transmitted_energy()`.
- **Use NTFF objectives selectively** — they are powerful, but more expensive than probe-only losses.
- **GPU acceleration** depends on the installed JAX/CUDA environment; verify device placement for performance-sensitive runs.
