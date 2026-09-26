---
title: "Inverse Design"
description: "Gradient-based design through the FDTD solver: the optimize() design-region driver, which objectives run inside the gradient loop, the manual jax.grad loop, and lumped-component value design."
sidebar:
  order: 16
---

rfx is differentiable end to end: `jax.grad` returns the gradient of a scalar
loss through every FDTD time step. This page shows how to optimize a
permittivity region with `optimize()`, which objectives work inside the
gradient loop, and how to write your own loop with `Simulation.forward()`. For
the background (and how this relates to adjoint solvers), see
[Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/).

| You want to | Use |
|---|---|
| optimize the permittivity of a box | `optimize(sim, DesignRegion(...), objective)` |
| optimize a metal/dielectric layout with filtering and projection | `topology_optimize(sim, TopologyDesignRegion(...), objective)` |
| a custom loss, several kinds of variable, or your own optimizer | `jax.grad` over `sim.forward(...)` |
| tune a lumped R, L or C value | `forward(rlc_values_override=...)` |

The examples use coarse 2 mm meshes and a few hundred steps so the page runs in
under a minute on a CPU. A real design uses a mesh that resolves the geometry,
runs long enough to capture the response you optimize, and ends with a
verification run of the final design.

## Optimize a design region

`optimize()` runs an Adam loop over the permittivity inside a `DesignRegion`.
The region is a box in metres plus an `eps_range`; the optimizer works on an
unbounded latent variable that a sigmoid maps into that range.

This toy model puts a dielectric region between a source and a probe and
maximizes the energy that reaches the probe:

```python
import jax
import jax.numpy as jnp
from rfx import (DesignRegion, GaussianPulse, Simulation,
                 maximize_transmitted_energy, optimize)

sim = Simulation(freq_max=10e9, domain=(0.06, 0.03, 0.03), dx=2e-3,
                 boundary="cpml", cpml_layers=6)
sim.add_source((0.012, 0.015, 0.015), "ez",
               waveform=GaussianPulse(f0=6e9, bandwidth=0.8),
               amplitude_kind="current")
sim.add_probe((0.048, 0.015, 0.015), "ez")   # time_series column 0

region = DesignRegion(corner_lo=(0.020, 0.008, 0.008),
                      corner_hi=(0.040, 0.022, 0.022),
                      eps_range=(1.0, 6.0))
objective = maximize_transmitted_energy(output_probe_idx=0)

result = optimize(sim, region, objective,
                  n_iters=5,        # a real design runs tens to hundreds
                  lr=0.5, n_steps=250, jit=True, verbose=False)

print("loss per iteration:", [f"{v:.3e}" for v in result.loss_history])
print("eps range in the design:",
      float(result.eps_design.min()), float(result.eps_design.max()))
```

`result` is an `OptimizeResult` with `eps_design` (the permittivity in the
region's cells), `loss_history`, and the final `latent`. Pass `latent` back as
`init_latent=` to continue a run.

Keep the region inside the declared `domain`. Any part that reaches into the
CPML padding outside it is clipped off, and `optimize()` raises `ValueError` if
nothing is left.

### Useful `optimize()` options

| Option | What it does |
|---|---|
| `n_steps` / `num_periods` | record length per iteration (default 20 periods at `freq_max`). Fewer steps use less memory, but the run must still contain the response you optimize. |
| `jit=True` | compile the loss and gradient once and reuse them in every iteration. Much faster on real models. See the caveats below. |
| `checkpoint_segments` | segmented checkpointing on a uniform mesh, for about `sqrt(n_steps)` memory at about 2x compute. Must divide `n_steps`. See [Memory Reduction](/rfx/guide/memory-reduction/). |
| `checkpoint_every` | the non-uniform-mesh counterpart (a chunk size in steps). |
| `n_starts=N`, `seed` | `N` restarts from random latents; returns the best one. Helps only on a multimodal loss. |
| `best_iterate=True` | return the lowest-loss iterate visited, not the last one. |
| `step_clamp=value` | cap the L2 norm of each Adam step, to stop overshoot past a sharp minimum. |
| `port_s11_freqs` | accumulate port S11 inside the loop, for the exact S11 objective below. |

With `jit=True` the objective is traced, not run. It must not read a traced
value on the host, for example `float(x)`, `np.asarray(x)` or a Python `if` on
an array. Two models still do not trace and stop with a
`TracerArrayConversionError` or `ConcretizationTypeError`; use the default
`jit=False` for them:

- a model with a microstrip (MSL) port;
- a model whose mesh is itself a design variable (a traced `dz_profile`) that
  has a wire port together with a conductor.

The jitted loss and gradient can differ from the eager ones in the last float32
bits, so a jitted optimization path can drift slightly from an eager one.

## Choose an objective

Objectives are functions `objective(result) -> scalar`, minimized by the
optimizer. There are two families, and only one of them works inside the
gradient loop.

### Objectives that work inside `optimize()`

`forward()` returns probe time series, NTFF data and, on request, port S11. It
does not build the full post-processed S-parameter matrix. These objectives
read only what `forward()` provides:

| Objective | Reads | Meaning |
|---|---|---|
| `minimize_reflected_energy(port_probe_idx=0, late_fraction=0.5)` | probe time series | late-time energy / early-time energy at the port probe (an S11 proxy) |
| `maximize_transmitted_energy(output_probe_idx=-1)` | probe time series | negated energy at an output probe (an S21 proxy) |
| `minimize_s11_at_freq_wave_decomp(target_freq, port_idx=0)` | `forward(port_s11_freqs=...)` | exact \|S11\|² of a lumped port from its V/I waves |
| `maximize_directivity(theta_target, phi_target)` | NTFF box data | directivity toward one direction |

`minimize_s11_at_freq_wave_decomp` lives in `rfx.optimize_objectives`. Pass
`port_s11_freqs=` to `optimize()` so the port accumulates the S11 it reads.
Because it separates incident and reflected waves exactly, it has no timing
precondition; prefer it over the reflected-energy proxy when your model has a
lumped port.

**The split window must contain the reflection.** `minimize_reflected_energy`
splits the port probe's time series at `late_fraction` (by default, at
half-way) and treats the late part as reflected energy. That only works if the
round trip from the port to the reflecting feature and back arrives *after* the
split and *before* the run ends. On a short round trip (a thin substrate, a
feature close to the port) the reflection lands in the early window, the late
window is almost empty, and the loss collapses toward zero. Its gradient is then
numerical noise, and nothing raises an error.

Check the loss value at the starting design before you optimize. A design that
reflects a meaningful fraction of the pulse gives a loss around `1e-2` to
`1e-1`; a loss near `1e-7` means the window is empty. Then fix it by cause:

- the reflection arrives after the run ends: increase `n_steps`;
- it arrives before the split: raise `late_fraction` (this moves the split
  earlier);
- the incident pulse and the reflection overlap in time: no split separates
  them. Narrow the source bandwidth, or use a port with
  `forward(port_s11_freqs=...)` and `minimize_s11_at_freq_wave_decomp`.

A finite-difference check of the gradient does not catch an empty window,
because both methods differentiate the same window and agree (see
[A passing finite-difference check is necessary, not sufficient](/rfx/guide/autodiff-adjoint/#a-passing-finite-difference-check-is-necessary-not-sufficient)).

**Directivity.** `maximize_directivity` reads `result.ntff_data`, so register a
box with `sim.add_ntff_box(corner_lo, corner_hi)` first; without it the
objective raises `ValueError`. Angles are in radians. Keep the default
`log_ratio=True`: it gives the correct gradient sign for every design variable,
including ones that change the total radiated power (conductors, loss,
large permittivity changes). `log_ratio=False` is correct only for variables
that leave the radiated power unchanged. NTFF objectives cost more than probe
objectives, so iterate on a coarse mesh and check the final design with
[Far-Field and RCS](/rfx/guide/farfield-rcs/).

### Objectives for a finished `run()`

These read `result.s_params` from `run(compute_s_params=True)`. They are for
scoring a design after a full run, not for the gradient loop: inside
`optimize()` or `forward()` they raise `ValueError`.

```python
from rfx import maximize_bandwidth, maximize_s21, minimize_s11, target_impedance

obj_s11 = minimize_s11(freqs=jnp.array([5e9]), target_db=-10)
obj_s21 = maximize_s21(freqs=jnp.linspace(4e9, 6e9, 20))
obj_z = target_impedance(freq=5e9, z_target=50.0)
obj_bw = maximize_bandwidth(f_center=5e9, f_bw=2e9, s11_threshold=-10)
```

An objective is only as good as the S-parameter extraction behind it. Check the
port's status in the
[S-parameter support matrix](https://github.com/bk-squared/rfx/blob/main/docs/guides/sparameter_support_matrix.md)
before you treat an optimized number as a physical result.

## Write your own gradient loop

For a custom loss, call `sim.forward(...)` inside a function and differentiate
it. Build the `Simulation` once, outside the loss. `eps_override` replaces the
whole permittivity grid, so its shape is the grid shape:

```python
sim2 = Simulation(freq_max=10e9, domain=(0.03, 0.02, 0.02), dx=2e-3,
                  boundary="cpml", cpml_layers=6)
sim2.add_source((0.008, 0.010, 0.010), "ez",
                waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                amplitude_kind="current")
sim2.add_probe((0.022, 0.010, 0.010), "ez")

grid = sim2.run(n_steps=1).grid        # a one-step run gives you the grid
eps0 = jnp.ones(grid.shape, dtype=jnp.float32)


def loss(eps_r):
    out = sim2.forward(eps_override=eps_r, n_steps=150)
    return -jnp.sum(out.time_series ** 2)


grad = jax.jit(jax.grad(loss))(eps0)
print("gradient shape:", grad.shape)
```

You can wrap the loss or its gradient in `jax.jit`. Keep anything that changes
array shapes (the mesh, the number of steps, which ports exist) outside the
jitted function, and pass continuous arrays or scalars in as arguments.

### Design-box overrides (less memory)

A whole-grid `eps_override` puts grid-sized arrays on the autodiff tape at every
step. If only a box is a design variable, pass the box and its values instead.
The gradient is the same, and the tape stores box-sized arrays. Pass
`checkpoint=False` with it, because per-step checkpointing saves the full
field state every step regardless.

The array must match the number of cells the box realizes on the grid (both
corners resolve to their nearest cell and both ends are included):

```python
box = ((0.012, 0.006, 0.006), (0.018, 0.014, 0.014))
lo, hi = grid.position_to_index(box[0]), grid.position_to_index(box[1])
box_shape = tuple(int(b) - int(a) + 1 for a, b in zip(lo, hi))


def box_loss(eps_box):
    out = sim2.forward(design_box=box, design_eps_override=eps_box,
                       n_steps=150, checkpoint=False)
    return -jnp.sum(out.time_series ** 2)


g_box = jax.grad(box_loss)(jnp.full(box_shape, 2.0))
print("box gradient shape:", g_box.shape)
```

`design_box` also accepts a `DesignRegion`. The box must not reach into the
CPML or contain a source, port, lumped element or surface-impedance sheet, and
it does not combine with dispersive (Debye/Lorentz) or Kerr materials,
subpixel smoothing, UPML, `stencil_order=4`, or the other whole-grid overrides.
Those cases raise an error; use `eps_override` for them.

**A design conductivity** goes in `design_sigma_override`, in one of two forms:

- A single box-shaped array is **per cell**. rfx averages the four cells around
  each edge, as it does everywhere on the grid, so the design value also reaches
  the layer of edges on the box's plus faces.
- A 3-tuple `(sigma_x, sigma_y, sigma_z)` is **per edge**: each array is written
  as-is to that component's edges at the box indices, with no averaging. This is
  what a conducting sheet needs, since its current flows only along its two
  in-plane edges. A box-shaped per-edge array cannot reach the plus-face edge
  layer. If your per-edge design needs that layer, declare the box one cell
  larger on that side.

For metal-shape design, `design_occupancy_override` does the same for a relaxed
PEC occupancy in `[0, 1]` on a uniform mesh.

## Tune a lumped R, L or C value

`add_lumped_rlc(...)` adds a circuit element to the FDTD update. It is not a
port and produces no S-parameters. To measure the load, add a port with
`add_port(..., impedance=Z0)` and read S11 from `forward(port_s11_freqs=...)`.
Scalar component values enter the gradient through `rlc_values_override`, keyed
by the 0-based order of the `add_lumped_rlc` calls. Keys you leave out keep the
registered value. This works on the uniform, single-device `forward()` path.

**Put the element one cell away from the port, not on the port cell.** A port
reads S11 from the voltage and current at its own cell. An element inside that
cell is in parallel with the source, not in the network the port measures, so
S11 barely sees it and its gradient is numerical noise.

```python
dx = 1.5e-3
sim3 = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=dx,
                  boundary="cpml", cpml_layers=6)
port_pos = (0.0093, 0.0093, 0.0093)
load_pos = (port_pos[0] + dx, port_pos[1], port_pos[2])   # one cell away
sim3.add_port(position=port_pos, component="ez", impedance=50.0)
sim3.add_lumped_rlc(position=load_pos, component="ez",
                    R=200.0, C=0.2e-12, topology="series")


def load_loss(R):
    out = sim3.forward(n_steps=800, port_s11_freqs=jnp.array([5e9]),
                       rlc_values_override={0: {"R": R}})
    return jnp.abs(out.s_params[0]) ** 2      # |S11|^2 at 5 GHz


R0 = 200.0
value, dloss_dR = jax.value_and_grad(load_loss)(R0)
assert jnp.isfinite(value) and jnp.isfinite(dloss_dR)

fd = (load_loss(R0 + 10.0) - load_loss(R0 - 10.0)) / 20.0
print(f"|S11|^2 = {float(value):.4f}, AD dL/dR = {float(dloss_dR):.3e}, "
      f"FD dL/dR = {float(fd):.3e}")
```

The gradient is taken at the value the element was registered with. Check that
the loss and gradient are finite at every value you evaluate, as the `assert`
does, and keep an optimizer's R inside a physical range (for example, optimize
a bounded latent variable).

## Limits

- **Loss near zero with a clean finite-difference check**: the observation
  window is probably empty. See the split-window rule above.
- **A proxy loss improved but the real metric did not**: re-run the final design
  through the calibrated workflow (port S-parameters, resonance extraction or
  far field) on a converged mesh. See [Validation](/rfx/guide/validation/).
- **`ValueError` from `minimize_s11` / `maximize_s21` inside `optimize()`**:
  those read post-processed S-parameters. Use a time-domain proxy or
  `minimize_s11_at_freq_wave_decomp` in the loop.
- **Mesh refinement**: `add_refinement` acts only through `run()`.
  `forward()`, `optimize()` and `topology_optimize()` refuse a refined model.
- **`jit=True` fails with a tracer error**: an MSL port, or a traced mesh with a
  wire port and a conductor. Use `jit=False`.
- **Out of memory in the backward pass**: use a design box, segmented
  checkpointing or fewer steps. See [Memory Reduction](/rfx/guide/memory-reduction/).
