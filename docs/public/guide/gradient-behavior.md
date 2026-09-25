---
title: "Gradient Behavior in rfx"
description: "Where rfx gradients are reliable, where they are noisy or meaningless, and three checks to run before you optimize: finite differences, loss magnitude, and record length."
sidebar:
  order: 14
---

An rfx gradient is the exact derivative of the discrete simulation you set up.
Whether it is *useful* depends on the design variable, the objective and the
run length. This page lists the cases that behave well, the ones that don't, and
three checks to run before you trust a gradient. For the background, see
[Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/).

## Three checks before you optimize

The examples share one small model: a lossy dielectric slab between a soft
source and a probe in open space, on a coarse 2 mm mesh so each check runs in
seconds.

```python
import jax
import jax.numpy as jnp
from rfx import Box, GaussianPulse, Simulation, gradient_record_length_witness

sim = Simulation(freq_max=4e9, domain=(0.06, 0.05, 0.04), dx=2e-3,
                 boundary="cpml", cpml_layers=6)
sim.add_material("slab", eps_r=4.0, sigma=0.01)
sim.add(Box((0.024, 0.019, 0.014), (0.036, 0.031, 0.026)), material="slab")
sim.add_source((0.018, 0.025, 0.020), "ez",
               waveform=GaussianPulse(f0=3e9, bandwidth=0.8),
               amplitude_kind="current")
sim.add_probe((0.042, 0.025, 0.020), "ez")

grid = sim.run(n_steps=1).grid
eps0 = jnp.full(grid.shape, 4.0, dtype=jnp.float32)
slab_cell = grid.position_to_index((0.030, 0.025, 0.020))


def objective(eps_r, n_steps=200):
    out = sim.forward(eps_override=eps_r, n_steps=n_steps)
    return jnp.sum(out.time_series ** 2)
```

### 1. Autodiff against finite differences

Compare the autodiff gradient with a central difference at a few cells:

```python
def fd_check(objective, eps_r, cell, h=1e-1):
    """Relative error between AD and a central finite difference at one cell."""
    fd = (objective(eps_r.at[cell].add(h))
          - objective(eps_r.at[cell].add(-h))) / (2 * h)
    ad = jax.grad(objective)(eps_r)[cell]
    rel_err = abs(ad - fd) / max(abs(fd), 1e-30)
    print(f"cell {cell}: AD={float(ad):.4e}  FD={float(fd):.4e}  "
          f"rel err={float(rel_err):.2e}")
    return rel_err


cell = tuple(int(i) for i in slab_cell)
fd_check(objective, eps0, cell, h=1e-2)
fd_check(objective, eps0, cell, h=1e-1)
```

rfx runs in float32 by default, which resolves a loss to about `1e-7` of its
value. Choose `h` so that the loss change, about `2 h |dL/dp|`, is well above
that. Here the loss is large and `h = 1e-2` leaves an error of a few percent,
while `h = 1e-1` agrees to about 0.2 %. For permittivity-like variables, start
between `1e-2` and `1e-1`. If the check still fails, look at the loss scale, the
run length and the monitor placement before touching optimizer settings.

### 2. The loss magnitude

A passing finite-difference *witness is also not sufficient*. Finite
differences and autodiff differentiate the same loss over the same observation
window. If the window is empty (the reflection never reaches the probe during
the run, or the monitor sits in the absorber), both agree on a gradient of
numerical noise.

So compare the loss **magnitude** with a physical expectation, not only the
relative error. A reflected-energy proxy on a design that reflects a meaningful
fraction of the pulse lands near `1e-2` to `1e-1`. A value like `~1e-7`
signals an empty window, not a matched design. See the split-window rule in
[Inverse Design](/rfx/guide/inverse-design/).

### 3. The record length

A resonance that is still ringing when the record ends barely changes the loss
value, but it can change the gradient a lot. `gradient_record_length_witness`
differentiates your objective at `n_steps` and at `factor * n_steps` (default
2x) and reports how much the gradient vector moved:

```python
for n in (100, 200):
    w = gradient_record_length_witness(objective, eps0, n, tol=0.05)
    print(f"n_steps={n}: passed={w.passed}, "
          f"gradient change={w.worst:.3f}, value change={w.worst_value_rel_change:.3f}")
```

Here 100 steps fails, because the pulse has not finished passing the probe, and
200 steps passes.

The objective must take `(params, n_steps)`. You choose `tol`; there is no
default because the right bar depends on the structure's Q and on what the
gradient is for. A few percent is a reasonable start. `w.cosine_by_bin` tells
you whether the direction changed or only the step length. A failed witness
means the record is too short for this gradient: lengthen the run and check
again. The witness costs about `1 + factor` differentiated runs, so use it
once per new setup, not every iteration. Pair it with the settling check in
[Probes and S-Parameters](/rfx/guide/probes-sparams/); it does not replace it.

## What behaves well

- **Smooth dielectric variables.** Continuous `eps_r` in a fixed design region
  is the safest case. Use a bounded parameterization so the optimizer stays in
  the material range you mean:

  ```text
  eps_design = eps_min + (eps_max - eps_min) * jax.nn.sigmoid(latent)
  ```

  `optimize()` does this for you through `DesignRegion(eps_range=...)`.
- **Band-averaged objectives.** A mean over a band, such as
  `jnp.mean(jnp.abs(s11) ** 2)`, is less dominated by a single noisy frequency
  point than one sample.
- **The differentiable port paths.** Lumped and wire ports through
  `forward(port_s11_freqs=...)`, lumped R/L/C values through
  `forward(rlc_values_override=...)`, and the waveguide and microstrip
  S-matrix calculators where they accept `eps_override`. Each is
  differentiable only within its own support entry; see the
  [S-parameter support matrix](https://github.com/bk-squared/rfx/blob/main/docs/guides/sparameter_support_matrix.md).

## What is noisy or stiff

| Case | Symptom | What to do |
|---|---|---|
| A PEC edge or topology boundary that moves | stair-stepping: the gradient jumps or flips sign when an edge crosses a cell | keep topology fixed in a gradient run; use relaxed occupancy or dielectric variables with filters and a minimum feature size; verify the final discrete geometry |
| Long records | weak signal, large dynamic range, round-off | use the shortest run that still captures the observable, and run the record-length witness |
| Near cutoff or a high-Q resonance | one frequency sample is very sensitive to mesh, material and run length | keep the band away from cutoff unless that is the target; use broadband objectives; verify with a convergence study |
| Float32 finite differences | FD disagrees with AD by a few percent | use a larger `h` (see above) |

**Precision.** The default is `precision="float32"` (complex64 DFT buffers).
The uniform single-device lane also accepts `precision="float64"` once JAX x64
is enabled (enabling x64 alone does not switch the fields to float64), and
`precision="mixed"` (float16 field storage, float32 accumulators). Mixed
precision with CPML has a higher absorber residual, so do not use it for
low-reflection or S-parameter observables near that floor. The non-uniform,
distributed and subgridded lanes refuse both non-default modes.

## What is not a design variable

- **CPML cells.** The absorber is artificial. A gradient there says nothing
  about a device. Keep design regions out of the CPML.
- **Integer and topology choices.** Grid size, CPML layer count, number of
  steps, inserting or removing a shape. Fix them per run, or search over them
  in an outer loop of separate differentiable problems.
- **Subpixel smoothing.** It is a `run()` option and is not part of the
  differentiable `forward()` path, so gradients see the staircased
  permittivity. Don't use an interface position as a design variable and
  expect subpixel-accurate sensitivity.
- **Unsupported combinations.** A finite gradient from a source, port or mesh
  combination outside the support matrix is not evidence of anything. rfx
  raises on the combinations it knows are unsupported.

## Limits

- **AD and FD agree but the loss is tiny**: the observation window is empty.
  Fix the run length or monitor placement.
- **The gradient changes a lot when you double the run**: the record is too
  short. Lengthen it until the record-length witness passes.
- **The gradient flips sign as a metal edge moves**: stair-stepping. Use relaxed
  or dielectric variables and verify the final geometry.
- **A gradient-optimized proxy is not a validated result**: re-run the final
  design through the port, resonance or far-field workflow on a converged mesh.
