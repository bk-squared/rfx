---
title: "Gradient Behavior in rfx"
sidebar:
  order: 14
---

rfx gradient workflows use JAX reverse-mode automatic differentiation through
the implemented discrete FDTD calculation. This guide documents where gradients
are usually well behaved, where they are numerically stiff or noisy, and how to
check them before using them for RF design decisions. For the conceptual
background, see [Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/).

## How it works

JAX traces the supported FDTD time-stepping and objective path as a computation
graph. Per-step `jax.checkpoint` (the `checkpoint=True` default) or the
segmented checkpoint knobs — `checkpoint_segments` on uniform grids,
`checkpoint_every` on non-uniform grids — reduce reverse-mode memory by
recomputing forward states during backpropagation. See the
[Memory-Reduction Path](/rfx/guide/memory-reduction/) for how to size and pick
these knobs.

```python
import jax
import jax.numpy as jnp

# sim is an already configured Simulation with sources/probes/ports.
# Discover the permittivity-array shape from a quick run, then build the
# starting array (result.grid is populated after every run() call):
eps0 = jnp.ones(sim.run(n_steps=1).grid.shape, dtype=jnp.float32)

def objective(eps_r):
    result = sim.forward(eps_override=eps_r, n_steps=400, checkpoint=True)
    return jnp.sum(result.time_series ** 2)

grad = jax.grad(objective)(eps0)  # discrete reverse-mode AD gradient
```

The gradient is a sensitivity of this discrete objective. It is not, by itself,
a calibrated S-parameter, far-field, or resonance validation result.

## Where gradients are usually well behaved

### Smooth dielectric variables

Continuous changes in `eps_r` inside a fixed design region are the safest first
case. Use bounded parameterizations, such as a sigmoid or projection from latent
variables, so the optimizer cannot leave the material range you intend to test.

```python
# Example bounded dielectric parameterization.
eps_design = eps_min + (eps_max - eps_min) * jax.nn.sigmoid(latent)
```

### Broadband or averaged objectives

Objectives averaged over a band are often less sensitive to a single noisy
frequency point:

```python
loss = jnp.mean(jnp.abs(s11) ** 2)
```

This does not remove the need for a final port-family validation run. It only
makes the optimization objective less dominated by one sample.

### Documented differentiable port paths

Some port paths carry a differentiable |S| channel: the lumped/wire path via
`forward(port_s11_freqs=...)`, and the `compute_waveguide_s_matrix` /
`compute_msl_s_matrix` calculators via their `eps_override` argument. Each is
differentiable only where its own support entry says so. Treat the AD path as a
contract for the sensitivity calculation; the physical claim still follows the
port-family evidence envelope. See [Inverse Design](/rfx/guide/inverse-design/)
for end-to-end optimization loops built on these paths.

## Where gradients are noisy or problematic

### Moving PEC and topology boundaries

Stairstepping creates discontinuities when a geometry parameter moves a PEC edge
across a Yee cell boundary. The gradient can be large, change sign abruptly, or
represent the rasterized discretization more than the intended CAD motion.

**Mitigation:** keep topology fixed during a gradient run, use smooth dielectric
or relaxed occupancy variables where the workflow documents them, apply filters
or minimum-feature constraints, and re-run a final discrete geometry validation.
Do not rely on a larger learning rate to repair a discontinuous objective.

### Long time windows

Long simulations can have weak gradient signal, large dynamic range, or more
roundoff sensitivity. Checkpointing preserves the mathematical reverse-mode
program for the chosen computation, but recomputation still occurs in finite
precision.

**Mitigation:** use the minimum run length that captures the observable, record
the time-window choice, and compare gradients against finite differences on a
small set of cells or scalar parameters.

### Near-cutoff and high-Q cases

Near waveguide cutoff, beta changes rapidly with frequency and geometry. High-Q
resonances can also make a single frequency sample highly sensitive to small
mesh, material, or run-length changes.

**Mitigation:** keep optimization bands away from cutoffs unless that is the
actual design target; use broadband or mode-tracking objectives; and verify the
final design with a convergence or cross-reference check.

### Float32 finite-difference checks

rfx runs these workflows in float32 by default (complex64 field and DFT
buffers), unless you enable JAX 64-bit precision. If the finite-difference step
is too small, cancellation can make the finite-difference witness look worse
than the AD path.

**Mitigation:** start finite-difference checks with a step that is meaningful for
the variable scale, often around `h = 1e-2` for permittivity-like variables, then
adjust based on the loss magnitude and local sensitivity.

## What is not differentiable

### CPML absorber cells

Gradients with respect to physical material variables inside CPML are not useful
for RF design. CPML is an artificial absorber, not a device material. Exclude
CPML cells from design regions.

### Integer and topology choices

Grid dimensions, CPML layer count, timestep count, and shape insertion/removal
are discrete choices. Use them as fixed setup parameters, or run an outer design
search that launches separate differentiable problems with fixed topology.

### Subpixel smoothing

Subpixel averaging of material properties at geometry boundaries is precomputed
once at setup and is not part of the JAX computation graph. Gradients do not
flow through the subpixel weights — the material boundary is treated as fixed
during reverse-mode differentiation. Exclude geometry boundary positions from
design variables when subpixel smoothing is active.

### Unsupported physics combinations

If a source, port, mesh, or monitor combination is outside the documented support
scope, finite AD output is not validation evidence. The correct outcome for an
unsupported combination is an explicit support error or a clearly scoped local
engineering check, not a public claim.

## Validating gradients

Always check a new objective against finite differences at a few representative
cells or scalar design parameters before trusting a full optimization run:

```python
def fd_check(objective, eps_r, cell=(10, 5, 5), h=1e-2):
    """Finite-difference gradient witness for one cell."""
    eps_p = eps_r.at[cell].add(h)
    eps_m = eps_r.at[cell].add(-h)
    fd = (objective(eps_p) - objective(eps_m)) / (2 * h)
    ad = jax.grad(objective)(eps_r)[cell]
    rel_err = abs(ad - fd) / max(abs(fd), 1e-30)
    print(f"AD={ad:.6e}, FD={fd:.6e}, err={rel_err:.2%}")
    return rel_err
```

Use the result as a witness, not as a universal tolerance. If the witness fails,
inspect the objective scale, perturbation size, run length, monitor placement,
and support envelope before changing optimizer settings.

A *passing* witness is also not sufficient on its own. Finite differences and AD
differentiate the same objective through the same observation window, so both
agree even when that window is empty — an objective that never sees the physics it
is supposed to (reflection that never reaches the probe, a monitor inside an
absorber) produces a self-consistent gradient of numerical noise. Guard against
this by checking the loss **magnitude** against a physical expectation, not only
the AD-vs-FD relative error: a reflected-energy proxy on a meaningfully reflecting
design lands near `1e-2`–`1e-1`, so a value like `~1e-7` signals an empty window,
not a matched design.

## Best practices

1. **Start with a small, supported setup** before scaling the grid or objective.
2. **Use bounded continuous variables** for dielectric or occupancy design.
3. **Exclude CPML and fixed metal from the design region** unless the guide for
   that workflow explicitly documents a relaxed variable.
4. **Check AD against finite differences** on a small set of cells or scalar
   parameters.
5. **Prefer broadband or averaged losses** when a single frequency point is
   noisy or near a resonance null.
6. **Use the segmented checkpoint knobs** (`checkpoint_segments` on uniform
   grids, `checkpoint_every` on non-uniform grids) when reverse-mode memory is
   the limiting factor.
7. **Validate the final RF observable** through the relevant port, resonance,
   far-field, or convergence workflow.

## Gradient-path framing

| Path | Gradient-path framing | RF evidence reminder |
|---|---|---|
| Yee E/H update | differentiable inside supported runners | validate the observable, not just the tape |
| CPML absorber | may be on the AD tape | exclude from physical design variables |
| Lumped/wire `forward(port_s11_freqs=...)` | differentiable S11-vector path on the uniform single-device runner | inherits the lumped/wire support envelope |
| MSL or waveguide S-matrix AD paths | differentiable only where the calculator documents `eps_override` support | use the matching calculator and support entry |
| DFT probes / time series | useful proxy-objective signals | not an impedance-defined port by themselves |
| Dispersive or lossy materials | case-dependent AD path | verify with finite differences and physics evidence |
| S-parameter post-processing helpers | reporting/objective utilities | do not promote an undocumented port workflow |
