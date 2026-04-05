---
title: "Gradient Behavior in rfx"
sidebar:
  order: 14
---

rfx guarantees that `jax.grad` flows correctly through the entire FDTD
simulation. This guide documents where gradients are reliable, where they
are noisy, and best practices for using them.

## How It Works

JAX traces the entire FDTD time-stepping loop as a computation graph.
`jax.checkpoint` (rematerialization) reduces memory from O(n_steps) to
O(sqrt(n_steps)) by recomputing forward states during backpropagation.

```python
import jax
from rfx.simulation import run

def objective(eps_r):
    result = run(grid, materials(eps_r), n_steps,
                 sources=[src], probes=[probe],
                 checkpoint=True)
    return jnp.sum(result.time_series ** 2)

grad = jax.grad(objective)(eps_r)  # exact gradient via reverse-mode AD
```

## Where Gradients Work Well

### Smooth dielectric variations
Continuous changes in eps_r produce smooth objective landscapes. Gradient
descent converges reliably in 10-30 iterations.

```python
# Optimizing eps_r in a design region: well-behaved
eps_r = eps_r.at[design_region].set(sigmoid(latent) * 5 + 1)
```

### Lumped-port S-parameter objectives
|S11|², |S21|², impedance matching — all produce smooth, differentiable
objectives when computed from DFT probes.

### Broadband objectives
Objectives averaged over multiple frequencies smooth out individual
frequency-point noise:
```python
loss = jnp.mean(jnp.abs(s11) ** 2)  # smoother than single-frequency
```

### CW steady-state
CWSource with adequate ramp_steps produces stable gradients for
frequency-domain objectives.

## Where Gradients Are Noisy or Problematic

### Sharp PEC boundaries
Stairstepping creates discontinuities in the objective landscape when
geometry parameters move a PEC edge across a cell boundary. The gradient
at these transitions is technically correct but can be very large or change
sign abruptly.

**Mitigation**: Use subpixel smoothing for dielectric boundaries. For PEC,
accept that topology changes are non-smooth — use larger learning rates or
stochastic methods.

### Very long simulations (> 5000 steps)
Gradient magnitude can decay through many timesteps (vanishing gradient) or
accumulate numerical errors. The `jax.checkpoint` mechanism is exact in
theory but float32 rounding compounds over many recomputation segments.

**Mitigation**: Use the minimum n_steps needed. `run_until_decay(decay_by=1e-3)`
automatically determines sufficient length.

### Near-cutoff waveguide modes
At frequencies near cutoff, the propagation constant beta approaches zero,
causing large sensitivity to small parameter changes. Gradients are correct
but numerically stiff.

**Mitigation**: Avoid including near-cutoff frequencies in the objective.
Use a frequency band starting at 1.3 * f_cutoff.

### Float32 precision limits
JAX defaults to float32. For very small perturbations, finite-difference
validation may show large disagreement with AD due to cancellation.

**Mitigation**: When validating with FD, use h >= 1e-2 (not 1e-4).
The AD gradient is correct; the FD estimate is imprecise at small h.

## What Is NOT Differentiable

### CPML absorber region
Gradients w.r.t. eps_r inside the CPML layers are not physically meaningful.
The CPML is an artificial absorber, not a physical material. Exclude CPML
cells from your design region.

### Integer parameters
Grid dimensions, CPML layer count, timestep count — these are discrete and
not differentiable. Use them as fixed hyperparameters.

### Geometry topology
Adding or removing a shape (e.g., "should there be a hole here?") is a
discrete decision. rfx gradients optimize continuous parameters within a
fixed topology. For topology changes, use genetic algorithms or RL.

## Validating Gradients

Always validate AD gradients against finite differences for new problem setups:

```python
def fd_check(objective, eps_r, cell=(10, 5, 5), h=1e-2):
    """Finite-difference gradient validation."""
    eps_p = eps_r.at[cell].add(h)
    eps_m = eps_r.at[cell].add(-h)
    fd = (objective(eps_p) - objective(eps_m)) / (2 * h)
    ad = jax.grad(objective)(eps_r)[cell]
    rel_err = abs(ad - fd) / max(abs(fd), 1e-30)
    print(f"AD={ad:.6e}, FD={fd:.6e}, err={rel_err:.2%}")
    return rel_err

# Rules of thumb:
# - h=1e-2: reliable for float32 (< 5% error expected)
# - h=1e-3: may work, check case by case
# - h=1e-4: often unreliable in float32 (cancellation)
```

## Best Practices

1. **Always use `checkpoint=True`** — 10-100x memory savings, no accuracy loss
2. **Start with small grids** — iterate fast, scale up for final design
3. **Learning rate 0.01-0.1** — for eps_r optimization with Adam
4. **Validate with FD first** — before trusting gradient on a new problem type
5. **Average over frequencies** — broadband objectives are smoother
6. **Exclude CPML from design region** — gradients there are artifacts
7. **Use `until_decay`** — don't run longer than needed
8. **GPU for speed** — same code, 10-50x faster, identical gradients

## Supported Gradient Paths

| Physics path | Gradient flows? | Validated? |
|-------------|----------------|------------|
| Yee E/H update | Yes | FD < 2% |
| CPML absorber | Yes (but not useful) | — |
| Lumped port excitation | Yes | FD < 2% |
| TFSF plane wave | Yes | FD < 2% |
| Waveguide port | Yes | FD < 2% |
| DFT probe accumulation | Yes | FD < 50% (tiny values) |
| Debye dispersion | Yes | Needs validation |
| Lorentz/Drude dispersion | Yes | Needs validation |
| Lossy conductors (sigma) | Yes | Needs validation |
| Magnetic materials (mu_r) | Yes | Needs validation |
| S-parameter extraction | Yes | Verified via optimizer convergence |
| Subpixel smoothing | No (precomputed, not in AD graph) | N/A |
