---
title: "Inverse Design"
sidebar:
  order: 16
---

rfx runs a JAX-differentiable FDTD: `jax.grad` returns reverse-mode gradients of
a scalar loss through the FDTD time-stepping, so the same solver that computes
the fields also computes design sensitivities. If you are coming from Meep's
adjoint terminology, start with
[Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/).

## How it works

JAX traces the solver and objective path as a computation graph. Gradient
checkpointing recomputes forward states during backpropagation to bound
reverse-mode memory (see [Memory Reduction](/rfx/guide/memory-reduction/)).

```text
Forward:  eps_r → FDTD steps → probes / NTFF → scalar loss
Backward: jax.grad(loss)(eps_r) → ∂loss/∂eps_r
```

## Manual gradient loop

Most designs should use the higher-level [`optimize()` driver](#design-region-api),
which builds this loop for you. Drop to a manual `jax.grad` loop only for custom
objectives or training loops. Stay on the differentiable `Simulation.forward()`
API instead of hand-writing FDTD updates: build the `Simulation` once, pass an
`eps_override` array shaped like the grid permittivity, and differentiate a
scalar loss:

```python
import jax
import jax.numpy as jnp

# sim is an already configured Simulation with sources/probes/ports.
# Discover the grid shape from a quick run (result.grid is always populated):
eps0 = jnp.ones(sim.run(n_steps=1).grid.shape, dtype=jnp.float32)

def objective(eps_r):
    result = sim.forward(eps_override=eps_r, n_steps=150, checkpoint=True)
    return -jnp.sum(result.time_series ** 2)  # example proxy loss

grad = jax.grad(objective)(eps0)
```

This differentiates the proxy objective only. It does not by itself validate the
final RF observable; re-run the relevant port, resonance, far-field, or
convergence check on the optimized design (see [Validation](/rfx/guide/validation/)).

## Built-in objectives: choose the right family

### 1) Post-processed S-parameter objectives

Each factory returns `objective(result) -> scalar` that reads `result.s_params`,
so it applies to a completed `run(compute_s_params=True)` result. These raise
`ValueError` inside the default `optimize()`/`forward()` loop, which does not
build the post-processed S-parameter matrix (see family 2). An objective value
is only as trustworthy as the S-parameter extraction behind it — confirm that
port's S-parameter path is validated (see [Validation](/rfx/guide/validation/))
before treating the optimized number as a physical result.

```python
from rfx import minimize_s11, maximize_s21, target_impedance, maximize_bandwidth

obj_s11 = minimize_s11(freqs=jnp.array([5e9]), target_db=-10)
obj_s21 = maximize_s21(freqs=jnp.linspace(4e9, 6e9, 20))
obj_z = target_impedance(freq=5e9, z_target=50.0)
obj_bw = maximize_bandwidth(f_center=5e9, f_bw=2e9, s11_threshold=-10)
```

### 2) Differentiable loop objectives for `optimize()`

The `optimize()`/`forward()` pass emits `result.time_series` but not the
post-processed `result.s_params` matrix, so the family-1 objectives cannot run
inside a gradient loop. Use these time-domain proxies instead — they read only
`result.time_series` and compose with both `optimize()` and `topology_optimize()`:

```python
from rfx import minimize_reflected_energy, maximize_transmitted_energy

obj_reflect = minimize_reflected_energy(port_probe_idx=0)
obj_transmit = maximize_transmitted_energy(output_probe_idx=-1)
```

`minimize_reflected_energy` is a late-time-reflection S11 proxy;
`maximize_transmitted_energy` is an output-power S21 proxy. Both index
`result.time_series` columns: `port_probe_idx` selects the probe co-located with
the excitation port, `output_probe_idx` the downstream probe.

**Precondition — the split window must contain the reflection.**
`minimize_reflected_energy` splits the probe time series at `late_fraction`
(default: the second half) and treats the late half as reflected energy. That
premise only holds if the round trip from the port to the reflecting feature and
back arrives *after* the split point and *before* the run ends. On a short
round-trip geometry (a thin substrate, a feature close to the port), the
reflection lands in the early "incident" half and the late window is nearly empty,
so the proxy collapses to `~0` and its gradient becomes numerical noise — with no
error raised. Sanity-check the loss magnitude (a meaningfully reflecting design
sits near `1e-2`–`1e-1`, not `~1e-7`), then fix the window by its failure mode:
enlarge `n_steps` if the reflection arrives after the run ends; **raise**
`late_fraction` (which moves the split earlier) if it arrives before the split; or,
if the incident pulse and reflection overlap in time on a very short round trip,
narrow the source bandwidth to compress the incident pulse — or use an
impedance-referenced port (`add_port(..., impedance=Z0)` with
`forward(port_s11_freqs=...)`), which separates incident and reflected waves
exactly. Note that an AD-vs-finite-difference check will *not* catch an empty
window: both differentiate the same window and agree (see
[Autodiff and Adjoint Background](/rfx/guide/autodiff-adjoint/#a-passing-finite-difference-check-is-necessary-not-sufficient)).

For NTFF/directivity optimization, pass
`maximize_directivity(..., log_ratio=True)` when the design variable can change
total radiated power (conductors/PEC, lossy, or magnitude-changing dielectric
DoFs). The default `log_ratio=False` mode drops a quotient-rule term and yields
wrong-sign gradients for those DoFs; it is correct only for shape-preserving,
constant-radiated-power DoFs.

## Design-region API

`optimize()` is the high-level driver: give it a configured `Simulation`, a
`DesignRegion` box (physical `corner_lo`/`corner_hi` in metres plus an
`eps_range` the design is clamped to), and a proxy objective from family 2. It
runs an Adam gradient loop over the region's permittivity and returns an
`OptimizeResult` with `eps_design` (optimized permittivity in the box),
`loss_history`, and the final `latent` parameters. For port/probe setup on the
base simulation, see [Sources & Ports](/rfx/guide/sources-ports/).

```python
from rfx import Simulation, DesignRegion, optimize, minimize_reflected_energy

sim = Simulation(freq_max=10e9, domain=(0.1, 0.04, 0.02), boundary="cpml")
sim.add_port(...)  # see Sources & Ports for the concrete call

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
# result.eps_design, result.loss_history, result.latent
```

The region is clamped to the grid interior; `optimize()` raises `ValueError` if
it lies entirely inside the CPML absorber, so keep `corner_lo`/`corner_hi` within
the physical domain rather than the padding.

## Far-field objectives with NTFF data

`maximize_directivity` optimizes the directivity ratio toward a target direction.
It reads `result.ntff_data`, `result.ntff_box`, and `result.grid`, so the base
simulation must register a near-to-far-field box first via
`sim.add_ntff_box(corner_lo, corner_hi)`; without it the objective raises
`ValueError`. `theta_target`/`phi_target` are in radians.

```python
from rfx import maximize_directivity

objective = maximize_directivity(
    theta_target=0.0,  # radians
    phi_target=0.0,
    log_ratio=True,
)
```

Pass this to `optimize()` like any family-2 proxy. (`optimize()` also forwards
`result.ntff_box` as a keyword to any custom objective whose signature includes
an `ntff_box` parameter; the built-in objective does not need that — it reads the
box from the result.) NTFF passes cost more than probe-only losses, so iterate on
a coarse grid, then re-run the final design through the far-field validation path
(see [Far-field & RCS](/rfx/guide/farfield-rcs/)).

## Tips

- **Memory**: `checkpoint=True` is the default; for large `n_steps`, use `checkpoint_segments` (uniform) or `checkpoint_every` (non-uniform) to trade ~2x compute for ~√n_steps memory. See [Memory Reduction](/rfx/guide/memory-reduction/).
- **Start with small grids** for design iteration, then scale up for the final verification run.
- **Learning rate**: `0.01–0.1` is a reasonable first range for permittivity optimization; see [Gradient Behavior](/rfx/guide/gradient-behavior/).
- **Proxy objectives first**: start with `minimize_reflected_energy()` or `maximize_transmitted_energy()`.
- **NTFF objectives cost more** than probe-only losses; reserve them for radiation targets.
- **GPU acceleration** depends on the installed JAX/CUDA environment; verify device placement for performance-sensitive runs.
