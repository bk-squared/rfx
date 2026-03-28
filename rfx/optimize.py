"""Gradient-based inverse design using JAX autodiff.

Leverages jax.grad / jax.value_and_grad on the differentiable FDTD
pipeline (with jax.checkpoint) to optimize material layouts for
target electromagnetic responses.

Typical workflow:
    1. Define a parameterized material layout (design region).
    2. Define an objective function over SimResult (e.g., S11 < -10 dB).
    3. Call ``optimize()`` to run gradient descent.

Example
-------
>>> from rfx.optimize import optimize, DesignRegion
>>> region = DesignRegion(corner_lo=(0.01, 0.01, 0), corner_hi=(0.04, 0.04, 0.001),
...                       eps_range=(1.0, 4.4))
>>> result = optimize(sim, region, objective_fn, n_iters=50, lr=0.01)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DesignRegion:
    """Rectangular region where material parameters are optimizable.

    Parameters
    ----------
    corner_lo, corner_hi : (x, y, z) in metres
        Bounding box of the design region.
    eps_range : (eps_min, eps_max)
        Permittivity bounds.  The optimizer works in a continuous
        latent space and projects to this range via sigmoid.
    """
    corner_lo: tuple[float, float, float]
    corner_hi: tuple[float, float, float]
    eps_range: tuple[float, float] = (1.0, 12.0)


@dataclass
class OptimizeResult:
    """Result of an optimization run.

    Attributes
    ----------
    eps_design : (nx_d, ny_d, nz_d) float array
        Optimized permittivity in the design region.
    loss_history : list of float
        Objective value at each iteration.
    latent : jnp.ndarray
        Final latent parameters (pre-sigmoid).
    """
    eps_design: jnp.ndarray
    loss_history: list[float]
    latent: jnp.ndarray


def _latent_to_eps(latent: jnp.ndarray, eps_min: float, eps_max: float) -> jnp.ndarray:
    """Map unbounded latent parameters to bounded permittivity via sigmoid."""
    return eps_min + (eps_max - eps_min) * jax.nn.sigmoid(latent)


def optimize(
    sim,
    region: DesignRegion,
    objective: Callable,
    *,
    n_iters: int = 50,
    lr: float = 0.01,
    init_latent: jnp.ndarray | None = None,
    verbose: bool = True,
) -> OptimizeResult:
    """Run gradient-based optimization on a design region.

    Parameters
    ----------
    sim : Simulation
        Base simulation (the design region overrides eps_r in its box).
    region : DesignRegion
        Design region specification.
    objective : callable(Result) -> scalar
        Objective function to minimize.  Must be JAX-differentiable
        through the simulation.
    n_iters : int
        Number of optimization iterations.
    lr : float
        Learning rate (Adam).
    init_latent : array or None
        Initial latent parameters.  If None, initialized to zeros
        (maps to midpoint of eps_range).
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    OptimizeResult
    """
    grid = sim._build_grid()

    # Compute design-region grid indices
    lo_idx = grid.position_to_index(region.corner_lo)
    hi_idx = grid.position_to_index(region.corner_hi)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))

    if init_latent is None:
        init_latent = jnp.zeros(design_shape, dtype=jnp.float32)

    eps_min, eps_max = region.eps_range

    # Adam state
    m = jnp.zeros_like(init_latent)
    v = jnp.zeros_like(init_latent)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    latent = init_latent
    loss_history = []

    def forward(lat):
        """Forward pass: latent -> eps -> simulation -> objective."""
        eps_design = _latent_to_eps(lat, eps_min, eps_max)

        # Build materials with design region override
        materials, debye = sim._build_materials(grid)
        eps_r = materials.eps_r

        # Inject design region
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)

        from rfx.core.yee import MaterialArrays
        materials = MaterialArrays(eps_r=eps_r, sigma=materials.sigma, mu_r=materials.mu_r)

        # Run simulation
        from rfx.simulation import run as _run, make_source, make_probe, make_port_source
        from rfx.sources.sources import LumpedPort, setup_lumped_port

        n_steps = grid.num_timesteps(num_periods=20.0)
        sources = []
        probes = []

        for pe in sim._ports:
            lp = LumpedPort(
                position=pe.position, component=pe.component,
                impedance=pe.impedance, excitation=pe.waveform,
            )
            materials = setup_lumped_port(grid, lp, materials)
            sources.append(make_port_source(grid, lp, materials, n_steps))

        for pe in sim._probes:
            probes.append(make_probe(grid, pe.position, pe.component))

        result = _run(
            grid, materials, n_steps,
            boundary=sim._boundary,
            debye=debye,
            sources=sources,
            probes=probes,
            checkpoint=True,
        )
        return objective(result)

    grad_fn = jax.value_and_grad(forward)

    for it in range(n_iters):
        loss, grad = grad_fn(latent)
        loss_val = float(loss)
        loss_history.append(loss_val)

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        latent = latent - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        if verbose and (it % 10 == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}  loss = {loss_val:.6e}")

    eps_design = _latent_to_eps(latent, eps_min, eps_max)
    return OptimizeResult(
        eps_design=eps_design,
        loss_history=loss_history,
        latent=latent,
    )
