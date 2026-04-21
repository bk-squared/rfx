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

import inspect
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


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


def _call_optimize_objective(
    objective: Callable,
    result,
    *,
    accepts_ntff_box: bool,
) -> jnp.ndarray:
    """Evaluate an optimize objective against the minimal forward result contract."""
    if accepts_ntff_box:
        return objective(result, ntff_box=result.ntff_box)
    return objective(result)


def _resolve_optimize_hybrid_context(
    sim,
    *,
    eps_r: jnp.ndarray,
    n_steps: int,
    adjoint_mode: str,
):
    """Build the supported hybrid replay context once when optimize opts in."""
    if adjoint_mode not in {"pure_ad", "hybrid", "auto"}:
        raise ValueError(
            f"adjoint_mode must be 'pure_ad', 'hybrid', or 'auto', got {adjoint_mode!r}"
        )
    if adjoint_mode == "pure_ad":
        return None

    inputs = sim.build_hybrid_phase1_inputs(eps_override=eps_r, n_steps=n_steps)
    report = sim.inspect_hybrid_phase1_from_inputs(inputs)
    if report.supported:
        return sim.build_hybrid_phase1_context_from_inputs(inputs)
    if adjoint_mode == "hybrid":
        raise ValueError(report.reason_text)
    return None


def optimize(
    sim,
    region: DesignRegion,
    objective: Callable,
    *,
    n_iters: int = 50,
    lr: float = 0.01,
    init_latent: jnp.ndarray | None = None,
    n_steps: int | None = None,
    num_periods: float = 20.0,
    verbose: bool = True,
    adjoint_mode: str = "pure_ad",
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
    n_steps : int or None
        Number of simulation timesteps per iteration. Controls memory
        usage — fewer steps = less memory for ``jax.grad``. If None,
        auto-computed from *num_periods*.
    num_periods : float
        Periods at freq_max for auto n_steps (default 20). Reduce to
        10 for lower memory usage with minimal accuracy loss.
    verbose : bool
        Print progress every 10 iterations.
    adjoint_mode : {"pure_ad", "hybrid", "auto"}
        Forward/adjoint routing policy for each optimize iteration.
        ``pure_ad`` preserves the current default behavior.
        ``hybrid`` requires the current Phase 1 hybrid seam to support the
        configured simulation and raises otherwise.
        ``auto`` uses the hybrid seam only when support inspection passes and
        otherwise falls back to the existing pure-AD path.

    Returns
    -------
    OptimizeResult
    """
    grid = sim._build_grid()

    # Compute design-region grid indices, clamped to interior (exclude CPML).
    lo_idx = list(grid.position_to_index(region.corner_lo))
    hi_idx = list(grid.position_to_index(region.corner_hi))
    pads = (grid.pad_x, grid.pad_y, grid.pad_z)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads[d])
    lo_idx = tuple(lo_idx)
    hi_idx = tuple(hi_idx)

    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
    if any(s <= 0 for s in design_shape):
        raise ValueError(
            f"Design region is empty after clamping to interior "
            f"(lo_idx={lo_idx}, hi_idx={hi_idx}). Ensure the design "
            f"region does not lie entirely within the CPML boundary."
        )

    if init_latent is None:
        init_latent = jnp.zeros(design_shape, dtype=jnp.float32)

    eps_min, eps_max = region.eps_range
    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    base_sigma = base_materials.sigma
    base_mu_r = base_materials.mu_r
    _n_steps = n_steps if n_steps is not None else grid.num_timesteps(num_periods=num_periods)
    objective_accepts_ntff_box = "ntff_box" in inspect.signature(objective).parameters
    hybrid_context = _resolve_optimize_hybrid_context(
        sim,
        eps_r=base_eps_r,
        n_steps=_n_steps,
        adjoint_mode=adjoint_mode,
    )

    # Adam state
    m = jnp.zeros_like(init_latent)
    v = jnp.zeros_like(init_latent)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    latent = init_latent
    loss_history = []
    it_count = [0]  # mutable counter for verbose inside forward()

    def forward(lat):
        """Forward pass: latent -> eps -> simulation -> objective."""
        eps_design = _latent_to_eps(lat, eps_min, eps_max)

        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)

        if verbose and it_count[0] == 0:
            cells = grid.nx * grid.ny * grid.nz
            print(f"  optimize: n_steps={_n_steps}, grid={grid.shape} "
                  f"({cells/1e6:.1f}M cells)")

        if hybrid_context is not None:
            result = sim.forward_hybrid_phase1_from_context(hybrid_context, eps_override=eps_r)
        else:
            from rfx.core.yee import MaterialArrays

            materials = MaterialArrays(eps_r=eps_r, sigma=base_sigma, mu_r=base_mu_r)
            result = sim._forward_from_materials(
                grid,
                materials,
                debye_spec,
                lorentz_spec,
                n_steps=_n_steps,
                checkpoint=True,
                pec_mask=base_pec_mask,
            )
        return _call_optimize_objective(
            objective,
            result,
            accepts_ntff_box=objective_accepts_ntff_box,
        )

    grad_fn = jax.value_and_grad(forward)

    for it in range(n_iters):
        loss, grad = grad_fn(latent)
        it_count[0] = it + 1
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


# ---------------------------------------------------------------------------
# Gradient checking (AD vs finite-difference)
# ---------------------------------------------------------------------------

@dataclass
class GradientCheckResult:
    """Result of AD-vs-FD gradient comparison.

    Attributes
    ----------
    ad_grad : jnp.ndarray
        Gradient from reverse-mode autodiff.
    fd_grad : jnp.ndarray
        Gradient from central finite differences.
    relative_error : float
        max |ad - fd| / (max |ad| + max |fd| + 1e-30).
    """
    ad_grad: jnp.ndarray
    fd_grad: jnp.ndarray
    relative_error: float


def gradient_check(
    sim,
    design_params: jnp.ndarray,
    objective_fn: Callable,
    *,
    eps: float = 1e-3,
    n_steps: int | None = None,
    num_periods: float = 20.0,
) -> GradientCheckResult:
    """Compare AD gradient with finite-difference gradient.

    Runs the simulation via ``sim.forward()`` with permittivity
    overrides derived from *design_params*, computes the gradient
    of *objective_fn* with respect to *design_params* using both
    reverse-mode AD and central finite differences, and returns
    the comparison.

    Parameters
    ----------
    sim : Simulation
        Configured simulation (ports, probes already added).
    design_params : jnp.ndarray
        Flat array of design parameters.  These are added element-wise
        to the baseline ``eps_r`` from ``sim._assemble_materials()``.
        Shape must broadcast with ``grid.shape``.
    objective_fn : callable(SimResult) -> scalar
        Differentiable objective function.
    eps : float
        Finite-difference perturbation (default 1e-3).
    n_steps : int or None
        Number of timesteps.  Passed to ``sim.forward()``.
    num_periods : float
        Periods at freq_max if *n_steps* is None.

    Returns
    -------
    GradientCheckResult
        Contains ad_grad, fd_grad, and relative_error.
    """
    grid = sim._build_grid()
    base_materials, _, _, _, _, _ = sim._assemble_materials(grid)
    base_eps = base_materials.eps_r

    fwd_kw = {}
    if n_steps is not None:
        fwd_kw["n_steps"] = n_steps
    else:
        fwd_kw["num_periods"] = num_periods

    def loss_fn(params):
        eps_r = base_eps + params
        result = sim.forward(eps_override=eps_r, checkpoint=True, **fwd_kw)
        return objective_fn(result)

    # AD gradient
    ad_grad = jax.grad(loss_fn)(design_params)

    # Finite-difference gradient (central)
    flat = design_params.ravel()
    fd_grad_flat = jnp.zeros_like(flat)
    n_params = flat.shape[0]

    for i in range(n_params):
        e_vec = jnp.zeros_like(flat).at[i].set(eps)
        p_plus = (flat + e_vec).reshape(design_params.shape)
        p_minus = (flat - e_vec).reshape(design_params.shape)
        f_plus = loss_fn(p_plus)
        f_minus = loss_fn(p_minus)
        fd_grad_flat = fd_grad_flat.at[i].set((f_plus - f_minus) / (2 * eps))

    fd_grad = fd_grad_flat.reshape(design_params.shape)

    # Relative error
    ad_abs = float(jnp.max(jnp.abs(ad_grad)))
    fd_abs = float(jnp.max(jnp.abs(fd_grad)))
    diff = float(jnp.max(jnp.abs(ad_grad - fd_grad)))
    rel_err = diff / (ad_abs + fd_abs + 1e-30)

    return GradientCheckResult(
        ad_grad=ad_grad,
        fd_grad=fd_grad,
        relative_error=rel_err,
    )
