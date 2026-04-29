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
    n_steps: int | None = None,
    num_periods: float = 20.0,
    verbose: bool = True,
    skip_preflight: bool = False,
    checkpoint_every: int | None = None,
    emit_time_series: bool = True,
    n_warmup: int = 0,
    design_mask: jnp.ndarray | None = None,
    distributed: bool = False,
    port_s11_freqs: object | None = None,
    checkpoint_segments: int | None = None,
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
    port_s11_freqs : array-like or None
        Frequencies (Hz) at which to accumulate per-port V/I DFTs in the
        JIT scan body so that ``result.s_params`` is populated with
        wave-decomposition |S11| values.  Required when the objective is
        :func:`minimize_s11_at_freq_wave_decomp` (issue #72).
    checkpoint_segments : int or None
        If set, route ``sim.forward`` through the segmented-checkpoint
        path (issue #73) which trades ≈2× compute for ≈√n_steps memory.
        Must divide ``n_steps``; useful when ``n_steps`` is large enough
        that the linear-memory scan would OOM.  ``None`` (default)
        preserves the legacy per-step ``jax.checkpoint`` behaviour.

        Currently wired only on the uniform single-device forward path.
        Non-uniform meshes and ``distributed=True`` will raise
        ``NotImplementedError``; NU support is tracked as a follow-up
        on issue #73.

    Returns
    -------
    OptimizeResult
    """
    sim._auto_preflight(skip=skip_preflight, context="optimize")

    # #64: dispatch to NU or uniform grid. The differentiable pipeline
    # is then the same (sim.forward with eps_override + pec_mask_override),
    # so optimize() automatically inherits checkpoint_every, emit_time_series,
    # and NU support as forward() gains them.
    is_nonuniform = (
        sim._dz_profile is not None
        or sim._dx_profile is not None
        or sim._dy_profile is not None
    )
    if is_nonuniform:
        grid = sim._build_nonuniform_grid()
        from rfx.nonuniform import position_to_index as _nu_pos_to_idx
        lo_idx = list(_nu_pos_to_idx(grid, region.corner_lo))
        hi_idx = list(_nu_pos_to_idx(grid, region.corner_hi))
        base_materials, _, _, base_pec_mask = sim._assemble_materials_nu(grid)
        period = 1.0 / float(sim._freq_max)
        # float(grid.dt): host-boundary context — called after _build_nonuniform_grid()
        # outside any JIT trace, so grid.dt is always a Python float here.
        _n_steps_auto = int(np.ceil(num_periods * period / float(grid.dt)))
    else:
        grid = sim._build_grid()
        lo_idx = list(grid.position_to_index(region.corner_lo))
        hi_idx = list(grid.position_to_index(region.corner_hi))
        base_materials, _, _, base_pec_mask, _, _ = sim._assemble_materials(grid)
        _n_steps_auto = grid.num_timesteps(num_periods=num_periods)

    # Per-face clamp (v1.7.5): using the symmetric ``pad_{axis}`` would
    # over-clamp a design region against a PMC/PEC reflector face whose
    # ``pad_lo`` or ``pad_hi`` is 0. Clamp each side to its own pad.
    pads_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)
    pads_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)
    dims = (grid.nx, grid.ny, grid.nz)
    for d in range(3):
        lo_idx[d] = max(lo_idx[d], pads_lo[d])
        hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads_hi[d])
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
    base_eps_r = base_materials.eps_r
    _n_steps = n_steps if n_steps is not None else _n_steps_auto

    # Adam state
    m = jnp.zeros_like(init_latent)
    v = jnp.zeros_like(init_latent)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    latent = init_latent
    loss_history = []
    it_count = [0]  # mutable counter for verbose inside forward()

    def forward(lat):
        """Forward pass: latent -> eps_override -> sim.forward() -> objective."""
        eps_design = _latent_to_eps(lat, eps_min, eps_max)

        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_override = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)

        if verbose and it_count[0] == 0:
            cells = grid.nx * grid.ny * grid.nz
            mesh_kind = "NU" if is_nonuniform else "uniform"
            print(f"  optimize: n_steps={_n_steps}, grid={grid.shape} "
                  f"({cells/1e6:.1f}M cells, {mesh_kind} mesh)")

        result = sim.forward(
            eps_override=eps_override,
            pec_mask_override=base_pec_mask,
            n_steps=_n_steps,
            checkpoint=True,
            checkpoint_segments=checkpoint_segments,
            checkpoint_every=checkpoint_every,
            emit_time_series=emit_time_series,
            n_warmup=n_warmup,
            design_mask=design_mask,
            distributed=distributed,
            port_s11_freqs=port_s11_freqs,
            skip_preflight=True,  # already done at optimize() entry
        )
        import inspect
        sig = inspect.signature(objective)
        if 'ntff_box' in sig.parameters:
            return objective(result, ntff_box=result.ntff_box)
        return objective(result)

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


# ---------------------------------------------------------------------------
# Progressive multi-resolution orchestrator (issue #42)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgressiveStage:
    """One resolution stage in a progressive_optimize schedule.

    Attributes
    ----------
    dx : float
        Target cell size for this stage (metres). Passed to ``sim_factory``.
    n_iters : int
        Number of Adam iterations at this resolution.
    lr : float
        Learning rate for this stage (defaults to 0.01).
    num_periods : float
        Periods at freq_max for auto n_steps computation (default 20).
    n_steps : int or None
        Explicit step count override. None -> auto from ``num_periods``.
    """
    dx: float
    n_iters: int
    lr: float = 0.01
    num_periods: float = 20.0
    n_steps: int | None = None


@dataclass
class ProgressiveOptimizeResult:
    """Result of a progressive_optimize run.

    Attributes
    ----------
    stages : list of OptimizeResult
        Per-stage optimize() output, in schedule order.
    final_eps_design : jnp.ndarray
        Optimized permittivity at the finest stage's design-region shape.
    final_latent : jnp.ndarray
        Final latent at the finest stage's shape.
    loss_history : list of float
        Concatenated loss across all stages.
    stage_boundaries : list of int
        Iteration index where each stage starts (cumulative sum of
        ``n_iters``). Useful for plotting.
    """
    stages: list[OptimizeResult]
    final_eps_design: jnp.ndarray
    final_latent: jnp.ndarray
    loss_history: list[float]
    stage_boundaries: list[int]


def _resize_latent(latent: jnp.ndarray, new_shape: tuple[int, int, int],
                   method: str = "linear") -> jnp.ndarray:
    """Resize a 3D latent tensor to ``new_shape`` via ``jax.image.resize``."""
    if tuple(latent.shape) == tuple(new_shape):
        return latent
    return jax.image.resize(latent, shape=tuple(new_shape), method=method)


def progressive_optimize(
    sim_factory: Callable[[float], object],
    region: DesignRegion,
    objective: Callable,
    schedule: list[ProgressiveStage],
    *,
    init_latent: jnp.ndarray | None = None,
    interp_method: str = "linear",
    verbose: bool = True,
    skip_preflight: bool = False,
    checkpoint_every: int | None = None,
    emit_time_series: bool = True,
    n_warmup: int = 0,
    distributed: bool = False,
) -> ProgressiveOptimizeResult:
    """Progressive multi-resolution inverse design (issue #42).

    Runs ``optimize()`` at each stage in ``schedule``, upsampling the
    latent between stages. Coarse stages produce a fast loss-landscape
    scan; finer stages refine on top.

    Parameters
    ----------
    sim_factory : callable(dx) -> Simulation
        Builds a Simulation at the given dx. Called once per stage.
    region : DesignRegion
        Design region in physical coordinates (resolution-independent).
    objective : callable(Result) -> scalar
        JAX-differentiable loss.
    schedule : list of ProgressiveStage
        Must be non-empty. Stages usually go coarse -> fine.
    init_latent : jnp.ndarray or None
        Initial latent at the FIRST stage's design-region shape.
        If None, optimize() initialises to zeros.
    interp_method : str
        Latent-upsample interpolation ("linear", "nearest", "cubic").
    verbose : bool
        Print per-stage progress banner.

    Returns
    -------
    ProgressiveOptimizeResult

    Notes
    -----
    The design variable basis is the per-stage latent (not a stage-
    independent per-wavelength grid). ``jax.image.resize`` bridges
    stages, which is adequate when the mesh refinement is modest.
    For dramatic resolution jumps, consider a dedicated per-wavelength
    basis.
    """
    if not schedule:
        raise ValueError("progressive_optimize: schedule must be non-empty")

    stage_results: list[OptimizeResult] = []
    loss_history: list[float] = []
    stage_boundaries: list[int] = [0]
    current_latent = init_latent

    for stage_idx, stage in enumerate(schedule):
        if verbose:
            print(f"\n=== stage {stage_idx + 1}/{len(schedule)}: "
                  f"dx={stage.dx * 1e3:.3f}mm, n_iters={stage.n_iters}, "
                  f"lr={stage.lr} ===")

        sim = sim_factory(stage.dx)

        is_nonuniform = (
            sim._dz_profile is not None
            or sim._dx_profile is not None
            or sim._dy_profile is not None
        )
        grid = (sim._build_nonuniform_grid()
                if is_nonuniform else sim._build_grid())

        # Compute this stage's design-region shape using the same logic
        # as optimize() so upsample target is exact.
        if is_nonuniform:
            from rfx.nonuniform import position_to_index as _nu_pos_to_idx
            lo_idx = list(_nu_pos_to_idx(grid, region.corner_lo))
            hi_idx = list(_nu_pos_to_idx(grid, region.corner_hi))
        else:
            lo_idx = list(grid.position_to_index(region.corner_lo))
            hi_idx = list(grid.position_to_index(region.corner_hi))

        pads_lo = (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)
        pads_hi = (grid.pad_x_hi, grid.pad_y_hi, grid.pad_z_hi)
        dims = (grid.nx, grid.ny, grid.nz)
        for d in range(3):
            lo_idx[d] = max(lo_idx[d], pads_lo[d])
            hi_idx[d] = min(hi_idx[d], dims[d] - 1 - pads_hi[d])
        stage_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))
        if any(s <= 0 for s in stage_shape):
            raise ValueError(
                f"progressive_optimize stage {stage_idx}: design region "
                f"collapses after CPML clamping at dx={stage.dx}"
            )

        if current_latent is None:
            stage_init = None
        else:
            stage_init = _resize_latent(
                current_latent, stage_shape, method=interp_method,
            ).astype(jnp.float32)

        stage_result = optimize(
            sim, region, objective,
            n_iters=stage.n_iters,
            lr=stage.lr,
            init_latent=stage_init,
            n_steps=stage.n_steps,
            num_periods=stage.num_periods,
            verbose=verbose,
            skip_preflight=skip_preflight,
            checkpoint_every=checkpoint_every,
            emit_time_series=emit_time_series,
            n_warmup=n_warmup,
            distributed=distributed,
        )

        stage_results.append(stage_result)
        loss_history.extend(stage_result.loss_history)
        stage_boundaries.append(stage_boundaries[-1] + len(stage_result.loss_history))
        current_latent = stage_result.latent

    return ProgressiveOptimizeResult(
        stages=stage_results,
        final_eps_design=stage_results[-1].eps_design,
        final_latent=stage_results[-1].latent,
        loss_history=loss_history,
        stage_boundaries=stage_boundaries,
    )
