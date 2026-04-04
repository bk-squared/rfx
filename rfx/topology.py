"""Density-based topology optimization for differentiable FDTD.

Leverages ``jax.grad`` through the entire pipeline:
    density field -> permittivity -> FDTD -> objective -> gradient

This is the killer feature for a differentiable FDTD simulator: the
ability to optimize *arbitrary* material layouts (not just parametric
shapes) using gradient information from the full electromagnetic solve.

Typical workflow
----------------
>>> from rfx import Simulation
>>> from rfx.topology import TopologyDesignRegion, topology_optimize
>>> sim = Simulation(freq_max=10e9, domain=(0.05, 0.05, 0.025))
>>> sim.add_port((0.005, 0.025, 0.001), "ez")
>>> sim.add_probe((0.045, 0.025, 0.001), "ez")
>>> region = TopologyDesignRegion(
...     corner_lo=(0.01, 0.01, 0), corner_hi=(0.04, 0.04, 0.001),
...     material_bg="air", material_fg="fr4",
... )
>>> result = topology_optimize(sim, region,
...     objective=lambda r: -jnp.sum(r.time_series ** 2),
...     n_iterations=100)

Key concepts
------------
- **Density field**: Each cell in the design region has a continuous
  density rho in [0, 1].  rho=0 -> background material, rho=1 -> foreground.
- **Density filter**: Spatial averaging (cone kernel) enforces a minimum
  feature size and regularizes the optimization.
- **Threshold projection**: A smooth Heaviside pushes densities toward
  0 or 1 (binary), controlled by sharpness parameter beta.
- **Beta continuation**: Gradually increasing beta during optimization
  starts with a smooth (convex-ish) landscape and ends with near-binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# TopologyDesignRegion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TopologyDesignRegion:
    """Density-based design region for topology optimization.

    Each cell in the region has a continuous density rho in [0, 1].
    rho=0 corresponds to *material_bg* (background) and rho=1 to
    *material_fg* (foreground).

    Parameters
    ----------
    corner_lo, corner_hi : (x, y, z) in metres
        Bounding box of the design region.
    material_bg : str
        Background material name (from MATERIAL_LIBRARY).
    material_fg : str
        Foreground material name.
    filter_radius : float or None
        Radius (in metres) for the cone density filter.
        If None, no filtering is applied.
    beta_projection : float
        Initial sharpness of the threshold projection.
        beta=1 is gentle; beta>=64 is near-binary.
    min_feature_size : float or None
        Alias for filter_radius (diameter = 2 * filter_radius).
        If provided, overrides filter_radius.
    """
    corner_lo: tuple[float, float, float]
    corner_hi: tuple[float, float, float]
    material_bg: str = "air"
    material_fg: str = "pec"
    filter_radius: float | None = None
    beta_projection: float = 1.0
    min_feature_size: float | None = None

    @property
    def effective_filter_radius(self) -> float | None:
        """Return the filter radius, accounting for min_feature_size alias."""
        if self.min_feature_size is not None:
            return self.min_feature_size / 2.0
        return self.filter_radius


# ---------------------------------------------------------------------------
# Filtering and projection (all differentiable)
# ---------------------------------------------------------------------------

def apply_density_filter(rho: jnp.ndarray, radius_cells: float) -> jnp.ndarray:
    """Cone-shaped density filter for minimum feature size control.

    Convolves the density field with a normalized cone kernel:
        w(r) = max(0, 1 - r/R)
    where R is the filter radius in cells.

    This is a standard technique from topology optimization (Bruns & Tortorelli,
    2001) that ensures smooth density fields and imposes a minimum length scale.

    Parameters
    ----------
    rho : array, shape (..., nx, ny, nz) or (nx, ny) or (nx, ny, nz)
        Density field in [0, 1].
    radius_cells : float
        Filter radius in grid cells.

    Returns
    -------
    rho_filtered : same shape as rho, values in [0, 1]
    """
    if radius_cells < 0.5:
        return rho

    ndim = rho.ndim
    r_int = int(jnp.ceil(radius_cells))

    # Build cone kernel for 2D or 3D.
    # We use an *unnormalized* kernel and divide by the local weight sum
    # to handle boundaries correctly (standard topology-optimization trick:
    # cells near edges see fewer kernel neighbours, so the raw convolution
    # would dip below the true average).
    if ndim == 2:
        ax = jnp.arange(-r_int, r_int + 1, dtype=jnp.float32)
        xx, yy = jnp.meshgrid(ax, ax, indexing="ij")
        dist = jnp.sqrt(xx ** 2 + yy ** 2)
        kernel = jnp.maximum(0.0, 1.0 - dist / radius_cells)
        # Do NOT normalize kernel here — we normalise per-cell below
        rho_4d = rho[None, None, :, :]  # (1, 1, nx, ny)
        kernel_4d = kernel[None, None, :, :]  # (1, 1, kx, ky)
        numerator = jax.lax.conv(rho_4d, kernel_4d,
                                 window_strides=(1, 1),
                                 padding="SAME")
        # Local weight sum: convolve ones with the same kernel
        ones_4d = jnp.ones_like(rho_4d)
        denominator = jax.lax.conv(ones_4d, kernel_4d,
                                   window_strides=(1, 1),
                                   padding="SAME")
        return (numerator / denominator)[0, 0]
    elif ndim == 3:
        ax = jnp.arange(-r_int, r_int + 1, dtype=jnp.float32)
        xx, yy, zz = jnp.meshgrid(ax, ax, ax, indexing="ij")
        dist = jnp.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
        kernel = jnp.maximum(0.0, 1.0 - dist / radius_cells)
        rho_5d = rho[None, None, :, :, :]
        kernel_5d = kernel[None, None, :, :, :]
        dn = jax.lax.conv_dimension_numbers(
            rho_5d.shape, kernel_5d.shape,
            ("NCDHW", "IODHW", "NCDHW"))
        numerator = jax.lax.conv_general_dilated(
            rho_5d, kernel_5d,
            window_strides=(1, 1, 1),
            padding="SAME",
            dimension_numbers=dn)
        ones_5d = jnp.ones_like(rho_5d)
        denominator = jax.lax.conv_general_dilated(
            ones_5d, kernel_5d,
            window_strides=(1, 1, 1),
            padding="SAME",
            dimension_numbers=dn)
        return (numerator / denominator)[0, 0]
    else:
        raise ValueError(f"Density field must be 2D or 3D, got ndim={ndim}")


def apply_projection(rho: jnp.ndarray, beta: float, eta: float = 0.5) -> jnp.ndarray:
    """Smooth Heaviside projection for binarization.

    Projects continuous densities toward 0 or 1 using:
        rho_proj = (tanh(beta*eta) + tanh(beta*(rho - eta)))
                   / (tanh(beta*eta) + tanh(beta*(1 - eta)))

    Parameters
    ----------
    rho : array
        Filtered density field, values in [0, 1].
    beta : float
        Sharpness parameter. beta=1 is gentle, beta>=64 is near-binary.
    eta : float
        Threshold level (default 0.5).

    Returns
    -------
    rho_projected : same shape, values in [0, 1]

    References
    ----------
    Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods,
    convergence and robust formulations in topology optimization.
    """
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


def density_to_eps(
    rho: jnp.ndarray,
    eps_bg: float,
    eps_fg: float,
    filter_radius_cells: float | None = None,
    beta: float = 1.0,
) -> jnp.ndarray:
    """Convert density field to permittivity via filter + projection + interpolation.

    The full pipeline is:
        1. Density filter (cone, radius=filter_radius_cells)
        2. Threshold projection (smooth Heaviside, sharpness=beta)
        3. Linear interpolation: eps = eps_bg + rho_proj * (eps_fg - eps_bg)

    All operations are differentiable w.r.t. rho.

    Parameters
    ----------
    rho : array, shape (nx, ny) or (nx, ny, nz)
        Raw density field in [0, 1].
    eps_bg, eps_fg : float
        Background and foreground permittivity.
    filter_radius_cells : float or None
        Cone filter radius in grid cells.  None = no filtering.
    beta : float
        Projection sharpness.

    Returns
    -------
    eps : same shape as rho
        Permittivity field, values in [eps_bg, eps_fg].
    """
    rho_f = rho
    if filter_radius_cells is not None and filter_radius_cells >= 0.5:
        rho_f = apply_density_filter(rho, filter_radius_cells)
    rho_p = apply_projection(rho_f, beta)
    return eps_bg + rho_p * (eps_fg - eps_bg)


# ---------------------------------------------------------------------------
# TopologyResult
# ---------------------------------------------------------------------------

@dataclass
class TopologyResult:
    """Result of a topology optimization run.

    Attributes
    ----------
    density : jnp.ndarray
        Final (raw) density field in the design region.
    density_projected : jnp.ndarray
        Final density after filtering and projection.
    eps_design : jnp.ndarray
        Final permittivity distribution in the design region.
    history : list of float
        Objective value at each iteration.
    beta_history : list of float
        Projection sharpness at each iteration.
    final_result : object
        Simulation result from the last iteration.
    """
    density: jnp.ndarray
    density_projected: jnp.ndarray
    eps_design: jnp.ndarray
    history: list[float]
    beta_history: list[float]
    final_result: object = None


# ---------------------------------------------------------------------------
# Default beta schedule
# ---------------------------------------------------------------------------

_DEFAULT_BETA_SCHEDULE: list[tuple[int, float]] = [
    (0, 1.0),
    (30, 4.0),
    (60, 16.0),
    (80, 64.0),
]


def _get_beta(iteration: int, schedule: list[tuple[int, float]]) -> float:
    """Look up beta for the current iteration from a schedule.

    The schedule is a list of (iteration, beta) pairs. Beta is held
    constant until the next breakpoint.
    """
    beta = schedule[0][1]
    for it_break, b in schedule:
        if iteration >= it_break:
            beta = b
    return beta


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def topology_optimize(
    sim,
    design_region: TopologyDesignRegion,
    objective: Callable,
    *,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    beta_schedule: list[tuple[int, float]] | None = None,
    init_density: jnp.ndarray | None = None,
    verbose: bool = True,
) -> TopologyResult:
    """Run density-based topology optimization.

    Optimizes a continuous density field within the design region to
    minimize an objective function, using gradient information from
    ``jax.value_and_grad`` through the full FDTD pipeline.

    Uses ``optax.adam`` for the optimizer.

    Parameters
    ----------
    sim : Simulation
        Base simulation (the design region overrides eps_r in its box).
    design_region : TopologyDesignRegion
        Region and material specification.
    objective : callable(Result) -> scalar
        Objective function to minimize.  Must be JAX-differentiable
        through the simulation.
    n_iterations : int
        Number of optimization iterations.
    learning_rate : float
        Learning rate for Adam optimizer.
    beta_schedule : list of (iteration, beta) or None
        Schedule for increasing projection sharpness.
        Default: [(0, 1), (30, 4), (60, 16), (80, 64)]
    init_density : array or None
        Initial density field.  If None, initialized to 0.5 everywhere
        (uniform mixture of bg and fg materials).
    verbose : bool
        Print progress every 10 iterations.

    Returns
    -------
    TopologyResult
        Contains final density, permittivity, loss history, and beta history.
    """
    try:
        import optax
    except ImportError:
        raise ImportError(
            "topology_optimize requires optax. "
            "Install it with: pip install optax  "
            "or: pip install rfx-fdtd[optimization]"
        )

    if beta_schedule is None:
        beta_schedule = _DEFAULT_BETA_SCHEDULE

    # Resolve material permittivities
    from rfx.api import MATERIAL_LIBRARY
    mat_bg = MATERIAL_LIBRARY.get(design_region.material_bg)
    mat_fg = MATERIAL_LIBRARY.get(design_region.material_fg)
    if mat_bg is None:
        raise ValueError(
            f"Unknown background material: {design_region.material_bg!r}. "
            f"Available: {list(MATERIAL_LIBRARY.keys())}"
        )
    if mat_fg is None:
        raise ValueError(
            f"Unknown foreground material: {design_region.material_fg!r}. "
            f"Available: {list(MATERIAL_LIBRARY.keys())}"
        )
    eps_bg = mat_bg["eps_r"]
    eps_fg = mat_fg["eps_r"]

    # Build grid and compute design region indices
    grid = sim._build_grid()
    lo_idx = grid.position_to_index(design_region.corner_lo)
    hi_idx = grid.position_to_index(design_region.corner_hi)
    design_shape = tuple(hi_idx[d] - lo_idx[d] + 1 for d in range(3))

    # Compute filter radius in cells
    filt_r = design_region.effective_filter_radius
    filter_radius_cells = None
    if filt_r is not None:
        filter_radius_cells = filt_r / grid.dx

    # Initialize density
    if init_density is None:
        init_density = 0.5 * jnp.ones(design_shape, dtype=jnp.float32)

    # We optimize in logit space for unconstrained optimization,
    # then map back via sigmoid to keep density in [0, 1].
    # logit(0.5) = 0
    init_logit = jnp.log(init_density / (1.0 - init_density + 1e-8) + 1e-8)

    # Set up optax optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(init_logit)

    logit = init_logit
    history = []
    beta_history = []

    def forward(logit_param, beta):
        """Forward pass: logit -> density -> eps -> simulation -> objective."""
        # Sigmoid to get density in [0, 1]
        rho = jax.nn.sigmoid(logit_param)

        # Density to permittivity (filter + projection + interpolation)
        eps_design = density_to_eps(
            rho, eps_bg, eps_fg,
            filter_radius_cells=filter_radius_cells,
            beta=beta,
        )

        # Build materials with design-region override
        materials, debye_spec, lorentz_spec, _ = sim._assemble_materials(grid)
        eps_r = materials.eps_r

        # Inject design region permittivity
        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(eps_design)

        from rfx.core.yee import MaterialArrays
        materials = MaterialArrays(
            eps_r=eps_r, sigma=materials.sigma, mu_r=materials.mu_r,
        )

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

        _, debye, lorentz = sim._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec,
        )

        result = _run(
            grid, materials, n_steps,
            boundary=sim._boundary,
            debye=debye,
            lorentz=lorentz,
            sources=sources,
            probes=probes,
            checkpoint=True,
        )
        return objective(result)

    for it in range(n_iterations):
        beta = _get_beta(it, beta_schedule)
        beta_history.append(beta)

        loss, grad = jax.value_and_grad(lambda l: forward(l, beta))(logit)
        loss_val = float(loss)
        history.append(loss_val)

        # optax update
        updates, opt_state = optimizer.update(grad, opt_state, logit)
        logit = optax.apply_updates(logit, updates)

        if verbose and (it % 10 == 0 or it == n_iterations - 1):
            rho_cur = jax.nn.sigmoid(logit)
            binarization = float(jnp.mean(4.0 * rho_cur * (1.0 - rho_cur)))
            print(
                f"  iter {it:4d}  loss = {loss_val:.6e}  "
                f"beta = {beta:.1f}  binarization = {binarization:.3f}"
            )

    # Final density and eps
    final_rho = jax.nn.sigmoid(logit)
    final_eps = density_to_eps(
        final_rho, eps_bg, eps_fg,
        filter_radius_cells=filter_radius_cells,
        beta=_get_beta(n_iterations - 1, beta_schedule),
    )

    # Compute projected density for reporting
    rho_filtered = final_rho
    if filter_radius_cells is not None and filter_radius_cells >= 0.5:
        rho_filtered = apply_density_filter(final_rho, filter_radius_cells)
    rho_projected = apply_projection(
        rho_filtered, _get_beta(n_iterations - 1, beta_schedule),
    )

    return TopologyResult(
        density=final_rho,
        density_projected=rho_projected,
        eps_design=final_eps,
        history=history,
        beta_history=beta_history,
    )
