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

Hybrid note
-----------
The current experimental hybrid subset is intentionally narrower than the
general pure-AD topology surface: zero-sigma, source/probe-only dielectric
topology on PEC or CPML boundary. Port-based, PEC-foreground, sigma-bearing,
and dispersive-positive topology remain on the pure-AD lane unless explicitly
expanded in a later phase.

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

import inspect
from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp


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


@dataclass(frozen=True)
class TopologyMaterialFields:
    """Material fields produced by the density pipeline."""
    eps: jnp.ndarray
    sigma: jnp.ndarray
    pec_occupancy: jnp.ndarray | None = None

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


def density_to_material_fields(
    rho: jnp.ndarray,
    eps_bg: float,
    eps_fg: float,
    filter_radius_cells: float | None = None,
    beta: float = 1.0,
    sigma_bg: float = 0.0,
    sigma_fg: float = 0.0,
    *,
    pec_bg: bool = False,
    pec_fg: bool = False,
) -> TopologyMaterialFields:
    """Convert density to differentiable material fields.

    For finite materials, this interpolates ``eps`` and ``sigma`` directly.
    When one side of the interpolation is PEC, it instead keeps the
    non-conducting material properties and emits a relaxed conductor
    occupancy field for the solver-level PEC operator.
    """
    rho_f = rho
    if filter_radius_cells is not None and filter_radius_cells >= 0.5:
        rho_f = apply_density_filter(rho, filter_radius_cells)
    rho_p = apply_projection(rho_f, beta)

    if pec_bg and pec_fg:
        eps = jnp.full_like(rho_p, eps_bg)
        sigma = jnp.full_like(rho_p, sigma_bg)
        return TopologyMaterialFields(eps=eps, sigma=sigma, pec_occupancy=jnp.ones_like(rho_p))

    if pec_fg and not pec_bg:
        eps = jnp.full_like(rho_p, eps_bg)
        sigma = jnp.full_like(rho_p, sigma_bg)
        return TopologyMaterialFields(eps=eps, sigma=sigma, pec_occupancy=rho_p)

    if pec_bg and not pec_fg:
        eps = jnp.full_like(rho_p, eps_fg)
        sigma = jnp.full_like(rho_p, sigma_fg)
        return TopologyMaterialFields(eps=eps, sigma=sigma, pec_occupancy=1.0 - rho_p)

    eps = eps_bg + rho_p * (eps_fg - eps_bg)
    sigma = sigma_bg + rho_p * (sigma_fg - sigma_bg)
    return TopologyMaterialFields(eps=eps, sigma=sigma, pec_occupancy=None)


def density_to_eps(
    rho: jnp.ndarray,
    eps_bg: float,
    eps_fg: float,
    filter_radius_cells: float | None = None,
    beta: float = 1.0,
    sigma_bg: float = 0.0,
    sigma_fg: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward-compatible density -> (eps, sigma) helper."""
    fields = density_to_material_fields(
        rho,
        eps_bg,
        eps_fg,
        filter_radius_cells=filter_radius_cells,
        beta=beta,
        sigma_bg=sigma_bg,
        sigma_fg=sigma_fg,
    )
    return fields.eps, fields.sigma


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
    pec_occupancy_design: jnp.ndarray | None = None

    @property
    def loss_history(self) -> list[float]:
        """Alias for ``history``, consistent with ``OptimizeResult.loss_history``."""
        return self.history


def _call_topology_objective(objective: Callable, result, *, accepts_ntff_box: bool) -> jnp.ndarray:
    """Evaluate a topology objective against the minimal forward result contract."""
    if accepts_ntff_box:
        return objective(result, ntff_box=result.ntff_box)
    return objective(result)


def _build_topology_material_state(
    base_eps_r: jnp.ndarray,
    base_sigma: jnp.ndarray,
    base_mu_r: jnp.ndarray,
    lo_idx: tuple[int, int, int],
    hi_idx: tuple[int, int, int],
    fields: TopologyMaterialFields,
):
    """Material-state helper shared by topology inspection and routing."""
    from rfx.core.yee import MaterialArrays

    si, sj, sk = lo_idx
    ei, ej, ek = hi_idx
    eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.eps)
    sigma = base_sigma.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.sigma)
    materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=base_mu_r)

    pec_occupancy = None
    if fields.pec_occupancy is not None:
        pec_occupancy = jnp.zeros(base_eps_r.shape, dtype=jnp.float32)
        pec_occupancy = pec_occupancy.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.pec_occupancy)
    return materials, pec_occupancy


def _topology_hybrid_reason_texts(
    sim,
    *,
    bg_has_dispersion: bool,
    fg_has_dispersion: bool,
    debye_spec,
    lorentz_spec,
    materials,
    pec_mask,
    pec_occupancy,
    port_metadata,
) -> tuple[str, ...]:
    """Topology-specific guardrails layered on top of seam-owned inspection."""
    reasons: list[str] = []
    if sim._boundary not in {"pec", "cpml"}:
        reasons.append("Phase II topology hybrid supports PEC or CPML boundary only")
    if debye_spec is not None or lorentz_spec is not None or bg_has_dispersion or fg_has_dispersion:
        reasons.append("Phase II topology hybrid supports nondispersive materials only")
    if port_metadata is not None and getattr(port_metadata, "total_ports", 0) > 0:
        reasons.append("Phase II topology hybrid requires zero ports")
    if pec_mask is not None:
        reasons.append("Phase II topology hybrid requires pec_mask-free fixtures")
    if pec_occupancy is not None:
        reasons.append("Phase II topology hybrid supports dielectric-only topology (no pec_occupancy)")
    if bool(jnp.any(jnp.abs(materials.sigma) > 0.0)):
        reasons.append("Phase II topology hybrid requires zero sigma everywhere")
    return tuple(reasons)


def _inspect_topology_hybrid_support(
    sim,
    design_region: TopologyDesignRegion,
    *,
    init_density: jnp.ndarray | None = None,
    n_steps: int | None = None,
    num_periods: float = 20.0,
):
    """Inspect the bounded Phase 4C topology subset via the seam-owned materials path."""
    try:
        mat_bg = sim._resolve_material(design_region.material_bg)
        mat_fg = sim._resolve_material(design_region.material_fg)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc

    eps_bg = mat_bg.eps_r
    eps_fg = mat_fg.eps_r
    sigma_bg = mat_bg.sigma
    sigma_fg = mat_fg.sigma
    bg_has_dispersion = bool(mat_bg.debye_poles or mat_bg.lorentz_poles)
    fg_has_dispersion = bool(mat_fg.debye_poles or mat_fg.lorentz_poles)
    pec_threshold = sim._PEC_SIGMA_THRESHOLD
    bg_is_pec = sigma_bg >= pec_threshold
    fg_is_pec = sigma_fg >= pec_threshold

    grid = sim._build_grid()
    lo_idx = list(grid.position_to_index(design_region.corner_lo))
    hi_idx = list(grid.position_to_index(design_region.corner_hi))
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
            f"Design region is empty after clamping to interior (lo_idx={lo_idx}, hi_idx={hi_idx})."
        )

    filt_r = design_region.effective_filter_radius
    filter_radius_cells = None if filt_r is None else filt_r / grid.dx
    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)
    density = (
        0.5 * jnp.ones(design_shape, dtype=jnp.float32)
        if init_density is None
        else init_density
    )
    fields = density_to_material_fields(
        density,
        eps_bg,
        eps_fg,
        filter_radius_cells=filter_radius_cells,
        beta=design_region.beta_projection,
        sigma_bg=sigma_bg,
        sigma_fg=sigma_fg,
        pec_bg=bg_is_pec,
        pec_fg=fg_is_pec,
    )
    materials, pec_occupancy = _build_topology_material_state(
        base_materials.eps_r,
        base_materials.sigma,
        base_materials.mu_r,
        lo_idx,
        hi_idx,
        fields,
    )
    resolved_n_steps = n_steps if n_steps is not None else grid.num_timesteps(num_periods=num_periods)
    inputs = sim._build_hybrid_phase1_inputs_from_materials(
        grid,
        materials,
        debye_spec,
        lorentz_spec,
        n_steps=resolved_n_steps,
        pec_mask=base_pec_mask,
        pec_occupancy=pec_occupancy,
    )
    report = sim.inspect_hybrid_phase1_from_inputs(inputs)
    extra_reasons = _topology_hybrid_reason_texts(
        sim,
        bg_has_dispersion=bg_has_dispersion,
        fg_has_dispersion=fg_has_dispersion,
        debye_spec=debye_spec,
        lorentz_spec=lorentz_spec,
        materials=materials,
        pec_mask=base_pec_mask,
        pec_occupancy=pec_occupancy,
        port_metadata=report.port_metadata,
    )
    if extra_reasons:
        all_reasons = tuple(dict.fromkeys(report.reasons + extra_reasons))
        report = replace(report, supported=False, reasons=all_reasons, inventory=None)
    return inputs, report, grid, lo_idx, hi_idx, filter_radius_cells, base_materials, debye_spec, lorentz_spec, fields


def inspect_topology_hybrid_support(
    sim,
    design_region: TopologyDesignRegion,
    *,
    init_density: jnp.ndarray | None = None,
    n_steps: int | None = None,
    num_periods: float = 20.0,
):
    """Inspect whether a topology fixture fits the bounded Phase 4C hybrid subset."""
    _, report, *_ = _inspect_topology_hybrid_support(
        sim,
        design_region,
        init_density=init_density,
        n_steps=n_steps,
        num_periods=num_periods,
    )
    return report


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
    adjoint_mode: str = "pure_ad",
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
    adjoint_mode : {"pure_ad", "hybrid", "auto"}
        Forward/adjoint routing policy for each topology iteration.
        ``pure_ad`` preserves the current default behavior.
        ``hybrid`` requires the bounded Phase 4C topology hybrid subset to
        pass support inspection and raises otherwise.
        ``auto`` uses the hybrid seam only when the Phase 4C support
        inspection passes and otherwise falls back to the current pure-AD path.

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

    # Resolve material permittivities and conductivities.
    # _resolve_material checks user-registered materials (sim._materials)
    # first, then falls back to MATERIAL_LIBRARY.
    try:
        mat_bg = sim._resolve_material(design_region.material_bg)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    try:
        mat_fg = sim._resolve_material(design_region.material_fg)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    eps_bg = mat_bg.eps_r
    eps_fg = mat_fg.eps_r
    sigma_bg = mat_bg.sigma
    sigma_fg = mat_fg.sigma
    pec_threshold = sim._PEC_SIGMA_THRESHOLD
    bg_is_pec = sigma_bg >= pec_threshold
    fg_is_pec = sigma_fg >= pec_threshold

    # Build grid and compute design region indices
    grid = sim._build_grid()
    lo_idx = list(grid.position_to_index(design_region.corner_lo))
    hi_idx = list(grid.position_to_index(design_region.corner_hi))

    # Clamp indices to the interior region (exclude CPML padding).
    # Without this, a design region at the domain edge can overlap
    # with CPML cells, causing shape mismatches and incorrect gradients.
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

    # Compute filter radius in cells
    filt_r = design_region.effective_filter_radius
    filter_radius_cells = None
    if filt_r is not None:
        filter_radius_cells = filt_r / grid.dx

    base_materials, debye_spec, lorentz_spec, base_pec_mask, _, _ = sim._assemble_materials(grid)
    base_eps_r = base_materials.eps_r
    base_sigma = base_materials.sigma
    base_mu_r = base_materials.mu_r

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

    n_steps = grid.num_timesteps(num_periods=20.0)
    objective_accepts_ntff_box = "ntff_box" in inspect.signature(objective).parameters
    hybrid_context = None
    if adjoint_mode not in {"pure_ad", "hybrid", "auto"}:
        raise ValueError(
            f"adjoint_mode must be 'pure_ad', 'hybrid', or 'auto', got {adjoint_mode!r}"
        )
    if adjoint_mode != "pure_ad":
        inputs, report, *_ = _inspect_topology_hybrid_support(
            sim,
            design_region,
            init_density=init_density,
            n_steps=n_steps,
        )
        if report.supported:
            hybrid_context = sim.build_hybrid_phase1_context_from_inputs(inputs)
        elif adjoint_mode == "hybrid":
            raise ValueError(report.reason_text)

    def forward(logit_param, beta):
        """Forward pass: logit -> material fields -> simulation -> objective."""
        rho = jax.nn.sigmoid(logit_param)
        fields = density_to_material_fields(
            rho,
            eps_bg,
            eps_fg,
            filter_radius_cells=filter_radius_cells,
            beta=beta,
            sigma_bg=sigma_bg,
            sigma_fg=sigma_fg,
            pec_bg=bg_is_pec,
            pec_fg=fg_is_pec,
        )

        si, sj, sk = lo_idx
        ei, ej, ek = hi_idx
        eps_r = base_eps_r.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.eps)
        sigma = base_sigma.at[si:ei+1, sj:ej+1, sk:ek+1].set(fields.sigma)

        if hybrid_context is not None:
            result = sim.forward_hybrid_phase1_from_context(hybrid_context, eps_override=eps_r)
        else:
            materials, pec_occupancy = _build_topology_material_state(
                base_eps_r,
                base_sigma,
                base_mu_r,
                lo_idx,
                hi_idx,
                fields,
            )
            result = sim._forward_from_materials(
                grid,
                materials,
                debye_spec,
                lorentz_spec,
                n_steps=n_steps,
                checkpoint=True,
                pec_mask=base_pec_mask,
                pec_occupancy=pec_occupancy,
            )
        return _call_topology_objective(
            objective,
            result,
            accepts_ntff_box=objective_accepts_ntff_box,
        )

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
            rho_raw = jax.nn.sigmoid(logit)
            rho_proj = apply_projection(rho_raw, beta)
            # Binarization: 0 = fully binary, 1 = maximally intermediate
            binarization = float(jnp.mean(4.0 * rho_proj * (1.0 - rho_proj)))
            print(
                f"  iter {it:4d}  loss = {loss_val:.6e}  "
                f"beta = {beta:.1f}  binarization = {binarization:.3f}"
            )

    # Final density and eps
    final_rho = jax.nn.sigmoid(logit)
    final_fields = density_to_material_fields(
        final_rho,
        eps_bg,
        eps_fg,
        filter_radius_cells=filter_radius_cells,
        beta=_get_beta(n_iterations - 1, beta_schedule),
        sigma_bg=sigma_bg,
        sigma_fg=sigma_fg,
        pec_bg=bg_is_pec,
        pec_fg=fg_is_pec,
    )
    final_eps = final_fields.eps

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
        pec_occupancy_design=final_fields.pec_occupancy,
    )
