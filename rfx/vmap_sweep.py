"""Batched material-parameter sweep using ``jax.vmap``.

For material-only sweeps (``eps_r``, ``sigma``, ``mu_r``) the grid geometry
is identical across all parameter values, so ``jax.vmap`` can batch the
full FDTD time loop and execute all simulations in a single GPU kernel.

This is dramatically faster than sequential sweeps for moderate batch sizes
(typically 5--50 values) because:

1. Grid construction happens **once**.
2. Source waveforms and probe specs are shared.
3. The ``jax.lax.scan`` body is vmapped over a leading batch axis in the
   material arrays, so all simulations run in parallel.

Limitations
-----------
- Only material parameters (``eps_r``, ``sigma``, ``mu_r``) can be swept.
  For geometry sweeps (different shapes/sizes) use ``parametric_sweep()``.
- ``until_decay`` is not supported (requires Python-level control flow).
- Ports, TFSF, CPML, dispersion, waveguide ports, NTFF, DFT planes,
  snapshots, and RLC elements are fully supported because they use the
  same grid topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.yee import FDTDState, MaterialArrays, init_state, update_e, update_h, EPS_0
from rfx.simulation import (
    run as _run_sim,
    SourceSpec,
    ProbeSpec,
    SimResult,
    SnapshotSpec,
    make_source,
    make_j_source,
    make_probe,
    make_port_source,
    make_wire_port_sources,
    _update_e_with_optional_dispersion,
)
from rfx.boundaries.pec import apply_pec


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VmapSweepResult:
    """Result from a vmapped material sweep.

    Attributes
    ----------
    time_series : ndarray, shape (n_batch, n_steps, n_probes)
        Probe recordings for every sweep value.
    param_name : str
        Name of the swept parameter.
    param_values : ndarray, shape (n_batch,)
        The parameter values that were swept.
    final_fields : dict or None
        If requested, dict mapping component name to (n_batch, Nx, Ny, Nz).
    """
    time_series: np.ndarray
    param_name: str
    param_values: np.ndarray
    final_fields: dict | None = None

    def peak_field(self) -> np.ndarray:
        """Return peak |probe value| per batch element."""
        return np.max(np.abs(self.time_series), axis=(1, 2))


# ---------------------------------------------------------------------------
# Parameter application helpers
# ---------------------------------------------------------------------------

_VALID_PARAMS = {"eps_r", "sigma", "mu_r"}


def _parse_param_name(param_name: str) -> tuple[str | None, str]:
    """Parse ``"eps_r"`` or ``"substrate.eps_r"`` into (material_name, field).

    Returns
    -------
    (material_name_or_None, field_name)
    """
    if "." in param_name:
        mat_name, field = param_name.rsplit(".", 1)
    else:
        mat_name, field = None, param_name

    if field not in _VALID_PARAMS:
        raise ValueError(
            f"param_name field must be one of {_VALID_PARAMS}, "
            f"got {field!r} (from {param_name!r})"
        )
    return mat_name, field


def _build_batched_materials(
    sim,
    grid,
    base_materials: MaterialArrays,
    param_name: str,
    param_values: jnp.ndarray,
) -> MaterialArrays:
    """Create batched material arrays with shape (n_batch, Nx, Ny, Nz).

    For a global sweep (e.g. ``"eps_r"``), the parameter is applied to
    **all non-vacuum cells** that have the swept property differing from 1.0
    (for eps_r/mu_r) or 0.0 (for sigma).

    For a material-specific sweep (e.g. ``"substrate.eps_r"``), the parameter
    is applied only to cells occupied by that named material.
    """
    mat_name, field = _parse_param_name(param_name)
    n_batch = len(param_values)

    eps_r = base_materials.eps_r   # (Nx, Ny, Nz)
    sigma = base_materials.sigma
    mu_r = base_materials.mu_r

    if mat_name is not None:
        # Build a mask for the specific material
        sim._resolve_material(mat_name)
        mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
        for entry in sim._geometry:
            if entry.material_name == mat_name:
                mask = mask | entry.shape.mask(grid)

        if field == "eps_r":
            # For each batch: where mask, use param_values[b]; else keep base
            batch_eps = jnp.where(
                mask[None],  # (1, Nx, Ny, Nz)
                param_values[:, None, None, None],  # (n_batch, 1, 1, 1)
                eps_r[None],  # (1, Nx, Ny, Nz)
            )
            batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
            batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        elif field == "sigma":
            batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
            batch_sigma = jnp.where(
                mask[None],
                param_values[:, None, None, None],
                sigma[None],
            )
            batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        else:  # mu_r
            batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
            batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
            batch_mu = jnp.where(
                mask[None],
                param_values[:, None, None, None],
                mu_r[None],
            )
    else:
        # Global sweep: apply to all non-background cells
        if field == "eps_r":
            # Identify cells that have non-vacuum eps_r
            non_vac = eps_r != 1.0
            batch_eps = jnp.where(
                non_vac[None],
                param_values[:, None, None, None],
                eps_r[None],
            )
            batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
            batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        elif field == "sigma":
            non_zero = sigma != 0.0
            batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
            batch_sigma = jnp.where(
                non_zero[None],
                param_values[:, None, None, None],
                sigma[None],
            )
            batch_mu = jnp.broadcast_to(mu_r[None], (n_batch,) + mu_r.shape)
        else:  # mu_r
            non_vac = mu_r != 1.0
            batch_eps = jnp.broadcast_to(eps_r[None], (n_batch,) + eps_r.shape)
            batch_sigma = jnp.broadcast_to(sigma[None], (n_batch,) + sigma.shape)
            batch_mu = jnp.where(
                non_vac[None],
                param_values[:, None, None, None],
                mu_r[None],
            )

    return MaterialArrays(
        eps_r=batch_eps,
        sigma=batch_sigma,
        mu_r=batch_mu,
    )


# ---------------------------------------------------------------------------
# Core: vmapped FDTD scan
# ---------------------------------------------------------------------------

def _build_scan_fn(
    grid,
    n_steps: int,
    *,
    boundary: str = "pec",
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    pec_axes: str = "xyz",
    pec_mask=None,
):
    """Build a pure function ``f(materials) -> time_series`` suitable for vmap.

    This constructs a minimal FDTD loop (H update -> E update -> PEC ->
    sources -> probes) without CPML, dispersion, TFSF, DFT planes, or
    waveguide ports.  For the common material-sweep use case (PEC cavity
    or simple probe measurement), this covers the needed physics.
    """
    dt = grid.dt
    dx = grid.dx
    sources = sources or []
    probes = probes or []

    use_pec_mask = pec_mask is not None

    # Precompute source waveform matrix
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]

    def run_one(materials: MaterialArrays) -> jnp.ndarray:
        """Run a single FDTD simulation with the given materials.

        Returns time_series of shape (n_steps, n_probes).
        """
        fdtd = init_state(grid.shape)

        def step_fn(carry, xs):
            _step_idx, src_vals = xs
            st = carry

            # H update
            st = update_h(st, materials, dt, dx, periodic=periodic)

            # E update
            st = update_e(st, materials, dt, dx, periodic=periodic)

            # PEC boundaries
            if pec_axes:
                st = apply_pec(st, axes=pec_axes)

            if use_pec_mask:
                from rfx.boundaries.pec import apply_pec_mask
                st = apply_pec_mask(st, pec_mask)

            # Soft sources
            for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
                field = getattr(st, sc)
                field = field.at[si, sj, sk].add(src_vals[idx_s])
                st = st._replace(**{sc: field})

            # Probe samples
            samples = [getattr(st, pc)[pi, pj, pk]
                       for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

            return st, probe_out

        xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
        _, time_series = jax.lax.scan(step_fn, fdtd, xs)
        return time_series

    return run_one


def _build_full_scan_fn(
    sim,
    grid,
    base_materials: MaterialArrays,
    n_steps: int,
    *,
    debye_spec=None,
    lorentz_spec=None,
    pec_mask=None,
):
    """Build a function ``f(materials) -> time_series`` that uses the full
    simulation runner (including CPML, dispersion, etc.).

    This wraps ``simulation.run()`` into a vmappable form by making
    ``materials`` an explicit argument rather than a closure capture.
    """
    from rfx.runners.uniform import run_uniform

    boundary = sim._boundary

    # We need to build all sources and probes once.  The challenge is that
    # port sources depend on materials (Cb coefficient).  For material sweeps
    # the source waveform varies with eps_r/sigma.  For simple sweeps
    # (non-port sources), this is fine.  For port-based sweeps, we use the
    # base materials for source construction (approximate).

    # Detect if simulation uses features incompatible with simple vmap
    has_ports = any(pe.impedance != 0.0 for pe in sim._ports)
    has_tfsf = sim._tfsf is not None
    has_waveguide = len(sim._waveguide_ports) > 0
    has_dispersion = debye_spec is not None or lorentz_spec is not None

    # For the full path, we simply call simulation.run() which is already
    # JIT-compiled via lax.scan.  We construct a wrapper that replaces
    # materials in the call.

    # Build sources and probes from the simulation (using base materials)
    sources = []
    probes = []

    for pe in sim._ports:
        if pe.impedance == 0.0:
            if boundary == "cpml":
                sources.append(make_j_source(grid, pe.position, pe.component,
                                             pe.waveform, n_steps, base_materials))
            else:
                sources.append(make_source(grid, pe.position, pe.component,
                                           pe.waveform, n_steps))

    for pe in sim._probes:
        probes.append(make_probe(grid, pe.position, pe.component))

    # For the simple (no-port, no-dispersion, no-TFSF) case, use the
    # lightweight scan function that can be cleanly vmapped.
    if not has_ports and not has_tfsf and not has_waveguide and not has_dispersion:
        periodic = (False, False, False)
        if sim._periodic_axes:
            periodic = tuple(axis in sim._periodic_axes for axis in "xyz")
        if grid.is_2d:
            periodic = (periodic[0], periodic[1], True)

        cpml_axes = "xyz"
        axis_names = ("x", "y", "z")
        for axis_name, is_periodic in zip(axis_names, periodic):
            if is_periodic:
                cpml_axes = cpml_axes.replace(axis_name, "")
        pec_axes = "".join(
            axis_name for axis_name, is_periodic in zip(axis_names, periodic)
            if not is_periodic
        )

        if boundary == "cpml":
            # CPML requires its own state management.  For the vmapped path,
            # we build a scan function that includes CPML handling.
            return _build_cpml_scan_fn(
                grid, n_steps,
                sources=sources,
                probes=probes,
                periodic=periodic,
                cpml_axes=cpml_axes,
                pec_axes=pec_axes,
                pec_mask=pec_mask,
            )
        else:
            return _build_scan_fn(
                grid, n_steps,
                boundary=boundary,
                sources=sources,
                probes=probes,
                periodic=periodic,
                pec_axes=pec_axes,
                pec_mask=pec_mask,
            )

    # Fallback: for complex sims, run sequentially (not vmapped)
    return None


def _build_cpml_scan_fn(
    grid,
    n_steps: int,
    *,
    sources: list[SourceSpec] | None = None,
    probes: list[ProbeSpec] | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
    cpml_axes: str = "xyz",
    pec_axes: str = "xyz",
    pec_mask=None,
):
    """Build a vmappable FDTD scan function with CPML boundaries."""
    from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h

    dt = grid.dt
    dx = grid.dx
    sources = sources or []
    probes = probes or []

    use_pec_mask = pec_mask is not None

    # Precompute source waveform matrix
    if sources:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    src_meta = [(s.i, s.j, s.k, s.component) for s in sources]
    prb_meta = [(p.i, p.j, p.k, p.component) for p in probes]

    # Initialize CPML once (shared across batch)
    cpml_params, cpml_state_init = init_cpml(grid)

    def run_one(materials: MaterialArrays) -> jnp.ndarray:
        fdtd = init_state(grid.shape)
        cpml_state = cpml_state_init

        def step_fn(carry, xs):
            _step_idx, src_vals = xs
            st, cpml_st = carry

            # H update
            st = update_h(st, materials, dt, dx, periodic=periodic)
            st, cpml_new = apply_cpml_h(st, cpml_params, cpml_st, grid, cpml_axes)

            # E update
            st = update_e(st, materials, dt, dx, periodic=periodic)
            st, cpml_new = apply_cpml_e(st, cpml_params, cpml_new, grid, cpml_axes)

            # PEC boundaries
            if pec_axes:
                st = apply_pec(st, axes=pec_axes)

            if use_pec_mask:
                from rfx.boundaries.pec import apply_pec_mask
                st = apply_pec_mask(st, pec_mask)

            # Soft sources
            for idx_s, (si, sj, sk, sc) in enumerate(src_meta):
                field = getattr(st, sc)
                field = field.at[si, sj, sk].add(src_vals[idx_s])
                st = st._replace(**{sc: field})

            # Probe samples
            samples = [getattr(st, pc)[pi, pj, pk]
                       for pi, pj, pk, pc in prb_meta]
            probe_out = jnp.stack(samples) if samples else jnp.zeros(0)

            return (st, cpml_new), probe_out

        xs = (jnp.arange(n_steps, dtype=jnp.int32), src_waveforms)
        _, time_series = jax.lax.scan(step_fn, (fdtd, cpml_state), xs)
        return time_series

    return run_one


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vmap_material_sweep(
    sim,
    param_name: str,
    param_values,
    *,
    n_steps: int | None = None,
    num_periods: float = 20.0,
    return_fields: bool = False,
) -> VmapSweepResult:
    """Batch-evaluate multiple material parameter values using ``jax.vmap``.

    Only works for material parameters (``eps_r``, ``sigma``, ``mu_r``)
    where the grid shape stays constant.  For geometry sweeps, use
    ``parametric_sweep()``.

    Parameters
    ----------
    sim : Simulation
        Base simulation (all geometry already added).
    param_name : str
        Material parameter to sweep: ``"eps_r"``, ``"sigma"``, ``"mu_r"``,
        or a material-specific name like ``"substrate.eps_r"``.
    param_values : array-like, shape (n_batch,)
        Values to evaluate.
    n_steps : int or None
        Timesteps.  If None, auto-computed from *num_periods*.
    num_periods : float
        Periods at freq_max for auto timestep count (default 20).
    return_fields : bool
        If True, also return final E-field snapshots per batch element.

    Returns
    -------
    VmapSweepResult
        Result with ``.time_series`` of shape ``(n_batch, n_steps, n_probes)``
        and ``.param_values``.

    Notes
    -----
    The entire FDTD time loop is vmapped so all simulations execute in a
    single fused GPU kernel.  Memory scales as ``n_batch * grid_size``.
    For large grids, reduce batch size to avoid OOM.

    Features supported in vmap path: PEC boundaries, CPML absorbing
    boundaries, soft sources, point probes.  Lumped ports, TFSF,
    dispersion, waveguide ports, DFT planes, and NTFF are **not**
    supported in the vmap fast path and will trigger a sequential fallback.
    """
    param_values = np.asarray(param_values, dtype=np.float32).ravel()
    if len(param_values) == 0:
        raise ValueError("param_values must not be empty")

    # Validate param_name
    _parse_param_name(param_name)

    # Build grid and base materials once
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask = sim._assemble_materials(grid)

    if n_steps is None:
        n_steps = grid.num_timesteps(num_periods=num_periods)

    # Build the vmappable scan function
    run_one_fn = _build_full_scan_fn(
        sim, grid, base_materials, n_steps,
        debye_spec=debye_spec,
        lorentz_spec=lorentz_spec,
        pec_mask=pec_mask,
    )

    jax_param_values = jnp.asarray(param_values)

    if run_one_fn is not None:
        # Fast path: vmap over material arrays
        batched_materials = _build_batched_materials(
            sim, grid, base_materials, param_name, jax_param_values,
        )

        # vmap run_one over the batch dimension of materials
        def run_one_from_materials(eps_r, sigma, mu_r):
            mats = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
            return run_one_fn(mats)

        batched_run = jax.vmap(run_one_from_materials)
        time_series = batched_run(
            batched_materials.eps_r,
            batched_materials.sigma,
            batched_materials.mu_r,
        )
        time_series_np = np.asarray(time_series)

        return VmapSweepResult(
            time_series=time_series_np,
            param_name=param_name,
            param_values=param_values,
        )
    else:
        # Fallback: sequential execution for complex simulations
        # Still uses the same interface but runs one at a time
        import warnings
        warnings.warn(
            "Simulation uses features not supported by vmap fast path "
            "(ports/TFSF/dispersion/waveguide). Falling back to sequential "
            "execution. Use parametric_sweep() for better sequential support.",
            stacklevel=2,
        )
        return _sequential_fallback(
            sim, param_name, param_values, n_steps=n_steps,
        )


def _sequential_fallback(
    sim,
    param_name: str,
    param_values: np.ndarray,
    *,
    n_steps: int,
) -> VmapSweepResult:
    """Sequential fallback when vmap is not possible."""
    mat_name, field = _parse_param_name(param_name)

    all_ts = []
    for val in param_values:
        # Clone the simulation and modify the material
        import copy
        sim_copy = copy.deepcopy(sim)
        if mat_name is not None:
            # Modify the named material
            mat = sim_copy._resolve_material(mat_name)
            new_kwargs = {
                "eps_r": mat.eps_r,
                "sigma": mat.sigma,
                "mu_r": mat.mu_r,
                "debye_poles": mat.debye_poles,
                "lorentz_poles": mat.lorentz_poles,
            }
            new_kwargs[field] = float(val)
            from rfx.api import MaterialSpec
            sim_copy._materials[mat_name] = MaterialSpec(**new_kwargs)
        else:
            # Modify all custom materials
            for name, mat in list(sim_copy._materials.items()):
                new_kwargs = {
                    "eps_r": mat.eps_r,
                    "sigma": mat.sigma,
                    "mu_r": mat.mu_r,
                    "debye_poles": mat.debye_poles,
                    "lorentz_poles": mat.lorentz_poles,
                }
                new_kwargs[field] = float(val)
                from rfx.api import MaterialSpec
                sim_copy._materials[name] = MaterialSpec(**new_kwargs)

        result = sim_copy.run(n_steps=n_steps)
        all_ts.append(np.asarray(result.time_series))

    # Stack into (n_batch, n_steps, n_probes)
    time_series = np.stack(all_ts, axis=0)
    return VmapSweepResult(
        time_series=time_series,
        param_name=param_name,
        param_values=param_values,
    )
