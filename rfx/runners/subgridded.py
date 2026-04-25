"""Subgridded (SBP-SAT) run path extracted from Simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.subgridding.sbp_sat_3d import phase1_3d_dt


_AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
_STRICT_INTERIOR_MESSAGE = (
    "private SBP-SAT benchmark flux planes must be fine-owned "
    "strict-interior placements"
)


@dataclass(frozen=True)
class _BenchmarkFluxPlaneRequest:
    """Private benchmark-only flux plane request for the SBP-SAT lane.

    This deliberately mirrors only the minimum geometry needed by tests and
    internal benchmarks.  It is not accepted by ``Simulation.run()`` and does
    not widen the public ``add_flux_monitor`` / ``add_dft_plane_probe`` API.
    """

    name: str
    axis: str
    coordinate: float
    freqs: object
    size: tuple[float, float]
    center: tuple[float, float]
    window: str = "rect"
    window_alpha: float = 0.25


class _BenchmarkFluxRun(NamedTuple):
    """Private benchmark run result returned only by helper-level tests."""

    result: object
    benchmark_flux_planes: tuple


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, str):
        try:
            return _AXIS_TO_INDEX[axis.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported benchmark flux axis {axis!r}") from exc
    axis_i = int(axis)
    if axis_i not in (0, 1, 2):
        raise ValueError(f"unsupported benchmark flux axis {axis!r}")
    return axis_i


def _local_strict_normal_index(
    *,
    coordinate: float,
    offset: float,
    dx: float,
    n_cells: int,
) -> int:
    idx = int(round((float(coordinate) - float(offset)) / float(dx)))
    if not (1 <= idx <= n_cells - 2):
        raise ValueError(
            f"{_STRICT_INTERIOR_MESSAGE}; local normal index {idx} is outside "
            f"the accepted range 1..{n_cells - 2}"
        )
    return idx


def _local_strict_tangential_bounds(
    *,
    center: float,
    size: float,
    offset: float,
    dx: float,
    n_cells: int,
    label: str,
) -> tuple[int, int]:
    if size is None or center is None:
        raise ValueError(
            f"{_STRICT_INTERIOR_MESSAGE}; finite {label} size and center are required"
        )
    size = float(size)
    center = float(center)
    if not (np.isfinite(size) and np.isfinite(center) and size > 0.0):
        raise ValueError(
            f"{_STRICT_INTERIOR_MESSAGE}; finite positive {label} size is required"
        )

    lo = int(round((center - 0.5 * size - offset) / dx))
    hi = int(round((center + 0.5 * size - offset) / dx))
    if not (1 <= lo < hi <= n_cells - 1):
        raise ValueError(
            f"{_STRICT_INTERIOR_MESSAGE}; {label} tangential bounds "
            f"{lo}:{hi} must stay inside 1..{n_cells - 1}"
        )
    return lo, hi


def _build_benchmark_flux_plane_specs(
    requests: tuple[_BenchmarkFluxPlaneRequest, ...] | list[_BenchmarkFluxPlaneRequest],
    *,
    shape_f: tuple[int, int, int],
    offsets: tuple[float, float, float],
    dx_f: float,
    n_steps: int,
):
    """Validate and lower private benchmark requests to JIT plane specs.

    The placement contract intentionally fails closed: planes must be owned by
    the fine grid, must not sit on the first/last normal slice, and must keep
    their finite aperture away from tangential edges so no SBP-SAT interface or
    corner samples enter the benchmark accumulator.
    """

    from rfx.subgridding.jit_runner import _BenchmarkFluxPlaneSpec

    specs = []
    for request in requests:
        axis_i = _axis_index(request.axis)
        tangential_axes = tuple(ax for ax in range(3) if ax != axis_i)
        idx = _local_strict_normal_index(
            coordinate=request.coordinate,
            offset=offsets[axis_i],
            dx=dx_f,
            n_cells=shape_f[axis_i],
        )
        if request.size is None or request.center is None:
            raise ValueError(
                f"{_STRICT_INTERIOR_MESSAGE}; finite tangential size and "
                "center are required"
            )
        if len(request.size) != 2 or len(request.center) != 2:
            raise ValueError(
                f"{_STRICT_INTERIOR_MESSAGE}; tangential size and center "
                "must contain exactly two coordinates"
            )
        bounds = [
            _local_strict_tangential_bounds(
                center=float(request.center[t_i]),
                size=float(request.size[t_i]),
                offset=offsets[t_axis],
                dx=dx_f,
                n_cells=shape_f[t_axis],
                label=f"axis {'xyz'[t_axis]}",
            )
            for t_i, t_axis in enumerate(tangential_axes)
        ]
        (lo1, hi1), (lo2, hi2) = bounds
        freqs = jnp.asarray(request.freqs, dtype=jnp.float64)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError(
                f"{_STRICT_INTERIOR_MESSAGE}; at least one benchmark "
                "frequency is required"
            )
        specs.append(
            _BenchmarkFluxPlaneSpec(
                name=str(request.name),
                axis=axis_i,
                index=idx,
                freqs=freqs,
                dx=float(dx_f),
                total_steps=int(n_steps),
                window=str(request.window),
                window_alpha=float(request.window_alpha),
                lo1=int(lo1),
                hi1=int(hi1),
                lo2=int(lo2),
                hi2=int(hi2),
            )
        )
    return tuple(specs)


def run_subgridded_path(
    sim,
    grid_coarse,
    base_materials_coarse,
    pec_mask_coarse,
    n_steps,
):
    """Run the canonical experimental SBP-SAT subgridding path (JIT-compiled).

    Parameters
    ----------
    sim : Simulation
        The Simulation instance (read-only access to its fields).
    grid_coarse : Grid
        Coarse uniform grid.
    base_materials_coarse : MaterialArrays
        Material arrays on the coarse grid.
    pec_mask_coarse : jnp.ndarray or None
        PEC mask on the coarse grid.
    n_steps : int
        Number of timesteps.

    Returns
    -------
    Result
    """
    return _run_subgridded_path_impl(
        sim,
        grid_coarse,
        base_materials_coarse,
        pec_mask_coarse,
        n_steps,
    ).result


def _run_subgridded_path_impl(
    sim,
    grid_coarse,
    base_materials_coarse,
    pec_mask_coarse,
    n_steps,
    *,
    _benchmark_flux_planes: tuple[_BenchmarkFluxPlaneRequest, ...] | None = None,
) -> _BenchmarkFluxRun:
    """Internal implementation shared by public and benchmark-only paths."""

    from rfx.api import Result
    from rfx.subgridding.face_ops import build_zface_ops
    from rfx.subgridding.sbp_sat_3d import SubgridConfig3D
    from rfx.subgridding.jit_runner import run_subgridded_jit as _run_sg

    if hasattr(sim, "_validate_phase1_subgrid_feature_surface"):
        sim._validate_phase1_subgrid_feature_surface()
    else:
        if hasattr(sim, "_validate_phase1_subgrid_boundaries"):
            sim._validate_subgrid_boundary_mode()
        elif sim._boundary != "pec":
            raise ValueError(
                "SBP-SAT subgridding supports boundary='pec' only in the legacy path"
            )
        if getattr(sim, "_coaxial_ports", None):
            raise ValueError(
                "Phase-1 SBP-SAT z-slab subgridding does not support coaxial ports"
            )
        if any(pe.impedance != 0.0 or pe.extent is not None for pe in sim._ports):
            raise ValueError(
                "Phase-1 SBP-SAT z-slab subgridding supports soft point sources only; "
                "impedance point ports and wire/extent ports are deferred"
            )

    ref = sim._refinement
    ratio = ref["ratio"]
    z_lo, z_hi = ref["z_range"]
    tau = ref.get("tau", 0.5)
    dx_c = grid_coarse.dx
    dx_f = dx_c / ratio
    if ref.get("xy_margin") is not None:
        raise ValueError(
            "Phase-1 SBP-SAT z-slab subgridding does not support xy_margin"
        )

    def _range_to_indices(axis_range, n_cells, pad_lo, pad_hi, label):
        interior_lo = int(pad_lo)
        interior_hi = int(n_cells - pad_hi)
        if axis_range is None:
            lo_i, hi_i = interior_lo, interior_hi
        else:
            lo, hi = axis_range
            lo_i = max(int(round(lo / dx_c)) + interior_lo, interior_lo)
            hi_i = min(int(round(hi / dx_c)) + 1 + interior_lo, interior_hi)
        if hi_i <= lo_i:
            raise ValueError(f"{label}={axis_range} maps to an empty coarse interval")
        return lo_i, hi_i

    fi_lo, fi_hi = _range_to_indices(
        ref.get("x_range"), grid_coarse.nx,
        grid_coarse.pad_x_lo, grid_coarse.pad_x_hi, "x_range"
    )
    fj_lo, fj_hi = _range_to_indices(
        ref.get("y_range"), grid_coarse.ny,
        grid_coarse.pad_y_lo, grid_coarse.pad_y_hi, "y_range"
    )
    fk_lo, fk_hi = _range_to_indices(
        ref["z_range"], grid_coarse.nz,
        grid_coarse.pad_z_lo, grid_coarse.pad_z_hi, "z_range"
    )

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    dt = phase1_3d_dt(dx_f)

    config = SubgridConfig3D(
        nx_c=grid_coarse.nx, ny_c=grid_coarse.ny, nz_c=grid_coarse.nz,
        dx_c=dx_c,
        fi_lo=fi_lo, fi_hi=fi_hi,
        fj_lo=fj_lo, fj_hi=fj_hi,
        fk_lo=fk_lo, fk_hi=fk_hi,
        nx_f=nx_f, ny_f=ny_f, nz_f=nz_f,
        dx_f=dx_f, dt=float(dt), ratio=ratio, tau=tau,
        face_ops=build_zface_ops((fi_hi - fi_lo, fj_hi - fj_lo), ratio, dx_c),
    )

    overlap = (slice(fi_lo, fi_hi), slice(fj_lo, fj_hi), slice(fk_lo, fk_hi))
    mats_c = base_materials_coarse._replace(
        eps_r=base_materials_coarse.eps_r.at[overlap].set(1.0),
        sigma=base_materials_coarse.sigma.at[overlap].set(0.0),
        mu_r=base_materials_coarse.mu_r.at[overlap].set(1.0),
    )
    pec_mask_c = pec_mask_coarse
    if pec_mask_c is not None:
        pec_mask_c = pec_mask_c.at[overlap].set(False)

    # Build fine-grid materials by rasterizing geometry at fine resolution
    shape_f = (nx_f, ny_f, nz_f)

    # Create a Grid for fine region (for position_to_index utility)
    fine_domain = (nx_f * dx_f, ny_f * dx_f, nz_f * dx_f)
    fine_grid = Grid(
        freq_max=sim._freq_max,
        domain=fine_domain,
        dx=dx_f,
        cpml_layers=0,
    )
    # Override shape to match exactly (Grid may add +1 rounding)
    fine_grid._shape_override = shape_f

    # Rasterize geometry into fine grid materials using shared function.
    # Uses cell-center coordinates (not cell edges) for correct placement.
    x_off = (fi_lo - grid_coarse.pad_x_lo) * dx_c
    y_off = (fj_lo - grid_coarse.pad_y_lo) * dx_c
    z_off = (fk_lo - grid_coarse.pad_z_lo) * dx_c

    from rfx.geometry.rasterize import coords_from_fine_grid, rasterize_geometry

    coords_f = coords_from_fine_grid(nx_f, ny_f, nz_f, dx_f, x_off, y_off, z_off)
    mats_f, _, _, pec_mask_f, _, _ = rasterize_geometry(
        sim._geometry,
        sim._resolve_material,
        coords_f,
        pec_sigma_threshold=sim._PEC_SIGMA_THRESHOLD,
    )
    has_pec_f = bool(jnp.any(pec_mask_f)) if pec_mask_f is not None else False

    def _pos_to_fine_idx(pos):
        idx = (
            int(round((pos[0] - x_off) / dx_f)),
            int(round((pos[1] - y_off) / dx_f)),
            int(round((pos[2] - z_off) / dx_f)),
        )
        if not (0 <= idx[0] < nx_f and 0 <= idx[1] < ny_f and 0 <= idx[2] < nz_f):
            raise ValueError(
                f"Position {pos} maps to fine-grid index {idx} outside "
                f"the SBP-SAT fine grid shape ({nx_f}, {ny_f}, {nz_f}). "
                "Adjust x_range/y_range/z_range to cover all sources and probes."
            )
        return idx

    # Build sources on fine grid
    sources_f = []
    times = jnp.arange(n_steps, dtype=jnp.float32) * dt

    for pe in sim._ports:
        # Phase 1 supports soft point sources only; impedance and wire
        # ports are rejected before this runner is entered.
        idx = _pos_to_fine_idx(pe.position)
        i, j, k = idx
        waveform = jax.vmap(pe.waveform)(times)
        sources_f.append((i, j, k, pe.component, np.array(waveform)))

    # Build probes on fine grid
    probe_indices_f = []
    probe_components = []
    for pe in sim._probes:
        idx = _pos_to_fine_idx(pe.position)
        probe_indices_f.append(idx)
        probe_components.append(pe.component)

    benchmark_flux_specs = ()
    if _benchmark_flux_planes:
        benchmark_flux_specs = _build_benchmark_flux_plane_specs(
            tuple(_benchmark_flux_planes),
            shape_f=shape_f,
            offsets=(x_off, y_off, z_off),
            dx_f=dx_f,
            n_steps=n_steps,
        )

    result = _run_sg(
        grid_coarse,
        mats_c,
        mats_f,
        config,
        n_steps,
        pec_mask_c=pec_mask_c,
        pec_mask_f=pec_mask_f if has_pec_f else None,
        sources_f=sources_f,
        probe_indices_f=probe_indices_f,
        probe_components=probe_components,
        outer_pec_faces=frozenset(sim._boundary_spec.pec_faces()),
        outer_pmc_faces=frozenset(sim._boundary_spec.pmc_faces()),
        periodic=tuple(axis in (sim._periodic_axes or "") for axis in "xyz"),
        fine_periodic=tuple(
            axis in (sim._periodic_axes or "") and lo == pad_lo and hi == n - pad_hi
            for axis, (lo, hi, n, pad_lo, pad_hi) in zip(
                "xyz",
                (
                    (fi_lo, fi_hi, grid_coarse.nx, grid_coarse.pad_x_lo, grid_coarse.pad_x_hi),
                    (fj_lo, fj_hi, grid_coarse.ny, grid_coarse.pad_y_lo, grid_coarse.pad_y_hi),
                    (fk_lo, fk_hi, grid_coarse.nz, grid_coarse.pad_z_lo, grid_coarse.pad_z_hi),
                ),
            )
        ),
        absorber_boundary=sim._boundary,
        _benchmark_flux_planes=benchmark_flux_specs,
    )

    public_result = Result(
        state=result.state_f,
        time_series=result.time_series,
        s_params=None,
        freqs=None,
        grid=fine_grid,
        dt=dt,
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
    return _BenchmarkFluxRun(
        result=public_result,
        benchmark_flux_planes=tuple(result.benchmark_flux_planes or ()),
    )


def run_subgridded_benchmark_flux(
    sim,
    *,
    n_steps: int,
    planes: tuple[_BenchmarkFluxPlaneRequest, ...] | list[_BenchmarkFluxPlaneRequest],
) -> _BenchmarkFluxRun:
    """Run the private SBP-SAT benchmark-only flux accumulator path.

    Public DFT/flux requests still fail in the regular API validator.  This
    helper accepts only private fine-owned plane requests and returns private
    raw accumulators alongside the ordinary public ``Result`` to prove the
    benchmark does not leak into ``Result.dft_planes`` or
    ``Result.flux_monitors``.
    """

    if sim._dx is None and sim._geometry:
        sim._auto_configure_mesh()
    sim._validate_mesh_quality()
    sim._validate_simulation_config()
    if sim._refinement is None:
        raise ValueError("private SBP-SAT benchmark flux requires refinement")

    grid = sim._build_grid()
    base_materials, _, _, pec_mask, _, _ = sim._assemble_materials(grid)
    return _run_subgridded_path_impl(
        sim,
        grid,
        base_materials,
        pec_mask,
        int(n_steps),
        _benchmark_flux_planes=tuple(planes),
    )
