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
_PRIVATE_SHEET_MESSAGE = (
    "private SBP-SAT benchmark analytic sheet sources must be fine-owned "
    "strict-interior placements"
)
_PRIVATE_TFSF_MESSAGE = (
    "private SBP-SAT benchmark TFSF-style incident fields must be fine-owned "
    "strict-interior placements"
)
_PRIVATE_PLANE_WAVE_MESSAGE = (
    "private SBP-SAT benchmark plane-wave sources must be fine-owned "
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


@dataclass(frozen=True)
class _PrivateAnalyticSheetSourceRequest:
    """Private benchmark-only analytic sheet source for SBP-SAT evidence.

    This is deliberately accepted only by ``run_subgridded_benchmark_flux``.
    It is not a public source API, not public TFSF, and not surfaced through
    ``Simulation.run()`` or ``Result``.
    """

    name: str
    axis: str
    coordinate: float
    component: str
    propagation_sign: int
    amplitude: float
    f0_hz: float
    bandwidth: float
    phase_rad: float
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    window: str = "rect"
    window_alpha: float = 0.25


@dataclass(frozen=True)
class _PrivateTFSFIncidentRequest:
    """Private benchmark-only TFSF-style incident field for SBP-SAT evidence.

    This request is deliberately narrower than public TFSF.  It is accepted
    only by ``run_subgridded_benchmark_flux`` and exists to test whether a
    paired incident E/H correction can recover private fixture quality without
    widening ``Simulation.add_tfsf_source`` or public ``Result`` surfaces.
    """

    name: str
    axis: str
    coordinate: float
    electric_component: str
    magnetic_component: str
    propagation_sign: int
    amplitude: float
    f0_hz: float
    bandwidth: float
    phase_rad: float
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    window: str = "rect"
    window_alpha: float = 0.25


@dataclass(frozen=True)
class _PrivatePlaneWaveSourceRequest:
    """Private benchmark-only uniform plane-wave source for SBP-SAT evidence.

    This request is the private W1/R1 source contract carrier.  It is accepted
    only by private benchmark helpers and is intentionally not public TFSF,
    not a public source API, and not surfaced through ``Simulation.run()`` or
    public ``Result`` fields.
    """

    name: str
    axis: str
    coordinate: float
    electric_component: str
    magnetic_component: str
    propagation_sign: int
    amplitude: float
    f0_hz: float
    bandwidth: float
    phase_rad: float
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    window: str = "rect"
    window_alpha: float = 0.25


class _BenchmarkFluxRun(NamedTuple):
    """Private benchmark run result returned only by helper-level tests."""

    result: object
    benchmark_flux_planes: tuple


_PRIVATE_REFERENCE_PUBLIC_SURFACE_MESSAGE = (
    "private SBP-SAT same-contract reference flux rejects public observables "
    "and sources"
)


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


def _local_strict_span_bounds(
    *,
    span: tuple[float, float],
    offset: float,
    dx: float,
    n_cells: int,
    label: str,
    message: str,
) -> tuple[int, int]:
    if span is None or len(span) != 2:
        raise ValueError(f"{message}; finite {label} span is required")
    lo_coord, hi_coord = float(span[0]), float(span[1])
    if not (np.isfinite(lo_coord) and np.isfinite(hi_coord) and hi_coord > lo_coord):
        raise ValueError(f"{message}; finite increasing {label} span is required")

    lo = int(round((lo_coord - offset) / dx))
    hi = int(round((hi_coord - offset) / dx))
    if not (1 <= lo < hi <= n_cells - 1):
        raise ValueError(
            f"{message}; {label} span bounds {lo}:{hi} must stay inside "
            f"1..{n_cells - 1}"
        )
    return lo, hi


def _sheet_temporal_window(
    n_steps: int,
    window: str,
    alpha: float,
) -> np.ndarray:
    window = str(window).lower()
    if window == "rect":
        return np.ones(n_steps, dtype=np.float64)
    if window == "hann":
        return np.hanning(n_steps).astype(np.float64)
    if window == "tukey":
        alpha = float(alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                "private analytic sheet source tukey alpha must be in [0, 1]"
            )
        if alpha <= 0.0:
            return np.ones(n_steps, dtype=np.float64)
        if alpha >= 1.0:
            return np.hanning(n_steps).astype(np.float64)
        x = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
        weights = np.ones(n_steps, dtype=np.float64)
        lo = x < alpha / 2.0
        hi = x >= 1.0 - alpha / 2.0
        weights[lo] = 0.5 * (1.0 + np.cos(2.0 * np.pi * (x[lo] / alpha - 0.5)))
        weights[hi] = 0.5 * (
            1.0 + np.cos(2.0 * np.pi * (x[hi] / alpha - 1.0 / alpha + 0.5))
        )
        return weights
    raise ValueError(f"unsupported private analytic sheet source window {window!r}")


def _analytic_sheet_waveform(
    request: _PrivateAnalyticSheetSourceRequest,
    *,
    dt: float,
    n_steps: int,
) -> jnp.ndarray:
    f0 = float(request.f0_hz)
    bandwidth = float(request.bandwidth)
    amplitude = float(request.amplitude)
    if not (np.isfinite(f0) and f0 > 0.0):
        raise ValueError(f"{_PRIVATE_SHEET_MESSAGE}; f0_hz must be positive")
    if not (np.isfinite(bandwidth) and bandwidth > 0.0):
        raise ValueError(f"{_PRIVATE_SHEET_MESSAGE}; bandwidth must be positive")
    if not np.isfinite(amplitude):
        raise ValueError(f"{_PRIVATE_SHEET_MESSAGE}; amplitude must be finite")

    times = np.arange(n_steps, dtype=np.float64) * float(dt)
    tau = 1.0 / (np.pi * f0 * bandwidth)
    t0 = 5.0 * tau
    envelope = np.exp(-(((times - t0) / tau) ** 2))
    carrier = np.sin(2.0 * np.pi * f0 * times + float(request.phase_rad))
    taper = _sheet_temporal_window(n_steps, request.window, request.window_alpha)
    return jnp.asarray(amplitude * carrier * envelope * taper, dtype=jnp.float32)


def _private_tfsf_electric_waveform(
    request: _PrivateTFSFIncidentRequest,
    *,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    f0 = float(request.f0_hz)
    bandwidth = float(request.bandwidth)
    amplitude = float(request.amplitude)
    if not (np.isfinite(f0) and f0 > 0.0):
        raise ValueError(f"{_PRIVATE_TFSF_MESSAGE}; f0_hz must be positive")
    if not (np.isfinite(bandwidth) and bandwidth > 0.0):
        raise ValueError(f"{_PRIVATE_TFSF_MESSAGE}; bandwidth must be positive")
    if not np.isfinite(amplitude):
        raise ValueError(f"{_PRIVATE_TFSF_MESSAGE}; amplitude must be finite")

    times = np.arange(n_steps, dtype=np.float64) * float(dt)
    tau = 1.0 / (np.pi * f0 * bandwidth)
    t0 = 5.0 * tau
    envelope = np.exp(-(((times - t0) / tau) ** 2))
    carrier = np.sin(2.0 * np.pi * f0 * times + float(request.phase_rad))
    taper = _sheet_temporal_window(n_steps, request.window, request.window_alpha)
    return amplitude * carrier * envelope * taper


def _private_plane_wave_electric_waveform(
    request: _PrivatePlaneWaveSourceRequest,
    *,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    f0 = float(request.f0_hz)
    bandwidth = float(request.bandwidth)
    amplitude = float(request.amplitude)
    if not (np.isfinite(f0) and f0 > 0.0):
        raise ValueError(f"{_PRIVATE_PLANE_WAVE_MESSAGE}; f0_hz must be positive")
    if not (np.isfinite(bandwidth) and bandwidth > 0.0):
        raise ValueError(f"{_PRIVATE_PLANE_WAVE_MESSAGE}; bandwidth must be positive")
    if not np.isfinite(amplitude):
        raise ValueError(f"{_PRIVATE_PLANE_WAVE_MESSAGE}; amplitude must be finite")

    times = np.arange(n_steps, dtype=np.float64) * float(dt)
    tau = 1.0 / (np.pi * f0 * bandwidth)
    t0 = 5.0 * tau
    envelope = np.exp(-(((times - t0) / tau) ** 2))
    carrier = np.sin(2.0 * np.pi * f0 * times + float(request.phase_rad))
    taper = _sheet_temporal_window(n_steps, request.window, request.window_alpha)
    return amplitude * carrier * envelope * taper


def _build_private_tfsf_incident_specs(
    requests: (
        tuple[_PrivateTFSFIncidentRequest, ...] | list[_PrivateTFSFIncidentRequest]
    ),
    *,
    shape_f: tuple[int, int, int],
    offsets: tuple[float, float, float],
    dx_f: float,
    dt: float,
    n_steps: int,
):
    """Validate and lower private benchmark TFSF-style incident fields."""

    from rfx.core.yee import EPS_0, MU_0
    from rfx.subgridding.jit_runner import _PrivateTFSFIncidentSpec

    specs = []
    eta0 = float(np.sqrt(MU_0 / EPS_0))
    for request in requests:
        axis_i = _axis_index(request.axis)
        if axis_i != 2:
            raise ValueError(
                f"{_PRIVATE_TFSF_MESSAGE}; only z-axis incidence is supported"
            )
        if int(request.propagation_sign) != 1:
            raise ValueError(
                f"{_PRIVATE_TFSF_MESSAGE}; only +z propagation is supported"
            )
        if request.electric_component != "ex" or request.magnetic_component != "hy":
            raise ValueError(
                f"{_PRIVATE_TFSF_MESSAGE}; only ex/hy polarization is supported"
            )

        try:
            index = _local_strict_normal_index(
                coordinate=request.coordinate,
                offset=offsets[axis_i],
                dx=dx_f,
                n_cells=shape_f[axis_i],
            )
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(_STRICT_INTERIOR_MESSAGE, _PRIVATE_TFSF_MESSAGE)
            ) from exc
        lo1, hi1 = _local_strict_span_bounds(
            span=request.x_span,
            offset=offsets[0],
            dx=dx_f,
            n_cells=shape_f[0],
            label="x",
            message=_PRIVATE_TFSF_MESSAGE,
        )
        lo2, hi2 = _local_strict_span_bounds(
            span=request.y_span,
            offset=offsets[1],
            dx=dx_f,
            n_cells=shape_f[1],
            label="y",
            message=_PRIVATE_TFSF_MESSAGE,
        )
        electric_values = _private_tfsf_electric_waveform(
            request,
            dt=float(dt),
            n_steps=int(n_steps),
        )
        magnetic_values = electric_values / eta0
        specs.append(
            _PrivateTFSFIncidentSpec(
                name=str(request.name),
                axis=axis_i,
                index=int(index),
                electric_component=str(request.electric_component),
                magnetic_component=str(request.magnetic_component),
                propagation_sign=int(request.propagation_sign),
                electric_values=jnp.asarray(electric_values, dtype=jnp.float32),
                magnetic_values=jnp.asarray(magnetic_values, dtype=jnp.float32),
                lo1=int(lo1),
                hi1=int(hi1),
                lo2=int(lo2),
                hi2=int(hi2),
            )
        )
    return tuple(specs)


def _build_private_plane_wave_source_specs(
    requests: (
        tuple[_PrivatePlaneWaveSourceRequest, ...] | list[_PrivatePlaneWaveSourceRequest]
    ),
    *,
    shape_f: tuple[int, int, int],
    offsets: tuple[float, float, float],
    dx_f: float,
    dt: float,
    n_steps: int,
):
    """Validate and lower private benchmark plane-wave sources."""

    from rfx.core.yee import EPS_0, MU_0
    from rfx.subgridding.jit_runner import _PrivatePlaneWaveSourceSpec

    specs = []
    eta0 = float(np.sqrt(MU_0 / EPS_0))
    for request in requests:
        axis_i = _axis_index(request.axis)
        if axis_i != 2:
            raise ValueError(
                f"{_PRIVATE_PLANE_WAVE_MESSAGE}; only z-axis incidence is supported"
            )
        if int(request.propagation_sign) != 1:
            raise ValueError(
                f"{_PRIVATE_PLANE_WAVE_MESSAGE}; only +z propagation is supported"
            )
        if request.electric_component != "ex" or request.magnetic_component != "hy":
            raise ValueError(
                f"{_PRIVATE_PLANE_WAVE_MESSAGE}; only ex/hy polarization is supported"
            )

        try:
            index = _local_strict_normal_index(
                coordinate=request.coordinate,
                offset=offsets[axis_i],
                dx=dx_f,
                n_cells=shape_f[axis_i],
            )
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(
                    _STRICT_INTERIOR_MESSAGE, _PRIVATE_PLANE_WAVE_MESSAGE
                )
            ) from exc
        lo1, hi1 = _local_strict_span_bounds(
            span=request.x_span,
            offset=offsets[0],
            dx=dx_f,
            n_cells=shape_f[0],
            label="x",
            message=_PRIVATE_PLANE_WAVE_MESSAGE,
        )
        lo2, hi2 = _local_strict_span_bounds(
            span=request.y_span,
            offset=offsets[1],
            dx=dx_f,
            n_cells=shape_f[1],
            label="y",
            message=_PRIVATE_PLANE_WAVE_MESSAGE,
        )
        electric_values = _private_plane_wave_electric_waveform(
            request,
            dt=float(dt),
            n_steps=int(n_steps),
        )
        magnetic_values = electric_values / eta0
        specs.append(
            _PrivatePlaneWaveSourceSpec(
                name=str(request.name),
                axis=axis_i,
                index=int(index),
                electric_component=str(request.electric_component),
                magnetic_component=str(request.magnetic_component),
                propagation_sign=int(request.propagation_sign),
                electric_values=jnp.asarray(electric_values, dtype=jnp.float32),
                magnetic_values=jnp.asarray(magnetic_values, dtype=jnp.float32),
                lo1=int(lo1),
                hi1=int(hi1),
                lo2=int(lo2),
                hi2=int(hi2),
                contract="private_uniform_plane_wave_source",
            )
        )
    return tuple(specs)


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


def _build_private_analytic_sheet_source_specs(
    requests: (
        tuple[_PrivateAnalyticSheetSourceRequest, ...]
        | list[_PrivateAnalyticSheetSourceRequest]
    ),
    *,
    shape_f: tuple[int, int, int],
    offsets: tuple[float, float, float],
    dx_f: float,
    dt: float,
    n_steps: int,
):
    """Validate and lower private benchmark analytic sheet sources."""

    from rfx.subgridding.jit_runner import _PrivateAnalyticSheetSourceSpec

    specs = []
    for request in requests:
        axis_i = _axis_index(request.axis)
        if axis_i != 2:
            raise ValueError(
                f"{_PRIVATE_SHEET_MESSAGE}; only z-axis sheets are supported"
            )
        if request.component not in ("ex", "ey"):
            raise ValueError(
                f"{_PRIVATE_SHEET_MESSAGE}; sheet component must be ex or ey"
            )
        if int(request.propagation_sign) != 1:
            raise ValueError(
                f"{_PRIVATE_SHEET_MESSAGE}; only +z propagation is supported"
            )

        try:
            index = _local_strict_normal_index(
                coordinate=request.coordinate,
                offset=offsets[axis_i],
                dx=dx_f,
                n_cells=shape_f[axis_i],
            )
        except ValueError as exc:
            raise ValueError(
                str(exc).replace(_STRICT_INTERIOR_MESSAGE, _PRIVATE_SHEET_MESSAGE)
            ) from exc
        lo1, hi1 = _local_strict_span_bounds(
            span=request.x_span,
            offset=offsets[0],
            dx=dx_f,
            n_cells=shape_f[0],
            label="x",
            message=_PRIVATE_SHEET_MESSAGE,
        )
        lo2, hi2 = _local_strict_span_bounds(
            span=request.y_span,
            offset=offsets[1],
            dx=dx_f,
            n_cells=shape_f[1],
            label="y",
            message=_PRIVATE_SHEET_MESSAGE,
        )
        specs.append(
            _PrivateAnalyticSheetSourceSpec(
                name=str(request.name),
                axis=axis_i,
                index=int(index),
                component=str(request.component),
                propagation_sign=int(request.propagation_sign),
                amplitude=float(request.amplitude),
                f0_hz=float(request.f0_hz),
                bandwidth=float(request.bandwidth),
                phase_rad=float(request.phase_rad),
                source_values=_analytic_sheet_waveform(
                    request,
                    dt=float(dt),
                    n_steps=int(n_steps),
                ),
                lo1=int(lo1),
                hi1=int(hi1),
                lo2=int(lo2),
                hi2=int(hi2),
            )
        )
    return tuple(specs)


def _uniform_private_offsets(grid: Grid) -> tuple[float, float, float]:
    """Map user-domain coordinates to full-grid indices for private fixtures."""

    return (
        -float(grid.pad_x_lo) * float(grid.dx),
        -float(grid.pad_y_lo) * float(grid.dx),
        -float(grid.pad_z_lo) * float(grid.dx),
    )


def _validate_private_reference_surface(sim) -> None:
    """Fail closed before running the private uniform/reference harness."""

    public_fields = (
        ("DFT plane probes", getattr(sim, "_dft_planes", None)),
        ("flux monitors", getattr(sim, "_flux_monitors", None)),
        ("public TFSF sources", getattr(sim, "_tfsf", None)),
        ("NTFF boxes", getattr(sim, "_ntff", None)),
        ("waveguide ports", getattr(sim, "_waveguide_ports", None)),
        ("coaxial ports", getattr(sim, "_coaxial_ports", None)),
        ("Floquet ports", getattr(sim, "_floquet_ports", None)),
        ("lumped RLC", getattr(sim, "_lumped_rlc", None)),
        ("soft point sources or ports", getattr(sim, "_ports", None)),
    )
    for label, value in public_fields:
        if value:
            raise ValueError(f"{_PRIVATE_REFERENCE_PUBLIC_SURFACE_MESSAGE}: {label}")

    if getattr(sim, "_refinement", None) is not None:
        raise ValueError(
            "private SBP-SAT same-contract reference flux requires a uniform "
            "simulation with no refinement"
        )

    if getattr(sim, "_boundary", None) not in ("pec", "cpml"):
        raise ValueError(
            "private SBP-SAT same-contract reference flux supports only "
            "boundary='pec' or boundary='cpml'"
        )
    if getattr(sim, "_periodic_axes", None):
        raise ValueError(
            "private SBP-SAT same-contract reference flux does not support "
            "periodic axes"
        )
    boundary_spec = getattr(sim, "_boundary_spec", None)
    if boundary_spec is not None:
        if boundary_spec.absorber_type == "upml":
            raise ValueError(
                "private SBP-SAT same-contract reference flux does not support UPML"
            )
        if boundary_spec.periodic_axes():
            raise ValueError(
                "private SBP-SAT same-contract reference flux does not support "
                "periodic BoundarySpec faces"
            )
        if boundary_spec.pmc_faces():
            raise ValueError(
                "private SBP-SAT same-contract reference flux does not support "
                "PMC BoundarySpec faces"
            )


def _state_to_private_subgrid(state):
    """Wrap a uniform FDTD state so private SBP-SAT accumulators can sample it."""

    from rfx.subgridding.sbp_sat_3d import SubgridState3D

    zeros = jnp.zeros_like(state.ex)
    return SubgridState3D(
        ex_c=zeros,
        ey_c=zeros,
        ez_c=zeros,
        hx_c=zeros,
        hy_c=zeros,
        hz_c=zeros,
        ex_f=state.ex,
        ey_f=state.ey,
        ez_f=state.ez,
        hx_f=state.hx,
        hy_f=state.hy,
        hz_f=state.hz,
        step=state.step,
    )


def _state_from_private_subgrid(state, private_state):
    """Copy private wrapper fine arrays back into a uniform FDTD state."""

    return state._replace(
        ex=private_state.ex_f,
        ey=private_state.ey_f,
        ez=private_state.ez_f,
        hx=private_state.hx_f,
        hy=private_state.hy_f,
        hz=private_state.hz_f,
    )


def run_private_tfsf_reference_flux(
    sim,
    *,
    n_steps: int,
    planes: tuple[_BenchmarkFluxPlaneRequest, ...] | list[_BenchmarkFluxPlaneRequest],
    private_tfsf_incidents: (
        tuple[_PrivateTFSFIncidentRequest, ...] | list[_PrivateTFSFIncidentRequest]
    ) = (),
    private_plane_wave_sources: (
        tuple[_PrivatePlaneWaveSourceRequest, ...]
        | list[_PrivatePlaneWaveSourceRequest]
    ) = (),
) -> _BenchmarkFluxRun:
    """Run a private same-contract uniform reference for SBP-SAT TFSF evidence.

    This is a benchmark-only sibling to ``run_subgridded_benchmark_flux``.
    It intentionally does not route through ``Simulation.run()``,
    ``rfx.runners.uniform``, public TFSF, public DFT plane probes, or public
    flux monitors.  The step ordering mirrors the public uniform TFSF slots:
    ``update_h`` -> private H -> CPML-H, then ``update_e`` -> private E ->
    CPML-E.
    """

    import warnings

    from rfx.api import Result
    from rfx.boundaries.cpml import apply_cpml_e, apply_cpml_h, init_cpml
    from rfx.boundaries.pec import apply_pec, apply_pec_faces, apply_pec_mask
    from rfx.boundaries.pmc import apply_pmc_faces
    from rfx.core.yee import FDTDState, init_state, update_e, update_h
    from rfx.subgridding.jit_runner import (
        _BenchmarkFluxPlaneResult,
        _accumulate_benchmark_flux_plane,
        _apply_private_plane_wave_source_e,
        _apply_private_plane_wave_source_h,
        _apply_private_tfsf_incident_e,
        _apply_private_tfsf_incident_h,
        _empty_benchmark_flux_accumulator,
    )

    _validate_private_reference_surface(sim)
    if sim._dx is None and sim._geometry:
        sim._auto_configure_mesh()
    sim._validate_mesh_quality()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No sources, ports, TFSF, or waveguide/Floquet ports configured.*",
            category=UserWarning,
        )
        sim._validate_simulation_config()

    grid = sim._build_grid()
    materials, debye_spec, lorentz_spec, pec_mask, _, kerr_chi3 = (
        sim._assemble_materials(grid)
    )
    if debye_spec is not None or lorentz_spec is not None:
        raise ValueError(
            "private SBP-SAT same-contract reference flux does not support "
            "dispersive materials"
        )
    if kerr_chi3 is not None and bool(jnp.any(kerr_chi3)):
        raise ValueError(
            "private SBP-SAT same-contract reference flux does not support "
            "nonlinear materials"
        )
    if pec_mask is not None and bool(jnp.any(pec_mask)):
        raise ValueError(
            "private SBP-SAT same-contract reference flux does not support "
            "PEC geometry masks"
        )

    n_steps = int(n_steps)
    shape = tuple(int(v) for v in grid.shape)
    offsets = _uniform_private_offsets(grid)
    benchmark_flux_specs = _build_benchmark_flux_plane_specs(
        tuple(planes),
        shape_f=shape,
        offsets=offsets,
        dx_f=float(grid.dx),
        n_steps=n_steps,
    )
    private_tfsf_specs = _build_private_tfsf_incident_specs(
        tuple(private_tfsf_incidents),
        shape_f=shape,
        offsets=offsets,
        dx_f=float(grid.dx),
        dt=float(grid.dt),
        n_steps=n_steps,
    )
    private_plane_wave_specs = _build_private_plane_wave_source_specs(
        tuple(private_plane_wave_sources),
        shape_f=shape,
        offsets=offsets,
        dx_f=float(grid.dx),
        dt=float(grid.dt),
        n_steps=n_steps,
    )

    use_cpml = (
        sim._boundary == "cpml"
        and grid.cpml_layers > 0
        and bool(getattr(grid, "cpml_axes", ""))
    )
    cpml_params = cpml_state_init = None
    if use_cpml:
        cpml_params, cpml_state_init = init_cpml(grid)

    pec_axes = "xyz" if sim._boundary == "pec" else ""
    pec_faces = frozenset(getattr(sim, "_pec_faces", set()) or set())
    pmc_faces = frozenset(sim._boundary_spec.pmc_faces())
    periodic = tuple(axis in (sim._periodic_axes or "") for axis in "xyz")
    probe_meta = [
        (grid.position_to_index(pe.position), pe.component)
        for pe in getattr(sim, "_probes", ())
    ]
    flux_acc_init = tuple(
        _empty_benchmark_flux_accumulator(plane, shape)
        for plane in benchmark_flux_specs
    )

    def _apply_private_h_all(state: FDTDState, step_idx):
        private_state = _state_to_private_subgrid(state)
        for incident in private_tfsf_specs:
            private_state = _apply_private_tfsf_incident_h(
                private_state,
                incident,
                incident.magnetic_values[step_idx],
            )
        for source in private_plane_wave_specs:
            private_state = _apply_private_plane_wave_source_h(
                private_state,
                source,
                source.magnetic_values[step_idx],
            )
        return _state_from_private_subgrid(state, private_state)

    def _apply_private_e_all(state: FDTDState, step_idx):
        private_state = _state_to_private_subgrid(state)
        for incident in private_tfsf_specs:
            private_state = _apply_private_tfsf_incident_e(
                private_state,
                incident,
                incident.electric_values[step_idx],
            )
        for source in private_plane_wave_specs:
            private_state = _apply_private_plane_wave_source_e(
                private_state,
                source,
                source.electric_values[step_idx],
            )
        return _state_from_private_subgrid(state, private_state)

    def _sample_probes(state: FDTDState) -> jnp.ndarray:
        if not probe_meta:
            return jnp.zeros(0, dtype=jnp.float32)
        samples = []
        for (i, j, k), component in probe_meta:
            samples.append(getattr(state, component)[i, j, k])
        return jnp.stack(samples)

    def _accumulate_flux(accs, state: FDTDState):
        private_state = _state_to_private_subgrid(state)
        return tuple(
            _accumulate_benchmark_flux_plane(acc, private_state, plane, grid.dt)
            for acc, plane in zip(accs, benchmark_flux_specs)
        )

    def step_fn(carry, step_idx):
        state, cpml_state, flux_accs = carry
        state = update_h(state, materials, grid.dt, grid.dx, periodic=periodic)
        # Same pre-CPML slot used by public TFSF in ``Simulation.run``.
        state = _apply_private_h_all(state, step_idx)
        if use_cpml:
            state, cpml_new = apply_cpml_h(
                state,
                cpml_params,
                cpml_state,
                grid,
                grid.cpml_axes,
                materials=materials,
            )
        else:
            cpml_new = cpml_state
        if pmc_faces:
            state = apply_pmc_faces(state, pmc_faces)

        state = update_e(state, materials, grid.dt, grid.dx, periodic=periodic)
        # Same pre-CPML slot used by public TFSF in ``Simulation.run``.
        state = _apply_private_e_all(state, step_idx)
        if use_cpml:
            state, cpml_new = apply_cpml_e(
                state,
                cpml_params,
                cpml_new,
                grid,
                grid.cpml_axes,
                materials=materials,
            )
        if pec_axes:
            state = apply_pec(state, axes=pec_axes)
        if pec_faces:
            state = apply_pec_faces(state, set(pec_faces))
        if pec_mask is not None:
            state = apply_pec_mask(state, pec_mask)

        flux_accs = _accumulate_flux(flux_accs, state)
        return (state, cpml_new, flux_accs), _sample_probes(state)

    state_init = init_state(shape)
    initial_carry = (state_init, cpml_state_init, flux_acc_init)
    final_carry, time_series = jax.lax.scan(
        step_fn,
        initial_carry,
        jnp.arange(n_steps, dtype=jnp.int32),
    )
    final_state, _, final_flux_accs = final_carry
    final_state = final_state._replace(step=jnp.array(n_steps, dtype=jnp.int32))
    benchmark_flux_results = tuple(
        _BenchmarkFluxPlaneResult(
            name=plane.name,
            axis=plane.axis,
            index=plane.index,
            freqs=plane.freqs,
            dx=plane.dx,
            e1_dft=accs[0],
            e2_dft=accs[1],
            h1_dft=accs[2],
            h2_dft=accs[3],
            lo1=plane.lo1,
            hi1=plane.hi1 if plane.hi1 >= 0 else accs[0].shape[1],
            lo2=plane.lo2,
            hi2=plane.hi2 if plane.hi2 >= 0 else accs[0].shape[2],
        )
        for plane, accs in zip(benchmark_flux_specs, final_flux_accs)
    )

    public_result = Result(
        state=final_state,
        time_series=time_series,
        s_params=None,
        freqs=None,
        grid=grid,
        dt=float(grid.dt),
        freq_range=(sim._freq_max / 10, sim._freq_max, sim._boundary),
    )
    return _BenchmarkFluxRun(
        result=public_result,
        benchmark_flux_planes=benchmark_flux_results,
    )


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
    _private_sheet_sources: tuple[_PrivateAnalyticSheetSourceRequest, ...]
    | None = None,
    _private_tfsf_incidents: tuple[_PrivateTFSFIncidentRequest, ...] | None = None,
    _private_plane_wave_sources: tuple[_PrivatePlaneWaveSourceRequest, ...]
    | None = None,
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
        ref.get("x_range"),
        grid_coarse.nx,
        grid_coarse.pad_x_lo,
        grid_coarse.pad_x_hi,
        "x_range",
    )
    fj_lo, fj_hi = _range_to_indices(
        ref.get("y_range"),
        grid_coarse.ny,
        grid_coarse.pad_y_lo,
        grid_coarse.pad_y_hi,
        "y_range",
    )
    fk_lo, fk_hi = _range_to_indices(
        ref["z_range"],
        grid_coarse.nz,
        grid_coarse.pad_z_lo,
        grid_coarse.pad_z_hi,
        "z_range",
    )

    nx_f = (fi_hi - fi_lo) * ratio
    ny_f = (fj_hi - fj_lo) * ratio
    nz_f = (fk_hi - fk_lo) * ratio

    dt = phase1_3d_dt(dx_f)

    config = SubgridConfig3D(
        nx_c=grid_coarse.nx,
        ny_c=grid_coarse.ny,
        nz_c=grid_coarse.nz,
        dx_c=dx_c,
        fi_lo=fi_lo,
        fi_hi=fi_hi,
        fj_lo=fj_lo,
        fj_hi=fj_hi,
        fk_lo=fk_lo,
        fk_hi=fk_hi,
        nx_f=nx_f,
        ny_f=ny_f,
        nz_f=nz_f,
        dx_f=dx_f,
        dt=float(dt),
        ratio=ratio,
        tau=tau,
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
    private_sheet_specs = ()
    if _private_sheet_sources:
        private_sheet_specs = _build_private_analytic_sheet_source_specs(
            tuple(_private_sheet_sources),
            shape_f=shape_f,
            offsets=(x_off, y_off, z_off),
            dx_f=dx_f,
            dt=float(dt),
            n_steps=n_steps,
        )
    private_tfsf_specs = ()
    if _private_tfsf_incidents:
        private_tfsf_specs = _build_private_tfsf_incident_specs(
            tuple(_private_tfsf_incidents),
            shape_f=shape_f,
            offsets=(x_off, y_off, z_off),
            dx_f=dx_f,
            dt=float(dt),
            n_steps=n_steps,
        )
    private_plane_wave_specs = ()
    if _private_plane_wave_sources:
        private_plane_wave_specs = _build_private_plane_wave_source_specs(
            tuple(_private_plane_wave_sources),
            shape_f=shape_f,
            offsets=(x_off, y_off, z_off),
            dx_f=dx_f,
            dt=float(dt),
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
                    (
                        fi_lo,
                        fi_hi,
                        grid_coarse.nx,
                        grid_coarse.pad_x_lo,
                        grid_coarse.pad_x_hi,
                    ),
                    (
                        fj_lo,
                        fj_hi,
                        grid_coarse.ny,
                        grid_coarse.pad_y_lo,
                        grid_coarse.pad_y_hi,
                    ),
                    (
                        fk_lo,
                        fk_hi,
                        grid_coarse.nz,
                        grid_coarse.pad_z_lo,
                        grid_coarse.pad_z_hi,
                    ),
                ),
            )
        ),
        absorber_boundary=sim._boundary,
        _benchmark_flux_planes=benchmark_flux_specs,
        _private_sheet_sources=private_sheet_specs,
        _private_tfsf_incidents=private_tfsf_specs,
        _private_plane_wave_sources=private_plane_wave_specs,
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
    sheet_sources: (
        tuple[_PrivateAnalyticSheetSourceRequest, ...]
        | list[_PrivateAnalyticSheetSourceRequest]
        | None
    ) = None,
    private_tfsf_incidents: (
        tuple[_PrivateTFSFIncidentRequest, ...]
        | list[_PrivateTFSFIncidentRequest]
        | None
    ) = None,
    private_plane_wave_sources: (
        tuple[_PrivatePlaneWaveSourceRequest, ...]
        | list[_PrivatePlaneWaveSourceRequest]
        | None
    ) = None,
) -> _BenchmarkFluxRun:
    """Run the private SBP-SAT benchmark-only flux accumulator path.

    Public DFT/flux requests still fail in the regular API validator.  This
    helper accepts only private fine-owned plane requests plus optional
    private analytic sheet sources, private TFSF-style incident fields, or
    private plane-wave sources, and returns private raw accumulators alongside
    the ordinary public ``Result`` to prove the benchmark does not leak into
    ``Result.dft_planes`` or ``Result.flux_monitors``.
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
        _private_sheet_sources=tuple(sheet_sources or ()),
        _private_tfsf_incidents=tuple(private_tfsf_incidents or ()),
        _private_plane_wave_sources=tuple(private_plane_wave_sources or ()),
    )
