"""Coax plane V/I extraction, line reflection and two-port wave solve."""

from __future__ import annotations
from rfx.errors import AccuracyWarning
from typing import NamedTuple, TYPE_CHECKING
import jax
import jax.numpy as jnp
from rfx._internal._precision import HIGHEST
from rfx.core.jax_utils import complex_inv
import numpy as np
from rfx.grid import Grid
from rfx.model.coax_stamp import SMA_PIN_RADIUS, SMA_OUTER_RADIUS, PTFE_EPS_R, _require_coaxial_aperture, _coaxial_port_geometry

if TYPE_CHECKING:
    from rfx.sources.coaxial_port import CoaxialPort


class CoaxialTEMReferencePlaneVI(NamedTuple):
    """Calibrated TEM V/I extracted from a coaxial reference plane.

    ``voltage`` is the radial electric-field line integral from the inner
    conductor to the outer conductor. ``current`` is the azimuthal magnetic
    field loop integral around the center conductor.  Both quantities use the
    public port convention: positive current flows into the device under test.
    """

    voltage: np.ndarray
    current: np.ndarray
    radial_positions_m: np.ndarray
    h_sample_radius_m: float
    inner_radius_m: float
    outer_radius_m: float


class CoaxialTEMCartesianPlaneVI(NamedTuple):
    """TEM V/I extracted from tangential Cartesian field-plane samples.

    The plane-local coordinates are ``u`` and ``v``.  ``e_u/e_v`` and
    ``h_u/h_v`` are tangential field samples on that plane, with arbitrary
    leading axes such as frequency.  This is the adapter needed between a
    frequency-domain FDTD plane dump and :func:`coaxial_tem_reference_plane_vi`.
    """

    vi: CoaxialTEMReferencePlaneVI
    radial_positions_m: np.ndarray
    azimuthal_angles_rad: np.ndarray
    center_uv_m: tuple[float, float]


def coaxial_tem_reference_plane_vi(
    radial_positions_m,
    e_radial_samples,
    h_phi_samples,
    *,
    h_sample_radius_m: float,
    inner_radius: float,
    outer_radius: float,
    eps_r: float = PTFE_EPS_R,
    voltage_sign: float = 1.0,
    current_sign: float = 1.0,
) -> CoaxialTEMReferencePlaneVI:
    """Extract calibrated coaxial TEM voltage/current from field samples.

    Parameters
    ----------
    radial_positions_m:
        1-D sample positions on a radial line from inner to outer conductor.
    e_radial_samples:
        Radial electric field samples.  The last axis must match
        ``radial_positions_m``. Leading axes (for example frequency) are
        preserved in the returned voltage.
    h_phi_samples:
        Azimuthal magnetic field samples around a circular loop at
        ``h_sample_radius_m``. The last axis is averaged over azimuthal sample
        angle. Leading axes are preserved in the returned current.
    h_sample_radius_m:
        Radius of the magnetic-field sampling loop.
    inner_radius, outer_radius, eps_r:
        Coax geometry. ``eps_r`` is validated for consistency with the other
        TEM helpers even though it is not needed by the field integrals.
    voltage_sign, current_sign:
        Optional convention flips for callers whose field orientation differs
        from the public port convention.

    Notes
    -----
    This helper is the calibration primitive for a coaxial reference-plane
    TEM extractor.  It does not run FDTD and does not by itself promote
    ``add_coaxial_port(...)`` to a high-level S-parameter API.
    """

    _validate_coaxial_tem_geometry(inner_radius, outer_radius, eps_r=eps_r)
    radial_positions = np.asarray(radial_positions_m, dtype=np.float64)
    if radial_positions.ndim != 1 or radial_positions.size < 2:
        raise ValueError("radial_positions_m must be a 1-D array with at least two samples")
    if np.any(~np.isfinite(radial_positions)):
        raise ValueError("radial_positions_m must be finite")
    if np.any(np.diff(radial_positions) <= 0.0):
        raise ValueError("radial_positions_m must be strictly increasing")
    if radial_positions[0] < inner_radius or radial_positions[-1] > outer_radius:
        raise ValueError(
            "radial_positions_m must lie within [inner_radius, outer_radius], got "
            f"{radial_positions[0]}..{radial_positions[-1]} outside "
            f"{inner_radius}..{outer_radius}"
        )
    if not np.isfinite(h_sample_radius_m):
        raise ValueError("h_sample_radius_m must be finite")
    if h_sample_radius_m < inner_radius or h_sample_radius_m > outer_radius:
        raise ValueError(
            "h_sample_radius_m must lie within [inner_radius, outer_radius], got "
            f"{h_sample_radius_m}"
        )

    e_radial = np.asarray(e_radial_samples, dtype=np.complex128)
    if e_radial.shape[-1:] != (radial_positions.size,):
        raise ValueError(
            "last axis of e_radial_samples must match radial_positions_m length; "
            f"got {e_radial.shape} vs {radial_positions.size}"
        )
    h_phi = np.asarray(h_phi_samples, dtype=np.complex128)
    if h_phi.ndim == 0:
        raise ValueError("h_phi_samples must have at least one azimuthal sample")
    if h_phi.shape[-1] < 1:
        raise ValueError("h_phi_samples must have at least one azimuthal sample")

    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    voltage = float(voltage_sign) * trapezoid(
        e_radial,
        radial_positions,
        axis=-1,
    )
    current = (
        float(current_sign)
        * 2.0
        * np.pi
        * float(h_sample_radius_m)
        * np.mean(h_phi, axis=-1)
    )
    return CoaxialTEMReferencePlaneVI(
        voltage=np.asarray(voltage, dtype=np.complex128),
        current=np.asarray(current, dtype=np.complex128),
        radial_positions_m=radial_positions,
        h_sample_radius_m=float(h_sample_radius_m),
        inner_radius_m=float(inner_radius),
        outer_radius_m=float(outer_radius),
    )


def coaxial_tem_reference_plane_vi_from_cartesian_plane(
    u_coords_m,
    v_coords_m,
    e_u_samples,
    e_v_samples,
    h_u_samples,
    h_v_samples,
    *,
    center_u_m: float,
    center_v_m: float,
    inner_radius: float,
    outer_radius: float,
    eps_r: float = PTFE_EPS_R,
    radial_positions_m=None,
    h_sample_radius_m: float | None = None,
    azimuthal_angles_rad=None,
    voltage_sign: float = 1.0,
    current_sign: float = 1.0,
) -> CoaxialTEMCartesianPlaneVI:
    """Extract coaxial TEM V/I from Cartesian tangential field-plane samples.

    Parameters
    ----------
    u_coords_m, v_coords_m:
        Strictly increasing coordinate arrays for the two tangential plane
        axes.  Field arrays must use these as their last two dimensions.
    e_u_samples, e_v_samples:
        Tangential electric-field DFT samples on the plane.
    h_u_samples, h_v_samples:
        Tangential magnetic-field DFT samples on the same plane.
    center_u_m, center_v_m:
        Coax center on the sampled plane.
    inner_radius, outer_radius:
        Coax conductor radii in metres.
    radial_positions_m:
        Optional radial integration points.  By default, all available
        positive-``u`` grid coordinates that lie inside ``[inner, outer]`` are
        used.  The caller may pass a denser grid for synthetic or interpolated
        data.
    h_sample_radius_m:
        Optional radius for the azimuthal H-loop.  Defaults to the geometric
        mean of the inner and outer conductor radii.
    azimuthal_angles_rad:
        Optional loop sample angles.  Defaults to 16 equiangular samples.

    Notes
    -----
    This adapter performs bilinear interpolation and then delegates the actual
    TEM V/I integration to :func:`coaxial_tem_reference_plane_vi`.  It does not
    run FDTD and is not a promoted coaxial S-parameter API.
    """

    _validate_coaxial_tem_geometry(inner_radius, outer_radius, eps_r=eps_r)
    u_coords = _validate_monotone_axis(u_coords_m, name="u_coords_m")
    v_coords = _validate_monotone_axis(v_coords_m, name="v_coords_m")
    center_u = float(center_u_m)
    center_v = float(center_v_m)
    if not np.isfinite(center_u) or not np.isfinite(center_v):
        raise ValueError("center_u_m and center_v_m must be finite")

    e_u = _validate_plane_field(e_u_samples, u_coords, v_coords, name="e_u_samples")
    e_v = _validate_plane_field(e_v_samples, u_coords, v_coords, name="e_v_samples")
    h_u = _validate_plane_field(h_u_samples, u_coords, v_coords, name="h_u_samples")
    h_v = _validate_plane_field(h_v_samples, u_coords, v_coords, name="h_v_samples")
    if not (e_u.shape == e_v.shape == h_u.shape == h_v.shape):
        raise ValueError(
            "e_u/e_v/h_u/h_v samples must have identical shapes, got "
            f"{e_u.shape}, {e_v.shape}, {h_u.shape}, {h_v.shape}"
        )

    if radial_positions_m is None:
        radial_positions = u_coords - center_u
        radial_positions = radial_positions[
            (radial_positions >= inner_radius) & (radial_positions <= outer_radius)
        ]
        if radial_positions.size < 2:
            radial_positions = np.linspace(inner_radius, outer_radius, 33)
    else:
        radial_positions = np.asarray(radial_positions_m, dtype=np.float64)

    if h_sample_radius_m is None:
        h_radius = float(np.sqrt(float(inner_radius) * float(outer_radius)))
    else:
        h_radius = float(h_sample_radius_m)
    if azimuthal_angles_rad is None:
        angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float64)
    else:
        angles = np.asarray(azimuthal_angles_rad, dtype=np.float64)
        if angles.ndim != 1 or angles.size < 4:
            raise ValueError("azimuthal_angles_rad must be a 1-D array with at least 4 samples")
        if np.any(~np.isfinite(angles)):
            raise ValueError("azimuthal_angles_rad must be finite")

    # Voltage line integral along the +u radial ray.  On this ray
    # e_radial == e_u.
    e_radial = _interp_bilinear_plane(
        e_u,
        u_coords,
        v_coords,
        center_u + radial_positions,
        np.full_like(radial_positions, center_v),
    )

    # Current loop integral around radius h_radius.  In plane-local
    # coordinates, r_hat=(cosθ, sinθ), phi_hat=(-sinθ, cosθ).
    loop_u = center_u + h_radius * np.cos(angles)
    loop_v = center_v + h_radius * np.sin(angles)
    h_u_loop = _interp_bilinear_plane(h_u, u_coords, v_coords, loop_u, loop_v)
    h_v_loop = _interp_bilinear_plane(h_v, u_coords, v_coords, loop_u, loop_v)
    h_phi = -h_u_loop * np.sin(angles) + h_v_loop * np.cos(angles)

    vi = coaxial_tem_reference_plane_vi(
        radial_positions,
        e_radial,
        h_phi,
        h_sample_radius_m=h_radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        eps_r=eps_r,
        voltage_sign=voltage_sign,
        current_sign=current_sign,
    )
    return CoaxialTEMCartesianPlaneVI(
        vi=vi,
        radial_positions_m=radial_positions,
        azimuthal_angles_rad=angles,
        center_uv_m=(center_u, center_v),
    )


def coaxial_tem_reference_plane_s11(voltage, current, reference_impedance: float):
    """Convert calibrated TEM V/I phasors to one-port S11.

    The convention is the same power-wave split used by the raw V/I replay
    helpers: ``a = (V + Z0 I) / 2`` and ``b = (V - Z0 I) / 2``.
    """

    voltage = np.asarray(voltage, dtype=np.complex128)
    current = np.asarray(current, dtype=np.complex128)
    z0 = complex(reference_impedance)
    if not np.isfinite(z0) or z0.real <= 0.0 or abs(z0.imag) > 0.0:
        raise ValueError(
            "reference_impedance must be a positive real finite impedance, got "
            f"{reference_impedance}"
        )
    incident = 0.5 * (voltage + z0 * current)
    reflected = 0.5 * (voltage - z0 * current)
    return reflected / incident


def _validate_monotone_axis(values, *, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError(f"{name} must be a 1-D array with at least two samples")
    if np.any(~np.isfinite(axis)):
        raise ValueError(f"{name} must be finite")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _validate_plane_field(
    values,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    field = np.asarray(values, dtype=np.complex128)
    if field.shape[-2:] != (u_coords.size, v_coords.size):
        raise ValueError(
            f"last two axes of {name} must match u/v coordinate lengths; "
            f"got {field.shape[-2:]} vs {(u_coords.size, v_coords.size)}"
        )
    return field


def _interp_bilinear_plane(
    field: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    sample_u: np.ndarray,
    sample_v: np.ndarray,
) -> np.ndarray:
    sample_u = np.asarray(sample_u, dtype=np.float64)
    sample_v = np.asarray(sample_v, dtype=np.float64)
    if sample_u.shape != sample_v.shape:
        raise ValueError(
            "sample_u and sample_v must have identical shapes, got "
            f"{sample_u.shape} vs {sample_v.shape}"
        )
    if np.any(~np.isfinite(sample_u)) or np.any(~np.isfinite(sample_v)):
        raise ValueError("sample coordinates must be finite")
    if (
        sample_u.min(initial=u_coords[0]) < u_coords[0]
        or sample_u.max(initial=u_coords[-1]) > u_coords[-1]
        or sample_v.min(initial=v_coords[0]) < v_coords[0]
        or sample_v.max(initial=v_coords[-1]) > v_coords[-1]
    ):
        raise ValueError("sample coordinates must lie inside the field plane bounds")

    iu1 = np.searchsorted(u_coords, sample_u, side="right")
    iv1 = np.searchsorted(v_coords, sample_v, side="right")
    iu1 = np.clip(iu1, 1, u_coords.size - 1)
    iv1 = np.clip(iv1, 1, v_coords.size - 1)
    iu0 = iu1 - 1
    iv0 = iv1 - 1

    u0 = u_coords[iu0]
    u1 = u_coords[iu1]
    v0 = v_coords[iv0]
    v1 = v_coords[iv1]
    wu = (sample_u - u0) / (u1 - u0)
    wv = (sample_v - v0) / (v1 - v0)

    f00 = field[..., iu0, iv0]
    f10 = field[..., iu1, iv0]
    f01 = field[..., iu0, iv1]
    f11 = field[..., iu1, iv1]
    return (
        f00 * (1.0 - wu) * (1.0 - wv)
        + f10 * wu * (1.0 - wv)
        + f01 * (1.0 - wu) * wv
        + f11 * wu * wv
    )


def _validate_coaxial_tem_geometry(
    inner_radius: float,
    outer_radius: float,
    eps_r: float = 1.0,
    mu_r: float = 1.0,
) -> None:
    if inner_radius <= 0.0:
        raise ValueError(f"inner_radius must be positive, got {inner_radius}")
    if outer_radius <= inner_radius:
        raise ValueError(
            "outer_radius must be larger than inner_radius, got "
            f"{outer_radius} <= {inner_radius}"
        )
    if eps_r <= 0.0:
        raise ValueError(f"eps_r must be positive, got {eps_r}")
    if mu_r <= 0.0:
        raise ValueError(f"mu_r must be positive, got {mu_r}")




def extract_coaxial_plane_vi_from_dft(
    *,
    grid: "Grid",
    port: CoaxialPort,
    plane_axial_index: int,
    ex_dft: np.ndarray,
    ey_dft: np.ndarray,
    hx_dft: np.ndarray,
    hy_dft: np.ndarray,
    eps_r: float = PTFE_EPS_R,
) -> CoaxialTEMCartesianPlaneVI:
    """Pull V/I phasors from DFT plane probes at the coaxial reference plane.

    A convenience wrapper: it takes the DFT-plane outputs (one per
    Ex/Ey/Hx/Hy component) and delegates to
    :func:`coaxial_tem_reference_plane_vi_from_cartesian_plane`, which
    already implements the radial line-integral / azimuthal loop-integral
    extraction with bilinear interpolation.

    The plane probes must be on the same cross-section that
    :func:`build_coaxial_tem_plane_source_specs` injected on, i.e.
    ``plane_axial_index``.

    The ``current_sign`` is derived from the port's face direction. The
    raw azimuthal-loop integral ``2π·r·⟨H_φ⟩`` measures the line current
    flowing in the ``+ẑ`` direction; the standard ``V/I = +Z_TEM`` forward-
    wave convention requires this to align with the *forward* direction of
    the source. For ``face='top'`` the source emits in ``-z`` (pin extends
    into ``-z``), so ``current_sign = -1`` gives ``V/I = +Z_TEM`` for a
    clean forward wave; ``face='bottom'`` emits ``+z`` and uses
    ``current_sign = +1``. Without this, a power-wave S11 built from these
    phasors reads ``|S11| = 1/Γ`` instead of ``Γ`` for ``face='top'`` ports —
    same magnitude for lossless reflections but the wrong phase, and
    biased away from 1 by numerical asymmetries.
    """

    _, direction, _, _, _, _ = _coaxial_port_geometry(grid, port)
    current_sign = float(direction)

    u_coords = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v_coords = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    return coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u_coords,
        v_coords,
        np.asarray(ex_dft, dtype=np.complex128),
        np.asarray(ey_dft, dtype=np.complex128),
        np.asarray(hx_dft, dtype=np.complex128),
        np.asarray(hy_dft, dtype=np.complex128),
        center_u_m=float(port.position[0]),
        center_v_m=float(port.position[1]),
        inner_radius=float(port.pin_radius),
        outer_radius=float(port.outer_radius),
        eps_r=float(eps_r),
        current_sign=current_sign,
    )


# ===========================================================================
# Coaxial transmission-line S-parameter extraction (broad-E5 redesign)
#
# A single-plane V/I power-wave measurement reported non-physical |S11|>1
# for a lossless short. Root cause was NOT the V/I extractor (its math is exact for a clean
# lossless TEM field) but the measurement TOPOLOGY: a ~5mm coax stub in a
# hardcoded closed PEC box has no transmission line for a clean reflection.
#
# The validated method below builds a real coax line terminated in a matched
# resistive feed (conductors must NOT run into a PML — that is unstable) and
# extracts the reflection from the modal voltage V(z)=∫E_r dr sampled at >=3
# equally spaced reference planes, using a matrix-pencil estimate of the
# complex propagation constant. It is Z0-free and self-measures beta, so it is
# immune to the coarse-mesh |V/I| magnitude bias. The recurrence residual is a
# built-in single-TEM-mode validity gate. Resolution recipe: >=~4 cells across
# the (outer-inner) annulus (dx <~ 0.38 mm for SMA); below that the coax mode
# is under-resolved and the high-frequency reflection degrades.
# ===========================================================================


class CoaxialLineReflection(NamedTuple):
    """Reflection at a coaxial reference plane from a multi-plane line scan.

    ``gamma`` is the matrix-pencil estimate of the complex propagation constant
    (rad/m, with a positive imaginary part = phase constant beta). ``reflection``
    is the complex reflection coefficient Gamma at ``reference_plane_m``.
    ``recurrence_residual`` is the single-TEM-mode cleanliness gate (0 = a clean
    two-wave field; >~0.1 flags multi-mode/under-resolved contamination).
    ``fit_residual`` is the relative two-wave least-squares residual.
    """

    gamma: complex
    reflection: complex
    recurrence_residual: float
    fit_residual: float
    forward_amp: complex
    backward_amp: complex


def coaxial_line_plane_voltage(
    grid: Grid,
    ex_dft,
    ey_dft,
    *,
    center_xy: tuple[float, float],
    pin_radius: float = SMA_PIN_RADIUS,
    outer_radius: float = SMA_OUTER_RADIUS,
    eps_r: float = PTFE_EPS_R,
):
    """Modal TEM voltage ``V = ∫ E_r dr`` at a z-plane from Ex/Ey DFT planes.

    Reuses the radial line-integral of
    :func:`coaxial_tem_reference_plane_vi_from_cartesian_plane`; only the voltage
    is returned (the line method does not need the azimuthal current). Leading
    axes of ``ex_dft``/``ey_dft`` (for example frequency) are preserved.
    """

    _require_coaxial_aperture(grid, (0, 1), center_xy, outer_radius)
    u = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    ex = np.asarray(ex_dft, dtype=np.complex128)
    ey = np.asarray(ey_dft, dtype=np.complex128)
    res = coaxial_tem_reference_plane_vi_from_cartesian_plane(
        u, v, ex, ey, ex, ey,  # H args unused for the voltage line-integral
        center_u_m=float(center_xy[0]),
        center_v_m=float(center_xy[1]),
        inner_radius=float(pin_radius),
        outer_radius=float(outer_radius),
        eps_r=float(eps_r),
    )
    return np.asarray(res.vi.voltage, dtype=np.complex128)


def coaxial_line_plane_voltage_jnp(
    grid: Grid,
    ex_dft,
    *,
    center_xy: tuple[float, float],
    pin_radius: float = SMA_PIN_RADIUS,
    outer_radius: float = SMA_OUTER_RADIUS,
):
    """Differentiable ``jax.numpy`` modal TEM voltage ``V = ∫ E_r dr`` at a z-plane.

    jnp-native counterpart of :func:`coaxial_line_plane_voltage` for the AD path
    (``compute_coaxial_line_reflection(eps_scale=...)``). The radial sample
    geometry and the bilinear-interpolation weights are STATIC (grid coordinates,
    concrete NumPy); only ``ex_dft`` (the DFT E-field plane) carries the tape.
    Byte-close to the numpy path (difference is float reassociation in the
    trapezoid only, ~1e-10). Voltage-only: the coax line-reflection method does
    not use the azimuthal current, so ``ey_dft``/the H-loop are not needed here.

    On the ``+u`` radial ray ``e_radial == e_u == ex``. ``ex_dft`` may carry
    leading axes (e.g. frequency); they are preserved and the integral is over
    the trailing radial axis.
    """
    _require_coaxial_aperture(grid, (0, 1), center_xy, outer_radius)
    u = (np.arange(grid.nx, dtype=np.float64) - grid.pad_x_lo) * grid.dx
    v = (np.arange(grid.ny, dtype=np.float64) - grid.pad_y_lo) * grid.dx
    cu, cv = float(center_xy[0]), float(center_xy[1])
    radial = u - cu
    sel = (radial >= float(pin_radius)) & (radial <= float(outer_radius))
    radial_positions = radial[sel]
    if radial_positions.size < 2:
        radial_positions = np.linspace(float(pin_radius), float(outer_radius), 33)
    su = cu + radial_positions
    sv = np.full_like(su, cv)
    # Concrete bilinear indices + weights (mirrors _interp_bilinear_plane); only
    # the gathered field values below are traced.
    iu = np.clip(np.searchsorted(u, su) - 1, 0, u.size - 2)
    iv = np.clip(np.searchsorted(v, sv) - 1, 0, v.size - 2)
    tu = jnp.asarray((su - u[iu]) / (u[iu + 1] - u[iu]))
    tv = jnp.asarray((sv - v[iv]) / (v[iv + 1] - v[iv]))
    ex = jnp.asarray(ex_dft)
    f00 = ex[..., iu, iv]
    f10 = ex[..., iu + 1, iv]
    f01 = ex[..., iu, iv + 1]
    f11 = ex[..., iu + 1, iv + 1]
    e_radial = (
        f00 * (1 - tu) * (1 - tv)
        + f10 * tu * (1 - tv)
        + f01 * (1 - tu) * tv
        + f11 * tu * tv
    )
    return jnp.trapezoid(e_radial, jnp.asarray(radial_positions), axis=-1)


def _coaxial_line_reflection_jnp(
    z, plane_voltages, *, reference_plane_m, D, z0, load_below
) -> CoaxialLineReflection:
    """``jax.numpy`` core of :func:`coaxial_line_reflection_from_plane_voltages`.

    Reached ONLY for *traced* voltages (``jax.grad`` / ``jax.jit``); the concrete
    path stays in NumPy (float64) so the validated-precision residual gates are
    byte-identical. ``z`` / ``D`` / ``z0`` / ``load_below`` are static geometry
    (plain Python / NumPy) — only ``plane_voltages`` carries the tape. Singular
    points (all-zero field, zero incident wave) use double-``where`` so the
    gradient stays finite at a perfect match / null.
    """
    V = jnp.asarray(plane_voltages)
    Vm, Vc, Vp = V[:-2], V[1:-1], V[2:]
    den = jnp.sum(jnp.abs(Vc) ** 2)
    safe_den = jnp.where(den > 0.0, den, 1.0)          # AD-safe divide (all-zero field)
    p = jnp.sum((Vm + Vp) * jnp.conj(Vc)) / safe_den
    gamma = jnp.arccosh(p / 2.0 + 0j) / D
    gamma = jnp.where(jnp.imag(gamma) < 0.0, jnp.conj(gamma), gamma)  # enforce beta > 0
    rec_arg = jnp.sum(jnp.abs((Vm + Vp) - p * Vc) ** 2) / (jnp.abs(p) ** 2 * den + 1e-300)
    rec_ok = rec_arg > 0.0
    rec_resid = jnp.where(rec_ok, jnp.sqrt(jnp.where(rec_ok, rec_arg, 1.0)), 0.0)

    zc = jnp.asarray(z) - z0
    Phi = jnp.stack([jnp.exp(+gamma * zc), jnp.exp(-gamma * zc)], axis=1)
    AB, *_ = jnp.linalg.lstsq(Phi, V, rcond=None)
    A, B = AB[0], AB[1]
    resid = jnp.matmul(Phi, AB, precision=HIGHEST) - V
    # AD-safe norm (double-where): a perfect fit makes the residual exactly 0,
    # where d(sqrt)/dx = inf; symmetric with rec_resid above.
    fit_sq = jnp.sum(jnp.abs(resid) ** 2)
    fit_ok = fit_sq > 0.0
    fit_num = jnp.where(fit_ok, jnp.sqrt(jnp.where(fit_ok, fit_sq, 1.0)), 0.0)
    fit_resid = fit_num / (jnp.sqrt(jnp.sum(jnp.abs(V) ** 2)) + 1e-300)

    zr = reference_plane_m - z0
    a_wave = A * jnp.exp(+gamma * zr)    # travels -z
    b_wave = B * jnp.exp(-gamma * zr)    # travels +z
    if load_below:
        v_inc, v_ref = a_wave, b_wave
    else:
        v_inc, v_ref = b_wave, a_wave
    inc_ok = jnp.abs(v_inc) > 0.0
    reflection = jnp.where(
        inc_ok, v_ref / jnp.where(inc_ok, v_inc, 1.0), jnp.asarray(jnp.nan + 1j * jnp.nan)
    )
    return CoaxialLineReflection(
        gamma=gamma,
        reflection=reflection,
        recurrence_residual=rec_resid,
        fit_residual=fit_resid,
        forward_amp=v_inc,
        backward_amp=v_ref,
    )


def coaxial_line_reflection_from_plane_voltages(
    plane_positions_m,
    plane_voltages,
    *,
    reference_plane_m: float,
    spacing_rtol: float = 1e-6,
    _prefer_jnp: bool = False,
) -> CoaxialLineReflection:
    """Reflection coefficient from modal voltages at >=3 equally spaced planes.

    For a single TEM mode the modal voltage is a two-wave sum
    ``V(z) = A e^{+γz} + B e^{-γz}``. Equally spaced samples obey the recurrence
    ``V(z-Δ) + V(z+Δ) = 2 cosh(γΔ) V(z)``; a least-squares estimate of
    ``2 cosh(γΔ)`` over all interior planes gives the complex propagation
    constant ``γ`` directly from the field (β self-measured, no analytic input,
    Z0-free). ``A``/``B`` follow by least squares, and ``Γ`` is evaluated at
    ``reference_plane_m`` with the incident wave defined as the one travelling
    from the probe region toward the reference plane.

    Parameters
    ----------
    plane_positions_m : (N,) strictly increasing, EQUALLY spaced (N>=3).
    plane_voltages : (N,) complex modal voltages at those planes (one frequency).
    reference_plane_m : axial position at which Γ is reported (the load plane);
        may lie outside the probe span (the two-wave model is extrapolated).

    AD note
    -------
    ``Γ`` is differentiable w.r.t. the modal ``plane_voltages``: *traced* inputs
    (``jax.grad`` / ``jax.jit``) run a ``jax.numpy`` core
    (:func:`_coaxial_line_reflection_jnp`), while *concrete* inputs keep the
    NumPy (float64) path so the validated-precision residual gates stay
    byte-identical. Plane *positions* are static geometry. This is an
    ``extractor``-level AD property. The enclosing
    ``Simulation.compute_coaxial_line_reflection(..., eps_scale=...)`` method
    uses ``coaxial_line_plane_voltage_jnp`` and is checked end to end by
    ``tests/unit/autodiff/test_coax_end_to_end_ad.py``. This extractor is checked separately
    by
    ``tests/unit/autodiff/test_ad_surface_contract.py::test_coaxial_reflection_extraction_is_traceable``
    and ``tests/unit/sparams/test_coaxial_line_reflection.py``.
    """

    z = np.asarray(plane_positions_m, dtype=np.float64)
    if z.ndim != 1 or z.size < 3:
        raise ValueError("plane_positions_m must be 1-D with at least 3 planes")
    if np.shape(plane_voltages) != z.shape:
        raise ValueError(
            f"plane_voltages shape {np.shape(plane_voltages)} must match positions {z.shape}"
        )
    d = np.diff(z)
    if np.any(d <= 0.0):
        raise ValueError("plane_positions_m must be strictly increasing")
    D = float(d.mean())
    if np.any(np.abs(d - D) > spacing_rtol * D):
        raise ValueError("plane_positions_m must be equally spaced for the matrix pencil")
    z0 = float(z.mean())
    # Reference (load) plane side vs the probe centroid; ``<=`` keeps the tie
    # case (load exactly at the probe mean — not a valid one-port geometry) on
    # the load-below branch deterministically. Static (positions only).
    load_below = bool(reference_plane_m <= z0)

    # Traced voltages -> jax.numpy core (differentiable); concrete -> NumPy.
    # ``_prefer_jnp`` forces the jnp core even for concrete inputs (used by the
    # end-to-end AD path so a concrete FD probe uses the same tolerant jnp lstsq
    # instead of numpy's strict SVD, which raises on marginal fits).
    if _prefer_jnp or isinstance(plane_voltages, jax.core.Tracer):
        return _coaxial_line_reflection_jnp(
            z, plane_voltages, reference_plane_m=float(reference_plane_m),
            D=D, z0=z0, load_below=load_below,
        )

    # --- concrete path: NumPy (float64), byte-identical to the pre-AD code ---
    V = np.asarray(plane_voltages, dtype=np.complex128)

    # Matrix-pencil estimate of p = 2 cosh(γΔ) (LS over interior planes).
    Vm, Vc, Vp = V[:-2], V[1:-1], V[2:]
    den = float(np.sum(np.abs(Vc) ** 2))
    if den <= 0.0:
        raise ValueError("plane_voltages are all zero; cannot estimate gamma")
    p = np.sum((Vm + Vp) * np.conj(Vc)) / den
    # cosh is even, so p is invariant under gamma -> -gamma. We resolve the
    # two-fold branch by forcing beta = Im(gamma) > 0 (a forward-propagating
    # phase). This implies the line is PASSIVE (driven from the feed, field
    # decaying away from the load): alpha = Re(gamma) is reported as a magnitude;
    # a genuinely growing wave would be (correctly) unsupported here.
    gamma = np.arccosh(p / 2.0 + 0j) / D
    if np.imag(gamma) < 0.0:           # enforce beta = Im(gamma) > 0
        gamma = np.conj(gamma)
    rec_resid = float(
        np.sqrt(np.sum(np.abs((Vm + Vp) - p * Vc) ** 2) / (np.abs(p) ** 2 * den + 1e-300))
    )

    # Two-wave least squares for A, B (centred for conditioning).
    Phi = np.stack([np.exp(+gamma * (z - z0)), np.exp(-gamma * (z - z0))], axis=1)
    AB, *_ = np.linalg.lstsq(Phi, V, rcond=None)
    A, B = complex(AB[0]), complex(AB[1])
    fit_resid = float(np.linalg.norm(Phi @ AB - V) / (np.linalg.norm(V) + 1e-300))

    zr = float(reference_plane_m) - z0
    a_wave = A * np.exp(+gamma * zr)    # travels -z
    b_wave = B * np.exp(-gamma * zr)    # travels +z
    if load_below:                      # load below the probes: incident is -z (a_wave)
        v_inc, v_ref = a_wave, b_wave
    else:                               # load above the probes: incident is +z (b_wave)
        v_inc, v_ref = b_wave, a_wave
    reflection = v_ref / v_inc if abs(v_inc) > 0.0 else complex(np.nan, np.nan)
    return CoaxialLineReflection(
        gamma=complex(gamma),
        reflection=complex(reflection),
        recurrence_residual=rec_resid,
        fit_residual=fit_resid,
        forward_amp=complex(v_inc),
        backward_amp=complex(v_ref),
    )


class TwoPortWaveSolve(NamedTuple):
    """Result of :func:`solve_two_port_from_wave_amplitudes`.

    ``s_params`` has shape ``(2, 2, n_freqs)`` with ``S[j, i]`` the response at
    port *j* when driving port *i*. ``cond_a`` is the per-frequency 2-norm
    condition number of the incident-wave matrix that was inverted.

    ``cond_a`` bounds DEGENERACY of the two drives — it is not a reliability
    score. It is blind to a systematic incident/outgoing mislabel (stays ~40 on
    a defect that puts ``|S21|`` at 16), and it does not bound accuracy either,
    since it multiplies whatever noise rides on the caller's amplitudes. See
    ``solve_two_port_from_wave_amplitudes`` for the full witness/blind-spot
    table; passivity is the required downstream check.
    """

    s_params: np.ndarray
    cond_a: np.ndarray


def _solve_two_port_from_wave_amplitudes_jnp(a_inc, b_out) -> TwoPortWaveSolve:
    """``jax.numpy`` core of :func:`solve_two_port_from_wave_amplitudes`.

    Reached ONLY for *traced* wave amplitudes (``jax.grad`` / ``jax.jit``); the
    concrete path stays in NumPy, UNCHANGED, so the validated cond/NaN-handling
    behaviour stays byte-identical (untouched code, not just numerically equal).
    ``S = B @ inv(A)`` per frequency, batched over the trailing frequency axis
    via ``jax.numpy``'s native batched ``linalg.inv``/``linalg.cond`` (no
    explicit ``vmap`` needed — both already broadcast over leading axes).

    The concrete path's ill-conditioning WARNING and per-bin NaN-skip-with-
    inv-catch are Python control flow keyed on concrete values with no jnp
    equivalent under trace (mirrors ``_coaxial_line_reflection_jnp``'s own
    note that its contamination check is "NOT evaluated here — can't
    Python-branch on a tracer"); a differentiating caller is responsible for
    keeping the two drives non-degenerate. ``cond_a`` is still RETURNED (not
    skipped), so a caller can inspect it after concretizing the traced result.
    """
    a = jnp.asarray(a_inc)
    b = jnp.asarray(b_out)
    if a.shape != b.shape:
        raise ValueError(
            f"a_inc and b_out must have the same shape, got {a.shape} vs {b.shape}."
        )
    if a.ndim != 3 or a.shape[0] != 2 or a.shape[1] != 2:
        raise ValueError(
            "a_inc/b_out must have shape (2, 2, n_freqs) indexed "
            f"[measured_port, driven_port, freq]; got {a.shape}."
        )
    A = jnp.moveaxis(a, -1, 0)   # (n_freqs, 2, 2)
    B = jnp.moveaxis(b, -1, 0)   # (n_freqs, 2, 2)
    s = jnp.matmul(B, complex_inv(A), precision=HIGHEST)    # (n_freqs, 2, 2)
    cond_a = jnp.linalg.cond(A)             # (n_freqs,)
    return TwoPortWaveSolve(s_params=jnp.moveaxis(s, 0, -1), cond_a=cond_a)


def solve_two_port_from_wave_amplitudes(
    a_inc: np.ndarray,
    b_out: np.ndarray,
    *,
    cond_warn: float = 1.0e3,
    _prefer_jnp: bool = False,
) -> TwoPortWaveSolve:
    """Two-drive 2x2 S-matrix solve from measured wave amplitudes.

    Parameters
    ----------
    a_inc, b_out : (2, 2, n_freqs) complex
        ``a_inc[j, i]`` / ``b_out[j, i]`` are the incident / outgoing wave
        amplitudes measured AT port ``j`` during the run that DRIVES port ``i``,
        both referred to port ``j``'s own reference plane.
    cond_warn : float
        Emit a warning when the incident matrix's condition number exceeds this
        at any frequency. The solve is still returned — ``cond_a`` carries the
        per-frequency number so a caller can mask bins itself.

        **This threshold flags DEGENERACY, not accuracy, and a value below it
        is not a reliability certificate.** The recovered ``S`` inherits the
        caller's amplitude-measurement noise amplified by ``cond(A)``: measured
        on this solve, a 1e-4 relative perturbation of ``A`` gives ~1.3e-3
        relative error in ``S`` at ``cond = 19`` and ~1.3e-2 at ``cond = 199``,
        both far below this default. Bounding that is the caller's job, because
        only the caller knows its own noise floor.

    Returns
    -------
    TwoPortWaveSolve

    Notes
    -----
    Why a two-DRIVE solve rather than the obvious ``S[j, i] = b_j / a_i``:
    that ratio silently assumes the non-driven port receives nothing, so a
    terminator reflection ``Gamma_t`` lands entirely in the reported ``S11``.
    On a through line this makes ``|1 + S11 - S21| = |Gamma_t|`` exactly — a
    floor of 0.08 with rfx's validated matched terminator
    (``tests/unit/sparams/test_coaxial_line_reflection.py`` asserts ``|Gamma| < 0.08``),
    independent of how good the implementation is.

    Both drives together give four equations in four unknowns::

        A = [[a1(1), a1(2)],        B = [[b1(1), b1(2)],
             [a2(1), a2(2)]]             [b2(1), b2(2)]]

        S = B @ inv(A)

    which needs no assumption about the terminator at all — the non-driven
    port's incident wave is one of the measured inputs. Measured: the recovered
    ``|S21|`` stays exact to ~1e-13 for terminator reflections up to
    ``|Gamma_t| = 0.5``. For an ideal planted through line the conditioning has
    the closed form ``cond(A) = (1+|Gamma_t|)/(1-|Gamma_t|)``, i.e. exactly 3.0
    at ``|Gamma_t| = 0.5``; a run through the production extractor at 12 GHz
    (where propagation phase enters) measured 2.4 for the same terminator.
    Either way the matrix is nowhere near singular, which is why no terminator
    improvement is a prerequisite for the two-port wave solve.

    This function is pure linear algebra on amplitudes someone else measured.
    It cannot detect a mis-measured amplitude — that is what the reciprocity
    and analytic-DUT checks around it are for.

    **What reciprocity around this solve can and cannot catch.** Any per-port
    factor ``c`` applied to both ``a`` and ``b`` at one port is the similarity
    ``S -> D S D^-1`` with ``D = diag(1, c)``, which sends ``S21 -> c*S21``
    and ``S12 -> S12/c``:

    * ``|c| != 1`` (an amplitude-calibration error) moves the ratio
      ``|S21|/|S12|`` by ``c^2`` — reciprocity SEES it, including a 5% error.
    * ``|c| == 1`` (a wave-split sign flip, or a reference-plane shift) leaves
      both magnitudes untouched — reciprocity is structurally BLIND to it.

    **And one more family, non-diagonal, that neither local witness sees.**
    Mislabelling incident vs outgoing at one port CONSISTENTLY (i.e. on both
    drives — what an actual wiring bug looks like, as opposed to a one-off) is
    ``S' = N M^-1`` with ``M = [[1,0],[S21,S22]]`` and ``N = [[S11,S12],[0,1]]``,
    giving ``S'21 = -S21/S22`` and ``S'12 = S12/S22``. Those share a magnitude,
    so reciprocity is exactly blind to it, and ``cond(A)`` stays small (measured
    40.8 on an asymmetric DUT, 4% of the default ``cond_warn``) so the
    degeneracy guard is silent as well. What DOES catch it is **passivity**:
    ``S'22 = 1/S22`` explodes for any well-matched passive DUT (measured
    ``|S22'| = 32.8``, column power 317 against a passive limit of 1).
    **Anything built on this solve must
    therefore route its result through the repo's passivity self-check** —
    that check is not optional decoration here, it is the only handle on this
    defect family.

    Note that a symmetric fixture hides this: a through line (``S22 = 0``) or a
    shunt element (``S11 = S22``) makes ``M`` singular, so the defect collapses
    and appears "caught" by accident. Discriminating it needs an ASYMMETRIC DUT.

    All three families are pinned in ``tests/unit/sparams/test_coax_two_port_smatrix.py``.

    AD note
    -------
    ``S`` is differentiable w.r.t. traced wave amplitudes (``jax.grad`` /
    ``jax.jit``): a ``jax.numpy`` core computes ``S = B @ inv(A)`` per
    frequency, batched (:func:`_solve_two_port_from_wave_amplitudes_jnp`),
    while concrete inputs keep this NumPy path UNCHANGED, byte-for-byte
    (including the ill-conditioning warning and per-bin NaN handling, which
    are Python control flow on concrete values with no jnp equivalent under
    trace). Reached by ``Simulation.compute_coaxial_two_port(...,
    eps_scale=...)`` via :func:`rfx.api._sparams._assemble_coaxial_two_port_from_voltages`.

    ``_prefer_jnp`` forces the jnp core even for CONCRETE inputs — used by
    the ``eps_scale`` path so a concrete FD probe (``eps_scale`` provided but
    not under ``jax.grad``) still routes through this SAME jnp core rather
    than silently falling back to this NumPy path (mirrors
    ``coaxial_line_reflection_from_plane_voltages``'s own ``_prefer_jnp``,
    because an AD-vs-FD comparator must hit the
    identical branch on both sides, not two functions that merely agree in
    the well-conditioned case).
    """
    # Traced amplitudes -> jax.numpy core (differentiable); concrete -> NumPy,
    # UNCHANGED below (mirrors coaxial_line_reflection_from_plane_voltages's
    # own dispatch).
    if (
        _prefer_jnp
        or isinstance(a_inc, jax.core.Tracer)
        or isinstance(b_out, jax.core.Tracer)
    ):
        return _solve_two_port_from_wave_amplitudes_jnp(a_inc, b_out)

    a = np.asarray(a_inc, dtype=np.complex128)
    b = np.asarray(b_out, dtype=np.complex128)
    if a.shape != b.shape:
        raise ValueError(
            f"a_inc and b_out must have the same shape, got {a.shape} vs {b.shape}."
        )
    if a.ndim != 3 or a.shape[0] != 2 or a.shape[1] != 2:
        raise ValueError(
            "a_inc/b_out must have shape (2, 2, n_freqs) indexed "
            f"[measured_port, driven_port, freq]; got {a.shape}."
        )
    n_f = int(a.shape[2])
    s = np.full((2, 2, n_f), np.nan + 1j * np.nan, dtype=np.complex128)
    cond = np.full(n_f, np.inf, dtype=np.float64)
    for k in range(n_f):
        A = a[:, :, k]
        B = b[:, :, k]
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue
        cond[k] = float(np.linalg.cond(A))
        if not np.isfinite(cond[k]):
            continue
        try:
            s[:, :, k] = B @ np.linalg.inv(A)
        except np.linalg.LinAlgError:
            pass
    bad = int(np.sum(~np.isfinite(cond) | (cond > float(cond_warn))))
    if bad:
        import warnings as _w

        _w.warn(
            f"solve_two_port_from_wave_amplitudes: the two drives are nearly "
            f"linearly dependent at {bad}/{n_f} "
            f"frequencies (cond > {cond_warn:g}; worst {np.nanmax(cond):.3g}) "
            "— usually both ports seeing essentially the same field, e.g. a "
            "symmetric structure driven identically, or one drive that failed "
            "to excite. S at those bins is degenerate. NOTE this threshold "
            "flags DEGENERACY only: cond below it is NOT a reliability "
            "certificate, because cond(A) also multiplies whatever noise is on "
            "the measured amplitudes (~1.3e-2 relative error in S from 1e-4 "
            "noise at cond=199, which never trips this warning).", AccuracyWarning,
            stacklevel=2,
        )
    return TwoPortWaveSolve(s_params=s, cond_a=cond)
