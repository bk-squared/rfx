"""Coaxial (SMA-style) probe port for cavity coupling.

Models an SMA connector as:
  - PEC outer conductor (cylinder shell)
  - PTFE dielectric fill (eps_r=2.1) between outer and inner conductors
  - PEC center pin (solid cylinder)
  - Lumped excitation gap at the base (cavity wall interface)

Standard SMA dimensions:
  - Center pin diameter: 1.27 mm (radius 0.635 mm)
  - Outer conductor OD:  4.11 mm (radius 2.055 mm)
  - PTFE fill:           eps_r = 2.1

The pin protrudes into the cavity; direction depends on the face:
  top    → pin extends in -Z direction
  bottom → pin extends in +Z direction
  front  → pin extends in -Y direction
  back   → pin extends in +Y direction
  left   → pin extends in +X direction
  right  → pin extends in -X direction
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import EPS_0, MU_0
from rfx.geometry.csg import Cylinder


# ---------------------------------------------------------------------------
# SMA standard constants
# ---------------------------------------------------------------------------

SMA_PIN_RADIUS   = 0.635e-3   # m — center pin radius (1.27 mm OD)
SMA_OUTER_RADIUS = 2.055e-3   # m — outer conductor radius (4.11 mm OD)
PTFE_EPS_R       = 2.1        # PTFE relative permittivity
PEC_SIGMA        = 1e10       # S/m — effective PEC conductivity


# ---------------------------------------------------------------------------
# Face → (pin axis, pin direction sign, E-field component)
# ---------------------------------------------------------------------------

_FACE_CONFIG: dict[str, tuple[str, float, str]] = {
    "top":    ("z", -1.0, "ez"),   # pin goes -Z into cavity
    "bottom": ("z", +1.0, "ez"),   # pin goes +Z into cavity
    "front":  ("y", -1.0, "ey"),   # pin goes -Y into cavity
    "back":   ("y", +1.0, "ey"),   # pin goes +Y into cavity
    "left":   ("x", +1.0, "ex"),   # pin goes +X into cavity
    "right":  ("x", -1.0, "ex"),   # pin goes -X into cavity
}


# ---------------------------------------------------------------------------
# CoaxialPort descriptor
# ---------------------------------------------------------------------------

class CoaxialPort(NamedTuple):
    """SMA-style coaxial probe port for cavity coupling.

    Models: outer conductor (PEC cylinder), PTFE dielectric fill,
    center pin (PEC rod), lumped excitation gap at the base.

    Parameters
    ----------
    position : (x, y, z) in metres
        Center of port on the cavity wall (gap cell location).
    face : str
        Which wall the port passes through: "top", "bottom", "front",
        "back", "left", or "right".
    pin_length : float
        Protrusion of center pin into the cavity (metres). Default 5 mm.
    pin_radius : float
        Center pin radius (metres). Default 0.635 mm (SMA standard).
    outer_radius : float
        Outer conductor radius (metres). Default 2.055 mm (SMA standard).
    impedance : float
        Port reference impedance in ohms. Default 50.
    excitation : object
        Source waveform (GaussianPulse, ModulatedGaussian, etc.).
    """

    position: tuple       # (x, y, z) — center of port on cavity wall
    face: str             # "top", "bottom", "front", "back", "left", "right"
    pin_length: float     # protrusion into cavity (m), default 5 mm
    pin_radius: float     # center pin radius (m), default 0.635 mm (SMA)
    outer_radius: float   # outer conductor radius (m), default 2.055 mm (SMA)
    impedance: float      # port impedance, default 50 ohm
    excitation: object    # waveform (GaussianPulse etc.)


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


def coaxial_tem_capacitance_per_m(
    inner_radius: float,
    outer_radius: float,
    eps_r: float = PTFE_EPS_R,
) -> float:
    """Analytic capacitance per metre for a lossless coaxial TEM line."""

    import math

    _validate_coaxial_tem_geometry(inner_radius, outer_radius, eps_r)
    return 2.0 * math.pi * EPS_0 * float(eps_r) / math.log(outer_radius / inner_radius)


def coaxial_tem_inductance_per_m(
    inner_radius: float,
    outer_radius: float,
    mu_r: float = 1.0,
) -> float:
    """Analytic inductance per metre for a lossless coaxial TEM line."""

    import math

    _validate_coaxial_tem_geometry(inner_radius, outer_radius, eps_r=1.0, mu_r=mu_r)
    return MU_0 * float(mu_r) * math.log(outer_radius / inner_radius) / (2.0 * math.pi)


def coaxial_tem_characteristic_impedance(
    inner_radius: float,
    outer_radius: float,
    eps_r: float = PTFE_EPS_R,
    mu_r: float = 1.0,
) -> float:
    """Analytic characteristic impedance ``Z0 = sqrt(L'/C')`` for coax TEM."""

    import math

    c_per_m = coaxial_tem_capacitance_per_m(inner_radius, outer_radius, eps_r)
    l_per_m = coaxial_tem_inductance_per_m(inner_radius, outer_radius, mu_r)
    return math.sqrt(l_per_m / c_per_m)


def coaxial_tem_phase_constant(
    freqs_hz,
    eps_r: float = PTFE_EPS_R,
    mu_r: float = 1.0,
):
    """Analytic lossless TEM phase constant ``beta`` for frequency array."""

    from rfx.grid import C0

    freqs = np.asarray(freqs_hz, dtype=np.float64)
    return 2.0 * np.pi * freqs * np.sqrt(float(eps_r) * float(mu_r)) / C0


def coaxial_load_reflection(load_impedance, reference_impedance: float):
    """Closed-form reflection coefficient for a coaxial load."""

    z_load = np.asarray(load_impedance, dtype=np.complex128)
    z0 = complex(reference_impedance)
    if not np.isfinite(z0) or z0.real <= 0.0 or abs(z0.imag) > 0.0:
        raise ValueError(
            "reference_impedance must be a positive real finite impedance, got "
            f"{reference_impedance}"
        )
    with np.errstate(invalid="ignore"):
        gamma = (z_load - z0) / (z_load + z0)
    return np.where(np.isinf(z_load), 1.0 + 0.0j, gamma)


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


def _coaxial_port_geometry(grid: Grid, port: CoaxialPort):
    """Compute center-pin and outer-conductor geometry for the given port.

    Returns
    -------
    axis : str          — "x", "y", or "z"
    direction : float   — +1.0 or -1.0 (pin protrusion direction)
    component : str     — "ex", "ey", or "ez"
    pin_center : tuple  — (x, y, z) center of the pin cylinder
    pin_tip : tuple     — (x, y, z) coordinate of pin tip inside cavity
    gap_index : tuple   — (i, j, k) grid cell for the excitation gap
    """
    if port.face not in _FACE_CONFIG:
        raise ValueError(
            f"face must be one of {list(_FACE_CONFIG.keys())}, got {port.face!r}"
        )

    axis, direction, component = _FACE_CONFIG[port.face]
    px, py, pz = port.position

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]

    # Pin center is offset by half pin_length in the protrusion direction
    half = port.pin_length / 2.0
    offsets = [0.0, 0.0, 0.0]
    offsets[axis_idx] = direction * half

    pin_center = (px + offsets[0], py + offsets[1], pz + offsets[2])
    pin_tip_offsets = [0.0, 0.0, 0.0]
    pin_tip_offsets[axis_idx] = direction * port.pin_length
    pin_tip = (
        px + pin_tip_offsets[0],
        py + pin_tip_offsets[1],
        pz + pin_tip_offsets[2],
    )

    gap_index = grid.position_to_index(port.position)
    return axis, direction, component, pin_center, pin_tip, gap_index


# ---------------------------------------------------------------------------
# setup_coaxial_port
# ---------------------------------------------------------------------------

def setup_coaxial_port(grid: Grid, port: CoaxialPort, materials):
    """Stamp coaxial port geometry into the material arrays.

    Adds three nested cylindrical regions:
      1. Outer PEC conductor (annular shell approximated by outer - inner mask)
      2. PTFE dielectric fill (eps_r=2.1, between outer and pin radii)
      3. PEC center pin (solid cylinder along pin axis)

    Also folds the port impedance into the gap cell conductivity so that
    ``apply_coaxial_port_source`` only needs to inject the source term.

    Parameters
    ----------
    grid : Grid
    port : CoaxialPort
    materials : MaterialArrays

    Returns
    -------
    Updated MaterialArrays with coaxial geometry and port impedance loaded.
    """
    axis, direction, component, pin_center, pin_tip, gap_idx = \
        _coaxial_port_geometry(grid, port)

    _validate_coaxial_tem_geometry(port.pin_radius, port.outer_radius, PTFE_EPS_R)

    # ---- 1. Outer PEC conductor shell ----
    #
    # The original helper painted the full outer cylinder as PEC and then
    # overwrote all non-pin cells inside that cylinder as PTFE, leaving no
    # annular outer-conductor shell at the nominal outer radius.  Keep an
    # explicit one-cell-ish shell while preserving dielectric fill between the
    # center pin and the shell.  This is still a probe-style helper, not a
    # promoted TEM S-parameter API.
    shell_thickness = min(
        float(grid.dx),
        0.5 * (float(port.outer_radius) - float(port.pin_radius)),
    )
    shell_inner_radius = float(port.outer_radius) - shell_thickness
    outer_cyl = Cylinder(
        center=pin_center,
        radius=port.outer_radius,
        height=port.pin_length,
        axis=axis,
    )
    outer_mask = outer_cyl.mask(grid)
    shell_inner_mask = Cylinder(
        center=pin_center,
        radius=shell_inner_radius,
        height=port.pin_length,
        axis=axis,
    ).mask(grid)
    outer_shell_mask = outer_mask & ~shell_inner_mask
    eps_r  = jnp.where(outer_shell_mask, 1.0,         materials.eps_r)
    sigma  = jnp.where(outer_shell_mask, PEC_SIGMA,   materials.sigma)

    # ---- 2. PTFE fill between the pin and outer shell ----
    pin_cyl_mask = Cylinder(
        center=pin_center,
        radius=port.pin_radius,
        height=port.pin_length,
        axis=axis,
    ).mask(grid)
    ptfe_mask = shell_inner_mask & ~pin_cyl_mask
    eps_r  = jnp.where(ptfe_mask, PTFE_EPS_R,  eps_r)
    sigma  = jnp.where(ptfe_mask, 0.0,          sigma)

    # ---- 3. PEC center pin ----
    eps_r  = jnp.where(pin_cyl_mask, 1.0,       eps_r)
    sigma  = jnp.where(pin_cyl_mask, PEC_SIGMA, sigma)

    materials = materials._replace(eps_r=eps_r, sigma=sigma)

    # ---- 4. Fold port impedance into gap cell conductivity ----
    from rfx.sources.sources import port_sigma as _port_sigma
    i, j, k = gap_idx
    sp = _port_sigma(grid, gap_idx, component, port.impedance)
    new_sigma = materials.sigma.at[i, j, k].add(sp)
    materials = materials._replace(sigma=new_sigma)

    return materials


def add_coaxial_matched_termination(
    grid: Grid,
    port: CoaxialPort,
    materials,
    *,
    target_impedance: float,
    axial_offset_cells: int = 1,
):
    """Add a distributed annular resistive load to a coaxial port.

    Places conductive material in the PTFE annulus on a single
    cross-section, with sigma chosen so the radial resistance from the
    inner conductor (pin) to the outer conductor (shell) matches
    ``target_impedance``. This is the line-end equivalent of soldering a
    Z₀-valued resistor between pin and shell at one axial position.

    For an annulus of inner radius ``a``, outer radius ``b``, and axial
    thickness ``dz``, the radial resistance through a uniform conductor of
    conductivity ``sigma`` is::

        R = log(b/a) / (2π · dz · sigma)

    so the matching ``sigma = log(b/a) / (2π · dz · Z_target)``. This
    produces a clean impedance match at the FDTD's design frequency; the
    discrete absorption at off-design frequencies has the usual lossy-
    line behaviour and is not unity-power.

    Parameters
    ----------
    grid : Grid
    port : CoaxialPort
        Must already have been stamped via :func:`setup_coaxial_port`
        before calling this helper.
    materials : MaterialArrays
    target_impedance : float
        The line impedance to match, typically the closed-form
        ``coaxial_tem_characteristic_impedance(...)``.
    axial_offset_cells : int
        How many Yee cells away from the pin tip the load slice sits,
        stepping *into* the coax line (toward the gap, opposite the
        forward direction). ``1`` puts the load one cell inside the line
        from the pin tip — the load slice still sits inside the PTFE
        annulus between pin and shell, so the radial pin-to-shell
        integration path is well-defined. Larger values leave more
        unloaded PTFE line between the source and the termination.

    Returns
    -------
    Updated MaterialArrays with the matched-termination slice stamped.

    Notes
    -----
    The helper deliberately writes only into PTFE-region cells (radii in
    ``[pin_radius, shell_inner]``); the PEC pin and PEC shell at the load
    z-slice are left untouched, so the analytic ``log(b/a)`` integration
    path from pin to shell is preserved and the resulting termination has
    the correct discrete radial impedance.
    """

    if not np.isfinite(target_impedance) or target_impedance <= 0.0:
        raise ValueError(
            f"target_impedance must be a positive finite resistance, got "
            f"{target_impedance}"
        )

    axis, direction, _, pin_center, pin_tip, _ = _coaxial_port_geometry(
        grid, port
    )
    if axis != "z":
        raise NotImplementedError(
            "add_coaxial_matched_termination currently supports only z-axis "
            "coaxial ports (face='top'/'bottom')."
        )

    shell_thickness = min(
        float(grid.dx),
        0.5 * (float(port.outer_radius) - float(port.pin_radius)),
    )
    shell_inner_radius = float(port.outer_radius) - shell_thickness
    log_ratio = float(np.log(shell_inner_radius / float(port.pin_radius)))
    dz = float(grid.dx)

    sigma_load = log_ratio / (2.0 * np.pi * dz * float(target_impedance))

    # Forward direction along z (face='top' is -z, face='bottom' is +z).
    # axial_offset_cells > 0 means "step into the line" (against forward
    # direction, toward the gap), so the load slice sits inside the
    # stamped PTFE annulus between the pin and the outer shell.
    pin_tip_idx = int(grid.position_to_index(pin_tip)[2])
    load_idx_z = pin_tip_idx - int(direction) * int(axial_offset_cells)
    if load_idx_z < 0 or load_idx_z >= grid.shape[2]:
        raise ValueError(
            f"matched-termination slice z-index {load_idx_z} falls outside "
            f"grid (axial_offset_cells={axial_offset_cells})"
        )

    sigma_np = np.array(materials.sigma)
    eps_r_np = np.array(materials.eps_r)
    cells_stamped = 0
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            r = float(np.hypot(x - port.position[0], y - port.position[1]))
            if not (float(port.pin_radius) <= r <= shell_inner_radius):
                continue
            # Skip cells already stamped as PEC by setup_coaxial_port.
            if sigma_np[i, j, load_idx_z] >= 0.5 * PEC_SIGMA:
                continue
            sigma_np[i, j, load_idx_z] = sigma_load
            eps_r_np[i, j, load_idx_z] = 1.0
            cells_stamped += 1

    if cells_stamped == 0:
        raise ValueError(
            "add_coaxial_matched_termination stamped 0 PTFE annulus cells; "
            "check axial_offset_cells and grid resolution"
        )

    return materials._replace(
        sigma=jnp.asarray(sigma_np),
        eps_r=jnp.asarray(eps_r_np),
    )


def add_coaxial_pec_end_cap(
    grid: Grid,
    port: CoaxialPort,
    materials,
    *,
    axial_offset_cells: int = 0,
):
    """Close the outer shell of a coaxial port with a PEC end-cap disk.

    Stamps a PEC-filled disk across the full outer-conductor radius at a
    chosen axial position (defaulting to the original line end at
    ``pin_tip``). Combined with :func:`add_coaxial_open_termination`,
    this forms a proper open-circuit cup: a section of PTFE-filled
    circular waveguide closed at the far end by PEC, where the inner
    conductor terminates inside the cup and the outer conductor + cap
    isolate the line from the surrounding vacuum cavity.

    Without an end-cap, a retracted pin alone leaves the line "open" to
    the rest of the cavity through the shell end (since the shell
    rasterises only along ``pin_length``); below-cutoff evanescent
    decay is small over a few Yee cells, and a cavity-floor reflection
    dominates the observed Γ. The end-cap eliminates that escape path.

    Parameters
    ----------
    grid : Grid
    port : CoaxialPort
        Must already have been stamped via :func:`setup_coaxial_port`.
    materials : MaterialArrays
    axial_offset_cells : int
        Axial offset (in Yee cells) from the original pin tip. ``0``
        places the cap exactly at the line end (one cell past the
        last pin/shell cell). Negative values move the cap into the
        line; positive values move it out into the surrounding vacuum.

    Returns
    -------
    Updated MaterialArrays with the PEC end-cap stamped.
    """

    axis, direction, _, pin_center, pin_tip, _ = _coaxial_port_geometry(
        grid, port
    )
    if axis != "z":
        raise NotImplementedError(
            "add_coaxial_pec_end_cap currently supports only z-axis "
            "coaxial ports (face='top'/'bottom')."
        )

    sigma_np = np.array(materials.sigma)
    eps_r_np = np.array(materials.eps_r)

    # Find the forward-most z index where the outer shell is stamped,
    # then place the cap one cell past it (in the forward direction).
    # The shell is identified by PEC at radii in the shell band.
    shell_thickness = min(
        float(grid.dx),
        0.5 * (float(port.outer_radius) - float(port.pin_radius)),
    )
    shell_inner_radius = float(port.outer_radius) - shell_thickness
    nz = grid.shape[2]
    forward = int(direction)
    z_iter = range(nz - 1, -1, -1) if forward > 0 else range(nz)
    shell_tip_z: int | None = None
    for z_idx in z_iter:
        for i in range(grid.nx):
            x = (i - grid.pad_x_lo) * grid.dx
            for j in range(grid.ny):
                y = (j - grid.pad_y_lo) * grid.dx
                r = float(np.hypot(x - port.position[0], y - port.position[1]))
                if not (shell_inner_radius <= r <= float(port.outer_radius)):
                    continue
                if sigma_np[i, j, z_idx] >= 0.5 * PEC_SIGMA:
                    shell_tip_z = z_idx
                    break
            if shell_tip_z is not None:
                break
        if shell_tip_z is not None:
            break
    if shell_tip_z is None:
        raise ValueError(
            "add_coaxial_pec_end_cap could not locate the shell tip; ensure "
            "setup_coaxial_port has been called for this port"
        )

    # Cap one cell past the shell tip, in the forward direction.
    cap_z = shell_tip_z + forward * (1 + int(axial_offset_cells))
    if cap_z < 0 or cap_z >= nz:
        raise ValueError(
            f"PEC end-cap z-index {cap_z} falls outside grid "
            f"(axial_offset_cells={axial_offset_cells})"
        )

    cells_stamped = 0
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            r = float(np.hypot(x - port.position[0], y - port.position[1]))
            if r > float(port.outer_radius):
                continue
            sigma_np[i, j, cap_z] = float(PEC_SIGMA)
            eps_r_np[i, j, cap_z] = 1.0
            cells_stamped += 1

    if cells_stamped == 0:
        raise ValueError(
            "add_coaxial_pec_end_cap stamped 0 cells; check geometry"
        )

    return materials._replace(
        sigma=jnp.asarray(sigma_np),
        eps_r=jnp.asarray(eps_r_np),
    )


def add_coaxial_open_termination(
    grid: Grid,
    port: CoaxialPort,
    materials,
    *,
    pin_retract_cells: int = 1,
):
    """Open-circuit termination via pin retraction (inner-conductor cut-back).

    The center pin is shortened from its original length by
    ``pin_retract_cells`` Yee cells, while the outer shell, PTFE fill,
    and gap excitation are left intact. Beyond the new pin tip the
    cross-section is *PTFE-filled outer shell with no inner conductor*,
    which is the geometry of a PTFE-loaded circular waveguide. The
    lowest such waveguide TE/TM mode for an SMA shell sits well above
    the GHz–10 GHz test band (TE₁₁ cutoff ≈ 45 GHz inside ε_r=2.1
    PTFE), so any wave reaching the pin-tip step is below cutoff in the
    rest of the line and decays evanescently. The reflection back
    toward the source is therefore an open-circuit-like ``Γ ≈ +1``
    (with a small fringing-capacitance phase) at all sub-cutoff
    frequencies.

    Parameters
    ----------
    grid : Grid
    port : CoaxialPort
        Must already have been stamped via :func:`setup_coaxial_port`
        before calling this helper.
    materials : MaterialArrays
    pin_retract_cells : int
        Number of Yee cells to retract the pin from its original
        ``port.pin_length`` end. ``1`` removes one cell of pin at the
        far end and replaces it with PTFE. Larger values produce a
        deeper "cup" in the outer shell.

    Returns
    -------
    Updated MaterialArrays with the retracted pin geometry stamped.
    """

    if pin_retract_cells <= 0:
        raise ValueError(
            f"pin_retract_cells must be positive, got {pin_retract_cells}"
        )

    axis, direction, _, pin_center, pin_tip, _ = _coaxial_port_geometry(
        grid, port
    )
    if axis != "z":
        raise NotImplementedError(
            "add_coaxial_open_termination currently supports only z-axis "
            "coaxial ports (face='top'/'bottom')."
        )

    sigma_np = np.array(materials.sigma)
    eps_r_np = np.array(materials.eps_r)

    # Pre-compute the (i, j) cells that fall inside the pin radius and
    # were stamped by setup_coaxial_port as PEC. This avoids relying on
    # a separate ``pin_tip_idx`` computation that may be off-by-one
    # against the Cylinder mask's rasterisation convention.
    pin_ij: list[tuple[int, int]] = []
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            r = float(np.hypot(x - port.position[0], y - port.position[1]))
            if r <= float(port.pin_radius):
                pin_ij.append((i, j))

    # Walk z indices along the forward direction and collect the first
    # ``pin_retract_cells`` z-indices where any pin (i, j) cell is
    # stamped PEC. These are the "tip" cells in the wave-forward sense.
    nz = grid.shape[2]
    forward = int(direction)
    z_iter = range(nz - 1, -1, -1) if forward > 0 else range(nz)
    pin_tip_z_indices: list[int] = []
    for z_idx in z_iter:
        if any(sigma_np[i, j, z_idx] >= 0.5 * PEC_SIGMA for (i, j) in pin_ij):
            pin_tip_z_indices.append(z_idx)
            if len(pin_tip_z_indices) >= int(pin_retract_cells):
                break

    cells_retracted = 0
    for z_idx in pin_tip_z_indices:
        for (i, j) in pin_ij:
            if sigma_np[i, j, z_idx] >= 0.5 * PEC_SIGMA:
                sigma_np[i, j, z_idx] = 0.0
                eps_r_np[i, j, z_idx] = float(PTFE_EPS_R)
                cells_retracted += 1

    if cells_retracted == 0:
        raise ValueError(
            "add_coaxial_open_termination retracted 0 pin cells; check "
            "pin_retract_cells and the original pin geometry"
        )

    return materials._replace(
        sigma=jnp.asarray(sigma_np),
        eps_r=jnp.asarray(eps_r_np),
    )


# ---------------------------------------------------------------------------
# make_coaxial_port_source / apply_coaxial_port_source
# ---------------------------------------------------------------------------

def make_coaxial_port_source(grid: Grid, port: CoaxialPort, materials, n_steps: int):
    """Return a callable that injects the coaxial port excitation each timestep.

    The source term is injected at the gap cell (cavity-wall interface).
    Port impedance is already folded into materials by ``setup_coaxial_port``,
    so this function only adds the voltage source term:

        E[gap] += Cb * V_src / dx

    Parameters
    ----------
    grid : Grid
    port : CoaxialPort
    materials : MaterialArrays
        Must already have port impedance loaded (call setup_coaxial_port first).
    n_steps : int
        Total number of time steps (used for pre-computing Cb if desired).

    Returns
    -------
    apply_fn : callable(state, t) -> state
        Stateless function that injects the source at time t.
    """
    _, _, _, _, _, gap_idx = _coaxial_port_geometry(grid, port)
    i, j, k = gap_idx
    axis, _, component, _, _, _ = _coaxial_port_geometry(grid, port)

    from rfx.sources.sources import port_d_parallel as _port_d_parallel
    d_par = _port_d_parallel(grid, gap_idx, component)
    dt = grid.dt

    eps   = float(materials.eps_r[i, j, k]) * EPS_0
    sigma_val = float(materials.sigma[i, j, k])
    loss  = sigma_val * dt / (2.0 * eps)
    cb    = (dt / eps) / (1.0 + loss)

    def apply_fn(state, t: float):
        """Inject coaxial port source at time t. Call AFTER update_e()."""
        v_src = port.excitation(t)
        field = getattr(state, component)
        field = field.at[i, j, k].add(cb * v_src / d_par)
        return state._replace(**{component: field})

    return apply_fn


# ---------------------------------------------------------------------------
# Distributed TEM plane-source builder (M67 prototype → public)
# ---------------------------------------------------------------------------
#
# These helpers build the per-cell ``SourceSpec`` / ``MagneticSourceSpec`` lists
# for a transverse E/M coaxial plane source.  They are the public surface that
# ``compute_coaxial_s_matrix`` uses to assemble each driven-port FDTD run.  The
# helpers are intentionally low-level: a higher-level Simulation method is
# responsible for plane index, DFT probe setup, V/I extraction, and S-matrix
# bookkeeping.
#
# Reference plane convention:
#   The plane source is injected on a single transverse cross-section of the
#   coaxial port at axial position ``z = pin_center[axis]``.  The same plane
#   is the V/I extraction reference plane.  Callers can shift the reference
#   plane downstream by passing ``reference_plane_axial_index_offset`` to
#   ``build_coaxial_tem_plane_source_specs``; the V/I extractor must use the
#   matching offset for the calibration to remain self-consistent.


class CoaxialPlaneSourceSpec(NamedTuple):
    """Specs returned by :func:`build_coaxial_tem_plane_source_specs`.

    The lists carry one ``SourceSpec`` / ``MagneticSourceSpec`` per cell on
    the coaxial cross-section between ``pin_radius`` and ``outer_radius``.
    Cells whose radius falls outside that annulus are skipped.
    """

    electric_sources: tuple
    magnetic_sources: tuple
    plane_axial_index: int
    plane_axis: str
    source_cell_count: int
    field_scale: float
    magnetic_ratio: float
    z_tem_ohm: float


def build_coaxial_tem_plane_source_specs(
    *,
    grid: "Grid",
    port: CoaxialPort,
    n_steps: int,
    field_scale: float = 1.0e4,
    magnetic_ratio: float = 1.0,
    reference_plane_axial_index_offset: int = 0,
    eps_r: float = PTFE_EPS_R,
) -> CoaxialPlaneSourceSpec:
    """Return TFSF-style transverse E/M source specs for a coaxial port.

    Bakes a Yee-half-step-correct one-side TFSF correction pair into per-
    cell ``SourceSpec`` / ``MagneticSourceSpec`` lists. The additive
    ``src_vals`` / ``mag_src_vals`` injection in :func:`rfx.simulation.run`
    has the same leapfrog timing as the waveguide-port TFSF apply pair
    (``apply_waveguide_port_h`` / ``apply_waveguide_port_e``) so the
    additive specs reproduce its unidirectional injection without
    requiring a separate apply hook in the simulation runner.

    Three structural fixes vs the M67/M72 prototype-promotion of the
    plane-source (whose bidirectional emission held PEC-short ``|S11|``
    near 0.2-0.4 regardless of termination):

    * **Cross-coupling**: H sources are driven by the E mode profile
      (``e_radial`` decomposed into Cartesian ``(cos φ, sin φ)``), and
      E sources are driven by the H mode profile (``h_φ`` decomposed
      into Cartesian ``(-sin φ, cos φ)``). The previous implementation
      drove H from H and E from E, which does not enforce the curl
      coupling that makes the wave unidirectional.
    * **Spatial half-cell offset**: H sources move from the source plane
      to the upstream half-cell on the scattered side of the TFSF
      boundary (``plane_index - 1`` for ``+z`` forward, ``plane_index``
      for ``-z`` forward).
    * **Yee half-step phase**: ``h_inc(t_n) = waveform(t_n + Δt) / Z₀``
      with ``Δt = dt/2 + dz·sqrt(εr)/(2c)``; the previous version sampled
      both E and H at integer ``n·dt`` so the H source carried no Yee
      half-step phase relative to E.

    The :class:`CoaxialPlaneSourceSpec` return contract is unchanged:
    ``plane_axial_index`` is still the *E* reference plane (where the V/I
    extractor reads). The mag-source list now lives one cell upstream.

    For ``face='top'`` (pin going ``-z`` into cavity), forward = ``-z``
    and the pattern follows the waveguide ``-x``-direction conventions.
    For ``face='bottom'``, forward = ``+z`` and the ``+x`` pattern is
    used.

    Parameters
    ----------
    grid:
        The simulation grid.
    port:
        :class:`CoaxialPort` with geometry and excitation waveform.
    n_steps:
        Number of FDTD timesteps the simulation will execute. The waveform
        is materialised at each step so this must match the runner.
    field_scale:
        Linear scale on the radial E waveform. Increase to lift the plane
        signal above DFT noise; the V/I extraction is amplitude-linear.
    magnetic_ratio:
        Multiplier on the ``H`` waveform after the analytic ``1/Z_TEM``
        factor (already baked into ``h_inc`` so a unit ratio means
        Poynting-balanced TEM). Kept for diagnostic ablation; production
        callers should leave it at ``1.0``.
    reference_plane_axial_index_offset:
        Shift of the source plane (and therefore the V/I reference plane)
        relative to ``port.pin_center`` along the port axis. ``0`` injects
        at the pin centre plane.
    eps_r:
        Coaxial dielectric permittivity for the analytic ``Z_TEM`` and the
        Yee-half-step delay (``v_phase = c / sqrt(εr)``). Default
        :data:`PTFE_EPS_R` matches the SMA helper.

    Returns
    -------
    CoaxialPlaneSourceSpec
        Lists of per-cell source specs plus bookkeeping needed by the V/I
        extractor.
    """

    # Lazy imports so the source helpers stay JIT-friendly for callers that
    # import this module without a full simulation pipeline.
    from rfx.simulation import MagneticSourceSpec, SourceSpec
    from rfx.grid import C0 as _C0
    import jax
    import jax.numpy as jnp

    axis, direction, _component, pin_center, _pin_tip, _gap_idx = (
        _coaxial_port_geometry(grid, port)
    )
    if axis != "z":
        raise NotImplementedError(
            "build_coaxial_tem_plane_source_specs currently supports only the "
            "default z-axis coaxial port (face='top'/'bottom')."
        )

    # Forward direction: +1 for face='bottom' (pin goes +z), -1 for face='top'.
    forward_sign = float(direction)

    plane_index = int(grid.position_to_index(pin_center)[2]) + int(
        reference_plane_axial_index_offset
    )

    # TFSF plane indices and signs mirror waveguide_port.py:1303-1308 / 1374-1386.
    if forward_sign > 0:  # +z forward, mirrors waveguide "+x" pattern.
        h_plane_idx = plane_index - 1
        e_plane_idx = plane_index
        h_sign = -1.0
        e_sign = -1.0
    else:  # -z forward, mirrors waveguide "-x" pattern.
        h_plane_idx = plane_index
        e_plane_idx = plane_index + 1
        h_sign = +1.0
        e_sign = -1.0

    z0 = coaxial_tem_characteristic_impedance(
        port.pin_radius,
        port.outer_radius,
        eps_r,
    )
    log_ratio = float(np.log(port.outer_radius / port.pin_radius))

    # Local (intrinsic) impedance of the dielectric fill: η = sqrt(μ/ε).
    # In a coax TEM mode the LOCAL field ratio is E_r/H_φ = η, while
    # V/I = Z_TEM = η · log(b/a)/(2π) is the LINE impedance. The TFSF
    # injection works on local fields, so h_inc must carry the 1/η
    # factor — using 1/Z_TEM here would over-inject E by the geometry
    # factor 2π/log(b/a) (Z_med/Z_TEM ratio).
    z_med = float(np.sqrt(MU_0 / (float(eps_r) * EPS_0)))

    # Yee-staggered time delay: h_inc samples the wave at the upstream H
    # half-cell at Yee-H time (n+1/2)·dt. Forward TEM phase velocity in
    # the dielectric is v = c / sqrt(εr); the half-cell propagation delay
    # plus the leapfrog half-step is the same for both directions because
    # "upstream" is always one half-cell against the wave's motion.
    dz = float(grid.dx)
    dt_step = float(grid.dt)
    delta_t = 0.5 * dt_step + 0.5 * dz * float(np.sqrt(eps_r)) / float(_C0)

    # E and H source amplitude tables.
    times_e = jnp.arange(int(n_steps), dtype=jnp.float32) * jnp.float32(dt_step)
    times_h = times_e + jnp.float32(delta_t)
    e_inc_table = jax.vmap(port.excitation)(times_e)
    h_inc_table = jax.vmap(port.excitation)(times_h) / jnp.float32(z_med)

    # TFSF coefficients (mirror waveguide_port.py:1297, 1338, with ε
    # replaced by εr·ε₀ for the dielectric fill — the source plane sits
    # inside PTFE, so the E update at this cell uses the dielectric cb
    # coefficient. ``waveguide_port.py`` uses bare ``EPS_0`` because
    # waveguides are vacuum-filled by convention; coax in PTFE needs the
    # εr correction so the per-step E-injection matches the physical
    # ``cb = dt/(εr·ε₀·dz)`` of the FDTD update.
    coeff_h = jnp.float32(dt_step / (MU_0 * dz))
    coeff_e = jnp.float32(dt_step / (float(eps_r) * EPS_0 * dz))

    # Inner edge of the PEC outer-conductor shell stamped by
    # ``setup_coaxial_port``. Source cells must stay strictly inside the
    # PTFE annulus [pin_radius, shell_inner]; injecting at radii in
    # [shell_inner, outer_radius] hits PEC cells whose E is zeroed every
    # step by ``apply_pec_mask``, breaking the TFSF cancellation
    # symmetry.
    shell_thickness = min(
        float(dz),
        0.5 * (float(port.outer_radius) - float(port.pin_radius)),
    )
    shell_inner_radius = float(port.outer_radius) - shell_thickness

    # Time-series amplitude scales (cell-independent factors lifted out).
    h_factor = jnp.float32(h_sign) * coeff_h * jnp.float32(field_scale) * e_inc_table
    e_factor = (
        jnp.float32(-e_sign)
        * coeff_e
        * jnp.float32(field_scale)
        * jnp.float32(magnetic_ratio)
        * h_inc_table
    )

    electric_sources: list = []
    magnetic_sources: list = []
    source_cell_count = 0
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            du = x - port.position[0]
            dv = y - port.position[1]
            radius = float(np.hypot(du, dv))
            if not (
                float(port.pin_radius) <= radius <= shell_inner_radius
            ):
                continue
            cos_phi = du / radius
            sin_phi = dv / radius

            # TEM mode shapes (1/r), normalised so V = ∫E_r dr = 1 for unit
            # ``field_scale``. h_phi shape carries the same 1/r profile; the
            # 1/Z_TEM factor relating H_φ to E_r is baked into h_inc_table.
            mode_shape = 1.0 / (radius * log_ratio)
            source_cell_count += 1

            # H-side TFSF correction at h_plane_idx, driven by E mode profile
            # (cf. apply_waveguide_port_h: h_u += sign·coeff·src·ez_profile,
            #                              h_v += -sign·coeff·src·ey_profile).
            # For z-normal coax: e_u=Ex, e_v=Ey, h_u=Hx, h_v=Hy.
            #   Hx (h_u) += h_sign·coeff_h·e_inc·Ey_profile
            #            = h_sign·coeff_h·e_inc·(mode_shape·sin_phi)
            #   Hy (h_v) += -h_sign·coeff_h·e_inc·Ex_profile
            #            = -h_sign·coeff_h·e_inc·(mode_shape·cos_phi)
            hx_shape = jnp.float32(mode_shape * sin_phi)
            hy_shape = jnp.float32(-mode_shape * cos_phi)

            if abs(sin_phi) > 1e-12:
                magnetic_sources.append(
                    MagneticSourceSpec(
                        i=i,
                        j=j,
                        k=h_plane_idx,
                        component="hx",
                        waveform=(hx_shape * h_factor).astype(jnp.float32),
                    )
                )
            if abs(cos_phi) > 1e-12:
                magnetic_sources.append(
                    MagneticSourceSpec(
                        i=i,
                        j=j,
                        k=h_plane_idx,
                        component="hy",
                        waveform=(hy_shape * h_factor).astype(jnp.float32),
                    )
                )

            # E-side TFSF correction at e_plane_idx, driven by H mode profile
            # (cf. apply_waveguide_port_e: e_v += sign·coeff·h_inc·hy_profile,
            #                               e_u += -sign·coeff·h_inc·hz_profile).
            #   Ex (e_u) += -e_sign·coeff_e·h_inc·Hy_profile
            #            = -e_sign·coeff_e·h_inc·(mode_shape·cos_phi)
            #   Ey (e_v) += e_sign·coeff_e·h_inc·Hx_profile
            #            = e_sign·coeff_e·h_inc·(-mode_shape·sin_phi)
            #            = -e_sign·coeff_e·h_inc·(mode_shape·sin_phi)
            # Both Ex and Ey end up scaled by the same `-e_sign·coeff_e·h_inc`
            # which is the e_factor lifted out above.
            ex_shape = jnp.float32(mode_shape * cos_phi)
            ey_shape = jnp.float32(mode_shape * sin_phi)

            if abs(cos_phi) > 1e-12:
                electric_sources.append(
                    SourceSpec(
                        i=i,
                        j=j,
                        k=e_plane_idx,
                        component="ex",
                        waveform=(ex_shape * e_factor).astype(jnp.float32),
                    )
                )
            if abs(sin_phi) > 1e-12:
                electric_sources.append(
                    SourceSpec(
                        i=i,
                        j=j,
                        k=e_plane_idx,
                        component="ey",
                        waveform=(ey_shape * e_factor).astype(jnp.float32),
                    )
                )

    return CoaxialPlaneSourceSpec(
        electric_sources=tuple(electric_sources),
        magnetic_sources=tuple(magnetic_sources),
        plane_axial_index=plane_index,
        plane_axis=axis,
        source_cell_count=source_cell_count,
        field_scale=float(field_scale),
        magnetic_ratio=float(magnetic_ratio),
        z_tem_ohm=float(z0),
    )


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

    This is the convenience wrapper used by ``compute_coaxial_s_matrix``: it
    takes the DFT-plane outputs (one per Ex/Ey/Hx/Hy component) and delegates
    to :func:`coaxial_tem_reference_plane_vi_from_cartesian_plane`, which
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
    ``current_sign = +1``. Without this, ``compute_coaxial_s_matrix``
    reports ``|S11| = 1/Γ`` instead of ``Γ`` for ``face='top'`` ports —
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
