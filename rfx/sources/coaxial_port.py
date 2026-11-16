"""Coaxial (SMA-style) probe port for cavity coupling.

Models an SMA connector as:
  - PEC outer conductor (cylinder shell)
  - PTFE dielectric fill (eps_r=2.1) between outer and inner conductors
  - PEC center pin (solid cylinder)

The removed ``setup_coaxial_port`` API also built a lumped excitation gap.
The remaining coaxial-line helpers use TEM excitation and PEC edge conductors.

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

from rfx.errors import AccuracyWarning
from typing import NamedTuple

import jax
import jax.numpy as jnp

from rfx._internal._precision import HIGHEST
from rfx.core.jax_utils import complex_inv
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import EPS_0, MU_0
from rfx.model.coax_stamp import (
    SMA_PIN_RADIUS,
    SMA_OUTER_RADIUS,
    PTFE_EPS_R,
    SHELL_THICKNESS_M,
    PEC_SIGMA,
    _FACE_CONFIG,
    _require_coaxial_aperture,
    _coaxial_port_geometry,
    CoaxialPlaneSourceSpec,
    _coaxial_tem_edge_profile,
    build_coaxial_tem_plane_source_specs,
    stamp_coaxial_line,
    stamp_coaxial_short_plane,
    stamp_coaxial_annular_resistor,
)

from rfx.measurement.coax_records import (
    CoaxialTEMReferencePlaneVI,
    CoaxialTEMCartesianPlaneVI,
    coaxial_tem_reference_plane_vi,
    coaxial_tem_reference_plane_vi_from_cartesian_plane,
    coaxial_tem_reference_plane_s11,
    _validate_monotone_axis,
    _validate_plane_field,
    _interp_bilinear_plane,
    _validate_coaxial_tem_geometry,
    extract_coaxial_plane_vi_from_dft,
    CoaxialLineReflection,
    coaxial_line_plane_voltage,
    coaxial_line_plane_voltage_jnp,
    _coaxial_line_reflection_jnp,
    coaxial_line_reflection_from_plane_voltages,
    TwoPortWaveSolve,
    _solve_two_port_from_wave_amplitudes_jnp,
    solve_two_port_from_wave_amplitudes,
)



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
    terminates: tuple = ()


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
