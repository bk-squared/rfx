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

from rfx.grid import Grid
from rfx.core.yee import EPS_0
from rfx.geometry.csg import Cylinder, Box


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

    # ---- 1. Outer PEC conductor (full outer cylinder, then overwrite inside) ----
    outer_cyl = Cylinder(
        center=pin_center,
        radius=port.outer_radius,
        height=port.pin_length,
        axis=axis,
    )
    outer_mask = outer_cyl.mask(grid)
    eps_r  = jnp.where(outer_mask, 1.0,         materials.eps_r)
    sigma  = jnp.where(outer_mask, PEC_SIGMA,   materials.sigma)

    # ---- 2. PTFE fill (overwrite inside outer conductor) ----
    ptfe_cyl = Cylinder(
        center=pin_center,
        radius=port.outer_radius,   # same outer boundary — sigma will be 0 here
        height=port.pin_length,
        axis=axis,
    )
    # PTFE region = outer cylinder body minus the pin radius
    pin_cyl_mask = Cylinder(
        center=pin_center,
        radius=port.pin_radius,
        height=port.pin_length,
        axis=axis,
    ).mask(grid)
    ptfe_mask = outer_mask & ~pin_cyl_mask
    eps_r  = jnp.where(ptfe_mask, PTFE_EPS_R,  eps_r)
    sigma  = jnp.where(ptfe_mask, 0.0,          sigma)

    # ---- 3. PEC center pin ----
    eps_r  = jnp.where(pin_cyl_mask, 1.0,       eps_r)
    sigma  = jnp.where(pin_cyl_mask, PEC_SIGMA, sigma)

    materials = materials._replace(eps_r=eps_r, sigma=sigma)

    # ---- 4. Fold port impedance into gap cell conductivity ----
    sigma_port = 1.0 / (port.impedance * grid.dx)
    i, j, k = gap_idx
    new_sigma = materials.sigma.at[i, j, k].add(sigma_port)
    materials = materials._replace(sigma=new_sigma)

    return materials


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

    dx = grid.dx
    dt = grid.dt

    # Precompute Cb at the gap cell (constant — materials don't change at runtime)
    eps   = float(materials.eps_r[i, j, k]) * EPS_0
    sigma_val = float(materials.sigma[i, j, k])  # includes sigma_port
    loss  = sigma_val * dt / (2.0 * eps)
    cb    = (dt / eps) / (1.0 + loss)

    def apply_fn(state, t: float):
        """Inject coaxial port source at time t. Call AFTER update_e()."""
        v_src = port.excitation(t)
        field = getattr(state, component)
        field = field.at[i, j, k].add(cb * v_src / dx)
        return state._replace(**{component: field})

    return apply_fn
