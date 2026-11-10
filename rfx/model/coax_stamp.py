"""Coaxial line stamping into material arrays and TEM plane-source specs.

The CoaxialPort declaration and V/I measurement stay in rfx.sources.coaxial_port.
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import EPS_0, MU_0
from rfx.geometry.csg import Cylinder
from rfx.sources._laplace import _solve_laplace_2d
from rfx.sources.coax_impedance import coaxial_built_line_impedance
from rfx.sources.sources import stamp_lumped_sigma

if TYPE_CHECKING:
    from rfx.sources.coaxial_port import CoaxialPort



# ---------------------------------------------------------------------------
# SMA standard constants
# ---------------------------------------------------------------------------

SMA_PIN_RADIUS   = 0.635e-3   # m — center pin radius (1.27 mm OD)
SMA_OUTER_RADIUS = 2.055e-3   # m — outer conductor radius (4.11 mm OD)
PTFE_EPS_R       = 2.1        # PTFE relative permittivity
# Outer-conductor wall thickness, in METRES. Fixed in metres and not in cells
# on purpose: a thickness of one cell makes the line's own cross-section move
# with the mesh, so a dx ladder refines the GEOMETRY as well as the mesh and the
# rungs do not realize one guide. 1 mm rasterizes to at least one cell at every
# cell size this lane supports and leaves clearance to the absorber on the 8 mm
# lateral domain every coax fixture here uses.
SHELL_THICKNESS_M = 1.0e-3
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


def _require_coaxial_aperture(grid, axes, center, radius):
    """A radial node aperture needs both rims; periodic images are unsupported."""
    from rfx._internal._periodic import interval_coordinates
    for axis, coordinate in zip(axes, center):
        if "xyz"[axis] not in getattr(grid, "periodic_axes", ""):
            continue
        lo, hi = interval_coordinates(grid, axis, coordinate-radius, coordinate+radius)
        if lo <= 0 or hi >= float(grid.domain[axis]):
            raise ValueError(
                f"Coaxial node-inclusive aperture on periodic axis {'xyz'[axis]!r}: "
                f"interval [{lo}, {hi}] m touches the seam of period "
                f"L={grid.domain[axis]} m; radial sampling requires an interior aperture.")


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

    from rfx._internal._periodic import interval_coordinates
    for a in range(3):
        if a == axis_idx:
            interval_coordinates(grid, a, port.position[a], pin_tip[a])
        else:
            _require_coaxial_aperture(grid, (a,), (port.position[a],), port.outer_radius)
    gap_index = grid.position_to_index(port.position)
    return axis, direction, component, pin_center, pin_tip, gap_index


# ---------------------------------------------------------------------------
# Distributed TEM plane-source builder (M67 prototype → public)
# ---------------------------------------------------------------------------
#
# These helpers build the per-cell ``SourceSpec`` / ``MagneticSourceSpec`` lists
# for a transverse E/M coaxial plane source.  The coax line lanes
# (``compute_coaxial_line_reflection`` and the lanes built like it) drive their
# FDTD runs with them.  The helpers are intentionally low-level: a higher-level method is
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

    The lists carry the nonzero transverse edges of the realized TEM mode.
    The electric profile is the discrete gradient of a unit-voltage
    electrostatic potential on the staircased conductor cross-section.
    ``z_tem_ohm`` is the built line impedance from that unit-voltage profile.
    """

    electric_sources: tuple
    magnetic_sources: tuple
    plane_axial_index: int
    plane_axis: str
    source_cell_count: int
    field_scale: float
    magnetic_ratio: float
    z_tem_ohm: float


def _coaxial_tem_edge_profile(grid, port, e_plane_idx, shell_inner_radius,
                            pec_cell_mask=None):
    """Unit-voltage TEM E on the realized Ex/Ey edges (host float64).

    The potential lives on Yee nodes. A shorted Ex/Ey edge joins two nodes
    of one equipotential conductor; the component containing the pin centre
    is at 1 V and the other conductor nodes are at 0 V. Forward differences
    place -grad(phi) at Ex[i+1/2,j] and Ey[i,j+1/2]. The dual transverse H
    profile has the same indices: Hx=-Ey/eta, Hy=Ex/eta.

    Production callers supply the stamper's conductor cells. Standalone
    callers realize the same cylinders on one cross-section. All of this is
    geometry-only, including when the enclosing material solve is traced.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid

    with jax.ensure_compile_time_eval():
        if pec_cell_mask is None:
            coords = coords_from_uniform_grid(grid)
            center = (float(port.position[0]), float(port.position[1]), 0.0)
            pin = Cylinder(center, float(port.pin_radius), 1.0, axis="z")
            bore = Cylinder(center, float(shell_inner_radius), 1.0, axis="z")
            pin_cells = np.asarray(pin.mask_on_coords(coords.x, coords.y, np.array([0.0])))
            bore_cells = np.asarray(bore.mask_on_coords(coords.x, coords.y, np.array([0.0])))
            cells = pin_cells | ~bore_cells
            edge_plane = 0
        else:
            cells = np.asarray(pec_cell_mask, dtype=bool)
            if cells.shape != grid.shape:
                raise ValueError("coax TEM conductor mask must match the grid shape")
            edge_plane = int(e_plane_idx)
        # Match the runner: realize all four incident cells in 3-D before
        # selecting Ex/Ey at the E source plane (including the previous z cell).
        mx, my, _ = (np.asarray(m)[:, :, edge_plane] for m in
                     realized_pec_edge_masks(cells, periodic=tuple(
                         a in getattr(grid, 'periodic_axes', '') for a in 'xyz')))

    # Mark both endpoints of every shorted transverse edge, using the
    # repository's realized masks rather than a second conductor rule.
    metal = mx | my
    metal[1:, :] |= mx[:-1, :]
    metal[:, 1:] |= my[:, :-1]
    nx, ny = metal.shape
    i0, j0 = grid.position_to_index(port.position)[:2]
    if not metal[i0, j0]:
        raise ValueError("coax TEM cross-section has no realized centre pin")
    pin_nodes = np.zeros_like(metal)
    pin_nodes[i0, j0] = True
    pending = [(int(i0), int(j0))]
    while pending:
        i, j = pending.pop()
        neighbours = []
        if i + 1 < nx and mx[i, j]:
            neighbours.append((i + 1, j))
        if i > 0 and mx[i - 1, j]:
            neighbours.append((i - 1, j))
        if j + 1 < ny and my[i, j]:
            neighbours.append((i, j + 1))
        if j > 0 and my[i, j - 1]:
            neighbours.append((i, j - 1))
        for node in neighbours:
            if not pin_nodes[node]:
                pin_nodes[node] = True
                pending.append(node)
    ground_nodes = metal & ~pin_nodes
    if not ground_nodes.any():
        raise ValueError("coax TEM cross-section has no separate outer conductor")

    # Homogeneous fill: its constant epsilon cancels from Laplace. In
    # particular, eps_scale is a material design variable, not this profile.
    dx = float(grid.dx)
    phi = _solve_laplace_2d(np.ones((nx, ny)), pin_nodes, ground_nodes, dx, dx)
    phi[pin_nodes] = 1.0
    phi[ground_nodes] = 0.0
    ex = np.zeros_like(phi)
    ey = np.zeros_like(phi)
    ex[:-1, :] = -np.diff(phi, axis=0) / dx
    ey[:, :-1] = -np.diff(phi, axis=1) / dx
    return ex, ey


def build_coaxial_tem_plane_source_specs(
    *,
    grid: "Grid",
    port: CoaxialPort,
    n_steps: int,
    field_scale: float = 1.0e4,
    magnetic_ratio: float = 1.0,
    reference_plane_axial_index_offset: int = 0,
    plane_axial_index: int | None = None,
    eps_r: float = PTFE_EPS_R,
    shell_inner_radius: float | None = None,
    pec_cell_mask=None,
) -> CoaxialPlaneSourceSpec:
    """Return TFSF-style transverse E/M source specs for a coaxial port.

    ``shell_inner_radius`` is the inner face of the outer conductor, i.e. the
    outer edge of the PTFE annulus. Pass the value
    :func:`stamp_coaxial_line` returned for this line; omitted, it defaults to
    the declared ``port.outer_radius``, which is where that function puts it.
    ``pec_cell_mask`` is that stamper's realized conductor-cell mask; the
    source solves Laplace on its cross-section and injects the resulting
    discrete TEM edge profile. Without a mask it realizes the declared
    cylinders on the same grid. The fill is homogeneous with relative
    permittivity ``eps_r``; the profile depends only on concrete geometry.

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
      (the realized electrostatic ``Ex, Ey``), and E sources are driven
      by its magnetic dual ``(-Ey, Ex) / eta``. The previous implementation
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
        Multiplier on the ``H`` waveform after the local ``1/eta``
        factor (already baked into ``h_inc`` so a unit ratio means
        Poynting-balanced TEM). Kept for diagnostic ablation; production
        callers should leave it at ``1.0``.
    reference_plane_axial_index_offset:
        Shift of the source plane (and therefore the V/I reference plane)
        relative to ``port.pin_center`` along the port axis. ``0`` injects
        at the pin centre plane.
    plane_axial_index:
        Explicit E-plane node for a calculator's internally indexed source.
        If supplied, bypasses the physical pin-centre lookup; the offset
        above is still applied. Declared ports use the default lookup.
    eps_r:
        Coaxial dielectric permittivity for the analytic ``Z_TEM`` and the
        Yee-half-step delay (``v_phase = c / sqrt(εr)``). Default
        :data:`PTFE_EPS_R` matches the SMA helper.
    pec_cell_mask:
        Optional conductor-cell mask returned by ``stamp_coaxial_line``.
        Its source-plane cross-section sets the Laplace Dirichlet nodes.

    Returns
    -------
    CoaxialPlaneSourceSpec
        Lists of per-cell source specs plus bookkeeping needed by the V/I
        extractor.
    """

    # Lazy imports so the source helpers stay JIT-friendly for callers that
    # import this module without a full simulation pipeline.
    from rfx._internal._uniform_specs import MagneticSourceSpec, SourceSpec
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

    plane_index = (int(grid.position_to_index(pin_center)[2])
                   if plane_axial_index is None else int(plane_axial_index)) + int(
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
    from rfx.sources._sampling import concrete_waveform_eval
    with concrete_waveform_eval(port.excitation, dt_step, delta_t):
        times_e = jnp.arange(int(n_steps), dtype=jnp.float32) * jnp.float32(dt_step)
        times_h = times_e + jnp.float32(delta_t)
        e_inc_table = jax.vmap(port.excitation)(times_e)
        h_samples = jax.vmap(port.excitation)(times_h)
    h_inc_table = h_samples / jnp.float32(z_med)

    # TFSF coefficients (mirror waveguide_port.py:1297, 1338, with ε
    # replaced by εr·ε₀ for the dielectric fill — the source plane sits
    # inside PTFE, so the E update at this cell uses the dielectric cb
    # coefficient. ``waveguide_port.py`` uses bare ``EPS_0`` because
    # waveguides are vacuum-filled by convention; coax in PTFE needs the
    # εr correction so the per-step E-injection matches the physical
    # ``cb = dt/(εr·ε₀·dz)`` of the FDTD update.
    from rfx.core.yee import MaterialArrays, component_h_materials
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0), high_wall=(False, False, False))[0] * MU_0
    coeff_h = jnp.float32(dt_step / (mu_abs * dz))
    coeff_e = jnp.float32(dt_step / (float(eps_r) * EPS_0 * dz))

    # The declared inner wall is only needed for standalone mask realization;
    # production callers provide the stamper's actual conductor cells.
    if shell_inner_radius is None:
        shell_inner_radius = float(port.outer_radius)
    shell_inner_radius = float(shell_inner_radius)
    _require_coaxial_aperture(grid, (0, 1), port.position[:2], shell_inner_radius)

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
    ex_profile, ey_profile = _coaxial_tem_edge_profile(
        grid, port, e_plane_idx, shell_inner_radius, pec_cell_mask,
    )
    z0 = coaxial_built_line_impedance(grid, ex_profile, ey_profile, eps_r)
    source_cell_count = 0
    for i in range(grid.nx):
        for j in range(grid.ny):
            ex_value = ex_profile[i, j]
            ey_value = ey_profile[i, j]
            if ex_value == 0.0 and ey_value == 0.0:
                continue
            source_cell_count += 1

            # H-side correction: Hx += h_sign * coeff_h * Ey,
            #                    Hy -= h_sign * coeff_h * Ex.
            # The dual H edges have exactly these transverse locations.
            if ey_value != 0.0:
                magnetic_sources.append(
                    MagneticSourceSpec(
                        i=i, j=j, k=h_plane_idx, component="hx",
                        waveform=(jnp.float32(ey_value) * h_factor).astype(jnp.float32),
                    )
                )
            if ex_value != 0.0:
                magnetic_sources.append(
                    MagneticSourceSpec(
                        i=i, j=j, k=h_plane_idx, component="hy",
                        waveform=(-jnp.float32(ex_value) * h_factor).astype(jnp.float32),
                    )
                )

            # E-side correction from H = z_hat cross E / eta. The local
            # 1/eta, propagation signs and Yee timing remain in e_factor.
            if ex_value != 0.0:
                electric_sources.append(
                    SourceSpec(
                        i=i, j=j, k=e_plane_idx, component="ex",
                        waveform=(jnp.float32(ex_value) * e_factor).astype(jnp.float32),
                    )
                )
            if ey_value != 0.0:
                electric_sources.append(
                    SourceSpec(
                        i=i, j=j, k=e_plane_idx, component="ey",
                        waveform=(jnp.float32(ey_value) * e_factor).astype(jnp.float32),
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


def stamp_coaxial_line(
    grid: Grid,
    materials,
    *,
    center_xy: tuple[float, float],
    z_lo_index: int,
    z_hi_index: int,
    pin_radius: float = SMA_PIN_RADIUS,
    outer_radius: float = SMA_OUTER_RADIUS,
    eps_r: float = PTFE_EPS_R,
    shell_thickness_m: float = SHELL_THICKNESS_M,
    realized_entities: list | None = None,
    port_id: str = "coaxial_port[0]",
):
    """Stamp a coextensive coax line (z-axis) and return its conductor cells.

    Fills the axial index range ``[z_lo_index, z_hi_index]`` with the dielectric
    annulus (``eps_r``) between the pin and the DECLARED outer radius, and
    reports the pin and the outer wall as a PEC CELL MASK for the caller to
    realize through :func:`rfx.boundaries.pec.realized_pec_edge_masks`.

    Returns ``(materials, shell_inner_radius, pec_cell_mask)``.

    Two things changed here, and both were measured before they were changed
    (``docs/design_notes/coax_conductor_realization.md``):

    * **The conductors are no longer a per-node conductivity.** ``sigma`` is
      applied in ``rfx/core/yee.py`` to the three E components co-indexed with a
      node, so a conductor cell damped only its three plus-side edges and the
      edges entering it from the minus side stayed live on the neighbouring
      dielectric node. That ragged, one-sided wall carried extra inductance: the
      line's fitted phase constant sat 8-18 % above ``omega sqrt(eps)/c``, and
      the open corner contacts of the rasterized ring let the mode couple to the
      region outside it, costing up to 15 % of the column power. Realizing the
      same cells through the lattice ownership contract instead — walls on BOTH
      faces, every normal edge between them shorted — puts the phase constant
      within 1 % and the column power within 0.007 of unity at every cell size
      measured.
    * **The wall's inner radius is the declared outer radius**, and its
      thickness is fixed in metres. It used to be ``outer_radius - min(dz, ...)``,
      so the dielectric annulus and hence the line's own characteristic impedance
      moved with the mesh (40.7 -> 45.3 ohm over a 4-to-9-cell ladder against a
      declared 48.59). The annulus is now ``pin_radius .. outer_radius`` as
      drawn, at every cell size.

    The conductors must NOT extend into a CPML region: running PEC into the PML
    is numerically unstable (verified: max|E| diverges to NaN). Stop the line at
    least ~2 cells short of the absorbing boundary and terminate the feed with
    :func:`stamp_coaxial_annular_resistor`.
    """

    _require_coaxial_aperture(grid, (0, 1), center_xy, outer_radius + shell_thickness_m)
    dz = float(grid.dx)
    z_lo = (int(z_lo_index) - grid.pad_z_lo) * dz
    z_hi = (int(z_hi_index) - grid.pad_z_lo) * dz
    zc = 0.5 * (z_lo + z_hi)
    # Cylinder.mask samples node coordinates with a closed axial predicate.
    # Put each end half a cell beyond the requested endpoint sample: exactly
    # z_lo_index .. z_hi_index, without a floating-point tie on either end.
    height = (z_hi - z_lo) + dz
    center = (float(center_xy[0]), float(center_xy[1]), zc)

    a = float(pin_radius)
    b = float(outer_radius)
    thickness = float(shell_thickness_m)
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError(
            f"shell_thickness_m must be positive and finite, got {shell_thickness_m}"
        )

    # The wall is realized as shorted E edges, so it needs no depth to be a
    # conductor -- one cell is as perfect as a hundred. What its thickness must
    # not do is reach an absorber: PEC inside a CPML pad is unstable and the
    # field diverges to NaN. Neither preflight check can catch that here,
    # because both iterate the REGISTERED geometry and this wall is stamped,
    # never registered. So the thickness is clamped against the room the board
    # actually has, and refused outright when there is not a cell of it.
    #
    # Only faces that HAVE a pad constrain the wall; a PEC or periodic face has
    # no absorber to run into. Physical coordinates put node i of an axis at
    # ``(i - pad_lo) * dx``, so the pad's inner edge is 0 on the low side and
    # ``(n - 1 - pad_lo - pad_hi) * dx`` on the high side.
    cx, cy = float(center_xy[0]), float(center_xy[1])
    nx, ny = int(grid.shape[0]), int(grid.shape[1])
    pad_x_lo, pad_x_hi = int(grid.pad_x_lo), int(grid.pad_x_hi)
    pad_y_lo, pad_y_hi = int(grid.pad_y_lo), int(grid.pad_y_hi)
    span_x = (nx - 1 - pad_x_lo - pad_x_hi) * dz
    span_y = (ny - 1 - pad_y_lo - pad_y_hi) * dz
    room = []
    if pad_x_lo:
        room.append(("x-lo", pad_x_lo, cx))
    if pad_x_hi:
        room.append(("x-hi", pad_x_hi, span_x - cx))
    if pad_y_lo:
        room.append(("y-lo", pad_y_lo, cy))
    if pad_y_hi:
        room.append(("y-hi", pad_y_hi, span_y - cy))
    if room:
        pad_name, pad_cells, clearance = min(room, key=lambda t: t[2])
        allowed = clearance - b - dz          # keep one cell clear of the pad
        if allowed < dz:
            raise ValueError(
                f"the coaxial line at ({cx * 1e3:.4g}, {cy * 1e3:.4g}) mm with "
                f"outer radius {b * 1e3:.4g} mm has no room for a wall before "
                f"the {pad_name} absorber: its centre is {clearance * 1e3:.4g} mm "
                f"from that pad's inner edge ({pad_cells} cells deep), which "
                f"leaves {allowed / dz:.2f} cells outside the declared radius "
                f"once one cell of clearance is kept. A conductor inside a CPML "
                f"pad diverges. Move the line further from that face, shrink "
                f"outer_radius, or enlarge the board."
            )
        thickness = min(thickness, allowed)

    shell_inner_radius = b
    shell_outer_radius = b + thickness

    outer_mask = Cylinder(center=center, radius=shell_outer_radius, height=height,
                          axis="z").mask(grid)
    shell_inner_mask = Cylinder(center=center, radius=shell_inner_radius, height=height,
                                axis="z").mask(grid)
    pin_mask = Cylinder(center=center, radius=a, height=height, axis="z").mask(grid)

    shell = outer_mask & ~shell_inner_mask
    if not shell.any():
        raise ValueError(
            f"stamp_coaxial_line: a {thickness * 1e3:.4f} mm wall outside "
            f"r = {b * 1e3:.4f} mm rasterizes to no cell at dx = {dz * 1e6:.2f} um. "
            "The mesh is too coarse for this line; refine it or widen the wall."
        )

    # The clamp above keeps a cell of clearance in continuous arithmetic. The
    # conductor is realized as shorted edges, and a cell i shorts edges on
    # nodes i AND i + 1, so on the high side of an axis the realized wall
    # reaches one node past its last cell and the clamp's cell of clearance
    # can realize as none: a conductor on an absorber's inner face. The coax
    # lanes absorb on x and y, so the realized clearance is
    # checked, in whole cells, against every padded lateral face.
    conductor = np.asarray(shell | pin_mask)
    for axis, name, n_axis, pad_lo_, pad_hi_ in (
        (0, "x", nx, pad_x_lo, pad_x_hi), (1, "y", ny, pad_y_lo, pad_y_hi),
    ):
        cells = np.nonzero(conductor.any(axis=tuple(t for t in range(3) if t != axis)))[0]
        gaps = []
        if pad_lo_:
            gaps.append((f"{name}-lo", int(cells.min()) - pad_lo_))
        if pad_hi_:
            gaps.append((f"{name}-hi", (n_axis - 1 - pad_hi_) - (int(cells.max()) + 1)))
        for face, gap in gaps:
            if gap < 1:
                raise ValueError(
                    f"the coaxial line at ({cx * 1e3:.4g}, {cy * 1e3:.4g}) mm with "
                    f"outer radius {b * 1e3:.4g} mm realizes its outer wall "
                    f"{gap} cell(s) from the {face} absorber's inner face; at "
                    f"least one whole cell of vacuum is required, because a "
                    f"conductor inside or on a CPML pad diverges. Move the line "
                    f"further from that face, shrink outer_radius, or enlarge "
                    f"the board."
                )

    eps = np.array(materials.eps_r)
    sig = np.array(materials.sigma)
    # Only the dielectric is written into the material arrays. The conductors
    # are returned as a mask and realized as PEC edges by the caller, so nothing
    # here sets a conductivity.
    dielectric = shell_inner_mask & ~pin_mask
    eps = np.where(dielectric, float(eps_r), eps)
    sig = np.where(dielectric, 0.0, sig)
    materials = materials._replace(eps_r=jnp.asarray(eps), sigma=jnp.asarray(sig))
    if realized_entities is not None:
        from rfx.model.coax import coax_stamp_entities
        realized_entities.extend(coax_stamp_entities(
            grid, center=center, height=height, a=a, b=b, outer_radius=shell_outer_radius,
            pin=pin_mask, shell=shell, bore=shell_inner_mask, outer=outer_mask,
            dielectric=dielectric, port_id=port_id))
    return materials, shell_inner_radius, np.asarray(shell | pin_mask)


def stamp_coaxial_short_plane(
    grid: Grid,
    materials,
    *,
    center_xy: tuple[float, float],
    z_index: int,
    outer_radius: float = SMA_OUTER_RADIUS,
):
    """The calibration short (``Gamma = -1``): a PEC disk at one z-plane.

    Returns ``(materials, pec_cell_mask)``. Like :func:`stamp_coaxial_line` the
    disk is reported as cells rather than written as a conductivity, so the
    caller realizes it through the same lattice ownership contract as the rest
    of the conductor; ``materials`` comes back with the disk's permittivity set
    to vacuum and is otherwise unchanged.
    """

    _require_coaxial_aperture(grid, (0, 1), center_xy, outer_radius)
    eps = np.array(materials.eps_r)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    z = int(z_index)
    mask = np.zeros(tuple(int(n) for n in (grid.nx, grid.ny, grid.nz)), dtype=bool)
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            if np.hypot(x - cx, y - cy) <= float(outer_radius):
                mask[i, j, z] = True
                eps[i, j, z] = 1.0
    if not mask.any():
        raise ValueError(
            "stamp_coaxial_short_plane stamped 0 cells; check geometry/resolution")
    return materials._replace(eps_r=jnp.asarray(eps)), mask


def stamp_coaxial_annular_resistor(
    grid: Grid,
    materials,
    *,
    center_xy: tuple[float, float],
    z_index: int,
    pin_radius: float = SMA_PIN_RADIUS,
    outer_radius: float = SMA_OUTER_RADIUS,
    target_impedance: float,
    shell_inner_radius: float | None = None,
    pec_cell_mask=None,
):
    """Stamp a radial annular resistor (``Γ → 0`` match) at one z-plane.

    Each Ex and Ey edge in the annulus at this node plane receives
    ``σ = log(b'/a) / (2π·dz·Z)`` so the radial pin-to-shell resistance matches
    ``target_impedance``. The edge-owned load is not averaged into adjacent
    planes or Ez. Used both for the matched feed termination and a matched DUT.

    ``pec_cell_mask`` is the conductor mask :func:`stamp_coaxial_line` returns;
    cells in it are skipped. It replaces the old test ``sigma >= PEC_SIGMA/2``,
    which stopped finding the conductor when the conductor stopped being a
    conductivity. Passing nothing falls back to that test, which is correct only
    for a caller that still stamps its conductors into ``sigma``.

    Returns updated materials.
    """

    _require_coaxial_aperture(grid, (0, 1), center_xy,
                             outer_radius if shell_inner_radius is None else shell_inner_radius)
    if not np.isfinite(target_impedance) or target_impedance <= 0.0:
        raise ValueError(f"target_impedance must be positive finite, got {target_impedance}")
    dz = float(grid.dx)
    if shell_inner_radius is None:
        shell_inner_radius = float(outer_radius)
    pec = None if pec_cell_mask is None else np.asarray(pec_cell_mask)
    sigma_load = float(np.log(shell_inner_radius / float(pin_radius))) / (
        2.0 * np.pi * dz * float(target_impedance)
    )
    eps = np.array(materials.eps_r)
    sig = np.array(materials.sigma)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    z = int(z_index)
    stamped = 0
    for i in range(grid.nx):
        x = (i - grid.pad_x_lo) * grid.dx
        for j in range(grid.ny):
            y = (j - grid.pad_y_lo) * grid.dx
            r = float(np.hypot(x - cx, y - cy))
            occupied = (bool(pec[i, j, z]) if pec is not None
                        else sig[i, j, z] >= 0.5 * PEC_SIGMA)
            if float(pin_radius) <= r <= shell_inner_radius and not occupied:
                materials = stamp_lumped_sigma(materials, (i, j, z), sigma_load, "ex")
                materials = stamp_lumped_sigma(materials, (i, j, z), sigma_load, "ey")
                eps[i, j, z] = 1.0
                stamped += 1
    if stamped == 0:
        raise ValueError("stamp_coaxial_annular_resistor stamped 0 cells; check geometry/resolution")
    return materials._replace(eps_r=jnp.asarray(eps))
