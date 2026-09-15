"""Differentiable rectangular spiral inductor on the 3-D Yee FDFD pipeline:
body-fitted (r_out, spacing, width) parameterisation, lumped ports on
vertical lead columns, open/short de-embedding, L and Q.

What is built (``build_spiral``, host numpy, once)
--------------------------------------------------
1. Geometry at the NOMINAL parameters ``theta = (r_out, spacing, width)``:
   :func:`rfx.fdfd.gds.rect_spiral` polygons (strip on the top metal, a
   straight underpass on the lower metal, the via square), x/y grid lines
   from :func:`rfx.fdfd.gds.mesh_lines` (every axis-aligned polygon edge is
   a grid line) and z lines from :func:`rfx.fdfd.gds.z_lines` on a small
   stack: PEC ground at z = 0 (the outer wall of the Yee box), a silicon
   slab, oxide with the two metal layers and the via, air above, PEC lid.
   The polygons are rasterised into static conductor cell masks and those
   into PEC edge masks (:func:`rfx.fdfd.yee3d.pec_edges_from_cells`). This
   fixes the TOPOLOGY: cell counts, masks and sparsity patterns never change.
2. Two lumped ports (:class:`rfx.fdfd.ports3d.LumpedElement`) between the
   spiral terminals and the ground: each terminal gets a vertical lead
   column of PEC cells (footprint one strip width square at the terminal
   centre) from its metal layer down to ``port_gap_cells`` cells above the
   ground; the port is the Ez-edge box of the remaining gap cells under the
   column footprint. Three fixtures share the grid: DUT (spiral + columns),
   OPEN (columns only) and SHORT (columns + a PEC bar on each terminal's
   metal level from the lead's DUT-side end to a common PEC post between
   the columns that runs from the ground up through both metal levels).
   See "the short standard" below for why it is built that way.

What is traced (``solve_spiral``, jax.numpy, differentiable)
------------------------------------------------------------
Only the METRIC moves. :func:`rect_spiral_edge_coordinates` mirrors the
polygon construction analytically and returns the x and y coordinates of
every axis-aligned polygon edge as a function of ``theta``; these are the
breakpoints. Every nominal grid line is then moved by piecewise-linear
interpolation between the breakpoints it lies between (the outer walls are
fixed breakpoints), i.e. ``x(theta) = interp(x_nominal, bps_nominal,
bps(theta))`` -- the body-fitted stretch of :mod:`rfx.fdfd.hplane` in 3-D.
It is written in delta form ``x_nom + d_lo + frac (d_hi - d_lo)`` with
``d = bps(theta) - bps_nominal`` so the nominal grid is reproduced exactly
(to the bit). Steps ``dx = diff(x)``, ``dy = diff(y)`` feed
:func:`rfx.fdfd.ports3d.s_matrix` (one LU per fixture), the three
S-matrices go through :func:`rfx.fdfd.deembed.open_short_deembed` and
:func:`rfx.fdfd.deembed.inductor_metrics`. Losses are optional: a
Leontovich surface impedance on every metal cell of the fixture
(:mod:`rfx.fdfd.conductor`, traced ``sigma_metal``) and a bulk silicon
conductivity folded into the permittivity of the silicon cells as
``-j sigma_si / (omega eps0)``. Everything downstream of the build is
differentiable in ``theta``, ``sigma_metal``, ``sigma_si`` and ``freq``.

Topological constraint (the feasibility check)
----------------------------------------------
The breakpoints must keep their nominal ORDER: a strip edge may not cross
another strip edge, the port line or a wall. With ``pitch = width +
spacing`` and apothems ``c_j = r_out - width/2 - j pitch`` this means
``spacing > 0`` (adjacent turns keep a gap), ``r_out`` inside the box,
``-c_0 - width/2 - lead`` (the port line) above the lower wall, plus the
generator's own validity rules (``a_in > width/2``, last side >= width).
:func:`breakpoint_gaps` returns the signed gaps between consecutive
breakpoints (all must be positive); :func:`check_feasible` raises on a
violation. Inside the feasible region every derivative is smooth; the
masks never change so there is no staircase jump anywhere.

Inductor conventions (2-port with both terminals to ground)
-----------------------------------------------------------
``L_diff = Im(Z11 - Z12 - Z21 + Z22) / omega`` of the de-embedded Z is the
SERIES inductance between the two terminals (the differential drive; the
shunt parasitics enter only as the series combination of the two terminal
capacitances). ``L_se`` is reported from ``1/Y11`` (port 2 GROUNDED,
:func:`rfx.fdfd.deembed.l_from_y11`): the ``Z11`` (port 2 OPEN) definition
is capacitive for a floating inductor -- with port 2 open the only path to
ground is the terminal capacitance -- and is exposed separately as
``L_z11_open`` for completeness. ``L_raw`` is ``L_diff`` of the raw DUT
S-matrix (leads and ports included).

The short standard
------------------
Open/short de-embedding (Koolen) needs ``Y_short - Y_open`` to be the
admittance of the two lead impedances, i.e. each lead shorted to ground at
the DUT reference plane. Joining the two leads by a bar alone is a THRU,
whose ``Y_short - Y_open`` is singular; the post to ground makes each lead
a short. The bars sit on each terminal's own metal level and cover the
cell where the DUT's current leaves the lead (the column top, or the last
stub cell), so the short's current traverses exactly the lead the DUT's
current does (a bar exiting the outer column at the lower metal level
would miss the column's upper segment and any stub: measured as a 2 %
error on the de-embedded L when a one-cell stub was added). Because both
shorts share the post, its residual impedance appears in every entry of
``Z_short'`` and cancels exactly in the differential combination
``Z11 - Z12 - Z21 + Z22``: the only residual on ``L_diff`` is the partial
inductance of the two bar halves (one cell of strip each). ``L_se`` from
``1/Y11`` does carry the post's residual, which is why ``L_diff`` is the
primary metric.

Leontovich validity (a frequency FLOOR, not a ceiling)
------------------------------------------------------
The sheet model replaces each metal cell by a surface impedance and is
valid only when the skin depth ``delta = sqrt(2 / (omega mu0 sigma))`` is
well below the metal thickness (``t_m1``, ``t_m2``, ``t_via``: 2 um in the
default stack). Below that the sheet over-estimates the internal
inductance and the resistance: copper at 100 MHz has ``delta = 6.6 um >
2 um`` and gives ``L_diff`` 42 % ABOVE the PEC value and ``Q_diff = 3.5``
for a 2 um strip (measured; unphysical). With ``sigma = 1e14`` S/m the
sheet reproduces the PEC ``L_diff`` to 3.5e-4, so the model itself is
consistent; it is the ``delta << t`` condition that is violated. Copper on
2 um reaches ``delta = t`` at 1.1 GHz and ``delta = t / 2`` at 4.4 GHz;
the 2.4 GHz used in the tests (``delta / t = 0.67``) is marginal.
:func:`leontovich_validity` returns ``delta / t_min`` for a model, and
``solve_spiral`` does not check it (``sigma_metal`` and ``freq`` may be
traced): the CALLER must keep it well below 1. Nothing here resolves the
current distribution inside the metal.

The straight-bar DUT (``dut_kind = "bar"``)
-------------------------------------------
``SpiralSpec(dut_kind="bar", bar_length=l)`` swaps the spiral for a single
straight strip of length ``l`` (centre to centre of the two lead columns)
and width ``width`` on the TOP metal, spanning ``x in [-(l+W)/2, (l+W)/2]``,
``y in [-W/2, W/2]``, with the two column footprints the ``W x W`` squares at
its ends and both ports therefore on M2. Nothing else changes: the same
build, the same three fixtures (the short's bars now both sit on M2), the
same de-embedding, the same traced metric, the same ``solve_spiral``.
``theta = (bar_length, width)`` -- a TWO-tuple, not the spiral's
three-tuple with an ignored spacing. The Greenhouse referee for it is the
self partial inductance of ONE ``l_eff x width x t_m2`` bar plus its
PEC-ground image; ``l_eff`` is a reference-plane choice and is NOT simply
``bar_length``. For the spiral the short standard's bar leaves the lead
column perpendicular to the DUT current, so the plane is the column
footprint CENTRE (+-W/2, measured as +-1.46 % there); for the BAR that same
bar is COLLINEAR with the DUT current and the ground post starts one cell
inside the column (``post_i0 = inner.i1 + 1``, an isolating cell the port
edges require), so the short removes ``W/2 + base_dx`` of strip at each
end. ``validation/fdfd/straight_bar_convergence.py`` reports the gap
against ``l``, ``l - W`` and ``l - W - 2 base_dx`` and uses the
length DIFFERENTIAL, in which the plane cancels, as its primary gate.

The right-angle bend DUT (``dut_kind = "bend"``)
-----------------------------------------------
``SpiralSpec(dut_kind="bend", bend_l1=l1, bend_l2=l2)`` swaps the spiral for
ONE L-shaped strip with a single right angle on the TOP metal: a leg of
length ``l1`` in ``+y`` (the direction in which the DUT current leaves the
OUTER lead column, i.e. the spiral's lead direction), the corner, then a
leg of length ``l2`` in ``-x``. ``l1`` and ``l2`` are centre-to-centre
lengths -- outer footprint centre ``(l2/2, -l1/2)``, corner
``(l2/2, l1/2)``, inner footprint centre ``(-l2/2, l1/2)`` -- so the
conductor between the two reference planes is exactly ``l1 + l2`` long and
the drawn (mitred) strip is ``l1 + l2 + W``, overhanging each footprint
centre by ``W/2`` as the bar does. ``theta = (l1, l2, width)``; the
breakpoints come from :func:`bend_edge_coordinates` (``+-(l2 +- W)/2`` on x,
``+-(l1 +- W)/2`` on y) and the geometry from :func:`bend_geometry`. Both
terminals are on M2; ``n_turns``, ``r_out``, ``spacing``, ``lead`` and
``bar_length`` are unused.

The two lead columns differ in BOTH x and y -- unlike the bar's (collinear)
and the spiral's (one shared port line) -- and that is the only thing the
build needed generalising for: each column's terminal row is taken from its
OWN port, the short standard's bar runs on each column's own row, and the
grounded post spans the UNION of the two rows. For the spiral and the bar
the two rows coincide and the result is bit-identical to the single-row
construction. The short standard is therefore, in cell indices with
``outer`` the ``+x`` column:

* post: ``i in [inner.i1 + 1, outer.i0 - 1)`` (one isolating cell on each
  side, which the port's own Ez edges require), ``j`` over the union of the
  two terminal rows, ``z`` from the ground up through the metal levels;
* outer arm: ``i in [post_i0, outer.i1)`` on the outer row; inner arm:
  ``i in [inner.i0, post_i1)`` on the inner row. Both overlap the post's x
  range on their own row, which is what connects them.

Driven differentially the post carries no current, so the de-embedded
quantity is ``L(strip between the planes) - L(the two arms)`` and each arm
is ``W/2 + one cell`` long, x-directed. That is perpendicular to the DUT
current at the outer column (as for the spiral) and collinear with it at
the inner column (as for the bar).
``validation/fdfd/corner_convergence.py`` is the study this was added for:
it compares the bend against the straight bar of the same total length
``l1 + l2`` to isolate ONE right-angle corner.

The level-invariant fixture (``wall_margin_m``, ``lid_height_m``,
``short_gap_m``, ``port_gap_m``, ``refine``, ``z_refine``, ``pad_refine``)
------------------------------------------------------------------------
Three things in the fixture above are defined in CELLS, so the physical
problem changes with ``base_dx``: the graded padding (``pad_lines``) grows
``pad_cells`` cells from the last meshed cell, so the side walls and the lid
move IN as the grid refines; the short standard's post spans
``[inner.i1 + 1, outer.i0 - 1)``, one CELL from each column, so the post and
the bridge arms change width with the grid; and ``port_gap_cells`` counts z
cells. Every option below replaces one of those by a physical coordinate;
each is independent (so the old fixture can be moved one artefact at a time)
and every default is the original build, statement for statement:

* ``wall_margin_m``: the four side walls at the DUT bounding box -+ this
  distance, reached from the meshed box by :func:`pad_to` -- a geometric
  grading of ratio <= ``pad_ratio`` whose LAST line is pinned to the wall;
  ``lid_height_m``: the lid at this z, the same way. They replace
  ``pad_cells`` (for the walls) and ``pad_cells_z`` (for the lid).
* ``short_gap_m``: the post's x edges at the inner column's ``+x`` edge plus
  the gap and the outer column's ``-x`` edge minus the gap, both made
  mandatory grid lines (with the column footprints), so the post, the arms
  (post to each column's far edge) and the reference planes are the same
  physical objects at every resolution.
* ``port_gap_m``: the lead columns end at this height, a mandatory z line;
  the port spans however many z cells lie below it.
* ``refine = m``: NESTED refinement. The level-1 lines (all the mandatory
  lines above, polygon edges and stack interfaces included, graded to
  ``base_dx`` / ``base_dz`` / ``metal_cells`` as usual) are built first and
  every interval of the meshed box is then cut into ``m`` equal cells in x
  and y and ``z_refine or m`` in z (joint refinement; the metal slabs are
  subdivided too), so every level-1 line is a line of every level. The
  padding is refined by :func:`_refine_pad`: ``pad_refine="uniform"``
  subdivides every padding interval ``m`` times as well (the whole mesh is
  nested); ``"graded"`` keeps the level-1 padding lines and only grades the
  transition (cheaper when the level-1 padding grows well below
  ``pad_ratio``; with the steepest allowed grading it cascades through the
  whole padding and saves nothing).

The traced metric is unchanged: breakpoints are still the polygon edges and
every other line (walls, post, subdivisions) moves by the same piecewise
linear interpolation between them, so ``theta`` derivatives work at every
level. ``validation/fdfd/invariant_ladder.py`` is the study these options
were added for; its gate P1 reads every fixture coordinate back from the
built cell masks at every level.

Scope fence. Closed PEC box (no PML; fine below the first box resonance);
staircase PEC or Leontovich metal, no finite-thickness skin effect (see
above for the frequency floor that implies); grid resolution is a cost
choice (``base_dx``), the accuracy of L at a given resolution is NOT
validated here -- what is validated is the discrete model's
self-consistency (reciprocity, passivity, de-embedding invariance) and its
derivatives against finite differences. Sizes are CPU-SuperLU sizes (see
the tests for the measured factorisation times).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rfx.fdfd import conductor as cd
from rfx.fdfd import deembed as de
from rfx.fdfd import gds
from rfx.fdfd import ports3d as p3
from rfx.fdfd import yee3d as y

__all__ = [
    "SmallStack", "SpiralSpec", "SpiralModel", "SpiralResult", "LineMap",
    "rect_spiral_edge_coordinates", "straight_bar_edge_coordinates", "straight_bar_geometry",
    "bend_edge_coordinates", "bend_geometry",
    "build_spiral", "pad_lines", "spiral_lines", "breakpoint_gaps",
    "check_feasible", "solve_spiral", "nominal_record", "skin_depth", "leontovich_validity",
    "KEY_M2", "KEY_M1", "KEY_VIA", "DUT_KINDS",
    "pad_to", "subdivide_lines", "PAD_REFINE_MODES",
]

DUT_KINDS = ("spiral", "bar", "bend")

KEY_M2 = (134, 0)      # top metal: the spiral strip (rect_spiral default ``layer``)
KEY_M1 = (126, 0)      # lower metal: the underpass
KEY_VIA = (133, 0)     # via square joining them


# ----------------------------------------------------------------------------
# specification

@dataclass(frozen=True)
class SmallStack:
    """Small vertical stack (metres, bottom up): PEC ground at z = 0, silicon
    ``t_si``, oxide ``t_ox_low``, metal M1 ``t_m1``, via ``t_via``, metal M2
    ``t_m2``, oxide ``t_ox_high``, air ``t_air``, PEC lid. ``base_dz`` is
    the largest z cell, ``metal_cells`` the minimum number of cells across
    each metal / via slab. The silicon's conductivity is NOT part of the
    stack: it is the traced ``sigma_si`` of :func:`solve_spiral`."""
    t_si: float = 30e-6
    t_ox_low: float = 1e-6
    t_m1: float = 2e-6
    t_via: float = 2e-6
    t_m2: float = 2e-6
    t_ox_high: float = 3e-6
    t_air: float = 20e-6
    eps_si: float = 11.9
    eps_ox: float = 4.1
    base_dz: float = 10e-6
    metal_cells: int = 2

    @property
    def z_m1(self) -> float:
        return self.t_si + self.t_ox_low

    @property
    def z_via(self) -> float:
        return self.z_m1 + self.t_m1

    @property
    def z_m2(self) -> float:
        return self.z_via + self.t_via

    @property
    def z_ox_top(self) -> float:
        return self.z_m2 + self.t_m2 + self.t_ox_high

    @property
    def z_top(self) -> float:
        return self.z_ox_top + self.t_air

    def layer_stack(self) -> gds.LayerStack:
        return gds.LayerStack((
            gds.Layer("silicon", "dielectric", 0.0, self.t_si, eps_r=self.eps_si),
            gds.Layer("oxide", "dielectric", self.t_si, self.z_ox_top - self.t_si, eps_r=self.eps_ox),
            gds.Layer("M1", "conductor", self.z_m1, self.t_m1, gds=KEY_M1, sigma=5.8e7),
            gds.Layer("VIA", "via", self.z_via, self.t_via, gds=KEY_VIA, sigma=5.8e7),
            gds.Layer("M2", "conductor", self.z_m2, self.t_m2, gds=KEY_M2, sigma=5.8e7),
        ), name="spiral-small")


@dataclass(frozen=True)
class SpiralSpec:
    """Nominal spiral and grid. ``theta = (r_out, spacing, width)`` are the
    continuous parameters; ``n_turns`` (integer), ``lead`` (the straight
    outer lead, >= ``width`` recommended so the outer lead column sits on
    the lead and not on the first side) and the stack are static.
    ``base_dx`` defaults to ``width`` -- ONE cell across the strip -- which
    with ``spacing = width`` keeps the grid uniform inside the spiral and
    the unknown count near 1e4 (12739 for the defaults); ``width / 2`` is
    the design resolution and costs 3.7x the unknowns (47742) and ~13x the
    SuperLU/COLAMD factorisation time (17 s vs 1.3 s per fixture, measured).
    ``margin`` is the distance from the outermost polygon edge to the PEC
    side walls; ``port_gap_cells`` the number of z cells between the ground
    and the bottom of the lead columns (the port spans them);
    ``lead_stub_cells`` moves each column that many cells outward (-y,
    into the margin below the port line) and joins it to the terminal by a
    PEC stub on the terminal's metal layer -- a longer lead with the spiral
    and the ground plane untouched. This is NOT a valid de-embedding
    configuration and ``build_spiral`` warns when it is used: the stub is
    collinear with the lead strip / underpass and couples to the DUT by
    mutual inductance, which open/short de-embedding cannot represent, and
    it OVER-corrects -- with one 10 um cell the de-embedded L_diff moves
    by +2.3 % while the raw L moves by only +0.76 % (+6.2 % de-embedded
    for two cells; measured at 100 MHz on the defaults). It is kept only
    to reproduce that measurement.

    ``pml_cells`` / ``pml_order`` / ``pml_kappa_max`` / ``pml_r0`` put a PML
    on the four side walls and the lid (never on the ground, see
    :attr:`pml`); ``pad_cells`` / ``pad_cells_z`` / ``pad_ratio`` instead
    append geometrically growing cells outside the meshed box on those same
    five faces (:func:`pad_lines`), which moves the PEC walls far away for a
    few cells each (``pad_cells_z = 0`` means "same as ``pad_cells``").
    Both default to OFF (the closed PEC box of the original model). They
    were added for the wall-independence protocol of
    ``validation/fdfd/spiral_convergence.py``; that study measured the PML to
    be useless at 100 MHz (the quasi-static stretch is dominated by
    ``sigma_w / (j omega eps0)``, ``pml_kappa_max`` has no effect at all and
    ``L_dut`` moves 5-15 % non-monotonically with the PML depth) and the
    padding to converge monotonically (+0.06 % from 4 to 6 cells at
    ``base_dx = W``).

    ``dut_kind`` selects WHAT sits between the two lead columns. ``"spiral"``
    (the default, unchanged) is the ``rect_spiral`` strip + underpass + via.
    ``"bar"`` replaces it by ONE straight strip of length ``bar_length`` on
    the top metal (no underpass, no via, both terminals on M2), everything
    else -- grid, graded padding, lead columns, ports, open/short fixtures,
    de-embedding, the traced body-fitted metric -- identical; ``n_turns``,
    ``r_out``, ``spacing`` and ``lead`` are then unused. Its continuous
    parameters are ``theta = (bar_length, width)`` (see :attr:`theta`) and
    its breakpoints come from :func:`straight_bar_edge_coordinates`. It was
    added for ``validation/fdfd/straight_bar_convergence.py``, which
    decomposes the spiral study's residual gap: a bar has no corners, no
    underpass and no via, so it isolates the discretisation of the strip
    cross-section and the de-embedding residual from the spiral-specific
    geometry.

    ``wall_margin_m`` / ``lid_height_m`` / ``short_gap_m`` / ``port_gap_m``
    / ``refine`` / ``z_refine`` / ``pad_refine`` build the LEVEL-INVARIANT
    fixture (module doc, "The level-invariant fixture"): walls, lid, the
    short standard's post and the port gap at physical coordinates, and
    nested joint refinement of a level-1 mesh. All default to the original
    cell-defined build (``invariant`` is then False and ``build_spiral``
    runs the original statements)."""
    n_turns: int = 2
    r_out: float = 60e-6
    spacing: float = 10e-6
    width: float = 10e-6
    lead: float = 20e-6
    base_dx: float | None = None
    margin: float = 20e-6
    stack: SmallStack = field(default_factory=SmallStack)
    port_gap_cells: int = 1
    lead_stub_cells: int = 0
    z0: float = 50.0
    pml_cells: int = 0
    pml_order: int = 3
    pml_kappa_max: float = 1.0
    pml_r0: float = 1e-8
    pad_cells: int = 0
    pad_cells_z: int = 0
    pad_ratio: float = 1.5
    dut_kind: str = "spiral"
    bar_length: float = 0.0
    bend_l1: float = 0.0
    bend_l2: float = 0.0
    # -- the level-invariant fixture (``validation/fdfd/invariant_ladder.py``);
    #    every default below is the original cell-defined fixture, unchanged
    wall_margin_m: float | None = None
    lid_height_m: float | None = None
    short_gap_m: float | None = None
    port_gap_m: float | None = None
    refine: int = 1
    z_refine: int = 0
    pad_refine: str = "uniform"

    @property
    def invariant(self) -> bool:
        """True when any option of the level-invariant fixture is set (see
        "The level-invariant fixture" in the module doc); False is the
        original cell-defined build, statement for statement."""
        return (self.wall_margin_m is not None or self.lid_height_m is not None
                or self.short_gap_m is not None or self.port_gap_m is not None
                or int(self.refine) != 1 or int(self.z_refine) != 0)

    @property
    def pml(self) -> tuple[int, int, int, int, int, int]:
        """PML cell counts for :class:`rfx.fdfd.yee3d.Yee3DSpec`:
        ``pml_cells`` on the four side walls and the lid, ZERO on ``z_lo``
        -- the ground plane is the physical PEC of the fixture and must stay
        a bare wall. ``pml_cells = 0`` (the default) is the closed PEC box
        of the original model."""
        p = int(self.pml_cells)
        if p < 0:
            raise ValueError("pml_cells must be >= 0")
        return (p, p, p, p, 0, p)

    @property
    def theta(self) -> tuple[float, ...]:
        """Continuous parameters of the DUT: ``(r_out, spacing, width)`` for
        ``dut_kind="spiral"``, ``(bar_length, width)`` for
        ``dut_kind="bar"`` (the straight-bar DUT has no spacing; the tuple
        is two long, NOT a three-tuple with an ignored entry, so that
        ``jax.grad`` returns exactly the two derivatives that exist) and
        ``(bend_l1, bend_l2, width)`` for ``dut_kind="bend"``."""
        if self.dut_kind == "bar":
            return (float(self.bar_length), float(self.width))
        if self.dut_kind == "bend":
            return (float(self.bend_l1), float(self.bend_l2), float(self.width))
        return (float(self.r_out), float(self.spacing), float(self.width))

    @property
    def dx(self) -> float:
        return float(self.width if self.base_dx is None else self.base_dx)


# ----------------------------------------------------------------------------
# analytic edge coordinates (the breakpoints)

def _edge_coordinates_unsorted(theta, n_turns: int, lead: float):
    """Unsorted breakpoints ``(x (4 n + 2,), y (4 n + 2,))`` in a FIXED
    analytic order (used by the line map); see
    :func:`rect_spiral_edge_coordinates` for the sorted public version."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    r_out, spacing, width = theta[0], theta[1], theta[2]
    hw = 0.5 * width
    pitch = width + spacing
    a0 = r_out - hw
    a_in = a0 - pitch * n_turns
    c = a0 - pitch * jnp.arange(n_turns, dtype=jnp.float64)     # apothems of turns 0..n-1
    x = jnp.concatenate([c + hw, c - hw, -c + hw, -c - hw,
                         jnp.stack([a_in + hw, a_in - hw])])
    y_port = -a0 - lead
    ext = width + hw * jnp.tan(0.5 * (2.0 * jnp.pi / 4.0))
    end_y = -c[-1] + ext
    yv = jnp.concatenate([c + hw, c - hw, -c + hw, -c - hw, jnp.stack([y_port, end_y])])
    return x, yv


def rect_spiral_edge_coordinates(theta, n_turns: int, lead: float = 0.0):
    """Sorted x and y coordinates of every axis-aligned edge of the polygons
    :func:`rfx.fdfd.gds.rect_spiral` emits for ``theta = (r_out, spacing,
    width)``, as ``jax.numpy`` functions of ``theta`` (mirrors the generator:
    the strip edges sit at ``+-c_j +- width/2`` for the apothems ``c_j =
    r_out - width/2 - j (width + spacing)``, the inner extension / underpass
    / via at ``a_in +- width/2`` with ``a_in = c_0 - n_turns pitch``; the y
    lines add the port line ``-c_0 - lead`` and the extension's end
    ``-c_{n-1} + 1.5 width``). The via's lower edge coincides analytically
    with the last side's upper strip edge (``-c_{n-1} + width/2``) and is
    not listed twice. Returns ``(x (4 n + 2,), y (4 n + 2,))``."""
    x, yv = _edge_coordinates_unsorted(theta, int(n_turns), float(lead))
    return jnp.sort(x), jnp.sort(yv)


# ----------------------------------------------------------------------------
# straight-bar DUT (``dut_kind = "bar"``): the SAME fixture with the spiral
# replaced by one straight strip between the two lead columns

def _bar_edge_coordinates_unsorted(theta):
    """Unsorted breakpoints ``(x (4,), y (2,))`` of the straight bar in a
    FIXED analytic order; ``theta = (bar_length, width)``."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    l_bar, width = theta[0], theta[1]
    hw, hl = 0.5 * width, 0.5 * l_bar
    x = jnp.stack([-(hl + hw), -(hl - hw), hl - hw, hl + hw])
    yv = jnp.stack([-hw, hw])
    return x, yv


def straight_bar_edge_coordinates(theta):
    """Sorted x and y coordinates of every axis-aligned polygon edge of the
    straight-bar DUT for ``theta = (bar_length, width)``, as ``jax.numpy``
    functions of ``theta``.

    The bar is one rectangle on the TOP metal (``KEY_M2``) centred on the
    origin: ``x`` in ``[-(l + W)/2, (l + W)/2]``, ``y`` in ``[-W/2, W/2]``,
    with the two lead-column footprints the ``W x W`` squares at its two
    ends (``|x| in [(l - W)/2, (l + W)/2]``). Their inner edges
    ``+-(l - W)/2`` are breakpoints too, which is what makes the column
    footprint grid-exact at every ``base_dx`` and puts the two port centres
    exactly at ``x = +-l/2``, so ``l`` is a centre-to-centre length that
    does not move with the grid (where the de-embedding's reference planes
    actually land is a separate question -- see the module doc).
    Returns ``(x (4,), y (2,))``."""
    x, yv = _bar_edge_coordinates_unsorted(theta)
    return jnp.sort(x), jnp.sort(yv)


def straight_bar_geometry(l_bar: float, width: float, layer=KEY_M2) -> gds.Spiral:
    """Host-numpy geometry of the straight-bar DUT in the shape
    :func:`build_spiral` consumes (a :class:`rfx.fdfd.gds.Spiral` record):
    the bar rectangle AND the two column-footprint squares on ``layer``
    (overlapping polygons on one layer; :func:`rfx.fdfd.gds.fill_fractions`
    takes their union, so the raster is the bar and the squares only add
    their edges to :func:`rfx.fdfd.gds.mesh_lines`), no underpass, no via.
    ``ports`` are the two footprint centres ``(+l/2, -W/2)`` and
    ``(-l/2, -W/2)`` -- the ``+x`` one first, so it plays the role of the
    spiral's OUTER terminal (the one that must be right of the other)."""
    if l_bar <= width:
        raise ValueError("bar_length must exceed width (the two column footprints must not touch)")
    if width <= 0:
        raise ValueError("width must be > 0")
    hw, hl = 0.5 * width, 0.5 * l_bar
    bar = gds.rect_from_bounds(-(hl + hw), -hw, hl + hw, hw)
    foot_hi = gds.rect_from_bounds(hl - hw, -hw, hl + hw, hw)
    foot_lo = gds.rect_from_bounds(-(hl + hw), -hw, -(hl - hw), hw)
    centre = np.array([[-hl - hw, 0.0], [hl + hw, 0.0]], dtype=np.float64)
    return gds.Spiral(polygons={layer: [bar, foot_hi, foot_lo]}, centreline=centre,
                      segment_lengths=np.array([l_bar + width], dtype=np.float64),
                      length=float(l_bar + width), underpass_length=0.0,
                      ports=((hl, -hw), (-hl, -hw)))


# ----------------------------------------------------------------------------
# right-angle bend DUT (``dut_kind = "bend"``): the SAME fixture with the
# spiral replaced by ONE L-shaped strip with a single right angle

def _bend_edge_coordinates_unsorted(theta):
    """Unsorted breakpoints ``(x (4,), y (4,))`` of the L-shaped bend in a
    FIXED analytic order; ``theta = (l1, l2, width)``."""
    theta = jnp.asarray(theta, dtype=jnp.float64)
    l1, l2, width = theta[0], theta[1], theta[2]
    hw = 0.5 * width
    x = jnp.stack([-(0.5 * l2 + hw), -(0.5 * l2 - hw), 0.5 * l2 - hw, 0.5 * l2 + hw])
    yv = jnp.stack([-(0.5 * l1 + hw), -(0.5 * l1 - hw), 0.5 * l1 - hw, 0.5 * l1 + hw])
    return x, yv


def bend_edge_coordinates(theta):
    """Sorted x and y coordinates of every axis-aligned polygon edge of the
    right-angle bend DUT for ``theta = (l1, l2, width)``, as ``jax.numpy``
    functions of ``theta``.

    The bend is one L-shaped strip on the TOP metal (``KEY_M2``) whose
    centreline runs ``(l2/2, -(l1+W)/2) -> (l2/2, l1/2) -> (-(l2+W)/2,
    l1/2)``: a leg of length ``l1`` in ``+y`` (the LEAD direction of the
    spiral fixture, i.e. the direction in which the DUT current leaves the
    outer lead column), one right angle at ``(l2/2, l1/2)``, and a leg of
    length ``l2`` in ``-x``. ``l1`` and ``l2`` are CENTRE-TO-CENTRE lengths:
    from the outer lead-column footprint centre ``(l2/2, -l1/2)`` to the
    corner and from the corner to the inner footprint centre
    ``(-l2/2, l1/2)``, so the de-embedded centreline length is exactly
    ``l1 + l2`` and the drawn strip is ``l1 + l2 + W`` long (it overhangs
    each footprint centre by ``W/2``, exactly as the straight bar does).
    The two ``W x W`` footprint squares are the legs' end squares, so the
    four x breakpoints are ``+-(l2 +- W)/2`` and the four y breakpoints
    ``+-(l1 +- W)/2`` -- the legs' two edges plus each footprint's DUT-side
    edge -- which makes both footprints grid-exact at every ``base_dx``.
    Returns ``(x (4,), y (4,))``."""
    x, yv = _bend_edge_coordinates_unsorted(theta)
    return jnp.sort(x), jnp.sort(yv)


def bend_geometry(l1: float, l2: float, width: float, layer=KEY_M2) -> gds.Spiral:
    """Host-numpy geometry of the right-angle bend DUT in the shape
    :func:`build_spiral` consumes (a :class:`rfx.fdfd.gds.Spiral` record):
    the mitred L polygon (:func:`rfx.fdfd.gds.offset_polyline` of the
    three-point centreline, so its area is exactly ``width x`` centreline
    length and the corner square is drawn once) AND the two
    column-footprint squares on ``layer`` (overlapping polygons on one
    layer, unioned by :func:`rfx.fdfd.gds.fill_fractions`; they only add
    their edges to :func:`rfx.fdfd.gds.mesh_lines`), no underpass, no via.
    ``ports`` are ``((l2/2, -(l1+W)/2), (-l2/2, (l1-W)/2))`` in the
    ``(x centre, y LOW edge)`` convention :func:`build_spiral` reads them
    in -- the ``+x`` one first, so it plays the role of the spiral's OUTER
    terminal (the one that must be right of the other). The two footprints
    therefore differ in BOTH x and y: unlike the bar's, these two lead
    columns are not collinear (see the module doc for the short standard
    that follows)."""
    if width <= 0:
        raise ValueError("width must be > 0")
    if l1 <= width or l2 <= width:
        raise ValueError("bend_l1 and bend_l2 must exceed width (the legs must clear the "
                         "footprint squares and leave the corner one full width)")
    hw = 0.5 * width
    x_o, y_c1 = 0.5 * l2, -0.5 * l1
    y_c2, x_i = 0.5 * l1, -0.5 * l2
    centre = np.array([[x_o, y_c1 - hw], [x_o, y_c2], [x_i - hw, y_c2]], dtype=np.float64)
    strip = gds.offset_polyline(centre, width)
    foot_o = gds.rect_from_bounds(x_o - hw, y_c1 - hw, x_o + hw, y_c1 + hw)
    foot_i = gds.rect_from_bounds(x_i - hw, y_c2 - hw, x_i + hw, y_c2 + hw)
    seg = np.array([l1 + hw, l2 + hw], dtype=np.float64)
    return gds.Spiral(polygons={layer: [strip, foot_o, foot_i]}, centreline=centre,
                      segment_lengths=seg, length=float(l1 + l2 + width),
                      underpass_length=0.0,
                      ports=((x_o, y_c1 - hw), (x_i, y_c2 - hw)))


def _edge_coordinates_for(spec: "SpiralSpec", theta):
    """Unsorted breakpoints of whichever DUT ``spec`` describes (the single
    dispatch point of the ``dut_kind`` hook; everything downstream -- line
    maps, traced grid, fixtures, de-embedding -- is shared)."""
    if spec.dut_kind == "bar":
        return _bar_edge_coordinates_unsorted(theta)
    if spec.dut_kind == "bend":
        return _bend_edge_coordinates_unsorted(theta)
    if spec.dut_kind != "spiral":
        raise ValueError(f"unknown dut_kind {spec.dut_kind!r} (one of {DUT_KINDS})")
    return _edge_coordinates_unsorted(theta, spec.n_turns, spec.lead)


# ----------------------------------------------------------------------------
# line map: nominal grid lines -> (segment, fraction) between breakpoints

class LineMap(NamedTuple):
    """Static bookkeeping of one axis: ``bp_nom (K,)`` sorted nominal
    breakpoints including the two walls, ``perm`` (K-2,) the argsort of the
    analytic (unsorted) breakpoints, ``seg (n+1,)`` / ``frac (n+1,)`` the
    segment index and position of every nominal grid line."""
    bp_nom: np.ndarray
    perm: np.ndarray
    seg: np.ndarray
    frac: np.ndarray


def _line_map(lines: np.ndarray, bps_unsorted: np.ndarray) -> LineMap:
    lines = np.asarray(lines, dtype=np.float64)
    perm = np.argsort(bps_unsorted, kind="stable")
    inner = bps_unsorted[perm]
    if np.any(np.diff(inner) <= 0):
        raise ValueError("nominal breakpoints are not strictly increasing")
    bp = np.concatenate([[lines[0]], inner, [lines[-1]]])
    if not (inner[0] > lines[0] and inner[-1] < lines[-1]):
        raise ValueError("breakpoints must lie strictly inside the box")
    span = lines[-1] - lines[0]
    tol = 1e-9 * span
    # every breakpoint must be a grid line (mesh_lines(snap=True) guarantees it)
    hit = np.abs(lines[:, None] - bp[None, :]) <= tol          # (n+1, K)
    if not np.all(hit.sum(axis=0) == 1):
        raise ValueError("a breakpoint is not on the nominal grid (or hits two lines)")
    seg = np.clip(np.searchsorted(bp, lines, side="right") - 1, 0, len(bp) - 2)
    on_bp = hit.any(axis=1)
    bp_index = np.argmax(hit, axis=1)
    seg = np.where(on_bp, np.minimum(bp_index, len(bp) - 2), seg)
    lo, hi = bp[seg], bp[seg + 1]
    frac = np.where(on_bp, np.where(bp_index == len(bp) - 1, 1.0, 0.0), (lines - lo) / (hi - lo))
    return LineMap(bp_nom=bp, perm=perm, seg=seg.astype(np.int64), frac=frac)


def _traced_lines(lm: LineMap, lines_nom: np.ndarray, bps_unsorted) -> jax.Array:
    """``x_nom + d_lo + frac (d_hi - d_lo)`` with ``d = bps(theta) - bps_nom``
    (zero at the walls): exact at the nominal parameters."""
    inner = jnp.asarray(bps_unsorted, dtype=jnp.float64)[lm.perm] - lm.bp_nom[1:-1]
    delta = jnp.concatenate([jnp.zeros((1,)), inner, jnp.zeros((1,))])
    d_lo, d_hi = delta[lm.seg], delta[lm.seg + 1]
    return jnp.asarray(lines_nom) + d_lo + lm.frac * (d_hi - d_lo)


# ----------------------------------------------------------------------------
# static model

@dataclass(frozen=True)
class Column:
    """Lead column footprint: cell index ranges ``[i0, i1) x [j0, j1)`` and
    the z cells ``[k0, k1)`` of the PEC column; the port is the Ez box on
    the nodes ``[i0, i1] x [j0, j1]`` over the z cells ``[0, k0)``. With
    ``lead_stub_cells > 0`` the footprint sits that many cells below the
    terminal and a PEC stub on the terminal's metal layer joins them."""
    i0: int
    i1: int
    j0: int
    j1: int
    k0: int
    k1: int


@dataclass(frozen=True)
class SpiralModel:
    """Everything static: grid, masks, patterns, ports, line maps."""
    spec: SpiralSpec
    stack: gds.LayerStack
    spiral: gds.Spiral
    yee: y.Yee3DModel
    x_nom: np.ndarray
    y_nom: np.ndarray
    z: np.ndarray
    xmap: LineMap
    ymap: LineMap
    cells: dict[str, np.ndarray]                 # fixture -> (nx, ny, nz) bool metal cells
    pec: dict[str, tuple]                        # fixture -> (Ex, Ey, Ez) bool PEC edge masks
    geo: dict[str, cd.ConductorGeometry]         # fixture -> Leontovich geometry
    ports: tuple[p3.LumpedElement, p3.LumpedElement]
    columns: tuple[Column, Column]
    eps_static: np.ndarray                       # (nx, ny, nz) complex, lossless
    si_cells: np.ndarray                         # (nx, ny, nz) bool, silicon (non-metal) cells
    theta_nominal: tuple[float, ...]
    build_seconds: float
    fixture: dict[str, Any] = field(default_factory=dict)   # invariant-fixture bookkeeping

    @property
    def n_unknowns(self) -> int:
        return self.yee.n_unknowns

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.yee.shape

    @property
    def fixtures(self) -> tuple[str, ...]:
        return ("dut", "open", "short")


def _footprint(lines: np.ndarray, lo: float, hi: float) -> tuple[int, int]:
    """Cell index range whose centres lie in ``[lo, hi]``."""
    centres = 0.5 * (lines[:-1] + lines[1:])
    idx = np.nonzero((centres >= lo - 1e-12 * (hi - lo)) & (centres <= hi + 1e-12 * (hi - lo)))[0]
    if len(idx) == 0:
        raise ValueError("empty column footprint")
    if not np.all(np.diff(idx) == 1):
        raise AssertionError("footprint cells not contiguous")
    return int(idx[0]), int(idx[-1]) + 1


def _z_cell_range(z: np.ndarray, z0: float, z1: float) -> tuple[int, int]:
    centres = 0.5 * (z[:-1] + z[1:])
    idx = np.nonzero((centres > z0) & (centres < z1))[0]
    return int(idx[0]), int(idx[-1]) + 1


def pad_lines(lines: np.ndarray, n: int, ratio: float, lo: bool = True, hi: bool = True) -> np.ndarray:
    """Append ``n`` geometrically growing cells (factor ``ratio``) outside
    ``lines`` on the ``lo`` / ``hi`` end, starting from the end cell of
    ``lines``: the cheap way to move a PEC wall far away (``n`` cells add
    ``d_end (ratio^(n+1) - ratio) / (ratio - 1)``, e.g. 31 x the end cell
    for ``n = 6``, ``ratio = 1.5``). ``n = 0`` returns the input unchanged."""
    lines = np.asarray(lines, dtype=np.float64)
    if n <= 0:
        return lines
    if ratio <= 1.0:
        raise ValueError("pad_ratio must be > 1")
    out = lines
    if lo:
        d, v, pre = lines[1] - lines[0], lines[0], []
        for _ in range(int(n)):
            d *= ratio
            v -= d
            pre.append(v)
        out = np.concatenate([np.asarray(pre[::-1]), out])
    if hi:
        d, v, post = lines[-1] - lines[-2], out[-1], []
        for _ in range(int(n)):
            d *= ratio
            v += d
            post.append(v)
        out = np.concatenate([out, np.asarray(post)])
    return out


# ----------------------------------------------------------------------------
# the level-invariant fixture: physical walls, nested refinement

PAD_REFINE_MODES = ("uniform", "graded")


def pad_to(lines: np.ndarray, wall: float, ratio: float, hi: bool = True) -> np.ndarray:
    """Graded padding from the end cell of ``lines`` to the FIXED coordinate
    ``wall`` (on the ``hi`` end, or the ``lo`` end with ``hi=False``).

    With ``d`` the end cell and ``D`` the distance to the wall, the number of
    cells ``n`` is the smallest for which ``d (r + r^2 + ... + r^n) >= D`` at
    ``r = ratio``; the common ratio is then solved (bisection) so the ``n``
    cells ``d q^k`` sum to ``D`` exactly and the last line is PINNED to
    ``wall`` bit-exactly. Every neighbour ratio is therefore in ``[1/ratio,
    ratio]`` (``q <= ratio`` by the choice of ``n``; ``q > 1/ratio`` whenever
    ``D >= d / ratio``, which is required). Unlike :func:`pad_lines`, where
    the wall is wherever ``pad_cells`` cells happen to end (so it moves with
    the end cell, i.e. with the grid level), the wall here is a property of
    the geometry."""
    lines = np.asarray(lines, dtype=np.float64)
    if ratio <= 1.0:
        raise ValueError("pad ratio must be > 1")
    if hi:
        d, edge, span = lines[-1] - lines[-2], lines[-1], float(wall) - lines[-1]
    else:
        d, edge, span = lines[1] - lines[0], lines[0], lines[0] - float(wall)
    if span < d / ratio:
        raise ValueError(f"wall {wall!r} is closer to the meshed box than end cell / ratio "
                         f"({span:.3g} m < {d / ratio:.3g} m)")

    def total(q: float, n: int) -> float:
        return d * float(np.sum(q ** np.arange(1, n + 1, dtype=np.float64)))

    n = 1
    while total(ratio, n) < span:
        n += 1
        if n > 400:
            raise RuntimeError("pad_to: no cell count reaches the wall (report this)")
    lo_q, hi_q = 1.0 / ratio, float(ratio)
    for _ in range(200):
        mid = 0.5 * (lo_q + hi_q)
        if total(mid, n) < span:
            lo_q = mid
        else:
            hi_q = mid
    q = 0.5 * (lo_q + hi_q)
    cells = d * q ** np.arange(1, n + 1, dtype=np.float64)
    cells *= span / cells.sum()
    steps = np.cumsum(cells)
    if hi:
        new = edge + steps
        new[-1] = float(wall)
        return np.concatenate([lines, new])
    new = (edge - steps)[::-1]
    new[0] = float(wall)
    return np.concatenate([new, lines])


def subdivide_lines(lines: np.ndarray, m: int) -> np.ndarray:
    """Every interval of ``lines`` cut into ``m`` equal cells (nested
    refinement): every input line is an output line bit-exactly
    (``np.linspace`` returns both end points exactly). ``m = 1`` returns
    the input unchanged."""
    lines = np.asarray(lines, dtype=np.float64)
    m = int(m)
    if m < 1:
        raise ValueError("refinement factor must be >= 1")
    if m == 1:
        return lines
    out = [lines[:1]]
    for a, b in zip(lines[:-1], lines[1:]):
        out.append(np.linspace(a, b, m + 1)[1:])
    return np.concatenate(out)


def _refine_pad(pad: np.ndarray, edge: float, cell: float, m: int, mode: str,
                ratio: float) -> np.ndarray:
    """Refine one padding run ``pad`` (the level-1 lines strictly beyond the
    meshed box's ``edge`` up to and including the wall, ordered away from the
    box) for level ``m``, given the refined box's end cell ``cell``.
    ``"uniform"`` subdivides every padding interval ``m`` times; ``"graded"``
    keeps the level-1 padding lines and inserts only the lines a neighbour
    ratio <= ``0.95 ratio`` needs between the refined end cell and them
    (:func:`rfx.fdfd.gds.grade_lines`). Both keep every level-1 line and
    both end exactly on the wall."""
    if m == 1 or len(pad) == 0:
        return pad
    pad = np.asarray(pad, dtype=np.float64)
    sgn = 1.0 if pad[0] > edge else -1.0
    u = sgn * (pad - edge)                                           # > 0, increasing
    if mode == "uniform":
        out = subdivide_lines(np.concatenate([[0.0], u]), m)[1:]
        idx = np.arange(m - 1, len(out), m)
    elif mode == "graded":
        # grade_lines enforces 0.95 x its ratio argument; ask for ``ratio``
        # itself so the level-1 padding (neighbour ratio <= ratio by
        # construction) is left alone and only the transition is filled
        g = gds.grade_lines(np.concatenate([[-cell, 0.0], u]),
                            float(np.max(np.diff(u, prepend=0.0))),
                            ratio * (1.0 + 1e-9) / 0.95)
        out = g[g > 0.0]
        idx = np.searchsorted(out, u - 1e-9 * u[-1])
        if not np.allclose(out[idx], u, rtol=0.0, atol=1e-12 * u[-1]):
            raise RuntimeError("graded padding lost a level-1 line (report this)")
    else:
        raise ValueError(f"pad_refine must be one of {PAD_REFINE_MODES}, got {mode!r}")
    res = edge + sgn * out
    res[idx] = pad          # the level-1 lines (walls included) bit-exactly
    return res


def _stack_z_lines(st: "SmallStack", stack: gds.LayerStack, extra: Sequence[float]) -> np.ndarray:
    """:func:`rfx.fdfd.gds.z_lines` over ``[0, z_top]`` with ``extra``
    mandatory z lines (the physical port gap): every stack interface and
    every extra line exactly, uniform cells of at most ``base_dz`` between
    them, at least ``metal_cells`` across every metal / via slab. With no
    extra line inside an interval this is ``z_lines`` itself."""
    zi = stack.interfaces()
    lo, hi = 0.0, float(st.z_top)
    ex = np.asarray([v for v in extra if lo < v < hi], dtype=np.float64)
    z = np.unique(np.concatenate([[lo, hi], zi[(zi > lo) & (zi < hi)], ex]))
    out = [z[0]]
    for a, b in zip(z[:-1], z[1:]):
        zc = 0.5 * (a + b)
        need = 1
        for lay in stack.layers:
            if lay.is_metal and lay.z0 <= zc < lay.z1:
                need = max(need, st.metal_cells)
        n = max(need, int(np.ceil((b - a) / st.base_dz - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:].tolist())
    return np.asarray(out, dtype=np.float64)


def _fixture_rects(spec: "SpiralSpec", sp: gds.Spiral) -> list:
    """Plan-view rectangles whose edges the invariant fixture makes grid
    lines: the two lead-column footprints (``W x W`` at each port) and, with
    ``short_gap_m``, the short standard's post (``x`` from the inner
    column's ``+x`` edge plus the gap to the outer column's ``-x`` edge minus
    the gap, ``y`` over the union of the two terminal rows). The bars run
    from the post to the columns' far edges, which are these same lines."""
    w = float(spec.width)
    hw = 0.5 * w
    rects = [gds.rect_from_bounds(xc - hw, yc, xc + hw, yc + w) for (xc, yc) in sp.ports]
    if spec.short_gap_m is not None:
        g = float(spec.short_gap_m)
        (xo, yo), (xi, yi) = sp.ports
        x0, x1 = xi + hw + g, xo - hw - g
        if x1 <= x0 or g <= 0:
            raise ValueError(f"short_gap_m = {g!r} leaves no post between the lead columns "
                             f"(post width {x1 - x0:.3g} m)")
        rects.append(gds.rect_from_bounds(x0, min(yo, yi), x1, max(yo, yi) + w))
    return rects


def _fixture_grid(spec: "SpiralSpec", sp: gds.Spiral, stack: gds.LayerStack,
                  bounds: tuple[float, float, float, float],
                  bbox: tuple[float, float, float, float]):
    """x, y, z lines of the level-invariant fixture and their bookkeeping.

    LEVEL 1: x/y from :func:`rfx.fdfd.gds.mesh_lines` over the meshed box
    ``bounds`` with the polygons AND :func:`_fixture_rects` as mandatory
    edges; z from :func:`_stack_z_lines` with the port gap as a mandatory
    line. Padding (level 1): ``wall_margin_m`` -> :func:`pad_to` the four
    walls at ``bbox -+ wall_margin_m`` (else ``pad_cells`` as before);
    ``lid_height_m`` -> :func:`pad_to` the lid (else ``pad_cells_z``).
    LEVEL m: the meshed box's intervals cut into ``refine`` (x, y) and
    ``z_refine or refine`` (z) equal cells, the padding refined by
    :func:`_refine_pad` in the ``pad_refine`` mode. The ground (z = 0) is
    never padded."""
    st = spec.stack
    m = int(spec.refine)
    mz = int(spec.z_refine) or m
    if m < 1 or mz < 1:
        raise ValueError("refine and z_refine must be >= 1 (z_refine = 0: follow refine)")
    if spec.pad_refine not in PAD_REFINE_MODES:
        raise ValueError(f"pad_refine must be one of {PAD_REFINE_MODES}")
    polys = [p for ps in sp.polygons.values() for p in ps] + _fixture_rects(spec, sp)
    x1c, y1c = gds.mesh_lines(polys, bounds, spec.dx, snap=True)
    extra = [] if spec.port_gap_m is None else [float(spec.port_gap_m)]
    z1c = (gds.z_lines(stack, st.base_dz, metal_cells=st.metal_cells, z_min=0.0, z_max=st.z_top)
           if spec.port_gap_m is None else _stack_z_lines(st, stack, extra))
    ratio = float(spec.pad_ratio)

    def one_axis(core: np.ndarray, lo_wall: float | None, hi_wall: float | None,
                 n_pad: int, lo: bool, k: int) -> tuple[np.ndarray, dict[str, Any]]:
        lvl1 = core
        if hi_wall is not None:
            lvl1 = pad_to(lvl1, hi_wall, ratio, hi=True)
            if lo and lo_wall is not None:
                lvl1 = pad_to(lvl1, lo_wall, ratio, hi=False)
        elif n_pad:
            lvl1 = pad_lines(lvl1, n_pad, ratio, lo=lo, hi=True)
        i0 = int(np.searchsorted(lvl1, core[0]))
        i1 = i0 + len(core) - 1
        fine = subdivide_lines(core, k)
        lo_pad = _refine_pad(lvl1[:i0][::-1], core[0], fine[1] - fine[0], k, spec.pad_refine,
                             ratio)[::-1]
        hi_pad = _refine_pad(lvl1[i1 + 1:], core[-1], fine[-1] - fine[-2], k, spec.pad_refine,
                             ratio)
        out = np.concatenate([lo_pad, fine, hi_pad])
        if not np.all(np.diff(out) > 0):
            raise RuntimeError("invariant grid is not strictly increasing (report this)")
        info = {"level1": lvl1, "core_level1": core, "core_index": (len(lo_pad), len(lo_pad) + len(fine) - 1)}
        return out, info

    wm = spec.wall_margin_m
    if wm is not None and spec.pad_cells:
        raise ValueError("wall_margin_m and pad_cells are alternative wall placements")
    if spec.lid_height_m is not None and (spec.pad_cells_z or (spec.pad_cells and wm is None)):
        raise ValueError("lid_height_m and pad_cells(_z) are alternative lid placements")
    bx0, by0, bx1, by1 = bbox
    x, xi = one_axis(x1c, None if wm is None else bx0 - wm, None if wm is None else bx1 + wm,
                     spec.pad_cells, True, m)
    y, yi = one_axis(y1c, None if wm is None else by0 - wm, None if wm is None else by1 + wm,
                     spec.pad_cells, True, m)
    z, zi = one_axis(z1c, None, None if spec.lid_height_m is None else float(spec.lid_height_m),
                     (spec.pad_cells_z or spec.pad_cells) if spec.lid_height_m is None else 0,
                     False, mz)
    info = {"refine": m, "z_refine": mz, "pad_refine": spec.pad_refine,
            "x": xi, "y": yi, "z": zi}
    return x, y, z, info


def build_spiral(spec: SpiralSpec) -> SpiralModel:
    """Static model at the nominal ``theta`` (see the module doc)."""
    t0 = time.time()
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError("build_spiral needs x64: the nominal breakpoints are evaluated by the "
                           "same float64 jnp function the solve traces")
    st = spec.stack
    stack = st.layer_stack()
    if spec.dut_kind == "bar":
        sp = straight_bar_geometry(spec.bar_length, spec.width)
    elif spec.dut_kind == "bend":
        sp = bend_geometry(spec.bend_l1, spec.bend_l2, spec.width)
    elif spec.dut_kind == "spiral":
        sp = gds.rect_spiral(spec.n_turns, spec.r_out, spec.width, spec.spacing, spec.lead,
                             layer=KEY_M2, underpass_layer=KEY_M1, via_layer=KEY_VIA)
    else:
        raise ValueError(f"unknown dut_kind {spec.dut_kind!r} (one of {DUT_KINDS})")
    polys = [p for ps in sp.polygons.values() for p in ps]
    allp = np.concatenate(polys)
    xlo, ylo = allp.min(axis=0) - spec.margin
    xhi, yhi = allp.max(axis=0) + spec.margin
    fixture: dict[str, Any] = {}
    if spec.invariant:
        # the level-invariant fixture (module doc): physical walls / lid / post
        # / port gap and nested refinement; the branch below is untouched
        pmin, pmax = allp.min(axis=0), allp.max(axis=0)
        x_nom, y_nom, z, fixture = _fixture_grid(
            spec, sp, stack, (xlo, ylo, xhi, yhi),
            (float(pmin[0]), float(pmin[1]), float(pmax[0]), float(pmax[1])))
    else:
        x_nom, y_nom = gds.mesh_lines(sp.polygons, (xlo, ylo, xhi, yhi), spec.dx, snap=True)
        z = gds.z_lines(stack, st.base_dz, metal_cells=st.metal_cells, z_min=0.0, z_max=st.z_top)
        if spec.pad_cells:
            # graded padding outside the meshed box: the four side walls and the
            # lid move away, the GROUND (z = 0) stays where it is
            x_nom = pad_lines(x_nom, spec.pad_cells, spec.pad_ratio)
            y_nom = pad_lines(y_nom, spec.pad_cells, spec.pad_ratio)
            z = pad_lines(z, spec.pad_cells_z or spec.pad_cells, spec.pad_ratio, lo=False, hi=True)
    nx, ny, nz = len(x_nom) - 1, len(y_nom) - 1, len(z) - 1
    if spec.port_gap_cells < 1:
        raise ValueError("port_gap_cells must be >= 1")
    if spec.lead_stub_cells < 0:
        raise ValueError("lead_stub_cells must be >= 0")
    if spec.lead_stub_cells > 0:
        warnings.warn("lead_stub_cells > 0: a collinear lead stub couples to the DUT by mutual "
                      "inductance and open/short de-embedding over-corrects it (+2.3 % on L_diff "
                      "per 10 um cell, measured); not a valid de-embedding configuration",
                      stacklevel=2)

    # breakpoints and line maps (nominal analytic breakpoints from the SAME
    # jnp function the solve uses, so the delta form is exactly zero there)
    bx, by = (np.asarray(v, dtype=np.float64) for v in
              _edge_coordinates_for(spec, np.asarray(spec.theta)))
    xmap = _line_map(x_nom, bx)
    ymap = _line_map(y_nom, by)

    # materials (lossless): eps per cell from the stack, conductor cells
    ras = gds.rasterise(sp.polygons, stack, x_nom, y_nom, z, freq=None)
    spiral_cells = np.asarray(ras.conductor_mask, dtype=bool)
    zc = 0.5 * (z[:-1] + z[1:])
    si_cells = np.broadcast_to((zc < st.t_si)[None, None, :], (nx, ny, nz)).copy()

    # lead columns and ports
    hw = 0.5 * spec.width
    k_m1 = _z_cell_range(z, st.z_m1, st.z_m1 + st.t_m1)
    k_m2 = _z_cell_range(z, st.z_m2, st.z_m2 + st.t_m2)
    gap = spec.port_gap_cells
    if spec.port_gap_m is not None:
        # physical port gap: the column bottom is the z line at port_gap_m
        # (a mandatory line of the invariant grid), however many cells lie below
        hit = np.nonzero(np.abs(z - float(spec.port_gap_m)) <= 1e-12 * (z[-1] - z[0]))[0]
        if len(hit) != 1:
            raise AssertionError("port_gap_m is not a z line of the grid (report this)")
        gap = int(hit[0])
    stub = spec.lead_stub_cells
    cols = []
    stub_cells = np.zeros((nx, ny, nz), dtype=bool)
    # the metal level each terminal sits on: the spiral's inner terminal is on
    # the underpass (M1), the straight bar has both terminals on M2
    metal_ks = (k_m2, k_m2) if spec.dut_kind in ("bar", "bend") else (k_m2, k_m1)
    terminal_rows: list[tuple[int, int]] = []
    # each port carries its own ``(x centre, y LOW edge)``; for the spiral and
    # the bar both ports sit on the SAME y line (the port line / the bar row)
    # and this loop is bit-identical to reading it once, but the bend's two
    # terminals differ in y as well as in x
    for (xc, yc), k_metal in zip(sp.ports, metal_ks):
        i0, i1 = _footprint(x_nom, xc - hw, xc + hw)
        jt0, jt1 = _footprint(y_nom, yc, yc + spec.width)              # terminal cell(s)
        if gap >= k_metal[0]:
            raise ValueError("port gap reaches the metal layer; fewer port_gap_cells")
        j0, j1 = jt0 - stub, jt1 - stub
        if j0 < 1:
            raise ValueError("lead_stub_cells exceeds the margin cells below the port line")
        cols.append(Column(i0, i1, j0, j1, gap, k_metal[0]))
        terminal_rows.append((jt0, jt1))
        if stub:
            stub_cells[i0:i1, j0:jt0, k_metal[0]:k_metal[1]] = True
    columns = (cols[0], cols[1])
    ports = tuple(p3.LumpedElement(2, (c.i0, c.j0, 0), (c.i1 + 1, c.j1 + 1, gap)) for c in columns)

    col_cells = np.zeros((nx, ny, nz), dtype=bool)
    for c in columns:
        col_cells[c.i0:c.i1, c.j0:c.j1, c.k0:c.k1] = True
    col_cells |= stub_cells
    si_cells &= ~(spiral_cells | col_cells)

    # short standard: each lead shorted to ground where the DUT begins -- a
    # bar on the terminal's OWN metal level (M2 for the outer, M1 for the
    # inner terminal) covering the column top / the last stub cell and
    # running to a common post between the columns; the post goes from
    # the ground up through both levels. The short's current then leaves
    # each lead at the same cell the DUT's current does.
    outer, inner = columns
    if inner.i1 >= outer.i0:
        raise AssertionError("column order: the inner terminal must be left of the outer one")
    post_i0, post_i1 = inner.i1 + 1, outer.i0 - 1
    if spec.short_gap_m is not None:
        # physical short standard: post edges at the column edges -+ the gap
        # (mandatory x lines of the invariant grid), not one cell from them
        g = float(spec.short_gap_m)
        post_i0, post_i1 = _footprint(x_nom, x_nom[inner.i1] + g, x_nom[outer.i0] - g)
        span_x = x_nom[-1] - x_nom[0]
        if (abs(x_nom[post_i0] - (x_nom[inner.i1] + g)) > 1e-12 * span_x
                or abs(x_nom[post_i1] - (x_nom[outer.i0] - g)) > 1e-12 * span_x):
            raise AssertionError("short-standard post edges are not grid lines (report this)")
    if post_i1 <= post_i0:
        raise ValueError("no room for the short's ground post between the lead columns "
                         "(need >= 3 cells between them); larger n_turns * pitch or finer base_dx")
    # bar row of each column: the DUT-side end of ITS OWN lead. For the spiral
    # and the bar the two rows coincide (both terminals on the port line) and
    # this is the single row of the original code; for the bend they differ and
    # the post spans their union so that both arms reach it.
    rows = [(max(c.j0, jt - 1), max(c.j0, jt - 1) + (c.j1 - c.j0))
            for c, (jt, _) in zip(columns, terminal_rows)]
    jp0, jp1 = min(r[0] for r in rows), max(r[1] for r in rows)
    bar_cells = np.zeros((nx, ny, nz), dtype=bool)
    bar_cells[post_i0:outer.i1, rows[0][0]:rows[0][1], metal_ks[0][0]:metal_ks[0][1]] = True
    bar_cells[inner.i0:post_i1, rows[1][0]:rows[1][1], metal_ks[1][0]:metal_ks[1][1]] = True
    post_cells = np.zeros((nx, ny, nz), dtype=bool)
    post_cells[post_i0:post_i1, jp0:jp1, 0:max(metal_ks[0][1], metal_ks[1][1])] = True

    cells = {
        "dut": spiral_cells | col_cells,
        "open": col_cells.copy(),
        "short": col_cells | bar_cells | post_cells,
    }
    if np.any(cells["open"] & spiral_cells):
        raise AssertionError("lead columns / stubs overlap the spiral metal")
    yee = y.build(y.Yee3DSpec(nx, ny, nz, pml=spec.pml, pml_order=spec.pml_order,
                             pml_kappa_max=spec.pml_kappa_max, pml_r0=spec.pml_r0))
    pec = {k: y.pec_edges_from_cells(yee.spec, v) for k, v in cells.items()}
    # the port edges must be free in every fixture
    for k, masks in pec.items():
        for port in ports:
            ids = p3.element_edges(yee, port)
            if np.any(np.concatenate([m.ravel() for m in masks])[ids]):
                raise AssertionError(f"port edges are PEC in fixture {k}")
    geo = {k: cd.conductor_geometry(yee, cd.Conductor(cells=v)) for k, v in cells.items()}
    if spec.invariant:
        fixture = dict(fixture, post=(post_i0, post_i1, jp0, jp1, 0,
                                      max(metal_ks[0][1], metal_ks[1][1])),
                       arms=((post_i0, outer.i1) + rows[0] + metal_ks[0],
                             (inner.i0, post_i1) + rows[1] + metal_ks[1]),
                       port_gap_index=gap)
    return SpiralModel(
        spec=spec, stack=stack, spiral=sp, yee=yee, x_nom=x_nom, y_nom=y_nom, z=z,
        xmap=xmap, ymap=ymap, cells=cells, pec=pec, geo=geo, ports=(ports[0], ports[1]),
        columns=columns, eps_static=np.asarray(ras.eps_r, dtype=np.complex128), si_cells=si_cells,
        theta_nominal=spec.theta, build_seconds=time.time() - t0, fixture=fixture)


# ----------------------------------------------------------------------------
# traced grid

def spiral_lines(model: SpiralModel, theta=None):
    """Traced grid lines ``(x (nx+1,), y (ny+1,), z (nz+1,))`` at ``theta``
    (nominal if ``None``); ``z`` is static."""
    if theta is None:
        theta = model.theta_nominal
    bx, by = _edge_coordinates_for(model.spec, theta)
    return (_traced_lines(model.xmap, model.x_nom, bx),
            _traced_lines(model.ymap, model.y_nom, by),
            jnp.asarray(model.z, dtype=jnp.float64))


def breakpoint_gaps(model: SpiralModel, theta):
    """Signed gaps between consecutive breakpoints (walls included) on x and
    y, concatenated; the parameterisation is valid iff all are positive
    (use as an optimiser constraint)."""
    bx, by = _edge_coordinates_for(model.spec, theta)
    out = []
    for lm, b in ((model.xmap, bx), (model.ymap, by)):
        inner = jnp.asarray(b)[lm.perm]
        full = jnp.concatenate([lm.bp_nom[:1], inner, lm.bp_nom[-1:]])
        out.append(jnp.diff(full))
    return jnp.concatenate(out)


def check_feasible(model: SpiralModel, theta) -> None:
    """Raise ``ValueError`` if ``theta`` breaks the breakpoint order or the
    generator's validity rules (host check, concrete values only)."""
    if model.spec.dut_kind == "bar":
        l_bar, width = (float(v) for v in np.asarray(theta, dtype=np.float64))
        if min(l_bar, width) <= 0:
            raise ValueError("bar_length, width must be > 0")
        if l_bar <= width:
            raise ValueError("infeasible: the two lead-column footprints touch (bar_length <= width)")
        gaps_b = np.asarray(breakpoint_gaps(model, np.asarray(theta, dtype=np.float64)))
        if np.any(gaps_b <= 0):
            raise ValueError(f"infeasible: breakpoint order violated (min gap {gaps_b.min():.3g} m)")
        return
    if model.spec.dut_kind == "bend":
        l1, l2, width = (float(v) for v in np.asarray(theta, dtype=np.float64))
        if min(l1, l2, width) <= 0:
            raise ValueError("bend_l1, bend_l2, width must be > 0")
        if min(l1, l2) <= width:
            raise ValueError("infeasible: a bend leg is shorter than the width "
                             "(the footprint squares would swallow the corner)")
        gaps_n = np.asarray(breakpoint_gaps(model, np.asarray(theta, dtype=np.float64)))
        if np.any(gaps_n <= 0):
            raise ValueError(f"infeasible: breakpoint order violated (min gap {gaps_n.min():.3g} m)")
        return
    r_out, spacing, width = (float(v) for v in np.asarray(theta, dtype=np.float64))
    if min(r_out, spacing, width) <= 0:
        raise ValueError("r_out, spacing, width must be > 0")
    n = model.spec.n_turns
    pitch = width + spacing
    a0 = r_out - 0.5 * width
    a_in = a0 - pitch * n
    if a_in <= 0.5 * width:
        raise ValueError("infeasible: the inner end reaches the axis (a_in <= width/2)")
    last_side = 2.0 * (a0 - pitch * (n - 1)) - pitch
    if last_side < width:
        raise ValueError("infeasible: last side shorter than the width")
    gaps = np.asarray(breakpoint_gaps(model, np.asarray(theta, dtype=np.float64)))
    if np.any(gaps <= 0):
        raise ValueError(f"infeasible: breakpoint order violated (min gap {gaps.min():.3g} m)")


def skin_depth(freq, sigma):
    """``delta = sqrt(2 / (omega mu0 sigma))`` in metres (jnp; traced inputs
    allowed): copper (5.8e7 S/m) gives 6.6 um at 100 MHz, 1.35 um at
    2.4 GHz."""
    omega = 2.0 * jnp.pi * jnp.asarray(freq, dtype=jnp.float64)
    return jnp.sqrt(2.0 / (omega * y.MU0 * jnp.asarray(sigma, dtype=jnp.float64)))


def leontovich_validity(model: SpiralModel, freq, sigma) -> float:
    """``delta / t_min``: the skin depth over the thinnest metal slab of the
    stack (M1, via, M2). The Leontovich sheet of ``solve_spiral`` is valid
    for values well below 1 (host float; see the module doc for the
    measured consequences of violating it)."""
    st = model.spec.stack
    t_min = min(st.t_m1, st.t_via, st.t_m2)
    return float(skin_depth(freq, sigma)) / t_min


# ----------------------------------------------------------------------------
# solve

class SpiralResult(NamedTuple):
    """One frequency. S-matrices are ``(2, 2)``; ``z_dut`` the de-embedded
    impedance matrix; the metrics scalars (see the module doc for the
    conventions); ``x_lines`` / ``y_lines`` / ``z_lines`` the traced grid."""
    freq: jax.Array
    s_raw: jax.Array
    s_open: jax.Array
    s_short: jax.Array
    z_dut: jax.Array
    s_dut: jax.Array
    L_diff: jax.Array
    Q_diff: jax.Array
    L_se: jax.Array
    Q_se: jax.Array
    L_z11_open: jax.Array
    L_raw: jax.Array
    x_lines: jax.Array
    y_lines: jax.Array
    z_lines: jax.Array


def _fixture_s(model: SpiralModel, name: str, freq, eps_r, dx, dy, dz, z0, sigma_metal,
               sigma_volumetric=None):
    if sigma_volumetric is not None:
        if sigma_metal is not None:
            raise ValueError("sigma_volumetric and sigma_metal are alternative metal models")
        omega = 2.0 * jnp.pi * jnp.asarray(freq, dtype=jnp.float64)
        eps_c = 1.0 - 1j * jnp.asarray(sigma_volumetric, dtype=jnp.float64) / (omega * y.EPS0)
        eps_v = jnp.where(jnp.asarray(model.cells[name]), eps_c, eps_r)
        return p3.s_matrix(model.yee, freq, eps_v, dx, dy, dz, list(model.ports), z0, pec=None)
    if sigma_metal is None:
        return p3.s_matrix(model.yee, freq, eps_r, dx, dy, dz, list(model.ports), z0,
                           pec=model.pec[name])
    cond = cd.Conductor(cells=model.cells[name])
    terms = cd.surface_terms(model.yee, freq, cond, sigma_metal, dx, dy, dz,
                             eps_r=eps_r, geo=model.geo[name])
    return p3.s_matrix(model.yee, freq, eps_r, dx, dy, dz, list(model.ports), z0, terms=terms)


def solve_spiral(model: SpiralModel, freq, theta=None, sigma_metal=None, sigma_si=0.0,
                 z0=None, fixtures: Sequence[str] = ("dut", "open", "short"),
                 sigma_volumetric=None) -> SpiralResult:
    """Solve the three fixtures at ``freq`` (Hz) and ``theta`` (nominal if
    ``None``), de-embed and evaluate the metrics. ``sigma_metal`` (S/m,
    traced) switches the metal from PEC to Leontovich; ``sigma_si`` (S/m,
    traced) is the silicon bulk conductivity. ``fixtures`` restricts the
    solves (e.g. ``("dut",)`` for the raw S only; the de-embedded fields are
    then ``nan``). Differentiable in ``theta``, ``sigma_metal``, ``sigma_si``
    and ``freq``; jit-able. A finite ``sigma_metal`` is only meaningful
    when the skin depth is well below the metal thickness
    (:func:`leontovich_validity` << 1; copper on the default 2 um metal
    needs f >> 1.1 GHz) -- this is not checked here because both inputs
    may be traced.

    ``sigma_volumetric`` (S/m, traced) is the THIRD metal model, added for
    the validation study of ``validation/fdfd/spiral_convergence.py``: every
    conductor cell of the fixture becomes a lossy dielectric ``eps_r = 1 -
    j sigma / (omega eps0)`` and NO interior PEC edge mask is applied, so
    the current density is resolved inside the metal. When the skin depth is
    much larger than every cross-section dimension the current is uniform
    and the extracted inductance includes the DC internal inductance --
    exactly the model of the Greenhouse referee. It excludes
    ``sigma_metal`` (the Leontovich sheet); the outer box walls stay PEC."""
    freq = jnp.asarray(freq, dtype=jnp.float64)
    z0 = model.spec.z0 if z0 is None else z0
    if theta is None:
        theta = model.theta_nominal
    dx_lines, dy_lines, dz_lines = spiral_lines(model, theta)
    dx, dy, dz = jnp.diff(dx_lines), jnp.diff(dy_lines), jnp.diff(dz_lines)
    omega = 2.0 * jnp.pi * freq
    eps_r = jnp.asarray(model.eps_static) + jnp.asarray(model.si_cells, dtype=jnp.float64) * (
        -1j * jnp.asarray(sigma_si, dtype=jnp.float64) / (omega * y.EPS0))
    nan2 = jnp.full((2, 2), jnp.nan + 0j, dtype=jnp.complex128)
    s = {name: _fixture_s(model, name, freq, eps_r, dx, dy, dz, z0, sigma_metal, sigma_volumetric)
         if name in fixtures else nan2 for name in ("dut", "open", "short")}
    f1 = freq[None]
    z_raw = de.s_to_z(s["dut"][:, :, None], z0)
    l_raw = de.l_diff(z_raw, f1)[0]
    z_dut = de.open_short_deembed(s["dut"][:, :, None], s["open"][:, :, None],
                                  s["short"][:, :, None], z0)
    m = de.inductor_metrics(z_dut, f1)
    s_dut = de.z_to_s(z_dut, z0)
    return SpiralResult(
        freq=freq, s_raw=s["dut"], s_open=s["open"], s_short=s["short"],
        z_dut=z_dut[:, :, 0], s_dut=s_dut[:, :, 0],
        L_diff=m["L_diff"][0], Q_diff=m["Q_diff"][0], L_se=m["L_y11"][0], Q_se=m["Q_y11"][0],
        L_z11_open=m["L_se"][0], L_raw=l_raw,
        x_lines=dx_lines, y_lines=dy_lines, z_lines=dz_lines)


# ----------------------------------------------------------------------------
# record

def _c2(a) -> list:
    a = np.asarray(a)
    return np.stack([a.real, a.imag], axis=-1).tolist()


def nominal_record(model: SpiralModel, res: SpiralResult, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON-serialisable summary of a solve (theta, grid, N, L, Q, S)."""
    th = model.theta_nominal
    if model.spec.dut_kind == "bar":
        theta_rec: dict[str, float] = {"bar_length": th[0], "width": th[1]}
    elif model.spec.dut_kind == "bend":
        theta_rec = {"bend_l1": th[0], "bend_l2": th[1], "width": th[2]}
    else:
        theta_rec = {"r_out": th[0], "spacing": th[1], "width": th[2]}
    rec: dict[str, Any] = {
        "theta": theta_rec, "dut_kind": model.spec.dut_kind,
        "n_turns": model.spec.n_turns, "lead": model.spec.lead, "base_dx": model.spec.dx,
        "margin": model.spec.margin, "port_gap_cells": model.spec.port_gap_cells,
        "lead_stub_cells": model.spec.lead_stub_cells,
        "z0": model.spec.z0,
        "grid": {"nx": model.shape[0], "ny": model.shape[1], "nz": model.shape[2],
                 "z_lines": model.z.tolist()},
        "n_unknowns": model.n_unknowns,
        "freq": float(res.freq),
        "L_diff": float(res.L_diff), "Q_diff": float(res.Q_diff),
        "L_se_y11": float(res.L_se), "Q_se_y11": float(res.Q_se),
        "L_z11_open": float(res.L_z11_open), "L_raw": float(res.L_raw),
        "S": {"raw": _c2(res.s_raw), "open": _c2(res.s_open), "short": _c2(res.s_short),
              "dut": _c2(res.s_dut)},
        "Z_dut": _c2(res.z_dut),
    }
    if extra:
        rec.update(extra)
    return rec
