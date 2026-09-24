"""The one metric contract both grid classes answer to.

Step 0a of the NU grid core change; design note
``docs/design_notes/20260922_nu_grid_core_predeclaration.md``, decision 3
("two metrics, one table"). Nothing here is imported by the solver kernels —
this module is the place the CONVENTION is written down once, plus the two
small pure helpers the accessors on :class:`rfx.grid.Grid` and
:class:`rfx.nonuniform.NonUniformGrid` share.

Why it exists
-------------
rfx keeps two grids. ``Grid`` carries one scalar ``dx``; ``NonUniformGrid``
carries per-cell arrays AND a scalar ``dx`` that means "the boundary cell".
A consumer that reads the scalar is right on a uniform mesh and silently
wrong on a graded one wherever the local cell differs from the boundary
cell. The accessors below give a consumer one place to ask, and this module
records which metric each consumer is entitled to.

Axis naming
-----------
``axis`` is ``"x"``, ``"y"`` or ``"z"``; the integers ``0``, ``1``, ``2`` are
accepted for callers that already hold a component axis. Anything else is a
``ValueError`` — there is no default axis.

Index origin
------------
Every array an accessor returns is indexed on the FULL PADDED axis: index 0
is the first absorber/pad cell, and the array length is the grid's ``n``
along that axis. This is the indexing ``dx_arr`` / ``dy_arr`` / ``dz`` and
``inv_dx`` / ``inv_dx_h`` already use, so an index taken from
``position_to_index`` indexes them all the same way.

The first INTERIOR entry sits at ``pad_<axis>_lo``. Physical coordinates are
measured from that first interior NODE, the convention
``coords_from_uniform_grid`` and ``coords_from_nonuniform_grid`` both use::

    node_of(axis, pad_lo) == 0.0
    index_of(axis, node_of(axis, i)) == i

and ``node_of`` equals the corresponding entry of the coordinate spine. On an
axis with no pad (``pad_lo == 0``) ``node_of(axis, i)`` is the bare closed
form ``i * dx``.

The round trip holds where both halves are defined, and that range differs
between the two classes. ``Grid.index_of`` addresses the whole padded axis,
so the identity holds for every ``i`` in ``[0, n)``. The graded class
resolves a coordinate against the INTERIOR edge list, so it holds for ``i``
in ``[pad_lo, n - pad_hi)`` -- a node inside an absorber pad has no interior
edge to match. Outside that range both classes raise rather than answer;
neither clamps. (The legacy ``position_to_index`` does clamp; 0b retires it.)

The LAST entry of ``cells(axis)`` is a node-provider, not physical extent.
``NonUniformGrid`` appends one duplicate boundary cell so ``N`` cells are
bounded by ``N+1`` E nodes (#562, ``_append_bounding_node``); the uniform
``Grid`` allocates the same extra node through its ``+1`` fence-post term.
``sum(cells(axis))`` therefore OVERSHOOTS the realized wall-to-wall extent by
that one cell on both classes. Read extents from ``node_of``, or from
``rfx.nonuniform.interior_cells``, never from the sum.

The two metrics, and the end entries
------------------------------------
``d = cells(axis)`` are the PRIMAL cell widths (length ``N``).
``dual = duals(axis)`` are the DUAL spacings an E node's material acts over::

    dual[0] = d[0]
    dual[k] = (d[k-1] + d[k]) / 2            for k >= 1

The two inverse-metric arrays the stencil carries are derived from those,
with one end entry each overridden (``_profile_to_inv_arrays``, CORE-C2)::

    inv_d_h = 1 / d        except   inv_d_h[N-1] = 0
    inv_d_e = 1 / dual     except   inv_d_e[0]   = 1 / d[0]

``inv_d_h[N-1] = 0`` is the forward difference having no ``d[N]`` to reach;
``inv_d_e[0] = 1/d[0]`` is the backward-difference boundary, and it is the
same number ``1/dual[0]`` would give because ``dual[0] = d[0]``.

Bit-for-bit, the reproducing spelling is ``_profile_to_inv_arrays(cells)``
and NOT ``1/duals``: the inverse arrays are built by ``2/(d[k-1]+d[k])`` in
float32, while ``duals`` is ``(d[k-1]+d[k])/2`` in float64 and a float32
reciprocal of it need not land on the same bits. ``duals`` is spelled as the
mean rather than as ``1/inv_d_e`` so a constant axis returns the cell size
exactly (``0.5*(d+d) == d`` in floating point; ``1/(2/(d+d))`` need not be).

Who is entitled to which metric
-------------------------------
Every row was read at the site named, not taken from the design note.

=========================  =========================================  ==================================
consumer                   site                                       metric
=========================  =========================================  ==================================
H update                   ``core/yee.update_h_nu`` via ``inv_d*_h``  PRIMAL on the difference axis
E update                   ``core/yee.update_e_nu`` via ``inv_d*``    DUAL on the difference axis
current-source ``dV``      ``nonuniform.make_current_source`` (#672)  PRIMAL on the component's own
                                                                      axis, DUAL on the two transverse
lumped port sigma          ``sources.sources.port_sigma`` (#691)      same mixed rule
                                                                      (``port_d_parallel`` primal,
                                                                      ``port_dual_transverse`` dual)
lumped R / C fold          ``lumped.setup_rlc_materials`` and         same mixed rule
                           ``setup_rlc_materials_traced``
surface-impedance sheet    ``materials.thin_conductor``               DUAL along the sheet NORMAL
                           ``.SheetImpedanceSpec`` (#677)             (``sigma_sheet = G / d_dual``)
node positions             ``geometry.rasterize_grid``                PRIMAL cumulative sum; a
                           ``._axis_node_positions`` (#562, #807)     constant axis takes the closed
                                                                      form ``(i - pad) * dx``
CPML sigma / kappa         ``boundaries.cpml._grid_spacings`` and     x and y: the ONE scalar each,
                           the ``CPMLAxisParams`` built beside it     on BOTH faces. z: the cell
                                                                      array's two ends (G11)
waveguide-port injection   ``nonuniform``                            PRIMAL at the H plane, DUAL at
                           ``._waveguide_port_axis_metrics``, used    the E plane, on the port's own
                           by ``run_nonuniform``'s H/E apply steps    propagation axis
=========================  =========================================  ==================================

The CPML row is the one open defect left in this table; it is listed as what
the code does now, not as what it is entitled to.

The waveguide-port row moved in 0b: each half of the port's one-sided TFSF
boundary now reads the plane it acts on (G13). One site of that gap is NOT
closed and #810 should not be recorded as closed on its account: the
half-cell reference shift stores the boundary scalar as ``cfg.dx``
(``sources.waveguide_port``, at config build), so a port in a refined band
shifts by half the boundary cell where its own half-cell is smaller. On the
WR-90 band that is 2.16, 3.55 and 4.74 degrees of phase at 8.2, 10.3 and
12.4 GHz. It does not reach a normalized S-parameter, and it is untouched
here.

The CPML row is the one worth reading twice, because the gap is narrower than
"one scalar everywhere". Four of the six per-face slots are filled from two
numbers: ``dx_x_lo`` and ``dx_x_hi`` both take ``grid.dx``, ``dx_y_lo`` and
``dx_y_hi`` both take ``grid.dy``. The z pair does NOT: ``dz_lo`` and
``dz_hi`` are read from the first and last entries of the cell array, so the
z faces are already calibrated on the cells that sit under them. G11 is
therefore about x and y, and 0b fills those four slots from
``boundary_cell(axis, side)``.

Two places where a site differs from the design note's one-line summary of
it. Recorded, not changed (0a moves no consumer):

* The note's fact table says "source ``dV`` and RLC the primal". Both sites
  are MIXED, not primal: primal on the component's own axis and DUAL on the
  two transverse axes (``nonuniform.make_current_source``, #672;
  ``lumped.setup_rlc_materials``, #691). The primal half of that sentence is the
  own-axis half.
* Two consumers reach the dual rule through the float32 store widened back
  to float64 (``sources.sources._nu_cell_arrays``,
  ``sources.msl_port._axis_dual_size``) rather than through the float64 spine
  ``cells`` returns. On a graded fixture the two disagree at a few nodes in
  the last float32 bits; see the test module for the measured figure.
"""

from __future__ import annotations

import numpy as np

#: Canonical axis order. Index into this for the integer spelling.
AXES = ("x", "y", "z")

#: How close the realized interior cells must sum to the declared profile
#: (metres). ``make_band_profile`` already guarantees its own edges to 1e-12 m
#: by float64 accumulation, so the grid re-check is held to the same figure.
DECLARED_SPAN_TOL_M = 1e-12

#: A span overlapping a cell by no more than this fraction of the axis's
#: smallest cell does not cross it. A plane declared on a node lands a few
#: 1e-18 m to either side of the cumulative sum a mesher or the grid's
#: coordinate spine gives for that node, and without the slack it drags in
#: the cell on the far side.
NODE_TOUCH_REL = 1e-3

#: Two cells are one size when they agree to this relative tolerance. A
#: profile built as ``np.diff`` of node coordinates carries 1e-14-relative
#: noise between cells that are meant to be equal.
SAME_CELL_RTOL = 1e-9


def normalize_axis(axis) -> int:
    """``"x"``/``"y"``/``"z"`` or ``0``/``1``/``2`` to an axis index.

    Rejects everything else, including ``None`` and a bare bool. There is no
    default axis: a consumer that does not know which axis it is asking about
    is the bug this interface exists to remove.
    """
    if isinstance(axis, str):
        key = axis.lower()
        if key in AXES:
            return AXES.index(key)
        raise ValueError(
            f"axis must be one of {AXES} or 0/1/2, got {axis!r}"
        )
    if isinstance(axis, bool):
        raise ValueError(f"axis must be one of {AXES} or 0/1/2, got {axis!r}")
    if isinstance(axis, (int, np.integer)):
        if 0 <= int(axis) <= 2:
            return int(axis)
        raise ValueError(
            f"axis index must be 0, 1 or 2, got {int(axis)}"
        )
    raise ValueError(f"axis must be one of {AXES} or 0/1/2, got {axis!r}")


def axis_name(axis) -> str:
    """The ``"x"``/``"y"``/``"z"`` spelling of ``axis``, for messages."""
    return AXES[normalize_axis(axis)]


def dual_spacings_from_cells(cells: np.ndarray) -> np.ndarray:
    """``dual[0] = d[0]``, ``dual[k] = (d[k-1]+d[k])/2`` on the host.

    Same rule as ``rfx.nonuniform.e_node_dual_spacings``, kept in float64 and
    in numpy: this is the host-side accessor spelling, the other is the
    solver-facing one that rounds to float32 through ``jnp``. Written as the
    mean rather than as ``1/inv_d_e`` so a constant axis returns the cell
    size bit-exactly.
    """
    d = np.asarray(cells, dtype=np.float64)
    if d.size == 0:
        return d.copy()
    dual = np.empty_like(d)
    dual[0] = d[0]
    if d.size > 1:
        dual[1:] = 0.5 * (d[:-1] + d[1:])
    return dual


def cells_crossed(cells, lo: float, hi: float) -> np.ndarray:
    """The interior cells a span ``[lo, hi]`` crosses, in order.

    ``cells`` are interior cell widths with the first interior node at 0. A
    cell counts as crossed when the span overlaps it by more than
    ``NODE_TOUCH_REL`` of the smallest cell, so a span that starts or ends
    on a node does not reach the cell beyond it. The S-parameter
    reference-span check and the preflight checks both read this one rule.
    """
    d = np.asarray(cells, dtype=np.float64)
    if d.size == 0:
        return d.copy()
    edges = np.insert(np.cumsum(d), 0, 0.0)
    tol = NODE_TOUCH_REL * float(np.min(d))
    crossed = (edges[:-1] < float(hi) - tol) & (edges[1:] > float(lo) + tol)
    return d[crossed]


def distinct_cell_sizes(sizes) -> list[float]:
    """The different cell sizes among ``sizes``, first occurrence first,
    two sizes being the same when they agree to ``SAME_CELL_RTOL``."""
    distinct: list[float] = []
    for v in np.asarray(sizes, dtype=np.float64).ravel():
        if not any(np.isclose(v, d, rtol=SAME_CELL_RTOL, atol=0.0)
                   for d in distinct):
            distinct.append(float(v))
    return distinct


def is_one_cell_size(sizes) -> bool:
    """``len(distinct_cell_sizes(sizes)) <= 1`` without the per-size loop.

    Every size is compared with the first, as that function does. Its loop
    grows with cells times distinct sizes: 18 s for a 2000-cell profile
    whose cells all differ, on every preflight of such a model.
    """
    s = np.asarray(sizes, dtype=np.float64).ravel()
    if s.size == 0:
        return True
    return bool(np.all(np.isclose(s, s[0], rtol=SAME_CELL_RTOL, atol=0.0)))
