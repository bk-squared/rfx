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
    index_of(axis, node_of(axis, i)) == i        (concrete axes)

so ``node_of`` and ``index_of`` are inverses and ``node_of`` equals the
corresponding entry of the coordinate spine. On an axis with no pad
(``pad_lo == 0``) ``node_of(axis, i)`` is the bare closed form ``i * dx``.

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

===========================  ==========================================  =================================
consumer                     site                                        metric
===========================  ==========================================  =================================
H update (``update_h_nu``)   ``rfx/core/yee.py`` via ``inv_d*_h``        PRIMAL on the difference axis
E update (``update_e_nu``)   ``rfx/core/yee.py`` via ``inv_d*``          DUAL on the difference axis
current-source ``dV``        ``rfx/nonuniform.py:1369-1403`` (#672)      PRIMAL on the component's own
                                                                         axis, DUAL on the two transverse
lumped port sigma            ``rfx/sources/sources.py:136-153`` (#691)   same mixed rule
                                                                         (``port_d_parallel`` primal,
                                                                         ``port_dual_transverse`` dual)
lumped R / C material fold   ``rfx/lumped.py:287-298, 406-422``          same mixed rule
surface-impedance sheet      ``rfx/materials/thin_conductor.py:374-396`` DUAL along the sheet NORMAL
                             (#677)                                      (``sigma_sheet = G / d_dual``)
node positions, rasterizing  ``rfx/geometry/rasterize_grid.py:102-125``  PRIMAL cumulative sum; a
                             (#562, #807)                                 constant axis takes the closed
                                                                         form ``(i - pad) * dx``
CPML sigma / kappa           ``rfx/boundaries/cpml.py:192-206, 505-510`` PRIMAL boundary cell -- ONE
                                                                         scalar filling all six per-face
                                                                         slots today (G11)
waveguide-port injection     ``rfx/nonuniform.py:2091, 2134, 2202``      scalar ``grid.dx`` today (G13)
===========================  ==========================================  =================================

The last two rows are the open defects step 0b closes; they are listed as
what the code does now, not as what it is entitled to.

Two places where a site differs from the design note's one-line summary of
it. Recorded, not changed (0a moves no consumer):

* The note's fact table says "source ``dV`` and RLC the primal". Both sites
  are MIXED, not primal: primal on the component's own axis and DUAL on the
  two transverse axes (``rfx/nonuniform.py:1369-1403``, #672;
  ``rfx/lumped.py:287-298``, #691). The primal half of that sentence is the
  own-axis half.
* Two consumers reach the dual rule through the float32 store widened back
  to float64 (``rfx/sources/sources.py:31-41``,
  ``rfx/sources/msl_port.py:446-453``) rather than through the float64 spine
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
