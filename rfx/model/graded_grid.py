"""The realized graded (non-uniform) Yee grid.

The grid record, its per-axis cell profiles and inverse spacings, the
time-step pin against the realized Courant limit, and the position-to-index
lookups. Moved verbatim from ``rfx/nonuniform.py``, which re-exports every
name here.
"""

from __future__ import annotations

from typing import NamedTuple

from rfx.boundaries.depths import resolve_face_depths

import jax
import jax.numpy as jnp

import numpy as np

from rfx.core.yee import (
    EPS_0, MU_0,
)
from rfx.core.jax_utils import is_tracer
from rfx._grid_metric import (
    DECLARED_SPAN_TOL_M,
    Float32WidthResolutionWarning,
    NODE_TOUCH_REL,
    axis_name as _axis_name,
    dual_spacings_from_cells as _dual_spacings_from_cells,
    nearest_node_index,
    normalize_axis as _normalize_axis,
)

C0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))


class NonUniformGrid(NamedTuple):
    """Non-uniform grid with per-axis cell-size arrays.

    ``dx`` / ``dy`` are the BOUNDARY cell sizes (used by CPML and any
    legacy code that reads a scalar spacing); ``dx_arr`` / ``dy_arr`` /
    ``dz`` hold the per-cell spacings. In the uniform-xy case,
    ``dx_arr`` is ``jnp.full(nx, dx)`` and ``dy_arr`` is analogous.

    ``pad_{axis}_{lo|hi}`` mirror ``rfx.grid.Grid`` per-face padding —
    a face whose token is ``pmc``/``pec``/``periodic`` gets 0 cells on
    that side even when the axis as a whole uses CPML. For back-compat
    the six fields default to ``cpml_layers`` on every face (the
    pre-per-face-allocation symmetric layout).
    """
    nx: int
    ny: int
    nz: int
    dx: float              # BOUNDARY x cell size (CPML + legacy scalars)
    dy: float              # BOUNDARY y cell size
    dx_arr: jnp.ndarray    # (nx,) — per-cell dx (includes CPML padding)
    dy_arr: jnp.ndarray    # (ny,) — per-cell dy
    dz: jnp.ndarray        # (nz,) z cell sizes (already per-cell)
    dt: float | jax.Array  # timestep; eager float in normal use, traced
                           # jax.Array on the run_nonuniform traced path.
                           # Readers must not apply float() / .item() without
                           # an is_tracer(dt) guard or a host-boundary context.
    cpml_layers: int
    # Non-periodic cell-size arrays carry pads and a trailing duplicate so N
    # cells are bounded by N+1 E-nodes (#562, see _append_bounding_node). The
    # duplicate is a node-provider, not physical extent: its H term is the one
    # the stencil zeroes. Read extents from node positions
    # (coords_from_nonuniform_grid, or cumsum edges up to index nx-1) — NEVER
    # from sum(dx_arr), which overshoots by that trailing cell.
    # Periodic axes have N cells and N nodes, no pads or duplicate, and both
    # inverse metrics are 1/d at every node, including the seam.
    # Pre-computed inverse spacing arrays (length N, padded).
    # CORE-C2: inv_d* feed the E update (mean spacing), inv_d*_h feed
    # the H update (local cell width). See _profile_to_inv_arrays.
    inv_dx: jnp.ndarray    # (nx,) — E update: 2/(dx[i-1]+dx[i]); [0]=1/dx[0]
    inv_dy: jnp.ndarray    # (ny,) — E update: 2/(dy[j-1]+dy[j]); [0]=1/dy[0]
    inv_dz: jnp.ndarray    # (nz,) — E update: 2/(dz[k-1]+dz[k]); [0]=1/dz[0]
    inv_dx_h: jnp.ndarray  # (nx,) — H update: 1/dx[i]; [nx-1]=0
    inv_dy_h: jnp.ndarray  # (ny,) — H update: 1/dy[j]; [ny-1]=0
    inv_dz_h: jnp.ndarray  # (nz,) — H update: 1/dz[k]; [nz-1]=0
    # Per-face CPML padding (PMC+CPML composition fix, 2026-04)
    pad_x_lo: int = 0
    pad_x_hi: int = 0
    pad_y_lo: int = 0
    pad_y_hi: int = 0
    pad_z_lo: int = 0
    pad_z_hi: int = 0
    # Exact float64 copies of the padded cell-size profiles (#802/#807).
    # The node-position spine (coords_from_nonuniform_grid) reads these, so
    # realized geometry does not inherit the float32 quantization of the
    # solver-facing stores above — a cumsum of f32-widened cell sizes lands
    # ~1e-10 m off the exact node positions, which flips half-open inclusion
    # at node-aligned faces and splits the one-cell-sheet plane between the
    # uniform and NU lanes. Populated by make_nonuniform_grid on the
    # concrete path only (None when the profile is traced, and on grids
    # built by hand — the coordinate producer then widens the f32 store,
    # the pre-#802 values, and emits ExactNodeSpineMissingWarning so the
    # degraded node line is never silent). Solver arithmetic keeps reading
    # the float32 fields; both derive from the same padded profile at
    # construction, so they cannot drift independently.
    dx_arr_f64: np.ndarray | None = None
    dy_arr_f64: np.ndarray | None = None
    dz_f64: np.ndarray | None = None
    face_layers: dict[str, int] | None = None
    # The caller's concrete nominal interior widths of each TRACED axis
    # (``Simulation(nominal_profiles=)``), float64 numpy keyed "x"/"y"/"z",
    # or None. Read only by ``_axis_position_to_index`` to resolve a metre
    # coordinate on a traced axis; never part of the node spine above.
    nominal_cells: dict[str, np.ndarray] | None = None
    traced_cells: dict[str, tuple[int, int]] | None = None
    periodic_axes: str = ""
    domain: tuple[float, float, float] | None = None

    @property
    def shape(self):
        """Grid shape (nx, ny, nz) — duck-typing compatible with Grid."""
        return (self.nx, self.ny, self.nz)

    @property
    def axis_pads(self):
        """Leading (``lo``) per-axis pad — the number coordinate-offset
        callers subtract from array indices to recover user coords."""
        return (self.pad_x_lo, self.pad_y_lo, self.pad_z_lo)

    # ------------------------------------------------------------------
    # Grid metric interface (step 0a, design note
    # docs/design_notes/20260922_nu_grid_core_predeclaration.md decision 2).
    #
    # THE CONVENTION -- index origin, the primal/dual rule, the end entries
    # of ``inv_d_e`` / ``inv_d_h``, and which consumer is entitled to which
    # metric -- is written once, in ``rfx/_grid_metric.py``. Read that module
    # before adding a metric formula at a call site.
    #
    # ``rfx.grid.Grid`` carries the same seven methods with the same meaning,
    # so a consumer can hold either class and never branch on its type. 0a
    # adds the interface only: no consumer moves onto it here, and the scalar
    # ``dx`` / ``dy`` fields keep their present meaning (the BOUNDARY cell).
    #
    # Source of truth (decision 1): the float64 cell arrays
    # ``dx_arr_f64`` / ``dy_arr_f64`` / ``dz_f64``. ``make_nonuniform_grid``
    # fills all three for every CONCRETE axis, and re-checks that the cells
    # it realized between the declared domain faces sum to the declared
    # profile. Nodes, duals and both inverse-metric arrays are derived from
    # the cells; none of them is stored independently.
    # ------------------------------------------------------------------

    def _axis_layout(self, axis) -> tuple[int, int, int, int]:
        """``(axis_index, n, pad_lo, pad_hi)`` for a normalized axis."""
        ax = _normalize_axis(axis)
        n = (self.nx, self.ny, self.nz)[ax]
        pad_lo = (self.pad_x_lo, self.pad_y_lo, self.pad_z_lo)[ax]
        pad_hi = (self.pad_x_hi, self.pad_y_hi, self.pad_z_hi)[ax]
        return ax, int(n), int(pad_lo), int(pad_hi)

    def _axis_store(self, axis):
        """The solver-facing float32 cell store for ``axis``."""
        return (self.dx_arr, self.dy_arr, self.dz)[_normalize_axis(axis)]

    def _axis_exact(self, axis):
        """The float64 cell spine for ``axis``, or ``None`` if absent."""
        return (self.dx_arr_f64, self.dy_arr_f64,
                self.dz_f64)[_normalize_axis(axis)]

    def is_traced(self, axis) -> bool:
        """Whether ``axis``'s cell widths are a JAX tracer (decision 6).

        True on the mesh-as-design-variable path, where the profile is a
        trace leaf and has no host copy. A traced axis still steps the solver
        and still differentiates; what it cannot do is answer a host-side
        structural question, which is why ``index_of`` refuses on it.
        """
        return bool(is_tracer(self._axis_store(axis)))

    def cells(self, axis) -> np.ndarray:
        """PRIMAL cell widths along ``axis``, full padded length ``n``.

        float64 on the host for a concrete axis -- the float64 spine, which
        is the source of truth (decision 1), NOT the float32 solver store.
        A traced axis returns its tracer unchanged, in jnp.

        On a non-periodic axis the last entry is the duplicate (#562), supplying
        ``N+1``-th E node and carries no physical extent, so ``sum(cells)``
        overshoots the realized wall-to-wall extent by one cell. Read extents
        from ``node_of`` or ``interior_cells``. A periodic axis has no duplicate:
        its N widths sum to the declared period and all N cells are stepped.

        A spine whose length disagrees with the axis is REFUSED rather than
        returned: the spine and the float32 store are two views of one padded
        profile, so a length mismatch means they are not, and every node
        derived from it would sit at the wrong index. A MISSING spine is a
        different case, handled below.

        A grid built by hand (the ``NonUniformGrid(...)`` constructor called
        directly, rather than ``make_nonuniform_grid``) may carry no spine.
        It then gets the float32 store widened back to float64 AND an
        ``ExactNodeSpineMissingWarning``, the same degraded line and the same
        warning ``coords_from_nonuniform_grid`` emits -- silently falling
        back is the #802 realization with no trace of why.
        """
        ax, n, _pad_lo, _pad_hi = self._axis_layout(axis)
        store = self._axis_store(ax)
        if is_tracer(store):
            return store
        exact = self._axis_exact(ax)
        if exact is not None and np.asarray(exact).shape != (n,):
            raise ValueError(
                f"axis {_axis_name(ax)!r} declares {n} cells but its float64 "
                f"spine holds {np.asarray(exact).shape}. The spine and the "
                "solver's float32 store are two views of one padded profile; "
                "a length that disagrees with the axis means they are not, "
                "and every node position derived from the spine would sit at "
                "the wrong index. Rebuild the grid with make_nonuniform_grid."
            )
        if exact is None or np.asarray(exact).dtype != np.float64:
            import warnings
            from rfx.geometry.rasterize_grid import (
                ExactNodeSpineMissingWarning,
            )
            warnings.warn(
                f"NonUniformGrid axis {_axis_name(ax)!r} has no float64 "
                "cell spine; cells() widened the float32 store instead. Node "
                "positions derived from it carry the pre-#802 quantization "
                "(~1e-10 m), which flips half-open inclusion at node-aligned "
                "faces. Build the grid with make_nonuniform_grid.",
                ExactNodeSpineMissingWarning, stacklevel=2)
            return np.asarray(store, dtype=np.float64)
        return np.asarray(exact, dtype=np.float64)

    def duals(self, axis) -> np.ndarray:
        """DUAL (E-node) spacings along ``axis``.

        ``dual[0] = d[0]``, ``dual[k] = (d[k-1]+d[k])/2`` -- the control
        volume an E node's material acts over, and the metric the NU E update
        divides its curl by. float64 for a concrete axis; a traced axis goes
        through ``e_node_dual_spacings`` so the gradient survives.

        Reproduces ``e_node_dual_spacing_at(grid.cells(axis), k)`` for every
        ``k`` bit for bit. Against the float32 solver store the two spellings
        agree only to float32 roundoff -- that gap is the store's, not the
        rule's, and it is why the accessor reads the spine.
        """
        d = self.cells(axis)
        if is_tracer(d):
            return e_node_dual_spacings(d)
        return _dual_spacings_from_cells(d)

    def index_of(self, axis, x: float) -> int:
        """Padded index of the node nearest physical coordinate ``x``.

        Refuses on a traced axis (decision 6, closes G9). The existing
        ``position_to_index`` answers there by substituting a NOMINAL uniform
        mesh of ``fallback_dx`` cells and returns an index for a mesh the
        solver is not running -- silently the wrong cell whenever the traced
        profile is graded. That fallback stays in ``position_to_index`` for
        0a because consumers still call it; 0b retires it as each consumer
        moves onto this accessor.

        Which node (#1295, #1342): the nearest one by float64 distance, and
        at a tie (two distances agreeing to 1e-9 of the local cell) the
        LOWER one -- ``nearest_node_index``, the function the sheet snap
        itself calls, on the node line the conductors are rasterized on
        (``_axis_node_positions`` of the float64 spine: the closed form
        ``(i - pad_lo) * cell`` on a constant axis, the cumulative sum on a
        graded one). A port, source or probe and a sheet declared at the
        same coordinate therefore land on the same node, bit for bit.
        ``position_to_index`` and ``pos_to_nu_index`` resolve through the
        same function. The uniform ``Grid`` still rounds an exact tie to the
        even node until #1342.

        A coordinate outside the interior span is REFUSED, as on the uniform
        grid. ``_axis_position_to_index`` clamps instead: ``argmin`` over the
        edge list always returns some index, so asking for a position a metre
        outside a 6 mm domain silently gives the last face and the caller
        stamps its material there. That legacy clamp stays where it is for
        0a -- consumers still call ``position_to_index`` -- and 0b retires it
        as each consumer moves onto this accessor.

        The bound is the interior, ``[node_of(pad_lo), node_of(last interior
        node)]``, because that is the range the edge list can represent; a
        coordinate inside an absorber pad has no interior node to name. The
        comparison carries ``DECLARED_SPAN_TOL_M`` (1e-12 m) of slack at
        each end. That is not a fudge factor: it is the node tolerance the
        mesh builder itself claims. ``make_band_profile`` guarantees that
        every declared interface lands on a cumulative node coordinate to
        1e-12 m and that the cells sum to the declared span to the same
        figure, and ``assert_cells_span_declared_profile`` holds the
        constructor to it. A grid that only places a node to 1e-12 m cannot
        tell a coordinate 1e-13 m past the face from the face, so refusing
        one would be refusing a distinction the grid does not make. Outside
        that band the refusal is exact.

        That figure grows with the column: the builder documents 1e-11 m on
        a 10 m column of 1e5 cells. On a board or a waveguide, the sizes
        this lane meshes, 1e-12 m is the right order. A caller meshing
        metres should read the builder's guarantee before trusting the
        boundary case.
        """
        ax, _n, pad_lo, pad_hi = self._axis_layout(axis)
        if self.is_traced(ax):
            raise ValueError(
                f"index_of: axis {_axis_name(ax)!r} carries a TRACED cell "
                "profile (mesh-as-design-variable), so a physical coordinate "
                "cannot be resolved to an index on the host. The nominal "
                "uniform fallback the old position_to_index used returns an "
                "index for a mesh the solver is not running (gap G9). "
                "Resolve structural positions before tracing the mesh, or "
                "pass the index directly."
            )
        if _axis_name(ax) in self.periodic_axes:
            return _periodic_index(self, ax, float(x))
        nodes, cells = self._node_line(ax)
        pos = float(x)
        lo, hi = float(nodes[0]), float(nodes[-1])
        if not (lo - DECLARED_SPAN_TOL_M <= pos <= hi + DECLARED_SPAN_TOL_M):
            raise ValueError(
                f"position {pos} on axis {_axis_name(ax)!r} lies outside "
                f"this axis's interior span [{lo}, {hi}] m. Check the "
                f"coordinate lies inside the simulation domain."
            )
        return _nearest_on_line(nodes, cells, pos) + pad_lo

    def _node_line(self, axis):
        """``(nodes, cells)`` a coordinate on a CONCRETE ``axis`` resolves on.

        ``nodes`` are the INTERIOR entries of ``_axis_node_positions`` on the
        float64 spine -- the node coordinates ``coords_from_nonuniform_grid``
        gives the rasterizer, first interior node at 0 -- and ``cells`` the
        interior cells between them.
        """
        from rfx.geometry.rasterize_grid import _axis_node_positions
        ax, n, pad_lo, pad_hi = self._axis_layout(axis)
        d = self.cells(ax)
        nodes = _axis_node_positions(d, pad_lo)[pad_lo:n - pad_hi]
        return nodes, interior_cells(d, pad_lo, pad_hi, periodic=_axis_name(ax) in self.periodic_axes)

    def node_of(self, axis, i: int):
        """Physical coordinate of the E node at padded index ``i``.

        Origin at the first interior node, so ``node_of(axis, pad_lo) ==
        0.0``. Derived from the cells by cumulative sum, except that a
        CONSTANT axis takes the closed form ``(i - pad_lo) * d[0]`` -- which
        is what keeps a uniform-valued NU axis bit-identical to the uniform
        lane's own node line (#807). Both branches are
        ``_axis_node_positions``, so this equals
        ``coords_from_nonuniform_grid(grid).<axis>[i]`` bit for bit.

        A traced axis returns a traced scalar (jnp cumulative sum), the same
        arithmetic ``coords_from_nonuniform_grid`` runs in-trace.

        Cost: this builds the whole axis to return one entry, because reusing
        the spine producer is what makes the answer bit-identical to the
        coordinate arrays. Nothing calls it in a loop in 0a. A consumer in 0b
        that needs many nodes asks ``coords_from_nonuniform_grid`` once, or
        this grows a cached array -- not a second spelling of the arithmetic.
        """
        ax, n, pad_lo, _pad_hi = self._axis_layout(axis)
        idx = int(i)
        if not (0 <= idx < n):
            raise IndexError(
                f"padded index {idx} is outside axis {_axis_name(ax)!r}, "
                f"which has {n} entries"
            )
        d = self.cells(ax)
        if is_tracer(d):
            d_j = jnp.asarray(d)
            cum = jnp.concatenate(
                [jnp.zeros((1,), dtype=d_j.dtype), jnp.cumsum(d_j)])
            return cum[idx] - cum[pad_lo]
        from rfx.geometry.rasterize_grid import _axis_node_positions
        return float(_axis_node_positions(d, pad_lo)[idx])

    def boundary_cell(self, axis, side: str):
        """The cell width at the ``"lo"`` or ``"hi"`` face of ``axis``.

        The number a CPML profile is entitled to calibrate against, per face.
        ``make_nonuniform_grid`` pins the first and last interior cells of x
        and y to the scalar ``dx`` / ``dy`` and pads each face with copies of
        them, so on a built grid both sides equal that scalar; z has no
        scalar and this is the only way to ask.

        ``cpml.py`` fills its six per-face slots from this accessor.
        """
        if side not in ("lo", "hi"):
            raise ValueError(f"side must be 'lo' or 'hi', got {side!r}")
        d = self.cells(axis)
        entry = d[0] if side == "lo" else d[-1]
        return entry if is_tracer(d) else float(entry)

    def is_constant(self, axis) -> bool:
        """Whether every cell on ``axis`` has the same width.

        The predicate decision 4 selects the uniform kernel on, instead of
        "was a profile given" (``rfx/api/_mesh.py:152-156``).

        A TRACED axis returns ``False``. Constancy is not decidable on the
        host there, and ``False`` is the safe direction: it routes the axis
        to the non-uniform kernel, which is correct for a constant mesh too,
        whereas ``True`` would hand a possibly graded mesh to a kernel that
        folds one scalar.
        """
        d = self._axis_store(axis)
        if is_tracer(d):
            return False
        arr = self.cells(axis)
        return bool(arr.size and np.all(arr == arr[0]))





def _pad_profile(profile, pad_lo: int, pad_hi: int | None = None):
    """Pad a 1-D cell-size profile with CPML cells on each end.

    ``pad_lo`` and ``pad_hi`` may differ (per-face allocation, 2026-04 —
    a PMC/PEC face gets 0 cells on that side while the opposing CPML
    face keeps its allocation). If ``pad_hi`` is omitted the symmetric
    ``pad_lo`` count is used on both ends (pre-per-face-allocation behaviour).

    CPML uses constant spacing matching the boundary cell size, so the
    ``pad_lo`` cells on the leading side carry ``profile[0]`` and the
    ``pad_hi`` cells on the trailing side carry ``profile[-1]``.

    When ``profile`` is a JAX tracer the padding stays in-trace (needed
    for ``jax.grad`` w.r.t. ``dz_profile`` — mesh-as-design-variable).
    Otherwise the numpy path is used, preserving the Python-float ``dt``
    that Simulation-level callers depend on.
    """
    if pad_hi is None:
        pad_hi = pad_lo
    if is_tracer(profile):
        prof = jnp.asarray(profile, dtype=jnp.float32)
        lo_pad = jnp.full(pad_lo, prof[0])
        hi_pad = jnp.full(pad_hi, prof[-1])
        return jnp.concatenate([lo_pad, prof, hi_pad])
    lo_pad = np.full(pad_lo, float(profile[0]))
    hi_pad = np.full(pad_hi, float(profile[-1]))
    return np.concatenate([lo_pad, np.asarray(profile, dtype=np.float64), hi_pad])


def _append_bounding_node(profile_full, *, periodic=False):
    """Append one duplicate boundary cell so an N-cell profile yields N+1
    E-nodes — the node count the uniform ``Grid`` allocates for N cells.

    N cells are bounded by N+1 nodes, and the E-nodes this grid steps sit on
    cell EDGES (``inv_d_e[i] = 2/(d[i-1]+d[i])`` is the dual spacing of an
    edge-centred node). Without this the array carried only N nodes, the
    stencil zeroed the last cell's H term (``inv_d_h[N-1] = 0``), and the
    realized wall-to-wall extent came out ``sum(d) - d[-1]`` — one cell
    short of what the caller asked for (#562: +2.47 % of TM110 on a
    45 x 39-cell PEC box, +37 MHz of WR-90 TE101 centre frequency from the
    guide-width axis alone).

    Duplicating the boundary cell is what makes the far node's coefficient
    the mirror of the near one's: ``inv_d_e[N] = 2/(d[N-1]+d[N])``, which is
    ``1/d[N-1]`` exactly when ``d[N] == d[N-1]``, matching the existing
    ``inv_d_e[0] = 1/d[0]`` boundary treatment. The appended cell's own H
    term is the one the stencil zeroes, so it adds a node without adding a
    cell of physical extent. Periodic axes already have N nodes for N cells
    and return the profile unchanged.
    """
    if periodic:
        return profile_full
    if is_tracer(profile_full):
        return jnp.concatenate([profile_full, profile_full[-1:]])
    return np.concatenate([profile_full, profile_full[-1:]])


def interior_cells(d_full, pad_lo: int, pad_hi: int, *, periodic=False):
    """The physical interior cells of a padded NU cell-size array.

    The array is ``[lo pad | interior | hi pad | bounding-node duplicate]``
    (#562, see ``_append_bounding_node``), so the interior ends one entry
    before ``len - pad_hi``. Slicing without that ``-1`` pulls in a pad cell
    — or the duplicate itself on a face with no pad — and overstates the
    extent by one cell, which is the same class of error #562 was.
    A periodic axis has neither pads nor duplicate, so every entry is interior.

    Returns the CELLS; ``np.insert(np.cumsum(cells), 0, 0.0)`` then gives the
    ``N+1`` interior NODE positions, the last of which lands exactly on the
    requested domain face.
    """
    return d_full[pad_lo : len(d_full) - pad_hi - int(not periodic)]


def assert_cells_span_declared_profile(d_full, pad_lo, pad_hi, profile,
                                       axis: str = "?", *, periodic=False) -> None:
    """Refuse a padded cell array whose interior does not add up to the
    profile that was declared (step 0a, decision 1).

    The realized interior of a padded array is
    ``[lo pad | interior | hi pad | bounding-node duplicate]``, so the cells
    that carry physical extent are ``interior_cells(d_full, pad_lo, pad_hi)``
    and nothing else. Their count must equal the declared profile's, and
    their sum must equal the declared profile's sum to
    ``DECLARED_SPAN_TOL_M`` (1e-12 m).

    That tolerance is the one ``make_band_profile`` already meets between any
    two DECLARED LINES it was given: it returns float64 cells whose running
    sum lands on every declared edge to 1e-12 m, and whose total equals
    ``edges[-1] - edges[0]`` to the same figure. This grid-side check is the
    re-check of that guarantee after padding and the bounding node have been
    applied -- the step where the #562 class of error lives, and the step the
    band builder cannot see.

    ``make_nonuniform_grid`` takes a flat profile, not a set of declared
    lines, so the per-LINE form of the check has no input here and this is
    the strongest statement the constructor can make. The per-line check
    belongs with whatever owns the lines, i.e. the mesher of step 1: when
    ``GridSpec`` auto-meshing lands it carries the declared edges into the
    grid and re-checks each interval, and that is where the mesher note
    records it. Until then a caller holding declared lines gets the per-line
    guarantee from ``make_band_profile`` itself.

    Raises ``ValueError``; a traced profile is skipped by the caller, since
    neither side of the comparison is a host number there.
    """
    interior = interior_cells(np.asarray(d_full, dtype=np.float64),
                              int(pad_lo), int(pad_hi), periodic=periodic)
    declared = np.asarray(profile, dtype=np.float64)
    if interior.shape[0] != declared.shape[0]:
        raise ValueError(
            f"{axis} axis: the padded cell array realizes "
            f"{interior.shape[0]} interior cells but {declared.shape[0]} "
            "were declared. The interior of a padded array ends one entry "
            "before len - pad_hi (the bounding-node duplicate, #562); a "
            "count that disagrees means the padding or that duplicate was "
            "applied differently than interior_cells reads it."
        )
    miss = float(interior.sum() - declared.sum())
    if abs(miss) > DECLARED_SPAN_TOL_M:
        raise ValueError(
            f"{axis} axis: the realized interior cells span "
            f"{float(interior.sum()):.12e} m but the declared profile spans "
            f"{float(declared.sum()):.12e} m, a miss of {miss:.3e} m "
            f"(tolerance {DECLARED_SPAN_TOL_M:.0e} m). Every declared "
            "interface is placed by the running sum of these cells, so a "
            "miss of one cell puts a material plane one cell away from the "
            "face that declared it."
        )



def node_positions_from_profile(profile):
    """Interior E-node positions for a RAW (unpadded) cell-size profile.

    ``N`` cells give ``N+1`` nodes, the first at 0 and the last exactly on the
    profile's total extent — the same positions
    ``coords_from_nonuniform_grid`` produces for the interior of a built grid.
    Public because callers outside this module legitimately need the
    convention (the #325 graded-rasterization preflight models what the
    rasterizer will do, and modelling it with a private-import composition or,
    worse, a hand-rolled copy, is how that check drifted in the first place — #562
    review F2, #568).
    """
    from rfx.geometry.rasterize_grid import _axis_node_positions
    return _axis_node_positions(_append_bounding_node(profile), 0)


def _profile_to_inv_arrays(profile_full: np.ndarray, *, periodic=False) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(inv_d_e, inv_d_h)`` — the E-update and H-update inverse
    cell-spacing arrays for a padded 1-D cell-size profile ``d``.

    CORE-C2 fix (2026-05-16). The non-uniform Yee stencil needs a
    *different* metric for each update:

    * **H update** — the H-curl differences two E nodes that straddle
      one cell, so it divides by the **local cell width**::

          inv_d_h[k] = 1/d[k]       (k < N-1);   inv_d_h[N-1] = 0

    * **E update** — the E-curl differences two H cell-centres, whose
      separation is the **mean** of the two adjacent cell widths::

          inv_d_e[k] = 2/(d[k-1]+d[k])   (k >= 1);   inv_d_e[0] = 1/d[0]

    On a uniform mesh both collapse to ``1/d`` so the uniform path is
    bit-identical. The pre-fix code had the two swapped (H got the mean,
    E got the local width) — a stencil inconsistency that scaled the
    curl by ``2 d[k]/(d[k]+d[k±1])`` on every graded cell. The boundary
    entries (``inv_d_h[N-1]=0``, ``inv_d_e[0]=1/d[0]``) reproduce the
    prior boundary behaviour exactly, so PEC/CPML faces are untouched.
    On a constant-spacing periodic axis both metrics are 1/d everywhere.

    The return order ``(inv_d_e, inv_d_h)`` matches the caller's
    ``inv_dx, inv_dx_h = _profile_to_inv_arrays(...)`` unpacking: the
    first slot feeds the E update, the second feeds ``update_h_nu``.
    """
    arr = jnp.asarray(profile_full, dtype=jnp.float32)
    inv_local = 1.0 / arr                          # 1/d[k]
    if periodic:
        inv_periodic = jnp.asarray(1.0 / np.asarray(profile_full, dtype=np.float64), dtype=jnp.float32)
        return inv_periodic, inv_periodic
    inv_mean = 2.0 / (arr[:-1] + arr[1:])          # 2/(d[k]+d[k+1])
    # H update: local cell width; trailing 0 (forward-diff has no d[N]).
    inv_d_h = jnp.concatenate([inv_local[:-1], jnp.zeros(1, dtype=jnp.float32)])
    # E update: mean of (d[k-1], d[k]); leading 1/d[0] (backward-diff
    # boundary — reproduces the pre-fix value so the face cell is
    # unchanged).
    inv_d_e = jnp.concatenate([inv_local[:1], inv_mean])
    return inv_d_e, inv_d_h


def e_node_dual_spacings(profile_full):
    """Per-node DUAL spacing — the length an E node's material acts over.

    ``inv_d_e[k] = 2/(d[k-1]+d[k])`` (``inv_d_e[0] = 1/d[0]``) is the metric
    the non-uniform E update divides the curl by, so the control volume of
    the E node at index ``k`` extends half a cell either side::

        dual[k] = (d[k-1] + d[k]) / 2     (k >= 1);   dual[0] = d[0]

    which is exactly ``1/inv_d_e``. An f0 sheet uses this node dual;
    a DC fold uses the occupied cell's primal width ``d[k]`` before the
    E-edge mean, realizing ``sigma_bulk * thickness`` integrated over duals.
    The #669 Leontovich measurement gave ratios 1.2021 on a 0.25/0.50 mm
    transition and 0.6214 on a 1.00/0.25 mm transition against a matched mesh
    (expected 1.000), motivating the node dual for f0 sheets.

    Computed as ``(d[k-1]+d[k])/2`` rather than ``1/inv_d_e`` so a uniform
    profile returns the cell size bit-exactly (``0.5*(d+d) == d`` in
    floating point, while ``1/(2/(d+d))`` need not be). ``jnp`` throughout so
    a traced ``dz_profile`` (mesh-as-design-variable) stays differentiable.
    """
    arr = jnp.asarray(profile_full)
    return jnp.concatenate([arr[:1], 0.5 * (arr[:-1] + arr[1:])])


def e_node_dual_spacing_at(profile_full, k: int):
    """Scalar dual spacing at node ``k`` — same rule as
    :func:`e_node_dual_spacings`, evaluated on the caller's array WITHOUT the
    dtype-normalizing ``jnp.asarray``.

    Two callers need that: a float64 numpy profile must keep float64 so a
    uniform mesh stays bit-identical to the old primal spelling (a float32
    round-trip alone moves the last bits), and a traced profile must stay a
    tracer so the mesh-as-design-variable path keeps its gradient.

    ``tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py`` pins this against
    :func:`e_node_dual_spacings` element-by-element, so the two spellings
    cannot drift.
    """
    if k <= 0:
        return profile_full[0]
    return 0.5 * (profile_full[k - 1] + profile_full[k])


def _pinned_dt(dt_pin, dt_min_cell, profiles, axis_names,
               where="make_nonuniform_grid") -> float:
    """Validate a caller-pinned time step against the REALIZED Courant limit.

    The limit is ``1 / (c0 * sqrt(sum_a 1/d_a_min^2))`` with ``d_a_min`` the
    smallest cell on axis ``a``. Every axis whose profile is concrete
    contributes its realized minimum, measured off the padded profile the
    grid actually carries. A traced axis has no host value at trace time, so
    it contributes ``dt_min_cell`` — the caller's declared floor for that
    axis — and ``dt_min_cell`` is REQUIRED as soon as one axis is traced.

    The step is refused, not clipped: a silently shortened step would run a
    different physics problem from the one the caller pinned, and the whole
    point of pinning is that every member of a deformation family shares one
    step.
    """
    if is_tracer(dt_pin):
        raise ValueError(
            f"{where}(dt=...) must be a concrete float; a traced "
            "dt puts the time step itself into the derivative, which is the "
            "thing pinning it is meant to remove."
        )
    dt_val = float(dt_pin)
    if not np.isfinite(dt_val) or dt_val <= 0.0:
        raise ValueError(
            f"{where}(dt=...) must be finite and positive, got "
            f"{dt_pin!r}."
        )
    traced_axes = [n for n, p in zip(axis_names, profiles) if is_tracer(p)]
    if traced_axes and dt_min_cell is None:
        raise ValueError(
            f"{where}(dt={dt_val:.6e}) with a traced "
            f"{'/'.join(traced_axes)} profile also needs dt_min_cell=: a "
            "tracer carries no host cell size, so the Courant limit of the "
            "realized mesh cannot be measured here. Pass the smallest cell "
            "the traced axes reach anywhere in the family you will run."
        )
    if dt_min_cell is not None:
        if is_tracer(dt_min_cell) or not np.isfinite(float(dt_min_cell)) \
                or float(dt_min_cell) <= 0.0:
            raise ValueError(
                f"{where}(dt_min_cell=...) must be a concrete "
                f"positive float (m), got {dt_min_cell!r}."
            )
    inv_sq = 0.0
    realized = {}
    for name, prof in zip(axis_names, profiles):
        if is_tracer(prof):
            d_min = float(dt_min_cell)
            realized[name] = (d_min, "declared")
        else:
            d_min = float(np.min(np.asarray(prof, dtype=np.float64)))
            realized[name] = (d_min, "realized")
        inv_sq += 1.0 / d_min ** 2
    limit = 1.0 / (C0 * np.sqrt(inv_sq))
    if dt_val > limit:
        detail = ", ".join(
            f"{n} {realized[n][0] * 1e3:.6g} mm ({realized[n][1]})"
            for n in axis_names)
        raise ValueError(
            f"{where}(dt={dt_val:.6e} s) exceeds the Courant "
            f"limit {limit:.6e} s of the smallest cells [{detail}] — the run "
            f"would diverge. Ratio dt/limit = {dt_val / limit:.4f}."
        )
    # A DECLARED floor is a claim about cells nobody has measured yet. Declare
    # it too large and the check above passes on a mesh whose smallest cell is
    # actually below the pinned step: the fields then diverge to NaN with
    # nothing said, while the same profile handed over concretely is refused
    # here. Measure the claim against the realized cells when they arrive.
    for name, prof in zip(axis_names, profiles):
        if is_tracer(prof):
            _assert_declared_floor_at_runtime(prof, float(dt_min_cell), name,
                                              dt_val, limit)
    return dt_val


def _assert_declared_floor_at_runtime(profile_full, declared_min, axis,
                                      dt_val, limit) -> None:
    """Check a traced axis's REALIZED smallest cell against the floor the
    caller declared, at the moment the values exist.

    A tracer carries no value at trace time, so the host-side check in
    :func:`_pinned_dt` can only compare the caller's number with itself. This
    stages a host callback that runs when the mesh does, reads
    ``min(realized cells)`` and raises if it is below the declared floor. The
    message names both numbers and the step they were checked against; under
    ``jax.jit`` it arrives wrapped in the runtime's callback error, which is
    still the message rather than a silent NaN.
    """
    def _check(realized_min):
        realized = float(realized_min)
        if realized < declared_min * (1.0 - 1e-9):
            raise ValueError(
                f"dt_min_cell={declared_min * 1e3:.6g} mm was declared for "
                f"the traced {axis} profile, but the mesh this run built has "
                f"a smallest {axis} cell of {realized * 1e3:.6g} mm. The "
                f"pinned dt={dt_val:.6e} s was checked against the declared "
                f"floor (Courant limit {limit:.6e} s there) and is not "
                f"guaranteed stable on the realized mesh — the fields would "
                "diverge with nothing said. Declare the smallest cell the "
                "deformation family actually reaches, or shorten dt."
            )

    from rfx._traced_check import traced_check
    traced_check("graded mesh", "declared cell floor", _check, jnp.min(jnp.asarray(profile_full)))


def _warn_float32_width_resolution(profile, axis):
    """Check shared interior nodes before padding or solver dtype conversion."""
    if is_tracer(profile):
        return
    widths = np.asarray(profile, dtype=np.float64)
    if widths.size < 2 or not np.all(widths.astype(np.float32).astype(np.float64) == widths):
        return
    nodes = np.cumsum(widths)[:-1]
    local = np.minimum(widths[:-1], widths[1:])
    ratios = (float(np.finfo(np.float32).eps) / 2) * np.abs(nodes) / local
    worst = int(np.argmax(ratios))
    cap = 0.5 * NODE_TOUCH_REL
    if ratios[worst] > cap:
        import warnings
        warnings.warn(
            f"axis {axis}: float32-valued widths at interior node "
            f"x={nodes[worst]:.12g} m have r={ratios[worst]:.12g}, "
            f"exceeding the node-band cap {cap:.12g}; faces declared near "
            "such nodes may move one cell. Widths built in float64 avoid it.",
            Float32WidthResolutionWarning, stacklevel=3)


def make_nonuniform_grid(
    domain_xy: tuple[float, float],
    dz_profile: np.ndarray,
    dx: float,
    cpml_layers: int = 8,
    *,
    dx_profile: np.ndarray | None = None,
    dy_profile: np.ndarray | None = None,
    pec_faces: set[str] | None = None,
    pmc_faces: set[str] | None = None,
    cpml_axes: str = "xyz",
    face_layers: dict[str, int] | None = None,
    dt: float | None = None,
    dt_min_cell: float | None = None,
    dt_caller: str | None = None,
    nominal_profiles: dict | None = None,
    periodic_axes: str = "",
    domain: tuple | None = None,
    boundary_depths: tuple | None = None,
) -> NonUniformGrid:
    """Create a non-uniform Yee grid.

    Parameters
    ----------
    domain_xy : (Lx, Ly) in metres
        Only used when ``dx_profile`` / ``dy_profile`` are None — in
        that case the xy mesh is uniform with spacing ``dx``.
    dz_profile : 1D array of z cell sizes in metres (physical domain only)
    dx : float
        Boundary cell size (also used for CPML padding and as the
        uniform-xy spacing when no xy profile is provided).
    cpml_layers : int
        Number of CPML cells added to each face (when that face is
        absorbing — see ``pec_faces`` / ``pmc_faces``).
    dx_profile, dy_profile : 1D arrays or None
        Optional per-cell x / y spacings for the physical (non-CPML)
        interior. When provided, the first and last values must match
        ``dx`` (they set the boundary cell size used by the CPML).
    pec_faces, pmc_faces : set of str or None
        Face labels (``x_lo``, ``x_hi``, ``y_lo``, ``y_hi``, ``z_lo``,
        ``z_hi``) where the boundary is PEC / PMC. Per-face pad count
        is forced to 0 on these faces so the reflector plane aligns
        with the user domain edge. Added 2026-04 to close the
        PMC+CPML composition gap on the non-uniform mesh path.
    cpml_axes : str
        Axes that participate in CPML allocation (default ``"xyz"``).
        Per-face allocation still applies; an axis absent from
        ``cpml_axes`` gets pad=0 on both faces.
    dt : float or None
        Concrete time step (s) to use INSTEAD of the Courant step this
        builder derives from ``min(cell)``. ``None`` (default) keeps the
        derived step and is byte-identical to the historical behaviour.

        Why it exists: the derived step is a function of the smallest
        cell, so a mesh that is a design variable moves ``dt`` with the
        deformation, and every observable read off the run then carries
        the step change as well as the geometry change. Pinning one step
        across a deformation family makes the sweep one antenna at one
        step. The value must be CONCRETE — a traced ``dt`` would put the
        step itself in the derivative.
    dt_min_cell : float or None
        Smallest cell size (m) the caller's TRACED axes reach anywhere in
        the family they will run. Required with ``dt`` when any profile is
        a tracer, because a tracer carries no host value to measure the
        realized Courant limit against. Concrete axes are always measured
        from their realized profile; this declared number is used only for
        the axes that cannot be.
    nominal_profiles : dict or None
        Concrete interior cell widths of each TRACED axis at the primal
        point (see ``Simulation``); a metre coordinate on that axis resolves
        on this profile's node line. Validated by
        :func:`rfx.model.graded_admission.admit_nominal_profiles`.
    """
    _pec = pec_faces or set()
    _pmc = pmc_faces or set()
    boundary_depths = boundary_depths if boundary_depths is not None else resolve_face_depths(
        budget=cpml_layers, absorbing_axes="".join(a for a in cpml_axes if a not in periodic_axes),
        pec_faces=_pec, pmc_faces=_pmc, face_layers=face_layers, periodic_axes=periodic_axes,
    )
    (pad_x_lo, pad_x_hi, pad_y_lo, pad_y_hi, pad_z_lo, pad_z_hi) = (
        record.realized for record in boundary_depths
    )
    from rfx.grid import axis_kinds, realized_cells
    kinds = axis_kinds(boundary_depths)
    # --- x profile (uniform or provided) ---
    if dx_profile is None:
        nx_interior = realized_cells(domain_xy[0], dx, kinds["x"])
        dx_prof_phys = np.full(nx_interior, float(dx))
    elif is_tracer(dx_profile):
        # Tracer path (mesh-as-design-variable): stay in jnp and skip the
        # concrete boundary validation. Caller is responsible for keeping
        # `dx_profile[0] == dx_profile[-1] == dx` (the scalar `dx` still
        # names the boundary cell; the CPML reads the traced end cells
        # through `boundary_cell`). Mirrors the 2026-04-17 dz tracer refactor.
        dx_prof_phys = jnp.asarray(dx_profile, dtype=jnp.float32)
    else:
        dx_prof_phys = np.asarray(dx_profile, dtype=np.float64)
        # Guard: CPML cells must have the same size as the boundary interior
        # cells, otherwise the CPML σ/κ profile is miscalibrated.
        if abs(float(dx_prof_phys[0]) - float(dx)) > 1e-12:
            raise ValueError(
                f"dx_profile[0]={float(dx_prof_phys[0])} must equal boundary "
                f"dx={float(dx)} (CPML cells use the boundary spacing)."
            )
        if abs(float(dx_prof_phys[-1]) - float(dx)) > 1e-12:
            raise ValueError(
                f"dx_profile[-1]={float(dx_prof_phys[-1])} must equal boundary "
                f"dx={float(dx)}."
            )
    dx_full = _append_bounding_node(_pad_profile(dx_prof_phys, pad_x_lo, pad_x_hi), periodic="x" in periodic_axes)
    if not is_tracer(dx_full):
        assert_cells_span_declared_profile(
            dx_full, pad_x_lo, pad_x_hi, dx_prof_phys, "x", periodic="x" in periodic_axes)
    nx = int(dx_full.shape[0])

    # --- y profile ---
    if dy_profile is None:
        # Uniform y uses `dx` as the cell size (legacy behaviour)
        ny_interior = realized_cells(domain_xy[1], dx, kinds["y"])
        dy_prof_phys = np.full(ny_interior, float(dx))
        dy_boundary = float(dx)
    elif is_tracer(dy_profile):
        # Tracer path: stay in jnp. The scalar `dy` is the concrete `dx`
        # (the CPML reads the traced end cells through `boundary_cell`) — the caller must align `dy_profile[0]` and
        # `dy_profile[-1]` with `dx` (same contract as the concrete path).
        dy_prof_phys = jnp.asarray(dy_profile, dtype=jnp.float32)
        dy_boundary = float(dx)
    else:
        dy_prof_phys = np.asarray(dy_profile, dtype=np.float64)
        dy_boundary = float(dy_prof_phys[0])
        if abs(float(dy_prof_phys[-1]) - dy_boundary) > 1e-12:
            raise ValueError(
                f"dy_profile boundary cells must match each other "
                f"(got lo={dy_boundary}, hi={float(dy_prof_phys[-1])})."
            )
    dy_full = _append_bounding_node(_pad_profile(dy_prof_phys, pad_y_lo, pad_y_hi), periodic="y" in periodic_axes)
    if not is_tracer(dy_full):
        assert_cells_span_declared_profile(
            dy_full, pad_y_lo, pad_y_hi, dy_prof_phys, "y", periodic="y" in periodic_axes)
    ny = int(dy_full.shape[0])

    # --- z profile ---
    dz_full = _append_bounding_node(_pad_profile(dz_profile, pad_z_lo, pad_z_hi), periodic="z" in periodic_axes)
    if not is_tracer(dz_full):
        assert_cells_span_declared_profile(
            dz_full, pad_z_lo, pad_z_hi, dz_profile, "z", periodic="z" in periodic_axes)
    nz = int(dz_full.shape[0])

    for axis, profile in zip("xyz", (dx_prof_phys, dy_prof_phys, dz_profile)):
        _warn_float32_width_resolution(profile, axis)

    # --- CFL from minimum cell size on every axis ---
    dt_pin = dt

    def _axis_min(d_full):
        return jnp.min(d_full) if is_tracer(d_full) else float(np.min(d_full))

    any_traced = (
        is_tracer(dx_full) or is_tracer(dy_full) or is_tracer(dz_full)
    )
    dx_min = _axis_min(dx_full)
    dy_min = _axis_min(dy_full)
    dz_min = _axis_min(dz_full)
    if any_traced:
        dt = 0.99 / (C0 * jnp.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))
    else:
        dt = float(
            0.99 / (C0 * np.sqrt(1 / dx_min ** 2 + 1 / dy_min ** 2 + 1 / dz_min ** 2))
        )
    if dt_pin is not None:
        dt = _pinned_dt(dt_pin, dt_min_cell,
                        (dx_full, dy_full, dz_full), ("x", "y", "z"),
                        where=dt_caller or "make_nonuniform_grid")

    # --- Per-cell arrays + inverse spacings ---
    dx_arr = jnp.asarray(dx_full, dtype=jnp.float32)
    dy_arr = jnp.asarray(dy_full, dtype=jnp.float32)
    dz_arr = jnp.asarray(dz_full, dtype=jnp.float32)

    # Exact float64 spine for node positions (#802/#807): concrete profiles
    # only — a traced profile has no host copy and its axis keeps the traced
    # coordinate path.
    def _exact_f64(d_full):
        return None if is_tracer(d_full) else np.asarray(d_full,
                                                         dtype=np.float64)

    dx_arr_f64 = _exact_f64(dx_full)
    dy_arr_f64 = _exact_f64(dy_full)
    dz_f64 = _exact_f64(dz_full)

    from rfx.model.graded_admission import admit_nominal_profiles, admit_traced_windows
    nominal_cells = admit_nominal_profiles(
        nominal_profiles, (dx_profile, dy_profile, dz_profile),
        (domain_xy[0], domain_xy[1], None))
    traced_cells = admit_traced_windows(nominal_profiles)
    for axis, window in (traced_cells or {}).items():
        profile = (dx_profile, dy_profile, dz_profile)["xyz".index(axis)]
        _assert_traced_cells_match_nominal(profile, nominal_cells[axis], window, axis)

    inv_dx, inv_dx_h = _profile_to_inv_arrays(dx_full, periodic="x" in periodic_axes)
    inv_dy, inv_dy_h = _profile_to_inv_arrays(dy_full, periodic="y" in periodic_axes)
    inv_dz, inv_dz_h = _profile_to_inv_arrays(dz_full, periodic="z" in periodic_axes)

    return NonUniformGrid(
        nx=nx, ny=ny, nz=nz,
        dx=float(dx), dy=dy_boundary,
        dx_arr=dx_arr, dy_arr=dy_arr, dz=dz_arr,
        dt=dt, cpml_layers=cpml_layers,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
        inv_dx_h=inv_dx_h, inv_dy_h=inv_dy_h, inv_dz_h=inv_dz_h,
        pad_x_lo=pad_x_lo, pad_x_hi=pad_x_hi,
        pad_y_lo=pad_y_lo, pad_y_hi=pad_y_hi,
        pad_z_lo=pad_z_lo, pad_z_hi=pad_z_hi,
        dx_arr_f64=dx_arr_f64, dy_arr_f64=dy_arr_f64, dz_f64=dz_f64,
        face_layers=dict(x_lo=pad_x_lo, x_hi=pad_x_hi,
                         y_lo=pad_y_lo, y_hi=pad_y_hi,
                         z_lo=pad_z_lo, z_hi=pad_z_hi),
        nominal_cells=nominal_cells,
        traced_cells=traced_cells, periodic_axes=periodic_axes, domain=domain,
    )


def _assert_traced_cells_match_nominal(profile, nominal, window, axis):
    """Check fixed widths within four float32 eps times the nominal width.

    This reads values, not tangents: at the primal point a tangent outside
    the window is unseen. It is caught once values move, for example on a
    later optimisation step reusing the declaration.
    """
    realized = jnp.asarray(profile, dtype=jnp.float32)
    nominal32 = jnp.asarray(nominal, dtype=jnp.float32)
    from rfx.model.graded_admission import window_mask
    outside = ~window_mask(window, realized.shape[0])
    tolerance = 4 * jnp.finfo(jnp.float32).eps * nominal32
    bad = outside & ~(jnp.abs(realized - nominal32) <= tolerance)
    first = jnp.argmax(bad)

    def _check(failed, cell, width, expected):
        if bool(failed):
            raise ValueError(
                f"traced {axis} axis interior cell {int(cell)} has realized "
                f"width {float(width)!r}, nominal width {float(expected)!r}; "
                f"outside declared traced_cells window {window!r}.")

    from rfx._traced_check import traced_check
    traced_check("graded mesh", "fixed cells", _check, jnp.any(bad), first, realized[first], nominal32[first])


def _assert_traced_index_matches_realized(
    d_arr, total_pad: int, pad_lo: int, pos: float, nominal_idx: int,
    axis: str, declared: bool = False,
) -> None:
    """The node a coordinate names must be the same node on the mesh the run
    actually builds.

    On a traced axis the index above is resolved against a UNIFORM reference
    mesh of the boundary cell size, because a tracer has no host node line to
    measure against. That reference is the realized mesh only where the
    declared nominal profile is itself uniform up to the queried coordinate.
    Where it is not — a graded band, a fine window — the same declared
    position names one node concretely and a different node under tracing, so
    the traced run feeds or probes a different point of the board than the
    concrete run of the same script. Silently.

    A tracer has values at RUN time, so the comparison is made there: the
    realized cumulative edges pick their own nearest node and the callback
    refuses when the two disagree, naming both. Nothing is raised while the
    two agree, which is the case a deformation that keeps its outer node
    lines fixed (the mesh-as-design-variable pattern) is in. ``declared``
    says the index came from the caller's ``nominal_profiles=`` rather than
    the uniform reference mesh; it changes the message only.

    A coordinate midway between two nodes is equally near both. The nominal
    index takes the lower one (``nearest_node_index``, #1295), but the
    realized edges are cumulative sums in the traced dtype, so one neighbour
    comes out a few ulp nearer, either one (at most 5 float32 ulp of the
    coordinate measured on constant 0.127 to 1 mm columns of up to 400
    cells). The nominal index is therefore accepted when it is the realized
    nearest node, or its neighbour at a realized distance within 16 ulp of
    the coordinate of the nearest one. That is under 1e-3 of a cell on
    those columns; a deformation that moves a node line moves it by far more.
    """
    n_total = int(d_arr.shape[0])
    pad_hi = total_pad - pad_lo
    if n_total - total_pad <= 0:
        return
    interior = jnp.asarray(d_arr)[pad_lo:n_total - pad_hi]
    edges = jnp.concatenate(
        [jnp.zeros((1,), dtype=interior.dtype), jnp.cumsum(interior)])
    dist = jnp.abs(edges - float(pos))
    nearest = jnp.argmin(dist)
    nominal = min(int(nominal_idx), int(dist.shape[0]) - 1)
    tol = 16 * jnp.finfo(dist.dtype).eps * jnp.maximum(
        abs(float(pos)), edges[nearest])
    agrees = (nearest == nominal) | (
        (jnp.abs(nearest - nominal) == 1)
        & (dist[nominal] - dist[nearest] <= tol))

    mesh = ("on the declared nominal_profiles= profile" if declared else
            "on the nominal uniform mesh this lane must use for a structural "
            "index")
    hint = ("" if declared else f" Or declare the concrete {axis} profile "
            "with nominal_profiles=.")

    def _check(realized, agrees):
        realized = int(realized)
        if not bool(agrees):
            raise ValueError(
                f"position {pos:.9g} m on the traced {axis} axis resolves to "
                f"interior node {nominal_idx} {mesh}, but to node "
                f"{realized} on the mesh the run built. The two runs would "
                f"place this source / probe / port at different points of the "
                f"board. Declare the {axis} deformation so the node lines "
                "outside it do not move (stretch and shrink inside a window), "
                "or give the position by node index." + hint
            )

    from rfx._traced_check import traced_check
    traced_check("graded mesh", "node index", _check, nearest, agrees)


def _traced_fallback_width(grid: NonUniformGrid, axis) -> float:
    """Width of the uniform reference cells a traced axis resolves positions
    on when no ``nominal_profiles`` entry is declared: the y scalar on y, the
    x scalar on x and z (one definition for the resolver and host code)."""
    return float(grid.dy if grid._axis_layout(axis)[0] == 1 else grid.dx)


def z_position_to_index(grid: NonUniformGrid, z_phys: float) -> int:
    """Convert physical z-coordinate to (cpml-offset) grid index."""
    return _axis_position_to_index(grid, "z", z_phys,
                                   fallback_dx=_traced_fallback_width(grid, "z"))


def _axis_position_to_index(
    grid: NonUniformGrid,
    axis,
    pos: float,
    fallback_dx: float | None = None,
) -> int:
    """Padded index of the node nearest ``pos`` on ``axis``, clamped.

    The node is the one ``index_of`` names: ``nearest_node_index`` on the
    node line the conductors are rasterized on, a tie to the lower node
    (#1295, #1342). Two legacy behaviours stay here and not in ``index_of``:
    a coordinate outside the interior CLAMPS to the end node, and a traced
    axis answers from a nominal mesh.

    When the axis's cell widths are a JAX tracer (mesh-as-design-variable
    path) there is no host node line. If the caller declared the concrete
    profile (``grid.nominal_cells``, from ``nominal_profiles=``) the index is
    resolved on its node line ``[0, cumsum(w)]``; otherwise on a uniform
    reference mesh of ``fallback_dx`` cells. Either way it is checked against
    the realized mesh at run time (``_assert_traced_index_matches_realized``).
    The traced cell sizes drive the FDTD physics downstream; only the
    structural index is resolved from the nominal mesh, with the same rule.
    """
    ax, n, pad_lo, pad_hi = grid._axis_layout(axis)
    if _axis_name(ax) in grid.periodic_axes:
        return _periodic_index(grid, ax, pos)
    store = grid._axis_store(ax)
    from rfx.model.graded_admission import nominal_cells
    widths = nominal_cells(grid, ax)
    if widths is not None:
        idx = _nearest_on_line(np.concatenate([[0.0], np.cumsum(widths)]),
                               widths, pos)
        _assert_traced_index_matches_realized(
            store, pad_lo + pad_hi, pad_lo, float(pos), idx,
            _axis_name(ax), declared=True)
        return idx + pad_lo
    if is_tracer(store):
        if fallback_dx is None or fallback_dx <= 0:
            raise ValueError(
                "tracer-valued cell-size profile requires a concrete "
                "fallback_dx for position->index resolution."
            )
        last = n - pad_lo - pad_hi
        from rfx.geometry.rasterize_grid import _uniform_axis_nodes
        nominal = _uniform_axis_nodes(last + 1, 0, float(fallback_dx))
        idx = _nearest_on_line(nominal, np.full(last, float(fallback_dx)),
                               pos)
        _assert_traced_index_matches_realized(
            store, pad_lo + pad_hi, pad_lo, float(pos), idx,
            _axis_name(ax))
        return idx + pad_lo
    nodes, cells = grid._node_line(ax)
    return _nearest_on_line(nodes, cells, pos) + pad_lo


def host_node_of(grid: NonUniformGrid, axis, i: int) -> float:
    """Coordinate of padded node ``i`` on the line ``_axis_position_to_index``
    resolves positions on, so the two round-trip on every axis.

    A concrete axis gives ``node_of``. A traced axis gives the nominal mesh
    that resolver uses: the declared ``nominal_profiles`` widths, else uniform
    cells of the axis's fallback width (:func:`_traced_fallback_width`).
    """
    ax, n, pad_lo, _pad_hi = grid._axis_layout(axis)
    if not is_tracer(grid._axis_store(ax)) or _axis_name(ax) in grid.periodic_axes:
        return float(grid.node_of(ax, i))
    idx = int(i)
    if not (0 <= idx < n):
        raise IndexError(f"padded index {idx} is outside axis {_axis_name(ax)!r}, which has {n} entries")
    from rfx.model.graded_admission import nominal_cells
    widths = nominal_cells(grid, ax)
    if widths is not None:
        nodes = np.concatenate([[0.0], np.cumsum(widths)])
        return float(nodes[min(max(idx - pad_lo, 0), nodes.size - 1)])
    from rfx.geometry.rasterize_grid import _uniform_axis_nodes
    fallback = _traced_fallback_width(grid, ax)
    return float(_uniform_axis_nodes(n, pad_lo, fallback)[idx])


def host_nominal_grid(grid: NonUniformGrid) -> NonUniformGrid:
    """``grid`` with each traced axis's cells replaced by the nominal widths
    positions resolve on (the line :func:`host_node_of` reads), so host code
    can size a port on concrete numbers. Concrete axes are returned as they are.
    """
    from rfx.model.graded_admission import nominal_cells
    fields = {}
    for ax, (store, exact) in enumerate((("dx_arr", "dx_arr_f64"), ("dy_arr", "dy_arr_f64"), ("dz", "dz_f64"))):
        cells = getattr(grid, store)
        if not is_tracer(cells):
            continue
        widths = nominal_cells(grid, ax, padded=True)
        if widths is None:
            widths = np.full(cells.shape[0], _traced_fallback_width(grid, ax))
        fields[store] = jnp.asarray(widths, dtype=cells.dtype)
        fields[exact] = np.asarray(widths, dtype=np.float64)
    return grid._replace(**fields) if fields else grid


def _nearest_on_line(nodes, cells, pos) -> int:
    """``nearest_node_index`` on a node line, with the local cell the sheet
    snap uses (``_local_cell``). A coordinate outside the line gets its end
    node, which is the legacy clamp of ``position_to_index``."""
    from rfx.geometry.rasterize_grid import _local_cell
    return nearest_node_index(nodes, float(pos),
                              _local_cell(nodes, cells, float(pos)))


def position_to_index(grid: NonUniformGrid, pos: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert physical (x, y, z) to grid indices for NonUniformGrid.

    Accounts for per-face CPML padding (``pad_{axis}_lo`` leading offset).
    Each axis resolves through ``_axis_position_to_index``: the node
    ``index_of`` names, clamped into the interior. On a constant axis, for a
    coordinate inside the interior that is not a tie, that is the uniform
    ``Grid`` lookup. Both lanes take the lower node within the tie band. Outside the
    interior this clamps to the end node, while the uniform grid returns a
    pad index or refuses.
    """
    i = _axis_position_to_index(grid, "x", pos[0],
                                fallback_dx=_traced_fallback_width(grid, "x"))
    j = _axis_position_to_index(grid, "y", pos[1],
                                fallback_dx=_traced_fallback_width(grid, "y"))
    k = z_position_to_index(grid, pos[2])
    return (i, j, k)


def _periodic_index(grid, axis, coordinate):
    """Grid's half-open periodic point admission and lower-node tie rule."""
    from rfx._grid_metric import nearest_uniform_index
    ratio = coordinate / float(grid.cells(axis)[0])
    n = grid.shape[axis]
    if ratio < -.5 or ratio >= n + .5:
        raise ValueError(f"position {coordinate} on axis {'xyz'[axis]!r} lies outside this axis's {n} entries")
    return nearest_uniform_index(ratio) % n
