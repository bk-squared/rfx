"""Conductor realization for preflight.

This module is not a check
family: it emits no advisory at all. It is the DATA layer every family reads
when it needs to know where metal actually is on this lattice, under the
lattice ownership contract: a declaration and its realization are the same
statement, so every conductor number preflight prints is read from
``rfx.boundaries.pec.realized_pec_edge_masks``, never from a cell mask or a
bounding box.

Three module-level classes and four leaves carry that:

* :class:`_RealizedPEC` -- the WHOLE model's realized conductor set on one
  grid (volume cells, sheets, wires, and the ``(Mx, My, Mz)`` edge masks the
  solver zeroes), read once from the production assembly.
* :class:`_EntryRealization` -- what ONE conductor declaration realizes as
  (``volume`` / ``sheet`` / ``wire`` / ``lossy`` / ``refused``), with its own
  edge masks, wall planes and shifted-origin re-realization.
* :class:`_CampaignStaticsContext` -- the per-configuration shared state the
  campaign checks and every other conductor-reading check build once:
  grid, production ``GridCoords``, cell centres, per-axis spacings, periodic
  flags, the realized set and the per-entry realizations.
* ``_shape_bounds``, ``_shift_back_np``, ``_wall_nodes_on_plane`` and
  ``_realized_edges_np`` -- the numpy leaves those three read and nothing
  else does.

Plus the four ``_PreflightMixin`` methods that hand them to the families:
``_assemble_realized`` (the production assembly call), ``_port_realized_edges``
and its kept-name alias ``_port_pec_mask``, and ``_campaign_ctx`` (the
configuration-keyed cache).

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic. The two ``rfx.preflight._common`` leaves below are the legs-2 and -6
shared leaves coming home: ``_sorted_box_corners`` went to ``_common`` because
``_shape_bounds`` and ``_CampaignStaticsContext`` (here) and
``_validate_cfg_sheet_cavity_thickness`` (``pec_geometry``) straddled the
split, and ``_component_is_dead`` because ``_RealizedPEC.component_is_dead``
(here) and ``_validate_cfg_port_inside_pec`` (``ports``) did. Both readers on
this side are now in one module, but the OTHER reader of each is still in a
different one, so both leaves stay in ``_common``; pulling either in here
would make ``pec_geometry`` or ``ports`` import a sibling for a nine- or
thirty-line helper.

``_CampaignStaticsContext.entry_realizations`` function-locally imports
``rfx.geometry.rasterize_grid._local_cell as _rasterize_local_cell``. That
alias is leg 5's and is load-bearing prose, not style: a DIFFERENT
``_local_cell``, the coarsest-cell helper with signature
``(profile, lo, hi, fallback)``, lives in ``rfx/preflight/mesh.py`` and is
re-exported by the facade. The local import always shadowed the module global
inside that method; the alias says so.

Nothing in this module emits a :class:`~rfx.preflight._common.PreflightIssue`
or warns -- an AST scan over the moved text finds zero constructions of any
issue-carrying class -- so the split's frozen emission contract
(113 sites / 74 literal codes) does not move, and the committed advisory-text
snapshot in ``tests/locks/test_preflight_split_snapshot.py`` has nothing of
this module's own to pin. What it pins instead is that the families keep
reaching it: 64 of the 65 corpus fixtures build a ``_CampaignStaticsContext``
and 35 run the production assembly through ``_assemble_realized``, so a broken
edge here reds most of the corpus rather than one line of one report.
"""

from __future__ import annotations

from rfx.preflight._not_judged import is_tracer_error, not_judged

from contextlib import contextmanager
from contextvars import ContextVar

import jax
import jax.numpy as jnp
import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box

from rfx.preflight._common import (
    PreflightWarning,
    _component_is_dead,
    _sorted_box_corners,
)


def _shape_bounds(shape):
    """``(lo, hi, exact)`` for any Shape that reports a bounding box.

    ``exact`` is True when the bounding box IS the shape (a :class:`Box`),
    False when it merely bounds it. ``None`` when the shape reports no
    bounds at all — the caller must then count the entry as skipped rather
    than treat it as clean.
    """
    lo, hi = _sorted_box_corners(shape)
    if lo is not None:
        return lo, hi, isinstance(shape, Box)
    bb = getattr(shape, "bounding_box", None)
    if bb is None:
        return None
    try:
        blo, bhi = bb()
        blo = np.asarray(blo, dtype=np.float64)
        bhi = np.asarray(bhi, dtype=np.float64)
    except (TypeError, ValueError, AttributeError, IndexError,
            NotImplementedError) as _traced_exc:
        if is_tracer_error(_traced_exc):
            not_judged(None, "_validate_cfg_campaign_statics")
        return None
    if blo.shape != (3,) or bhi.shape != (3,):
        return None
    return np.minimum(blo, bhi), np.maximum(blo, bhi), False


def _shift_back_np(arr, ax: int, periodic: bool):
    """``arr[i-1]`` along ``ax`` under the convention (numpy twin of
    ``rfx.boundaries.pec._shift``): wrap on a periodic or length-1 axis,
    zero pad otherwise."""
    if arr.shape[ax] == 1 or periodic:
        return np.roll(arr, 1, axis=ax)
    out = np.zeros_like(arr)
    sl_dst = [slice(None)] * arr.ndim
    sl_src = [slice(None)] * arr.ndim
    sl_dst[ax] = slice(1, None)
    sl_src[ax] = slice(None, -1)
    out[tuple(sl_dst)] = arr[tuple(sl_src)]
    return out


def _wall_nodes_on_plane(edges, axis: int, k: int, periodic) -> np.ndarray:
    """In-plane boolean node map of plane ``k`` along ``axis``: True where
    the node carries a tangential PEC wall.

    A node carries a wall when any E edge tangential to ``axis`` and
    INCIDENT to the node is PEC — the edge stored at the node's own index
    or at the backward index along that edge's own axis (the same rule
    :func:`rfx.boundaries.pec.realized_wall_planes` applies to one column,
    evaluated for the whole plane). The result's two axes are the in-plane
    axes in xyz order.
    """
    tangential = [c for c in range(3) if c != axis]
    out = None
    for pos, t in enumerate(tangential):
        m = np.asarray(np.take(edges[t], k, axis=axis), dtype=bool)
        hit = m | _shift_back_np(m, pos, bool(periodic[t]))
        out = hit if out is None else (out | hit)
    return out


def _realized_edges_np(pec_mask, sheets, wires, periodic, shape):
    """``(Mx, My, Mz)`` as host numpy from the ONE realization function."""
    from rfx.boundaries.pec import realized_pec_edge_masks
    if pec_mask is None and not sheets and not wires:
        return tuple(np.zeros(tuple(shape), dtype=bool) for _ in range(3))
    import jax
    with jax.ensure_compile_time_eval():
        cells = None if pec_mask is None else jnp.asarray(pec_mask)
        edges = realized_pec_edge_masks(cells, sheets, wires, periodic)
        return tuple(np.asarray(m, dtype=bool) for m in edges)


class _RealizedPEC:
    """The run's realized conductor set, read once from the production
    assembly with the collectors: volume cells (``pec_mask``),
    :class:`~rfx.boundaries.pec.SheetSpec` sheets, wires, and the
    ``(Mx, My, Mz)`` edge masks the solver zeroes — host numpy.
    """

    def __init__(self, *, lane, grid, materials, pec_mask, sheets, wires,
                 periodic, sheet_specs=(), geometry_masks=()):
        self.geometry_masks = dict(geometry_masks)
        self.lane = lane
        self.grid = grid
        self.materials = materials
        self.pec_mask = (None if pec_mask is None
                         else np.asarray(pec_mask, dtype=bool))
        self.sheets = list(sheets)
        self.wires = list(wires)
        self.sheet_specs = list(sheet_specs)
        self.periodic = tuple(bool(p) for p in periodic)
        self.edges = _realized_edges_np(
            pec_mask, self.sheets, self.wires, self.periodic, grid.shape)

    @property
    def empty(self) -> bool:
        return self.pec_mask is None and not self.sheets and not self.wires

    def wall_planes(self, axis: int, *, ij=None, region=None) -> list:
        from rfx.boundaries.pec import realized_wall_planes
        return realized_wall_planes(self.edges, axis, ij=ij, region=region,
                                    periodic=self.periodic)

    def is_pec_edge(self, component, i, j, k) -> bool:
        from rfx.boundaries.pec import edge_is_pec
        return edge_is_pec(self.edges, component, i, j, k)

    def wall_nodes_on_plane(self, axis: int, k: int) -> np.ndarray:
        return _wall_nodes_on_plane(self.edges, axis, k, self.periodic)

    def component_is_dead(self, component: str, idx) -> bool:
        return _component_is_dead(self.edges, component, idx)


class _EntryRealization:
    """What ONE conductor declaration realizes as under the lattice ownership contract.

    ``kind`` is ``"volume"`` (centre-sampled cells), ``"sheet"`` (a
    :class:`SheetSpec`: one plane, closed footprint, no cell), ``"wire"``
    (a filament), ``"lossy"`` (a sigma-fold thin conductor — fenced, not
    PEC realization) or ``"refused"`` (the classifier raised; ``error``
    carries its message). Nothing here is inferred from an extent: the
    kind is what the shared classifier says.
    """

    def __init__(self, *, label, name, shape, kind, cells=None, sheet=None,
                 wire=None, error=None, lo=None, hi=None, declared_mid=None,
                 tie_planes=None, solved_shape=None, subject_label=None):
        self.label = label
        self.subject_label = label if subject_label is None else subject_label
        self.name = name
        self.shape = shape
        self.solved_shape = shape if solved_shape is None else solved_shape
        self.kind = kind
        self.cells = None if cells is None else np.asarray(cells, dtype=bool)
        self.sheet = sheet
        self.wire = wire
        self.error = error
        self.lo = lo
        self.hi = hi
        self.declared_mid = declared_mid
        self.tie_planes = tie_planes
        self._edges = None

    @property
    def is_pec(self) -> bool:
        return self.kind in ("volume", "sheet", "wire")

    def edges(self, periodic, shape):
        """This entry's OWN realized ``(Mx, My, Mz)`` (numpy), cached."""
        if self._edges is None:
            if self.kind == "volume":
                self._edges = _realized_edges_np(
                    self.cells, (), (), periodic, shape)
            elif self.kind == "sheet":
                self._edges = _realized_edges_np(
                    None, [self.sheet], (), periodic, shape)
            elif self.kind == "wire":
                self._edges = _realized_edges_np(
                    None, (), [self.wire], periodic, shape)
            else:
                self._edges = tuple(np.zeros(tuple(shape), dtype=bool)
                                    for _ in range(3))
        return self._edges

    def wall_planes(self, axis: int, periodic, shape) -> list:
        from rfx.boundaries.pec import realized_wall_planes
        return realized_wall_planes(self.edges(periodic, shape), axis,
                                    periodic=periodic)

    def edge_count(self, periodic, shape) -> int:
        return int(sum(int(m.sum()) for m in self.edges(periodic, shape)))

    def edge_count_shifted(self, ctx, axis: int, shift_m: float):
        """This entry's realized edge count when the lattice origin is
        slid by ``shift_m`` along ``axis`` — re-realized through the
        PRODUCTION rule (§1.1 centre-sampled volume / §1.3 closed sheet
        footprint, then ``realized_pec_edge_masks``), never a hand copy.
        ``None`` when the shifted realization is refused (a sheet plane
        that leaves the node range, an emptied volume)."""
        from rfx.geometry.rasterize_grid import (
            GridCoords, cell_centres_from_nodes, pec_volume_cell_mask,
            sheet_spec_from_shape, interior_lattice_mask,
        )
        from dataclasses import replace
        nodes = [np.asarray(ctx.coords.x), np.asarray(ctx.coords.y),
                 np.asarray(ctx.coords.z)]
        nodes[axis] = nodes[axis] + np.float64(shift_m)
        coords = GridCoords(x=nodes[0], y=nodes[1], z=nodes[2],
                            shape=tuple(ctx.grid.shape))
        shape = tuple(ctx.grid.shape)
        try:
            if self.kind == "volume":
                centres = cell_centres_from_nodes(coords, ctx.cell_sizes)
                cells = interior_lattice_mask(
                    pec_volume_cell_mask(self.solved_shape, centres, ctx.cell_sizes, grid=ctx.grid), ctx.grid,
                    cell_axes=(True, True, True))
                if not cells.any():
                    return None
                edges = _realized_edges_np(cells, (), (), ctx.periodic, shape)
            elif self.kind == "sheet":
                sp = sheet_spec_from_shape(
                    self.solved_shape, coords, ctx.cell_sizes,
                    normal_axis=int(self.sheet.normal_axis), name=self.name, grid=ctx.grid)
                sp = replace(sp, footprint=interior_lattice_mask(sp.footprint, ctx.grid))
                edges = _realized_edges_np(None, [sp], (), ctx.periodic, shape)
            else:
                return None
        except (ValueError, TypeError, IndexError) as _traced_exc:
            if is_tracer_error(_traced_exc):
                not_judged(None, "_validate_cfg_campaign_statics")
            return None
        return int(sum(int(m.sum()) for m in edges))

    def footprint_on_plane(self, axis: int, k: int, periodic, shape):
        """In-plane node map of this entry's wall on plane ``k``."""
        return _wall_nodes_on_plane(self.edges(periodic, shape), axis, k,
                                    periodic)


_assembly_warning_readers = ContextVar("assembly_warning_readers", default=None)


@contextmanager
def assembly_warning_scope():
    readers = []
    token = _assembly_warning_readers.set(readers)
    try:
        yield
    finally:
        try:
            for ctx in readers:
                ctx._emit_assembly_warnings()
        finally:
            _assembly_warning_readers.reset(token)


class _CampaignStaticsContext:
    """Shared lazily-built state for the campaign checks and
    every other preflight check that reads conductor geometry.

    Built once per CONFIGURATION through :meth:`_PreflightMixin
    ._campaign_ctx` (a cache keyed on the declarations and their mesh
    parameters, so an ``add()`` after a preflight misses it — a plain
    instance cache would have gone stale). The grid, sample coordinates,
    cell centres and the realized conductor set come from the PRODUCTION
    builders — the check must see the run's realization, not a hand model
    of it (the ``_validate_cfg_graded_box_rasterization`` lesson: a hand
    model of the sampling diverged from the rasterizer on exactly the
    cases it existed to catch). Under the lattice ownership contract that means one thing: every
    conductor number a check prints is read from
    ``rfx.boundaries.pec.realized_pec_edge_masks`` (through
    :meth:`realized` for the whole model and :meth:`entry_realizations`
    per declaration), never from a cell mask or a bounding box.
    ``coords`` contains the production grid's E-node sample coordinates.
    """

    _NARROW_EXCS = (ValueError, TypeError, NotImplementedError, KeyError,
                    AttributeError, IndexError)

    def __init__(self, sim):
        self.sim = sim
        self.error: str | None = None
        self.lane: str | None = None
        self.grid = None
        self.coords = None
        self.centres = None
        self.cell_sizes = None
        self.nodes = None
        self.spacings = None
        self.periodic = (False, False, False)
        self._realized = None
        self._entries = None
        self.omitted_pec_products: list[str] = []
        self.assembly_error: str | None = None
        self._assembly_exception: Exception | None = None
        self._build()

    def _build(self) -> None:
        sim = self.sim
        profiles = (sim._dx_profile, sim._dy_profile, sim._dz_profile)
        if is_tracer(sim._dx) or any(
                p is not None and is_tracer(p) for p in profiles):
            self.error = "traced-mesh"
            return
        from rfx.geometry.rasterize_grid import (
            GridCoords, coords_from_nonuniform_grid,
            centres_from_nonuniform_grid, cell_sizes_from_nonuniform_grid,
            centres_from_uniform_grid, cell_sizes_from_uniform_grid,
            periodic_flags_from_axes,
        )
        try:
            if any(p is not None for p in profiles):
                self.lane = "nonuniform"
                grid = sim._build_nonuniform_grid()
                self.coords = coords_from_nonuniform_grid(grid)
                self.centres = centres_from_nonuniform_grid(grid, self.coords)
                self.cell_sizes = cell_sizes_from_nonuniform_grid(grid)
                nodes = [np.asarray(c, dtype=np.float64)
                         for c in (self.coords.x, self.coords.y, self.coords.z)]
                spacings = [np.asarray(d, dtype=np.float64)
                            for d in self.cell_sizes]
            else:
                self.lane = "uniform"
                grid = sim._build_grid()
                from rfx.geometry.csg import _grid_coords
                cx, cy, cz = _grid_coords(grid)
                self.coords = GridCoords(x=cx, y=cy, z=cz, shape=grid.shape)
                self.centres = centres_from_uniform_grid(grid)
                self.cell_sizes = cell_sizes_from_uniform_grid(grid)
                d = float(grid.dx)
                nodes, spacings = [], []
                for n, pad in zip(grid.shape, grid.axis_pads):
                    nodes.append((np.arange(n, dtype=np.float64) - pad) * d)
                    spacings.append(np.full(n, d, dtype=np.float64))
            self.grid = grid
            self.nodes = nodes
            self.spacings = spacings
            pf = getattr(sim, "_periodic_flags", None)
            self.periodic = (tuple(bool(p) for p in pf()) if callable(pf)
                             else periodic_flags_from_axes(
                                 getattr(sim, "_periodic_axes", "")))
        except self._NARROW_EXCS as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._assembly_exception = exc


    def realized(self, *, strict=False, defer_warnings=False):
        """:class:`_RealizedPEC` from the PRODUCTION assembly (with the
        sheet/wire collectors), once; ``None`` when the assembly
        raised (``assembly_error`` says why). Strict execution/build readers
        re-raise the original exception, including after a diagnostic read.
        """
        import warnings
        from contextlib import nullcontext
        readers = _assembly_warning_readers.get()
        defer = defer_warnings and readers is not None
        if self._realized is None and self.assembly_error is None:
            with (warnings.catch_warnings(record=True) if defer else nullcontext([])) as emitted:
                try:
                    self._realized = self.sim._assemble_realized(
                        self.grid, nonuniform=(self.lane == "nonuniform"))
                except self._NARROW_EXCS as exc:
                    self.assembly_error = f"{type(exc).__name__}: {exc}"
                    self._assembly_exception = exc
            if emitted:
                self._assembly_warnings = tuple(emitted)
                readers.append(self)
        if not defer:
            self._emit_assembly_warnings()
        if strict and self._assembly_exception is not None:
            raise self._assembly_exception
        return self._realized

    def bind_realization(self):
        """Keep record assembly failures in the diagnostic assembly channel."""
        from rfx.model.realization_table import bind_table
        root = self.realized() if self.grid is not None else None
        if root is None or self._assembly_exception is not None:
            return None
        try:
            bind_table(self.sim, root, lane=root.lane, compact=False)
        except self._NARROW_EXCS as exc:
            self._assembly_exception = exc
            return None
        return root

    def _emit_assembly_warnings(self):
        import warnings
        emitted = getattr(self, '_assembly_warnings', ())
        self._assembly_warnings = ()
        for warning in emitted:
            warnings.warn_explicit(warning.message, warning.category,
                                   warning.filename, warning.lineno)

    def assembled(self):
        """``(materials, pec_mask)`` — kept for callers that only read
        cells; ``pec_mask`` holds VOLUME cells only (a sheet owns none)."""
        r = self.realized()
        if r is None:
            return None
        return r.materials, r.pec_mask

    def entry_realizations(self) -> list:
        """One :class:`_EntryRealization` per conductor declaration —
        PEC geometry entries (``geometry[i]``) and thin conductors
        (``thin_conductor[i]``) — through the shared classifier."""
        if self._entries is not None:
            return self._entries
        from rfx.geometry.rasterize_grid import (
            classify_pec_entry, sheet_spec_from_shape,
            _local_cell as _rasterize_local_cell,
        )
        from rfx.materials.thin_conductor import sheet_bounds
        from rfx.geometry.smoothing import continued_conductor_shape
        sim = self.sim
        out = []
        assembled = self.realized(defer_warnings=True)
        products = {key: (cells, sheet, wire, shape)
                    for key, cells, sheet, wire, shape in
                    getattr(assembled, "assembly_entries", ())}

        def _bounds(shape):
            lo, hi = _sorted_box_corners(shape)
            if lo is not None:
                return lo, hi
            try:
                blo, bhi = sheet_bounds(shape)
            except self._NARROW_EXCS as _traced_exc:
                if is_tracer_error(_traced_exc):
                    not_judged(None, "_validate_cfg_campaign_statics")
                return None, None
            if blo is None or bhi is None:
                return None, None
            if any(is_tracer(v) for v in jax.tree_util.tree_leaves((blo, bhi))):
                not_judged(None, "_validate_cfg_campaign_statics")
                return None, None
            blo = np.asarray(blo, dtype=np.float64)
            bhi = np.asarray(bhi, dtype=np.float64)
            return np.minimum(blo, bhi), np.maximum(blo, bhi)

        def _tie(sheet, lo, hi):
            """(k_lo, k_hi) when the declared mid-plane is an exact
            half-cell tie between two node planes, else None — the same
            comparison ``_nearest_plane`` resolves LOWER."""
            if lo is None or hi is None:
                return None, None
            a = int(sheet.normal_axis)
            mid = 0.5 * (float(lo[a]) + float(hi[a]))
            from rfx._internal._periodic import plane_coordinate
            mid = plane_coordinate(self.grid, a, mid)
            nodes = self.nodes[a]
            d_local = _rasterize_local_cell(nodes, self.cell_sizes[a], mid)
            k = int(sheet.plane)
            if k + 1 < len(nodes):
                d_lo = abs(mid - float(nodes[k]))
                d_hi = abs(float(nodes[k + 1]) - mid)
                from rfx._internal._grid_metric import node_tie_tol
                if abs(d_lo - d_hi) <= node_tie_tol(mid, d_local) and d_lo > 0.0:
                    return mid, (k, k + 1)
            return mid, None

        from rfx.geometry.cross_section import section_entries
        sections = section_entries(sim, self.grid)
        for i, entry in enumerate(sim._geometry):
            section = sections[i].shape
            try:
                mat = sim._resolve_material(entry.material_name)
            except KeyError:
                continue
            if mat.sigma < sim._PEC_SIGMA_THRESHOLD:
                continue
            label = f"geometry[{i}]"
            lo, hi = _bounds(section)
            try:
                if id(entry) in products:
                    cells, sheet, wire, solved = products[id(entry)]
                else:
                    solved = continued_conductor_shape(
                        sim, self.grid, section, entry=entry)
                    cells, sheet, wire = classify_pec_entry(
                        solved, self.coords, self.centres, self.cell_sizes,
                        name=entry.material_name, grid=self.grid)
            except self._NARROW_EXCS as exc:
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=section,
                    kind="refused",
                    error=(str(exc) if isinstance(exc, ValueError)
                           else f"{type(exc).__name__}: {exc}"),
                    lo=lo, hi=hi))
                continue
            if cells is not None:
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=section,
                    kind="volume", cells=cells, lo=lo, hi=hi, solved_shape=solved))
            elif sheet is not None:
                mid, tie = _tie(sheet, lo, hi)
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=section,
                    kind="sheet", sheet=sheet, lo=lo, hi=hi, solved_shape=solved,
                    declared_mid=mid, tie_planes=tie))
            else:
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=section,
                    kind="wire", wire=wire, lo=lo, hi=hi))
        for i, tc in enumerate(getattr(sim, "_thin_conductors", ())):
            label = f"thin_conductor[{i}]"
            lo, hi = _bounds(tc.shape)
            if not getattr(tc, "is_pec", False):
                out.append(_EntryRealization(
                    label=label, name=label, shape=tc.shape, kind="lossy",
                    lo=lo, hi=hi, solved_shape=continued_conductor_shape(
                        sim, self.grid, tc.shape, entry=tc)))
                continue
            try:
                if id(tc) in products:
                    _, sheet, _, solved = products[id(tc)]
                else:
                    solved = continued_conductor_shape(
                        sim, self.grid, tc.shape, entry=tc)
                    sheet = sheet_spec_from_shape(
                        solved, self.coords, self.cell_sizes, name=label,
                        lane=self.lane or "", refuse_thick=True, grid=self.grid)
            except ValueError as exc:
                out.append(_EntryRealization(
                    label=label, name=label, shape=tc.shape, kind="refused",
                    error=str(exc), lo=lo, hi=hi))
                continue
            mid, tie = _tie(sheet, lo, hi)
            out.append(_EntryRealization(
                label=label, name=label, shape=tc.shape, kind="sheet",
                sheet=sheet, lo=lo, hi=hi, declared_mid=mid, tie_planes=tie,
                solved_shape=solved))
        self._entries = out
        return out

    def pec_entries(self) -> list:
        """Realized (not refused, not lossy) conductor entries."""
        return [e for e in self.entry_realizations() if e.is_pec]

    def impedance_sheet_entries(self) -> list:
        """f0 sheets for the solved-size check, without classifying them as PEC."""
        from rfx.geometry.rasterize_grid import sheet_spec_from_shape
        from rfx.geometry.smoothing import continued_conductor_shape
        out = []
        for i, tc in enumerate(self.sim._thin_conductors):
            if tc.surface_impedance_f0 is None:
                continue
            bounds = _shape_bounds(tc.shape)
            if bounds is None:
                continue
            lo, hi, _ = bounds
            solved = continued_conductor_shape(self.sim, self.grid, tc.shape, entry=tc)
            try:
                sheet = sheet_spec_from_shape(
                    solved, self.coords, self.cell_sizes, name=f'thin_conductor[{i}]',
                    lane=self.lane or '', refuse_thick=True, grid=self.grid)
            except self._NARROW_EXCS as _traced_exc:
                if is_tracer_error(_traced_exc):
                    not_judged(None, "_validate_cfg_campaign_statics")
                continue
            out.append(_EntryRealization(
                label='add_thin_conductor(surface_impedance_f0=...)',
                subject_label=f'thin_conductor[{i}]',
                name=f'thin_conductor[{i}]', shape=tc.shape, kind='sheet',
                sheet=sheet, lo=lo, hi=hi, solved_shape=solved))
        return out

    def interior_pec_entries(self) -> list:
        """Per-entry conductors for comparisons with the user's drawing.

        Absorber continuation is excluded from cell, footprint and edge
        counts; physical overlap checks retain the full ``pec_entries``.
        """
        import copy
        from dataclasses import replace
        from rfx.geometry.rasterize_grid import interior_lattice_mask

        if hasattr(self, "_interior_entries"):
            return self._interior_entries
        out = []
        for entry in self.pec_entries():
            e = copy.copy(entry)
            e._edges = None
            if e.cells is not None:
                e.cells = interior_lattice_mask(
                    e.cells, self.grid, cell_axes=(True, True, True))
            if e.sheet is not None:
                e.sheet = replace(e.sheet, footprint=interior_lattice_mask(
                    e.sheet.footprint, self.grid))
            if e.wire is not None:
                e.wire = replace(e.wire, edges=tuple(
                    interior_lattice_mask(mask, self.grid,
                                          cell_axes=tuple(a == c for a in range(3)))
                    for c, mask in enumerate(e.wire.edges)))
            out.append(e)
        self._interior_entries = out
        return out

    def rasterize(self, shape) -> np.ndarray:
        """This shape's PRODUCTION node mask on this lane's grid (dielectric
        sampling — node, half-open). Conductors do NOT go through here;
        their realization is :meth:`entry_realizations`."""
        if self.lane == "uniform":
            return np.asarray(shape.mask(self.grid))
        return np.asarray(shape.mask_on_coords(
            self.coords.x, self.coords.y, self.coords.z))

    def congruence_entries(self):
        """``(keyed, unkeyed)`` conductor entries for the congruence check.

        ``keyed`` is ``(realization, lo, hi, exact)`` for every realized
        conductor (geometry PEC entries AND PEC thin conductors — a sheet
        is a conductor with an extent) whose shape reports a bounding box
        through the public ``Shape`` protocol; ``unkeyed`` is the rest,
        which the message must name as skipped. A patterned metal LAYER
        cannot be a ``Box`` (a Box fills its clearance holes), so the
        census keys on bounds, not on ``isinstance(shape, Box)``, and
        carries whether the bounds ARE the shape (``exact``).
        """
        keyed, unkeyed = [], []
        for e in self.interior_pec_entries():
            bounds = _shape_bounds(e.shape)
            if bounds is None:
                unkeyed.append(e)
            else:
                keyed.append((e, bounds[0], bounds[1], bounds[2]))
        return keyed, unkeyed

    def local_spacing(self, axis: int, coord: float) -> float:
        nodes = self.nodes[axis]
        j = int(np.searchsorted(nodes, coord)) - 1
        j = min(max(j, 0), len(self.spacings[axis]) - 1)
        return float(self.spacings[axis][j])

    def sub_lattice_offsets(self, lo_corner) -> list[float]:
        """Fractional cell offset from the node below, zero on an invariant axis."""
        offs = []
        for a in range(3):
            nodes = self.nodes[a]
            if len(nodes) == 1:
                offs.append(0.0)
                continue
            c = float(lo_corner[a])
            j = int(np.searchsorted(nodes, c + 1e-15)) - 1
            j = min(max(j, 0), len(nodes) - 2)
            d = float(nodes[j + 1] - nodes[j])
            o = ((c - float(nodes[j])) / d) % 1.0
            if o > 1.0 - 1e-6 or o < 1e-6:
                o = 0.0
            offs.append(o)
        return offs


def _assemble_realized(self, grid, *, nonuniform: bool):
    """The run's realized conductor set on ``grid``.

    Runs the PRODUCTION assembly with the sheet / wire collectors and
    realizes the ``(Mx, My, Mz)`` edge masks through
    ``rfx.boundaries.pec.realized_pec_edge_masks`` under the run's own
    periodic flags — the same call the solver lanes make. Every
    preflight consumer that needs "where is metal" reads the returned
    :class:`_RealizedPEC` (wall planes, ``is_pec_edge``), never the
    cell mask: a sheet owns no cell, and a volume's far face is a
    wall the cell mask does not mark (the class).
    """
    from rfx.model.conductors import realized_conductors
    return realized_conductors(self, grid, nonuniform=nonuniform, mode="audit")

def _port_realized_edges(self, grid):
    """:class:`_RealizedPEC` for the uniform lane, or ``None``.

    the guide a waveguide port sits in is defined
    by its WALLS, and on the committed sub-aperture fixtures those
    walls are interior PEC shapes, not the domain faces. Read from
    the PRODUCTION assembly so preflight sees the same realized
    conductors the solve does — no geometric re-derivation from the
    shape list. Called once per preflight and threaded through, so
    the assembly runs once. A simulation with no geometry entries at
    all has no interior walls by construction, so the assembly is
    skipped there rather than run to produce an all-False set.
    """
    if (not self._geometry
            and not getattr(self, "_thin_conductors", None)
            and not getattr(self, "_pinned_sheets", None)):
        return None
    try:
        realized = self._assemble_realized(grid, nonuniform=False)
    except (ValueError, TypeError, NotImplementedError, KeyError,
            AttributeError, IndexError):
        return None
    if realized.empty:
        return None
    return realized

def _port_pec_mask(self, grid):
    """Kept name for ``tests/_waveguide_chain_battery_fixture.py``
    (``transverse_spans``), which reads the guide the way preflight
    does. Returns :meth:`_port_realized_edges` — the run's realized
    conductor set, NOT a cell mask (a cell mask cannot carry a
    volume's far face or a sheet); the object is what
    :meth:`_port_transverse_spans` takes. Callers should move to the
    new name; delete this once the fixture does."""
    return self._port_realized_edges(grid)


def context_from_conductors(sim, conductors):
    from rfx.model.realization_context import context_from_conductors as prepared_context
    return prepared_context(sim, conductors)


def release_inspection(sim):
    """Release the model-owned inspection products."""
    sim._pf_campaign_ctx = None
    sim._realized_geometry_record = None
    sim._coax_geometry = None


def clear_inspection_cache(self) -> None:
    """Release this model's cached inspection products.

    Clears preflight's assembly context, the realized-geometry record and
    the coax stamp. They are rebuilt on the next call that needs them.
    They are released automatically while the solve runs. Call this after
    inspecting a model you will not solve. Returns None.
    """
    release_inspection(self)


def _campaign_ctx(self):
    """The shared :class:`_CampaignStaticsContext` for THIS preflight.

    Built once per configuration and reused by every check that reads
    conductor geometry (thin metal on NU, port/probe liveness, NTFF
    walls, the campaign checks, the realization findings):
    the context runs the production assembly, which on a large model
    is the most expensive thing preflight does, and five checks each
    building their own would run it five times. The cache key is the
    shared declaration snapshot. Entry references prevent address reuse;
    concrete profile contents detect in-place edits. Observer-only changes
    also rebuild: both public records describe the same declaration state.
    """
    from rfx._declarations import declaration_key

    key = declaration_key(self)
    cached = getattr(self, "_pf_campaign_ctx", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    ctx = _CampaignStaticsContext(self)
    self._pf_campaign_ctx = (key, ctx)
    return ctx


_assemble_realized.__qualname__ = "_PreflightMixin._assemble_realized"
_port_realized_edges.__qualname__ = "_PreflightMixin._port_realized_edges"
_port_pec_mask.__qualname__ = "_PreflightMixin._port_pec_mask"
_campaign_ctx.__qualname__ = "_PreflightMixin._campaign_ctx"
clear_inspection_cache.__qualname__ = "_PreflightMixin.clear_inspection_cache"


def _same_assembly_failure(sim, exc):
    """Recognize a later validator re-raising the same admission failure."""
    original = sim._campaign_ctx()._assembly_exception
    if type(original) is not type(exc) or str(original) != str(exc):
        return False

    def origin(error):
        tb = error.__traceback__
        if tb is None:
            return None
        while tb.tb_next is not None:
            tb = tb.tb_next
        return tb.tb_frame.f_code, tb.tb_lineno

    return origin(original) is not None and origin(original) == origin(exc)


def _validate_cfg_assembly(self, _w) -> None:
    ctx = self._campaign_ctx()
    if getattr(ctx, "_assembly_lent", False):
        return
    if ctx.error == "traced-mesh":
        return
    if ctx.error is None:
        ctx.realized(defer_warnings=True)
    if ctx._realized is not None and ctx._assembly_exception is None:
        from rfx.model.pad_fill import check_pad_fill
        from rfx.model.source_admission import conductor_operator, refuse_dead_soft_sources, refuse_shorted_lumped_rlc
        try:
            root = ctx._realized
            check_pad_fill(self, root.grid, root.geometry_masks)
            refuse_dead_soft_sources(self, root)
            conformal = getattr(self, '_conformal_run_call', None)
            if conformal is None:
                conformal = bool(self._boundary_spec.conformal_faces())
            operator = getattr(self, '_conductor_operator_call', None)
            refuse_shorted_lumped_rlc(self, root, operator=(operator or
                conductor_operator(conformal=conformal)))
        except ctx._NARROW_EXCS as exc:
            ctx._assembly_exception = exc
    exc = ctx._assembly_exception
    if exc is None:
        return
    if isinstance(exc, (ValueError, NotImplementedError)):
        _w.warn(PreflightWarning(
            f"The model cannot be assembled: {exc}",
            severity="error", code="assembly_refusal", source="_validate_cfg_assembly",
        ))
    else:
        _w.warn(PreflightWarning(
            f"The model assembly raised {type(exc).__name__}: {exc}; "
            "this is a defect, please report it.",
            severity="error", code="assembly_fault", source="_validate_cfg_assembly",
        ))


_validate_cfg_assembly.__qualname__ = "_PreflightMixin._validate_cfg_assembly"
