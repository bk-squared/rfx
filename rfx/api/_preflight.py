"""Preflight and configuration-validation methods for :class:`Simulation`.

Import contract (Part B Stage 1a refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

The methods here were moved verbatim out of ``rfx/api/__init__.py``'s
``class Simulation`` body. They are pure structural relocations — same
indentation, decorators, signatures, and logic. ``Simulation`` inherits
``_PreflightMixin`` so every method below remains a bound method on
``Simulation`` instances; ~79 test call-sites are unaffected.
"""

from __future__ import annotations

import json
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 0).
#
# The report/result types and the formatting + absorber leaves that used to sit
# here now live in ``rfx.preflight._common``, moved verbatim. They are re-bound
# as module globals of THIS module because three things depend on that
# namespace and none of them would fail loudly if it quietly emptied out:
#
#   * 17 files do ``from rfx.api._preflight import <name>``, including
#     ``rfx/sparams/_common.py`` (``PreflightWarning``) and
#     ``validation/research/convergence_floor/fixture.py``;
#   * ``tests/unit/preflight/test_preflight_rasterization.py`` and
#     ``tests/unit/ports/test_msl_clearance_diagnostic.py`` hold the MODULE
#     OBJECT and ``monkeypatch.setattr`` a global on it across 9 call sites, so
#     a name that stops living here aims those patches at a module nobody
#     reads. (None of the 9 patched names is moved by this leg; the re-export
#     is what keeps that true for the readers that stayed behind.)
#   * ``_PreflightMixin`` below calls every one of these by BARE NAME, so they
#     have to resolve as globals of this module.
#
# ``PreflightWarning`` in particular stays ONE class object reached by two
# names — re-exported, never redefined — because ``pytest.warns`` and
# ``isinstance`` across the suite compare identity.
#
# Listed explicitly, not ``import *``: the surface is the contract, and
# ``tests/locks/test_preflight_split_snapshot.py`` pins it by set equality in
# both directions. 13 names move and 13 names come back, so the module
# namespace that lock pins is exactly as wide after this leg as before it.
#
# Leg 2 added a 14th, ``_sorted_box_corners``. It is a pure leaf with THREE
# readers that this split separates: ``_validate_cfg_sheet_cavity_thickness``
# (leaving with leg 2) and the two module-level ones below,
# ``_shape_bounds`` and ``_CampaignStaticsContext``, which stay here until
# the realization leg. A leg module may not import from this facade -- that
# is the cycle the import contract forbids -- so the leaf had to move
# somewhere both sides can reach, and ``_common`` is the module that exists
# for exactly that. Put in ``pec_geometry`` instead it would have made the
# realization leg import from a port-family peer for a nine-line helper.
# ---------------------------------------------------------------------------
from rfx.preflight._common import (
    _fmt_len,
    _fmt_signed,
    _fmt_freq,
    _absorber_boundary_for_axis,
    _axis_pad_thickness_m,
    _coord_in_absorber,
    _ABSORBER_PROXIMITY_CELLS,
    _coord_near_absorber,
    _sorted_box_corners,
    PreflightWarning,
    PreflightErrorWarning,
    PreflightConfigError,
    PreflightIssue,
    PreflightReport,
)


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 1).
#
# The MSL probe-clearance geometry block and the MSL check-2c constants moved
# verbatim to ``rfx.preflight.msl``. They are re-bound as module globals of
# THIS module for the three reasons the leg-0 block above lists, plus a fourth
# that belongs to this family alone:
#
#   * ``_validate_forward_sparameter_request`` STAYS on ``_PreflightMixin``
#     below and calls ``msl_source_near_field_standoff_cells`` by BARE NAME,
#     so that name must resolve as a global of this module or the narrow
#     ``forward(port_s11_freqs=)`` path raises NameError.
#
# ``_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY`` and
# ``_MSL_REALIZED_THICKNESS_Z0_BUDGET`` have no reader inside the package at
# all: ``tests/unit/ports/test_msl_port_preflight.py`` imports them from here
# and nothing else does, which is precisely why a split cannot drop them.
#
# 11 names move out and 11 come back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.msl import (
    _MSL_REALIZED_THICKNESS_Z0_SENSITIVITY,
    _MSL_REALIZED_THICKNESS_Z0_BUDGET,
    _MSL_REALIZED_THICKNESS_TOL,
    MSL_EPS_EFF_PROXY,
    msl_min_probe_clearance,
    _MSL_NEAR_FIELD_STANDOFF_H_SUB,
    _MSL_NEAR_FIELD_MIN_OFFSET_CELLS,
    msl_source_near_field_standoff_cells,
    msl_nearest_downstream_reflector,
    msl_probe_clearance_for_port,
    msl_absorber_compliant_offset_max,
)


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 2).
#
# The five issue-#703 gate constants and the banner that derives them moved
# verbatim to ``rfx.preflight.pec_geometry``. They are re-bound as module
# globals of THIS module for the reasons the leg-0 block above lists -- the
# namespace is pinned by set equality, and 17 files import names from here.
#
# What the re-export does NOT do, and must not be mistaken for: it is not the
# patch point. ``tests/unit/preflight/test_preflight_rasterization.py`` proves
# three of these gates load-bearing by mutating them in both directions, and
# their readers are the check bodies, which resolve them in
# ``rfx.preflight.pec_geometry``'s globals. A patch aimed here would rebind a
# name no reader consults and BOTH arms of each such test would pass on an
# unmutated gate -- the failure this split has to avoid quietly. Those four
# tests were repointed at the leg module in the same commit that moved the
# bodies; falsification of the repoint is recorded there.
#
# Leg 2 takes 6 module-level names out of this file: these 5 and
# ``_sorted_box_corners``, which comes back through the ``_common`` block
# above rather than this one. 6 out, 6 back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.pec_geometry import (
    _CONGRUENCE_EXTENT_QUANTUM_M,
    _CONGRUENCE_SPREAD_TOL_EDGES,
    _CAVITY_THICKNESS_TOL,
    _OFF_LATTICE_EDGE_TOL,
    _CAMPAIGN_MAX_OFFENDERS,
)


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 3).
#
# The three module-level waveguide leaves -- ``_waveguide_skipped_note``,
# ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` and ``resolve_waveguide_port_freqs`` --
# moved verbatim to ``rfx.preflight.waveguide``. They are re-bound as module
# globals of THIS module for the reasons the leg-0 block above lists, and for
# one that is specific to this family and does NOT go away when the check
# bodies follow:
#
#   * ``preflight_sparameters`` STAYS on ``_PreflightMixin`` below --
#     permanently, it is the per-calculator routing entry point rather than a
#     family check -- and reads BOTH ``resolve_waveguide_port_freqs`` and
#     ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` by BARE NAME when it builds the
#     waveguide setup-audit call. Drop either from this namespace and
#     ``preflight_sparameters(calculator="waveguide")`` raises NameError.
#
# ``_waveguide_skipped_note``'s two readers are the layout and record audits,
# which call it by bare name and are still on the mixin as this block lands;
# they leave in the commit that moves the bodies, and resolve it in
# ``rfx.preflight.waveguide``'s globals from then on. The re-export is what
# keeps both arrangements working, and it is required either way:
# ``tests/unit/preflight/test_waveguide_setup_audits.py`` imports
# ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` from HERE and ``rfx/sparams/waveguide.py``
# imports ``resolve_waveguide_port_freqs`` from HERE.
#
# Unlike the leg-2 block, this one is also a legitimate patch point -- there
# is simply nothing patching it. An AST sweep over ``tests/ rfx/ validation/
# scripts/ examples/`` resolving aliases, ``importlib`` forms and string-form
# targets finds no ``monkeypatch``/``setattr``/``mock.patch`` site on any of
# the three names, which is why ``rfx/sparams/waveguide.py`` is left importing
# the resolver from here rather than being repointed at the leg module the way
# leg 1 had to repoint its own S-matrix readers.
#
# 3 names move out and 3 come back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.waveguide import (
    _waveguide_skipped_note,
    WAVEGUIDE_DEFAULT_NUM_PERIODS,
    resolve_waveguide_port_freqs,
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
            NotImplementedError):
        return None
    if blo.shape != (3,) or bhi.shape != (3,):
        return None
    return np.minimum(blo, bhi), np.maximum(blo, bhi), False


def _local_cell(profile, lo, hi, fallback):
    """Coarsest cell a body spans on one axis (#743).

    ``profile`` is a per-cell size array whose cumulative sum gives node
    positions from the padded array's origin; ``lo``/``hi`` are the body's
    physical bounds. Returns ``fallback`` when there is no profile or the
    span selects no cell, so callers keep their previous behaviour on a
    uniform axis.
    """
    if profile is None:
        return fallback
    import numpy as _np
    d = _np.asarray(profile, dtype=float)
    edges = _np.concatenate([[0.0], _np.cumsum(d)])
    inside = (edges[1:] > min(lo, hi)) & (edges[:-1] < max(lo, hi))
    if not inside.any():
        return fallback
    return float(d[inside].max())


def _shift_back_np(arr, ax: int, periodic: bool):
    """``arr[i-1]`` along ``ax`` under the #689 convention (numpy twin of
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
    cells = None if pec_mask is None else jnp.asarray(pec_mask)
    edges = realized_pec_edge_masks(cells, sheets, wires, periodic)
    return tuple(np.asarray(m, dtype=bool) for m in edges)


_H_LOOP = {
    # H component -> the four E edges of its curl loop, as (component,
    # di, dj, dk) offsets from the H index. Hx[i,j,k] sits at
    # (x_i, y_{j+1/2}, z_{k+1/2}); its loop is Ey[i,j,k], Ey[i,j,k+1],
    # Ez[i,j,k], Ez[i,j+1,k]; cyclically for Hy, Hz.
    "hx": ((1, 0, 0, 0), (1, 0, 0, 1), (2, 0, 0, 0), (2, 0, 1, 0)),
    "hy": ((2, 0, 0, 0), (2, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 1)),
    "hz": ((0, 0, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0), (1, 1, 0, 0)),
}


def _component_is_dead(edges, component: str, idx) -> bool:
    """Whether a field component at grid index ``idx`` is frozen by the
    realized PEC edges — the one rule for ``port_in_pec`` (#929).

    An E component is dead iff its OWN edge is PEC (the contract's one
    sentence). An H component is dead iff ALL FOUR E edges of its curl
    loop are PEC: then ``curl E = 0`` around it every step and it never
    moves. Half a cell above a sheet only one loop edge is PEC, so H there
    is live — the sheet exemption falls out of the rule instead of a
    thickness heuristic; inside a one-cell volume all four are PEC and H
    is frozen, which is what a volume declaration means.
    """
    comp = component.lower()
    i, j, k = (int(v) for v in idx)
    shape = edges[0].shape
    if comp in ("ex", "ey", "ez"):
        c = "xyz".index(comp[1])
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            return False
        return bool(edges[c][i, j, k])
    loop = _H_LOOP.get(comp)
    if loop is None:
        return False
    for c, di, dj, dk in loop:
        ii, jj, kk = i + di, j + dj, k + dk
        if not (0 <= ii < shape[0] and 0 <= jj < shape[1]
                and 0 <= kk < shape[2]):
            return False
        if not edges[c][ii, jj, kk]:
            return False
    return True


class _RealizedPEC:
    """The run's realized conductor set, read once from the production
    assembly with the #931 collectors: volume cells (``pec_mask``),
    :class:`~rfx.boundaries.pec.SheetSpec` sheets, wires, and the
    ``(Mx, My, Mz)`` edge masks the solver zeroes — host numpy.
    """

    def __init__(self, *, lane, grid, materials, pec_mask, sheets, wires,
                 periodic, sheet_specs=()):
        self.lane = lane
        self.grid = grid
        self.materials = materials
        self.pec_mask = (None if pec_mask is None
                         else np.asarray(pec_mask, dtype=bool))
        self.sheets = list(sheets)
        self.wires = list(wires)
        self.sheet_specs = list(sheet_specs)  # observational f0 geometry, not PEC
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
        # thin delegate; named differently from the owner so the single-owner
        # lock (tests/contracts/test_pec_single_owner_lock.py) reads one rule
        from rfx.boundaries.pec import edge_is_pec
        return edge_is_pec(self.edges, component, i, j, k)

    def wall_nodes_on_plane(self, axis: int, k: int) -> np.ndarray:
        return _wall_nodes_on_plane(self.edges, axis, k, self.periodic)

    def component_is_dead(self, component: str, idx) -> bool:
        return _component_is_dead(self.edges, component, idx)


class _EntryRealization:
    """What ONE conductor declaration realizes as under #931.

    ``kind`` is ``"volume"`` (centre-sampled cells), ``"sheet"`` (a
    :class:`SheetSpec`: one plane, closed footprint, no cell), ``"wire"``
    (a filament), ``"lossy"`` (a sigma-fold thin conductor — fenced, not
    PEC realization) or ``"refused"`` (the classifier raised; ``error``
    carries its message). Nothing here is inferred from an extent: the
    kind is what the shared classifier says.
    """

    def __init__(self, *, label, name, shape, kind, cells=None, sheet=None,
                 wire=None, error=None, lo=None, hi=None, declared_mid=None,
                 tie_planes=None):
        self.label = label
        self.name = name
        self.shape = shape
        self.kind = kind
        self.cells = None if cells is None else np.asarray(cells, dtype=bool)
        self.sheet = sheet
        self.wire = wire
        self.error = error
        self.lo = lo
        self.hi = hi
        self.declared_mid = declared_mid   # sheet: declared mid-plane coord
        self.tie_planes = tie_planes       # sheet: (k_lo, k_hi) on an exact tie
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
            sheet_spec_from_shape,
        )
        nodes = [np.asarray(ctx.coords.x), np.asarray(ctx.coords.y),
                 np.asarray(ctx.coords.z)]
        nodes[axis] = nodes[axis] + np.float64(shift_m)
        coords = GridCoords(x=nodes[0], y=nodes[1], z=nodes[2],
                            shape=tuple(ctx.grid.shape))
        shape = tuple(ctx.grid.shape)
        try:
            if self.kind == "volume":
                centres = cell_centres_from_nodes(coords, ctx.cell_sizes)
                cells = np.asarray(pec_volume_cell_mask(self.shape, centres),
                                   dtype=bool)
                if not cells.any():
                    return None
                edges = _realized_edges_np(cells, (), (), ctx.periodic, shape)
            elif self.kind == "sheet":
                sp = sheet_spec_from_shape(
                    self.shape, coords, ctx.cell_sizes,
                    normal_axis=int(self.sheet.normal_axis), name=self.name)
                edges = _realized_edges_np(None, [sp], (), ctx.periodic, shape)
            else:
                return None
        except (ValueError, TypeError, IndexError):
            return None
        return int(sum(int(m.sum()) for m in edges))

    def footprint_on_plane(self, axis: int, k: int, periodic, shape):
        """In-plane node map of this entry's wall on plane ``k``."""
        return _wall_nodes_on_plane(self.edges(periodic, shape), axis, k,
                                    periodic)


class _CampaignStaticsContext:
    """Shared lazily-built state for the issue-#703 campaign checks and
    every other preflight check that reads conductor geometry.

    Built once per CONFIGURATION through :meth:`_PreflightMixin
    ._campaign_ctx` (a cache keyed on the entry identities and the mesh
    parameters, so an ``add()`` after a preflight misses it — a plain
    instance cache would have gone stale). The grid, sample coordinates,
    cell centres and the realized conductor set come from the PRODUCTION
    builders — the check must see the run's realization, not a hand model
    of it (the ``_validate_cfg_graded_box_rasterization`` lesson: a hand
    model of the sampling diverged from the rasterizer on exactly the
    cases it existed to catch). Under #931 that means one thing: every
    conductor number a check prints is read from
    ``rfx.boundaries.pec.realized_pec_edge_masks`` (through
    :meth:`realized` for the whole model and :meth:`entry_realizations`
    per declaration), never from a cell mask or a bounding box.
    """

    _NARROW_EXCS = (ValueError, TypeError, NotImplementedError, KeyError,
                    AttributeError, IndexError)

    def __init__(self, sim):
        self.sim = sim
        self.error: str | None = None
        self.lane: str | None = None
        self.grid = None
        self.coords = None       # production GridCoords (E-node samples)
        self.centres = None      # production primal-cell centres (§1.1)
        self.cell_sizes = None   # per-axis per-cell size arrays
        self.nodes = None        # 3x float64 node-position arrays
        self.spacings = None     # 3x float64 per-cell spacing arrays
        self.periodic = (False, False, False)
        self._realized = None
        self._entries = None
        self.assembly_error: str | None = None
        self._build()

    def _build(self) -> None:
        sim = self.sim
        profiles = (sim._dx_profile, sim._dy_profile, sim._dz_profile)
        if is_tracer(sim._dx) or any(
                p is not None and is_tracer(p) for p in profiles):
            # Same precedent as _validate_cfg_graded_box_rasterization:
            # a traced mesh has no concrete node positions to check.
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
                # ONE node line: the classifier's own coordinates (the f64
                # spine when the grid carries one). Re-deriving nodes from
                # the float32 cell-size store put preflight's printed
                # positions ~1e-8 relative off the planes the classifier
                # resolved — enough to miss an exact half-cell tie.
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
        except self._NARROW_EXCS as exc:  # malformed config; preflight
            self.error = f"{type(exc).__name__}: {exc}"  # must not crash

    # ------------------------------------------------------------------
    # The realized conductor set (whole model) and per-entry realization
    # ------------------------------------------------------------------

    def realized(self):
        """:class:`_RealizedPEC` from the PRODUCTION assembly (with the
        #931 sheet/wire collectors), once; ``None`` when the assembly
        raised (``assembly_error`` says why)."""
        if self._realized is None and self.assembly_error is None:
            try:
                self._realized = self.sim._assemble_realized(
                    self.grid, nonuniform=(self.lane == "nonuniform"))
            except self._NARROW_EXCS as exc:
                self.assembly_error = f"{type(exc).__name__}: {exc}"
        return self._realized

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
            classify_pec_entry, sheet_spec_from_shape, _local_cell,
        )
        from rfx.materials.thin_conductor import sheet_bounds
        sim = self.sim
        out = []

        def _bounds(shape):
            lo, hi = _sorted_box_corners(shape)
            if lo is not None:
                return lo, hi
            try:
                blo, bhi = sheet_bounds(shape)
            except self._NARROW_EXCS:
                return None, None
            if blo is None or bhi is None:
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
            nodes = self.nodes[a]
            d_local = _local_cell(nodes, self.cell_sizes[a], mid)
            k = int(sheet.plane)
            if k + 1 < len(nodes):
                d_lo = abs(mid - float(nodes[k]))
                d_hi = abs(float(nodes[k + 1]) - mid)
                if abs(d_lo - d_hi) <= 1e-9 * d_local and d_lo > 0.0:
                    return mid, (k, k + 1)
            return mid, None

        for i, entry in enumerate(sim._geometry):
            try:
                mat = sim._resolve_material(entry.material_name)
            except KeyError:
                continue
            if mat.sigma < sim._PEC_SIGMA_THRESHOLD:
                continue
            label = f"geometry[{i}]"
            lo, hi = _bounds(entry.shape)
            try:
                cells, sheet, wire = classify_pec_entry(
                    entry.shape, self.coords, self.centres, self.cell_sizes,
                    name=entry.material_name)
            except self._NARROW_EXCS as exc:
                # A shape that cannot be rasterized is a FINDING, not a
                # crash: preflight's job is to report. One unplaceable
                # conductor used to kill the whole per-entry loop, so every
                # other entry lost its realization too (measured on the
                # #685 `_NoBBox` fixture: NotImplementedError out of
                # classify_pec_entry, no advisory at all).
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=entry.shape,
                    kind="refused",
                    error=(str(exc) if isinstance(exc, ValueError)
                           else f"{type(exc).__name__}: {exc}"),
                    lo=lo, hi=hi))
                continue
            if cells is not None:
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=entry.shape,
                    kind="volume", cells=cells, lo=lo, hi=hi))
            elif sheet is not None:
                mid, tie = _tie(sheet, lo, hi)
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=entry.shape,
                    kind="sheet", sheet=sheet, lo=lo, hi=hi,
                    declared_mid=mid, tie_planes=tie))
            else:
                out.append(_EntryRealization(
                    label=label, name=entry.material_name, shape=entry.shape,
                    kind="wire", wire=wire, lo=lo, hi=hi))
        for i, tc in enumerate(getattr(sim, "_thin_conductors", ())):
            label = f"thin_conductor[{i}]"
            lo, hi = _bounds(tc.shape)
            if not getattr(tc, "is_pec", False):
                out.append(_EntryRealization(
                    label=label, name=label, shape=tc.shape, kind="lossy",
                    lo=lo, hi=hi))
                continue
            try:
                sheet = sheet_spec_from_shape(
                    tc.shape, self.coords, self.cell_sizes, name=label,
                    lane=self.lane or "", refuse_thick=True)
            except ValueError as exc:
                out.append(_EntryRealization(
                    label=label, name=label, shape=tc.shape, kind="refused",
                    error=str(exc), lo=lo, hi=hi))
                continue
            mid, tie = _tie(sheet, lo, hi)
            out.append(_EntryRealization(
                label=label, name=label, shape=tc.shape, kind="sheet",
                sheet=sheet, lo=lo, hi=hi, declared_mid=mid, tie_planes=tie))
        self._entries = out
        return out

    def pec_entries(self) -> list:
        """Realized (not refused, not lossy) conductor entries."""
        return [e for e in self.entry_realizations() if e.is_pec]

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
        for e in self.pec_entries():
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
        """Per-axis fractional offset (in cells) of a corner from the node below."""
        offs = []
        for a in range(3):
            nodes = self.nodes[a]
            c = float(lo_corner[a])
            j = int(np.searchsorted(nodes, c + 1e-15)) - 1
            j = min(max(j, 0), len(nodes) - 2)
            d = float(nodes[j + 1] - nodes[j])
            o = ((c - float(nodes[j])) / d) % 1.0
            # A corner within float rounding of a node is ON the node
            # (cumsum-built NU nodes carry ~1e-15 m rounding, which the
            # bare modulo would print as 1.000 cells instead of 0).
            if o > 1.0 - 1e-6 or o < 1e-6:
                o = 0.0
            offs.append(o)
        return offs


class _PreflightMixin:
    """Preflight / validation methods mixed into :class:`Simulation`."""

    @staticmethod
    def _validate_tfsf_vacuum_boundary(materials: MaterialArrays, tfsf_cfg) -> None:
        """Ensure the TFSF boundary planes remain vacuum.

        The TFSF correction assumes vacuum on and immediately adjacent to
        the TFSF boundaries. Fail loudly instead of allowing silently wrong
        scattered fields. For the 4-edge Method-B box this means the y planes
        as well as the x planes (issue #471 F5: the x-only check let a PEC
        strip on the y_lo plane pass silently); the check for that path lives
        with the source in ``tfsf_oblique_open.validate_vacuum_boundary`` so
        ``compute_rcs`` can run the identical check.
        """
        from rfx.sources.tfsf import is_tfsf_methodB

        if is_tfsf_methodB(tfsf_cfg):
            from rfx.sources.tfsf_oblique_open import validate_vacuum_boundary

            validate_vacuum_boundary(materials, tfsf_cfg)
            return

        boundary_slices = (
            ("x_lo-1", slice(tfsf_cfg.x_lo - 1, tfsf_cfg.x_lo)),
            ("x_lo", slice(tfsf_cfg.x_lo, tfsf_cfg.x_lo + 1)),
            ("x_hi", slice(tfsf_cfg.x_hi, tfsf_cfg.x_hi + 1)),
            ("x_hi+1", slice(tfsf_cfg.x_hi + 1, tfsf_cfg.x_hi + 2)),
        )

        for plane_name, xs in boundary_slices:
            eps = np.asarray(materials.eps_r[xs, :, :])
            sigma = np.asarray(materials.sigma[xs, :, :])
            mu = np.asarray(materials.mu_r[xs, :, :])
            if not (
                np.allclose(eps, 1.0)
                and np.allclose(sigma, 0.0)
                and np.allclose(mu, 1.0)
            ):
                raise ValueError(
                    "TFSF plane-wave source requires vacuum on and adjacent to "
                    f"the TFSF x boundaries; non-vacuum material found at {plane_name}"
                )

    def _validate_run_sparameter_request(
        self,
        *,
        compute_s_params: bool | None,
        s_param_freqs,
        s_param_n_steps: int | None,
        devices: list | None = None,
    ) -> None:
        """Reject explicit ``run`` S-parameter requests outside its contract."""

        requested = (
            compute_s_params is True
            or s_param_freqs is not None
            or s_param_n_steps is not None
        )
        if not requested:
            return

        port_entries = self._port_sparameter_entries()
        source_only_entries = [pe for pe in self._ports if pe.impedance == 0.0]
        messages: list[str] = []

        if self._msl_ports:
            messages.append(
                "add_msl_port(...) uses compute_msl_s_matrix(); "
                "run(compute_s_params=True) does not include MSL ports in "
                "Result.s_params"
            )
        if self._waveguide_ports:
            messages.append(
                "add_waveguide_port(...) uses compute_waveguide_s_matrix() "
                "for the full S-matrix; run() may return per-port "
                "result.waveguide_sparams but not Result.s_params"
            )
        if self._floquet_ports:
            messages.append(
                "add_floquet_port(...) is experimental and has no "
                "claims-bearing run(compute_s_params=True) S-matrix path"
            )
        if self._tfsf is not None:
            messages.append(
                "add_tfsf_source(...) is a plane-wave source, not a port"
            )
        if self._coaxial_ports:
            messages.append(
                "add_coaxial_port(...) is not wired into run(compute_s_params=True); "
                "use Simulation.compute_coaxial_s_matrix(...) (experimental TEM "
                "plane-source API) or add_port(extent=...) for the current "
                "probe-feed S-parameter path"
            )

        if not port_entries:
            if source_only_entries:
                messages.append(
                    "add_source(...) / add_polarized_source(...) are "
                    "source-only observables and cannot populate "
                    "Result.s_params"
                )
            detail = "; ".join(messages) if messages else (
                "register at least one add_port(...) impedance port"
            )
            raise ValueError(
                "run(compute_s_params=True) computes Result.s_params only "
                f"for add_port(...) lumped or wire ports; {detail}."
            )

        if messages:
            raise NotImplementedError(
                "run(compute_s_params=True) has a single result schema for "
                "add_port(...) lumped/wire ports. Mixed or specialized port "
                "families must use their documented calculators: "
                + "; ".join(messages)
                + "."
            )

        if self._solver == "adi":
            raise NotImplementedError(
                "run(compute_s_params=True) is not supported with "
                "solver='adi'; use the uniform Yee solver."
            )
        if devices is not None and len(devices) > 1:
            raise NotImplementedError(
                "run(compute_s_params=True) is not supported on the "
                "distributed multi-device path; run a single-device "
                "uniform S-parameter calculation."
            )
        if self._refinement is not None:
            if source_only_entries:
                raise NotImplementedError(
                    "subgrid compute_s_params ignores ordinary "
                    "add_source(...) entries like the uniform S-matrix "
                    "extractor; remove source-only entries and drive through "
                    "add_port(...) waveforms."
                )
            if any(pe.waveform is None for pe in port_entries):
                raise ValueError(
                    "subgrid compute_s_params needs a waveform "
                    "on every impedance port so each port can be driven in "
                    "turn. Pass waveform=... even for ports whose main-run "
                    "excite flag is False."
                )

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if is_nonuniform and any(pe.extent is None for pe in port_entries):
            raise NotImplementedError(
                "run(compute_s_params=True) on a non-uniform mesh is wired "
                "only for add_port(..., extent=...) WirePort extraction. "
                "Single-cell lumped-port S-parameters require the uniform "
                "reference lane."
            )

        # The lumped/wire S-parameter extractor runs a SEPARATE eager FDTD
        # re-run that does NOT apply periodic boundaries, so it would silently
        # ignore set_periodic_axes() and return an S-matrix for the wrong
        # (non-periodic) boundary-value problem (issue #206). Fail loudly
        # instead of returning silently-wrong S-parameters.
        if self._periodic_axes and port_entries:
            raise NotImplementedError(
                "run(compute_s_params=True) for lumped/wire add_port(...) does "
                "not honor periodic axes: the S-parameter extraction re-run uses "
                "non-periodic boundaries, so the returned S-matrix would silently "
                f"ignore set_periodic_axes({self._periodic_axes!r}). Remove the "
                "periodic axes for the S-parameter run, or use a port family that "
                "supports periodicity (e.g. a Floquet port)."
            )

    def _validate_forward_sparameter_request(self) -> None:
        """Reject ``forward(port_s11_freqs=...)`` outside its narrow path."""

        port_entries = self._port_sparameter_entries()
        messages: list[str] = []
        if self._msl_ports:
            messages.append("MSL ports use compute_msl_s_matrix()")
            # Near-field guard (issue #80): probe 0 must clear the source
            # FRINGING transient (~5·h_sub), which decays over a few substrate
            # thicknesses, not over λ. Inside it the V·I-split S11 of a high-Q
            # resonant load is corrupted (the issue-#80 edge-fed patch read
            # |S11|=8.94/1.11 at offset~5; passive ~0.99 once cleared). The
            # default n_probe_offset already floors to max(λ-clearance,
            # 5·h_sub/dx); this warns when an EXPLICIT value under-provisions it.
            for pe in self._msl_ports:
                # Issue #823: the SAME predicate _check_msl_port_geometry's
                # check 5 and add_msl_port's auto floor use. This site used to
                # spell it as a raw float comparison (< 5.0*h/dx) while the
                # other two rounded to cells, so the three disagreed in a band
                # (5h/dx = 15.2, offset 15: rounded compliant, float violating).
                # One helper, one answer.
                _nf_cells = msl_source_near_field_standoff_cells(
                    float(pe.height), float(self._dx) if self._dx else 0.0)
                if (self._dx and pe.n_probe_offset is not None
                        and int(pe.n_probe_offset) < _nf_cells):
                    messages.append(
                        f"MSL port {pe.name!r}: n_probe_offset="
                        f"{pe.n_probe_offset} sits within the source fringing "
                        f"transient ({_nf_cells} cells = max(3, round("
                        f"5·h_sub/dx))); probe 0 may corrupt the V·I-split S11 "
                        f"of a high-Q resonant load (issue #80) — increase "
                        f"n_probe_offset or leave it None for the safe default."
                    )
        if self._waveguide_ports:
            messages.append("waveguide ports use compute_waveguide_s_matrix()")
        if self._floquet_ports:
            messages.append(
                "Floquet ports are experimental and have no forward S11 path"
            )
        if self._tfsf is not None:
            messages.append("TFSF is a plane-wave source, not a port")
        if self._coaxial_ports:
            messages.append(
                "coaxial ports are not wired into forward(port_s11_freqs=...); "
                "use Simulation.compute_coaxial_s_matrix(...) for the "
                "experimental coaxial S-matrix path"
            )
        if not port_entries:
            source_only = any(pe.impedance == 0.0 for pe in self._ports)
            if source_only:
                messages.append("add_source(...) is not an impedance port")
            detail = "; ".join(messages) if messages else (
                "register add_port(...) first"
            )
            raise ValueError(
                "forward(port_s11_freqs=...) computes S11 only for "
                f"add_port(...) lumped or wire ports on the uniform "
                f"single-device path; {detail}."
            )
        if messages:
            raise NotImplementedError(
                "forward(port_s11_freqs=...) cannot be combined with "
                "specialized or non-port excitation families: "
                + "; ".join(messages)
                + "."
            )

    def _validate_mesh_quality(self) -> None:
        """Pre-simulation mesh quality check (P0).

        Scans all geometry elements against the grid cell size and warns
        about under-resolved features. Prevents silent garbage results
        from mesh-related setup errors.
        """
        import warnings as _w

        # Tracer-valued profiles (mesh-as-design-variable gradient) cannot
        # participate in host-side min/len/indexing. Advisory warnings
        # are skipped in that case — correctness is preserved downstream.
        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            return

        dx = self._dx
        if dx is None:
            dx = C0 / self._freq_max / 20.0

        # Determine minimum cell size per axis — use profile min when
        # non-uniform xy is active, so we don't flag features that are
        # actually well-resolved in their local fine-mesh region.
        min_dx = float(min(self._dx_profile)) if self._dx_profile is not None else dx
        min_dy = float(min(self._dy_profile)) if self._dy_profile is not None else dx
        if self._dz_profile is not None:
            min_dz = min(self._dz_profile)
        else:
            min_dz = dx

        for entry in self._geometry:
            shape = entry.shape
            mat_name = entry.material_name

            # Imported CAD mesh (#358): warn when the thinnest dimension (bbox extent — a
            # tessellation-independent proxy, e.g. a plate/wall thickness) falls below ~2 cells;
            # rasterisation is a cell-centre staircase, so a sub-2-cell dimension is lost or
            # misregistered (the #330 thin-conductor class).
            if hasattr(shape, "min_feature_size"):
                try:
                    feat = float(shape.min_feature_size())
                    cell = float(min(min_dx, min_dy, min_dz))
                    if 0.0 < feat < 2.0 * cell:
                        _w.warn(PreflightWarning(
                            f"Imported mesh (material '{mat_name}') thinnest dimension "
                            f"~{feat * 1e3:.3f} mm is below 2 cells "
                            f"(~{2 * cell * 1e3:.3f} mm at dx={cell * 1e3:.3f} mm); it will "
                            f"be lost or staircased by rasterisation. Refine the mesh region "
                            f"(finer dx / subpixel smoothing) or thicken the feature.",
                            code="mesh_import_underresolved", source="_validate_mesh_quality"))
                except (ValueError, AttributeError, TypeError):
                    pass

            # Get bounding box dimensions
            if hasattr(shape, "bounding_box"):
                try:
                    c1, c2 = shape.bounding_box()
                    dims = [abs(c2[i] - c1[i]) for i in range(3)]
                except (NotImplementedError, TypeError):
                    continue
            else:
                continue

            # Score against the LOCAL cell where this body actually sits,
            # not the global minimum. Using the finest cell anywhere made
            # the check vacuously green exactly where grading hurts — a
            # body in a coarse region judged by a fine cell it never sees
            # (#743). Falls back to the global minimum when a profile has
            # no usable extent for this body.
            cell_sizes = [
                _local_cell(self._dx_profile, c1[0], c2[0], min_dx),
                _local_cell(self._dy_profile, c1[1], c2[1], min_dy),
                _local_cell(self._dz_profile, c1[2], c2[2], min_dz),
            ]

            # FP1 refinement (2026-05-06): the partial-volume warning
            # at 3-5 cells along one axis is meaningful only for actual
            # *volumes* (≥3 cells in every axis).  A thin strip
            # (e.g. an MSL trace at LX × W_trace × dx → many × 4.7 × 1
            # cells) is a sheet, not a volume, and the per-axis 4.7
            # signal must not fire.  Compute cells on every axis up
            # front and gate the volume branch on the minimum.
            cells_per_axis = [
                (dim / cell) if cell > 0 else float("inf")
                for dim, cell in zip(dims, cell_sizes)
            ]
            is_thin_along_some_axis = min(cells_per_axis) < 3.0

            for axis, (dim, cell) in enumerate(zip(dims, cell_sizes)):
                mat = self._resolve_material(mat_name)
                is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                axis_name = "xyz"[axis]
                if dim <= 0:
                    # Zero-thickness geometry. For a PEC Box that IS the
                    # sheet declaration (lattice ownership contract #931
                    # §1.5: "zero thickness is a statement of intent") —
                    # it is realized on one node plane and reported by
                    # ``sheet_plane_realized``, so nothing to say here.
                    # For a dielectric a zero-extent Box samples no node
                    # and rasterizes to nothing.
                    if is_pec:
                        continue
                    _w.warn(
                        PreflightWarning(
                            f"Zero-thickness dielectric '{mat_name}' along "
                            f"{axis_name}-axis rasterizes to nothing (the "
                            f"node sampler is half-open [lo, hi), so a "
                            f"zero-extent span contains no node). Give it "
                            f"at least one cell of thickness "
                            f"({_fmt_len(cell)}). (A zero-thickness PEC Box "
                            f"is the sheet declaration and is fine.)",
                            code="mesh_resolution",
                            source="_validate_mesh_quality",
                        ),
                        stacklevel=3,
                    )
                elif dim < cell:
                    cells_count = dim / cell
                    # #931 §1.5: a PEC Box (or any PEC shape with a
                    # bounding box) thinner than one local cell is REFUSED
                    # at assembly — nothing is inferred from raster
                    # thickness. This advisory only documents the rule;
                    # the ``pec_box_subcell`` ERROR carries the refusal.
                    hint = (
                        " A PEC shape passed to sim.add() is a VOLUME "
                        "(realized with both faces and a shorted interior, "
                        "lattice ownership contract #931 §1.2), and a "
                        "volume thinner than one local cell is refused at "
                        "assembly (§1.5) rather than snapped to a plane. "
                        "If this is foil (a ground plane, patch, trace), "
                        "declare a SHEET: a zero-thickness Box via add(), "
                        "or add_thin_conductor(shape) — realized on the "
                        "node plane nearest its mid-plane, with the normal "
                        "E edge through it live. Otherwise resolve the "
                        "thickness with a finer local cell."
                        if is_pec else
                        " Use non-uniform mesh or reduce dx."
                    )
                    _w.warn(
                        PreflightWarning(
                            f"'{mat_name}' {axis_name}-extent {_fmt_len(dim)} = "
                            f"{cells_count:.1f} cells — below 1 cell resolution."
                            + hint,
                            code="mesh_resolution",
                            source="_validate_mesh_quality",
                        ),
                        stacklevel=3,
                    )
                else:
                    # Physics-based resolution thresholds (issue #37).
                    # A PEC volume 1-2 cells thick is a FILLED SLAB with
                    # walls on both faces (#931 §1.2 — ``pec_box_one_cell``
                    # says so for the 1-cell case); its thickness is
                    # realized as drawn, so there is no partial-volume
                    # question. Only warn on partial volume: 3-5 cells
                    # thick PEC bodies that are volumes on every axis.
                    # Dielectric: cells per local λ_eff, not cells per
                    # geometry extent.
                    cells = dim / cell
                    if is_pec:
                        if (3.0 <= cells < 5.0
                                and not is_thin_along_some_axis):
                            _w.warn(
                                PreflightWarning(
                                    f"PEC '{mat_name}' {axis_name}-extent "
                                    f"{_fmt_len(dim)} = {cells:.1f} cells — "
                                    "volume under-resolved (a PEC volume's "
                                    "curved/edge features need ≥5 cells; a "
                                    "1-2 cell slab is realized as drawn, "
                                    "with walls on both faces).",
                                    code="mesh_resolution",
                                    source="_validate_mesh_quality",
                                ),
                                stacklevel=3,
                            )
                    else:
                        eps_r = float(mat.eps_r) if mat.eps_r else 1.0
                        lam_eff = (
                            C0 / self._freq_max / math.sqrt(max(eps_r, 1.0))
                        )
                        cells_per_lam = lam_eff / cell
                        # rfx's Yee update is 2nd-order in bulk but
                        # degrades to 1st-order at ε-discontinuities
                        # because subpixel smoothing is default OFF
                        # (Meep ships it ON and stays 2nd-order). For
                        # phase-accurate propagation we need ≥15 cells
                        # per λ_eff — the traditional λ/10 rule applies
                        # to subpixel-smoothed codes. S-parameter
                        # extraction with a port or flux monitor
                        # amplifies dielectric-interface phase error
                        # into |S| magnitude error (see
                        # validation/crossval/11 rfx-vs-analytic audit,
                        # 2026-04-24): at 17.7 cells/λ_eff we measure
                        # ~5% |S21| deficit at Fabry-Perot peaks; at
                        # 35 cells/λ_eff (dx halved) it halves to ~2%.
                        # Require 20 cells/λ_eff when S-param
                        # extraction is active.
                        sparam_active = bool(
                            self._waveguide_ports
                            or self._flux_monitors
                        )
                        threshold = 20.0 if sparam_active else 15.0
                        if cells_per_lam < threshold:
                            suffix = (
                                " S-parameter extraction amplifies "
                                "ε-interface phase error into |S| "
                                "magnitude error; ~5% |S21| deficit "
                                "expected at 17 cells/λ_eff."
                                if sparam_active else
                                " Yee without subpixel smoothing has "
                                "1st-order convergence at ε interfaces."
                            )
                            _w.warn(
                                PreflightWarning(
                                    f"dielectric '{mat_name}' on {axis_name}: "
                                    f"{cells_per_lam:.1f} cells per λ_eff "
                                    f"(eps_r={eps_r:.2f}, freq_max="
                                    f"{_fmt_freq(self._freq_max)}, "
                                    f"dx={_fmt_len(cell)}). Need ≥"
                                    f"{threshold:.0f} cells/λ_eff for "
                                    f"phase-accurate propagation."
                                    f"{suffix}",
                                    code="mesh_resolution",
                                    source="_validate_mesh_quality",
                                ),
                                stacklevel=3,
                            )

        # Check gaps between PEC structures
        pec_entries = [e for e in self._geometry if e.material_name == "pec"]
        if len(pec_entries) >= 2:
            for i in range(len(pec_entries)):
                for j in range(i + 1, min(i + 5, len(pec_entries))):
                    try:
                        c1a, c2a = pec_entries[i].shape.bounding_box()
                        c1b, c2b = pec_entries[j].shape.bounding_box()
                        # Min gap along each axis
                        for ax in range(3):
                            gap = max(0, max(c1b[ax] - c2a[ax], c1a[ax] - c2b[ax]))
                            cell = [dx, dx, min_dz][ax]
                            if 0 < gap < 3 * cell:
                                _w.warn(
                                    PreflightWarning(
                                        f"Gap between PEC structures: "
                                        f"{_fmt_len(gap)} = {gap/cell:.1f} cells "
                                        f"along {'xyz'[ax]} — coupling may be "
                                        f"under-resolved.",
                                        code="mesh_resolution",
                                        source="_validate_mesh_quality",
                                    ),
                                    stacklevel=3,
                                )
                    except (NotImplementedError, TypeError, AttributeError):
                        continue

        # Physics-based numerical dispersion check (Taflove Ch. 4).
        # Instead of a fixed aspect-ratio heuristic, compute the actual
        # per-axis phase velocity error at freq_max from the FDTD
        # dispersion relation. This is application-independent.
        self._check_numerical_dispersion()

        # Thin-metal-on-NU-mesh symmetry (Meep/OpenEMS convention — issue #48).
        self._validate_thin_metal_on_nu_mesh()

    def _check_numerical_dispersion(self) -> None:
        """Warn when per-axis FDTD phase velocity error at freq_max
        exceeds a threshold (Taflove Ch. 4 dispersion relation).

        For each axis the worst-case phase velocity is:
            v_ph = (omega·dt) / (2·arcsin(nu_i · sin(k·d_i/2)))
        where nu_i = c·dt/d_i, k = 2π/λ, d_i = cell size along axis i.

        Reports the per-axis error so the user sees which axis is under-
        resolved or has Courant mismatch — no arbitrary ratio threshold.
        """
        import warnings as _w

        # Skip host-side min when any profile is a tracer. The dispersion
        # warning is advisory only; mesh-as-design-variable optimisation
        # runs under tracing and the warning cannot fire correctly there.
        if any(
            p is not None and is_tracer(p)
            for p in (self._dx_profile, self._dy_profile, self._dz_profile)
        ):
            return

        dx_nom = self._dx or (C0 / self._freq_max / 20.0)
        d = [dx_nom, dx_nom, dx_nom]
        if self._dx_profile is not None:
            d[0] = float(np.min(self._dx_profile))
        if self._dy_profile is not None:
            d[1] = float(np.min(self._dy_profile))
        if self._dz_profile is not None:
            d[2] = float(np.min(self._dz_profile))

        inv_sq = sum(1.0 / di ** 2 for di in d)
        dt_cfl = 0.99 / (C0 * math.sqrt(inv_sq))
        omega = 2.0 * math.pi * self._freq_max

        errors = {}
        sin_wdt2 = math.sin(omega * dt_cfl / 2.0)
        for ax, (name, di) in enumerate(zip("xyz", d)):
            # Taflove Eq. 4.44: v_ph along axis i
            # = omega * d_i / (2 * arcsin(d_i * sin(omega*dt/2) / (c*dt)))
            arg = di * sin_wdt2 / (C0 * dt_cfl)
            if abs(arg) >= 1.0:
                errors[name] = float("inf")
                continue
            v_ph = omega * di / (2.0 * math.asin(arg))
            errors[name] = abs(v_ph - C0) / C0

        max_err = max(errors.values())
        if max_err > 0.02:
            parts = ", ".join(
                f"{name}={err*100:.1f}%" for name, err in errors.items()
            )
            worst = max(errors, key=errors.get)
            _w.warn(
                PreflightWarning(
                    f"FDTD numerical dispersion at freq_max="
                    f"{self._freq_max/1e9:.2f}GHz exceeds 2%: {parts}. "
                    f"Worst axis: {worst} (cell {d['xyz'.index(worst)]*1e3:.3f}mm). "
                    f"Phase velocity error causes resonance frequency bias. "
                    f"Refine the coarse axis or co-refine all axes together "
                    f"(Taflove Ch. 4).",
                    code="numerical_dispersion",
                    source="_check_numerical_dispersion",
                ),
                stacklevel=4,
            )

    def _validate_thin_metal_on_nu_mesh(self) -> None:
        """Warn when a realized metal PLANE sits on a NU axis without
        symmetric neighbouring cells (Meep/OpenEMS require equal dz on
        both sides of a metal plane, else surface currents pick up O(1)
        error and the far-field pattern is corrupted — issue #48).

        Lattice ownership contract (#931): the plane examined is the one
        the run REALIZES — a sheet's node plane, or each wall plane of a
        thin volume (a slab at most two cells thick along the NU axis,
        whose two faces are both surface-current planes) — read from the
        shared realization, never from a bounding-box midpoint (which
        could name the wrong cell). Thick volumes are not "thin metal"
        and are skipped.
        """
        import warnings as _w
        profiles = (self._dx_profile, self._dy_profile, self._dz_profile)
        if all(p is None for p in profiles):
            return
        if any(p is not None and is_tracer(p) for p in profiles):
            # Tracer profiles can't be host-scanned for edge / ratio
            # checks. The warning is advisory only; correctness is
            # preserved downstream.
            return
        ctx = self._campaign_ctx()
        if ctx.error is not None:
            return
        shape = tuple(ctx.grid.shape)
        for axis_idx, prof in enumerate(profiles):
            if prof is None:
                continue
            axis_name = "xyz"[axis_idx]
            d = ctx.spacings[axis_idx]
            if d.size < 3:
                continue
            for e in ctx.pec_entries():
                if e.kind == "wire":
                    continue
                planes = e.wall_planes(axis_idx, ctx.periodic, shape)
                if not planes:
                    continue
                if e.kind == "volume":
                    # thin slab: at most two cell layers between its faces
                    if max(planes) - min(planes) > 2:
                        continue
                for k in planes:
                    if k - 1 < 0 or k >= d.size:
                        continue
                    d_below = float(d[k - 1])
                    d_above = float(d[k])
                    ratio = max(d_below, d_above) / min(d_below, d_above)
                    if ratio <= 1.5:
                        continue
                    _w.warn(
                        PreflightWarning(
                            f"Thin PEC {e.label} ('{e.name}', realized as a "
                            f"{e.kind}) has a metal plane at {axis_name} = "
                            f"{_fmt_len(float(ctx.nodes[axis_idx][k]))} "
                            f"(node {k}) with asymmetric neighbouring cells "
                            f"({_fmt_len(d_below)} below, {_fmt_len(d_above)} "
                            f"above, ratio {ratio:.2f}). Meep/OpenEMS require "
                            f"equal cell sizes across a metal plane; "
                            f"radiation pattern may be corrupted (issue #48). "
                            f"Put the metal plane on a preserved-region "
                            f"boundary or refine the neighbouring cell.",
                            code="thin_metal_nu_mesh",
                            source="_validate_thin_metal_on_nu_mesh",
                        ),
                        stacklevel=4,
                    )

    def _assemble_realized(self, grid, *, nonuniform: bool):
        """The run's realized conductor set on ``grid`` (#931 §1.7).

        Runs the PRODUCTION assembly with the sheet / wire collectors and
        realizes the ``(Mx, My, Mz)`` edge masks through
        ``rfx.boundaries.pec.realized_pec_edge_masks`` under the run's own
        periodic flags — the same call the solver lanes make. Every
        preflight consumer that needs "where is metal" reads the returned
        :class:`_RealizedPEC` (wall planes, ``is_pec_edge``), never the
        cell mask: a sheet owns no cell, and a volume's far face is a
        wall the cell mask does not mark (the #868 class).
        """
        sheets: list = []
        wires: list = []
        sheet_specs: list = []
        if nonuniform:
            mats, _, _, pec_mask = self._assemble_materials_nu(
                grid, sheet_specs=sheet_specs, pec_sheets=sheets,
                pec_wires=wires)
        else:
            mats, _, _, pec_mask, _, _, _ = self._assemble_materials(
                grid, sheet_specs=sheet_specs, pec_sheets=sheets,
                pec_wires=wires)
        return _RealizedPEC(
            lane="nonuniform" if nonuniform else "uniform", grid=grid,
            materials=mats, pec_mask=pec_mask, sheets=sheets, wires=wires,
            periodic=self._periodic_flags(), sheet_specs=sheet_specs)

    def _port_realized_edges(self, grid):
        """:class:`_RealizedPEC` for the uniform lane, or ``None``.

        Issue #738 review: the guide a waveguide port sits in is defined
        by its WALLS, and on the committed sub-aperture fixtures those
        walls are interior PEC shapes, not the domain faces. Read from
        the PRODUCTION assembly so preflight sees the same realized
        conductors the solve does — no geometric re-derivation from the
        shape list. Called once per preflight and threaded through, so
        the assembly runs once. A simulation with no geometry entries at
        all has no interior walls by construction, so the assembly is
        skipped there rather than run to produce an all-False set.
        """
        if not self._geometry and not getattr(self, "_thin_conductors", None):
            return None
        try:
            realized = self._assemble_realized(grid, nonuniform=False)
        except (ValueError, TypeError, NotImplementedError, KeyError,
                AttributeError, IndexError):
            # Deliberately NOT ``except Exception`` (PR #555): an async
            # worker timeout must propagate through this advisory. On a
            # narrow failure the caller falls back to the aperture,
            # which is always defined. Issue #482 also made the timeout/
            # cancel exceptions themselves ``BaseException``-derived, so
            # this tuple is belt-and-suspenders, not the only guard --
            # see the longer comment at the other ``_assemble_materials``
            # call site in this file.
            return None
        if realized.empty:
            return None
        return realized

    def _port_pec_mask(self, grid):
        """Kept name for ``tests/_waveguide_chain_battery_fixture.py``
        (``transverse_spans``), which reads the guide the way preflight
        does. Returns :meth:`_port_realized_edges` — the run's realized
        conductor set, NOT a cell mask (a cell mask cannot carry a
        volume's far face or a sheet, #931 §1.9); the object is what
        :meth:`_port_transverse_spans` takes. Callers should move to the
        new name; delete this once the fixture does."""
        return self._port_realized_edges(grid)

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the realized-aperture pair moved VERBATIM to
    # ``rfx/preflight/waveguide.py`` and is bound back here, AT THE
    # POSITION it held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls these checks in a fixed
    # sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_wg`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._port_transverse_spans(...)`` a bound method with its name,
    # signature and ``__doc__`` intact, and ``self._range_to_slice``
    # inside it keeps resolving through the composed ``Simulation`` MRO
    # exactly as it did before the move.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import (
        _port_transverse_spans,
        _check_waveguide_port_aperture_snap,
    )

    def _check_coaxial_port_junction_aperture(self) -> None:
        """Advise when a coaxial port's pin meets REGISTERED PEC at its
        junction plane — a short by geometry, not by the port (issue #589).

        Root cause this names: ``_assemble_materials`` is PEC-OR-only
        (``pec_mask = pec_mask | mask``, rfx/api/_compile.py) and there is
        no CSG subtraction shape, so a ground plane declared as a full PEC
        sheet with a dielectric "clearance hole" declared AFTER it stays
        solid. The committed coax-MSL junction fixture was built exactly
        that way; its settled run measured S00 = (-0.9928, -0.0048) at
        6 GHz — the pin was terminated in a short by the ground sheet, and
        no check said so (the fixture's only structural test asserted
        pin-column PEC continuity, trivially true through a solid sheet).

        What is measured: on the PRODUCTION assembly's REALIZED conductor
        set (:meth:`_port_realized_edges`, sim.add geometry and thin
        conductors only — the coax stub the compute_coaxial_* /
        compute_coax_msl_transition methods stamp into eps/sigma is not
        registered geometry and is not read here), at the port's
        junction plane ``k = position_to_index(position)[axis]``, the
        count of NODES CARRYING A TANGENTIAL WALL in the FIRST dielectric
        ring outside the pin. "Carries a wall" is the lattice ownership
        contract's own test (#931 §1.9): some E edge tangential to the
        port axis and incident to the node is PEC — which is what a sheet
        ground at plane ``k`` and a volume ground whose face lies on
        plane ``k`` both put there, and what a primal CELL mask never
        marked for a volume's far face (the #868 class). The ring is
        defined ON THE LATTICE: ``a + dx/sqrt(2) < r <= min(a +
        dx/sqrt(2) + dx, shell_inner)`` with ``r`` the node distance from
        the port centre and ``shell_inner = b - min(dx, (b - a)/2)`` the
        radius ``stamp_coaxial_line`` realizes for the shell. The inner
        bound is the pin's own reach: a PEC volume is centre-sampled
        (§1.1) and every node of an occupied cell carries its wall, so
        the pin's OWN nodes extend to at most ``a + dx/sqrt(2)`` (half
        the cell diagonal past its radius); a node beyond that cannot be
        a corner of a pin cell, so anything there is OTHER registered
        conductor. The earlier ``a + dx/2`` bound was calibrated to the
        pre-#931 node-sampled knife-edge footprint and would count the
        pin's own realized rim.

        Deliberately NOT the full ``a < r < shell_inner`` annulus: a
        clearance hole narrower than the shell (the fixture's predeclared
        0.4 mm hole under a 0.5 mm shell_inner leaves a one-cell ground
        lip, 32/68 of that annulus PEC) is a valid launch and the wide
        rule would flag the fix.

        Report-only (severity "warning", no refusal): a calibration short
        built from REGISTERED PEC would trip it and is the user's call.
        The repo's own calibration short is stamped via
        ``stamp_coaxial_short_plane`` (sigma), so no current lane does.
        Not audited (silent by construction, disclosed here): the
        non-uniform lane, 2-D mode, and a port whose position does not map
        into the grid (the compiler rejects that separately). Measured on
        the example snapshot (tests/contracts/test_example_fidelity_contract.py):
        the one coaxial variant has 0 PEC cells in the ring, so no
        snapshot row changes.
        """
        import warnings as _w

        if not self._coaxial_ports:
            return
        if (self._dx_profile is not None or self._dy_profile is not None
                or self._dz_profile is not None):
            return
        from rfx.sources.coaxial_port import _FACE_CONFIG

        grid = self._build_grid()
        if getattr(grid, "is_2d", False):
            return
        realized = self._port_realized_edges(grid)
        if realized is None:
            return
        dx = float(grid.dx)
        pads = (int(grid.pad_x_lo), int(grid.pad_y_lo), int(grid.pad_z_lo))
        axis_names = ("x", "y", "z")
        for n, port in enumerate(self._coaxial_ports):
            cfg = _FACE_CONFIG.get(str(port.face))
            if cfg is None:
                continue
            axis = axis_names.index(cfg[0])
            pos = tuple(float(v) for v in port.position)
            try:
                idx = grid.position_to_index(pos)
            except ValueError:
                continue
            k = int(idx[axis])
            t1, t2 = [a for a in range(3) if a != axis]
            a = float(port.pin_radius)
            b = float(port.outer_radius)
            shell_inner = b - min(dx, 0.5 * (b - a))
            c1 = (np.arange(grid.shape[t1]) - pads[t1]) * dx - pos[t1]
            c2 = (np.arange(grid.shape[t2]) - pads[t2]) * dx - pos[t2]
            r = np.hypot(c1[:, None], c2[None, :])
            if not (0 <= k < grid.shape[axis]):
                continue
            plane = realized.wall_nodes_on_plane(axis, k)
            r_lo = a + dx / math.sqrt(2.0)
            r_hi = min(r_lo + dx, shell_inner)
            ring = (r > r_lo) & (r <= r_hi)
            n_ring = int(np.count_nonzero(ring))
            if n_ring == 0:
                continue
            n_pec = int(np.count_nonzero(plane & ring))
            if n_pec == 0:
                continue
            outside = plane & (r > r_lo) & (r <= b)
            r_first = float(r[outside].min()) if outside.any() else float("nan")
            _w.warn(
                PreflightWarning(
                    f"Coaxial port {n} (face='{port.face}', pin r="
                    f"{a * 1e6:.0f} um, outer r={b * 1e6:.0f} um): at its "
                    f"junction plane ({axis_names[axis]}="
                    f"{pos[axis] * 1e3:.3f} mm, node {k - pads[axis]}) "
                    f"{n_pec}/{n_ring} nodes of the FIRST dielectric ring "
                    f"outside the pin ({r_lo * 1e6:.0f} < r <= "
                    f"{r_hi * 1e6:.0f} um; lattice-based bounds, so the "
                    f"pin's own realized rim, at most r = {a * 1e6:.0f} um "
                    f"+ dx/sqrt(2), is excluded) carry a REALIZED PEC wall "
                    f"— the first registered conductor outside the pin "
                    f"sits at r = {r_first * 1e6:.1f} um. The pin is "
                    f"terminated in a short by registered geometry at this "
                    f"plane. The assembly is PEC-OR-only: a ground sheet "
                    f"declared as a full plane with a dielectric 'hole' "
                    f"declared AFTER it stays solid (issue #589: S00 = "
                    f"-0.9928 at 6 GHz measured on exactly that fixture). "
                    f"If a clearance aperture was intended, build the "
                    f"conductor WITH the hole (a patterned sheet shape via "
                    f"add_thin_conductor, or an annular volume); if this "
                    f"short is the intended calibration standard, no "
                    f"action is needed (report-only).",
                    code="coaxial_port_junction_short",
                    source="_check_coaxial_port_junction_aperture",
                    loc=f"coaxial_port[{n}] face={port.face}",
                ),
                stacklevel=4,
            )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the cutoff emitter and its two lanes moved
    # VERBATIM to ``rfx/preflight/waveguide.py``, bound back at their
    # original position for the reason the block above gives.
    #
    # ``_check_waveguide_port_evanescent`` keeps three ``self.`` calls to
    # bodies that STAY here -- ``_build_grid``, ``_port_realized_edges``
    # (realization) -- and they resolve through the composed
    # ``Simulation`` MRO, so they survive the move untouched.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import (
        _emit_waveguide_port_cutoff_findings,
        _check_waveguide_port_evanescent_declared_geometry,
        _check_waveguide_port_evanescent,
    )

    def _collect_flux_regions(self, report: PreflightReport) -> None:
        """Record the same finite windows the uniform/NU runners consume.

        Only geometry is evaluated, never material arrays or field values.
        Traced geometry cannot provide metre bounds; keep that limitation
        explicit without forcing a material-AD caller off its tape.
        """
        from rfx.probes.flux_region import (
            flux_region_message, resolve_flux_region, validate_flux_region_inputs,
        )

        source = "_collect_flux_regions"

        def finding(entry, message, code, severity="warning"):
            report.append(PreflightIssue(
                message, severity=severity, code=code,
                loc=getattr(entry, "name", None), source=source,
            ))

        def traced(value):
            return any(is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(value))

        finite = []
        for entry in getattr(self, "_flux_monitors", ()):
            geometry = (entry.size, entry.center, entry.coordinate)
            if traced(geometry):
                finding(entry, "Flux monitor region is unavailable: traced geometry.",
                        "flux_region_unavailable")
                continue
            try:
                size, _ = validate_flux_region_inputs(entry.size, entry.center)
            except (ValueError, TypeError) as exc:
                finding(entry, f"Flux monitor {entry.name!r}: {exc}",
                        "flux_region_invalid", "error")
                continue
            if size is not None:
                finite.append(entry)
        if not finite:
            return

        # Resolve the selected/frozen mesh, including inferred NU profiles.
        # No independent rounding or declaration-only grid choice belongs here.
        try:
            if traced(self._resolve_mesh()):
                raise ValueError("traced mesh has no concrete metre bounds")
            grid = self._build_realized_grid()
        except (ValueError, TypeError, NotImplementedError, AttributeError,
                IndexError, KeyError) as exc:
            for entry in finite:
                finding(entry, f"Flux monitor {entry.name!r} region is unavailable: {exc}",
                        "flux_region_unavailable")
            return

        for entry in finite:
            try:
                record = resolve_flux_region(grid, entry, self._domain, warn=False)
            except (ValueError, TypeError, IndexError) as exc:
                finding(entry, f"Flux monitor {entry.name!r}: {exc}",
                        "flux_region_invalid", "error")
                continue
            report.flux_regions.append(record)
            if record["clamped"]:
                finding(entry, flux_region_message(record), "flux_region_clamped")

    def preflight(
        self,
        *,
        strict: bool = False,
        check_ntff: bool | str = True,
        check_resolution: bool = True,
        check_ad_memory: bool = False,
        n_steps_for_memory: int | None = None,
        available_memory_gb: float | None = None,
    ) -> "PreflightReport":
        """Run all pre-simulation checks and return warnings.

        Resolves and caches the mesh from the current static declaration,
        without changing declared dx/domain/profiles. Auto-mesh selection is
        emitted as a UserWarning, separately from validation findings.

        Parameters
        ----------
        strict : bool
            If True, raise ValueError on the first issue instead of
            collecting warnings.
        check_ntff : bool or "advisory"
            ``True`` (default): run the full NTFF check family (PEC-overlap
            hard error + λ/4 / λ/2 near-field gap advisories + the
            sub-wavelength ground-plane pattern advisory, issue #334).
            ``"advisory"``: run only the advisories — the tier ``run()``
            uses, because the λ/4 and small-ground-plane warnings are
            physics-relevant to any far-field computation while the
            PEC-overlap hard error remains an inverse-design gate
            (issue #303).
            ``False``: skip the family entirely.
        check_resolution : bool
            Run the tightened resolution check (existing _validate_mesh_quality
            uses per-material thresholds already — this flag kept for
            symmetry and future tightening). Default True.
        check_ad_memory : bool
            Run AD memory estimate and warn if > 85% of available VRAM.
            Requires n_steps_for_memory. Default False (diagnostic only).
        n_steps_for_memory : int or None
            Step count for AD memory sizing. Required when check_ad_memory.
        available_memory_gb : float or None
            Override VRAM detection. If None, best-effort via JAX devices.

        Returns
        -------
        PreflightReport
            A ``list`` subclass of :class:`PreflightIssue` (each a ``str``
            subclass), back-compatible with the legacy ``list[str]`` return.
            Empty if no issues found. Finite flux-window geometry is recorded
            separately in ``flux_regions``, including for issue-free reports.
        """
        import warnings
        # Selection information belongs outside the captured legality findings.
        # Resolve before any validator reads a profile or fallback spacing.
        self._resolve_mesh()
        issues = PreflightReport()

        # Static geometry diagnostics must stay host-side under an outer
        # jit, just like mesh selection. Otherwise even a concrete Box's
        # mask becomes a tracer before validators convert it to numpy.
        # Incoming mesh/design tracers remain tracers and retain the
        # validators' existing not-evaluable guards.
        with warnings.catch_warnings(record=True) as caught, jax.ensure_compile_time_eval():
            warnings.simplefilter("always")
            self._collect_flux_regions(issues)
            try:
                if check_resolution:
                    self._validate_mesh_quality()
                self._validate_simulation_config()
                if check_ntff:
                    self._validate_ntff_inverse_design(
                        include_pec_overlap_error=(check_ntff != "advisory"),
                    )
            except ValueError as e:
                # Collect (do NOT fail-on-first): the aggregated raise at the
                # end escalates every finding at once under strict.
                # Structurally-impossible configs raise PreflightConfigError
                # with the slug set at the check site; any other ValueError is
                # error-severity but uncoded.
                issues.append(PreflightIssue(
                    f"ERROR: {e}",
                    severity="error",
                    code=getattr(e, "code", "uncoded"),
                    loc=getattr(e, "loc", None),
                    source=getattr(e, "source", None),
                ))

        for w in caught:
            msg = str(w.message)
            # Collect (do NOT fail-on-first): aggregated raise at the end.
            # Prefer the structured fields carried on the warning INSTANCE
            # (PreflightWarning); fall back to the category-derived severity for
            # the legacy ``warnings.warn(msg, PreflightErrorWarning)`` form, and
            # to severity="warning"/code="uncoded" for any plain UserWarning.
            inst = w.message
            if isinstance(inst, PreflightWarning):
                severity = inst.severity
                code = inst.code
                loc = inst.loc
                source = inst.source
            else:
                severity = (
                    "error" if issubclass(w.category, PreflightErrorWarning)
                    else "warning"
                )
                code = "uncoded"
                loc = None
                source = None
            issues.append(PreflightIssue(
                msg, severity=severity, code=code, loc=loc, source=source
            ))

        if check_ad_memory:
            if n_steps_for_memory is None:
                raise ValueError("check_ad_memory=True requires n_steps_for_memory")
            est = self.estimate_ad_memory(
                n_steps_for_memory,
                available_memory_gb=available_memory_gb,
            )
            if est.warning:
                issues.append(PreflightIssue(
                    est.warning, severity="warning", code="ad_memory"
                ))

        if strict and len(issues):   # PreflightReport refuses bool() (#980)
            # Aggregate-then-raise: escalate ALL findings at once. Preserves the
            # historical "strict escalates any issue to ValueError" contract,
            # but reports every problem in one pass instead of fail-on-first
            # (pydantic / Tidy3D pattern). For an errors-only gate that lets
            # advisories through, call ``report.raise_for_failure()`` on a
            # ``strict=False`` report instead.
            raise ValueError(
                f"preflight (strict) found {len(issues)} issue(s):\n  - "
                + "\n  - ".join(issues)
            )

        if len(issues):              # PreflightReport refuses bool() (#980)
            for iss in issues:
                print(f"  [PREFLIGHT] {iss}")
        elif check_ntff is True:
            print("  [PREFLIGHT] All checks passed.")
        elif check_ntff == "advisory":
            print("  [PREFLIGHT] All checks passed (NTFF advisory tier; the "
                  "PEC-overlap error check runs on forward()/preflight()).")
        else:
            print("  [PREFLIGHT] All checks passed (NTFF checks skipped; "
                  "run sim.preflight() for the full set).")

        if issues.flux_regions:
            from rfx.probes.flux_region import flux_region_message
            for record in issues.flux_regions:
                print(f"  [FLUX REGION] {flux_region_message(record)}")

        return issues

    def preflight_sparameters(
        self,
        *,
        calculator: str = "run",
        strict: bool = False,
        normalize: bool | str | None = None,
        include_general: bool = False,
    ) -> "PreflightReport":
        """Preflight the selected S-parameter calculator without running FDTD.

        This is a routing/contract check for the port-family-specific
        S-parameter APIs.  It answers "which calculator should this simulation
        use?" before an expensive run starts:

        - ``calculator="run"`` checks ``run(compute_s_params=True)`` for
          lumped/wire ``add_port(...)`` families.
        - ``calculator="forward"`` checks ``forward(port_s11_freqs=...)`` for
          uniform single-device S11 vectors.
        - ``calculator="msl"`` checks ``compute_msl_s_matrix(...)``.
        - ``calculator="waveguide"`` checks ``compute_waveguide_s_matrix(...)``.

        Parameters
        ----------
        calculator:
            One of ``"run"``, ``"forward"``, ``"msl"``, or ``"waveguide"``
            (the corresponding method names are accepted as aliases).
        strict:
            If True, escalate ERROR-severity findings to a raise: collect
            everything, then raise a single ``ValueError`` listing the errors
            (aggregate-then-raise). The underlying ``NotImplementedError`` is
            recorded as an error-severity issue and re-surfaced as part of that
            aggregated ``ValueError`` (its exact type is not preserved).

            Errors only, not "any issue": the waveguide setup audits report
            advisory and informational findings on a perfectly valid routing,
            and a healthy two-port guide ALWAYS carries the informational
            E-plane note, so escalating on emptiness would make
            ``strict=True`` raise on every correct waveguide setup. Advisories
            are still returned in the report and printed; read them there.
            (This differs from ``preflight(strict=True)``, whose report has no
            informational tier.)
        normalize:
            Waveguide non-uniform preflight uses this to mirror
            ``compute_waveguide_s_matrix(normalize=...)``.  ``None`` means the
            method default, currently ``False``.
        include_general:
            If True, append the ordinary geometry/material ``preflight()``
            issues after the S-parameter routing check.

        Returns
        -------
        PreflightReport
            A ``list`` subclass of :class:`PreflightIssue` (back-compatible with
            the historical ``list[str]``). Carries no ERROR-severity issue when
            the selected calculator is valid for the registered port families —
            read ``report.ok`` / ``report.errors`` rather than emptiness.
            ``calculator="waveguide"`` additionally runs the three setup
            audits (:meth:`_validate_cfg_record_vs_far_boundary`,
            :meth:`_validate_cfg_port_index_mirror_covariance` and
            :meth:`_validate_cfg_layout_from_band_low_edge`), which report
            advisory and informational findings on a perfectly valid routing —
            a healthy two-port guide always carries at least the known
            E-plane-offset note. Those audits are evaluated at
            ``compute_waveguide_s_matrix``'s own default ``num_periods``
            (:data:`WAVEGUIDE_DEFAULT_NUM_PERIODS`); the record-length message
            names the ``num_periods`` that clears the threshold, which does not
            depend on the value it was evaluated at.
        """

        aliases = {
            "run": "run",
            "result": "run",
            "compute_s_params": "run",
            "forward": "forward",
            "forward_s11": "forward",
            "port_s11_freqs": "forward",
            "msl": "msl",
            "compute_msl_s_matrix": "msl",
            "waveguide": "waveguide",
            "compute_waveguide_s_matrix": "waveguide",
            "coaxial": "coaxial",
            "compute_coaxial_s_matrix": "coaxial",
        }
        key = aliases.get(calculator.lower())
        if key is None:
            allowed = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unknown S-parameter calculator {calculator!r}. "
                f"Choose one of: {allowed}."
            )

        issues = PreflightReport()

        try:
            if key == "run":
                self._validate_run_sparameter_request(
                    compute_s_params=True,
                    s_param_freqs=None,
                    s_param_n_steps=None,
                    devices=None,
                )
            elif key == "forward":
                self._validate_forward_sparameter_request()
                is_nonuniform = (
                    self._dz_profile is not None
                    or self._dx_profile is not None
                    or self._dy_profile is not None
                )
                if is_nonuniform:
                    raise NotImplementedError(
                        "forward(port_s11_freqs=...) is currently wired only "
                        "on the uniform single-device forward path. Drop "
                        "port_s11_freqs or use a uniform mesh."
                    )
            elif key == "msl":
                self._validate_msl_sparameter_request_for_preflight()
            elif key == "waveguide":
                wg_normalize = False if normalize is None else normalize
                self._validate_waveguide_sparameter_request_for_preflight(
                    normalize=wg_normalize,
                )
            elif key == "coaxial":
                self._validate_coaxial_sparameter_request_for_preflight()
        except (ValueError, NotImplementedError) as exc:
            # Collect as a coded error-severity issue (aggregated raise below
            # under strict) — consistent with preflight()'s PreflightIssue
            # contract instead of the old bare f-string.
            issues.append(PreflightIssue(
                f"{type(exc).__name__}: {exc}",
                severity="error",
                code=getattr(exc, "code", f"sparam_routing_{key}"),
                source="preflight_sparameters",
            ))

        # Waveguide setup audits (post-v1.8 plan item 2, sections 2-1/2-4 of
        # docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md).
        # Only when the routing check above found nothing: with fewer than two
        # waveguide ports there is no layout to audit. Emitted as warnings by
        # the check sites (the repo idiom) and folded into the report here, so
        # the coded fields survive into PreflightIssue.
        if key == "waveguide" and not len(issues):   # refuses bool() (#980)
            _wg_entries = list(self._waveguide_ports)
            if _wg_entries:
                import warnings as _wmod
                with _wmod.catch_warnings(record=True) as _wg_caught:
                    _wmod.simplefilter("always")
                    self._preflight_waveguide_setup(
                        _wmod,
                        freqs=resolve_waveguide_port_freqs(self, _wg_entries[0]),
                        num_periods=WAVEGUIDE_DEFAULT_NUM_PERIODS,
                    )
                for _rec in _wg_caught:
                    _inst = _rec.message
                    issues.append(PreflightIssue(
                        str(_inst),
                        severity=getattr(_inst, "severity", "warning"),
                        code=getattr(_inst, "code", "uncoded"),
                        loc=getattr(_inst, "loc", None),
                        source=getattr(_inst, "source", None),
                    ))

        if include_general:
            # strict=False here: collect the general findings, then aggregate
            # everything in one raise below (don't fail-on-first).
            general = self.preflight(strict=False)
            issues.extend(general)
            issues.flux_regions.extend(general.flux_regions)

        _errors = issues.errors
        if strict and _errors:
            _advisory = len(issues) - len(_errors)
            raise ValueError(
                f"preflight_sparameters (strict) found {len(_errors)} "
                f"error-severity issue(s)"
                + (f" (plus {_advisory} advisory/informational finding(s), "
                   "returned but not escalated)" if _advisory else "")
                + ":\n  - " + "\n  - ".join(_errors)
            )

        if len(issues):              # PreflightReport refuses bool() (#980)
            for issue in issues:
                print(f"  [SPARAM PREFLIGHT] {issue}")
        else:
            print(f"  [SPARAM PREFLIGHT] {key}: all checks passed.")
        return issues

    def _validate_msl_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_msl_s_matrix`` family-routing checks."""

        if not self._msl_ports:
            raise ValueError("No MSL ports registered. Call add_msl_port() first.")
        if self._ports or self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() is defined only for add_msl_port(...) "
                "families in the current simulation. Use separate simulations "
                "for add_port(...), add_waveguide_port(...), or "
                "add_floquet_port(...) S-parameter workflows."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported together with TFSF; "
                "TFSF is a plane-wave source, not an MSL port."
            )
        if self._coaxial_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() does not include add_coaxial_port(...); "
                "coaxial-port S-parameters need a separate validated V/I "
                "extraction and calibration contract."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ) and any(
            getattr(pe, "mode", "laplace") == "eigenmode"
            for pe in self._msl_ports
        ):
            raise NotImplementedError(
                "compute_msl_s_matrix() on a non-uniform mesh supports "
                "mode='laplace'/'uniform' (Ez static-Laplace feed) only; the "
                "eigenmode J+M launch needs the magnetic-source channel that "
                "the non-uniform runner does not carry. Use mode='laplace' "
                "(the add_msl_port default) on the graded-mesh lane."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with SBP-SAT "
                "subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with solver='adi'; "
                "use the uniform Yee solver."
            )

    def _validate_waveguide_sparameter_request_for_preflight(
        self,
        *,
        normalize: bool | str,
    ) -> None:
        """Mirror ``compute_waveguide_s_matrix`` family-routing checks.

        Only the simulation-visible clauses are mirrored (``normalize`` and
        multi-mode on the non-uniform fence); ``compute_waveguide_s_matrix``
        call-time parameters (``subpixel_smoothing``, ``port_reference_sims``,
        ``eps_override`` / ``sigma_override``) are not exposed to preflight and
        stay method-only checks.
        """

        if not self._waveguide_ports:
            raise ValueError(
                "No waveguide ports registered. Call add_waveguide_port() first."
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with "
                "lumped ports or TFSF"
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported with manual "
                "periodic-axis overrides"
            )
        if len(self._waveguide_ports) < 2:
            raise ValueError(
                "compute_waveguide_s_matrix() requires at least two "
                "waveguide ports"
            )

        entries = list(self._waveguide_ports)
        if any(entry.probe_plane is not None for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() does not use per-port "
                "probe_plane; use reference_plane only or leave probe_plane unset"
            )
        if any(entry.calibration_preset not in (None, "measured") for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() currently supports only "
                "measured/default reference planes or explicit reference_plane "
                "overrides"
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            unsupported = []
            if normalize is not True and normalize != "flux":
                unsupported.append("normalize=True or normalize='flux' is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile / dz_profile) supports "
                    "normalize=True or "
                    "normalize='flux' and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy/dz profile to use the uniform lane."
                )

    def _validate_coaxial_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_coaxial_s_matrix`` family-routing checks."""

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )
        if (
            self._ports
            or self._waveguide_ports
            or self._floquet_ports
            or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is defined only for "
                "add_coaxial_port(...) families in the current simulation."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported together with "
                "TFSF; TFSF is a plane-wave source, not a coaxial port."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() supports the uniform Yee lane only."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with SBP-SAT subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with solver='adi'."
            )

    def _validate_ntff_inverse_design(
        self, *, include_pec_overlap_error: bool = True,
    ) -> None:
        """NTFF checks: PEC overlap (error) and λ/4 gap (warn).

        CHECK 2: NTFF face plane strictly intersecting a PEC bbox
        (hard error; skipped when ``include_pec_overlap_error=False`` —
        the ``run()`` advisory tier, issue #303).
        CHECK 3: NTFF face closer than λ/4 to any geometry or port/source.
        Passive DFT probes are NOT counted: they read field state without
        perturbing it, so a probe on a box face is a measurement choice,
        not a radiating/scattering culprit (issue #303).
        CHECK 4: source backed by a PEC sheet under ~1λ across
        (warning-severity advisory; both tiers, issue #334) — the far-field
        pattern will be shaped by ground-plane edge diffraction. Expected
        physics, not a solver defect; the advisory exists so a resonance
        fixture is not mistaken for a pattern fixture.
        """
        import warnings as _w

        if self._ntff is None:
            return

        corner_lo, corner_hi, freqs = self._ntff
        # face = (axis, sign, coord, tangential bbox: [(lo_a, hi_a), (lo_b, hi_b)])
        faces = []
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            tang = ((corner_lo[other[0]], corner_hi[other[0]]),
                    (corner_lo[other[1]], corner_hi[other[1]]))
            faces.append(("lo", axis, corner_lo[axis], tang))
            faces.append(("hi", axis, corner_hi[axis], tang))

        # CHECK 2: PEC intersection, CLOSED on the REALIZED wall planes
        # (#931 §1.7). A PEC volume drawn z_a -> z_b realizes walls at
        # BOTH planes, so an NTFF face lying exactly ON a wall plane sits
        # on a conductor surface (tangential E zeroed there) and is an
        # overlap; a sheet (thin conductor or zero-thickness Box) is one
        # plane and is in the census too. Bounds along the normal axis are
        # the entry's own realized wall planes in physical units, read
        # through the shared context; the tangential overlap uses the
        # declared bounds. A traced mesh has no concrete planes, so that
        # lane keeps the declared-bounds test (closed).
        try:
            has_pec_geom = any(
                self._resolve_material(e.material_name).sigma
                >= self._PEC_SIGMA_THRESHOLD
                for e in self._geometry)
        except KeyError:
            has_pec_geom = False   # unresolved material: add()/run() raise
        realized_entries = []
        if include_pec_overlap_error and (
                has_pec_geom or getattr(self, "_thin_conductors", None)):
            ctx = self._campaign_ctx()
            if ctx.error is None:
                shape = tuple(ctx.grid.shape)
                for e in ctx.pec_entries():
                    if e.kind == "wire" or e.lo is None:
                        continue
                    walls = []
                    for a in range(3):
                        planes = e.wall_planes(a, ctx.periodic, shape)
                        if not planes:
                            walls = None
                            break
                        nodes = ctx.nodes[a]
                        walls.append((float(nodes[min(planes)]),
                                      float(nodes[max(planes)]),
                                      ctx.local_spacing(a, float(nodes[min(planes)]))))
                    if walls is not None:
                        realized_entries.append((e, walls))
            else:
                for e in ctx.entry_realizations():
                    if e.kind in ("wire", "lossy") or e.lo is None:
                        continue
                    realized_entries.append((e, [
                        (float(e.lo[a]), float(e.hi[a]), 0.0)
                        for a in range(3)]))
        for side, axis, coord, tang in faces:
            for e, walls in realized_entries:
                w_lo, w_hi, d_loc = walls[axis]
                tol = 1e-9 * d_loc
                if not (w_lo - tol <= coord <= w_hi + tol):
                    continue
                # Tangential overlap along the other two axes (declared)
                other = [a for a in range(3) if a != axis]
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if e.hi[idx] <= tlo or e.lo[idx] >= thi:
                        overlap = False
                        break
                if overlap:
                    raise PreflightConfigError(
                        f"NTFF face {'xyz'[axis]}_{side} at {coord*1e3:.2f}mm "
                        f"lies on or inside PEC {e.label} '{e.name}' "
                        f"(realized as a {e.kind}: {'xyz'[axis]} walls at "
                        f"[{w_lo*1e3:.3f}, {w_hi*1e3:.3f}] mm; declared "
                        f"bbox {tuple(e.lo)}–{tuple(e.hi)}). NTFF box must "
                        f"enclose all radiators with no conductor surface "
                        f"on or crossing any face. Shrink or move the NTFF "
                        f"box.",
                        code="ntff_pec_overlap",
                        source="_validate_ntff_inverse_design",
                    )

        # CHECK 3: λ/2 (Huygens) and λ/4 (reactive-near-field) gaps to any
        # geometry/source (probes excluded, issue #303). Issue #77: the λ/2 Huygens-equivalence rule
        # was documented but only the λ/4 strong
        # tier was enforced; a face at λ/30 above a ground-plane PEC silently
        # ran and produced corrupted directivity. The two-tier check below
        # warns mildly in [λ/4, λ/2) (results may degrade) and strongly in
        # < λ/4 (directivity / pattern likely corrupted).
        if freqs is None:
            return
        try:
            f_max = float(jnp.max(jnp.asarray(freqs)))
        except Exception:
            f_max = float(self._freq_max)
        lam_min = C0 / max(f_max, 1.0)
        gap_thresh = lam_min / 4.0
        huygens_thresh = lam_min / 2.0

        # Collect candidate bboxes and point positions
        bboxes: list[tuple[str, tuple, tuple]] = []
        for entry in self._geometry:
            try:
                c1, c2 = entry.shape.bounding_box()
                bboxes.append((entry.material_name, c1, c2))
            except (NotImplementedError, TypeError, AttributeError):
                continue
        # #931: PEC thin conductors are sheets — conductors in the census.
        from rfx.materials.thin_conductor import sheet_bounds as _sheet_bounds
        for _ti, tc in enumerate(getattr(self, "_thin_conductors", ())):
            if not getattr(tc, "is_pec", False):
                continue
            try:
                c1, c2 = _sheet_bounds(tc.shape)
            except (NotImplementedError, TypeError, AttributeError):
                continue
            if c1 is None or c2 is None:
                continue
            bboxes.append((f"thin_conductor[{_ti}]", tuple(c1), tuple(c2)))
        points: list[tuple[str, tuple]] = []
        for pe in self._ports:
            points.append(("port/source", tuple(pe.position)))
        # Probes intentionally excluded (issue #303): a DFT probe is a
        # passive observer and does not radiate or scatter.

        for side, axis, coord, tang in faces:
            other = [a for a in range(3) if a != axis]
            min_gap = float("inf")
            culprit = None
            # bbox distances
            for name, c1, c2 in bboxes:
                # tangential overlap check — only meaningful gap if the face
                # is "above" the feature in the normal direction
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if c2[idx] <= tlo or c1[idx] >= thi:
                        overlap = False
                        break
                if not overlap:
                    continue
                if coord <= c1[axis]:
                    d = c1[axis] - coord
                elif coord >= c2[axis]:
                    d = coord - c2[axis]
                else:
                    d = 0.0  # already handled by CHECK 2 for PEC; skip
                    continue
                if d < min_gap:
                    min_gap, culprit = d, f"geometry '{name}'"
            # points
            for name, pos in points:
                # require tangential in-box for relevance
                in_tang = all(
                    tang[i][0] <= pos[other[i]] <= tang[i][1] for i in range(2)
                )
                if not in_tang:
                    continue
                d = abs(coord - pos[axis])
                if d < min_gap:
                    min_gap, culprit = d, f"{name} at {pos}"

            if culprit is not None and min_gap < gap_thresh:
                _w.warn(
                    PreflightWarning(
                        f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                        f"from {culprit} — below λ/4 = {gap_thresh*1e3:.2f}mm at "
                        f"f_max={f_max/1e9:.2f}GHz. NTFF will integrate reactive "
                        f"near-field; directivity / pattern likely corrupted. "
                        f"Move NTFF box ≥ λ/2 from any radiating/scattering "
                        f"structure (Huygens-equivalence rule).",
                        code="ntff_near_field",
                        source="_validate_ntff_inverse_design",
                    ),
                    stacklevel=3,
                )
            elif culprit is not None and min_gap < huygens_thresh:
                _w.warn(
                    PreflightWarning(
                        f"NTFF face {'xyz'[axis]}_{side} is {min_gap*1e3:.2f}mm "
                        f"from {culprit} — below λ/2 = {huygens_thresh*1e3:.2f}mm "
                        f"at f_max={f_max/1e9:.2f}GHz. Close to reactive near-"
                        f"field; far-field pattern accuracy may degrade. Move "
                        f"NTFF box ≥ λ/2 from radiating/scattering structures.",
                        code="ntff_near_field",
                        source="_validate_ntff_inverse_design",
                    ),
                    stacklevel=3,
                )

        # CHECK 4 (issue #334): electrically small ground plane under a
        # radiator. Advisory in BOTH tiers — it is pattern physics, not an
        # inverse-design structural gate.
        self._validate_ntff_small_ground_plane(f_max, lam_min)

    def _validate_ntff_small_ground_plane(
        self, f_max: float, lam: float,
    ) -> None:
        """CHECK 4 (issue #334): finite PEC sheet backing a radiator that is
        under ~1λ across → edge-diffraction-shaped far-field pattern.

        Background: a 0.48λ × 0.44λ ground plane produces a pattern dominated
        by ground-plane edge diffraction (broadside dip, off-axis side peaks)
        — correct physics for that geometry, but a trap when the fixture was
        built for resonance/impedance work and its pattern is then read as a
        solver defect. The advisory names the mechanism up front.

        Predicate (warning-severity, fires at most ONCE per preflight):
        - a PEC geometry entry is sheet-like: thin-axis extent
          ``t <= max(λ/20, L_small/10)`` with both lateral extents >= λ/8;
        - a radiator backs it: an ``add_source()`` / lumped-wire
          ``add_port()`` entry sits laterally inside the sheet footprint and
          within half a wavelength of the sheet along the thin axis
          (image-theory coupling zone);
        - among qualifying sheets only the LARGEST footprint is judged — in a
          patch stack that is the ground plane, never the (intentionally
          sub-wavelength) resonant patch element itself;
        - fire iff that sheet's smaller lateral extent < 1λ at the highest
          requested NTFF frequency (sub-wavelength across the whole
          requested pattern band — the conservative direction).

        λ is evaluated at ``f_max`` of the NTFF frequencies: if the sheet is
        sub-wavelength even at the shortest requested wavelength, every bin
        of the requested pattern carries the edge-diffraction shaping.

        TFSF and MSL/waveguide/coax excitations are not counted as radiators
        here (same scope as the CHECK 3 point list): a sub-wavelength PEC
        plate as a scattering target is a legitimate RCS fixture, not a
        ground-plane misuse.
        """
        import warnings as _w

        ports = [tuple(pe.position) for pe in self._ports]
        if not ports:
            return

        best = None  # (lateral_area, L_small, L_big, c1, c2)
        # #931: a ground plane is canonically a SHEET declaration
        # (add_thin_conductor / zero-thickness Box), so PEC thin
        # conductors are in the census beside PEC geometry entries.
        from rfx.materials.thin_conductor import sheet_bounds as _sheet_bounds
        candidates = []
        for entry in self._geometry:
            if entry.material_name != "pec":
                continue
            try:
                c1, c2 = entry.shape.bounding_box()
            except (NotImplementedError, TypeError, AttributeError):
                continue
            candidates.append((c1, c2))
        for tc in getattr(self, "_thin_conductors", ()):
            if not getattr(tc, "is_pec", False):
                continue
            try:
                c1, c2 = _sheet_bounds(tc.shape)
            except (NotImplementedError, TypeError, AttributeError):
                continue
            if c1 is None or c2 is None:
                continue
            candidates.append((tuple(c1), tuple(c2)))
        for c1, c2 in candidates:
            ext = [c2[a] - c1[a] for a in range(3)]
            thin = min(range(3), key=lambda a: ext[a])
            lat = [a for a in range(3) if a != thin]
            l_small = min(ext[lat[0]], ext[lat[1]])
            l_big = max(ext[lat[0]], ext[lat[1]])
            # sheet-like: electrically thin, or thin relative to its own
            # footprint (covers coarse-meshed few-cell-thick ground planes)
            if ext[thin] > max(lam / 20.0, l_small / 10.0):
                continue
            # electrically non-negligible in BOTH lateral dims — wires,
            # narrow straps and tiny pads are not ground planes
            if l_small < lam / 8.0:
                continue
            backed = False
            for pos in ports:
                if not all(c1[a] <= pos[a] <= c2[a] for a in lat):
                    continue
                d = max(c1[thin] - pos[thin], pos[thin] - c2[thin], 0.0)
                if d <= lam / 2.0:
                    backed = True
                    break
            if not backed:
                continue
            area = ext[lat[0]] * ext[lat[1]]
            if best is None or area > best[0]:
                best = (area, l_small, l_big, c1, c2)

        if best is None:
            return
        _, l_small, l_big, c1, c2 = best
        if l_small >= lam:
            return  # ground plane >= ~1λ both ways: clean-pattern regime

        _w.warn(
            PreflightWarning(
                f"Far-field pattern advisory: the PEC sheet backing a source "
                f"(bbox ({c1[0]*1e3:.1f}, {c1[1]*1e3:.1f}, {c1[2]*1e3:.1f})"
                f"–({c2[0]*1e3:.1f}, {c2[1]*1e3:.1f}, {c2[2]*1e3:.1f}) mm) "
                f"spans {l_big*1e3:.1f}mm × {l_small*1e3:.1f}mm = "
                f"{l_big/lam:.2f}λ × {l_small/lam:.2f}λ at "
                f"f_max={f_max/1e9:.2f}GHz — a ground plane under ~1λ "
                f"across. Expect the radiation pattern to be shaped by "
                f"ground-plane edge diffraction (broadside dip, off-axis "
                f"side peaks). This is expected physics, not a solver "
                f"defect, and the fixture stays fine for resonance / "
                f"impedance work. For a clean broadside pattern enlarge the "
                f"ground plane to at least ~1.4λ; if the small ground plane "
                f"is intentional, interpret the pattern accordingly.",
                code="ntff_small_ground_plane",
                source="_validate_ntff_small_ground_plane",
            ),
            stacklevel=4,
        )

    def _validate_simulation_config(self) -> None:
        """Comprehensive pre-simulation configuration validation.

        Checks for common setup mistakes that produce silent wrong results:
        probe/source in CPML, boundary type mismatch, feature compatibility,
        NTFF precision, normalize defaults.

        Called from run() after _validate_mesh_quality().

        Stage 1b refactor (2026-05-17): the original ~592-line body was
        decomposed into per-check ``_validate_cfg_*`` helpers. This method
        keeps its signature and remains the public entry point; its body
        computes the shared local state (``dx``, CPML thicknesses,
        ``absorber_label``) and then calls each helper IN THE SAME ORDER
        as the original checks. No logic, ordering, or warning text
        changed — pure readability decomposition.
        """
        import warnings as _w

        dx = self._dx or C0 / self._freq_max / 20.0
        cpml_thickness = self._cpml_layers * dx if self._boundary in ("cpml", "upml") else 0

        cpml_thick_lo, cpml_thick_hi, _pmc_faces_set = (
            self._validate_cfg_compute_cpml_thickness(cpml_thickness)
        )
        absorber_label = "UPML" if self._boundary == "upml" else "CPML"

        # --- checks in original order ---------------------------------
        self._validate_cfg_precision_x64(_w)
        self._validate_cfg_pec_faces_with_finite_pec(_w)
        self._validate_cfg_upml_refinement()
        self._validate_cfg_upml_nonuniform_lane(_w)
        self._validate_cfg_floquet_nonuniform()
        self._validate_cfg_absorber_placement(
            _w, dx, cpml_thickness, cpml_thick_lo, cpml_thick_hi, absorber_label
        )
        self._validate_cfg_source_on_reflector_plane(_w, dx, _pmc_faces_set)
        self._validate_cfg_ntff_absorber_overlap(
            _w, cpml_thickness, cpml_thick_lo, cpml_thick_hi, absorber_label
        )
        self._validate_cfg_ntff_min_steps(dx)
        self._validate_cfg_settling_witness_present(_w)
        self._validate_cfg_geometry_in_cpml(
            _w, cpml_thickness, cpml_thick_lo, cpml_thick_hi, absorber_label
        )
        self._validate_cfg_port_inside_pec(_w, dx)
        self._validate_cfg_floating_single_cell_port(_w)
        self._validate_cfg_pec_boundary_open_structure(_w)
        self._validate_cfg_no_sources(_w)
        self._validate_cfg_tfsf_with_lumped_rlc(_w)
        self._validate_cfg_unresolved_pulse(_w, dx)
        self._validate_cfg_thin_conductor_surface_impedance(_w)
        self._validate_cfg_thin_conductor_graded_node(_w)
        self._validate_cfg_source_on_graded_node(_w)
        self._validate_cfg_wire_port_on_graded_node(_w)
        self._validate_cfg_nonuniform_limitations(_w, cpml_thickness)
        self._validate_cfg_multiband_grading(_w)
        self._validate_cfg_graded_box_rasterization(_w)
        self._validate_cfg_subgrid_limitations(_w)
        self._validate_cfg_conformal_fine_dx(dx)
        self._validate_cfg_adi_3d_accuracy(_w)
        self._validate_cfg_adi_interior_pec(_w)
        self._validate_cfg_lossless_resonator_in_absorber(_w)
        self._validate_cfg_dispersive_pole_at_absorber_face(
            _w, dx, cpml_thick_lo, cpml_thick_hi
        )
        self._validate_cfg_waveguide_reference_plane(
            _w, cpml_thick_lo, cpml_thick_hi
        )
        self._validate_cfg_refplane_placement(_w)
        self._validate_cfg_absorber_budget_vs_grid(_w, dx)
        self._validate_cfg_campaign_statics(_w)

        self._check_waveguide_port_evanescent()
        self._check_msl_port_geometry(dx, cpml_thick_lo, cpml_thick_hi)
        self._check_coaxial_port_junction_aperture()

    def _validate_cfg_precision_x64(self, _w) -> None:
        """Warn when ``precision`` cannot actually take effect.

        Two independent ways ``precision != "float32"`` silently degrades
        back to float32 with no error:

        1. ``precision="float64"`` requires JAX x64 mode already enabled by
           the caller (``jax.config.update("jax_enable_x64", True)`` or
           ``jax.experimental.enable_x64()``) — process-global JAX
           behavior, not something ``Simulation`` can flip on its own: this
           package never flips ``jax_enable_x64`` at import/module scope,
           since that would be permanent for the rest of the process and
           is the caller's decision to make, not a library's to make for
           them. Without this check, ``precision="float64"`` would look
           accepted (no error) while silently running float32 fields
           (issue #630: the Yee update arithmetic used to re-quantize
           float64 fields to float32 every timestep even when storage WAS
           float64 -- that half is fixed, but storage never becoming
           float64 in the first place is a distinct, still-live footgun).
        2. The non-uniform mesh runner (``rfx/runners/nonuniform.py``) does
           not thread ``field_dtype`` at all (issue #630 review) -- a
           non-uniform-mesh sim with ``precision="mixed"`` or
           ``"float64"`` silently runs float32 fields regardless. This is
           an ADVISORY heads-up only: ``_dispatch_plan`` (the single
           lane-decision point) hard-rejects the same combination with
           ``NotImplementedError`` before any compute runs, so this warning
           cannot actually be missed in practice -- it exists to explain
           the coming error before the run gets that far, and to cover the
           ``skip_preflight=True`` escape hatch, which does NOT bypass
           ``_dispatch_plan``'s own guard. The distributed lanes
           (``distributed.py``/``distributed_nu.py``/``distributed_v2.py``)
           have the same gap but ``distributed=True`` is a call-time
           ``run()``/``forward()`` kwarg, not visible here at
           ``Simulation``-construction-time preflight (matches the
           established "P3 (Distributed path)" precedent in
           ``_validate_cfg_subgrid_limitations`` below) -- ``_dispatch_plan``
           is therefore the ONLY enforcement point for the distributed
           case, not merely a backstop.

        Both are exactly the SILENT_WRONG class this preflight system
        exists to catch.
        """
        if self._precision == "float64" and not jax.config.jax_enable_x64:
            _w.warn(PreflightWarning(
                "precision='float64' was requested, but JAX x64 mode is not "
                "enabled (jax.config.jax_enable_x64 is False). JAX silently "
                "downcasts float64 arrays to float32, so fields will run at "
                "float32 despite this setting. Enable x64 before constructing "
                "this Simulation: jax.config.update('jax_enable_x64', True) "
                "(process-global) or wrap the call in "
                "jax.experimental.enable_x64() (scoped).",
                code="precision_float64_without_x64",
                source="_validate_cfg_precision_x64",
            ))

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if self._precision != "float32" and is_nonuniform:
            _w.warn(PreflightWarning(
                f"precision={self._precision!r} was requested on a "
                "non-uniform mesh (dx/dy/dz profile set), but the "
                "non-uniform runner does not thread field_dtype -- fields "
                "would silently run float32 regardless of this setting "
                "(issue #630). run()/forward() will raise NotImplementedError "
                "at dispatch rather than proceed silently; this warning "
                "exists to explain that error before you hit it. Use "
                "precision='float32' (the default) with a non-uniform mesh, "
                "or drop the mesh profile to reach the uniform lane where "
                "this precision knob is currently supported.",
                code="precision_nonuniform_lane_unsupported",
                source="_validate_cfg_precision_x64",
            ))

    def _validate_cfg_tfsf_with_lumped_rlc(self, _w) -> None:
        """Warn: a lumped RLC element illuminated by a TFSF plane wave is unstable.

        A ``add_lumped_rlc(...)`` element driven by a TFSF plane wave diverges
        (measured: blow-up to ~1e35 by ~250 steps, C-independent). The root cause is
        the TFSF total/scattered-field decomposition coupling into the lumped ADE
        current, NOT a missing circuit path: embedding the element in a PEC-gap
        structure does NOT cure it (tested 2026-07-22, #425 — a two-electrode PEC gap
        still grows 0.1→1e25→NaN over ~800 steps; see
        docs/research_notes/experiments/tfsf_lumped_pec_gap_stability.py). So there is
        no geometry fix at the API level; a stable plane-wave lumped lane needs a
        solver-level fix to the TFSF↔lumped coupling. The tunable-load (varactor)
        gradient IS validated on the PORT-fed lane (``add_port`` +
        ``forward(rlc_values_override=...)`` — tests/unit/autodiff/test_lumped_rlc_ad.py); use that
        for varactor/RIS design. See the tracking issue (#425).
        """
        if self._tfsf is None or not self._lumped_rlc:
            return
        _w.warn(PreflightWarning(
            "add_lumped_rlc(...) + a TFSF plane-wave source is numerically unstable "
            "(fields diverge, C-independent). This is the TFSF↔lumped-ADE coupling, "
            "not a missing circuit path — a PEC-gap structure does NOT cure it "
            "(tested, #425). Use the validated PORT-fed lane (add_port + "
            "forward(rlc_values_override=...)) for varactor/tunable-load design.",
            code="tfsf_lumped_rlc_unstable",
            source="_validate_cfg_tfsf_with_lumped_rlc",
        ))

    def _validate_cfg_graded_box_rasterization(self, _w) -> None:
        """Warn when a Box misses the fine cells implied by its z-span."""
        if self._dz_profile is None or is_tracer(self._dz_profile):
            return

        dz = np.asarray(self._dz_profile, dtype=np.float64)
        if dz.size == 0:
            return
        edges = np.concatenate(([0.0], np.cumsum(dz)))
        # Sample positions must be the ones the RUN uses, and the occupancy
        # rule must be the run's rule. This validator previously modelled both
        # by hand — cell centres plus a bare half-open span — and was wrong on
        # each count: dielectric coordinates are E-NODES (cell edges) since
        # #562 (the node sampler's thin-sheet branch snaps a sub-cell
        # dielectric Box onto its nearest node, #48/#75/#371). With the hand
        # model this check diverged from the rasterizer on boxes straddling a
        # grading transition — the #325 signature it exists to catch — and
        # went SILENT on one of them (#562 review, F2). So call the production
        # path instead of imitating it: node positions through the same two
        # steps the grid builder composes, and the shape's own mask for a
        # dielectric; for a PEC Box the production rule is the #931 §1.1
        # CENTRE-sampled volume rule (half-open on cell centres), spelled by
        # ``rasterize_grid._box_axis_volume`` and called here. x/y are passed
        # as single-sample arrays because only the z profile is needed here;
        # the mask admits the box's own midpoint, so the combined mask's z
        # profile is the z-axis mask.
        from rfx.nonuniform import node_positions_from_profile
        z_nodes = np.asarray(node_positions_from_profile(dz), dtype=np.float64)

        z_centres = z_nodes[:-1] + 0.5 * np.diff(z_nodes)
        for entry in self._geometry:
            if not isinstance(entry.shape, Box):
                continue
            c1, c2 = entry.shape.bounding_box()
            z_lo, z_hi = sorted((float(c1[2]), float(c2[2])))
            thickness = z_hi - z_lo
            if thickness <= 0.0:
                continue

            x_mid = np.array([0.5 * (float(c1[0]) + float(c2[0]))])
            y_mid = np.array([0.5 * (float(c1[1]) + float(c2[1]))])
            try:
                is_pec = (self._resolve_material(entry.material_name).sigma
                          >= self._PEC_SIGMA_THRESHOLD)
            except KeyError:
                is_pec = False
            if is_pec:
                # #931 §1.1: a PEC VOLUME is centre-sampled, half-open on
                # the cell centres — the production rule
                # (rasterize_grid._box_axis_volume), not the node sampler.
                from rfx.geometry.rasterize_grid import _box_axis_volume
                actual = int(np.count_nonzero(
                    np.asarray(_box_axis_volume(z_centres, z_lo, z_hi))))
            else:
                mask = np.asarray(entry.shape.mask_on_coords(x_mid, y_mid, z_nodes))
                actual = int(np.count_nonzero(mask))
            local = (edges[:-1] < z_hi) & (edges[1:] > z_lo)
            if not np.any(local):
                continue
            # The #325 signature is a fine band SHIFTED OUT of the Box span by
            # smooth_grading's transition insertion — the intended fine cells
            # then sit ADJACENT to the span, so measure min dz over a padded
            # neighborhood (±5 cells), not the span alone (span-only misses
            # the shifted-substrate case entirely).
            idx = np.flatnonzero(local)
            lo_i = max(0, int(idx[0]) - 5)
            hi_i = min(dz.size, int(idx[-1]) + 6)
            implied = thickness / float(np.min(dz[lo_i:hi_i]))

            if actual < math.ceil(0.5 * implied) and actual <= 4:
                _w.warn(
                    PreflightWarning(
                        f"Box material '{entry.material_name}' rasterizes to "
                        f"{actual} z cells (implied {implied:.1f}) over z-span "
                        f"[{_fmt_len(z_lo)}, {_fmt_len(z_hi)}). "
                        "smooth_grading transition cells may have shifted the "
                        "fine band — derive z coordinates from the actual "
                        "fine-band edges and assert the rasterized cell count "
                        "(issue #325)",
                        code="graded_box_rasterization",
                        loc=f"z=[{z_lo}, {z_hi})",
                        source="_validate_cfg_graded_box_rasterization",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_refplane_placement(self, _w) -> None:
        """Advisories for ``add_port(reference_plane_cells=...)`` (issue #313).

        (a) ``reference_plane_cells < 10`` puts the measurement planes in the
        port near field.  Measured on the canonical 16 mm thru (dx = 0.5 mm,
        gap-trimmed V, 2026-07-10 battery): N=3 planes read Zc = 52-53 ohm
        with |Im/Re| up to 8.2%, beta/(w/c) = 1.16-1.20, and a -3.1%
        closed-box-referee |S21| residual, while N=10 planes read the clean
        mid-line constants.  The Phase-0 pre-registration rule places BOTH
        planes (N and 2N cells) >= 10 cells from every port.

        (b) When SOME but not ALL impedance-carrying wire ports opt in, the
        off-diagonal S entries involving a non-opted port silently stay on
        the legacy port-cell path (only pairs where both ports opt in use
        the plane waves) — surface that at preflight instead of letting the
        mixed matrix pass unremarked.
        """
        wire_ports = [
            pe for pe in self._ports
            if pe.impedance != 0.0 and pe.extent is not None
        ]
        opted = [
            pe for pe in wire_ports
            if getattr(pe, "reference_plane_cells", None) is not None
        ]
        if not opted:
            return
        near = [pe for pe in opted if pe.reference_plane_cells < 10]
        if near:
            _w.warn(
                PreflightWarning(
                    f"{len(near)} reference-plane port(s) use "
                    "reference_plane_cells < 10 — planes this close sit in "
                    "the port near field. Measured on the canonical thru "
                    "(dx = 0.5 mm, 2026-07-10 battery): N=3 planes read "
                    "Zc = 52-53 ohm with |Im(Zc)/Re(Zc)| up to 8.2% and "
                    "beta/(w/c) = 1.16-1.20 (vs the clean mid-line "
                    "constants at N=10), and the closed-box-referee |S21| "
                    "residual was -3.1%. The Phase-0 pre-registration rule "
                    "(issue #313) places BOTH planes (N and 2N cells) >= 10 "
                    "cells from every port — prefer "
                    "reference_plane_cells >= 10 when the line length "
                    "allows it.",
                    code="refplane_near_field",
                    source="_validate_cfg_refplane_placement",
                ),
                stacklevel=2,
            )
        if len(opted) != len(wire_ports):
            _w.warn(
                PreflightWarning(
                    f"{len(opted)} of {len(wire_ports)} impedance-carrying "
                    "wire ports opt into reference_plane_cells — "
                    "off-diagonal S entries involving a non-opted port "
                    "SILENTLY stay on the legacy port-cell path (only "
                    "pairs where BOTH ports opt in use the plane waves). "
                    "Opt in every wire port of the S-matrix, or none.",
                    code="refplane_partial_optin",
                    source="_validate_cfg_refplane_placement",
                ),
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 2: the conformal-fine-dx guard moved VERBATIM to
    # ``rfx/preflight/pec_geometry.py`` and is bound back here, AT THE
    # POSITION it held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls these checks in a fixed
    # sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    # ------------------------------------------------------------------
    from rfx.preflight.pec_geometry import _validate_cfg_conformal_fine_dx


    def _validate_adi_interior_pec(self, pec_edge_masks) -> None:
        """Unskippable lane guard shared by run and forward after assembly."""
        from rfx.adi import _validate_interior_pec
        _validate_interior_pec(pec_edge_masks)

    def _validate_cfg_adi_interior_pec(self, _w) -> None:
        """Report the same refusal from the production conductor assembly."""
        if self._solver != "adi" or self._mode not in ("3d", "2d_tmz"):
            return
        if not self._geometry and not self._thin_conductors:
            return
        from rfx.adi import ADI_INTERIOR_PEC_MESSAGE
        ctx = self._campaign_ctx()
        realized = None if ctx.error else ctx.realized()
        if realized is not None and realized.empty:
            return
        detail = ""
        if realized is None:
            detail = (
                f" Interior PEC compatibility could not be evaluated: "
                f"{ctx.error or ctx.assembly_error}. Resolve assembly first.")
        _w.warn(PreflightWarning(
            ADI_INTERIOR_PEC_MESSAGE + detail,
            code="adi_interior_pec_unsupported",
            severity="error",
            source="_validate_cfg_adi_interior_pec",
        ), stacklevel=2)

    def _validate_cfg_adi_3d_accuracy(self, _w) -> None:
        """Advise on the 3D ADI large-timestep accuracy envelope (OPT-C1 fixed).

        HISTORY: until 2026-07-13 the 3D ADI path was an LOD split with
        artificial diffusion (OPT-C1) and this validator flagged it as
        KNOWN-INACCURATE unconditionally. ``adi_step_3d`` now implements the
        full Zheng–Chen–Zhang two-sub-step 3D ADI (issue #338 follow-up):
        eigenfrequency error measured 1.2% at 2x CFL on the 12^3 PEC-cavity
        adjudication test (``tests/unit/misc/test_review_tier1_validation_battery.py::
        test_optc1_adi_3d_cavity_eigenfrequency``, 2% gate, ~15 cells/wave).

        What remains is the honest large-dt envelope of ANY Crank–Nicolson-
        class implicit scheme: dispersion error grows ~dt^2, so at ~15
        cells per wavelength the <2% eigenfrequency envelope holds only for
        CFL factors up to ~2x (von Neumann: -1.4% at 2x, -2.8% at 3x, -6.7%
        at 5x). This envelope is for a cavity without interior PEC; it is
        not a stability guarantee for projected conductors or arbitrary
        material interfaces. Advise (WARNING severity) when
        ``adi_cfl_factor > 2.0`` on a 3D grid.
        The 2D TMz cavity accuracy check is separate. Interior PEC on
        either lane is refused by the independent conductor guard.
        """
        if self._solver != "adi" or self._mode != "3d":
            return
        if self._adi_cfl_factor <= 2.0:
            return
        _w.warn(
            PreflightWarning(
                f"solver='adi' with a 3D grid at adi_cfl_factor="
                f"{self._adi_cfl_factor:g}: the homogeneous, lossless ADI "
                f"split with compatible domain boundaries removes the "
                f"explicit CFL stability restriction. This does not "
                f"guarantee stability with interior PEC projection "
                f"(refused) or arbitrary material interfaces. For the "
                f"cavity without interior PEC, dispersion error grows "
                f"~dt^2 — at ~15 cells/wavelength the <2% eigenfrequency "
                f"envelope holds only up to ~2x CFL (measured -1.4% at 2x; "
                f"-2.8% at 3x, -6.7% at 5x by von Neumann analysis). Use "
                f"adi_cfl_factor <= 2 for wavelength-scale accuracy, or "
                f"reserve large factors for geometrically stiff meshes "
                f"(features far below the wavelength).",
                code="adi_3d_accuracy",
                severity="warning",
                source="_validate_cfg_adi_3d_accuracy",
            ),
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the eight contiguous absorber-configuration
    # checks -- lossless-Q, the #636 dispersive pole at a face, per-face
    # CPML thickness, the #647/#742 budget advisory, the shared
    # ``_preflight_face_layers``, pec_faces-vs-finite-PEC, and the two
    # UPML refusals -- moved VERBATIM to ``rfx/preflight/absorber.py`` and
    # are bound back here, AT THE POSITION they held in this class body.
    # Position is not cosmetic: ``_validate_simulation_config`` calls these
    # checks in a fixed sequence and the resulting advisory ORDER is the
    # observable ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_abs`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._preflight_face_layers()`` a bound method with its name,
    # signature and ``__doc__`` intact.
    #
    # ``_preflight_face_layers`` is the family's shared member and the
    # reason this block cannot be split: three bodies OUTSIDE the absorber
    # family call it -- ``_validate_cfg_multiband_grading`` and
    # ``_validate_cfg_nonuniform_limitations`` below, and
    # ``_validate_cfg_pec_face_short_of_domain_wall`` in
    # ``rfx/preflight/pec_geometry.py``. All three reach it through
    # ``self.`` on the composed ``Simulation``, so the rebind below is what
    # keeps them working and no call site changed.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import (
        _validate_cfg_lossless_resonator_in_absorber,
        _validate_cfg_dispersive_pole_at_absorber_face,
        _validate_cfg_compute_cpml_thickness,
        _validate_cfg_absorber_budget_vs_grid,
        _preflight_face_layers,
        _validate_cfg_pec_faces_with_finite_pec,
        _validate_cfg_upml_refinement,
        _validate_cfg_upml_nonuniform_lane,
    )

    def _validate_cfg_floquet_nonuniform(self) -> None:
        """P1.1: Floquet + non-uniform mesh — no silent fallback allowed."""
        if self._floquet_ports and self._dz_profile is not None:
            raise PreflightConfigError(
                "Floquet ports do not support non-uniform z mesh (dz_profile). "
                "Use the uniform reference lane and set dx explicitly.",
                code="floquet_nonuniform",
                source="_validate_cfg_floquet_nonuniform",
            )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the probe/source placement check moved VERBATIM
    # to ``rfx/preflight/absorber.py``, bound back at its original position
    # for the reason the block above gives. It is separated from that block
    # by ``_validate_cfg_floquet_nonuniform``, which is not an absorber
    # check and stays here.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import _validate_cfg_absorber_placement

    def _validate_cfg_source_on_reflector_plane(
        self, _w, dx: float, _pmc_faces_set: set
    ) -> None:
        """P1.6: Source / port placed ON a PEC or PMC face plane. Both
        reflectors zero specific field components at the plane every
        time step (PEC: tangential E; PMC: tangential H); a source
        that drives a zeroed component is silently discarded. A
        source that drives a component forced to zero by the mirror
        image (e.g. normal E on a PMC face) fights the symmetry and
        yields numerically inconsistent results.

        Component-specific rule:
          PEC face (axis = ax_name): tangential E (Ex/Ey/Ez with
            component axis != ax_name) is zeroed every E update.
            Normal E (component axis == ax_name) is the legitimate
            way to drive a PEC mirror.
          PMC face (axis = ax_name): tangential H (Hx/Hy/Hz with
            component axis != ax_name) is zeroed; the outgoing
            wave from an on-plane tangential E source is killed via
            this H zeroing. Normal E (component axis == ax_name) is
            odd-symmetric and must be zero at the plane by image,
            so injecting it fights the mirror.

        This follows the industry convention (Meep / OpenEMS /
        Tidy3D all follow the same rule).
        """
        _all_reflector_faces = set(self._pec_faces) | set(_pmc_faces_set)
        if _all_reflector_faces:
            _dx_axis = [float(dx), float(dx), float(dx)]
            if (self._dz_profile is not None
                    and not is_tracer(self._dz_profile)):
                _dx_axis[2] = float(self._dz_profile[0])
            for face in _all_reflector_faces:
                ax_name = face[0]
                side = face[2:]
                ax_i = "xyz".index(ax_name)
                face_kind = "PMC" if face in _pmc_faces_set else "PEC"
                d_ext = self._domain[ax_i] if ax_i < len(self._domain) else self._domain[-1]
                plane_coord = 0.0 if side == "lo" else float(d_ext)
                tol = 0.5 * _dx_axis[ax_i]
                for pe in self._ports:
                    pos = pe.position
                    coord = pos[ax_i]
                    if abs(coord - plane_coord) > tol:
                        continue
                    # Classify the source component vs. the face axis.
                    comp = pe.component.lower()
                    comp_field = comp[0]       # 'e' or 'h'
                    comp_axis = comp[1:]       # 'x' / 'y' / 'z'
                    is_tangential = (comp_axis != ax_name)
                    if face_kind == "PMC":
                        if comp_field == "e" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane. The outgoing "
                                f"tangential H is zeroed every step by "
                                f"apply_pmc_faces, so no wave radiates — the "
                                f"probe records silent zero field. Offset by "
                                f"one cell ({_dx_axis[ax_i]*1e3:.3g} mm) off "
                                f"the plane to let the Yee curl run normally."
                            )
                        elif comp_field == "e" and not is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane and drives the "
                                f"NORMAL E component. PMC imposes odd symmetry "
                                f"on normal E (it must be zero at the plane), "
                                f"so the source fights the mirror image. Use a "
                                f"tangential E source offset by one cell "
                                f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        elif comp_field == "h" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PMC {face} plane and drives a "
                                f"tangential H. apply_pmc_faces zeros this "
                                f"component at the plane every step, so the "
                                f"source has no effect."
                            )
                        else:
                            msg = None      # normal H on PMC plane is legit
                    else:                    # PEC
                        if comp_field == "e" and is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PEC {face} plane and drives a "
                                f"tangential E. PEC zeros E_tan at the plane "
                                f"every step, so the source is silently "
                                f"discarded. Use a normal E source at this "
                                f"face, or offset by one cell "
                                f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        elif comp_field == "h" and not is_tangential:
                            msg = (
                                f"Source/port at {pos} (component={pe.component}) "
                                f"sits on the PEC {face} plane and drives the "
                                f"NORMAL H component. PEC imposes odd symmetry "
                                f"on normal H (it must be zero at the plane). "
                                f"Use a tangential H source or offset by one "
                                f"cell ({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                            )
                        else:
                            msg = None      # tangential H or normal E on PEC is legit
                    if msg is not None:
                        _w.warn(
                            PreflightWarning(
                                msg,
                                code="source_decoupled",
                                source="_validate_cfg_source_on_reflector_plane",
                            ),
                            stacklevel=3,
                        )

    def _validate_cfg_ntff_absorber_overlap(
        self,
        _w,
        cpml_thickness: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
        absorber_label: str,
    ) -> None:
        """P1.4: NTFF box overlap with absorber.

        Issue #500: uses :func:`_absorber_boundary_for_axis` — the CPML
        pad is EXTERIOR to ``[0, domain_extent]`` (see that helper), so an
        NTFF corner is only in the absorber when it is genuinely outside
        the requested domain, not merely within ``ct_{lo,hi}`` of an edge.
        """
        if self._ntff is not None and cpml_thickness > 0:
            corner_lo, corner_hi, _ = self._ntff
            for ax in range(3):
                domain_ext = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                ax_i = min(ax, 2)
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                lo_b, hi_b = _absorber_boundary_for_axis(domain_ext, ct_lo, ct_hi)
                if (lo_b is not None and corner_lo[ax] < lo_b) or (
                    hi_b is not None and corner_hi[ax] > hi_b
                ):
                    _w.warn(
                        PreflightWarning(
                            f"NTFF box extends into {absorber_label} region along "
                            f"{'xyz'[ax]}-axis. Far-field results will be "
                            f"corrupted. Shrink NTFF box to interior.",
                            code="absorber_overlap",
                            source="_validate_cfg_ntff_absorber_overlap",
                        ),
                        stacklevel=3,
                    )
                    break

        # P1.5: non-uniform + NTFF is SUPPORTED (stale "unsupported" note removed
        # 2026-07-02). The NU runner accumulates the NTFF box and
        # compute_far_field handles graded-z per-cell dS + z-edges; a graded-z
        # dipole directivity benchmarks within ~0.05 dB of theory
        # (tests/unit/farfield/test_farfield_nonuniform.py). No guard needed.

    def _validate_cfg_ntff_min_steps(self, dx: float) -> None:
        """P1.7: NTFF with too few steps."""
        if self._ntff is not None:
            _, _, ntff_freqs = self._ntff
            if ntff_freqs is not None:
                min_freq = float(min(ntff_freqs))
                period = 1.0 / max(min_freq, 1.0)
                dt_est = dx / (C0 * 1.732) * 0.99  # CFL estimate
                min_steps_for_ntff = int(10 * period / dt_est)
                # Can't check n_steps here (not known yet), but store hint
                self._ntff_min_steps_hint = min_steps_for_ntff

    def _validate_cfg_settling_witness_present(self, _w) -> None:
        """Warn when the declared inputs cannot produce a ring-down witness.

        Input-side fact, not a result prediction: ``run()`` and ``forward()`` score their
        energy ring-down settling witness (#885) from the probe time series,
        so a simulation that registers NTFF or a field-DFT plane and NO
        point probe will come back with ``settling_db=None`` -- the
        claims-bearing open-domain DFT numbers this project's -40 dB
        settling rule governs, with the truncation guard absent. Cheaper to
        say here than after the run.

        Silent when a probe exists, and silent when the run asks for neither
        NTFF nor a field DFT (the rule scopes to those).

        Both entry points use the same recorded-probe arithmetic. Forward's
        host diagnostic is available on concrete results, including after
        an outer JIT returns; while records are traced it is unavailable.
        A registered probe does not imply coverage if its recording is
        disabled, too short, or below the storage-format underflow floor.
        """
        if self._ntff is None and not self._dft_planes:
            return
        if self._probes:
            return
        wants = []
        if self._ntff is not None:
            wants.append("NTFF far-field output")
        if self._dft_planes:
            wants.append("field-DFT plane probe(s)")
        _w.warn(PreflightWarning(
            f"this simulation requests {' and '.join(wants)} but registers no "
            "point probe. run() and forward() score their ring-down settling witness from "
            "the recorded probe time series, so the result will carry settling_db=None "
            "and those DFT numbers will have no truncation guard (#885); add "
            "sim.add_probe(position, component) somewhere the field is live "
            "and retain its time series. Inspect forward's diagnostic on "
            "the concrete result after JIT/AD evaluation.",
            code="settling_witness_will_be_absent",
            source="_validate_cfg_settling_witness_present",
        ))

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the #660 geometry-in-absorber reporting check
    # moved VERBATIM to ``rfx/preflight/absorber.py``, bound back at its
    # original position for the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import _validate_cfg_geometry_in_cpml

    def _validate_cfg_port_inside_pec(self, _w, dx: float) -> None:
        """P1.8: Port/source/probe frozen by realized PEC.

        Lattice ownership contract (#931 §1.7 / #929): every question this
        check asks is answered by the run's REALIZED edge set, read once
        from the production assembly with the sheet/wire collectors:

        * a wire-port extent cell is DEAD iff the port component's own E
          edge at that index is PEC (``_wire_port_live_cells``, the same
          primitive the assembler uses — issue #544 made the two share
          it, and #931 made the primitive read edges, so a sheet trace,
          which owns no cell, is visible here);
        * a point port / source / probe on an E component is dead iff its
          own edge is PEC; on an H component iff all four E edges of its
          curl loop are PEC (``curl E = 0`` freezes it). Half a cell above
          a SHEET only one loop edge is PEC, so an MSL diagnostic Hy probe
          at the trace plane stays live; inside a one-cell VOLUME all four
          are PEC and H is frozen, which is what a volume declaration
          means. The former ``<= 1.5 dx`` thickness exemption inferred
          sheet-ness from a bounding box — the #929 class — and is gone;
        * the #556/#929 end-gap advisory measures the separation from
          the actual source-end node to the nearest conductor on its own
          column. A volume face, sheet plane, or axial PEC edge ending at
          that node supplies contact. Contact is silent; a separation
          is reported without guessing the intended coupling mechanism.

        Non-uniform lane: the wire-port primitive cannot index a
        ``NonUniformGrid`` (no ``position_to_index``), so the wire-port
        advisories emit a "classification unavailable" note there rather
        than guess (issue #544 review item 6; #303 class); the point
        port/probe rule runs on both lanes through the shared context.
        """
        ctx = self._campaign_ctx()
        is_nonuniform = ctx.lane == "nonuniform"
        realized = None
        classification_unavailable_reason: str | None = None
        if ctx.error == "traced-mesh":
            classification_unavailable_reason = (
                "traced mesh (mesh-as-design-variable) -- no concrete "
                "node positions")
        elif ctx.error is not None:
            classification_unavailable_reason = ctx.error
        elif is_nonuniform:
            classification_unavailable_reason = (
                "non-uniform mesh (dz_profile/dx_profile/"
                "dy_profile set) -- the shared wire-port primitive "
                "only covers the uniform-grid path (issue #544)"
            )
        # A model with no conductor declaration has nothing that can
        # freeze a port: skip the production assembly (the expensive part
        # of the context) rather than run it to read an all-False set.
        has_conductor = bool(getattr(self, "_thin_conductors", None))
        if not has_conductor:
            try:
                has_conductor = any(
                    self._resolve_material(e.material_name).sigma
                    >= self._PEC_SIGMA_THRESHOLD for e in self._geometry)
            except KeyError:
                has_conductor = False
        if ctx.error is None and has_conductor:
            realized = ctx.realized()
            if realized is None:
                classification_unavailable_reason = ctx.assembly_error
        grid = ctx.grid
        entries = ctx.entry_realizations() if ctx.error is None else []
        shape = tuple(grid.shape) if grid is not None else None

        def _owners(component, idx):
            """Names of the declarations whose OWN realized edges freeze
            ``component`` at ``idx`` (attribution through the same
            function, never through a per-entry cell mask)."""
            names = []
            for e in entries:
                if not e.is_pec:
                    continue
                if _component_is_dead(e.edges(ctx.periodic, shape),
                                      component, idx):
                    if e.name not in names:
                        names.append(e.name)
            return names or ["unknown"]

        for pe in self._ports:
            if not getattr(pe, "extent", None):
                continue
            rasterized = self._wire_port_cell_centers(pe)
            if rasterized is None:
                continue
            centers, mid_idx = rasterized
            mid_center = centers[mid_idx]

            dead_indices: list[int] = []
            dead_names: list[str] = []
            if realized is not None and not is_nonuniform:
                from rfx.sources.sources import (
                    WirePort, _wire_port_cells, _wire_port_live_cells,
                )
                axis = {"ex": 0, "ey": 1, "ez": 2}[pe.component]
                end = list(pe.position)
                end[axis] += pe.extent
                wp = WirePort(
                    start=tuple(pe.position), end=tuple(end),
                    component=pe.component, impedance=pe.impedance,
                )
                try:
                    cells, live_flags, _ = _wire_port_live_cells(
                        grid, wp, realized.edges)
                except ValueError:
                    # Every extent cell is dead: _wire_port_live_cells
                    # raises there (issue #318 — such a port has no live
                    # cell to terminate or drive) instead of returning a
                    # degenerate split. Report all cells dead.
                    cells = _wire_port_cells(grid, wp)
                    live_flags = [False] * len(cells)
                dead_indices = [
                    idx for idx, live in enumerate(live_flags) if not live
                ]
                if dead_indices:
                    for idx in dead_indices:
                        for name in _owners(pe.component, cells[idx]):
                            if name not in dead_names:
                                dead_names.append(name)

                # Issue #556 (D5 follow-up, #488 arc): the OPPOSITE
                # failure mode of the #314/#319 advisories above. There,
                # a port extent cell lands ON PEC (dead cell). Here, the
                # port terminates SHORT of a conductor. On the
                # D5 "end-fed trace" fixture (dx=80um, h_sub=254um) the
                # trace's realization landed one full cell above the
                # wire's top. That historical D5 report described
                # |S21| rising with frequency (docs/research_notes/
                # 20260728_i488_falsifier_ledger.md). It motivated this
                # check, but a geometric gap alone does not identify its
                # electromagnetic coupling mechanism. No cell is dead in
                # that fixture, so #314/#319 are correctly silent.
                #
                # Contact (#931 §1.9, #929) is read from realized walls
                # and PEC source-axis edges on the terminal's own column.
                # The wire's last edge on the + side runs
                # from node ``cells[-1]`` to ``cells[-1] + 1``, so its end
                # node is ``cells[-1] + 1``; on the - side the end node
                # is ``cells[0]``. "Fires" = the end edge is live, the
                # end node has no conductor contact but a further node
                # on that column does. Search every realized candidate
                # (#929), not only a one-cell neighbour. An axial PEC edge
                # can supply contact without a tangential wall (filaments).
                # A dipole ending in open vacuum stays silent. A measured
                # separation alone does not determine the intended coupling.
                if cells and live_flags:
                    axis_letter = "xyz"[axis]
                    for end_idx, step in ((0, -1), (len(cells) - 1, +1)):
                        if not live_flags[end_idx]:
                            continue
                        end_node = list(cells[end_idx])
                        if step > 0:
                            end_node[axis] += 1
                        ij = tuple(end_node[t] for t in range(3)
                                   if t != axis)
                        if not all(0 <= end_node[t] < shape[t]
                                   for t in range(3)):
                            continue
                        planes = set(realized.wall_planes(axis, ij=ij))
                        line_index = tuple(slice(None) if a == axis else end_node[a]
                                           for a in range(3))
                        axial_edges = np.flatnonzero(
                            np.asarray(realized.edges[axis])[line_index])
                        # Looking downward, edge k ends at node k+1;
                        # looking upward, it begins at node k.
                        edge_nodes = {int(k)+(1 if step < 0 else 0)
                                      for k in axial_edges}
                        contacts = planes | edge_nodes
                        if end_node[axis] in contacts:
                            continue            # galvanic contact
                        candidates = [k for k in contacts
                                      if 0 <= k < shape[axis]
                                      and (k-end_node[axis])*step > 0]
                        if not candidates:
                            continue
                        beyond = min(candidates, key=lambda k: abs(k-end_node[axis]))
                        edge_index = list(end_node)
                        edge_index[axis] = beyond-(1 if step < 0 else 0)
                        adj_names = []
                        for e in entries:
                            if not e.is_pec:
                                continue
                            if beyond in e.wall_planes(axis, ctx.periodic,
                                                       shape):
                                foot = e.footprint_on_plane(
                                    axis, beyond, ctx.periodic, shape)
                                if bool(foot[ij]) and e.name not in adj_names:
                                    adj_names.append(e.name)
                        if beyond in edge_nodes:
                            for name in _owners(pe.component, tuple(edge_index)):
                                if name not in adj_names:
                                    adj_names.append(name)
                        adj_names = adj_names or ["unknown"]
                        side = ("+" if step > 0 else "-") + axis_letter
                        nodes_ax = ctx.nodes[axis]
                        gap_cells = abs(beyond-end_node[axis])
                        gap_m = abs(float(nodes_ax[beyond])-float(nodes_ax[end_node[axis]]))
                        kind = "wall plane" if beyond in planes else f"{pe.component}-edge endpoint"
                        _w.warn(
                            PreflightWarning(
                                f"Wire port at {pe.position} (extent "
                                f"{pe.extent}, component {pe.component}): "
                                f"its {side}-side end node "
                                f"{tuple(end_node)} ({axis_letter} = "
                                f"{_fmt_len(float(nodes_ax[end_node[axis]]))}) "
                                f"carries no realized PEC contact, but the "
                                f"nearest outward conductor node {beyond} "
                                f"({axis_letter} = "
                                f"{_fmt_len(float(nodes_ax[beyond]))}) is a "
                                f"realized {kind} of {adj_names}. The "
                                f"port end is "
                                f"{gap_cells} cell(s) (gap = {gap_m:g} m) short "
                                f"of that conductor. This is a geometric "
                                f"separation; it does not determine the "
                                f"intended electromagnetic coupling. If "
                                f"galvanic contact is intended, adjust the "
                                f"position and extent together so its end node "
                                f"lands on that wall plane or conductor endpoint, "
                                f"or correct the conductor declaration.",
                                code="wire_port_end_gap_to_conductor",
                                source="_validate_cfg_port_inside_pec",
                            ),
                            stacklevel=3,
                        )
            elif classification_unavailable_reason is not None:
                _w.warn(
                    PreflightWarning(
                        f"Wire port at {pe.position} (extent {pe.extent}): "
                        f"dead-cell classification unavailable "
                        f"({classification_unavailable_reason}). The "
                        f"#314/#319 PEC-overlap advisories and the #556 "
                        f"end-gap-to-conductor advisory are skipped "
                        f"for this port -- inspect the realized live/dead "
                        f"source edges and the intended terminal contacts "
                        f"on that mesh. Conductor overlap at an endpoint "
                        f"can be intentional; do not infer contact from "
                        f"the absence of this advisory.",
                        code="wire_port_dead_cell_classification_unavailable",
                        source="_validate_cfg_port_inside_pec",
                    ),
                    stacklevel=3,
                )

            if mid_idx in dead_indices:
                # Kept verbatim from the #314 fix (PR #317): probe-cell
                # corruption is the stronger, measured failure mode.
                name = dead_names[0] if dead_names else "an assembled PEC region"
                _w.warn(
                    PreflightWarning(
                        f"Wire port at {pe.position} (extent "
                        f"{pe.extent}): its MIDPOINT V/I probe cell "
                        f"(center {tuple(round(x, 6) for x in mid_center)}) "
                        f"lands inside PEC geometry "
                        f"'{name}'. S-parameters from "
                        f"this port are silently corrupted (measured: "
                        f"near-null forward transmission + over-unity "
                        f"reverse). Shorten/lengthen the extent or move "
                        f"the port so the midpoint cell sits in "
                        f"dielectric (issue #314).",
                        code="wire_port_midpoint_in_pec",
                        source="_validate_cfg_port_inside_pec",
                    ),
                    stacklevel=3,
                )

            non_midpoint_dead = [i for i in dead_indices if i != mid_idx]
            if non_midpoint_dead:
                n = len(centers)
                n_live = n - len(dead_indices)
                z0 = getattr(pe, "impedance", 0.0) or 0.0
                z_eff = z0 * n_live / n
                _w.warn(
                    PreflightWarning(
                        f"Wire port at {pe.position} (extent {pe.extent}) "
                        f"rasterizes to n={n} cells of which "
                        f"{len(dead_indices)} have their {pe.component} "
                        f"edge inside realized PEC "
                        f"{dead_names} (n_live/n = {n_live}/{n}). Dead "
                        f"cells are shorted by the PEC and are excluded "
                        f"from the port's resistance distribution, drive "
                        f"injection, and wave normalization (issue #318 "
                        f"fix): the port terminates at {z0:g} ohm across "
                        f"its {n_live} live cells. (rfx versions before "
                        f"the #318 fix counted all {n} cells and "
                        f"physically terminated at Z0*(n_live/n) = "
                        f"{z_eff:.1f} ohm — the issue-#313 finding.) "
                        f"Verify the extent was MEANT to end on/inside "
                        f"the conductor, and keep the midpoint V/I probe "
                        f"cell live; to silence, shorten the extent or "
                        f"move the port so none of its cells has its "
                        f"{pe.component} edge inside a conductor (per the "
                        f"realized edge set -- a volume shorts every edge "
                        f"between its two faces, a sheet shorts only the "
                        f"edges IN its plane and leaves the normal edge "
                        f"through it live).",
                        code="wire_port_dead_extent_cells",
                        source="_validate_cfg_port_inside_pec",
                    ),
                    stacklevel=3,
                )

        if realized is None:
            return
        _internal = getattr(self, "_internal_probe_indices", frozenset())
        _probe_entries = [
            pe for _pi, pe in enumerate(self._probes) if _pi not in _internal
        ]  # skip library-internal witness probes (issue #470; see
        #    _validate_cfg_absorber_placement for the rationale)
        for pe in list(self._ports) + _probe_entries:
            if getattr(pe, "extent", None):
                continue        # wire ports: the per-cell rule above
            pos = tuple(float(v) for v in pe.position)
            component = (getattr(pe, "component", "") or "").lower()
            if component not in ("ex", "ey", "ez", "hx", "hy", "hz"):
                continue
            try:
                if is_nonuniform:
                    from rfx.nonuniform import position_to_index as _nu_p2i
                    idx = tuple(int(v) for v in _nu_p2i(grid, pos))
                else:
                    idx = tuple(int(v) for v in grid.position_to_index(pos))
            except (ValueError, TypeError, IndexError, AttributeError):
                continue
            if not realized.component_is_dead(component, idx):
                continue
            names = _owners(component, idx)
            what = ("its own E edge is a realized PEC edge"
                    if component[0] == "e" else
                    "all four E edges of its curl loop are realized PEC "
                    "edges, so curl E = 0 and the component never moves")
            _w.warn(
                PreflightWarning(
                    f"Port/source/probe at {pos} ({component}) is frozen "
                    f"by realized PEC geometry {names}: {what} (lattice "
                    f"ownership contract, #931). Field will be zero. "
                    f"Move it off the conductor, or drive/read a "
                    f"component the conductor leaves live (the normal E "
                    f"through a sheet plane; tangential H beside a sheet).",
                    code="port_in_pec",
                    source="_validate_cfg_port_inside_pec",
                ),
                stacklevel=3,
            )

    def _wire_port_cell_centers(self, pe):
        """Per-cell physical sample centers of a wire port's rasterization.

        Uses the production endpoint snap and shared half-open edge span on
        either grid lane. Returns ``(centers, midpoint_index)`` with one
        physical E-edge center per driven cell and the midpoint V/I probe
        at ``cells[len(cells) // 2]``. Returns None when the declaration
        cannot be rasterized (diagnostics must not crash a run).
        """
        try:
            from rfx.sources.sources import (
                WirePort, _wire_port_cells, wire_port_edge_span,
            )
            from rfx.nonuniform import NonUniformGrid, position_to_index
            from rfx.geometry.rasterize_grid import (
                coords_from_nonuniform_grid, coords_from_uniform_grid,
            )
            axis = {"ex": 0, "ey": 1, "ez": 2}[pe.component]
            end = list(pe.position)
            end[axis] += pe.extent
            grid = self._build_realized_grid()
            if isinstance(grid, NonUniformGrid):
                start_idx = position_to_index(grid, pe.position)
                end_idx = position_to_index(grid, tuple(end))
                lo, hi = sorted((start_idx[axis], end_idx[axis]))
                first, last = wire_port_edge_span(
                    grid, axis, lo, hi, float(pe.position[axis]),
                    float(end[axis]))
                cells = []
                for k in range(first, last + 1):
                    cell = list(start_idx)
                    cell[axis] = k
                    cells.append(tuple(cell))
                coords = coords_from_nonuniform_grid(grid)
            else:
                wp = WirePort(start=tuple(pe.position), end=tuple(end),
                              component=pe.component, impedance=pe.impedance)
                cells = _wire_port_cells(grid, wp)
                coords = coords_from_uniform_grid(grid)
            if not cells:
                return None
            nodes = [np.asarray(line, dtype=float)
                     for line in (coords.x, coords.y, coords.z)]
            centers = []
            for cell in cells:
                # An E edge is at its node on the transverse axes and at
                # the midpoint of its bounding nodes on its own axis.
                pos = [float(nodes[ax][cell[ax]]) for ax in range(3)]
                pos[axis] = 0.5 * (pos[axis] + nodes[axis][cell[axis] + 1])
                centers.append(tuple(pos))
            return centers, len(cells) // 2
        except Exception:
            return None

    def _validate_cfg_floating_single_cell_port(self, _w) -> None:
        """P1.9: Single-cell port in dielectric with no adjacent PEC pin
        (issue #71). A single-cell LumpedPort placed mid-substrate with
        no conducting pin or microstrip does not couple to patch-antenna
        TM modes — the optimiser reads a nonsense loss from the
        floating Ez source. Recommend extent=<substrate_height> to
        promote to a WirePort spanning ground → patch.
        """
        _PORT_COMP_AXIS = {"ex": 0, "ey": 1, "ez": 2}
        for pe in self._ports:
            # Filter: only true ports (impedance > 0), single-cell
            # (extent is None), actively excited (excite is True).
            # add_source() creates _PortEntry with impedance=0.0 and is
            # intentionally a soft source — not a port footgun.
            if not pe.impedance or pe.impedance <= 0.0:
                continue
            if pe.extent is not None:
                continue
            if not pe.excite:
                continue
            pos = pe.position
            # Find the dielectric geometry enclosing the port cell.
            enclosing_eps_r = None
            enclosing_name = None
            for entry in self._geometry:
                if entry.material_name == "pec":
                    continue
                if not hasattr(entry.shape, "bounding_box"):
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError):
                    continue
                inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
                if not inside:
                    continue
                mspec = self._materials.get(entry.material_name)
                if mspec is not None and float(mspec.eps_r) > 1.0 + 1e-3:
                    enclosing_eps_r = float(mspec.eps_r)
                    enclosing_name = entry.material_name
                    break
            if enclosing_eps_r is None:
                continue
            # Check for a PEC geometry one cell away along the port's
            # component axis (coax-style pin or microstrip feed edge).
            # Without such a pin, the port cell cannot drive a vertical
            # current that couples to the patch TM mode.
            comp_axis = _PORT_COMP_AXIS.get(pe.component)
            if comp_axis is None:
                continue
            nudge = float(self._dx or 0.0) * 1.01
            adj_positions = (
                tuple(pos[i] + (nudge if i == comp_axis else 0.0) for i in range(3)),
                tuple(pos[i] - (nudge if i == comp_axis else 0.0) for i in range(3)),
            )
            # #931: a pin / ground may be a SHEET declaration
            # (add_thin_conductor); sheets are conductors in this census.
            from rfx.materials.thin_conductor import sheet_bounds as _sb
            adjacent_bounds = []
            for entry in self._geometry:
                if entry.material_name != "pec":
                    continue
                if not hasattr(entry.shape, "bounding_box"):
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError):
                    continue
                adjacent_bounds.append((c1, c2))
            for tc in getattr(self, "_thin_conductors", ()):
                if not getattr(tc, "is_pec", False):
                    continue
                try:
                    c1, c2 = _sb(tc.shape)
                except (NotImplementedError, TypeError, AttributeError):
                    continue
                if c1 is not None and c2 is not None:
                    adjacent_bounds.append((tuple(c1), tuple(c2)))
            has_adjacent_pec = False
            for apos in adj_positions:
                for c1, c2 in adjacent_bounds:
                    if all(c1[ax] <= apos[ax] <= c2[ax] for ax in range(3)):
                        has_adjacent_pec = True
                        break
                if has_adjacent_pec:
                    break
            if has_adjacent_pec:
                continue
            _w.warn(
                PreflightWarning(
                    f"Single-cell port at {pos} ({pe.component}) sits inside "
                    f"dielectric '{enclosing_name}' (eps_r={enclosing_eps_r:.2f}) "
                    f"with no adjacent PEC along the {pe.component[1]}-axis. A "
                    f"floating single-cell port inside substrate does not "
                    f"couple to patch-antenna TM modes. Pass "
                    f"extent=<substrate_height> to create a WirePort spanning "
                    f"ground → patch plane (issue #71).",
                    code="floating_port",
                    source="_validate_cfg_floating_single_cell_port",
                ),
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the P0.4 PEC-boundary-on-an-open-structure
    # advisory moved VERBATIM to ``rfx/preflight/absorber.py``, bound back
    # at its original position for the reason the blocks above give. The
    # split inventory filed this one as ambiguous; that module's docstring
    # records why reading the body puts it with the absorber family.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import _validate_cfg_pec_boundary_open_structure

    def _validate_cfg_no_sources(self, _w) -> None:
        """P0.5: No sources configured."""
        if (
            not self._ports
            and self._tfsf is None
            and not self._waveguide_ports
            and not self._floquet_ports
            and not self._msl_ports
        ):
            _w.warn(
                PreflightWarning(
                    "No sources, ports, TFSF, or waveguide/Floquet/MSL ports configured. "
                    "Simulation will produce zero fields.",
                    code="no_sources",
                    source="_validate_cfg_no_sources",
                ),
                stacklevel=3,
            )

    # Validated multi-band grading envelope (SPEC-01 WP6, #780; witness
    # battery validation/research/multiband_nu/, pre-declaration note
    # docs/design_notes/20260829_spec01_multiband_predeclaration.md,
    # support row docs/guides/support_matrix.md "Multi-band graded mesh").
    #
    # What the witnesses establish for fine bands ALONG Z (small-large-
    # small-large included), witnessed only up to 3 fine bands / 4
    # transitions — the widest profile in the battery — with EVERY
    # adjacent-cell ratio <= 1.4. Nothing below is an in-plane statement:
    # every witness holds the transverse mesh uniform, and no witness
    # grades dx_profile or dy_profile (adversarial review 2026-08-29,
    # BL1; note section WP6R.1/WP6R.7 item 1). More bands are expected to
    # behave the same way (each transition is local, and the measured
    # quantity is per-transition) but are NOT witnessed:
    #   F-S1  Remis-class dual-cell discrete energy is bounded at the
    #         float-accumulation class over 1e6 steps, 1D and 3D.
    #   F-S2  per-transition reflection at r <= 1.4 sits inside the window
    #         computed from the exact discrete scattering chain (-54 dB
    #         class at 30 cells/wavelength; -44 dB measured at r = 2.0).
    #   F-S3  symmetric round-trip amplitude asymmetry stays under the
    #         3e-4 differencing floor.
    #   F-S4  global 2nd-order supraconvergence is preserved on the
    #         multi-band grid (p_mb = 2.01, p_uc = 2.02 against the
    #         analytic TE_{1,0,4} of an empty 60 x 3 x 64 mm PEC cavity,
    #         the z-DOMINANT W4R3 fixture) — Monk & Suli 1994 / Li &
    #         Shields 2016 as realized by this solver. The earlier W4R2
    #         reading (p = 1.95) is superseded: that fixture carried ~1 %
    #         of its error on the graded axis (note WP6R.2).
    #   F-S5  the mesh-gradient AD path is unchanged by multi-band.
    #
    # 1.4 is a CAP, not a threshold fitted to data: it is the value the
    # spec claims and the witnesses were run at, and it matches the
    # commercial default (Tidy3D max_scale = 1.4). Beyond it the run is
    # still stable (Remis JCP 218:594, 2006 — min-cell CFL is sufficient
    # on ANY tensor grid), so both checks below are ADVISORY, never
    # blocking; what is lost beyond the cap is the validated accuracy
    # class, not stability.
    #
    # EXCLUSION the witnesses force: every witness that measures an
    # ACCURACY observable (F-S1 1D/3D, F-S2, F-S3, F-S4, both
    # revert-proofs) ran PEC-closed with cpml_layers = 0. F-S5 is the one
    # exception and is NOT PEC-closed — validation/research/multiband_nu/
    # w5_ad_consistency.py builds its grid with cpml_layers = 4 on all six
    # faces (and its runway is non-uniform, so that fixture draws the
    # second check below) — but F-S5 compares jax.grad to central FD, not
    # a field to a reference, so it is evidence that a graded mesh beside
    # an absorber constructs, runs and differentiates, and no evidence at
    # all about accuracy. So: NO witness measures an accuracy observable
    # with an absorber present, and this envelope says NOTHING about
    # grading that interacts with one. That exclusion is what the second
    # check reports (BL1 repair, review 2026-08-30; note WP6R.13).
    # z: the witnessed multi-band envelope cap. x/y: the pre-existing
    # in-plane threshold, deliberately NOT moved to 1.4 — every witness in
    # the multi-band battery grades z with a uniform transverse mesh, so
    # there is no in-plane provenance for moving an in-plane lock
    # (SPEC-00 0.2-4; adversarial review 2026-08-29, finding BL1).
    _MULTIBAND_RATIO_CAP = 1.4
    _INPLANE_RATIO_CAP = 1.3

    def _validate_cfg_multiband_grading(self, _w) -> None:
        """P2: multi-band graded-mesh envelope advisories (SPEC-01 WP6).

        Two advisory-tier checks on every explicit per-axis profile. A
        profile whose adjacent ratios are all <= 1.4 draws NEITHER — that
        is the "allow" half of WP6: a small-large-small-large profile is
        now a documented, witnessed configuration and must construct and
        preflight clean.

        1. ``nu_grading_ratio_beyond_validated_cap`` — some adjacent-cell
           ratio exceeds the 1.4 cap. Quotes the measured accuracy class
           on both sides of the cap so the reader can price the choice.
        2. ``nu_grading_reaches_absorber`` — an axis has an active
           absorber face and its adjacent interior runway is not uniform.

        Why check 2 exists, and where its DEPTH comes from. The absorber
        pad replicates the outermost interior cell
        (``rfx/nonuniform._pad_profile``), so the absorber itself is
        always uniformly meshed; what a boundary-adjacent transition does
        is change the discrete medium in the boundary-NORMAL direction
        right where the absorber begins. Normal-direction inhomogeneity
        is the documented breakdown class for PML generally (Meep's PML
        documentation: PML tolerates media varying only in the
        boundary-PARALLEL directions), and it leaves the transition's own
        reflection no interior runway in which to separate from the
        absorber's. The depth used is the FACE'S OWN allocated layer
        count — the only length the absorber itself defines, and the span
        over which its conductivity ramp acts. No measurement sets this
        depth and none could: no multi-band witness measures an accuracy
        observable with an absorber present (every accuracy witness is
        PEC-closed at cpml_layers = 0; the one absorber-bearing witness,
        F-S5, checks the AD path and scores no accuracy quantity), which
        is precisely why the combination is flagged rather than scored. Tolerance 1e-6 on the adjacent ratio, i.e. a runway is
        "uniform" up to 1 ppm of cell size — far below any grading a user
        can mean, and above float round-tripping in a computed profile.

        Inherited caveat: ``_preflight_face_layers`` over-reports the
        absorber on the non-port axes of a waveguide-port simulation (the
        grid drops those axes from ``cpml_axes``, the helper does not).
        This check inherits that, as every other consumer of the helper
        does; the effect is a possible advisory on an axis whose absorber
        the grid will not actually allocate. Advisory tier, so it costs a
        line of noise, never a run.
        """
        face_layers = None
        for ax_name, profile in (("x", self._dx_profile),
                                 ("y", self._dy_profile),
                                 ("z", self._dz_profile)):
            if profile is None or is_tracer(profile) or len(profile) < 2:
                continue
            p = np.asarray(profile, dtype=float)
            ratios = p[1:] / p[:-1]
            max_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
            cap = (self._MULTIBAND_RATIO_CAP if ax_name == "z"
                   else self._INPLANE_RATIO_CAP)
            scope = ("validated multi-band grading cap"
                     if ax_name == "z" else
                     "in-plane grading threshold (the validated 1.4 "
                     "multi-band cap is a z-axis envelope: no witness in "
                     "it grades an in-plane axis)")
            if max_ratio > cap + 1e-6:
                _w.warn(
                    PreflightWarning(
                        f"d{ax_name}_profile max adjacent cell ratio "
                        f"{max_ratio:.3f} exceeds the {scope} "
                        f"{cap:g} "
                        f"(docs/guides/support_matrix.md, 'Multi-band graded "
                        f"mesh'). WHY: stability is unaffected (dt is the "
                        f"global min-cell CFL, sufficient on any tensor "
                        f"grid), but per-transition accuracy is witnessed "
                        f"only along z and only up to ratio 1.4 — the -54 dB "
                        f"reflection class at 30 cells/wavelength (it scales "
                        f"as (dz/lambda)^2), against -44 dB measured at "
                        f"ratio 2.0. COST: an unquantified reflection and "
                        f"grading-dispersion error at that transition. "
                        f"REMEDY: split it into ratio<={cap:g} steps (e.g. "
                        f"rfx.smooth_grading) to stay inside the "
                        f"{'validated envelope' if ax_name == 'z' else 'pre-existing in-plane threshold (no in-plane envelope is validated)'}, "
                        f"or accept it and say so in the report. STALE-IF: "
                        f"the support-matrix row raises the cap.",
                        code="nu_grading_ratio_beyond_validated_cap",
                        source="_validate_cfg_multiband_grading",
                    ),
                    stacklevel=3,
                )
            if face_layers is None:
                face_layers = self._preflight_face_layers()
            for side in ("lo", "hi"):
                layers = face_layers.get(f"{ax_name}_{side}", 0)
                if layers < 2 or len(p) < layers:
                    # layers < 2: "one uniform cell" is vacuous (no adjacent
                    # ratio exists inside a single-cell runway).
                    continue
                # The REMEDY asks for `layers` uniform interior cells
                # against the face, i.e. `layers - 1` adjacent ratios among
                # p[0..layers-1]; taking layers+1 cells here demanded one
                # uniform cell MORE than the remedy and fired on an exactly
                # compliant profile (review 2026-08-29).
                runway = p[:layers] if side == "lo" else p[-layers:]
                r_run = runway[1:] / runway[:-1]
                dev = float(np.max(np.abs(r_run - 1.0)))
                if dev > 1e-6:
                    _w.warn(
                        PreflightWarning(
                            f"d{ax_name}_profile is not uniform within the "
                            f"{layers} interior cells adjacent to the "
                            f"{ax_name}_{side} absorber face (max adjacent "
                            f"cell-ratio deviation {dev:.3g}). WHY: the "
                            f"absorber pad replicates the outermost interior "
                            f"cell, so a transition in that runway changes "
                            f"the discrete medium in the boundary-NORMAL "
                            f"direction exactly where the absorber starts — "
                            f"the documented PML breakdown class — and gives "
                            f"the transition's own reflection no runway to "
                            f"separate from the absorber's. COST: the "
                            f"validated multi-band envelope does not cover "
                            f"it at all; no witness in it MEASURES an "
                            f"accuracy observable with an absorber present "
                            f"(every accuracy witness is PEC-closed at "
                            f"cpml_layers = 0; the one absorber-bearing "
                            f"witness checks the AD path only) "
                            f"(docs/guides/support_matrix.md, 'Multi-band "
                            f"graded mesh'). REMEDY: keep at least "
                            f"{layers} uniform interior cells against that "
                            f"face, or close it with PEC/PMC. STALE-IF: a "
                            f"witness that SCORES an accuracy observable "
                            f"with an absorber present is added to that "
                            f"row.",
                            code="nu_grading_reaches_absorber",
                            source="_validate_cfg_multiband_grading",
                        ),
                        stacklevel=3,
                    )

    def _validate_cfg_thin_conductor_graded_node(self, _w) -> None:
        """Advisory: a LOSSY thin conductor landing on a grading transition.

        A lossy sheet folds into ``materials.sigma`` at ONE E node along its
        normal, and the length that node's sigma acts over is the DUAL spacing
        ``(d[k-1]+d[k])/2`` — not either adjacent cell. Where the two adjacent
        cells are equal the distinction is invisible, which is precisely how
        the pre-#669-review fold (which divided by the primal cell ``d[k]``)
        stayed silent: every uniform mesh and every NU node away from a step
        agrees. This check surfaces the case that does not agree.

        Advisory tier, concrete profiles only, LOSSY sheets only — a PEC thin
        sheet is a :class:`~rfx.boundaries.pec.SheetSpec` (#931 §1.3) and
        folds no sigma, so it has no sheet resistance to normalize.
        Threshold: adjacent cells differing by more than 10%.

        The node is located through the lossy fold's OWN production path
        (node positions + the shape's node mask): the sigma-fill sheet is
        fenced out of the #931 contract (design note §1.8 — a lossy volume
        model, unchanged), so mirroring its mask here IS reading what the
        run realizes, not a re-derivation.
        """
        from rfx.nonuniform import node_positions_from_profile
        from rfx.materials.thin_conductor import sheet_bounds

        tcs = [tc for tc in getattr(self, "_thin_conductors", ())
               if not getattr(tc, "is_pec", False)]
        if not tcs:
            return
        profiles = (self._dx_profile, self._dy_profile, self._dz_profile)
        if all(p is None or is_tracer(p) for p in profiles):
            return

        for i, tc in enumerate(tcs):
            # Bounds source (issue #674): a surface-impedance sheet may be any
            # ``mask_on_coords`` shape, so read its bounding box; the legacy DC
            # fold is still Box-only on this lane (it warn-and-skips a non-Box
            # sheet), and advising about a sheet that is not folded would be
            # worse than silence.
            if getattr(tc, "surface_impedance_f0", None) is not None:
                lo, hi = sheet_bounds(tc.shape)
            else:
                lo = getattr(tc.shape, "corner_lo", None)
                hi = getattr(tc.shape, "corner_hi", None)
            if lo is None or hi is None:
                continue
            extents = [float(hi[a]) - float(lo[a]) for a in range(3)]
            n_axis = min(range(3), key=lambda a: extents[a])
            prof = profiles[n_axis]
            if prof is None or is_tracer(prof):
                continue
            d = np.asarray(prof, dtype=np.float64)
            if d.size < 2:
                continue

            # Locate the node the RUN will realize the sheet on, through the
            # production path (node positions + the shape's own mask), not a
            # hand-rolled nearest-node rule (#562 review F2, #568).
            nodes = np.asarray(node_positions_from_profile(d),
                               dtype=np.float64)
            _other = tuple(a for a in range(3) if a != n_axis)

            def _occupied_layers(fracs, _tc=tc, _lo=lo, _hi=hi,
                                 _n_axis=n_axis, _nodes=nodes,
                                 _other=_other):
                args = []
                for a in range(3):
                    if a == _n_axis:
                        args.append(_nodes)
                        continue
                    a_lo, a_hi = float(_lo[a]), float(_hi[a])
                    args.append(np.array(
                        [a_lo + f * (a_hi - a_lo) for f in fracs],
                        dtype=np.float64))
                m = np.asarray(_tc.shape.mask_on_coords(*args))
                return np.flatnonzero(m.any(axis=_other))

            # Bounding-box centre first — for a Box that is the whole story,
            # and it is bit-identically the probe this check has always run.
            hit = _occupied_layers((0.5,))
            if hit.size == 0:
                # #674: a PATTERNED sheet can have its bbox centre inside a
                # clearance hole, which would read as "no sheet here" and
                # silently drop the advisory. Fan out before giving up.
                hit = _occupied_layers((0.1, 0.3, 0.5, 0.7, 0.9))
            if hit.size == 0:
                continue
            k = int(hit[0])

            # cells adjacent to node k: d[k-1] below, d[k] above. The lo face
            # (k == 0) and the last node (k == d.size, backed by the bounding
            # -node duplicate d[k] == d[k-1]) are matched by construction.
            if k == 0 or k >= d.size:
                continue
            d_below, d_above = float(d[k - 1]), float(d[k])
            small, large = sorted((d_below, d_above))
            if small <= 0.0 or (large / small) - 1.0 <= 0.10:
                continue
            axis_name = "xyz"[n_axis]
            dual = 0.5 * (d_below + d_above)
            _w.warn(
                PreflightWarning(
                    f"lossy thin conductor #{i} sits at "
                    f"{axis_name} = {_fmt_len(float(nodes[k]))}, an E node "
                    f"whose adjacent cells differ by "
                    f"{(large / small - 1.0):.0%} ({_fmt_len(d_below)} below, "
                    f"{_fmt_len(d_above)} above). Its sheet fold is "
                    f"normalized by the E-node DUAL spacing "
                    f"{_fmt_len(dual)} — the length that node's sigma acts "
                    f"over — which is neither adjacent cell. That IS the "
                    f"correct normalization, but a sheet on a grading step is "
                    f"where the realized sheet resistance is most "
                    f"mesh-sensitive: move the sheet onto a locally uniform "
                    f"node (or flatten the grading there) if its loss is "
                    f"claims-bearing.",
                    code="thin_conductor_graded_node",
                    source="_validate_cfg_thin_conductor_graded_node",
                ),
                stacklevel=3,
            )

    # ---- issue #672: primal-vs-dual metrics at a source / wire-port node --

    _AXIS_OF_COMPONENT = {"ex": 0, "ey": 1, "ez": 2}

    def _graded_node_report(self, axis: int, coord: float):
        """``(node_pos, d_below, d_above, dual, ratio)`` when the E node
        nearest ``coord`` on ``axis`` sits on a >10% grading step, else None.

        Concrete profiles only (a tracer profile is skipped, matching the
        #671 check). The node is located with ``node_positions_from_profile``
        — the production node convention — not a hand-rolled rule (#562
        review F2, #568). ``k == 0`` and ``k >= d.size`` are matched by
        construction and never report.
        """
        from rfx.nonuniform import node_positions_from_profile

        prof = (self._dx_profile, self._dy_profile, self._dz_profile)[axis]
        if prof is None or is_tracer(prof):
            return None
        d = np.asarray(prof, dtype=np.float64)
        if d.size < 2:
            return None
        nodes = np.asarray(node_positions_from_profile(d), dtype=np.float64)
        k = int(np.argmin(np.abs(nodes - float(coord))))
        if k == 0 or k >= d.size:
            return None
        d_below, d_above = float(d[k - 1]), float(d[k])
        small, large = sorted((d_below, d_above))
        if small <= 0.0 or (large / small) - 1.0 <= 0.10:
            return None
        return (float(nodes[k]), d_below, d_above,
                0.5 * (d_below + d_above), large / small)

    def _validate_cfg_source_on_graded_node(self, _w) -> None:
        """Advisory: a current source sitting on a grading transition (#672).

        ``make_current_source`` divides the waveform by the E node's control
        volume, which is the PRIMAL per-cell width on the component's own
        axis and the DUAL spacing on the two TRANSVERSE axes. The parallel
        axis is exact by construction, so only the transverse axes are
        checked. Advisory tier: the normalization IS correct now — this
        flags where the realized current moment is most mesh-sensitive.

        The counterfactual ratio in the message is ``d[k] / dual``, where
        ``d[k]`` is the cell ABOVE the node (``d_above``) — that is the width
        the pre-#672 code actually used on every axis, so it is the one that
        makes the number and the sentence describe the same thing. Using
        ``max(d_below, d_above)`` instead, as this message did until the #673
        split review, silently reported the wrong cell on every DOWN-step
        node (where ``d_above < d_below``).
        """
        entries = [pe for pe in getattr(self, "_ports", ())
                   if float(getattr(pe, "impedance", 0.0)) == 0.0]
        if not entries:
            return
        for pe in entries:
            axis = self._AXIS_OF_COMPONENT.get(pe.component)
            if axis is None:
                continue
            for a in (ax for ax in range(3) if ax != axis):
                rep = self._graded_node_report(a, pe.position[a])
                if rep is None:
                    continue
                node_pos, d_below, d_above, dual, ratio = rep
                _w.warn(
                    PreflightWarning(
                        f"current source at {pe.position} "
                        f"(component {pe.component}) sits at "
                        f"{'xyz'[a]} = {_fmt_len(node_pos)}, an E node whose "
                        f"adjacent cells differ by {(ratio - 1.0):.0%} "
                        f"({_fmt_len(d_below)} below, {_fmt_len(d_above)} "
                        f"above). {'xyz'[a]} is one of this component's two "
                        f"TRANSVERSE axes, so its control volume takes the "
                        f"DUAL spacing {_fmt_len(dual)} there — not the "
                        f"primal cell {_fmt_len(d_above)} — and the realized "
                        f"current moment would be off by "
                        f"{d_above / dual:.3f}x "
                        f"on this axis alone if the primal cell were used "
                        f"(issue #672). That IS handled, but a source on a "
                        f"grading step is where the injected amplitude is "
                        f"most mesh-sensitive: move it onto a locally "
                        f"uniform node if its amplitude is claims-bearing.",
                        code="source_on_graded_node",
                        source="_validate_cfg_source_on_graded_node",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_wire_port_on_graded_node(self, _w) -> None:
        """Advisory: an impedance port whose control volume straddles a step.

        The port current is the discrete Ampere loop on the DUAL face
        pierced by the port's E edge, each leg weighted by the dual spacing
        along that H component's own axis — the two axes TRANSVERSE to the
        port component. ``V`` uses the primal edge length and is exact
        either way, so only the transverse axes are checked. An EXCITED port
        also drives through ``make_current_source``, so the source-side
        normalization above applies to it as well.

        Since #688 the same two dual spacings also size the TERMINATION
        conductance (``sigma = n_live * d_par / (Z0 * dual_b * dual_c)``),
        which is why this now covers single-cell lumped ports too: they
        carry no ``extent``, so the old filter skipped them entirely while
        they sat on the identical metric (measured, a lumped ez port on a
        2:1 step on both transverse axes printed ``[PREFLIGHT] All checks
        passed.`` while carrying the same 1.7778x conductance error).
        """
        entries = [pe for pe in getattr(self, "_ports", ())
                   if getattr(pe, "extent", None) is not None
                   or float(getattr(pe, "impedance", 0.0)) > 0.0]
        if not entries:
            return
        for pe in entries:
            axis = self._AXIS_OF_COMPONENT.get(pe.component)
            if axis is None:
                continue
            for a in (ax for ax in range(3) if ax != axis):
                rep = self._graded_node_report(a, pe.position[a])
                if rep is None:
                    continue
                node_pos, d_below, d_above, dual, ratio = rep
                kind = ("wire port" if getattr(pe, "extent", None) is not None
                        else "lumped port")
                _w.warn(
                    PreflightWarning(
                        f"{kind} at {pe.position} "
                        f"(component {pe.component}) sits at "
                        f"{'xyz'[a]} = {_fmt_len(node_pos)}, an E node whose "
                        f"adjacent cells differ by {(ratio - 1.0):.0%} "
                        f"({_fmt_len(d_below)} below, {_fmt_len(d_above)} "
                        f"above). {'xyz'[a]} is one of the port's two "
                        f"Ampere-loop axes, so the DUAL spacing "
                        f"{_fmt_len(dual)} weights BOTH that leg of the loop "
                        f"that measures I (issue #672) and the termination "
                        f"conductance that realizes Z0 (issue #688). The "
                        f"extracted Z_in = -V/I, and every S-parameter built "
                        f"on it, is most mesh-sensitive here: move the port "
                        f"onto a locally uniform node (or flatten the "
                        f"grading there) if its S-parameters are "
                        f"claims-bearing.",
                        code="wire_port_on_graded_node",
                        source="_validate_cfg_wire_port_on_graded_node",
                    ),
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 2: the #669 Leontovich surface-impedance advisories
    # moved VERBATIM to ``rfx/preflight/pec_geometry.py``, bound back at
    # their original position for the reason the block above gives.
    # ------------------------------------------------------------------
    from rfx.preflight.pec_geometry import (
        _validate_cfg_thin_conductor_surface_impedance,
    )


    def _validate_cfg_unresolved_pulse(self, _w, dx: float) -> None:
        """Warn when a pulse waveform is unresolved by the time step (#386).

        ``tau < 3*dt`` means the sampled excitation is a sub-dt spike: the
        pulse's spectrum extends far past the grid Nyquist limit and the
        discrete time integral no longer cancels, so a soft source leaves a
        static charge field that CPML cannot absorb. The canonical way to
        get here is passing an absolute-Hz number as ``bandwidth`` where a
        FRACTIONAL one is expected (``tau = 1/(f0*bandwidth*pi)`` then
        misses by ~9 orders of magnitude), so this fires regardless of
        ``until_decay`` — a sub-dt spike is always broken.

        ``dt`` is estimated from the preflight ``dx`` via the uniform-lane
        3D Courant formula (``Grid.courant_dt``). A refining ``dz_profile``
        makes the actual dt smaller, so the estimate errs toward firing; a
        strictly coarsening profile can raise the NU dt above this estimate
        by at most sqrt(3/2) ~ 1.22x (the NU dt combines per-axis minimum
        cell sizes, ``rfx/nonuniform.py``), so the check can under-fire by
        <= 22% — harmless against a mistake that misses the threshold by
        ~9 orders of magnitude, not by percent.
        """
        dt = dx / (C0 * math.sqrt(3.0)) * 0.99  # Grid.courant_dt(dx, ndim=3)
        entries = list(self._ports) + list(self._msl_ports)
        if self._tfsf is not None:
            entries.append(self._tfsf)
        for entry in entries:
            wf = getattr(entry, "waveform", None)
            tau = None
            if wf is not None and not isinstance(wf, str):
                try:
                    tau = float(wf.tau)
                except (AttributeError, TypeError, ValueError,
                        ZeroDivisionError):
                    tau = None
            else:
                # String-named waveforms (the TFSF entry's
                # "differentiated_gaussian" / "modulated_gaussian"): both
                # pulse families share tau = 1/(pi*f0*bandwidth), so the
                # absolute-Hz-bandwidth footgun on
                # add_tfsf_source(bandwidth=...) is computable from the
                # entry's own f0/bandwidth attributes when both are set.
                f0 = getattr(entry, "f0", None)
                bw = getattr(entry, "bandwidth", None)
                if f0 and bw:
                    try:
                        tau = 1.0 / (math.pi * float(f0) * float(bw))
                    except (TypeError, ValueError, ZeroDivisionError):
                        tau = None
            if tau is None or not math.isfinite(tau) or tau <= 0.0:
                continue
            if tau < 3.0 * dt:
                _wf_name = wf if isinstance(wf, str) else type(wf).__name__
                _w.warn(
                    PreflightWarning(
                        f"waveform tau={tau:.3g}s is below 3*dt "
                        f"(dt~{dt:.3g}s, tau/dt={tau/dt:.3g}): pulse "
                        "unresolved by the time step — an absolute-Hz "
                        "bandwidth was likely passed where a FRACTIONAL "
                        "one is expected; the discrete DC residue leaves "
                        "a static charge field CPML cannot absorb "
                        "(issue #386)",
                        code="unresolved_pulse",
                        loc=f"waveform {_wf_name} at "
                            f"{getattr(entry, 'position', None)}",
                        source="_validate_cfg_unresolved_pulse",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_nonuniform_limitations(
        self, _w, cpml_thickness: float
    ) -> None:
        """P2: Non-uniform mesh shadow-lane limitations."""
        if self._dz_profile is not None:
            # P2.3: TFSF on nonuniform mesh — narrowed scope.
            # Axis-aligned ±x incidence with angle_deg=0 runs the 1D
            # auxiliary along the uniform x axis and is supported. The
            # z-directed and oblique cases would need a z-nonuniform 1D
            # aux (resp. nonuniform 2D aux) and are deferred.
            if self._tfsf is not None:
                if self._tfsf.direction in ("+z", "-z"):
                    raise PreflightConfigError(
                        "TFSF z-directed incidence is not yet supported on "
                        "nonuniform z mesh. Axis-aligned incidence along x "
                        "(direction='+x' or '-x') is supported.",
                        code="nonuniform_tfsf",
                        source="_validate_cfg_nonuniform_limitations",
                    )
                if abs(self._tfsf.angle_deg) > 0.01:
                    raise PreflightConfigError(
                        "TFSF oblique incidence is not yet supported on "
                        "nonuniform z mesh. Use angle_deg=0.",
                        code="nonuniform_tfsf",
                        source="_validate_cfg_nonuniform_limitations",
                    )

            # P2.6: CPML z-thickness on non-uniform mesh.
            # Skip on tracer profiles — advisory warning only.
            # Issue #647: the cell count is the z faces' OWN allocation, not
            # the global budget. Keyed off `_boundary_spec` via
            # `_preflight_face_layers`, so a per-face spec whose z faces are
            # PEC/PMC (allocation 0) no longer reports a thin absorber that
            # does not exist, and a per-face `hi_thickness` is measured at the
            # thickness it actually allocates.
            _z_layers = max(self._preflight_face_layers()["z_lo"],
                            self._preflight_face_layers()["z_hi"])
            if (self._boundary == "cpml"
                    and _z_layers > 0
                    and not is_tracer(self._dz_profile)):
                cpml_z_thick = sum(float(d) for d in self._dz_profile[:_z_layers])
                if cpml_z_thick < cpml_thickness * 0.3:
                    _w.warn(
                        PreflightWarning(
                            f"CPML z-thickness is {cpml_z_thick*1e3:.1f}mm "
                            f"({_z_layers} cells), much thinner than "
                            f"xy-thickness {cpml_thickness*1e3:.1f}mm. "
                            f"Absorbing performance may be asymmetric. "
                            f"Consider more z cells or fewer CPML layers.",
                            code="nonuniform_cpml_thin",
                            source="_validate_cfg_nonuniform_limitations",
                        ),
                        stacklevel=3,
                    )

    def _validate_cfg_subgrid_limitations(self, _w) -> None:
        """P4: Subgridded path limitations.

        P3 (Distributed path): distributed warnings are emitted at
        run() dispatch time in distributed_v2.py — no preflight check
        here.
        """
        if self._refinement is not None:
            if self._dft_planes:
                _w.warn(
                    PreflightWarning(
                        "DFT plane probes are not supported with SBP-SAT "
                        "subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._waveguide_ports:
                _w.warn(
                    PreflightWarning(
                        "Waveguide ports are not supported with SBP-SAT "
                        "subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._floquet_ports:
                _w.warn(
                    PreflightWarning(
                        "Floquet ports are not supported with SBP-SAT subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._tfsf is not None:
                _w.warn(
                    PreflightWarning(
                        "TFSF source is not supported with SBP-SAT subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )
            if self._lumped_rlc:
                _w.warn(
                    PreflightWarning(
                        "Lumped RLC elements are not supported with SBP-SAT "
                        "subgridding.",
                        code="subgrid_unsupported_feature",
                        source="_validate_cfg_subgrid_limitations",
                    ),
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the P2.8 reference-plane check moved VERBATIM to
    # ``rfx/preflight/waveguide.py``, bound back at its original position
    # for the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import _validate_cfg_waveguide_reference_plane

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the three post-v1.8 waveguide S-parameter setup
    # audits, their shared plane/geometry builders and the single hook
    # both waveguide entry points call moved VERBATIM to
    # ``rfx/preflight/waveguide.py`` -- one contiguous class-body block,
    # its own section banner included -- and are bound back here at the
    # position they held, for the reason the blocks above give.
    #
    # ``_preflight_waveguide_setup`` is the hook, and it is reached from
    # TWO places by ``self.``: ``preflight_sparameters`` below, which
    # never leaves this facade, and ``rfx/sparams/waveguide.py``'s
    # ``compute_waveguide_s_matrix`` path. Both resolve through the
    # composed ``Simulation`` MRO and neither needed touching.
    #
    # ``_waveguide_setup_planes`` keeps three ``self.`` calls to builders
    # that live on ``Simulation`` itself -- ``_build_grid``,
    # ``_build_nonuniform_grid``, ``_build_waveguide_port_config`` -- which
    # is the point of those audits: they read the RUNNER's own plane
    # indices rather than recomputing them.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import (
        _waveguide_setup_planes,
        _waveguide_far_geometry,
        _validate_cfg_layout_from_band_low_edge,
        _validate_cfg_record_vs_far_boundary,
        _validate_cfg_port_index_mirror_covariance,
        _preflight_waveguide_setup,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 1: the five MSL-family bodies moved VERBATIM to
    # ``rfx/preflight/msl.py`` and are bound back here, AT THE POSITION
    # they held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls ``_check_msl_port_geometry`` in
    # a fixed sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_msl`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._check_msl_port_geometry(...)`` a bound method with its name,
    # signature and ``__doc__`` intact, and ``self._assemble_realized``
    # inside ``_msl_assemble_once`` keeps resolving through the composed
    # ``Simulation`` MRO exactly as it did before the move.
    # ------------------------------------------------------------------
    from rfx.preflight.msl import (
        _msl_assemble_once,
        _msl_declared_face_geometry,
        _msl_realized_substrate,
        _msl_conductor_gap,
        _check_msl_port_geometry,
    )

    # ------------------------------------------------------------------
    # Issue #703: campaign statics checks. Four failure classes a month
    # of external cross-validation hit, all detectable before the first
    # time step. Shared state (grid, coords, per-entry masks, assembly)
    # is built once in _CampaignStaticsContext; each check emits at most
    # ONE aggregated advisory (the #697 duplication lesson).
    # ------------------------------------------------------------------

    def _campaign_ctx(self):
        """The shared :class:`_CampaignStaticsContext` for THIS preflight.

        Built once per configuration and reused by every check that reads
        conductor geometry (thin metal on NU, port/probe liveness, NTFF
        walls, the #703 campaign checks, the #931 realization findings):
        the context runs the production assembly, which on a large model
        is the most expensive thing preflight does, and five checks each
        building their own would run it five times. The cache key is the
        identity of every geometry / thin-conductor entry plus the mesh
        parameters, so an ``add()`` after a preflight (a new entry object)
        or a mesh change misses the cache and rebuilds — the staleness a
        plain instance cache would have had.
        """
        sim = self
        key = (
            tuple(id(e) for e in sim._geometry),
            tuple(id(tc) for tc in getattr(sim, "_thin_conductors", ())),
            id(sim._dx), id(sim._dx_profile), id(sim._dy_profile),
            id(sim._dz_profile), tuple(sim._domain),
            getattr(sim, "_periodic_axes", None), sim._cpml_layers,
            id(getattr(sim, "_refinement", None)),
            tuple(sorted(sim._materials)),
        )
        cached = getattr(self, "_pf_campaign_ctx", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        ctx = _CampaignStaticsContext(self)
        self._pf_campaign_ctx = (key, ctx)
        return ctx

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 2: the #703 campaign-statics umbrella, its four
    # checks and the #931 per-declaration realization findings moved
    # VERBATIM to ``rfx/preflight/pec_geometry.py`` -- one contiguous
    # class-body block -- and are bound back here at the position they
    # held, for the reason the conformal block above gives.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_pec`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._validate_cfg_pec_realization(...)`` a bound method with its
    # name, signature and ``__doc__`` intact, and the two cross-family
    # ``self.`` calls -- ``_validate_cfg_campaign_statics`` ->
    # ``self._campaign_ctx()`` and
    # ``_validate_cfg_pec_face_short_of_domain_wall`` ->
    # ``self._preflight_face_layers()`` -- keep resolving through the
    # composed ``Simulation`` MRO exactly as they did before the move.
    #
    # ``_congruence_origin_shift`` was a ``@staticmethod`` and has to be
    # re-wrapped as one. Its single caller,
    # ``_validate_cfg_congruent_rasterization_parity``, invokes it as
    # ``self._congruence_origin_shift(ctx, members, counts)``; left as the
    # plain function the import binds, that call would pass ``self`` as
    # ``ctx`` and every argument would shift by one. The decorator could
    # not travel with the body -- at module level ``@staticmethod`` makes
    # a staticmethod OBJECT, not a callable -- so it is re-applied here,
    # which also reproduces the pre-move ``__qualname__``: the rewrite
    # loop in ``rfx/api/__init__.py`` tests ``inspect.isfunction`` and has
    # always skipped this name for exactly this reason.
    # ------------------------------------------------------------------
    from rfx.preflight.pec_geometry import (
        _validate_cfg_campaign_statics,
        _validate_cfg_pec_realization,
        _validate_cfg_sheet_slot_vacuum,
        _validate_cfg_pec_face_short_of_domain_wall,
        _congruence_origin_shift,
        _validate_cfg_congruent_rasterization_parity,
        _validate_cfg_sheet_cavity_thickness,
        _validate_cfg_off_lattice_design_edges,
    )

    _congruence_origin_shift = staticmethod(_congruence_origin_shift)


    def _validate_adi_configuration(self, materials: MaterialArrays, debye_spec, lorentz_spec) -> None:
        """Validate that the current simulation is compatible with the ADI path."""
        if self._mode not in ("2d_tmz", "3d"):
            raise ValueError("solver='adi' supports mode='3d' or mode='2d_tmz'")
        if self._boundary == "upml":
            raise ValueError("solver='adi' does not support boundary='upml'")
        if self._boundary not in ("pec", "cpml"):
            raise ValueError("solver='adi' supports boundary='pec' or 'cpml'")
        if self._refinement is not None:
            raise ValueError("solver='adi' does not support subgridding yet")
        if self._tfsf is not None:
            raise ValueError("solver='adi' does not support TFSF sources yet")
        if self._waveguide_ports or self._floquet_ports:
            raise ValueError("solver='adi' does not support waveguide or Floquet ports yet")
        if self._periodic_axes:
            raise ValueError("solver='adi' does not support manual periodic axes yet")
        if self._dft_planes:
            raise ValueError("solver='adi' does not support DFT plane probes yet")
        if self._ntff is not None:
            raise ValueError("solver='adi' does not support NTFF accumulation yet")
        if self._coaxial_ports:
            raise ValueError("solver='adi' does not support coaxial ports yet")
        if self._lumped_rlc:
            raise ValueError("solver='adi' does not support lumped RLC elements yet")
        if self._thin_conductors:
            raise ValueError("solver='adi' does not support thin-conductor corrections yet")
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError("solver='adi' does not support dispersive materials yet")
        # Conductivity is now supported: implicit sigma in ADI tridiagonal.
        # Internal absorbing layers also use sigma, so no restriction needed.
        for pe in self._ports:
            if pe.impedance != 0.0 or pe.extent is not None:
                raise ValueError("solver='adi' currently supports only add_source()-style soft sources")
            if self._mode == "2d_tmz" and pe.component != "ez":
                raise ValueError("solver='adi' in 2D TMz mode supports only Ez soft sources")
        _valid_adi_probes = {"ez", "hx", "hy"} if self._mode == "2d_tmz" else {"ex", "ey", "ez", "hx", "hy", "hz"}
        for probe in self._probes:
            if probe.component not in _valid_adi_probes:
                raise ValueError(f"solver='adi' supports probes on {_valid_adi_probes} only")
