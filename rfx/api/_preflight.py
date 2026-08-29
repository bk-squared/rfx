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

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box
# Bound at import time ON PURPOSE (issue #703 check 2): the live-edge check
# must keep its own reference to the real resample even when a test (or a
# regression) replaces the module attribute the ASSEMBLY resolves — the
# check exists to notice exactly that divergence.
from rfx.geometry.rasterize_grid import (
    resample_sheet_node_materials as _resample_sheet_node_materials,
)


def _fmt_len(meters: float) -> str:
    """Unit-adaptive length for warning text (issue #166).

    Fixed-mm formatting rendered µm-scale setups as ``0.0mm`` / ``0.000mm``
    and misled optical-scale users into reading a routine value as a bug;
    pick the unit that keeps the digits visible.
    """
    m = abs(meters)
    if m == 0.0:
        return "0mm"
    if m >= 1.0:
        return f"{meters:.4g}m"
    if m >= 1e-3:
        return f"{meters*1e3:.4g}mm"
    if m >= 1e-6:
        return f"{meters*1e6:.4g}µm"
    return f"{meters*1e9:.4g}nm"


def _fmt_freq(hz: float) -> str:
    """Unit-adaptive frequency for warning text (issue #166).

    Fixed-GHz formatting printed optical-scale ``freq_max`` as
    ``74950.00GHz`` instead of ``74.95THz``.
    """
    h = abs(hz)
    if h == 0.0:
        return "0Hz"
    if h >= 1e12:
        return f"{hz/1e12:.4g}THz"
    if h >= 1e9:
        return f"{hz/1e9:.4g}GHz"
    if h >= 1e6:
        return f"{hz/1e6:.4g}MHz"
    return f"{hz:.4g}Hz"


def _absorber_boundary_for_axis(
    domain_extent: float, ct_lo: float, ct_hi: float,
) -> tuple[float | None, float | None]:
    """The single canonical frame for "is/how-far a coordinate from the
    CPML/UPML absorber" on one axis (issue #500).

    Ground truth (proved in ``tests/test_preflight_absorber_frame.py`` for
    both the uniform (``rfx/grid.py`` ``Grid.__init__``: ``nx =
    ceil(domain/dx) + 1 + pad_lo + pad_hi``) and non-uniform
    (``rfx/nonuniform.py`` ``make_nonuniform_grid``: ``dz_profile`` covers
    only the physical domain and CPML cells are appended EXTERIOR to it via
    ``_pad_profile``) grid builders: the absorber is padded OUTSIDE the
    requested domain, never inside it. ``Grid.position_to_index`` maps node
    ``pad_{axis}_lo`` to user coordinate 0, so the absorber occupies user
    coordinates ``< 0`` (lo side) and ``> domain_extent`` (hi side); the
    requested ``[0, domain_extent]`` is absorber-FREE by construction.

    Every ``_validate_cfg_*`` / ``_check_msl_port_geometry`` consumer of
    ``cpml_thick_lo`` / ``cpml_thick_hi`` used to compare user coordinates
    against an INTERIOR reading instead (``[0, ct_lo]`` /
    ``[domain_extent - ct_hi, domain_extent]``) — verified false positives
    on geometry nowhere near the absorber (waveguide ports comfortably
    inside the domain, an NTFF box, a probe at the domain centre). This
    helper is the one place that encodes the correct (exterior) frame;
    every membership-style consumer must convert through it rather than
    re-deriving the comparison.

    Returns ``(lo_boundary, hi_boundary)``: the user-coordinate position at
    which the lo-side / hi-side absorber begins, or ``None`` on a side with
    no active absorber there (``ct <= 0`` — PEC/PMC/periodic face, 2D-z, or
    a non-absorbing global boundary). When active, the position is
    conservative by up to one cell (lo: ``0.0``; hi: ``domain_extent``) —
    NOT exactly the true interior/absorber interface. ``Grid.nx`` is sized
    from ``ceil(domain_extent/dx)``, so on the hi side the true last
    interior node can sit up to one ``dx`` past ``domain_extent`` (e.g.
    domain_extent=0.0101, dx=1e-3 -> ceil(10.1)=11 -> true interior extends
    to 0.011, one cell beyond the nominal 0.0101); a coordinate in that
    residual band reads as "in the absorber" here even though the real
    grid still treats it as interior. This makes every consumer
    conservative (may warn slightly early) rather than permissive (never
    silently misses a genuine overlap) — the thickness itself is not
    needed for a membership test, only whether it is active.
    """
    lo_boundary = 0.0 if ct_lo > 0 else None
    hi_boundary = domain_extent if ct_hi > 0 else None
    return lo_boundary, hi_boundary


def _coord_in_absorber(
    coord: float, domain_extent: float, ct_lo: float, ct_hi: float,
) -> bool:
    """True iff ``coord`` (one axis, user coordinates) is inside the
    EXTERIOR-padded absorber on that axis. See
    :func:`_absorber_boundary_for_axis` for the frame this rests on."""
    lo_b, hi_b = _absorber_boundary_for_axis(domain_extent, ct_lo, ct_hi)
    return (lo_b is not None and coord < lo_b) or (hi_b is not None and coord > hi_b)


# Proximity-advisory margin for _validate_cfg_absorber_placement (issue
# #500 review finding M3 / H1). NOT a dedicated calibration sweep like the
# MSL 8*dx buffer below (_check_msl_port_geometry) — #510 established that
# probe-to-absorber clearance is a real, previously-ungated hazard class
# (an MSL probe span could sit inside the CPML or past another port's feed
# with preflight silent), and #478 is the PR that introduced the
# internal/external probe distinction (`_internal_probe_indices`) this
# advisory must respect so library witness probes stay exempt. Neither
# pins a specific cell count for a GENERIC (non-MSL) probe/source
# proximity check, so 2 cells is a deliberately modest, conservative
# default: small enough to stay a "you are suspiciously close to the
# edge" advisory rather than a broad interior band, large enough to
# still catch the #470/#500-H1 regression-lock fixtures (a probe one
# grid cell inside the domain) this margin must keep firing on.
# Issue #510 nit 2: an earlier version of this comment justified the "2"
# by invoking _absorber_boundary_for_axis's own up-to-one-cell hi-side
# conservatism — that was a non-sequitur. That conservatism is a
# MEMBERSHIP concern (it can shift whether a coordinate reads as
# absorber_overlap at all, via the boundary position _coord_in_absorber
# uses); this margin is a PROXIMITY concern — it measures distance from
# whatever boundary _absorber_boundary_for_axis returns, unaffected by
# how that boundary itself was derived. The two are independent; neither
# bounds the other.
_ABSORBER_PROXIMITY_CELLS = 2


def _coord_near_absorber(
    coord: float, domain_extent: float, ct_lo: float, ct_hi: float,
    dx: float, n_cells: int = _ABSORBER_PROXIMITY_CELLS,
) -> bool:
    """True iff ``coord`` is NOT in the absorber (see
    :func:`_coord_in_absorber`) but sits within ``n_cells * dx`` of the
    boundary where an active absorber begins. Callers should check
    :func:`_coord_in_absorber` first — the two are meant to be mutually
    exclusive (membership is the more severe finding)."""
    lo_b, hi_b = _absorber_boundary_for_axis(domain_extent, ct_lo, ct_hi)
    margin = n_cells * dx
    near_lo = lo_b is not None and lo_b <= coord < lo_b + margin
    near_hi = hi_b is not None and hi_b - margin < coord <= hi_b
    return near_lo or near_hi


# --------------------------------------------------------------------------
# Issue #703: campaign statics checks — tunables + shared lazy context.
#
# Four failure classes a month-long external cross-validation hit, all
# statically detectable before the first time step (issue #703; message
# design per docs/design_notes/preflight_lessons_from_a_long_crossval.md:
# every finding carries OBSERVED / WHY / COST / REMEDY / STALE-IF plus a
# COVERAGE clause, and each check aggregates into ONE message per run —
# the #697 failure mode was 84 advisories with 93% duplication).
#
# The gate values are module-level on purpose: the falsification tests
# monkeypatch them in BOTH directions (loosen -> firing fixture goes
# silent; tighten -> silent fixture fires) to prove each gate is
# load-bearing (tests/test_preflight_campaign_statics*.py).
# --------------------------------------------------------------------------

# Check 1 — congruence key quantum (extents equal within 1e-9 m) and the
# tolerated rasterized-cell-count spread inside one congruence group. The
# tolerance is one cell: the smallest extent of the incident class (a
# node-thin sheet) rasterizes to a single cell, so any spread beyond one
# cell means the lattice sees two different solids where the design has one.
_CONGRUENCE_EXTENT_QUANTUM_M = 1e-9
_CONGRUENCE_SPREAD_TOL_CELLS = 1
# Check 2 — relative tolerance for "assigned statics == statics at the live
# edge" (float32 arrays; 1e-4 is ~1e3 ULP at eps_r ~ 4, far below any real
# material contrast).
_LIVE_EDGE_RTOL = 1e-4
# Check 3 — advisory threshold on either electrical-thickness measure.
_CAVITY_THICKNESS_TOL = 0.01
# Check 3 — a node-thin sheet "owns" a mesh cell when one cell holds this
# fraction of the sheet AND is that fraction full of it. A face-registered
# sheet (both faces on nodes) fills its cell ~1.0 and owns it; a mid-plane
# registered sheet straddles two cells, filling neither by more than half,
# and owns none — it is realized as a zero-thickness node. This decides
# WHICH of the two mechanisms the message attributes a pair's excess to,
# not how far the cavity is summed: an owned cell is still inside the
# electrical cavity, because rfx zeroes only tangential E on a one-cell
# PEC sheet and the cell's normal-E edge stays live.
_CAVITY_SHEET_CELL_FILL_FRAC = 0.9
# Check 4 — off-lattice face residual as a fraction of the axis extent.
_OFF_LATTICE_EDGE_TOL = 5e-3
# Shared cap on named offenders per aggregated message.
_CAMPAIGN_MAX_OFFENDERS = 5
# "Thinner than its local cell" margin — same 1% slack as
# rfx.geometry.rasterize_grid._subcell_box_axis_window.
_CAMPAIGN_SUBCELL_FACTOR = 1.01


def _sorted_box_corners(shape):
    """``(lo, hi)`` float64 arrays for a Box-like shape, else ``(None, None)``."""
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    if lo is None or hi is None:
        return None, None
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.minimum(lo, hi), np.maximum(lo, hi)


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


class _CampaignStaticsContext:
    """Shared lazily-built state for the four issue-#703 campaign checks.

    Built fresh per ``_validate_simulation_config`` call and deliberately
    NOT cached on the ``Simulation`` (an ``add_box`` after a preflight
    would leave a cached grid/mask stale). The grid, sample coordinates
    and per-entry masks come from the PRODUCTION builders — the check must
    see the run's rasterization, not a hand model of it (the
    ``_validate_cfg_graded_box_rasterization`` lesson: a hand model of the
    sampling diverged from the rasterizer on exactly the cases it existed
    to catch).
    """

    _NARROW_EXCS = (ValueError, TypeError, NotImplementedError, KeyError,
                    AttributeError, IndexError)

    def __init__(self, sim):
        self.sim = sim
        self.error: str | None = None
        self.lane: str | None = None
        self.grid = None
        self.coords = None       # production GridCoords (E-node samples)
        self.nodes = None        # 3x float64 node-position arrays
        self.spacings = None     # 3x float64 per-cell spacing arrays
        self._entry_masks: dict[int, np.ndarray] = {}
        self._assembled = None
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
            GridCoords, _axis_node_positions, coords_from_nonuniform_grid,
        )
        try:
            if any(p is not None for p in profiles):
                self.lane = "nonuniform"
                grid = sim._build_nonuniform_grid()
                self.coords = coords_from_nonuniform_grid(grid)
                nodes, spacings = [], []
                for d_arr, pad in (
                    (grid.dx_arr, getattr(grid, "pad_x_lo", grid.cpml_layers)),
                    (grid.dy_arr, getattr(grid, "pad_y_lo", grid.cpml_layers)),
                    (grid.dz, getattr(grid, "pad_z_lo", grid.cpml_layers)),
                ):
                    d_np = np.asarray(d_arr, dtype=np.float64)
                    nodes.append(_axis_node_positions(d_np, int(pad)))
                    spacings.append(d_np)
            else:
                self.lane = "uniform"
                grid = sim._build_grid()
                from rfx.geometry.csg import _grid_coords
                cx, cy, cz = _grid_coords(grid)
                self.coords = GridCoords(x=cx, y=cy, z=cz, shape=grid.shape)
                d = float(grid.dx)
                nodes, spacings = [], []
                for n, pad in zip(grid.shape, grid.axis_pads):
                    nodes.append((np.arange(n, dtype=np.float64) - pad) * d)
                    spacings.append(np.full(n, d, dtype=np.float64))
            self.grid = grid
            self.nodes = nodes
            self.spacings = spacings
        except self._NARROW_EXCS as exc:  # malformed config; preflight
            self.error = f"{type(exc).__name__}: {exc}"  # must not crash

    def rasterize(self, shape) -> np.ndarray:
        """This shape's PRODUCTION mask on this lane's grid, cached."""
        m = self._entry_masks.get(id(shape))
        if m is None:
            if self.lane == "uniform":
                m = np.asarray(shape.mask(self.grid))
            else:
                m = np.asarray(shape.mask_on_coords(
                    self.coords.x, self.coords.y, self.coords.z))
            self._entry_masks[id(shape)] = m
        return m

    def conductor_entries(self):
        """``(box_entries, other_entries)`` of PEC-class geometry entries."""
        sim = self.sim
        boxes, others = [], []
        for i, entry in enumerate(sim._geometry):
            mat = sim._resolve_material(entry.material_name)
            if mat.sigma < sim._PEC_SIGMA_THRESHOLD:
                continue
            lo, _hi = _sorted_box_corners(entry.shape)
            if isinstance(entry.shape, Box) and lo is not None:
                boxes.append((i, entry))
            else:
                others.append((i, entry))
        return boxes, others

    def congruence_entries(self):
        """``(keyed, unkeyed)`` conductor entries for the congruence check.

        ``keyed`` is ``(index, entry, lo, hi, exact)`` for every conductor
        entry that reports a bounding box through the public ``Shape``
        protocol; ``unkeyed`` is the rest, which the message must name as
        skipped.

        Deliberately NOT :meth:`conductor_entries`'s Box/non-Box split. A
        patterned metal LAYER is the incident class, and a ``Box`` cannot
        represent one — a Box fills the layer's clearance holes with metal
        — so such a layer arrives as a user-defined ``Shape`` (CAD sheet
        collapsed to one cell). An ``isinstance(shape, Box)`` census put
        every one of those into "skipped", which is how the check stayed
        silent on the very mirror pairs it was written for. Congruence
        needs an extent, not a Box: any shape that reports its bounds can
        be grouped, and whether the bounds ARE the shape is carried along
        (``exact``) so the message can say what it inferred.
        """
        sim = self.sim
        keyed, unkeyed = [], []
        for i, entry in enumerate(sim._geometry):
            mat = sim._resolve_material(entry.material_name)
            if mat.sigma < sim._PEC_SIGMA_THRESHOLD:
                continue
            bounds = _shape_bounds(entry.shape)
            if bounds is None:
                unkeyed.append((i, entry))
            else:
                keyed.append((i, entry, bounds[0], bounds[1], bounds[2]))
        return keyed, unkeyed

    def sheet_own_cell(self, axis: int, lo_a: float, hi_a: float,
                       node_idx: int):
        """The mesh cell this node-thin sheet FILLS along ``axis``, or None.

        A sheet whose two faces are registered as nodes occupies exactly
        one cell; a sheet registered at its mid-plane straddles two and
        fills neither, so it owns none and is realized as a
        zero-thickness node. Only the two cells touching the sheet's own
        node are candidates.

        Owning a cell does NOT take that cell out of the cavity — the
        cell's normal-E edge is live (``apply_pec_mask`` zeroes only
        tangential E on a one-cell PEC sheet), so its permittivity is in
        series with the gap. The caller uses this to say WHICH mechanism
        a cavity's excess came from.
        """
        thick = float(hi_a) - float(lo_a)
        if thick <= 0.0:
            return None
        nodes = self.nodes[axis]
        best = None
        for j in (node_idx - 1, node_idx):
            if j < 0 or j + 1 >= len(nodes):
                continue
            c_lo, c_hi = float(nodes[j]), float(nodes[j + 1])
            width = c_hi - c_lo
            if width <= 0.0:
                continue
            overlap = min(float(hi_a), c_hi) - max(float(lo_a), c_lo)
            if overlap <= 0.0:
                continue
            score = min(overlap / thick, overlap / width)
            if score < _CAVITY_SHEET_CELL_FILL_FRAC:
                continue
            if best is None or score > best[1]:
                best = (j, score)
        return None if best is None else best[0]

    def local_spacing(self, axis: int, coord: float) -> float:
        nodes = self.nodes[axis]
        j = int(np.searchsorted(nodes, coord)) - 1
        j = min(max(j, 0), len(self.spacings[axis]) - 1)
        return float(self.spacings[axis][j])

    def node_thin_axes(self, shape) -> list[int]:
        """Axes along which this Box is thinner than its local cell."""
        lo, hi = _sorted_box_corners(shape)
        if lo is None:
            return []
        mid = 0.5 * (lo + hi)
        return [
            a for a in range(3)
            if float(hi[a] - lo[a])
            <= self.local_spacing(a, float(mid[a])) * _CAMPAIGN_SUBCELL_FACTOR
        ]

    def assembled(self):
        """``(materials, pec_mask)`` from the PRODUCTION assembly, once."""
        if self._assembled is None and self.assembly_error is None:
            sim = self.sim
            try:
                if self.lane == "uniform":
                    mats, _, _, pec_mask, _, _, _ = (
                        sim._assemble_materials(self.grid))
                else:
                    mats, _, _, pec_mask = sim._assemble_materials_nu(self.grid)
                self._assembled = (mats, pec_mask)
            except self._NARROW_EXCS as exc:
                self.assembly_error = f"{type(exc).__name__}: {exc}"
        return self._assembled

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


class PreflightWarning(UserWarning):
    """Base for structured preflight findings carried on the warning instance.

    Mirrors the in-repo report idioms (:class:`SubgridValidationIssue`,
    :class:`PortValidationIssue`): the check site sets a stable lowercase-slug
    ``code`` and a ``severity`` on the warning instance, plus optional ``loc``
    (where in the setup the finding applies) and ``source`` (the check method
    name). ``preflight()`` reads these fields off ``w.message`` so the issue
    record is coded at the check site rather than inferred from text.

    Emit with ``warnings.warn(PreflightWarning(msg, code="...", source="..."))``.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        severity: str = "warning",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code
        self.severity = severity
        self.loc = loc
        self.source = source

    def __str__(self) -> str:  # back-compat: warning prints as its message
        return self.message


class PreflightErrorWarning(PreflightWarning):
    """An error-severity preflight finding emitted as a warning.

    Re-parented under :class:`PreflightWarning` (Phase A). Emitting (rather than
    raising) keeps the rest of the preflight suite running so the user sees ALL
    issues at once, while ``preflight()`` still tags the resulting
    :class:`PreflightIssue` with ``severity="error"`` so an automation agent can
    gate on it. Use for known-bad configurations that should stop a run.

    ``severity`` defaults to ``"error"``; the legacy
    ``warnings.warn("msg", PreflightErrorWarning)`` form (category, no instance
    attrs) still surfaces as error-severity via ``preflight()``'s
    ``issubclass(w.category, PreflightErrorWarning)`` derivation.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        severity: str = "error",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message, code=code, severity=severity, loc=loc, source=source
        )


class PreflightConfigError(ValueError):
    """A structurally-impossible-config raise carrying a check-site ``code``.

    The structurally-impossible config validators (``upml``+refinement,
    Floquet+non-uniform-z, ...) raise this so ``preflight()`` can record the
    error-severity :class:`PreflightIssue` with the slug set at the check site
    instead of inferring it from the message. It subclasses ``ValueError`` so
    every existing ``except ValueError`` / ``pytest.raises(ValueError)`` site
    (including the run() regression locks) is unaffected.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.loc = loc
        self.source = source


class PreflightIssue(str):
    """One preflight finding.

    Subclasses ``str`` so it is 100% back-compatible with the plain
    ``list[str]`` that ``preflight()`` has always returned — it prints, joins,
    compares, and regex-matches exactly like its message — while also carrying
    machine-readable fields so an automation agent can gate deterministically::

        report = sim.preflight()
        errors = [i for i in report if i.severity == "error"]
        if errors:
            ...  # stop before spending GPU minutes on a doomed run

    ``severity`` is ``"error"`` for hard contradictions / known-bad configs and
    ``"warning"`` for advisories. ``code`` is the lowercase-slug category set at
    the check site (e.g. ``"conformal_nan"``, ``"mesh_resolution"``,
    ``"absorber_overlap"``). ``loc`` and ``source`` are optional provenance.

    The str subclass silently drops these attrs under ``json.dumps`` — never
    serialize a :class:`PreflightIssue` directly; use :meth:`to_dict` or the
    owning :class:`PreflightReport`'s :meth:`PreflightReport.to_dict` /
    :meth:`PreflightReport.to_json`.
    """

    severity: str
    code: str
    loc: str | None
    source: str | None

    def __new__(
        cls,
        message,
        *,
        severity: str = "warning",
        code: str = "uncoded",
        loc: str | None = None,
        source: str | None = None,
    ):
        obj = super().__new__(cls, str(message))
        obj.severity = severity
        obj.code = code
        obj.loc = loc
        obj.source = source
        return obj

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable record of this finding."""
        return {
            "message": str(self),
            "code": self.code,
            "severity": self.severity,
            "loc": self.loc,
            "source": self.source,
        }


class PreflightReport(list):
    """Structured result of :meth:`Simulation.preflight`.

    A ``list`` subclass holding :class:`PreflightIssue` items, so it IS a list
    and every legacy ``list[str]`` call site (iterate / ``"\\n".join`` / ``len``
    / truthiness) keeps working unchanged. It also exposes the canonical report
    API shared with :class:`rfx.validation.PortValidationReport` and
    :class:`rfx.subgridding.validation.SubgridValidationReport`.
    """

    @property
    def issues(self) -> list:
        """All findings as a plain list (mirrors the other report classes)."""
        return list(self)

    @property
    def errors(self) -> list:
        """Error-severity findings only."""
        return [i for i in self if getattr(i, "severity", "warning") == "error"]

    @property
    def warnings(self) -> list:
        """Non-error (advisory) findings only."""
        return [i for i in self if getattr(i, "severity", "warning") != "error"]

    @property
    def ok(self) -> bool:
        """Whether the report contains no error-severity finding."""
        return not self.errors

    def by_code(self, code: str) -> list:
        """Return all findings with diagnostic ``code``."""
        return [i for i in self if getattr(i, "code", None) == code]

    def format(self) -> str:
        """Return a compact human-readable multiline summary."""
        status = "PASS" if self.ok else "FAIL"
        if not self:
            return f"preflight: {status} (no issues)"
        lines = [f"preflight: {status} ({len(self)} issue(s))"]
        for issue in self:
            sev = getattr(issue, "severity", "warning")
            code = getattr(issue, "code", "uncoded")
            lines.append(f"- {sev.upper()} [{code}] {issue}")
        return "\n".join(lines)

    def raise_for_failure(self) -> "PreflightReport":
        """Raise ``ValueError`` listing every error-severity finding.

        Returns ``self`` on success so callers can use it as both a fail-fast
        gate and an artifact (the R3 pre-VESSL gate). No-op when :attr:`ok`.
        """
        errors = self.errors
        if errors:
            detail = "\n  - ".join(str(e) for e in errors)
            raise ValueError(
                f"preflight found {len(errors)} blocking error(s):\n  - {detail}"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable validation artifact.

        Real serialization (unlike ``json.dumps`` of a bare
        :class:`PreflightIssue`, which drops the code/severity attrs).
        """
        return {
            "ok": self.ok,
            "n_issues": len(self),
            "n_errors": len(self.errors),
            "issues": [
                i.to_dict() if isinstance(i, PreflightIssue)
                else {
                    "message": str(i),
                    "code": getattr(i, "code", "uncoded"),
                    "severity": getattr(i, "severity", "warning"),
                    "loc": getattr(i, "loc", None),
                    "source": getattr(i, "source", None),
                }
                for i in self
            ],
        }

    def to_json(self, **kwargs: object) -> str:
        """Serialize the report for research-note artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)


# --------------------------------------------------------------------------
# MSL probe-clearance geometry (module-level so compute_msl_s_matrix can
# reuse the identical arithmetic for the issue-#469 interval solve).
# --------------------------------------------------------------------------

# Conservative ε_eff proxy → upper bound on β → lower bound on λ_g → most
# stringent (smallest) recommended clearance. For air-only lines this is
# overly conservative, but the cost of a false-positive advisory is low.
MSL_EPS_EFF_PROXY = 5.0


def msl_min_probe_clearance(freq_max: float) -> float:
    """Minimum probe-to-reflector clearance: λ_g/4 at ``freq_max``.

    At lower frequencies λ_g is larger and the same physical clearance
    represents fewer cells of standing-wave-free zone — f_max is the
    worst case.
    """
    c0 = 2.998e8
    lambda_g_min = c0 / (float(freq_max) * (MSL_EPS_EFF_PROXY ** 0.5))
    return 0.25 * lambda_g_min


def msl_nearest_downstream_reflector(
    geometry,
    *,
    x_probe: float,
    x_feed: float,
    y_feed: float,
    w_trace: float,
    dx: float,
    domain_y: float,
    direction: str,
    resolve_material=None,
    thin_conductors=(),
    pec_sigma_threshold: float = 1e6,
):
    """Distance from ``x_probe`` to the nearest downstream conductor edge.

    Walks every registered CONDUCTOR and returns
    ``(distance_m, label, unevaluated)`` for the nearest one at or beyond
    ``x_probe`` along the propagation direction — ``(inf, None,
    unevaluated)`` when nothing qualifies. ``unevaluated`` is the list of
    conductors this scan could NOT place (one string each); it is what
    lets the caller distinguish "nothing is nearby" from "I could not
    look" (issue #685).

    What counts as a conductor (issue #685). This used to be
    ``isinstance(shape, Box) and str(material_name).lower() == "pec"``,
    which was blind to two whole classes:

    * **PEC-PROMOTED materials** — a conductor registered with
      ``sigma >= 1e6`` under any other name. That is the common case for
      imported CAD, where every conductor may be called ``"metal"``. Pass
      ``resolve_material`` (normally ``sim._resolve_material``) and the
      test becomes the same ``sigma >= pec_sigma_threshold`` rule the
      assembler itself uses to build ``pec_mask``. Without it the legacy
      name test is kept so old callers do not change behaviour.
    * **non-``Box`` shapes** — ``Sheet``, ``MeshShape`` (#358), CSG
      results. Any shape with a ``bounding_box()`` is now placed by that
      box; a bounding box OVERSTATES a non-convex outline, which is the
      conservative direction for a clearance advisory (it can only bring
      the reported reflector nearer, never push it away). A shape with no
      usable bounding box goes to ``unevaluated`` instead of being
      skipped in silence.

    ``thin_conductors`` (normally ``sim._thin_conductors``) are scanned
    too: a thin PEC sheet, a DC-fold lossy sheet and a
    ``surface_impedance_f0`` sheet are all metal to a wave on the line,
    and since #677 the f0 one is in neither ``pec_mask`` nor
    ``materials.sigma``, so nothing else would ever see it.

    Observed consequence of the old scan: on a board whose reflectors were
    all thin conductors, it found none. Probe 0 then sat well inside this
    repo's own downstream rule (``msl_min_probe_clearance``) and its
    upstream rule (``5*h_sub``), and two further probes landed physically
    on metal -- with nothing warned. The old scan read only volumetric
    ``geometry``, so a board built from sheets was invisible to it.

    Axis generality (issue #661): the parameter names are the ``"+x"``-frame
    names. ``x_probe`` / ``x_feed`` are coordinates on the PROPAGATION axis,
    ``y_feed`` is the trace centreline on the WIDTH axis and ``domain_y``
    that axis's domain extent — the two axes are resolved from ``direction``
    via :func:`~rfx.sources.msl_port.msl_axis_roles`, so a ``"+y"`` port
    compares box y-extents along the feed axis and box x-extents across the
    trace width. Callers pass coordinates already projected onto those roles.

    Exclusions (both are #469-arc corrections to the pre-existing
    heuristic):

    * **the line being measured** — any trace-width box
      (|y-extent − w_trace| ≤ dx, y-range containing the feed centreline)
      whose x-range contains the FEED plane. This covers the through-line
      AND a port's own feed trace; the old rule (x-extent ≥ 80 % of an
      inter-port-extent estimate) missed the latter for '-x' ports —
      measured d=0 false positive against the port's OWN output feed
      (validation/crossval/07_sheen_lpf.py known-residual note) — and its
      '+x' extent estimate evaluated to the CPML thickness instead of the
      far-wall coordinate (latent arithmetic bug, now moot: the estimate
      is gone). A same-width series element that does NOT contain the
      feed plane is a genuine discontinuity and is still counted.
    * **ground-plane-like boxes** (y-extent ≥ 80 % of the domain y).
    """
    from rfx.geometry.csg import Box as _Box
    from rfx.sources.msl_port import _MSL_AXIS_INDEX, msl_axis_roles

    _prop_ax, _width_ax, _n_ax, sign = msl_axis_roles(direction)
    _ip = _MSL_AXIS_INDEX[_prop_ax]
    _iw = _MSL_AXIS_INDEX[_width_ax]
    nearest_d = float("inf")
    nearest_label = None
    unevaluated: list[str] = []

    def _is_conductor(material_name) -> bool:
        if resolve_material is None:
            # Legacy name test, kept for callers that cannot resolve.
            return str(material_name).lower() == "pec"
        try:
            mat = resolve_material(material_name)
        except Exception:
            unevaluated.append(
                f"geometry entry with material {material_name!r}: the "
                f"material could not be resolved, so its conductivity is "
                f"unknown")
            return False
        sigma = getattr(mat, "sigma", None)
        if sigma is None:
            return str(material_name).lower() == "pec"
        try:
            return float(sigma) >= float(pec_sigma_threshold)
        except (TypeError, ValueError):
            # A traced sigma (material-as-design-variable) cannot be
            # compared host-side.
            unevaluated.append(
                f"geometry entry with material {material_name!r}: sigma is "
                f"not a concrete number (traced design variable), so PEC "
                f"promotion cannot be decided host-side")
            return False

    def _bounds(shape, what: str):
        lo = getattr(shape, "corner_lo", None)
        hi = getattr(shape, "corner_hi", None)
        if lo is not None and hi is not None:
            return lo, hi
        bbox = getattr(shape, "bounding_box", None)
        if bbox is None:
            unevaluated.append(
                f"{what} ({type(shape).__name__}): no corner_lo/corner_hi "
                f"and no bounding_box(), so it cannot be placed on the line")
            return None, None
        try:
            lo, hi = bbox()
        except Exception as exc:
            unevaluated.append(
                f"{what} ({type(shape).__name__}): bounding_box() raised "
                f"{type(exc).__name__}, so it cannot be placed on the line")
            return None, None
        return lo, hi

    # (shape, label_prefix, is_bbox_derived) for every registered conductor.
    candidates: list = []
    for _gi, ge in enumerate(geometry):
        shape = getattr(ge, "shape", None)
        mat = getattr(ge, "material_name", "")
        if shape is None or not _is_conductor(mat):
            continue
        candidates.append((shape, f"conductor '{mat}'",
                           not isinstance(shape, _Box)))
    for _ti, tc in enumerate(thin_conductors or ()):
        tshape = getattr(tc, "shape", None)
        if tshape is None:
            continue
        candidates.append((tshape, f"thin_conductor[{_ti}]",
                           not isinstance(tshape, _Box)))

    for shape, _what, _from_bbox in candidates:
        lo, hi = _bounds(shape, _what)
        if lo is None or hi is None:
            continue
        # "x" = propagation axis, "y" = trace-width axis (issue #661).
        box_x_lo, box_x_hi = float(lo[_ip]), float(hi[_ip])
        box_y_lo, box_y_hi = float(lo[_iw]), float(hi[_iw])
        box_y_extent = box_y_hi - box_y_lo
        # Skip the line being measured (see docstring).
        if (
            abs(box_y_extent - w_trace) <= dx
            and box_y_lo - dx <= y_feed <= box_y_hi + dx
            and box_x_lo - dx <= x_feed <= box_x_hi + dx
        ):
            continue
        # Skip ground-plane-like boxes.
        if box_y_extent >= 0.8 * domain_y:
            continue
        # Distance from x_probe to the nearest edge of this box,
        # measured ALONG the propagation direction.
        if sign > 0:
            if box_x_lo > x_probe:
                d = box_x_lo - x_probe
            elif box_x_hi < x_probe:
                continue  # behind the probe
            else:
                d = 0.0
        else:
            if box_x_hi < x_probe:
                d = x_probe - box_x_hi
            elif box_x_lo > x_probe:
                continue
            else:
                d = 0.0
        if d < nearest_d:
            nearest_d = d
            _how = " (bounding box)" if _from_bbox else ""
            nearest_label = (
                f"{_what}{_how} at {_prop_ax}∈[{box_x_lo*1e3:.2f},"
                f"{box_x_hi*1e3:.2f}]mm "
                f"{_width_ax}∈[{box_y_lo*1e3:.2f},{box_y_hi*1e3:.2f}]mm"
            )
    return nearest_d, nearest_label, unevaluated


def msl_absorber_compliant_offset_max(
    grid,
    port,
    *,
    n_probes: int,
    n_spacing: int,
    off_lo: int,
    domain_x: float,
    ct_lo: float,
    ct_hi: float,
    dx: float,
    guess_hi: int,
) -> int | None:
    """Largest ``n_probe_offset >= off_lo`` (up to ``guess_hi``) whose
    resulting GRID-SNAPPED deepest probe clears both
    :func:`_coord_in_absorber` and :func:`_coord_near_absorber`.

    Issue #510 review finding (BLOCKING 1): the first version of this
    check derived the advertised interval endpoint by algebraically
    INVERTING the two predicates — ``int((headroom - margin) / dx) -
    (n_probes-1)*n_spacing``. Float division plus truncation, combined
    with :func:`_coord_near_absorber`'s boundary sitting at exactly
    ``n_cells*dx``, put the computed endpoint on the wrong side of an FP
    knife edge whenever the true boundary landed within about one ULP
    of an exact multiple of ``dx`` — reviewer's brute-force sweep found
    roughly 12,000 ``(dx, n_spacing, feed, domain)`` combinations where
    the ADVERTISED endpoint itself still tripped the warning it claimed
    to clear.

    This walks candidate offsets DOWN from ``guess_hi`` and asks the
    REAL predicate at each one (via ``msl_probe_x_coords_n``, the same
    grid-index/clamping arithmetic the extractor uses), rather than
    computing an answer algebraically. The returned value, if any, is
    verified compliant by construction — it cannot land on the wrong
    side of the boundary the way an algebraic inversion can.

    Returns ``None`` if no offset in ``[off_lo, guess_hi]`` clears
    (the compliant interval is empty).

    Axis generality (issue #661): ``domain_x`` / ``ct_lo`` / ``ct_hi``
    describe the port's PROPAGATION axis, not x specifically —
    ``msl_probe_x_coords_n`` already walks whichever axis ``port.direction``
    names, so the caller passes that axis's domain extent and CPML
    thicknesses.
    """
    from rfx.sources.msl_port import msl_probe_x_coords_n as _probe_x_coords_n

    off = guess_hi
    while off >= off_lo:
        ladder = _probe_x_coords_n(
            grid, port, n_probes=n_probes,
            n_offset_cells=off, n_spacing_cells=n_spacing,
        )
        x_deep_candidate = ladder[-1]
        if not (
            _coord_in_absorber(x_deep_candidate, domain_x, ct_lo, ct_hi)
            or _coord_near_absorber(x_deep_candidate, domain_x, ct_lo, ct_hi, dx)
        ):
            return off
        off -= 1
    return None


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
                if self._dx and pe.n_probe_offset < 5.0 * pe.height / self._dx:
                    messages.append(
                        f"MSL port {pe.name!r}: n_probe_offset="
                        f"{pe.n_probe_offset} sits within the source fringing "
                        f"transient (~{5.0 * pe.height / self._dx:.0f} cells = "
                        f"5·h_sub/dx); probe 0 may corrupt the V·I-split S11 of "
                        f"a high-Q resonant load (issue #80) — increase "
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
                if dim <= 0:
                    # Zero-thickness geometry
                    axis_name = "xyz"[axis]
                    _w.warn(
                        PreflightWarning(
                            f"Zero-thickness geometry '{mat_name}' along "
                            f"{axis_name}-axis. On non-uniform mesh this may "
                            f"produce empty rasterization. Consider giving it "
                            f"at least one cell of thickness ({_fmt_len(cell)}).",
                            code="mesh_resolution",
                            source="_validate_mesh_quality",
                        ),
                        stacklevel=3,
                    )
                elif dim < cell:
                    axis_name = "xyz"[axis]
                    cells_count = dim / cell
                    # Check if this is a PEC material that could use thin sheet
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    # This used to advise add_thin_conductor() here. Measured:
                    # for a PEC material that advice is a NO-OP — a sub-cell PEC
                    # Box and add_thin_conductor() on the same footprint produce
                    # a BIT-IDENTICAL pec_mask (verified in
                    # tests/test_thin_conductor_honesty.py). Neither models a
                    # sub-cell thickness; both give one cell of mask, which
                    # rfx/boundaries/pec.py treats as a surface. Say what is
                    # actually true instead of offering a change that changes
                    # nothing (issue #504).
                    hint = (
                        " A conductor thinner than a cell is modelled as a "
                        "one-cell PEC surface — tangential E is zeroed on it "
                        "and the normal component survives as surface charge. "
                        "That is usually what you want for metal many skin "
                        "depths thick, and switching to add_thin_conductor() "
                        "would not change it; but it means the sheet carries "
                        "no conductor loss and its thickness is not modelled. "
                        "For band-centre conductor loss use "
                        "add_thin_conductor(..., surface_impedance_f0=...) "
                        "(Leontovich surface resistance, issue #669)."
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
                    # PEC with extent <3 cells is a thin sheet — 1-cell
                    # rasterization is canonical. Only warn on partial
                    # volume: 3-5 cells thick PEC slabs.
                    # Dielectric: cells per local λ_eff, not cells per
                    # geometry extent.
                    mat = self._resolve_material(mat_name)
                    is_pec = mat.sigma >= self._PEC_SIGMA_THRESHOLD
                    axis_name = "xyz"[axis]
                    cells = dim / cell
                    if is_pec:
                        if (3.0 <= cells < 5.0
                                and not is_thin_along_some_axis):
                            _w.warn(
                                PreflightWarning(
                                    f"PEC '{mat_name}' {axis_name}-extent "
                                    f"{_fmt_len(dim)} = {cells:.1f} cells — "
                                    "volume under-resolved (PEC volume needs "
                                    "≥5 cells; thin sheets <3 cells are fine).",
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
        """Warn when a thin PEC sheet sits on a NU axis without symmetric
        neighbouring cells (Meep/OpenEMS require equal dz on both sides of
        a metal plane, else surface currents pick up O(1) error and the
        far-field pattern is corrupted — issue #48).
        """
        import warnings as _w
        profiles = (
            ("x", self._dx_profile),
            ("y", self._dy_profile),
            ("z", self._dz_profile),
        )
        for axis_name, prof in profiles:
            if prof is None:
                continue
            if is_tracer(prof):
                # Tracer profiles can't be host-scanned for edge / ratio
                # checks. The warning is advisory only; correctness is
                # preserved downstream.
                continue
            prof_arr = np.asarray(prof, dtype=np.float64)
            if len(prof_arr) < 3:
                continue
            axis_idx = "xyz".index(axis_name)
            for entry in self._geometry:
                mat = self._resolve_material(entry.material_name)
                if mat.sigma < self._PEC_SIGMA_THRESHOLD:
                    continue
                try:
                    c1, c2 = entry.shape.bounding_box()
                except Exception:
                    continue
                lo, hi = float(c1[axis_idx]), float(c2[axis_idx])
                extent = hi - lo
                min_d = float(prof_arr.min())
                if extent > min_d * 1.5:
                    continue
                # _dz_profile is the user's interior profile (no CPML
                # padding). Geometry coordinates are in interior space
                # starting at 0, so cumsum gives the cell edges directly.
                edges = np.concatenate([[0.0], np.cumsum(prof_arr)])
                mid = 0.5 * (lo + hi)
                k = int(np.searchsorted(edges, mid) - 1)
                if k < 0 or k + 1 >= len(prof_arr) or k - 1 < 0:
                    continue
                dz_here = prof_arr[k]
                dz_above = prof_arr[k + 1]
                dz_below = prof_arr[k - 1]
                # Check ratio both directions — metal-in-coarse-cell
                # next to a fine region is just as bad as the reverse.
                def _ratio(a, b):
                    return max(a, b) / min(a, b)
                ratio_above = _ratio(dz_above, dz_here)
                ratio_below = _ratio(dz_below, dz_here)
                if max(ratio_above, ratio_below) > 1.5:
                    _w.warn(
                        PreflightWarning(
                            f"Thin PEC '{entry.material_name}' on axis "
                            f"{axis_name} sits in a cell of dz={dz_here*1e3:.3f}"
                            f"mm with asymmetric neighbours "
                            f"(below {dz_below*1e3:.3f}, above "
                            f"{dz_above*1e3:.3f} mm). Meep/OpenEMS require "
                            f"equal cell sizes across a metal plane; "
                            f"radiation pattern may be corrupted (issue #48). "
                            f"Put the metal on a preserved-region boundary "
                            f"or refine the neighbouring cell.",
                            code="thin_metal_nu_mesh",
                            source="_validate_thin_metal_on_nu_mesh",
                        ),
                        stacklevel=4,
                    )

    def _port_pec_mask(self, grid):
        """Assembled ``pec_mask`` for the uniform lane, or ``None``.

        Issue #738 review: the guide a waveguide port sits in is defined
        by its WALLS, and on the committed sub-aperture fixtures those
        walls are interior PEC shapes, not the domain faces. Read from
        the PRODUCTION assembly (``_assemble_materials``) so preflight
        sees the same rasterized conductors the solve does — no
        geometric re-derivation from the shape list. Called once per
        preflight and threaded through, so the assembly runs once. A
        simulation with no geometry entries at all has no interior walls
        by construction, so the assembly is skipped there rather than
        run to produce an all-False mask.
        """
        if not self._geometry and not getattr(self, "_thin_conductors", None):
            return None
        try:
            _, _, _, pec_mask, _, _, _ = self._assemble_materials(grid)
        except (ValueError, TypeError, NotImplementedError, KeyError,
                AttributeError, IndexError):
            # Deliberately NOT ``except Exception`` (PR #555): an async
            # worker timeout must propagate through this advisory. On a
            # narrow failure the caller falls back to the aperture,
            # which is always defined.
            return None
        if pec_mask is None:
            return None
        return np.asarray(pec_mask).astype(bool)

    def _port_transverse_spans(self, entry, grid, pec_np=None):
        """Per transverse axis, the widths one waveguide port has on THIS grid.

        Returns ``{axis_name: dict}`` with keys:

        - ``declared``: what the config states — ``entry.{axis}_range``
          width, or the full axis domain when the range is left unset.
        - ``aperture``: the span :meth:`_range_to_slice` REPORTS on this
          grid — the identical call :meth:`_build_waveguide_port_config`
          makes to build ``WaveguidePort.a``/``.b``, i.e. exactly the
          mode-template / cutoff dimension the solve uses. ``None`` when
          the range does not resolve to a valid slice at all
          (:meth:`_range_to_slice` raises — the run would fail to
          compile).
        - ``rasterized``: the span the returned slice actually covers on
          the grid, ``(hi_idx - lo_idx - 1) * dx``. Equal to
          ``aperture`` on the explicit branch by construction; on the
          ``value_range is None`` branch ``_range_to_slice`` reports
          ``domain_max`` instead (issue #729 site 2), so the two differ
          whenever ``dx`` does not divide the domain.
        - ``guide``: the wall-to-wall transverse extent of the guide the
          port sits in, measured on the assembled ``pec_mask`` along the
          port's own transverse line. ``guide_source`` says where it
          came from: ``"pec_walls"`` (an interior PEC wall was found on
          both sides), ``"domain_faces"`` (no interior wall, and the
          axis' two domain faces are both PEC/PMC, so the closed domain
          IS the guide), or ``"aperture"`` (neither — the transverse
          axis is not closed, so no guide wider than the port's own
          aperture can be asserted and ``guide`` falls back to
          ``rasterized``).

        Issue #738 (family #737) lead measurement,
        ``examples/inverse_design/differentiable_s11_design.py`` at its
        then-committed dx = 2 mm (declared WR-90 a = 22.860 mm): the grid
        rasterized a 22.000 mm aperture, and preflight — which read only
        ``declared`` — printed "All checks passed". That example now
        carries a commensurate dx = 1.27 mm, so reproducing the
        measurement needs the old value.

        The first version of this helper set ``guide`` to the rasterized
        extent of the whole transverse DOMAIN. Review measurement on
        ``tests/test_waveguide_port_reference_sims.py::_tj_device`` (PEC
        Boxes fill y in [0, 0.04] and [0.08, 0.12], ports
        ``y_range=(0.04, 0.08)``) showed that is wrong wherever the walls
        are interior PEC: it reported fc_TE10 = 1.249 GHz / fc_TE20 =
        2.498 GHz, the cutoffs of the 120 mm DOMAIN. Measured on the
        mask instead, that fixture's walls are 42.0000 mm apart (the
        Boxes rasterize half-open to y-nodes 10..29 and 50..69 at
        dx = 2 mm, so the outermost PEC nodes are 29 and 50) and the
        cutoffs are 3.569 / 7.138 GHz; the DECLARED 40 mm guide would
        give 3.747 / 7.495 GHz. Hence the mask read below.

        ``aperture`` can land on either side of ``declared``:
        :meth:`_range_to_slice`'s explicit branch rounds range endpoints
        to the nearest cell, so it snaps above OR below depending on
        which side of a half-cell the endpoints fall (see the round-up
        case in ``tests/test_port_aperture_rasterization.py``).

        On a conformal (Dey-Mittra) axis the wall sits at the exact
        declared coordinate via a fractional-cell eps_correction
        (``tests/test_subpixel_pec.py``, "Stage 1 step 3"), not the
        ``(n-1)*dx`` staircase — disclosed, unmeasured non-regression:
        a conformal closed axis keeps the declared domain extent, which
        is what this code did before #738.
        """
        normal = entry.direction[1]
        axes = [a for a in "xyz" if a != normal]
        conformal = (self._boundary_spec.conformal_faces()
                     if self._boundary_spec is not None else set())
        closed = set()
        if self._boundary_spec is not None:
            closed = (self._boundary_spec.pec_faces()
                      | self._boundary_spec.pmc_faces())

        out: dict[str, dict] = {}
        slices: dict[str, tuple[int, int]] = {}
        for axis_name in axes:
            axis_idx = "xyz".index(axis_name)
            value_range = getattr(entry, f"{axis_name}_range")
            n_axis = (grid.nx, grid.ny, grid.nz)[axis_idx]
            declared = (float(value_range[1] - value_range[0])
                        if value_range is not None
                        else float(self._domain[axis_idx]))
            rec = {
                "declared": declared,
                "aperture": None,
                "rasterized": None,
                "guide": None,
                "guide_source": "aperture",
                "error": None,
                "explicit": value_range is not None,
            }
            try:
                slc, aperture = self._range_to_slice(
                    value_range, self._domain[axis_idx], grid.dx, n_axis,
                    grid.axis_pads[axis_idx],
                )
            except ValueError as exc:
                # _range_to_slice raises at COMPILE time on an
                # unrasterizable range; preflight(strict=False) is
                # contracted to COLLECT findings, never crash, so this
                # records the failure instead of propagating it.
                rec["error"] = str(exc)
                out[axis_name] = rec
                continue
            rec["aperture"] = float(aperture)
            rec["rasterized"] = float((slc[1] - slc[0] - 1) * grid.dx)
            rec["guide"] = rec["rasterized"]
            slices[axis_name] = (int(slc[0]), int(slc[1]))
            out[axis_name] = rec

        if len(slices) != 2:
            # One transverse axis did not resolve: no line to scan along.
            return out

        normal_idx = "xyz".index(normal)
        try:
            pos_vec = [0.0, 0.0, 0.0]
            pos_vec[normal_idx] = entry.x_position
            plane_idx = int(
                grid.position_to_index(tuple(pos_vec))[normal_idx])
        except (ValueError, TypeError, IndexError, AttributeError):
            plane_idx = None

        for axis_name in axes:
            axis_idx = "xyz".index(axis_name)
            other = [a for a in axes if a != axis_name][0]
            rec = out[axis_name]
            lo_idx, hi_idx = slices[axis_name]
            pad_lo = grid.face_pads[2 * axis_idx]
            pad_hi = grid.face_pads[2 * axis_idx + 1]
            n_axis = (grid.nx, grid.ny, grid.nz)[axis_idx]
            interior_lo = pad_lo
            interior_hi = n_axis - pad_hi - 1
            axis_closed = (f"{axis_name}_lo" in closed
                           and f"{axis_name}_hi" in closed)
            axis_conformal = (f"{axis_name}_lo" in conformal
                              and f"{axis_name}_hi" in conformal)

            wall_lo = wall_hi = None
            if pec_np is not None and plane_idx is not None:
                other_idx = "xyz".index(other)
                o_lo, o_hi = slices[other]
                mid_other = (o_lo + o_hi - 1) // 2
                idx = [0, 0, 0]
                idx[normal_idx] = plane_idx
                idx[other_idx] = mid_other
                if all(0 <= idx[k] < pec_np.shape[k]
                       for k in (normal_idx, other_idx)):
                    line = pec_np[
                        idx[0] if 0 != axis_idx else slice(None),
                        idx[1] if 1 != axis_idx else slice(None),
                        idx[2] if 2 != axis_idx else slice(None),
                    ]
                    line = np.asarray(line).reshape(-1)
                    # Scan OUTWARD from the aperture's own edge nodes,
                    # INCLUSIVE: on a sub-aperture port a wall can sit on
                    # the aperture's own edge node (measured on
                    # _tj_device: the upper Box's first PEC node IS the
                    # aperture's last node, index 50), so an exclusive
                    # scan would step straight past it.
                    for i in range(min(lo_idx, len(line) - 1),
                                   interior_lo - 1, -1):
                        if line[i]:
                            wall_lo = i
                            break
                    for i in range(max(hi_idx - 1, 0), interior_hi + 1):
                        if i >= len(line):
                            break
                        if line[i]:
                            wall_hi = i
                            break

            if wall_lo is not None and wall_hi is not None and wall_hi > wall_lo:
                rec["guide"] = float((wall_hi - wall_lo) * grid.dx)
                rec["guide_source"] = "pec_walls"
            elif axis_closed:
                if axis_conformal:
                    rec["guide"] = float(self._domain[axis_idx])
                else:
                    rec["guide"] = float((interior_hi - interior_lo) * grid.dx)
                rec["guide_source"] = "domain_faces"
            # else: guide stays == rasterized aperture, source "aperture".
        return out

    def _check_waveguide_port_aperture_snap(self, grid, pec_np) -> None:
        """Warn when the DECLARED port width is not what the grid rasterizes.

        Issue #738 (family #737). Fires on exactly one condition — the
        declared width differs from the span the port's grid slice
        actually covers, ``(hi_idx - lo_idx - 1) * dx``. Both branches of
        :meth:`_range_to_slice` are covered by that single comparison:

        - explicit range: the endpoints round to the nearest node, so a
          declared 22.860 mm becomes a 22.000 mm aperture at dx = 2 mm;
        - ``value_range is None``: the slice spans ``(pad, n - pad)`` but
          the reported span is ``domain_max``, so the solve's mode
          template gets the DECLARED number while the grid covers
          ``(n_interior - 1) * dx`` — issue #729 site 2, still open. The
          message names it, keyed on the None branch itself (``rec
          ["explicit"]``), not inferred from a width comparison.

        It does NOT fire on ``declared != guide``: a port whose aperture
        is narrower than the guide it sits in is the normal sub-aperture
        pattern (``tests/test_waveguide_port_reference_sims.py``,
        ``tests/test_api.py``, ``tests/test_distributed.py``) and nothing
        snapped there.
        """
        import warnings as _w

        for entry in self._waveguide_ports:
            spans = self._port_transverse_spans(entry, grid, pec_np)
            for axis_name in sorted(spans):
                rec = spans[axis_name]
                declared = rec["declared"]
                if rec["aperture"] is None:
                    _w.warn(
                        PreflightWarning(
                            f"Waveguide port '{entry.name}': declared "
                            f"{axis_name}-width {declared * 1e3:.4f} mm does "
                            f"not rasterize to a valid aperture on this grid "
                            f"(dx={grid.dx * 1e3:.4f} mm): {rec['error']}. "
                            f"This range is rejected by the compiler — the "
                            f"run would fail before stepping.",
                            code="port_aperture_unrasterizable",
                            source="_check_waveguide_port_aperture_snap",
                            severity="error",
                        ),
                        stacklevel=4,
                    )
                    continue
                rasterized = rec["rasterized"]
                if abs(declared - rasterized) <= 1e-12:
                    continue
                note = (
                    ""
                    if rec["explicit"] else
                    " This axis has no explicit range: the None-range "
                    "branch of _range_to_slice reports the declared domain "
                    "rather than the rasterized span its own explicit "
                    "branch computes (issue #729 site 2, still open — not "
                    "a new defect)."
                )
                _w.warn(
                    PreflightWarning(
                        f"Waveguide port '{entry.name}': declared "
                        f"{axis_name}-width {declared * 1e3:.4f} mm is not "
                        f"what this grid rasterizes "
                        f"(dx={grid.dx * 1e3:.4f} mm): the port's slice "
                        f"covers {rasterized * 1e3:.4f} mm, and the solve "
                        f"builds its mode template and cutoff from "
                        f"{rec['aperture'] * 1e3:.4f} mm. Cutoffs, |S| "
                        f"references, and any analytic comparison computed "
                        f"from {declared * 1e3:.4f} mm describe a structure "
                        f"this run does not solve.{note} Choose dx so it "
                        f"divides the declared width, or declare the width "
                        f"the grid can represent.",
                        code="port_aperture_snap",
                        source="_check_waveguide_port_aperture_snap",
                    ),
                    stacklevel=4,
                )

    def _emit_waveguide_port_cutoff_findings(
        self, entry, a_ap, b_ap, a_gd, b_gd, guide_label,
    ) -> None:
        """The three cutoff findings for one waveguide port.

        Shared by both lanes of :meth:`_check_waveguide_port_evanescent`
        (uniform grid: rasterized aperture + measured guide; non-uniform
        grid: declared geometry for both), so the two cannot drift and
        both carry ``source="_check_waveguide_port_evanescent"`` because
        that IS the check they belong to. Issue #738 review: the NU lane
        used to be a verbatim copy of this body.

        ``a_ap``/``b_ap`` set the #150 LOWER bounds — the dimensions the
        solve builds ``WaveguidePort.a``/``.b`` from. ``a_gd``/``b_gd``
        set the 0.90 x fc_next margin heuristic — the guide that decides
        which higher-order modes exist. ``guide_label`` is the
        human-readable provenance of the latter.
        """
        import warnings as _w

        m0, n0 = entry.mode

        def _fc_ap(m, n, _a=a_ap, _b=b_ap):
            return (C0 / 2.0) * math.sqrt((m / _a) ** 2 + (n / _b) ** 2)

        def _fc_gd(m, n, _a=a_gd, _b=b_gd):
            return (C0 / 2.0) * math.sqrt((m / _a) ** 2 + (n / _b) ** 2)

        fc_excited = _fc_ap(m0, n0)

        # --- LOWER bound (issue #150): source center / measurement bins
        # at or below the excited mode's own cutoff. Below fc the launch
        # is evanescent and near-cutoff content crawls at vanishing group
        # velocity: the extracted S is junk that GROWS with n_steps (the
        # in-band incident reference sits in the source spectral tail),
        # and below-cutoff DFT bins additionally NaN the gradient.
        if entry.freqs is not None:
            f_arr = np.asarray(entry.freqs, dtype=float)
            f_min = float(f_arr.min())
            band_center = float((f_arr.min() + f_arr.max()) / 2.0)
        else:
            f_min = None
            band_center = self._freq_max / 2.0
        f0_resolved = entry.f0 if entry.f0 is not None else band_center
        if f0_resolved <= fc_excited:
            _w.warn(
                PreflightWarning(
                    f"Waveguide port '{entry.name}': source center "
                    f"f0={f0_resolved / 1e9:.3f} GHz is at or below the "
                    f"{entry.mode_type}{m0}{n0} cutoff "
                    f"fc={fc_excited / 1e9:.3f} GHz"
                    f"{' (defaulted from the measurement band)' if entry.f0 is None else ''}. "
                    f"The launch is evanescent — extracted S-parameters are "
                    f"physically meaningless and grow with n_steps. Set "
                    f"f0 well above fc (e.g. the center of the measurement "
                    f"band).",
                    code="port_source_below_cutoff",
                    source="_check_waveguide_port_evanescent",
                ),
                stacklevel=4,
            )
        if f_min is not None and f_min <= fc_excited:
            _w.warn(
                PreflightWarning(
                    f"Waveguide port '{entry.name}': minimum measurement "
                    f"frequency {f_min / 1e9:.3f} GHz is at or below the "
                    f"{entry.mode_type}{m0}{n0} cutoff "
                    f"fc={fc_excited / 1e9:.3f} GHz. Below-cutoff bins "
                    f"produce junk S-parameters and NaN gradients under "
                    f"jax.grad. Restrict freqs to f > fc.",
                    code="port_freqs_below_cutoff",
                    source="_check_waveguide_port_evanescent",
                ),
                stacklevel=4,
            )

        fc_excited_gd = _fc_gd(m0, n0)
        fc_next = min(
            (
                _fc_gd(m, n)
                for m in range(0, 4)
                for n in range(0, 4)
                if not (m == 0 and n == 0)
                and not (m == m0 and n == n0)
                and _fc_gd(m, n) > fc_excited_gd * (1 + 1e-6)
            ),
            default=None,
        )
        if fc_next is None:
            return

        if entry.freqs is not None:
            f_check = float(np.max(np.asarray(entry.freqs)))
        else:
            f_check = self._freq_max

        threshold = 0.90 * fc_next
        if f_check > threshold:
            mn_next = min(
                ((m, n) for m in range(0, 4) for n in range(0, 4)
                 if not (m == 0 and n == 0) and not (m == m0 and n == n0)
                 and abs(_fc_gd(m, n) - fc_next) < 1.0),
                default=(None, None),
            )
            next_label = (f"TE{mn_next[0]}{mn_next[1]}"
                          if mn_next[0] is not None else "next")
            _w.warn(
                PreflightWarning(
                    f"Waveguide port '{entry.name}': max measurement frequency "
                    f"{f_check / 1e9:.3f} GHz exceeds 0.90 × fc_next="
                    f"{threshold / 1e9:.3f} GHz on the REALIZED guide "
                    f"({guide_label}) "
                    f"(fc_{entry.mode_type}{m0}{n0}={fc_excited_gd / 1e9:.3f} GHz, "
                    f"fc_{next_label}={fc_next / 1e9:.3f} GHz). "
                    f"Evanescent {next_label} contamination may exceed 1 % and "
                    f"registers as |S11| < 1 in a lossless structure. "
                    f"Restrict measurement freqs below {threshold / 1e9:.3f} GHz "
                    f"or increase port-to-obstacle distance.",
                    code="port_evanescent",
                    source="_check_waveguide_port_evanescent",
                ),
                stacklevel=4,
            )

    def _check_waveguide_port_evanescent_declared_geometry(self) -> None:
        """Pre-#738 behavior: cutoffs from the DECLARED geometry only.

        The non-uniform-mesh branch of
        :meth:`_check_waveguide_port_evanescent` — disclosed, unmeasured
        non-regression, not this issue's fix surface. It shares
        :meth:`_emit_waveguide_port_cutoff_findings` with the uniform
        lane and only differs in what it feeds that emitter.
        """
        for entry in self._waveguide_ports:
            axis = entry.direction[1]  # 'x', 'y', or 'z'
            if axis == "x":
                dim0 = (entry.y_range[1] - entry.y_range[0]
                        if entry.y_range is not None else self._domain[1])
                dim1 = (entry.z_range[1] - entry.z_range[0]
                        if entry.z_range is not None else self._domain[2])
            elif axis == "y":
                dim0 = (entry.x_range[1] - entry.x_range[0]
                        if entry.x_range is not None else self._domain[0])
                dim1 = (entry.z_range[1] - entry.z_range[0]
                        if entry.z_range is not None else self._domain[2])
            else:
                dim0 = (entry.x_range[1] - entry.x_range[0]
                        if entry.x_range is not None else self._domain[0])
                dim1 = (entry.y_range[1] - entry.y_range[0]
                        if entry.y_range is not None else self._domain[1])

            a, b = max(dim0, dim1), min(dim0, dim1)
            if a <= 0 or b <= 0:
                continue
            self._emit_waveguide_port_cutoff_findings(
                entry, a, b, a, b,
                f"{a * 1e3:.4f} x {b * 1e3:.4f} mm, declared geometry "
                f"(non-uniform mesh)",
            )

    def _check_waveguide_port_evanescent(self) -> None:
        """Warn when measurement frequencies cross a cutoff on the
        RASTERIZED grid, not the declared geometry.

        Issue #738 (family #737): this used to derive its ``a``/``b``
        from ``entry.*_range`` / ``self._domain`` — both DECLARED
        numbers — never consulting the grid the solve actually builds.
        Lead-measured on
        ``examples/inverse_design/differentiable_s11_design.py`` at its
        then-committed dx = 2 mm (declared WR-90 a = 22.860 mm): the grid
        rasterized a 22.000 mm aperture and the solve built its mode
        template from that, while this check read 22.860 mm and printed
        "All checks passed".

        Now uses :meth:`_port_transverse_spans` for both:
        - the #150 lower-bound checks (source / measurement at-or-below
          cutoff) are evaluated on the APERTURE, since that is the
          literal dimension ``WaveguidePort.a``/``.b`` feed
          ``_compute_beta`` and the mode template — this aligns the gate
          with what is actually solved rather than loosening it; the
          aperture can snap to either side of the declared width (see
          the round-up case in
          ``tests/test_port_aperture_rasterization.py``), so this is not
          a one-directional relaxation.
        - the 0.90 x fc_next margin heuristic is evaluated on the GUIDE,
          i.e. the wall-to-wall extent measured on the assembled
          ``pec_mask`` along the port's own transverse line, because
          higher-order modes are supported by the walled guide rather
          than by the port's aperture alone. When no walls can be
          established (the transverse axis is open, or the mask is
          unavailable) ``guide`` falls back to the rasterized aperture,
          so the heuristic never asserts a guide the geometry does not
          show. A finding here is a violation of the heuristic on the
          REALIZED guide, not a claim that a higher mode is actually
          propagating.

        At f/fc_next > 0.90 the evanescent decay constant is short enough
        that the next higher mode leaks into the single-mode extractor.
        Empirically (40 mm × 20 mm guide, 74 mm port-short spacing):
          f/fc_next = 0.87 → 0.3 % contamination — acceptable for |S11| gate 0.99
          f/fc_next = 0.93 → 1.5 % contamination — registers as |S11| < 1

        Uses port.freqs (measurement freqs) when set; falls back to freq_max.

        Non-uniform mesh (``_dx_profile``/``_dy_profile``/``_dz_profile``
        set): disclosed, unmeasured non-regression — falls back to
        :meth:`_check_waveguide_port_evanescent_declared_geometry` (the
        pre-#738 declared-geometry behavior) rather than rasterizing a
        non-uniform profile here; #738 does not extend to NU.
        """
        if not self._waveguide_ports:
            return

        if (self._dx_profile is not None or self._dy_profile is not None
                or self._dz_profile is not None):
            self._check_waveguide_port_evanescent_declared_geometry()
            return

        grid = self._build_grid()
        pec_np = self._port_pec_mask(grid)
        self._check_waveguide_port_aperture_snap(grid, pec_np)

        for entry in self._waveguide_ports:
            spans = self._port_transverse_spans(entry, grid, pec_np)
            axes = sorted(spans)
            # UNRASTERIZABLE (aperture=None) must not silence the cutoff
            # checks below: fall back to the DECLARED width, which is the
            # pre-#738 behavior and is always defined. Measured
            # regression: without this fallback, the committed fixture
            # tests/test_interop_design_document.py::
            # _waveguide_with_dispersive_slab LOST its port_evanescent /
            # port_source_below_cutoff findings entirely.
            ap = [spans[ax]["aperture"] if spans[ax]["aperture"] is not None
                  else spans[ax]["declared"] for ax in axes]
            gd = [spans[ax]["guide"] if spans[ax]["guide"] is not None
                  else spans[ax]["declared"] for ax in axes]
            a_ap, b_ap = max(ap), min(ap)
            a_gd, b_gd = max(gd), min(gd)
            if a_ap <= 0 or b_ap <= 0 or a_gd <= 0 or b_gd <= 0:
                continue
            # Name where each guide dimension came from, so a reader can
            # tell a wall-measured number from an aperture fallback
            # without re-deriving it.
            guide_label = " x ".join(
                f"{(spans[ax]['guide'] if spans[ax]['guide'] is not None else spans[ax]['declared']) * 1e3:.4f} mm "
                f"({ax}, {spans[ax]['guide_source'] if spans[ax]['guide'] is not None else 'declared'})"
                for ax in sorted(
                    axes,
                    key=lambda a: -(spans[a]["guide"]
                                    if spans[a]["guide"] is not None
                                    else spans[a]["declared"]),
                )
            )
            self._emit_waveguide_port_cutoff_findings(
                entry, a_ap, b_ap, a_gd, b_gd, guide_label)

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
            Empty if no issues found.
        """
        import warnings
        issues = PreflightReport()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
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

        if strict and issues:
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

        if issues:
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
            If True, escalate findings to a raise: collect every issue, then
            raise a single ``ValueError`` listing them all (aggregate-then-raise,
            matching ``preflight(strict=True)``). The underlying
            ``NotImplementedError`` is recorded as an error-severity issue and
            re-surfaced as part of that aggregated ``ValueError`` (its exact type
            is not preserved).
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
            the historical ``list[str]``). Empty when the selected calculator is
            valid for the registered port families.  Otherwise contains
            actionable, coded issues.
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

        if include_general:
            # strict=False here: collect the general findings, then aggregate
            # everything in one raise below (don't fail-on-first).
            issues.extend(self.preflight(strict=False))

        if strict and issues:
            raise ValueError(
                f"preflight_sparameters (strict) found {len(issues)} issue(s):"
                "\n  - " + "\n  - ".join(issues)
            )

        if issues:
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
        if self._dx_profile is not None or self._dy_profile is not None:
            unsupported = []
            if normalize is not True and normalize != "flux":
                unsupported.append("normalize=True or normalize='flux' is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile) supports normalize=True or "
                    "normalize='flux' and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy profile to use the uniform lane."
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

        # CHECK 2: strict PEC intersection
        pec_entries = (
            [e for e in self._geometry if e.material_name == "pec"]
            if include_pec_overlap_error else []
        )
        for side, axis, coord, tang in faces:
            for entry in pec_entries:
                try:
                    c1, c2 = entry.shape.bounding_box()
                except (NotImplementedError, TypeError, AttributeError):
                    continue
                # Strict interior along normal axis
                if not (c1[axis] < coord < c2[axis]):
                    continue
                # Tangential overlap along the other two axes
                other = [a for a in range(3) if a != axis]
                overlap = True
                for idx, (tlo, thi) in zip(other, tang):
                    if c2[idx] <= tlo or c1[idx] >= thi:
                        overlap = False
                        break
                if overlap:
                    raise PreflightConfigError(
                        f"NTFF face {'xyz'[axis]}_{side} at {coord*1e3:.2f}mm "
                        f"intersects PEC geometry '{entry.material_name}' "
                        f"(bbox {c1}–{c2}). NTFF box must enclose all radiators "
                        f"with no PEC crossing any face. Shrink or move the NTFF box.",
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
        for entry in self._geometry:
            if entry.material_name != "pec":
                continue
            try:
                c1, c2 = entry.shape.bounding_box()
            except (NotImplementedError, TypeError, AttributeError):
                continue
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
        self._validate_cfg_graded_box_rasterization(_w)
        self._validate_cfg_subgrid_limitations(_w)
        self._validate_cfg_conformal_fine_dx(dx)
        self._validate_cfg_adi_3d_accuracy(_w)
        self._validate_cfg_lossless_resonator_in_absorber(_w)
        self._validate_cfg_waveguide_reference_plane(
            _w, cpml_thick_lo, cpml_thick_hi
        )
        self._validate_cfg_refplane_placement(_w)
        self._validate_cfg_absorber_budget_vs_grid(_w, dx)
        self._validate_cfg_campaign_statics(_w)

        self._check_waveguide_port_evanescent()
        self._check_msl_port_geometry(dx, cpml_thick_lo, cpml_thick_hi)

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
        ``forward(rlc_values_override=...)`` — tests/test_lumped_rlc_ad.py); use that
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
        # each count: coordinates are E-NODES (cell edges) since #562, and
        # ``Box.mask_on_coords`` has a THIN-SHEET branch that snaps a box no
        # thicker than its local cell onto a single nearest node (#48/#75/#371).
        # With the hand model this check diverged from the rasterizer on boxes
        # straddling a grading transition — the #325 signature it exists to
        # catch — and went SILENT on one of them (#562 review, F2). So call the
        # production path instead of imitating it: node positions through the
        # same two steps the grid builder composes, and the shape's own mask.
        # x/y are passed as single-sample arrays because only the z profile is
        # needed here; either mask branch admits the box's own midpoint, so the
        # combined mask's z profile is the z-axis mask.
        from rfx.nonuniform import node_positions_from_profile
        z_nodes = np.asarray(node_positions_from_profile(dz), dtype=np.float64)

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

    def _validate_cfg_conformal_fine_dx(self, dx: float) -> None:
        """Flag the KNOWN conformal-PEC fine-mesh NaN before it wastes a run.

        ``Boundary(conformal=True)`` / conformal faces at a min cell size
        <= ~2 mm drives the field to NaN. Root cause is a discrete-adjointness
        break (the E-update-only ``eps_eff=eps/w`` makes the update operator
        non-SPSD), NOT a CFL issue — reducing dt does not cure it, and four fix
        methods are falsified. ``normalize=False`` is NOT a safe workaround.
        Surfacing this at preflight converts a silent-NaN GPU run into an
        instant redirect. Emitted as a WARNING-severity ``PreflightWarning``
        (code ``conformal_nan``), NOT error — conformal is actively-worked and
        convergence/development tests must still RUN this config, so it must not
        hard-fail; agents gate on the code, not a hard-stop.
        """
        spec = getattr(self, "_boundary_spec", None)
        if spec is None or not hasattr(spec, "conformal_faces"):
            return
        try:
            cf = spec.conformal_faces()
        except Exception:
            return
        if not cf:
            return
        cells = [
            c for c in (
                self._dx,
                getattr(self, "_dy", None),
                getattr(self, "_dz", None),
            )
            if c
        ]
        min_cell = min(cells) if cells else dx
        if min_cell <= 2.0e-3:
            import warnings as _w
            # WARNING severity (NOT error/forbid): the conformal-fine-dx NaN is
            # a KNOWN, actively-worked bug, and convergence/development tests
            # must still be able to RUN this config — a hard-fail would block
            # the very work fixing it. Agents gate on the code (conformal_nan),
            # not on a hard-stop.
            # MAINTENANCE: delete this guard when the BCK/USC contour-FIT
            # redesign lands. The strict-xfail tracker
            # tests/test_subpixel_pec.py::test_mesh_convergence_s21_with_conformal_pec
            # will hard-fail (XPASS) to force this removal (conformal-PEC
            # known issue).
            _w.warn(
                PreflightWarning(
                    f"conformal PEC is enabled on faces {sorted(cf)} with a min "
                    f"cell size {min_cell * 1e3:.3f} mm <= 2 mm — this is a KNOWN "
                    f"NaN (discrete-adjointness break, not CFL; 4 fix methods "
                    f"falsified — tracked at https://github.com/bk-squared/rfx/issues). "
                    f"normalize=False is NOT a safe workaround at fine dx. Use "
                    f"conformal=False (staircase PEC) or a coarser mesh "
                    f"(dx > 2 mm) until the contour-FIT redesign lands.",
                    code="conformal_nan",
                    severity="warning",
                    source="_validate_cfg_conformal_fine_dx",
                ),
                stacklevel=2,
            )

    def _validate_cfg_adi_3d_accuracy(self, _w) -> None:
        """Advise on the 3D ADI large-timestep accuracy envelope (OPT-C1 fixed).

        HISTORY: until 2026-07-13 the 3D ADI path was an LOD split with
        artificial diffusion (OPT-C1) and this validator flagged it as
        KNOWN-INACCURATE unconditionally. ``adi_step_3d`` now implements the
        full Zheng–Chen–Zhang two-sub-step 3D ADI (issue #338 follow-up):
        eigenfrequency error measured 1.2% at 2x CFL on the 12^3 PEC-cavity
        adjudication test (``tests/test_review_tier1_validation_battery.py::
        test_optc1_adi_3d_cavity_eigenfrequency``, 2% gate, ~15 cells/wave).

        What remains is the honest large-dt envelope of ANY Crank–Nicolson-
        class implicit scheme: dispersion error grows ~dt^2, so at ~15
        cells per wavelength the <2% eigenfrequency envelope holds only for
        CFL factors up to ~2x (von Neumann: -1.4% at 2x, -2.8% at 3x, -6.7%
        at 5x). Runs stay unconditionally STABLE at any factor — accuracy,
        not stability, is what degrades. Advise (WARNING severity, envelope
        advisory — not an error) when ``adi_cfl_factor > 2.0`` on a 3D grid.
        The 2D TMz path (2% resonance gate at 5x CFL, and stable — verified
        bounded — well beyond) is not flagged here.
        """
        if self._solver != "adi" or self._mode != "3d":
            return
        if self._adi_cfl_factor <= 2.0:
            return
        _w.warn(
            PreflightWarning(
                f"solver='adi' with a 3D grid at adi_cfl_factor="
                f"{self._adi_cfl_factor:g}: the 3D ADI scheme is "
                f"unconditionally stable, but its dispersion error grows "
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

    def _validate_cfg_lossless_resonator_in_absorber(self, _w) -> None:
        """Warn when EVERY dielectric is perfectly lossless in an open (CPML/
        UPML) domain — design-guide Anti-Pattern #1: a lossless substrate in an
        open boundary yields an artificially infinite Q that reads as a
        plausible-but-wrong resonance (an R5 surface-metric trap detectable
        purely from setup). Deliberately narrow + single-shot to avoid noise:
        it fires only when no dielectric carries any loss, and is hedged
        because it is harmless if you are not measuring Q.
        """
        if self._boundary not in ("cpml", "upml"):
            return
        try:
            from rfx.api._spec import MATERIAL_LIBRARY
        except Exception:
            MATERIAL_LIBRARY = {}

        def _resolve(name):
            mspec = self._materials.get(name) if self._materials else None
            if mspec is not None:
                return (
                    float(getattr(mspec, "eps_r", 1.0)),
                    float(getattr(mspec, "sigma", 0.0)),
                    bool(getattr(mspec, "debye_poles", None)
                         or getattr(mspec, "lorentz_poles", None)),
                )
            lib = MATERIAL_LIBRARY.get(name)
            if isinstance(lib, dict):
                return (
                    float(lib.get("eps_r", 1.0)),
                    float(lib.get("sigma", 0.0)),
                    bool(lib.get("debye_poles") or lib.get("lorentz_poles")),
                )
            return None

        lossless_names: list[str] = []
        any_lossy_dielectric = False
        for entry in self._geometry:
            resolved = _resolve(entry.material_name)
            if resolved is None:
                continue
            eps_r, sigma, has_poles = resolved
            # Dielectric (not vacuum/air, not a conductor/PEC).
            if not (eps_r > 1.05 and sigma < 1.0):
                continue
            if sigma <= 0.0 and not has_poles:
                lossless_names.append(entry.material_name)
            else:
                any_lossy_dielectric = True

        if lossless_names and not any_lossy_dielectric:
            uniq = sorted(set(lossless_names))
            _w.warn(
                PreflightWarning(
                    f"all dielectric(s) {uniq} are perfectly lossless in an open "
                    f"({self._boundary.upper()}) domain. If you are measuring "
                    f"Q / resonance, this gives an ARTIFICIALLY infinite Q "
                    f"(design-guide Anti-Pattern #1, an R5 surface-metric trap) — "
                    f"add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. "
                    f"(Harmless if you are not measuring Q.)",
                    code="lossless_q",
                    source="_validate_cfg_lossless_resonator_in_absorber",
                ),
                stacklevel=2,
            )

    def _validate_cfg_compute_cpml_thickness(
        self, cpml_thickness: float
    ) -> tuple[list[float], list[float], set]:
        """Per-face CPML thickness (2026-04). Mirrors Grid._face_pad:
        pec_faces / pmc_faces / periodic-axis faces consume 0 cells;
        remaining faces get the axis CPML thickness (non-uniform z
        aggregates the leading dz_profile entries). Under asymmetric
        composition (half-symmetric PMC + CPML, one-sided reflector)
        the lo and hi sides of a single axis can differ — the legacy
        symmetric scalar forced both sides to the max and produced
        false positives on the reflector face.

        Issue #647: the per-face LAYER COUNT now comes from
        :meth:`_preflight_face_layers`, which reads
        ``Boundary.lo_thickness`` / ``hi_thickness`` off the normalized
        ``_boundary_spec``. Before that, every absorbing face was reported
        at the global ``cpml_layers`` budget, so
        ``z=Boundary(lo='pec', hi='cpml', hi_thickness=2)`` with
        ``cpml_layers=16`` told every consumer the z_hi absorber was
        ``16*dx`` when the grid allocates ``2*dx`` — an 8x over-report that
        biases the MSL / waveguide clearance advisories (which use the
        magnitude as a calibrated buffer, not just as a boolean).

        Returns ``(cpml_thick_lo, cpml_thick_hi, _pmc_faces_set)``.
        """
        _pmc_faces_set = set(self._boundary_spec.pmc_faces())
        _face_layers = self._preflight_face_layers()
        # ``cpml_thickness`` is the BUDGET thickness (cpml_layers * dx, or 0
        # on a non-absorbing boundary); per-cell thickness is that divided
        # by the budget layer count.
        _per_layer = (cpml_thickness / self._cpml_layers
                      if self._cpml_layers else 0.0)

        def _face_thickness(ax_idx: int, side: str) -> float:
            ax_name = "xyz"[ax_idx]
            n_face = _face_layers[f"{ax_name}_{side}"]
            if n_face <= 0:
                return 0.0
            if (ax_name == "z"
                    and self._dz_profile is not None
                    and not is_tracer(self._dz_profile)):
                # Non-uniform z aggregates real cell sizes rather than
                # n*dx. The LEADING entries are used on both sides, as
                # before -- a hi-face trailing aggregation would be more
                # faithful on a graded profile but is a separate change
                # with its own regression surface.
                n = min(n_face, len(self._dz_profile))
                return float(sum(self._dz_profile[:n]))
            return n_face * _per_layer

        cpml_thick_lo = [_face_thickness(ax, "lo") for ax in range(3)]
        cpml_thick_hi = [_face_thickness(ax, "hi") for ax in range(3)]
        return cpml_thick_lo, cpml_thick_hi, _pmc_faces_set

    def _validate_cfg_absorber_budget_vs_grid(self, _w, dx: float) -> None:
        """``cpml_layers`` wider than an axis of the grid it is applied to.

        Issue #647. ``cpml_layers`` is an ALLOCATION BUDGET shared by all
        six faces, and the CPML scratch buffers are cut from it on every
        axis — including axes that allocate no absorber at all. When the
        budget exceeds an axis's own cell count the face slices ``[:n]`` /
        ``[-n:]`` shrink to the axis while the length-``n`` coefficient
        profile does not, and the run used to die inside the scan with a
        broadcasting ``TypeError`` and NOTHING from preflight
        ("All checks passed" was the measured output on the reported
        fixture). ``rfx.boundaries.cpml`` now clamps the buffer per axis,
        which is exact — the clamped region carries no active absorber
        layers — so this is an advisory about a mis-sized number, not a
        rejection: the requested absorber is realized in full.

        Only reachable with a per-face ``BoundarySpec``. With a scalar
        ``boundary='cpml'`` every axis is padded by ``2*cpml_layers``, so
        the budget can never exceed an axis extent.

        The extent arithmetic mirrors ``rfx.grid.Grid.__init__``
        (``ceil(domain/dx) + 1 + pad_lo + pad_hi``) through
        :meth:`_preflight_face_layers`; it inherits that helper's
        waveguide-axis divergence, which makes this check UNDER-fire (never
        over-fire) on waveguide-port simulations.

        Issue #737/#742: skips any axis whose ``pad_lo`` and ``pad_hi`` are
        both 0 -- a PEC-closed or periodic-closed axis allocates no absorber
        on either face, so there is nothing for ``cpml_layers`` to exceed
        and the advisory was firing on a boundary condition it should never
        have been conditioned on. This adopts the allocation>0 convention
        already used by every OTHER consumer of :meth:`_preflight_face_layers`:
        :meth:`_validate_cfg_compute_cpml_thickness` (``if n_face <= 0:
        return 0.0``), the nonuniform z-thickness check below (``_z_layers
        > 0``, whose #647 comment states this identical rationale), and
        ``cpml_axes_eff`` in ``rfx.nonuniform`` (``if (lo + hi) > 0``) --
        this was the sole consumer that had not adopted it.
        """
        n_budget = int(self._cpml_layers or 0)
        if n_budget <= 0 or dx <= 0:
            return
        face_layers = self._preflight_face_layers()
        for ax_idx, ax_name in enumerate("xyz"):
            if ax_name == "z" and self._mode.startswith("2d"):
                continue
            extent_m = (self._domain[ax_idx] if ax_idx < len(self._domain)
                        else self._domain[-1])
            pad_lo = face_layers[f"{ax_name}_lo"]
            pad_hi = face_layers[f"{ax_name}_hi"]
            if pad_lo <= 0 and pad_hi <= 0:
                # Issue #737/#742: no allocation on either face of this
                # axis (PEC/PMC-closed or periodic-closed) -- nothing to
                # budget. See docstring for the allocation>0 precedent.
                continue
            n_cells = int(math.ceil(extent_m / dx)) + 1 + pad_lo + pad_hi
            if n_budget <= n_cells:
                continue
            axis_boundary = getattr(self._boundary_spec, ax_name)
            _w.warn(
                PreflightWarning(
                    f"cpml_layers={n_budget} exceeds the {ax_name}-axis grid "
                    f"extent ({n_cells} cells: "
                    f"ceil({_fmt_len(extent_m)}/{_fmt_len(dx)})+1 interior "
                    f"nodes + {pad_lo}/{pad_hi} absorber cells, from "
                    f"{ax_name}=Boundary(lo={axis_boundary.lo!r}, "
                    f"hi={axis_boundary.hi!r})). The absorber you asked for "
                    f"is unaffected — the layer budget is clamped to the "
                    f"axis for the CPML scratch buffers only, and the "
                    f"clamped region carries no active absorber layers "
                    f"(issue #647). It does mean the layer count was sized "
                    f"for a coarser mesh or a larger domain than this one; "
                    f"set that face's lo_thickness/hi_thickness explicitly "
                    f"if you meant a thinner absorber.",
                    code="absorber_budget_exceeds_axis",
                    source="_validate_cfg_absorber_budget_vs_grid",
                ),
                stacklevel=3,
            )

    def _preflight_face_layers(self) -> dict[str, int]:
        """Allocated absorbing layers per face — preflight's mirror of
        ``rfx.grid.Grid._face_pad`` (issue #647).

        Keyed off the normalized ``_boundary_spec``, which is correct for
        BOTH legacy scalar construction (``boundary='cpml'`` +
        ``pec_faces=`` + ``set_periodic_axes()`` are all folded into it by
        ``_build_spec_from_legacy``) and per-face ``BoundarySpec``
        construction. A face is allocated
        ``Boundary.resolved_{lo,hi}_thickness(cpml_layers)`` cells when its
        own token absorbs, and 0 otherwise — which is what makes PEC / PMC
        / periodic faces fall out without a separate rule.

        Known divergence from ``Grid._face_pad``, deliberately not
        mirrored: the grid drops non-port axes from ``cpml_axes`` when
        waveguide ports are present, so on such a simulation this
        over-reports the absorber on the non-port axes. That is the
        pre-existing behaviour every current warning is calibrated
        against; changing it belongs to a waveguide-lane change, not here.
        """
        n_default = int(self._cpml_layers or 0)
        out: dict[str, int] = {}
        spec = self._boundary_spec
        for ax_name, boundary in (("x", spec.x), ("y", spec.y),
                                  ("z", spec.z)):
            for side in ("lo", "hi"):
                face = f"{ax_name}_{side}"
                token = getattr(boundary, side)
                if token not in ("cpml", "upml"):
                    out[face] = 0
                elif ax_name == "z" and self._mode.startswith("2d"):
                    # 2D modes collapse z to a single cell with NO absorber
                    # (Grid sets pad_z_lo = pad_z_hi = 0 and strips z from
                    # cpml_axes — rfx/grid.py). Without this mirror rule the
                    # z thickness is the full cpml budget and every 2D
                    # source/probe at z=0 false-trips absorber_overlap
                    # (issue #166).
                    out[face] = 0
                else:
                    resolve = getattr(boundary, f"resolved_{side}_thickness")
                    out[face] = int(resolve(n_default))
        return out

    def _validate_cfg_pec_faces_with_finite_pec(self, _w) -> None:
        """Warn about pec_faces + finite PEC objects co-existing.

        pec_faces creates an INFINITE PEC boundary face across the whole
        domain side. Users building antennas or finite-GP structures
        often use pec_faces thinking it's a "ground plane" — but it's
        a full-domain boundary condition, not a finite structure.
        """
        if self._pec_faces and self._geometry:
            has_finite_pec = any(
                entry.material_name == "pec"
                for entry in self._geometry
            )
            if has_finite_pec:
                pec_face_list = ", ".join(sorted(self._pec_faces))
                _w.warn(
                    PreflightWarning(
                        f"pec_faces={{{pec_face_list}}} creates an INFINITE PEC "
                        f"boundary AND the geometry contains finite PEC objects. "
                        f"For antennas or finite-GP structures, the pec_faces "
                        f"boundary makes the ground plane cover the entire domain "
                        f"face, which changes the physics (cavity vs radiating "
                        f"antenna). If you need a finite ground plane, remove "
                        f"pec_faces and use an explicit PEC Box instead.",
                        code="pec_faces_finite_pec",
                        source="_validate_cfg_pec_faces_with_finite_pec",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_upml_refinement(self) -> None:
        """UPML boundary does not support subgridding/refinement."""
        if self._boundary == "upml" and self._refinement is not None:
            raise PreflightConfigError(
                "boundary='upml' does not support subgridding/refinement",
                code="upml_refinement",
                source="_validate_cfg_upml_refinement",
            )

    def _validate_cfg_upml_nonuniform_lane(self, _w) -> None:
        """Advisory: ``boundary='upml'`` + a mesh profile is refused at run.

        Same shape as ``_validate_cfg_precision_x64``'s non-uniform arm:
        the enforcement point is ``_reject_upml_on_nonuniform`` at lane
        entry (``rfx/api/_execute.py``), which ``skip_preflight=True``
        does NOT bypass. This warning exists so the coming ``ValueError``
        is explained before the run reaches it, and so a
        ``skip_preflight=True`` caller still sees the reason in the one
        place they do look. Without it ``preflight()`` printed "All checks
        passed" and ``run()`` then refused, which reads as a preflight
        that does not know what the runner will do.

        The advisory carries its BASIS, not just its verdict, so a reader
        can tell a live guard from a stale one:

        * observed — on a 4x4x3 mm ez-dipole domain, two configs identical
          apart from the mesh profile: ``apply_upml_e`` ran 1x on the
          uniform lane and 0x on the non-uniform lane, while
          ``sim._boundary`` still read ``'upml'`` afterwards;
        * mechanism — ``rfx/nonuniform.py`` picks its absorber with
          ``use_cpml = grid.cpml_layers > 0`` (line 1051) and never reads
          the boundary type; every ``apply_upml_e``/``apply_upml_h`` call
          site is in the uniform scan body in ``rfx/simulation.py``. There
          is no UPML code on that lane to reach;
        * cost — CPML and UPML differ in reflection and in how material
          inside the pad is handled, so the run was a different absorber's
          result, and any post-hoc audit of ``sim._boundary`` reported the
          absorber that never ran;
        * alternative — ``boundary='cpml'`` runs what IS implemented here;
          dropping the mesh profile(s) reaches the uniform lane, which
          does implement UPML;
        * falsifier — this guard is stale the moment ``rfx/nonuniform.py``
          dispatches its absorber on the boundary type instead of on
          ``cpml_layers``, or an ``apply_upml_*`` call site appears
          outside ``rfx/simulation.py``. Both are one grep.

        Fires on exactly the condition ``_reject_upml_on_nonuniform``
        raises on (``boundary == 'upml'`` and ``cpml_layers > 0``), plus
        the mesh-profile test that its call sites supply — so the advisory
        and the error cannot disagree about which configs are refused.
        """
        is_nonuniform = (
            self._dx_profile is not None
            or self._dy_profile is not None
            or self._dz_profile is not None
        )
        if not is_nonuniform or self._boundary != "upml":
            return
        if self._cpml_layers <= 0:
            return
        _w.warn(PreflightWarning(
            "boundary='upml' was requested with a non-uniform mesh "
            "(dx/dy/dz profile set), but the non-uniform runner has no "
            "UPML code at all: rfx/nonuniform.py selects its absorber "
            "with `use_cpml = grid.cpml_layers > 0` and never reads the "
            "boundary type, and every apply_upml_e/apply_upml_h call site "
            "is in the uniform scan body (rfx/simulation.py). Measured "
            "before this was guarded (4x4x3 mm ez-dipole, configs "
            "identical apart from the mesh profile): apply_upml_e ran 1x "
            "on the uniform lane and 0x on the non-uniform lane while "
            "sim._boundary still read 'upml', so the run was CPML and "
            "even a post-hoc audit reported the absorber that never ran "
            "(issue #680). That is not a slightly worse UPML — CPML and "
            "UPML differ in reflection and in how material inside the pad "
            "is treated. Use boundary='cpml' to run the absorber that is "
            "implemented on this lane, or drop the mesh profile(s) to "
            "reach the uniform lane, which does implement UPML. "
            "run()/forward() raise ValueError at lane entry rather than "
            "proceed; this advisory explains that error in advance and "
            "covers skip_preflight=True, which does NOT bypass the lane "
            "guard. This guard is stale if rfx/nonuniform.py ever "
            "dispatches its absorber on the boundary type, or if an "
            "apply_upml_* call site appears outside rfx/simulation.py.",
            code="upml_nonuniform_lane_unsupported",
            source="_validate_cfg_upml_nonuniform_lane",
        ))

    def _validate_cfg_floquet_nonuniform(self) -> None:
        """P1.1: Floquet + non-uniform mesh — no silent fallback allowed."""
        if self._floquet_ports and self._dz_profile is not None:
            raise PreflightConfigError(
                "Floquet ports do not support non-uniform z mesh (dz_profile). "
                "Use the uniform reference lane and set dx explicitly.",
                code="floquet_nonuniform",
                source="_validate_cfg_floquet_nonuniform",
            )

    def _validate_cfg_absorber_placement(
        self,
        _w,
        dx: float,
        cpml_thickness: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
        absorber_label: str,
    ) -> None:
        """P1.2/P1.3: Probe or source inside, or suspiciously close to, the
        absorber region.

        Issue #500: membership goes through :func:`_coord_in_absorber` —
        the requested domain ``[0, domain_extent]`` is absorber-free by
        construction (exterior padding), so a probe/port is only ever
        "inside" the absorber when its coordinate is genuinely outside
        that interval (previously this compared against an interior
        reading of the CPML thickness and false-fired on geometry
        anywhere within roughly the outer half of the thickness from an
        edge, e.g. a probe at the domain centre — verified false positive
        in #500 repro 2).

        Review finding M3 / H1: dropping the interior-frame comparison
        also removed the only proximity coverage the pre-#500 code
        happened to provide — a probe genuinely INSIDE the domain but
        right at its edge used to warn (for the wrong reason) and, after
        the #500 fix alone, went silent. Two regressions surfaced this as
        load-bearing: ``tests/test_run_preflight_parity.py`` (a probe one
        cell inside a 0.02m domain was the run()-parity fixture's only
        warning trigger) and ``tests/test_msl_internal_probe_advisories.py::
        test_user_probe_advisories_and_332_still_fire`` (the #470
        regression lock's "user probe near the x-CPML" case, node 9 of a
        pad=8 grid — one cell inside). :func:`_coord_near_absorber`
        restores this honestly: a coordinate that is interior but within
        ``_ABSORBER_PROXIMITY_CELLS`` cells of an active absorber boundary
        gets a distinct, lower-severity ``absorber_proximity`` advisory
        instead of being silently indistinguishable from a comfortably
        interior placement — fields that close to the boundary still
        carry CPML fringe/reflection error even though they are not
        literally inside the absorbing medium.
        """
        if cpml_thickness > 0:
            _internal = getattr(self, "_internal_probe_indices", frozenset())
            for _pi, pe in enumerate(self._probes):
                if _pi in _internal:
                    # Library-registered diagnostic probes (e.g. the MSL
                    # settling-witness probes, issue #470): the library
                    # placed them deliberately and must not warn about
                    # itself — that self-noise buried the genuine MSL
                    # port-clearance advisories. User probes are
                    # unaffected (precedent: passive DFT probes are
                    # excluded from the geometry-extent check, #303).
                    continue
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ax_i = min(ax, 2)
                    ct_lo = cpml_thick_lo[ax_i]
                    ct_hi = cpml_thick_hi[ax_i]
                    if _coord_in_absorber(coord, domain_extent, ct_lo, ct_hi):
                        _w.warn(
                            PreflightWarning(
                                f"Probe at {pos} is near/inside {absorber_label} region "
                                f"({absorber_label} {'xyz'[ax]}-thickness: "
                                f"lo={_fmt_len(ct_lo)}, hi={_fmt_len(ct_hi)}). "
                                f"Signal will be attenuated. Move probe to interior.",
                                code="absorber_overlap",
                                source="_validate_cfg_absorber_placement",
                            ),
                            stacklevel=3,
                        )
                        break
                    if _coord_near_absorber(coord, domain_extent, ct_lo, ct_hi, dx):
                        # Issue #510 nit 1: worded off the domain edge, not
                        # off "where the absorber begins" — the edge is
                        # exactly what _coord_near_absorber measures from
                        # (_absorber_boundary_for_axis's lo_b/hi_b), so this
                        # framing is unconditionally true. The prior wording
                        # ("within N cells of the {label} absorber") could
                        # overstate proximity to the absorbing MEDIUM itself,
                        # which (hi side only) can start up to one cell
                        # further out than the boundary this margin is
                        # measured from (see _absorber_boundary_for_axis's
                        # docstring, and nit 3 below). Review follow-up:
                        # "just past which" (not "where") the absorber is
                        # active, because _coord_in_absorber's own
                        # membership predicate is a STRICT less-than
                        # (coord < lo_b) -- the boundary coordinate itself
                        # reads as interior, so the absorber is active
                        # strictly beyond it, not exactly at it.
                        _w.warn(
                            PreflightWarning(
                                f"Probe at {pos} is within "
                                f"{_ABSORBER_PROXIMITY_CELLS} cells "
                                f"({_fmt_len(_ABSORBER_PROXIMITY_CELLS * dx)}) of the "
                                f"domain edge on the {'xyz'[ax]}-axis, just past which "
                                f"the {absorber_label} absorber is active. "
                                f"Fields there carry CPML fringe/reflection error; move "
                                f"inward for claims-bearing measurement.",
                                code="absorber_proximity",
                                source="_validate_cfg_absorber_placement",
                            ),
                            stacklevel=3,
                        )
                        break

            for pe in self._ports:
                pos = pe.position
                for ax, coord in enumerate(pos):
                    domain_extent = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                    ax_i = min(ax, 2)
                    ct_lo = cpml_thick_lo[ax_i]
                    ct_hi = cpml_thick_hi[ax_i]
                    if _coord_in_absorber(coord, domain_extent, ct_lo, ct_hi):
                        _w.warn(
                            PreflightWarning(
                                f"Source/port at {pos} is near/inside {absorber_label} region "
                                f"({absorber_label} {'xyz'[ax]}-thickness: "
                                f"lo={_fmt_len(ct_lo)}, hi={_fmt_len(ct_hi)}). "
                                f"Energy will be absorbed. Move source to interior.",
                                code="absorber_overlap",
                                source="_validate_cfg_absorber_placement",
                            ),
                            stacklevel=3,
                        )
                        break
                    if _coord_near_absorber(coord, domain_extent, ct_lo, ct_hi, dx):
                        # Issue #510 nit 1 — see the matching probe-loop
                        # comment above; same unconditionally-true
                        # domain-edge framing.
                        _w.warn(
                            PreflightWarning(
                                f"Source/port at {pos} is within "
                                f"{_ABSORBER_PROXIMITY_CELLS} cells "
                                f"({_fmt_len(_ABSORBER_PROXIMITY_CELLS * dx)}) of the "
                                f"domain edge on the {'xyz'[ax]}-axis, just past which "
                                f"the {absorber_label} absorber is active. "
                                f"Fields there carry CPML fringe/reflection error; move "
                                f"inward for claims-bearing measurement.",
                                code="absorber_proximity",
                                source="_validate_cfg_absorber_placement",
                            ),
                            stacklevel=3,
                        )
                        break

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
        # (tests/test_farfield_nonuniform.py). No guard needed.

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

    def _validate_cfg_geometry_in_cpml(
        self,
        _w,
        cpml_thickness: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
        absorber_label: str,
    ) -> None:
        """P1.9: Geometry (dielectric OR PEC) extending into CPML region.

        CPML modifies field-update equations with absorbing coefficients;
        any structure placed there is effectively eaten by the absorber
        and produces physically meaningless results (issue #61).
        Periodic axes have no CPML (see _build_grid — issue #68), so
        the per-axis thresholds above already carry `cpml_thick_xyz[ax]
        == 0` on those axes and the check naturally skips.

        Issue #500 rewrite: the pre-#500 version treated the CPML as
        occupying ``[0, thick_lo]`` / ``[d - thick_hi, d]`` *inside* the
        requested domain (an "intentional full-domain edge" heuristic
        existed specifically to claw back the resulting false positives
        on the canonical transmission-line/MSL-substrate pattern —
        ``Box((0,0,0), (LX,LY,H_SUB))``). That frame was wrong: every rfx
        grid builder pads CPML EXTERIOR to the requested domain (see
        :func:`_absorber_boundary_for_axis`), so a Box entirely within
        ``[0, d]`` can never touch the absorber regardless of how close it
        sits to an edge — the heuristic is now unnecessary rather than
        merely refined. The only genuine issue-#61 case left is a Box
        whose bounding box extends to a NEGATIVE coordinate or past
        ``domain_extent`` — i.e. literally drawn into the exterior pad.

        Issue #660 reporting change: one warning per crossed AXIS naming the
        entry count, the worst offender and its overshoot distance, instead of
        one message per geometry entry that named neither.
        """
        if cpml_thickness > 0 and self._boundary == "cpml":
            # Issue #660: collect every crossing first, warn ONCE per axis.
            # The pre-#660 loop warned inside the entry loop with a message
            # carrying only the material name and the axis, so a 61-solid CAD
            # import emitted 61 lines of which 56 were byte-identical
            # (measured), and the overshoot distance — the one number that
            # separates a one-cell rounding artefact from an 11mm
            # coordinate-origin error — was computed and thrown away. The
            # crossed boundary and the offending bbox face are printed for the
            # same reason; ``_GeometryEntry`` carries no id/name field (only
            # ``shape`` + ``material_name``, verified), so the entry is
            # identified by its index and shape type, and the full per-entry
            # index list rides in the structured finding's ``loc``.
            per_axis: dict[int, list[tuple]] = {}
            for idx, entry in enumerate(self._geometry):
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        for ax in range(min(3, len(self._domain))):
                            thick_lo = cpml_thick_lo[ax]
                            thick_hi = cpml_thick_hi[ax]
                            if thick_lo <= 0 and thick_hi <= 0:
                                continue
                            d = self._domain[ax] if ax < len(self._domain) else self._domain[-1]
                            lo_b, hi_b = _absorber_boundary_for_axis(d, thick_lo, thick_hi)
                            over_lo = (
                                lo_b - c1[ax]
                                if (lo_b is not None and c1[ax] < lo_b) else None
                            )
                            over_hi = (
                                c2[ax] - hi_b
                                if (hi_b is not None and c2[ax] > hi_b) else None
                            )
                            if over_lo is None and over_hi is None:
                                continue
                            # A bbox can cross both faces (a shape wider than
                            # the domain); report the deeper crossing.
                            if over_hi is not None and (over_lo is None or over_hi >= over_lo):
                                over, side, coord, bound = over_hi, "hi", c2[ax], hi_b
                            else:
                                over, side, coord, bound = over_lo, "lo", c1[ax], lo_b
                            per_axis.setdefault(ax, []).append((
                                over, idx, entry.material_name,
                                type(entry.shape).__name__, side, coord, bound,
                            ))
                            # One finding per entry, first crossing axis —
                            # unchanged from pre-#660.
                            break
                    except (NotImplementedError, TypeError):
                        pass

            for ax in sorted(per_axis):
                recs = per_axis[ax]
                axis = "xyz"[ax]
                over, idx, mat, kind, side, coord, bound = max(recs, key=lambda r: r[0])
                msg = (
                    f"Material '{mat}' (geometry entry #{idx}, {kind}) extends "
                    f"into CPML region along {axis}-axis: bbox {side} face at "
                    f"{_fmt_len(coord)} is {_fmt_len(over)} past the "
                    f"{axis}-{side} absorber boundary at {_fmt_len(bound)}."
                )
                if len(recs) > 1:
                    msg += (
                        f" {len(recs)} geometry entries cross the {axis}-axis "
                        f"absorber (worst shown; overshoot "
                        f"{_fmt_len(min(r[0] for r in recs))} to "
                        f"{_fmt_len(over)}); per-entry index, face and "
                        f"overshoot in this finding's loc."
                    )
                msg += (
                    f" {absorber_label} modifies field updates — geometry "
                    f"inside the absorber is physically meaningless (issue #61)."
                )
                # Per-entry detail lives here rather than in N warning lines:
                # every crossing entry's index, crossed face and overshoot.
                _w.warn(
                    PreflightWarning(
                        msg,
                        code="geometry_in_absorber",
                        loc="geometry[" + ",".join(
                            f"#{r[1]} {r[4]} {_fmt_len(r[0])}" for r in recs
                        ) + "]",
                        source="_validate_cfg_geometry_in_cpml",
                    ),
                    stacklevel=3,
                )

    def _validate_cfg_port_inside_pec(self, _w, dx: float) -> None:
        """P1.8: Port/source/probe inside PEC geometry.

        FP4 refinement (2026-05-06): tangential H is non-zero on a
        PEC surface and well-defined within a thin (≤ 1.5·dx) PEC
        sheet — for example, an MSL diagnostic Hy probe placed at
        z = h_sub + 0.5·dx (the centre of a 1-cell trace) measures
        the trace surface current and must not warn.  Inside a thick
        PEC volume H still decays to zero, so the warning still
        fires there.
        """
        # Issue #314: for a WIRE port (extent set) the V/I reference plane
        # is the MIDPOINT probe cell of the production rasterization, not
        # the start position — a midpoint landing inside PEC silently
        # corrupts S-parameters (measured on the 2-port thru fixture:
        # forward |S21| collapses to ~0.005-0.01 while the reverse channel
        # reads over-unity ~1.06-1.15, because the voltage probe samples a
        # PEC-shorted environment and the port column punches a hole in
        # the trace). Check the exact cell production will use.
        #
        # Issue #319 generalization: test EVERY rasterized extent cell, not
        # just the midpoint. A non-midpoint cell inside PEC is a DEAD cell:
        # it is shorted by the surrounding conductor, so it carries no port
        # current. Since the issue-#318 fix, dead cells are EXCLUDED from
        # the port's sigma distribution, drive injection, and Z0c wave
        # normalization (the port terminates at Z0 across its live cells;
        # pre-fix versions counted all n cells and physically terminated at
        # Z0*(n_live/n) — the issue-#313 33.3-ohm finding). This advisory
        # remains because a port extent overlapping a conductor is usually
        # a geometry mistake worth confirming. Behavior when the midpoint
        # AND another cell are dead: BOTH warnings fire (the midpoint one
        # for probe-cell corruption, the dead-extent one for the geometry/
        # normalization consequence), and the dead-extent live-cell count
        # includes every dead cell — midpoint included — because the live
        # split does not care which cell holds the probe.
        #
        # Issue #544: the dead/live classification below is derived from
        # the SAME primitive the assembler uses to compute ``n_live_lw``
        # (``_wire_port_live_cells`` against the actual assembled
        # ``pec_mask``) — not a standalone bounding-box-vs-cell-center
        # approximation, which is what this advisory used before and which
        # had DRIFTED from the assembler. The approximation compared a
        # CENTER coordinate (node position + 0.5*dx along the component
        # axis) against a closed-interval ``[lo, hi]`` PEC bounding box;
        # the real rasterization (``Box.mask``, ``rfx/geometry/csg.py``)
        # evaluates occupancy at NODE coordinates with a half-open
        # interval ``[lo, hi)`` (or the thin-sheet nearest-node rule for a
        # sub-cell-thick box) — a different reference point that can, and
        # on the #488 mixed lumped/wire<->MSL fixture did, put the "dead"
        # node one cell away from what the approximation assumed: the
        # advisory reported n_live/n=3/4 while the assembler's
        # ground-truth count (and the measured passive-port Z_in = Z0/4 =
        # 12.5 ohm, PR #543) was 4/4 — every cell live. Sharing the
        # primitive makes the two paths unable to drift again by
        # construction ON THE UNIFORM-MESH PATH — see the NU disclosure
        # immediately below; the shared primitive itself never runs on a
        # non-uniform mesh, so the invariant does not (and is not claimed
        # to) extend there.
        #
        # Known limitation (issue #544 adversarial review; same
        # non-regression disclosure pattern as the #510 review nit A —
        # grep this file for "issue #510 review nit A" to find that
        # comment; citing it by line number/direction rots as the file
        # grows, which is exactly what happened to an earlier draft of
        # this comment): ``self._build_grid()`` /
        # ``self._assemble_materials(grid)`` are unconditionally UNIFORM.
        # A ``dz_profile``/``dx_profile``/``dy_profile`` (NU) sim's REAL
        # run instead uses ``_build_nonuniform_grid()`` +
        # ``_assemble_materials_nu()`` — a different ``Grid`` class with
        # different shape/padding (measured: uniform PEC node k=[4] vs NU
        # PEC node(s) k=[4,5] on the same nominal geometry), and
        # ``_wire_port_live_cells`` cannot even run against a
        # ``NonUniformGrid`` (no ``position_to_index``). Building a
        # uniform substitute for an NU sim would therefore MISCLASSIFY
        # cells rather than reflect the real run, so this advisory does
        # NOT attempt that — it emits a "classification unavailable" note
        # (issue #544 review item 6: a silent skip here is the #303
        # silently-skipped-check-family class) instead of either guessing
        # or going fully silent. This is the SAME pre-existing scope limit
        # the OLD (pre-#544) advisory had (it was equally NU-blind:
        # ``_wire_port_cell_centers`` also calls ``self._build_grid()``)
        # — not a new limitation, just now disclosed instead of silent.
        grid = None
        pec_mask = None
        pec_entry_masks: list[tuple[str, np.ndarray]] | None = None
        materials_attempted = False
        classification_unavailable_reason: str | None = None
        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )

        def _ensure_pec_entry_masks(grid):
            # Lazy (issue #544 review item 8): only rasterize per-entry
            # masks the first time any port actually needs a conductor
            # NAME -- a dead cell to attribute (#319) or an adjacent-
            # conductor gap to attribute (#556) -- so the common clean
            # case (no dead cells, no gap anywhere) never pays for it.
            # Covers BOTH ``self._geometry`` PEC entries AND
            # ``self._thin_conductors`` PEC sheets (issue #544 review
            # item 3: thin conductors also feed the assembled
            # ``pec_mask`` -- ``rfx/api/_compile.py``:232-238 -- and were
            # missing here, which produced an empty ``dead_names`` and a
            # false ``'pec'`` fallback label for a thin-conductor-only
            # dead cell).
            nonlocal pec_entry_masks
            if pec_entry_masks is not None:
                return pec_entry_masks
            pec_entry_masks = []
            for entry in self._geometry:
                mat = self._resolve_material(entry.material_name)
                if mat.sigma < self._PEC_SIGMA_THRESHOLD:
                    continue
                pec_entry_masks.append(
                    (entry.material_name,
                     np.asarray(entry.shape.mask(grid)))
                )
            for _tc_i, tc in enumerate(self._thin_conductors):
                if not tc.is_pec:
                    continue
                pec_entry_masks.append(
                    (f"thin_conductor[{_tc_i}]",
                     np.asarray(tc.shape.mask(grid)))
                )
            return pec_entry_masks

        for pe in self._ports:
            if not getattr(pe, "extent", None):
                continue
            rasterized = self._wire_port_cell_centers(pe)
            if rasterized is None:
                continue
            centers, mid_idx = rasterized
            mid_center = centers[mid_idx]

            if not materials_attempted:
                materials_attempted = True
                if is_nonuniform:
                    classification_unavailable_reason = (
                        "non-uniform mesh (dz_profile/dx_profile/"
                        "dy_profile set) -- the shared ground-truth "
                        "primitive only covers the uniform-grid path "
                        "(issue #544)"
                    )
                else:
                    try:
                        grid = self._build_grid()
                        _, _, _, pec_mask, _, _, _ = \
                            self._assemble_materials(grid)
                    except (ValueError, TypeError, NotImplementedError,
                            KeyError, AttributeError, IndexError) as exc:
                        # Defensive (matches ``_wire_port_cell_centers``):
                        # preflight must never crash a run over a
                        # diagnostics helper -- but this tuple is
                        # DELIBERATELY NOT ``except Exception`` (CI
                        # incident on PR #555's own head: a broad
                        # ``except Exception`` here caught
                        # ``rfx.experiments.worker.RunTimedOut``, a
                        # ``TimeoutError`` subclass a SIGALRM handler
                        # raises asynchronously while this exact call is
                        # in flight -- ``execute_run`` arms the alarm
                        # BEFORE calling ``compiled.preflight()``, so a
                        # slow ``_assemble_materials`` here is squarely
                        # inside the timeout window. Swallowing it here
                        # meant the worker's top-level
                        # ``except RunTimedOut`` handler never ran, so a
                        # 1-second-timeout experiment kept simulating for
                        # its full 500k-step run instead of exiting —
                        # the worker process hung rather than dying,
                        # timing out the parent's ``subprocess.wait``
                        # (tests/test_durable_worker_lifecycle.py
                        # ::test_worker_timeout_is_durable_failed_outcome,
                        # 60s). ``rfx/api`` must not import
                        # ``rfx.experiments`` to name that exception type
                        # directly (wrong dependency direction), so the
                        # general fix is this narrow, purpose-scoped
                        # tuple: only the errors ``_build_grid``/
                        # ``_assemble_materials`` are actually documented
                        # to raise for a malformed config (bad domain,
                        # unresolved material -- ``_resolve_material``
                        # raises ``KeyError`` -- unsupported shape
                        # method). Any signal-driven exception
                        # (``TimeoutError``, ``RuntimeError``-based
                        # cancellation, ``KeyboardInterrupt``,
                        # ``MemoryError``, ...) now propagates through
                        # this advisory untouched, exactly like it did
                        # before this advisory existed.
                        grid = None
                        pec_mask = None
                        classification_unavailable_reason = str(exc)

            dead_indices: list[int] = []
            dead_names: list[str] = []
            if grid is not None and pec_mask is not None:
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
                        grid, wp, pec_mask)
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
                    # Build (or reuse) the per-entry PEC masks -- see
                    # ``_ensure_pec_entry_masks`` above for the laziness
                    # and thin-conductor rationale (issue #544).
                    _ensure_pec_entry_masks(grid)
                    for idx in dead_indices:
                        ci, cj, ck = cells[idx]
                        for name, mask_arr in pec_entry_masks:
                            if (bool(mask_arr[ci, cj, ck])
                                    and name not in dead_names):
                                dead_names.append(name)
                    if not dead_names:
                        # Defensive net only: pec_mask is exactly the OR
                        # of the per-entry masks above, so this should be
                        # unreachable -- kept honest (an explicit
                        # "unknown" label) rather than silently inventing
                        # a specific, possibly wrong material name.
                        dead_names = ["unknown"]

                # Issue #556 (D5 follow-up, #488 arc): the OPPOSITE
                # failure mode of the #314/#319 advisories above. There,
                # a port extent cell lands ON PEC (false contact / dead
                # cell). Here, the port terminates one cell SHORT of a
                # rasterized conductor: its own end cell is live (not
                # PEC), but the cell immediately beyond it along the port
                # axis IS PEC in the assembled ``pec_mask`` -- read via
                # the SAME ground-truth arrays the #544 fix reads
                # (``_wire_port_live_cells`` against the assembled mask;
                # no geometric re-derivation from shapes). On the D5
                # "end-fed trace" fixture (dx=80um, h_sub=254um,
                # h_sub/dx=3.175 in the [0.10,0.40] mixed-cell danger
                # zone) the trace's rasterization snapped to the node one
                # full cell above the wire's rasterized top, so the feed
                # never galvanically reached the conductor and coupled
                # only capacitively (measured: |S21| RISING with
                # frequency, docs/research_notes/
                # 20260728_i488_falsifier_ledger.md, D5). No cell is dead
                # in that geometry, so #314/#319 are correctly silent --
                # nothing warned.
                #
                # Gating (the mis-gated-guard class the LLM-footgun audit
                # documents): fire ONLY on the exact one-cell-short-of-
                # adjacent-PEC signature. A wire port that legitimately
                # ends in open vacuum (dipole feed, nothing PEC in the
                # adjacent cell) has no adjacent-PEC hit and stays
                # silent; a port whose end cell IS on/in PEC has galvanic
                # contact (and #319 above already flags it as a dead
                # cell), so the live-end-cell requirement keeps this
                # silent there. BOTH ends of the extent are candidates
                # and both are checked (a port can miss a trace at its
                # far end or a ground plane at its near end). Pure numpy
                # reads of already-built arrays -- no new exception-prone
                # calls inside a try (PR #555 lesson: nothing here may
                # swallow an async worker-timeout exception).
                if cells and live_flags:
                    mask_np = np.asarray(pec_mask)
                    axis_letter = "xyz"[axis]
                    d_ax = (grid.dx, getattr(grid, "dy", grid.dx),
                            getattr(grid, "dz", grid.dx))[axis]
                    for end_idx, step in ((0, -1), (len(cells) - 1, +1)):
                        # Degenerate 1-cell rasterization: both loop
                        # iterations share end_idx 0 but probe DIFFERENT
                        # neighbor cells (nb-1 vs nb+1), so no duplicate
                        # warning is possible by construction.
                        if not live_flags[end_idx]:
                            continue
                        nb = list(cells[end_idx])
                        nb[axis] += step
                        if not (0 <= nb[axis] < mask_np.shape[axis]):
                            continue
                        if not bool(mask_np[nb[0], nb[1], nb[2]]):
                            continue
                        adj_names = [
                            name for name, mask_arr
                            in _ensure_pec_entry_masks(grid)
                            if bool(mask_arr[nb[0], nb[1], nb[2]])
                        ] or ["unknown"]
                        side = ("+" if step > 0 else "-") + axis_letter
                        _w.warn(
                            PreflightWarning(
                                f"Wire port at {pe.position} (extent "
                                f"{pe.extent}, component {pe.component}): "
                                f"its {side}-side end cell "
                                f"{tuple(cells[end_idx])} is live "
                                f"(vacuum/dielectric), but the cell "
                                f"immediately beyond it "
                                f"({tuple(nb)}) is PEC {adj_names} in "
                                f"the assembled geometry. The port "
                                f"terminates in vacuum/dielectric one "
                                f"cell (gap = 1 cell = {d_ax:g} m) short "
                                f"of a rasterized conductor, so the feed "
                                f"never galvanically reaches it and "
                                f"coupling is capacitive only (measured "
                                f"signature: |S21| rising with "
                                f"frequency; issue #556, the #488-lane "
                                f"D5 finding). Remedy: extend the port "
                                f"extent by one cell so its end lands on "
                                f"the conductor's rasterized node, or "
                                f"refine dx so the conductor interface "
                                f"aligns with a grid node (e.g. dx = h/N "
                                f"for an interface at height h).",
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
                        f"for this port -- verify manually that its "
                        f"rasterized extent does not land on PEC and "
                        f"does not stop one cell short of a conductor "
                        f"it is meant to contact.",
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
                        f"{len(dead_indices)} land inside PEC geometry "
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
                        f"move the port so none of its rasterized cells "
                        f"land on PEC (per the assembled geometry -- not "
                        f"a cell-center guess; a thin PEC sheet snaps to "
                        f"its nearest grid NODE, which can be a full cell "
                        f"away from the sheet's midpoint).",
                        code="wire_port_dead_extent_cells",
                        source="_validate_cfg_port_inside_pec",
                    ),
                    stacklevel=3,
                )

        _internal = getattr(self, "_internal_probe_indices", frozenset())
        _probe_entries = [
            pe for _pi, pe in enumerate(self._probes) if _pi not in _internal
        ]  # skip library-internal witness probes (issue #470; see
        #    _validate_cfg_absorber_placement for the rationale)
        for pe in list(self._ports) + _probe_entries:
            pos = pe.position
            component = (getattr(pe, "component", "") or "").lower()
            is_h_component = component in ("hx", "hy", "hz")
            for entry in self._geometry:
                if entry.material_name != "pec":
                    continue
                if hasattr(entry.shape, "bounding_box"):
                    try:
                        c1, c2 = entry.shape.bounding_box()
                        inside = all(c1[ax] <= pos[ax] <= c2[ax] for ax in range(3))
                        if inside:
                            pec_min_thickness = min(
                                c2[i] - c1[i] for i in range(3)
                            )
                            is_thin_pec = pec_min_thickness <= 1.5 * dx
                            if is_h_component and is_thin_pec:
                                continue
                            _w.warn(
                                PreflightWarning(
                                    f"Port/source at {pos} is inside PEC geometry "
                                    f"'{entry.material_name}'. Field will be zero. "
                                    f"Move source outside PEC.",
                                    code="port_in_pec",
                                    source="_validate_cfg_port_inside_pec",
                                ),
                                stacklevel=3,
                            )
                    except (NotImplementedError, TypeError):
                        pass

    def _wire_port_cell_centers(self, pe):
        """Per-cell physical sample centers of a wire port's rasterization.

        Mirrors the production rasterization exactly (issues #314/#319):
        ``_wire_port_cells`` on the same WirePort that
        ``forward()``/``run()`` build from this entry. Returns
        ``(centers, midpoint_index)`` where ``centers`` is one physical
        sample center per rasterized cell and ``midpoint_index`` is the
        index of the midpoint V/I probe cell, ``cells[len(cells) // 2]``
        (rfx/api/_execute.py). Returns None when the geometry cannot be
        resolved (defensive: preflight must never crash a run over a
        diagnostics helper).
        """
        try:
            from rfx.sources.sources import WirePort, _wire_port_cells
            axis = {"ex": 0, "ey": 1, "ez": 2}[pe.component]
            end = list(pe.position)
            end[axis] += pe.extent
            wp = WirePort(start=tuple(pe.position), end=tuple(end),
                          component=pe.component, impedance=pe.impedance)
            grid = self._build_grid()
            cells = _wire_port_cells(grid, wp)
            if not cells:
                return None
            d = (grid.dx, getattr(grid, "dy", grid.dx),
                 getattr(grid, "dz", grid.dx))
            pad = (getattr(grid, "pad_x_lo", 0),
                   getattr(grid, "pad_y_lo", 0),
                   getattr(grid, "pad_z_lo", 0))
            centers = []
            for cell in cells:
                # Inverse of grid.position_to_index (index = round(pos/d)
                # + pad_lo): node coordinate, plus the Yee half-cell
                # offset of the E component along its own axis (where the
                # V probe actually samples).
                pos = [(cell[ax] - pad[ax]) * d[ax] for ax in range(3)]
                pos[axis] += 0.5 * d[axis]
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
            has_adjacent_pec = False
            for apos in adj_positions:
                for entry in self._geometry:
                    if entry.material_name != "pec":
                        continue
                    if not hasattr(entry.shape, "bounding_box"):
                        continue
                    try:
                        c1, c2 = entry.shape.bounding_box()
                    except (NotImplementedError, TypeError):
                        continue
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

    def _validate_cfg_pec_boundary_open_structure(self, _w) -> None:
        """P0.4: PEC boundary on likely open structure."""
        if self._boundary == "pec" and self._ntff is not None:
            _w.warn(
                PreflightWarning(
                    "PEC boundary with NTFF far-field: PEC reflects all energy "
                    "back into domain. Use boundary='cpml' or boundary='upml' for open structures "
                    "(antennas, scatterers).",
                    code="pec_boundary_open",
                    source="_validate_cfg_pec_boundary_open_structure",
                ),
                stacklevel=3,
            )

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
        sheet goes into ``pec_mask`` and folds no sigma, so it has no sheet
        resistance to normalize. Threshold: adjacent cells differing by more
        than 10%.
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

    def _validate_cfg_thin_conductor_surface_impedance(self, _w) -> None:
        """Advisories for Leontovich (surface_impedance_f0) sheets (#669).

        Two advisory-tier checks per f0-mode thin conductor, both on
        CONCRETE values only (traced f0/sigma_bulk/thickness skip them):

        (a) ``thickness < 3 * delta(f0)`` with skin depth
            ``delta = sqrt(2/(2*pi*f0*mu0*sigma_bulk))`` — the
            thick-conductor (Leontovich) model is invalid for thin films;
            the DC sheet path (omit ``surface_impedance_f0``) is the
            correct model there.
        (b) ``|f0 - source centre| / source centre > 0.20`` — Rs is frozen
            at ``f0`` with relative band error ``|sqrt(f/f0)-1|``; a source
            band centred far from f0 makes that error claims-relevant.

        Thresholds 3x and 20% are fixed by the issue #669 implementation
        contract.
        """
        from rfx.core.yee import MU_0 as _MU0

        f0_sheets = [tc for tc in getattr(self, "_thin_conductors", ())
                     if getattr(tc, "surface_impedance_f0", None) is not None]
        if not f0_sheets:
            return

        src_f0s: list[float] = []
        for family in ("_ports", "_msl_ports", "_waveguide_ports",
                       "_coaxial_ports", "_floquet_ports"):
            for entry in getattr(self, family, ()) or ():
                wf = getattr(entry, "waveform", None)
                wf0 = getattr(wf, "f0", None)
                if wf0 is not None and not is_tracer(wf0):
                    try:
                        src_f0s.append(float(wf0))
                    except (TypeError, ValueError):
                        pass

        for i, tc in enumerate(f0_sheets):
            f0 = tc.surface_impedance_f0
            sb = tc.sigma_bulk
            t = tc.thickness
            if is_tracer(f0) or is_tracer(sb):
                continue
            f0 = float(f0)
            sb = float(sb)
            delta = math.sqrt(2.0 / (2.0 * math.pi * f0 * _MU0 * sb))
            if not is_tracer(t) and 0.0 < float(t) < 3.0 * delta:
                _w.warn(
                    PreflightWarning(
                        f"surface_impedance_f0 thin conductor #{i}: "
                        f"thickness {_fmt_len(float(t))} is below 3 skin "
                        f"depths ({_fmt_len(3.0 * delta)} at f0 = "
                        f"{f0:.4g} Hz, delta = {_fmt_len(delta)}). The "
                        f"thick-conductor (Leontovich) surface-resistance "
                        f"model is invalid for thin films — omit "
                        f"surface_impedance_f0 and use the DC sheet path "
                        f"(sigma_bulk*t/d), which is the correct model "
                        f"there.",
                        code="thin_conductor_leontovich_thin_film",
                        source="_validate_cfg_thin_conductor_surface_impedance",
                    ),
                    stacklevel=3,
                )
            for sf in src_f0s:
                if sf > 0.0 and abs(f0 - sf) / sf > 0.20:
                    _w.warn(
                        PreflightWarning(
                            f"surface_impedance_f0 thin conductor #{i}: "
                            f"f0 = {f0:.4g} Hz is more than 20% away from "
                            f"the source centre frequency {sf:.4g} Hz. Rs "
                            f"is frozen at f0 with relative band error "
                            f"|sqrt(f/f0)-1| — at the source centre that "
                            f"is {abs(math.sqrt(sf / f0) - 1.0):.1%}. Set "
                            f"surface_impedance_f0 to the band centre you "
                            f"actually analyse.",
                            code="thin_conductor_leontovich_band_offset",
                            source=(
                                "_validate_cfg_thin_conductor_"
                                "surface_impedance"),
                        ),
                        stacklevel=3,
                    )
                    break

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

    def _validate_cfg_waveguide_reference_plane(
        self,
        _w,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
    ) -> None:
        """P2.8: Waveguide-port reference plane sanity.

        The S-matrix returned by ``compute_waveguide_s_matrix`` is
        evaluated AT the reference plane (either ``entry.reference_plane``
        if user-specified, or the port's ``x_position`` by default after
        2026-04-22). The phase of reported S-params is therefore tied to
        that plane. Physical correctness requires the plane lies inside
        the simulation domain, outside the CPML absorbing region, and
        preferably inside a uniform-cross-section segment of guide so the
        modal decomposition is defined.

        P2.7 (obsolete): PMC / PEC + CPML on the same axis used to emit
        a warning for the architectural offset between the reflector
        plane and the user domain edge. The per-face allocation (2026-04) closed that gap on both
        the uniform (rfx/grid.py) and non-uniform (rfx/nonuniform.py)
        paths via per-face ``pad_{axis}_{lo,hi}`` allocation. The
        warning is retained as a no-op anchor so external references
        ("[P2.7]") don't break and as a reminder that the fix is
        regression-locked via tests/test_silent_drop_warnings.py and
        tests/test_boundary_pmc_hi_faces.py.
        """
        if self._waveguide_ports:
            axis_map = {"x": 0, "y": 1, "z": 2}
            for entry in self._waveguide_ports:
                direction = entry.direction  # e.g., "+x", "-x"
                ax_i = axis_map[direction[-1]]
                domain_ext = self._domain[ax_i]
                ct_lo = cpml_thick_lo[ax_i]
                ct_hi = cpml_thick_hi[ax_i]
                effective = (entry.reference_plane if entry.reference_plane is not None
                             else entry.x_position)
                if effective < 0 or effective > domain_ext:
                    raise PreflightConfigError(
                        f"waveguide_port reference plane = {effective:.4g} m is "
                        f"outside the {direction[-1]}-domain [0, {domain_ext:.4g}] m. "
                        f"Check x_position / reference_plane.",
                        code="waveguide_reference_plane",
                        source="_validate_cfg_waveguide_reference_plane",
                    )
                # Issue #500: this used to compare `effective` against an
                # INTERIOR reading of the CPML thickness
                # (`[0, ct_lo]`/`[domain_ext-ct_hi, domain_ext]`) —
                # verified false positive (repro 1: WR-90 ports at 20mm /
                # 70.678mm on a 90.678mm domain, comfortably interior,
                # warned anyway). The exterior-padding frame
                # (:func:`_absorber_boundary_for_axis`) makes the absorber
                # boundary exactly `0.0` / `domain_ext` — identical to the
                # hard bounds check immediately above, which already
                # raises `PreflightConfigError` for any `effective` this
                # branch could otherwise catch. So this warning is now
                # provably unreachable on both the uniform and
                # non-uniform (dz_profile) lanes; kept (routed through the
                # canonical helper, not deleted) as a documented no-op —
                # same precedent as the P2.7 anchor above — in case a
                # future change decouples the hard check from this one.
                lo_b, hi_b = _absorber_boundary_for_axis(domain_ext, ct_lo, ct_hi)
                if (lo_b is not None and effective < lo_b) or (
                    hi_b is not None and effective > hi_b
                ):
                    _w.warn(
                        PreflightWarning(
                            f"waveguide_port reference plane = {effective*1e3:.3g} mm is "
                            f"inside the CPML absorbing region along the "
                            f"{direction[-1]}-axis (CPML extent (exterior pad): "
                            f"[{-ct_lo*1e3:.3g}, 0] and "
                            f"[{domain_ext*1e3:.3g}, {(domain_ext + ct_hi)*1e3:.3g}] mm). "
                            f"S-matrix phase will be distorted by CPML stretching. "
                            f"Move x_position / reference_plane to the interior or "
                            f"reduce cpml_layers.",
                            code="waveguide_reference_plane",
                            source="_validate_cfg_waveguide_reference_plane",
                        ),
                        stacklevel=3,
                    )
                # Device overlap warning: check if any geometry box spans
                # the port's x-plane.
                if self._geometry:
                    for g in self._geometry:
                        try:
                            lo, hi = g.bounds
                        except Exception:
                            continue
                        if lo[ax_i] <= effective <= hi[ax_i]:
                            _w.warn(
                                PreflightWarning(
                                    f"waveguide_port reference plane at "
                                    f"{effective*1e3:.3g} mm intersects geometry "
                                    f"'{getattr(g, 'material', '?')}' "
                                    f"(bounds {lo[ax_i]*1e3:.3g}–{hi[ax_i]*1e3:.3g} mm "
                                    f"on {direction[-1]}). Modal decomposition "
                                    f"assumes a uniform cross-section at the port "
                                    f"plane; reported S-params will mix modes. Move "
                                    f"the reference plane into the empty-guide region.",
                                    code="waveguide_reference_plane",
                                    source="_validate_cfg_waveguide_reference_plane",
                                ),
                                stacklevel=3,
                            )
                            break

    def _msl_assemble_once(self):
        """Grid + materials + per-axis cell sizes on the grid the RUN uses,
        built ONCE per MSL preflight pass and shared by every port's
        :meth:`_msl_realized_substrate` call. None if the build raises."""
        try:
            nonuniform = any(getattr(self, a, None) is not None
                             for a in ("_dx_profile", "_dy_profile", "_dz_profile"))
            if nonuniform:
                grid = self._build_nonuniform_grid()
                from rfx.runners.nonuniform import assemble_materials_nu
                mats = assemble_materials_nu(self, grid)[0]
                sizes = (np.asarray(grid.dx_arr, dtype=float),
                         np.asarray(grid.dy_arr, dtype=float),
                         np.asarray(grid.dz, dtype=float))
            else:
                grid = self._build_grid()
                mats = self._assemble_materials(grid)[0]
                d = float(grid.dx)
                sizes = tuple(np.full(int(n), d) for n in grid.shape)
            return grid, mats, sizes, bool(nonuniform)
        except Exception:
            return None

    def _msl_realized_substrate(self, pe, inr, assembled=None):
        """Substrate under an MSL port as the RUN GRID realizes it, or None.

        Issue #752 review (#766 BLOCK): the substrate checks below used to
        derive "realized" thickness as ``n_cells * dx`` with the UNIFORM
        grid's scalar ``dx``. On a ``dz_profile`` simulation that is not a
        thickness at all -- it asserted "3 substrate cells = 300 um (+18%)"
        on a board ``fidelity_report()`` measured at 254.00 um on the same
        simulation. Two surfaces of one codebase disagreeing about what the
        solver built is exactly the class #752 was filed against.

        So this reads the substrate the way ``rfx.fidelity`` does: build
        the grid the run will use (non-uniform when any profile is set),
        assemble materials on it, walk the permittivity column under the
        port along the substrate-normal axis from the ground plane, and
        sum the ACTUAL cell sizes of the cells that carry the substrate's
        permittivity. Returns a dict with

          n          realized substrate cell count under the port,
          h_real     their summed thickness (m),
          frac       where the DECLARED top face sits inside the cell that
                     contains it, as a fraction of that cell (0 = on a node),
          d_iface    that cell's size (m),
          nonuniform whether a mesh profile was in force,

        or None when it cannot be derived (no substrate permittivity at the
        ground plane, or the build raised) -- callers then use the scalar
        estimate and SAY so in the message.

        ``assembled`` is the per-check cache from :meth:`_msl_assemble_once`
        (grid, materials, per-axis cell sizes, nonuniform flag) so N MSL
        ports cost one rasterization, not N (#766 review, non-blocking).

        The walk starts at the PORT'S OWN ground plane -- ``pe.position``'s
        substrate-normal coordinate -- not at the domain floor. The first
        version started at ``pad_lo`` (domain z = 0), which is only right
        when the ground plane sits on the floor; on a stripline-like or
        multi-layer stack it walked whatever dielectric lies BELOW the
        ground and reported that as "the substrate" (#766 review: a
        ground plane at 800 um over an eps_r 9 filler read back n=10,
        h_real=800 um, never seeing the 254 um / eps_r 3.66 substrate above
        it). The #752 class, reintroduced in a new form -- fixed here.
        """
        try:
            if assembled is None:
                assembled = self._msl_assemble_once()
            if assembled is None:
                return None
            grid, mats, sizes, nonuniform = assembled
            pos = tuple(float(v) for v in pe.position)
            if nonuniform:
                from rfx.nonuniform import position_to_index as _nu_p2i
                idx = list(_nu_p2i(grid, pos))
            else:
                idx = list(grid.position_to_index(pos))
            eps = np.asarray(mats.eps_r, dtype=float)
            k0 = int(idx[inr])  # the port's own ground plane, NOT pads[inr]
            sl = [int(idx[0]), int(idx[1]), int(idx[2])]
            sl[inr] = slice(k0, None)
            col = np.asarray(eps[tuple(sl)], dtype=float).ravel()
            if col.size == 0 or col[0] <= 1.0 + 1e-6:
                return None
            eps_sub = float(col[0])
            n = 0
            while n < col.size and abs(col[n] - eps_sub) <= 1e-3 * eps_sub:
                n += 1
            ax_sizes = sizes[inr][k0:]
            h_real = float(np.sum(ax_sizes[:n]))
            nodes = np.concatenate([[0.0], np.cumsum(ax_sizes)])
            h_sub = float(pe.height)
            k = int(np.searchsorted(nodes, h_sub, side="right") - 1)
            k = min(max(k, 0), len(ax_sizes) - 1)
            d_iface = float(ax_sizes[k])
            frac = (h_sub - float(nodes[k])) / d_iface if d_iface > 0 else 0.0
            if frac > 1.0 - 1e-9:  # numerical dust just below the next node
                frac = 0.0
            return dict(n=int(n), h_real=h_real, frac=float(frac),
                        d_iface=d_iface, nonuniform=bool(nonuniform))
        except Exception:
            return None

    def _check_msl_port_geometry(
        self,
        dx: float,
        cpml_thick_lo: list[float],
        cpml_thick_hi: list[float],
    ) -> None:
        """MSL port setup correctness checks (issue: silent Z0 / |S11| bias).

        Microstrip Z0 and |S11| are extremely sensitive to lateral box
        size and substrate resolution. Wrong setup can give 15-30% Z0
        bias or anti-convergent mesh-conv with no error message.
        Catches the common mistakes here so users find them in <1 min
        instead of after a full mesh sweep.

        Three checks per MSL port:

        1. **Lateral clearance** from trace edge to nearest absorbing
           boundary (CPML/PML) or PEC sidewall must be ≥ 2·h_sub.
           Microstrip fringing fields decay as exp(-π·d/h_sub); the
           5%-amplitude tail sits at d ≈ 0.95·h_sub. A ≥ 2·h_sub margin
           keeps Z0 bias under ~5% (verified by fixed-LY mesh-conv
           sweep, 2026-05-04 — see rfx-known-issues.md).

           Issue #500 / review finding MH2: the reference position is
           ``_absorber_boundary_for_axis``'s exterior-frame edge (y=0 /
           y=domain[1], x=0 / x=domain[0] when that face is CPML/UPML- or
           PEC/PMC-backed) PLUS an EXPLICIT, separately-calibrated buffer
           equal to the active CPML depth on that face
           (``cpml_thick_{lo,hi}`` = n_cpml·dx). The buffer is not a
           restatement of "where the absorber starts" (that question is
           answered by the helper alone, and the absorber does start
           exactly at the edge — issue #500's core finding still holds);
           it is a SEPARATE, empirically-measured MSL-specific margin.
           the maintainer's internal issue ledger (primary checkout
           only, not in this public tree), "Status 2026-05-04
           (CALIBRATED, OpenEMS-class)" entry: with the pre-calibration
           ``LY = W + 6·dx`` at dx=80µm, cpml_layers=8, "the trace ended
           up INSIDE the CPML overlap region (negative clearance)" and
           Z0 drifted UP with mesh refinement (54→60Ω) instead of
           converging to Hammerstad's 47.89Ω; the fix widened the
           required geometry to ``LY >= W + 2·(2·h_sub + 8·dx)`` — i.e.
           per-side clearance >= ``2·h_sub`` beyond a buffer of
           ``8·dx`` = ``cpml_layers·dx`` in that fixture (the "8" is
           that calibration's ``cpml_layers``, not a hardcoded constant
           — this reads it back out as ``cpml_thick_{lo,hi}`` so it
           scales with whatever ``cpml_layers`` is actually configured).
           A PEC/PMC face (``cpml_thick=0`` there) gets no buffer: the
           calibration's concern is CPML near-field stretching, which
           does not apply to a hard reflector. Dropping this buffer
           entirely (as an earlier #500 pass did) silently re-admits the
           negative-clearance configuration the 2026-05-04 calibration
           closed — measured on this repo's own
           ``tests/test_msl_port_preflight.py`` fixture: 3 advisories at
           base (pre-drop), 0 on the buffer-dropped branch.

        2. **Substrate resolution** n_z_sub = h_sub/dx ≥ 4 cells, on an
           ALIGNED mesh (h_sub/dx integer). Yee staircase at the
           dielectric interface is O(dx) (not O(dx²)) for inhomogeneous
           ε; <4 cells gives Z0 staircase error >5%. Re-verified post-
           #511/#507 by ``scripts/diagnostics/msl_z0_bias_floor_sweep.py``
           (2026-08-02, committed artifact under that directory): aligned
           dx=h_sub/{3,4,5,6} measured Z0 deviation -7.9%/-3.8%/-1.2%/+0.7%
           FROM THE DECLARED-board Hammerstad-Jensen anchor (S is
           normalized to that anchor — issue #723 — so this deviation is
           real and user-facing) — the "<5% at 4+ cells" promise holds
           when aligned.

           ISSUE #752 CORRECTION (2026-08-27): this docstring, and check
           2b below, used to also report that a misaligned mesh (h_sub/dx
           fractional part in [0.10, 0.40]) measured "+20.2%/+11.0% Z0
           bias, 2.56-2.94x worse than the aligned case" at dx=80/60µm —
           implying the misalignment class itself, independent of board
           identity, degrades extraction. That framing compared the
           misaligned run's DECLARED-board deviation against the aligned
           run's DECLARED-board deviation, but the misaligned mesh's
           half-open rasterizer rule (``rfx/geometry/csg.py``) ALSO
           thickens the realized substrate to 320µm/300µm at dx=80/60µm
           (+26%/+18% vs the declared 254µm; ``sim.fidelity_report()``
           confirms this) — a genuinely different physical board, not a
           worse extraction of the same one. Scored against the board
           each mesh point actually solves (Hammerstad-Jensen on the
           REALIZED h/W from ``fidelity_report()``; see the sibling
           ``msl_z0_bias_floor_sweep_realized_anchor.json`` artifact next
           to the pre-declared sweep JSON), the extractor tracks
           Hammerstad-Jensen to within 0.4% at EVERY point in the sweep,
           aligned or misaligned alike. The "2.56-2.94x worse" ratio and
           the "+20.2%/+11.0% Z0 bias" framing are RETRACTED as
           extractor-bias claims (the pre-declared sweep JSON and its
           as-run verdict block are left untouched — they remain the
           auditable record of what was measured; only this prose
           reading of them is corrected). See check 2b for what
           alignment advice survives on other grounds.

           The same sweep also asked whether alignment class shifts the
           |S11| floor itself, not just Z0 (issue #487). |S11|_floor
           tracks |Gamma_implied| = |(Z0-Z0_HJ)/(Z0+Z0_HJ)| within ~1.3x
           over 5 of 6 points (ratio 0.95-1.27), but BREAKS at the finest
           aligned point (h_sub/6, ratio 1.96), where Gamma_implied =
           0.0033 is nearly zero (the rasterized Z0 crosses the analytic
           anchor between n=5 and n=6) while mean|S11| stays at order
           0.006. Below that scale this sweep cannot RESOLVE whether the
           floor~=headroom*|Gamma_implied| mechanism still holds (headroom
           being that same ~0.95-1.27x ratio: floor / |Gamma_implied|),
           for three reasons, so this is reported as a resolution limit of
           the sweep, not a confirmed second mechanism: (1) Gamma_implied
           is one BAND-MEAN Z0 compared against a band-MEAN |S11|(f); the
           artifact has no per-bin Z0(f), and whenever Gamma(f) changes
           sign inside the band — exactly this case — mean|S11(f)|
           generically exceeds |Gamma(mean Z0)| by Jensen's inequality
           alone; (2) the two sides come from different estimators — the
           fitted-Z0's own honesty guard (``strict_extractor``, the
           documented +/-10% fitted-Z0 health bound) only calls Z0 healthy
           to that tolerance, and at the COARSEST MISALIGNED mesh (80um,
           not n=6) the two per-port Z0 reads — each that port's own
           max-deviation frequency bin, not the same frequency, so this
           is an order-of-magnitude reference, not a same-mesh
           measurement — differ by 0.17 ohm, already over half of the
           ENTIRE n=6 signal (Z0-Z0_HJ = 0.315 ohm). A coarser, misaligned
           mesh plausibly has MORE estimator noise than the much finer,
           aligned n=6 point, so 0.17 ohm likely OVERSTATES the true n=6
           noise floor — using it anyway is the conservative (cautious)
           choice for an argument that only needs to show noise CANNOT be
           excluded, not that it explains n=6 exactly; (3) only ONE point
           departs — h_sub/5 (Gamma_implied=0.0063) is fully consistent
           with floor=|Gamma_implied| (ratio 0.95). This single-point
           ratio breakdown is why no derived-dB formula ships from this
           sweep.

           No MEASURED-dB advisory ships either, for a separate reason:
           even where the mechanism DOES hold, the measured floors
           (-26.0 dB at 3 aligned cells, -33.0 dB at 4) are specific to
           THIS thru fixture, and the support matrix explicitly forbids
           generalizing thru/matched/notch evidence to other structures
           — quoting those numbers as an engineer-facing dB promise for
           an arbitrary MSL port would be exactly that overclaim. They
           are cited here as fixture-specific informational context only.
           The sweep JSON (``scripts/diagnostics/msl_z0_bias_floor_sweep/
           msl_z0_bias_floor_sweep.json``) is the artifact of record.

        3. **Port-to-CPML distance** in propagation direction ≥ 2·h_sub.
           Source-side CPML reflection inflates |S11| if the port is
           too close.
        """
        import warnings as _w
        if not self._msl_ports:
            return
        domain = self._domain

        # Issue #510 review (BLOCKING 2): the deepest-probe coordinate
        # used to be a pure continuous-coordinate extrapolation
        # (x_feed + offset*dx). The extractor places probes by GRID
        # INDEX, with rounding AND clamping into the in-domain range
        # (rfx.sources.msl_port._msl_x_for_index /
        # msl_probe_x_coords_n) — for a feed not exactly grid-aligned
        # that is an O(dx) model error (the same order as the 2-cell
        # absorber-proximity decision margin), and when the
        # offset+spacing ladder runs past the grid edge the continuous
        # formula names a coordinate the real extractor never visits
        # (several probes clamp onto the SAME cell). Build the real
        # grid once here — same precedent as ``_wire_port_cell_centers``
        # above, which mirrors production rasterization exactly for the
        # same reason — so every check below quotes the coordinates
        # ``compute_msl_s_matrix`` actually samples. Defensive: preflight
        # must never crash a run over a diagnostics helper, so a failed
        # grid build falls back to the pre-fix continuous formula at the
        # site below rather than raising or skipping checks 4/4a/4b.
        from rfx.sources.msl_port import (
            _MSL_AXIS_INDEX as _MSL_AX,
            msl_axis_roles as _msl_axis_roles,
            msl_port_from_entry as _msl_port_from_entry,
            msl_probe_x_coords_n as _probe_x_coords_n,
        )
        try:
            _msl_grid = self._build_grid()
        except Exception:
            _msl_grid = None

        # Issue #752 / #766 review: one rasterization for all ports.
        _msl_assembled = self._msl_assemble_once()

        for pe in self._msl_ports:
            # Issue #661: every check below runs on the port's OWN axes.
            # ``prop`` is the propagation axis (checks 3/4/4a fire along
            # it), ``width`` the trace-width axis (check 1 fires across
            # it). For a "+x" port these are x and y, i.e. the historical
            # behaviour; for a "+y" port they swap, and quoting the wrong
            # one would measure clearance against the wrong wall.
            _prop_ax, _width_ax, _norm_ax, _dir_sign = _msl_axis_roles(
                pe.direction
            )
            _ip = _MSL_AX[_prop_ax]
            _iw = _MSL_AX[_width_ax]
            _inr = _MSL_AX[_norm_ax]
            x_feed = float(pe.position[_ip])
            y_centre = float(pe.position[_iw])
            w_trace = float(pe.width)
            h_sub = float(pe.height)
            recommended = 2.0 * h_sub

            # ---- 1. Lateral (trace-width axis) clearance ----
            trace_y_lo = y_centre - w_trace / 2.0
            trace_y_hi = y_centre + w_trace / 2.0
            ly = float(domain[_iw])
            # Reference position on each y side = the exterior-frame edge
            # (_absorber_boundary_for_axis; None on an inactive/periodic
            # face, treated as the plain domain edge) PLUS the explicit
            # calibrated buffer (issue #500 / MH2 — see class docstring):
            # cpml_thick_{lo,hi} = n_cpml*dx, zero on a PEC/PMC face.
            _ly_lo_b, _ly_hi_b = _absorber_boundary_for_axis(
                ly, cpml_thick_lo[_iw], cpml_thick_hi[_iw]
            )
            y_abs_lo = (_ly_lo_b if _ly_lo_b is not None else 0.0) + cpml_thick_lo[_iw]
            y_abs_hi = (_ly_hi_b if _ly_hi_b is not None else ly) - cpml_thick_hi[_iw]
            clearance_lo = trace_y_lo - y_abs_lo
            clearance_hi = y_abs_hi - trace_y_hi
            for side, c, buf in (
                (f"−{_width_ax}", clearance_lo, cpml_thick_lo[_iw]),
                (f"+{_width_ax}", clearance_hi, cpml_thick_hi[_iw]),
            ):
                if c < recommended:
                    pct = max(0.0, (1.0 - c / recommended)) * 15.0
                    _w.warn(
                        PreflightWarning(
                            f"MSL port '{pe.name}' (trace W={w_trace*1e6:.0f}µm, "
                            f"h_sub={h_sub*1e6:.0f}µm): lateral clearance to "
                            f"{side} absorbing boundary = {c*1e6:.0f}µm "
                            f"(domain edge + {_fmt_len(buf)} calibrated CPML "
                            f"buffer) < recommended {recommended*1e6:.0f}µm "
                            f"(= 2·h_sub). Fringing field will be clipped → Z0 "
                            f"may be biased HIGH by ~{pct:.0f}%, mesh-conv may "
                            f"diverge. Increase domain {_width_ax}-extent OR "
                            f"move port further from sidewall.",
                            code="msl_port_geometry",
                            source="_check_msl_port_geometry",
                        ),
                        stacklevel=3,
                    )

            # ---- 2. Substrate cells ----
            # Issue #752 (#766 review): the cell count and any "realized"
            # thickness come from the RUN grid's assembled permittivity
            # (uniform or profiled), never from n * scalar dx -- see
            # _msl_realized_substrate. The scalar estimate is used only
            # when that cannot be derived, and the message says so.
            _real = self._msl_realized_substrate(pe, _inr, assembled=_msl_assembled)
            if _real is not None:
                n_z_sub = max(1, int(_real["n"]))
            else:
                n_z_sub = max(1, int(round(h_sub / dx)))
            if n_z_sub < 4:
                _extra = ""
                if _real is not None:
                    _h_real_um = _real["h_real"] * 1e6
                    if abs(_real["h_real"] - h_sub) > 0.005 * h_sub:
                        _pct_thick = (
                            (_h_real_um - h_sub * 1e6) / (h_sub * 1e6) * 100.0
                        )
                        _extra = (
                            f" On the grid this run uses the substrate "
                            f"actually realizes {_real['n']} cell(s) = "
                            f"{_h_real_um:.0f}µm ({_pct_thick:+.0f}% vs the "
                            f"declared {h_sub*1e6:.0f}µm) — read off the "
                            f"assembled permittivity under the port, not "
                            f"n*dx; the half-open rasterizer rule "
                            f"(rfx/geometry/csg.py) rounds a face that is "
                            f"not on a node UP to the next cell. A "
                            f"substrate-thickening effect, separate from "
                            f"staircase resolution; sim.fidelity_report() "
                            f"reports the same number, and see the "
                            f"mixed-cell-danger-zone check below."
                        )
                    _dx_note = (
                        f"base dx={dx*1e6:.0f}µm, mesh profile in force"
                        if _real["nonuniform"] else f"dx={dx*1e6:.0f}µm"
                    )
                else:
                    _frac_here = (h_sub / dx) - int(h_sub / dx)
                    if _frac_here > 1e-9:
                        _n_ceil = int(h_sub / dx) + 1
                        _h_real_um = _n_ceil * dx * 1e6
                        _pct_thick = (
                            (_h_real_um - h_sub * 1e6) / (h_sub * 1e6) * 100.0
                        )
                        _extra = (
                            f" SCALAR-dx ESTIMATE (the run grid could not "
                            f"be assembled for this check): h_sub/dx="
                            f"{h_sub/dx:.3f} is not an integer, so the "
                            f"half-open rasterizer rule would realize "
                            f"{_n_ceil} substrate cell(s) = {_h_real_um:.0f}µm "
                            f"({_pct_thick:+.0f}% vs the declared "
                            f"{h_sub*1e6:.0f}µm) on a uniform mesh; "
                            f"confirm with sim.fidelity_report()."
                        )
                    _dx_note = f"dx={dx*1e6:.0f}µm, scalar estimate"
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}': only {n_z_sub} substrate cell(s) "
                        f"in z (h_sub={h_sub*1e6:.0f}µm, {_dx_note}). "
                        f"Yee staircase at dielectric interface is O(dx) — "
                        f"Z0 staircase error >5% expected. Refine to dx ≤ "
                        f"{h_sub*1e6/4:.1f}µm (4+ substrate cells) AND keep "
                        f"h_sub/dx an integer (aligned) for <5% Z0 bias — "
                        f"measured post-#511/#507 at -3.8%/-1.2%/+0.7% for "
                        f"h_sub/4, h_sub/5, h_sub/6 vs the DECLARED-board "
                        f"Hammerstad-Jensen anchor (scripts/diagnostics/"
                        f"msl_z0_bias_floor_sweep.py).{_extra}",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 2b. Substrate-boundary cell alignment for
            # ``pec_occupancy_override`` users.  When h_sub/dx has a
            # fractional part in [0.10, 0.40], the substrate-air
            # interface lands in the lower portion of a Yee cell that
            # ALSO contains the trace at z=h_sub..h_sub+dx; the cell is
            # mixed substrate + PEC.  Hard-PEC ``Box(material="pec")``
            # avoids the specific bug below because it occupies WHOLE
            # cells (this build has no anisotropic/subpixel eps assembly
            # — rfx/api/__init__.py's Simulation docstring — so there is
            # no subpixel path for a hard PEC box to use; it simply never
            # enters ``pec_occupancy_override``). The AD-traceable
            # ``pec_occupancy_override`` path zeros the whole cell and
            # produces unphysical |S21| (cited, not remeasured on this
            # checkout: 2026-05-08, runs #563/#567: |S21|² > 1 across all
            # stub lengths at dx ∈ [75, 82]µm with h_sub=254µm; no
            # committed artifact, no regression test). Snap dx so
            # h_sub/dx is integer or its fractional part is > 0.6 to
            # stay in a safe alignment window.
            #
            # ISSUE #752 CORRECTION (2026-08-27): this check used to also
            # say Hard PEC is "NOT" exempt from Z0 bias, quoting "+20.2%
            # vs -7.9% at ~3 cells, +11.0% vs -3.8% at ~4 cells ...
            # 2.56-2.94x worse". Those four percentages are all measured
            # against the DECLARED 600/254µm board's Hammerstad-Jensen
            # anchor, but the +20.2%/+11.0% (misaligned, dx=80/60µm) rows
            # and the -7.9%/-3.8% (aligned, dx≈84.7/63.5µm) rows are NOT
            # the same physical board: the half-open rasterizer rule
            # thickens the misaligned meshes' realized substrate to
            # 320µm/300µm (+26%/+18% vs declared) while the aligned
            # meshes realize h_sub exactly. Comparing declared-board
            # deviations across different realized boards measures board
            # rasterization, not extractor bias. Scored against the board
            # each point actually solves (Hammerstad-Jensen on the
            # REALIZED h/W; see the sibling
            # ``msl_z0_bias_floor_sweep_realized_anchor.json`` next to
            # the pre-declared sweep JSON), the extractor tracks
            # Hammerstad-Jensen to within 0.4% at every one of the six
            # sweep points, aligned or misaligned. The "2.56-2.94x worse"
            # / "+20.2%/+11.0%" framing is RETRACTED as an extractor-bias
            # claim (the pre-declared JSON and its as-run verdict are
            # left untouched as the auditable record; only this reading
            # of them is corrected). What survives, on separate grounds:
            # (i) the |S21|² > 1 override risk above (cited, not
            # remeasured here), and (ii) the substrate-thickening effect
            # itself is real and measured (+26%/+18% at dx=80/60µm) — it
            # is a genuine board-fidelity change from what was declared,
            # even though it is not the "worse Z0 extraction" the old
            # text claimed. The alignment advice below is kept on those
            # two grounds, downgraded from a Z0-bias-magnitude claim.
            # Issue #752 (#766 review): the interface position and the
            # realized thickness come from the run grid (see
            # _msl_realized_substrate); on a uniform grid frac reduces to
            # the old (h_sub/dx) fractional part exactly.
            if _real is not None:
                frac = _real["frac"]
                _nu_here = _real["nonuniform"]
            else:
                frac = (h_sub / dx) - int(h_sub / dx)
                _nu_here = False
            if 0.10 <= frac <= 0.40:
                # Snap suggestions come from the SAME grid ``frac`` came
                # from (#766 review B1). They used to be derived from
                # ``int(h_sub / dx)`` on the SCALAR dx while ``frac`` came
                # from the run grid: on a mesh profile that scalar is not
                # a cell count at all, and on ANY mesh whose base dx
                # exceeds h_sub it is exactly 0 -- ``h_sub / n_below``
                # then raised ZeroDivisionError out of preflight, i.e.
                # out of run()/compute_msl_s_matrix(), aborting the solve
                # (the uniform-dx half of that crash predates this PR).
                # ``_msl_realized_substrate`` already returns the realized
                # count, so use it and treat "one cell below" as the only
                # coarser aligned option, which does not exist once the
                # substrate is down to a single cell.
                if _real is not None:
                    n_above = max(1, int(_real["n"]))
                else:
                    n_above = max(1, int(h_sub / dx) + 1)
                n_below = n_above - 1
                dx_low = h_sub / n_above                        # frac=0
                dx_high = h_sub / n_below if n_below >= 1 else None
                if _real is not None:
                    h_real_um = _real["h_real"] * 1e6
                    _iface_txt = (
                        f"the declared substrate top sits {frac:.3f} of a "
                        f"cell above the nearest mesh node (that cell is "
                        f"{_real['d_iface']*1e6:.1f}µm)"
                    )
                else:
                    h_real_um = n_above * dx * 1e6
                    _iface_txt = (
                        f"h_sub/dx = {h_sub/dx:.3f} (fractional part "
                        f"{frac:.3f}; scalar estimate, run grid unavailable)"
                    )
                pct_thick = (h_real_um - h_sub * 1e6) / (h_sub * 1e6) * 100.0
                _snap_txt = (
                    f"On a non-uniform profile, place a mesh node exactly "
                    f"at h_sub={h_sub*1e6:.1f}µm (the substrate top) instead "
                    f"of grading through it."
                    if _nu_here else
                    f"To snap onto a mesh matching the DECLARED board "
                    f"instead, set dx = {dx_low*1e6:.1f}µm (= h_sub/"
                    f"{n_above}) or {dx_high*1e6:.1f}µm "
                    f"(= h_sub/{n_below})."
                    if dx_high is not None else
                    f"To snap onto a mesh matching the DECLARED board "
                    f"instead, set dx = {dx_low*1e6:.1f}µm (= h_sub/"
                    f"{n_above}); there is no coarser aligned option — "
                    f"the substrate already realizes a single cell, and "
                    f"h_sub/dx cannot drop below 1. Check 2 above asks "
                    f"for dx ≤ {h_sub*1e6/4:.1f}µm (4+ substrate cells) "
                    f"anyway."
                )
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}': {_iface_txt} — this lands "
                        f"in the [0.10, 0.40] mixed-cell danger zone. The "
                        f"substrate-air interface bisects the same Yee cell "
                        f"that holds the trace; AD-traceable "
                        f"``pec_occupancy_override`` zeros the whole cell "
                        f"and produces unphysical |S21|² > 1 in this regime "
                        f"(cited, not remeasured on this checkout: runs "
                        f"#563/#567, 2026-05-08, dx∈[75,82]µm h_sub=254µm). "
                        f"Hard ``Box(material='pec')`` avoids that specific "
                        f"bug (this build has no subpixel eps assembly, so "
                        f"a hard PEC box never enters that path). "
                        f"Separately: the half-open rasterizer rule rounds "
                        f"that face UP, so this mesh actually realizes "
                        f"{n_above} cell(s) of substrate = {h_real_um:.0f}µm "
                        f"({pct_thick:+.0f}% THICKER than the declared "
                        f"{h_sub*1e6:.0f}µm — read off the run grid's "
                        f"assembled permittivity; sim.fidelity_report() "
                        f"reports the same number). That board-thickening, "
                        f"not extractor bias, is most of what a naive "
                        f"declared-board Z0 comparison used to attribute "
                        f"to 'misalignment' (retracted: see "
                        f"msl_z0_bias_floor_sweep_realized_anchor.json — "
                        f"the extractor tracks Hammerstad-Jensen on the "
                        f"board it actually solves to within 0.4% at every "
                        f"point in scripts/diagnostics/"
                        f"msl_z0_bias_floor_sweep.py's sweep, aligned or "
                        f"not). {_snap_txt}",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 3. Port-to-CPML distance along the PROPAGATION axis ----
            # Issue #500 / MH2: same reference as check 1 — exterior-frame
            # edge PLUS the explicit calibrated buffer. Issue #661: the
            # source-side wall is the LOW wall for a positive-going port
            # and the HIGH wall for a negative-going one, on whichever
            # axis ``direction`` names.
            _lx_lo_b, _lx_hi_b = _absorber_boundary_for_axis(
                float(domain[_ip]), cpml_thick_lo[_ip], cpml_thick_hi[_ip]
            )
            x_abs_lo = (_lx_lo_b if _lx_lo_b is not None else 0.0) + cpml_thick_lo[_ip]
            x_abs_hi = (
                (_lx_hi_b if _lx_hi_b is not None else float(domain[_ip]))
                - cpml_thick_hi[_ip]
            )
            x_clearance = (
                x_feed - x_abs_lo if _dir_sign > 0
                else x_abs_hi - x_feed
            )
            if x_clearance < recommended:
                _x_buf = (
                    cpml_thick_lo[_ip] if _dir_sign > 0
                    else cpml_thick_hi[_ip]
                )
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' at {_prop_ax}="
                        f"{x_feed*1e3:.2f}mm, "
                        f"direction={pe.direction!r}: distance to nearest "
                        f"{_prop_ax}-CPML = {x_clearance*1e6:.0f}µm (domain "
                        f"edge + {_fmt_len(_x_buf)} calibrated CPML buffer) < "
                        f"recommended {recommended*1e6:.0f}µm (= 2·h_sub). "
                        f"Source-side CPML reflection may inflate |S11|. Move "
                        f"port further from boundary OR increase domain "
                        f"{_prop_ax}-extent.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 4. Probe-to-reflector distance — standing-wave bias ----
            # The N-probe Z0 extractor in compute_msl_s_matrix assumes a
            # CLEAN travelling-wave regime at the probe locations.
            # When a strong reflector (PEC stub, open termination, mismatch)
            # sits within ≲ λ_g/4 of the probes, V_i contains substantial
            # standing-wave content and the recovered (α, γ, Z0) get
            # biased — typically reading |S11| ≪ 1 even when physics
            # demands full reflection.  Catches the cv06b-vs-Y2-demo
            # divergence (cv06b's L_LINE=30mm passes; Y2's L_LINE=5mm
            # fails by ~7 dB on |S11|@notch).
            min_probe_clear = msl_min_probe_clearance(float(self._freq_max))

            # Deepest probe position. Pre-#469 this used the legacy
            # 3-probe V₃ convention (offset + 2·spacing) which UNDERCOUNTS
            # the span for the default n_probes=5 (deepest probe sits at
            # offset + (n_probes-1)·spacing, rfx/sources/msl_port.py) —
            # the check now uses the true deepest probe (stricter/correct).
            n_off = pe.n_probe_offset if pe.n_probe_offset is not None else 5
            n_sp = pe.n_probe_spacing if pe.n_probe_spacing is not None else 3
            n_pr = getattr(pe, "n_probes", 5) or 5
            sign = float(_dir_sign)

            _mp = _msl_port_from_entry(pe)
            _probe_ladder = None
            if _msl_grid is not None:
                try:
                    _probe_ladder = _probe_x_coords_n(
                        _msl_grid, _mp, n_probes=n_pr,
                        n_offset_cells=n_off, n_spacing_cells=n_sp,
                    )
                except Exception:
                    _probe_ladder = None
            if _probe_ladder is not None:
                x_deep = _probe_ladder[-1]
                _ladder_dup_count = n_pr - len(set(_probe_ladder))
            else:
                # Fallback (issue #510 review, BLOCKING 2): grid build
                # or probe-ladder computation failed -- keep the
                # pre-fix continuous extrapolation rather than skip
                # checks 4/4a/4b outright. Degeneracy cannot be
                # detected on this path (no real ladder to inspect).
                x_deep = x_feed + sign * (n_off + (n_pr - 1) * n_sp) * dx
                _ladder_dup_count = 0

            if _ladder_dup_count > 0:
                # Issue #510 review (BLOCKING 2b): a ladder that runs
                # past the grid edge CLAMPS -- several probes land on
                # the same cell instead of spreading out, which makes
                # the N-probe least-squares fit rank-deficient. This is
                # a distinct hazard from "the deepest probe is near the
                # absorber" (4a below still fires separately on the
                # honest, clamped x_deep).
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"the {n_pr}-probe ladder (n_probe_offset={n_off}, "
                        f"n_probe_spacing={n_sp} cells) runs past the grid "
                        f"and CLAMPS — only {n_pr - _ladder_dup_count} of "
                        f"{n_pr} probes land on distinct grid cells "
                        f"({_ladder_dup_count} duplicate probe position(s): "
                        f"{tuple(round(c * 1e3, 2) for c in _probe_ladder)}mm). "
                        f"The N-probe least-squares wave-decomposition fit "
                        f"is rank-deficient on duplicated positions; "
                        f"`compute_msl_s_matrix`'s Z0/S11 extraction is "
                        f"unreliable for this port. Shorten "
                        f"n_probe_offset/n_probe_spacing or extend the "
                        f"domain so the full ladder stays in-grid.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            nearest_d, nearest_label, _unevaluated = \
                msl_nearest_downstream_reflector(
                    getattr(self, "_geometry", []),
                    x_probe=x_deep,
                    x_feed=x_feed,
                    y_feed=y_centre,
                    w_trace=w_trace,
                    dx=dx,
                    domain_y=float(domain[_iw]),
                    direction=pe.direction,
                    # Issue #685: decide conductor-ness by the SAME
                    # sigma >= threshold rule the assembler uses, and
                    # scan thin conductors, instead of matching the
                    # literal material name "pec" on Box shapes only.
                    resolve_material=self._resolve_material,
                    thin_conductors=getattr(self, "_thin_conductors", ()),
                    pec_sigma_threshold=self._PEC_SIGMA_THRESHOLD,
                )

            if _unevaluated:
                # Issue #685: this scan could not distinguish "nothing is
                # nearby" from "I could not look". Say which, rather than
                # letting an unplaceable conductor read as a clean line.
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"the downstream-reflector clearance scan could NOT "
                        f"evaluate {len(_unevaluated)} registered "
                        f"conductor(s), so a 'clear' result here is not "
                        f"evidence that the probes are clear — "
                        + "; ".join(_unevaluated)
                        + ". Give those shapes an axis-aligned bounding box "
                        "(or place the probes explicitly with "
                        "n_probe_offset) before trusting "
                        "`compute_msl_s_matrix`'s Z₀ / |S11| here.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            if nearest_d < min_probe_clear and nearest_label is not None:
                # Interval framing (issue #469): the compliant window is
                # offset ∈ [offset_min, offset_max]. INCREASING the offset
                # moves the probes TOWARD the reflector, so the pre-#469
                # "bump n_probe_offset" mitigation pointed the wrong way
                # on short feeds.
                # NOTE (issue #510 review, BLOCKING 1): this algebraic
                # inversion shares the FP-knife-edge pattern fixed for
                # 4a below via msl_absorber_compliant_offset_max's
                # walk-down search -- not converted here because
                # msl_nearest_downstream_reflector's continuous
                # PEC-box-distance predicate is a different shape (a
                # scalar clearance threshold, not the two boolean
                # membership/proximity predicates the walk-down helper
                # was built around) and does not drop into that helper
                # cleanly without its own re-derivation. Revisit with
                # the same walk-down technique if this interval is ever
                # found to mislead the same way.
                d_feed_to_refl = nearest_d + (n_off + (n_pr - 1) * n_sp) * dx
                off_max = int((d_feed_to_refl - min_probe_clear) / dx) - (n_pr - 1) * n_sp
                _hsub_cells = int(round(5.0 * h_sub / dx))
                interval_txt = (
                    f"compliant n_probe_offset interval ≈ "
                    f"[{max(3, _hsub_cells)}, {off_max}] cells"
                    if off_max >= max(3, _hsub_cells)
                    else "no compliant n_probe_offset exists on this feed "
                    "length (interval empty)"
                )
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"deepest probe at x={x_deep*1e3:.2f}mm sits "
                        f"{nearest_d*1e6:.0f}µm "
                        f"from a strong reflector ({nearest_label}); recommended "
                        f"≥ {min_probe_clear*1e6:.0f}µm "
                        f"(= λ_g/4 at f_max with ε_eff_proxy={MSL_EPS_EFF_PROXY:.1f}). "
                        f"Standing-wave content at the probes will bias "
                        f"`compute_msl_s_matrix`'s Z₀ extraction and |S11|@notch — "
                        f"physical |S11|→1 at a quarter-wave open stub may read "
                        f"as -5 to -10 dB instead of 0 dB.  Mitigation: "
                        f"extend L_LINE so the line between port and reflector "
                        f"is ≥ λ_g/2, OR set n_probe_offset inside the "
                        f"{interval_txt} (do NOT simply increase it — that "
                        f"moves the probes closer to the reflector).",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 4a. Probe SPAN vs the absorber — the deepest probe,
            # not just x_feed (issue #510). Checks 1 and 3 above measure
            # clearance AT x_feed only; the probe span can leave x_feed
            # comfortably clear of the CPML while x_deep (just computed
            # for check 4) lands inside or near it — check 4's reflector
            # scan only sees PEC geometry, not the absorber. Routed
            # through the #542 canonical membership/proximity helpers
            # directly (_coord_in_absorber / _coord_near_absorber), NOT
            # the buffered x_abs_lo/x_abs_hi built for check 3 above, so
            # the lo/hi-swap mutation falsifier
            # (tests/test_preflight_absorber_frame.py module docstring)
            # covers this comparison the same way it covers every other
            # consumer of those two helpers.
            #
            # Known limitation (issue #510 review nit A, disclosed
            # non-regression): ``_msl_grid`` is built via
            # ``self._build_grid()``, which is unconditionally UNIFORM (see
            # the grid-build comment above) -- on an x-graded mesh
            # (``dx_profile``) this can produce an observable false
            # positive, e.g. warning about x=0.08mm when the REAL,
            # NU-grid deepest probe sits at 3.24mm. Exact parity with
            # every other quantity this whole function already computes
            # off the scalar ``dx`` parameter, so not a new limitation.
            _domain_x = float(domain[_ip])
            _deep_idx = n_pr - 1
            _abs_margin = _ABSORBER_PROXIMITY_CELLS * dx
            _abs_headroom = (
                _domain_x - x_feed if _dir_sign > 0 else x_feed
            )
            _abs_hsub_cells = int(round(5.0 * h_sub / dx))
            _abs_off_lo = max(3, _abs_hsub_cells)
            if _msl_grid is not None:
                # Issue #510 review (BLOCKING 1): the advertised endpoint
                # is now VERIFIED against the real predicate via a
                # walk-down search, not computed algebraically -- see
                # msl_absorber_compliant_offset_max's docstring.
                # ``_abs_guess_hi`` is a deliberately generous, INEXACT
                # starting point (ceil, no proximity margin subtracted,
                # +4 cells of slack) -- only the walked-down RESULT
                # below is ever reported.
                _abs_guess_hi = (
                    int(math.ceil(_abs_headroom / dx)) - (n_pr - 1) * n_sp + 4
                )
                _abs_off_max = msl_absorber_compliant_offset_max(
                    _msl_grid, _mp,
                    n_probes=n_pr, n_spacing=n_sp, off_lo=_abs_off_lo,
                    domain_x=_domain_x, ct_lo=cpml_thick_lo[_ip],
                    ct_hi=cpml_thick_hi[_ip], dx=dx, guess_hi=_abs_guess_hi,
                )
            else:
                # Fallback (grid build failed): the pre-fix algebraic
                # estimate -- imprecise (issue #510 review, BLOCKING 1)
                # but better than no guidance at all.
                _abs_off_max = (
                    int((_abs_headroom - _abs_margin) / dx) - (n_pr - 1) * n_sp
                )
            _abs_interval_txt = (
                f"compliant n_probe_offset interval ≈ "
                f"[{_abs_off_lo}, {_abs_off_max}] cells"
                if _abs_off_max is not None and _abs_off_max >= _abs_off_lo
                else "no compliant n_probe_offset exists on this feed "
                "length (interval empty)"
            )
            if _coord_in_absorber(x_deep, _domain_x, cpml_thick_lo[_ip], cpml_thick_hi[_ip]):
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"probe {_deep_idx} (deepest, {_prop_ax}="
                        f"{x_deep*1e3:.2f}mm) is "
                        f"past the domain edge (domain {_prop_ax}-extent [0, "
                        f"{_domain_x*1e3:.2f}]mm) — inside the CPML absorbing "
                        f"region. The N-probe extractor's clean-travelling-"
                        f"wave assumption is void there: signal is attenuated "
                        f"and the fitted Z0/S11 are corrupted. "
                        f"{_abs_interval_txt}.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )
            elif _coord_near_absorber(
                x_deep, _domain_x, cpml_thick_lo[_ip], cpml_thick_hi[_ip], dx
            ):
                # Issue #510 review round-2 (nit B): this site was missed
                # when the "just past which" rephrasing (see the matching
                # comments at :2561/:2601, _validate_cfg_absorber_placement)
                # was applied elsewhere — same reasoning: _coord_in_absorber's
                # membership predicate is strict less-than, so the domain
                # edge coordinate itself still reads as interior; the
                # absorber is active strictly beyond it, not at it.
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"probe {_deep_idx} (deepest, {_prop_ax}="
                        f"{x_deep*1e3:.2f}mm) is "
                        f"within {_ABSORBER_PROXIMITY_CELLS} cells "
                        f"({_fmt_len(_abs_margin)}) of the domain edge, just "
                        f"past which the CPML absorber is active. Fields "
                        f"there carry CPML fringe/reflection error, biasing "
                        f"the fitted Z0/S11. {_abs_interval_txt}.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

            # ---- 4b. Probe span crossing another port's feed plane
            # (issue #510). A second port's feed is a source
            # discontinuity — check 4's reflector scan walks registered
            # PEC ``Box`` shapes only, so it cannot see it. Probes
            # sampling across it violate the N-probe extractor's
            # uniform-line assumption even with zero PEC geometry
            # nearby. Advisory tier, not an error: an intentional
            # multi-port line with internal witness probes between ports
            # is a legitimate research configuration as long as those
            # probes do not cross the OPPOSITE port's own feed. This is
            # an x-only crossing test (no y/z filtering on the other
            # port) — a deliberately simple, conservative check; two
            # independent lines that merely share an x-coordinate at
            # very different y positions would also warn here. Walks
            # only the two registries #510 named in scope -- MSL ports
            # (self._msl_ports) and lumped/wire ports (self._ports);
            # self._coaxial_ports and self._waveguide_ports also carry a
            # feed / reference plane but are out of scope here (issue
            # #510 review, disclosed rather than fixed).
            _span_lo, _span_hi = (
                (x_feed, x_deep) if _dir_sign > 0 else (x_deep, x_feed)
            )
            for _other in list(self._msl_ports) + list(self._ports):
                if _other is pe:
                    continue
                # Issue #661: compare on THIS port's propagation axis.
                _other_x = _other.position[_ip]
                if _span_lo < _other_x < _span_hi:
                    _other_name = getattr(_other, "name", None)
                    # Issue #510 review (BLOCKING 3): the previous label
                    # glued a possessive onto a parenthetical component
                    # tag and repeated the crossing coordinate twice
                    # ("...port at x=6.40mm (component='ez')'s feed
                    # plane at x=6.40mm"). State the owner once, with no
                    # trailing possessive, and let "feed plane at x=..."
                    # below carry the coordinate exactly once.
                    _other_owner_txt = (
                        f"MSL port '{_other_name}'" if _other_name is not None
                        else f"the lumped/wire port (component={_other.component!r})"
                    )
                    _w.warn(
                        PreflightWarning(
                            f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                            f"probe span {_prop_ax}∈[{_span_lo*1e3:.2f}, "
                            f"{_span_hi*1e3:.2f}]mm crosses the feed plane "
                            f"of {_other_owner_txt} at {_prop_ax}="
                            f"{_other_x*1e3:.2f}mm. "
                            f"A feed is a source discontinuity the "
                            f"reflector scan above cannot see; probes "
                            f"sampling across it break the N-probe "
                            f"extractor's uniform-line assumption. If this "
                            f"crossing is intentional, verify the "
                            f"extracted Z0/S11 independently.",
                            code="msl_port_geometry",
                            source="_check_msl_port_geometry",
                        ),
                        stacklevel=3,
                    )

    # ------------------------------------------------------------------
    # Issue #703: campaign statics checks. Four failure classes a month
    # of external cross-validation hit, all detectable before the first
    # time step. Shared state (grid, coords, per-entry masks, assembly)
    # is built once in _CampaignStaticsContext; each check emits at most
    # ONE aggregated advisory (the #697 duplication lesson).
    # ------------------------------------------------------------------

    def _validate_cfg_campaign_statics(self, _w) -> None:
        """Umbrella for the four issue-#703 checks; builds the shared context.

        Skips silently when the model has no conductor at all (nothing any
        of the four checks looks at), and on a traced mesh (no concrete
        node positions — the ``_validate_cfg_graded_box_rasterization``
        precedent). On a context-build failure it says so instead of
        reading as clean: a guard that cannot evaluate the model must not
        be indistinguishable from a guard that found nothing (#685 class).
        """
        try:
            has_conductor = any(
                self._resolve_material(e.material_name).sigma
                >= self._PEC_SIGMA_THRESHOLD
                for e in self._geometry
            ) or bool(self._thin_conductors)
        except KeyError:
            return  # unresolved material name; add_box/run raise elsewhere
        if not has_conductor:
            return
        ctx = _CampaignStaticsContext(self)
        if ctx.error == "traced-mesh":
            return
        if ctx.error is not None:
            _w.warn(PreflightWarning(
                "the issue-#703 campaign statics checks (congruent-conductor "
                "rasterization parity, sheet live-edge material consistency, "
                "sheet-cavity electrical thickness, off-lattice design-edge "
                f"census) could NOT run: {ctx.error}. Their silence on this "
                "run means 'not evaluated', not 'clean' (#685 class: a guard "
                "that cannot see the model must say so).",
                code="campaign_statics_unavailable",
                source="_validate_cfg_campaign_statics",
            ))
            return
        self._validate_cfg_congruent_rasterization_parity(_w, ctx)
        self._validate_cfg_sheet_live_edge_materials(_w, ctx)
        self._validate_cfg_sheet_cavity_thickness(_w, ctx)
        self._validate_cfg_off_lattice_design_edges(_w, ctx)

    @staticmethod
    def _congruence_origin_shift(ctx, members, counts):
        """Predict the best lattice-origin slide for a flagged group.

        Uniform lane only (a per-axis "origin shift" is well-defined only
        when the spacing is one number). Candidates are the shifts that
        snap a member's lo face onto a node, plus the shifts that snap
        each pairwise symmetry plane onto a node or half-node (a mirror
        pair rasterizes symmetrically when its mirror plane sits on a node
        or half-node). Each candidate is scored by RE-RASTERIZING the
        members through the production ``mask_on_coords`` on shifted node
        coordinates — no second copy of the occupancy rule.

        Restricted to groups whose members are all analytic Boxes: the
        scoring re-rasterizes every member once per candidate, and for a
        shape whose occupancy is a point-in-mesh query that is a preflight
        that runs for minutes. A group with an inexact member gets the
        geometry-move remedy instead, and the message says which.

        Returns ``(axis_index, shift_m, predicted_spread)`` or ``None``.
        """
        if ctx.lane != "uniform":
            return None
        if not all(exact for (_i, _e, _lo, _hi, exact) in members):
            return None
        d = float(ctx.grid.dx)
        fracs = np.array([ctx.sub_lattice_offsets(lo)
                          for (_i, _e, lo, _hi, _x) in members])

        def _wrap_spread(col):
            s = np.sort(np.asarray(col))
            gaps = np.diff(np.concatenate([s, s[:1] + 1.0]))
            return 1.0 - float(np.max(gaps))

        ax = int(np.argmax([_wrap_spread(fracs[:, a]) for a in range(3)]))
        cands: set[float] = set()
        for f in fracs[:, ax]:
            cands.add(round(-float(f) * d, 15))
            cands.add(round((1.0 - float(f)) * d, 15))
        centers = [0.5 * float(lo[ax] + hi[ax])
                   for (_i, _e, lo, hi, _x) in members]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                plane = 0.5 * (centers[i] + centers[j])
                r = plane % (d / 2.0)
                cands.add(round(-r, 15))
                cands.add(round(d / 2.0 - r, 15))
        cands.discard(0.0)
        cand_list = sorted(c for c in cands if abs(c) <= d)[:16]

        base = [np.asarray(ctx.coords.x), np.asarray(ctx.coords.y),
                np.asarray(ctx.coords.z)]
        best = None
        for s in cand_list:
            shifted = list(base)
            shifted[ax] = base[ax] + np.float32(s)
            cnts = [int(np.asarray(e.shape.mask_on_coords(*shifted)).sum())
                    for (_i, e, _lo, _hi, _x) in members]
            spread = max(cnts) - min(cnts)
            if best is None or (spread, abs(s)) < (best[2], abs(best[1])):
                best = (ax, s, spread)
        if best is None or best[2] >= (max(counts) - min(counts)):
            return None
        return best

    def _validate_cfg_congruent_rasterization_parity(self, _w, ctx) -> None:
        """#703 check 1: congruent conductors must rasterize congruently.

        Groups conductor entries by congruence (same shape class, sorted
        bounding-box extents equal within
        ``_CONGRUENCE_EXTENT_QUANTUM_M`` — mirror images and right-angle
        rotations share the key by construction) and compares each
        member's rasterized cell count from the production mask path.
        A spread beyond ``_CONGRUENCE_SPREAD_TOL_CELLS`` means the lattice
        broke a symmetry the design has. Runs on BOTH the uniform and the
        non-uniform lane (the counts and offsets come from that lane's own
        builders); the origin-shift suggestion is uniform-lane only.

        The census is :meth:`_CampaignStaticsContext.congruence_entries`,
        NOT the Box-only one: a patterned metal LAYER — the incident class
        — cannot be a ``Box`` (a Box fills its clearance holes), so it
        arrives as a user-defined ``Shape`` and a Box-only census skipped
        every member of every mirror pair. For a member whose bounding box
        is not the shape, equal bounds do not PROVE congruence, so the
        message says how many members were keyed that way.
        """
        keyed, unkeyed = ctx.congruence_entries()
        n_tc = len(self._thin_conductors)
        if len(keyed) < 2:
            return
        groups: dict[tuple, list] = {}
        for i, entry, lo, hi, exact in keyed:
            ext = np.sort(hi - lo)
            key = (
                type(entry.shape).__name__,
                tuple(int(round(float(e) / _CONGRUENCE_EXTENT_QUANTUM_M))
                      for e in ext),
            )
            groups.setdefault(key, []).append((i, entry, lo, hi, exact))

        flagged = []
        n_groups = 0
        for key, members in groups.items():
            if len(members) < 2:
                continue
            n_groups += 1
            counts = [int(ctx.rasterize(e.shape).sum())
                      for (_i, e, _lo, _hi, _x) in members]
            spread = max(counts) - min(counts)
            if spread > _CONGRUENCE_SPREAD_TOL_CELLS:
                flagged.append((key, members, counts, spread))
        if not flagged:
            return

        flagged.sort(key=lambda t: -t[3])
        key, members, counts, spread = flagged[0]
        ext_key = key[1]
        smallest_ext = ext_key[0] * _CONGRUENCE_EXTENT_QUANTUM_M
        n_inexact = sum(1 for (_i, _e, _lo, _hi, x) in members if not x)
        member_desc = "; ".join(
            f"geometry[{i}] '{e.material_name}' {c} cells, "
            "lo-corner sub-lattice offsets (x,y,z)=("
            + ", ".join(f"{o:.3f}" for o in ctx.sub_lattice_offsets(lo))
            + ") cells"
            for (i, e, lo, _hi, _x), c in zip(members, counts)
        )
        inferred = (
            f"INFERRED: {n_inexact} member(s) of the worst group are keyed "
            "by a bounding box that is not the shape itself, so equal "
            "bounds bound congruence rather than proving it — read the "
            "named members before acting. " if n_inexact else "")
        shift = self._congruence_origin_shift(ctx, members, counts)
        if shift is not None:
            ax_name = "xyz"[shift[0]]
            remedy = (
                f"REMEDY: slide the lattice origin by {_fmt_len(shift[1])} "
                f"along {ax_name} (re-rasterized with that slide, the "
                f"group's spread drops to {shift[2]} cell(s)), or move the "
                "members' shared symmetry plane onto a node or half-node."
            )
        elif ctx.lane == "nonuniform":
            remedy = (
                "REMEDY: place the members at positions congruent modulo "
                "the local cell size (no origin-shift prediction on the "
                "non-uniform lane — a single per-axis slide is not "
                "well-defined when the spacing varies)."
            )
        elif n_inexact:
            remedy = (
                "REMEDY: place the members at positions congruent modulo "
                "the cell size (no origin-shift prediction for this group "
                f"— {n_inexact} member(s) are not analytic Boxes, and "
                "scoring candidate slides would re-rasterize each of them "
                "16 times inside preflight)."
            )
        else:
            remedy = (
                "REMEDY: place the members at positions congruent modulo "
                "the cell size (no candidate origin slide improved the "
                "spread — the members' offsets differ on more than one "
                "axis, so equalizing them needs a geometry move)."
            )
        _w.warn(PreflightWarning(
            f"{len(flagged)} congruent-conductor group(s) rasterize to "
            "UNEQUAL cell counts on this lattice (design-identical solids, "
            f"different meshes). Worst group ({key[0]}, sorted extents "
            + " x ".join(_fmt_len(k * _CONGRUENCE_EXTENT_QUANTUM_M)
                         for k in ext_key)
            + f"): {member_desc}. OBSERVED: cell-count spread {spread} > "
            f"tolerance {_CONGRUENCE_SPREAD_TOL_CELLS} cell (one cell along "
            f"the group's smallest extent, {_fmt_len(smallest_ext)}). WHY: "
            "congruent solids whose faces sit at different sub-cell offsets "
            "are sampled by different node sets, so the mesh invents an "
            "asymmetry the design does not have — a mirror pair whose "
            "mirror plane is off-lattice rasterizes asymmetrically with the "
            "same sign in every pair. COST (measured, issue #703): mirror "
            "pairs 173 vs 183 cells (5.6%) from a mirror plane 0.26 cells "
            "off the lattice; an A/B run pair differing ONLY by a 13µm "
            "lattice-origin slide moved |S11| up to 3.5 dB per bin and "
            f"improved every aggregate agreement metric. {remedy} "
            f"COVERAGE: examined {len(keyed)} conductor entr(y/ies) that "
            f"report a bounding box, in {n_groups} congruence group(s) of "
            f">=2 members on the {ctx.lane} lane; skipped {len(unkeyed)} "
            "conductor entr(y/ies) whose shape reports no bounding box "
            f"(no congruence key) and {n_tc} thin-conductor sheet(s) (not "
            f"congruence-grouped). {inferred}STALE IF: "
            "re-rasterizing the named members gives equal counts (spread "
            "<= tolerance), or conductors stop being sampled on the E-node "
            "lattice.",
            code="congruent_conductor_rasterization_parity",
            source="_validate_cfg_congruent_rasterization_parity",
        ))

    def _validate_cfg_sheet_live_edge_materials(self, _w, ctx) -> None:
        """#703 check 2: a node-thin sheet's live edge runs on the right material.

        For exactly the cells :func:`sheet_normal_live_axis_masks` classifies
        as carrying one live (sheet-normal) E component — the operator's own
        rule, reused, never re-copied — compare the ASSEMBLED ``eps_r`` /
        ``sigma`` against the geometry re-sampled at ``node + d/2`` along
        the live axis. On current main this passes by construction (#702:
        the assembly itself calls ``resample_sheet_node_materials``); the
        check is the regression guard for that call being dropped or moved,
        and the named coverage gap for the subgridded FINE lane, which
        still inherits the original defect.
        """
        from rfx.geometry.rasterize_grid import (
            collect_thin_conductor_sheet_inputs, periodic_flags_from_axes,
        )
        boxes, others = ctx.conductor_entries()
        node_thin = [
            (i, e) for i, e in boxes
            if len(ctx.node_thin_axes(e.shape)) == 1
        ]
        has_tc = bool(self._thin_conductors)
        refinement = getattr(self, "_refinement", None)
        if not node_thin and not has_tc:
            return

        assembled = ctx.assembled()
        if assembled is None:
            _w.warn(PreflightWarning(
                "the sheet live-edge material check could NOT run (the "
                f"production assembly failed: {ctx.assembly_error}); its "
                "silence means 'not evaluated', not 'clean' (#685 class).",
                code="campaign_statics_unavailable",
                source="_validate_cfg_sheet_live_edge_materials",
            ))
            return
        mats, _pec_mask_assembled = assembled

        # Rebuild the exact conductor union the assembly's resample saw:
        # geometry PEC entries + PEC thin sheets (NOT the returned
        # pec_mask, which may carry later injections), plus the declared
        # f0 sheets — same helper, same lane mask function.
        cond = None
        for _i, e in boxes + others:
            m = ctx.rasterize(e.shape)
            cond = m if cond is None else (cond | m)
        pec_tc_masks, f0_sheets = collect_thin_conductor_sheet_inputs(
            self._thin_conductors, ctx.rasterize)
        for m in pec_tc_masks:
            cond = np.asarray(m) if cond is None else (cond | np.asarray(m))
        periodic = periodic_flags_from_axes(
            getattr(self, "_periodic_axes", ""))

        from rfx.geometry.rasterize_grid import sheet_normal_live_axis_masks
        axis_masks = sheet_normal_live_axis_masks(
            cond, declared_sheets=f0_sheets, periodic=periodic)

        fine_clause = ""
        if refinement is not None:
            fine_clause = (
                "; the add_refinement FINE region is NOT examined and is "
                "KNOWN-UNFIXED — rfx/runners/subgridded.py rasterizes the "
                "fine region at cell centres and never calls "
                "resample_sheet_node_materials, so its sheet cells inherit "
                "the original #702 defect (that clause is stale the moment "
                "that file gains the call — one grep)"
            )

        offenders = []
        n_mismatch = 0
        n_f0_cells = 0
        if axis_masks is not None:
            if ctx.lane == "uniform":
                h = float(ctx.grid.dx) * 0.5
                half_steps = (h, h, h)
            else:
                half_steps = (jnp.asarray(ctx.grid.dx_arr) * 0.5,
                              jnp.asarray(ctx.grid.dy_arr) * 0.5,
                              jnp.asarray(ctx.grid.dz) * 0.5)
            eps_exp, sigma_exp = _resample_sheet_node_materials(
                self._geometry, self._resolve_material, ctx.coords,
                mats.eps_r, mats.sigma,
                half_steps=half_steps,
                conductor_cell_mask=cond,
                declared_sheets=f0_sheets,
                periodic=periodic,
                pec_sigma_threshold=self._PEC_SIGMA_THRESHOLD,
            )
            eps_a = np.asarray(mats.eps_r, dtype=np.float64)
            eps_e = np.asarray(eps_exp, dtype=np.float64)
            sig_a = np.asarray(mats.sigma, dtype=np.float64)
            sig_e = np.asarray(sigma_exp, dtype=np.float64)
            f0_union = np.zeros(eps_a.shape, dtype=bool)
            for m, _ax in f0_sheets:
                f0_union |= np.asarray(m)
            n_f0_cells = int(f0_union.sum())
            for a in range(3):
                live = np.asarray(axis_masks[a])
                if not live.any():
                    continue
                bad_eps = live & (
                    np.abs(eps_a - eps_e)
                    > _LIVE_EDGE_RTOL * np.maximum(np.abs(eps_e), 1.0))
                # sigma is NOT compared at f0 sheet cells: the sheet fold
                # deliberately writes sigma_eff there after the resample.
                bad_sig = live & ~f0_union & (
                    np.abs(sig_a - sig_e)
                    > _LIVE_EDGE_RTOL * np.maximum(np.abs(sig_e), 1.0))
                bad = bad_eps | bad_sig
                n_mismatch += int(bad.sum())
                for idx in np.argwhere(bad)[:3]:
                    t = tuple(int(v) for v in idx)
                    offenders.append(
                        f"cell {t} (live axis {'xyz'[a]}): assigned "
                        f"eps_r {eps_a[t]:.4g} / sigma {sig_a[t]:.4g} vs "
                        f"live-edge sample eps_r {eps_e[t]:.4g} / sigma "
                        f"{sig_e[t]:.4g}")
        if n_mismatch == 0 and not fine_clause:
            return
        if n_mismatch:
            head = (
                f"{n_mismatch} node-thin conductor cell(s) carry statics "
                "that DISAGREE with the material at their live "
                "(sheet-normal) E edge. Worst: "
                + "; ".join(offenders[:_CAMPAIGN_MAX_OFFENDERS]) + ". "
                "OBSERVED: assigned cell eps_r/sigma differ from the "
                "geometry re-sampled at node + d/2 along the live axis by "
                f"more than rtol {_LIVE_EDGE_RTOL:g}. "
            )
        else:
            head = (
                "the coarse-lane sheet live-edge material check PASSED, "
                "but this run also configures add_refinement. "
            )
        _w.warn(PreflightWarning(
            head +
            "WHY: a sub-cell conductor is registered on one node and "
            "nothing writes eps_r there, while the one E component the "
            "sheet leaves alive sits half a cell away inside the "
            "neighbouring material — the #702 class: the live edge runs on "
            "eps_r 1.0 where the physical stack has no air at all. COST "
            "(measured, #702/#703): one such cell made a cavity read 17.3% "
            "wider as a series capacitance (sum d/eps, which governs "
            "coupling across a gap far below a wavelength) and dropped the "
            "coupling capacitance 14.8%, while the phase measure "
            "(sum d*sqrt(eps)) moved only 3.2% — a reader checking phase "
            "alone calls it benign. REMEDY: this is a RASTERIZATION "
            "regression, not a modelling choice — the assembly is expected "
            "to re-sample sheet-node statics at the live edge "
            "(resample_sheet_node_materials, called from "
            "rfx/api/_compile.py and rfx/runners/nonuniform.py); do NOT "
            "paper over it with a filler dielectric box. COVERAGE: "
            f"examined the {ctx.lane} lane's assembled arrays "
            f"({len(node_thin)} node-thin conductor Box entr(y/ies), "
            f"{len(self._thin_conductors)} thin-conductor sheet(s)); sigma "
            f"not compared at {n_f0_cells} surface-impedance sheet cell(s) "
            "(the sheet fold writes sigma there on purpose)"
            + fine_clause + ". STALE IF: the resample call sites move into "
            "rasterize_geometry (then compare against that shared body), "
            "or apply_pec_mask stops leaving the sheet-normal edge alive.",
            code="sheet_live_edge_material_mismatch",
            source="_validate_cfg_sheet_live_edge_materials",
        ))

    def _validate_cfg_sheet_cavity_thickness(self, _w, ctx) -> None:
        """#703 check 3: report each sheet-bounded cavity's electrical thickness.

        For adjacent node-thin conductor sheets along an axis with
        dielectric between, compare the MESH electrical thickness
        (node-to-node, over the run's own cells and assembled ``eps_r``)
        against the PHYSICAL face-to-face stack from the geometry Box
        spans, in BOTH measures: ``sum(d/eps)`` (series capacitance) and
        ``sum(d*sqrt(eps))`` (phase length). Advisory above
        ``_CAVITY_THICKNESS_TOL`` on either — a quantified limit of the
        zero-thickness sheet model, not a defect.
        """
        boxes, _others = ctx.conductor_entries()
        # Sheet census: geometry Boxes thin along exactly one axis, plus
        # PEC thin-conductor Box sheets (same node-thin realization).
        sheet_sources = [(f"geometry[{i}]", e.shape, e.material_name)
                         for i, e in boxes]
        for ti, tc in enumerate(self._thin_conductors):
            if getattr(tc, "is_pec", False) and isinstance(tc.shape, Box):
                sheet_sources.append(
                    (f"thin_conductor[{ti}]", tc.shape, "pec"))
        sheets = []
        for label, shape, _matname in sheet_sources:
            axes = ctx.node_thin_axes(shape)
            if len(axes) != 1:
                continue
            a = axes[0]
            m = ctx.rasterize(shape)
            if not m.any():
                continue
            occ = np.flatnonzero(
                m.any(axis=tuple(x for x in range(3) if x != a)))
            if occ.size != 1:
                continue  # did not snap to a single node layer
            s_lo, s_hi = _sorted_box_corners(shape)
            own = ctx.sheet_own_cell(a, float(s_lo[a]), float(s_hi[a]),
                                     int(occ[0]))
            sheets.append(
                (label, shape, a, int(occ[0]), m.any(axis=a), own))
        if len(sheets) < 2:
            return
        assembled = ctx.assembled()
        if assembled is None:
            return  # check 2 already reported the assembly failure
        mats, pec_mask = assembled
        eps_arr = np.asarray(mats.eps_r, dtype=np.float64)
        pec_np = (np.asarray(pec_mask) if pec_mask is not None
                  else np.zeros(eps_arr.shape, dtype=bool))

        nonbox_diel = sum(
            1 for e in self._geometry
            if not isinstance(e.shape, Box)
            and self._resolve_material(e.material_name).sigma
            < self._PEC_SIGMA_THRESHOLD)

        results = []
        n_pairs = 0
        n_skipped_pec_between = 0
        lam0 = C0 / float(self._freq_max)
        for a in range(3):
            axis_sheets = sorted((s for s in sheets if s[2] == a),
                                 key=lambda s: s[3])
            for si in range(len(axis_sheets)):
                for sj in range(si + 1, len(axis_sheets)):
                    lab1, sh1, _a1, k1, foot1, own1 = axis_sheets[si]
                    lab2, sh2, _a2, k2, foot2, _own2 = axis_sheets[sj]
                    if k2 <= k1:
                        continue
                    overlap = foot1 & foot2
                    if not overlap.any():
                        continue
                    # adjacency: no third sheet node strictly between over
                    # the shared footprint
                    blocked = any(
                        k1 < s[3] < k2 and (s[4] & overlap).any()
                        for s in axis_sheets)
                    if blocked:
                        continue
                    n_pairs += 1
                    idxs = np.argwhere(overlap)
                    cen = idxs.mean(axis=0)
                    rep = idxs[int(np.argmin(
                        ((idxs - cen) ** 2).sum(axis=1)))]
                    inplane = [x for x in range(3) if x != a]
                    col = [0, 0, 0]
                    col[inplane[0]] = int(rep[0])
                    col[inplane[1]] = int(rep[1])
                    # Mesh sums: cells k1 .. k2-1, node-to-node. That
                    # span is what the fields see, INCLUDING the lower
                    # sheet's own cell when the sheet fills one:
                    # apply_pec_mask zeroes only TANGENTIAL E on a
                    # one-cell PEC sheet, so the sheet cell's
                    # normal-E edge stays live and its permittivity is
                    # in the cavity (rfx/boundaries/pec.py; pinned by
                    # test_face_registered_sheet_cell_is_a_live_edge).
                    # Face registration therefore does NOT shorten the
                    # electrical cavity — it swaps a collapsed sheet for
                    # a live vacuum gap — so the sum stays and
                    # ``own1`` is used to ATTRIBUTE the excess below,
                    # not to trim the span.
                    # A sheet puts a live edge INSIDE this cavity only
                    # when its own PEC node's cell is the cell it fills
                    # — the lower-face-registered case, ``own == k``. An
                    # upper-face-registered sheet fills cell ``k-1``
                    # while its live edge is cell ``k``: above the cavity
                    # for the upper sheet, and for the lower sheet a
                    # dielectric cell whose eps the #702 resample already
                    # takes from the live edge. And the upper sheet's own
                    # node ``k2`` is never inside ``range(k1, k2)``. So
                    # ``own1 == k1`` is the whole condition.
                    own_cell = k1 if (own1 is not None and own1 == k1) \
                        else None
                    mesh_cap = mesh_phase = 0.0
                    own_cap = own_d = 0.0
                    own_eps = None
                    pec_between = False
                    for kk in range(k1, k2):
                        col[a] = kk
                        t = tuple(col)
                        if kk > k1 and pec_np[t]:
                            pec_between = True
                            break
                        d_loc = float(ctx.spacings[a][kk])
                        ee = float(eps_arr[t])
                        mesh_cap += d_loc / ee
                        mesh_phase += d_loc * math.sqrt(ee)
                        if kk == own_cell:
                            own_cap, own_d, own_eps = d_loc / ee, d_loc, ee
                    if pec_between:
                        n_skipped_pec_between += 1
                        continue
                    lo1, hi1 = _sorted_box_corners(sh1)
                    lo2, hi2 = _sorted_box_corners(sh2)
                    g_lo, g_hi = float(hi1[a]), float(lo2[a])
                    if g_hi <= g_lo:
                        continue
                    # in-plane physical point: centre of the two boxes'
                    # in-plane intersection
                    p = [0.0, 0.0, 0.0]
                    for ia in inplane:
                        lo_ov = max(float(lo1[ia]), float(lo2[ia]))
                        hi_ov = min(float(hi1[ia]), float(hi2[ia]))
                        p[ia] = 0.5 * (lo_ov + hi_ov)
                    cuts = {g_lo, g_hi}
                    diel_boxes = []
                    for e in self._geometry:
                        mat = self._resolve_material(e.material_name)
                        if mat.sigma >= self._PEC_SIGMA_THRESHOLD:
                            continue
                        blo, bhi = _sorted_box_corners(e.shape)
                        if blo is None:
                            continue  # non-Box: counted in coverage
                        if not all(blo[ia] <= p[ia] < bhi[ia]
                                   for ia in inplane):
                            continue
                        diel_boxes.append((blo, bhi, float(mat.eps_r)))
                        for c in (float(blo[a]), float(bhi[a])):
                            if g_lo < c < g_hi:
                                cuts.add(c)
                    edges = sorted(cuts)
                    phys_cap = phys_phase = 0.0
                    for e_lo, e_hi in zip(edges[:-1], edges[1:]):
                        zm = 0.5 * (e_lo + e_hi)
                        ee = 1.0
                        for blo, bhi, be in diel_boxes:
                            if blo[a] <= zm < bhi[a]:
                                ee = be  # entry order: later wins
                        phys_cap += (e_hi - e_lo) / ee
                        phys_phase += (e_hi - e_lo) * math.sqrt(ee)
                    if phys_cap <= 0.0 or phys_phase <= 0.0:
                        continue
                    d_cap = mesh_cap / phys_cap - 1.0
                    d_phase = mesh_phase / phys_phase - 1.0
                    if (abs(d_cap) > _CAVITY_THICKNESS_TOL
                            or abs(d_phase) > _CAVITY_THICKNESS_TOL):
                        gap = g_hi - g_lo
                        node_span = float(ctx.nodes[a][k2]
                                          - ctx.nodes[a][k1])
                        governs = ("the capacitance measure (sum d/eps) "
                                   "governs (gap << lambda at freq_max)"
                                   if gap < 0.1 * lam0 else
                                   "the phase measure (sum d*sqrt(eps)) "
                                   "governs (gap not << lambda)")
                        results.append((max(abs(d_cap), abs(d_phase)), (
                            f"[{'xyz'[a]}] {lab1}(node k={k1})/"
                            f"{lab2}(k={k2}) at in-plane column "
                            f"({int(rep[0])},{int(rep[1])}): "
                            f"sum(d/eps) mesh "
                            f"{_fmt_len(mesh_cap)} vs physical "
                            f"{_fmt_len(phys_cap)} ({d_cap:+.1%}); "
                            f"sum(d*sqrt(eps)) mesh {_fmt_len(mesh_phase)} "
                            f"vs physical {_fmt_len(phys_phase)} "
                            f"({d_phase:+.1%}); node-to-node "
                            f"{_fmt_len(node_span)} vs face-to-face "
                            f"{_fmt_len(gap)}; {governs}"
                            + ("" if own_eps is None else (
                                "; of the sum(d/eps) mesh total, "
                                f"{_fmt_len(own_cap)} is {lab1}'s OWN "
                                f"cell ({_fmt_len(own_d)} at eps_r "
                                f"{own_eps:.3f}) — that sheet fills one "
                                "cell, and rfx zeroes only TANGENTIAL E "
                                "on a one-cell PEC sheet, so the cell's "
                                "normal-E edge stays live and sits "
                                "INSIDE the cavity")))))
        if not results:
            return
        results.sort(key=lambda t: -t[0])
        lines = " | ".join(r[1] for r in results[:_CAMPAIGN_MAX_OFFENDERS])
        _w.warn(PreflightWarning(
            f"{len(results)} sheet-bounded cavit(y/ies) differ from the "
            "physical stack by more than "
            f"{_CAVITY_THICKNESS_TOL:.0%} in electrical thickness: {lines}. "
            "OBSERVED: mesh sums run node-to-node across the run's own "
            "cells and assembled eps_r; physical sums run face-to-face "
            "through the geometry Box spans at the pair's shared column — "
            "the difference is the zero-thickness sheet model's honest "
            "cost (each sheet's thickness collapses onto its node) plus "
            "any off-lattice registration. TWO MECHANISMS, and any pair "
            "naming an OWN cell above has the second one: a sheet "
            "registered at its MID-PLANE collapses onto one node and the "
            "gap reads mid-plane to mid-plane (a modelling trade — which "
            "face the plane sits on — not a fixable defect), while a "
            "sheet whose two FACES are registered fills one cell, and "
            "rfx zeroes only tangential E on a one-cell PEC sheet, so "
            "that cell's normal-E edge is live and its permittivity sits "
            "inside the cavity. Face registration therefore does not "
            "shorten the electrical cavity; it trades a collapsed sheet "
            "for a live gap, which reads WORSE when that cell is vacuum. "
            "WHY BOTH MEASURES: the same "
            "defect class measured 17.3% as a series capacitance and 3.2% "
            "as phase length (#703) — a bare percentage invites "
            "'correcting' a right number into a wrong one, so both are "
            "printed and the governing one is named. REMEDY: none required "
            "for the sheet model itself (this is a quantified limit, not a "
            "defect); if the governing measure's delta matters for a "
            "claims-bearing number, resolve the sheet thickness with cells "
            "or correct the extracted quantity by the printed delta. For "
            "an OWN-cell term the eps_r printed for that cell IS "
            "addressable: it is whatever the geometry puts on the live "
            "edge (issue #702), so a stack whose dielectric abuts the "
            "sheet's faces leaves vacuum there — extend the abutting "
            "dielectric across the sheet's cell, or register the sheet's "
            "mid-plane instead of its faces, and that term goes. "
            f"COVERAGE: examined {n_pairs} adjacent sheet pair(s) from "
            f"{len(sheets)} node-thin conductor sheet(s) on the {ctx.lane} "
            f"lane; physical stack computed from Box entries only — "
            f"{nonbox_diel} non-Box dielectric entr(y/ies) ignored (said "
            f"so, per #703); {n_skipped_pec_between} pair(s) skipped "
            "(conductor between). STALE IF: re-summing the printed column "
            "disagrees with these numbers, or sheets stop being registered "
            "node-thin.",
            code="sheet_cavity_electrical_thickness",
            source="_validate_cfg_sheet_cavity_thickness",
        ))

    def _validate_cfg_off_lattice_design_edges(self, _w, ctx) -> None:
        """#703 check 4: census of conductor design edges landing off-lattice.

        Per conductor Box and axis (sub-cell axes excluded — the node-thin
        snap is checks 2/3's domain), the largest distance from a face to
        its nearest E-node, relative to the axis extent. A resonant
        dimension realized ``dL`` off its design length detunes
        ``df/f ~ dL/L``. One aggregated advisory above
        ``_OFF_LATTICE_EDGE_TOL``, worst offenders first.
        """
        boxes, others = ctx.conductor_entries()
        if not boxes:
            return
        offenders = []
        n_axes = 0
        n_thin_axes = 0
        for i, e in boxes:
            lo, hi = _sorted_box_corners(e.shape)
            mid = 0.5 * (lo + hi)
            for a in range(3):
                ext = float(hi[a] - lo[a])
                if ext <= (ctx.local_spacing(a, float(mid[a]))
                           * _CAMPAIGN_SUBCELL_FACTOR):
                    n_thin_axes += 1
                    continue
                n_axes += 1
                nodes = ctx.nodes[a]
                res = max(
                    float(np.min(np.abs(nodes - float(lo[a])))),
                    float(np.min(np.abs(nodes - float(hi[a])))),
                )
                rel = res / ext
                if rel > _OFF_LATTICE_EDGE_TOL:
                    offenders.append((rel, i, e.material_name, a, ext, res))
        if not offenders:
            return
        offenders.sort(key=lambda t: -t[0])
        lines = "; ".join(
            f"geometry[{i}] '{name}' {'xyz'[a]}: extent {_fmt_len(ext)}, "
            f"worst face residual {_fmt_len(res)} ({rel:.2%} of the "
            f"extent, df/f ~ {rel:.2%})"
            for rel, i, name, a, ext, res
            in offenders[:_CAMPAIGN_MAX_OFFENDERS])
        _w.warn(PreflightWarning(
            f"{len(offenders)} conductor-Box design edge(s) sit off-lattice "
            f"by more than {_OFF_LATTICE_EDGE_TOL:.1%} of their extent "
            f"(worst {min(len(offenders), _CAMPAIGN_MAX_OFFENDERS)} "
            f"listed): {lines}. OBSERVED: distance from each Box face to "
            "its nearest E-node on this run's own node coordinates; the "
            "rasterized edge quantizes onto a node, so the realized extent "
            "can differ from the design by up to the printed residual, and "
            "a resonant dimension realized dL off detunes df/f ~ dL/L. "
            "COST (measured, #703): a uniform-mesh sweep rounded ONE "
            "substrate thickness by 8-10% across three 'convergence' "
            "points — three different boards solved under one name; the "
            "same campaign's board survived at dx=50µm only because every "
            "patterned dimension happened to be an exact multiple of 50µm. "
            "REMEDY: choose dx commensurate with the patterned dimensions, "
            "slide the lattice origin onto the worst face, or (non-uniform "
            "lane) place mesh nodes on the design edges. COVERAGE: "
            f"examined {n_axes} axis extent(s) on {len(boxes)} conductor "
            f"Box entr(y/ies) on the {ctx.lane} lane; {n_thin_axes} "
            "sub-cell axis extent(s) excluded (the node-thin snap is the "
            f"live-edge/cavity checks' domain); {len(others)} non-Box "
            "conductor entr(y/ies) skipped (no analytic face coordinates). "
            "STALE IF: |face - nearest node| on the run's node coordinates "
            "does not reproduce the printed residuals, or box faces stop "
            "rasterizing on the E-node lattice.",
            code="off_lattice_design_edges",
            source="_validate_cfg_off_lattice_design_edges",
        ))

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
