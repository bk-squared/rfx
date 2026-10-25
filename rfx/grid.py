"""Simulation grid: domain definition, auto-resolution, index helpers."""

from __future__ import annotations

import numpy as np
import warnings
from fractions import Fraction
from math import lcm

from rfx.boundaries.depths import resolve_face_depths

from rfx._grid_metric import (
    node_tie_tol,
    nearest_uniform_index,
    axis_name as _axis_name,
    dual_spacings_from_cells as _dual_spacings_from_cells,
    normalize_axis as _normalize_axis,
)

# Speed of light in vacuum (m/s)
C0 = 299_792_458.0

#: Arithmetic-dust budget in machine epsilons of ``length/dx`` (#1070).
#: ``cells_spanning`` keeps this as a floor under the shared NODE_TIE_REL
#: band: at very large ratios the ULP budget exceeds 1e-9 of one cell.
CELL_COUNT_ULP_BUDGET = 8

# Adopt a common spacing only within this fraction of the planner request.
PERIODIC_AUTO_MIN_FRACTION = 0.75
# Search farther only to suggest an explicit spacing, never to adopt it.
PERIODIC_COMMON_SEARCH_REFINEMENT = 16


def _divides_period(length: float, dx: float) -> bool:
    ratio = length / dx
    return round(ratio) >= 1 and abs(ratio - round(ratio)) <= (
        CELL_COUNT_ULP_BUDGET * np.finfo(float).eps * max(1.0, abs(ratio)))


def _nearest_period_dx(length: float, dx: float) -> float:
    ratio = length / dx
    counts = {max(1, int(np.floor(ratio))), max(1, int(np.ceil(ratio)))}
    return min((length / n for n in counts), key=lambda candidate: abs(candidate - dx))


def _wall_closed_axes(pec_faces, pmc_faces, *, is_2d=False):
    walls = set(pec_faces) | set(pmc_faces)
    return ''.join(a for a in 'xyz' if not (is_2d and a == 'z')
                   and all(f'{a}_{side}' in walls for side in ('lo', 'hi')))


def _periodic_resolution(domain, axes: str, dx: float, *, automatic: bool,
                         wall_axes: str = '') -> float:
    """Fit automatic spacing to periods and the one-percent wall envelope.

    Candidates never coarsen the request. Search below the 0.75 adoption
    bound only to report the largest fitting explicit spacing (down to /16).
    """
    lengths = [float(domain['xyz'.index(a)]) for a in axes]
    if any(not np.isfinite(length) or length <= 0 for length in lengths):
        raise ValueError(f"periodic axes {axes!r} require positive finite lengths; L={lengths} m")
    walls = {a: float(domain['xyz'.index(a)]) for a in wall_axes if a not in axes}

    def fits(spacing):
        return (all(_divides_period(length, spacing) for length in lengths)
                and all(abs(realized_cells(length, spacing, "wall") * spacing / length - 1) <= .01
                        for length in walls.values()))

    if automatic:
        if fits(dx):
            return dx
        minimum = dx / PERIODIC_COMMON_SEARCH_REFINEMENT
        anchors = list(walls.values()) + lengths
        candidates = sorted({length / n for length in anchors
                             for n in range(max(1, int(np.ceil(length / dx))),
                                            int(np.floor(length / minimum)) + 1)}, reverse=True)
        fitting = next((candidate for candidate in candidates if fits(candidate)), None)
        if fitting is not None and fitting >= PERIODIC_AUTO_MIN_FRACTION * dx:
            warnings.warn(
                f"Automatic dx={dx:.12g} m snapped to dx={fitting:.12g} m: "
                f"periodic axes {axes!r} divide exactly and wall axes {''.join(walls)!r} "
                f"are within 1 % (minimum {PERIODIC_AUTO_MIN_FRACTION:g} * requested dx).",
                UserWarning, stacklevel=3)
            return fitting
        advice = (f"largest fitting dx={fitting:.12g} m ({fitting * 1e3:.12g} mm); "
                  "set dx explicitly to use it" if fitting is not None
                  else "no fitting spacing found")
        raise ValueError(
            f"Automatic dx={dx:.12g} m cannot fit periodic axes {axes!r} and wall axes "
            f"{''.join(walls)!r} within 1 % at >= {PERIODIC_AUTO_MIN_FRACTION:g} * requested dx; "
            f"{advice}. Suggestion search limit: requested dx/{PERIODIC_COMMON_SEARCH_REFINEMENT}.")
    if all(_divides_period(length, dx) for length in lengths):
        return dx
    ratios = [Fraction(length / lengths[0]).limit_denominator(10000) for length in lengths]
    common_length = lengths[0] / lcm(*(r.denominator for r in ratios))
    suggestion = _nearest_period_dx(common_length, dx)
    advice = (f"nearest common dividing dx={suggestion:.12g} m"
              if all(_divides_period(length, suggestion) for length in lengths)
              else "choose commensurate lengths and an explicit dx that divides every period")
    details = "; ".join(
        f"axis {axis}: L={length:.12g} m, dx={dx:.12g} m, "
        f"nearest dividing dx={_nearest_period_dx(length, dx):.12g} m"
        for axis, length in zip(axes, lengths))
    raise ValueError(f"Periodic grid requires a common dx dividing each L ({details}); {advice}.")


def cells_spanning(length: float, dx: float, *,
                   ulp_budget: int = CELL_COUNT_ULP_BUDGET) -> int:
    """Cells needed to span *length* at cell size *dx* (issue #1070).

    ``ceil(length / dx)``, except that a ratio sitting within
    ``max(node_tie_tol(r, 1), ulp_budget * eps * max(1, abs(r)))`` of an integer
    is taken to BE that integer. The shared on-lattice band keeps sizing
    and Box faces on the same node (#1138); the existing ULP rule remains
    a floor for very large cell counts. Plain ``ceil`` buys an extra cell for float
    dust, and nothing downstream fills it: it lies outside every declared
    Box, so it rasterizes as vacuum, and at a face that abuts an absorber
    the pad extension then replicates that vacuum outward (#1070, the
    staircase-lane twin of #831).

    This is a LENGTH question only. A step count is not one -- see
    ``Grid.num_timesteps``, which still takes a plain ``ceil`` because
    running a fraction of a step over is right and running one short is not.
    """
    ratio = length / dx
    nearest = round(ratio)
    tolerance = max(
        node_tie_tol(ratio, 1.0),
        ulp_budget * float(np.finfo(float).eps) * max(1.0, abs(ratio)))
    if abs(ratio - nearest) <= tolerance:
        cells = int(nearest)
    else:
        cells = int(np.ceil(ratio))
    # A POSITIVE length always needs a cell (review of PR #1136, G). Zero is
    # the one integer the dust band can reach from above while the length is
    # still real: for 0 < r <= NODE_TIE_REL the nearest integer is 0, and snapping
    # there would return fewer cells than the declared length needs, where
    # every other snap returns one fewer cell than a ratio that did not need
    # it. ``ceil`` gave 1 and so does this.
    if ratio > 0.0 and cells < 1:
        return 1
    return cells


def realized_cells(length: float, dx: float, kind: str) -> int:
    """Count wall, absorber or periodic cells from the realized axis kind.

    Walls round to nearest, with true half ties rounding half up. The same
    node-tie/ULP band as ``cells_spanning`` snaps the shifted ratio at a
    half tie; absorbers and periodic axes retain ``cells_spanning``.
    """
    if kind != "wall":
        if kind not in ("absorber", "periodic"):
            raise ValueError(f"Unknown axis kind {kind!r}")
        return cells_spanning(length, dx)
    shifted = length / dx + .5
    nearest = round(shifted)
    tolerance = max(node_tie_tol(shifted, 1.0),
                    CELL_COUNT_ULP_BUDGET * np.finfo(float).eps * max(1.0, abs(shifted)))
    return max(1, int(np.floor(nearest if abs(shifted - nearest) <= tolerance else shifted)))


def axis_kinds(boundary_depths):
    """Classify axes from resolved faces, including mixed absorber faces."""
    from rfx.boundaries.depths import Kind
    faces = {face.name: face for face in boundary_depths}
    result = {}
    for axis in "xyz":
        pair = [faces[f"{axis}_{side}"] for side in ("lo", "hi")]
        result[axis] = ("periodic" if all(f.kind == Kind.PERIODIC for f in pair)
                        else "wall" if all(f.kind != Kind.PERIODIC and f.realized == 0
                                           for f in pair) else "absorber")
    return result


def domain_cells(domain, dx, boundary_depths):
    """Interior cell count per axis of ``domain`` from its resolved faces."""
    kinds = axis_kinds(boundary_depths)
    return tuple(realized_cells(float(length), dx, kinds[axis])
                 for axis, length in zip("xyz", domain))


class Grid:
    """Rectilinear FDTD grid with uniform cell size.

    Supports 3D and 2D (TMz/TEz) modes.  In 2D mode the grid has
    ``nz=1`` and the Courant factor uses √2 instead of √3.  The
    existing 3D update equations naturally reduce to 2D because
    z-derivatives vanish when the z-axis is periodic with one cell.

    Parameters
    ----------
    freq_max : float
        Maximum simulation frequency (Hz). Used for auto-resolution.
    domain : tuple[float, float, float]
        Physical domain size (Lx, Ly, Lz) in meters.
        For 2D modes, Lz is ignored (set nz=1 internally).
    dx : float | None
        Cell size override. If None, auto-computed as λ_min / 20.
    cpml_layers : int
        Number of CPML absorbing layers on each face.
    cpml_axes : str
        Axes that receive CPML padding. Default ``"xyz"``.
    mode : str
        ``"3d"`` (default), ``"2d_tmz"`` (Ez, Hx, Hy), or
        ``"2d_tez"`` (Hz, Ex, Ey).
    """

    def __init__(
        self,
        freq_max: float,
        domain: tuple[float, float, float],
        dx: float | None = None,
        cpml_layers: int = 8,
        cpml_axes: str = "xyz",
        mode: str = "3d",
        kappa_max: float | None = None,
        pec_faces: set[str] | None = None,
        pmc_faces: set[str] | None = None,
        face_layers: dict | None = None,
        conformal_faces: set[str] | None = None,
        periodic_axes: str = "",
        boundary_depths: tuple | None = None,
    ):
        if mode not in ("3d", "2d_tmz", "2d_tez"):
            raise ValueError(f"mode must be '3d', '2d_tmz', or '2d_tez', got {mode!r}")
        invalid_axes = sorted(set(cpml_axes) - set("xyz"))
        if invalid_axes:
            raise ValueError(f"cpml_axes must be drawn from 'xyz', got invalid axes {invalid_axes}")

        self.freq_max = freq_max
        self.domain = domain
        self.cpml_layers = cpml_layers
        self.kappa_max = kappa_max
        self.pec_faces = pec_faces or set()
        self.pmc_faces = pmc_faces or set()
        # Stage 1 conformal PEC: face labels whose enclosing axis is
        # declared ``Boundary(conformal=True)``. ``init_waveguide_port``
        # consults this set to skip the binary +face DROP on the modal
        # V/I aperture — the Dey-Mittra eps_correction at the boundary
        # cell is the principled handler when a conformal Box is in
        # ``pec_shapes``.
        self.conformal_faces = conformal_faces or set()
        self.cpml_axes = "".join(axis for axis in "xyz" if axis in cpml_axes)
        self.mode = mode
        self.is_2d = mode.startswith("2d")
        invalid_periodic = sorted(set(periodic_axes) - set("xyz"))
        if invalid_periodic:
            raise ValueError(f"periodic_axes must be drawn from 'xyz', got {invalid_periodic}")
        self.periodic_axes = "".join(a for a in "xyz" if a in periodic_axes and not (a == "z" and self.is_2d))
        self.cpml_axes = "".join(a for a in self.cpml_axes if a not in self.periodic_axes)

        # Auto-resolution: λ_min / 20
        lambda_min = C0 / freq_max
        requested_dx = dx if dx is not None else lambda_min / 20.0
        self.boundary_depths = boundary_depths if boundary_depths is not None else resolve_face_depths(
            budget=cpml_layers, absorbing_axes=self.cpml_axes,
            pec_faces=self.pec_faces, pmc_faces=self.pmc_faces,
            periodic_axes=self.periodic_axes, mode=mode, face_layers=face_layers,
        )
        kinds = axis_kinds(self.boundary_depths)
        self.dx = _periodic_resolution(
            domain, self.periodic_axes, requested_dx, automatic=dx is None,
            wall_axes="".join(a for a in "xyz" if kinds[a] == "wall"
                              and not (self.is_2d and a == "z")))
        ndim = 2 if self.is_2d else 3
        self.dt = self.dx / (C0 * np.sqrt(float(ndim))) * 0.99
        if self.is_2d:
            self.cpml_axes = self.cpml_axes.replace("z", "")

        # Legacy view retains declared depths, including scalar defaults on walls.
        self.face_layers = {record.name: record.declared for record in self.boundary_depths}
        for record in self.boundary_depths:
            setattr(self, f"pad_{record.name}", record.realized)

        # Legacy scalar ``pad_{axis}`` kept for callers that use it as
        # "nominal CPML thickness on this axis". When lo / hi differ
        # (asymmetric reflector + absorber composition) ``pad_{axis}``
        # reports the max of the two — callers that need a specific
        # side must use the per-face attributes.
        self.pad_x = max(self.pad_x_lo, self.pad_x_hi)
        self.pad_y = max(self.pad_y_lo, self.pad_y_hi)
        self.pad_z = max(self.pad_z_lo, self.pad_z_hi)
        # ``axis_pads`` carries the LEADING (``lo``) pad per axis, i.e.
        # the same number that callers subtract from array indices to
        # recover user-domain coordinates (``(idx - axis_pads[ax]) * dx``).
        # In the legacy symmetric layout this happened to equal
        # ``pad_{axis}``; under per-face allocation the two are different
        # whenever one face is PMC/PEC/periodic. Existing callers that
        # treated ``axis_pads`` as a coordinate offset continue to work
        # automatically for asymmetric configurations.
        self.axis_pads = (self.pad_x_lo, self.pad_y_lo, self.pad_z_lo)
        # Six-tuple of per-face pads; preferred over ``axis_pads`` for
        # new code that needs both sides (shape math, exterior fill).
        self.face_pads = (
            self.pad_x_lo, self.pad_x_hi,
            self.pad_y_lo, self.pad_y_hi,
            self.pad_z_lo, self.pad_z_hi,
        )

        # Grid dimensions (including CPML padding)
        # +1 fence-post correction: N cells need N+1 nodes so that
        # PEC walls at index 0 and index N span exactly N*dx.
        #
        # ``cells_spanning`` rather than a bare ``ceil`` (#1070): a declared
        # length is an arithmetic expression, and one ULP of float dust in
        # ``domain/dx`` used to buy a whole cell that no declared Box reaches.
        self.nx = (realized_cells(domain[0], self.dx, kinds["x"]) + ("x" not in self.periodic_axes)
                   + self.pad_x_lo + self.pad_x_hi)
        self.ny = (realized_cells(domain[1], self.dx, kinds["y"]) + ("y" not in self.periodic_axes)
                   + self.pad_y_lo + self.pad_y_hi)

        if self.is_2d:
            self.nz = 1  # single cell in z, use periodic z BC
        else:
            self.nz = (realized_cells(domain[2], self.dx, kinds["z"]) + ("z" not in self.periodic_axes)
                       + self.pad_z_lo + self.pad_z_hi)

        self.shape = (self.nx, self.ny, self.nz)

        # Interior region (excluding CPML)
        if self.is_2d:
            self.interior = (
                slice(self.pad_x_lo, self.nx - self.pad_x_hi),
                slice(self.pad_y_lo, self.ny - self.pad_y_hi),
                slice(0, 1),
            )
        else:
            self.interior = (
                slice(self.pad_x_lo, self.nx - self.pad_x_hi),
                slice(self.pad_y_lo, self.ny - self.pad_y_hi),
                slice(self.pad_z_lo, self.nz - self.pad_z_hi),
            )

    @staticmethod
    def courant_dt(dx: float, ndim: int = 3) -> float:
        """Courant-stable timestep for 2D or 3D FDTD."""
        return dx / (C0 * np.sqrt(float(ndim))) * 0.99

    def num_timesteps(self, num_periods: float = 20.0) -> int:
        """Estimate timesteps for given number of periods at freq_max."""
        from rfx.stepping.step_count import steps_for_periods
        return steps_for_periods(self.dt, self.freq_max, num_periods)

    def position_to_index(self, pos: tuple[float, float, float]) -> tuple[int, int, int]:
        """Convert physical position to grid index (accounting for the
        leading per-face CPML offset ``pad_{axis}_lo``).

        Raises
        ------
        ValueError
            If ``pos`` maps outside the grid. The pre-guard code
            returned an out-of-range index silently, which then indexed
            the wrong cell (or wrapped negatively) downstream.
        """
        # Preserve this API's input-dtype division, including half-node ties.
        i = self._rounded_index(0, pos[0] / self.dx)
        j = self._rounded_index(1, pos[1] / self.dx)
        k = 0 if self.is_2d else self._rounded_index(2, pos[2] / self.dx)
        nx, ny, nz = self.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise ValueError(
                f"position {pos} maps to grid index ({i}, {j}, {k}), "
                f"outside the grid shape {(nx, ny, nz)}. Check the "
                f"position lies inside the simulation domain."
            )
        return (i, j, k)

    def _rounded_index(self, axis: int, ratio, *, wrap=True) -> int:
        """Round the caller's quotient without changing its scalar dtype."""
        idx = nearest_uniform_index(ratio) + self.axis_pads[axis]
        if wrap and _axis_name(axis) in self.periodic_axes:
            if ratio < -0.5 or ratio >= self.shape[axis] + 0.5:
                # Let each public API issue its historical out-of-grid message.
                return idx
            return idx % self.shape[axis]
        return idx

    def interval_to_indices(self, start, end):
        """Map a cell/edge extent without wrapping its exclusive high endpoint.

        A constant coordinate denotes a plane and uses the point lookup.
        Non-periodic axes retain position_to_index's input-dtype division.
        """
        from rfx._periodic import interval_coordinates
        corners = [list(start), list(end)]
        spanning = [start[a] != end[a] for a in range(3)]
        for a in range(3):
            if spanning[a]:
                corners[0][a], corners[1][a] = interval_coordinates(
                    self, a, start[a], end[a])
        indices = []
        for pos in corners:
            idx = tuple(0 if a == 2 and self.is_2d else self._rounded_index(
                a, pos[a] / self.dx, wrap=not spanning[a]) for a in range(3))
            if not all(0 <= idx[a] < self.shape[a] + int(
                    spanning[a] and 'xyz'[a] in self.periodic_axes) for a in range(3)):
                raise ValueError(
                    f"position {tuple(pos)} maps to grid index {idx}, "
                    f"outside the grid shape {self.shape}. Check the "
                    "position lies inside the simulation domain.")
            indices.append(idx)
        return tuple(indices)

    # ------------------------------------------------------------------
    # Grid metric interface (step 0a, design note
    # docs/design_notes/20260922_nu_grid_core_predeclaration.md decision 2).
    #
    # THE CONVENTION -- index origin, the primal/dual rule, the end entries
    # of the two inverse-metric arrays, and which consumer is entitled to
    # which metric -- is written once, in ``rfx/_grid_metric.py``. Read that
    # module before adding a formula at a call site; decision 3 is that a
    # consumer needing a third quantity adds a named accessor here instead.
    #
    # ``NonUniformGrid`` carries the same seven methods with the same meaning,
    # so a consumer can hold either class and never branch on its type. On
    # THIS class every axis is constant by construction, so every array is
    # ``np.full(n, dx)`` and primal == dual bit-exactly.
    # ------------------------------------------------------------------

    def _axis_extent(self, axis) -> tuple[int, int]:
        """``(n, pad_lo)`` for a normalized axis index."""
        ax = _normalize_axis(axis)
        n = (self.nx, self.ny, self.nz)[ax]
        pad_lo = (self.pad_x_lo, self.pad_y_lo, self.pad_z_lo)[ax]
        return int(n), int(pad_lo)

    def cells(self, axis) -> np.ndarray:
        """PRIMAL cell widths along ``axis``, float64, full padded length.

        Constant by construction: ``np.full(n, dx)``. The last entry is the
        fence-post node-provider (see ``rfx/_grid_metric.py``), so the sum
        overshoots the realized extent by one cell -- use ``node_of``.
        """
        n, _ = self._axis_extent(axis)
        return np.full(n, float(self.dx), dtype=np.float64)

    def duals(self, axis) -> np.ndarray:
        """DUAL (E-node) spacings along ``axis``, float64.

        On a constant axis ``dual[k] = (d[k-1]+d[k])/2 = d[k]`` exactly, and
        ``dual[0] = d[0]`` by the same rule, so this equals ``cells`` bit for
        bit. Spelled through the shared helper anyway so the two classes
        cannot drift apart in how the rule is written.
        """
        return _dual_spacings_from_cells(self.cells(axis))

    def index_of(self, axis, x: float) -> int:
        """Padded index of the node nearest physical coordinate ``x``.

        Divides in float64 before applying the shared lower-node tie rule.
        ``position_to_index`` retains its input scalar's division dtype.
        Declared periodic axes wrap modulo their node count. In 2-D
        mode the z axis holds one cell and the answer is always 0, which is
        what ``position_to_index`` returns there.
        """
        ax = _normalize_axis(axis)
        n, _ = self._axis_extent(ax)
        if ax == 2 and self.is_2d:
            return 0
        idx = self._rounded_index(ax, float(x) / self.dx)
        if not (0 <= idx < n):
            raise ValueError(
                f"position {float(x)} on axis {_axis_name(ax)!r} maps to "
                f"padded index {idx}, outside this axis's {n} entries. Check "
                f"the coordinate lies inside the simulation domain."
            )
        return idx

    def node_of(self, axis, i: int) -> float:
        """Physical coordinate of the E node at padded index ``i``.

        ``(i - pad_lo) * dx`` -- the closed form ``_uniform_axis_nodes``
        (``rfx.geometry.rasterize_grid._uniform_axis_nodes``) evaluates for the whole
        axis, so this equals ``coords_from_uniform_grid(grid).<axis>[i]`` bit
        for bit, and equals the bare ``i * dx`` on an axis with no pad.
        """
        ax = _normalize_axis(axis)
        n, pad_lo = self._axis_extent(ax)
        idx = int(i)
        if not (0 <= idx < n):
            raise IndexError(
                f"padded index {idx} is outside axis {_axis_name(ax)!r}, "
                f"which has {n} entries"
            )
        return (float(idx) - pad_lo) * self.dx

    def boundary_cell(self, axis, side: str) -> float:
        """The cell width at the ``"lo"`` or ``"hi"`` face of ``axis``.

        The number a CPML profile is entitled to calibrate against. Constant
        here, so both sides give ``dx``.
        """
        if side not in ("lo", "hi"):
            raise ValueError(f"side must be 'lo' or 'hi', got {side!r}")
        cells = self.cells(axis)
        return float(cells[0] if side == "lo" else cells[-1])

    def is_constant(self, axis) -> bool:
        """Whether every cell on ``axis`` has the same width.

        Always ``True`` on this class. Decision 4 selects the uniform kernel
        on this predicate rather than on "was a profile given", so that the
        kernel and the metric have one source.
        """
        _normalize_axis(axis)
        return True

    def is_traced(self, axis) -> bool:
        """Whether ``axis``'s cell widths are a JAX tracer (decision 6).

        Always ``False`` here: a uniform ``Grid`` holds a Python float.
        """
        _normalize_axis(axis)
        return False

    def __repr__(self) -> str:
        return (
            f"Grid(shape={self.shape}, dx={self.dx:.4e} m, "
            f"dt={self.dt:.4e} s, freq_max={self.freq_max:.2e} Hz, "
            f"cpml_axes={self.cpml_axes!r})"
        )
