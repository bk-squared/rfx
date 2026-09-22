"""Simulation grid: domain definition, auto-resolution, index helpers."""

from __future__ import annotations

import numpy as np

from rfx._grid_metric import (
    axis_name as _axis_name,
    dual_spacings_from_cells as _dual_spacings_from_cells,
    normalize_axis as _normalize_axis,
)

# Speed of light in vacuum (m/s)
C0 = 299_792_458.0

#: How many machine epsilons of ``length/dx`` count as float dust rather than
#: as declared sub-cell intent (issue #1070).
#:
#: A declared domain length is an ARITHMETIC EXPRESSION, not a literal: the
#: rig in #1070 writes ``38*h + 2*(pad*h)``, and the #801 patch fixture,
#: the WR-90 lanes and the MSL boards all build theirs the same way. Each
#: multiply, each add and the division itself rounds, and each rounding costs
#: at most half an ULP of the result. Eight of them is a generous bound on a
#: handful of summed terms -- the rig above spends three -- and the tolerance
#: it buys is ``8 * eps * max(1, r)``, which at r = 232 is 4.12e-13, or 14.5
#: ULP of r. So the rule absorbs a deviation of up to 14 ULP there and takes
#: 15 as real.
#:
#: The distance to the nearest case anyone could MEAN is the reason this is
#: safe: a domain one part in 1e6 longer than 232 cells sits 8.2e9 ULP away,
#: nine orders of magnitude outside the band. An ABSOLUTE tolerance has no
#: such separation at either end of the useful range -- ``round(r, 9)`` is
#: coarser than an ULP for a million-cell domain and finer than one for a
#: domain of a few cells -- which is why this is relative.
CELL_COUNT_ULP_BUDGET = 8


def cells_spanning(length: float, dx: float, *,
                   ulp_budget: int = CELL_COUNT_ULP_BUDGET) -> int:
    """Cells needed to span *length* at cell size *dx* (issue #1070).

    ``ceil(length / dx)``, except that a ratio sitting within
    ``ulp_budget * eps * max(1, r)`` of an integer is taken to BE that
    integer. Plain ``ceil`` buys a whole extra cell for one ULP of float
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
    if abs(ratio - nearest) <= ulp_budget * float(np.finfo(float).eps) * max(
            1.0, abs(ratio)):
        cells = int(nearest)
    else:
        cells = int(np.ceil(ratio))
    # A POSITIVE length always needs a cell (review of PR #1136, G). Zero is
    # the one integer the dust band can reach from above while the length is
    # still real: for 0 < r < 8*eps the nearest integer is 0, and snapping
    # there would return fewer cells than the declared length needs, where
    # every other snap returns one fewer cell than a ratio that did not need
    # it. ``ceil`` gave 1 and so does this.
    if ratio > 0.0 and cells < 1:
        return 1
    return cells


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
        # T7 Phase 2 PR2: per-face active CPML layer counts (thickness).
        # Defaults to the scalar ``cpml_layers`` on every face (the
        # symmetric fast path). Asymmetric thickness is achieved by
        # capping active layers below ``cpml_layers`` per face — the
        # unused allocation stays as no-op padding in the CPML profile
        # so the Yee grid + CPMLState shape stay uniform.
        _default_face_n = {f: cpml_layers for f in
                           ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")}
        self.face_layers = {**_default_face_n, **(face_layers or {})}
        for _face, _n in self.face_layers.items():
            if _n > cpml_layers:
                raise ValueError(
                    f"face_layers[{_face!r}]={_n} exceeds cpml_layers="
                    f"{cpml_layers}; the scalar is the allocation budget "
                    f"and per-face active layers must be <= that budget."
                )
        self.cpml_axes = "".join(axis for axis in "xyz" if axis in cpml_axes)
        self.mode = mode
        self.is_2d = mode.startswith("2d")

        # Auto-resolution: λ_min / 20
        lambda_min = C0 / freq_max
        self.dx = dx if dx is not None else lambda_min / 20.0

        # Courant-stable timestep: √2 for 2D, √3 for 3D
        ndim = 2 if self.is_2d else 3
        self.dt = self.dx / (C0 * np.sqrt(float(ndim))) * 0.99

        if self.is_2d:
            self.cpml_axes = self.cpml_axes.replace("z", "")

        # Per-face CPML allocation (2026-04). A face whose BoundarySpec
        # token is ``pec``/``pmc``/``periodic`` gets ``pad=0`` on that
        # side even when the axis as a whole participates in CPML —
        # this is the Meep / OpenEMS / Tidy3D convention and the
        # architectural fix for PMC + CPML composition.
        def _face_pad(axis: str, side: str) -> int:
            face = f"{axis}_{side}"
            if face in self.pec_faces or face in self.pmc_faces:
                return 0
            if axis not in self.cpml_axes:
                return 0
            return int(self.face_layers.get(face, cpml_layers))

        self.pad_x_lo = _face_pad("x", "lo")
        self.pad_x_hi = _face_pad("x", "hi")
        self.pad_y_lo = _face_pad("y", "lo")
        self.pad_y_hi = _face_pad("y", "hi")
        if self.is_2d:
            self.pad_z_lo = 0
            self.pad_z_hi = 0
        else:
            self.pad_z_lo = _face_pad("z", "lo")
            self.pad_z_hi = _face_pad("z", "hi")

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
        self.nx = (cells_spanning(domain[0], self.dx) + 1
                   + self.pad_x_lo + self.pad_x_hi)
        self.ny = (cells_spanning(domain[1], self.dx) + 1
                   + self.pad_y_lo + self.pad_y_hi)

        if self.is_2d:
            self.nz = 1  # single cell in z, use periodic z BC
        else:
            self.nz = (cells_spanning(domain[2], self.dx) + 1
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
        period = 1.0 / self.freq_max
        return int(np.ceil(num_periods * period / self.dt))

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
        i = int(round(pos[0] / self.dx)) + self.pad_x_lo
        j = int(round(pos[1] / self.dx)) + self.pad_y_lo
        k = 0 if self.is_2d else int(round(pos[2] / self.dx)) + self.pad_z_lo
        nx, ny, nz = self.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise ValueError(
                f"position {pos} maps to grid index ({i}, {j}, {k}), "
                f"outside the grid shape {(nx, ny, nz)}. Check the "
                f"position lies inside the simulation domain."
            )
        return (i, j, k)

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
    # ``NonUniformGrid`` carries the same six methods with the same meaning,
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

        Same arithmetic as the ``axis`` component of ``position_to_index``:
        ``round(x/dx) + pad_lo``, with the same out-of-grid refusal. In 2-D
        mode the z axis holds one cell and the answer is always 0, which is
        what ``position_to_index`` returns there.
        """
        ax = _normalize_axis(axis)
        n, pad_lo = self._axis_extent(ax)
        if ax == 2 and self.is_2d:
            return 0
        idx = int(round(float(x) / self.dx)) + pad_lo
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
        (``rfx/geometry/rasterize_grid.py:76-81``) evaluates for the whole
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
