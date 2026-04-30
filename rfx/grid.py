"""Simulation grid: domain definition, auto-resolution, index helpers."""

from __future__ import annotations

import numpy as np

# Speed of light in vacuum (m/s)
C0 = 299_792_458.0


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

        # Per-face CPML allocation (v1.7.5). A face whose BoundarySpec
        # token is ``pec``/``pmc``/``periodic`` gets ``pad=0`` on that
        # side even when the axis as a whole participates in CPML —
        # this is the Meep / OpenEMS / Tidy3D convention and the
        # architectural fix for PMC + CPML composition. See
        # ``docs/research_notes/2026-04-19_v175_t10_half_symmetric_pmc.md``.
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
        # In the pre-v1.7.5 symmetric layout this happened to equal
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
        self.nx = (int(np.ceil(domain[0] / self.dx)) + 1
                   + self.pad_x_lo + self.pad_x_hi)
        self.ny = (int(np.ceil(domain[1] / self.dx)) + 1
                   + self.pad_y_lo + self.pad_y_hi)

        if self.is_2d:
            self.nz = 1  # single cell in z, use periodic z BC
        else:
            self.nz = (int(np.ceil(domain[2] / self.dx)) + 1
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
        leading per-face CPML offset ``pad_{axis}_lo``)."""
        k = 0 if self.is_2d else int(round(pos[2] / self.dx)) + self.pad_z_lo
        return (
            int(round(pos[0] / self.dx)) + self.pad_x_lo,
            int(round(pos[1] / self.dx)) + self.pad_y_lo,
            k,
        )

    def __repr__(self) -> str:
        return (
            f"Grid(shape={self.shape}, dx={self.dx:.4e} m, "
            f"dt={self.dt:.4e} s, freq_max={self.freq_max:.2e} Hz, "
            f"cpml_axes={self.cpml_axes!r})"
        )
