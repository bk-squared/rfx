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

        self.pad_x = cpml_layers if "x" in self.cpml_axes else 0
        self.pad_y = cpml_layers if "y" in self.cpml_axes else 0
        self.pad_z = 0 if self.is_2d else (cpml_layers if "z" in self.cpml_axes else 0)
        self.axis_pads = (self.pad_x, self.pad_y, self.pad_z)

        # Grid dimensions (including CPML padding)
        # +1 fence-post correction: N cells need N+1 nodes so that
        # PEC walls at index 0 and index N span exactly N*dx.
        self.nx = int(np.ceil(domain[0] / self.dx)) + 1 + 2 * self.pad_x
        self.ny = int(np.ceil(domain[1] / self.dx)) + 1 + 2 * self.pad_y

        if self.is_2d:
            self.nz = 1  # single cell in z, use periodic z BC
        else:
            self.nz = int(np.ceil(domain[2] / self.dx)) + 1 + 2 * self.pad_z

        self.shape = (self.nx, self.ny, self.nz)

        # Interior region (excluding CPML)
        if self.is_2d:
            self.interior = (
                slice(self.pad_x, self.nx - self.pad_x),
                slice(self.pad_y, self.ny - self.pad_y),
                slice(0, 1),
            )
        else:
            self.interior = (
                slice(self.pad_x, self.nx - self.pad_x),
                slice(self.pad_y, self.ny - self.pad_y),
                slice(self.pad_z, self.nz - self.pad_z),
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
        """Convert physical position to grid index (accounting for CPML offset)."""
        k = 0 if self.is_2d else int(round(pos[2] / self.dx)) + self.pad_z
        return (
            int(round(pos[0] / self.dx)) + self.pad_x,
            int(round(pos[1] / self.dx)) + self.pad_y,
            k,
        )

    def __repr__(self) -> str:
        return (
            f"Grid(shape={self.shape}, dx={self.dx:.4e} m, "
            f"dt={self.dt:.4e} s, freq_max={self.freq_max:.2e} Hz, "
            f"cpml_axes={self.cpml_axes!r})"
        )
