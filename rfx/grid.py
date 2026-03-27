"""Simulation grid: domain definition, auto-resolution, index helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Speed of light in vacuum (m/s)
C0 = 299_792_458.0


class Grid:
    """3D rectilinear FDTD grid with uniform cell size.

    Parameters
    ----------
    freq_max : float
        Maximum simulation frequency (Hz). Used for auto-resolution.
    domain : tuple[float, float, float]
        Physical domain size (Lx, Ly, Lz) in meters.
    dx : float | None
        Cell size override. If None, auto-computed as λ_min / 20.
    cpml_layers : int
        Number of CPML absorbing layers on each face.
    """

    def __init__(
        self,
        freq_max: float,
        domain: tuple[float, float, float],
        dx: float | None = None,
        cpml_layers: int = 10,
    ):
        self.freq_max = freq_max
        self.domain = domain
        self.cpml_layers = cpml_layers

        # Auto-resolution: λ_min / 20
        lambda_min = C0 / freq_max
        self.dx = dx if dx is not None else lambda_min / 20.0
        self.dt = self.courant_dt(self.dx)

        # Grid dimensions (including CPML padding)
        # +1 fence-post correction: N cells need N+1 nodes so that
        # PEC walls at index 0 and index N span exactly N*dx.
        pad = 2 * cpml_layers
        self.nx = int(np.ceil(domain[0] / self.dx)) + 1 + pad
        self.ny = int(np.ceil(domain[1] / self.dx)) + 1 + pad
        self.nz = int(np.ceil(domain[2] / self.dx)) + 1 + pad
        self.shape = (self.nx, self.ny, self.nz)

        # Interior region (excluding CPML)
        self.interior = (
            slice(cpml_layers, self.nx - cpml_layers),
            slice(cpml_layers, self.ny - cpml_layers),
            slice(cpml_layers, self.nz - cpml_layers),
        )

    @staticmethod
    def courant_dt(dx: float) -> float:
        """Courant-stable timestep for 3D FDTD."""
        return dx / (C0 * np.sqrt(3.0)) * 0.99  # 0.99 safety factor

    def num_timesteps(self, num_periods: float = 20.0) -> int:
        """Estimate timesteps for given number of periods at freq_max."""
        period = 1.0 / self.freq_max
        return int(np.ceil(num_periods * period / self.dt))

    def position_to_index(self, pos: tuple[float, float, float]) -> tuple[int, int, int]:
        """Convert physical position to grid index (accounting for CPML offset)."""
        return (
            int(round(pos[0] / self.dx)) + self.cpml_layers,
            int(round(pos[1] / self.dx)) + self.cpml_layers,
            int(round(pos[2] / self.dx)) + self.cpml_layers,
        )

    def __repr__(self) -> str:
        return (
            f"Grid(shape={self.shape}, dx={self.dx:.4e} m, "
            f"dt={self.dt:.4e} s, freq_max={self.freq_max:.2e} Hz)"
        )
