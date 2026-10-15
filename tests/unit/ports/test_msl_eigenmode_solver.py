"""Refusal contract for the eigenmode source removed on 2026-05-04."""

from __future__ import annotations

import numpy as np
import pytest

from rfx.grid import Grid
from rfx.sources.msl_eigenmode import (
    compute_msl_eigenmode_profile,
)
from rfx.sources.msl_port import MSLPort


# Integration-test geometry constants (RO4350B, 50 Ω microstrip)
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
DX_COARSE = 80e-6  # 3 substrate cells
DX_FINE = 40e-6    # 6 substrate cells


def _build_grid_and_port(dx: float):
    """Build a uniform Grid + MSLPort matching the integration-test geometry."""
    LX = 14e-3
    LY = W_TRACE + 6 * dx
    LZ = H_SUB + 1.5e-3
    grid = Grid(freq_max=5e9, domain=(LX, LY, LZ), dx=dx, cpml_layers=8)
    y_centre = LY / 2.0
    port = MSLPort(
        feed_x=2e-3,
        y_lo=y_centre - W_TRACE / 2.0,
        y_hi=y_centre + W_TRACE / 2.0,
        z_lo=0.0,
        z_hi=H_SUB,
        direction="+x",
        impedance=50.0,
        excitation=None,
    )
    return grid, port


def test_eigenmode_profile_request_is_refused():
    """The eigenmode source was removed; Laplace is the supported lane."""
    grid, port = _build_grid_and_port(DX_COARSE)
    freqs = np.linspace(3e9, 4.5e9, 16)
    with pytest.raises(NotImplementedError, match="Path B from-scratch FDFD eigenmode solver"):
        compute_msl_eigenmode_profile(grid, port, eps_r_sub=EPS_R, freqs=freqs)
