"""Tests for Holland thin-wire subcell model."""

import numpy as np
import jax.numpy as jnp

from rfx.grid import Grid
from rfx.geometry.thin_wire import ThinWire, compute_thin_wire_correction


def test_thin_wire_creates_correction():
    """Wire should produce non-zero eps/sigma along its path."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.02, 0.02), dx=0.002, cpml_layers=0)
    wire = ThinWire(start=(0.01, 0.01, 0.01), end=(0.04, 0.01, 0.01),
                    radius=0.0001)

    eps_corr, sigma_corr = compute_thin_wire_correction(grid, wire)

    assert float(jnp.max(eps_corr)) > 0, "Should have non-zero eps correction"
    assert float(jnp.max(sigma_corr)) > 0, "Should have non-zero sigma correction"

    n_modified = int(jnp.sum(eps_corr > 0))
    print(f"\nThin wire (x-aligned, r=0.1mm):")
    print(f"  Modified cells: {n_modified}")
    print(f"  Max eps_corr: {float(jnp.max(eps_corr)):.4f}")
    print(f"  Max sigma_corr: {float(jnp.max(sigma_corr)):.2e}")
    assert n_modified > 0


def test_thin_wire_axis_aligned():
    """X/Y/Z aligned wires should modify correct cells."""
    grid = Grid(freq_max=5e9, domain=(0.03, 0.03, 0.03), dx=0.003, cpml_layers=0)

    for axis, start, end in [
        ("x", (0.006, 0.015, 0.015), (0.024, 0.015, 0.015)),
        ("y", (0.015, 0.006, 0.015), (0.015, 0.024, 0.015)),
        ("z", (0.015, 0.015, 0.006), (0.015, 0.015, 0.024)),
    ]:
        wire = ThinWire(start=start, end=end, radius=0.0001)
        eps_corr, _ = compute_thin_wire_correction(grid, wire)
        n_mod = int(jnp.sum(eps_corr > 0))
        print(f"  {axis}-wire: {n_mod} cells modified")
        assert n_mod > 0, f"{axis}-aligned wire should modify cells"


def test_thin_wire_preserves_bulk():
    """Cells far from wire should have zero correction."""
    grid = Grid(freq_max=5e9, domain=(0.05, 0.05, 0.05), dx=0.005, cpml_layers=0)
    wire = ThinWire(start=(0.025, 0.025, 0.01), end=(0.025, 0.025, 0.04),
                    radius=0.0001)

    eps_corr, sigma_corr = compute_thin_wire_correction(grid, wire)

    # Corner cells should be unmodified
    assert float(eps_corr[0, 0, 0]) == 0.0
    assert float(sigma_corr[0, 0, 0]) == 0.0
    assert float(eps_corr[-1, -1, -1]) == 0.0
