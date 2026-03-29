"""Tests for Grid class."""

import numpy as np
import pytest

from rfx.grid import Grid, C0


def test_auto_resolution():
    """dx should be <= lambda_min / 20."""
    freq = 10e9
    grid = Grid(freq_max=freq, domain=(0.05, 0.05, 0.05))
    lambda_min = C0 / freq
    assert grid.dx <= lambda_min / 20.0


def test_courant_stability():
    """dt should satisfy Courant condition: dt < dx / (c * sqrt(3))."""
    grid = Grid(freq_max=5e9, domain=(0.1, 0.1, 0.05))
    courant_limit = grid.dx / (C0 * np.sqrt(3.0))
    assert grid.dt < courant_limit


def test_position_to_index():
    """Position conversion should account for CPML padding."""
    grid = Grid(freq_max=3e9, domain=(0.1, 0.1, 0.1), cpml_layers=10)
    # Origin should map to (cpml_layers, cpml_layers, cpml_layers)
    idx = grid.position_to_index((0.0, 0.0, 0.0))
    assert idx == (10, 10, 10)


def test_custom_dx():
    """User-specified dx should override auto-resolution."""
    grid = Grid(freq_max=10e9, domain=(0.1, 0.1, 0.1), dx=0.002)
    assert grid.dx == 0.002


def test_num_timesteps():
    """Timestep count should scale with number of periods."""
    grid = Grid(freq_max=5e9, domain=(0.1, 0.1, 0.1))
    n10 = grid.num_timesteps(num_periods=10)
    n20 = grid.num_timesteps(num_periods=20)
    assert abs(n20 / n10 - 2.0) < 0.1


def test_cpml_axes_padding_is_per_axis():
    """x-only CPML should not pad y/z coordinates or indices."""
    grid = Grid(freq_max=5e9, domain=(0.1, 0.04, 0.02), dx=0.002, cpml_layers=8, cpml_axes="x")
    assert grid.shape == (67, 21, 11)
    assert grid.position_to_index((0.0, 0.0, 0.0)) == (8, 0, 0)
