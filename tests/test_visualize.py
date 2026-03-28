"""Tests for field visualization utilities.

Validates that plot functions run without error and return Figure objects.
Uses matplotlib Agg backend to avoid display requirements.
"""

import numpy as np
import pytest

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from rfx.core.yee import init_state


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_field_slice():
    """plot_field_slice should return a Figure."""
    from rfx.visualize import plot_field_slice
    from rfx.grid import Grid

    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02))
    state = init_state(grid.shape)
    # Put some data in ez
    state = state._replace(ez=state.ez.at[5, :, :].set(1.0))

    fig = plot_field_slice(state, grid, component="ez", axis="z")
    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close(fig)
    print("\nplot_field_slice: OK")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_s_params():
    """plot_s_params should return a Figure."""
    from rfx.visualize import plot_s_params

    n_ports, n_freqs = 2, 50
    s = np.random.randn(n_ports, n_ports, n_freqs) * 0.3 + \
        1j * np.random.randn(n_ports, n_ports, n_freqs) * 0.3
    freqs = np.linspace(1e9, 10e9, n_freqs)

    fig = plot_s_params(s, freqs, db=True)
    assert fig is not None
    plt.close(fig)

    fig2 = plot_s_params(s, freqs, db=False)
    assert fig2 is not None
    plt.close(fig2)
    print("\nplot_s_params: OK (dB and linear)")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_time_series():
    """plot_time_series should return a Figure."""
    from rfx.visualize import plot_time_series

    n_steps, n_probes = 100, 3
    ts = np.sin(np.linspace(0, 10 * np.pi, n_steps))[:, None] * np.array([1.0, 0.5, 0.3])
    dt = 1e-12

    fig = plot_time_series(ts, dt, labels=["P1", "P2", "P3"])
    assert fig is not None
    plt.close(fig)
    print("\nplot_time_series: OK")


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_field_slice_axes():
    """plot_field_slice works for all three slice axes."""
    from rfx.visualize import plot_field_slice
    from rfx.grid import Grid

    grid = Grid(freq_max=10e9, domain=(0.02, 0.02, 0.02))
    state = init_state(grid.shape)

    for axis in ["x", "y", "z"]:
        fig = plot_field_slice(state, grid, component="ez", axis=axis)
        assert fig is not None
        plt.close(fig)
    print("\nplot_field_slice all axes: OK")
