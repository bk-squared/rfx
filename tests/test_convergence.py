"""Tests for the mesh convergence study module."""

import numpy as np
import pytest

from rfx.api import Simulation, Result
from rfx.geometry.csg import Box
from rfx.convergence import (
    convergence_study,
    richardson_extrapolation,
    ConvergenceResult,
    quick_convergence,
)


# ---------------------------------------------------------------------------
# Helper: lightweight sim factory
# ---------------------------------------------------------------------------

def _make_sim(dx):
    """Create a minimal PEC-cavity simulation at the given cell size."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.03, 0.03, 0.03),
        boundary="pec",
        dx=dx,
    )
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    return sim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_convergence_study_runs():
    """Convergence study produces results for multiple dx values."""
    dx_values = [4e-3, 3e-3, 2e-3]
    cr = convergence_study(
        sim_factory=_make_sim,
        dx_values=dx_values,
        metric_fn=lambda r: float(np.max(np.abs(np.asarray(r.time_series)))),
        n_steps=30,
        run_kwargs={"compute_s_params": False},
    )

    assert isinstance(cr, ConvergenceResult)
    assert len(cr.dx_values) == 3
    assert len(cr.metrics) == 3
    assert len(cr.results) == 3
    assert len(cr.errors) == 3
    # dx_values should be sorted coarse to fine
    assert cr.dx_values[0] >= cr.dx_values[-1]
    # Extrapolated value should be finite
    assert np.isfinite(cr.extrapolated)
    assert np.isfinite(cr.order)


def test_richardson_extrapolation():
    """Richardson extrapolation improves accuracy estimate.

    Synthetic test: f(dx) = f_exact + C * dx^2.
    Extrapolation should recover f_exact.
    """
    f_exact = 7.5e9  # exact resonance
    C = 1e15  # coefficient for O(dx^2) error

    dx_vals = np.array([3e-3, 2e-3, 1e-3])
    values = f_exact + C * dx_vals ** 2

    extrapolated, order = richardson_extrapolation(values, dx_vals, expected_order=2.0)

    # Should recover the exact value to high precision
    assert abs(extrapolated - f_exact) / f_exact < 1e-6, (
        f"Extrapolated {extrapolated:.6e} should be close to exact {f_exact:.6e}"
    )
    # Estimated order should be ~2
    assert abs(order - 2.0) < 0.1, (
        f"Estimated order {order:.2f} should be close to 2.0"
    )


def test_richardson_extrapolation_two_points():
    """Two-point Richardson extrapolation uses the expected order."""
    f_exact = 3.0e9
    C = 5e14
    dx_vals = np.array([2e-3, 1e-3])
    values = f_exact + C * dx_vals ** 2

    extrapolated, order = richardson_extrapolation(values, dx_vals, expected_order=2.0)

    assert abs(extrapolated - f_exact) / f_exact < 1e-6
    # With only 2 points, order defaults to expected_order
    assert abs(order - 2.0) < 1e-10


def test_convergence_order():
    """Estimated order should be ~2 for standard Yee (synthetic data)."""
    f_exact = 5.0e9
    C = 2e15

    dx_vals = np.array([4e-3, 3e-3, 2e-3, 1e-3])
    values = f_exact + C * dx_vals ** 2

    _, order = richardson_extrapolation(values, dx_vals, expected_order=2.0)

    assert abs(order - 2.0) < 0.1, (
        f"Convergence order {order:.2f} should be close to 2.0 for Yee"
    )


def test_convergence_order_cubic():
    """Estimated order should be ~3 for cubic error term."""
    f_exact = 1.0e9
    C = 1e18

    dx_vals = np.array([5e-3, 3e-3, 2e-3, 1e-3])
    values = f_exact + C * dx_vals ** 3

    _, order = richardson_extrapolation(values, dx_vals, expected_order=2.0)

    assert abs(order - 3.0) < 0.2, (
        f"Convergence order {order:.2f} should be close to 3.0"
    )


def test_convergence_plot():
    """Plot function creates a matplotlib figure."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    # Build a synthetic ConvergenceResult
    dx_vals = np.array([4e-3, 3e-3, 2e-3, 1e-3])
    f_exact = 5.0e9
    C = 2e15
    metrics = f_exact + C * dx_vals ** 2
    extrapolated, order = richardson_extrapolation(metrics, dx_vals)
    errors = np.abs(metrics - extrapolated) / abs(extrapolated)

    cr = ConvergenceResult(
        dx_values=dx_vals,
        metrics=metrics,
        extrapolated=extrapolated,
        order=order,
        errors=errors,
        results=[None] * len(dx_vals),
    )

    fig = cr.plot()
    assert fig is not None
    assert hasattr(fig, "savefig")  # quacks like a Figure

    # Also test passing an existing axes
    fig2, ax = plt.subplots()
    fig3 = cr.plot(ax=ax, title="Custom title")
    assert fig3 is fig2

    plt.close("all")


def test_convergence_summary():
    """Summary returns a formatted string."""
    dx_vals = np.array([3e-3, 2e-3, 1e-3])
    metrics = np.array([5.1e9, 5.03e9, 5.005e9])
    extrapolated = 5.0e9
    order = 2.0
    errors = np.abs(metrics - extrapolated) / abs(extrapolated)

    cr = ConvergenceResult(
        dx_values=dx_vals,
        metrics=metrics,
        extrapolated=extrapolated,
        order=order,
        errors=errors,
        results=[None] * 3,
    )

    text = cr.summary()
    assert "Mesh Convergence Summary" in text
    assert "Extrapolated value" in text
    assert "Convergence order" in text
    assert "2.00" in text


def test_convergence_too_few_dx_raises():
    """Passing fewer than 2 dx values should raise ValueError."""
    with pytest.raises(ValueError, match="at least 2"):
        convergence_study(
            sim_factory=_make_sim,
            dx_values=[1e-3],
            metric_fn=lambda r: 0.0,
        )


def test_richardson_too_few_values_raises():
    """Richardson extrapolation with fewer than 2 values should raise."""
    with pytest.raises(ValueError, match="at least 2"):
        richardson_extrapolation([1.0], [1e-3])


def test_richardson_mismatched_lengths_raises():
    """Mismatched values and dx_values should raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        richardson_extrapolation([1.0, 2.0], [1e-3, 2e-3, 3e-3])


def test_quick_convergence_runs():
    """quick_convergence should run without error on a simple simulation."""
    sim = Simulation(
        freq_max=5e9,
        domain=(0.03, 0.03, 0.03),
        boundary="pec",
    )
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")

    cr = quick_convergence(
        sim,
        metric="peak_field",
        dx_factors=[2.0, 1.5],
        n_steps=20,
        run_kwargs={"compute_s_params": False},
    )

    assert isinstance(cr, ConvergenceResult)
    assert len(cr.dx_values) == 2
    assert all(m > 0 for m in cr.metrics)
