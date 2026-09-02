"""Tests for the parametric sweep API."""

import numpy as np
import pytest

from rfx.api import Simulation, Result
from rfx.geometry.csg import Box
from rfx.sweep import parametric_sweep, SweepResult, plot_sweep


# ---------------------------------------------------------------------------
# Helper: lightweight sim factory
# ---------------------------------------------------------------------------

def _make_sim(eps_r_value):
    """Create a minimal PEC-cavity simulation with a dielectric fill."""
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), boundary="pec")
    sim.add_material("fill", eps_r=eps_r_value)
    sim.add(Box((0.005, 0.005, 0.005), (0.025, 0.025, 0.025)), material="fill")
    sim.add_port((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.02, 0.02, 0.02), "ez")
    return sim


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sweep_runs_multiple_sims():
    """parametric_sweep should run one simulation per parameter value."""
    values = [2.0, 4.0, 6.0]
    sr = parametric_sweep(
        _make_sim,
        param_name="eps_r",
        param_values=values,
        n_steps=30,
        run_kwargs={"compute_s_params": False},
    )

    assert isinstance(sr, SweepResult)
    assert len(sr.results) == 3
    # Each result should have a non-zero time series
    for r in sr.results:
        assert isinstance(r, Result)
        peak = float(np.max(np.abs(np.asarray(r.time_series))))
        assert peak > 0, "Each simulation should produce non-zero probe data"


def test_sweep_result_has_all_values():
    """SweepResult should store param_name and all param_values."""
    values = [1.5, 3.0]
    sr = parametric_sweep(
        _make_sim,
        param_name="eps_r",
        param_values=values,
        n_steps=20,
        run_kwargs={"compute_s_params": False},
    )

    assert sr.param_name == "eps_r"
    np.testing.assert_array_equal(sr.param_values, np.array([1.5, 3.0]))
    assert len(sr.results) == len(values)

    # peak_field extractor should return one value per sweep point
    pf = sr.peak_field()
    assert pf.shape == (2,)
    assert np.all(pf > 0)


def test_sweep_plot_creates_figure():
    """plot_sweep should return a matplotlib Figure without error."""
    pytest.importorskip("matplotlib")

    values = [2.0, 4.0]
    sr = parametric_sweep(
        _make_sim,
        param_name="eps_r",
        param_values=values,
        n_steps=20,
        run_kwargs={"compute_s_params": False},
    )

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    # Built-in metric: peak_field (s11_min_db needs s_params)
    fig = plot_sweep(sr, metric="peak_field")
    assert fig is not None
    assert hasattr(fig, "savefig")  # quacks like a Figure

    # Custom callable metric
    fig2 = plot_sweep(sr, metric=lambda r: float(np.max(np.abs(np.asarray(r.time_series)))))
    assert fig2 is not None

    plt.close("all")


def test_sweep_empty_values_raises():
    """Passing empty param_values should raise ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        parametric_sweep(_make_sim, "eps_r", [])


def test_sweep_unknown_metric_raises():
    """plot_sweep should raise on unknown metric string."""
    pytest.importorskip("matplotlib")

    sr = SweepResult(results=[], param_name="x", param_values=np.array([]))
    with pytest.raises(ValueError, match="Unknown metric"):
        plot_sweep(sr, metric="nonexistent_metric")
