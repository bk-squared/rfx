"""Parametric sweep API for design-space exploration.

Provides a simple ``parametric_sweep`` function that runs multiple simulations
across a range of parameter values, collecting results into a structured
``SweepResult``.  A companion ``plot_sweep`` visualizes how a chosen metric
varies with the swept parameter.

Design note
-----------
For geometry parameter sweeps each simulation may have different grid shapes,
so ``jax.vmap`` cannot batch the full FDTD loop.  This module therefore uses
the *sim_factory* pattern which creates and runs each simulation sequentially.
For material-only sweeps (``eps_r``, ``sigma``, etc.) where the grid geometry
is identical, ``jax.vmap`` over the material arrays is possible in principle
and may be added as a future fast-path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _require_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for sweep visualization")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    """Structured result from a parametric sweep.

    Attributes
    ----------
    results : list
        One ``rfx.api.Result`` (or equivalent) per parameter value.
    param_name : str
        Label for the swept parameter.
    param_values : ndarray
        1-D array of parameter values that were swept.
    """

    results: list
    param_name: str
    param_values: np.ndarray

    # -- convenience extractors ------------------------------------------------

    def s11_min_db(self) -> np.ndarray:
        """Return the minimum S11 (dB) for each sweep point.

        Works with results that have ``s_params`` (n_ports, n_ports, n_freqs).
        """
        vals = []
        for r in self.results:
            sp = getattr(r, "s_params", None)
            if sp is not None:
                s11_mag = np.abs(np.asarray(sp)[0, 0, :])
                s11_db = 20.0 * np.log10(np.maximum(s11_mag, 1e-30))
                vals.append(float(np.min(s11_db)))
            else:
                vals.append(np.nan)
        return np.asarray(vals)

    def peak_field(self) -> np.ndarray:
        """Return peak absolute value of time-series for each sweep point."""
        vals = []
        for r in self.results:
            ts = getattr(r, "time_series", None)
            if ts is not None:
                vals.append(float(np.max(np.abs(np.asarray(ts)))))
            else:
                vals.append(np.nan)
        return np.asarray(vals)


# ---------------------------------------------------------------------------
# Main sweep function
# ---------------------------------------------------------------------------

def parametric_sweep(
    sim_factory: Callable,
    param_name: str,
    param_values,
    *,
    n_steps: int | None = None,
    until_decay: float | None = None,
    run_kwargs: dict | None = None,
) -> SweepResult:
    """Run multiple simulations with different parameter values.

    Parameters
    ----------
    sim_factory : callable
        ``sim_factory(param_value) -> Simulation``.  Called once per value
        in *param_values* to create a fully configured ``Simulation`` object.
    param_name : str
        Human-readable name of the parameter being swept (used for labels).
    param_values : array-like
        1-D sequence of parameter values to sweep over.
    n_steps : int or None
        If given, passed as ``n_steps`` to each ``sim.run()``.
    until_decay : float or None
        If given, passed as ``until_decay`` to each ``sim.run()``.
    run_kwargs : dict or None
        Additional keyword arguments forwarded to ``sim.run()``.

    Returns
    -------
    SweepResult
        Container with ``.results``, ``.param_values``, and ``.param_name``.
    """
    param_values = np.asarray(param_values).ravel()
    if len(param_values) == 0:
        raise ValueError("param_values must not be empty")

    kw: dict[str, Any] = dict(run_kwargs) if run_kwargs else {}
    if n_steps is not None:
        kw["n_steps"] = n_steps
    if until_decay is not None:
        kw["until_decay"] = until_decay

    results = []
    for val in param_values:
        sim = sim_factory(float(val))
        result = sim.run(**kw)
        results.append(result)

    return SweepResult(
        results=results,
        param_name=param_name,
        param_values=param_values,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_BUILTIN_METRICS = {
    "s11_min_db": ("s11_min_db", "Min S11 (dB)"),
    "peak_field": ("peak_field", "Peak |E| (V/m)"),
}


def plot_sweep(
    sweep_result: SweepResult,
    metric: str = "s11_min_db",
    *,
    title: str | None = None,
    ylabel: str | None = None,
    marker: str = "o-",
) -> object:
    """Visualize how a metric varies across the swept parameter.

    Parameters
    ----------
    sweep_result : SweepResult
        Output of ``parametric_sweep``.
    metric : str or callable
        Built-in metric name (``"s11_min_db"`` or ``"peak_field"``), or a
        callable ``metric(result) -> float`` applied to each result.
    title : str or None
        Plot title.  Defaults to ``"<param_name> sweep"`` .
    ylabel : str or None
        Y-axis label.  Defaults to the metric name.
    marker : str
        Matplotlib line/marker style (default ``"o-"``).

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()

    if callable(metric):
        y = np.asarray([metric(r) for r in sweep_result.results])
        metric_label = getattr(metric, "__name__", "metric")
    elif metric in _BUILTIN_METRICS:
        method_name, metric_label = _BUILTIN_METRICS[metric]
        y = getattr(sweep_result, method_name)()
    else:
        raise ValueError(
            f"Unknown metric {metric!r}. "
            f"Use one of {list(_BUILTIN_METRICS)} or pass a callable."
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_result.param_values, y, marker)
    ax.set_xlabel(sweep_result.param_name)
    ax.set_ylabel(ylabel or metric_label)
    ax.set_title(title or f"{sweep_result.param_name} sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
