"""Automatic mesh convergence study with Richardson extrapolation.

Runs a simulation at multiple cell sizes and estimates the grid-independent
result using Richardson extrapolation.  Useful for verifying that a design
is adequately resolved before trusting quantitative metrics.

Usage
-----
>>> result = convergence_study(
...     sim_factory=lambda dx: build_my_sim(dx=dx),
...     dx_values=[2e-3, 1.5e-3, 1e-3, 0.75e-3],
...     metric_fn=lambda r: float(r.find_resonances()[0].freq),
... )
>>> result.summary()
>>> result.plot()
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
        raise ImportError("matplotlib is required for convergence visualization")


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

def richardson_extrapolation(
    values,
    dx_values,
    expected_order: float = 2.0,
) -> tuple[float, float]:
    """Richardson extrapolation to estimate the grid-independent value.

    Uses the three finest grid points to estimate both the converged value
    and the effective convergence order.  Falls back to two-point
    extrapolation when only two values are available.

    Parameters
    ----------
    values : array-like
        Metric values at each cell size (must have >= 2 elements).
    dx_values : array-like
        Cell sizes corresponding to *values* (same length).
    expected_order : float
        Initial guess for the convergence order (default 2 for Yee).

    Returns
    -------
    (extrapolated, order) : (float, float)
        Estimated grid-independent value and observed convergence order.

    Notes
    -----
    When three or more values are available the order *p* is estimated by
    solving the ratio of successive differences:

        (f3 - f2) / (f2 - f1) = (h3^p - h2^p) / (h2^p - h1^p)

    where h = dx and f = metric.  A Newton iteration refines *p* from the
    initial guess.  With only two values the *expected_order* is used
    directly.
    """
    values = np.asarray(values, dtype=np.float64)
    dx_values = np.asarray(dx_values, dtype=np.float64)
    if len(values) < 2:
        raise ValueError("Need at least 2 values for Richardson extrapolation")
    if len(values) != len(dx_values):
        raise ValueError("values and dx_values must have the same length")

    # Sort by decreasing dx (coarsest first) so indices go coarse->fine
    order_idx = np.argsort(dx_values)[::-1]
    h = dx_values[order_idx]
    f = values[order_idx]

    if len(f) >= 3:
        # Use the three finest (last three in coarse->fine order)
        h1, h2, h3 = h[-3], h[-2], h[-1]
        f1, f2, f3 = f[-3], f[-2], f[-1]

        # Estimate order p from the ratio of successive differences
        diff21 = f2 - f1
        diff32 = f3 - f2

        if abs(diff21) < 1e-30 or abs(diff32) < 1e-30:
            # Metrics are essentially identical — already converged
            return float(f[-1]), expected_order

        ratio = diff32 / diff21
        h2 / h1
        h3 / h2

        # Newton iteration to solve for p:
        # g(p) = (h3^p - h2^p) / (h2^p - h1^p) - ratio = 0
        p = expected_order
        for _ in range(50):
            h1p = h1 ** p
            h2p = h2 ** p
            h3p = h3 ** p
            num = h3p - h2p
            den = h2p - h1p
            if abs(den) < 1e-30:
                break
            g = num / den - ratio
            # Numerical derivative
            dp = 1e-6
            h1pd = h1 ** (p + dp)
            h2pd = h2 ** (p + dp)
            h3pd = h3 ** (p + dp)
            num_d = h3pd - h2pd
            den_d = h2pd - h1pd
            if abs(den_d) < 1e-30:
                break
            g_d = (num_d / den_d - ratio - g) / dp
            if abs(g_d) < 1e-30:
                break
            p_new = p - g / g_d
            if abs(p_new - p) < 1e-10:
                p = p_new
                break
            p = p_new

        # Clamp order to reasonable range
        p = float(np.clip(p, 0.5, 10.0))

        # Extrapolate using two finest points
        r = (h3 / h2) ** p
        extrapolated = float((f3 - r * f2) / (1.0 - r))
    else:
        # Two-point extrapolation with assumed order
        h1, h2 = h[0], h[1]
        f1, f2 = f[0], f[1]
        p = expected_order
        r = (h2 / h1) ** p
        extrapolated = float((f2 - r * f1) / (1.0 - r))

    return extrapolated, float(p)


# ---------------------------------------------------------------------------
# ConvergenceResult
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """Structured result from a mesh convergence study.

    Attributes
    ----------
    dx_values : ndarray
        Cell sizes tested (coarse to fine).
    metrics : ndarray
        Metric value at each cell size.
    extrapolated : float
        Richardson-extrapolated grid-independent value.
    order : float
        Estimated convergence order.
    errors : ndarray
        Relative error of each metric w.r.t. the extrapolated value.
    results : list
        Raw simulation results (one per dx).
    """

    dx_values: np.ndarray
    metrics: np.ndarray
    extrapolated: float
    order: float
    errors: np.ndarray
    results: list

    def plot(self, *, ax=None, title: str = "Mesh Convergence"):
        """Plot convergence: metric vs dx and log-log error vs dx.

        Parameters
        ----------
        ax : matplotlib Axes or None
            If provided, only the metric-vs-dx plot is drawn on this axes.
            If None, a two-panel figure is created.
        title : str
            Figure/axes title.

        Returns
        -------
        matplotlib Figure
        """
        _require_mpl()
        import matplotlib
        matplotlib.use("Agg")

        if ax is not None:
            ax.plot(self.dx_values * 1e3, self.metrics, "o-", label="Simulated")
            ax.axhline(self.extrapolated, color="r", ls="--", alpha=0.7,
                       label=f"Extrapolated = {self.extrapolated:.6g}")
            ax.set_xlabel("Cell size dx (mm)")
            ax.set_ylabel("Metric")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            return ax.figure

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: metric vs dx
        ax1.plot(self.dx_values * 1e3, self.metrics, "o-", color="C0",
                 label="Simulated")
        ax1.axhline(self.extrapolated, color="r", ls="--", alpha=0.7,
                     label=f"Extrapolated = {self.extrapolated:.6g}")
        ax1.set_xlabel("Cell size dx (mm)")
        ax1.set_ylabel("Metric")
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right: log-log error
        nonzero = self.errors > 0
        if np.any(nonzero):
            ax2.loglog(self.dx_values[nonzero] * 1e3,
                       self.errors[nonzero] * 100, "s-", color="C1")
            # Reference slope
            if np.sum(nonzero) >= 2:
                dx_ref = self.dx_values[nonzero]
                err_fit = self.errors[nonzero][-1] * (dx_ref / dx_ref[-1]) ** self.order
                ax2.loglog(dx_ref * 1e3, err_fit * 100, "k--", alpha=0.4,
                           label=f"Order {self.order:.1f} slope")
                ax2.legend()
        ax2.set_xlabel("Cell size dx (mm)")
        ax2.set_ylabel("Relative error (%)")
        ax2.set_title("Convergence Rate")
        ax2.grid(True, alpha=0.3, which="both")

        fig.tight_layout()
        return fig

    def summary(self) -> str:
        """Return a formatted convergence summary table.

        Returns
        -------
        str
            Multi-line summary table.
        """
        lines = ["Mesh Convergence Summary", "=" * 50]
        lines.append(f"{'dx (mm)':>10s}  {'Metric':>14s}  {'Rel. Error':>12s}")
        lines.append("-" * 50)
        for dx, m, e in zip(self.dx_values, self.metrics, self.errors):
            lines.append(f"{dx*1e3:10.4f}  {m:14.6g}  {e*100:11.4f}%")
        lines.append("-" * 50)
        lines.append(f"Extrapolated value : {self.extrapolated:.6g}")
        lines.append(f"Convergence order  : {self.order:.2f}")
        text = "\n".join(lines)
        print(text)
        return text


# ---------------------------------------------------------------------------
# Main convergence study
# ---------------------------------------------------------------------------

def convergence_study(
    sim_factory: Callable,
    dx_values,
    metric_fn: Callable,
    *,
    n_steps: int | None = None,
    until_decay: float | None = None,
    run_kwargs: dict | None = None,
    expected_order: float = 2.0,
) -> ConvergenceResult:
    """Run a mesh convergence study.

    Parameters
    ----------
    sim_factory : callable(dx) -> Simulation
        Builds a simulation at the given cell size.
    dx_values : list of float
        Cell sizes to test (coarse to fine).
    metric_fn : callable(Result) -> float
        Extracts the metric of interest (e.g., resonance frequency).
    n_steps : int or None
        If given, passed as ``n_steps`` to each ``sim.run()``.
    until_decay : float or None
        If given, passed as ``until_decay`` to each ``sim.run()``.
    run_kwargs : dict or None
        Additional keyword arguments forwarded to ``sim.run()``.
    expected_order : float
        Initial guess for convergence order (default 2 for standard Yee).

    Returns
    -------
    ConvergenceResult
        Container with dx_values, metrics, extrapolated value, order,
        relative errors, and raw results.
    """
    dx_values = np.asarray(dx_values, dtype=np.float64).ravel()
    if len(dx_values) < 2:
        raise ValueError("Need at least 2 dx values for a convergence study")

    # Sort coarse to fine
    sort_idx = np.argsort(dx_values)[::-1]
    dx_values = dx_values[sort_idx]

    kw: dict[str, Any] = dict(run_kwargs) if run_kwargs else {}
    if n_steps is not None:
        kw["n_steps"] = n_steps
    if until_decay is not None:
        kw["until_decay"] = until_decay

    results = []
    metrics = []
    for dx in dx_values:
        sim = sim_factory(float(dx))
        result = sim.run(**kw)
        results.append(result)
        metrics.append(float(metric_fn(result)))

    metrics_arr = np.asarray(metrics, dtype=np.float64)

    # Richardson extrapolation
    extrapolated, order = richardson_extrapolation(
        metrics_arr, dx_values, expected_order=expected_order,
    )

    # Relative errors w.r.t. extrapolated value
    if abs(extrapolated) > 0:
        errors = np.abs(metrics_arr - extrapolated) / abs(extrapolated)
    else:
        errors = np.abs(metrics_arr - extrapolated)

    return ConvergenceResult(
        dx_values=dx_values,
        metrics=metrics_arr,
        extrapolated=extrapolated,
        order=order,
        errors=errors,
        results=results,
    )


# ---------------------------------------------------------------------------
# Quick convenience wrapper
# ---------------------------------------------------------------------------

def quick_convergence(
    sim: object,
    *,
    metric: str | Callable = "resonance",
    dx_factors: list[float] | None = None,
    n_steps: int | None = None,
    until_decay: float | None = None,
    run_kwargs: dict | None = None,
) -> ConvergenceResult:
    """Run a convergence study on an existing Simulation with dx scaling.

    Creates copies of *sim* at different cell sizes by scaling the base dx
    and re-running.  This is a convenience wrapper around
    ``convergence_study`` for quick checks.

    Parameters
    ----------
    sim : rfx.api.Simulation
        A fully configured simulation.  Its ``_dx`` (or auto-computed dx)
        is used as the reference cell size scaled by each factor.
    metric : str or callable
        ``"resonance"`` extracts the first resonance frequency.
        ``"peak_field"`` extracts the peak probe amplitude.
        A callable ``metric(Result) -> float`` is used directly.
    dx_factors : list of float or None
        Scaling factors applied to the base dx.  Default ``[2.0, 1.5, 1.0, 0.75]``.
        Values > 1 mean coarser; < 1 mean finer.
    n_steps : int or None
        Forwarded to ``sim.run()``.
    until_decay : float or None
        Forwarded to ``sim.run()``.
    run_kwargs : dict or None
        Additional keyword arguments forwarded to ``sim.run()``.

    Returns
    -------
    ConvergenceResult
    """
    if dx_factors is None:
        dx_factors = [2.0, 1.5, 1.0, 0.75]

    # Resolve the base dx from the simulation
    from rfx.auto_config import auto_configure
    base_dx = getattr(sim, "_dx", None)
    if base_dx is None:
        # Auto-compute the default dx the same way Simulation would
        C0 = 299792458.0
        lam_min = C0 / sim._freq_max
        base_dx = lam_min / 20.0  # standard Yee rule of thumb
    dx_values = [base_dx * f for f in dx_factors]

    # Build the metric callable
    if isinstance(metric, str):
        if metric == "resonance":
            def metric_fn(result):
                modes = result.find_resonances()
                if not modes:
                    return float("nan")
                return float(modes[0].freq)
        elif metric == "peak_field":
            def metric_fn(result):
                ts = np.asarray(result.time_series)
                return float(np.max(np.abs(ts)))
        else:
            raise ValueError(
                f"Unknown metric {metric!r}. "
                f"Use 'resonance', 'peak_field', or pass a callable."
            )
    else:
        metric_fn = metric

    # Build a sim_factory that clones the configuration with a new dx
    def sim_factory(dx):
        from rfx.api import Simulation
        new_sim = Simulation(
            freq_max=sim._freq_max,
            domain=sim._domain,
            boundary=sim._boundary,
            cpml_layers=sim._cpml_layers,
            dx=dx,
            mode=sim._mode,
            dz_profile=sim._dz_profile,
            precision=sim._precision,
        )
        # Copy registered items
        new_sim._materials = dict(sim._materials)
        new_sim._geometry = list(sim._geometry)
        new_sim._ports = list(sim._ports)
        new_sim._probes = list(sim._probes)
        new_sim._thin_conductors = list(sim._thin_conductors)
        new_sim._ntff = sim._ntff
        new_sim._tfsf = sim._tfsf
        new_sim._dft_planes = list(sim._dft_planes)
        new_sim._waveguide_ports = list(sim._waveguide_ports)
        new_sim._periodic_axes = sim._periodic_axes
        new_sim._refinement = sim._refinement
        new_sim._lumped_rlc = list(sim._lumped_rlc)
        new_sim._floquet_ports = list(sim._floquet_ports)
        return new_sim

    return convergence_study(
        sim_factory=sim_factory,
        dx_values=dx_values,
        metric_fn=metric_fn,
        n_steps=n_steps,
        until_decay=until_decay,
        run_kwargs=run_kwargs,
    )
