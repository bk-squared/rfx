"""Simulation-time breakdown utilities.

Profiles the main phases of a Simulation.forward or Simulation.run call so
users can see where wall-clock time is going (compile vs scan, NU grid
construction, material assembly, etc.) without rolling their own
stopwatch around every call.

Typical usage::

    from rfx.profiling import profile_forward
    report = profile_forward(sim, n_steps=2000, checkpoint_every=100)
    print(report)  # ASCII table

    # Or programmatic:
    report["scan_exec"]  # seconds
    report["compile"]    # seconds
"""

from __future__ import annotations

import time
from typing import Any

import jax


def _sync(x):
    """Block on any JAX arrays in x so wall-time measurements don't
    include async dispatch."""
    jax.block_until_ready(x)


def profile_forward(sim, *, n_steps: int, warmup_trace: bool = True,
                    **forward_kwargs) -> dict:
    """Time the phases of ``sim.forward(n_steps=...)``.

    Returns a dict with wall-clock seconds for each phase:

      - ``grid_build``   : ``_build_(non)uniform_grid``
      - ``compile``      : first ``forward`` call (includes JAX trace + JIT)
      - ``scan_exec``    : second ``forward`` call (amortised — compile cached)
      - ``total``        : end-to-end wall time of the profiling loop

    Parameters
    ----------
    sim : Simulation
    n_steps : int
        Timesteps per call.
    warmup_trace : bool
        If True, a third call is issued to confirm compile cache hit;
        its time is also returned as ``verify_exec``.
    **forward_kwargs
        Forwarded to ``sim.forward``.
    """
    report: dict[str, Any] = {}
    t0 = time.time()

    # 1. Grid build — NU or uniform, whichever this sim uses.
    g0 = time.time()
    if sim._dz_profile is not None:
        sim._build_nonuniform_grid()
    else:
        sim._build_grid()
    report["grid_build"] = time.time() - g0

    # 2. First forward call — trace + compile + one scan execution.
    c0 = time.time()
    fr = sim.forward(n_steps=n_steps, **forward_kwargs)
    _sync(fr.time_series)
    report["compile"] = time.time() - c0

    # 3. Second forward call — pure scan execution (compile cached).
    s0 = time.time()
    fr2 = sim.forward(n_steps=n_steps, **forward_kwargs)
    _sync(fr2.time_series)
    report["scan_exec"] = time.time() - s0

    if warmup_trace:
        v0 = time.time()
        fr3 = sim.forward(n_steps=n_steps, **forward_kwargs)
        _sync(fr3.time_series)
        report["verify_exec"] = time.time() - v0

    report["total"] = time.time() - t0
    report["n_steps"] = n_steps
    try:
        g = (sim._build_nonuniform_grid() if sim._dz_profile is not None
             else sim._build_grid())
        report["cells"] = int(g.nx * g.ny * g.nz)
    except Exception:
        report["cells"] = None
    return report


def format_report(report: dict) -> str:
    """Render a dict from ``profile_forward`` as an ASCII table."""
    keys = ["grid_build", "compile", "scan_exec"]
    if "verify_exec" in report:
        keys.append("verify_exec")
    keys.append("total")
    lines = []
    cells = report.get("cells")
    n_steps = report.get("n_steps")
    header = f"rfx profile — n_steps={n_steps}"
    if cells:
        header += f"  cells={cells:,}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in keys:
        v = report.get(k)
        if v is None:
            continue
        pct = 100.0 * v / report["total"] if report.get("total") else 0.0
        lines.append(f"{k:<14s}  {v:>7.2f}s  ({pct:>5.1f}%)")
    if "scan_exec" in report and report["scan_exec"] > 0 and cells:
        cps = n_steps * cells / report["scan_exec"] / 1e6
        lines.append(f"throughput    {cps:>7.2f} Mcell-steps/s")
    return "\n".join(lines)
