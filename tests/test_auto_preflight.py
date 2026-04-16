"""Issue #66: forward()/optimize()/topology_optimize() auto-run preflight.

Before #66 users discovered physics violations only after minutes of GPU
compute, when mesh-quality warnings fired from inside ``run()``. Now the
differentiable entry points emit a single UserWarning up front with the
full preflight summary, unless ``skip_preflight=True``.
"""

from __future__ import annotations

import warnings

from rfx import Simulation, Box


def _make_sim_with_preflight_issue():
    """Partial PEC volume (3-cell extent) — matches the known-warning case
    from ``test_preflight_physics_thresholds::test_partial_pec_volume_warns``.
    """
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3,
                     cpml_layers=4)
    sim.add_source((0.01, 0.01, 0.002), "ez")
    sim.add_probe((0.01, 0.01, 0.01), "ez")
    sim.add(Box((0.005, 0.005, 0.005), (0.010, 0.010, 0.008)),
            material="pec")
    return sim


def _auto_preflight_warnings(caught):
    return [w for w in caught
            if issubclass(w.category, UserWarning)
            and "preflight found" in str(w.message)]


def test_forward_auto_preflight_fires():
    sim = _make_sim_with_preflight_issue()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.forward(n_steps=4)
    hits = _auto_preflight_warnings(caught)
    assert hits, (
        f"forward() should auto-fire preflight UserWarning; got: "
        f"{[str(w.message) for w in caught]}"
    )


def test_forward_skip_preflight_silences():
    sim = _make_sim_with_preflight_issue()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.forward(n_steps=4, skip_preflight=True)
    hits = _auto_preflight_warnings(caught)
    assert not hits, (
        f"skip_preflight=True must silence the auto-preflight warning; "
        f"got: {[str(w.message) for w in hits]}"
    )


def test_optimize_auto_preflight_fires():
    """optimize() goes through the same hook."""
    from rfx.optimize import DesignRegion, optimize

    sim = _make_sim_with_preflight_issue()
    region = DesignRegion(
        corner_lo=(0.006, 0.006, 0.008),
        corner_hi=(0.009, 0.009, 0.012),
        eps_range=(1.0, 4.0),
    )

    def obj(result):
        import jax.numpy as jnp
        return jnp.sum(result.time_series[:, 0] ** 2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimize(sim, region, obj, n_iters=1, n_steps=4, verbose=False)
    hits = _auto_preflight_warnings(caught)
    assert hits, (
        f"optimize() should auto-fire preflight UserWarning; got: "
        f"{[str(w.message) for w in caught]}"
    )
