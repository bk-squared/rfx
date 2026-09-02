"""Issue #42: progressive_optimize orchestrator.

Runs optimize() at a sequence of increasing resolutions, upsampling the
latent between stages via ``jax.image.resize``. Coarse stages produce a
fast loss-landscape scan; finer stages refine on top.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx import Simulation
from rfx.optimize import (
    DesignRegion,
    ProgressiveOptimizeResult,
    ProgressiveStage,
    progressive_optimize,
)


def _make_factory(domain=(0.02, 0.02, 0.02)):
    """Return a sim_factory(dx) -> Simulation closure."""
    def factory(dx):
        sim = Simulation(
            freq_max=10e9, domain=domain, dx=dx, cpml_layers=4,
        )
        sim.add_source((domain[0] / 2, domain[1] / 2, 0.004), "ez")
        sim.add_probe((domain[0] / 2, domain[1] / 2, 0.010), "ez")
        return sim
    return factory


def _loss(result):
    return jnp.sum(result.time_series[:, 0] ** 2)


def test_two_stage_progressive_runs():
    factory = _make_factory()
    region = DesignRegion(
        corner_lo=(0.008, 0.008, 0.009),
        corner_hi=(0.012, 0.012, 0.012),
        eps_range=(1.0, 4.0),
    )
    schedule = [
        ProgressiveStage(dx=1.0e-3, n_iters=2, lr=0.05, n_steps=15),
        ProgressiveStage(dx=0.5e-3, n_iters=2, lr=0.02, n_steps=15),
    ]
    result = progressive_optimize(
        factory, region, _loss, schedule,
        verbose=False, skip_preflight=True,
    )
    assert isinstance(result, ProgressiveOptimizeResult)
    assert len(result.stages) == 2
    assert len(result.loss_history) == 4
    for v in result.loss_history:
        assert np.isfinite(v), f"stage loss not finite: {v}"
    assert result.stage_boundaries == [0, 2, 4]


def test_latent_shape_grows_between_stages():
    """Finest stage's latent must have larger shape than coarse stage's."""
    factory = _make_factory()
    region = DesignRegion(
        corner_lo=(0.008, 0.008, 0.009),
        corner_hi=(0.012, 0.012, 0.012),
        eps_range=(1.0, 4.0),
    )
    schedule = [
        ProgressiveStage(dx=1.0e-3, n_iters=1, lr=0.05, n_steps=15),
        ProgressiveStage(dx=0.5e-3, n_iters=1, lr=0.02, n_steps=15),
    ]
    result = progressive_optimize(
        factory, region, _loss, schedule,
        verbose=False, skip_preflight=True,
    )
    coarse_shape = result.stages[0].latent.shape
    fine_shape = result.stages[1].latent.shape
    assert all(f >= c for f, c in zip(fine_shape, coarse_shape)), (
        f"expected fine shape {fine_shape} >= coarse shape {coarse_shape} "
        "after 2x refinement"
    )
    assert any(f > c for f, c in zip(fine_shape, coarse_shape)), (
        f"expected at least one axis to grow: coarse={coarse_shape} "
        f"fine={fine_shape}"
    )


def test_empty_schedule_raises():
    factory = _make_factory()
    region = DesignRegion(
        corner_lo=(0.008, 0.008, 0.009),
        corner_hi=(0.012, 0.012, 0.012),
        eps_range=(1.0, 4.0),
    )
    import pytest
    with pytest.raises(ValueError, match="schedule must be non-empty"):
        progressive_optimize(factory, region, _loss, [])


# ---------------------------------------------------------------------------
# Issue #69 — distributed=True thread-through
# ---------------------------------------------------------------------------

def _make_nu_factory(domain=(0.01, 0.01, 0.01), dx=5e-3):
    """NU-mesh sim_factory (distributed=True requires a non-uniform profile)."""
    import numpy as np

    def factory(stage_dx):
        # dz_profile length is stage-driven; use stage_dx as the cell size.
        nz = max(3, int(round(domain[2] / stage_dx)))
        dz_profile = np.full(nz, stage_dx, dtype=np.float64)
        sim = Simulation(
            freq_max=5e9,
            domain=(domain[0], domain[1], float(dz_profile.sum())),
            dx=stage_dx,
            cpml_layers=2,
            boundary="pec",
            dz_profile=dz_profile,
        )
        sim.add_source(
            (domain[0] / 2, domain[1] / 2, float(dz_profile[:2].sum())),
            "ez",
        )
        sim.add_probe(
            (domain[0] / 2, domain[1] / 2, float(dz_profile[:-1].sum())),
            "ez",
        )
        return sim

    return factory


def test_progressive_optimize_distributed_true_runs_on_nu_mesh():
    """Issue #69: `distributed=True` threads into every inner `forward()`.

    Minimum smoke: 2-stage schedule with `distributed=True` on a NU mesh
    must produce a finite loss history matching the shape of the
    non-distributed run (same loss_history length, both finite).
    """
    import jax
    import pytest

    if len(jax.devices()) < 2:
        pytest.skip("requires 2 devices (conftest XLA_FLAGS)")

    factory = _make_nu_factory()
    region = DesignRegion(
        corner_lo=(0.003, 0.003, 0.004),
        corner_hi=(0.007, 0.007, 0.006),
        eps_range=(1.0, 4.0),
    )
    schedule = [
        ProgressiveStage(dx=2.0e-3, n_iters=1, lr=0.05, n_steps=10),
        ProgressiveStage(dx=1.0e-3, n_iters=1, lr=0.02, n_steps=10),
    ]

    baseline = progressive_optimize(
        factory, region, _loss, schedule,
        verbose=False, skip_preflight=True, distributed=False,
    )
    distributed = progressive_optimize(
        factory, region, _loss, schedule,
        verbose=False, skip_preflight=True, distributed=True,
    )

    assert len(baseline.loss_history) == len(distributed.loss_history) == 2
    assert all(np.isfinite(v) for v in distributed.loss_history), (
        f"distributed loss not finite: {distributed.loss_history}"
    )
    assert all(np.isfinite(v) for v in baseline.loss_history), (
        f"baseline loss not finite: {baseline.loss_history}"
    )


def test_progressive_optimize_distributed_rejects_uniform_mesh():
    """Issue #69: `distributed=True` on a uniform mesh must raise up front.

    Inherits the reject rule from `Simulation.forward(distributed=True)`
    (NU-only in v1.6.2 per DP3).
    """
    import jax
    import pytest

    if len(jax.devices()) < 2:
        pytest.skip("requires 2 devices (conftest XLA_FLAGS)")

    factory = _make_factory()  # uniform mesh
    region = DesignRegion(
        corner_lo=(0.008, 0.008, 0.009),
        corner_hi=(0.012, 0.012, 0.012),
        eps_range=(1.0, 4.0),
    )
    schedule = [
        ProgressiveStage(dx=1.0e-3, n_iters=1, lr=0.05, n_steps=10),
    ]
    with pytest.raises(NotImplementedError, match="non-uniform"):
        progressive_optimize(
            factory, region, _loss, schedule,
            verbose=False, skip_preflight=True, distributed=True,
        )
