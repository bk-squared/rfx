"""A traced Box refusal makes optimize retry eagerly with the same losses."""

import sys
import warnings

import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation
from rfx.errors import ConfigurationWarning
from rfx.optimize import DesignRegion, optimize


def _model():
    sim = Simulation(freq_max=10e9, domain=(.008, .007, .006),
                     dx=.001, boundary="pec")
    sim.add_source((.002, .002, .002), "ez")
    sim.add_probe((.003, .002, .002), "ez")
    return sim


def test_optimize_traced_box_falls_back_to_eager_losses():
    def objective(result):
        # As in test_optimize_wire_conductor_mesh, assemble a second model
        # inside the objective. This extent is traced by outer jit, but is
        # concrete during eager AD: differentiating Box dimensions themselves
        # is separately refused by forward().
        extent = jnp.asarray(.004, dtype=jnp.float32)
        nested = _model()
        nested.add_material("dielectric", eps_r=3.7)
        nested.add(Box((.001, .001, .001), (extent, .003, .003)),
                   material="dielectric")
        record = nested.forward(n_steps=4, skip_preflight=True).time_series
        return jnp.sum(result.time_series**2) + jnp.sum(record**2)

    region = DesignRegion((.001, .001, .001), (.004, .003, .003))
    histories = {}
    for jit in (False, True):
        sim = _model()
        mesh = sim.freeze_mesh().shape
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = optimize(sim, region, objective, n_iters=2, n_steps=4,
                              jit=jit, verbose=False, skip_preflight=True)
        history = np.asarray(result.loss_history)
        fallback = [w for w in caught
                    if "every iteration now recompiles" in str(w.message)]
        print(f"row=jit-{jit} loss_history={history.tolist()} "
              f"fallback_warnings={len(fallback)} expected={int(jit)} "
              f"iterations=2 steps=4 mesh={mesh} dx_m=0.001",
              file=sys.stderr)
        assert history.shape == (2,), "optimize must complete both iterations"
        assert np.isfinite(history).all(), "loss history must be finite"
        assert np.any(history != 0), "loss comparison must not be all zero"
        assert len(fallback) == int(jit), "fallback must warn once only for jit=True"
        if jit:
            assert fallback[0].category is ConfigurationWarning
            assert "TracedShapeDimensionError" in str(fallback[0].message)
            print(f"row=jit-True warning={fallback[0].message}", file=sys.stderr)
        histories[jit] = history

    # Existing fallback/eager parity bar (same eager computation):
    # tests/unit/autodiff/test_optimize_jit_default.py:116, exact equality.
    difference = np.max(np.abs(histories[True] - histories[False]))
    print(f"row=fallback-vs-eager max_abs_loss_difference={difference:.17g} bar=0",
          file=sys.stderr)
    np.testing.assert_array_equal(histories[True], histories[False],
                                  err_msg="traced Box fallback must preserve eager losses")
