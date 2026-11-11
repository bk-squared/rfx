"""Stage B: a single SI normalization guard, no relative-map underflow guard."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import LorentzPoleMap, RelativeStrength
from rfx.errors import AccuracyWarning
from tests.unit.materials.test_pole_map_gradients import BINS, KAPPA, parameters, scene
from tests.unit.materials.test_pole_map_runs import record


def objective_for(graded, relative):
    # Construct the declared mesh before recording the reverse pass warnings;
    # its existing adjacent-cell-ratio AccuracyWarning is independent of poles.
    sim, grid = scene(parameters(), graded)
    def objective(params):
        log_kappa, slope = params
        if relative:
            kappa = RelativeStrength(float(KAPPA), lambda x,y,z:
                jnp.exp(log_kappa-jnp.log(KAPPA))*(1+slope*(x-.015)/.0093))
        else:
            kappa = lambda x,y,z: jnp.exp(log_kappa)*(1+slope*(x-.015)/.0093)
        sim._materials['plasma'].lorentz_poles[:] = [LorentzPoleMap(0., 2e9, kappa)]
        result = sim.forward(n_steps=200, checkpoint=True, skip_preflight=True)
        trace = jnp.asarray(result.time_series).reshape(200, -1)[:, 0]
        kernel = jnp.exp(-2j*jnp.pi*jnp.asarray(BINS)[:, None]*jnp.arange(200)[None, :]*grid.dt)*grid.dt
        return jnp.sum(jnp.abs(kernel@trace)**2)
    return objective


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b6_2_si_guard(monkeypatch, graded):
    objective = objective_for(graded, False)
    monkeypatch.setenv('RFX_POLE_UNDERFLOW_CHECK', '1')
    with pytest.warns(AccuracyWarning) as caught:
        checked = jax.block_until_ready(jax.value_and_grad(objective)(parameters()))
    warnings_found = [w.message for w in caught if issubclass(w.category, AccuracyWarning)]
    record(f'B6.2-{graded}', dict(warnings=len(warnings_found), counts=[getattr(w, 'count', None) for w in warnings_found],
           objective=float(checked[0]), gradient=np.asarray(checked[1]).tolist()))
    assert len(warnings_found) == 1
    monkeypatch.delenv('RFX_POLE_UNDERFLOW_CHECK')
    unchecked = jax.block_until_ready(jax.value_and_grad(objective)(parameters()))
    for a, b in zip(jax.tree.leaves(checked), jax.tree.leaves(unchecked)):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize('graded', [False, True], ids=['uniform', 'graded'])
def test_b6_3_relative_guard(monkeypatch, graded):
    objective = objective_for(graded, True)
    monkeypatch.setenv('RFX_POLE_UNDERFLOW_CHECK', '1')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = jax.block_until_ready(jax.value_and_grad(objective)(parameters()))
    found = [w.message for w in caught if issubclass(w.category, AccuracyWarning)]
    record(f'B6.3-{graded}', dict(warnings=len(found), objective=float(result[0]), gradient=np.asarray(result[1]).tolist()))
    assert not found, found
