"""ADI default-current differentiation and traced conductivity refusal (#1373)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation


def _simulation(mode, kind=None):
    sim = Simulation(freq_max=1e10, domain=(.01, .01, .01 if mode == "3d" else .001),
                     dx=.001, boundary="pec", mode=mode, solver="adi", adi_cfl_factor=1)
    z = .005 if mode == "3d" else 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sim.add_source((.005, .005, z), "ez", waveform=GaussianPulse(f0=5e9),
                       amplitude_kind=kind)
    sim.add_probe((.006, .005, z), "ez")
    return sim


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
def test_default_adi_eps_gradient_matches_richardson(mode):
    sim = _simulation(mode)
    shape = sim._build_grid().shape

    def objective(eps):
        trace = sim.forward(eps_override=jnp.ones(shape, dtype=jnp.float32) * eps, n_steps=20,
                            skip_preflight=True).time_series
        return jnp.sum(trace**2)

    value, gradient = jax.value_and_grad(objective)(jnp.float32(1.))
    assert np.asarray(gradient).dtype == np.float32
    assert np.isfinite(value)
    assert np.isfinite(gradient)
    # Eager concrete overrides, with host subtraction to avoid cancellation
    # introduced by a second float32 subtraction of the two scalar losses.
    differences = [(float(objective(1. + h)) - float(objective(1. - h))) / (2 * h)
                   for h in (.02, .01)]
    richardson = (4 * differences[1] - differences[0]) / 3
    residual = abs(float(gradient) / richardson - 1)
    print(f"{mode}: value={float(value)}, grad={float(gradient)}, "
          f"FD={differences}, Richardson={richardson}, residual={residual}")
    assert abs(richardson) > 0
    np.testing.assert_allclose(float(gradient), richardson, rtol=1e-3, atol=0)


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
@pytest.mark.parametrize("kind", [None, "current"])
def test_adi_traced_sigma_refused_before_stepping(mode, kind, monkeypatch):
    import rfx.adi as adi

    sim = _simulation(mode, kind)
    shape = sim._build_grid().shape

    def premature_step(*args, **kwargs):
        pytest.fail("ADI stepping reached before traced-sigma refusal")

    monkeypatch.setattr(adi, "run_adi_3d", premature_step)
    monkeypatch.setattr(adi, "run_adi_2d", premature_step)

    def objective(sigma):
        return jnp.sum(sim.forward(sigma_override=jnp.zeros(shape, dtype=jnp.float32) + sigma,
                                  n_steps=20, skip_preflight=True).time_series**2)

    with pytest.raises(ValueError, match=(
            "verified only in a lossless source cell; a traced sigma_override "
            "can make it lossy.*amplitude_kind='field'.*Yee")):
        jax.grad(objective)(jnp.float32(0.))


@pytest.mark.parametrize("mode", ["3d", "2d_tmz"])
def test_adi_field_sigma_gradient_remains_finite(mode):
    sim = _simulation(mode, "field")
    shape = sim._build_grid().shape

    def objective(sigma):
        return jnp.sum(sim.forward(sigma_override=jnp.zeros(shape, dtype=jnp.float32) + sigma,
                                  n_steps=20, skip_preflight=True).time_series**2)

    assert np.isfinite(jax.grad(objective)(jnp.float32(0.)))
