"""Exercise the real parameter map and Adam projection without an FDTD scan."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.errors import AccuracyWarning, RfxRefusal
from rfx.material_fit import DebyeFitResult
from rfx.materials.debye import DebyePole


fit_module = importlib.import_module("rfx.differentiable_material_fit")


def test_permittivity_floor_can_be_left_and_enforced(monkeypatch):
    calls = []
    freqs = np.array([2e9, 3e9, 4e9])
    measured = np.zeros((1, 1, 3), complex)
    real_value_and_grad = jax.value_and_grad
    target = 2.0

    def factory(eps_inf, debye_poles, lorentz_poles):
        calls.append(eps_inf)
        assert not isinstance(eps_inf, jax.core.Tracer)
        sim = Simulation(freq_max=5e9, domain=(.012, .010, .008), dx=.002,
                         boundary="pec")
        sim.add_material("dut", eps_r=eps_inf, debye_poles=debye_poles,
                         lorentz_poles=lorentz_poles)
        sim.add_probe((.006, .004, .004), component="ez")
        return sim

    def simulated_loss(p):
        eps_inf, _ = fit_module._params_to_debye_poles(p, 1)
        return (eps_inf - target) ** 2, jnp.zeros_like(measured)

    def replace_forward_loss(forward, *, has_aux):
        # Seam: replace only the loss supplied to jax.value_and_grad;
        # its differentiation, the parameter map and Adam remain real.
        assert has_aux
        return real_value_and_grad(simulated_loss, has_aux=True)

    def unexpected_fdtd(*args, **kwargs):
        raise AssertionError("the floor test must not run FDTD")

    monkeypatch.setattr(fit_module.jax, "value_and_grad", replace_forward_loss)
    monkeypatch.setattr("rfx.simulation.run", unexpected_fdtd)

    def fit(eps_inf):
        return fit_module.differentiable_material_fit(
            factory, measured, freqs,
            initial_guess=DebyeFitResult(eps_inf, [DebyePole(1.0, 2e-11)], 0.0),
            n_iterations=40, learning_rate=0.05, verbose=False)

    up = fit(1.0)
    assert calls == [1.0]
    assert up.eps_inf >= 1.5, f"up eps_inf={up.eps_inf}"
    target = 0.5
    with pytest.warns(AccuracyWarning, match="eps_inf=1 is at the floor 1") as caught:
        down = fit(1.0)
        assert down.eps_inf == 1.0, f"down eps_inf={down.eps_inf}"
    assert len(caught) == 1
    calls.clear()
    with pytest.raises(RfxRefusal, match="0.999.*fit_debye.*fit_lorentz"):
        fit(0.999)
    assert calls == []
