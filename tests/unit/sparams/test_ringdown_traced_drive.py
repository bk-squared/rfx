"""Traced drive refusals (R1-R4, R6).

CPU budget: <60 s per file; 12 x 11 x 5 cells, 400 sampled times, zero steps.
The runner guards also keep refusal mutations from starting a field solve.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse
from rfx._traced_refusal import TracedStructureRefusal
from rfx.ringdown import RingdownForward, RingdownRun, RingdownSpec, RingdownStop
from tests.unit.sparams.test_ringdown_run import PULSE, _open_box, _wire


@pytest.fixture(autouse=True)
def no_field_steps(monkeypatch):
    def reached_runner(*args, **kwargs):
        pytest.fail("refusal must happen before the field runner")
    for plan in (RingdownRun, RingdownForward, RingdownStop):
        monkeypatch.setattr(plan, "run", reached_runner)


def _call(sim, entry, **kwargs):
    options = dict(n_steps=400, ringdown=RingdownSpec(), skip_preflight=True)
    if entry == "forward":
        return sim.forward(port_s11_freqs=np.array([13e9]), **options, **kwargs)
    return sim.run(compute_s_params=True, **options, **kwargs)


@pytest.mark.parametrize("entry, transform, early_stop", [
    pytest.param("forward", jax.grad, False, id="R1-forward-grad"),
    pytest.param("forward", jax.jit, False, id="R2-forward-jit"),
    pytest.param("run", jax.grad, False, id="R3-run-grad"),
    pytest.param("run", jax.grad, True, id="R4-until-identified-grad"),
])
def test_traced_drive_refuses(entry, transform, early_stop):
    """Physical anchor: identification needs free ringing after every drive is off."""
    def objective(amplitude):
        pulse = GaussianPulse(f0=PULSE.f0, bandwidth=PULSE.bandwidth,
                              cutoff=PULSE.cutoff, amplitude=amplitude)
        sim = _wire(_open_box(), waveform=pulse)
        options = {"until_identified": True} if early_stop else {}
        _call(sim, entry, **options)
        return amplitude * 0.0

    with pytest.raises(TracedStructureRefusal) as caught:
        transform(objective)(1.0)
    message = str(caught.value)
    assert f"{entry}(ringdown=...)" in message
    assert "wire port at (0.004, 0.004, 0.001) (ez)" in message
    assert "traced" in message


@pytest.mark.parametrize("entry, argument", [("run", "s_param_freqs"),
                                             ("forward", "port_s11_freqs")])
@pytest.mark.parametrize("graded", [False, True])
def test_traced_bins_name_the_entry_and_argument(entry, argument, graded):
    """Physical anchor: host pole identification requires concrete frequency bins."""
    from rfx import Simulation

    def objective(bins):
        sim = (Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001,
                          boundary="cpml", cpml_layers=4,
                          dz_profile=np.full(5, .001)) if graded else _open_box())
        _wire(sim)
        options = dict(n_steps=400, ringdown=RingdownSpec(), skip_preflight=True)
        if entry == "run":
            sim.run(compute_s_params=True, **options, **{argument: bins})
        else:
            sim.forward(**options, **{argument: bins})
        return jnp.sum(bins)

    with pytest.raises(jax.errors.ConcretizationTypeError) as caught:
        jax.jit(objective)(jnp.array([13e9]))
    assert f"{entry}(ringdown=...)" in str(caught.value)
    assert argument in str(caught.value)
