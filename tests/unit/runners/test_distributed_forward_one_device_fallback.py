"""Graded sources refuse explicitly, including transformed forward calls."""
import gc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from tests._multi_device_scene import host_devices



@pytest.fixture(autouse=True)
def release_models(monkeypatch):
    # The once-per-process experimental-lane notice is independent of fallback.
    monkeypatch.setattr("rfx.api._execute._DISTRIBUTED_FIRST_CALL_WARNED", True)
    yield
    jax.clear_caches()
    gc.collect()


def scene(kind):
    if kind == "waveguide":
        sim = Simulation(12e9, (.06, .02, .01), dx=.002, cpml_layers=4,
                         dz_profile=np.array([.002, .0018, .0022, .0018, .0022]),
                         boundary={"x": "cpml", "y": "pec", "z": "pec"})
        sim.add_waveguide_port(.010, direction="+x", freqs=np.array([10e9, 11e9]),
                               f0=11e9, bandwidth=.6, probe_offset=2, ref_offset=1)
        sim.add_probe((.030, .010, .006), "ez")
    else:
        sim = Simulation(20e9, (.030, .020, .012), dx=.001, cpml_layers=4,
                         dz_profile=.001 * np.array([1.2, 1.1, 1., .9, .8, .7, .6,
                                                     .6, .7, .8, .9, 1., 1.1, .6]))
        sim.add_tfsf_source()
        sim.add_probe((.023, .012, .007), "ez")
    shape = sim._build_nonuniform_grid().shape
    eps = jnp.ones(shape, dtype=jnp.float32)
    eps = eps.at[shape[0] // 2 - 1:shape[0] // 2 + 2, :, :].set(1.7)
    return sim, eps


@pytest.mark.parametrize("kind", ["waveguide", "tfsf"])
@pytest.mark.parametrize("automatic_devices", [False, True])
@pytest.mark.parametrize("skip_preflight", [False, True])
def test_graded_forward_refuses_instead_of_fallback(kind, automatic_devices, skip_preflight):
    sim, eps = scene(kind)
    devices = None if automatic_devices else host_devices(2)
    with pytest.raises(NotImplementedError, match="waveguide ports" if kind == "waveguide" else "TFSF.*graded multi-device"):
        sim.forward(eps_override=eps, n_steps=48, checkpoint=False,
            checkpoint_every=4, n_warmup=4, skip_preflight=skip_preflight,
            distributed=True, devices=devices)


@pytest.mark.parametrize("kind", ["waveguide", "tfsf"])
@pytest.mark.parametrize("transform", ["grad", "jit", "vmap"])
def test_transformed_graded_forward_refuses(kind, transform):
    sim, eps = scene(kind)
    devices = host_devices(2)
    def objective(e):
        return jnp.sum(sim.forward(eps_override=e, n_steps=4, checkpoint=False,
            skip_preflight=True, distributed=True, devices=devices).time_series ** 2)
    fn = {"grad": jax.grad, "jit": jax.jit, "vmap": jax.vmap}[transform](objective)
    with pytest.raises(NotImplementedError, match="waveguide ports" if kind == "waveguide" else "TFSF.*graded multi-device"):
        fn(eps[None] if transform == "vmap" else eps)


@pytest.mark.parametrize("kind", ["waveguide", "tfsf"])
def test_sources_refused_even_above_grading_limit(kind):
    sim, _ = scene(kind)
    profile = np.array(sim._dz_profile, dtype=float)
    profile[-1] = 6.5 * profile.min()
    sim._dz_profile = profile
    # The declared z follows the lengthened profile, or the sum rule refuses
    # the model before the source refusal this test pins (#1941).
    sim._domain = (*sim._domain[:2], float(profile.sum()))
    for entry in ("forward", "run"):
        options = dict(devices=host_devices(2))
        if entry == "forward":
            options.update(distributed=True, checkpoint=False)
        with pytest.raises(NotImplementedError, match="waveguide ports" if kind == "waveguide" else "TFSF.*graded multi-device"):
            getattr(sim, entry)(n_steps=2, skip_preflight=True, **options)
