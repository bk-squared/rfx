"""Uniform distributed forward uses the public run entry's execution pipeline."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.ringdown import RingdownSpec
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.runners import distributed_v2


def scene(lossy=False):
    """CPML vacuum or an asymmetric lossy block with a single PEC wall."""
    boundary = (BoundarySpec(x=Boundary(lo="pec", hi="cpml"), y="cpml", z="cpml")
                if lossy else "cpml")
    sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                     cpml_layers=4, boundary=boundary)
    if lossy:
        sim.add_material("lossy", eps_r=3.7, sigma=.02)
        sim.add(Box((.013, .006, .003), (.022, .014, .009)), material="lossy")
    sim.add_source((.009, .008, .005), "ez", amplitude_kind="field")
    sim.add_probe((.024, .012, .007), "ez")
    sim.add_probe((.018, .011, .008), "hy")
    return sim


def devices():
    result = jax.devices("cpu")[:2]
    assert len(result) == 2, "requires two CPU devices configured before Python starts"
    return result


@pytest.mark.parametrize("lossy", [False, True], ids=["vacuum", "lossy"])
def test_run_forward_bit_identical(lossy):
    expected = scene(lossy).run(n_steps=60, devices=devices(),
                                compute_s_params=False, skip_preflight=True)
    # ForwardResult exposes observables only. Observe its actual runner Result
    # for final fields; expected fields and probes come from the public run.
    original = distributed_v2.run_distributed
    observed = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result)
        return result

    with patch.object(distributed_v2, "run_distributed", capture):
        actual = scene(lossy).forward(n_steps=60, distributed=True,
                                      devices=devices(), skip_preflight=True)
    assert len(observed) == 1
    for component in ("ex", "ey", "ez", "hx", "hy", "hz"):
        assert np.array_equal(getattr(expected.state, component),
                              getattr(observed[0].state, component)), component
    assert np.array_equal(expected.time_series, actual.time_series)
    assert np.any(np.asarray(expected.time_series) != 0)
    assert actual.dt == expected.dt


REFUSED = {
    "eps_override": jnp.asarray(2.0),
    "sigma_override": jnp.asarray(.02),
    "mu_r_override": jnp.asarray(1.5),
    "pec_mask_override": jnp.asarray(True),
    "pec_occupancy_override": jnp.asarray(.5),
    "design_box": ((.013, .006, .003), (.020, .012, .008)),
    "design_eps_override": jnp.asarray(3.7),
    "design_sigma_override": jnp.asarray(.02),
    "design_occupancy_override": jnp.asarray(.5),
    "design_box_holds_ports": True,
    "rlc_values_override": {0: {"R": 50.0}},
    "checkpoint": True,
    "checkpoint_every": 4,
    "checkpoint_segments": 4,
    "n_warmup": 4,
    "emit_time_series": False,
    "ringdown": RingdownSpec(),
}


@pytest.mark.parametrize("keyword", REFUSED)
def test_unsupported_keyword_refused(keyword):
    with pytest.raises(NotImplementedError, match=keyword):
        scene().forward(n_steps=8, distributed=True, devices=devices(),
                        skip_preflight=True, **{keyword: REFUSED[keyword]})


@pytest.mark.parametrize("keyword,works_today", [
    ("eps_override", "use a graded mesh"), ("sigma_override", "use a graded mesh"),
    ("design_box", "omit distributed=True"), ("mu_r_override", "omit distributed=True"),
    ("checkpoint_every", "omit the checkpoint arguments")])
def test_refusal_says_what_works_today(keyword, works_today):
    with pytest.raises(NotImplementedError) as caught:
        scene().forward(n_steps=8, distributed=True, devices=devices(),
                        skip_preflight=True, **{keyword: REFUSED[keyword]})
    message = str(caught.value)
    assert works_today in message and "omit distributed=True" in message
    assert "run_distributed" not in message and "#" not in message


def test_refused_in_plain_words_inside_jax_jit():
    """The uniform multi-device runner reads concrete values while it runs; without
    this refusal jax.jit of the call fails inside it with a tracer error."""
    import jax
    with pytest.raises(NotImplementedError, match="cannot run inside jax.jit"):
        jax.jit(lambda: scene().forward(n_steps=8, distributed=True, devices=devices(),
                                        skip_preflight=True).time_series)()


def test_requested_port_s_matrix_matches_run_default():
    def port_scene():
        sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                         cpml_layers=4, boundary="cpml")
        sim.add_port((.009, .008, .005), "ez", impedance=50.0)
        sim.add_probe((.024, .012, .007), "ez")
        return sim

    expected = port_scene().run(n_steps=60, devices=devices(), skip_preflight=True)
    actual = port_scene().forward(n_steps=60, distributed=True, devices=devices(),
                                  port_s11_freqs=expected.freqs, skip_preflight=True)
    assert np.array_equal(expected.s_params, actual.s_params)
    assert np.array_equal(expected.freqs, actual.freqs)
    assert np.array_equal(expected.time_series, actual.time_series)
