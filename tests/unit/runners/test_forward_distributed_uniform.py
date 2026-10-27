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


def test_ntff_records_match_run_bit_identically():
    from tests.contracts.run_forward_census import build_o_ntff

    expected = build_o_ntff("uniform").run(
        n_steps=60, devices=devices(), compute_s_params=False, skip_preflight=True)
    actual = build_o_ntff("uniform").forward(
        n_steps=60, distributed=True, devices=devices(), skip_preflight=True)
    assert expected.ntff_data is not None and actual.ntff_data is not None
    assert expected.ntff_box is not None and actual.ntff_box is not None
    assert actual.ntff_data._fields == expected.ntff_data._fields
    for name in expected.ntff_data._fields:
        assert np.array_equal(getattr(actual.ntff_data, name),
                              getattr(expected.ntff_data, name)), name
    assert any(np.any(np.asarray(a)) for a in expected.ntff_data)


def test_conformal_request_refused_by_both_entries():
    def conformal_scene():
        boundary = BoundarySpec(
            x="cpml", y=Boundary(lo="pec", hi="pec", conformal=True), z="cpml")
        sim = Simulation(freq_max=20e9, domain=(.030, .020, .012), dx=.001,
                         cpml_layers=4, boundary=boundary)
        sim.add(Box((.013, .006, .003), (.022, .014, .009)), material="pec")
        sim.add_source((.009, .008, .005), "ez", amplitude_kind="field")
        sim.add_probe((.024, .012, .007), "ez")
        return sim

    with pytest.raises(NotImplementedError, match="conformal_pec") as expected:
        conformal_scene().run(n_steps=8, devices=devices(),
                              compute_s_params=False, skip_preflight=True)
    with pytest.raises(type(expected.value), match="conformal_pec"):
        conformal_scene().forward(n_steps=8, distributed=True,
                                  devices=devices(), skip_preflight=True)


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


@pytest.mark.parametrize("family", ["lumped", "wire"])
def test_requested_port_fields_match_one_device_forward(family):
    from tests.contracts.run_forward_census import build_p_lumped, build_p_wire
    from rfx.probes import sparam_driver
    from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp

    def port_scene():
        sim = (build_p_wire if family == "wire" else build_p_lumped)("uniform")
        if family == "wire":
            # Run's S driver admits an impedance port without a plain source.
            sim._ports = [pe for pe in sim._ports if pe.impedance != 0]
        return sim

    freqs = np.array([4., 8., 12., 16., 20.]) * 1e9
    expected = port_scene().forward(n_steps=150, port_s11_freqs=freqs,
                                   checkpoint=False, skip_preflight=True)
    observed = []
    steps = []
    original = sparam_driver._distributed_lumped_accumulators
    original_run = distributed_v2.run_distributed

    def capture(*args, **kwargs):
        raw = original(*args, **kwargs)
        if kwargs.get("_main_options") is None:  # S-driver drive runs only
            observed.append(raw)
        return raw

    def count_steps(*args, **kwargs):
        steps.append(kwargs.get("_source_port_indices"))
        return original_run(*args, **kwargs)

    with patch.object(sparam_driver, "_distributed_lumped_accumulators", capture), \
            patch.object(distributed_v2, "run_distributed", count_steps):
        actual = port_scene().forward(n_steps=150, distributed=True, devices=devices(),
                                     port_s11_freqs=freqs, skip_preflight=True)
    assert len(observed) == 1  # The existing S driver ran once, without a second computation.
    # Time-stepping runs for this request: the main run and one S drive, as
    # measured on main 1215d3a35 (before the main run took the port records).
    assert len(steps) == 2, steps

    def compare_field(name, value, reference):
        value, reference = np.asarray(value), np.asarray(reference)
        assert value.shape == reference.shape, name
        peak = np.max(np.abs(reference), initial=0)
        error = np.max(np.abs(value.astype(np.complex128) - reference), initial=0)
        assert error <= 1e-4 * peak, (name, error, peak)
        print(f"{family}/{name}: max_relative_difference={error / peak if peak else 0:.12g}")

    compare_field("s_params", actual.s_params, expected.s_params)
    compare_field("freqs", actual.freqs, expected.freqs)
    field = f"{family}_port_sparams"
    native_accs = getattr(expected, field)[0][1]
    driver_accs = observed[0][family][0][1]
    for index, name in ((0, "v_dft"), (1, "i_dft")):
        compare_field(name, driver_accs[index], native_accs[index])
    if family == "wire":
        compare_field("v_port_dft", driver_accs[3], native_accs[3])
    # Raw tuples include unmeasured reference channels, so the whole public
    # quantity refuses rather than masquerading as a native accumulator tuple.
    reason = "does not record this port quantity; run on one device"
    assert getattr(expected, field) is not None
    with pytest.raises(NotImplementedError, match=reason):
        getattr(actual, field)
    # The port time records are the main forward run's, as run(devices=) returns
    # them: within 9 float32 ULP of each channel's peak of the run() records.
    reference = port_scene().run(n_steps=150, devices=devices(), compute_s_params=False,
                                 skip_preflight=True).sparam_time_records
    records = actual.sparam_time_records
    assert reference and len(records) == len(reference)
    for record, ref in zip(records, reference, strict=True):
        assert (record.id, record.kind, record.channels) == (ref.id, ref.kind, ref.channels)
        value, ref_samples = np.asarray(record.samples), np.asarray(ref.samples)
        assert value.shape == ref_samples.shape
        for channel in range(ref_samples.shape[1]):
            ulp = float(np.spacing(np.float32(np.max(np.abs(ref_samples[:, channel])))))
            error = float(np.max(np.abs(value[:, channel].astype(np.float64)
                                        - ref_samples[:, channel])))
            assert error <= 9 * ulp, (record.channels[channel], error / ulp)
    other = "lumped_port_sparams" if family == "wire" else "wire_port_sparams"
    assert getattr(actual, other) is None and getattr(expected, other) is None
    objective = minimize_s11_at_freq_wave_decomp(12e9)
    assert np.isfinite(np.asarray(objective(actual)))
    compare_field("objective", objective(actual), objective(expected))
    # Refusing raw records must leave the recorded-probe settling report usable.
    assert isinstance(actual.settling_witness, dict)


def test_multi_port_s_requests_are_refused_on_this_path():
    """Two excited ports in one record: both entries refuse S with the same words
    (#1660's isolated-drive rule). A passive second port, which one-device forward()
    takes, is refused here by the multi-device lane's passive-port refusal."""
    from tests.contracts.run_forward_census import build_p_lumped

    def port_scene(excite_second):
        return build_p_lumped("uniform").add_port(
            (.022, .009, .0055), "ez", impedance=50., excite=excite_second)

    freqs = np.array([4., 8., 12., 16., 20.]) * 1e9
    messages = []
    for kwargs in (dict(checkpoint=False),
                   dict(distributed=True, devices=devices())):
        with pytest.raises(ValueError, match="isolated drive") as caught:
            port_scene(True).forward(n_steps=150, port_s11_freqs=freqs,
                                     skip_preflight=True, **kwargs)
        messages.append(str(caught.value))
    assert messages[0] == messages[1]

    with pytest.raises(NotImplementedError, match="passive port"):
        port_scene(False).forward(n_steps=150, distributed=True, devices=devices(),
                                  port_s11_freqs=freqs, skip_preflight=True)


def test_gradient_of_port_s_refused_in_plain_words():
    """jax.grad through port_s11_freqs on this path used to fail inside the S driver
    with a JAX tracer error; it is refused with a reason instead."""
    from tests.contracts.run_forward_census import build_p_lumped
    from rfx.optimize_objectives import minimize_s11_at_freq_wave_decomp

    def loss(eps):
        sim = build_p_lumped("uniform")
        sim.add_material("design", eps_r=eps)
        sim.add(Box((.013, .006, .003), (.020, .012, .008)), material="design")
        result = sim.forward(n_steps=60, distributed=True, devices=devices(),
                             port_s11_freqs=np.array([8e9, 12e9]), skip_preflight=True)
        return minimize_s11_at_freq_wave_decomp(12e9)(result)

    with pytest.raises(NotImplementedError, match="cannot trace this request") as failure:
        jax.grad(loss)(2.0)
    assert isinstance(failure.value.__cause__, (
        jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError))
