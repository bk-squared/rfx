"""Identification channels and frequency-local gradient judgments."""
import inspect
import sys

import numpy as np
import pytest

from rfx import ringdown as rd
from tests.unit.sparams.test_ringdown_run import _box, FREQS

PROBES = (((0.006, 0.005, 0.002), "ez"),)


def test_gradient_witness_cannot_hide_a_bin_or_parameter():
    g = {"large": np.array([1000., 100.]), "weak": np.array([1., .01])}
    other = {"large": g["large"], "weak": np.array([1., .012])}
    w = rd.gradient_witness(g, other, against="longer_record", bin_axis=0)
    assert w.judged and not w.ok
    assert w.value == pytest.approx(.2)
    assert rd.gradient_witness(g, other, against="longer_record").value == pytest.approx(.002)
    other["weak"] = np.array([1.1, .01])
    assert not rd.gradient_witness(g, other, against="longer_record").ok


def test_gradient_witness_zero_bins_axis_and_nan():
    g = np.array([[0., 1., .01], [0., 2., .02]])
    other = g.copy()
    other[:, 0] = 100
    other[:, 2] *= 1.1
    w = rd.gradient_witness(g, other, against="longer_record", bin_axis=-1)
    assert w.value == pytest.approx(.1) and not w.ok
    assert w.note == "skipped 1 zero-reference bins"
    assert not rd.gradient_witness(np.zeros(2), np.ones(2), against="longer_record", bin_axis=0).judged
    other[0, 0] = np.nan
    assert np.isnan(rd.gradient_witness(g, other, against="longer_record", bin_axis=1).value)
    with pytest.raises(ValueError, match="bin_axis"):
        rd.gradient_witness(g, g, against="longer_record", bin_axis=2)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_identification_channels_reach_both_pencils_and_preserve_ports(lane, monkeypatch):
    seen = []
    models = []
    original = rd.identify

    def identify(series, *args, **kwargs):
        completion_call = inspect.currentframe().f_back.f_code.co_filename == rd.__file__
        model = original(series, *args, **kwargs)
        if completion_call:
            seen.append(np.asarray(series).copy())
            models.append(model)
        return model

    monkeypatch.setattr(rd, "identify", identify)
    sim = _box(lane)
    declared = tuple(sim._probes)
    spec = rd.RingdownSpec(identification_probes=PROBES)
    kw = {"s_param_freqs": FREQS} if lane == "uniform" else {}
    run = sim.run(n_steps=600, compute_s_params=True, skip_preflight=True, ringdown=spec, **kw)
    assert tuple(sim._probes) == declared
    model = models[0]
    expected = {float(p.imag / (2 * np.pi)): float(abs(c[-1]) / model.window_rms[-1])
                for p, c in zip(model.s, model.c)}
    for pole, shares in zip(run.ringdown.report.poles, run.ringdown.report.identification_amplitude_shares):
        assert shares[0] == pytest.approx(expected[pole.f_hz])
    run_channels = seen[0]
    assert run_channels.shape == (600, 3)
    seen.clear()
    kw = {"port_s11_freqs": FREQS} if lane == "uniform" else {}
    forward = sim.forward(n_steps=600, skip_preflight=True, ringdown=spec, **kw)
    np.asarray(forward.ringdown.s_params)  # wait for host callback
    assert seen and all(y.shape[1] == 3 for y in seen)
    assert tuple(sim._probes) == declared
    report = forward.ringdown.report
    assert report.completed and run.ringdown.report.completed
    assert report.identification_channels == run.ringdown.report.identification_channels
    assert len(report.identification_channels) == 1
    assert np.asarray(report.identification_amplitude_shares).shape == (report.n_kept, 1)
    assert report.poles == run.ringdown.report.poles
    assert report.witness("W0").ok and report.witness("W1").ok
    assert np.all(np.asarray(forward.ringdown._pole_counts) <= np.asarray(forward.ringdown._pole_budgets))
    assert np.max(np.abs(np.asarray(forward.ringdown.s_params).ravel() - run.ringdown.s_params.ravel())) < 1.5e-5


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("position", [(-.001, .005, .002), (.05, .005, .002)])
def test_identification_probe_outside_interior_refuses(lane, position):
    sim = _box(lane)
    with pytest.raises(ValueError, match="identification probe position"):
        rd.RingdownRun(sim, rd.RingdownSpec(identification_probes=((position, "ez"),)),
                       lane=lane, n_steps=600, grid=(sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()))


def test_identification_probe_bad_component_and_nonfinite_refuse():
    for probe in [((0., 0., 0.), "bad"), ((np.nan, 0., 0.), "ez")]:
        with pytest.raises(ValueError, match="identification probe"):
            rd.RingdownSpec(identification_probes=(probe,))


def test_identification_probe_unsupported_lane_refuses():
    sim = _box("uniform")
    with pytest.raises(ValueError, match="lane"):
        rd.RingdownRun(sim, rd.RingdownSpec(identification_probes=PROBES),
                       lane="unsupported", n_steps=600, grid=sim._build_grid())


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_early_stop_identification_uses_extra_channels(lane, monkeypatch):
    seen = []
    original = rd._judge_record

    def judge(Y, *args, **kwargs):
        seen.append(Y.shape[1])
        return original(Y, *args, **kwargs)

    monkeypatch.setattr(rd, "_judge_record", judge)
    r = _box(lane).run(n_steps=1500, compute_s_params=True, s_param_freqs=FREQS,
                      skip_preflight=True, until_identified=True,
                      ringdown=rd.RingdownSpec(identification_probes=PROBES))
    assert seen and set(seen) == {3}
    assert r.ringdown.report.completed
    assert len(r.ringdown.report.identification_channels) == 1


@pytest.mark.parametrize("caller", ["run", "forward"])
def test_distributed_lane_refuses_identification_probes(caller):
    sim = _box("graded")
    kw = {"devices": [0, 1]} if caller == "run" else {"distributed": True}
    with pytest.raises(NotImplementedError, match="distributed"):
        getattr(sim, caller)(n_steps=600, skip_preflight=True, **kw,
                             ringdown=rd.RingdownSpec(identification_probes=PROBES))


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_absorber_probe_refuses(lane):
    from rfx import Simulation
    kw = {"dz_profile": np.full(5, .001)} if lane == "graded" else {}
    sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001,
                     boundary="cpml", cpml_layers=3, **kw)
    grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
    # The requested node exists in the padded grid, but is not an interior probe.
    pos = tuple(grid.node_of(ax, 1 if ax == 0 else 4) for ax in range(3))
    with pytest.raises(ValueError, match="absorber pad"):
        rd.RingdownRun(sim, rd.RingdownSpec(identification_probes=((pos, "ez"),)),
                       lane=lane, n_steps=600, grid=grid)


def test_identification_node_that_resolves_elsewhere_refuses(monkeypatch):
    sim = _box("uniform")
    grid = sim._build_grid()
    original = rd.field_index

    def shifted(grid, position, component):
        idx = original(grid, position, component)
        return (idx[0] + 1, *idx[1:]) if position == (.006, .005, .002) else idx

    monkeypatch.setattr(rd, "field_index", shifted)
    with pytest.raises(RuntimeError, match="resolves to"):
        rd.RingdownRun(sim, rd.RingdownSpec(identification_probes=(((.0061, .005, .002), "ez"),)),
                       lane="uniform", n_steps=600, grid=grid)


def test_weak_port_resonance_gradient_needs_an_identification_probe():
    """A short corner wire weakly observes TM110 in a lossy PEC box.

    At 12.425 GHz, compare dS11/d ln(eps_r) with a longer completion,
    itself witnessed by twice the record. The fixed 1% bar is the product's
    gradient bar, not a tolerance fitted to a measured error envelope.

    The record is 1200 steps (#1458): at 600 steps a source scale that is
    not a power of two moved the probe-assisted error between 0.22 % and
    3.45 % through float32 rounding of which poles sit at the keep boundary
    (the completion itself is scale-invariant: powers of two are
    bit-identical). At 1200 steps the error is 0.18-0.20 % for six scales
    and WE tracks it (ratio 0.94-1.0); record: rfx-archive
    rfx/records/20261003-1458-ringdown-scale/. The short arms run at two
    source scales, one of them not a power of two, so a configuration that
    is again rounding-marginal shows here.
    """
    import jax
    import jax.numpy as jnp
    from rfx import Box, GaussianPulse, Simulation

    def evaluate(n, probes, amplitude=1.):
        sim = Simulation(freq_max=20e9, domain=(.012, .011, .005), dx=.001, boundary="pec")
        sim.add_material("fill", eps_r=2.2,
                         sigma=2 * np.pi * 12.5e9 * 8.854187817e-12 * 2.2 / 5000)
        sim.add(Box((0, 0, 0), (.012, .011, .005)), material="fill")
        sim.add_port(position=(.001, .001, 0), component="ez", impedance=50., extent=.00025,
                     waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5,
                                            amplitude=amplitude))
        grid = sim._build_grid()
        assert grid.shape == (13, 12, 6)
        eps = jnp.full(grid.shape, 2.2, jnp.float32)

        def f(p):
            r = sim.forward(n_steps=n, skip_preflight=True,
                            port_s11_freqs=np.array([12.425e9]), eps_override=eps * p,
                            ringdown=rd.RingdownSpec(identification_probes=probes)).ringdown
            return jnp.stack([r.s_params, r.s_params_long]), r

        _s, g, result = jax.jit(lambda p: jax.jvp(
            f, (p,), (jnp.ones_like(p),), has_aux=True))(jnp.float32(1.))
        g = np.asarray(g).reshape(2, -1)
        assert np.all(np.isfinite(g))
        assert np.all(np.asarray(result._pole_counts) <= np.asarray(result._pole_budgets))
        return g, result

    # The reference is port-only (independent of the probe code under test): a
    # record long enough that the completion has nothing left to recover.
    ref, _ = evaluate(3000, ())
    longer, _ = evaluate(6000, ())
    ref_w = rd.gradient_witness(ref[0], longer[0], against="longer_record", bin_axis=0)
    assert ref_w.judged and ref_w.ok
    for amplitude in (1., 7.):
        for probes in ((), PROBES):
            g, result = evaluate(1200, probes, amplitude)
            w = rd.gradient_witness(g[0], g[1], ringdown=result, bin_axis=0)
            error = float(np.max(np.abs(g[0] - ref[0]) / np.abs(ref[0])))
            print(f"amplitude={amplitude} probes={bool(probes)}: resonance error={error:.6g}, "
                  f"WE={w.value:.6g}, reference witness={ref_w.value:.6g}", file=sys.stderr)
            assert w.judged
            if probes:
                # Both, not an implication: a regression that raises the
                # witness and the error together must fail here.
                assert w.ok and w.value <= .01 and error <= .01, (amplitude, w.value, error)
                # The witness reads the actual error (PI 2026-09-25: 0.78-2.6x).
                assert .5 <= w.value / error <= 2., (amplitude, w.value, error)
            else:
                assert not w.ok and error > .01, (amplitude, w.value, error)
