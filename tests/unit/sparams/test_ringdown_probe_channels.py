"""Probe spectra, clock, relative witnesses and admission (T0-T4).

Physical anchor: the predeclared lossy PEC TM110 cavity, Q_d=300 near 12.4 GHz.
4-CPU budget estimate: 90-115 s including compilation and identification;
T1 uniform S0 uses 56,100 steps (1.2 s stepping, estimated 30-60 s total).
The graded S0 reference (93,505 steps) is reserved for the predeclared job;
its unit clock pin uses a captured short-window model and an explicit sum.
T0 is the existing, unmodified port-mode and early-stop test suite.
These are estimates, not measured timings; this file is never run on the Mac.
"""
from functools import lru_cache
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, ringdown as rd
from rfx.errors import AccuracyWarning, RfxRefusal, RfxUnsupported
from tests.unit.sparams.test_ringdown_run import DXP, DYP, DZP, FREQS, MM

PROBES = (((8 * MM, 7 * MM, 3 * MM), "ez"), ((3 * MM, 7 * MM, 3 * MM), "hy"))
SHORTEST = {"uniform": 600, "graded": 1000}


def box(lane="uniform", amplitude=1., user=True):
    kw = dict(dx_profile=DXP, dy_profile=DYP, dz_profile=DZP) if lane == "graded" else {}
    sim = Simulation(freq_max=20e9, domain=(12 * MM, 11 * MM, 5 * MM), dx=MM,
                     boundary="pec", **kw)
    sim.add_material("fill", eps_r=2.2, sigma=2 * np.pi * 12.5e9 * 8.854187817e-12 * 2.2 / 300)
    sim.add(Box((0, 0, 0), (12 * MM, 11 * MM, 5 * MM)), material="fill")
    sim.add_source((2 * MM, 2 * MM, MM), "ez", amplitude_kind="current",
                   waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5, amplitude=amplitude))
    if user:
        for position, component in PROBES:
            sim.add_probe(position=position, component=component)
    return sim


def spec(**kw):
    return rd.RingdownSpec(identification_probes=kw.pop("identification_probes", PROBES),
                           probe_freqs=kw.pop("probe_freqs", FREQS), **kw)


def stamped_sum(series, dt, bins=FREQS, start=0):
    """Independent numpy float64 physical-clock sum, including dt exactly once."""
    y = np.asarray(series, dtype=np.float64)
    result = np.zeros((len(bins), y.shape[1]), dtype=np.complex128)
    for c, offset in enumerate((1., .5)):
        for lo in range(0, len(y), 2048):
            hi = min(lo + 2048, len(y))
            times = (np.arange(start + lo, start + hi, dtype=np.float64) + offset) * float(dt)
            result[:, c] += float(dt) * (np.exp(-2j * np.pi * np.asarray(bins)[:, None] * times) @ y[lo:hi, c])
    return result


def channel_error(a, b):
    return np.max(np.abs(np.asarray(a) - b), axis=0) / np.max(np.abs(b), axis=0)


@lru_cache(maxsize=None)
def pair(lane, user=True):
    kw = dict(n_steps=SHORTEST[lane], skip_preflight=True)
    r = box(lane, user=user).run(**kw, ringdown=spec())
    f = box(lane, user=user).forward(**kw, ringdown=spec())
    return r, f


def test_t1_uniform_reference_s0():
    """Physical anchor: a Q=300 TM110 tail after 14 tau pins both E and H clocks."""
    r = box().run(n_steps=56100, ringdown=spec(), skip_preflight=True)
    assert r.ringdown.report.completed, r.ringdown.report.failure
    reference = stamped_sum(r.time_series, r.grid.dt)
    decay = stamped_sum(r.time_series[50490:], r.grid.dt, start=50490)
    assert np.all(np.max(np.abs(decay), axis=0) / np.max(np.abs(reference), axis=0) <= 1e-5)
    assert np.all(channel_error(r.ringdown.probe_spectra, reference) <= 1e-5)
    short = box().run(n_steps=1500, ringdown=spec(), skip_preflight=True)
    assert np.all(channel_error(short.ringdown.probe_spectra, reference) <= 1e-3)


def test_t1_graded_clock_from_captured_model(monkeypatch):
    """Physical anchor: Ez and Hy of the graded cavity differ by half a time step."""
    captured = {}
    original = rd.two_window_witness

    def capture(*args, **kwargs):
        w = original(*args, **kwargs)
        captured.update(model=w.model, dt=args[1])
        return w

    monkeypatch.setattr(rd, "two_window_witness", capture)
    r = box("graded").run(n_steps=1000, ringdown=spec(), skip_preflight=True)
    model, dt = captured["model"], captured["dt"]
    expected = stamped_sum(r.time_series, dt)
    # The geometric sum is written here, independent of stamp/tail helpers.
    # Apply the declared resolved/static classes directly, without helpers.
    for pole, residue in zip(model.s, model.c):
        resolved = abs(pole.real * dt) >= np.log1p(1e-6) / model.D
        static = abs(pole.imag / (2 * np.pi)) < 1 / (1000 * dt) and abs(np.exp(pole.real * dt) - 1) < 1e-6
        if not (resolved or static):
            continue
        for c, offset in enumerate((1., .5)):
            expected[:, c] += dt * residue[c] * np.exp(pole * dt * (1000 - model.n_ref)) * (
                np.exp(-2j * np.pi * FREQS * dt * (1000 + offset))
                / (-np.expm1((pole - 2j * np.pi * FREQS) * dt)))
    assert np.all(channel_error(r.ringdown.probe_spectra, expected) <= 1e-12)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
@pytest.mark.parametrize("user", [True, False])
def test_t2_equal_requests_and_per_entry_series(lane, user):
    """Physical anchor: observing a field adds no load or source to the cavity."""
    r, f = pair(lane, user)
    assert r.ringdown.report.completed and f.ringdown.report.completed
    assert r.ringdown.s_params is None and f.ringdown.s_params is None
    assert f.ringdown.s_params_long is None
    assert r.ringdown.probe_spectra.dtype == np.complex128
    assert np.array_equal(r.ringdown.freqs, FREQS)
    assert np.array_equal(np.asarray(f.ringdown.freqs), FREQS.astype(np.asarray(f.ringdown.freqs).dtype))
    assert np.all(channel_error(f.ringdown.probe_spectra, r.ringdown.probe_spectra) <= 1e-3)
    for entry, completed in (("run", r), ("forward", f)):
        plain = getattr(box(lane, user=user), entry)(n_steps=SHORTEST[lane], skip_preflight=True)
        assert np.array_equal(np.asarray(plain.time_series), np.asarray(completed.time_series))
    if not user and lane == "uniform":
        assert f.time_series.shape == (600, 1)  # the soft source's default column
    if user:
        assert r.ringdown.report.identification_channels == f.ringdown.report.identification_channels


def test_t2_equal_refusal_messages():
    """Physical anchor: mixing port S and raw field completion has no shared scale."""
    messages = []
    for entry in ("run", "forward"):
        sim = box()
        sim.add_port((2 * MM, 2 * MM, 0), "ez", extent=MM, impedance=50.)
        with pytest.raises(RfxUnsupported) as caught:
            getattr(sim, entry)(n_steps=600, ringdown=spec(), skip_preflight=True)
        messages.append(str(caught.value))
    assert messages[0] == messages[1]
    assert "run them as two calls" in messages[0]


@pytest.mark.parametrize("bins", [[], [[1.]], [0.], [-1.], [np.nan], [np.inf]])
def test_t3_invalid_probe_bins(bins):
    """Physical anchor: an infinite-record spectrum requires finite positive bins."""
    with pytest.raises(ValueError, match="probe_freqs.*1-D, non-empty, finite and positive"):
        spec(probe_freqs=bins)


def test_t3_bins_hashable_and_traced_by_name():
    """Physical anchor: host pole identification requires concrete frequency bins."""
    s = spec(probe_freqs=np.array([12.400000001e9]))
    assert s.probe_freqs == (12.400000001e9,) and isinstance(hash(s), int)
    with pytest.raises(jax.errors.ConcretizationTypeError, match="RingdownSpec.*probe_freqs"):
        jax.jit(lambda bins: spec(probe_freqs=bins).window_start)(np.array([12e9]))


def admission_model():
    return SimpleNamespace(_ports=[SimpleNamespace(is_soft_source=True, waveform=object())],
                           _mode="3d", _materials={})


@pytest.mark.parametrize("row, words, error", [
    ("wire", "completed probe channels and port S-parameters", RfxUnsupported),
    ("lumped", "run them as two calls", RfxUnsupported),
    ("special", "run them as two calls", RfxUnsupported),
    ("empty", "none is declared", RfxRefusal),
    ("no_drive", "nothing drives the model", RfxRefusal),
    ("no_waveform", "nothing drives the model", RfxRefusal),
    ("tfsf", "TFSF plane wave", RfxUnsupported),
    ("distributed", "distributed lane", RfxUnsupported),
    ("subgrid", "subgridded lane", RfxUnsupported),
    ("periodic", "periodic axes", RfxUnsupported),
    ("kerr", "Kerr", RfxUnsupported),
    ("stencil", "stencil_order=4", RfxUnsupported),
    ("decay", "until_decay", RfxUnsupported),
])
def test_t3_admission_rows(row, words, error):
    """Physical anchor: completion requires fixed linear free ringing without ports."""
    sim, request, kw = admission_model(), spec(), dict(devices=None, until_decay=None, compute_s_params=None)
    if row in ("wire", "lumped"):
        sim._ports.append(SimpleNamespace(is_soft_source=False, extent=MM if row == "wire" else None))
    elif row == "special":
        sim._waveguide_ports = [object()]
    elif row == "empty":
        request = spec(identification_probes=())
    elif row == "no_drive":
        sim._ports = []
    elif row == "no_waveform":
        sim._ports[0].waveform = None
    elif row == "tfsf":
        sim._tfsf = SimpleNamespace(waveform="gaussian")
    elif row == "distributed":
        kw["devices"] = [0, 1]
    elif row == "subgrid":
        sim._refinement = object()
    elif row == "periodic":
        sim._periodic_axes = "x"
    elif row == "kerr":
        sim._materials = {"nonlinear": SimpleNamespace(chi3=1.)}
    elif row == "stencil":
        sim._stencil_order = 4
    elif row == "decay":
        kw["until_decay"] = -60
    with pytest.raises(error, match=words):
        rd.refuse_run_request(sim, request, **kw)


def test_t3_early_stop_refused():
    """Physical anchor: the port-S stopping floor does not bound raw probe spectra."""
    with pytest.raises(RfxUnsupported, match="early stop.*port S witness.*not offered for probe channels"):
        rd.refuse_until_identified(admission_model(), spec(), True, until_decay=None, snapshot=None)


def test_t3_explicit_s_preflight_refused():
    """Physical anchor: a soft source has no incident port wave from which to form S."""
    with pytest.raises(ValueError, match="source-only observables.*Result.s_params"):
        box().run(n_steps=600, ringdown=spec(), compute_s_params=True, skip_preflight=True)


@pytest.mark.parametrize("compute", [None, False])
def test_t3_no_s_admitted_and_uniform_wire_row_not_reached(compute):
    """Physical anchor: an unloaded driven cavity supplies field channels without a port."""
    rd.refuse_run_request(admission_model(), spec(), devices=None, until_decay=None, compute_s_params=compute)
    rd.refuse_run_lane("run_uniform", 0)
    rd.refuse_forward_request(admission_model(), spec(), distributed=False, emit_time_series=True)
    rd.refuse_forward_lane("fwd_uniform", 0, port_s11_freqs=None, spec=spec())
    if compute is False:
        r = box().run(n_steps=600, ringdown=spec(), compute_s_params=False, skip_preflight=True)
        assert r.ringdown.report.completed and r.ringdown.s_params is None


def test_t3_pec_zero_channel_named():
    """Physical anchor: tangential E on a PEC wall is zero at every step."""
    probes = (PROBES[0], ((6.5e-3, 7e-3, 0.0), "ex"))
    with pytest.warns(AccuracyWarning, match="identification probe ex at .*records no field over the window"):
        r = box().run(n_steps=600, ringdown=spec(identification_probes=probes), skip_preflight=True)
    assert not r.ringdown.report.completed and r.ringdown.probe_spectra is None
    assert "identification probe ex" in r.ringdown.report.failure

    # The cluster measured WE=1.486 for rounding-level Hz: it is not an exact zero.
    hz_request = spec(identification_probes=(PROBES[0], (PROBES[1][0], "hz")))
    with pytest.warns(AccuracyWarning, match=r"witness failed -- .*WE .*\(bar "):
        hz = box().run(n_steps=600, ringdown=hz_request, skip_preflight=True).ringdown
    assert hz.report.completed and hz.probe_spectra is not None
    assert not hz.report.ok
    assert hz.report.witness("WE").value > hz_request.witness_tol


def test_t4a_power_of_two_drive_scale():
    """Physical anchor: field linearity makes relative witnesses independent of drive amplitude."""
    outputs = [box(amplitude=a).forward(n_steps=600, ringdown=spec(), skip_preflight=True).ringdown
               for a in (1., 2.**-10)]
    a, b = outputs
    assert np.array_equal(np.asarray(b.probe_spectra), np.asarray(a.probe_spectra) * 2.**-10)
    for name in ("WE", "W2", "traced"):
        wa, wb = a.report.witness(name), b.report.witness(name)
        assert np.isfinite(wa.value) and np.isfinite(wb.value)
        assert abs(wa.value - wb.value) <= abs(np.spacing(np.float64(wa.value)))


def test_t4b_report_formula_and_not_formed(monkeypatch):
    """Physical anchor: each channel's own peak weights E and the smaller H field equally."""
    captured = {}
    two, early = rd.two_window_witness, rd.early_start_witness

    def w2(*args, **kwargs):
        captured["w2"] = two(*args, **kwargs)
        assert kwargs["observable"] is None
        return captured["w2"]

    def we(*args, **kwargs):
        captured["we"] = early(*args, **kwargs)
        return captured["we"]

    monkeypatch.setattr(rd, "two_window_witness", w2)
    monkeypatch.setattr(rd, "early_start_witness", we)
    f = box().forward(n_steps=600, ringdown=spec(), skip_preflight=True).ringdown
    report = f.report
    a = captured["w2"].spectra
    for name, b in (("W2", captured["w2"].spectra_short), ("WE", captured["we"].spectra_long)):
        expected = np.max(np.max(np.abs(a - b), axis=0) / np.max(np.abs(a), axis=0))
        assert report.witness(name).value == pytest.approx(expected, rel=1e-12, abs=0.)
        assert not np.isclose(expected, np.max(np.abs(a - b) / np.abs(a)), rtol=1e-6, atol=0.)
        assert "max_f |X_main,c|" in report.witness(name).rule
    for name in ("W0", "W1", "passivity", "consistency"):
        w = report.witness(name)
        assert not w.judged and np.isnan(w.value) and w.note.startswith("not formed:")
    assert report.w0_ulps is None and report.w0_relative is None and report.s_accumulator_roundoff is None
    assert report.identification_channels and np.shape(report.identification_amplitude_shares)[1] == 2
    assert f._accs == () and f._s_plain is None


@pytest.mark.parametrize("distributed, emit, words", [
    (True, True, "distributed=True"), (False, False, "emit_time_series=False"),
])
def test_t3_forward_record_requirements(distributed, emit, words):
    """Physical anchor: host identification needs a recorded single-device field window."""
    with pytest.raises((RfxUnsupported, RfxRefusal), match=words):
        rd.refuse_forward_request(admission_model(), spec(), distributed=distributed, emit_time_series=emit)


def test_probe_bins_above_freq_max_use_one_reference(monkeypatch):
    """Physical anchor: decimation must retain modes at every requested spectral bin."""
    import importlib

    _pencil = importlib.import_module("rfx.ringdown._pencil")  # the module; rd._pencil is a function

    seen = []
    original = _pencil.decimation_plan

    def capture(n, dt, freq_max):
        seen.append(float(freq_max))
        return original(n, dt, freq_max)

    # Patch the name where the identification plan looks it up.
    monkeypatch.setattr(_pencil, "decimation_plan", capture)
    request = spec(probe_freqs=(8e9, 18e9, 20.000000123e9))
    result = box().forward(n_steps=600, ringdown=request, skip_preflight=True).ringdown
    assert result.report.freq_max_hz == request.probe_freqs[-1]
    assert seen and all(value == request.probe_freqs[-1] for value in seen)
