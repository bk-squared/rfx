"""``Simulation.run(..., ringdown=RingdownSpec())`` on a small resonant box (issue #1254).

Physics. A 12 x 11 x 5 mm metal box filled with eps_r 2.2 (conductivity for a
dielectric Q of 300 at 12.5 GHz) rings in its TM110 mode at 12.42 GHz with a
loaded Q of about 230 when a 1 mm wire standing on the floor near a corner is
driven by a pulse through its 50 ohm port: an amplitude decay time
Q / (pi f) = 5.9 ns, about 3100 steps of the 1 mm grid. A record stopped at
2.86 ns (1500 steps, half a decay time) misreads S11 by about 1 dB and 16
degrees; the record's ringing tail completed in closed form from the poles of
its second half gives the 30.5 ns record's S11 to 2.4e-4 dB and 1.5e-3 degrees.

What is pinned here, on the uniform lane and on the graded lane (a constant
dz_profile routes the same box through the graded runner):

* W0 -- the port voltage and current rebuilt from the port-channel probes and
  accumulated the run's way equal the run's own accumulators bit for bit
  (``run()`` raises otherwise; a rebuild reading the wrong H samples is caught);
* every output the run returns without ``ringdown=`` is byte-identical with it
  (SHA-256 of every array and the repr of every other value);
* the short record's completed S11 matches the long record's within
  0.05 dB and 0.5 degrees while its plain S11 does not, and the witnesses read ok;
* the driven column of a two-port graded board (S11 and S21) likewise;
* every model the completion does not cover is refused with its reason.
"""

from __future__ import annotations

import hashlib

import jax
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx import ringdown as rd
from rfx.ringdown import RingdownSpec
from rfx.sources.sources import CWSource

MM = 1.0e-3
FREQS = np.linspace(8.0e9, 18.0e9, 101)
N_SHORT = 1500          # 2.86 ns: half a TM110 decay time
N_LONG = 16000          # 30.5 ns: 5.2 decay times
PULSE = GaussianPulse(f0=13.0e9, bandwidth=0.8, cutoff=4.5)


def _box(lane, *, second_port=False):
    kw = {"dz_profile": np.full(5, 1.0 * MM)} if lane == "graded" else {}
    sim = Simulation(freq_max=20.0e9, domain=(12 * MM, 11 * MM, 5 * MM),
                     dx=1.0 * MM, boundary="pec", **kw)
    eps_r, q_d, f_ref = 2.2, 300.0, 12.5e9
    sim.add_material("fill", eps_r=eps_r,
                     sigma=2.0 * np.pi * f_ref * 8.854187817e-12 * eps_r / q_d)
    sim.add(Box((0, 0, 0), (12 * MM, 11 * MM, 5 * MM)), material="fill")
    sim.add_port(position=(2 * MM, 2 * MM, 0.0), component="ez", impedance=50.0,
                 extent=1.0 * MM, waveform=PULSE)
    if second_port:
        sim.add_port(position=(9 * MM, 8 * MM, 0.0), component="ez", impedance=50.0,
                     extent=1.0 * MM, waveform=PULSE, excite=False)
    sim.add_probe(position=(8 * MM, 7 * MM, 2 * MM), component="ez")
    return sim


def _run(sim, n, **kw):
    return sim.run(n_steps=n, compute_s_params=True, s_param_freqs=FREQS,
                   skip_preflight=True, **kw)


def _db_deg(s, ref):
    s = np.asarray(s).astype(np.complex128)
    ref = np.asarray(ref).astype(np.complex128)
    d_db = np.abs(20.0 * np.log10(np.abs(s)) - 20.0 * np.log10(np.abs(ref)))
    d_deg = np.abs(np.degrees(np.angle(s / ref)))
    return float(d_db.max()), float(d_deg.max())


LANES = pytest.mark.parametrize("short_runs", ["uniform", "graded"], indirect=True)


@pytest.fixture(scope="module")
def short_runs(request):
    """The short record on one lane, without and with ``ringdown=``."""
    lane = request.param
    plain = _run(_box(lane), N_SHORT)
    completed = _run(_box(lane), N_SHORT, ringdown=RingdownSpec())
    return lane, plain, completed


# ---------------------------------------------------------------------------
# W0 and byte identity
# ---------------------------------------------------------------------------

@LANES
def test_w0_holds_and_the_completion_has_the_shape_of_the_run(short_runs):
    lane, plain, completed = short_runs
    rdr = completed.ringdown
    assert isinstance(rdr, rd.RingdownResult), lane
    w0 = rdr.report.witness("W0")
    assert w0.ok and w0.value == 0.0, (lane, w0)
    s_run = np.asarray(completed.s_params)
    assert rdr.s_params.shape == s_run.shape and rdr.s_params.dtype == s_run.dtype
    assert np.array_equal(np.asarray(rdr.freqs), np.asarray(completed.freqs))
    rep = rdr.report
    assert rep.window_steps == (N_SHORT // 2, N_SHORT)
    assert rep.n_record == N_SHORT and rep.freq_max_hz == 20.0e9
    assert plain.ringdown is None


def _digest(obj, path, out):
    if obj is None or isinstance(obj, (bool, int, float, complex, str)):
        out[path] = repr(obj)
    elif isinstance(obj, np.generic):
        out[path] = f"{obj.dtype.str}:{obj.item()!r}"
    elif isinstance(obj, (np.ndarray, jax.Array)):
        a = np.ascontiguousarray(np.asarray(obj))
        out[path] = (a.dtype.str, a.shape, hashlib.sha256(a.tobytes()).hexdigest())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=repr):
            _digest(obj[k], f"{path}[{k!r}]", out)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        for f in obj._fields:
            _digest(getattr(obj, f), f"{path}.{f}", out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _digest(v, f"{path}[{i}]", out)
    elif hasattr(obj, "__dict__"):
        for k in sorted(vars(obj)):
            _digest(vars(obj)[k], f"{path}.{k}", out)
    else:
        out[path] = repr(obj)
    return out


@LANES
def test_every_other_output_is_byte_identical(short_runs):
    lane, plain, completed = short_runs
    fields = [f for f in plain._fields if f != "ringdown"]
    a, b = {}, {}
    for f in fields:
        _digest(getattr(plain, f), f, a)
        _digest(getattr(completed, f), f, b)
    assert a.keys() == b.keys(), (lane, sorted(set(a) ^ set(b)))
    differ = [k for k in a if a[k] != b[k]]
    assert not differ, (lane, differ)
    n_arrays = sum(1 for v in a.values() if isinstance(v, tuple))
    assert n_arrays >= 8, (lane, n_arrays)       # state, series, S, freqs ...
    assert np.asarray(completed.time_series).shape == (N_SHORT, 1)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_w0_and_byte_identity_hold_under_scoped_x64(lane):
    """Under x64 the uniform lane's port accumulators are complex128; the
    rebuild accumulates in the run's own dtype and assembles from it, so W0
    still holds bit for bit and the run's outputs are unchanged."""
    from tests._x64_compat import enable_x64

    with enable_x64():
        plain = _run(_box(lane), 600)
        completed = _run(_box(lane), 600, ringdown=RingdownSpec())
    assert completed.ringdown.report.witness("W0").value == 0.0
    for f in ("time_series", "s_params", "freqs"):
        assert np.array_equal(np.asarray(getattr(plain, f)),
                              np.asarray(getattr(completed, f))), (lane, f)


def test_w0_catches_a_rebuild_that_reads_the_wrong_h_samples(monkeypatch):
    """Swap the axes the Ez port's Ampere-loop legs are differenced along: the
    rebuilt current is then a different loop, and the run refuses to complete."""
    monkeypatch.setitem(rd._LOOP_LEGS, "ez", (("hy", 1), ("hx", 0)))
    with pytest.raises(RuntimeError, match="W0 failed"):
        _run(_box("uniform"), 600, ringdown=RingdownSpec())


# ---------------------------------------------------------------------------
# The resonant fixture: short completed = long, short plain != long
# ---------------------------------------------------------------------------

@LANES
def test_a_short_record_completed_is_the_long_record(short_runs):
    lane, _plain, short = short_runs
    long = _run(_box(lane), N_LONG, ringdown=RingdownSpec())
    ref = long.ringdown.s_params[0, 0]
    for r in (short, long):
        rep = r.ringdown.report
        assert rep.ok, rep.summary()
    db_c, deg_c = _db_deg(short.ringdown.s_params[0, 0], ref)
    db_p, deg_p = _db_deg(np.asarray(short.s_params)[0, 0], ref)
    db_l, deg_l = _db_deg(np.asarray(long.s_params)[0, 0], ref)
    print(f"\n[{lane}] vs completed {N_LONG}-step S11: completed {N_SHORT}-step "
          f"{db_c:.2e} dB / {deg_c:.2e} deg; plain {N_SHORT}-step {db_p:.3f} dB / "
          f"{deg_p:.2f} deg; plain {N_LONG}-step {db_l:.2e} dB / {deg_l:.2e} deg; "
          f"W2 {short.ringdown.report.witness('W2').value:.2e}")
    assert db_c < 0.05 and deg_c < 0.5, (db_c, deg_c)
    assert db_p > 0.5, (db_p, deg_p)             # the truncation is real
    assert db_l < 0.05 and deg_l < 0.5, (db_l, deg_l)   # the reference is settled
    # the TM110 pole, where the box puts it
    f = np.array([p.f_hz for p in short.ringdown.report.poles])
    q = np.array([p.q for p in short.ringdown.report.poles])
    k = int(np.argmin(np.abs(f - 12.42e9)))
    assert abs(f[k] - 12.42e9) < 0.01e9 and 150.0 < q[k] < 350.0, (f[k], q[k])


def test_the_driven_column_of_a_two_port_graded_box():
    """Two wire ports, the second passive: W0 holds on both, and the completed
    S11 and S21 from the short record match the long record's."""
    short = _run(_box("graded", second_port=True), N_SHORT, ringdown=RingdownSpec())
    long = _run(_box("graded", second_port=True), N_LONG, ringdown=RingdownSpec())
    assert short.ringdown.report.ok, short.ringdown.report.summary()
    assert long.ringdown.report.ok, long.ringdown.report.summary()
    S, S_ref = short.ringdown.s_params, long.ringdown.s_params
    assert S.shape == np.asarray(short.s_params).shape == (2, 2, FREQS.size)
    for j in (0, 1):
        db_c, deg_c = _db_deg(S[j, 0], S_ref[j, 0])
        db_p, deg_p = _db_deg(np.asarray(short.s_params)[j, 0], S_ref[j, 0])
        print(f"\n[two-port] S{j + 1}1 vs the completed {N_LONG}-step one: completed "
              f"{db_c:.2e} dB / {deg_c:.2e} deg, plain {db_p:.3f} dB / {deg_p:.2f} deg")
        assert db_c < 0.05 and deg_c < 0.5, (j, db_c, deg_c)
        assert db_p > 0.2 and deg_p > 2.0, (j, db_p, deg_p)
    assert np.all(S[:, 1] == 0)                  # the undriven column stays empty


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def _open_box():
    return Simulation(freq_max=20.0e9, domain=(12 * MM, 11 * MM, 5 * MM),
                      dx=1.0 * MM, boundary="cpml", cpml_layers=4)


def _wire(sim, **kw):
    sim.add_port(position=(4 * MM, 4 * MM, 1 * MM), component="ez", impedance=50.0,
                 extent=1.0 * MM, waveform=kw.pop("waveform", PULSE), **kw)
    return sim


def _case_lumped():
    sim = _open_box()
    sim.add_port(position=(4 * MM, 4 * MM, 2 * MM), component="ez", impedance=50.0,
                 waveform=PULSE)
    return sim, {}


def _case_msl():
    sim = _wire(_open_box())
    sim.add_material("sub", eps_r=3.0)
    sim.add(Box((0, 0, 0), (12 * MM, 11 * MM, 1 * MM)), material="sub")
    sim.add_msl_port((2 * MM, 5.5 * MM, 1 * MM), width=2 * MM, height=1 * MM)
    return sim, {}


def _case_waveguide():
    # add_waveguide_port refuses a model that already has an add_port port, so
    # this one has none: the kind of port is what is refused.
    sim = Simulation(freq_max=20.0e9, domain=(30 * MM, 11 * MM, 5 * MM),
                     dx=1.0 * MM, boundary="cpml", cpml_layers=4)
    sim.add_waveguide_port(5 * MM)
    return sim, {}


def _case_coax():
    sim = _wire(Simulation(freq_max=20.0e9, domain=(12 * MM, 11 * MM, 5 * MM),
                           dx=1.0 * MM, boundary="pec"))
    sim.add_coaxial_port((6 * MM, 5 * MM, 5 * MM), "top", pin_length=2 * MM)
    return sim, {}


def _case_floquet():
    sim = _wire(_open_box())
    sim.set_periodic_axes("xy")
    sim.add_floquet_port(4 * MM)
    return sim, {}


def _case_tfsf_cw():
    # add_tfsf_source refuses a model that already has an add_port port.
    sim = _open_box()
    sim.add_tfsf_source(f0=10e9, waveform="continuous_wave")
    return sim, {}


def _case_kerr():
    sim = _wire(_open_box())
    sim.add_material("kerr", eps_r=2.0, chi3=1e-18)
    sim.add(Box((6 * MM, 6 * MM, 1 * MM), (9 * MM, 9 * MM, 3 * MM)), material="kerr")
    return sim, {}


def _case_devices():
    return _wire(_open_box()), {"devices": list(jax.devices()) * 2}


def _case_source_on():
    return _wire(_open_box(), waveform=CWSource(f0=10e9)), {}


def _case_no_wire_port():
    sim = _open_box()
    sim.add_source((4 * MM, 4 * MM, 2 * MM), "ez", waveform=PULSE)
    return sim, {}


def _case_no_driven_port():
    sim = _wire(_open_box(), excite=False)
    sim.add_source((8 * MM, 7 * MM, 2 * MM), "ez", waveform=PULSE)
    return sim, {}


def _case_until_decay():
    return _wire(_open_box()), {"until_decay": 1e-3}


def _case_no_sparams():
    return _wire(_open_box()), {"compute_s_params": False}


def _case_uniform_two_ports():
    sim = _wire(_open_box())
    sim.add_port(position=(8 * MM, 7 * MM, 1 * MM), component="ez", impedance=50.0,
                 extent=1.0 * MM, waveform=PULSE, excite=False)
    return sim, {}


def _case_not_a_spec():
    return _wire(_open_box()), {"ringdown": {"window_start": 0.5}}


@pytest.mark.parametrize("case, match", [
    (_case_lumped, "one-cell lumped port"),
    (_case_msl, "microstrip"),
    (_case_waveguide, "waveguide"),
    (_case_coax, "coaxial"),
    (_case_floquet, "periodic axes|Floquet"),
    (_case_tfsf_cw, "continuous wave"),
    (_case_kerr, "Kerr"),
    (_case_devices, "distributed"),
    (_case_source_on, "still on inside the identification window"),
    (_case_no_wire_port, "needs a wire port"),
    (_case_no_driven_port, "needs a driven wire port"),
    (_case_until_decay, "until_decay"),
    (_case_no_sparams, "compute_s_params=False"),
    (_case_uniform_two_ports, "supports one wire port"),
    (_case_not_a_spec, "RingdownSpec"),
])
def test_what_the_completion_does_not_cover_is_refused(case, match):
    sim, kw = case()
    spec = kw.pop("ringdown", RingdownSpec())
    kw.setdefault("compute_s_params", True)
    with pytest.raises((NotImplementedError, ValueError, TypeError), match=match):
        sim.run(n_steps=400, ringdown=spec, skip_preflight=True, **kw)
