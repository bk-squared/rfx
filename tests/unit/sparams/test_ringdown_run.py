"""``Simulation.run(..., ringdown=RingdownSpec())`` on a small resonant box (issue #1254).

Physics. A 12 x 11 x 5 mm metal box filled with eps_r 2.2 (conductivity for a
dielectric Q of 300 at 12.5 GHz) rings in its TM110 mode near 12.4 GHz when a
short wire port inside it is driven by a pulse through 50 ohm. On the uniform
lane (1 mm cells, a 1 mm wire on the floor near a corner) the loaded Q is about
230, an amplitude decay time Q / (pi f) = 5.9 ns: a record stopped at 2.86 ns
misreads S11 by about 1 dB and 16 degrees, and the record's ringing tail
completed in closed form from the poles of its second half gives the 30.5 ns
record's S11 to 2.4e-4 dB and 1.5e-3 degrees. The graded lane runs the same box
on cells of 0.6-1.4 mm on all three axes, the port a 0.6 mm wire between a
0.8 mm and a 1.4 mm z cell, where every primal and dual metric of the port
differs (TM110 at 12.40 GHz, loaded Q about 200).

What is pinned here, on both lanes:

* W0 -- the port voltage and current rebuilt from the port-channel probes,
  accumulated the run's way, are the run's own accumulators to <= 9 ULPs, and
  the lane's S assembly on those accumulators is ``Result.s_params``; a rebuild
  reading the wrong H samples is caught and the run is returned uncompleted;
* every output the run returns without ``ringdown=`` is byte-identical with it
  (SHA-256 of every array and the repr of every other value);
* the short record's completed S11 matches the long record's within the
  measured envelope while its plain S11 does not, and the witnesses read ok;
* the driven column of a two-port graded box and a three-cell port likewise;
  W1 catches a completion fed the midpoint cell's voltage of that port;
* the decimation reference covers the requested band, and a port on a CPML
  floor of the graded lane is probed inside the grid;
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
PULSE = GaussianPulse(f0=13.0e9, bandwidth=0.8, cutoff=4.5)
#: 2.86 ns short records (about half a TM110 decay time) and settled long ones.
N_SHORT = {"uniform": 1500, "graded": 2500}
N_LONG = {"uniform": 16000, "graded": 30000}
#: The measured envelope of the short record's completed S11 against the long
#: record's, with about 4x margin: uniform 2.4e-4 dB / 1.5e-3 deg, graded
#: 1.2e-3 dB / 6.3e-3 deg (JAX 0.10.2 and 0.4.33 within 4e-6 dB).
BAR_DB = {"uniform": 1.0e-3, "graded": 5.0e-3}
BAR_DEG = {"uniform": 1.0e-2, "graded": 3.0e-2}
#: 0.6-1.4 mm cells on every axis (sums 12, 11, 5 mm; equal end cells).
DXP = np.array([1.0, 0.8, 0.6, 0.9, 1.2, 1.4, 1.2, 1.1, 1.0, 0.9, 0.9, 1.0]) * MM
DYP = np.array([1.0, 1.4, 0.8, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.2, 1.0]) * MM
DZP = np.array([0.8, 0.6, 1.4, 1.4, 0.8]) * MM


def _box(lane, *, second_port=False, cells=1, freq_max=20.0e9):
    """The box on one lane; ``cells`` sets the port's length in cells."""
    kw = ({"dx_profile": DXP, "dy_profile": DYP, "dz_profile": DZP}
          if lane == "graded" else {})
    sim = Simulation(freq_max=freq_max, domain=(12 * MM, 11 * MM, 5 * MM),
                     dx=1.0 * MM, boundary="pec", **kw)
    eps_r, q_d, f_ref = 2.2, 300.0, 12.5e9
    sim.add_material("fill", eps_r=eps_r,
                     sigma=2.0 * np.pi * f_ref * 8.854187817e-12 * eps_r / q_d)
    sim.add(Box((0, 0, 0), (12 * MM, 11 * MM, 5 * MM)), material="fill")
    if lane == "graded":
        # nodes x 3.3 (cells 0.9 | 1.2), y 3.8 (0.6 | 0.8); one cell: the
        # 0.6 mm z cell between 0.8 and 1.4; three cells: z 0 .. 2.8 mm
        pos, ext = ((3.3 * MM, 3.8 * MM, 0.8 * MM), 0.6 * MM) if cells == 1 \
            else ((3.3 * MM, 3.8 * MM, 0.0), 2.8 * MM)
        pos2, ext2 = (9.2 * MM, 8.2 * MM, 0.8 * MM), 0.6 * MM
    else:
        pos, ext = (2 * MM, 2 * MM, 0.0), float(cells) * MM
        pos2, ext2 = (9 * MM, 8 * MM, 0.0), 1.0 * MM
    sim.add_port(position=pos, component="ez", impedance=50.0, extent=ext,
                 waveform=PULSE)
    if second_port:
        sim.add_port(position=pos2, component="ez", impedance=50.0, extent=ext2,
                     waveform=PULSE, excite=False)
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
    plain = _run(_box(lane), N_SHORT[lane])
    completed = _run(_box(lane), N_SHORT[lane], ringdown=RingdownSpec())
    return lane, plain, completed


_LONG = {}


def _long(lane):
    """The settled long record on one lane (computed once per session)."""
    if lane not in _LONG:
        _LONG[lane] = _run(_box(lane), N_LONG[lane], ringdown=RingdownSpec())
    return _LONG[lane]


# ---------------------------------------------------------------------------
# W0 and byte identity
# ---------------------------------------------------------------------------

@LANES
def test_w0_holds_and_the_completion_has_the_shape_of_the_run(short_runs):
    lane, plain, completed = short_runs
    rdr = completed.ringdown
    assert isinstance(rdr, rd.RingdownResult), lane
    w0 = rdr.report.witness("W0")
    print(f"\n[{lane}] W0 = {w0.value:.3g} ULP; per array {rdr.report.w0_ulps}")
    assert w0.ok and w0.value <= rd.W0_ULP_BAR, (lane, w0)
    s_run = np.asarray(completed.s_params)
    assert rdr.s_params.shape == s_run.shape and rdr.s_params.dtype == s_run.dtype
    assert np.array_equal(np.asarray(rdr.freqs), np.asarray(completed.freqs))
    rep = rdr.report
    n = N_SHORT[lane]
    assert rep.window_steps == (round(0.5 * n), n)
    assert rep.n_record == n and rep.freq_max_hz == 20.0e9
    assert rep.completed and rep.failure is None
    amp = [p.amplitude for p in rep.poles]
    assert amp == sorted(amp, reverse=True)
    assert 0.0 < rep.tail_share < 1.0 and rep.slowest_decay_over_window > 0.0
    gw = rep.witness("growing_poles")
    assert gw.ok and gw.value == 0.0 and "not judged" in gw.rule
    assert rep.discarded_growth_max is not None
    print(f"[{lane}] W1 {rep.witness('W1').value:.3g}, accumulator round-off "
          f"{rep.s_accumulator_roundoff:.3g}; {gw.note}")
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
    assert np.asarray(completed.time_series).shape == (N_SHORT[lane], 1)


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_w0_and_byte_identity_hold_under_scoped_x64(lane):
    """Under x64 the uniform lane's port accumulators are complex128; the
    rebuild accumulates in the run's own dtype and assembles from it, so W0
    still holds and the run's outputs are unchanged."""
    from tests._x64_compat import enable_x64

    with enable_x64():
        plain = _run(_box(lane), 600)
        completed = _run(_box(lane), 600, ringdown=RingdownSpec())
    w0 = completed.ringdown.report.witness("W0").value
    print(f"\n[{lane}, x64] W0 = {w0:.3g} ULP")
    assert w0 <= rd.W0_ULP_BAR
    for f in ("time_series", "s_params", "freqs"):
        assert np.array_equal(np.asarray(getattr(plain, f)),
                              np.asarray(getattr(completed, f))), (lane, f)


def test_w0_catches_a_rebuild_that_reads_the_wrong_h_samples(monkeypatch):
    """Swap the axes the Ez port's Ampere-loop legs are differenced along: the
    rebuilt current is then a different loop. The run is returned as it is,
    uncompleted, with W0's number in the report and a warning."""
    plain = _run(_box("uniform"), 600)
    monkeypatch.setitem(rd._LOOP_LEGS, "ez", (("hy", 1), ("hx", 0)))
    with pytest.warns(UserWarning, match="W0 failed"):
        r = _run(_box("uniform"), 600, ringdown=RingdownSpec())
    rep = r.ringdown.report
    assert r.ringdown.s_params is None and not rep.completed and not rep.ok
    assert rep.failure.startswith("W0") and rep.witness("W0").value > rd.W0_ULP_BAR
    for f in ("time_series", "s_params", "freqs"):
        assert np.array_equal(np.asarray(getattr(plain, f)), np.asarray(getattr(r, f)))


# ---------------------------------------------------------------------------
# The resonant fixture: short completed = long, short plain != long
# ---------------------------------------------------------------------------

@LANES
def test_a_short_record_completed_is_the_long_record(short_runs):
    lane, _plain, short = short_runs
    long = _long(lane)
    ref = long.ringdown.s_params[0, 0]
    for r in (short, long):
        rep = r.ringdown.report
        assert rep.ok, rep.summary()
    db_c, deg_c = _db_deg(short.ringdown.s_params[0, 0], ref)
    db_p, deg_p = _db_deg(np.asarray(short.s_params)[0, 0], ref)
    db_l, deg_l = _db_deg(np.asarray(long.s_params)[0, 0], ref)
    print(f"\n[{lane}] vs completed {N_LONG[lane]}-step S11: completed "
          f"{N_SHORT[lane]}-step {db_c:.2e} dB / {deg_c:.2e} deg; plain "
          f"{N_SHORT[lane]}-step {db_p:.3f} dB / {deg_p:.2f} deg; plain "
          f"{N_LONG[lane]}-step {db_l:.2e} dB / {deg_l:.2e} deg; "
          f"W2 {short.ringdown.report.witness('W2').value:.2e}")
    assert db_c < BAR_DB[lane] and deg_c < BAR_DEG[lane], (db_c, deg_c)
    assert db_p > 0.5, (db_p, deg_p)             # the truncation is real
    assert db_l < 0.05 and deg_l < 0.5, (db_l, deg_l)   # the reference is settled
    # the TM110 pole, where the box puts it
    tm110 = {"uniform": 12.42e9, "graded": 12.40e9}[lane]
    f = np.array([p.f_hz for p in short.ringdown.report.poles])
    q = np.array([p.q for p in short.ringdown.report.poles])
    k = int(np.argmin(np.abs(f - tm110)))
    assert abs(f[k] - tm110) < 0.01e9 and 150.0 < q[k] < 350.0, (f[k], q[k])


def test_the_driven_column_of_a_two_port_graded_box():
    """Two wire ports, the second passive: W0 holds on both, and the completed
    S11 and S21 from the short record match the long record's."""
    short = _run(_box("graded", second_port=True), N_SHORT["graded"],
                 ringdown=RingdownSpec())
    long = _run(_box("graded", second_port=True), N_LONG["graded"],
                ringdown=RingdownSpec())
    assert short.ringdown.report.ok, short.ringdown.report.summary()
    assert long.ringdown.report.ok, long.ringdown.report.summary()
    S, S_ref = short.ringdown.s_params, long.ringdown.s_params
    assert S.shape == np.asarray(short.s_params).shape == (2, 2, FREQS.size)
    for j in (0, 1):
        db_c, deg_c = _db_deg(S[j, 0], S_ref[j, 0])
        db_p, deg_p = _db_deg(np.asarray(short.s_params)[j, 0], S_ref[j, 0])
        print(f"\n[two-port] S{j + 1}1 vs the completed {N_LONG['graded']}-step one: "
              f"completed {db_c:.2e} dB / {deg_c:.2e} deg, plain {db_p:.3f} dB / "
              f"{deg_p:.2f} deg")
        assert db_c < 0.05 and deg_c < 0.5, (j, db_c, deg_c)
        assert db_p > 0.2 and deg_p > 2.0, (j, db_p, deg_p)
    assert np.all(S[:, 1] == 0)                  # the undriven column stays empty


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_a_three_cell_port_completes_its_whole_gap_voltage(lane):
    """A port three cells long: its voltage is the whole-gap sum, not the
    midpoint cell's. The settled record's completed S11 must equal its plain
    S11 (a completion fed the midpoint voltage is 0.9 off in |S|), and the short
    record's completed S11 must match the settled one."""
    short = _run(_box(lane, cells=3), N_SHORT[lane], ringdown=RingdownSpec())
    long = _run(_box(lane, cells=3), N_LONG[lane], ringdown=RingdownSpec())
    if lane == "graded":                        # the uniform lane hands back no metadata
        assert len(long.wire_port_sparams[0][0][13]) == 3
    for r in (short, long):
        assert r.ringdown.report.completed, r.ringdown.report.summary()
    ref = long.ringdown.s_params[0, 0]
    db_l, deg_l = _db_deg(np.asarray(long.s_params)[0, 0], ref)
    db_c, deg_c = _db_deg(short.ringdown.s_params[0, 0], ref)
    print(f"\n[{lane}, 3-cell] settled plain vs completed {db_l:.2e} dB / {deg_l:.2e} "
          f"deg; short completed vs settled {db_c:.2e} dB / {deg_c:.2e} deg; W1 "
          f"{short.ringdown.report.witness('W1').value:.2e}")
    assert db_l < 0.05 and deg_l < 0.5, (db_l, deg_l)
    assert db_c < 0.05 and deg_c < 0.5, (db_c, deg_c)
    for r in (short, long):
        w1 = r.ringdown.report.witness("W1")
        assert w1.ok and w1.value <= rd.W1_BAR, w1


def test_the_decimation_reference_covers_the_requested_band():
    """The box's freq_max says 6.5 GHz but S11 is asked up to 18 GHz, where
    TM110 rings at 12.42 GHz. Decimating toward 6.5 GHz would filter the mode
    out of the window; the reference is the higher of the two, 18 GHz, and the
    completed S11 of a 9000-step record is the settled one."""
    r = _run(_box("uniform", freq_max=6.5e9), 9000, ringdown=RingdownSpec())
    rep = r.ringdown.report
    assert rep.freq_max_hz == 18.0e9 and rep.ok, rep.summary()
    ref = _long("uniform").ringdown.s_params[0, 0]
    db_c, deg_c = _db_deg(r.ringdown.s_params[0, 0], ref)
    db_p, deg_p = _db_deg(np.asarray(r.s_params)[0, 0], ref)
    print(f"\n[reference 18 GHz] completed {db_c:.2e} dB / {deg_c:.2e} deg; plain "
          f"{db_p:.3f} dB / {deg_p:.2f} deg")
    assert db_c < BAR_DB["uniform"] and deg_c < BAR_DEG["uniform"], (db_c, deg_c)
    assert db_p > 0.1, db_p


def test_a_graded_port_on_an_absorbing_floor_is_probed_inside_the_grid():
    """A wire port standing on the z = 0 face of a graded board with CPML below:
    the extra edge below the span would lie in the absorber pad, where the
    graded lane's positions cannot reach; it is not probed, and the run
    completes with W0 holding."""
    sim = Simulation(freq_max=12e9, domain=(24 * MM, 20 * MM, 7.5 * MM), dx=1.0 * MM,
                     boundary="cpml", cpml_layers=6,
                     dz_profile=np.array([0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1]) * MM)
    sim.add_material("sub", eps_r=3.5)
    sim.add(Box((0, 0, 0), (24 * MM, 20 * MM, 1.5 * MM)), material="sub")
    sim.add(Box((6 * MM, 5 * MM, 1.5 * MM), (18 * MM, 15 * MM, 1.5 * MM)), material="pec")
    sim.add_port(position=(9 * MM, 10 * MM, 0.0), component="ez", impedance=50.0,
                 extent=1.5 * MM, waveform=GaussianPulse(f0=8e9, bandwidth=0.9))
    r = sim.run(n_steps=1500, compute_s_params=True,
                s_param_freqs=np.linspace(4e9, 12e9, 41), skip_preflight=True,
                ringdown=RingdownSpec())
    rep = r.ringdown.report
    assert rep.witness("W0").value <= rd.W0_ULP_BAR, rep.summary()
    assert rep.completed, rep.summary()


def test_the_passivity_witness_reads_the_plain_record_too():
    """A graded wire port spanning a 0.8 mm and a 0.6 mm z cell reads |S11| up
    to 1.025 on its own plain 2.86 ns record (1.002 settled): the port reads
    non-passive before any completion, and the witness says so."""
    sim = Simulation(freq_max=20.0e9, domain=(12 * MM, 11 * MM, 5 * MM), dx=1.0 * MM,
                     boundary="pec", dx_profile=DXP, dy_profile=DYP, dz_profile=DZP)
    sim.add_material("fill", eps_r=2.2,
                     sigma=2.0 * np.pi * 12.5e9 * 8.854187817e-12 * 2.2 / 300.0)
    sim.add(Box((0, 0, 0), (12 * MM, 11 * MM, 5 * MM)), material="fill")
    sim.add_port(position=(3.3 * MM, 3.8 * MM, 0.0), component="ez", impedance=50.0,
                 extent=1.4 * MM, waveform=PULSE)
    with pytest.warns(UserWarning, match="passivity"):
        r = _run(sim, N_SHORT["graded"], ringdown=RingdownSpec())
    w = r.ringdown.report.witness("passivity")
    plain = float(np.max(np.abs(np.asarray(r.s_params)[0, 0])))
    assert not w.ok and plain > 1.0 + 1e-3
    assert f"{plain:.6g}" in w.note and "non-passive without the completion" in w.note


def test_w1_catches_a_completion_fed_the_midpoint_voltage(monkeypatch):
    """Feed the completion the midpoint cell's voltage of a three-cell port
    instead of the whole-gap voltage (the float64 rebuild only; the replay W0
    checks is left alone). W1 sees the two V/I constructions disagree by about
    0.9 in |S|, and the run comes back uncompleted with W1's number."""
    plain = _run(_box("uniform", cells=3), 600)
    orig = rd._port_vi

    def midpoint_voltage(lane, pm, metrics, e, hx, hy, hz):
        v, v_port, i_val = orig(lane, pm, metrics, e, hx, hy, hz)
        if isinstance(e, np.ndarray) and e.dtype == np.float64:
            return v, v, i_val
        return v, v_port, i_val

    monkeypatch.setattr(rd, "_port_vi", midpoint_voltage)
    with pytest.warns(UserWarning, match="W1 failed"):
        r = _run(_box("uniform", cells=3), 600, ringdown=RingdownSpec())
    rep = r.ringdown.report
    assert r.ringdown.s_params is None and rep.failure.startswith("W1")
    assert rep.witness("W0").ok and rep.witness("W1").value > 0.1
    for f in ("time_series", "s_params", "freqs"):
        assert np.array_equal(np.asarray(getattr(plain, f)), np.asarray(getattr(r, f)))


def test_a_failed_identification_returns_the_run_uncompleted(monkeypatch):
    """When the pole identification fails, nothing is completed and nothing is
    discarded: the plain result comes back with the failure named."""
    plain = _run(_box("uniform"), 600)

    def _no_pencil(*_a, **_k):
        raise ValueError("a window of 3 decimated samples is too short for a "
                         "matrix pencil")

    monkeypatch.setattr(rd, "_pencil", _no_pencil)
    with pytest.warns(UserWarning, match="identification failed"):
        r = _run(_box("uniform"), 600, ringdown=RingdownSpec())
    rep = r.ringdown.report
    assert r.ringdown.s_params is None and rep.failure.startswith("identification")
    assert rep.witness("W0").ok and rep.witness("W1").ok
    for f in ("time_series", "s_params", "freqs"):
        assert np.array_equal(np.asarray(getattr(plain, f)), np.asarray(getattr(r, f)))


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


def _case_graded_loop_in_the_pad():
    sim = Simulation(freq_max=20.0e9, domain=(12 * MM, 11 * MM, 5 * MM), dx=1.0 * MM,
                     boundary="cpml", cpml_layers=4, dz_profile=np.full(5, 1.0 * MM))
    sim.add_port(position=(0.0, 4 * MM, 1 * MM), component="ez", impedance=50.0,
                 extent=1.0 * MM, waveform=PULSE)
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
    (_case_graded_loop_in_the_pad, "absorber pad"),
    (_case_not_a_spec, "RingdownSpec"),
])
def test_what_the_completion_does_not_cover_is_refused(case, match):
    sim, kw = case()
    spec = kw.pop("ringdown", RingdownSpec())
    kw.setdefault("compute_s_params", True)
    with pytest.raises((NotImplementedError, ValueError, TypeError), match=match):
        sim.run(n_steps=400, ringdown=spec, skip_preflight=True, **kw)
