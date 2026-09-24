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
  accumulated the run's way, are the run's own accumulators to 1e-5 of each
  array's peak (the ULP counts printed), and the lane's S assembly on those
  accumulators is ``Result.s_params``; a rebuild reading the wrong H samples,
  dropping the current's half-step phase or reading the series one step late
  is caught and the run is returned uncompleted;
* every output the run returns without ``ringdown=`` is byte-identical with it
  (SHA-256 of every array and the repr of every other value);
* the short record's completed S11 matches the long record's within the
  measured envelope while its plain S11 does not, and the witnesses read ok;
* the error witness WE (completions from [T/2, T] and [T/4, T]) is judged:
  a 1.72 ns graded record whose completion is 1.6e-3 off reads 1.6e-3 on WE
  and fails; W2 is reported, not judged; with the pulse still on at T/4, WE
  is not formed and W2 is judged;
* every completed S is the completion from [T/2, T] computed by hand from the
  run's own port channels, bit for bit (``_run``);
* the driven column of a two-port graded box and a three-cell port likewise;
  W1 catches a completion fed the midpoint cell's voltage of that port;
* the decimation reference covers the requested band, and a port on a CPML
  floor of the graded lane is probed inside the grid;
* every model the completion does not cover is refused with its reason.
"""

from __future__ import annotations

import hashlib
import math

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


def _box(lane, *, second_port=False, cells=1, freq_max=20.0e9, pulse=PULSE,
         precision="float32", stencil_order=2):
    """The box on one lane; ``cells`` sets the port's length in cells."""
    kw = ({"dx_profile": DXP, "dy_profile": DYP, "dz_profile": DZP}
          if lane == "graded" else {})
    sim = Simulation(freq_max=freq_max, domain=(12 * MM, 11 * MM, 5 * MM),
                     dx=1.0 * MM, boundary="pec", precision=precision,
                     stencil_order=stencil_order, **kw)
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
                 waveform=pulse)
    if second_port:
        sim.add_port(position=pos2, component="ez", impedance=50.0, extent=ext2,
                     waveform=PULSE, excite=False)
    sim.add_probe(position=(8 * MM, 7 * MM, 2 * MM), component="ez")
    return sim


def _run_captured(sim, n, **kw):
    """``_run`` that also hands back the port channels the completion saw.

    With ``ringdown=``, ``rd.two_window_witness`` is wrapped for the call to
    record its inputs (the float64 port V/I series, the bins, the S assembly
    ``observable``, the plain DFT). When the run completes, its S must be the
    completion from ``[window_start T, T]`` recomputed here from those
    channels with ``identify``/``tail_dft`` -- bit for bit: the witnesses
    beside it may change, the completed number may not.
    """
    cap = {}
    orig = rd.two_window_witness

    def spy(*a, **k):
        out = orig(*a, **k)
        cap.update(args=a, kw=k)
        return out

    rd.two_window_witness = spy
    try:
        r = sim.run(n_steps=n, compute_s_params=True, s_param_freqs=FREQS,
                    skip_preflight=True, **kw)
    finally:
        rd.two_window_witness = orig
    if r.ringdown is not None and r.ringdown.s_params is not None:
        assert cap, "a completed run did not go through two_window_witness"
        Y, dt, freqs, n_record, n_start = cap["args"]
        k = cap["kw"]
        main = rd.identify(np.asarray(Y)[:n_record], dt, n_start, n_record,
                           freq_max=k["freq_max"], guard=k["guard"],
                           sv_rel=k["sv_rel"], unit_tol=k["unit_tol"])
        S_hand = np.asarray(k["observable"](
            k["plain"] + rd.tail_dft(main, n_record - 1, freqs)))
        assert S_hand.dtype == r.ringdown.s_params.dtype
        assert np.array_equal(S_hand, r.ringdown.s_params), (
            "the completed S is not the completion from the main window")
    return r, cap


def _run(sim, n, **kw):
    return _run_captured(sim, n, **kw)[0]


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
    print(f"\n[{lane}] W0 = {w0.value:.3g} of the peak; ULPs per array "
          f"{rdr.report.w0_ulps}")
    assert w0.ok and w0.value <= rd.W0_REL_BAR, (lane, w0)
    assert set(rdr.report.w0_relative) == set(rdr.report.w0_ulps)
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
    print(f"\n[{lane}, x64] W0 = {w0:.3g} of the peak; ULPs per array "
          f"{completed.ringdown.report.w0_ulps}")
    assert w0 <= rd.W0_REL_BAR
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
    print(f"\n[wrong H samples] W0 = {rep.witness('W0').value:.3g} of the peak")
    assert rep.failure.startswith("W0") and rep.witness("W0").value > rd.W0_REL_BAR
    for f in ("time_series", "s_params", "freqs"):
        assert np.array_equal(np.asarray(getattr(plain, f)), np.asarray(getattr(r, f)))


def _replay_without_half_step(monkeypatch):
    """The W0 replay accumulates the current without its half-step phase."""
    import jax.numpy as jnp

    import rfx.core.dft_utils as dft_utils
    orig = rd._emulate_accumulators

    def replay(*a, **k):
        with monkeypatch.context() as m:
            m.setattr(dft_utils, "half_step_current_phase",
                      lambda freqs, dt: jnp.ones(jnp.shape(freqs), dtype=jnp.complex64))
            return orig(*a, **k)

    monkeypatch.setattr(rd, "_emulate_accumulators", replay)


def _replay_one_step_late(monkeypatch):
    """The W0 replay is fed the probe series shifted one step later."""
    orig = rd._emulate_accumulators

    def shift(a):
        a = np.asarray(a)
        return np.concatenate([np.zeros_like(a[:1]), a[:-1]], axis=0)

    def replay(lane, pm, e32, h32, dt):
        return orig(lane, pm, shift(e32), {k: shift(v) for k, v in h32.items()}, dt)

    monkeypatch.setattr(rd, "_emulate_accumulators", replay)


@pytest.mark.parametrize("defect", [_replay_without_half_step, _replay_one_step_late])
def test_w0_catches_a_replay_that_is_not_the_run_s(defect, monkeypatch):
    """W0 judged at 1e-5 of each array's peak still sees a rebuild that drops
    the current's half-step phase (a phase of pi f dt, 0.1 rad at 18 GHz) or
    reads the series one step late: the run comes back uncompleted."""
    defect(monkeypatch)
    with pytest.warns(UserWarning, match="W0 failed"):
        r = _run(_box("uniform"), 600, ringdown=RingdownSpec())
    rep = r.ringdown.report
    w0 = rep.witness("W0").value
    print(f"\n[{defect.__name__}] W0 = {w0:.3g} of the peak; per array "
          f"{ {k: float(f'{v:.3g}') for k, v in rep.w0_relative.items()} }")
    assert r.ringdown.s_params is None and rep.failure.startswith("W0")
    assert w0 > 100 * rd.W0_REL_BAR, w0


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
        assert rep.witness("WE").judged and not rep.witness("W2").judged
    db_c, deg_c = _db_deg(short.ringdown.s_params[0, 0], ref)
    db_p, deg_p = _db_deg(np.asarray(short.s_params)[0, 0], ref)
    db_l, deg_l = _db_deg(np.asarray(long.s_params)[0, 0], ref)
    print(f"\n[{lane}] vs completed {N_LONG[lane]}-step S11: completed "
          f"{N_SHORT[lane]}-step {db_c:.2e} dB / {deg_c:.2e} deg; plain "
          f"{N_SHORT[lane]}-step {db_p:.3f} dB / {deg_p:.2f} deg; plain "
          f"{N_LONG[lane]}-step {db_l:.2e} dB / {deg_l:.2e} deg; "
          f"WE {short.ringdown.report.witness('WE').value:.2e}, "
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
    for r in (short, long):
        assert r.ringdown.report.ok, r.ringdown.report.summary()
        assert r.ringdown.report.witness("WE").judged
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


def test_the_error_witness_fails_a_graded_record_cut_at_a_fifth_of_its_decay_time():
    """The graded box stopped at 1200 steps (1.37 ns, about a fifth of TM110's
    amplitude decay time): its completion is genuinely off the completed
    30,000-step answer, above the 1e-3 bar, and WE -- the completion from
    [T/2, T] against the one from the window started twice as early,
    [T/4, T] -- reads it and is the only judged witness that fails. W2
    ([T/2, T] against [T/2, 0.9 T]) is reported, not judged, and only
    printed here.

    TM110 of this box from the 30,000-step completion. #1236 moved it: the
    port used to put its 50 ohm load also on the Ex and Ey edges at its node,
    a spurious loss that damped the box.

        tree                     f (GHz)   loaded Q   amplitude decay time
        main before #1236        12.402    201.7      5.18 ns
        after #1236              12.406    244.4      6.27 ns

    Completed-S error against the 30,000-step completion, and WE (JAX 0.10.2):

        record              tree     actual     WE         judged failures
        1500 (1.72 ns)      before   1.649e-3   1.615e-3   WE  (0.33 tau)
        1500 (1.72 ns)      after    1.420e-4   1.406e-4   none (0.27 tau)
        1827 (2.09 ns)      after    4.3e-5     4.3e-5     none (tau/3)
        1200 (1.37 ns)      before   2.17e-3    2.17e-3    WE  (0.27 tau)
        1200 (1.37 ns)      after    1.98e-3    1.71e-3    WE  (0.22 tau)
        1000 (1.14 ns)      after    1.39e-3    1.42e-3    WE

    This test was written at 1500 steps, a third of main's decay time. After
    #1236 a record cut at a third of the box's own decay time completes to
    4.3e-5, and WE rightly passes it, so the cut moved to 1200 steps, where the
    completion is off on both trees. The error is not monotonic in the record
    length (1.39e-3 at 1000 steps, 1.98e-3 at 1200); at 800 and 600 steps WE
    is not formed (the pulse is still on in [T/4, T]) and W2 is judged.
    W2 at 1200 steps: 2.99e-3 after #1236, 1.74e-2 before.
    """
    with pytest.warns(UserWarning, match="witness failed -- WE"):
        short, cap = _run_captured(_box("graded"), 1200, ringdown=RingdownSpec())
    rep = short.ringdown.report
    we, w2 = rep.witness("WE"), rep.witness("W2")
    ref = _long("graded").ringdown.s_params.astype(np.complex128)
    actual = float(np.max(np.abs(short.ringdown.s_params.astype(np.complex128) - ref)))
    # WE by hand from the run's own channels: [T/2, T] against [T/4, T]
    Y, dt, freqs, n, n_start = cap["args"]
    k = cap["kw"]
    kw = {key: k[key] for key in ("freq_max", "guard", "sv_rel", "unit_tol")}
    Y = np.asarray(Y)[:n]
    S_half, S_quarter = (
        np.asarray(k["observable"](k["plain"] + rd.tail_dft(
            rd.identify(Y, dt, n0, n, **kw), n - 1, freqs))).astype(np.complex128)
        for n0 in (600, 300))
    we_hand = float(np.max(np.abs(S_half - S_quarter)))
    print(f"\n[graded, 1200 steps] actual {actual:.3e}; WE {we.value:.3e} "
          f"(by hand {we_hand:.3e}, WE/actual {we.value / actual:.3g}); W2 "
          f"{w2.value:.3e} (W2/actual {w2.value / actual:.3g})\n{rep.summary()}")
    assert n_start == 600 and rep.long_window_steps == (300, 1200)
    assert we.value == we_hand
    assert we.judged and not we.ok and we.value > we.bar
    assert not w2.judged
    assert not rep.ok and rep.completed
    assert [w.name for w in rep.witnesses if w.judged and not w.ok] == ["WE"]
    assert "WE" in rep.summary() and "not judged" in rep.summary()
    assert actual > 1.0e-3                      # the record really is short
    assert 0.5 < we.value / actual < 2.0, (we.value, actual)


def test_w2_is_judged_when_the_pulse_is_still_on_in_the_longer_window():
    """The uniform box driven by the same pulse delayed to peak at 0.71 ns
    (cutoff 24 tau): at 1500 steps it is off over the main window [T/2, T]
    but at its peak at T/4, inside WE's window [T/4, T], so the completion
    from that window would fit the drive, not the ringing. WE is not formed,
    W2 is judged in its place and says so, and the report's ok is W2's."""
    pulse = GaussianPulse(f0=13.0e9, bandwidth=0.8, cutoff=24.0)
    r = _run(_box("uniform", pulse=pulse), N_SHORT["uniform"],
             ringdown=RingdownSpec())
    rep = r.ringdown.report
    we, w2 = rep.witness("WE"), rep.witness("W2")
    print(f"\n[uniform, delayed pulse] {rep.summary()}")
    assert rep.completed and rep.witness("source_off").ok
    assert not we.judged and math.isnan(we.value) and not we.ok
    assert we.note.startswith("not formed") and "of its peak in [0.25 T, T]" in we.note
    assert w2.judged and "could not be formed" in w2.note and "R-i" in w2.note
    assert rep.long_rank is None and rep.long_n_kept is None
    assert w2.ok and rep.ok
    assert rep.ok == all(w.ok for w in rep.witnesses if w.name != "WE")


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
    assert rep.witness("W0").value <= rd.W0_REL_BAR, rep.summary()
    assert rep.completed, rep.summary()


def _graded_wire_port_box():
    """A graded wire port spanning a 0.8 mm and a 0.6 mm z cell, in the box."""
    sim = Simulation(freq_max=20.0e9, domain=(12 * MM, 11 * MM, 5 * MM), dx=1.0 * MM,
                     boundary="pec", dx_profile=DXP, dy_profile=DYP, dz_profile=DZP)
    sim.add_material("fill", eps_r=2.2,
                     sigma=2.0 * np.pi * 12.5e9 * 8.854187817e-12 * 2.2 / 300.0)
    sim.add(Box((0, 0, 0), (12 * MM, 11 * MM, 5 * MM)), material="fill")
    sim.add_port(position=(3.3 * MM, 3.8 * MM, 0.0), component="ez", impedance=50.0,
                 extent=1.4 * MM, waveform=PULSE)
    return sim


def test_the_passivity_witness_reads_the_plain_record_too():
    """The plain 2.86 ns record of a graded wire port spanning a 0.8 mm and a
    0.6 mm z cell reads non-passive; the witness note says so, and the verdict
    follows the COMPLETED S.

    Measured on this fixture: plain max |S11| 1.02396 on main before #1236 and
    1.05279 after; completed 1.01428 before (the witness failed) and 0.99796
    after (passive, no warning). The settled 30000-step records agree with the
    completions: 1.01428 before, 0.997956 after. The cause is #1236: the
    port's load used to sit also on the Ex and Ey edges at each of the wire's
    nodes, so the port read this passive box as non-passive even settled.
    With the load on the wire's own Ez edges the settled S is passive, and the
    plain short record, cut further from settled now that the spurious loss
    no longer damps the box, reads higher. The witness's failing branch is pinned by
    ``test_the_passivity_witness_fails_a_non_passive_completion`` below,
    which does not rely on a physical defect.
    """
    import warnings
    sim = _graded_wire_port_box()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = _run(sim, N_SHORT["graded"], ringdown=RingdownSpec())
    rep = r.ringdown.report
    w = rep.witness("passivity")
    plain = float(np.max(np.abs(np.asarray(r.s_params)[0, 0])))
    assert plain > 1.0 + 1e-3, plain
    assert f"{plain:.6g}" in w.note and "non-passive without the completion" in w.note
    completed = float(np.max(np.abs(r.ringdown.s_params[0, 0])))
    assert w.ok and w.value == pytest.approx(completed, rel=1e-12), (w, completed)
    assert completed <= w.bar, (completed, w.bar)
    assert rep.ok, rep.summary()
    assert not [c for c in caught if "passivity" in str(c.message)]


def test_the_passivity_witness_fails_a_non_passive_completion(monkeypatch):
    """The witness's failing branch, with no physical defect behind it. The
    uniform box with a three-cell port, 1500 steps: plain max |S11| 0.99714
    and completed 0.98016, both passive, every witness ok. The completed S is
    scaled to max |S11| = 1.010 at the one place the completion turns its
    spectra into S; the witness must fail it, warn, and keep the plain
    record's number in its note without the non-passive clause.

    Seam: ``rd._assemble`` also builds the W0, W1 and WE comparisons, so only
    the call that follows ``rd.two_window_witness`` -- ``S = to_s(w2.spectra)``,
    the completed S the passivity witness reads -- is scaled. The run is made
    with ``sim.run`` directly: ``_run``'s bit-for-bit recomputation of the
    completed S would, by design, see the injected scale.
    """
    orig_assemble, orig_two_window = rd._assemble, rd.two_window_witness
    armed = {"on": False}

    def two_window(*a, **k):
        out = orig_two_window(*a, **k)
        armed["on"] = True
        return out

    def assemble(*a, **k):
        s = orig_assemble(*a, **k)
        if armed["on"]:
            armed["on"] = False
            return s * (1.010 / float(np.max(np.abs(s[0, 0]))))
        return s

    monkeypatch.setattr(rd, "two_window_witness", two_window)
    monkeypatch.setattr(rd, "_assemble", assemble)
    with pytest.warns(UserWarning, match="witness failed -- passivity"):
        r = _box("uniform", cells=3).run(
            n_steps=1500, compute_s_params=True, s_param_freqs=FREQS,
            skip_preflight=True, ringdown=RingdownSpec())
    rep = r.ringdown.report
    w = rep.witness("passivity")
    plain = float(np.max(np.abs(np.asarray(r.s_params)[0, 0])))
    assert plain <= 1.0 + 1e-3, plain              # the plain record is passive
    assert not w.ok and w.value == pytest.approx(1.010, rel=1e-6), (w.value, w.bar)
    assert w.value == pytest.approx(
        float(np.max(np.abs(r.ringdown.s_params[0, 0]))), rel=1e-12)
    assert not rep.ok
    assert f"{plain:.6g}" in w.note
    assert "non-passive without the completion" not in w.note
    assert [x.name for x in rep.witnesses if x.judged and not x.ok] == ["passivity"]


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
    assert "NOT COMPLETED" in rep.summary() and "round-off" not in rep.summary()
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
    assert "NOT COMPLETED" in rep.summary()
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


def _case_stencil_order_4():
    # the uniform PEC box: (2,4) runs there, and steps at 0.857 of grid.dt
    return _box("uniform", stencil_order=4), {}


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
    (_case_stencil_order_4, r"run\(ringdown=...\) is not supported with stencil_order=4.*grid's dt"),
    (_case_not_a_spec, "RingdownSpec"),
])
def test_what_the_completion_does_not_cover_is_refused(case, match):
    sim, kw = case()
    spec = kw.pop("ringdown", RingdownSpec())
    kw.setdefault("compute_s_params", True)
    with pytest.raises((NotImplementedError, ValueError, TypeError), match=match):
        sim.run(n_steps=400, ringdown=spec, skip_preflight=True, **kw)
