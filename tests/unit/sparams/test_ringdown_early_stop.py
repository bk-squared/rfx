"""``run(..., ringdown=RingdownSpec(), until_identified=True)``: the automatic early stop (issue #1254, PR 2).

Physics. The 12 x 11 x 5 mm eps_r 2.2 metal box of ``test_ringdown_run.py``
rings in its TM110 mode near 12.4 GHz after a short pulse through a 50 ohm wire
port, with an amplitude decay time Q / (pi f) of 5.9 ns on the uniform lane and
6.3 ns on the graded one. The ring-down completion gives the settled S11 from a
record cut while the box still rings; the early stop ends the run at the first
record T the completion can stand on: every source off over [T/4, T], the
error witness WE (the completions from [T/2, T] and [T/4, T] against each
other) within its 1e-3 bar at two checks in a row, and -- the PI's floor
(2026-09-27) -- T at least half the decay time of the slowest identified
mode (a pole whose own tail moves the completed S by the bar; a static field
left out), because a pair of modes that no window of a record separates is completed as
one mode while every witness agrees (rfx #1254, R-i: 0.03 % apart, recorded
for 7 % of their decay time, 8.8 % off).

What is pinned here, on both lanes:

* the run stops at a fraction of the settled record and its completed S11 is
  the settled record's within the PR 1 bars (``BAR_DB``, ``BAR_DEG``), while the
  plain S11 at the stop is not;
* the stopped result is ``run(n_steps=T_stop, ringdown=...)``'s, bit for bit
  (every ``Result`` field, the completed S and its report), and the check that
  stopped read the WE that report reads;
* on these boxes WE is within its bar at two checks in a row already at a
  third of the decay time: the floor holds the stop until the record passes
  half of it, and the decay time the floor reads is the settled record's own;
* a static field under the ringing (a 0 Hz pole that does not decay) does not
  hold the floor: the stop fires at half the ringing mode's decay time;
* what the floor is for, on synthetic records through the stop's own check and
  schedule: a pair no short record resolves (the R-i case), a decaying mode
  below 1/record, and a weakly coupled high-Q pair each hold the stop where WE
  already agrees and the completion is 1-10 % off; a non-passive completion
  does not stop; the floor reads the largest decay time of the checks the
  two-in-a-row rule rests on;
* no check reads the record while a source is on over [T/4, T], and the same
  pulse delayed to end at 1.34 ns moves the first check past four times its end;
* a maximum below the floor completes the whole record and the report says
  the floor stopped the stop;
* the stop rule needs two checks in a row (a pure-logic sequence);
* the chunk hooks with a stop that never fires leave both loops bit-identical,
  and a stop at k steps returns k steps;
* refusals.
"""

from __future__ import annotations

import contextlib
import io
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse
from rfx import ringdown as rd
from rfx.ringdown import RingdownSpec
from tests.unit.sparams.test_ringdown_run import (
    BAR_DB, BAR_DEG, FREQS, N_LONG, _box, _db_deg, _digest, _long)

#: The PI's floor (2026-09-27), written here rather than read from the module
#: so that changing the module's constant is caught.
FLOOR = 0.5
LANES = ("uniform", "graded")


def _stop_run(sim, n, **kw):
    return sim.run(n_steps=n, compute_s_params=True, s_param_freqs=FREQS,
                   skip_preflight=True, ringdown=RingdownSpec(),
                   until_identified=True, **kw)


def _fixed_run(sim, n):
    return sim.run(n_steps=n, compute_s_params=True, s_param_freqs=FREQS,
                   skip_preflight=True, ringdown=RingdownSpec())


_STOPPED = {}


def _stopped(lane):
    """The early-stopped run on one lane and the fixed run at its stop (cached)."""
    if lane not in _STOPPED:
        r = _stop_run(_box(lane), N_LONG[lane])
        _STOPPED[lane] = (r, _fixed_run(_box(lane), r.ringdown.stop.n_stop))
    return _STOPPED[lane]


def _strong_slowest_tau(poles):
    """1 / decay rate of the slowest pole of amplitude >= 1e-3 of the largest."""
    a_max = max(p.amplitude for p in poles)
    return 1.0 / min(p.decay_per_s for p in poles if p.amplitude >= 1e-3 * a_max)


# ---------------------------------------------------------------------------
# The stop on the resonant box
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lane", LANES)
def test_the_run_stops_early_with_the_settled_s_parameters(lane):
    r, _fixed = _stopped(lane)
    st = r.ringdown.stop
    print(f"\n[{lane}]\n{st.summary()}")
    assert st.fired, st.reason
    assert st.n_stop < N_LONG[lane] / 4, (st.n_stop, N_LONG[lane])
    assert np.asarray(r.time_series).shape[0] == st.n_stop
    assert st.checks[-1].n_record == st.n_stop and st.checks[-1].stop
    assert r.ringdown.report.n_record == st.n_stop and r.ringdown.report.ok
    long = _long(lane)
    d_db, d_deg = _db_deg(r.ringdown.s_params[0, 0], long.ringdown.s_params[0, 0])
    p_db, p_deg = _db_deg(r.s_params[0, 0], long.ringdown.s_params[0, 0])
    print(f"[{lane}] stopped at {st.n_stop} steps: completed S11 against the "
          f"{N_LONG[lane]}-step record {d_db:.2e} dB / {d_deg:.2e} deg, plain "
          f"{p_db:.3g} dB / {p_deg:.3g} deg")
    assert d_db <= BAR_DB[lane] and d_deg <= BAR_DEG[lane], (d_db, d_deg)
    assert p_db > 10 * BAR_DB[lane], "the stopped record must still ring"


@pytest.mark.parametrize("lane", LANES)
def test_the_stopped_result_is_the_fixed_record_run_bit_for_bit(lane):
    r, fixed = _stopped(lane)
    a, b = {}, {}
    for f in r._fields:
        if f != "ringdown":
            _digest(getattr(r, f), f, a)
            _digest(getattr(fixed, f), f, b)
    assert a.keys() == b.keys()
    differ = [k for k in a if a[k] != b[k]]
    assert not differ, (lane, differ)
    ra, rb = {}, {}
    _digest(r.ringdown._replace(stop=None), "ringdown", ra)
    _digest(fixed.ringdown, "ringdown", rb)
    assert ra == rb, (lane, [k for k in ra if ra[k] != rb.get(k)])
    assert fixed.ringdown.stop is None
    we = fixed.ringdown.report.witness("WE").value
    assert r.ringdown.stop.checks[-1].we == we, (r.ringdown.stop.checks[-1].we, we)


@pytest.mark.parametrize("lane", LANES)
def test_the_floor_holds_the_stop_where_the_witness_already_agrees(lane):
    r, _fixed = _stopped(lane)
    st = r.ringdown.stop
    held = [c for c in st.checks[:-1] if c.sources_off and c.we_ok
            and c.consecutive >= 2 and not c.floor_ok]
    assert held, ("on this box WE agrees at two checks in a row before half a "
                  f"decay time; checks {[(c.n_record, c.we, c.record_over_tau) for c in st.checks]}")
    for c in st.checks:
        assert c.stop == (c is st.checks[-1])
        if c.record_over_tau < FLOOR:
            assert not c.stop, c
    last = st.checks[-1]
    assert last.record_over_tau >= FLOOR, last
    assert last.n_record * st.dt >= FLOOR * last.tau_slowest_s
    # the decay time the floor reads is the settled record's own slowest pole
    tau_long = _strong_slowest_tau(_long(lane).ringdown.report.poles)
    print(f"\n[{lane}] held at {[c.n_record for c in held]} steps "
          f"({[round(c.record_over_tau, 3) for c in held]} decay times); stopped "
          f"at {last.n_record} ({last.record_over_tau:.3f}); tau at the stop "
          f"{last.tau_slowest_s * 1e9:.4g} ns, settled record {tau_long * 1e9:.4g} ns")
    assert abs(last.tau_slowest_s / tau_long - 1.0) < 0.02


def test_no_check_reads_the_record_while_the_pulse_is_on():
    """The same pulse delayed by 40 of its widths instead of 4.5 (same
    spectrum) ends at about 1.34 ns instead of 0.25 ns. The oracle is the
    pulse itself, evaluated here: the first step after which it stays below
    source_off_tol of its peak. The first check lies at or past four times
    that step, so [T/4, T] of every check is free of it, and the run stops
    later than it does with the pulse on time."""
    long_pulse = GaussianPulse(f0=13.0e9, bandwidth=0.8, cutoff=40.0)
    sim = _box("uniform", pulse=long_pulse)
    dt = float(sim._build_grid().dt)
    n = N_LONG["uniform"]
    w = np.abs(np.asarray(jax.vmap(long_pulse)(jnp.arange(n, dtype=jnp.float32) * dt),
                          dtype=np.float64))
    n_off = int(np.nonzero(w > 1e-6 * w.max())[0][-1]) + 1
    r = _stop_run(sim, n)
    st = r.ringdown.stop
    print(f"\n[long pulse] off after {n_off} steps; first check {st.first_check}\n"
          f"{st.summary()}")
    assert st.first_check >= 4 * n_off - 2
    assert st.checks and all(c.sources_off for c in st.checks)
    assert all(c.n_record >= st.first_check for c in st.checks)
    assert st.fired and r.ringdown.report.witness("source_off").ok
    assert st.n_stop > _stopped("uniform")[0].ringdown.stop.n_stop


def test_a_check_with_the_source_on_does_not_read_the_record():
    """Condition (a) is judged before the record is read: at a record whose
    [T/4, T] holds the pulse, the check fails (a) without calling ``peek``."""
    sim = _box("uniform")
    stop = rd.RingdownStop(sim, RingdownSpec(), lane="uniform", n_max=4000,
                           grid=sim._build_grid())
    stop.n_user = 1

    def peek():
        raise AssertionError("the record was read with a source on")

    assert stop.after_chunk(stop.first_check - 1, peek) is False
    assert not stop.checks
    c = stop._check(400, peek)
    print(f"\n[source on] {c}")
    assert not c.sources_off and not c.stop and c.consecutive == 0
    assert c.source_ratio_long > 1e-6 or "still on" in c.note


def test_a_maximum_below_the_floor_completes_the_whole_record_and_says_why():
    """1501 steps are 0.48 of the uniform box's decay time: WE is within its
    bar at every check, the floor never passes, so the whole record is
    completed -- exactly as ``run(n_steps=1501, ringdown=...)`` completes it,
    every field bit for bit. 1501 is one step past a chunk boundary: the loop
    must not end on a one-step chunk, which XLA compiles differently."""
    n = 1501
    r = _stop_run(_box("uniform"), n)
    st = r.ringdown.stop
    print(f"\n[maximum reached]\n{st.summary()}")
    assert not st.fired and st.n_stop == n
    assert np.asarray(r.time_series).shape[0] == n
    assert st.checks and "(d)" in st.reason and "maximum" in st.reason
    assert all(c.we_ok and not c.floor_ok for c in st.checks)
    fixed = _fixed_run(_box("uniform"), n)
    a, b = {}, {}
    for f in r._fields:
        if f != "ringdown":
            _digest(getattr(r, f), f, a)
            _digest(getattr(fixed, f), f, b)
    assert not [k for k in a if a[k] != b[k]]
    ra, rb = {}, {}
    _digest(r.ringdown._replace(stop=None), "ringdown", ra)
    _digest(fixed.ringdown, "ringdown", rb)
    assert ra == rb


def test_a_static_field_does_not_hold_the_floor():
    """A closed box can keep a static field after the pulse (the E4 Q 533
    cavity did): the pencil identifies it as a pole at 0 Hz with |lambda|
    within rounding of 1, a decay time without end. Here a 10 GHz Q 200 mode
    (amplitude decay time 6.366 ns) rings on two channels on top of such a
    constant. The floor must read the ringing mode, so the stop fires at the
    first record past half its decay time -- 3250 steps of 1 ps, checked every
    250 steps with the real check function and stop rule -- and the report
    names the 10 GHz pole and the static pole it left out."""
    dt, f0, q = 1.0e-12, 10.0e9, 200.0
    alpha = math.pi * f0 / q
    t = np.arange(3250) * dt
    ring = np.exp(-alpha * t)
    Y = np.stack([ring * np.cos(2 * np.pi * f0 * t) + 0.5,
                  ring * np.sin(2 * np.pi * f0 * t + 0.3) - 0.3], axis=1)
    f_bins = np.linspace(5e9, 15e9, 41)
    # the completed spectrum is 2.5x the record's at half a decay time: scale
    # so the synthetic "S" stays below 1 and reads passive
    scale = 4.0 * float(np.max(np.abs(rd.plain_dft(Y - np.array([0.5, -0.3]),
                                                   dt, f_bins))))
    spec = RingdownSpec()
    sched = rd._StopSchedule(2500)
    for T in range(2500, 3251, 250):
        fields = rd._judge_record(Y[:T], dt, f_bins, T, round(spec.window_start * T),
                                  spec, ref_hz=20e9, observable=lambda a: a / scale)
        c = sched.record(T, T * dt, dict(sources_off=True, **fields), 0.0)
        print(f"\n[static field] T {T}: WE {c.we:.2e}, floor pole "
              f"{rd._pole_text(c.floor_pole)}, record {c.record_over_tau:.3f} "
              f"decay times, left out {[(p.f_hz, p.abs_lambda) for p in c.zero_frequency_poles]}")
        if c.stop:
            break
    tau = 1.0 / alpha
    last = sched.checks[-1]
    assert last.stop and last.n_record == 3250, [(c.n_record, c.we, c.record_over_tau)
                                                  for c in sched.checks]
    assert last.record_s >= FLOOR * tau > (last.n_record - 250) * dt
    assert abs(last.tau_slowest_s / tau - 1.0) < 1e-3
    assert abs(abs(last.floor_pole.f_hz) / f0 - 1.0) < 1e-4
    left = last.zero_frequency_poles
    assert left and all(abs(p.f_hz) < 1.0 / last.record_s for p in left)
    # the pole left out would have held the floor: it does not decay
    assert min(p.decay_per_s for p in left) < 1e-3 * alpha


# ---------------------------------------------------------------------------
# What the floor and the stop's other conditions are for: synthetic records
# through the stop's own check (_judge_record) and schedule (_StopSchedule)
# ---------------------------------------------------------------------------

_DT = 1.2148552052451076e-12


def _modes_record(modes, n, dt=_DT):
    """``sum_k amp_k exp(s_k t) + c.c.`` for ``(f, Q, amp)`` modes, rounded to
    float32 like an FDTD record, as one channel; and its settled spectrum."""
    s, r = [], []
    for f, q, amp in modes:
        w = 2 * np.pi * f
        s += [-w / (2 * q) + 1j * w, -w / (2 * q) - 1j * w]
        r += [amp, np.conj(amp)]
    s, r = np.array(s), np.array(r)
    t = np.arange(n) * dt
    y = (np.exp(np.outer(t, s)) @ r).real.astype(np.float32).astype(np.float64)

    def settled(freqs):
        z = np.exp(-2j * np.pi * np.asarray(freqs) * dt)
        return dt * (1.0 / (1.0 - np.exp(s * dt)[None, :] * z[:, None])) @ r

    return y[:, None], settled


def _one_port(modes, n, dt=1.0e-12, z0=50.0):
    """A one-port reflection ``S = -1 + sum_k h_k alpha_k / (alpha_k + j (w - w_k))``
    driven by a 10.5 GHz Gaussian pulse (``h`` the feature height), as the port
    channels ``V = a + b`` and ``I = (a - b) / Z0`` rounded to float32; with the
    settled S, the S observable and the step after which the pulse is off."""
    from scipy.signal import lfilter

    t = np.arange(n) * dt
    a = np.exp(-((t - 200e-12) ** 2) / (2 * (40e-12) ** 2)) * np.cos(
        2 * np.pi * 10.5e9 * (t - 200e-12))
    b = -1.0 * a
    poles = []
    for f, q, h in modes:
        alpha = math.pi * f / q
        lam = np.exp((-alpha + 2j * math.pi * f) * dt)
        b = b + 2.0 * np.real(h * alpha * lfilter([dt], [1.0, -lam],
                                                   a.astype(np.complex128)))
        poles.append((h * alpha, lam))
    Y = np.stack([(a + b).astype(np.float32).astype(np.float64),
                  ((a - b) / z0).astype(np.float32).astype(np.float64)], axis=1)
    n_off = int(np.nonzero(np.abs(a) > 1e-6 * np.abs(a).max())[0][-1]) + 1

    def settled(freqs):
        z = np.exp(-2j * np.pi * np.asarray(freqs) * dt)
        S = -np.ones(np.size(freqs), dtype=np.complex128)
        for R, lam in poles:
            S += R * dt / (1 - lam * z) + np.conj(R) * dt / (1 - np.conj(lam) * z)
        return S

    def to_s(sp):
        return (sp[:, 0] - z0 * sp[:, 1]) / (sp[:, 0] + z0 * sp[:, 1])

    return Y, settled, to_s, n_off


def _synthetic_stop(Y, dt, bins, obs, n_max, *, ref_hz, n_off=0):
    """The stop on a synthetic record: ``(checks, the completion's actual error
    at each check against the settled answer)``; the last check stopped when
    ``checks[-1].stop``."""
    spec = RingdownSpec()
    sched = rd._StopSchedule(rd._first_check_record(n_off, spec.window_start,
                                                    rd.STOP_CHUNK))
    errors = []
    for done in range(rd.STOP_CHUNK, n_max + 1, rd.STOP_CHUNK):
        if not sched.due(done):
            continue
        n0 = round(spec.window_start * done)
        fields = rd._judge_record(Y[:done], dt, bins, done, n0, spec,
                                  ref_hz=ref_hz, observable=obs)
        c = sched.record(done, done * dt, dict(sources_off=True,
                                               source_ratio_long=0.0, **fields), 0.0)
        model = rd.identify(Y[:done], dt, n0, done, freq_max=ref_hz)
        errors.append(np.asarray(obs(rd.plain_dft(Y[:done], dt, bins)
                                     + rd.tail_dft(model, done - 1, bins))))
        if c.stop:
            break
    return sched.checks, errors


def _print_checks(label, checks, errs):
    for c, e in zip(checks, errs):
        print(f"  [{label}] T {c.n_record}: WE {c.we:.2e}, {c.consecutive} in a row, "
              f"floor {rd._pole_text(c.floor_pole)} over {c.record_over_tau:.3f}, "
              f"max|S| {c.passivity:.3f}, error {e:.2e}, stop {c.stop}")


def test_the_floor_waits_for_a_pair_no_short_record_resolves():
    """The R-i blind spot, scaled to seconds: two Q 100 modes at 2.5 GHz, 0.12
    of a linewidth apart (0.12 %), beside a 2.03 GHz Q 30 mode. A short record
    identifies the pair as one blended pole in every window, so WE agrees at
    two checks in a row by 750 steps (0.07 of the pair's 12.7 ns decay time)
    while the completion is 4.5 % off. The floor holds the stop until half the
    decay time, where the pair is resolved and the completion is right."""
    modes = [(2.5e9, 100.0, 0.5 * np.exp(0.2j)), (2.5e9 * 1.0012, 100.0, 0.5 * np.exp(2.5j)),
             (2.03e9, 30.0, 0.3)]
    bins = 1.5e9 + 1e7 * np.arange(201)
    Y, settled = _modes_record(modes, 12000)
    truth = settled(bins)
    scale = 2.0 * float(np.max(np.abs(truth)))
    checks, outs = _synthetic_stop(Y, _DT, bins, lambda sp: np.asarray(sp)[:, 0] / scale,
                                   12000, ref_hz=4e9)
    errs = [float(np.max(np.abs(o - truth / scale))) for o in outs]
    _print_checks("R-i pair", checks, errs)
    tau = 100.0 / (math.pi * 2.5e9)
    early = [k for k, c in enumerate(checks) if c.consecutive >= 2 and c.record_s < FLOOR * tau]
    assert early and errs[early[0]] > 1e-2, "WE agreed early while the completion was off"
    assert checks[-1].stop and checks[-1].record_s >= FLOOR * tau
    assert errs[-1] < 1e-4, errs[-1]


def test_a_decaying_low_frequency_mode_holds_the_floor():
    """A decaying pair at 0.15 GHz (Q 60, 0.1 % apart, decay time 127 ns),
    period longer than the early records, beside a 2.5 GHz Q 60 mode: it is a
    mode, not a static field, so it sets the floor and nothing stops within
    6000 steps; at the checks where WE already agrees twice the completion is
    1 % off."""
    modes = [(2.5e9, 60.0, 0.5), (0.15e9, 60.0, 0.4 * np.exp(0.2j)),
             (0.15e9 * 1.001, 60.0, 0.4 * np.exp(2.5j))]
    bins = 0.05e9 + 1e7 * np.arange(396)
    Y, settled = _modes_record(modes, 6000)
    truth = settled(bins)
    scale = 2.0 * float(np.max(np.abs(truth)))
    checks, outs = _synthetic_stop(Y, _DT, bins, lambda sp: np.asarray(sp)[:, 0] / scale,
                                   6000, ref_hz=4e9)
    errs = [float(np.max(np.abs(o - truth / scale))) for o in outs]
    _print_checks("low pair", checks, errs)
    assert not any(c.stop for c in checks)
    agreed = [k for k, c in enumerate(checks) if c.consecutive >= 2]
    assert agreed and max(errs[k] for k in agreed) > 5e-3
    for k in agreed:
        p = checks[k].floor_pole
        assert p is not None and abs(abs(p.f_hz) / 0.15e9 - 1.0) < 0.01, checks[k]
        assert not checks[k].zero_frequency_poles


def test_a_weakly_coupled_high_q_pair_holds_the_floor():
    """A Q 40 mode at 10 GHz (feature 0.8 in S) and a pair of Q 50,000 modes at
    11 GHz, 0.1 % apart, coupled 8x more weakly (feature 0.1 each): small in
    time, tall in frequency. Their tails move S by more than the bar, so they
    set the floor (decay time 1.4 us) and nothing stops within 4000 steps; at
    the checks where WE agrees twice the completed S is 0.1 off."""
    Y, settled, to_s, n_off = _one_port(
        [(10.0e9, 40.0, 0.8), (11.0e9, 50000.0, 0.1), (11.011e9, 50000.0, 0.1)], 4000)
    bins = np.linspace(8e9, 13e9, 501)
    truth = settled(bins)
    checks, outs = _synthetic_stop(Y, 1.0e-12, bins, to_s, 4000, ref_hz=20e9, n_off=n_off)
    errs = [float(np.max(np.abs(o - truth))) for o in outs]
    _print_checks("weak pair", checks, errs)
    assert not any(c.stop for c in checks)
    agreed = [k for k, c in enumerate(checks) if c.consecutive >= 2]
    assert agreed and min(errs[k] for k in agreed) > 5e-2
    for k in agreed:
        p = checks[k].floor_pole
        assert abs(abs(p.f_hz) / 11.0e9 - 1.0) < 2e-3 and p.q > 5000, checks[k]


def test_a_blended_pair_between_coarse_bins_holds_the_floor():
    """The same kind of weak Q 50,000 pair, 0.37 % apart, seen on a common
    100 MHz sweep (8-18 GHz, 101 bins): its members sit 0.5 MHz and 40.5 MHz
    above the 11.0 GHz bin, and a short record identifies them as one blended
    pole between bins. Measured at the bins, that pole's tail read below the
    bar and the stop fired at 3000 steps with the completed S 2e-2 off (the
    fresh-eyes recheck of #1254 PR 2). A pole is known only to about 1/T, so
    the floor weighs it on the nearest bin within 1/T: the blend sets the
    floor and nothing stops within 8000 steps."""
    Y, settled, to_s, n_off = _one_port(
        [(10.0e9, 40.0, 0.8), (11.0005e9, 50000.0, 0.1), (11.0405e9, 50000.0, 0.1)], 8000)
    bins = np.linspace(8e9, 18e9, 101)
    truth = settled(bins)
    checks, outs = _synthetic_stop(Y, 1.0e-12, bins, to_s, 8000, ref_hz=20e9, n_off=n_off)
    errs = [float(np.max(np.abs(o - truth))) for o in outs]
    _print_checks("coarse-bin pair", checks, errs)
    assert not any(c.stop for c in checks)
    agreed = [k for k, c in enumerate(checks) if c.consecutive >= 2 and errs[k] > 1e-3]
    assert agreed, "no check where WE agreed twice while the completion was off"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "known limitation (PI 2026-09-27, #1381): a weak high-Q pair that no window "
    "of the record separates beats, the short record reads the beat as fast "
    "decay (apparent Q 619 for a true 50,000), and the blended pole's tail then "
    "weighs below the bar, so the stop can fire before the pair is resolved"))
def test_a_weak_pair_whose_beat_reads_as_decay_holds_the_floor():
    """A Q 40 mode at 10 GHz and a weak pair of Q 50,000 modes at 11.000 and
    11.060 GHz (feature 0.03 each) on a 100 MHz sweep: the stop fires at 2250
    steps with the completed S 3e-2 off at the 11.000 GHz bin while WE reads
    about 1e-4 (the fresh-eyes review of the 1/T rule). Stated in
    docs/guides/known_limitations.md; a floor rule that sees this case would
    turn this test XPASS."""
    Y, settled, to_s, n_off = _one_port(
        [(10.0e9, 40.0, 0.8), (11.0e9, 50000.0, 0.03), (11.06e9, 50000.0, 0.03)], 8000)
    bins = np.linspace(8e9, 18e9, 101)
    truth = settled(bins)
    checks, outs = _synthetic_stop(Y, 1.0e-12, bins, to_s, 8000, ref_hz=20e9, n_off=n_off)
    errs = [float(np.max(np.abs(o - truth))) for o in outs]
    _print_checks("beating weak pair", checks, errs)
    assert not any(c.stop and e > 1e-3 for c, e in zip(checks, errs))


def test_a_pole_the_record_cannot_tell_from_dc_is_not_moved_onto_a_bin():
    """For the floor a pole is weighed on the nearest bin within 1/T, but a
    pole within 1/T of DC stays where it is: moved onto a low bin, a slowly
    relaxing static field would weigh like a resonance there (the fresh-eyes
    review of the 1/T rule). T = 2.25 ns, 1/T = 444 MHz: the 0.2 GHz bin is
    within 1/T of the DC pole, which stays; the 10 GHz pole moves onto the
    10.01 GHz bin."""
    T = 2.25e-9
    s = np.array([-5.0e6 + 0.0j, -1.0e8 + 2j * np.pi * 10.0e9, -1.0e8 - 2j * np.pi * 10.0e9])
    bins = np.array([0.2e9, 5.0e9, 10.01e9])
    out = rd._on_nearest_bin(s, bins, T)
    assert out[0] == s[0]
    assert out[1] == s[1].real + 2j * np.pi * 10.01e9
    assert out[2] == s[2]          # no positive bin within 1/T of -10 GHz


def test_a_non_passive_completion_does_not_stop():
    """A single Q 40 mode whose feature (2.5) makes the settled |S| 1.5 at its
    resonance: WE agrees and the record passes the floor by 1750 steps, but the
    completion reads non-passive, as the report's passivity witness would, so
    the run does not stop there."""
    Y, settled, to_s, n_off = _one_port([(10.0e9, 40.0, 2.5)], 3000)
    bins = np.linspace(8e9, 13e9, 501)
    checks, _ = _synthetic_stop(Y, 1.0e-12, bins, to_s, 3000, ref_hz=20e9, n_off=n_off)
    for c in checks:
        print(f"  [non-passive] T {c.n_record}: WE {c.we:.2e}, {c.consecutive} in a row, "
              f"floor ok {c.floor_ok}, max|S| {c.passivity:.4f} ok {c.passive_ok}, stop {c.stop}")
    assert not any(c.stop for c in checks)
    last = checks[-1]
    assert last.consecutive >= 2 and last.floor_ok and last.growing_ok
    assert not last.passive_ok and abs(last.passivity - 1.5) < 0.01


def test_the_floor_reads_the_largest_decay_time_the_consecutive_rule_rests_on():
    """Pure rule: two checks in a row agree, the first read a slow pole (100 ns)
    and the second only a fast one (1 ns). The floor uses 100 ns, so a record
    of 10 ns (0.1 of it) does not stop; a third agreeing check that also reads
    1 ns rests on checks two and three only and stops."""
    sched = rd._StopSchedule(250)
    ok = dict(sources_off=True, we=1e-5, we_ok=True, passive_ok=True, growing_ok=True,
              passivity=0.9)
    a = sched.record(250, 5e-9, dict(ok, tau_own_s=100e-9), 0.0)
    b = sched.record(500, 10e-9, dict(ok, tau_own_s=1e-9), 0.0)
    c = sched.record(750, 11e-9, dict(ok, tau_own_s=1e-9), 0.0)
    assert (a.stop, b.stop, c.stop) == (False, False, True)
    assert b.tau_slowest_s == 100e-9 and b.floor_from == 250 and not b.floor_ok
    assert c.tau_slowest_s == 1e-9 and c.floor_from == 750


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def _decide(sequence):
    """Stop index of a sequence of (sources_off, we_ok, floor_ok) checks, or None."""
    run = 0
    for k, (a, b, d) in enumerate(sequence):
        run, stop = rd._stop_decision(run, a, b, d)
        if stop:
            return k
    return None


@pytest.mark.parametrize("sequence, expected", [
    ([(True, True, True)], None),                                  # one check is not two
    ([(True, False, True), (True, True, True)], None),             # WE failed just before
    ([(False, False, False), (True, True, True)], None),           # a source was on before
    ([(True, True, False), (True, True, True)], 1),                # floor only at the stop
    ([(True, True, True), (True, False, True), (True, True, True),
      (True, True, True)], 3),                                     # a failure restarts the count
    ([(True, True, False)] * 5, None),                             # never past the floor
])
def test_the_stop_needs_two_checks_in_a_row(sequence, expected):
    assert _decide(sequence) == expected


# ---------------------------------------------------------------------------
# The hooks: a stop that never fires changes nothing
# ---------------------------------------------------------------------------

def _uniform_low_level(n, **kw):
    """A CPML box with a dielectric inclusion, a DC-free pulse and two probes."""
    from rfx import simulation as _sim
    from rfx.core.yee import MaterialArrays
    from rfx.grid import Grid

    grid = Grid(freq_max=20e9, domain=(0.006, 0.006, 0.006), dx=5e-4, cpml_layers=6)
    shape = (grid.nx, grid.ny, grid.nz)
    eps = jnp.ones(shape, dtype=jnp.float32).at[10:14, 10:14, 10:14].set(4.0)
    mats = MaterialArrays(eps_r=eps, sigma=jnp.zeros(shape, dtype=jnp.float32),
                          mu_r=jnp.ones(shape, dtype=jnp.float32))
    t = jnp.arange(220, dtype=jnp.float32) * grid.dt
    x = (t - 30 * grid.dt) / (10 * grid.dt)
    wf = (-2.0 * x * jnp.exp(-(x ** 2)))[:n]
    c = (grid.nx // 2, grid.ny // 2, grid.nz // 2)
    src = [_sim.SourceSpec(i=c[0], j=c[1], k=c[2], component="ez", waveform=wf)]
    probes = [_sim.ProbeSpec(i=c[0] - 4, j=c[1], k=c[2], component="ez"),
              _sim.ProbeSpec(i=c[0] + 4, j=c[1], k=c[2], component="ez")]
    return _sim.run(grid, mats, n, boundary="cpml", sources=src, probes=probes, **kw)


def _same(a, b):
    assert np.array_equal(np.asarray(a.time_series), np.asarray(b.time_series))
    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        assert np.array_equal(np.asarray(getattr(a.state, comp)),
                              np.asarray(getattr(b.state, comp))), comp


def test_the_uniform_scan_hook_is_a_continuation():
    seen = []

    def never(steps, peek):
        seen.append((steps, np.asarray(peek().time_series).shape[0]))
        return False

    plain = _uniform_low_level(220)
    _same(_uniform_low_level(220, stop_fn=never, stop_interval=50), plain)
    assert seen == [(50, 50), (100, 100), (150, 150), (200, 200), (220, 220)]
    seen.clear()
    _same(_uniform_low_level(201, stop_fn=never, stop_interval=50),
          _uniform_low_level(201))
    assert [s for s, _ in seen] == [50, 100, 150, 201], "no one-step last chunk"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        stopped = _uniform_low_level(220, stop_fn=lambda s, p: s >= 100,
                                     stop_interval=50, report_every=80)
    assert np.asarray(stopped.time_series).shape[0] == 100
    _same(stopped, _uniform_low_level(100))
    assert out.getvalue().strip().splitlines()[-1].split()[1].startswith("100/220")


def test_the_graded_loop_hook_is_a_continuation():
    from rfx.nonuniform import run_nonuniform_until_decay
    from tests.unit.nonuniform.test_nonuniform_until_decay import _build_nu_case

    grid, materials, sources, probes = _build_nu_case(200)

    def loop(max_steps, **kw):
        r = run_nonuniform_until_decay(
            grid, materials, decay_by=0.0, check_interval=50,
            min_steps=max_steps + 1, max_steps=max_steps,
            decay_energy_consecutive=1, sources=sources, probes=probes, **kw)
        return SimpleNamespace(time_series=r["time_series"], state=r["state"])

    seen = []

    def never(steps, peek):
        seen.append((steps, np.asarray(peek()["time_series"]).shape[0]))
        return False

    plain = loop(200)
    _same(loop(200, stop_fn=never), plain)
    assert seen == [(50, 50), (100, 100), (150, 150), (200, 200)]
    stopped = loop(200, stop_fn=lambda s, p: s >= 100)
    assert np.asarray(stopped.time_series).shape[0] == 100
    _same(stopped, loop(100))
    # a stop between two progress lines (every 80 steps: 100, then 200) prints
    # its own line at the stop
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        stopped = loop(200, stop_fn=lambda s, p: s >= 150, report_every=80)
    assert np.asarray(stopped.time_series).shape[0] == 150
    lines = [ln.split()[1:3] for ln in out.getvalue().strip().splitlines()]
    assert lines == [["100/200", "(cap)"], ["150/200", "(cap)"]], lines
    seen.clear()
    loop(151, stop_fn=never)
    assert [s for s, _ in seen] == [50, 100, 151], "no one-step last chunk"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def _with_plane(sim):
    sim.add_dft_plane_probe(axis="z", coordinate=2.0e-3, component="ez",
                            freqs=np.array([12e9]), name="plane")
    return sim


def _with_flux(sim):
    sim.add_flux_monitor(axis="z", coordinate=2.0e-3, freqs=np.array([12e9]),
                         name="flux")
    return sim


def _with_ntff(sim):
    sim.add_ntff_box((2.0e-3, 2.0e-3, 1.0e-3), (10.0e-3, 9.0e-3, 4.0e-3),
                     freqs=np.array([12e9]))
    return sim


def _with_moments(sim):
    sim.add_current_moment_monitor((2.0e-3, 2.0e-3, 1.0e-3), (10.0e-3, 9.0e-3, 4.0e-3),
                                   block_size=2.0e-3, freqs=np.array([12e9]))
    return sim


@pytest.mark.parametrize("kwargs, extra, exc, match", [
    ({"until_identified": 1}, None, TypeError, "True or False"),
    ({"until_decay": 1e-3}, None, ValueError, "two rules"),
    ({"snapshot": "SNAP"}, None, NotImplementedError, "snapshot"),
    ({}, _with_plane, NotImplementedError, "DFT plane"),
    ({}, _with_flux, NotImplementedError, "flux monitor"),
    ({}, _with_ntff, NotImplementedError, "NTFF box"),
    ({}, _with_moments, NotImplementedError, "current-moment monitor"),
])
def test_refusals(kwargs, extra, exc, match):
    from rfx.simulation import SnapshotSpec

    sim = _box("uniform")
    if extra is not None:
        sim = extra(sim)
    kw = {"until_identified": True, **kwargs}
    if kw.get("snapshot") == "SNAP":
        kw["snapshot"] = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=2)
    with pytest.raises(exc, match=match):
        sim.run(n_steps=4000, compute_s_params=True, s_param_freqs=FREQS,
                skip_preflight=True, ringdown=RingdownSpec(), **kw)


def test_until_identified_needs_ringdown():
    with pytest.raises(ValueError, match="needs ringdown="):
        _box("uniform").run(n_steps=4000, skip_preflight=True, until_identified=True)


def test_a_maximum_whose_window_holds_the_pulse_is_refused_before_the_run():
    """The refusal of ``run(n_steps=n_max, ringdown=...)`` applies to the
    maximum record, before a step is taken."""
    with pytest.raises(ValueError, match="still on"):
        _stop_run(_box("uniform"), 150)


def test_the_floor_is_the_pis_half_decay_time():
    """The PI's decision (2026-09-27): no stop before half the amplitude decay
    time Q / (pi f) of the slowest identified mode. The rest of the rule is the
    lead's, not pinned here: which poles count as identified modes (a pole
    whose own tail moves the completed S by witness_tol of its peak, a static
    field left out), the largest decay time over the checks the two-in-a-row
    rule rests on, checks 1.25x apart on 250-step chunks, and passivity and
    growing poles read ok at the stopping check."""
    assert rd.STOP_FLOOR_DECAY_TIMES == FLOOR == 0.5
