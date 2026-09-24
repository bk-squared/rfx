"""Synthetic oracles for the ring-down completion (``rfx/ringdown.py``, issue #1254).

A record made of known damped oscillations, ``y_n = sum_k r_k lambda_k**n``,
has a known infinite-record DFT, ``dt * sum_k r_k / (1 - lambda_k z)``. The
completion must reproduce it from a window of a record that stops while the
oscillations are still large, and the poles it identifies must be the ones
the record was made of. The signals are real (conjugate pole pairs), several
channels share the poles with different residues, and they are sampled at the
steps of two research boards so harminv's decimation plan runs exactly as it
does there: 1.2149 ps (a patch antenna, freq_max 4 GHz) and 0.9533 ps (a
dielectric-filled cavity with a 2.5 % near-degenerate pair, freq_max 12 GHz).
numpy only; no JAX and no x64 flag here.
"""

from __future__ import annotations

import numpy as np
import pytest

from rfx import ringdown as rd

DT = 1.2148552052451076e-12            # patch board step (s)
F_MAX = 4.0e9                           # that board's freq_max
N_RECORD = 8000                         # 9.72 ns: the slow modes still ring
FREQS = 1.5e9 + 1.0e7 * np.arange(201)

# (f Hz, Q, residue channel 0, residue channel 1); each one conjugate pair.
MODES = (
    (2.03e9, 60.0, 0.50 * np.exp(0.3j), 0.15 * np.exp(-1.1j)),
    (2.51e9, 100.0, 0.40 * np.exp(-0.7j), 0.50 * np.exp(2.0j)),
    (2.95e9, 80.0, 0.20 * np.exp(1.9j), 0.25 * np.exp(0.4j)),
    (3.40e9, 40.0, 0.25 * np.exp(-2.5j), 0.10 * np.exp(1.2j)),
)


def _true_poles(modes=MODES):
    s, r = [], []
    for f, q, *res in modes:
        w = 2.0 * np.pi * f
        alpha = w / (2.0 * q)
        for sign in (+1, -1):
            s.append(-alpha + sign * 1j * w)
            r.append([v if sign > 0 else np.conj(v) for v in res])
    return np.asarray(s), np.asarray(r)


def _record(n=N_RECORD, dt=DT, modes=MODES):
    s, r = _true_poles(modes)
    t = np.arange(n) * dt
    y = np.exp(np.outer(t, s)) @ r
    assert np.max(np.abs(y.imag)) < 1e-12 * np.max(np.abs(y.real))
    return y.real


def _analytic_infinite_dft(dt=DT, freqs=FREQS, modes=MODES):
    s, r = _true_poles(modes)
    lam = np.exp(s * dt)
    z = np.exp(-2j * np.pi * freqs * dt)
    return dt * (1.0 / (1.0 - lam[None, :] * z[:, None])) @ r


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


def _model(s, c, n_ref, dt=DT):
    """A RingdownModel holding given poles and residues (the true ones)."""
    return rd.RingdownModel(
        s=np.asarray(s, dtype=np.complex128), c=np.asarray(c, dtype=np.complex128),
        n_ref=int(n_ref), dt=float(dt), freq_max=F_MAX, guard=None, factors=(),
        D=1, rank=len(s), n_window=0, n_decimated=0, n_growing_discarded=0,
        n_invalid=0, n_guard_dropped=0, singular_values=np.zeros(0))


def _identify(y, dt=DT, f_max=F_MAX, **kw):
    n = y.shape[0]
    return rd.identify(y, dt, int(round(0.5 * n)), n, freq_max=f_max, **kw)


# ---------------------------------------------------------------------------
# The closed form and its index convention
# ---------------------------------------------------------------------------

def test_the_tail_is_large_on_this_record():
    """The record stops while the slow modes ring, so a wrong tail cannot hide:
    the unrecorded part is a sizeable fraction of the infinite DFT."""
    y = _record()
    frac = _rel(_analytic_infinite_dft(), rd.plain_dft(y, DT, FREQS))
    assert frac > 0.05, frac


def test_tail_index_convention_with_the_true_model():
    """Plain DFT of samples 0..N plus the closed-form tail from sample N+1 is the
    infinite DFT, with the true residues referred to sample 0 and to a later
    sample. A tail that starts one sample late (or early) fails this by far."""
    s, r = _true_poles()
    y = _record()
    y_inf = _analytic_infinite_dft()
    got = rd.complete_spectra(y, DT, FREQS, _model(s, r, 0), N_RECORD)
    assert _rel(got, y_inf) < 1e-11, _rel(got, y_inf)
    n_ref = 3217
    got2 = rd.complete_spectra(y, DT, FREQS,
                               _model(s, r * np.exp(s * DT * n_ref)[:, None], n_ref),
                               N_RECORD)
    assert _rel(got2, y_inf) < 1e-11, _rel(got2, y_inf)
    # the same record cut shorter: its own tail, the same infinite DFT
    got3 = rd.complete_spectra(y, DT, FREQS, _model(s, r, 0), 5000)
    assert _rel(got3, y_inf) < 1e-11, _rel(got3, y_inf)


def test_tail_keeps_its_digits_next_to_a_slow_pole():
    """``1 - lambda z`` is taken as ``-expm1``: for a Q = 1e7 pole sitting on a
    bin the naive ``1 - exp(x)`` has ~8 digits left; the closed form here
    matches the same sum written with ``expm1`` in the other order."""
    f0, q = 2.5e9, 1.0e7
    alpha = np.pi * f0 / q
    s = np.array([-alpha + 2j * np.pi * f0])
    c = np.array([[1.0 + 0.0j]])
    freqs = np.array([f0])
    tail = rd.tail_dft(_model(s, c, 0), 99, freqs)[0, 0]
    x = s[0] * DT - 2j * np.pi * f0 * DT
    ref = DT * np.exp(x * 100) / (-np.expm1(x))
    naive = DT * np.exp(x * 100) / (1.0 - np.exp(x))
    assert abs(tail - ref) / abs(ref) < 1e-15
    assert abs(naive - ref) / abs(ref) > 1e-12     # the digits expm1 keeps


# ---------------------------------------------------------------------------
# Identification on a window
# ---------------------------------------------------------------------------

def test_completion_from_the_window_is_the_infinite_dft():
    """Identify on [N/2, N) and complete: the analytic infinite-record DFT to
    1e-9 on every bin, both channels, with harminv's decimation running."""
    y = _record()
    model = _identify(y)
    assert model.factors, "the board's dt must exercise harminv's decimation"
    got = rd.complete_spectra(y, DT, FREQS, model, N_RECORD)
    err = _rel(got, _analytic_infinite_dft())
    assert err < 1e-9, (err, model.rank, model.factors)


def test_the_poles_are_the_ones_the_record_was_made_of():
    y = _record()
    model = _identify(y)
    s_true, _ = _true_poles()
    assert model.rank == s_true.size and model.s.size == s_true.size
    assert model.n_growing_discarded == 0 and model.n_guard_dropped == 0
    for st in s_true:
        sg = model.s[int(np.argmin(np.abs(model.s - st)))]
        assert abs(sg.imag - st.imag) / abs(st.imag) < 1e-8, (st, sg)
        assert abs(sg.real - st.real) / abs(st.real) < 1e-8, (st, sg)


def test_float32_noise_grows_no_pole_and_stays_at_noise_level():
    """float32 quantization plus 1e-7-of-peak white noise: the rank rule keeps
    the true rank, no growing pole is kept, the error stays near the noise."""
    rng = np.random.default_rng(20260923)
    y = _record()
    y32 = y.astype(np.float32).astype(np.float64)
    y32 = y32 + 1e-7 * np.max(np.abs(y)) * rng.standard_normal(y.shape)
    model = _identify(y32)
    assert model.n_growing_discarded == 0
    assert model.rank == 2 * len(MODES), model.rank
    assert np.all(np.abs(model.lam) <= 1.0 + 1e-6)
    err = _rel(rd.complete_spectra(y32, DT, FREQS, model, N_RECORD),
               _analytic_infinite_dft())
    assert err < 1e-5, err


def test_a_growing_component_is_discarded_and_counted():
    y = _record()
    t = np.arange(N_RECORD) * DT
    grow = 1e-2 * np.exp(5.0e6 * t) * np.cos(2.0 * np.pi * 2.2e9 * t)
    model = _identify(y + grow[:, None])
    assert model.n_growing_discarded == 2, model.n_growing_discarded
    assert np.all(np.abs(model.lam) <= 1.0 + 1e-6)


@pytest.mark.parametrize("n_window", [1235, 2058, 6431])
def test_the_decimation_plan_is_harminvs(n_window):
    from importlib import import_module
    harminv = import_module("rfx.harminv")
    factors, kept = rd.decimation_plan(n_window, DT, F_MAX)
    ref = harminv._decimation_plan(n_window, DT, F_MAX, "auto")
    assert (factors, kept) == (tuple(ref[0]), ref[1])


# ---------------------------------------------------------------------------
# The decimation reference is the run's band
# ---------------------------------------------------------------------------

DT_CAV = 9.532874347654823e-13         # cavity board step (s)
F_MAX_CAV = 12.0e9
FREQS_CAV = 5.5e9 + 2.0e7 * np.arange(251)
MODES_CAV = (
    (6.086e9, 450.0, 0.40 * np.exp(0.2j), 0.30 * np.exp(-0.9j)),
    (9.499e9, 600.0, 0.30 * np.exp(-0.6j), 0.02 * np.exp(1.1j)),
    (9.745e9, 550.0, 0.28 * np.exp(1.7j), 0.03 * np.exp(-2.0j)),
)


def test_decimating_toward_a_reference_below_the_band_loses_the_pair():
    """At 0.95 ps a 4 GHz reference decimates a 12.4 ns window by 65, whose
    Nyquist (8.1 GHz) filters out the 9.50 / 9.75 GHz pair; the band's own
    12 GHz reference decimates by 21 and completes the pair to 1e-8."""
    n = 26000
    y = _record(n, DT_CAV, MODES_CAV)
    y_inf = _analytic_infinite_dft(DT_CAV, FREQS_CAV, MODES_CAV)
    assert int(np.prod(rd.decimation_plan(n // 2, DT_CAV, 4.0e9)[0])) == 65
    m_band = _identify(y, DT_CAV, F_MAX_CAV)
    assert m_band.D == 21
    err_band = _rel(rd.complete_spectra(y, DT_CAV, FREQS_CAV, m_band, n), y_inf)
    err_low = _rel(rd.complete_spectra(y, DT_CAV, FREQS_CAV,
                                       _identify(y, DT_CAV, 4.0e9), n), y_inf)
    assert err_band < 1e-8, err_band
    assert err_low > 1e-2, err_low
    f = m_band.s.imag / (2.0 * np.pi)
    for f0 in (9.499e9, 9.745e9):
        assert np.min(np.abs(f - f0)) < 1e3, f0


# ---------------------------------------------------------------------------
# The transition-band guard
# ---------------------------------------------------------------------------

def test_the_guard_removes_the_branch_cut():
    """A decimated eigenvalue on the negative real axis is the decimated
    Nyquist frequency. Which side of the log's cut it lands on is decided by
    the sign of a rounding-level imaginary part, and the two sides map to
    original-rate frequencies ``1 / (D dt)`` apart -- the completed spectrum
    jumps with a rounding error. The guard drops such a pole, so both roundings
    give the same (empty) model."""
    D = 10
    up = np.array([-0.8 + 1e-17j])
    down = np.array([-0.8 - 1e-17j])
    s_up, n_up = rd._to_original_rate(up, D, DT, None)
    s_dn, n_dn = rd._to_original_rate(down, D, DT, None)
    jump_hz = abs(s_up[0].imag - s_dn[0].imag) / (2.0 * np.pi)
    assert n_up == n_dn == 0
    assert abs(jump_hz * D * DT - 1.0) < 1e-12, jump_hz
    g_up, gn_up = rd._to_original_rate(up, D, DT, 0.9)
    g_dn, gn_dn = rd._to_original_rate(down, D, DT, 0.9)
    assert gn_up == gn_dn == 1 and g_up.size == g_dn.size == 0
    # an in-band pole passes the guard untouched, and without decimation there
    # is no cut to guard (exp(s dt) is lambda itself)
    inband = np.array([0.9 * np.exp(0.3j * np.pi)])
    assert rd._to_original_rate(inband, D, DT, 0.9)[0].size == 1
    assert rd._to_original_rate(up, 1, DT, 0.9)[0].size == 1


def test_the_guard_drops_no_in_band_content():
    """With 1e-5-of-peak noise the pencil keeps ~140 poles, a dozen of them in
    the anti-alias filter's transition band. The guarded and unguarded
    completions then differ by far less than either differs from the truth
    (measured 6e-9 against 9e-6): what the guard drops carries nothing in the
    band."""
    rng = np.random.default_rng(1254)
    y = _record()
    y = y + 1e-5 * np.max(np.abs(y)) * rng.standard_normal(y.shape)
    m_g = _identify(y, guard=0.9)
    m_0 = _identify(y, guard=None)
    assert m_g.n_guard_dropped > 0, "the case must exercise the guard"
    c_g = rd.complete_spectra(y, DT, FREQS, m_g, N_RECORD)
    c_0 = rd.complete_spectra(y, DT, FREQS, m_0, N_RECORD)
    err = _rel(c_g, _analytic_infinite_dft())
    assert err < 1e-4, err
    assert _rel(c_g, c_0) < 1e-2 * err, (_rel(c_g, c_0), err)


# ---------------------------------------------------------------------------
# The two-window witness and the growing-pole rule
# ---------------------------------------------------------------------------

def test_the_two_window_witness_is_small_on_a_clean_record_and_flags_a_short_one():
    y = _record()
    w = rd.two_window_witness(y, DT, FREQS, N_RECORD, N_RECORD // 2, freq_max=F_MAX)
    assert w.value < 1e-9 * np.max(np.abs(w.spectra)), w.value
    assert _rel(w.spectra, _analytic_infinite_dft()) < 1e-9
    # a record cut at 0.5 ns, while the pulse-like start dominates: the two
    # windows disagree, and W2 says so at the size of the actual error
    rng = np.random.default_rng(7)
    n_short = 1200
    ys = _record(n_short) + 1e-4 * rng.standard_normal((n_short, 2))
    ws = rd.two_window_witness(ys, DT, FREQS, n_short, n_short // 2, freq_max=F_MAX)
    actual = float(np.max(np.abs(ws.spectra - _analytic_infinite_dft())))
    assert ws.value > 1e-3 * np.max(np.abs(ws.spectra))
    assert 0.1 <= ws.value / actual <= 10.0, (ws.value, actual)


def test_the_growing_pole_rule_exempts_only_zero_frequency():
    record_s = 1e-8
    s = np.array([
        1e-9 / DT + 1e3j,                               # |lambda| = 1 + 1e-9, 159 Hz
        1e-9 / DT + 2j * np.pi * 3e9,                   # same growth at 3 GHz
        -1e8 + 2j * np.pi * 2e9,                        # decaying
    ])
    grow, exempt = rd.growing_poles(_model(s, np.ones((3, 1)), 0), record_s, 1e-6)
    assert len(exempt) == 1 and abs(exempt[0].f_hz) < 1e3
    assert len(grow) == 1 and abs(grow[0].f_hz - 3e9) < 1.0


@pytest.mark.parametrize("kw", [
    {"window_start": 0.95}, {"window_start": 0.0}, {"split": 1.0},
    {"guard": 0.0}, {"guard": 1.5}, {"witness_tol": 0.0}, {"freq_max": -1.0},
])
def test_the_spec_refuses_what_it_cannot_mean(kw):
    with pytest.raises(ValueError):
        rd.RingdownSpec(**kw)
