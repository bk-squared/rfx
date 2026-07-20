"""Tests for the RF-friendly ``Result`` accessors + one-call plotting.

Covers the thin convenience layer added to ``rfx.api._spec.Result``:
1-indexed S-parameter accessors (``s(m, n)``, ``s11``..``s22``,
``s_db``), the ``freqs_hz`` property, the clear errors when
S-parameters / dt are absent, and that the plot wrappers delegate to the
existing engine (``rfx.visualize`` / ``rfx.smith``) and return matplotlib
objects.  Uses synthetic, known arrays (no full sim) and the Agg backend
so the suite is headless-safe.
"""

import numpy as np
import pytest

from rfx.api._spec import Result

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# A synthetic 2-port S-matrix with a unique, recognisable value in every
# (m, n, k) slot so a wrong axis order is immediately caught:
#   s_params[i, j, k] = (i+1) + 1j*(j+1) + 0.01*k
N_FREQS = 5
FREQS = np.linspace(1e9, 5e9, N_FREQS)  # Hz


def _two_port():
    sp = np.empty((2, 2, N_FREQS), dtype=complex)
    for i in range(2):
        for j in range(2):
            sp[i, j, :] = (i + 1) + 1j * (j + 1) + 0.01 * np.arange(N_FREQS)
    return Result(
        state=None,
        time_series=np.zeros((10, 2)),
        s_params=sp,
        freqs=FREQS,
        dt=1e-12,
    )


def _no_sparams():
    return Result(
        state=None,
        time_series=np.zeros((10, 2)),
        s_params=None,
        freqs=None,
    )


def test_freqs_hz_matches_field():
    r = _two_port()
    np.testing.assert_array_equal(r.freqs_hz, FREQS)


def test_s_mn_uses_real_layout():
    """``s(m, n)`` must slice ``s_params[m-1, n-1, :]`` (1-indexed ports)."""
    r = _two_port()
    sp = np.asarray(r.s_params)
    for m in (1, 2):
        for n in (1, 2):
            np.testing.assert_array_equal(r.s(m, n), sp[m - 1, n - 1, :])


def test_s_shortcuts_match_s_mn():
    r = _two_port()
    np.testing.assert_array_equal(r.s11(), r.s(1, 1))
    np.testing.assert_array_equal(r.s21(), r.s(2, 1))
    np.testing.assert_array_equal(r.s12(), r.s(1, 2))
    np.testing.assert_array_equal(r.s22(), r.s(2, 2))


def test_s_db_matches_formula():
    r = _two_port()
    expected = 20.0 * np.log10(np.maximum(np.abs(r.s(2, 1)), 1e-10))
    np.testing.assert_allclose(r.s_db(2, 1), expected)


def test_s_db_guards_zero():
    sp = np.zeros((1, 1, N_FREQS), dtype=complex)
    r = Result(state=None, time_series=np.zeros((10, 1)),
               s_params=sp, freqs=FREQS)
    db = r.s_db(1, 1)
    assert np.all(np.isfinite(db))
    np.testing.assert_allclose(db, 20.0 * np.log10(1e-10))


def test_out_of_range_port_raises():
    r = _two_port()
    with pytest.raises(ValueError, match="2 port"):
        r.s(3, 1)
    with pytest.raises(ValueError, match="out of range"):
        r.s(1, 0)


def test_s_accessor_no_sparams_raises_clear_error():
    r = _no_sparams()
    with pytest.raises(ValueError, match="compute_s_params=True"):
        r.s(1, 1)
    with pytest.raises(ValueError, match="compute_s_params=True"):
        r.s11()


def test_freqs_hz_no_freqs_raises_clear_error():
    r = _no_sparams()
    with pytest.raises(ValueError, match="compute_s_params=True"):
        _ = r.freqs_hz


def test_single_port_s11_valid_but_no_s21():
    sp = np.full((1, 1, N_FREQS), 0.3 + 0.4j)
    r = Result(state=None, time_series=np.zeros((10, 1)),
               s_params=sp, freqs=FREQS)
    np.testing.assert_array_equal(r.s11(), sp[0, 0, :])
    with pytest.raises(ValueError, match="1 port"):
        r.s21()


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_s_params_returns_figure():
    import matplotlib
    r = _two_port()
    fig = r.plot_s_params()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_smith_returns_axes():
    import matplotlib
    r = _two_port()
    ax = r.plot_smith()
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_smith_selects_port():
    import matplotlib
    r = _two_port()
    ax = r.plot_smith(ports=(2, 1))
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(ax.figure)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_time_series_returns_figure():
    import matplotlib
    r = _two_port()
    fig = r.plot_time_series()
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_plot_s_params_no_sparams_raises():
    r = _no_sparams()
    with pytest.raises(ValueError, match="compute_s_params=True"):
        r.plot_s_params()


def _two_port_no_freqs():
    sp = np.zeros((2, 2, N_FREQS), dtype=complex)
    return Result(state=None, time_series=np.zeros((10, 2)),
                  s_params=sp, freqs=None, dt=1e-12)


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_no_freqs_raises():
    """S-params present but freqs absent must fail loud in both plotters."""
    r = _two_port_no_freqs()
    with pytest.raises(ValueError, match="compute_s_params=True"):
        r.plot_s_params()
    with pytest.raises(ValueError, match="compute_s_params=True"):
        r.plot_smith()


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_time_series_no_dt_raises():
    r = Result(state=None, time_series=np.zeros((10, 2)),
               s_params=None, freqs=None, dt=None)
    with pytest.raises(ValueError, match="store_dt=True"):
        r.plot_time_series()


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_smith_out_of_range_ports_raises():
    r = _two_port()
    with pytest.raises(ValueError, match="2 port"):
        r.plot_smith(ports=(3, 1))


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_smith_bad_arity_raises():
    r = _two_port()
    with pytest.raises(ValueError, match="two 1-indexed ports"):
        r.plot_smith(ports=(1,))


# ---------------------------------------------------------------------------
# Waveform-aware auto Harminv window (post-#392 review).
#
# ``find_resonances``'s auto ``source_decay_time`` hardcoded the cutoff=3
# GaussianPulse envelope (2.0*3.0*tau). The refactor to
# ``2.0*getattr(waveform, 't0', 3.0*tau)`` must be BITWISE-identical on
# the default path (doubling is exact in binary floating point) and scale
# with the waveform's own onset when one is provided.
# ---------------------------------------------------------------------------

def test_auto_harminv_window_default_bitwise_matches_historical():
    from rfx.api._spec import _auto_source_decay_time
    for f1, f2 in [(1e9, 5e9), (0.6e9, 12.3e9), (2.2e9, 2.9e9),
                   (8e9, 12e9)]:
        f_center = (f1 + f2) / 2
        tau = 1.0 / (f_center * 0.8 * np.pi)
        # main's exact formula, op-for-op:
        historical = 2.0 * 3.0 * tau
        assert _auto_source_decay_time((f1, f2)) == historical, (
            f"default auto window drifted from main at fr=({f1}, {f2})")


def test_two_t0_equals_six_tau_exactly_at_cutoff3():
    """The load-bearing float identity behind the refactor: for the
    default cutoff=3 waveform, 2*t0 == 2*(3*tau) == 6*tau EXACTLY
    (multiplying by 2.0 is an exact exponent shift, so rounding
    commutes: fl(6*tau) == 2*fl(3*tau))."""
    from rfx import GaussianPulse
    for f0 in (1e9, 2.2e9, 3.7e9, 18e9):
        for bwv in (0.5, 0.8, 1.2):
            wf = GaussianPulse(f0=f0, bandwidth=bwv)
            assert 2.0 * wf.t0 == 6.0 * wf.tau
            assert 2.0 * wf.t0 == 2.0 * 3.0 * wf.tau


def test_auto_harminv_window_scales_with_waveform_cutoff():
    """cutoff=4.5 waveform: the window start is 2*t0 = 9*tau of the
    WAVEFORM's own tau — no longer the hardcoded cutoff=3 envelope."""
    from rfx import GaussianPulse
    from rfx.api._spec import _auto_source_decay_time
    wf = GaussianPulse(f0=3e9, bandwidth=0.8, cutoff=4.5)
    got = _auto_source_decay_time((1e9, 5e9), wf)
    assert got == 2.0 * wf.t0
    assert got == 9.0 * wf.tau  # 2*(4.5*tau); doubling is exact
    assert got > _auto_source_decay_time((1e9, 5e9))


def test_find_resonances_threads_waveform_to_auto_window(monkeypatch):
    """The public method passes its ``waveform`` kwarg to the auto-window
    helper (None on the default path)."""
    import rfx.api._spec as spec_mod
    from rfx import GaussianPulse

    seen = {}
    real = spec_mod._auto_source_decay_time

    def spy(fr, waveform=None):
        seen["wf"] = waveform
        return real(fr, waveform)

    monkeypatch.setattr(spec_mod, "_auto_source_decay_time", spy)
    dt = 1e-12
    t = np.arange(4000) * dt
    ts = np.exp(-t / 2e-9) * np.sin(2 * np.pi * 3e9 * t)
    r = Result(state=None, time_series=ts[:, None], s_params=None,
               freqs=None, dt=dt, freq_range=(1e9, 5e9))

    wf = GaussianPulse(f0=3e9, bandwidth=0.8, cutoff=4.5)
    r.find_resonances(waveform=wf)
    assert seen["wf"] is wf

    r.find_resonances()
    assert seen["wf"] is None
