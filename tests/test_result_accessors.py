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
