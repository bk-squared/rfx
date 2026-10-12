"""A microstrip port's S is referred to its declared feed plane.

The port launches at the feed plane the user declares and measures V/I at
probe 0, ``n_probe_offset`` cells further in. S is formed at probe 0 and
shifted back by ``D_p = exp(-j beta_p L_p)`` per port (one resolver,
``rfx.sparams._msl_planes``, for both the MSL and the mixed calculator;
measurement plan MSL-PLANE.md, decision record 5).

(a) Moving probe 0 must not move S at the feed plane: the same settled thru
    solved with two ``n_probe_offset`` values agrees there within the
    declared bar |d arg| <= 2 * 0.01 * beta * |dL| (the shift's own 1 %
    beta error), while S at probe 0 differs by about 2 * beta * dL.
(b) On a bin where the beta fit rails, the shift uses the Hammerstad-Jensen
    beta, formed here from the closed form and the realized node
    coordinates, not from the resolver; the fitted beta then cannot move S.
(c) The MSL and mixed calculators give the same L and the same shift for
    the same port.

Base model (a), (b): the settled MSL thru of
``tests/unit/sparams/test_settling_witness.py::_msl_thru`` (12 mm board,
eps_r 2.2, 0.8 mm substrate, 1.2 mm trace, dx 0.2 mm, ports at 2 and 10 mm,
12 bins over 2-18 GHz, 40 periods: settled to about -123 dB with no railed
bin), rebuilt here with an explicit ``n_probe_offset``.
"""
from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation

_FREQS = jnp.linspace(2e9, 18e9, 12)
_PERIODS = 40.0
_EPS_R = 2.2
_H = 0.0008
_W = 0.0012
_OFFSETS = (20, 12)   # cells; 20 is what the auto rule picks on this board


def _thru(n_probe_offset):
    y_c = 0.004
    sim = Simulation(snap="declared", freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (0.012, 0.008, _H)), material="sub")
    sim.add(Box((0, 0, 0), (0.012, 0.008, 0)), material="pec")
    sim.add(Box((0.002, y_c - _W / 2, _H), (0.010, y_c + _W / 2, _H)),
            material="pec")
    for name, x, d in (("p1", 0.002, "+x"), ("p2", 0.010, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=_W, height=_H,
                         direction=d, impedance=50.0, eps_r_sub=_EPS_R,
                         name=name, n_probe_offset=n_probe_offset,
                         n_probe_spacing=3, n_probes=5)
    return sim


def _solve(n_probe_offset):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _thru(n_probe_offset).compute_msl_s_matrix(
            freqs=_FREQS, num_periods=_PERIODS)


def _probe0(res):
    """Undo the shift with the returned fields: S at probe 0."""
    d = np.exp(-1j * np.asarray(res.shift_beta, dtype=np.float64)
               * np.asarray(res.reference_plane_offsets_m)[:, None])
    return np.asarray(res.S) / (d[:, None, :] * d[None, :, :])


@pytest.fixture(scope="module")
def two_offsets():
    return tuple(_solve(o) for o in _OFFSETS)


def _table(ra, rb):
    """Per-bin record of (a): every quantity the verdict reads."""
    la = np.asarray(ra.reference_plane_offsets_m)
    lb = np.asarray(rb.reference_plane_offsets_m)
    dl = la - lb                                   # per port, realized
    beta = 0.5 * (np.asarray(ra.shift_beta) + np.asarray(rb.shift_beta))
    sa, sb = np.asarray(ra.S), np.asarray(rb.S)
    pa, pb = _probe0(ra), _probe0(rb)
    rows = {}
    for (i, j) in ((1, 0), (0, 1), (0, 0), (1, 1)):
        shift = beta[i] * abs(dl[i]) + beta[j] * abs(dl[j])   # 2 beta dL
        rows[(i, j)] = dict(
            d_feed=np.angle(sa[i, j] / sb[i, j]),
            d_probe=np.angle(pa[i, j] / pb[i, j]),
            shift=shift, bar=0.01 * shift,
            mag_a=np.abs(sa[i, j]), mag_b=np.abs(sb[i, j]))
    return dl, rows


def _non_railed(ra, rb):
    return ~(np.asarray(ra.shift_beta_closed_form).any(axis=0)
             | np.asarray(rb.shift_beta_closed_form).any(axis=0))


def _write_table(res, rows):
    import sys
    f = np.asarray(res.freqs)
    for (i, j), r in rows.items():
        sys.stderr.write(f"S{i + 1}{j + 1}: f_GHz d_arg_feed d_arg_probe0 "
                         "2*beta*dL bar |S|a |S|b\n")
        for k in range(f.size):
            sys.stderr.write(
                f"  {f[k] / 1e9:7.3f} {r['d_feed'][k]: .3e} "
                f"{r['d_probe'][k]: .4f} {r['shift'][k]:.4f} "
                f"{r['bar'][k]:.2e} {r['mag_a'][k]:.3e} {r['mag_b'][k]:.3e}\n")


def test_a_records_are_settled_and_realized(two_offsets):
    ra, rb = two_offsets
    for r in (ra, rb):
        assert np.all(np.asarray(r.settling_db) < -60.0), r.settling_db
    assert _non_railed(ra, rb).sum() >= 6, "too few non-railed bins to judge"
    dl, _ = _table(ra, rb)
    np.testing.assert_allclose(dl, (_OFFSETS[0] - _OFFSETS[1]) * 2e-4,
                               rtol=1e-6)


def test_a_feed_plane_s21_does_not_move_with_probe_0(two_offsets):
    """The declared bar: |d arg S21| <= 2 * 0.01 * beta * |dL| per bin.

    S11 of this thru is ~1e-9 at both offsets, so its phase is not read.
    """
    ra, rb = two_offsets
    ok = _non_railed(ra, rb)
    _, rows = _table(ra, rb)
    _write_table(ra, rows)
    for key in ((1, 0), (0, 1)):
        r = rows[key]
        assert np.all(np.abs(r["d_feed"][ok]) <= r["bar"][ok]), (
            f"S{key[0] + 1}{key[1] + 1} at the feed plane moved with probe "
            f"0: d arg {r['d_feed'][ok]} rad, bar {r['bar'][ok]} rad")


def test_a_probe_0_s21_moves_by_about_two_beta_dl(two_offsets):
    """Probe 0 deeper in by dL at both ports -> the line between the two
    probe-0 planes is 2 dL shorter -> probe-0 S21 phase higher by ~2 beta dL.

    Pins the sign the code's DFT convention gives: phase falls with line
    length, so the shift back to the feed plane multiplies by
    exp(-j beta L). With the sign flipped the feed-plane S21 would move by
    ~4 beta dL instead of ~0.
    """
    ra, rb = two_offsets
    ok = _non_railed(ra, rb)
    _, rows = _table(ra, rb)
    for key in ((1, 0), (0, 1)):
        r = rows[key]
        miss = np.abs(np.angle(np.exp(1j * (r["d_probe"] - r["shift"]))))
        assert np.all(miss[ok] <= 0.1 * r["shift"][ok]), (
            f"S{key[0] + 1}{key[1] + 1} at probe 0: d arg {r['d_probe'][ok]} "
            f"rad, expected about +{r['shift'][ok]} rad")


def _hj_beta(freqs, width, height, eps_r):
    from rfx.core.yee import EPS_0, MU_0
    from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff
    _, eps_eff = hammerstad_jensen_z0_eps_eff(width, height, eps_r)
    c0 = 1.0 / float(np.sqrt(MU_0 * EPS_0))
    return 2.0 * np.pi * np.asarray(freqs, dtype=np.float64) * np.sqrt(eps_eff) / c0


def _realized_offset(sim, x_feed, sign, n_cells):
    """Feed node to probe-0 node on the sim's own grid, read here."""
    from rfx.sources.msl_port import _msl_coord_for_index, _msl_position_to_index
    grid = sim._build_realized_grid()
    i = int(_msl_position_to_index(grid, (x_feed, 0.004, 0.0))[0])
    return sign * (_msl_coord_for_index(grid, "x", i + sign * n_cells)
                   - _msl_coord_for_index(grid, "x", i))


def _recording_shift(monkeypatch):
    """Record the S the calculator hands to the shift; pass it through."""
    import rfx.sparams._msl_planes as planes
    seen = []
    real = planes.apply_feed_plane_shift

    def recorder(S, offsets_m, shift_beta):
        seen.append(np.asarray(S))
        return real(S, offsets_m, shift_beta)

    monkeypatch.setattr(planes, "apply_feed_plane_shift", recorder)
    return seen


_SHORT = 8.0   # periods; (b) and the formula check read no settled value


def test_shift_is_applied_with_the_returned_fields(monkeypatch):
    """S returned = D_i S_probe0 D_j, D formed here from the returned fields.

    Supplementary (not one of the declared (a)-(c)): it names a red test for
    a flipped sign of D and for a missing shift without any phase bar.
    """
    seen = _recording_shift(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _thru(20).compute_msl_s_matrix(freqs=_FREQS, num_periods=_SHORT)
    assert len(seen) == 1
    s_in = seen[0]
    d = np.exp(-1j * np.asarray(res.shift_beta, dtype=np.float64)
               * np.asarray(res.reference_plane_offsets_m)[:, None])
    expected = d[:, None, :] * s_in * d[None, :, :]
    s_out = np.asarray(res.S)
    assert not np.allclose(s_out, s_in, rtol=1e-3, atol=0), "no shift applied"
    np.testing.assert_allclose(s_out, expected, rtol=2e-5,
                               atol=2e-6 * np.max(np.abs(s_in)))


def test_b_railed_bins_use_the_closed_form_beta(monkeypatch):
    """Every bin forced railed: shift beta = Hammerstad-Jensen, L realized.

    The expected D is formed here from the closed form and the grid's own
    node coordinates; then a fitted beta 20 % off must not move S at all.
    """
    from rfx.probes import msl_wave_decomp as wd
    real = wd._estimate_beta

    def railed(*args, **kwargs):
        beta, _ = real(*args, **kwargs)
        return beta, jnp.asarray(True)

    def railed_and_wrong(*args, **kwargs):
        beta, _ = real(*args, **kwargs)
        return 1.2 * beta, jnp.asarray(True)

    def solve(estimator):
        monkeypatch.setattr(wd, "_estimate_beta", estimator)
        seen = _recording_shift(monkeypatch)
        sim = _thru(20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_msl_s_matrix(freqs=_FREQS, num_periods=_SHORT)
        return sim, res, seen[-1]

    sim, res, s_in = solve(railed)
    assert np.all(np.asarray(res.shift_beta_closed_form))
    assert np.all(np.asarray(res.beta_railed))
    beta_hj = _hj_beta(np.asarray(res.freqs), _W, _H, _EPS_R)
    np.testing.assert_allclose(np.asarray(res.shift_beta, dtype=np.float64),
                               np.stack([beta_hj, beta_hj]), rtol=1e-6)
    lengths = np.asarray([_realized_offset(sim, 0.002, +1, 20),
                          _realized_offset(sim, 0.010, -1, 20)])
    np.testing.assert_allclose(np.asarray(res.reference_plane_offsets_m),
                               lengths, rtol=0, atol=1e-12)
    d = np.exp(-1j * beta_hj[None, :] * lengths[:, None])
    np.testing.assert_allclose(np.asarray(res.S),
                               d[:, None, :] * s_in * d[None, :, :],
                               rtol=2e-5, atol=2e-6 * np.max(np.abs(s_in)))

    _, res_wrong, _ = solve(railed_and_wrong)
    assert not np.allclose(np.asarray(res_wrong.beta), np.asarray(res.beta),
                           rtol=1e-3), "the patched estimator did not reach beta"
    np.testing.assert_array_equal(np.asarray(res_wrong.S), np.asarray(res.S))


def _mixed_board(with_feed):
    from tests.unit.sparams.test_mixed_port_sparam import (
        _add_feed, _add_msl, _base_sim)
    # snap="declared" on both twins: compute_msl_s_matrix has no
    # skip_preflight, and run()'s preflight refuses this board's 600 um
    # sheet on the 84.7 um lattice otherwise (-5.46 % solved width).
    sim, y_c = _base_sim(snap="declared")
    if with_feed:
        _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)
    return sim


def test_c_msl_and_mixed_calculators_share_l_and_shift(monkeypatch):
    """Same MSL port, same board: the two calculators give the same L and,
    on bins where both use the closed form, the same shift beta.

    ``compute_msl_s_matrix`` refuses a model carrying ``add_port``, so the
    MSL-only twin is the same board without the wire feed. On fitted bins
    the two shift betas come from different field solutions (the feed
    changes the field at the probes) and are not compared here.
    """
    from rfx.probes import msl_wave_decomp as wd
    real = wd._estimate_beta

    def railed(*args, **kwargs):
        beta, _ = real(*args, **kwargs)
        return beta, jnp.asarray(True)

    monkeypatch.setattr(wd, "_estimate_beta", railed)
    freqs = np.linspace(1e9, 4e9, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed = _mixed_board(True).compute_mixed_s_matrix(
            freqs=freqs, num_periods=4.0, skip_preflight=True)
        msl = _mixed_board(False).compute_msl_s_matrix(
            freqs=freqs, num_periods=4.0)
    assert mixed.port_families == ("wire", "msl")
    assert np.asarray(mixed.reference_plane_offsets_m)[0] == 0.0
    assert not np.any(np.asarray(mixed.shift_beta)[0])
    assert (np.asarray(mixed.reference_plane_offsets_m)[1]
            == np.asarray(msl.reference_plane_offsets_m)[0])
    assert np.asarray(msl.reference_plane_offsets_m)[0] > 0.0
    np.testing.assert_array_equal(np.asarray(mixed.shift_beta)[1],
                                  np.asarray(msl.shift_beta)[0])
    assert np.all(np.asarray(mixed.shift_beta_closed_form)[1])


def test_graded_feed_offset_is_the_realized_cell_sum():
    """On a graded propagation axis L is the sum of the realized cells,
    not ``n_probe_offset * dx``.

    The board of tests/contracts/test_grouped_drives.py::port_scene('msl',
    'graded'): 1 mm cells with four 1.1 mm cells at indices 6-9, the port
    feed at x = 6 mm, ``n_probe_offset = 3`` -> L = 3.3 mm (3 mm declared).
    """
    from tests.contracts.test_grouped_drives import port_scene
    sim = port_scene("msl", "graded")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_msl_s_matrix(n_steps=32,
                                       freqs=jnp.array([1.8e9, 2e9, 2.2e9]))
    profile = np.r_[np.full(6, .001), np.full(4, .0011), np.full(14, .001)]
    expected = float(np.sum(profile[6:9]))
    np.testing.assert_allclose(np.asarray(res.reference_plane_offsets_m),
                               [expected], rtol=1e-6)   # grid stores float32
