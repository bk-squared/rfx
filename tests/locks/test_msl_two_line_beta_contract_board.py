"""The closed-form microstrip beta against the simulated line's two-line beta.

A microstrip port's S is shifted from probe 0 to the declared feed plane by
``exp(-j beta L)`` with the closed-form beta (Hammerstad-Jensen eps_eff(0)
with Kirschning-Jansen dispersion, ``rfx.measurement.msl_planes``). The shift is
only as right as that beta is for the line the solver actually builds. This
lock measures the simulated line's beta by the two-line method -- two thrus
identical except for length, probes at the auto offset on both -- and pins
the closed form to it on one coarse board.

Board: 1.2 mm strip on 0.8 mm of eps_r 2.2 (4 substrate cells), dx 0.2 mm,
ports at the strip ends 8.0 mm apart, plus a twin 36 cells (7.2 mm) longer
(M2's ``choose_delta`` rule, 20-160 deg, applied to 2-12 GHz: no single
length meets it over 2-18 GHz); ``n_probe_offset`` 20 (the auto value),
spacing 3, 5 probes; 2-18 GHz, 12 bins, 40 periods (settled to about
-123 dB). beta_2L = -unwrap(arg S21_long / S21_short) / dL with S21 at probe 0
(the stored shift undone with the result's own fields). Undoing the shift is
not circular: both lines use the same offsets, so the same D_1 D_2 multiplies
S21 on both and cancels exactly in S21_long / S21_short -- the closed-form
beta under test never enters beta_2L.

Measured on branch ports/msl-plane (Mac CPU, 2026-10-13): beta_KJ/beta_2L - 1
= -1.226 % ... -0.220 % (outside 1 % at 2.0, 3.5 and 4.9 GHz); doubling the
absorber moved beta_2L by at most 0.033 %. Record: bk-squared/rfx-archive
``rfx/plans/20261005-s2-measurement-plan/MSL-PLANE.md``, decision records 11
and 12 (the bar 1.35e-2 is the measured 1.226 % plus about 10 % margin, set by the ports editor: decision record 16).
The shift's phase error on this board is at most 1.9e-2 rad over 2-18 GHz
(decision record 15). Moving
probe 0 on ONE line is not this measurement: within 5 substrate heights of
the feed the field is not one propagating mode (decision record 11, C1).
"""
from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "this module's two-line measurement (rfx-archive rfx/records/20261012-msl-plane-measure/m4; bar from MSL-PLANE.md decision record 16)",
    "commit": "a1cad87",
    "date": "2026-10-13",
    "run_id": "local",
    "host": "macOS jax cpu",
    "pinned_until": "2027-04-13",
}

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation

_EPS_R, _H, _W, _DX = 2.2, 0.0008, 0.0012, 2e-4
_DELTA_CELLS = 36
_OFFSET = 20
_FREQS = jnp.linspace(2e9, 18e9, 12)
_BAR = 1.35e-2


def _board(extra_cells):
    ext = extra_cells * _DX
    lx, y_c = 0.012 + ext, 0.004
    sim = Simulation(snap="declared", freq_max=20e9, domain=(lx, 0.008, 0.0032),
                     dx=_DX, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0, 0, 0), (lx, 0.008, _H)), material="sub")
    sim.add(Box((0, 0, 0), (lx, 0.008, 0)), material="pec")
    sim.add(Box((0.002, y_c - _W / 2, _H), (0.010 + ext, y_c + _W / 2, _H)),
            material="pec")
    for name, x, d in (("p1", 0.002, "+x"), ("p2", 0.010 + ext, "-x")):
        sim.add_msl_port(position=(x, y_c, 0.0), width=_W, height=_H,
                         direction=d, impedance=50.0, eps_r_sub=_EPS_R,
                         name=name, n_probe_offset=_OFFSET,
                         n_probe_spacing=3, n_probes=5)
    return sim


def _feed_nodes(sim):
    grid = sim._build_realized_grid()
    pad = int(grid.pad_x_lo)
    return [(int(grid.position_to_index((pe.position[0], 0.004, 0.0))[0]) - pad)
            * float(grid.dx) for pe in sim._msl_ports]


def _probe0_s21(res):
    beta = np.asarray(res.shift_beta, dtype=np.float64)
    offsets = np.asarray(res.reference_plane_offsets_m, dtype=np.float64)
    u = np.exp(1j * beta * offsets[:, None])
    return np.asarray(res.S)[1, 0] * u[1] * u[0]


@pytest.mark.slow
def test_closed_form_beta_matches_the_two_line_beta_on_the_coarse_board():
    from rfx.measurement.msl_planes import msl_closed_form_beta

    short, long_ = _board(0), _board(_DELTA_CELLS)
    fs, fl = _feed_nodes(short), _feed_nodes(long_)
    # Realized length difference, and the same port-to-strip-end geometry:
    # the strip runs feed node to feed node on both, 2 mm from each x face.
    d_l = (fl[1] - fl[0]) - (fs[1] - fs[0])
    np.testing.assert_allclose(d_l, _DELTA_CELLS * _DX, rtol=1e-9)
    np.testing.assert_allclose([fs[0], fl[0]], [0.002, 0.002], rtol=1e-9)
    np.testing.assert_allclose([0.012 - fs[1], 0.012 + _DELTA_CELLS * _DX - fl[1]],
                               [0.002, 0.002], rtol=1e-9)
    results = []
    for sim in (short, long_):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_msl_s_matrix(freqs=_FREQS, num_periods=40.0)
        assert np.all(np.asarray(res.settling_db) < -60.0), res.settling_db
        np.testing.assert_allclose(res.reference_plane_offsets_m,
                                   [_OFFSET * _DX] * 2, rtol=1e-9)
        results.append(res)
    ratio = _probe0_s21(results[1]) / _probe0_s21(results[0])
    beta_2l = -np.unwrap(np.angle(ratio)) / d_l
    freqs = np.asarray(results[0].freqs)
    beta_kj = msl_closed_form_beta(_W, _H, _EPS_R, freqs)
    err = beta_kj / beta_2l - 1.0
    assert np.all(np.abs(err) <= _BAR), (
        "closed-form beta vs two-line beta per bin (%): "
        + ", ".join(f"{f / 1e9:.2f} GHz {100 * e:+.3f}" for f, e in zip(freqs, err)))
