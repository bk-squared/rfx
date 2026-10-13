"""A microstrip port's S is referred to its declared feed plane.

The port launches at the feed plane the user declares and measures V/I at
probe 0, ``n_probe_offset`` cells further in. The calculator solves S at
probe 0 from the recorded V/I and shifts it back to the feed plane:
``S_feed[i, j] = D_i S_probe0[i, j] D_j``, ``D_p = exp(-j beta_p L_p)``,
beta_p the closed form (Hammerstad-Jensen eps_eff(0) with Kirschning-Jansen
dispersion) on every bin (one resolver, ``rfx.measurement.msl_planes``, for both
the MSL and the mixed calculator; rfx-archive MSL-PLANE.md, decision records
5, 7 and 12).

The identity below is checked with every factor formed in THIS file:
S_probe0 by a NumPy multi-drive solve of the raw probe-0 V/I the calculator
records (its ``raw_3probe_dump_path`` dump), beta by the two formulas
written out here, and L from the grid's node coordinates. None of them calls
the resolver. How close that closed-form beta is to the simulated line's own
beta is a separate, slow check: tests/locks/test_msl_two_line_beta_contract_board.py.

Base model: the MSL thru of
``tests/unit/sparams/test_settling_witness.py::_msl_thru`` (12 mm board,
eps_r 2.2, 0.8 mm substrate, 1.2 mm trace, dx 0.2 mm, ports at 2 and 10 mm),
rebuilt with an explicit ladder. ``n_probe_offset = 12`` puts the two probe-0
planes 3.2 mm apart (at the auto offset 20 they coincide at x = 6 mm and S
is trivially the identity map); the record is short on purpose: the identity
does not depend on settling.
"""
from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import numpy as np

from rfx import Box, Simulation

_EPS_R = 2.2
_H = 0.0008
_W = 0.0012
_DX = 2e-4
_OFFSET = 12
_FREQS = jnp.linspace(2e9, 18e9, 4)
_PERIODS = 3.0


def _thru(n_probe_offset=_OFFSET):
    y_c = 0.004
    sim = Simulation(snap="declared", freq_max=20e9, domain=(0.012, 0.008, 0.0032),
                     dx=_DX, boundary="cpml", cpml_layers=8)
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


def _closed_form_beta(freqs, w, h, er):
    """beta = (2 pi f / c) sqrt(eps_eff(f)), written out here on purpose.

    eps_eff(0): rfx's simplified Hammerstad-Jensen form
    (eps_r + 1)/2 + (eps_r - 1)/2 (1 + 12 h/w)^-0.5.
    Dispersion: Kirschning & Jansen, Electron. Lett. 18(6):272-274, 1982
    (Wadell, Transmission Line Design Handbook, sec. 3.5.1), fn in GHz*mm.
    """
    f = np.asarray(freqs, dtype=np.float64)
    u = w / h
    e0 = (er + 1) / 2 + (er - 1) / 2 * (1 + 12 / u) ** -0.5
    fn = f / 1e9 * h * 1e3
    p1 = 0.27488 + (0.6315 + 0.525 / (1 + 0.0157 * fn) ** 20) * u \
        - 0.065683 * np.exp(-8.7513 * u)
    p2 = 0.33622 * (1 - np.exp(-0.03442 * er))
    p3 = 0.0363 * np.exp(-4.6 * u) * (1 - np.exp(-(fn / 38.7) ** 4.97))
    p4 = 1 + 2.751 * (1 - np.exp(-(er / 15.916) ** 8))
    pf = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    eps_f = er - (er - e0) / (1 + pf)
    return 2 * np.pi * f * np.sqrt(eps_f) / 299792458.0


def _feed_to_probe0(sim, x_feed, sign, n_cells):
    """Distance from the feed node to the probe-0 node, from the grid's nodes."""
    grid = sim._build_realized_grid()
    i_feed = int(grid.position_to_index((x_feed, 0.004, 0.0))[0])
    pad = int(grid.pad_x_lo)
    x_node = lambda i: (i - pad) * float(grid.dx)   # uniform lattice nodes
    return sign * (x_node(i_feed + sign * n_cells) - x_node(i_feed))


def _probe0_from_raw(dump_path):
    """S at probe 0: power waves of the recorded probe-0 V/I, S = B A^-1."""
    with np.load(dump_path, allow_pickle=True) as raw:
        meta = json.loads(str(raw["metadata_json"]))
        v = np.asarray(raw["raw_v"][:, :, 0, :], dtype=np.complex128)
        i = np.asarray(raw["raw_i1"], dtype=np.complex128)
        production = np.asarray(raw["production_smatrix"])
    r = np.asarray(meta["s_reference_impedances_ohm"], dtype=np.float64)
    a = (v + r[None, :, None] * i) / (2 * np.sqrt(r)[None, :, None])
    b = (v - r[None, :, None] * i) / (2 * np.sqrt(r)[None, :, None])
    # a, b are [drive, port, f]; b_port = S a_port for every drive.
    s = np.stack([np.linalg.solve(a[:, :, k], b[:, :, k]).T
                  for k in range(v.shape[-1])], axis=-1)
    return s, meta, production


def _solve(sim, dump):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sim.compute_msl_s_matrix(freqs=_FREQS, num_periods=_PERIODS,
                                        raw_3probe_dump_path=str(dump))


def test_feed_plane_s_is_d_times_probe0_s_times_d(tmp_path):
    """S_feed = D S_probe0 D, every factor formed here, not by the resolver."""
    sim = _thru()
    res = _solve(sim, tmp_path / "raw.npz")
    assert res.assembly == "multi_drive_solve"
    s0, meta, production = _probe0_from_raw(tmp_path / "raw.npz")
    beta = _closed_form_beta(np.asarray(res.freqs), _W, _H, _EPS_R)
    lengths = np.asarray([_feed_to_probe0(sim, 0.002, +1, _OFFSET),
                          _feed_to_probe0(sim, 0.010, -1, _OFFSET)])
    np.testing.assert_allclose(lengths, _OFFSET * _DX, rtol=1e-12)
    d = np.exp(-1j * beta[None, :] * lengths[:, None])
    expected = d[:, None, :] * s0 * d[None, :, :]
    scale = float(np.max(np.abs(s0)))
    # The shift must be present and must not be the probe-0 matrix itself.
    assert np.max(np.abs(np.asarray(res.S) - s0)) > 1e-2 * scale
    np.testing.assert_allclose(np.asarray(res.S), expected, rtol=0,
                               atol=2e-5 * scale)
    # The returned fields carry the same numbers (rad/m and m).
    np.testing.assert_allclose(np.asarray(res.shift_beta, dtype=np.float64),
                               np.stack([beta, beta]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(res.reference_plane_offsets_m),
                               lengths, rtol=0, atol=1e-12)
    # The dump records the same feed-plane S as the result.
    assert meta["schema_version"] == 5
    np.testing.assert_array_equal(production, np.asarray(res.S))


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


def test_msl_and_mixed_calculators_share_l_and_shift():
    """Same MSL port, same board: the two calculators give the same L and
    the same shift beta, and the mixed result's S_wave carries the shift too.

    ``compute_msl_s_matrix`` refuses a model carrying ``add_port``, so the
    MSL-only twin is the same board without the wire feed.
    """
    freqs = np.linspace(1e9, 4e9, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mixed, diag = _mixed_board(True).compute_mixed_s_matrix(
            freqs=freqs, num_periods=2.0, skip_preflight=True,
            return_diagnostics=True)
        msl = _mixed_board(False).compute_msl_s_matrix(
            freqs=freqs, num_periods=2.0)
    assert mixed.port_families == ("wire", "msl")
    assert np.asarray(mixed.reference_plane_offsets_m)[0] == 0.0
    assert not np.any(np.asarray(mixed.shift_beta)[0])
    assert (np.asarray(mixed.reference_plane_offsets_m)[1]
            == np.asarray(msl.reference_plane_offsets_m)[0])
    assert np.asarray(msl.reference_plane_offsets_m)[0] > 0.0
    np.testing.assert_array_equal(np.asarray(mixed.shift_beta)[1],
                                  np.asarray(msl.shift_beta)[0])
    # The MSL entry itself, against a probe-0 value formed here: the lane's
    # MSL driven diagonal is the V*I split b/a = (V - Z I)/(V + Z I) of its
    # own probe-0 record (run 1 drives the MSL port), times D^2 with D from
    # this file's Hammerstad-Jensen + Kirschning-Jansen and the realized
    # offset (10 cells of the 84.7 um lattice). A mixed lane that forgets the
    # shift fails here even if S_wave is shifted.
    from tests.unit.sparams.test_mixed_port_sparam import _EPS_R as er_m
    from tests.unit.sparams.test_mixed_port_sparam import _H_SUB, _W_TRACE, _DX as dx_m
    s = np.asarray(mixed.S if mixed.S_raw is None else mixed.S_raw)
    v0 = np.asarray(diag["v0_msl"])[1, 0]
    i0 = np.asarray(diag["i_msl"])[1, 0]
    z = float(np.asarray(diag["z0_hj_msl"])[0])
    gamma0 = (v0 - z * i0) / (v0 + z * i0)
    beta = _closed_form_beta(freqs, _W_TRACE, _H_SUB, er_m)
    length = 10 * dx_m
    np.testing.assert_allclose(np.asarray(mixed.reference_plane_offsets_m)[1],
                               length, rtol=1e-6)
    expected = np.exp(-2j * beta * length) * gamma0
    np.testing.assert_allclose(s[1, 1], expected, rtol=0,
                               atol=2e-5 * float(np.max(np.abs(gamma0))))
    # Before the (default) passivity projection, S and S_wave differ only in
    # the flux-overridden off-diagonal magnitudes, so their phases agree:
    # both are referred to the same plane. Unshifted S_wave would differ by
    # beta*L = 0.03-0.12 rad here.
    s = np.asarray(mixed.S if mixed.S_raw is None else mixed.S_raw)
    sw = np.asarray(mixed.S_wave)
    # #1659 refuses (NaN) the entries formed from the per-cell lumped/wire
    # wave; the phase comparison reads the entries both results still carry.
    read = np.isfinite(s) & np.isfinite(sw) & (np.abs(sw) > 0)
    assert read[1, 1].all(), "the MSL diagonal must be carried by both S and S_wave"
    np.testing.assert_allclose(np.angle(s[read] / sw[read]), 0.0, atol=1e-5)


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
