"""The closed-form pieces behind the design films (issue 1359), held without a solver.

``scripts/showcase/design_taper.py`` builds its Klopfenstein baseline in closed
form and picks the profile's free parameter with a TE10 section cascade;
``scripts/showcase/design_beam.py`` writes each cover cell onto the cells nested
in it on a mesh twice as fine for the re-solve; ``scripts/showcase/_design_common.py`` judges the gradient
against central differences and writes every iterate as it goes.  Each test
holds one of those against an independent statement of the same physics or
rule, so a slip in the code cannot be matched by the same slip in the test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "showcase"))
import _design_common as dc  # noqa: E402
import design_beam  # noqa: E402
import design_taper  # noqa: E402

C0 = design_taper.C0
FC = C0 / (2 * 22.86e-3)          # WR-90 TE10 cutoff


def _abcd_s11(eps_sec, widths, f, eps_load):
    """S11 of the same cascade from chained ABCD matrices, source side first
    (a different route from the script's load-side impedance recursion)."""
    k0 = 2 * np.pi * f / C0
    m = np.eye(2, dtype=complex)
    for er, w in zip(eps_sec, widths):
        z = 1 / np.sqrt(er - (FC / f) ** 2)
        bl = k0 / z * w
        m = m @ np.array([[np.cos(bl), 1j * z * np.sin(bl)], [1j * np.sin(bl) / z, np.cos(bl)]])
    zl = 1 / np.sqrt(eps_load - (FC / f) ** 2)
    zin = (m[0, 0] * zl + m[0, 1]) / (m[1, 0] * zl + m[1, 1])
    z0 = 1 / np.sqrt(1 - (FC / f) ** 2)
    return (zin - z0) / (zin + z0)


def test_the_section_cascade_is_the_transmission_line_answer():
    f = np.array([8.2e9, 10.3e9, 12.4e9])
    # no section: the bare vacuum -> eps_r 9 step
    z0, z9 = (1 / np.sqrt(e - (FC / f) ** 2) for e in (1.0, 9.0))
    np.testing.assert_allclose(design_taper.cascade_s11([], [], f, FC, 9.0), (z9 - z0) / (z9 + z0),
                               rtol=1e-12)
    # an asymmetric three-section stack against the ABCD chain
    eps, w = np.array([1.7, 6.5, 3.2]), np.array([2.54e-3, 3.175e-3, 2.54e-3])
    for fi in f:
        np.testing.assert_allclose(design_taper.cascade_s11(eps, w, [fi], FC, 9.0)[0],
                                   _abcd_s11(eps, w, fi, 9.0), rtol=1e-10, atol=1e-13)
    # a quarter-wave section of impedance sqrt(z0 z9) matches at its design frequency
    f0 = 10.3e9
    zq = np.sqrt(z0[1] * z9[1])
    er_q = 1 / zq ** 2 + (FC / f0) ** 2
    lam_g = C0 / f0 * zq                      # 2 pi / beta with beta = k0 / Z
    assert abs(design_taper.cascade_s11([er_q], [lam_g / 4], [f0], FC, 9.0)[0]) < 1e-12


def test_klopfenstein_profile_has_its_closed_form_end_steps():
    # phi(1, A) = (cosh A - 1) / A^2, so ln Z(1) = ln z2 - Gamma0 / cosh A and
    # ln Z(0) = ln z1 + Gamma0 / cosh A: the two end steps of height Gamma_m
    z1, z2 = 1.3, 0.34
    g0 = 0.5 * np.log(z2 / z1)
    for A in (0.5, 3.0, 12.0):
        lnz = design_taper.klopfenstein_lnz(np.array([0.0, 0.5, 1.0]), A, z1, z2)
        np.testing.assert_allclose(lnz[0], np.log(z1) + g0 / np.cosh(A), atol=1e-6)
        np.testing.assert_allclose(lnz[1], 0.5 * np.log(z1 * z2), atol=1e-12)
        np.testing.assert_allclose(lnz[2], np.log(z2) - g0 / np.cosh(A), atol=1e-6)
    # monotone between the ends (a taper, not a ripple)
    lnz = design_taper.klopfenstein_lnz(np.linspace(0, 1, 401), 6.0, z1, z2)
    assert np.all(np.diff(lnz) < 0)


def test_the_klopfenstein_sections_run_from_the_guide_to_the_load():
    edges = design_taper.section_edges_cells(1) * 0.635e-3
    centres, length = 0.5 * (edges[:-1] + edges[1:]) - edges[0], edges[-1] - edges[0]
    eps, prof = design_taper.klopfenstein_sections(8.0, centres, length, 10.3e9, FC, 9.0)
    assert prof["x_m"][0] == 0.0 and abs(prof["x_m"][-1] - length) < 1e-15
    assert np.all(np.diff(eps) > 0) and 1.0 < eps[0] < eps[-1] < 9.0
    # beta = k0 / Z: the eps_r the profile reaches at each end is the guide's
    # own impedance plus the end step, read back through Z_TE10
    z_end = 1 / np.sqrt(prof["eps_r"][[0, -1]] - (FC / 10.3e9) ** 2)
    z1, z9 = (1 / np.sqrt(e - (FC / 10.3e9) ** 2) for e in (1.0, 9.0))
    gm = 0.5 * abs(np.log(z9 / z1)) / np.cosh(8.0)
    np.testing.assert_allclose(np.log(z_end), [np.log(z1) - gm, np.log(z9) + gm], atol=1e-6)


def test_taper_sections_keep_their_length_on_the_fine_mesh():
    e1, e2 = design_taper.section_edges_cells(1), design_taper.section_edges_cells(2)
    assert e1[0] == 94 and e1[-1] == 220 and len(e1) == 31
    assert set(np.diff(e1)) == {4, 5}
    np.testing.assert_array_equal(e2 * 0.3175e-3, e1 * 0.635e-3)


def test_the_fine_cover_nests_each_coarse_cell_in_place():
    # a lambda/20 eps_r index i covers [x_i, x_i+1); on the lambda/40 mesh that is
    # fine indices 2i and 2i+1, starting at the same physical position
    rng = np.random.default_rng(1359)
    coarse = rng.uniform(1.0, 10.0, (31, 31, 3))
    fine = design_beam.nested_block(coarse, 2)
    assert fine.shape == (62, 62, 6)
    for (i, j, k) in ((0, 0, 0), (30, 30, 2), (7, 15, 1), (23, 4, 0)):
        np.testing.assert_array_equal(fine[2 * i:2 * i + 2, 2 * j:2 * j + 2, 2 * k:2 * k + 2],
                                      coarse[i, j, k])
    # the block starts at the same physical node: coarse index lo on a grid with
    # 10 pad cells of dx sits where fine index start sits with 20 pads of dx / 2
    lo, pads_c, pads_f = (31, 31, 48), (10, 10, 10), (20, 20, 20)
    start = design_beam.nested_start(lo, pads_c, pads_f, 2)
    dx = 4.9965
    for c, pc, f, pf in zip(lo, pads_c, start, pads_f):
        assert (c - pc) * dx == (f - pf) * dx / 2
        assert (c + 1 - pc) * dx == (f + 2 - pf) * dx / 2      # the cell's far face, too


def test_the_beam_start_is_the_modules_ramp():
    eps0, psi0 = design_beam.module_start((31, 31, 3))
    np.testing.assert_allclose(eps0[:, 5, 1], np.linspace(2.0, 9.0, 31), atol=1e-6)
    assert np.all(eps0 == eps0[:, :1, :1])                        # uniform in y and z
    np.testing.assert_allclose(1.0 + 9.0 / (1.0 + np.exp(-psi0)), eps0, atol=1e-5)


def _ladder(values):
    return {n: {str(h): {"fd": v} for h, v in zip((0.2, 0.1, 0.05), fds)}
            for n, fds in values.items()}


def test_the_fd_verdict_judges_the_large_and_flags_round_off():
    ladder = _ladder({"big": (1.000, 1.010, 1.0125),      # truncation shrinking: clean
                      "noisy": (0.500, 0.520, 0.600),     # moves more when the step shrinks
                      "small": (0.050, 0.060, 0.070)})    # below 0.1 x the largest: reported
    ad = {"big": 1.02, "noisy": 0.52, "small": 0.2}
    v = dc.judge_fd(ad, ladder, (0.2, 0.1, 0.05), 0.1, 0.05, 0.1)
    assert v["judged"] == ["big", "noisy"]
    assert v["rows"]["small"]["passed"] is None
    assert v["rows"]["big"]["passed"] and v["rows"]["noisy"]["passed"]
    assert v["roundoff"] == ["noisy"]
    assert v["all_judged_passed"]
    v = dc.judge_fd({**ad, "big": 1.08}, ladder, (0.2, 0.1, 0.05), 0.1, 0.05, 0.1)
    assert not v["rows"]["big"]["passed"] and not v["all_judged_passed"]


def test_every_iterate_is_on_disk_before_the_next_starts(tmp_path):
    store = dc.IterateStore(tmp_path / "iterations.npz", static={"freqs_hz": np.arange(3.0)})
    for k in range(4):
        store.append(eps=np.full(2, float(k)), J=float(k), grad=np.full(2, -float(k)))
        store.persist()
        on_disk = np.load(tmp_path / "iterations.npz")
        assert on_disk["J"].tolist() == list(range(k + 1))
    store.append(eps=np.full(2, 9.0), J=9.0)            # the final design: no gradient
    store.persist()
    on_disk = np.load(tmp_path / "iterations.npz")
    assert on_disk["eps"].shape == (5, 2) and np.all(np.isnan(on_disk["grad"][-1]))
    np.testing.assert_array_equal(on_disk["freqs_hz"], np.arange(3.0))
    with pytest.raises(KeyError):
        store.append(eps=np.zeros(2), J=0.0, new_key=1.0)
