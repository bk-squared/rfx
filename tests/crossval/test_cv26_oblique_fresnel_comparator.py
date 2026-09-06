"""cv26 oblique slab -- unit witnesses of the comparator, no FDTD.

Everything here is analytic or lattice-exact and is what
``docs/design_notes/20260902_cv26_oblique_fresnel_predeclaration.md`` quotes:
the TE/TM oracle and the eps <-> mu duality, the realized-angle convention,
the Meep k_point mapping (and the wrong-2 pi convention failing at 1e-9), the
TE/TM swap failing at 45 deg+ and passing at normal incidence at 1e-9, the
numerical-dispersion term, the exact 2-D Yee lattice (reducing to an
independent 1-D march at k_y = 0, and to Fresnel as dx -> 0), the CPML
profile transcription against rfx's own ``_cpml_profile``, the derived
bandwidths, records, the primary-recipe rule, and every falsifier's margin.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from tests._gate_policy import gate_from_envelope

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = _load("cv26_oblique_fresnel", "validation/crossval/comparators/oblique_fresnel.py")
F0 = O.TFSF_F0_HZ
FREQS = np.linspace(6e9, 14e9, 41)


# ---------------------------------------------------------------------------
# Oracle, duality, realized angle
# ---------------------------------------------------------------------------

def test_te_tm_duality_is_exact():
    """TM on (eps = 4, mu = 1) == TE on (eps = 1, mu = 4) at every angle and frequency (bit level)."""
    for th in (0.0, 30.0, 45.0, 60.0, 63.43, 75.0, 85.0):
        ky = O.ky_from(F0, th)
        f = FREQS[FREQS > 1.01 * O.cutoff_hz(ky)]             # propagating bins only
        r_tm, t_tm = O.slab_rt(f, ky, 4.0, O.D_SLAB_M, "tm")
        r_te, t_te = O.slab_rt(f, ky, 1.0, O.D_SLAB_M, "te", mu_slab=4.0)
        assert np.max(np.abs(r_tm - r_te)) == 0.0 and np.max(np.abs(t_tm - t_te)) == 0.0
        R, T = O.slab_RT(f, ky, 4.0, O.D_SLAB_M, "te")
        assert np.allclose(R + T, 1.0, atol=1e-12)          # lossless closure
    assert O.rfx_slab_materials("tm") == (1.0, 4.0) and O.rfx_slab_materials("te") == (4.0, 1.0)


def test_normal_incidence_reduces_to_cv04s_transfer_matrix():
    """At k_y = 0 the oblique oracle equals cv04's normal-incidence characteristic matrix."""
    n = 2.0
    delta = 2 * np.pi * FREQS * n * O.D_SLAB_M / O.C0
    M00, M01, M10, M11 = np.cos(delta), 1j * np.sin(delta) / n, 1j * n * np.sin(delta), np.cos(delta)
    r04 = (M00 + M01 - M10 - M11) / (M00 + M01 + M10 + M11)
    R, T = O.oracle_RT(FREQS, 0.0, "te")
    assert np.allclose(R, np.abs(r04) ** 2, atol=1e-12)
    assert np.allclose(T, 1 - np.abs(r04) ** 2, atol=1e-12)


def test_brewster_angle_zeroes_tm_and_not_te():
    thB = O.theta_brewster_rad(4.0)
    assert math.degrees(thB) == pytest.approx(63.4349, abs=1e-3)
    ky = O.ky_from(F0, math.degrees(thB))
    R_tm, _ = O.oracle_RT(np.array([F0]), ky, "tm")
    R_te, _ = O.oracle_RT(np.array([F0]), ky, "te")
    assert R_tm[0] < 1e-28 and R_te[0] > 0.4


def test_realized_angle_convention():
    ky = O.ky_from(F0, 60.0)
    assert O.cutoff_hz(ky) == pytest.approx(F0 * math.sin(math.radians(60.0)))
    th = O.realized_theta_rad(np.array([O.cutoff_hz(ky) * 0.999, F0, 2 * F0]), ky)
    assert np.isnan(th[0]) and math.degrees(th[1]) == pytest.approx(60.0, abs=1e-9)
    assert math.degrees(th[2]) == pytest.approx(math.degrees(math.asin(0.5 * math.sin(math.radians(60)))), abs=1e-9)


def test_te_tm_swap_fails_at_oblique_and_passes_at_normal_to_1e9():
    """F2 at unit level: the swapped oracle differs from the declared one by >> 1e-9 at 45 deg+, and by 0 at normal."""
    for th in (45.0, 60.0):
        ky = O.ky_from(F0, th)
        R_te, _ = O.oracle_RT(FREQS, ky, "te"); R_tm, _ = O.oracle_RT(FREQS, ky, "tm")
        assert np.nanmax(np.abs(R_te - R_tm)) > 0.1
    R_te, _ = O.oracle_RT(FREQS, 0.0, "te"); R_tm, _ = O.oracle_RT(FREQS, 0.0, "tm")
    assert np.nanmax(np.abs(R_te - R_tm)) <= 1e-9


# ---------------------------------------------------------------------------
# Meep k_point mapping (F4 at unit level)
# ---------------------------------------------------------------------------

def test_meep_k_point_round_trips_to_1e9_and_the_wrong_2pi_convention_fails():
    for th in (30.0, 45.0, 60.0):
        kp = O.meep_k_point(F0, th)
        assert kp[0] == 0.0 and kp[2] == 0.0
        ky = O.ky_from_meep_k_point(kp)
        assert abs(ky - O.ky_from(F0, th)) <= 1e-9 * O.ky_from(F0, th)
        assert abs(math.degrees(O.realized_theta_rad(np.array([F0]), ky)[0]) - th) <= 1e-9
        # by hand: Meep's k is in units of 2 pi / a -> k_y a / (2 pi) = (f0 a / c) sin theta
        assert kp[1] == pytest.approx(F0 * O.MEEP_A_M / O.C0 * math.sin(math.radians(th)), rel=1e-12)
        bad = O.meep_k_point_wrong_2pi(F0, th)
        ky_bad = O.ky_from_meep_k_point(bad)
        assert abs(ky_bad - O.ky_from(F0, th)) > 1e-9 * O.ky_from(F0, th)
        th_bad = O.realized_theta_rad(np.array([F0]), ky_bad)[0]
        assert np.isnan(th_bad) or abs(math.degrees(th_bad) - th) > 1e-9
    assert O.meep_k_point(F0, 45.0)[1] == pytest.approx(0.2358654336749684, rel=1e-12)


def test_meep_fwidth_matches_the_rfx_spectrum():
    """exp(-2 pi^2 (f-f0)^2 / fwidth^2) == exp(-((f-f0)/(bw f0))^2) with fwidth = sqrt(2) pi bw f0."""
    bw = O.ARM_BW["te_45"]
    fw = O.meep_fwidth_for(bw, F0)
    f = np.linspace(8e9, 12e9, 9)
    meep = np.exp(-2 * math.pi ** 2 * (f - F0) ** 2 / fw ** 2)
    assert np.allclose(meep, O.incident_amp_rel(f, F0, bw), rtol=1e-12)


# ---------------------------------------------------------------------------
# Lattice terms
# ---------------------------------------------------------------------------

def test_dispersion_term_is_second_order_and_vanishes_as_dx_to_zero():
    ky = O.ky_from(F0, 60.0)
    f = np.linspace(9.3e9, 10.7e9, 15)
    w1 = O.dispersion_term(f, ky, "te", O.DX_M, O.DT_S)
    w2 = O.dispersion_term(f, ky, "te", O.DX_M / 2, O.DT_S / 2)
    w4 = O.dispersion_term(f, ky, "te", O.DX_M / 4, O.DT_S / 4)
    assert 3.5 < np.mean(w1["W_R"]) / np.mean(w2["W_R"]) < 4.5
    assert 3.5 < np.mean(w2["W_R"]) / np.mean(w4["W_R"]) < 4.5
    assert 0.03 < np.max(np.abs(w1["phase_err_rad"])) < 0.06     # section 3: ~0.04-0.05 rad at 60 deg, dx = 1 mm


def _march_1d(f_hz, eps_r, d_m, dx, dt, n_vac=20):
    """An INDEPENDENT 1-D Yee lattice march at normal incidence (cv23 section 12.2's form)."""
    mu0 = O.MU_0
    w = 2 * np.pi * np.asarray(f_hz, float); wh = 2 * np.sin(w * dt / 2) / dt
    n_slab = int(round(d_m / dx)); N = 2 * n_vac + n_slab + 1
    eps = np.full(N, O.EPS_0); eps[n_vac:n_vac + n_slab] = O.EPS_0 * eps_r
    k = (2 / dx) * np.arcsin(wh * dx / (2 * O.C0))
    R = np.empty(w.size)
    for m in range(w.size):
        E = np.zeros(N, complex); H = np.zeros(N - 1, complex)
        E[-1] = 1.0; E[-2] = np.exp(1j * k[m] * dx); H[-1] = (E[-1] - E[-2]) / (dx * 1j * wh[m] * mu0)
        for i in range(N - 2, 0, -1):
            H[i - 1] = H[i] - dx * 1j * wh[m] * eps[i] * E[i]
            E[i - 1] = E[i] - dx * 1j * wh[m] * mu0 * H[i - 1]
        M = np.array([[1, 1], [np.exp(-1j * k[m] * dx), np.exp(1j * k[m] * dx)]])
        a, b = np.linalg.solve(M, E[:2]); R[m] = abs(b / a) ** 2
    return R


def test_full_lattice_with_ideal_absorber_equals_the_independent_1d_march_at_normal_incidence():
    cells = O.rig_cells(100, 20)
    f = np.linspace(7e9, 13e9, 13)
    lat = O.yee_lattice_full(f, 0.0, cells, eps_slab=4.0, ideal_absorber=True, aux="plane")
    assert np.max(np.abs(lat["R"] - _march_1d(f, 4.0, O.D_SLAB_M, O.DX_M, O.DT_S))) < 1e-12
    assert np.allclose(lat["R"] + lat["T"], 1.0, atol=1e-10)


def test_lattice_converges_to_fresnel_at_second_order_at_oblique_incidence():
    ky = O.ky_from(F0, 45.0)
    f = np.linspace(8.5e9, 11.5e9, 7)
    R_an, _ = O.oracle_RT(f, ky, "te")
    errs = []
    for K in (1, 2, 4):
        cells = O.rig_cells(100, 20, dx_div=K)
        lat = O.yee_lattice_full(f, ky, cells, eps_slab=4.0, dx=cells["dx"], dt=O.DT_S / K, ideal_absorber=True, aux="plane")
        errs.append(np.mean(np.abs(lat["R"] - R_an)))
    assert 3.3 < errs[0] / errs[1] < 4.7 and 3.3 < errs[1] / errs[2] < 4.7


def test_vacuum_lattice_has_no_scattered_field_and_pec_gives_unit_reflection_with_an_ideal_absorber():
    cells = O.rig_cells(100, 20)
    ky = O.ky_from(F0, 82.0)
    f = np.array([9.95e9, 1.0e10, 1.005e10])
    vac = O.yee_lattice_full(f, ky, cells, n_cpml=20)
    assert np.max(vac["r_amp"]) < 1e-9 and np.allclose(vac["T"], 1.0, atol=1e-9)   # TFSF removes the incident at x_hi
    pec = O.yee_lattice_full(f, ky, cells, ideal_absorber=True, aux="plane", pec=True)
    assert np.allclose(pec["R"], 1.0, atol=1e-9)
    pec_c = O.yee_lattice_full(f, ky, cells, n_cpml=20, pec=True)
    assert np.max(np.abs(pec_c["R"] - 1.0)) > 0.05                   # the absorber echo is inside the compact box


def test_cpml_profile_transcription_equals_rfx():
    """The numpy profile equals rfx/boundaries/cpml.py _cpml_profile to float32 (lo face), and the aux
    profile equals tfsf_2d's construction."""
    pytest.importorskip("rfx")
    from rfx.boundaries.cpml import _cpml_profile
    p = _cpml_profile(20, O.DT_S, O.DX_M)
    mine = O.cpml_profile_np(20, O.DT_S, O.DX_M)
    for k in ("b", "c", "kappa", "sigma", "alpha"):
        assert np.allclose(np.asarray(getattr(p, k), float), mine[k], rtol=2e-7, atol=1e-9), k
    from rfx.sources.tfsf_2d import init_tfsf_2d
    cfg, _ = init_tfsf_2d(141, 45, O.DX_M, O.DT_S, cpml_layers=20, tfsf_margin=5, f0=F0, bandwidth=0.05, theta_deg=45.0)
    aux = O.aux_cpml_profile_np(O.DT_S, O.DX_M)
    assert np.allclose(np.asarray(cfg.b_cpml, float), aux["b"], rtol=2e-7)
    assert np.allclose(np.asarray(cfg.c_cpml, float), aux["c"], rtol=2e-7)
    assert np.allclose(np.asarray(cfg.kappa_cpml, float), aux["kappa"], rtol=2e-7)
    ac = O.aux_cells(O.rig_cells(100, 20))
    assert (cfg.n2x, cfg.i0_x, cfg.src_x) == (ac["n2x"], ac["i0_x"], ac["src_x"])
    assert cfg.src_t0 == pytest.approx(O.SRC_T0_OVER_TAU * cfg.src_tau)
    assert -cfg.k_transverse == pytest.approx(O.ky_from(F0, 45.0), rel=1e-9)


# ---------------------------------------------------------------------------
# The auxiliary layout is the MODULE's, not a copy (#888)
# ---------------------------------------------------------------------------
# Before #888 the comparator restated tfsf_2d's auxiliary constants as its own
# literals (30 / 25 / 3 / order 4 / kappa_max 7 / sigma factor 0.8) and the copy
# drifted a full absorber redesign behind the module: with the module at 200 /
# order 3 / kappa_max 1 / sigma_max derived from R = 1e-14, ``aux_cells`` and
# ``aux_cpml_profile_np`` modelled an auxiliary grid that no longer existed.
# These two tests fail if the copy is ever reintroduced.

def test_aux_layout_constants_are_the_modules_not_a_copy():
    """Every auxiliary constant the comparator exposes IS the module's object,
    and the derived source-to-x_lo run follows from them."""
    pytest.importorskip("rfx")
    import rfx.sources.tfsf_2d as T
    assert O.AUX_N_CPML == T.AUX_N_CPML
    assert O.AUX_N_MARGIN == T.AUX_N_MARGIN_X
    assert O.AUX_SRC_OFFSET == T.AUX_SRC_OFFSET
    assert O.AUX_CPML_ORDER == T.AUX_CPML_ORDER
    assert O.AUX_CPML_KAPPA_MAX == T.AUX_CPML_KAPPA_MAX
    assert O.AUX_CPML_R_ASYMPTOTIC == T.AUX_CPML_R_ASYMPTOTIC
    # the pre-#888 heuristic's sigma factor has no successor: sigma_max is
    # DERIVED from the reflection target, so a resurrected factor is a defect
    assert not hasattr(O, "AUX_SIGMA_FACTOR")
    assert O.AUX_SRC_TO_X_LO == T.AUX_N_MARGIN_X - T.AUX_SRC_OFFSET
    # and the rig's bookkeeping carries the same number (26_oblique_slab_fresnel
    # asserts cfg.i0_x - cfg.src_x against it)
    assert O.rig_cells(100, 20)["aux_src_to_x_lo"] == O.AUX_SRC_TO_X_LO


def test_aux_profile_equals_a_live_init_tfsf_2d_config_to_float32():
    """``aux_cpml_profile_np`` reproduces what the SHIPPED module builds, not a
    transcription of it: cast to float32 -- the dtype cfg stores -- the arrays
    are bit-identical, and in float64 they differ only by that rounding."""
    pytest.importorskip("rfx")
    from rfx.sources.tfsf_2d import init_tfsf_2d
    cfg, _ = init_tfsf_2d(141, 45, O.DX_M, O.DT_S, cpml_layers=20, tfsf_margin=5,
                          f0=F0, bandwidth=0.05, theta_deg=45.0)
    aux = O.aux_cpml_profile_np(O.DT_S, O.DX_M)
    for key, live in (("b", cfg.b_cpml), ("c", cfg.c_cpml), ("kappa", cfg.kappa_cpml)):
        live64 = np.asarray(live, dtype=np.float64)
        mine = np.asarray(aux[key], dtype=np.float64)
        assert live64.shape == mine.shape == (O.AUX_N_CPML,), key
        # bit-identical once rounded the way rfx stores it
        assert np.array_equal(np.float32(mine), np.asarray(live)), key
        # and the residual in float64 is float32 round-off (2^-23 = 1.19e-07)
        scale = float(np.max(np.abs(live64)))
        assert np.max(np.abs(live64 - mine)) <= 1.2e-7 * max(scale, 1.0), key
    # and the profile is the DERIVED one, end to end: index 0 is the outer cell
    # carrying sigma_max = -ln(R_asym)(m+1)/(2 eta d) kappa_max -- from the
    # reflection target, not a 0.8 (m+1)/(eta dx) factor -- falling to 0 at the
    # inner edge, kappa flat at kappa_max = 1 (the aux grid carries one
    # propagating mode, so no stretching), alpha the only term left inside.
    sigma_max = (-math.log(O.AUX_CPML_R_ASYMPTOTIC) * (O.AUX_CPML_ORDER + 1)
                 / (2.0 * O.ETA_0 * O.AUX_N_CPML * O.DX_M) * O.AUX_CPML_KAPPA_MAX)
    assert aux["sigma"][0] == pytest.approx(sigma_max, rel=1e-12)
    assert aux["sigma"][-1] == 0.0
    assert O.AUX_CPML_KAPPA_MAX == 1.0 and np.all(aux["kappa"] == 1.0)
    assert aux["b"][0] == pytest.approx(math.exp(-sigma_max * O.DT_S / O.EPS_0), rel=1e-12)
    assert aux["b"][-1] == pytest.approx(math.exp(-O.CPML_ALPHA_MAX * O.DT_S / O.EPS_0), rel=1e-12)
    assert aux["b"][0] < aux["b"][-1]
    # the pre-#888 heuristic would have put sigma_max ~2 orders of magnitude
    # higher on a 30-cell layer; nothing here may reproduce it
    # -ln(1e-28) * 4 / (2 eta * 200 dx) = 1.7114 S/m: the target re-derived at 82 deg
    # (lane A dc597063, note section 12). The 70-deg derivation read 0.8558 here.
    assert sigma_max == pytest.approx(1.7114, rel=1e-3)


# ---------------------------------------------------------------------------
# The auxiliary echo cannot cancel out of e_absorber (#888 fix candidate 2)
# ---------------------------------------------------------------------------
# ``predict_settling``'s e_absorber used to difference the rig against a control
# built with the DEFAULT aux="model", i.e. a control carrying the SAME auxiliary
# absorber.  Einc was then identical on both sides and the auxiliary echo -- the
# dominant term, 0.2431 in R at te_45 in round 2 -- subtracted out exactly, so
# the quantity was blind to the absorber it is named after.  The control is now
# clean on BOTH grids: ideal_absorber (3-D CPML) AND aux_echo_free (auxiliary
# CPML).  These three tests fail if the cancellation is ever reintroduced.

# an auxiliary absorber deliberately made terrible, in init_tfsf_2d's own
# override names: 6 cells at a 0.5 reflection target -> |B/A| ~ 0.4
AUX_AWFUL = {"aux_n_cpml": 6, "aux_cpml_r_asymptotic": 0.5}


def _aux_reflection(cells, ky, aux_kwargs=None, echo_free=False):
    """|B/A| of the two-mode fit E = A e^{-j kx x} + B e^{+j kx x} over the
    auxiliary grid's total-field span -- the auxiliary absorber's own amplitude
    reflection, measured on the comparator's own auxiliary lattice."""
    f = np.array([9.5e9, 10.0e9, 10.5e9])
    E = O.aux_lattice_field(f, ky, cells, echo_free=echo_free, aux_kwargs=aux_kwargs)
    ac = O.aux_cells(cells, aux_kwargs=aux_kwargs)
    kx = O.yee_kx(f, ky, 1.0, 1.0, O.DX_M, O.DT_S)
    i = np.arange(ac["i0_x"], ac["i0_x"] + (cells["x_hi"] - cells["x_lo"] + 2))
    out = []
    for m in range(f.size):
        M = np.stack([np.exp(-1j * kx[m] * i * O.DX_M), np.exp(+1j * kx[m] * i * O.DX_M)], axis=1)
        A, B = np.linalg.lstsq(M, E[m][i], rcond=None)[0]
        out.append(abs(B / A))
    return float(np.max(out))


def test_the_echo_free_auxiliary_grid_is_reflectionless_and_the_shipped_one_is_not():
    """``aux_lattice_field(echo_free=True)`` is the same grid, the same source at
    the same src_x, with an outgoing-wave termination in place of the auxiliary
    CFS-CPML: beyond the source the field is a PURE +x lattice wave, so the
    node-to-node ratio is exp(-j kx dx) to round-off and |B/A| is at the solve's
    floor.  With the absorber it is not."""
    cells = O.rig_cells(100, 20)
    f = np.array([9.5e9, 10.0e9, 10.5e9])
    for th in (0.0, 45.0, 60.0):
        ky = O.ky_from(F0, th)
        E = O.aux_lattice_field(f, ky, cells, echo_free=True)
        ac = O.aux_cells(cells)
        kx = O.yee_kx(f, ky, 1.0, 1.0, O.DX_M, O.DT_S)
        for m in range(f.size):
            seg = E[m][ac["src_x"] + 1:]
            ratio = seg[1:] / seg[:-1]
            assert np.max(np.abs(ratio - np.exp(-1j * kx[m] * O.DX_M))) < 1e-12, (th, f[m])
        assert _aux_reflection(cells, ky, echo_free=True) < 1e-12, th
        # the shipped 200-cell absorber is very good but NOT reflectionless, and
        # the deliberately terrible one is 5 decades worse
        assert 1e-14 < _aux_reflection(cells, ky) < 1e-4, th
        assert _aux_reflection(cells, ky, aux_kwargs=AUX_AWFUL) > 0.1, th
    # the layout follows the absorber, exactly as init_tfsf_2d makes it
    bad = O.aux_cells(cells, aux_kwargs=AUX_AWFUL)
    assert (bad["n_cpml"], bad["i0_x"], bad["src_x"]) == (6, 6 + O.AUX_N_MARGIN, 6 + O.AUX_SRC_OFFSET)
    with pytest.raises(ValueError):
        O.aux_overrides({"aux_n_cmpl": 6})                    # a typo must not model the shipped grid
    with pytest.raises(ValueError):
        O.yee_lattice_full(f, 0.0, cells, aux="plane", aux_echo_free=True)
    # and the surgical control does the job the note's blunter aux="plane"
    # suggestion would have done: with both grids terminated the measured R / T
    # are the ideal-plane-wave ones to round-off, at every angle and on the slab,
    # WITHOUT replacing the source or the normalisation
    for th in (0.0, 45.0, 82.0):
        ky = O.ky_from(F0, th)
        fp = f[f > 1.01 * O.cutoff_hz(ky)] if th > 0 else f
        clean = O.yee_lattice_full(fp, ky, cells, eps_slab=4.0, ideal_absorber=True, aux_echo_free=True)
        plane = O.yee_lattice_full(fp, ky, cells, eps_slab=4.0, ideal_absorber=True, aux="plane")
        assert np.max(np.abs(clean["R"] - plane["R"])) < 1e-12, th
        assert np.max(np.abs(clean["T"] - plane["T"])) < 1e-10, th
        # the identities the control must satisfy on its own
        vac = O.yee_lattice_full(fp, ky, cells, ideal_absorber=True, aux_echo_free=True)
        pec = O.yee_lattice_full(fp, ky, cells, ideal_absorber=True, aux_echo_free=True, pec=True)
        assert np.max(vac["r_amp"]) < 1e-12 and np.allclose(vac["T"], 1.0, atol=1e-12), th
        assert np.allclose(pec["R"], 1.0, atol=1e-12), th


def test_e_absorber_is_blind_to_the_auxiliary_absorber_when_the_control_shares_it():
    """The defect itself, reproduced: differencing against a control that carries
    the rig's own auxiliary absorber cancels the auxiliary echo EXACTLY, so the
    quantity does not move when that absorber is destroyed.  ``predict_settling``
    still reports this number -- as ``e_absorber_main_only``, the 3-D CPML term
    alone -- and it must keep being blind, because that is what makes it a
    decomposition rather than the gate."""
    spec = O.arm_spec("te_00")
    kw = dict(nx_interior=100, nfft=1 << 12)
    good = O.predict_settling(spec, **kw)
    bad = O.predict_settling(spec, aux_kwargs=AUX_AWFUL, **kw)
    # the auxiliary absorber really was destroyed
    cells = O.rig_cells(100, O.N_CPML)
    assert _aux_reflection(cells, spec["ky"]) < 1e-4
    assert _aux_reflection(cells, spec["ky"], aux_kwargs=AUX_AWFUL) > 0.1
    # ... and the shared-aux quantity does not notice: under 20 % either way
    ratio = bad["e_absorber_main_only"] / good["e_absorber_main_only"]
    assert 0.8 < ratio < 1.2, ratio
    # it is exactly the pre-#888 expression, recomputed here from the series
    for r, k in ((good, None), (bad, AUX_AWFUL)):
        ser = O.record_probe_series(spec, aux_kwargs=k, **kw)
        ctl = O.record_probe_series(spec, ideal_absorber=True, aux_kwargs=k, **kw)
        n = r["n_settle"]
        e = max(float(np.abs(ser[q][:n] - ctl[q][:n]).max()) / r["inc_peak"] for q in ("tot_r", "tot_t"))
        assert e == pytest.approx(r["e_absorber_main_only"], rel=1e-12)


def test_e_absorber_tracks_the_auxiliary_absorber_and_is_not_the_shared_aux_quantity():
    """The gate quantity. ``e_absorber`` is measured against a control with no
    absorber on EITHER grid, so destroying the auxiliary absorber moves it by
    almost an order of magnitude on a rig where the blind quantity moves by 1 %.
    Reverting the control to the rig's own auxiliary grid makes ``e_absorber``
    equal ``e_absorber_main_only`` and fails every assertion here."""
    spec = O.arm_spec("te_00")
    kw = dict(nx_interior=100, nfft=1 << 12)
    good = O.predict_settling(spec, **kw)
    bad = O.predict_settling(spec, aux_kwargs=AUX_AWFUL, **kw)
    assert good["e_absorber_control"] == "ideal_absorber + aux_echo_free"
    # it tracks: >= 5x on an absorber 5 decades worse
    assert bad["e_absorber"] / good["e_absorber"] > 5.0, (good["e_absorber"], bad["e_absorber"])
    # and it is NOT the shared-aux quantity -- that is the cancellation itself
    assert bad["e_absorber"] > 5.0 * bad["e_absorber_main_only"], bad
    # the control really is clean on both grids: the auxiliary term alone is what
    # separates the two controls, and it is the whole of the difference
    ser = O.record_probe_series(spec, aux_kwargs=AUX_AWFUL, **kw)
    shared = O.record_probe_series(spec, ideal_absorber=True, aux_kwargs=AUX_AWFUL, **kw)
    clean = O.record_probe_series(spec, ideal_absorber=True, aux_echo_free=True,
                                  aux_kwargs=AUX_AWFUL, **kw)
    n = bad["n_settle"]
    aux_only = max(float(np.abs(shared[q][:n] - clean[q][:n]).max()) / bad["inc_peak"]
                   for q in ("tot_r", "tot_t"))
    e_clean = max(float(np.abs(ser[q][:n] - clean[q][:n]).max()) / bad["inc_peak"]
                  for q in ("tot_r", "tot_t"))
    assert e_clean == pytest.approx(bad["e_absorber"], rel=1e-12)
    assert aux_only > 5.0 * bad["e_absorber_main_only"], (aux_only, bad["e_absorber_main_only"])
    # the a-priori R / T term the record has to answer for follows it
    assert bad["W_absorber_R_max"] > good["W_absorber_R_max"] > 0.0
    assert not bad["absorber_ok"]


def test_continuum_pml_reflection():
    assert O.cpml_continuum_reflection(0.0) == pytest.approx(1e-15)
    assert O.cpml_continuum_reflection(math.radians(82.0)) == pytest.approx(8.17e-3, rel=1e-2)
    assert O.cpml_continuum_reflection(math.radians(85.0)) == pytest.approx(4.93e-2, rel=1e-2)


# ---------------------------------------------------------------------------
# Windows, bandwidths, records, recipes (note sections 2, 4, 6)
# ---------------------------------------------------------------------------

def test_windows_are_cv04s_committed_envelope_through_the_shared_policy():
    import json
    golden = json.loads((_REPO / "tests/fixtures/golden_workflows/multilayer_fresnel.json").read_text())
    base = {m["id"]: m["observed_baseline"] for m in golden["expected_metrics"]}
    assert O.CV04_ENVELOPE["mean_dR"] == base["mean_reflectance_error"] == 0.0066
    assert O.CV04_ENVELOPE["mean_dT"] == base["mean_transmittance_error"] == 0.011
    assert O.CV04_ENVELOPE["per_bin_max_RT_closure"] == 0.0487
    assert O.W_BIN == gate_from_envelope(0.0487, quantum=1000) == 0.074
    assert O.W_MEAN_R == gate_from_envelope(0.0066, quantum=1000) == 0.010
    assert O.W_MEAN_T == gate_from_envelope(0.011, quantum=1000) == 0.017
    assert O.LEAK_BAR == 1e-3 and O.PML_FLOOR_R == pytest.approx(2.001e-3) and O.PML_REL == 0.5
    assert O.injection_term(1.0) == O.PML_FLOOR_R


def test_bandwidth_is_set_by_the_purity_bar_at_the_cutoff():
    assert O.CUTOFF_INC_AMP == O.TAIL_PURITY_LIMIT == 1e-3
    for arm, th in O.ARM_THETA0_DEG.items():
        bw = O.ARM_BW[arm]
        assert bw <= O.BW_MAX
        fc = O.cutoff_hz(O.ky_from(F0, th))
        if th > 0:
            assert O.incident_amp_rel(fc, F0, bw) <= 1e-3
    assert O.ARM_BW == {"te_00": 0.25, "te_30": 0.1902, "te_45": 0.1114, "te_60": 0.0509,
                        "tm_00": 0.25, "tm_45": 0.1114, "tm_60": 0.0509}
    assert O.GRAZE_BW == 0.0037 and O.GRAZE_THETA0_DEG == 82.0


def test_records_are_the_declared_lattice_settling_steps():
    """Section 13 (round 2). The record of every arm and rung is the DECLARED
    settling step of the exact lattice, not round 1's closed form -- which
    under-predicts by 1.42x at 30 deg and by 2.4-2.8x at 45 and 60 deg,
    because the witness is broadband and what binds it lives near the cutoff,
    not at the gated band edge (``theta_eff_deg``)."""
    for arm in O.ARM_ORDER:
        for K in sorted({1, O.ARM_DX_DIV[arm]}):
            r = O.derive_record(O.arm_spec(arm), dx_div=K)
            assert r["record_source"].startswith("declared"), (arm, K, r["record_source"])
            assert r["n_steps"] == O.RECORD_DECLARED[(arm, K, O.N_CPML, O.NX_INTERIOR)]["n_settle"]
            if O.ARM_THETA0_DEG[arm] > 0:
                # the closed form is an UNDER-estimate at every oblique rung (the
                # physical claim), and the record is set by content far outside the
                # gated band. The SIZE of the under-estimate is pinned per rung from
                # the derived table, not asserted as a round bar: round 2's "2.4-2.8x
                # at 45 and 60 deg" was the 30-cell auxiliary absorber's number.
                ratio = r["n_steps"] / r["n_closed_form"]
                assert ratio > 1.0, (arm, K, ratio)
                assert ratio == pytest.approx(O.RECORD_DECLARED_CLOSED_FORM_RATIO[(arm, K)], rel=0.02), (arm, K, ratio)
                assert r["theta_eff_deg"] > r["theta_gate_hi_deg"] + 10.0, (arm, K, r["theta_eff_deg"])
    for arm in O.GRAZE_ARMS:
        r = O.derive_record(O.arm_spec(arm))
        decl = O.RECORD_DECLARED[(arm, 1, O.N_CPML_COMPACT, O.NX_INTERIOR_GRAZE)]["n_settle"]
        assert r["record_source"].startswith("declared") and r["n_steps"] == decl, (arm, r["n_steps"], decl)


def test_the_record_of_round_1s_settled_arms_is_reproduced():
    """The arms the rig actually settles bracket the declared record: each settled
    inside one RECORD_EXTEND_STEPS quantum of it.

    Re-anchored 2026-09-06 to settling MEASURED on the rig that ships: primary rig at
    N_CPML_PRIMARY = 80 (close note section 9, pre-declaration section 17), compact box
    at N_CPML_COMPACT = 20, auxiliary absorber R_asym = 1e-28. Each arm was run by the
    case from the closed-form record (no declared entry existed at 80 yet) and extended
    in its quantum until the witnesses settled -- "record: derived N -> M (k ext)":
    te_00 dx 1597 (0 ext), tm_00 dx 1597, te_30 dx 2172 -> 3172 (10 ext of 100),
    te_30 dx/2 4296 -> 6296 (10 ext of 200), te_45 dx/2 6020 -> 10620 (23 of 200),
    te_60 dx/2 9870 -> 18470 (43 of 200), tm_45 dx/2 5839 -> 10239 (22 of 200),
    graze_pec 21735 -> 21835 (1 ext of 100, compact, unchanged rig). Every declared
    record must bracket its measured settle inside one extension quantum. Logs under
    the session scratchpad settle80_*/ and lad2_*_80/. The round-1 anchors (1597 /
    3172 / 6496 / 22001) were the 30-cell auxiliary absorber's and the 20-cell primary
    rig's; the 2026-09-05 anchors were the 1e-28 auxiliary on the 20-cell rig.
    """
    for arm, K, measured, quantum in (("te_00", 1, 1597, 100), ("tm_00", 1, 1597, 100),
                                      ("te_30", 1, 3172, 100), ("te_30", 2, 6296, 200),
                                      ("te_45", 2, 10620, 200), ("te_60", 2, 18470, 200),
                                      ("tm_45", 2, 10239, 200),
                                      ("graze_pec", 1, 21835, 100)):
        key = (arm, K, O.declared_n_cpml(O.arm_spec(arm)), O.arm_spec(arm)["nx_interior"])
        decl = O.RECORD_DECLARED[key]
        n = O.derive_record(O.arm_spec(arm), dx_div=K)["n_steps"]
        if decl.get("nfft_converged") is False:
            # the model cannot vouch for one number here; what it CAN say is that the
            # FDTD settles inside the range its own nfft ladder spans, and the declared
            # (largest) value is not more than one quantum beyond the measured settle
            lad = decl["n_settle_ladder_2e17_to_2e20"]
            assert min(lad) - 2 * quantum < measured <= max(lad) + quantum, (arm, K, measured, lad)
            assert n == max(lad)
        else:
            assert measured - 2 * quantum < n <= measured + quantum, (arm, K, n, measured)


def test_the_absorber_echo_over_the_record_is_what_picks_dx_over_2():
    """Section 13.4 said the echo picks dx/2: at dx the 20-cell CPML's grazing
    reflection put the 45 deg arms outside W_bin, and at dx/2 it did not.

    On the 80-cell primary rig (pre-declaration section 17) that argument no
    longer exists: the MAIN absorber's echo over the record is under 1e-3 at
    BOTH rungs for every oblique arm (te_45 dx/2 1.8e-04, te_60 dx 7.2e-04 --
    170x and 50x below the 20-cell values), so the absorber term is not what
    separates dx from dx/2 any more. What IS asserted: absorber_ok on both rungs
    at the declared depth, an echo under 1e-3 (a term of ~1e-3 in R against
    W_MEAN_R = 0.010, i.e. the absorber is no longer the leading term), and that
    the recipe's dx/2 choice survives -- it now rests on the lattice dispersion
    term W_disp (halved at dx/2), which is the case's own printed reason. The
    old assertion that dx FAILS absorber_ok is recorded here as what changed:
    a 20-cell absorber failure, not a property of the oblique arms."""
    for arm in ("te_45", "te_60", "tm_45", "tm_60"):
        r2 = O.derive_record(O.arm_spec(arm), dx_div=2)
        r1 = O.derive_record(O.arm_spec(arm), dx_div=1)
        for K, r in ((2, r2), (1, r1)):
            assert r["record_source"].startswith("declared"), (arm, K, r["record_source"])
            assert r["absorber_ok"], (arm, K, r["W_absorber_R_max"], r["W_absorber_T_max"])
            assert r["e_absorber"] < 1.0e-3, (arm, K, r["e_absorber"])
        assert O.ARM_DX_DIV[arm] == 2      # the recipe is unchanged; its reason moved to W_disp
    for arm in ("te_00", "tm_00", "te_30"):
        assert O.derive_record(O.arm_spec(arm), dx_div=O.ARM_DX_DIV[arm])["absorber_ok"], arm


def test_yee_group_velocity_along_x_vanishes_at_the_cutoff():
    """``yee_vgx``: the reason an oblique record is not the normal-incidence
    record scaled by a constant."""
    ky = O.ky_from(F0, 45.0)
    fc = O.cutoff_hz(ky)
    for K in (1, 2):
        dx, dt = O.DX_M / K, O.DT_S / K
        f = np.array([fc * 1.0001, 7.6e9, 10e9, 14e9])
        v = O.yee_vgx(f, ky, 1.0, 1.0, dx, dt) / O.C0
        cos_t = np.cos(O.realized_theta_rad(f, ky))
        # the x group velocity collapses at the cutoff (the lattice sits slightly
        # above the continuum there, which is the direction that does not flatter it)
        assert cos_t[0] < 0.02 and v[0] < 0.05 and v[0] > cos_t[0]
        assert np.all(np.abs(v[1:] - cos_t[1:]) < 0.01 / K ** 2)   # -> c cos(theta), second order
    assert O.yee_vgx(np.array([10e9]), 0.0, 1.0, 1.0, O.DX_M, O.DT_S)[0] / O.C0 > 0.99


@pytest.mark.slow
def test_predict_settling_reproduces_a_declared_record():
    """The declared table is not a pinned guess: re-derive one entry."""
    # at the length the table was derived at -- the 2**17 default is too short for it
    got = O.predict_settling(O.arm_spec("te_00"), dx_div=1, nfft=O.RECORD_DECLARED_NFFT)
    decl = O.RECORD_DECLARED[("te_00", 1, O.N_CPML, O.NX_INTERIOR)]
    assert got["n_settle"] == decl["n_settle"]
    assert abs(got["e_absorber"] - decl["e_absorber"]) <= 1e-3 * decl["e_absorber"]


def test_primary_recipe_follows_the_margin_rule():
    """Section 4.6: dx/2 iff the a-priori lattice term sits within 1.5x of the mean window at dx."""
    for arm in O.ARM_ORDER:
        assert O.primary_dx_div(arm) == O.ARM_DX_DIV[arm], arm
        m1 = O.lattice_margin(arm, 1)
        m = O.lattice_margin(arm, O.ARM_DX_DIV[arm])
        assert min(m["margin_R"], m["margin_T"]) >= O.LATTICE_MARGIN_MIN, (arm, m)
        if O.ARM_DX_DIV[arm] == 2:
            assert min(m1["margin_R"], m1["margin_T"]) < O.LATTICE_MARGIN_MIN and min(m["margin_R"], m["margin_T"]) >= 2.7, (arm, m1, m)
        else:
            assert min(m1["margin_R"], m1["margin_T"]) >= 1.8, (arm, m1)


@pytest.mark.parametrize("name", sorted(O.FALSIFIERS))
def test_every_declared_falsifier_has_an_analytic_margin(name):
    p = O.falsifier_prediction(name)
    if name.startswith("graze"):
        assert p["predicted_fails_G6"]
        assert p["bins_beyond_window"] >= (60 if name == "graze_pec_depth_half" else 20)
        # The pre-declaration (2026-09-02 note, line 383) declares F5b by |dR| max
        # against the window and bins beyond it -- both asserted above. The ">10"
        # ratio bar that used to sit here was an extra this file added, and it was
        # met (16.6) only because the pre-#888 auxiliary echo and the 3-D CPML's
        # grazing error happened to cancel at one bin, giving a near-zero
        # denominator. On the absorber that ships the ratio is 2.15 for sigma_half
        # and 18.2 for depth_half. The falsifier still FIRES (predicted_fails_G6,
        # 38 / 69 bins); what is asserted now is that it fires and by how much,
        # pinned, so a further collapse is visible.
        assert p["max_ratio_excess_def_over_decl"] > 1.0
        assert p["max_ratio_excess_def_over_decl"] == pytest.approx(
            {"graze_pec_sigma_half": 2.154, "graze_pec_depth_half": 18.234}[name], rel=0.05)
    else:
        assert p["predicted_fails"]
        assert p["ratio_mean_R"] >= 2.9, (name, p["ratio_mean_R"])
        if name == "tm_60_swap_te":
            assert p["brewster_bin"]["R_te_oracle"] > 4 * p["brewster_bin"]["floor"]


def test_rejected_falsifiers_were_coin_tosses():
    for name, ratio in O.FALSIFIERS_REJECTED.items():
        if isinstance(ratio, float):
            assert 0.8 < ratio < 1.5


# ---------------------------------------------------------------------------
# The Meep leg's acceptance (note section 14, round 2). Round 1's leg wrote
# R = -inf / T = +inf for all 400 bins on te_00 and te_30 and the E4 gate read
# them; nothing below touches Meep, only the contract the artifact must meet.
# ---------------------------------------------------------------------------

def _meep_arrays(arm, kind):
    spec = O.arm_spec(arm)
    f = np.linspace(spec["f0_hz"] * (1 - 1.2 * spec["bw"]), spec["f0_hz"] * (1 + 1.2 * spec["bw"]), 400)
    R_an, T_an = O.oracle_RT(f, spec["ky"], spec["pol"])
    R_an = np.nan_to_num(R_an); T_an = np.nan_to_num(T_an, nan=1.0)
    inc = np.ones_like(f)
    if kind == "good":
        return spec, f, R_an, T_an, inc
    if kind == "round1":                       # the exact shape of the two dead legs
        return spec, f, np.full_like(f, -np.inf), np.full_like(f, np.inf), np.zeros_like(f)
    if kind == "nonphysical":                  # finite, and still not a reference
        return spec, f, R_an, T_an + 1.0, inc
    raise ValueError(kind)


def test_meep_acceptance_rejects_round_1s_artifact_by_name():
    spec, f, R, T, inc = _meep_arrays("te_30", "round1")
    acc = O.meep_accept(f, R, T, inc, spec)
    assert not acc["accepted"]
    assert any("non-finite" in r for r in acc["reasons"])
    assert any("identically zero" in r for r in acc["reasons"])


def test_meep_acceptance_rejects_a_finite_but_non_physical_leg():
    spec, f, R, T, inc = _meep_arrays("te_45", "nonphysical")
    acc = O.meep_accept(f, R, T, inc, spec)
    assert not acc["accepted"]
    assert any("T outside" in r for r in acc["reasons"]) or any("R + T - 1" in r for r in acc["reasons"])


def test_meep_acceptance_accepts_the_oracle_itself_and_is_not_an_agreement_test():
    """It must pass the analytic answer, and it must NOT reject a leg merely for
    disagreeing with it -- that would turn an E4 disagreement into a SKIP."""
    spec, f, R, T, inc = _meep_arrays("te_45", "good")
    assert O.meep_accept(f, R, T, inc, spec, inc_flux_refl=inc)["accepted"]
    off = np.clip(R + 0.30, 0.0, 1.0)          # 0.30 wrong in R: far outside every E4 window
    assert O.meep_accept(f, off, np.clip(1.0 - off, 0.0, 1.0), inc, spec)["accepted"]


def test_meep_acceptance_catches_a_degenerate_normalisation_and_a_broken_vacuum_identity():
    spec, f, R, T, inc = _meep_arrays("te_45", "good")
    g = O.gated_mask(f, spec)
    bad = inc.copy(); bad[np.flatnonzero(g)[0]] = 1e-15
    assert not O.meep_accept(f, R, T, bad, spec)["accepted"]
    assert not O.meep_accept(f, R, T, inc, spec, inc_flux_refl=inc * 1.5)["accepted"]


def test_meep_unavailable_distinguishes_absent_rejected_and_usable():
    spec, f, R, T, inc = _meep_arrays("te_45", "good")
    good = {"accepted": True, "R": R.tolist(), "T": T.tolist(), "freqs_hz": f.tolist(), "k_point": [0, 0.2, 0]}
    assert O.meep_unavailable_reason(good, "x") is None
    assert "no Meep artifact" in O.meep_unavailable_reason(None, "x")
    rej = dict(good, accepted=False, rejection_reasons=["flux normalisation is identically zero"])
    why = O.meep_unavailable_reason(rej, "x")
    assert "REJECTED" in why and "identically zero" in why
    v1 = {"R": [float("-inf")] * 3, "T": [float("inf")] * 3, "freqs_hz": [1.0, 2.0, 3.0], "k_point": [0, 0, 0]}
    assert "non-finite" in O.meep_unavailable_reason(v1, "x")   # a pre-acceptance artifact still skips


def test_meep_minimum_run_time_covers_first_arrival_on_every_arm():
    """Round 1's stop condition could fire ``MEEP_STOP_DT`` = 50 after the
    sources ended; on te_00 and te_30 the pulse had not crossed the box, the
    monitored point was still identically zero, and the run stopped with an
    empty flux monitor."""
    src_x, trans_x = -74.5, 3.5                      # the leg's own geometry, in units of a
    for arm in O.MEEP_ARMS:
        t = O.meep_min_after_sources(O.arm_spec(arm), src_x, trans_x)
        assert t >= abs(trans_x - src_x), (arm, t)   # never below the vacuum transit
        assert t > O.MEEP_STOP_DT, (arm, t)          # ... which is exactly what round 1 violated


def test_a_declared_falsifier_still_reaches_the_e4_gate():
    """A defect injection exists to be judged: withholding its arrays would turn
    the F4 falsifier into a SKIP and the lane would stop detecting the defect it
    is there to detect.  The carve-out is keyed on the artifact's own
    ``falsifier`` field, which only ``--falsifier <declared name>`` sets."""
    spec, f, R, T, inc = _meep_arrays("te_45", "nonphysical")
    assert not O.meep_accept(f, R, T, inc, spec)["accepted"]
    doc = {"accepted": False, "rejection_reasons": ["T outside [0, 1] by more than 0.06 on a gated bin"],
           "R": R.tolist(), "T": T.tolist(), "freqs_hz": f.tolist(), "k_point": [0, 0.2, 0],
           "precheck": {"passed": False}}
    assert "REJECTED" in O.meep_unavailable_reason(doc, "x")
    assert O.meep_unavailable_reason(dict(doc, falsifier="k_2pi"), "x") is None
    assert set(O.MEEP_FALSIFIERS) == {"k_2pi"}


def test_the_records_the_table_cannot_vouch_for_say_so():
    """The settling step is a property of the inverse-DFT length: undecayed content
    wraps, and at 60 degrees on dx the first crossing moves with the transform
    length. On the 20-cell rig the derivation length 2**19 was a lone outlier
    (9627 / 9623 / 10258 / 9632); on the 80-cell rig the whole ladder drifts
    (9619 / 9424 / 10258 / 9245 for both te_60 and tm_60) and there is no lone
    outlier to blame. A sustained-window criterion was measured to change
    nothing. So those keys carry ``nfft_converged = False`` and their ladder; the
    declared value is the LARGEST of the ladder (the conservative record -- more
    settling, never less); every other key must NOT carry the flag. The table
    states what it cannot vouch for, and by how much."""
    flagged = {k for k, v in O.RECORD_DECLARED.items() if v.get("nfft_converged") is False}
    assert {("te_60", 1, O.N_CPML, O.NX_INTERIOR), ("tm_60", 1, O.N_CPML, O.NX_INTERIOR)} <= flagged, flagged
    for k in flagged:
        v = O.RECORD_DECLARED[k]
        ladder = v["n_settle_ladder_2e17_to_2e20"]
        assert len(ladder) == 4 and v["n_settle"] in ladder
        assert v["n_settle"] == max(ladder), (k, ladder, "the declared record must be the conservative end of its ladder")
        spread = (max(ladder) - min(ladder)) / min(ladder)
        # measured 11.0 % on the 80-cell rig for the 60-deg keys; a collapse to nothing
        # or a blow-up past 20 % is news either way
        assert 0.02 < spread < 0.20, (k, ladder, spread)
    for k, v in O.RECORD_DECLARED.items():
        if k not in flagged:
            assert "nfft_converged" not in v, k
