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
                # the closed form is an UNDER-estimate at every oblique rung, and the
                # record is set by content far outside the gated band
                ratio = r["n_steps"] / r["n_closed_form"]
                assert ratio > (2.3 if O.ARM_THETA0_DEG[arm] >= 45 else 1.4), (arm, K, ratio)
                assert r["theta_eff_deg"] > r["theta_gate_hi_deg"] + 10.0, (arm, K, r["theta_eff_deg"])
    for arm in O.GRAZE_ARMS:
        r = O.derive_record(O.arm_spec(arm))
        assert r["record_source"].startswith("declared") and r["n_steps"] == 22008


def test_the_record_of_round_1s_settled_arms_is_reproduced():
    """The two arms round 1 actually settled bracket the declared record: each
    settled inside one RECORD_EXTEND_STEPS quantum of it.  ``rfx_baseline.log``
    of run cv26-oblique-r1-20260902T162340Z: te_00 1597 (0 ext), te_30 3172
    (10 ext of 100) and 6496 (11 ext of 200), graze_pec 22001 (5 ext of 100)."""
    for arm, K, measured, quantum in (("te_00", 1, 1597, 100), ("te_30", 1, 3172, 100),
                                      ("te_30", 2, 6496, 200), ("graze_pec", 1, 22001, 100)):
        n = O.derive_record(O.arm_spec(arm), dx_div=K)["n_steps"]
        assert measured - 2 * quantum < n <= measured + quantum, (arm, K, n, measured)


def test_the_absorber_echo_over_the_record_is_what_picks_dx_over_2():
    """Section 13.4. The echo is gated by its AMPLITUDE inside the record, not
    by its arrival: at dx the 20-cell CPML's grazing reflection puts the 45 deg
    arms outside W_bin, and at dx/2 it does not."""
    for arm in ("te_45", "te_60", "tm_45", "tm_60"):
        r2 = O.derive_record(O.arm_spec(arm), dx_div=2)
        r1 = O.derive_record(O.arm_spec(arm), dx_div=1)
        assert r2["absorber_ok"], (arm, r2["W_absorber_R_max"], r2["W_absorber_T_max"])
        assert r1["e_absorber"] > 1.6 * r2["e_absorber"], (arm, r1["e_absorber"], r2["e_absorber"])
    for arm in ("te_45", "tm_45", "tm_60"):
        assert not O.derive_record(O.arm_spec(arm), dx_div=1)["absorber_ok"], arm
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
    got = O.predict_settling(O.arm_spec("te_00"), dx_div=1)
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
        assert p["max_ratio_excess_def_over_decl"] > 10
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
