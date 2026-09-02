"""cv23 lossy slab -- gate replay and window-derivation witnesses.

Two kinds of test live here:

1. **Artifact-free** (always run): the windows are cv22's cv04-derived ones
   plus the triangle sums for A; the sigma-update discrete-time factor
   sigma_eff = sigma x/tan x used in the windows reproduces the LIVE
   ``update_e`` coefficients (extracted from one update on a tiny grid and
   driven sinusoidally); the arms have the tan delta / skin depth / R, T, A
   the note says; the record lengths are derived from the slab's own
   ring-down; the documented ``Simulation.add_material`` path assembles the
   same arrays as the direct construction, bit-for-bit, on every arm; every
   pre-declared falsifier exceeds its window analytically (the note's section
   6 margins, recomputed), the gain arm breaks passivity, and the Meep
   wrong-scaling falsifiers fail E4.

2. **Artifact replay** (``pytest.skip`` while the VESSL artifacts are absent):
   ``validation/crossval/_23_lossy_results/rfx.json`` is replayed through the
   same evaluators from its raw per-bin R/T and must reproduce its stored
   gate verdicts, all passing, on R, T and A; each ``rfx__falsifier_<name>.json``
   must FAIL for the pre-declared reason; each ``meep_tand1__falsifier_<x>.json``
   must fail E4 against the baseline and carry ``precheck.passed == False``.

No FDTD runs here. Pre-declaration:
``docs/design_notes/20260902_cv23_lossy_slab_predeclaration.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests._gate_policy import gate_from_envelope

_REPO = Path(__file__).resolve().parents[1]
_RESULTS = _REPO / "validation/crossval/_23_lossy_results"
_GOLDEN_CV04 = _REPO / "tests/fixtures/golden_workflows/multilayer_fresnel.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("cv23_dispersive_eps", "validation/crossval/comparators/dispersive_eps.py")
G = _load("cv23_cv22_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")
L = _load("cv23_lossy_gates", "validation/crossval/comparators/cv23_lossy_gates.py")

EPS_0 = 8.8541878128e-12


def _rig_dt() -> float:
    from rfx.grid import Grid
    grid = Grid(freq_max=20e9, domain=(G.NX_INTERIOR_R3 * G.DX_M, 0.004, G.DX_M),
                dx=G.DX_M, cpml_layers=G.N_CPML, mode="2d_tmz")
    return float(grid.dt)


def _records(dt):
    return {arm: L.derive_record_length(L.ARMS[arm]["params"], dt) for arm in L.ARM_ORDER}


def _rfx_bins():
    """The rfx rFFT bin grid of the derived records (longest 1181 -> nfft 16384;
    all arms share it, see test_record_lengths_are_derived)."""
    dt = _rig_dt()
    n_steps = max(r["n_steps"] for r in _records(dt).values())
    nfft = int(2 ** np.ceil(np.log2(n_steps)) * G.NFFT_OVERSAMPLE)
    f = np.fft.rfftfreq(nfft, d=dt)
    f = f[(f > G.MASK_F_LO_HZ) & (f < G.MASK_F_HI_HZ)]
    return f, dt


# ---------------------------------------------------------------------------
# 1. Artifact-free witnesses
# ---------------------------------------------------------------------------

def test_windows_are_cv22s_plus_the_triangle_sums_for_absorption():
    golden = json.loads(_GOLDEN_CV04.read_text())
    baseline = {m["id"]: m["observed_baseline"] for m in golden["expected_metrics"]}
    assert L.W_BIN == G.W_BIN == 0.074 and L.W_MEAN_R == G.W_MEAN_R == 0.010 and L.W_MEAN_T == G.W_MEAN_T == 0.017
    assert L.W_BIN_A == 2 * G.W_BIN == pytest.approx(0.148)
    assert L.W_MEAN_A == G.W_MEAN_R + G.W_MEAN_T == pytest.approx(0.027)
    # the tighter A window is DERIVABLE from cv04's closure (reported, not gated)
    assert L.CV04_MEAN_CLOSURE == baseline["mean_energy_closure_error"] == 0.0091
    assert L.W_BIN_A_TIGHT == gate_from_envelope(0.0487, quantum=1000) == 0.074
    assert L.W_MEAN_A_TIGHT == gate_from_envelope(0.0091, quantum=1000) == 0.014
    assert L.W_BIN_A_TIGHT < L.W_BIN_A and L.W_MEAN_A_TIGHT < L.W_MEAN_A
    assert L.MEEP_PRIMARY_RESOLUTION == 40 and L.MEEP_EPS_AVERAGING is False


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_arm_properties_are_the_notes(arm):
    f, _ = _rfx_bins()
    g = G.gated_mask(f)
    p = L.ARMS[arm]["params"]
    td = de.tan_delta_of(f[g], p)
    R, T, A = L.analytic_rta(f[g], p)
    assert np.all((R + T) < 1.0) and np.all(A > 0.19)
    if arm == "tand0p1":
        assert 0.07 < td.min() < 0.071 and 0.17 < td.max() < 0.18
        assert R.min() < 0.01 and R.max() > 0.29 and T.min() > 0.5 and A.max() < 0.30
    elif arm == "tand1":
        assert 0.70 < td.min() < 0.71 and 1.73 < td.max() < 1.75
        assert 0.17 < R.min() and R.max() < 0.34 and 0.046 < T.min() and T.max() < 0.062 and A.max() > 0.77
    else:
        assert 2.1 < td.min() < 2.11 and 5.2 < td.max() < 5.23
        assert R.min() > 0.33 and T.max() < 0.004 and A.min() > 0.48
        # the T gate is vacuous on this arm (note section 2): T sits under the mean window everywhere
        assert T.max() < L.W_MEAN_T
    sd7 = float(de.skin_depth_m(7e9, p))
    assert {"tand0p1": 60e-3, "tand1": 7e-3, "tand3": 3e-3}[arm] < sd7 < {"tand0p1": 70e-3, "tand1": 8e-3, "tand3": 3.5e-3}[arm]
    # material path per arm, as declared
    assert L.ARMS[arm]["materials_path"] == {"tand0p1": "direct", "tand1": "api", "tand3": "api"}[arm]


def test_record_lengths_are_derived_from_the_slab_ringdown():
    """Note section 8: n_steps_min = n_pulse_end + max_f ln(100 w(f))/rate(f)/dt +
    TAIL_WINDOW over the incident ring band; no material pole (J = sigma E is
    memoryless), the etalon rate carries the absorption per pass; every arm on
    nfft 16384 inside the nx-1000 CPML gate."""
    dt = _rig_dt()
    recs = _records(dt)
    assert {a: r["n_steps"] for a, r in recs.items()} == {"tand0p1": 1067, "tand1": 1158, "tand3": 1181}
    for arm, r in recs.items():
        assert r["cpml_gate_ok"] and r["t_safe_cpml_steps"] == 1262
        assert r["n_pulse_end"] == 908 and r["settling_limit"] == 1e-2
        assert r["nx_interior"] == G.NX_INTERIOR_R3 == 1000
        assert r["rate_material_1_s"] is None           # memoryless sigma update
        assert 1.1e9 < r["f_ring_hz"] < 1.2e9            # the bottom of the ring band governs
        assert r["rate_ring_1_s"] == r["rate_etalon_slowest_1_s"]
    assert recs["tand0p1"]["n_ring"] < recs["tand1"]["n_ring"] < recs["tand3"]["n_ring"]
    nffts = {int(2 ** np.ceil(np.log2(r["n_steps"])) * G.NFFT_OVERSAMPLE) for r in recs.values()}
    assert nffts == {16384}
    f, _ = _rfx_bins()
    g = G.gated_mask(f)
    assert int(g.sum()) == 229
    assert G.incident_amplitude_rel(f[g]).min() >= G.GATED_BAND_MIN_INC_AMP_FRAC
    # falsifier records (note section 8)
    for name, want in (("tand0p1_sigma_zero", 1078), ("tand1_sigma_x1p5", 1169), ("tand0p1_sigma_neg", 1106)):
        _, bad = L.apply_falsifier(name)
        assert L.derive_record_length(bad, dt)["n_steps"] == want, name
    # cv22's committed records are unchanged by the sqrt-branch fix (note section 6, F4)
    assert {a: G.derive_record_length(G.ARMS[a]["model"], G.ARMS[a]["params"], dt)["n_steps"]
            for a in G.ARM_ORDER} == {"debye": 1108, "lorentz": 1228, "drude": 1168}


def test_gain_slab_is_below_threshold_and_a_lasing_one_is_refused():
    dt = _rig_dt()
    _, gain = L.apply_falsifier(L.PASSIVITY_FALSIFIER)
    assert gain["sigma"] < 0
    assert abs(gain["sigma"]) * L.D_SLAB_M * 376.73 / 2 < 1.0          # sigma d Z0/2 = 0.29
    rec = L.derive_record_length(gain, dt)
    assert rec["rho_etalon"] < 1.0
    with pytest.raises(ValueError):
        L.derive_record_length({**gain, "sigma": 5.0 * gain["sigma"]}, dt)  # sigma d Z0/2 = 1.47


def _live_update_e_coefficients(sigma: float, eps_r: float, dt: float):
    """Extract ca and cb of rfx's update_e on a tiny periodic grid: ez = 1 with
    H = 0 gives ez' = ca; ez = 0 with a unit step in hy at i = 3 gives
    ez'[3] = cb / dx."""
    import jax.numpy as jnp
    from rfx.core.yee import init_materials, init_state, update_e
    shape = (6, 3, 1)
    dx = G.DX_M
    mats = init_materials(shape)
    mats = mats._replace(eps_r=mats.eps_r * eps_r, sigma=mats.sigma + sigma)
    per = (True, True, True)
    s = init_state(shape)
    s1 = update_e(s._replace(ez=jnp.ones(shape, dtype=s.ez.dtype)), mats, dt, dx, per)
    ca = float(s1.ez[2, 1, 0])
    hy = jnp.zeros(shape, dtype=s.hy.dtype).at[3:, :, :].set(1.0)
    s2 = update_e(s._replace(hy=hy), mats, dt, dx, per)
    cb = float(s2.ez[3, 1, 0]) * dx
    return ca, cb


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_sigma_warp_matches_the_live_update_e_recurrence(arm):
    """Note section 3: drive E^{n+1} = ca E^n + cb C^{n+1/2} (the LIVE
    coefficients) with C = cos(w t) and recover E/C; it must equal
    1/(j w_hat eps0 eps_num), eps_num = eps' - j sigma (x/tan x)/(w eps0),
    and be closer to that than to the continuous eps."""
    dt = _rig_dt()
    p = L.ARMS[arm]["params"]
    ca, cb = _live_update_e_coefficients(p["sigma"], p["eps_inf"], dt)
    s = p["sigma"] * dt / (2 * EPS_0 * p["eps_inf"])
    assert ca == pytest.approx((1 - s) / (1 + s), rel=2e-6)            # yee.py:391-392
    assert cb == pytest.approx((dt / (EPS_0 * p["eps_inf"])) / (1 + s), rel=2e-6)
    for f in (4.5e9, 7.0e9, 9.5e9):
        w = 2 * np.pi * f
        n_per = int(round(1.0 / (f * dt)))
        n_tr, n_meas = 40 * n_per, 20 * n_per
        N = n_tr + n_meas
        t = np.arange(N + 1) * dt
        C = np.cos(w * (t + dt / 2))           # curl H at the half step
        E = np.zeros(N + 1)
        for n in range(N):
            E[n + 1] = ca * E[n] + cb * C[n]
        seg = slice(n_tr, n_tr + n_meas)
        basis = np.column_stack([np.cos(w * t[seg]), np.sin(w * t[seg])])
        (pa, pb), *_ = np.linalg.lstsq(basis, E[seg], rcond=None)
        ratio = pa - 1j * pb                   # E = Re[(A - jB) e^{jwt}], C amplitude 1 + 0j
        x = w * dt / 2
        w_hat = 2 * np.sin(x) / dt
        pred = 1.0 / (1j * w_hat * EPS_0 * de.eps_numerical_ade(f, "conductive", p, dt))
        cont = 1.0 / (1j * w_hat * EPS_0 * de.eps_analytic(f, "conductive", p))
        assert abs(ratio - pred) / abs(pred) < 2e-5, (arm, f, ratio, pred)
        assert abs(ratio - pred) < abs(ratio - cont)
    # the factor itself, at the band top
    assert de.sigma_warp(2 * np.pi * 10e9, dt) - 1 == pytest.approx(-1.79e-3, abs=2e-5)


def test_sigma_window_term_is_named_and_small_at_this_dt():
    f, dt = _rfx_bins()
    g = G.gated_mask(f)
    dt_meep = L.MEEP_COURANT * (L.MEEP_A_M / L.MEEP_PRIMARY_RESOLUTION) / de.C0
    for arm in L.ARM_ORDER:
        p = L.ARMS[arm]["params"]
        wR, wT, wA, *_ = L.sigma_window(f, p, dt)
        assert 0 < wR[g].max() < 4e-4 and 0 < wT[g].max() < 3e-4 and 0 < wA[g].max() < 4e-4
        assert max(wR[g].max(), wT[g].max(), wA[g].max()) < L.W_BIN / 100
        mp = de.to_meep("conductive", p, a_m=L.MEEP_A_M)
        wmA = L.meep_sigma_window_A(f, p, mp, dt_meep)
        assert 0 < wmA[g].max() < 2e-5


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_add_material_path_assembles_the_direct_arrays_bit_for_bit(arm):
    """Note section 2: Simulation.add_material + add(Box) + _assemble_materials
    reproduces init_materials + .at[slab].set on the nx-1000 rig grid."""
    import jax.numpy as jnp
    from rfx import Box, Simulation
    from rfx.core.yee import init_materials
    from rfx.geometry.csg import _grid_coords
    from rfx.grid import Grid
    p = L.ARMS[arm]["params"]
    domain = (G.NX_INTERIOR_R3 * G.DX_M, 0.004, G.DX_M)
    grid = Grid(freq_max=20e9, domain=domain, dx=G.DX_M, cpml_layers=G.N_CPML, mode="2d_tmz")
    cells = G.rig_cells(G.NX_INTERIOR_R3)
    lo, hi = cells["slab_lo"], cells["slab_hi"]
    assert (lo, hi) == (515, 525)
    direct = init_materials(grid.shape)
    direct = direct._replace(eps_r=direct.eps_r.at[lo:hi].set(p["eps_inf"]), sigma=direct.sigma.at[lo:hi].set(p["sigma"]))
    sim = Simulation(freq_max=20e9, domain=domain, dx=G.DX_M, cpml_layers=G.N_CPML, mode="2d_tmz")
    sim.add_material(L.API_MATERIAL_NAME, eps_r=p["eps_inf"], sigma=p["sigma"])
    xs, _, _ = _grid_coords(grid)
    sim.add(Box((float(xs[lo]), -1.0, -1.0), (float(xs[hi]), 1.0, 1.0)), material=L.API_MATERIAL_NAME)
    g2 = sim._build_grid()
    assert tuple(g2.shape) == tuple(grid.shape) and float(g2.dt) == float(grid.dt)
    mats, _, _, pec, *_ = sim._assemble_materials(g2)
    assert bool(jnp.array_equal(mats.eps_r, direct.eps_r)) and bool(jnp.array_equal(mats.sigma, direct.sigma))
    assert bool(jnp.array_equal(mats.mu_r, direct.mu_r))
    assert pec is None or not bool(jnp.any(pec))
    assert int((mats.sigma[:, 0, 0] > 0).sum()) == 10


@pytest.mark.parametrize("name", sorted(L.FALSIFIERS))
def test_rfx_falsifiers_exceed_the_windows_analytically(name):
    """Note section 6: every F1/F2/F4 defect must fail G2 (band-mean) on R, T
    or A with >= 2x margin on at least one of them (the coin-toss guard); the
    named coin tosses are recorded, and the gain arm breaks passivity."""
    f, dt = _rfx_bins()
    arm, bad = L.apply_falsifier(name)
    good = L.ARMS[arm]["params"]
    Rb, Tb, _ = L.analytic_rta(f, bad)
    e2 = L.evaluate_e2(f, Rb, Tb, good, dt)          # defective "measurement" vs the true oracle
    assert not e2["e2_ok"], name
    assert not (e2["gates"]["G2_R"] and e2["gates"]["G2_T"] and e2["gates"]["G2_A"]), name
    ratios = {"R": e2["mean_dR_gated"] / e2["mean_window_R"], "T": e2["mean_dT_gated"] / e2["mean_window_T"],
              "A": e2["mean_dA_gated"] / e2["mean_window_A"]}
    assert max(ratios.values()) >= 2.0, (name, ratios)
    if name == "tand0p1_sigma_x1p5":
        assert ratios["R"] < 1.05 and ratios["T"] > 5        # R is the coin toss; T carries it
    if name == "tand1_sigma_x1p5":
        assert ratios["A"] < 1.5 and ratios["R"] > 6         # A is the coin toss; R carries it
    if name == "tand3_sigma_x1p5":
        assert ratios["T"] < 0.2 and ratios["R"] > 7         # T vacuous; R carries it (8.1x of 0.010; 7.95x of the window incl. W_sig)
    if name.endswith("_sigma_zero"):
        assert e2["n_bins_A_over_window"] == 229 and ratios["A"] > 9
    if name == L.PASSIVITY_FALSIFIER:
        assert not e2["gates"]["G3_passivity"]
        assert np.all((Rb + Tb)[np.asarray(e2["gated"])] > 1.0 + G.CONS_MAX_LIMIT)
        assert 1.24 < (Rb + Tb)[np.asarray(e2["gated"])].min() and (Rb + Tb).max() < 1.5
        assert ratios["A"] > 20


@pytest.mark.parametrize("name", sorted(L.MEEP_FALSIFIERS))
def test_meep_falsifiers_exceed_the_e4_windows_analytically(name):
    f, dt = _rfx_bins()
    p = L.ARMS[L.MEEP_FALSIFIER_ARM]["params"]
    good = de.to_meep("conductive", p, a_m=L.MEEP_A_M)
    bad = L.apply_meep_falsifier(good, name)
    dt_meep = L.MEEP_COURANT * (L.MEEP_A_M / L.MEEP_PRIMARY_RESOLUTION) / de.C0
    eps_bad = np.conj(de.eps_meep_convention(f, bad))
    Rm, Tm = de.tmm_slab_rt(f, eps_bad, L.D_SLAB_M)
    R, T, _ = L.analytic_rta(f, p)
    e2 = L.evaluate_e2(f, R, T, p, dt)
    meep_doc = {"freqs_hz": f.tolist(), "R": Rm.tolist(), "T": Tm.tolist(), "dt_meep_s": dt_meep,
                "meep_params": bad, "precheck": {"passed": False}, "eps_averaging": False}
    e4 = L.evaluate_e4(e2, meep_doc)
    assert not e4["e4_ok"], name
    assert not e4["gates"]["G4_mean_R"] and not e4["gates"]["G5_mean_R"]
    assert e4["mean_dR_meep_tmm_gated"] / e4["mean_window4_R"] > 10
    # and the control: the RIGHT mapping passes the same E4 gates, R, T and A.
    eps_ok = np.conj(de.eps_meep_convention(f, good))
    Ro, To = de.tmm_slab_rt(f, eps_ok, L.D_SLAB_M)
    e4_ok = L.evaluate_e4(e2, {**meep_doc, "R": Ro.tolist(), "T": To.tolist(), "meep_params": good,
                               "precheck": {"passed": True}})
    assert e4_ok["e4_ok"] and all(e4_ok["gates"][k] for k in ("G4_A", "G4_mean_A", "G5_A", "G5_mean_A"))


# ---------------------------------------------------------------------------
# 2. Artifact replay (skips until the VESSL run lands)
# ---------------------------------------------------------------------------

def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _replay_e2(arm_doc: dict) -> dict:
    return L.evaluate_e2(arm_doc["freqs_hz"], arm_doc["R_rfx"], arm_doc["T_rfx"],
                         arm_doc["params"], arm_doc["dt_s"], tail=arm_doc["tail"])


def _baseline() -> dict:
    p = _RESULTS / "rfx.json"
    if not p.is_file():
        pytest.skip(f"cv23 baseline artifact absent: {p.relative_to(_REPO)} (VESSL run pending)")
    doc = _read(p)
    assert doc["schema"] == L.SCHEMA
    if doc.get("smoke"):
        pytest.skip("baseline artifact is a --smoke run, not evidence")
    return doc


def test_baseline_artifact_replays_and_passes_e2_on_all_arms():
    doc = _baseline()
    assert set(doc["arms"]) == set(L.ARM_ORDER), "all three arms must be present"
    assert doc["falsifier"] is None
    for arm, ad in doc["arms"].items():
        assert ad["model"] == "conductive"
        assert ad["params"] == pytest.approx(L.ARMS[arm]["params"])
        assert ad["params_run"] == pytest.approx(L.ARMS[arm]["params"])
        assert ad["materials_path"] == L.MATERIALS_PATH[arm]
        if ad["materials_path"] == "api":
            assert ad["materials"]["api_equals_direct"] is True and ad["materials"]["api_no_pec"] is True
        assert ad["run"]["recipe"] == G.RECIPE_R3
        rec = ad["run"]["record"]
        want = L.derive_record_length(ad["params"], ad["dt_s"], nx_interior=ad["run"]["nx_interior"])
        assert rec["n_steps_min"] == want["n_steps"], arm
        assert ad["run"]["n_steps"] == rec["n_steps"] == rec["n_steps_min"] + rec["extensions"] * G.RECORD_EXTEND_STEPS
        assert rec["n_steps"] <= rec["t_safe_cpml_steps"]
        assert ad["run"]["nx_interior"] >= G.NX_INTERIOR_R3
        assert ad["tail"]["limit"] == G.SETTLING_LIMIT and ad["tail"]["ok"], (arm, ad["tail"])
        assert ad["tail"]["fit_start_step"] == rec["n_pulse_end"] + G.TAIL_WINDOW
        refit = G.refit_tail(ad["tail"], ad["dt_s"], ad["run"]["n_steps"], rec["n_pulse_end"])
        rate, nb = refit["fitted_rate_scat_refl_1_s"], refit["fitted_rate_blocks"]
        assert nb >= 3 and rate == pytest.approx(ad["tail"]["fitted_rate_scat_refl_1_s"])
        assert np.isfinite(rate) and rate > 0
        print(f"cv23-summary rfx {arm}: n_steps_min {rec['n_steps_min']} reached {rec['n_steps']} "
              f"(+{rec['extensions']} ext); tail scat/trans {ad['tail']['scat_refl_rel']:.2e}/"
              f"{ad['tail']['total_trans_rel']:.2e}; fitted rate {ad['tail']['fitted_rate_scat_refl_1_s']:.3e}/"
              f"{ad['tail']['fitted_rate_total_trans_1_s']:.3e} vs derived {rec['rate_ring_1_s']:.3e}; "
              f"mean|dR|/|dT|/|dA| {ad['mean_dR_gated']:.4f}/{ad['mean_dT_gated']:.4f}/{ad['mean_dA_gated']:.4f}; "
              f"A_tight_ok {ad['A_tight_ok']}")
        assert ad["band_inc_ok"]
        re = _replay_e2(ad)
        assert re["gates"] == {k: v for k, v in ad["gates"].items() if k in re["gates"]}, arm
        assert re["e2_ok"], (arm, re["gates"], re["max_dR_gated"], re["max_dT_gated"], re["max_dA_gated"])
        assert abs(re["mean_dA_gated"] - ad["mean_dA_gated"]) < 1e-12
        assert re["A_tight_ok"] == ad["A_tight_ok"]
        assert re["n_bins_gated"] >= 200
    assert doc["verdict"]["rfx_self_ok"]


def test_baseline_artifact_e4_against_the_committed_meep_jsons():
    doc = _baseline()
    missing = [arm for arm in L.ARM_ORDER if not (_RESULTS / L.meep_json_name(arm)).is_file()]
    if missing:
        pytest.skip(f"Meep JSON absent for {missing}; E4 not replayable (exit 2 class)")
    for arm, ad in doc["arms"].items():
        md = _read(_RESULTS / L.meep_json_name(arm))
        assert md["falsifier"] is None and md["arm"] == arm
        assert md["resolution"] == L.MEEP_PRIMARY_RESOLUTION == 40
        assert md["eps_averaging"] is False and md["run"]["finite"]
        assert md["precheck"]["passed"] and md["precheck"]["max_rel_err"] < 1e-9
        assert md["meep_params"]["kind"] == "D_conductivity"
        e4 = L.evaluate_e4(_replay_e2(ad), md)
        assert e4["e4_ok"], (arm, e4["gates"], e4["max_dA_rfx_meep_gated"])
        assert ad["meep"]["present"] and e4["gates"] == ad["meep"]["gates"]
        print(f"cv23-summary meep {arm}: Meep-vs-TMM mean R/T/A {e4['mean_dR_meep_tmm_gated']:.4f}/"
              f"{e4['mean_dT_meep_tmm_gated']:.4f}/{e4['mean_dA_meep_tmm_gated']:.4f}; rfx-vs-Meep "
              f"{e4['mean_dR_rfx_meep_gated']:.4f}/{e4['mean_dT_rfx_meep_gated']:.4f}/{e4['mean_dA_rfx_meep_gated']:.4f}")
    assert doc["verdict"]["exit_code"] == 0


@pytest.mark.parametrize("name", sorted(L.FALSIFIERS))
def test_rfx_falsifier_artifacts_fail_for_the_declared_reason(name):
    p = _RESULTS / L.rfx_json_name(name)
    if not p.is_file():
        pytest.skip(f"falsifier artifact absent: {p.name}")
    doc = _read(p)
    assert doc["falsifier"] == name and not doc.get("smoke")
    arm, bad = L.apply_falsifier(name)
    ad = doc["arms"][arm]
    assert ad["params_run"] == pytest.approx(bad)
    assert ad["params"] == pytest.approx(L.ARMS[arm]["params"])
    re = L.evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], L.ARMS[arm]["params"], ad["dt_s"], tail=ad["tail"])
    assert re["gates"] == {k: v for k, v in ad["gates"].items() if k in re["gates"]}
    assert not re["e2_ok"]
    assert not (re["gates"]["G2_R"] and re["gates"]["G2_T"] and re["gates"]["G2_A"]), "must fail on a band-mean"
    if name == L.PASSIVITY_FALSIFIER:
        assert not re["gates"]["G3_passivity"], "the gain arm must break passivity"
    assert doc["verdict"]["exit_code"] == 1


@pytest.mark.parametrize("name", sorted(L.MEEP_FALSIFIERS))
def test_meep_falsifier_artifacts_fail_e4_against_the_baseline(name):
    mp_path = _RESULTS / L.meep_json_name(L.MEEP_FALSIFIER_ARM, name)
    if not mp_path.is_file():
        pytest.skip(f"Meep falsifier artifact absent: {mp_path.name}")
    doc = _baseline()
    md = _read(mp_path)
    assert md["falsifier"] == name
    assert md["precheck"]["passed"] is False, "the 1e-9 pre-check must have caught the wrong scaling"
    e4 = L.evaluate_e4(_replay_e2(doc["arms"][L.MEEP_FALSIFIER_ARM]), md)
    assert not e4["e4_ok"], e4["gates"]
    assert not e4["gates"]["G4_mean_R"]


def test_meep_ladder_summary_reproduces_from_its_rungs():
    p = _RESULTS / "meep_ladder_summary.json"
    if not p.is_file():
        pytest.skip("meep_ladder_summary.json absent")
    summ = _read(p)
    doc = _baseline()
    fresh = L.meep_ladder_summary(str(_RESULTS), doc)
    for arm, v in summ["arms"].items():
        assert v["rungs"].keys() == fresh["arms"][arm]["rungs"].keys()
        for res, rung in v["rungs"].items():
            assert rung.get("finite"), (arm, res)
            assert rung["mean_dT_meep_tmm_gated"] == pytest.approx(
                fresh["arms"][arm]["rungs"][res]["mean_dT_meep_tmm_gated"], rel=1e-9)
        r40 = v["rungs"]["40"]
        assert r40["mean_dR_meep_tmm_gated"] <= L.W_MEAN_R and r40["mean_dT_meep_tmm_gated"] <= L.W_MEAN_T
    print("cv23-summary meep ladder:", {a: {k: round(o, 2) for k, o in v["orders"].items()} for a, v in summ["arms"].items()})


# ---------------------------------------------------------------------------
# 3. Round 2 (note section 12): the derived Yee-lattice term and Meep's
#    thickness excess. Artifact-free predictions are LOCKED here; the replays
#    print measured-vs-predicted and assert structure only (skip until the r2
#    artifacts land). Selected on the pod with ``-k r2``.
# ---------------------------------------------------------------------------

# Note section 12 predictions (mean over the gated bins of |lattice - TMM|),
# rfx dt at Courant 0.700, per (arm, dx_div): (R, T, A).
_R2_LATTICE_PRED = {
    ("tand0p1", 1): (0.00392, 0.00567, 0.00194), ("tand0p1", 2): (0.00096, 0.00139, 0.00048), ("tand0p1", 4): (0.00024, 0.00035, 0.00012),
    ("tand1", 1): (0.00509, 0.00176, 0.00334), ("tand1", 2): (0.00126, 0.00044, 0.00081), ("tand1", 4): (0.00031, 0.00011, 0.00020),
    ("tand3", 1): (0.01256, 0.00011, 0.01244), ("tand3", 2): (0.00311, 0.00003, 0.00308), ("tand3", 4): (0.00078, 0.00001, 0.00077),
}
# Meep thickness-excess hypothesis TMM(d + a/res): mean |dR| per (arm, res).
_R2_MEEP_THICK_PRED = {("tand0p1", 10): 0.0593, ("tand0p1", 20): 0.0307, ("tand0p1", 40): 0.0155, ("tand0p1", 80): 0.0078,
                       ("tand1", 10): 0.0100, ("tand1", 20): 0.0057, ("tand1", 40): 0.0030, ("tand1", 80): 0.0015}
_R2_LATTICE_RESIDUAL_BAR = 3e-4     # 10x the r1 |rfx - lattice| mean (3e-5) -> "lattice confirmed"


def test_r2_lattice_term_is_derived_and_predicts_the_ladder():
    """The exact 1-D Yee lattice of the staircase slab converges to the TMM
    second-order and, at the rig's dx/dt, gives the section-12 numbers; the
    sqrt-free multilayer TMM reduces to the slab."""
    f, dt = _rfx_bins()
    g = G.gated_mask(f)
    p3 = L.ARMS["tand3"]["params"]
    R, T, _ = L.analytic_rta(f[g], p3)
    errs = []
    for dx in (1e-3, 2.5e-4, 6.25e-5):
        Rl, Tl = de.yee_lattice_slab_rt(f[g], 4.0, p3["sigma"], L.D_SLAB_M, dx, 0.7 * dx / de.C0)
        errs.append(np.abs(Rl - R).mean())
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5      # second order per x4 in dx
    Rm, Tm = de.tmm_layers_rt(f[g], [(de.eps_analytic(f[g], "conductive", p3), L.D_SLAB_M)])
    assert np.abs(Rm - R).max() < 1e-12 and np.abs(Tm - T).max() < 1e-12
    for (arm, K), want in _R2_LATTICE_PRED.items():
        p = L.ARMS[arm]["params"]
        wR, wT, wA = L.lattice_window(f[g], p, G.DX_M / K, dt / K)
        got = (wR.mean(), wT.mean(), wA.mean())
        for gv, wv in zip(got, want):
            assert abs(gv - wv) <= max(2e-5, 0.03 * wv), (arm, K, got, want)
    for (arm, res), want in _R2_MEEP_THICK_PRED.items():
        p = L.ARMS[arm]["params"]
        R0, _, _ = L.analytic_rta(f[g], p)
        Rt, _, _ = L.meep_thickness_excess_rta(f[g], p, res)
        assert abs(np.abs(Rt - R0).mean() - want) <= 2e-4, (arm, res)
    # the coordinator's two candidate mechanisms on tand3, a priori: both small
    Rp, _ = de.tmm_slab_rt(f[g], de.eps_analytic(f[g], "conductive", p3), L.D_SLAB_M + 0.5e-3)
    assert np.abs(Rp - R).mean() < 6e-4                              # +-dx/2 thickness: surface impedance is thickness-blind
    eh = {"eps_inf": 2.5, "sigma": p3["sigma"] / 2}
    Rh, _ = de.tmm_layers_rt(f[g], [(de.eps_analytic(f[g], "conductive", eh), 1e-3),
                                    (de.eps_analytic(f[g], "conductive", p3), 8e-3),
                                    (de.eps_analytic(f[g], "conductive", eh), 1e-3)])
    assert np.mean(Rh - R) < -0.015                                  # half-weighted cells: wrong sign, 2x too big


_R2_RFX_TAGS = {arm: [f"{arm}_dx2", f"{arm}_dx4"] for arm in L.ARM_ORDER}


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_r2_rfx_dx_ladder_against_the_lattice_prediction(arm):
    base = _baseline()["arms"][arm]
    f0 = np.asarray(base["freqs_hz"]); g0 = np.asarray(base["gated"])
    wR0, wT0, wA0 = L.lattice_window(f0[g0], base["params"], base["run"]["dx_m"], base["dt_s"])
    res0 = np.mean(np.abs(np.asarray(base["R_rfx"])[g0] - np.asarray(base["R_tmm"])[g0] - (
        L.lattice_rta(f0[g0], base["params"], base["run"]["dx_m"], base["dt_s"])[0] - np.asarray(base["R_tmm"])[g0])))
    lines = [f"r2-summary rfx {arm} dx: mean|dR| {base['mean_dR_gated']:.5f} (lattice pred {wR0.mean():.5f}, "
             f"|rfx - lattice| {res0:.2e}); |dT| {base['mean_dT_gated']:.5f} (pred {wT0.mean():.5f})"]
    for tag in _R2_RFX_TAGS[arm]:
        p = _RESULTS / f"rfx__{tag}.json"
        if not p.is_file():
            lines.append(f"r2-summary rfx {tag}: ABSENT")
            continue
        d = _read(p)["arms"][arm]
        assert d["params"] == pytest.approx(L.ARMS[arm]["params"]) and d["params_run"] == pytest.approx(L.ARMS[arm]["params"])
        K = d["run"]["dx_div"]
        assert K == int(tag[-1]) and d["run"]["dx_m"] == pytest.approx(G.DX_M / K)
        assert d["tail"]["ok"], (tag, d["tail"]["scat_refl_rel"], d["tail"]["total_trans_rel"])
        re = _replay_e2(d)
        assert re["gates"] == {k: v for k, v in d["gates"].items() if k in re["gates"]}
        f = np.asarray(d["freqs_hz"]); g = np.asarray(d["gated"])
        Rl, Tl, Al = L.lattice_rta(f[g], d["params"], d["run"]["dx_m"], d["dt_s"])
        Rx = np.asarray(d["R_rfx"])[g]; Tx = np.asarray(d["T_rfx"])[g]
        resid_R = np.mean(np.abs(Rx - Rl)); resid_T = np.mean(np.abs(Tx - Tl))
        ratio = d["mean_dR_gated"] / base["mean_dR_gated"] if base["mean_dR_gated"] > 0 else float("nan")
        pred = _R2_LATTICE_PRED[(arm, K)]
        reading = ("lattice-confirmed" if max(resid_R, resid_T) <= _R2_LATTICE_RESIDUAL_BAR else
                   "first-order" if (K == 2 and 0.4 <= ratio <= 0.6) or (K == 4 and 0.2 <= ratio <= 0.35) else
                   "no-fall" if ratio >= 0.7 else "unresolved")
        lines.append(f"r2-summary rfx {tag}: n_steps {d['run']['n_steps']} mean|dR| {d['mean_dR_gated']:.5f} "
                     f"(x{ratio:.3f} of dx; lattice pred {pred[0]:.5f}) |dT| {d['mean_dT_gated']:.5f} (pred {pred[1]:.5f}) "
                     f"|dA| {d['mean_dA_gated']:.5f} (pred {pred[2]:.5f}); |rfx - lattice| R {resid_R:.2e} T {resid_T:.2e} "
                     f"-> {reading}; E2 {'PASS' if re['e2_ok'] else 'FAIL'}")
    print("\n".join(lines))


def test_r2_meep_res80_and_thin_block_against_the_predictions():
    doc = _baseline()
    lines = []
    for arm, tag, pred, label in (("tand0p1", "res80", _R2_MEEP_THICK_PRED[("tand0p1", 80)], "thickness excess d + a/80"),
                                  ("tand3", "res80", 0.0002, "lattice, second order"),
                                  ("tand0p1", "res40_thin1", 0.0003, "block drawn d - a/40: lattice level")):
        p = _RESULTS / f"meep_{arm}__{tag}.json"
        if not p.is_file():
            lines.append(f"r2-summary meep {arm}__{tag}: ABSENT")
            continue
        md = _read(p)
        assert md["run"]["finite"] and md["precheck"]["passed"] and md["eps_averaging"] is False
        if tag == "res40_thin1":
            assert md["thickness_offset_cells"] == -1 and md["d_slab_m"] == pytest.approx(L.D_SLAB_M - L.MEEP_A_M / 40)
        else:
            assert md["resolution"] == 80 and md.get("thickness_offset_cells", 0) == 0
        e4 = L.evaluate_e4(_replay_e2(doc["arms"][arm]), md)
        lines.append(f"r2-summary meep {arm}__{tag}: Meep-vs-TMM mean|dR| {e4['mean_dR_meep_tmm_gated']:.5f} "
                     f"(pred {pred:.4f}, {label}) |dT| {e4['mean_dT_meep_tmm_gated']:.5f} |dA| {e4['mean_dA_meep_tmm_gated']:.5f}; "
                     f"G4_mean_R {'pass' if e4['gates']['G4_mean_R'] else 'FAIL'}")
    print("\n".join(lines))


def test_r2_meep_ladder_summary_carries_the_res80_rung():
    p = _RESULTS / "meep_ladder_summary.json"
    if not p.is_file():
        pytest.skip("meep_ladder_summary.json absent")
    summ = _read(p)
    if 80 not in summ.get("resolutions", []):
        pytest.skip("ladder summary predates the res-80 rung")
    fresh = L.meep_ladder_summary(str(_RESULTS), _baseline())
    for arm, v in summ["arms"].items():
        assert v["rungs"].keys() == fresh["arms"][arm]["rungs"].keys()
        for res, rung in v["rungs"].items():
            if rung.get("finite"):
                assert rung["mean_dR_meep_tmm_gated"] == pytest.approx(fresh["arms"][arm]["rungs"][res]["mean_dR_meep_tmm_gated"], rel=1e-9)
    print("r2-summary meep ladder orders:", {a: {k: round(o, 2) for k, o in v["orders"].items()} for a, v in summ["arms"].items()})
