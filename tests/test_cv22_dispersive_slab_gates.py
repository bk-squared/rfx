"""cv22 dispersive slab -- gate replay and window-derivation witnesses.

Two kinds of test live here:

1. **Artifact-free** (always run): the windows are derived from the committed
   cv04 envelope by the repo's shared rule; the ADE discrete-time transfer
   functions used in the windows reproduce the LIVE ``init_debye`` /
   ``init_lorentz`` coefficient recurrences; every pre-declared falsifier
   exceeds its window analytically (the margins table of the note, §6,
   recomputed); the Meep wrong-convention falsifiers exceed the E4 windows.

2. **Artifact replay** (``pytest.skip`` while the VESSL artifacts are absent):
   ``validation/crossval/_22_dispersive_results/rfx.json`` is replayed
   through the same evaluators from its raw per-bin R/T and must reproduce
   its stored gate verdicts, all passing; each ``rfx__falsifier_<name>.json``
   must FAIL for the pre-declared reason; each
   ``meep_lorentz__falsifier_<name>.json`` must fail E4 against the baseline
   and carry ``precheck.passed == False``.

No FDTD runs here. Pre-declaration:
``docs/design_notes/20260902_cv22_dispersive_slab_predeclaration.md``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER, gate_from_envelope

_REPO = Path(__file__).resolve().parents[1]
_RESULTS = _REPO / "validation/crossval/_22_dispersive_results"
_GOLDEN_CV04 = _REPO / "tests/fixtures/golden_workflows/multilayer_fresnel.json"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("cv22_dispersive_eps", "validation/crossval/comparators/dispersive_eps.py")
G = _load("cv22_dispersive_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")


def _rig_dt() -> float:
    from rfx.grid import Grid
    grid = Grid(freq_max=20e9, domain=(G.NX_INTERIOR * G.DX_M, 0.004, G.DX_M),
                dx=G.DX_M, cpml_layers=G.N_CPML, mode="2d_tmz")
    return float(grid.dt)


def _r3_records(dt):
    return {arm: G.derive_record_length(G.ARMS[arm]["model"], G.ARMS[arm]["params"], dt)
            for arm in G.ARM_ORDER}


def _rfx_bins():
    """The rfx rFFT bin grid of the r3 recipe (Lorentz record 1228 steps ->
    nfft 16384; all three arms share it, see test_r3_record_lengths_are_derived)."""
    dt = _rig_dt()
    n_steps = _r3_records(dt)["lorentz"]["n_steps"]
    nfft = int(2 ** np.ceil(np.log2(n_steps)) * G.NFFT_OVERSAMPLE)
    f = np.fft.rfftfreq(nfft, d=dt)
    f = f[(f > G.MASK_F_LO_HZ) & (f < G.MASK_F_HI_HZ)]
    return f, dt


# ---------------------------------------------------------------------------
# 1. Artifact-free witnesses
# ---------------------------------------------------------------------------

def test_windows_are_derived_from_the_committed_cv04_envelope():
    golden = json.loads(_GOLDEN_CV04.read_text())
    baseline = {m["id"]: m["observed_baseline"] for m in golden["expected_metrics"]}
    assert G.CV04_ENVELOPE["mean_dR"] == baseline["mean_reflectance_error"]
    assert G.CV04_ENVELOPE["mean_dT"] == baseline["mean_transmittance_error"]
    # The per-bin number is a code comment in cv04 (no artifact carries it).
    cv04_src = (_REPO / "validation/crossval/04_multilayer_fresnel.py").read_text()
    assert "max|R+T-1| = 0.0487" in cv04_src
    assert G.CV04_ENVELOPE["per_bin_max_RT_closure"] == 0.0487
    assert ENVELOPE_GATE_MULTIPLIER == 1.5
    assert G.W_BIN == gate_from_envelope(0.0487, quantum=1000) == 0.074
    assert G.W_MEAN_R == gate_from_envelope(0.0066, quantum=1000) == 0.010
    assert G.W_MEAN_T == gate_from_envelope(0.011, quantum=1000) == 0.017
    # cv04 witness constants carried unchanged.
    assert (G.TAIL_WINDOW, G.TAIL_PURITY_LIMIT, G.TAIL_LIMIT, G.CONS_MAX_LIMIT) == (50, 1e-3, 0.10, 0.06)


@pytest.mark.parametrize("arm", ["debye", "lorentz", "drude"])
def test_arms_have_strong_dispersion_in_the_gated_band(arm):
    f, _ = _rfx_bins()
    g = G.gated_mask(f)
    model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
    eps = de.eps_analytic(f[g], model, params)
    mag = np.abs(eps)
    assert (mag.max() - mag.min()) / mag.max() >= 0.30
    tand = -eps.imag / eps.real
    assert np.any((tand >= 0.1) & (tand <= 1.0)) or (tand.min() <= 1.0 <= tand.max())
    assert np.all(eps.imag < 0)  # passive in the rfx convention


def test_r3_record_lengths_are_derived_from_the_slab_ringdown():
    """§13 recipe: n_steps_min = n_pulse_end + max_f ln(100 w(f))/rate(f)/dt +
    TAIL_WINDOW over the incident ring band, per arm, inside the CPML
    round-trip gate of the nx 1000 rig; all three arms land on the same nfft
    so the gated bin grid is shared."""
    dt = _rig_dt()
    recs = _r3_records(dt)
    # §13: incident-weighted ring band (w >= 0.5 -> 1.13-15 GHz). Debye's slowest
    # component is the 1.4 GHz etalon (w 0.62, 1.18e10/s), not the 4 GHz one
    # r3 used (1.84e10/s, 1066 steps, witness -36 dB); Lorentz/Drude unchanged.
    assert {a: r["n_steps"] for a, r in recs.items()} == {"debye": 1108, "lorentz": 1228, "drude": 1168}
    f_lo, f_hi = G.ring_band_hz()
    assert 1.1e9 < f_lo < 1.2e9 and f_hi == 15e9
    for arm, r in recs.items():
        assert r["cpml_gate_ok"] and r["t_safe_cpml_steps"] == 1262
        assert r["n_pulse_end"] == 908 and r["settling_limit"] == 1e-2
        assert r["nx_interior"] == G.NX_INTERIOR_R3 == 1000
        assert r["ring_band_hz"] == [f_lo, f_hi]
    assert recs["debye"]["rate_ring_1_s"] < recs["debye"]["rate_material_1_s"]
    assert 1.4e9 < recs["debye"]["f_ring_hz"] < 1.5e9
    for arm in ("lorentz", "drude"):
        assert recs[arm]["rate_ring_1_s"] == recs[arm]["rate_material_1_s"]
    nffts = {int(2 ** np.ceil(np.log2(r["n_steps"])) * G.NFFT_OVERSAMPLE) for r in recs.values()}
    assert nffts == {16384}
    # cv04's own 719-step gate truncates every arm's ring-down (the r2 finding).
    assert all(r["n_steps"] > 719 for r in recs.values())


def test_ade_stability_constraints_of_the_note():
    dt = _rig_dt()
    tau = G.ARMS["debye"]["params"]["tau"]
    alpha = (2 * tau - dt) / (2 * tau + dt)
    assert abs(alpha) < 1
    lp = G.ARMS["lorentz"]["params"]
    w0 = 2 * np.pi * lp["f0"]
    assert w0 * dt < 2.0 and lp["delta"] >= 0
    dp = G.ARMS["drude"]["params"]
    assert dp["gamma"] / 2 * dt < 2.0
    # Meep-side mapped Debye pole.
    dt_meep = G.MEEP_COURANT * G.DX_M / de.C0
    assert 2 * np.pi * G.DEBYE_MEEP_MAP_FN_HZ * dt_meep < 2.0


@pytest.mark.parametrize("arm", ["debye", "lorentz", "drude"])
def test_discrete_transfer_function_matches_live_ade_recurrence(arm):
    """Drive the P recurrence built from the LIVE rfx coefficient arrays with
    a sinusoidal E and recover chi(omega); it must equal the closed-form
    eps_numerical_ade - eps_inf used in the windows."""
    from rfx.core.yee import EPS_0, init_materials
    from rfx.materials.debye import DebyePole, init_debye
    from rfx.materials.lorentz import drude_pole, init_lorentz, lorentz_pole

    dt = _rig_dt()
    model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
    args = de.rfx_pole_args(model, params)
    mats = init_materials((1, 1, 1))
    mats = mats._replace(eps_r=mats.eps_r * params["eps_inf"])
    if model == "debye":
        coeffs, _ = init_debye([DebyePole(**args)], mats, dt)
        alpha = float(coeffs.alpha[0, 0, 0, 0]); beta = float(coeffs.beta[0, 0, 0, 0])
        assert abs(alpha - (2 * params["tau"] - dt) / (2 * params["tau"] + dt)) < 1e-6
    else:
        pole = lorentz_pole(**args) if model == "lorentz" else drude_pole(**args)
        coeffs, _ = init_lorentz([pole], mats, dt)
        a = float(coeffs.a[0, 0, 0, 0]); b = float(coeffs.b[0, 0, 0, 0]); c = float(coeffs.c[0, 0, 0, 0])

    for f in (4.5e9, 7.0e9, 9.5e9):
        w = 2 * np.pi * f
        n_per = int(round(1.0 / (f * dt)))
        n_tr, n_meas = 40 * n_per, 20 * n_per   # settle, then project
        N = n_tr + n_meas
        t = np.arange(N + 1) * dt
        E = np.cos(w * t)
        P = np.zeros(N + 1)
        if model == "debye":
            for n in range(N):
                P[n + 1] = alpha * P[n] + beta * (E[n + 1] + E[n])
        else:
            for n in range(1, N):
                P[n + 1] = a * P[n] + b * P[n - 1] + c * E[n]
        # complex amplitude of P at omega on the steady-state window, by a
        # least-squares fit to [cos, sin] (exact for a pure sinusoid; no
        # integer-period assumption). E = cos(wt) has amplitude 1 + 0j.
        # The constant column absorbs the Drude recurrence's non-decaying DC
        # mode (omega_0 = 0 puts a root at z = 1: free carriers keep a
        # constant polarization offset from the transient, which is physical).
        seg = slice(n_tr, n_tr + n_meas)
        basis = np.column_stack([np.cos(w * t[seg]), np.sin(w * t[seg]), np.ones(n_meas)])
        (pa, pb, _dc), *_ = np.linalg.lstsq(basis, P[seg], rcond=None)
        chi_meas = (pa - 1j * pb) / EPS_0          # P = Re[(A - jB) e^{jwt}]
        chi_pred = de.eps_numerical_ade(f, model, params, dt) - params["eps_inf"]
        chi_cont = de.eps_analytic(f, model, params) - params["eps_inf"]
        # 2e-5: the live coefficients are float32 (x64 off), so ~1e-6 is the floor.
        assert abs(chi_meas - chi_pred) / abs(chi_pred) < 2e-5, (arm, f, chi_meas, chi_pred)
        # and the discrete form is what makes it agree: the continuous form
        # is measurably further away at the band top than the discrete one.
        assert abs(chi_meas - chi_pred) <= abs(chi_meas - chi_cont) + 1e-12


def test_ade_window_term_is_named_and_small_at_this_dt():
    f, dt = _rfx_bins()
    g = G.gated_mask(f)
    for arm in G.ARM_ORDER:
        model, params = G.ARMS[arm]["model"], G.ARMS[arm]["params"]
        wR, wT, _, _ = G.ade_window(f, model, params, dt)
        assert 0 < wR[g].max() < 2e-3 and 0 < wT[g].max() < 2e-3
        assert wR[g].max() < G.W_BIN / 10


@pytest.mark.parametrize("name", sorted(G.FALSIFIERS))
def test_rfx_falsifiers_exceed_the_windows_analytically(name):
    """§6 margins: every F1/F2 defect must fail G2 (band-mean) on R or T and
    the named per-bin failures must exist, before any FDTD is run."""
    f, dt = _rfx_bins()
    arm, model, bad = G.apply_falsifier(name)
    good = G.ARMS[arm]["params"]
    R, T = G.analytic_rt(f, model, good)
    Rb, Tb = G.analytic_rt(f, model, bad)
    e2 = G.evaluate_e2(f, Rb, Tb, model, good, dt)   # defective "measurement" vs the true oracle
    assert not (e2["gates"]["G2_R"] and e2["gates"]["G2_T"]), name
    # margin >= 2x on at least one band-mean window (coin-toss guard)
    ratio = max(e2["mean_dR_gated"] / e2["mean_window_R"], e2["mean_dT_gated"] / e2["mean_window_T"])
    assert ratio >= 2.0, (name, ratio)
    if name != "debye_tau_x2":
        assert e2["n_bins_R_over_window"] + e2["n_bins_T_over_window"] > 0
    else:
        assert e2["n_bins_T_over_window"] >= 60   # 72 in the note; T fails per-bin above ~7 GHz


def test_debye_tau_x1p3_would_be_a_coin_toss_and_is_not_a_falsifier():
    f, dt = _rfx_bins()
    p = G.ARMS["debye"]["params"]
    bad = {**p, "tau": 1.3 * p["tau"]}
    Rb, Tb = G.analytic_rt(f, "debye", bad)
    e2 = G.evaluate_e2(f, Rb, Tb, "debye", p, dt)
    assert e2["mean_dR_gated"] / e2["mean_window_R"] < 1.5
    assert "debye_tau_x1p3" not in G.FALSIFIERS


@pytest.mark.parametrize("name", sorted(G.MEEP_FALSIFIERS))
def test_meep_falsifiers_exceed_the_e4_windows_analytically(name):
    f, dt = _rfx_bins()
    lp = G.ARMS["lorentz"]["params"]
    good = de.to_meep("lorentz", lp, a_m=G.MEEP_A_M)
    bad = G.apply_meep_falsifier(good, name)
    dt_meep = G.MEEP_COURANT * G.DX_M / de.C0
    # A perfect Meep of the WRONG material, against a perfect rfx of the right one.
    eps_bad = np.conj(de.eps_meep_convention(f, bad))
    Rm, Tm = de.tmm_slab_rt(f, eps_bad, G.D_SLAB_M)
    R, T = G.analytic_rt(f, "lorentz", lp)
    e2 = G.evaluate_e2(f, R, T, "lorentz", lp, dt)
    meep_doc = {"freqs_hz": f.tolist(), "R": Rm.tolist(), "T": Tm.tolist(),
                "dt_meep_s": dt_meep, "meep_params": bad, "precheck": {"passed": False}}
    e4 = G.evaluate_e4(e2, meep_doc)
    assert not e4["e4_ok"], name
    assert not (e4["gates"]["G4_mean_R"] and e4["gates"]["G4_mean_T"]), name
    assert not (e4["gates"]["G5_mean_R"] and e4["gates"]["G5_mean_T"]), name
    # and the control: the RIGHT mapping passes the same E4 gates.
    eps_ok = np.conj(de.eps_meep_convention(f, good))
    Ro, To = de.tmm_slab_rt(f, eps_ok, G.D_SLAB_M)
    e4_ok = G.evaluate_e4(e2, {**meep_doc, "R": Ro.tolist(), "T": To.tolist(), "meep_params": good,
                               "precheck": {"passed": True}})
    assert e4_ok["e4_ok"]


# ---------------------------------------------------------------------------
# 2. Artifact replay (skips until the VESSL run lands)
# ---------------------------------------------------------------------------

def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _replay_e2(arm_doc: dict) -> dict:
    return G.evaluate_e2(arm_doc["freqs_hz"], arm_doc["R_rfx"], arm_doc["T_rfx"],
                         arm_doc["model"], arm_doc["params"], arm_doc["dt_s"], tail=arm_doc["tail"])


def _baseline() -> dict:
    p = _RESULTS / "rfx.json"
    if not p.is_file():
        pytest.skip(f"cv22 baseline artifact absent: {p.relative_to(_REPO)} (VESSL run pending)")
    doc = _read(p)
    assert doc["schema"] == "cv22-dispersive-slab/v1"
    if doc.get("smoke"):
        pytest.skip("baseline artifact is a --smoke run, not evidence")
    return doc


def test_baseline_artifact_replays_and_passes_e2_on_all_arms():
    doc = _baseline()
    assert set(doc["arms"]) == set(G.ARM_ORDER), "all three arms must be present"
    assert doc["falsifier"] is None
    for arm, ad in doc["arms"].items():
        assert ad["model"] == G.ARMS[arm]["model"]
        assert ad["params"] == pytest.approx(G.ARMS[arm]["params"])
        assert ad["params_run"] == pytest.approx(G.ARMS[arm]["params"])
        assert ad["run"]["recipe"] == G.RECIPE_R3, "the baseline must be the r3 (derived-record) recipe"
        rec = ad["run"]["record"]
        want = G.derive_record_length(ad["model"], ad["params"], ad["dt_s"], nx_interior=ad["run"]["nx_interior"])
        assert rec["n_steps_min"] == want["n_steps"], arm
        assert ad["run"]["n_steps"] == rec["n_steps"] == rec["n_steps_min"] + rec["extensions"] * G.RECORD_EXTEND_STEPS
        assert rec["n_steps"] <= rec["t_safe_cpml_steps"]
        assert ad["run"]["nx_interior"] >= G.NX_INTERIOR_R3
        assert ad["tail"]["limit"] == G.SETTLING_LIMIT and ad["tail"]["ok"], (arm, ad["tail"])
        # the stored envelope refits to the recorded rate, and the rate is a
        # physical decay (positive, finite)
        assert ad["tail"]["fit_start_step"] == rec["n_pulse_end"] + G.TAIL_WINDOW, "the fit must start after the pulse"
        refit = G.refit_tail(ad["tail"], ad["dt_s"], ad["run"]["n_steps"], rec["n_pulse_end"])
        rate, nb = refit["fitted_rate_scat_refl_1_s"], refit["fitted_rate_blocks"]
        assert nb >= 3 and rate == pytest.approx(ad["tail"]["fitted_rate_scat_refl_1_s"])
        assert refit["fitted_rate_total_trans_1_s"] == pytest.approx(ad["tail"]["fitted_rate_total_trans_1_s"])
        assert np.isfinite(rate) and rate > 0
        print(f"r4-summary rfx {arm}: n_steps_min {rec['n_steps_min']} reached {rec['n_steps']} "
              f"(+{rec['extensions']} ext, box grows {len(rec.get('nx_grows', []))}); tail scat/trans "
              f"{ad['tail']['scat_refl_rel']:.2e}/{ad['tail']['total_trans_rel']:.2e}; fitted rate scat/trans "
              f"{ad['tail']['fitted_rate_scat_refl_1_s']:.3e}/{ad['tail']['fitted_rate_total_trans_1_s']:.3e} /s "
              f"vs derived ring rate {rec['rate_ring_1_s']:.3e} (component {rec['f_ring_hz']/1e9:.2f} GHz)")
        assert ad["band_inc_ok"]
        re = _replay_e2(ad)
        assert re["gates"] == {k: v for k, v in ad["gates"].items() if k in re["gates"]}, arm
        assert re["e2_ok"], (arm, re["gates"], re["max_dR_gated"], re["max_dT_gated"])
        assert abs(re["mean_dR_gated"] - ad["mean_dR_gated"]) < 1e-12
        assert re["n_bins_gated"] >= 100
    assert doc["verdict"]["rfx_self_ok"]


def test_baseline_artifact_e4_against_the_committed_meep_jsons():
    doc = _baseline()
    missing = [arm for arm in G.ARM_ORDER if not (_RESULTS / G.meep_json_name(arm)).is_file()]
    if missing:
        pytest.skip(f"Meep JSON absent for {missing}; E4 not replayable (exit 2 class)")
    for arm, ad in doc["arms"].items():
        md = _read(_RESULTS / G.meep_json_name(arm))
        assert md["falsifier"] is None and md["arm"] == arm
        assert md["resolution"] == G.MEEP_PRIMARY_RESOLUTION == 40, "§12: the converged Meep reference"
        assert md["run"]["finite"]
        assert md["precheck"]["passed"], (arm, md["precheck"]["max_rel_err"])
        assert md["precheck"]["max_rel_err"] < 1e-9
        e4 = G.evaluate_e4(_replay_e2(ad), md)
        assert e4["e4_ok"], (arm, e4["gates"], e4["max_dR_rfx_meep_gated"], e4["max_dT_rfx_meep_gated"])
        assert e4["gates"]["precheck_passed"]
        # the committed r4 artifact predates the precheck_passed gate key
        # (review finding 2); every stored gate must still replay identically
        assert ad["meep"]["present"]
        assert {k: v for k, v in e4["gates"].items() if k in ad["meep"]["gates"]} == ad["meep"]["gates"]
    assert doc["verdict"]["exit_code"] == 0


@pytest.mark.parametrize("name", sorted(G.FALSIFIERS))
def test_rfx_falsifier_artifacts_fail_for_the_declared_reason(name):
    p = _RESULTS / G.rfx_json_name(name)
    if not p.is_file():
        pytest.skip(f"falsifier artifact absent: {p.name}")
    doc = _read(p)
    assert doc["falsifier"] == name and not doc.get("smoke")
    arm = G.FALSIFIERS[name][0]
    ad = doc["arms"][arm]
    # The artifact records the DEFECTIVE params the FDTD ran with (params_run)
    # and the DECLARED material it was judged against (params); replay both.
    _, model, bad = G.apply_falsifier(name)
    assert ad["params_run"] == pytest.approx(bad)
    assert ad["params"] == pytest.approx(G.ARMS[arm]["params"])
    re = G.evaluate_e2(ad["freqs_hz"], ad["R_rfx"], ad["T_rfx"], model, G.ARMS[arm]["params"],
                       ad["dt_s"], tail=ad["tail"])
    assert re["gates"] == {k: v for k, v in ad["gates"].items() if k in re["gates"]}
    assert not re["e2_ok"]
    assert not (re["gates"]["G2_R"] and re["gates"]["G2_T"]), "must fail on the band-mean, not only a witness"
    assert doc["verdict"]["exit_code"] == 1


@pytest.mark.parametrize("name", sorted(G.MEEP_FALSIFIERS))
def test_meep_falsifier_artifacts_fail_e4_against_the_baseline(name):
    mp_path = _RESULTS / G.meep_json_name(G.MEEP_FALSIFIER_ARM, name)
    if not mp_path.is_file():
        pytest.skip(f"Meep falsifier artifact absent: {mp_path.name}")
    doc = _baseline()
    md = _read(mp_path)
    assert md["falsifier"] == name
    assert md["precheck"]["passed"] is False, "the 1e-9 pre-check must have caught the wrong convention"
    e4 = G.evaluate_e4(_replay_e2(doc["arms"][G.MEEP_FALSIFIER_ARM]), md)
    assert not e4["e4_ok"], e4["gates"]
    assert not (e4["gates"]["G4_mean_R"] and e4["gates"]["G4_mean_T"])


# ---------------------------------------------------------------------------
# 3. Round-2 replay (note section 11.2): scaling ratios are COMPUTED and
#    PRINTED against the pre-declared predictions; only structural
#    acceptances are asserted. Skips until the r2 artifacts exist.
# ---------------------------------------------------------------------------

_R2_RFX = {"lorentz": ["lorentz_dx2", "lorentz_dx4", "lorentz_nx1500"],
           "debye": ["debye_dx2", "debye_dx4"]}
_R2_MEEP_TAGS = ["res20", "res40", "decay1e-6", "src7"]


def _r2_rfx_doc(tag: str) -> dict:
    p = _RESULTS / f"rfx__{tag}.json"
    if not p.is_file():
        pytest.skip(f"r2 artifact absent: {p.name}")
    return _read(p)


@pytest.mark.parametrize("arm", sorted(_R2_RFX))
def test_r2_rfx_refinement_ratios_against_predeclared_predictions(arm):
    base = _baseline()["arms"][arm]
    m0 = base["mean_dR_gated"]
    lines = [f"r2-summary rfx {arm}: baseline mean|dR| {m0:.4f} mean|dT| {base['mean_dT_gated']:.4f} "
             f"(window {base['mean_window_R']:.4f}/{base['mean_window_T']:.4f})"]
    for tag in _R2_RFX[arm]:
        p = _RESULTS / f"rfx__{tag}.json"
        if not p.is_file():
            lines.append(f"r2-summary rfx {tag}: ABSENT")
            continue
        d = _read(p)["arms"][arm]
        assert d["params"] == pytest.approx(G.ARMS[arm]["params"])
        re = _replay_e2(d)                      # same evaluators, its own dt
        assert re["gates"] == {k: v for k, v in d["gates"].items() if k in re["gates"]}
        ratio = d["mean_dR_gated"] / m0 if m0 > 0 else float("nan")
        reading = ("first-order" if 0.20 <= ratio <= 0.35 and tag.endswith("dx4") else
                   "second-order" if ratio <= 0.12 and tag.endswith("dx4") else
                   "no-fall" if ratio >= 0.7 else "unresolved")
        lines.append(f"r2-summary rfx {tag}: dx_div={d['run']['dx_div']} nx={d['run']['nx_interior']} "
                     f"n_steps={d['run']['n_steps']} mean|dR| {d['mean_dR_gated']:.4f} (x{ratio:.2f}) "
                     f"mean|dT| {d['mean_dT_gated']:.4f} max|dR| {d['max_dR_gated']:.4f} "
                     f"E2 {'PASS' if re['e2_ok'] else 'FAIL'}"
                     + (f" -> {reading}" if tag.endswith("dx4") else ""))
    print("\n".join(lines))


@pytest.mark.parametrize("arm", ["lorentz", "drude"])
def test_r2_meep_diagnostics_against_predeclared_predictions(arm):
    base_path = _RESULTS / G.meep_json_name(arm)
    if not base_path.is_file():
        pytest.skip("baseline Meep leg absent")
    doc = _baseline()["arms"][arm]
    e2 = _replay_e2(doc)
    e4b = G.evaluate_e4(e2, _read(base_path))
    lines = [f"r2-summary meep {arm}: baseline (10 px/cm) Meep-vs-TMM mean|dR| {e4b['mean_dR_meep_tmm_gated']:.4f} "
             f"mean|dT| {e4b['mean_dT_meep_tmm_gated']:.4f}"]
    for tag in _R2_MEEP_TAGS:
        p = _RESULTS / f"meep_{arm}__{tag}.json"
        if not p.is_file():
            lines.append(f"r2-summary meep {arm}__{tag}: ABSENT")
            continue
        md = _read(p)
        assert md["run"]["finite"] and md["precheck"]["passed"], tag
        e4 = G.evaluate_e4(e2, md)
        rT = e4["mean_dT_meep_tmm_gated"] / max(e4b["mean_dT_meep_tmm_gated"], 1e-12)
        lines.append(f"r2-summary meep {arm}__{tag}: res={md['resolution']} decay={md['decay']} "
                     f"fcen={md['fcen_meep']:.3f} Meep-vs-TMM mean|dR| {e4['mean_dR_meep_tmm_gated']:.4f} "
                     f"mean|dT| {e4['mean_dT_meep_tmm_gated']:.4f} (T x{rT:.2f}) max|dT| {e4['max_dT_meep_tmm_gated']:.4f} "
                     f"G4_mean_T={'pass' if e4['gates']['G4_mean_T'] else 'FAIL'}")
    print("\n".join(lines))


def test_r2_meep_debye_res40_primary_is_the_predeclared_fix():
    """Section 11.2(c) acceptance: the primary Debye leg is the 40 px/cm one,
    ran finite, passed the 1e-9 pre-check, and its mapped pole sits where the
    note says (omega_n dt = 0.262, eps_num(Nyquist) > 1)."""
    p = _RESULTS / G.meep_json_name("debye")
    if not p.is_file():
        pytest.skip("Meep Debye leg absent")
    md = _read(p)
    assert md["resolution"] == 40 and md["run"]["finite"] and md["precheck"]["passed"]
    assert md["fn_debye_map_hz"] == pytest.approx(G.DEBYE_MEEP_MAP_FN_HZ)
    x = 2 * np.pi * md["fn_debye_map_hz"] * md["dt_meep_s"]
    assert x == pytest.approx(0.262, abs=0.005)
    dp = G.ARMS["debye"]["params"]
    eps_nyq = dp["eps_inf"] - dp["delta_eps"] * x ** 2 / (4 - x ** 2)
    assert eps_nyq > 1.0
    print(f"r2-summary meep debye primary: res 40, omega_n dt {x:.3f}, eps_num(Nyq) {eps_nyq:.3f}, "
          f"precheck max_rel_err {md['precheck']['max_rel_err']:.2e}")
    q = _RESULTS / "meep_debye__fn40.json"
    if q.is_file():
        m40 = _read(q)
        assert m40["run"]["finite"]
        e4 = G.evaluate_e4(_replay_e2(_baseline()["arms"]["debye"]), m40)
        print(f"r2-summary meep debye fn40 cross-check: mean|dT| vs TMM {e4['mean_dT_meep_tmm_gated']:.4f} "
              f"(W_map mean T carried {np.mean(np.asarray(e4['w_map_T'])[np.asarray(_baseline()['arms']['debye']['gated'])]):.2e})")


# ---------------------------------------------------------------------------
# 4. Meep ladder (measured-in-r2 evidence of Meep's first-order term; §12)
# ---------------------------------------------------------------------------

def test_meep_ladder_summary_locks_the_measured_first_order_term():
    p = _RESULTS / "meep_ladder_summary.json"
    if not p.is_file():
        pytest.skip("meep_ladder_summary.json absent")
    summ = _read(p)
    doc = _baseline()
    # The committed summary must be what the committed rungs replay to.
    fresh = G.meep_ladder_summary(str(_RESULTS), doc)
    for arm, v in summ["arms"].items():
        assert v["rungs"].keys() == fresh["arms"][arm]["rungs"].keys()
        for res, rung in v["rungs"].items():
            if rung.get("finite"):
                assert rung["mean_dT_meep_tmm_gated"] == pytest.approx(
                    fresh["arms"][arm]["rungs"][res]["mean_dT_meep_tmm_gated"], rel=1e-9)
    for arm in ("lorentz", "drude"):
        o = summ["arms"][arm]["orders"]
        # first order per doubling (r2 measured 0.94-1.06 across R/T and both doublings)
        for k, val in o.items():
            assert 0.8 <= val <= 1.3, (arm, k, val)
        r40 = summ["arms"][arm]["rungs"]["40"]
        assert r40["mean_dR_meep_tmm_gated"] <= G.W_MEAN_R and r40["mean_dT_meep_tmm_gated"] <= G.W_MEAN_T
    # F-B witness (unconditional, review finding 7): the overdamped Debye pole
    # is unstable at 10 px/cm (eps_num(Nyq) = 0.486 < 1); the committed
    # meep_debye__res10.json is built from the run log (no JSON is written by a
    # leg that raises inside Meep) and the ladder summary carries it as
    # finite = false.
    w = _read(_RESULTS / "meep_debye__res10.json")
    assert w["run"]["finite"] is False and w["rc"] == 1 and w["resolution"] == 10
    assert w["omega_n_dt"] == pytest.approx(1.048, abs=0.002) and w["eps_num_nyq"] == pytest.approx(0.486, abs=0.002)
    assert summ["arms"]["debye"]["rungs"]["10"]["finite"] is False
    print("r3-summary meep ladder:", {a: v["orders"] for a, v in summ["arms"].items()})
