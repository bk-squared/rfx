"""The exact-lattice witness standard, replayed on the committed slab-family artifacts.

Pre-declaration: ``docs/design_notes/20260903_lattice_witness_standard.md``.

Two parts, the convention the cv22 / cv23 gate tests use:

1. **Artifact-free (always run).** The extended lattice solver against the one
   cv23 measured (bit-for-bit at sigma = 0, 1e-14 class otherwise); the source
   spectral-gain constant LAMBDA against a numerical rFFT of the rig's own
   waveform; the incident envelope-rate inverse; the monotonicity of the budget
   in every witness it reads; and the analytic falsifiers F1 / F2 / F3, whose
   verdict at every rung is pre-declared here.

2. **Artifact replay (skips while the artifacts are absent).** ``lattice_witness.json``
   rebuilds from the committed ``rfx*.json`` rungs of cv22 and cv23, every rung
   passes GL1 and GL2, and the three falsifiers fire exactly where the note says
   they do. Plus the cv04 material: cv23's ``sigma_zero`` arm IS cv04's lossless
   eps' = 4 slab on the settled rig, and it is gated here against its own
   ``params_run`` -- and must FAIL when judged against the declared lossy lattice.

No FDTD runs here. Seconds.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_R22 = _REPO / "validation/crossval/_22_dispersive_results"
_R23 = _REPO / "validation/crossval/_23_lossy_results"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("lw_dispersive_eps", "validation/crossval/comparators/dispersive_eps.py")
G = _load("lw_cv22_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")
L23 = _load("lw_cv23_gates", "validation/crossval/comparators/cv23_lossy_gates.py")
LW = _load("lw_lattice_witness", "validation/crossval/comparators/lattice_witness.py")
RIG = _load("lw_slab_rig", "validation/crossval/comparators/slab_rig.py")

_DT_DX = 2.335067793382187e-12   # the rig's dt at dx = 1 mm (Courant 0.700)

# Pre-declared per (case, rung, falsifier): does the falsifier FIRE, i.e. does
# the lattice gate reject the defective model at that rung? Note section 7.
# "False" rows are the two the note declares NON-DISCRIMINATING and says why:
# cv22's Debye arm (its record settles only to 7.2e-3, so the truncation term
# dominates its 2.2e-3 lattice term) and the finest cv23 rungs, where the
# second-order term has fallen below the window -- which is F2 measuring the
# convergence it is supposed to measure.
#
# ``eps_continuum`` (F4, note section 7) is the falsifier for the ONE ingredient
# this lane adds: carrying eps_num into the lattice. It fires on cv22's Drude arm
# only, and on the two cv23 rungs whose sigma term is large enough to see at
# their record (tand1 and tand3 = tand3_dx2, through T). Its "False" rows are a
# measurement, not an omission: at these rungs the ADE / sigma discrete-time term
# is BELOW the window, so it is not separately testable there.
_F_FIRES = {
    ("cv22", "debye"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": False, "eps_x1p01": False, "eps_continuum": False,
    },
    ("cv22", "lorentz"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": False,
    },
    ("cv22", "drude"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": True,
    },
    ("cv23", "tand0p1"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": False,
    },
    ("cv23", "tand1"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": True,
    },
    ("cv23", "tand3"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": True,
    },
    ("cv23", "tand0p1_dx2"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": False,
    },
    ("cv23", "tand0p1_dx4"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": False, "eps_x1p01": True, "eps_continuum": False,
    },
    ("cv23", "tand1_dx2"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": False,
    },
    ("cv23", "tand1_dx4"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": False,
    },
    ("cv23", "tand3_dx2"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": True,
    },
    ("cv23", "tand3_dx4"): {
        "thickness_plus_cell": True, "thickness_minus_cell": True,
        "continuum": True, "eps_x1p01": True, "eps_continuum": False,
    },
}


# ===========================================================================
# 1. Artifact-free
# ===========================================================================

def test_extended_lattice_reproduces_the_cv23_solver_it_generalizes():
    """The PI's condition for extending ``yee_lattice_slab_rt``: the model form
    must reproduce cv23's conductive solver exactly at sigma = 0 and to the
    1e-14 class otherwise -- so cv23's committed lattice numbers are unchanged."""
    f = np.linspace(4.0e9, 10.0e9, 120)
    R0, T0 = de.yee_lattice_slab_rt(f, 4.0, 0.0, G.D_SLAB_M, G.DX_M, _DT_DX)
    R1, T1 = de.yee_lattice_slab_rt_model(f, "conductive", {"eps_inf": 4.0, "sigma": 0.0},
                                          G.D_SLAB_M, G.DX_M, _DT_DX)
    assert np.array_equal(R0, R1) and np.array_equal(T0, T1), "sigma = 0 must be bit-for-bit"
    for arm, spec in L23.ARMS.items():
        p = spec["params"]
        R0, T0 = de.yee_lattice_slab_rt(f, p["eps_inf"], p["sigma"], G.D_SLAB_M, G.DX_M, _DT_DX)
        R1, T1 = de.yee_lattice_slab_rt_model(f, "conductive", p, G.D_SLAB_M, G.DX_M, _DT_DX)
        assert np.max(np.abs(R0 - R1)) < 1e-14, (arm, np.max(np.abs(R0 - R1)))
        assert np.max(np.abs(T0 - T1)) < 1e-14, (arm, np.max(np.abs(T0 - T1)))


def test_lattice_converges_to_the_transfer_matrix_at_second_order_for_every_model():
    """The lattice model is the rig's discretization of the SAME continuum
    problem the cv gates use, for the pole models too: |lattice - TMM| must fall
    like dx^2. (cv23 note section 12.2 measured this for the conductive arm.)"""
    f = np.linspace(4.0e9, 10.0e9, 60)
    cases = [("conductive", {"eps_inf": 4.0, "sigma": 4.6731})]
    cases += [(s["model"], s["params"]) for s in G.ARMS.values()]
    for model, params in cases:
        Ra, Ta = G.analytic_rt(f, model, params)
        errs = []
        for K in (1, 2, 4):
            dx, dt = G.DX_M / K, _DT_DX / K
            Rl, Tl = de.yee_lattice_slab_rt_model(f, model, params, G.D_SLAB_M, dx, dt)
            errs.append(float(np.mean(np.abs(Rl - Ra))))
        for lo, hi in zip(errs[:-1], errs[1:]):
            order = math.log2(lo / hi)
            assert 1.7 <= order <= 2.3, (model, errs, order)


def test_source_spectral_gain_matches_a_numerical_transform_of_the_rig_waveform():
    """LAMBDA = sqrt(pi) tau / dt is the whole calibration between a time-domain
    witness level (relative to the incident peak) and a spectral one. Checked
    against the rig's actual differentiated Gaussian, sampled at the rig's dt."""
    dt = _DT_DX
    tau = LW.TAU_SRC_S
    t0 = G.SRC_T0_OVER_TAU * tau
    n = 4096
    u = (np.arange(n) * dt - t0) / tau
    s = -2.0 * u * np.exp(-u ** 2)          # rfx/sources/tfsf.py, differentiated_gaussian
    S = np.abs(np.fft.rfft(s, n=8 * n))
    measured = float(S.max() / np.max(np.abs(s)))
    assert abs(measured / LW.source_spectral_gain(dt) - 1.0) < 2e-3, (measured, LW.source_spectral_gain(dt))
    # and the relative shape the artifacts store is the same function
    fr = np.fft.rfftfreq(8 * n, d=dt)
    band = (fr > 2e9) & (fr < 14e9)
    rel = S[band] / S.max()
    assert np.max(np.abs(rel - G.incident_amplitude_rel(fr[band]))) < 5e-3


def test_incident_tail_rate_inverts_the_source_envelope():
    for level in (1e-3, 2.1e-4, 1e-4):
        rate = LW.incident_tail_rate(level)
        a = rate * LW.TAU_SRC_S / 2.0
        assert a > 1.0 / math.sqrt(2.0)
        assert abs(2.0 * a * math.exp(-a * a) - level) < 1e-12 * max(level, 1e-12) + 1e-15
    with pytest.raises(ValueError):
        LW.incident_tail_rate(1.0)          # above the envelope peak: not a tail


def test_the_window_is_monotone_in_every_witness_it_reads():
    """A looser tail, a looser purity, a slower ring-down or a longer record can
    only WIDEN W_witness. This is what makes the gate honest: the run cannot buy
    a tighter window by being worse."""
    f = np.linspace(4.0e9, 10.0e9, 40)
    a = G.incident_amplitude_rel(f)
    base = dict(dt=_DT_DX, n_steps=1000, scat_tail_rel=1e-3, trans_tail_rel=1e-3,
                purity_rel=1e-4, rate_1_s=1.0e10)
    t0 = LW.budget_terms(f, a, **base)
    for key, worse in (("scat_tail_rel", 2e-3), ("purity_rel", 2e-4),
                       ("rate_1_s", 5.0e9), ("n_steps", 2000)):
        t1 = LW.budget_terms(f, a, **{**base, key: worse})
        which = {"scat_tail_rel": "delta_scat", "purity_rel": "delta_inc",
                 "rate_1_s": "delta_scat", "n_steps": "delta_round"}[key]
        assert np.all(t1[which] >= t0[which] - 1e-18), key
        assert np.any(t1[which] > t0[which]), key


class _ReachedSetup(Exception):
    """Raised from the ``setup`` hook to prove the guard let the call through."""


def _bar_probe(bar, *, recipe=None):
    """``run_slab_arm`` with one settling bar, stopped at the material hook.

    No FDTD: ``setup`` is the first thing the rig calls after the settling-bar
    guard, so raising there separates "the guard refused" from "the guard let
    it through".
    """
    def setup(rig):
        raise _ReachedSetup

    kw = {} if recipe is None else {"recipe": recipe}
    RIG.run_slab_arm("conductive", {"eps_inf": 4.0, "sigma": 0.0}, setup=setup,
                     nx_interior=200, n_steps_cap=10, smoke=True, settling_bar=bar, **kw)


@pytest.mark.parametrize("bar,recipe", [
    # The four probes the 2026-09-03 review ran against the pre-fix guard.
    (1.0, None),        # grossly looser than the -40 dB bar
    (2e-2, "r3"),       # 2x looser, refused before and after
    (5e-2, G.RECIPE_CV04),  # 5x looser -- ACCEPTED before the fix, because the
                            # guard compared against the ACTIVE recipe's bar and
                            # cv04's TAIL_LIMIT is 0.10, not SETTLING_LIMIT
    (0.0, "r3"),        # disables the witness -- accepted before the fix
    (-1.0, "r3"),       # ditto, and negative
])
def test_settling_bar_outside_the_declared_interval_is_refused(bar, recipe):
    """The bar may only TIGHTEN the family's -40 dB witness. The comparison is
    against the family constant ``SETTLING_LIMIT``, not against whatever bar the
    active recipe happens to carry, and a non-positive bar is not a tightening
    at all -- it switches the witness off."""
    with pytest.raises(ValueError, match="never widened"):
        _bar_probe(bar, recipe=recipe)


def test_settling_bar_can_only_be_tightened():
    """The other half of the same contract: a genuine tightening (the section
    8.1 Debye rung's 3e-4, and the bar itself) is let through -- the guard is
    not simply refusing every bar."""
    for bar in (G.SETTLING_LIMIT, 3e-4, 1e-6):
        with pytest.raises(_ReachedSetup):
            _bar_probe(bar)


def test_f1_one_cell_thickness_separation_is_computed_a_priori_at_every_rung():
    """F1, purely analytic (no artifact): at every declared rung of every arm,
    a one-cell thickness error moves the lattice prediction by a margin that is
    computed here from the material and the mesh alone. The note tabulates it;
    this pins that it is non-trivial on at least one observable everywhere --
    including the surface-impedance arm tand3, where R is thickness-blind and
    T carries it (cv23 note section 12.2)."""
    f = np.linspace(4.0e9 + 1e6, 10.0e9, 229)
    rungs = [("conductive", s["params"], L23.ARM_DX_DIV.get(arm, 1)) for arm, s in L23.ARMS.items()]
    rungs += [(s["model"], s["params"], 1) for s in G.ARMS.values()]
    rungs += [("conductive", {"eps_inf": 4.0, "sigma": 0.0}, 1)]
    for model, params, K in rungs:
        for Kr in {K, 2 * K, 4 * K}:
            dx, dt = G.DX_M / Kr, _DT_DX / Kr
            sR, sT, sA = LW.model_separation(f, model, params, dx, dt, "thickness_plus_cell")
            rel = max(float(np.mean(sR)), float(np.mean(sT)))
            assert rel > 1e-4, (model, params, Kr, np.mean(sR), np.mean(sT))


def test_f2_is_the_lattice_term_itself_and_falls_at_second_order():
    """F2's defect -- the CONTINUUM transfer matrix used as the witness model --
    separates from the declared lattice by exactly W_lat(f), the second-order
    term. So F2 must stop firing as the mesh is refined; the rung at which it
    does is the note's convergence statement, not a weakness of the gate."""
    f = np.linspace(4.0e9 + 1e6, 10.0e9, 229)
    p = L23.ARMS["tand3"]["params"]
    seps = []
    for K in (1, 2, 4, 8):
        sR, _sT, _sA = LW.model_separation(f, "conductive", p, G.DX_M / K, _DT_DX / K, "continuum")
        seps.append(float(np.mean(sR)))
    for lo, hi in zip(seps[:-1], seps[1:]):
        assert 1.7 <= math.log2(lo / hi) <= 2.3, seps


def test_f4_eps_continuum_isolates_the_one_ingredient_this_lane_adds():
    """F4 (``eps_continuum``): the lattice built on the CONTINUUM permittivity
    instead of the discrete-time ``eps_num`` the update realizes. It is the only
    falsifier that moves the ADE / sigma correction and nothing else -- same
    marcher, same geometry, same (dx, dt).

    Two things are asserted, both analytic:

    * on a LOSSLESS non-dispersive slab (cv04's material, eps' = 4, sigma = 0)
      the separation is IDENTICALLY zero, because eps_analytic IS eps_num there.
      So F4 cannot fire on that rung by construction, and a "silent" reading
      there is geometry, not detection power.
    * on every arm whose material HAS a discrete-time correction the separation
      is strictly positive and grows with the correction, so the falsifier is
      not vacuous where it is claimed to bite.
    """
    f = np.linspace(4.0e9 + 1e6, 10.0e9, 229)
    sR, sT, sA = LW.model_separation(f, "conductive", {"eps_inf": 4.0, "sigma": 0.0},
                                     G.DX_M, _DT_DX, "eps_continuum")
    assert max(float(np.max(sR)), float(np.max(sT)), float(np.max(sA))) == 0.0

    for arm, spec in L23.ARMS.items():
        K = L23.ARM_DX_DIV.get(arm, 1)
        sR, sT, _sA = LW.model_separation(f, "conductive", spec["params"],
                                          G.DX_M / K, _DT_DX / K, "eps_continuum")
        assert float(np.mean(sR)) > 0.0 and float(np.mean(sT)) > 0.0, arm
    for arm, spec in G.ARMS.items():
        sR, sT, _sA = LW.model_separation(f, spec["model"], spec["params"],
                                          G.DX_M, _DT_DX, "eps_continuum")
        assert float(np.mean(sR)) > 0.0 and float(np.mean(sT)) > 0.0, arm


# ===========================================================================
# 2. Artifact replay
# ===========================================================================

def _witness_doc(results: Path, case_id: str) -> dict:
    p = results / LW.witness_json_name()
    if not p.is_file():
        pytest.skip(f"lattice witness artifact absent: {p.relative_to(_REPO)} (run --lattice-witness)")
    doc = json.loads(p.read_text())
    assert doc["schema"] == LW.SCHEMA and doc["case_id"] == case_id
    return doc


def _rfx_rungs(results: Path) -> dict:
    if not (results / "rfx.json").is_file():
        pytest.skip(f"baseline artifact absent under {results.relative_to(_REPO)} (VESSL run pending)")
    return LW.rungs_from_results(str(results))


@pytest.mark.parametrize("case_id,results", [
    ("22_dispersive_slab_fresnel", _R22),
    ("23_lossy_slab_fresnel", _R23),
])
def test_committed_witness_artifact_rebuilds_from_the_committed_rungs(case_id, results):
    doc = _witness_doc(results, case_id)
    rebuilt = LW.build_from_results(case_id, str(results))
    assert set(rebuilt["rungs"]) == set(doc["rungs"]), (case_id, sorted(doc["rungs"]))
    for name, r in doc["rungs"].items():
        b = rebuilt["rungs"][name]
        for k in ("mean_dR_lattice_gated", "mean_W_witness_R_gated", "mean_W_ceiling_R_gated",
                  "worst_ratio_R", "worst_ratio_T", "worst_ratio_A"):
            assert b[k] == pytest.approx(r[k], rel=1e-9, abs=1e-15), (case_id, name, k)
        assert b["gates"] == r["gates"], (case_id, name)


@pytest.mark.parametrize("case,case_id,results", [
    ("cv22", "22_dispersive_slab_fresnel", _R22),
    ("cv23", "23_lossy_slab_fresnel", _R23),
])
def test_every_committed_rung_passes_the_lattice_gate(case, case_id, results):
    """The claim the standard makes: at EVERY dx rung the case runs, the
    residual against the exact lattice is inside a window derived from the
    lattice model's own error budget -- so the residual against the continuum
    IS the lattice term, rung by rung."""
    doc = _witness_doc(results, case_id)
    assert doc["verdict"]["all_rungs_ok"], doc["verdict"]
    for name, r in doc["rungs"].items():
        assert r["gates"]["precond_cpml_gate"], (name, "CPML round trip inside the record")
        assert r["gates"]["precond_tail_witness"], (name, "the case's own settling witness")
        for k, v in r["gates"].items():
            assert v, (case, name, k, r["worst_ratio_R"], r["worst_ratio_T"], r["worst_ratio_A"])
        # headroom, reported so a drift shows up as a number and not only a colour
        assert max(r["worst_ratio_R"], r["worst_ratio_T"], r["worst_ratio_A"]) <= 1.0
        print(f"lattice-witness-summary {case} {name}: dx {r['dx_m']*1e3:.2f} mm, "
              f"|rfx-lattice| mean R {r['mean_dR_lattice_gated']:.2e} vs "
              f"W {r['mean_W_witness_R_gated']:.2e}, worst bin ratio "
              f"{max(r['worst_ratio_R'], r['worst_ratio_T'], r['worst_ratio_A']):.2f}")


@pytest.mark.parametrize("case,case_id,results", [
    ("cv22", "22_dispersive_slab_fresnel", _R22),
    ("cv23", "23_lossy_slab_fresnel", _R23),
])
def test_falsifiers_fire_exactly_where_the_note_says_they_do(case, case_id, results):
    entries = _rfx_rungs(results)
    for name, arm_doc in entries.items():
        expect = _F_FIRES.get((case, name))
        if expect is None:
            # A rung the note pre-declares but has no committed measurement for
            # yet (the section-8.1 remedy rung). Report its verdict; F1 is still
            # asserted for it by test_f1_..._at_every_rung.
            got = {k: (not LW.evaluate_falsifier(arm_doc, k)["witness_ok"]) for k in LW.FALSIFIER_KINDS}
            print(f"lattice-witness-summary {case} {name}: NEW rung, falsifiers fired = {got}")
            continue
        for kind, should_fire in expect.items():
            fr = LW.evaluate_falsifier(arm_doc, kind)
            fired = not fr["witness_ok"]
            assert fired == should_fire, (
                case, name, kind, "fired" if fired else "silent",
                fr["separation_over_window_R"], fr["n_bins_R_over_window"])


@pytest.mark.parametrize("case,case_id,results", [
    ("cv22", "22_dispersive_slab_fresnel", _R22),
    ("cv23", "23_lossy_slab_fresnel", _R23),
])
def test_f1_one_cell_thickness_fails_the_lattice_gate_at_every_rung(case, case_id, results):
    """The strongest thing the standard buys: no rung of any slab arm survives a
    one-cell thickness error in the model. F1 is the falsifier the continuum
    gate cannot run (a one-cell error is inside its window on several arms)."""
    entries = _rfx_rungs(results)
    for name, arm_doc in entries.items():
        for kind in ("thickness_plus_cell", "thickness_minus_cell"):
            fr = LW.evaluate_falsifier(arm_doc, kind)
            assert not fr["witness_ok"], (case, name, kind)
            over = max(fr["n_bins_R_over_window"], fr["n_bins_T_over_window"])
            assert over >= 40, (case, name, kind, over, fr["n_bins_gated"])


def test_the_cv04_material_rung_is_the_committed_sigma_zero_arm():
    """cv04's slab is eps' = 4, sigma = 0, d = 10 mm at dx = 1 mm. cv23's
    ``tand0p1_sigma_zero`` falsifier runs exactly that material on the SETTLED
    version of the same rig, so the cv04 material has a committed lattice rung
    today. Judged against its own material it passes; judged against the
    DECLARED lossy lattice -- which is what makes it a cv23 falsifier -- the
    lattice gate rejects it."""
    p = _R23 / "rfx__falsifier_tand0p1_sigma_zero.json"
    if not p.is_file():
        pytest.skip(f"cv23 sigma_zero falsifier artifact absent: {p.relative_to(_REPO)}")
    ad = json.loads(p.read_text())["arms"]["tand0p1"]
    assert ad["params_run"]["sigma"] == 0.0 and ad["params_run"]["eps_inf"] == 4.0
    ok = LW.evaluate(ad, params=ad["params_run"], tag="cv04_material")
    assert ok["witness_ok"], ok["gates"]
    assert ok["worst_ratio_R"] <= 1.0
    # the same measurement against the declared (lossy) lattice must be rejected
    wrong = LW.evaluate(ad, tag="cv04_material_vs_declared")
    assert not wrong["witness_ok"] and wrong["worst_ratio_R"] > 10.0
    # and the continuum falsifier fires on this rung too
    f2 = LW.evaluate_falsifier(ad, "continuum", params=ad["params_run"])
    assert not f2["witness_ok"], f2
    print(f"lattice-witness-summary cv04-material sigma_zero: |rfx-lattice| mean R "
          f"{ok['mean_dR_lattice_gated']:.2e} vs W {ok['mean_W_witness_R_gated']:.2e}; "
          f"|rfx-TMM| is the lattice term")
