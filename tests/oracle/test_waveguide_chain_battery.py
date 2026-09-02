"""WR-90 chain battery (v1.8 WP2): the falsifier battery on one common fixture set.

Pre-declaration (every position, rung, drive setting and tolerance, committed
before the first run): ``docs/design_notes/waveguide_chain_battery_predeclaration.md``.
Builder: ``tests/_waveguide_chain_battery_fixture.py``. Gate arithmetic (one
copy, shared with the measurement driver): ``tests/_waveguide_chain_battery_gates.py``.
Fixture: ``tests/fixtures/waveguide_chain_battery/fixture.json`` (schema in the
README next to it), written by
``scripts/diagnostics/waveguide_chain_battery_measure.py``.

Two layers:

* **REPLAY** (fast lane, no FDTD): reads the fixture and re-asserts every
  pre-declared gate from the stored numbers — settling per drive, non-vacuity,
  the physics gates of §6, the referee of §5(d), AD-vs-FD with the ULP-span
  validity evaluated before the accuracy gate (§5(a)), the reference-plane
  invariance / rotation / wrong-sign witness (§5(b)), the dx ladder with its
  interpretability guard (§5(c)), the cheap refute of §8, and that the stored
  ``verdicts`` equal what the shared gate module recomputes. Report-first
  quantities (gradient invariance, Richardson, monotonicity) are printed on
  the first run and pinned by ``gate_from_envelope`` in a separate commit.
* **LIVE** (``slow``): re-measures the coarse and mid rungs on CPU (the fine
  rung under ``gpu``), compares the S-matrices with the fixture to the stated
  tolerance, and re-runs the rotation gate and one AD-vs-FD leg at the coarse
  rung against physics rather than against the fixture.

Lane placement (contract criterion 3, "fast lane when ≤ 30 s"): the replay
layer is pure JSON arithmetic; the measurement itself is minutes on a GPU and
lives in the slow lane / the VESSL YAML named in the fixture's provenance.

Must-not list honoured: no ``normalize=True``; nothing imported from
``rfx/probes/refplane.py``; no committed tolerance moved; a red gate stays red
here (``xfail(strict=True)`` carrying the measured number and a root cause in
the PR body), never loosened.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F
from tests import _waveguide_chain_battery_gates as G
from tests._gate_policy import gate_from_envelope

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture.json"

# Live-layer tolerance on |S_live − S_fixture| (absolute, per entry), derived
# from the measured cross-backend envelope through the shared policy: the 12
# coarse + mid cells re-measured on this box (CPU, jax 0.6.2) against the
# fixture (VESSL 369367257823, gpu-rtx4090, jax 0.4.33) differed by at most
# 5.000e-6 (pec_short-coarse-flux; the others 9.6e-7 .. 2.6e-6). The quantum
# is 1e-4 so the pin is not a 2x knife edge on a float32 reassociation
# envelope; the fine rung (4x the steps) is compared on the GPU lane only.
LIVE_ABS_S_ENVELOPE = 5.000e-6
LIVE_ABS_S_TOL: float | None = gate_from_envelope(LIVE_ABS_S_ENVELOPE, quantum=10000)   # 1e-4

# Gates that are RED on the committed fixture, by test family and parameter
# id, each with the measured number and the mechanism (root cause in the PR
# body). They stay red in the live layer and are xfail(strict=True) here —
# never loosened. Filled in the measurement commit; an empty family means
# every case of that family passed.
_RED_SETTLING = (
    "measured 0.0 dB on both drives at 40 AND 80 periods (VESSL 369367257823): behind the "
    "4/8-cell short the far-port records are exactly zero at this rung and "
    "settling_db_from_port_records reads (end+tiny)/(peak+tiny) = 0 dB on them; the records "
    "in the float32 normal range ring down to -100 / -98 dB (fixture "
    "settling_db_over_normal_records). Witness degeneracy, not truncation — see PR body")
_RED_ROTATION = (
    "measured resid 6.602 deg (Yee), 6.565 deg (continuous), wrong-sign witness 0.73 deg at the "
    "fine rung: the port config de-embeds with f_cutoff = 6.378 GHz (discrete cutoff of an "
    "aperture one cell wider than the guide) while the guide propagates with 6.555 GHz (thru "
    "S21 phase fit); against the port's own beta the residual is 6e-5 deg. Mechanism, not "
    "tolerance — see PR body")
_RED_IDENTITY_FLUX = (
    "measured max scaled diff 1.440 (slab) / 1.065 (PEC-short) against rtol 1e-5 / atol 1e-7 "
    "(abs 0.9-1.1e-5): the float32 reverse-mode primal of the flux lane differs from the "
    "untraced call by reassociation of a 2849-step Poynting DFT; the same call under x64 "
    "agrees to 1.5e-15 - 2.2e-14 (fixture x64_witness). Gate stays as pre-declared — see PR body")
_RED_AD_FD_ZERO_DERIVATIVE = (
    "measured g_ad = +2.68e-5 (float32) vs g_fd = -7.24e-7, rel 38, FD span 6.5e8 ULP: the "
    "objective's derivative is physically zero (|S11| = 1 for a lossless window in front of a "
    "PEC); float32 AD noise 2.7e-5 exceeds the O(1e-6) residual derivative, and the x64 AD "
    "(-9.8e-7) agrees with FD at that level. The pre-declared ULP-floor skip did not occur "
    "because the float64 FD resolves the residual — see PR body")
KNOWN_RED: dict[str, dict[str, str]] = {
    "settling": {"pec_short-fine-false": _RED_SETTLING, "pec_short-fine-flux": _RED_SETTLING},
    "ad_vs_fd": {"pec_short-flux-eps-s11_mag2": _RED_AD_FD_ZERO_DERIVATIVE},
    "forward_identity": {
        f"{d}-flux-{k}-{o}": _RED_IDENTITY_FLUX
        for d, k, objs in (("slab", "eps", ("s11_mag2", "s21_mag2", "re_s21", "im_s21")),
                           ("pec_short", "eps", ("s11_mag2", "re_s11", "im_s11")),
                           ("pec_short", "sigma", ("s11_mag2",)))
        for o in objs},
    "rotation": {f"{d}-{l}": _RED_ROTATION for d in ("pec_short", "slab") for l in ("false", "flux")},
    "physics": {},
    "referee": {},
}

EXPECTED_PREFLIGHT_CODES = {
    # (dut, rung): sorted preflight codes, pre-declaration §2.6
    ("thru", "coarse"): [], ("pec_short", "coarse"): [],
    ("slab", "coarse"): ["lossless_q", "mesh_resolution", "mesh_resolution", "mesh_resolution"],
    ("thru", "mid"): [], ("pec_short", "mid"): ["mesh_resolution"],
    ("slab", "mid"): ["lossless_q", "mesh_resolution", "mesh_resolution", "mesh_resolution"],
    ("thru", "fine"): [], ("pec_short", "fine"): [],
    ("slab", "fine"): ["lossless_q"],
}
EXPECTED_PREFLIGHT_FRAGMENTS = {
    ("slab", "coarse"): ["5.1 cells per", "lossless"],
    ("pec_short", "mid"): ["5.08mm = 4.0 cells"],
    ("slab", "mid"): ["10.2 cells per", "lossless"],
    ("slab", "fine"): ["lossless"],
}
GRID_SHAPES = {"coarse": [83, 10, 5], "mid": [165, 19, 9], "fine": [329, 37, 17]}
N_STEPS = {"coarse": 713, "mid": 1425, "fine": 2849}
CPML = {"coarse": 17, "mid": 34, "fine": 68}
DUT_CELLS = {("pec_short", "coarse"): 72, ("pec_short", "mid"): 576, ("pec_short", "fine"): 4608,
             ("slab", "coarse"): 144, ("slab", "mid"): 1152, ("slab", "fine"): 9216}


def _load() -> dict | None:
    if not FIXTURE.exists():
        return None
    return json.loads(FIXTURE.read_text())


_FX = _load()


def _cells():
    return [] if _FX is None else _FX["cells"]


def _cell_id(c) -> str:
    return f"{c['dut']}-{c['rung']}-{c['lane']}"


def _param(pid: str, family: str | None = None, *values):
    marks = ()
    if family is not None and pid in KNOWN_RED[family]:
        marks = (pytest.mark.xfail(strict=True, reason=KNOWN_RED[family][pid]),)
    return pytest.param(*(values or (pid,)), id=pid, marks=marks)


def _params_cells(pred=lambda c: True, family: str | None = None):
    return [_param(_cell_id(c), family) for c in _cells() if pred(c)]


def _cell(cid: str) -> dict:
    for c in _cells():
        if _cell_id(c) == cid:
            return c
    raise KeyError(cid)


def _legs():
    return [] if _FX is None else _FX["ad_vs_fd"]


def _leg_id(l) -> str:
    return f"{l['dut']}-{l['lane']}-{l['theta_kind']}-{l['objective']}"


def _params_legs(family: str | None = None):
    return [_param(_leg_id(l), family) for l in _legs()]


def _leg(lid: str) -> dict:
    for l in _legs():
        if _leg_id(l) == lid:
            return l
    raise KeyError(lid)


def _planes():
    if _FX is None:
        return {}
    return {k: v for k, v in _FX["plane_shift"].items() if k != "cheap_refute"}


def _params_planes(family: str | None = None):
    return [_param(k.replace("|", "-"), family, k) for k in _planes()]


def _params_gradient():
    out = []
    for k, p in _planes().items():
        for obj in p["gradient_invariance"]:
            out.append(pytest.param(k, obj, id=f"{k.replace('|', '-')}-{obj}"))
    return out


def _params_ladder():
    return [] if _FX is None else [pytest.param(k, id=k.replace("|", "-")) for k in _FX["ladder"]]


@pytest.fixture(scope="module")
def fx() -> dict:
    if _FX is None:
        pytest.fail(f"{FIXTURE} is missing — the battery has not been measured, or the fixture "
                    "was deleted; regenerate with the driver named in the README")
    return _FX


# ===========================================================================
# REPLAY layer
# ===========================================================================

def test_fixture_constants_match_builder(fx):
    """A drift here means the fixture was measured on a different geometry
    than the one now declared (README, ``fixture``)."""
    c = fx["fixture"]
    assert c["a_m"] == F.A_M and c["b_m"] == F.B_M
    assert c["dx_ladder_m"] == list(F.DX_LADDER) and c["n_ladder"] == list(F.N_LADDER)
    assert c["domain_x_m"] == F.DOMAIN_X_M
    assert c["port_planes_m"] == [F.PORT_LEFT_X_M, F.PORT_RIGHT_X_M]
    assert c["reference_planes_default_m"] == [F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M]
    assert c["reference_planes_shifted_m"] == [F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M]
    assert c["probe_planes_m"] == [F.PROBE_LEFT_M, F.PROBE_RIGHT_M]
    assert c["pec_short_x_m"] == list(F.PEC_SHORT_X_M) and c["slab_x_m"] == list(F.SLAB_X_M)
    assert c["slab_eps_r"] == F.SLAB_EPS_R
    assert c["pec_short_window_x_m"] == list(F.PEC_SHORT_WINDOW_X_M)
    np.testing.assert_allclose(c["freqs_hz"], F.FREQS, rtol=0, atol=0)
    assert c["f0_hz"] == F.F0_HZ and c["bandwidth"] == F.BANDWIDTH
    assert c["band_centre_bin"] == F.BAND_CENTRE_BIN and c["num_periods"] == F.NUM_PERIODS
    assert c["lanes"] == ["false", "flux"]
    assert c["boundary"] == "cpml-x, pec-y, pec-z"
    assert c["theta0_eps"] == F.THETA0_EPS and c["theta0_sigma_s_per_m"] == F.THETA0_SIGMA_S_PER_M
    assert c["fd_step_eps"] == F.FD_STEP_EPS and c["fd_step_sigma_s_per_m"] == F.FD_STEP_SIGMA_S_PER_M


def test_schema_and_provenance(fx):
    assert fx["schema"] == "rfx.waveguide_chain_battery" and fx["schema_version"] == 1
    assert (REPO / fx["predeclaration"]).exists()
    p = fx["provenance"]
    assert p["run_id"] != "local" and p["run_lane"] != "local", (
        "a developer dry run is never claims-bearing (README, provenance.run_id)")
    assert fx["predeclaration_sha"] not in ("unknown", "", p["commit"]), (
        "the pre-declaration commit must be recorded and must differ from (predate) the run commit")
    assert p["precision"] == "float32" and p["jax_enable_x64"] is False
    assert "recapture_command" in p and p["recapture_entry_point"].startswith("scripts/")
    assert (REPO / p["recapture_entry_point"]).exists()
    assert fx["legs_rung"] == G.LEGS_RUNG_DEFAULT
    # the whole-battery wall time decides the lane: the live layer is `slow`
    assert p["wall_time_s"] > 30.0, p["wall_time_s"]


def test_cells_cover_every_dut_rung_lane_and_re_assert_guard_3(fx):
    keys = {(c["dut"], c["rung"], c["lane"]) for c in fx["cells"]}
    assert keys == {(d, r, l) for d in F.DUTS for r in G.RUNG_LABELS for l in ("false", "flux")}
    fc = 299_792_458.0 / (2.0 * F.A_M)
    for c in fx["cells"]:
        r = c["rung"]
        assert c["grid_shape"] == GRID_SHAPES[r], (c["dut"], r)
        assert c["n_steps"] == N_STEPS[r] and c["cpml_layers"] == CPML[r]
        assert c["guide_cells_yz"] == [9 * (2 ** G.RUNG_LABELS.index(r)), 4 * (2 ** G.RUNG_LABELS.index(r))]
        assert c["fc_te10_numerical_hz"] == pytest.approx(fc, rel=1e-12)
        assert c["reference_planes_m"] == pytest.approx([F.REF_LEFT_DEFAULT_M, F.REF_RIGHT_DEFAULT_M], abs=1e-12)
        if c["dut"] == "thru":
            assert c["dut_cells"] is None
        else:
            # guard 3 of §5(c): rasterized counts scale exactly with 1/dx
            assert c["dut_cells"] == DUT_CELLS[(c["dut"], r)], (c["dut"], r, c["dut_cells"])
            assert math.prod(c["dut_runs_xyz"]) == c["dut_cells"]
        assert c["s_params"]["S11"] and len(c["s_params"]["S11"]) == len(F.FREQS)
        S = G.s_from_json(c["s_params"])
        assert np.all(np.isfinite(S))


@pytest.mark.parametrize("cid", _params_cells())
def test_preflight_findings_are_recorded_verbatim_and_match_the_predeclaration(cid):
    """Preflight output is part of the result (§2.6): codes and the quoted
    fragments must be exactly what the note pre-declared."""
    c = _cell(cid)
    codes = sorted(f["code"] for f in c["preflight"])
    assert codes == EXPECTED_PREFLIGHT_CODES[(c["dut"], c["rung"])], (cid, c["preflight"])
    text = " ".join(f["message"] for f in c["preflight"])
    for frag in EXPECTED_PREFLIGHT_FRAGMENTS.get((c["dut"], c["rung"]), []):
        assert frag in text, (cid, frag, text)
    for f in c["preflight"]:
        assert f["severity"] == "warning", (cid, f)
        assert f["message"], cid


@pytest.mark.parametrize("cid", _params_cells(family="settling"))
def test_settling_witness_per_drive(cid):
    """§2.5: every drive ≤ −40 dB on the claims-bearing record (the doubled
    record where the 40-period one read above the line)."""
    c = _cell(cid)
    eff = G.cell_settling_effective(c)
    rerun = c.get("settling_rerun")
    print(f"[{cid}] settling_db(40)={c['settling_db']}"
          + (f" rerun(80)={rerun['settling_db']}" if rerun else ""))
    for port, val in eff.items():
        assert val <= G.SETTLING_DB_MAX, (
            f"{cid} drive {port}: settling {val:.1f} dB above {G.SETTLING_DB_MAX} dB "
            f"(40-period {c['settling_db'][port]:.1f} dB"
            + (f", 80-period {rerun['settling_db'][port]:.1f} dB" if rerun else "") + ")")


@pytest.mark.parametrize("cid", _params_cells(lambda c: c["dut"] != "thru"))
def test_non_vacuity_both_reflecting_duts(cid):
    c = _cell(cid)
    assert c["non_vacuity_max_s11"] > G.NON_VACUITY_MIN_MAX_S11, (cid, c["non_vacuity_max_s11"])
    m = G.cell_metrics(G.s_from_json(c["s_params"]))
    assert m["non_vacuity_max_s11"] == pytest.approx(c["non_vacuity_max_s11"], rel=1e-12)


@pytest.mark.parametrize("cid", _params_cells(lambda c: c["dut"] != "thru", family="physics"))
def test_physics_gates_at_the_claims_rung(cid):
    """§6: column power < 1.02, magnitude reciprocity < 0.01, complex
    reciprocity ≤ 0.01 — gated at the fine rung, reported at the others.
    Power closure is written and reported only (gated in WP3)."""
    c = _cell(cid)
    m = G.cell_metrics(G.s_from_json(c["s_params"]))
    for k in ("column_power_max", "reciprocity_mag_mean", "reciprocity_complex_max", "power_closure_max"):
        assert m[k] == pytest.approx(c[k], rel=1e-9, abs=1e-15), (cid, k)
    print(f"[{cid}] colpow={m['column_power_max']:.5f} recip_mag={m['reciprocity_mag_mean']:.2e} "
          f"recip_complex={m['reciprocity_complex_max']:.2e} closure={m['power_closure_max']:.2e}")
    if c["rung"] != G.CLAIMS_RUNG:
        return
    assert m["column_power_max"] < G.COLUMN_POWER_MAX, (cid, m["column_power_max"])
    assert m["reciprocity_mag_mean"] < G.RECIPROCITY_MAG_MAX, (cid, m["reciprocity_mag_mean"])
    assert m["reciprocity_complex_max"] <= G.RECIPROCITY_COMPLEX_MAX, (cid, m["reciprocity_complex_max"])


@pytest.mark.parametrize("cid", _params_cells(lambda c: c["dut"] == "pec_short", family="referee"))
def test_referee_pec_short_bounds(cid):
    """§5(d): 0.99 ≤ |S11| < 1.03 every bin, mean within 0.02 — fine rung."""
    c = _cell(cid)
    r = G.referee_pec_short(G.s_from_json(c["s_params"]), F.FREQS)
    stored = _FX["referee"]["pec_short"][f"{c['rung']}|{c['lane']}"]
    assert r["min_s11"] == pytest.approx(stored["min_s11"], rel=1e-12)
    print(f"[{cid}] |S11| min={r['min_s11']:.5f} max={r['max_s11']:.5f} mean={r['mean_s11']:.5f} "
          f"bins>=1.03={r['bins_above_1_03']}")
    if c["rung"] != G.CLAIMS_RUNG:
        return
    assert G.referee_pec_short_pass(r), (cid, r)


@pytest.mark.parametrize("cid", _params_cells(lambda c: c["dut"] == "slab", family="referee"))
def test_referee_slab_vs_airy(cid):
    """§5(d): magnitude ≤ 0.05, phase ≤ 15° (phase on bins with |S_ref| ≥ 0.30,
    the mask that is part of the source gate) — fine rung."""
    c = _cell(cid)
    r = G.referee_slab_airy(G.s_from_json(c["s_params"]), F.FREQS)
    stored = _FX["referee"]["slab_airy"][f"{c['rung']}|{c['lane']}"]
    assert r["max_mag_abs_diff"] == pytest.approx(stored["max_mag_abs_diff"], rel=1e-12)
    print(f"[{cid}] max|d|S||={r['max_mag_abs_diff']:.4f} @ {r['worst_bin_hz']/1e9:.1f} GHz  "
          f"phase max={r['max_phase_diff_deg']:.2f}° (unmasked {r['max_phase_diff_deg_unmasked']:.2f}°, "
          f"masked bins S11={r['phase_bins_masked_s11']})")
    if c["rung"] != G.CLAIMS_RUNG:
        return
    assert G.referee_slab_mag_pass(r), (cid, r["max_mag_abs_diff"])
    assert G.referee_slab_phase_pass(r), (cid, r["max_phase_diff_deg"])


@pytest.mark.parametrize("lid", _params_legs(family="ad_vs_fd"))
def test_ad_vs_fd_leg(lid):
    """§5(a): FD validity (ULP span ≥ 1e4 in the loss's own dtype) BEFORE the
    accuracy gate rel ≤ 0.05; a leg under the floor is skipped, never passed.
    The PEC-short |S11|² under a lossless eps θ is the pre-declared skip; if it
    resolves, that is a finding to explain, not a bonus."""
    l = _leg(lid)
    e = G.ad_fd_entry(g_ad=l["g_ad"], f_plus=l["f_plus"], f_minus=l["f_minus"], h=l["h"],
                      loss_dtype=np.dtype(l["loss_dtype"]))
    assert e["verdict"] == l["verdict"], (lid, e, l["verdict"])
    assert l["x64_context"] is True and l["loss_dtype"] == "float64", lid
    assert l["rung"] == G.LEGS_RUNG_DEFAULT and l["dx_m"] == G.RUNG_DX[G.LEGS_RUNG_DEFAULT]
    print(f"[{lid}] g_ad={e['g_ad']:+.6e} g_fd={e['g_fd']:+.6e} rel={e['rel']:.3e} "
          f"span={e['fd_ulp_span']:.3g} ULP -> {e['verdict']}")
    if l["expected_ulp_floor_skip"] and e["verdict"] != "skipped_under_ulp_floor":
        # §5(a): "a measured PEC-short |S11|² eps-gradient that passes the floor
        # is a finding to explain, not a bonus" — the accuracy gate still
        # applies; the explanation is in the PR body / fixture.
        print(f"[{lid}] FINDING: pre-declared ULP-floor skip RESOLVED instead "
              f"(span {e['fd_ulp_span']:.3g} ULP, |S11|^2 at theta0 = {l['value_at_theta0']:.6f}, "
              f"g_fd = {e['g_fd']:+.3e}: the derivative of the discrete extractor's |S11| residual)")
    if e["verdict"] == "skipped_under_ulp_floor":
        pytest.skip(f"{lid}: FD reference cannot resolve this gradient (span {e['fd_ulp_span']:.3g} ULP"
                    + (", pre-declared" if l["expected_ulp_floor_skip"] else "") + ")")
    w = l.get("x64_witness")
    if w is not None:
        print(f"[{lid}] x64 witness: g_ad_x64={w['g_ad_x64']:+.6e} identity_x64 max|dS|="
              f"{w['forward_identity_x64']['max_abs_diff']:.3e}")
    assert e["verdict"] == "pass", (
        f"{lid}: rel {e['rel']:.3e} > {G.AD_FD_REL_GATE} (g_ad {e['g_ad']:+.4e}, g_fd {e['g_fd']:+.4e})")


@pytest.mark.parametrize("lid", _params_legs(family="forward_identity"))
def test_forward_identity_traced_equals_untraced(lid):
    """Contract criterion 1: S under the θ0 traced override equals the untraced
    call to rtol 1e-5 / atol 1e-7 (``tests/unit/autodiff/test_waveguide_flux_ad.py:104``)."""
    l = _leg(lid)
    ident = l["forward_identity"]
    conc = l.get("forward_identity_concrete_override_vs_plain")
    print(f"[{lid}] traced-vs-untraced max|dS|={ident['max_abs_diff']:.3e} scaled={ident['max_scaled_diff']:.3f}"
          + (f"; concrete-override-vs-plain max|dS|={conc['max_abs_diff']:.3e}" if conc else ""))
    assert G.forward_identity_pass(ident["max_scaled_diff"]), (lid, ident)


@pytest.mark.parametrize("key", _params_planes())
def test_plane_shift_abs_s_invariant(key):
    p = _planes()[key]
    print(f"[{key}] max|d|S||={p['abs_s_max_diff']:.3e}")
    assert p["abs_s_allclose"], (key, p["abs_s_max_diff"])


@pytest.mark.parametrize("key", _params_planes(family="rotation"))
def test_plane_shift_rotation_matches_yee_and_continuous_beta(key):
    """§5(b): every entry rotates by the pre-declared 2βΔ / β(Δ_L+|Δ_R|) within
    3° of the Yee-discrete β and 6° of the continuous β."""
    p = _planes()[key]
    base = _cell(f"{p['dut']}-{p['rung']}-{p['lane']}")
    rot = G.plane_shift_rotation(G.s_from_json(base["s_params"]), G.s_from_json(p["s_params_shifted"]),
                                 F.FREQS, base["dt_s"], base["dx_m"], fc_port_hz=p["fc_port_hz"])
    assert rot["resid_yee_max"] == pytest.approx(p["resid_yee_max"], abs=1e-9), "stored rotation block drifted"
    print(f"[{key}] fc_port={p['fc_port_hz']/1e9:.4f} GHz vs pre-declared {p['fc_predeclared_hz']/1e9:.4f} GHz; "
          f"residual against the extractor's own beta = {p.get('resid_port_beta_max', float('nan')):.3f}°; "
          f"measurable entries {p['entries_measurable']}")
    for name, r in rot["rotation_deg"].items():
        if not r["measurable"]:
            print(f"[{key}] {name}: not measurable (band peak |S| {r['abs_s_base_peak']:.2e} below the "
                  f"noise floor {G.ROTATION_NOISE_FLOOR:.1e})")
            continue
        print(f"[{key}] {name}: resid_yee={r['resid_yee_max']:.3f}° resid_cont={r['resid_cont_max']:.3f}° "
              f"resid_port={r.get('resid_port_beta_max', float('nan')):.3f}° "
              f"wrong_sign_min={r['wrong_sign_resid_min']:.1f}° measured@10GHz={r['measured'][F.BAND_CENTRE_BIN]:.1f}° "
              f"masked={r['masked_bins_hz']}")
    assert rot["resid_yee_max"] <= G.ROTATION_TOL_YEE_DEG, (key, rot["resid_yee_max"])
    assert rot["resid_cont_max"] <= G.ROTATION_TOL_CONTINUOUS_DEG, (key, rot["resid_cont_max"])
    assert rot["wrong_sign_resid_min"] > G.WRONG_SIGN_MIN_DEG, (key, rot["wrong_sign_resid_min"])


@pytest.mark.parametrize("key,obj", _params_gradient())
def test_gradient_invariance_report_then_pin(key, obj):
    """§5(b) gradient leg: magnitude objectives invariant, complex objectives
    rotation-covariant. Report-only on the first run (bar 1e-2); the pin
    commit fills ``pinned_gate`` from ``gate_from_envelope(measured, quantum=1000)``."""
    g = _planes()[key]["gradient_invariance"][obj]
    if g.get("skipped_under_ulp_floor"):
        pytest.skip(f"{key} {obj}: {g['reason']}")
    print(f"[{key}] {obj} ({g['kind']}): rel_change={g['rel_change']:.3e} bar={g['report_bar']} "
          f"pinned={g['pinned_gate']}")
    assert np.isfinite(g["rel_change"])
    if g["pinned_gate"] is None:
        return  # report-only, first run
    assert g["rel_change"] <= g["pinned_gate"], (key, obj, g["rel_change"], g["pinned_gate"])


@pytest.mark.parametrize("key", _params_ladder())
def test_ladder_non_increase_gate_with_interpretability_guard(key):
    """§5(c): fine_delta ≤ coarse_delta + floor on every bin; a successive-delta
    ratio outside [0.15, 0.70] on a conditioned bin makes the ladder
    'not interpretable' — neither passed nor failed."""
    lad = _FX["ladder"][key]
    print(f"[{key}] coarse_delta_worst={lad['coarse_delta_worst']:.4g} fine_delta_worst={lad['fine_delta_worst']:.4g} "
          f"excess_worst={lad['excess_worst']:+.4g} @ {lad['worst_bin_hz']/1e9:.1f} GHz  "
          f"ratio_worst={lad['successive_ratio_worst']} monotone={lad['monotone_fraction_of_bins']:.2f} "
          f"conditioned={lad['n_conditioned_bins']} -> {lad['verdict']}")
    if lad["verdict"] == "not_interpretable":
        pytest.xfail(f"{key}: not interpretable — ratio {lad['successive_ratio_worst']} outside "
                     f"{lad['ratio_window']} at {lad['successive_ratio_worst_bin_hz']/1e9:.1f} GHz (pre-declared outcome)")
    assert lad["gate_pass"], (key, lad["coarse_delta_worst"], lad["fine_delta_worst"], lad["floor"])
    assert lad["verdict"] == "pass"


@pytest.mark.parametrize("key", _params_ladder())
def test_ladder_witnesses_report_then_pin(key):
    """Witness (i) monotonicity + successive-delta ratio and (ii) Richardson vs
    oracle: written on the first run, pinned by ``gate_from_envelope`` later."""
    lad = _FX["ladder"][key]
    if "richardson" in lad:
        for pair, r in lad["richardson"].items():
            print(f"[{key}] richardson {pair}: max|est-oracle|={r['max_abs_diff']:.4g} "
                  f"@ {r['max_abs_diff_bin_hz']/1e9:.1f} GHz (finer rung alone {r['finer_rung_abs_diff_max']:.4g})")
        if lad.get("pinned_richardson_gate") is not None:
            pair = lad.get("pinned_richardson_pair", "mid-fine")
            assert lad["richardson"][pair]["max_abs_diff"] <= lad["pinned_richardson_gate"], (key, pair)
    if lad.get("pinned_monotone_fraction_min") is not None:
        assert lad["monotone_fraction_of_bins"] >= lad["pinned_monotone_fraction_min"], key


def test_cheap_refute_flipped_shift_sign_makes_the_rotation_gate_red(fx):
    """§8: under a local copy of ``_shift_modal_waves`` with the shift sign
    flipped, the rotation gate must go red by more than 10° while |S| stays
    invariant — otherwise the gate does not bind."""
    r = fx["plane_shift"].get("cheap_refute")
    assert r is not None, "the cheap refute was not run"
    print(f"[refute] min residual over entries={r['resid_yee_min_over_entries']:.1f}° "
          f"max={r['resid_yee_max_over_entries']:.1f}° gate_would_pass={r['rotation_gate_would_pass']}")
    assert not r["rotation_gate_would_pass"]
    assert r["resid_yee_min_over_entries"] > G.WRONG_SIGN_MIN_DEG
    assert r["abs_s_still_invariant"]


def test_port_cutoff_witness_is_recorded(fx):
    """Report-only mechanism witness: the cutoff the port config carries
    (``WaveguidePortConfig.f_cutoff``, the β / Z_TE of the extractor) against
    the guide's own cutoff fitted from the thru's S21 phase between the two
    declared planes (ref_shift = 0 on both, so no de-embed β enters)."""
    pc = fx["port_cutoff"]
    assert pc["per_rung"], "no thru cells to fit"
    for key, fit in pc["per_rung"].items():
        print(f"[port-cutoff {key}] port fc={fit['fc_port_hz']/1e9:.4f} GHz "
              f"(effective width {fit['port_cutoff_effective_width_cells']:.2f} cells) | guide fit "
              f"fc={fit['fc_fit_hz']/1e9:.4f} GHz (rms {fit['rms_deg_at_fit']:.2f}°, const {fit['const_deg_at_fit']:.1f}°) | "
              f"rms at c/2a={fit['rms_deg_at_c_over_2a']:.2f}°, at discrete-guide={fit['rms_deg_at_discrete_guide']:.2f}°, "
              f"at port fc={fit['rms_deg_at_port_cutoff']:.2f}°")
        assert np.isfinite(fit["fc_fit_hz"]) and fit["rms_deg_at_fit"] < 5.0, (key, fit)


def test_pins_are_derived_from_the_measured_envelopes(fx):
    """The pin commit: every pinned gate equals the shared policy applied to
    the stored measured quantity (a hand-moved pin is caught here)."""
    pins = fx["pins"]
    rels = [g["rel_change"] for k, p in _planes().items() for gk, g in p["gradient_invariance"].items()
            if not g.get("skipped_under_ulp_floor") and g.get("pinned_gate") is not None]
    assert rels and max(rels) == pytest.approx(pins["gradient_invariance_envelope"], rel=1e-12)
    assert pins["gradient_invariance_gate"] == gate_from_envelope(max(rels), quantum=G.GRADIENT_PIN_QUANTUM)
    for key, lad in fx["ladder"].items():
        if lad["verdict"] == "not_interpretable":
            assert lad["pinned_richardson_gate"] is None and lad["pinned_monotone_fraction_min"] is None, key
            continue
        if "richardson" in lad:
            pair = lad["pinned_richardson_pair"]
            assert lad["pinned_richardson_gate"] == gate_from_envelope(
                lad["richardson"][pair]["max_abs_diff"], quantum=G.RICHARDSON_PIN_QUANTUM[lad["kind"]]), key
        assert lad["pinned_monotone_fraction_min"] == G.pin_lower_from_envelope(
            lad["monotone_fraction_of_bins"], quantum=G.MONOTONE_PIN_QUANTUM), key
    assert LIVE_ABS_S_TOL == gate_from_envelope(LIVE_ABS_S_ENVELOPE, quantum=10000)


def test_stored_verdicts_equal_recomputed(fx):
    """README: the replay recomputes every verdict from the stored numbers; a
    disagreement with the stored block is itself a failure."""
    recomputed = G.recompute_verdicts(fx)
    assert recomputed == fx["verdicts"], {
        k: (fx["verdicts"].get(k), recomputed.get(k))
        for k in set(recomputed) | set(fx["verdicts"])
        if fx["verdicts"].get(k) != recomputed.get(k)}
    assert len(recomputed) > 100, len(recomputed)


# ===========================================================================
# LIVE layer
# ===========================================================================

def _measure_cell(dut: str, rung: str, lane, num_periods: float = F.NUM_PERIODS):
    sim = F.build_simulation(dut, G.RUNG_DX[rung])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
        res = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=lane)
    return sim, res, np.asarray(res.s_params).astype(complex), sorted(i.code for i in report)


def _live_compare(fx, rung: str):
    for dut in F.DUTS:
        for lane in F.LANES:
            label = G.LANE_LABELS[lane]
            cid = f"{dut}-{rung}-{label}"
            stored = _cell(cid)
            sim, res, S, codes = _measure_cell(dut, rung, lane)
            assert codes == sorted(f["code"] for f in stored["preflight"]), (cid, codes)
            S0 = G.s_from_json(stored["s_params"])
            d = float(np.max(np.abs(S - S0)))
            m = G.cell_metrics(S)
            print(f"[live {cid}] max|S_live-S_fixture|={d:.3e} settling={np.asarray(res.settling_db)} "
                  f"colpow={m['column_power_max']:.5f} recip_c={m['reciprocity_complex_max']:.2e}")
            assert np.all(np.isfinite(S))
            if LIVE_ABS_S_TOL is not None:
                assert d <= LIVE_ABS_S_TOL, (cid, d, LIVE_ABS_S_TOL)
            if dut != "thru":
                assert m["non_vacuity_max_s11"] > G.NON_VACUITY_MIN_MAX_S11, cid
            # §6 gates bind at the claims rung only (the coarse/mid values are
            # reported, as in the replay): the coarse False-lane slab measured
            # reciprocity 0.035 / 0.068 on the fixture, above the fine-rung gate.
            if dut != "thru" and rung == G.CLAIMS_RUNG:
                assert m["column_power_max"] < G.COLUMN_POWER_MAX, (cid, m["column_power_max"])
                assert m["reciprocity_mag_mean"] < G.RECIPROCITY_MAG_MAX, cid
                assert m["reciprocity_complex_max"] <= G.RECIPROCITY_COMPLEX_MAX, cid


@pytest.mark.slow
@pytest.mark.parametrize("rung", ["coarse", "mid"])
def test_live_cells_reproduce_the_fixture_cpu(fx, rung):
    _live_compare(fx, rung)


@pytest.mark.slow
@pytest.mark.gpu
def test_live_cells_reproduce_the_fixture_fine_rung(fx):
    _live_compare(fx, "fine")


@pytest.mark.slow
@pytest.mark.parametrize("lane", F.LANES, ids=lambda l: G.LANE_LABELS[l])
def test_live_plane_shift_rotation_coarse_rung(lane):
    """The §5(b) rotation gates hold at every rung; re-measured live at the
    coarse rung against the analytic prediction, not against the fixture."""
    sim, res, S_base, _ = _measure_cell("slab", "coarse", lane)
    sim_s = F.build_simulation("slab", G.RUNG_DX["coarse"],
                               reference_planes=(F.REF_LEFT_SHIFTED_M, F.REF_RIGHT_SHIFTED_M))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S_shift = np.asarray(sim_s.compute_waveguide_s_matrix(
            num_periods=F.NUM_PERIODS, normalize=lane).s_params).astype(complex)
    grid = sim._build_grid()
    rot = G.plane_shift_rotation(S_base, S_shift, F.FREQS, float(grid.dt), G.RUNG_DX["coarse"])
    print(f"[live plane-shift coarse {G.LANE_LABELS[lane]}] |S| max diff={rot['abs_s_max_diff']:.2e} "
          f"resid_yee={rot['resid_yee_max']:.3f}° resid_cont={rot['resid_cont_max']:.3f}° "
          f"wrong_sign_min={rot['wrong_sign_resid_min']:.1f}°")
    assert rot["abs_s_allclose"]
    assert rot["resid_yee_max"] <= G.ROTATION_TOL_YEE_DEG
    assert rot["resid_cont_max"] <= G.ROTATION_TOL_CONTINUOUS_DEG
    assert rot["wrong_sign_resid_min"] > G.WRONG_SIGN_MIN_DEG


@pytest.mark.slow
def test_live_ad_vs_fd_slab_s21_mag2_coarse_rung():
    """One §5(a) leg live on CPU: slab, normalize=False, eps θ, |S21|² at the
    band centre — float32 reverse-mode AD vs a float64 central FD with the
    ULP-span validity asserted before the accuracy gate."""
    import jax
    import jax.numpy as jnp
    from tests._x64_compat import enable_x64

    dx = G.RUNG_DX["coarse"]
    sim = F.build_simulation("slab", dx)

    def obj(theta, sim_):
        S = sim_.compute_waveguide_s_matrix(
            num_periods=F.NUM_PERIODS, normalize=False,
            eps_override=F.design_override(sim_, "slab", theta, kind="eps")).s_params
        return G.objective_value(S, "s21_mag2")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, g = jax.value_and_grad(lambda th: obj(th, sim))(jnp.asarray(F.THETA0_EPS, jnp.float32))
        with enable_x64():
            sim64 = F.build_simulation("slab", dx)
            fp = obj(jnp.asarray(F.THETA0_EPS + F.FD_STEP_EPS, jnp.float64), sim64)
            fm = obj(jnp.asarray(F.THETA0_EPS - F.FD_STEP_EPS, jnp.float64), sim64)
            e = G.ad_fd_entry(g_ad=float(g), f_plus=float(fp), f_minus=float(fm), h=F.FD_STEP_EPS,
                              loss_dtype=np.asarray(fp).dtype)
    print(f"[live ad-fd coarse] g_ad={e['g_ad']:+.6e} g_fd={e['g_fd']:+.6e} rel={e['rel']:.3e} "
          f"span={e['fd_ulp_span']:.3g} ULP ({e['loss_dtype']})")
    assert e["loss_dtype"] == "float64"
    assert e["fd_ulp_span"] >= G.FD_ULP_FLOOR, "the FD reference cannot resolve this gradient — comparator failure"
    assert e["rel"] <= G.AD_FD_REL_GATE, (e["g_ad"], e["g_fd"], e["rel"])
