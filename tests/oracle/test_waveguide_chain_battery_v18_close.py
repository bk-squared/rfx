"""WR-90 chain battery: live cells, plane rotation, AD-vs-FD and shift-sign control.

Live cells use the #1012 CPML half-cell artifact; the shift-sign falsifier
uses the enforced schema-4 realized-PEC / forward2 artifact. The live
AD-vs-FD leg and cross-backend tolerance were moved unchanged from the
original battery. Historical artifacts and pre-declarations remain in place.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F
from tests import _waveguide_chain_battery_gates as G
from tests import _waveguide_chain_battery_enforcement as E
from tests._gate_policy import gate_from_envelope

# Live-layer tolerance on |S_live − S_fixture| (absolute, per entry), derived
# from the measured cross-backend envelope through the shared policy: the 12
# coarse + mid cells re-measured on this box (CPU, jax 0.6.2) against the
# fixture (VESSL 369367257823, gpu-rtx4090, jax 0.4.33) differed by at most
# 5.000e-6 (pec_short-coarse-flux; the others 9.6e-7 .. 2.6e-6). The quantum
# is 1e-4 so the pin is not a 2x knife edge on a float32 reassociation
# envelope; the fine rung (4x the steps) is compared on the GPU lane only.
LIVE_ABS_S_ENVELOPE = 5.000e-6
LIVE_ABS_S_TOL: float | None = gate_from_envelope(LIVE_ABS_S_ENVELOPE, quantum=10000)   # 1e-4


def _measure_cell(dut: str, rung: str, lane, num_periods: float = F.NUM_PERIODS):
    sim = F.build_simulation(dut, G.RUNG_DX[rung])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = sim.preflight()
        res = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=lane)
    return sim, res, np.asarray(res.s_params).astype(complex), sorted(i.code for i in report)


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture_v18_close.json"
LIVE_CELLS_FIXTURE = FIXTURE.with_name("fixture_1012_cpml_half_cell_run369367264100.json")
LIVE_CELLS_SHA256 = "90ac2c88a7686baa888f9b7de5f0d145a81e89d3d153a25a235bf45e463ee74b"


@pytest.fixture(scope="module")
def live_fx() -> dict:
    # A missing live reference is an ingest defect, not a reason to skip.
    return E.load_enforced_fixture()


@pytest.fixture(scope="module")
def live_cells_fx() -> dict:
    # #1012 live cells include the slab re-freeze; historical replay stays on #931.
    raw = LIVE_CELLS_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == LIVE_CELLS_SHA256, "live cell record SHA-256 mismatch"
    return json.loads(raw)


def _cell(fx_: dict, dut: str, rung: str, lane: str) -> dict:
    return next(c for c in fx_["cells"] if (c["dut"], c["rung"], c["lane"]) == (dut, rung, lane))


def _live_compare(fx_, rung: str):
    # Every cell is measured and printed before any assertion, so one red cell
    # does not hide the cells after it.
    cells = []
    for dut in F.DUTS:
        for lane in F.LANES:
            label = G.LANE_LABELS[lane]
            stored = _cell(fx_, dut, rung, label)
            sim, res, S, codes = _measure_cell(dut, rung, lane)
            S0 = G.s_from_json(stored["s_params"])
            d = float(np.max(np.abs(S - S0)))
            m = G.cell_metrics(S)
            print(f"[live {dut}-{rung}-{label}] max|S_live-S_fixture|={d:.3e} "
                  f"settling={np.asarray(res.settling_db)} colpow={m['column_power_max']:.5f} "
                  f"recip_c={m['reciprocity_complex_max']:.2e}")
            cells.append((dut, label, stored, S, codes, d))
    for dut, label, stored, S, codes, d in cells:
        assert codes == sorted(f["code"] for f in stored["preflight"]), (dut, rung, label, codes)
        assert np.all(np.isfinite(S))
        assert d <= LIVE_ABS_S_TOL, (dut, rung, label, d, LIVE_ABS_S_TOL,
                                     "cross-backend excess is reported with both backends' "
                                     "numbers, never absorbed by widening the pin (§5.11)")


@pytest.mark.slow
@pytest.mark.parametrize("rung", ["coarse", "mid"])
def test_live_cells_reproduce_the_fixture_cpu(live_cells_fx, rung):
    """§5.11 row 1 against the realized-PEC contract-build measurement."""
    _live_compare(live_cells_fx, rung)


@pytest.mark.slow
@pytest.mark.gpu
def test_live_cells_reproduce_the_fixture_fine_rung(live_cells_fx):
    """§5.11 row 2, on the GPU lane (the fine rung is 4x the steps)."""
    _live_compare(live_cells_fx, "fine")


@pytest.mark.slow
@pytest.mark.parametrize("lane", F.LANES, ids=lambda l: G.LANE_LABELS[l])
def test_live_plane_shift_rotation_coarse_rung(lane):
    """§5.11 row 3: the coarse rung against physics, not against the artifact,
    with the sign-discriminating pair (predicted 0.512° / 0.669° / 64.07°)."""
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


def test_shift_sign_falsifier_uses_the_measured_operator_and_has_a_control(live_fx):
    control = E.replay_shift_sign_refute(live_fx, flip_sign=False)
    refute = E.replay_shift_sign_refute(live_fx)
    for case in control["per_case"]:
        assert case["max_abs_diff_from_stored_shift"] <= LIVE_ABS_S_TOL
    assert control["abs_s_still_invariant"]
    assert refute["abs_s_still_invariant"]
    stored = live_fx["plane_shift"]["cheap_refute"]
    for case, expected in zip(refute["per_case"], stored["per_case"], strict=True):
        assert (case["dut"], case["lane"]) == (expected["dut"], expected["lane"])
        assert case["entries_measurable"] == expected["entries_measurable"]
        for entry, residual in case["resid_yee_per_entry"].items():
            if residual is None:
                assert expected["resid_yee_per_entry"][entry] is None
            else:
                assert residual == pytest.approx(expected["resid_yee_per_entry"][entry])
                assert residual > G.WRONG_SIGN_MIN_DEG
    changed = deepcopy(live_fx)
    key = "cheap_refute_flip_shift_sign"
    changed["plane_shift"]["cheap_refute"] = refute
    assert G.recompute_verdicts(changed)[key] == "pass"
    changed["plane_shift"]["cheap_refute"] = control
    assert G.recompute_verdicts(changed)[key] == "fail"
    del changed["plane_shift"]["cheap_refute"]
    assert G.recompute_verdicts(changed)[key] == "owed"


def test_live_ad_vs_fd_slab_s21_mag2_coarse_rung():
    """One §5(a) leg live on CPU: slab, normalize=False, eps θ, |S21|² at the
    band centre — float32 reverse-mode AD vs a second-order forward x64 FD.
    Full-array arm validity blocks before AD; the ULP floor precedes accuracy."""
    import jax
    import jax.numpy as jnp
    from tests._x64_compat import enable_x64

    dx = G.RUNG_DX["coarse"]
    sim = F.build_simulation("slab", dx)
    F.assert_fd_stencil_admissible(sim, "slab", "eps")

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
            from scripts.diagnostics.waveguide_chain_battery_measure import measure_fd_samples
            _, samples, _ = measure_fd_samples(sim64, "slab", "eps", False, ("s21_mag2",))
            e = G.ad_fd_from_leg({"g_ad": float(g), **samples["s21_mag2"]})
    print(f"[live ad-fd coarse] g_ad={e['g_ad']:+.6e} g_fd={e['g_fd']:+.6e} rel={e['rel']:.3e} "
          f"span={e['fd_ulp_span']:.3g} ULP ({e['loss_dtype']})")
    assert e["loss_dtype"] == "float64"
    assert e["fd_ulp_span"] >= G.FD_ULP_FLOOR, "the FD reference cannot resolve this gradient — comparator failure"
    assert e["rel"] <= G.AD_FD_REL_GATE, (e["g_ad"], e["g_fd"], e["rel"])
