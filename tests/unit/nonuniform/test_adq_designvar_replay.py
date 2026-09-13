"""Replay of the NU design-variable AD witness (no FDTD).

Frozen rules: docs/design_notes/20260913_nu_ad_designvar_predeclaration.md.
Every verdict is re-derived from the stored loss ladders with the SAME judges
the instrument used, and must equal the recorded one -- including the FIRED
and INCONCLUSIVE ones, which are recorded results, not failures to hide.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from validation.research.multiband_nu import adq_designvar as adq

RESULTS = Path(__file__).resolve().parents[3] / "validation/research/multiband_nu/results"


def _load(arm):
    p = RESULTS / f"adq_{arm}.json"
    if not p.exists():
        pytest.skip(f"adq arm {arm} not measured yet")
    d = json.loads(p.read_text())
    assert "instrument_error" not in d, d.get("instrument_error")
    return d


def test_frozen_constants():
    assert adq.HS == tuple(2.0 ** -k for k in range(17, 2, -1))
    assert adq.N_QUANTA == 32 and adq.RHO_BAND == (2.0, 8.0)
    assert adq.COUNTS == (20, 4, 20, 20)
    np.testing.assert_array_equal(adq.P0, [0.014, 0.002, 0.014, 0.014, 4.3, 3.0, 4.3, 1.0])


def test_judges_selfcheck_live():
    """The judges must separate a true slope from a 10 %-wrong one (fast, no FDTD)."""
    sc = adq.selfcheck()
    assert sc["all_pass"], sc
    assert 1.8 <= sc["true_quadratic"]["r1_slope"] <= 2.2
    assert sc["wrong_by_10pct"]["r1_slope"] < 1.3


@pytest.mark.parametrize("arm", ["stack_l1", "stack_l2", "revert_l1_nodt"])
def test_selfcheck_recorded_before_arm(arm):
    d = _load(arm)
    assert d["selfcheck"]["all_pass"] is True


def _points(steps):
    return [{"h": s["h"], "loss_plus": s["loss_plus"], "loss_minus": s["loss_minus"]} for s in steps]


@pytest.mark.parametrize("arm", ["stack_l1", "stack_l2"])
def test_stack_verdicts_replay(arm):
    d = _load(arm)
    loss0 = d["loss0"]
    for name, r in d["directions"].items():
        pts = _points(r["fd"]["steps"])
        assert [p["h"] for p in pts] == list(adq.HS)
        order = adq.fit_order(pts, loss0, r["ad_relative"])
        fd = adq.fd_budget(pts, r["ad_relative"], adq.P0[adq.NAMES.index(name)])
        assert order["verdict"] == r["order"]["verdict"], name
        assert fd["verdict"] == r["fd"]["verdict"], name
        for lab in ("R0", "R1"):
            if lab in r["order"]:
                assert abs(order[lab]["slope"] - r["order"][lab]["slope"]) <= 1e-12, (name, lab)


@pytest.mark.parametrize("arm", ["stack_l1", "stack_l2"])
def test_chain_rule_and_coverage_replay(arm):
    d = _load(arm)
    jac, g = np.asarray(d["jacobian"]), np.asarray(d["g_cell"])
    np.testing.assert_allclose(jac.T @ g, d["chain_rule"], rtol=1e-12, atol=0)
    # J^T g (cell chain) vs the direct parameter gradient: same f32 solve, two AD routes.
    np.testing.assert_allclose(d["chain_rule"], d["g_param"], rtol=1e-4, atol=1e-5)
    for key, cols in (("coverage_raw_m_eps", slice(None)),):
        c = adq.coverage(g, jac[:, cols])
        assert abs(c["fraction"] - d[key]["fraction"]) <= 1e-12
    held = [adq.NAMES.index(n) for n in d["verified_controls"]]
    c = adq.coverage(g, jac[:, held])
    assert abs(c["fraction"] - d["coverage_verified_raw_m_eps"]["fraction"]) <= 1e-12


def test_induced_directions_constant_on_tied_set():
    """dt is differentiable along a control only if it moves every tied-minimum
    cell by the same amount; the stack map must guarantee it."""
    d = _load("stack_l1")
    jac, tied = np.asarray(d["jacobian"]), d["tied_cells_z"]
    for i in range(8):
        assert np.ptp(jac[tied, i]) == 0.0, adq.NAMES[i]


def test_revert_proof_has_power_against_the_dt_path():
    """With dt cut out of the gradient, h_thin must FIRE; the core controls, whose
    dt-path share is exactly zero, must return the true-gradient arm's verdicts."""
    d = _load("revert_l1_nodt")
    assert d["forward_values_identical"] is True
    share = dict(zip(adq.NAMES[:4], d["dt_path_share"]))
    assert share["h_thin"] > 0.05
    for n in ("h_core_left", "h_core_right", "h_air"):
        assert share[n] == 0.0, n
    r = d["directions"]["h_thin"]
    assert r["order"]["verdict"] == "FIRED" and r["order"]["R1"]["slope"] < 1.3
    true = _load("stack_l1")["directions"]
    for n in ("h_core_left", "h_core_right", "h_air"):
        assert d["directions"][n]["order"]["verdict"] == true[n]["order"]["verdict"], n


def test_h_air_fire_is_on_the_upper_side_of_the_band():
    """Recorded result, not a pass: h_air's order window FIRED with R1 slope ABOVE
    2.2. A wrong gradient drives R1 toward slope 1, never above 2; the upper-side
    fire is higher-order curvature in a 4-point coarse-end window, and its dt-path
    share is exactly zero (revert arm). Pinned so a change in either is noticed."""
    d = _load("stack_l1")["directions"]["h_air"]["order"]
    assert d["verdict"] == "FIRED" and d["R1"]["slope"] > 2.2 and d["points"] == 4
