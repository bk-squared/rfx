"""No-solve arithmetic and orchestration falsifiers for the PI's FD redesign."""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F
from tests import _waveguide_chain_battery_gates as G
from tests._x64_compat import enable_x64
from scripts.diagnostics import waveguide_chain_battery_measure as M


def certificate(dut="pec_short", kind="eps"):
    spec = F.fd_stencil(kind)
    dx = G.RUNG_DX["coarse"]
    return {**spec, "dut": dut, "arms": [
        {"theta": th, "min_eps_r": 1.0, "min_sigma_s_per_m": 0.0,
         "courant_ratio": 0.99, "dx_m": dx,
         "dt_s": 0.99 * dx / (F.C0 * math.sqrt(3)),
         "scope": "full_assembled_array", "materials_finite": True}
        for th in spec["thetas"]]}


def leg(dut, lane, kind, objective):
    valid = certificate(dut, kind)
    a, b, c = valid["thetas"]
    def f(x):
        return 2 + 3*x + x*x
    samples = {"g_ad": 3 + 2*a, "h": valid["h"], "loss_dtype": "float64",
               "stencil": valid["stencil"], "f_plus": f(b)}
    samples.update({"f0": f(a), "f_2h": f(c)} if kind == "eps" else {"f_minus": f(c)})
    return {**samples, **G.ad_fd_from_leg(samples), "dut": dut, "lane": lane,
            "theta_kind": kind, "objective": objective, "theta0": a,
            "fd_validity": valid, "dx_m": G.RUNG_DX["coarse"], "rung": "coarse",
            "x64_context": True, "primary_precision": "x64" if lane == "flux" else "float32",
            "forward_identity": {"max_scaled_diff": 0.0}}


def all_legs():
    return [leg(*key) for key in sorted(G.required_ad_leg_keys())]


@pytest.mark.parametrize("theta0", [0.0, 0.7, -0.2])
def test_forward_stencil_differentiates_quadratic(theta0):
    h = 0.05
    def f(x):
        return 2 + 3*x + x*x
    e = G.forward2_ad_fd_entry(g_ad=3+2*theta0, f0=f(theta0), f_plus=f(theta0+h),
                              f_2h=f(theta0+2*h), h=h, loss_dtype=np.float64)
    assert e["g_fd"] == pytest.approx(3+2*theta0, abs=3e-14)
    assert e["verdict"] == "pass"
    assert e["gate"] == 0.05


def test_forward_stencil_has_second_order_error():
    errors = []
    for h in (0.05, 0.025):
        def f(x):
            return 2+x+x**3
        e = G.forward2_ad_fd_entry(g_ad=1.0, f0=f(0), f_plus=f(h), f_2h=f(2*h),
                                  h=h, loss_dtype=np.float64)
        errors.append(abs(e["g_fd"]-1))
    assert errors[0]/errors[1] == pytest.approx(4, rel=1e-9)


def test_weighted_ulp_budget_and_zero_span_skip():
    e = G.forward2_ad_fd_entry(g_ad=0, f0=2, f_plus=2, f_2h=2, h=.05, loss_dtype=np.float64)
    assert e["fd_ulp_budget"] == 8*np.spacing(2.0)
    assert e["fd_ulp_span"] == 0
    assert e["ulp_floor"] == 1e4
    assert e["verdict"] == "skipped_under_ulp_floor"


def test_resolved_wrong_gradient_still_fails_existing_gate():
    e = G.forward2_ad_fd_entry(g_ad=1.051, f0=2, f_plus=2.05, f_2h=2.1, h=.05, loss_dtype=np.float64)
    assert e["fd_ulp_span"] > 1e4
    assert e["verdict"] == "fail"


def test_new_relative_family_has_exactly_fourteen_mandatory_legs():
    rows = all_legs()
    assert len(rows) == 14
    assert G.AD_LEGS["pec_short", "eps"] == ("re_s11", "im_s11")
    G.assert_ad_fd_records(rows, rung="coarse")


@pytest.mark.parametrize("defect", ["missing", "duplicate", "excluded", "stale_stencil", "invalid_arm", "missing_arm", "nonfinite", "wrong_step", "wrong_rung", "changed_dt"])
def test_bad_mandatory_record_blocks(defect):
    rows = all_legs()
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "excluded":
        rows.append(leg("pec_short", "flux", "eps", "s11_mag2"))
    elif defect == "stale_stencil":
        rows[0]["stencil"] = "central2"
    elif defect == "invalid_arm":
        rows[0]["fd_validity"]["arms"][2]["min_eps_r"] = .95
    elif defect == "missing_arm":
        rows[0]["fd_validity"]["arms"].pop()
    elif defect == "nonfinite":
        rows[0]["f_plus"] = np.nan
    elif defect == "wrong_step":
        rows[0]["h"] = .01
    elif defect == "changed_dt":
        for a in rows[0]["fd_validity"]["arms"]:
            a["dt_s"] *= .9
            a["courant_ratio"] *= .9
    else:
        rows[0]["rung"] = "fine"
    with pytest.raises(AssertionError):
        G.assert_ad_fd_records(rows, rung="coarse")


def test_assembly_and_replay_block_missing_mandatory_legs(tmp_path):
    with pytest.raises(AssertionError, match="mandatory legs missing"):
        M.assemble(SimpleNamespace(legs_rung="coarse"), tmp_path, {})
    with pytest.raises(AssertionError, match="mandatory legs missing"):
        G.recompute_verdicts({"schema_version": 4, "cells": [], "ad_vs_fd": []})


@pytest.mark.parametrize("kind", ["eps", "sigma"])
def test_driver_certifies_before_solving_and_reuses_baseline(monkeypatch, kind):
    events = []
    valid = certificate(kind=kind)
    def certify(*args):
        events.append("certified")
        return valid
    monkeypatch.setattr(F, "assert_fd_stencil_admissible", certify)
    monkeypatch.setattr(M, "_override_kw", lambda sim, dut, kind, theta: {"theta": theta})
    class Sim:
        def compute_waveguide_s_matrix(self, *, theta, **kw):
            assert events[0] == "certified"
            th = float(theta)
            events.append(th)
            return SimpleNamespace(s_params=np.full((2, 2, len(F.FREQS)), 2+3*th+th*th, dtype=np.complex128))
    with enable_x64():
        baseline, samples, recorded = M.measure_fd_samples(Sim(), "pec_short", kind, False, ("re_s11",))
    assert events == ["certified", *valid["thetas"]]
    assert recorded == valid
    assert baseline[0, 0, 0] == 2+3*valid["theta0"]+valid["theta0"]**2
    assert G.ad_fd_from_leg({"g_ad": 3+2*valid["theta0"], **samples["re_s11"]})["verdict"] == "pass"


def test_invalid_arm_blocks_before_first_solve(monkeypatch, tmp_path):
    def reject(*args):
        valid = certificate("slab")
        valid["arms"][2]["min_eps_r"] = .95
        F.assert_fd_validity(valid)
    monkeypatch.setattr(F, "assert_fd_stencil_admissible", reject)
    monkeypatch.setattr(F, "build_simulation", lambda *a: object())
    with pytest.raises(AssertionError, match="FD configuration defect BLOCKED"):
        M.stage_ad_fd(SimpleNamespace(overwrite=False), tmp_path, "coarse", {})
    assert not list(tmp_path.iterdir())


def test_stale_cached_ad_stage_blocks_instead_of_skipping(tmp_path):
    path = M.ad_fd_path(tmp_path, "slab", "false", "eps")
    path.write_text(json.dumps({"schema_version": 3, "legs": []}))
    with pytest.raises(AssertionError, match="BLOCKED: stale"):
        M.stage_ad_fd(SimpleNamespace(overwrite=False), tmp_path, "coarse", {})


def test_nonfinite_solved_arm_blocks(monkeypatch):
    monkeypatch.setattr(F, "assert_fd_stencil_admissible", lambda *a: certificate())
    monkeypatch.setattr(M, "_override_kw", lambda *a: {})
    sim = SimpleNamespace(compute_waveguide_s_matrix=lambda **kw: SimpleNamespace(s_params=np.array([np.nan])))
    with enable_x64(), pytest.raises(AssertionError, match="BLOCKED: nonfinite S"):
        M.measure_fd_samples(sim, "pec_short", "eps", False, ("re_s11",))


def test_complete_driver_stage_writes_current_family_and_reuses_cache(monkeypatch, tmp_path):
    import jax.numpy as jnp

    class Sim:
        def _build_grid(self):
            return object()

        def compute_waveguide_s_matrix(self, *, theta=0.0, **kw):
            value = 2 + 3*theta + theta*theta
            S = jnp.ones((2, 2, len(F.FREQS)), dtype=jnp.result_type(value, 1j)) * value
            return SimpleNamespace(s_params=S)

    monkeypatch.setattr(F, "build_simulation", lambda *a: Sim())
    monkeypatch.setattr(F, "assert_fd_stencil_admissible", lambda sim, dut, kind: certificate(dut, kind))
    monkeypatch.setattr(M, "_override_kw", lambda sim, dut, kind, theta: {"theta": theta})
    monkeypatch.setattr(M, "_checkpoint_segments", lambda grid: None)
    monkeypatch.setattr(M, "run_smatrix", lambda sim, lane, **kw:
                        (None, np.asarray(sim.compute_waveguide_s_matrix(**kw).s_params), None, None, None))
    args = SimpleNamespace(overwrite=False)
    M.stage_ad_fd(args, tmp_path, "coarse", {})
    files = sorted(tmp_path.glob("ad_fd__*.json"))
    assert len(files) == 6
    rows = [row for path in files for row in json.loads(path.read_text())["legs"]]
    G.assert_ad_fd_records(rows, rung="coarse")
    before = {path: path.read_bytes() for path in files}
    monkeypatch.setattr(F, "build_simulation", lambda *a: pytest.fail("valid cache must not solve again"))
    M.stage_ad_fd(args, tmp_path, "coarse", {})
    assert all(path.read_bytes() == content for path, content in before.items())
