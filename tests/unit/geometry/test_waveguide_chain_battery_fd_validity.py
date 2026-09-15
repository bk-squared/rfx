"""Build-only falsifiers for the amended chain battery's FD configuration.

No FDTD steps run here. A resolvable comparison and an admissible configuration
are separate requirements: invalid material or Courant arms must block.
"""

from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pytest

from tests import _waveguide_chain_battery_fixture as F
from tests._x64_compat import enable_x64


@pytest.fixture(scope="module")
def sims():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {(dut, dx): F.build_simulation(dut, dx)
                for dut in F.DUTS for dx in F.DX_LADDER}


def _record(sim, dut, kind, *, theta0=None, h=None):
    spec = F.fd_stencil(kind, theta0=theta0, h=h)
    return {**spec, "dut": dut,
            "arms": [F.fd_arm_validity(sim, dut, kind, theta)
                     for theta in spec["thetas"]]}


@pytest.mark.parametrize("dut", F.DUTS)
@pytest.mark.parametrize("dx", F.DX_LADDER)
def test_eps_stencil_all_duts_and_rungs_has_global_certificate(sims, dut, dx):
    record = F.assert_fd_stencil_admissible(sims[dut, dx], dut, "eps")
    assert record["stencil"] == "forward2"
    assert record["theta0"] == 0.0
    assert [arm["theta"] for arm in record["arms"]] == [0.0, 0.05, 0.1]
    for arm in record["arms"]:
        print(f"[{dut} dx={dx:g} theta={arm['theta']:g}] "
              f"global min_eps_r={arm['min_eps_r']:.9f} "
              f"Courant ratio={arm['courant_ratio']:.9f}")
        assert arm["scope"] == "full_assembled_array"
        assert arm["min_eps_r"] == 1.0
        assert arm["courant_ratio"] == pytest.approx(0.99, abs=1e-12)


@pytest.mark.parametrize("dx", F.DX_LADDER)
def test_sigma_central_stencil_all_arms_are_admissible(sims, dx):
    record = F.assert_fd_stencil_admissible(sims["pec_short", dx], "pec_short", "sigma")
    assert record["stencil"] == "central2"
    assert [arm["theta"] for arm in record["arms"]] == [
        F.THETA0_SIGMA_S_PER_M,
        F.THETA0_SIGMA_S_PER_M + F.FD_STEP_SIGMA_S_PER_M,
        F.THETA0_SIGMA_S_PER_M - F.FD_STEP_SIGMA_S_PER_M,
    ]
    assert all(arm["min_sigma_s_per_m"] >= 0 for arm in record["arms"])
    assert all(arm["courant_ratio"] == pytest.approx(0.99, abs=1e-12)
               for arm in record["arms"])


@pytest.mark.parametrize("h,expected", [
    (0.05, 1.015718569),
    (0.02, 1.000051019),
    (0.0199, 1.000000000),
    (0.01, 0.994987437),
])
def test_historical_negative_arm_courant_table(sims, h, expected):
    sim = sims["pec_short", F.DX_COARSE]
    with enable_x64():
        arm = F.fd_arm_validity(sim, "pec_short", "eps", -h)
        window_ratio = F.eps_fd_minus_window_interior_courant_ratio(sim, "pec_short", h=h)
    # The diagnostic table is real arithmetic. Production assembly explicitly
    # stores materials in float32 even inside enable_x64; the GLOBAL certificate
    # must read that actual override, not silently replace it by ideal 1-h.
    actual_min = float(np.float32(1.0) + np.float32(-h))
    assert arm["min_eps_r"] == actual_min
    assert arm["courant_ratio"] == pytest.approx(0.99 / math.sqrt(actual_min), abs=1e-12)
    assert window_ratio == pytest.approx(expected, abs=5e-10)


def test_window_diagnostic_does_not_certify_global_slab(sims):
    sim = sims["slab", F.DX_COARSE]
    with enable_x64():
        window_ratio = F.eps_fd_minus_window_interior_courant_ratio(sim, "slab")
        global_baseline = F.fd_arm_validity(sim, "slab", "eps", 0.0)
    assert window_ratio == pytest.approx(0.99 / math.sqrt(3.95), abs=1e-12)
    assert global_baseline["courant_ratio"] == pytest.approx(0.99, abs=1e-12)
    assert global_baseline["min_eps_r"] == 1.0


def test_numerically_stable_eps_below_declared_material_scope_blocks(sims):
    with enable_x64():
        record = _record(sims["pec_short", F.DX_COARSE], "pec_short", "eps", theta0=-0.01)
    assert record["arms"][0]["courant_ratio"] < 1
    assert record["arms"][0]["min_eps_r"] < 1
    with pytest.raises(AssertionError, match="BLOCKED.*min_eps_r=0.99"):
        F.assert_fd_validity(record)


def test_courant_boundary_blocks_even_with_admissible_material(sims):
    record = F.assert_fd_stencil_admissible(sims["thru", F.DX_COARSE], "thru", "eps")
    arm = record["arms"][0]
    arm["dt_s"] = arm["dx_m"] / (F.C0 * math.sqrt(3.0))
    arm["courant_ratio"] = 1.0
    with pytest.raises(AssertionError, match="BLOCKED.*global Courant ratio=1.000000000"):
        F.assert_fd_validity(record)


def test_negative_sigma_arm_blocks(sims):
    record = _record(sims["pec_short", F.DX_COARSE], "pec_short", "sigma", theta0=0.0)
    assert record["arms"][-1]["min_sigma_s_per_m"] < 0
    with pytest.raises(AssertionError, match="BLOCKED.*min_sigma=-"):
        F.assert_fd_validity(record)


@pytest.mark.parametrize("kind,bad_value", [
    ("eps", 0.5), ("eps", np.nan), ("eps", np.inf),
    ("sigma", -0.001), ("sigma", np.nan), ("sigma", np.inf),
])
def test_invalid_material_outside_design_window_blocks(sims, monkeypatch, kind, bad_value):
    sim = sims["slab", F.DX_COARSE]
    lo, hi = F.design_region_index_range(sim, "slab")
    assert not lo <= 0 < hi, "the injected defect must be outside the design window"
    original = F.design_override

    def corrupt_override(sim_, dut, theta, *, kind="eps"):
        result = np.array(original(sim_, dut, theta, kind=kind), copy=True)
        result[0, 0, 0] = bad_value
        return result

    monkeypatch.setattr(F, "design_override", corrupt_override)
    with pytest.raises(AssertionError, match="FD configuration defect BLOCKED"):
        F.assert_fd_stencil_admissible(sim, "slab", kind)


@pytest.mark.parametrize("missing_index", [0, 1, 2])
def test_missing_baseline_or_perturbed_arm_blocks(sims, missing_index):
    record = F.assert_fd_stencil_admissible(sims["pec_short", F.DX_COARSE], "pec_short", "eps")
    record = copy.deepcopy(record)
    del record["arms"][missing_index]
    with pytest.raises(AssertionError, match="FD configuration defect: missing/wrong arms"):
        F.assert_fd_validity(record)
