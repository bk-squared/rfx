"""PEC-cube RCS gates: rfx FDTD vs independent Bempp BEM (campaign Lane 3).

Extends the independent integral-equation cross-check (Lane 1, sphere) to a shape
with NO closed form. The Lane-1-validated Bempp EFIE harness (reproduces exact
Mie <=0.15 dB) is the arbiter; here it is shown CONVERGED at every H-plane angle
(main-vs-fine mesh), which is what licenses it to referee both the backscatter and
the oblique bins.

Findings this locks:
  * rfx NEAR-BACKSCATTER RCS agrees with independent BEM on a non-closed-form
    shape (an axis-aligned cube is grid-perfect in FDTD -> no staircase).
  * rfx's FORWARD-OBLIQUE bistatic bins read high -- the documented bistatic
    contamination (issue #280) -- confirmed on a SECOND shape by a non-FDTD
    method (Bempp is converged there, so the gap is rfx-side). RECORDED, not
    gated -- matching the sphere's non-gated bistatic posture.

All dB distances are recomputed here from the committed raw sigma arrays (the
producer's derived dB values are not trusted). Additive: no existing gate touched.
humble-crossval: distances are rfx-centric / method-distance facts; Bempp is the
converged arbiter, not a verdict that rfx is "wrong" in a documented non-validated
region.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/rcs_cube_bem/fixture.json"


@pytest.fixture(scope="module")
def fx():
    return json.loads(_FIXTURE.read_text())


def _phi(fx):
    return np.array(fx["phi_deg"])


def _dist_db(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 10.0 * np.log10(np.maximum(a, 1e-30) / np.maximum(b, 1e-30))


def test_bempp_arbiter_converged_at_all_angles(fx):
    """LOAD-BEARING: the Bempp arbiter is mesh-converged at EVERY H-plane angle
    (main-vs-fine <= 0.05 dB), recomputed from raw sigmas. This is what makes its
    referee verdict trustworthy in the oblique bins, not only at backscatter."""
    bm = np.array(fx["bempp"]["bistatic_sigma_m2"])
    bf = np.array(fx["bempp"]["bistatic_sigma_m2_fine"])
    conv = np.abs(_dist_db(bm, bf))
    assert conv.max() <= 0.05, (conv.max(), _phi(fx)[int(np.argmax(conv))])


def test_backscatter_rfx_agrees_with_bempp(fx):
    """rfx monostatic (phi=pi backscatter) agrees with independent BEM on a
    non-closed-form shape, within 1.0 dB (measured ~0.42)."""
    rfx = np.array(fx["rfx"]["bistatic_sigma_m2"])
    bm = np.array(fx["bempp"]["bistatic_sigma_m2"])
    d_back = float(_dist_db(rfx[-1], bm[-1]))
    assert abs(d_back) <= 1.0, d_back
    # cross-check the shipped monostatic_dbsm matches the phi=pi bistatic bin
    assert np.isclose(fx["rfx"]["monostatic_dbsm"],
                      10 * np.log10(max(rfx[-1], 1e-30)), atol=0.5)


def test_near_backscatter_region_agrees(fx):
    """The near-backscatter region (phi >= 135 deg) -- where rfx RCS is validated --
    agrees with BEM within 1.5 dB (measured max ~1.06)."""
    phi = _phi(fx)
    rfx = np.array(fx["rfx"]["bistatic_sigma_m2"])
    bm = np.array(fx["bempp"]["bistatic_sigma_m2"])
    d = np.abs(_dist_db(rfx, bm))[phi >= 135.0]
    assert d.max() <= 1.5, d.max()


def test_forward_oblique_contamination_recorded_and_attributed(fx):
    """The forward-oblique discrepancy is DISCLOSED (not hidden), is genuinely
    large (the documented issue-#280 contamination), and is attributable to rfx
    because the arbiter is converged there. This gate locks the honest record; it
    does NOT gate rfx's non-validated bistatic region.

    NOTE: this is a REVERSE gate (asserts the gap is LARGE) -- a regression-lock on
    a known-bad state, not an upper bound. If issue #280 (forward-face TFSF/NTFF
    contamination) is ever fixed, this assertion will FAIL and must be updated to
    reflect the improved agreement -- that is the intended signal, not a defect."""
    phi = _phi(fx)
    rfx = np.array(fx["rfx"]["bistatic_sigma_m2"])
    bm = np.array(fx["bempp"]["bistatic_sigma_m2"])
    bf = np.array(fx["bempp"]["bistatic_sigma_m2_fine"])
    dist = np.abs(_dist_db(rfx, bm))
    obl = (phi >= 15.0) & (phi <= 120.0)
    # disclosed: the fixture records a materially large oblique gap
    assert dist[obl].max() > 2.0, dist[obl].max()
    # attributable: the arbiter is converged in that same region (rfx-side gap)
    conv_obl = np.abs(_dist_db(bm, bf))[obl]
    assert conv_obl.max() <= 0.05, conv_obl.max()


def test_physical_optics_order_of_magnitude(fx):
    """Order-of-magnitude PO sanity: at kL~3.77 (resonant region) the flat-plate PO
    sigma_PO = 4pi L^4/lam^2 is only the kL->inf asymptote, so both rfx and Bempp
    backscatter should sit at ORDER 2x PO (measured ~2.2-2.4). PO is recomputed here
    from geometry (not the producer scalar). NOTE: the rfx-vs-Bempp backscatter/PO
    *ratio* agreement is arithmetically the backscatter match (PO cancels) already
    gated by test_backscatter_rfx_agrees_with_bempp; the only independent PO content
    is this order-of-magnitude bracket."""
    g = fx["geometry"]
    po = 4 * np.pi * g["L_m"] ** 4 / g["lambda_m"] ** 2
    rfx_back = np.array(fx["rfx"]["bistatic_sigma_m2"])[-1]
    bm_back = np.array(fx["bempp"]["bistatic_sigma_m2"])[-1]
    assert 1.0 < rfx_back / po < 4.0, rfx_back / po
    assert 1.0 < bm_back / po < 4.0, bm_back / po
