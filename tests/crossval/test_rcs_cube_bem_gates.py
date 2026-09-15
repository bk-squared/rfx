"""PEC-cube RCS gates: rfx FDTD vs independent Bempp BEM (campaign Lane 3).

Extends the independent integral-equation cross-check (Lane 1, sphere) to a shape
with NO closed form. The Lane-1-validated Bempp EFIE harness (reproduces exact
Mie <=0.15 dB) is the arbiter; here it is shown CONVERGED at every H-plane angle
(main-vs-fine mesh), which is what licenses it to referee both the backscatter and
the oblique bins.

Findings this locks:
  * rfx NEAR-BACKSCATTER RCS agrees with independent BEM on a non-closed-form
    shape (an axis-aligned cube is grid-perfect in FDTD -> no staircase). That
    claim is load-bearing for the gates below and used to be untested; it is a
    check now (``test_the_cube_is_a_whole_number_of_cells_on_its_own_mesh``),
    and the check is precise about what "grid-perfect" means here: the EXTENT
    is 18 cells per axis to 0.012 of a cell, but the cube is not registered on
    the node lattice, so WHICH 18 cells depends on the sampler.

#931 SCOPE (measured, not assumed): this lane does not go through the lattice
ownership contract at all. The producer builds its metal with the low-level
``rasterize(grid, [(cube, 1.0, PEC_SIGMA)])`` — a sigma = 1e7 CELL FILL, which
design note §1.8 explicitly fences out of the contract as a lossy-volume model.
So the committed rfx sigmas do not move, the gates below are not re-derived,
and no re-run is needed. The two models are pinned as DIFFERENT rather than
quietly equated (§1.8's own requirement); see
``test_the_sigma_fill_and_the_pec_contract_are_not_the_same_object``.
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

_FIXTURE = Path(__file__).resolve().parents[2] / "tests/fixtures/rcs_cube_bem/fixture.json"


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


# --------------------------------------------------------------------------- #
# #931: the geometry claim this file's gates rest on, and the model fence
# --------------------------------------------------------------------------- #

def _producer():
    """The fixture's own producer module, imported without running it."""
    import importlib.util
    path = _FIXTURE.parent / "generate.py"
    spec = importlib.util.spec_from_file_location("_rcs_cube_generate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_cube_is_a_whole_number_of_cells_on_its_own_mesh(fx):
    """Turn "an axis-aligned cube is grid-perfect" into a check.

    The gates in this file are justified by that sentence — if the cube were
    staircased, a 0.42 dB agreement with BEM would be a different claim. The
    producer's own constants say: L = 30 mm, dx = lambda/RES = 1.6655 mm, so
    L/dx = 18.012, i.e. eighteen whole cells to about a hundredth of a cell,
    and equal on all three axes. Read from the producer, not retyped, so a
    mesh change reds this before it reaches the physics.
    """
    m = _producer()
    dx = m.LAM / m.RES
    n = m.L / dx
    assert abs(n - round(n)) < 0.02, (
        f"the cube is {n:.4f} cells per side on its own mesh — no longer a "
        "whole number of cells, so 'grid-perfect, no staircase' is false and "
        "the gates below need a fresh justification")
    assert round(n) == 18
    assert fx["geometry"]["L_m"] == pytest.approx(m.L)
    assert fx["rfx"]["dx_m"] == pytest.approx(dx, rel=1e-9)


def test_the_sigma_fill_and_the_pec_contract_are_not_the_same_object():
    """#931 §1.8: a sigma fill and a realized PEC volume must not be equated.

    Design note §1.8 fences ``rasterize(..., sigma=1e7)`` out of the ownership
    contract as a LOSSY VOLUME model and asks for a test that the two models
    are not silently equated. Measured here on this file's own cube and on the
    cv16 sphere, at the cube fixture's mesh:

      cube    node-sampled (rasterize) 18x18x18 = 5832 cells
              centre-sampled (contract) 18x18x18 = 5832 cells
              SAME COUNT, DIFFERENT CELLS — the cube's faces sit 0.012 of a
              cell off the node lattice, so the two samplers pick different
              blocks of eighteen.
      sphere  node-sampled 3023 cells, centre-sampled 3082 cells — 59 rim
              cells apart, because a curved surface has no registration to
              share.

    The equality a reader might assume ("PEC is PEC") is therefore false, and
    that is the point: cv16 and this cube measure a conductivity fill, not the
    realized edge set, and their numbers are not evidence about the contract.
    """
    import numpy as _np
    from rfx.geometry.csg import Box, Sphere, rasterize
    from rfx.geometry.rasterize_grid import (centres_from_uniform_grid,
                                             pec_volume_cell_mask)
    from rfx.grid import Grid

    m = _producer()
    dx = m.LAM / m.RES
    grid = Grid(freq_max=m.F0 * 1.5, domain=(m.DOMAIN,) * 3, dx=dx,
                cpml_layers=m.CPML)
    c = m.DOMAIN / 2
    centres = centres_from_uniform_grid(grid)

    cube = Box(corner_lo=(c - m.L / 2,) * 3, corner_hi=(c + m.L / 2,) * 3)
    _eps, sigma = rasterize(grid, [(cube, 1.0, m.PEC_SIGMA)])
    filled = _np.asarray(sigma) > 0
    owned = _np.asarray(pec_volume_cell_mask(cube, centres), dtype=bool)

    def runs(mask):
        return tuple(int(mask.any(axis=tuple(k for k in range(3) if k != ax)).sum())
                     for ax in range(3))

    assert runs(filled) == runs(owned) == (18, 18, 18)
    assert int(filled.sum()) == int(owned.sum()) == 18 ** 3
    assert not _np.array_equal(filled, owned), (
        "the sigma fill and the contract's cell ownership now select the SAME "
        "cells for this cube. That may be an improvement, but §1.8 fences the "
        "two models apart on purpose — re-read the fence before treating this "
        "lane's numbers as evidence about PEC realization")

    sphere = Sphere(center=(c, c, c), radius=0.015)
    _e2, s2 = rasterize(grid, [(sphere, 1.0, m.PEC_SIGMA)])
    filled_s = _np.asarray(s2) > 0
    owned_s = _np.asarray(pec_volume_cell_mask(sphere, centres), dtype=bool)
    assert int(filled_s.sum()) != int(owned_s.sum()), (
        "node- and centre-sampling now agree on a curved body; the cv16 / "
        "rcs_scattering split between the audited sphere and the measured "
        "sphere has closed and both should be re-read")
