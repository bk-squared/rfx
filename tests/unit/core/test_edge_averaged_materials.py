"""The material an E component sees is the mean over its four cells (#1210).

A Yee E component lies on an edge shared by four primal cells, so its eps and
sigma are the tangential averages over those four. The tests below read the
coefficient the solver actually multiplies the field by -- not the helper's
output alone -- so a lane that quietly went back to the owning cell's value
turns them red.

The mutation control is the one that matters: the helper is still called, with
the same arguments, and only the mean is replaced by the owning cell's value.
A test that compares two calls of the same helper would survive that.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import (
    EPS_0,
    FDTDState,
    MaterialArrays,
    curl_h,
    e_update_coeffs,
    edge_averaged_e_update_coeffs,
    edge_averaged_materials,
    precompute_coeffs,
    update_e,
)
from rfx.boundaries.pec import _volume_occupancy_masks

DT = 1.0e-12
DX = 1.0e-3
SHAPE = (6, 7, 8)


def _half_space_eps(eps_lo=1.0, eps_hi=9.0, axis=2, cut=4):
    """eps_r = eps_lo below cell ``cut`` along ``axis``, eps_hi from it up."""
    idx = np.indices(SHAPE)[axis]
    return jnp.asarray(np.where(idx < cut, eps_lo, eps_hi).astype(np.float32))


def _cell_owned_coeffs(eps_r, sigma):
    """What the pre-#1210 rule gave: the owning cell's value, all three."""
    ca, cb = e_update_coeffs(eps_r, sigma, DT)
    return (ca, ca, ca), (cb, cb, cb)


# ---------------------------------------------------------------------------
# (ii) the coefficient ON a material face is the mean -- and the mutation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate", [False, True],
                         ids=["edge-average", "mutation-owning-cell"])
def test_the_coefficient_on_a_dielectric_face_is_the_mean_of_the_four_cells(
        monkeypatch, mutate):
    """Ez on the interface plane of an eps = 1 / eps = 9 step.

    The step is along z and Ez(i, j, k+1/2) is averaged over x and y only, so
    Ez is NOT the component that moves -- Ex and Ey are, and at the interface
    plane the mean of their four cells is (1 + 1 + 9 + 9)/4 = 5. The measured
    quantity is Cb = (dt/eps)/(1 + sigma dt/2eps) with sigma = 0, i.e.
    dt/(5 eps_0), against dt/(9 eps_0) for the owning cell.
    """
    if mutate:
        # Restore the defect: same call, same arguments, mean -> owning cell.
        import rfx.core.yee as _yee
        monkeypatch.setattr(
            _yee, "edge_averaged_materials",
            lambda eps_r, sigma, periodic=(False, False, False): (
                (eps_r, eps_r, eps_r), (sigma, sigma, sigma)))

    eps = _half_space_eps(1.0, 9.0, axis=2, cut=4)
    sigma = jnp.zeros(SHAPE, jnp.float32)

    # Read it where the solver reads it: the baked coefficients.
    coeffs = precompute_coeffs(MaterialArrays(eps, sigma, jnp.ones(SHAPE)),
                               DT, DX)
    # Ex(i+1/2, j, k) at k = 4 is incident on cells k = 3 (eps 1) and k = 4
    # (eps 9), twice each over the y pair.
    cb_ex_face = float(np.asarray(coeffs.cb_ex)[2, 3, 4]) * DX
    expected_mean = DT / (5.0 * EPS_0)
    expected_owner = DT / (9.0 * EPS_0)

    if mutate:
        assert cb_ex_face == pytest.approx(expected_owner, rel=1e-6), (
            "the mutation did not take -- the test cannot claim to detect it")
        return
    assert cb_ex_face == pytest.approx(expected_mean, rel=1e-6), (
        f"Ex on the eps = 1 / eps = 9 interface plane got "
        f"dt/({DT / (cb_ex_face * EPS_0):.3f} eps_0); the four incident cells "
        f"are (1, 1, 9, 9), mean 5. {expected_owner / expected_mean:.2f}x this "
        f"is the owning cell's eps = 9, the pre-#1210 rule.")


def test_a_lossy_face_averages_the_conductance_not_the_resistance():
    """sigma is a PARALLEL conductance along the edge: arithmetic, not harmonic.

    A sigma = 0 / sigma = 1e4 S/m step. The edge on the interface plane carries
    two conducting cells and two lossless ones, so it conducts half: the mean
    is 5e3 S/m. A harmonic mean would be 0 (an open circuit), which is the
    NORMAL-component rule and belongs to a different component.
    """
    idx = np.indices(SHAPE)[2]
    sigma = jnp.asarray(np.where(idx < 4, 0.0, 1e4).astype(np.float32))
    eps = jnp.ones(SHAPE, jnp.float32)
    (_, _, _), (sx, sy, sz) = edge_averaged_materials(eps, sigma)
    assert float(np.asarray(sx)[2, 3, 4]) == pytest.approx(5e3, rel=1e-6)
    assert float(np.asarray(sy)[2, 3, 4]) == pytest.approx(5e3, rel=1e-6)
    # Ez averages over x and y, both uniform here: the step along z cannot
    # reach it, so Ez keeps the owning cell's value.
    assert float(np.asarray(sz)[2, 3, 4]) == pytest.approx(1e4, rel=1e-6)


def test_update_e_multiplies_each_component_by_its_own_coefficient():
    """The field the solver writes, not the coefficient it built.

    One step from a seeded H on the eps = 1 / eps = 9 fixture, compared against
    the cell-owned coefficients applied by hand. The two must differ exactly on
    the cells whose four incident cells are not all the same.
    """
    rng = np.random.default_rng(1210)
    fields = [jnp.asarray(rng.standard_normal(SHAPE).astype(np.float32))
              for _ in range(3)]
    zeros = jnp.zeros(SHAPE, jnp.float32)
    st = FDTDState(ex=zeros, ey=zeros, ez=zeros,
                   hx=fields[0], hy=fields[1], hz=fields[2],
                   step=jnp.array(0, jnp.int32))
    eps = _half_space_eps(1.0, 9.0, axis=2, cut=4)
    sigma = jnp.zeros(SHAPE, jnp.float32)
    mats = MaterialArrays(eps, sigma, jnp.ones(SHAPE, jnp.float32))

    out = update_e(st, mats, DT, DX)
    _, (cb_x, cb_y, cb_z) = edge_averaged_e_update_coeffs(eps, sigma, DT)
    cx, cy, cz = curl_h(st.hx, st.hy, st.hz, DX, (False, False, False), 2, None)
    # Not bitwise: ``update_e`` is jitted and XLA fuses the multiply-add, so
    # the two orderings differ by a float32 ULP or two (measured 8e-6
    # relative). The cell-owned rule is 80 % away at the interface, so the
    # tolerance separates them by five orders of magnitude.
    for name, got, cb, curl in (("ex", out.ex, cb_x, cx),
                                ("ey", out.ey, cb_y, cy),
                                ("ez", out.ez, cb_z, cz)):
        ref = np.asarray(cb * curl)
        np.testing.assert_allclose(
            np.asarray(got), ref, rtol=1e-4, atol=1e-6 * np.max(np.abs(ref)),
            err_msg=f"{name} is not the per-component coefficient times the curl")

    _, owned_cb = _cell_owned_coeffs(eps, sigma)
    owned = np.asarray(owned_cb[0] * cx)
    got = np.asarray(out.ex)
    rel = np.abs(got - owned) / np.maximum(np.abs(owned), 1e-30)
    # eps 9 -> 5 on the interface plane is 4/5 of the owning cell's value.
    assert rel.max() > 0.5, (
        f"the per-component rule moved Ex by at most {rel.max():.2e} of the "
        f"cell-owned value on an eps = 1 / eps = 9 step -- update_e is still "
        f"cell-owned")


# ---------------------------------------------------------------------------
# (iii) homogeneous bit-identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps_r,sigma_v", [(1.0, 0.0), (4.2, 3.7), (12.9, 1e3)])
def test_a_uniform_material_gives_bit_identical_coefficients(eps_r, sigma_v):
    """((a + a) + (a + a)) * 0.25 == a exactly, so vacuum fixtures do not move."""
    eps = jnp.full(SHAPE, eps_r, jnp.float32)
    sigma = jnp.full(SHAPE, sigma_v, jnp.float32)
    (e_c, s_c) = edge_averaged_materials(eps, sigma)
    for a in e_c:
        assert np.array_equal(np.asarray(a), np.asarray(eps))
    for a in s_c:
        assert np.array_equal(np.asarray(a), np.asarray(sigma))

    ca, cb = e_update_coeffs(eps, sigma, DT)
    new_ca, new_cb = edge_averaged_e_update_coeffs(eps, sigma, DT)
    for a, b in zip(new_ca, (ca,) * 3):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    for a, b in zip(new_cb, (cb,) * 3):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_a_uniform_grid_bakes_bit_identical_fast_coefficients():
    eps = jnp.full(SHAPE, 2.55, jnp.float32)
    sigma = jnp.full(SHAPE, 0.0, jnp.float32)
    mu = jnp.ones(SHAPE, jnp.float32)
    c = precompute_coeffs(MaterialArrays(eps, sigma, mu), DT, DX)
    loss = sigma * np.float32(DT) / (np.float32(2.0) * eps * np.float32(EPS_0))
    denom = np.float32(1.0) + loss
    ca = (np.float32(1.0) - loss) / denom
    cb = (np.float32(DT) / (eps * np.float32(EPS_0))) / (denom * np.float32(DX))
    for got in (c.ca_ex, c.ca_ey, c.ca_ez):
        assert np.array_equal(np.asarray(got), np.asarray(ca))
    for got in (c.cb_ex, c.cb_ey, c.cb_ez):
        assert np.array_equal(np.asarray(got), np.asarray(cb))


# ---------------------------------------------------------------------------
# (iv) the boundary convention, against the PEC incidence rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("periodic", [
    (True, True, True), (True, False, False), (False, True, False),
    (False, False, True), (False, False, False),
])
def test_the_edge_average_and_the_pec_incidence_rule_cover_the_same_cells(periodic):
    """One convention for "which four cells" (#689/#931), two weightings.

    ``_volume_occupancy_masks`` noisy-ORs the same four cells the mean averages,
    so on a BINARY occupancy the mean is non-zero exactly where the mask is 1 --
    on a periodic axis because both wrap, on a non-periodic one because the
    mean's edge replication and the mask's zero pad agree about SUPPORT (they
    disagree about weight, which is the point of each).
    """
    rng = np.random.default_rng(0)
    occ = jnp.asarray((rng.random(SHAPE) < 0.4).astype(np.float32))
    masks = _volume_occupancy_masks(occ, periodic)
    means, _ = edge_averaged_materials(occ, occ, periodic)
    for c, (m, mean) in enumerate(zip(masks, means)):
        assert np.array_equal((np.asarray(mean) > 0).astype(np.float32),
                              np.asarray(m)), (
            f"component {c}: the edge average and the PEC incidence rule "
            f"disagree about which cells reach the edge (periodic={periodic})")


def test_a_periodic_axis_wraps_and_a_non_periodic_one_replicates():
    """The two out-of-domain rules, read off a single conducting layer."""
    idx = np.indices(SHAPE)[0]
    sigma = jnp.asarray(np.where(idx == SHAPE[0] - 1, 4.0, 0.0).astype(np.float32))
    eps = jnp.ones(SHAPE, jnp.float32)

    _, (_, sy_p, _) = edge_averaged_materials(eps, sigma, (True, False, False))
    # Ey averages over x and z; with x periodic, cell 0's backward neighbour is
    # the conducting last layer: (0 + 4 + 0 + 4)/4 = 2.
    assert float(np.asarray(sy_p)[0, 2, 3]) == pytest.approx(2.0)

    _, (_, sy_n, _) = edge_averaged_materials(eps, sigma, (False, False, False))
    # Non-periodic: cell 0 replicates itself, so it stays lossless.
    assert float(np.asarray(sy_n)[0, 2, 3]) == 0.0
    # And the eps at that same edge is 1, not 0 -- a zero pad would be a
    # division by zero in e_update_coeffs.
    (_, ey_n, _), _ = edge_averaged_materials(eps, sigma, (False, False, False))
    assert float(np.asarray(ey_n)[0, 2, 3]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# lumped stamps are edge-owned: they are not averaged
# ---------------------------------------------------------------------------

def test_a_lumped_stamp_reaches_its_own_edge_undiluted():
    """A 50 ohm port's conductance across one edge, not a quarter of it.

    The stamp is recorded in ``sigma`` AND in ``sigma_lumped``.
    ``component_e_materials`` averages ``sigma - sigma_lumped`` and adds the
    stamp back at its own cell, so the cell sees the whole conductance on all
    three components and its neighbours see none of it. Averaging the stamp
    instead left a wire port reading |S11| = 0.20 against the closed-form 1/3.
    """
    from rfx.core.yee import component_e_materials

    g = 1.0 / (50.0 * DX)          # the port_sigma of a 50 ohm cubic cell
    cell = (2, 3, 4)
    sigma = jnp.zeros(SHAPE, jnp.float32).at[cell].add(g)
    lumped = jnp.zeros(SHAPE, jnp.float32).at[cell].add(g)
    eps = jnp.ones(SHAPE, jnp.float32)

    mats = MaterialArrays(eps, sigma, jnp.ones(SHAPE), sigma_lumped=lumped)
    _, sig_c = component_e_materials(mats)
    for name, arr in zip(("ex", "ey", "ez"), sig_c):
        assert float(np.asarray(arr)[cell]) == pytest.approx(g, rel=1e-6), (
            f"{name}: the lumped stamp arrived diluted at its own cell")
    # and nowhere else
    for arr in sig_c:
        a = np.asarray(arr).copy()
        a[cell] = 0.0
        assert np.count_nonzero(a) == 0, (
            "the lumped stamp leaked onto a neighbouring edge")

    # Without the record, the same sigma array is averaged: a quarter here,
    # a quarter on each of three neighbours. That is the defect.
    plain = MaterialArrays(eps, sigma, jnp.ones(SHAPE))
    _, sig_plain = component_e_materials(plain)
    assert float(np.asarray(sig_plain[0])[cell]) == pytest.approx(g / 4, rel=1e-6)


def test_no_lumped_record_is_bit_identical_to_the_plain_average():
    from rfx.core.yee import component_e_materials

    eps = _half_space_eps(1.0, 9.0, axis=2, cut=4)
    idx = np.indices(SHAPE)[1]
    sigma = jnp.asarray(np.where(idx < 3, 0.0, 7.0).astype(np.float32))
    mats = MaterialArrays(eps, sigma, jnp.ones(SHAPE))
    eps_c, sig_c = component_e_materials(mats)
    eps_ref, sig_ref = edge_averaged_materials(eps, sigma)
    for a, b in zip(eps_c + sig_c, eps_ref + sig_ref):
        assert np.array_equal(np.asarray(a), np.asarray(b))
