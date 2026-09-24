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

@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_a_lumped_stamp_reaches_its_own_edge_undiluted(component):
    """A 50 ohm port's conductance across one edge, not a quarter of it.

    The stamp is recorded in ``sigma`` AND in ``sigma_lumped`` (per E
    component, #1236). ``component_e_materials`` averages ``sigma`` minus the
    stamps and adds each back at its own cell, on its own component, so the
    port's edge sees the whole conductance and no other edge sees any of it.
    Averaging the stamp instead left a wire port reading |S11| = 0.20 against
    the closed-form 1/3; adding it to all three components put the port's
    load on the two transverse edges at its node as well (#1236).
    """
    from rfx.core.yee import component_e_materials
    from rfx.sources.sources import stamp_lumped_sigma

    g = 1.0 / (50.0 * DX)          # the port_sigma of a 50 ohm cubic cell
    cell = (2, 3, 4)
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]
    eps = jnp.ones(SHAPE, jnp.float32)
    mats = stamp_lumped_sigma(
        MaterialArrays(eps, jnp.zeros(SHAPE, jnp.float32), jnp.ones(SHAPE)),
        cell, g, component)
    sigma = mats.sigma
    _, sig_c = component_e_materials(mats)
    for c, (name, arr) in enumerate(zip(("ex", "ey", "ez"), sig_c)):
        want = g if c == axis else 0.0
        assert float(np.asarray(arr)[cell]) == pytest.approx(want, rel=1e-6), (
            f"{name}: a stamp on {component} reads {float(np.asarray(arr)[cell])}"
            f" on {name} at its own cell, want {want}")
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


# ---------------------------------------------------------------------------
# the DRIVE coefficient is the update's own coefficient (#1210)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_the_cell_helper_agrees_with_the_grid_wide_one(component):
    """One rule, two evaluations: whole grid, and one node of it.

    The drive coefficients cannot afford a grid-sized array, so they index the
    four incident cells instead. If the two ever disagree there are two rules.
    """
    from rfx.core.yee import (cell_component_e_materials,
                              component_e_materials)

    rng = np.random.default_rng(1210)
    eps = jnp.asarray(1.0 + 9.0 * rng.random(SHAPE).astype(np.float32))
    sigma = jnp.asarray(1e3 * rng.random(SHAPE).astype(np.float32))
    lump_z = jnp.zeros(SHAPE, jnp.float32).at[2, 3, 4].add(77.0)
    lump_x = jnp.zeros(SHAPE, jnp.float32).at[2, 3, 4].add(13.0)
    # Two elements on one node, on two components (#1236).
    mats = MaterialArrays(eps, sigma + lump_x + lump_z, jnp.ones(SHAPE),
                          sigma_lumped=(lump_x, None, lump_z))
    axis = {"ex": 0, "ey": 1, "ez": 2}[component]

    for periodic in [(False, False, False), (True, True, True)]:
        eps_g, sig_g = component_e_materials(mats, periodic)
        for cell in [(0, 0, 0), (2, 3, 4), (5, 6, 7), (0, 6, 0)]:
            e_c, s_c = cell_component_e_materials(mats, cell, component,
                                                  periodic)
            assert float(e_c) == pytest.approx(
                float(np.asarray(eps_g[axis])[cell]), rel=1e-6), (
                f"eps disagrees at {cell}, {component}, periodic={periodic}")
            assert float(s_c) == pytest.approx(
                float(np.asarray(sig_g[axis])[cell]), rel=1e-6)


@pytest.mark.parametrize("mutate", [False, True],
                         ids=["per-component", "mutation-owning-cell"])
def test_a_current_source_injects_the_update_s_own_coefficient(monkeypatch,
                                                               mutate):
    """A current source on a material step drives with the update's Cb.

    ``make_j_source`` turns a current into the field increment the E update
    itself would have produced, so its Cb must be the update's Cb for the
    component it injects. On an eps = 1 / eps = 9 step along y, an Ex source at
    the interface plane sees the mean 5, not the owning cell's 9 — the
    cell-owned coefficient is 0.556x of it, which injects 0.556x the declared
    current. |S11| divides the drive out; an absolute field, a radiated power
    and a gain do not.

    Mutation (b): the drive reads the owning cell (the same call, the same
    arguments) and the assertion fires.
    """
    from rfx.core.yee import e_component_coeffs
    from rfx.grid import Grid
    import rfx.simulation as _sim

    if mutate:
        monkeypatch.setattr(
            _sim, "cell_component_e_coeffs",
            lambda m, cell, comp, dt, periodic=(False, False, False):
                e_update_coeffs(m.eps_r[tuple(cell)], m.sigma[tuple(cell)], dt))

    grid = Grid(freq_max=1e10, domain=(9e-3, 9e-3, 9e-3), dx=1e-3,
                cpml_layers=0, cpml_axes="")
    shape = tuple(grid.shape)
    idx = np.indices(shape)[1]
    eps = jnp.asarray(np.where(idx < 4, 1.0, 9.0).astype(np.float32))
    sigma = jnp.zeros(shape, jnp.float32)
    mats = MaterialArrays(eps, sigma, jnp.ones(shape, jnp.float32))
    cell = (4, 4, 4)
    pos = tuple((c + 0.5) * 1e-3 for c in cell)

    spec = _sim.make_j_source(grid, pos, "ex", lambda t: 1.0, 3, mats,
                              amplitude_kind="current")
    assert (spec.i, spec.j, spec.k) == cell, "fixture moved off the interface"
    # make_j_source scales Cb by the cell volume; divide it back out.
    drive = float(np.asarray(spec.waveform)[0]) * (1e-3 ** 3)
    update_cb = float(np.asarray(e_component_coeffs(mats, grid.dt)[1][0])[cell])
    owning_cb = float(e_update_coeffs(eps, sigma, grid.dt)[1][cell])

    if mutate:
        assert drive == pytest.approx(owning_cb, rel=1e-5), (
            "the mutation did not take -- the test cannot claim to detect it")
        return
    assert drive == pytest.approx(update_cb, rel=1e-5), (
        f"the Ex drive coefficient is {drive:.6e} but the E update multiplies "
        f"this node by {update_cb:.6e} ({owning_cb / update_cb:.4f}x is the "
        f"owning cell's value)")


def test_a_lumped_stamp_is_edge_owned_on_a_periodic_axis_too():
    """The stamp is not averaged, so a periodic seam cannot smear it.

    A stamp on the lo face of a periodic axis: the volume average there wraps
    to the far side, and the stamp must still land whole on its own cell and
    nowhere else — including on the cell it wraps to.
    """
    from rfx.core.yee import component_e_materials

    g = 1.0 / (50.0 * DX)
    cell = (0, 3, 4)
    wrapped = (SHAPE[0] - 1, 3, 4)
    sigma = jnp.zeros(SHAPE, jnp.float32).at[cell].add(g)
    lumped = jnp.zeros(SHAPE, jnp.float32).at[cell].add(g)
    # A non-uniform VOLUME sigma as well, so the wrap is exercised.
    vol = jnp.asarray((np.indices(SHAPE)[0] * 1.0).astype(np.float32))
    # The element sits on Ez (#1236: the record carries its component).
    mats = MaterialArrays(jnp.ones(SHAPE, jnp.float32), sigma + vol,
                          jnp.ones(SHAPE), sigma_lumped=(None, None, lumped))

    for periodic in [(True, True, True), (True, False, False)]:
        _, sig_c = component_e_materials(mats, periodic)
        for name, arr in zip(("ex", "ey", "ez"), sig_c):
            a = np.asarray(arr)
            # the volume part at this cell, computed without the stamp
            plain = MaterialArrays(jnp.ones(SHAPE, jnp.float32), vol,
                                   jnp.ones(SHAPE))
            base = np.asarray(component_e_materials(plain, periodic)[1][
                {"ex": 0, "ey": 1, "ez": 2}[name]])
            want = g if name == "ez" else 0.0
            assert a[cell] - base[cell] == pytest.approx(want, rel=1e-5,
                                                         abs=1e-3), (
                f"{name}: the stamp did not arrive whole at its own cell, "
                f"on its own component only (periodic={periodic})")
            assert a[wrapped] - base[wrapped] == pytest.approx(0.0, abs=1e-3), (
                f"{name}: the stamp wrapped onto the far face "
                f"(periodic={periodic})")


# ---------------------------------------------------------------------------
# the subpixel (Kottke) lane's sigma is per-component too
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mutate", [False, True],
                         ids=["per-component", "mutation-cell-owned-sigma"])
def test_the_aniso_updates_decay_with_the_edge_averaged_sigma(monkeypatch,
                                                              mutate):
    """A lossy sigma step under the Kottke inverse-eps update.

    The subpixel lane gives eps a per-component value by construction, but its
    sigma came from the owning cell. sigma is a VOLUME conductivity: the four
    cells' conduction paths lie in parallel along the edge, so it takes the
    same mean. Every committed subpixel fixture is LOSSLESS, so reverting the
    three ``*_aniso*`` lines stays green without this test.

    Measured on Ex at the interface plane of a sigma = 0 / sigma = 2e4 S/m
    step along z: the edge sees 1e4 S/m, the owning cell 2e4, and Ca
    (the decay factor, with E^n the only term) differs by the ratio below.
    """
    import jax
    import rfx.core.yee as _yee
    from rfx.core.yee import update_e_aniso_inv, update_e_nu_aniso

    # Both arms compile fresh: update_e_aniso_inv is jitted, and a cached
    # trace would hand the second arm the first arm's coefficients.
    jax.clear_caches()
    if mutate:
        monkeypatch.setattr(
            _yee, "component_e_materials",
            lambda m, periodic=(False, False, False): (
                (m.eps_r,) * 3, (m.sigma,) * 3))

    idx = np.indices(SHAPE)[2]
    sigma = jnp.asarray(np.where(idx < 4, 0.0, 2e4).astype(np.float32))
    eps = jnp.ones(SHAPE, jnp.float32)
    mats = MaterialArrays(eps, sigma, jnp.ones(SHAPE, jnp.float32))
    cell = (2, 3, 4)

    # H = 0, so E^{n+1} = Ca * E^n and Ca is read directly off the field.
    one = jnp.ones(SHAPE, jnp.float32)
    zero = jnp.zeros(SHAPE, jnp.float32)
    st = FDTDState(ex=one, ey=one, ez=one, hx=zero, hy=zero, hz=zero,
                   step=jnp.array(0, jnp.int32))
    inv = jnp.ones(SHAPE, jnp.float32)      # inv_eps = 1 -> eps_r = 1

    def ca_of(sig_value):
        """Ca = (1 - s*dt/2e) / (1 + s*dt/2e) at eps_r = 1."""
        return float(e_update_coeffs(jnp.float32(1.0),
                                     jnp.float32(sig_value), DT)[0])

    ca_edge, ca_owned = ca_of(1e4), ca_of(2e4)
    assert abs(ca_edge - ca_owned) > 1e-3, "the fixture cannot separate the two"

    out = update_e_aniso_inv(st, mats, inv, inv, inv, DT, DX)
    jax.clear_caches()
    got = float(np.asarray(out.ex)[cell])
    nu_inv = jnp.ones(SHAPE[0], jnp.float32)
    out_nu = update_e_nu_aniso(st, mats, eps, eps, eps, DT,
                               nu_inv, jnp.ones(SHAPE[1], jnp.float32),
                               jnp.ones(SHAPE[2], jnp.float32))
    got_nu = float(np.asarray(out_nu.ex)[cell])

    expect = ca_owned if mutate else ca_edge
    other = ca_edge if mutate else ca_owned
    for name, value in (("update_e_aniso_inv", got), ("update_e_nu_aniso", got_nu)):
        assert value == pytest.approx(expect, rel=1e-5), (
            f"{name}: Ex on the sigma = 0 / sigma = 2e4 face decayed by "
            f"{value:.6f}; the edge sees the mean 1e4 S/m (Ca = {ca_edge:.6f}), "
            f"the owning cell 2e4 (Ca = {other if mutate else ca_owned:.6f})")


@pytest.mark.parametrize("mutate", [False, True],
                         ids=["per-component", "mutation-owning-cell"])
def test_the_applied_port_drives_use_the_update_s_own_coefficient(monkeypatch,
                                                                  mutate):
    """``apply_lumped_port`` / ``apply_wire_port``, the S-parameter drives.

    The same port reaches the fields through two spellings: pre-baked, via
    ``make_port_source`` / ``make_wire_port_sources``, and per step, via these
    two. Until #1210 the second pair read the owning cell, so a lumped port on
    a material step transverse to its component drove 1.80x harder through one
    path than the other (eps 1|9 along y, Ez port: owning cell 9 against the
    mean 5).
    """
    import rfx.sources.sources as _src
    from rfx.core.yee import e_component_coeffs, init_state
    from rfx.grid import Grid
    from rfx.sources.sources import (LumpedPort, WirePort, apply_lumped_port,
                                     apply_wire_port)

    if mutate:
        monkeypatch.setattr(
            _src, "cell_component_e_coeffs",
            lambda m, cell, comp, dt, periodic=(False, False, False):
                e_update_coeffs(m.eps_r[tuple(cell)], m.sigma[tuple(cell)], dt))

    grid = Grid(freq_max=1e10, domain=(9e-3, 9e-3, 9e-3), dx=1e-3,
                cpml_layers=0, cpml_axes="")
    shape = tuple(grid.shape)
    idx = np.indices(shape)[1]
    eps = jnp.asarray(np.where(idx < 4, 1.0, 9.0).astype(np.float32))
    sigma = jnp.zeros(shape, jnp.float32)
    mats = MaterialArrays(eps, sigma, jnp.ones(shape, jnp.float32))
    cell = (4, 4, 4)
    pos = tuple((c + 0.5) * 1e-3 for c in cell)

    update_cb = float(np.asarray(e_component_coeffs(mats, grid.dt)[1][2])[cell])
    owning_cb = float(e_update_coeffs(eps, sigma, grid.dt)[1][cell])
    expect = owning_cb if mutate else update_cb
    assert update_cb / owning_cb == pytest.approx(1.8, rel=1e-2)

    st = init_state(shape)
    lp = LumpedPort(position=pos, component="ez", impedance=50.0,
                    excitation=lambda t: 1.0)
    out = apply_lumped_port(st, grid, lp, 0.0, mats)
    drive = float(np.asarray(out.ez)[cell]) * 1e-3          # undo 1/d_par
    assert drive == pytest.approx(expect, rel=1e-5), (
        f"apply_lumped_port drove with {drive:.6e}; the E update multiplies "
        f"Ez at that node by {update_cb:.6e} "
        f"({owning_cb:.6e}, {owning_cb / update_cb:.3f}x, is the owning cell)")

    wp = WirePort(start=pos, end=(pos[0], pos[1], pos[2] + 1e-3),
                  component="ez", impedance=50.0,
                  excitation=lambda t: 1.0)
    from rfx.sources.sources import _wire_port_live_cells
    _, _, n_live = _wire_port_live_cells(grid, wp, None)
    out_w = apply_wire_port(init_state(shape), grid, wp, 0.0, mats)
    # apply_wire_port splits the drive over the live cells and divides by
    # d_par; undo both so what is compared is the coefficient itself.
    drive_w = float(np.asarray(out_w.ez)[cell]) * 1e-3 * n_live
    assert drive_w == pytest.approx(expect, rel=1e-5), (
        f"apply_wire_port drove with {drive_w:.6e} against {update_cb:.6e}")
