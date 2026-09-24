"""A lumped element loads the E edge it sits on, and no other (#1236).

A port's load, an RLC resistor or capacitor is a device across ONE Yee edge:
an element declared on Ez at node n is a resistor (or capacitor) between node n
and node n + z. It is folded into the material as an equivalent conductivity or
permittivity at that node's cell, and until #1236 the fold carried no
component, so the E update added it to all three components at the node: a
50 ohm port on Ez was also a 50 ohm resistor on the Ex and Ey edges leaving the
node in +x and +y. On a centre-fed dipole that pinned the feed-tip radial field
on those two sides to 0.03 of the gap field against 0.37 on the free side, and
moved the resonance +0.34 % at lambda/43.

What is read here is the coefficient the solver multiplies each component by
(``e_component_coeffs``, the entry ``update_e`` uses) and the single-node
coefficient every port and source drive uses (``cell_component_e_coeffs``) --
not the stamp helper's own bookkeeping. The background is a random dielectric
and conductor, so the four-cell edge average is exercised around the stamp.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import (
    MaterialArrays,
    cell_component_e_coeffs,
    cell_component_e_materials,
    component_e_materials,
    e_component_coeffs,
)
from rfx.sources.sources import stamp_lumped_eps, stamp_lumped_sigma

DT = 1.0e-12
DX = 1.0e-3
SHAPE = (6, 7, 8)
CELL = (2, 3, 4)
COMPONENTS = ("ex", "ey", "ez")


def _background():
    rng = np.random.default_rng(1236)
    eps = jnp.asarray(1.0 + 9.0 * rng.random(SHAPE).astype(np.float32))
    sigma = jnp.asarray(5.0 * rng.random(SHAPE).astype(np.float32))
    return MaterialArrays(eps, sigma, jnp.ones(SHAPE, jnp.float32))


def _coeffs(mats):
    ca, cb = e_component_coeffs(mats, DT)
    return [np.asarray(a) for a in ca], [np.asarray(b) for b in cb]


@pytest.mark.parametrize("kind", ["sigma", "eps"])
@pytest.mark.parametrize("component", COMPONENTS)
def test_a_stamp_changes_only_its_own_component_at_its_own_node(component,
                                                                  kind):
    """50 ohm (or 1 pF) on one edge: that edge's coefficient changes, at that
    node only; the other two components are bit-identical to no stamp."""
    axis = COMPONENTS.index(component)
    base = _background()
    if kind == "sigma":
        value = 1.0 / (50.0 * DX)                     # 50 ohm, cubic cell
        mats = stamp_lumped_sigma(base, CELL, value, component)
    else:
        value = 1e-12 * DX / (8.8541878128e-12 * DX * DX)   # 1 pF fold
        mats = stamp_lumped_eps(base, CELL, value, component)

    ca0, cb0 = _coeffs(base)
    ca1, cb1 = _coeffs(mats)
    for c in range(3):
        for name, a0, a1 in (("Ca", ca0[c], ca1[c]), ("Cb", cb0[c], cb1[c])):
            # The volume is recovered as (total - stamp), which is the
            # background only to one rounding of the sum (#1210's record):
            # a neighbour moves by ~1 ulp (measured <= 1.3e-7 here). The
            # element moves its own Cb to 0.86x (50 ohm) and 0.057x (1 pF),
            # so 1e-5 separates the two by decades on both sides.
            rel = np.abs(a1 - a0) / np.maximum(np.abs(a0), 1e-30)
            changed = np.argwhere(rel > 1e-5)
            if c == axis:
                assert [tuple(x) for x in changed] == [CELL], (
                    f"{COMPONENTS[c]} {name}: a stamp on {component} at {CELL} "
                    f"must change this component there and nowhere else; "
                    f"changed at {[tuple(x) for x in changed]}")
            else:
                assert changed.size == 0, (
                    f"{COMPONENTS[c]} {name} changed at "
                    f"{[tuple(x) for x in changed]}: a lumped element on "
                    f"{component} loads the {COMPONENTS[c]} edge too (#1236)")

    # The realized edge value is the declared one, whole (not averaged).
    eps_c, sig_c = component_e_materials(mats)
    eps_b, sig_b = component_e_materials(base)
    got = (np.asarray(sig_c[axis])[CELL] - np.asarray(sig_b[axis])[CELL]
           if kind == "sigma" else
           np.asarray(eps_c[axis])[CELL] - np.asarray(eps_b[axis])[CELL])
    assert got == pytest.approx(value, rel=1e-5)

    # Every port / source DRIVE coefficient reads the single-node rule; it
    # must agree with the grid-wide one for all three components.
    for c, comp in enumerate(COMPONENTS):
        ca_n, cb_n = cell_component_e_coeffs(mats, CELL, comp, DT)
        assert float(ca_n) == pytest.approx(float(ca1[c][CELL]), rel=1e-6)
        assert float(cb_n) == pytest.approx(float(cb1[c][CELL]), rel=1e-6)


def test_two_elements_of_different_components_on_one_node_keep_both():
    """An Ez port and an Ex resistor + capacitor at the same node: each
    component carries its own element, Ey carries neither, and the cell total
    (read by the cell-owned lanes) carries both."""
    base = _background()
    g_z = 1.0 / (50.0 * DX)
    g_x = 1.0 / (300.0 * DX)
    c_x = 3.0
    mats = stamp_lumped_sigma(base, CELL, g_z, "ez")
    mats = stamp_lumped_sigma(mats, CELL, g_x, "ex")
    mats = stamp_lumped_eps(mats, CELL, c_x, "ex")

    eps_c, sig_c = component_e_materials(mats)
    eps_b, sig_b = component_e_materials(base)
    d_sig = [float(np.asarray(s)[CELL] - np.asarray(b)[CELL])
             for s, b in zip(sig_c, sig_b)]
    d_eps = [float(np.asarray(e)[CELL] - np.asarray(b)[CELL])
             for e, b in zip(eps_c, eps_b)]
    assert d_sig == pytest.approx([g_x, 0.0, g_z], rel=1e-5, abs=1e-4)
    assert d_eps == pytest.approx([c_x, 0.0, 0.0], rel=1e-5, abs=1e-5)

    for comp, want_s, want_e in zip(COMPONENTS, (g_x, 0.0, g_z),
                                    (c_x, 0.0, 0.0)):
        e_n, s_n = cell_component_e_materials(mats, CELL, comp)
        e_0, s_0 = cell_component_e_materials(base, CELL, comp)
        assert float(s_n - s_0) == pytest.approx(want_s, rel=1e-5, abs=1e-4)
        assert float(e_n - e_0) == pytest.approx(want_e, rel=1e-5, abs=1e-5)

    assert float(mats.sigma[CELL] - base.sigma[CELL]) == pytest.approx(
        g_x + g_z, rel=1e-5)
    sx, sy, sz = mats.sigma_lumped
    assert sy is None and sx is not None and sz is not None


def test_a_component_less_record_is_refused():
    """The pre-#1236 record (one array, no component) loaded all three edges.
    A caller still writing it must fail loudly, not keep that rule."""
    base = _background()
    bare = jnp.zeros(SHAPE, jnp.float32).at[CELL].add(20.0)
    mats = base._replace(sigma=base.sigma + bare, sigma_lumped=bare)
    with pytest.raises(TypeError, match="per E component"):
        component_e_materials(mats)
    with pytest.raises(TypeError, match="per E component"):
        cell_component_e_materials(mats, CELL, "ez")
    with pytest.raises(ValueError, match="one E edge"):
        stamp_lumped_sigma(base, CELL, 1.0, "hz")


def test_the_distributed_slab_update_loads_the_own_edge_only():
    """The distributed lane's cell-owned slab update, given the same record:
    in a homogeneous background it must equal the single-device update on all
    three components (the port node included)."""
    from rfx.core.yee import FDTDState, init_materials, update_e
    from rfx.runners.distributed import _update_e_local

    mats = init_materials(SHAPE)._replace(
        eps_r=jnp.full(SHAPE, 2.2, jnp.float32),
        sigma=jnp.full(SHAPE, 0.3, jnp.float32))
    mats = stamp_lumped_sigma(mats, CELL, 1.0 / (50.0 * DX), "ez")
    rng = np.random.default_rng(7)
    h = [jnp.asarray(rng.standard_normal(SHAPE).astype(np.float32))
         for _ in range(3)]
    z = jnp.zeros(SHAPE, jnp.float32)
    st = FDTDState(ex=z, ey=z, ez=z, hx=h[0], hy=h[1], hz=h[2],
                   step=jnp.array(0, jnp.int32))
    slab = _update_e_local(st, mats, DT, DX)
    ref = update_e(st, mats, DT, DX)
    for comp in COMPONENTS:
        a = np.asarray(getattr(slab, comp))
        b = np.asarray(getattr(ref, comp))
        # Interior only: the slab lane pads H with zeros at its faces the
        # same way; compare away from the lo faces of the backward curl.
        np.testing.assert_allclose(a[1:, 1:, 1:], b[1:, 1:, 1:], rtol=2e-6,
                                   atol=1e-6 * float(np.max(np.abs(b))),
                                   err_msg=comp)


def test_a_dispersive_lane_says_it_loads_all_three_edges():
    """Debye/Lorentz coefficients are one per cell for all three components
    (cell-owned), over the whole grid once any dispersive material is present
    (#1260). With a lumped record present that lane still puts every element
    on all three edges at its node; it must say so, not do it silently."""
    from rfx.materials.debye import DebyePole, init_debye
    from rfx.core.yee import init_materials

    mats = stamp_lumped_sigma(init_materials(SHAPE), CELL, 20.0, "ez")
    with pytest.warns(UserWarning, match=r"#1260.*#1236"):
        init_debye([DebyePole(delta_eps=1.0, tau=1e-11)], mats, DT)
    from rfx.materials.lorentz import init_lorentz, lorentz_pole
    with pytest.warns(UserWarning, match=r"#1260.*#1236"):
        init_lorentz([lorentz_pole(delta_eps=1.0, omega_0=2e10, delta=1e9)],
                     mats, DT)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        init_debye([DebyePole(delta_eps=1.0, tau=1e-11)],
                   init_materials(SHAPE), DT)
