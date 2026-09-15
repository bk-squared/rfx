"""#689 site 2 — the conductor neighbour rule at a domain face.

Defect (external review against 4eb7fa4). The conductor edge rule spelled
its neighbour lookup as ``jnp.roll`` on all three axes, which WRAPS. On a
NON-periodic axis a body on the ``n-1`` face then saw cell ``0`` as its
neighbour and put a wall on the far side of the domain, where there is no
metal. Whether an edge is inside a conductor is a property of the body and
of the lattice, not of where the body sits in the array, so the wrap has to
be conditional: it applies where cell ``0`` and cell ``n-1`` really ARE
neighbours (a PERIODIC axis, or a length-1 axis on the 2-D lane) and a zero
pad applies otherwise. ``rfx.boundaries.pec._shift`` is the one spelling.

#931 replaced the rule the convention is attached to. The old rule (masked
cell AND a masked neighbour along the component's own axis) was a SHEET
classification applied to volumes; a body's far face was never a wall. The
contract's volume rule is "an E component is PEC iff it is incident to an
occupied cell" — four cells, backward shifts along the two axes TRANSVERSE
to the component. This file re-pins the #689 convention on that rule.

ORACLE — a body on the last cell of an axis, and its far face.
Fixture: (6,6,10) domain, one 1-cell 4x4 z-plate of 16 cells.

    plate cell   z flag      [ex, ey, ez] nnz
    k = 8        either      [40, 40, 25]   walls on planes 8 and 9
    k = 9        NON-per.    [20, 20, 25]   far face is plane 10: outside the
                                            array, owned by the domain BC
    k = 9        PERIODIC    [40, 40, 25]   plane 10 == plane 0, a real
                                            neighbour, so the wall is there

The k=8 row is the placement-independent expected value and is measured
without touching the boundary branch. The k=9 rows are the discriminator:
under a zero pad the far face simply has no array entry (the domain
boundary condition owns that plane — design note §1.2), while a wrap on a
NON-periodic axis would paint 20 spurious Ex/Ey walls across the whole
z_lo face of a domain whose metal is at the top.

THE WRAP IS LOAD-BEARING ON TWO KINDS OF AXIS. Measured, not assumed:

  * a genuinely periodic axis, as in the table above (and the transverse
    case ``test_apply_pec_mask_forwards_the_periodic_flags`` below, where
    the wrap doubles a plate's tangential wall count);
  * a length-1 axis (the 2-D lane; ``rfx/simulation.py`` forces
    ``periodic[2] = True`` when ``grid.is_2d``). For a VOLUME the length-1
    wrap is a no-op under the contract — the backward neighbour along a
    length-1 axis is the cell itself and ``C | C == C`` — which is why the
    2-D lane keeps its interior PEC either way. It is load-bearing for a
    SHEET whose normal is that axis: there the plane has no thickness
    direction, and the length-1 branch is what makes the normal component
    PEC on the footprint at all (design note §1.3). Both are pinned below.

``apply_pec_mask`` / ``realized_pec_edge_masks`` and
``build_sheet_impedance_ctx`` both take the flags; handing them different
ones would compute the #677 G4 footprint identity against two different
neighbour rules.
"""

import numpy as np
import jax.numpy as jnp

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.pec import (
    SheetSpec, apply_pec_mask, realized_pec_edge_masks,
)
from rfx.core.yee import init_state


def _plates(ks, shape=(6, 6, 10)):
    m = np.zeros(shape, bool)
    for k in ks:
        m[1:5, 1:5, k] = True
    return jnp.asarray(m)


def _nnz(masks):
    return [int(jnp.sum(x)) for x in masks]


def test_far_face_of_a_last_cell_body_is_owned_by_the_domain_bc():
    """ORACLE — the k=8 / k=9 table in the module docstring."""
    interior = _nnz(realized_pec_edge_masks(_plates([8])))
    assert interior == [40, 40, 25], interior
    assert _nnz(realized_pec_edge_masks(_plates([7]))) == interior
    # same body on the last cell: its far face is plane 10, outside the
    # array. The domain BC owns that plane; the rule must NOT wrap it onto
    # plane 0, where there is no metal.
    on_face = realized_pec_edge_masks(_plates([9]))
    assert _nnz(on_face) == [20, 20, 25], _nnz(on_face)
    assert int(jnp.sum(on_face[0][:, :, 0])) == 0
    assert int(jnp.sum(on_face[1][:, :, 0])) == 0


def test_periodic_axis_keeps_the_wrap_so_the_far_face_lands_on_plane_zero():
    """On a periodic axis cell 0 and cell n-1 really are neighbours, so the
    far face of a body on the last cell is plane 0."""
    per = realized_pec_edge_masks(_plates([9]), periodic=(False, False, True))
    assert _nnz(per) == [40, 40, 25], _nnz(per)
    assert int(jnp.sum(per[0][:, :, 0])) == 20
    assert int(jnp.sum(per[1][:, :, 0])) == 20
    # A body straddling the seam does NOT discriminate under the volume
    # rule — each of its two cells already puts a wall on its own near
    # plane, so wrap and pad give the same set. Measured, so the oracle
    # above is not credited to the wrong fixture.
    m = np.zeros((8, 8, 8), bool)
    m[[7, 0], 3:5, 3:5] = True
    m = jnp.asarray(m)
    assert (_nnz(realized_pec_edge_masks(m, periodic=(True, False, False)))
            == _nnz(realized_pec_edge_masks(m)) == [18, 18, 18])


def test_length_one_axis_wrap_is_a_no_op_for_a_volume():
    """The 2-D lane, volume branch: the backward neighbour along a length-1
    axis is the cell itself, and ``C | C == C``. Measured so the guard is
    not credited with work it does not do."""
    m = np.zeros((10, 10, 1), bool)
    m[3:7, 3:7, 0] = True
    m = jnp.asarray(m)
    assert _nnz(realized_pec_edge_masks(m)) == [20, 20, 25]
    assert (_nnz(realized_pec_edge_masks(m, periodic=(False, False, True)))
            == [20, 20, 25])


def test_length_one_axis_is_load_bearing_for_a_sheet_normal():
    """The 2-D lane, sheet branch: a sheet whose normal is the length-1 axis
    has no through-sheet direction, so its 'normal' component is PEC at the
    footprint nodes — the same edge set the volume rule gives the rectangle
    the footprint's node corners draw (4x4 nodes = 3x3 cells)."""
    fp = np.zeros((10, 10, 1), bool)
    fp[3:7, 3:7, 0] = True
    sheet = SheetSpec(normal_axis=2, plane=0, footprint=jnp.asarray(fp))
    got = _nnz(realized_pec_edge_masks(None, sheets=[sheet]))
    assert got == [12, 12, 16], got
    cells = np.zeros((10, 10, 1), bool)
    cells[3:6, 3:6, 0] = True           # the 3x3 cells those nodes bound
    assert _nnz(realized_pec_edge_masks(jnp.asarray(cells))) == got


def test_apply_pec_mask_forwards_the_periodic_flags():
    """#931: a 1-cell plate on cell ``n-1`` of an axis has its far face on
    plane ``n`` — owned by the domain BC when the axis is non-periodic
    (no array entry), and plane ``0`` (≡ ``n``) when it is periodic.  The
    x-tangential ``Ey`` count doubles iff the x flag reaches the rule.
    (The pre-#931 version of this test pinned a 1-cell plate as a sheet
    with a live normal edge; under the volume rule a 1-cell Box shorts
    its normal edge on every axis and no flag changes that.)"""
    shape = (6, 6, 10)
    st = init_state(shape)
    st = st._replace(ex=jnp.ones(shape), ey=jnp.ones(shape),
                     ez=jnp.ones(shape))
    pm = np.zeros(shape, bool)
    pm[5, 1:5, 1:5] = True          # one cell thick along x, on face n-1
    pm = jnp.asarray(pm)
    ey_non = np.asarray(apply_pec_mask(st, pm).ey) == 0.0
    ey_per = np.asarray(apply_pec_mask(st, pm, (True, False, False)).ey) == 0.0
    assert int(ey_non.sum()) == 20 and ey_non[5].sum() == 20, int(ey_non.sum())
    assert int(ey_per.sum()) == 40, int(ey_per.sum())
    assert ey_per[0].sum() == 20 and ey_per[5].sum() == 20
    # the normal component is shorted inside the cell under both flags
    for per in [(False, False, False), (True, False, False)]:
        ex = np.asarray(apply_pec_mask(st, pm, per).ex) == 0.0
        assert int(ex.sum()) == 25 and ex[5].sum() == 25


def test_sheet_ctx_and_pec_edges_share_one_neighbour_rule():
    """#677 G4 footprint identity, re-checked under the contract: the f0
    sheet ctx and the PEC realization of the SAME footprint select the same
    edges, under every flag combination."""
    from rfx.materials.thin_conductor import (
        SheetImpedanceSpec, build_sheet_impedance_ctx)
    fp = np.zeros((6, 6, 10), bool)
    fp[1:5, 1:5, 0] = True
    m = jnp.asarray(fp)
    spec = SheetImpedanceSpec(mask=m, normal_axis=2, g_sheet=1.0,
                              sigma_sheet=jnp.where(m, 1.0, 0.0), plane=0)
    pec_sheet = SheetSpec(normal_axis=2, plane=0, footprint=m)
    for per in [(False, False, False), (False, False, True),
                (True, True, True)]:
        ctx = build_sheet_impedance_ctx([spec], periodic=per)
        ref = realized_pec_edge_masks(None, sheets=[pec_sheet], periodic=per)
        assert bool(jnp.all(ctx.mask_ex == ref[0])), per
        assert bool(jnp.all(ctx.mask_ey == ref[1])), per
        assert bool(jnp.all(ctx.mask_ez == ref[2])), per


def _tmz_sim():
    sim = Simulation(freq_max=30e9, domain=(0.02, 0.02, 0.001), dx=1e-3,
                     boundary="pec", mode="2d_tmz")
    sim.add(Box((0.008, 0.008, 0.0), (0.012, 0.012, 0.001)), material="pec")
    sim.add_source((0.004, 0.010, 0), "ez", waveform=GaussianPulse(f0=15e9),
                   amplitude_kind="field")
    sim.add_probe((0.010, 0.010, 0), "ez")   # INSIDE the PEC block
    sim.add_probe((0.004, 0.004, 0), "ez")   # outside control
    return sim


def test_two_d_interior_pec_still_zeroes_ez():
    """The 2-D lane end to end. The probe sits inside the PEC on purpose;
    ``skip_preflight`` only silences the expected 'source is inside PEC'
    line, it is not a config fix."""
    res = _tmz_sim().run(n_steps=200, skip_preflight=True)
    ts = np.abs(np.asarray(res.time_series))
    inside, outside = float(ts[:, 0].max()), float(ts[:, 1].max())
    assert inside == 0.0, (inside, outside)
    assert outside > 1e-3, (inside, outside)
