"""#689 site 2 — the tangential-edge neighbour rule at a domain face.

Defect (external review against 4eb7fa4).
``rfx.boundaries.pec.tangential_edge_masks`` spelled the neighbour lookup
as ``jnp.roll`` on all three axes, which WRAPS. On a NON-periodic axis two
one-cell bodies sitting on the ``0`` and ``n-1`` faces then saw each other
through the domain and both had their sheet-NORMAL component selected —
i.e. classified as if they were one two-cell slab. Whether a component is
tangential or normal is a property of the body, not of where it sits in the
array, so translating the same pair one cell inward must not change the
answer.

ORACLE — TRANSLATION INVARIANCE. Fixture: (6,6,10) domain, two 1-cell
4x4 z-plates. The interior placements never touch the boundary branch, so
they supply the expected value independently of the code under test.

    placement          [ex, ey, ez] nnz
    k=1 & k=8          [32, 32,  0]     <- the oracle value
    k=2 & k=7          [32, 32,  0]     <- placement-independent
    k=0 & k=9          [32, 32, 32]     <- measured at 4eb7fa4
    k=0 & k=9, fixed   [32, 32,  0]
    single plate k=0   [16, 16,  0]     <- control: needs BOTH faces

ex and ey read 32 in every placement, so the oracle isolates the wrapped
axis and would also catch a fix that over-corrects the other two.

THE WRAP IS LOAD-BEARING ON TWO KINDS OF AXIS. Measured, not assumed —
an unconditional zero pad is a worse bug than the one it fixes:

  * length-1 axis (the 2-D lane; ``rfx/simulation.py`` forces
    ``periodic[2] = True`` when ``grid.is_2d``). A (10,10,1) mask of 16
    cells: wrap selects 16 Ez edges, zero pad selects 0 — every 2-D run
    with interior PEC silently loses its PEC. End-to-end 2d_tmz witness,
    probe inside a PEC block, 200 steps:

        max|Ez| inside the block   outside control
        guards kept   0.000000e+00      1.588587e-02
        zero pad      7.549178e-02      3.035763e-02

    Preflight on that fixture, verbatim (the probe is inside the PEC on
    purpose, which is the whole point of the measurement):
    ``[PREFLIGHT] Port/source at (0.01, 0.01, 0) is inside PEC geometry
    'pec'. Field will be zero. Move source outside PEC.``

  * a genuinely periodic axis, where cell 0 and cell n-1 ARE neighbours.
    A body straddling the seam one cell either side ({n-1, 0}) goes from
    8 selected edges to 0 under a zero pad. A body straddling it two
    cells either side, or spanning the whole axis, is unaffected — so
    only the thin seam-straddling case discriminates, and it is exactly
    the RIS/FSS drawing that would break silently.

So the rule is conditional: zero pad on a non-periodic axis of length > 1,
wrap otherwise. ``apply_pec_mask`` and ``build_sheet_impedance_ctx`` both
take the flags; handing them different ones would compute the #677 G4
footprint identity against two different neighbour rules.

The two 2-D nets OVERLAP on the main uniform stepper and do NOT overlap
elsewhere, which is worth knowing before trusting either alone. Deleting
only the length-1 guard leaves ``test_two_d_interior_pec_still_zeroes_ez``
GREEN, because ``rfx/simulation.py`` forces ``periodic[2] = True`` on a 2-D
grid and threads it in — the periodic guard covers that lane by itself.
The callers that pass the DEFAULT flags (the eager lumped/wire S-parameter
re-runs in ``rfx/probes/probes.py``, the subgridding runners) have the
length-1 guard as their only net, which is what
``test_length_one_axis_stays_self_adjacent`` pins, and it does red under
that mutation. Removing BOTH guards reds the end-to-end test with
inside = 7.549178e-02.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.pec import apply_pec_mask, tangential_edge_masks
from rfx.core.yee import init_materials, init_state


def _plates(ks, shape=(6, 6, 10)):
    m = np.zeros(shape, bool)
    for k in ks:
        m[1:5, 1:5, k] = True
    return jnp.asarray(m)


def _nnz(masks):
    return [int(jnp.sum(x)) for x in masks]


def test_face_plates_classify_like_interior_plates():
    """ORACLE — translation invariance of a geometric classification."""
    interior = _nnz(tangential_edge_masks(_plates([1, 8])))
    assert interior == [32, 32, 0], interior
    assert _nnz(tangential_edge_masks(_plates([2, 7]))) == interior
    # the defect case: same body, translated onto the two z faces
    assert _nnz(tangential_edge_masks(_plates([0, 9]))) == interior


def test_single_face_plate_is_the_control():
    """The wrap only fired when BOTH faces were occupied."""
    assert _nnz(tangential_edge_masks(_plates([0]))) == [16, 16, 0]
    assert _nnz(tangential_edge_masks(_plates([9]))) == [16, 16, 0]


def test_periodic_axis_keeps_the_wrap_for_a_seam_straddling_body():
    """On a periodic axis cell 0 and cell n-1 really are neighbours."""
    m = np.zeros((8, 8, 8), bool)
    m[[7, 0], 3:5, 3:5] = True          # 1 cell either side of the seam
    m = jnp.asarray(m)
    per = tangential_edge_masks(m, (True, False, False))
    non = tangential_edge_masks(m, (False, False, False))
    assert int(jnp.sum(per[0])) == 8, int(jnp.sum(per[0]))
    assert int(jnp.sum(non[0])) == 0, int(jnp.sum(non[0]))
    # 2 cells either side is contiguous under either convention
    m2 = np.zeros((8, 8, 8), bool)
    m2[[6, 7, 0, 1], 3:5, 3:5] = True
    m2 = jnp.asarray(m2)
    assert (int(jnp.sum(tangential_edge_masks(m2, (True, False, False))[0]))
            == int(jnp.sum(tangential_edge_masks(m2)[0])) == 16)


def test_length_one_axis_stays_self_adjacent():
    """The 2-D lane: nz == 1, so the wrap is what makes a body self-adjacent
    along z. Without this guard every 2-D interior PEC vanishes."""
    m = np.zeros((10, 10, 1), bool)
    m[3:7, 3:7, 0] = True
    m = jnp.asarray(m)
    ex, ey, ez = tangential_edge_masks(m)     # default flags, no periodic
    assert int(jnp.sum(ez)) == int(jnp.sum(m)) == 16
    assert int(jnp.sum(ex)) == int(jnp.sum(ey)) == 16


@pytest.mark.parametrize("shape", [(9, 8, 7), (16, 16, 16), (5, 5, 5)])
def test_interior_is_bit_identical_to_the_pre_689_roll_rule(shape):
    """The change is confined to the two boundary slices of each axis."""
    rng = np.random.default_rng(689)
    m = jnp.asarray(rng.random(shape) > 0.5)
    new = tangential_edge_masks(m)
    for ax in range(3):
        old = m & (jnp.roll(m, 1, axis=ax) | jnp.roll(m, -1, axis=ax))
        sl = [slice(None)] * 3
        sl[ax] = slice(1, -1)
        sl = tuple(sl)
        assert bool(jnp.all(new[ax][sl] == old[sl])), (shape, ax)


def test_periodic_all_true_reproduces_the_pre_689_rule_exactly():
    """The wrap branch is still literally the old rule, byte for byte."""
    rng = np.random.default_rng(4)
    m = jnp.asarray(rng.random((9, 8, 7)) > 0.5)
    new = tangential_edge_masks(m, (True, True, True))
    for ax in range(3):
        old = m & (jnp.roll(m, 1, axis=ax) | jnp.roll(m, -1, axis=ax))
        assert bool(jnp.all(new[ax] == old)), ax


def test_apply_pec_mask_forwards_the_periodic_flags():
    shape = (6, 6, 10)
    st = init_state(shape)
    st = st._replace(ex=jnp.ones(shape), ey=jnp.ones(shape),
                     ez=jnp.ones(shape))
    pm = _plates([0, 9])
    zeroed_non = float(jnp.sum(1.0 - apply_pec_mask(st, pm).ez))
    zeroed_per = float(jnp.sum(
        1.0 - apply_pec_mask(st, pm, (False, False, True)).ez))
    assert zeroed_non == 0.0, zeroed_non
    assert zeroed_per == 32.0, zeroed_per


def test_sheet_ctx_and_pec_mask_share_one_neighbour_rule():
    """#677 G4 footprint identity, re-checked under the new argument."""
    from rfx.materials.thin_conductor import (
        SheetImpedanceSpec, build_sheet_impedance_ctx)
    m = _plates([0])
    spec = SheetImpedanceSpec(mask=m, normal_axis=2, g_sheet=1.0,
                              sigma_sheet=jnp.where(m, 1.0, 0.0))
    for per in [(False, False, False), (False, False, True),
                (True, True, True)]:
        ctx = build_sheet_impedance_ctx([spec], periodic=per)
        ref = tangential_edge_masks(m, per)
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
    """The length-1 guard, end to end. The probe sits inside the PEC on
    purpose; the expected preflight line is quoted in the module docstring
    and ``skip_preflight`` only silences it, it is not a config fix."""
    res = _tmz_sim().run(n_steps=200, skip_preflight=True)
    ts = np.abs(np.asarray(res.time_series))
    inside, outside = float(ts[:, 0].max()), float(ts[:, 1].max())
    # guards kept: 0.0 / 1.588587e-02.  zero pad: 7.549178e-02 / 3.035763e-02
    assert inside == 0.0, (inside, outside)
    assert outside > 1e-3, (inside, outside)
