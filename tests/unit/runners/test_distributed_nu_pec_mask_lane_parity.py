"""The two NU lanes must realize PEC by ONE rule (#931 §1.7).

History. ``rfx/runners/distributed_nu.py::_apply_pec_mask_nu_shmap``
carried a hand-written copy of the neighbour rule and did not follow #689
(zero pad on a non-periodic axis), so the lanes disagreed at the y and z
domain faces. The copy was replaced by a call to the shared
``tangential_edge_masks``; then #931 replaced the sheet rule itself with
the VOLUME rule (§1.2) on every single-device lane and this call site
became the last user of the old classifier — the same divergence, one
layer down. It now calls ``realized_pec_edge_masks``, the single owner.

ORACLE — an independent transcription of §1.2, written out per component
in this file with explicit zero-padded backward shifts, plus lane parity
and translation invariance. Two 1-cell 4x4 plates in a (6,6,10) domain,
counting only the distributed lane's REAL cells so the ghost convention
is not what is being compared:

    placement            §1.2 oracle    pre-#931 sheet rule
    z interior k=1, k=8  [80, 80, 50]   [32, 32,  0]
    z faces  k=0 & k=9   [60, 60, 50]   [32, 32,  0]
    y interior j=1, j=4  [80, 50, 80]   [32,  0, 32]
    y faces  j=0 & j=5   [60, 50, 60]   [32,  0, 32]

The old rule gave one wall per plate; the volume rule gives both faces,
which is the whole point of #931. The face rows lose one plane because
the far face of the plate on the last cell falls outside the array — the
domain BC owns that plane (§1.2).

THE SHARDED AXIS KEEPS THE WRAP: a slab's ghost rows carry the seam
neighbour's status and the wrap can only reach local indices 0 and
``nx_local-1``, both ghosts, both forced ``False`` before the field is
touched, so no real cell sees it. Since #931 the PHYSICAL-boundary ghost
is ``False`` (the zero pad), not ``True``: under the volume rule a
``True`` ghost would put a spurious wall on the whole x_lo/x_hi node
plane of the outer ranks.

Sheets are not on this lane: it shards a CELL mask and a sheet owns no
cell, so ``forward(distributed=True)`` refuses one loudly. That refusal is
pinned here too, next to the single-device lane's positive sheet result.
"""

import os
# Same convention as the sibling distributed modules: 2 virtual devices so
# the seam case has a real seam. ``setdefault`` so a caller that already
# set XLA_FLAGS wins.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh

from rfx.boundaries.pec import SheetSpec, apply_pec_mask
from rfx.core.yee import init_state
from rfx.runners.distributed_nu import _apply_pec_mask_nu_shmap

NX, NY, NZ = 6, 6, 10
GHOST = 1


def _plates(axis, idx):
    """Two 1-cell 4x4 plates, normal to ``axis``, at cell layers ``idx``."""
    m = np.zeros((NX, NY, NZ), bool)
    sl = [slice(1, 5)] * 3
    for i in idx:
        sl[axis] = i
        m[tuple(sl)] = True
    return m


def _oracle_counts(cell_mask):
    """§1.2 written out here, independent of the library implementation.

    ``Mx = C | C[j-1] | C[k-1] | C[j-1,k-1]`` and cyclically, with an
    explicit ZERO pad on every backward shift (the non-periodic #689
    convention). Returns the per-component count of PEC entries.
    """
    C = np.asarray(cell_mask, dtype=bool)

    def back(a, ax):
        out = np.zeros_like(a)
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        dst[ax] = slice(1, None)
        src[ax] = slice(0, -1)
        out[tuple(dst)] = a[tuple(src)]
        return out

    mx = C | back(C, 1) | back(C, 2) | back(back(C, 1), 2)
    my = C | back(C, 0) | back(C, 2) | back(back(C, 0), 2)
    mz = C | back(C, 0) | back(C, 1) | back(back(C, 0), 1)
    return [int(m.sum()) for m in (mx, my, mz)]


def _single_device_nnz(gmask, sheets=()):
    """What the single-device ``apply_pec_mask`` zeroes, per component."""
    st = init_state((NX, NY, NZ))
    one = jnp.ones((NX, NY, NZ), jnp.float32)
    out = apply_pec_mask(st._replace(ex=one, ey=one, ez=one),
                         None if gmask is None else jnp.asarray(gmask),
                         sheets=sheets)
    return [int(np.sum(np.asarray(c) == 0.0))
            for c in (out.ex, out.ey, out.ez)]


def _distributed_nnz(gmask, n_devices=1):
    """The same, through the real shard_map kernel; real cells only."""
    nx_per = NX // n_devices
    nx_local = nx_per + 2 * GHOST
    slabs = np.zeros((n_devices * nx_local, NY, NZ), bool)
    for d in range(n_devices):
        lo, hi = d * nx_per, (d + 1) * nx_per
        base = d * nx_local
        slabs[base + GHOST:base + GHOST + nx_per] = gmask[lo:hi]
        # ghost rows: interior seams carry the neighbour's PEC status; the
        # two physical-boundary ghosts are the #689 zero pad (False since
        # #931 — see shard_pec_mask_x_slab).
        slabs[base] = gmask[lo - 1] if d > 0 else False
        slabs[base + nx_local - 1] = gmask[hi] if d < n_devices - 1 else False
    shape = (n_devices * nx_local, NY, NZ)
    mesh = Mesh(np.asarray(jax.devices()[:n_devices]).reshape(n_devices), ("x",))
    st = init_state(shape)
    one = jnp.ones(shape, jnp.float32)
    out = _apply_pec_mask_nu_shmap(st._replace(ex=one, ey=one, ez=one),
                                   jnp.asarray(slabs), mesh, n_devices,
                                   nx_local)
    tot = [0, 0, 0]
    for c, comp in enumerate((out.ex, out.ey, out.ez)):
        a = np.asarray(comp)
        for d in range(n_devices):
            base = d * nx_local
            tot[c] += int(np.sum(a[base + GHOST:base + GHOST + nx_per] == 0.0))
    return tot


@pytest.mark.parametrize("axis,faces,interior,expected_interior,expected_faces", [
    (2, [0, NZ - 1], [1, NZ - 2], [80, 80, 50], [60, 60, 50]),
    (1, [0, NY - 1], [1, NY - 2], [80, 50, 80], [60, 50, 60]),
])
def test_lanes_agree_and_match_the_volume_rule(
        axis, faces, interior, expected_interior, expected_faces):
    """Same body, two placements, two lanes, one answer — and it is §1.2."""
    for idx, expected in ((interior, expected_interior),
                          (faces, expected_faces)):
        gmask = _plates(axis, idx)
        assert _oracle_counts(gmask) == expected, (axis, idx)
        assert _single_device_nnz(gmask) == expected, (axis, idx)
        assert _distributed_nnz(gmask) == expected, (axis, idx)


def test_a_one_cell_slab_realizes_both_faces_on_both_lanes():
    """The #931 headline, on the lane that used to disagree hardest."""
    for k in (2, 5):
        g = np.zeros((NX, NY, NZ), bool)
        g[1:5, 1:5, k] = True
        st = init_state((NX, NY, NZ))
        one = jnp.ones((NX, NY, NZ), jnp.float32)
        out = apply_pec_mask(st._replace(ex=one, ey=one, ez=one),
                             jnp.asarray(g))
        ex = np.asarray(out.ex)
        assert bool((ex[1:5, 1:5, k] == 0.0).all()), "near face missing"
        assert bool((ex[1:5, 1:5, k + 1] == 0.0).all()), "FAR face missing"
        assert _distributed_nnz(g) == _single_device_nnz(g)


def test_lanes_agree_across_a_real_shard_seam():
    """Same check on 2 ranks, so the plates straddle a slab seam."""
    if jax.device_count() < 2:
        pytest.skip("needs 2 devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    for axis, idx in ((2, [0, NZ - 1]), (2, [1, NZ - 2]), (1, [0, NY - 1])):
        g = _plates(axis, idx)
        assert _distributed_nnz(g, n_devices=2) == _oracle_counts(g), (axis, idx)
    # and the seam itself: an x-spanning slab must classify the same
    # whether it is cut into 1 or 2 ranks
    xbar = np.zeros((NX, NY, NZ), bool)
    xbar[:, 2:4, 2:4] = True        # spans x across the seam
    assert (_distributed_nnz(xbar, n_devices=2)
            == _distributed_nnz(xbar, n_devices=1)
            == _single_device_nnz(xbar))


def test_a_sheet_is_realized_on_the_single_device_lane():
    """A sheet owns no cell, so it exists only through ``sheets=``."""
    foot = np.zeros((NX, NY, NZ), bool)
    foot[1:5, 1:5, 4] = True
    sheet = SheetSpec(normal_axis=2, plane=4, footprint=jnp.asarray(foot))
    counts = _single_device_nnz(None, sheets=(sheet,))
    # one plane, in-plane components only, both end nodes in the footprint
    assert counts[2] == 0, "the sheet's NORMAL edge must stay live (#690)"
    assert counts[0] > 0 and counts[1] > 0
    # the footprint is realized CLOSED: 4x5 Ex edges and 5x4 Ey edges on
    # the plane (both end nodes of each edge inside the 4x4-node footprint
    # extended to 5 nodes per in-plane axis is what the closed rectangle
    # gives; here the footprint is 4 nodes wide, so 3 interior edges per
    # row on each axis)
    assert counts[0] == 3 * 4 and counts[1] == 4 * 3
    # ... and a bare cell mask of the same footprint would be a VOLUME
    # with two faces, i.e. a different object entirely
    vol = np.zeros((NX, NY, NZ), bool)
    vol[1:5, 1:5, 4] = True
    assert _single_device_nnz(vol)[2] > 0, "a volume shorts its normal edge"


def _distributed_occ_nnz(occ, n_devices=1):
    """The SOFT twin of ``_distributed_nnz``; real cells only."""
    from rfx.runners.distributed_nu import _apply_pec_occupancy_nu_shmap
    nx_per = NX // n_devices
    nx_local = nx_per + 2 * GHOST
    slabs = np.zeros((n_devices * nx_local, NY, NZ), np.float32)
    for d in range(n_devices):
        lo, hi = d * nx_per, (d + 1) * nx_per
        base = d * nx_local
        slabs[base + GHOST:base + GHOST + nx_per] = occ[lo:hi]
        slabs[base] = occ[lo - 1] if d > 0 else 0.0
        slabs[base + nx_local - 1] = occ[hi] if d < n_devices - 1 else 0.0
    shape = (n_devices * nx_local, NY, NZ)
    mesh = Mesh(np.asarray(jax.devices()[:n_devices]).reshape(n_devices), ("x",))
    st = init_state(shape)
    one = jnp.ones(shape, jnp.float32)
    out = _apply_pec_occupancy_nu_shmap(st._replace(ex=one, ey=one, ez=one),
                                        jnp.asarray(slabs), mesh, n_devices,
                                        nx_local)
    tot = [0, 0, 0]
    for c, comp in enumerate((out.ex, out.ey, out.ez)):
        a = np.asarray(comp)
        for d in range(n_devices):
            base = d * nx_local
            tot[c] += int(np.sum(a[base + GHOST:base + GHOST + nx_per] == 0.0))
    return tot


@pytest.mark.parametrize("axis,idx", [
    (2, [1, NZ - 2]), (2, [0, NZ - 1]), (1, [0, NY - 1]), (0, [0, NX - 1]),
])
def test_soft_equals_hard_on_the_distributed_lane_including_faces(axis, idx):
    """The differentiable lane realizes the SAME conductor as the hard one,
    on the shmap twin, with a body sitting on a non-periodic domain face.

    §1.6: the soft rule is the noisy-OR of the four incident cells,
    ``M = 1 - prod(1 - o_c)``, under the same #689 shifts — so at binary
    occupancy it is bit-identical to §1.2. That identity is pinned on the
    single-device lane in tests/contracts; this is the shmap twin, which
    is where the two spellings drifted apart before (``rfx/boundaries/pec.py``
    used to record that ``apply_pec_occupancy``,
    ``_apply_pec_occupancy_nu_shmap`` and ``geometry/smoothing.py`` still
    spelled the rule with ``roll`` while the hard path zero-padded, so hard
    and soft disagreed at a non-periodic domain face and no test
    discriminated). The face rows are the discriminating ones.
    """
    g = _plates(axis, idx)
    occ = g.astype(np.float32)
    assert _distributed_occ_nnz(occ) == _distributed_nnz(g) == _oracle_counts(g)


def test_the_distributed_nu_forward_lane_refuses_a_sheet():
    """It shards a CELL mask; a silently-absent sheet is not acceptable."""
    import warnings

    import rfx

    sim = rfx.Simulation(freq_max=10e9, domain=(0.06, 0.01, 0.01), dx=1e-3,
                         cpml_layers=4, dz_profile=np.full(18, 1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add(rfx.Box((0.002, 0.002, 0.005), (0.008, 0.008, 0.005)),
                material="pec")
    with pytest.raises(NotImplementedError, match="PEC sheets"):
        sim.forward(distributed=True, n_steps=2)
