"""The two NU lanes must classify PEC tangential edges by ONE rule.

Defect (external review against e1a19a9).
``rfx/boundaries/pec.py::tangential_edge_masks`` is the single source of the
thin-sheet neighbour rule, and #689 changed it: a NON-periodic axis of
length > 1 now uses a zero pad instead of ``jnp.roll``, so two one-cell
bodies on opposite domain faces stop seeing each other through the wrap.
``rfx/runners/distributed_nu.py::_apply_pec_mask_nu_shmap`` carried a
SECOND, hand-written copy of the same rule and did not follow, so the
single-device and distributed NU lanes disagreed on the same geometry.
The copy is gone; the shmap kernel now calls the shared helper.

There was no test discriminating the two lanes before this one.

ORACLE — LANE PARITY plus TRANSLATION INVARIANCE. Two 1-cell 4x4 plates in
a (6,6,10) domain, counting only the distributed lane's REAL cells so the
ghost convention is not what is being compared. The interior placements
never reach any boundary branch, so they supply the expected value
independently of both implementations:

    placement            single-device   distributed @ e1a19a9   fixed
    z faces  k=0 & k=9   [32, 32,  0]    [32, 32, 32]            [32, 32,  0]
    y faces  j=0 & j=5   [32,  0, 32]    [32, 32, 32]            [32,  0, 32]
    z interior k=1, k=8  [32, 32,  0]    [32, 32,  0]            [32, 32,  0]
    y interior j=1, j=4  [32,  0, 32]    [32,  0, 32]            [32,  0, 32]

The y row was not in the review report; the same inlined ``jnp.roll``
produced 32 spurious Ey edges there for the same reason.

THE SHARDED AXIS KEEPS THE WRAP, and that is not an oversight. A slab's
ghost rows carry the seam neighbour's PEC status
(``shard_pec_mask_x_slab``), and the wrap can only reach local indices 0
and ``nx_local-1`` — both ghosts, both forced ``False`` before the field is
touched. So no real cell ever sees it, and passing
``periodic=(True, False, False)`` reproduces the pre-fix x behaviour
bit-for-bit: ``test_sharded_axis_rule_is_bit_identical_to_the_old_roll``
checks that against the literal old expression on random masks.
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

from rfx.boundaries.pec import apply_pec_mask, tangential_edge_masks
from rfx.core.yee import init_state
from rfx.runners.distributed_nu import _apply_pec_mask_nu_shmap

NX, NY, NZ = 6, 6, 10
GHOST = 1


def _plates(axis, idx):
    """Two 1-cell 4x4 plates, normal to ``axis``, at layers ``idx``."""
    m = np.zeros((NX, NY, NZ), bool)
    sl = [slice(1, 5)] * 3
    for i in idx:
        sl[axis] = i
        m[tuple(sl)] = True
    return m


def _single_device_nnz(gmask):
    """What ``apply_pec_mask`` zeroes, per component."""
    st = init_state((NX, NY, NZ))
    one = jnp.ones((NX, NY, NZ), jnp.float32)
    out = apply_pec_mask(st._replace(ex=one, ey=one, ez=one),
                         jnp.asarray(gmask))
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
        # ghost rows: interior seams carry the neighbour's PEC status, the
        # two physical-boundary ghosts are PEC=True (shard_pec_mask_x_slab)
        slabs[base] = gmask[lo - 1] if d > 0 else True
        slabs[base + nx_local - 1] = gmask[hi] if d < n_devices - 1 else True
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


@pytest.mark.parametrize("axis,faces,interior,expected", [
    (2, [0, NZ - 1], [1, NZ - 2], [32, 32, 0]),
    (1, [0, NY - 1], [1, NY - 2], [32, 0, 32]),
])
def test_lanes_agree_and_are_translation_invariant(axis, faces, interior,
                                                   expected):
    """The oracle: same body, two placements, two lanes, one answer."""
    for idx in (interior, faces):
        single = _single_device_nnz(_plates(axis, idx))
        dist = _distributed_nnz(_plates(axis, idx))
        assert single == expected, (axis, idx, single)
        assert dist == expected, (axis, idx, dist)


def test_lanes_agree_across_a_real_shard_seam():
    """Same check on 2 ranks, so the plates straddle a slab seam."""
    if jax.device_count() < 2:
        pytest.skip("needs 2 devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    for axis, idx, expected in ((2, [0, NZ - 1], [32, 32, 0]),
                                (2, [1, NZ - 2], [32, 32, 0]),
                                (1, [0, NY - 1], [32, 0, 32])):
        g = _plates(axis, idx)
        assert _distributed_nnz(g, n_devices=2) == expected, (axis, idx)
        # and the seam itself: an x-spanning slab must classify the same
        # whether it is cut into 1 or 2 ranks
    xbar = np.zeros((NX, NY, NZ), bool)
    xbar[:, 2:4, 2:4] = True        # spans x across the seam
    assert (_distributed_nnz(xbar, n_devices=2)
            == _distributed_nnz(xbar, n_devices=1)
            == _single_device_nnz(xbar))


@pytest.mark.parametrize("shape", [(8, 6, 10), (5, 5, 5), (9, 8, 7)])
def test_sharded_axis_rule_is_bit_identical_to_the_old_roll(shape):
    """Regression guard for the x axis: passing ``periodic[0]=True`` must
    reproduce the inlined expression this kernel used to carry."""
    rng = np.random.default_rng(689)
    m = jnp.asarray(rng.random(shape) > 0.5)
    new = tangential_edge_masks(m, (True, False, False))
    old_x = m & (jnp.roll(m, 1, axis=0) | jnp.roll(m, -1, axis=0))
    assert bool(jnp.all(new[0] == old_x)), shape
    # ...and the other two axes must NOT be the old rule at the faces
    for ax in (1, 2):
        old = m & (jnp.roll(m, 1, axis=ax) | jnp.roll(m, -1, axis=ax))
        sl = [slice(None)] * 3
        sl[ax] = slice(1, -1)
        assert bool(jnp.all(new[ax][tuple(sl)] == old[tuple(sl)])), ax
