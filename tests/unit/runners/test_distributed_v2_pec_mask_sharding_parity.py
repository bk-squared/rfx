"""#1053 leg 1: v2 shards the PEC mask exactly the way distributed_nu does.

``distributed_nu`` has a purpose-built sharder, ``shard_pec_mask_x_slab``.
``distributed_v2`` does not use it -- it reuses the shared ``split_array_x``
primitive it already shards ``eps_r`` / ``sigma`` / ``mu_r`` with, because v2
has already padded ``pec_mask`` to ``nx_padded`` by the time it needs slabs and
``shard_pec_mask_x_slab`` would pad it a second time.

That is only safe while the two produce the SAME slabs, and "the same" has to
mean bit-identical, not equivalent-looking: the two lanes realize their edge
masks from these slabs independently, per rank, per step, so a one-row
disagreement about which ghost carries what is a physics divergence at the seam
and nowhere else. This file is that equality, checked rather than assumed.

The three conventions being compared, all load-bearing:

* the ``pad_x`` alignment cells at high x are filled with ``True``
  (``distributed_v2.py:601-602``; ``distributed_nu.py:525-527``, "consistent
  with high-x PEC padding");
* the PHYSICAL-boundary ghost rows -- device 0's left, device N-1's right --
  are filled with ``False``. They were ``True`` until #931, which "would put a
  spurious PEC wall on the whole x_lo / x_hi node plane of the outer ranks,
  shorting a CPML face" (``distributed_nu.py:538-547``);
* the INTERIOR ghost rows carry the seam neighbour's real value, which is what
  lets a rank's first and last real cell see their true x neighbour under the
  four-incident-cell rule.

No device is needed: both paths are pure array construction.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from rfx.runners._distributed_common import split_array_x
from rfx.runners.distributed_nu import shard_pec_mask_x_slab


class _GridStub:
    """The seven scalars ``shard_pec_mask_x_slab`` reads off a ShardedNUGrid.

    Deliberately a stub and not a real grid: the point is that the sharder
    needs nothing else, which is why v2 (which has no ``ShardedNUGrid``) can
    be compared against it at all.
    """

    def __init__(self, nx, pad_x, ny, nz, n_devices, ghost=1):
        self.n_devices = n_devices
        self.pad_x = pad_x
        self.nx_per_rank = (nx + pad_x) // n_devices
        self.ghost_width = ghost
        self.nx_local = self.nx_per_rank + 2 * ghost
        self.ny = ny
        self.nz = nz
        self.nx = nx


@pytest.mark.parametrize("nx,pad_x,n_devices", [
    (8, 0, 2),
    (7, 1, 2),      # odd nx: the pad lane, where the two padders must agree
    (16, 0, 2),
    (15, 1, 2),     # odd nx again, at a size where a rank owns 8 real cells
    (12, 0, 4),     # more than two ranks: two interior seams
])
def test_v2_pec_mask_slabs_are_bit_identical_to_the_nu_sharder(
        nx, pad_x, n_devices):
    ny = nz = 6
    ghost = 1
    grid = _GridStub(nx, pad_x, ny, nz, n_devices, ghost)
    rng = np.random.default_rng(1053)
    mask = jnp.asarray(rng.random((nx, ny, nz)) > 0.55)

    nu = shard_pec_mask_x_slab(mask, grid)

    padded = mask
    if pad_x:
        padded = jnp.pad(mask, ((0, pad_x), (0, 0), (0, 0)),
                         constant_values=True)
    v2 = split_array_x(padded, n_devices, ghost, pad_value=False)
    v2 = v2.reshape(n_devices * grid.nx_local, ny, nz)

    assert v2.dtype == nu.dtype == jnp.bool_, (
        f"dtype drift: v2 {v2.dtype}, nu {nu.dtype}")
    assert v2.shape == nu.shape, f"shape: v2 {v2.shape}, nu {nu.shape}"
    assert bool(jnp.array_equal(v2, nu)), (
        f"nx={nx} pad_x={pad_x} n_devices={n_devices}: v2's "
        "split_array_x(pad_value=False) path and distributed_nu's "
        "shard_pec_mask_x_slab disagree about the PEC slab layout. The two "
        "lanes realize edge masks from these slabs independently, so this is "
        "a seam physics divergence, not a formatting difference. "
        f"first differing row: "
        f"{int(np.argmax(np.any(np.asarray(v2) != np.asarray(nu), axis=(1, 2))))}")


def test_the_physical_boundary_ghosts_are_False_not_True():
    """The #931 fix, pinned on the v2 path specifically.

    A ``True`` here shorts the whole x_lo / x_hi node plane of the outer
    ranks. It is invisible in a closed PEC domain, where that plane is already
    metal, and wrong in a CPML domain, where it is an absorber.
    """
    nx, ny, nz, n_devices, ghost = 8, 6, 6, 2, 1
    nx_per = nx // n_devices
    nx_local = nx_per + 2 * ghost
    mask = jnp.ones((nx, ny, nz), dtype=bool)          # metal EVERYWHERE
    slabs = split_array_x(mask, n_devices, ghost, pad_value=False)

    assert not bool(jnp.any(slabs[0, 0])), (
        "device 0's LEFT ghost row is PEC even though the physical-boundary "
        "pad must be False (#931)")
    assert not bool(jnp.any(slabs[n_devices - 1, nx_local - 1])), (
        "device N-1's RIGHT ghost row is PEC; same #931 rule")
    # and the interior ghosts DO carry the neighbour, which is the other half
    # of the convention.
    assert bool(jnp.all(slabs[0, nx_local - 1])), (
        "device 0's right ghost lost the seam neighbour's PEC status")
    assert bool(jnp.all(slabs[1, 0])), (
        "device 1's left ghost lost the seam neighbour's PEC status")
