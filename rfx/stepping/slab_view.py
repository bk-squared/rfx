"""What a per-slab setup producer receives (rfx #1402, W1 staging).

A multi-device run realizes the model on each x slab, not on the whole grid and
then cut: a whole-grid setup leaves whole-domain arrays on every device. Each
producer that builds a grid-shaped quantity (realized materials, occupancy
edge factors, PEC edge masks, pole fractions, CPML state) has ONE
implementation, which runs on the whole grid on one device and on a
:class:`SlabView` on several. The contract, pinned by
``tests/contracts/path_equivalence/test_slab_view_producers.py``::

    view.own(producer(slab inputs, view)) == producer(whole inputs)[view.x0:view.x1]

bit for bit, on every slab, for two and three devices.

A view receives:

* the slab's input rows ``[lo, hi)`` of every x-first input array, cut with
  :meth:`SlabView.rows`: the owned rows ``[x0, x1)`` plus ``halo_lo`` rows
  below and ``halo_hi`` rows above;
* the GLOBAL grid. Its one-dimensional metrics are small and every predicate
  that selects an arithmetic path (``is_constant``, "is this axis graded")
  must be decided on the whole axis, never on the slab: a slab of equal cells
  inside a graded axis would otherwise take the equal-cell arithmetic. A
  producer slices its x metric with :meth:`SlabView.rows` after deciding;
* ``first`` / ``last``: whether the slab holds a global x face, so the
  outside-the-lattice rule (edge replication, zero padding, CPML faces) is
  applied there and only there. At an interior seam the halo supplies the
  neighbour instead;
* ``globals``: whole-model scalars a producer would otherwise compute from
  the whole grid (the Courant budget for the pole stability refusal, dt).

Halo: every grid-shaped producer on the graded path reads the cell one step
back along x and none reads forward (``_material_bwd_neighbour`` in
``edge_averaged_materials`` and ``component_h_materials``, the occupancy edge
factors). So ``HALO_LO = 1`` on every slab but the first and ``HALO_HI = 0``.
This is read from the code; the contract test is what verifies it, and a
producer that needs more is a stencil statement to make before changing it.
"""
from typing import Any, Mapping, NamedTuple
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from rfx.stepping.slab import Slab, place, forward_sharding

HALO_LO = 1
HALO_HI = 0


class UniformSlabRows(NamedTuple):
    """Global indices of [extra low | low ghost | owned | high ghost]."""
    indices: Any
    outside: Any
    alignment: Any


def uniform_slab_rows(layout):
    indices = (np.arange(layout.n_devices)[:, None] * layout.nx_per_rank
               + np.arange(layout.nx_local + 1)[None, :] - layout.ghost_width - 1)
    return UniformSlabRows(indices, (indices < 0) | (indices >= layout.nx),
                           (indices >= layout.nx) & (indices < layout.nx_padded))


def stack_and_place(array, rows, mesh, *, outside="replicate"):
    """Gather uniform input rows, then place once on the leading device axis.

    Material and metric inputs replicate global endpoints; occupancy and
    masks use zero. Product padding is a separate application of slab.FILL.
    """
    nx = int(np.max(np.where(rows.outside, -1, rows.indices))) + 1
    layout = Slab(nx, mesh.size)
    sharding = forward_sharding(array)
    if sharding is not None and tuple(sharding.spec)[:1] == ('x',):
        return uniform_from_owned(array, layout, mesh, outside=outside)
    xp = np if isinstance(array, np.ndarray) else jnp
    stacked = xp.take(array, np.clip(rows.indices, 0, nx - 1), axis=0)
    if outside != "replicate":
        mask = rows.outside.reshape(rows.outside.shape + (1,) * (array.ndim - 1))
        stacked = xp.where(mask, xp.asarray(outside, dtype=array.dtype), stacked)
    return place(jnp.asarray(stacked), NamedSharding(mesh, P('x')))


def uniform_from_owned(array, layout, mesh, *, outside='replicate'):
    """Exchange owned x slabs into two low rows and one high row, locally."""
    from rfx.stepping.rank import rank_shard_map, mesh_ranks
    return _owned_map(layout, mesh, outside)(mesh_ranks(mesh), array)


@lru_cache(maxsize=128)
def _owned_map(layout, mesh, outside):
    from rfx.stepping.rank import rank_shard_map
    right = [(r, r + 1) for r in range(layout.n_devices - 1)]
    left = [(r + 1, r) for r in range(layout.n_devices - 1)]
    def extend(local, *, rank):
        low = jax.lax.ppermute(local[-2:], 'x', right)
        high = jax.lax.ppermute(local[:1], 'x', left)
        values = jnp.concatenate((low, local, high))
        indices = rank * layout.nx_per_rank + jnp.arange(layout.nx_local + 1) - 2
        shape = (-1,) + (1,) * (local.ndim - 1)
        if outside == 'replicate':
            endpoint = local[jnp.clip(layout.nx - rank * layout.nx_per_rank - 1, 0, local.shape[0] - 1)]
            fill = jnp.where((indices < 0).reshape(shape), local[0], endpoint)
        else:
            fill = jnp.asarray(outside, local.dtype)
        values = jnp.where(((indices >= 0) & (indices < layout.nx)).reshape(shape), values, fill)
        return values[None]
    return jax.jit(rank_shard_map(extend, mesh=mesh, in_specs=P('x'), out_specs=P('x'),
                                check_rep=False))


class UniformSlabView(NamedTuple):
    """Rank as row-index data; component identity remains static under vmap.

    Static geometry is realized with concrete SlabViews before this array
    producer. Metrics are selected only after whole-axis decisions.
    """
    indices: Any
    grid: Any
    globals: Mapping

    rank = None
    x0 = 0
    halo_lo = 2
    halo_hi = 1

    @property
    def nx(self):
        return int(self.grid.nx)

    @property
    def x1(self):
        return self.indices.shape[0] - self.halo_lo - self.halo_hi

    @property
    def lo(self):
        return -self.halo_lo

    @property
    def hi(self):
        return self.x1 + self.halo_hi

    def rows(self, a, axis=0):
        if axis == 0 and 'x_widths' in self.globals:
            return self.globals['x_widths']
        return jnp.take(jnp.asarray(a), jnp.clip(self.indices, 0, a.shape[axis] - 1), axis=axis)

    def own(self, a, axis=0):
        index = [slice(None)] * a.ndim
        index[axis] = slice(self.halo_lo, self.halo_lo + self.x1)
        return a[tuple(index)]


class SlabView(NamedTuple):
    rank: int
    n_devices: int
    x0: int                 # first owned global x row
    x1: int                 # one past the last owned global x row
    halo_lo: int            # input rows below x0 (0 on the first slab)
    halo_hi: int            # input rows above x1 (0 on the last slab)
    grid: Any               # the GLOBAL grid (1-D metrics only)
    globals: Mapping        # whole-model scalars

    @property
    def nx(self):
        return int(self.grid.nx)

    @property
    def first(self):
        return self.x0 == 0

    @property
    def last(self):
        return self.x1 == self.nx

    @property
    def lo(self):
        """Global x row of local input row 0."""
        return self.x0 - self.halo_lo

    @property
    def hi(self):
        return self.x1 + self.halo_hi

    def rows(self, a, axis=0):
        """The slab's input rows of a global array (or 1-D x metric)."""
        index = [slice(None)] * a.ndim
        index[axis] = slice(self.lo, self.hi)
        return a[tuple(index)]

    def own(self, a, axis=0):
        """The owned rows of a slab-local result."""
        index = [slice(None)] * a.ndim
        index[axis] = slice(self.halo_lo, self.halo_lo + self.x1 - self.x0)
        return a[tuple(index)]

    def local(self, i):
        """Local input row of a global x row inside ``[lo, hi)``."""
        if not self.lo <= i < self.hi:
            raise ValueError(f"x row {i} outside slab rows [{self.lo}, {self.hi})")
        return int(i - self.lo)


def slab_views(grid, n_devices, *, globals=None, halo_lo=HALO_LO, halo_hi=HALO_HI):
    """One view per device, with the runtime's x ownership (:class:`Slab`)."""
    layout = Slab(int(grid.nx), int(n_devices))
    views = []
    for rank, (x0, x1) in enumerate(layout.owned):
        # A slab that owns no row (ceil-divided layouts) reads none either.
        owns = x1 > x0
        view = SlabView(rank, layout.n_devices, x0, x1,
                        halo_lo if owns and x0 > 0 else 0,
                        halo_hi if owns and x1 < layout.nx else 0,
                        grid, dict(globals or {}))
        if layout.nx > 1 and view.hi - view.lo == 1:
            # The neighbour rules treat a length-1 axis as the 2-D lane's
            # self-adjacency and wrap it (rfx #1402, found by idm's review).
            raise ValueError(
                f"slab {rank} of {layout.n_devices} would read a single x row of {layout.nx}; "
                "a one-row slab takes the 2-D self-adjacency rule. Use fewer devices.")
        views.append(view)
    return tuple(views)


def uniform_apply(producer, *inputs, mesh, row_count, row_axis=0):
    """Vmap array operands only; preserve the producer's static Python objects.

    The partition is made during the first trace, before vmap can convert a
    declaration scalar into a weakly typed device array. Pole-axis producers
    may name row_axis=1; mixed records are flattened at their array leaves.
    """
    saved = {}

    def spatial(value):
        return (isinstance(value, (jax.Array, jax.core.Tracer, np.ndarray))
                and value.ndim >= 1 and row_axis < value.ndim
                and value.shape[row_axis] == row_count)

    def metadata(value):
        return not any(spatial(a) for a in jax.tree.leaves(value))

    def traced(*args):
        result = producer(*args)
        leaves, tree = jax.tree.flatten(result, is_leaf=metadata)
        dynamic = tuple(spatial(a) for a in leaves)
        if not saved:
            saved.update(tree=tree, leaves=leaves, dynamic=dynamic)
        return tuple(a if keep else None for a, keep in zip(leaves, dynamic))

    # Arithmetic is eager over the placed device axis; R5 retains only
    # stage inputs and outputs for reverse mode. Metadata bypasses vmap.
    mapped = jax.checkpoint(jax.vmap(traced))(*inputs)
    return jax.tree.unflatten(saved['tree'], [
        a if keep else original for a, original, keep in
        zip(mapped, saved['leaves'], saved['dynamic'])])


def uniform_from_cut(array, layout, mesh, *, outside='replicate'):
    """Extend an existing sharded slab by its predecessor's extra low row.

    This compatibility adapter never assembles the global grid. Public input
    staging can use stack_and_place directly; low-level callers already hand
    the runner cut slabs, whose one-cell halo needs one additional row.
    """
    from rfx.stepping.rank import rank_shard_map, mesh_ranks
    if array is None or np.ndim(array) == 0:
        return array
    if array.ndim == 4:
        return array
    return _cut_map(layout, mesh, outside)(mesh_ranks(mesh), array)


@lru_cache(maxsize=128)
def _cut_map(layout, mesh, outside):
    from rfx.stepping.rank import rank_shard_map
    def extend(local, *, rank):
        extra = jax.lax.ppermute(local[-3:-2], 'x',
                                [(r, r + 1) for r in range(layout.n_devices - 1)])
        values = jnp.concatenate((extra, local))
        indices = rank * layout.nx_per_rank + jnp.arange(layout.nx_local + 1) - 2
        if outside == 'replicate':
            high = local[jnp.clip(layout.nx - rank * layout.nx_per_rank, 1, layout.nx_local - 1)]
            fill = jnp.where((indices < 0)[:, None, None], local[1], high)
        else:
            fill = jnp.asarray(outside, local.dtype)
        values = jnp.where(((indices >= 0) & (indices < layout.nx))[:, None, None], values, fill)
        return values[None]
    return jax.jit(rank_shard_map(extend, mesh=mesh, in_specs=P('x'),
        out_specs=P('x'), check_rep=False))


def product_rows(array, layout, mesh, kind, *, row_axis=1, dt=None, pole_rows=False):
    """Drop the extra input row and apply the step's exact named fill policy."""
    from rfx.stepping.slab import pole_fill
    if array is None:
        return None
    fill = pole_fill(kind, dt)
    rows = uniform_slab_rows(layout)
    shape = [1] * array.ndim
    if pole_rows:
        row_axis = 2
    shape[0], shape[row_axis] = layout.n_devices, layout.nx_local
    outside = place(jnp.asarray(rows.outside[:, 1:].reshape(shape)), NamedSharding(mesh, P('x')))
    alignment = place(jnp.asarray(rows.alignment[:, 1:].reshape(shape)), NamedSharding(mesh, P('x')))
    stage = _product_map(mesh, row_axis, pole_rows, fill.alignment, fill.physical)
    return jax.checkpoint(stage)(array, outside, alignment)


@lru_cache(maxsize=128)
def _product_map(mesh, row_axis, pole_rows, alignment_fill, physical_fill):
    """Compile only row copies/fills; the owners' arithmetic stays eager."""
    def local(value, outside, alignment):
        if pole_rows:
            value = jnp.moveaxis(value, 2, 1)
        index = [slice(None)] * value.ndim
        index[row_axis] = slice(1, None)
        value = value[tuple(index)]
        pad = jnp.where(alignment, jnp.asarray(alignment_fill, value.dtype),
                        jnp.asarray(physical_fill, value.dtype))
        value = jnp.where(outside, pad, value)
        return value.reshape((value.shape[0] * value.shape[1],) + value.shape[2:])
    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(P('x'),) * 3,
                                out_specs=P('x'), check_vma=False))


@lru_cache(maxsize=128)
def _axis_map(mesh, unpack):
    """Only copies: expose or merge the device axis inside each shard."""
    def local(a):
        if unpack:
            return a[None]
        return a.reshape((a.shape[0] * a.shape[1],) + a.shape[2:])
    return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=P('x'),
                                out_specs=P('x'), check_vma=False))


def device_rows(value, mesh, *, unpack):
    """Convert packed field leaves without a cross-device reshape.

    A leaf is a field array (packed ``(rows, ny, nz)``, unpacked
    ``(device, rows, ny, nz)``). Pole coefficients are staged pole-first and
    are split by their leading axis instead (rfx #1834), so any other rank
    is refused.
    """
    field = 3 if unpack else 4
    def convert(a):
        if a.ndim != field:
            raise ValueError(
                f"device_rows takes field ({field}-D) leaves when unpack={unpack}; "
                f"got shape {a.shape} (pole coefficients are pole-first)")
        return _axis_map(mesh, unpack)(a)
    return jax.tree.map(convert, value)
