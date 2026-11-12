"""Join rank-owned records without replicating the packed carry.

Mesh position is an input. Static owner offsets select disjoint writes inside
rank_shard_map; only the reduced chunks and the returned record are replicated.
"""
from functools import lru_cache
from math import prod

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from rfx.stepping.rank import mesh_ranks, rank_shard_map


@jax.custom_jvp
def _restore_zero(total, negative):
    return jnp.where((total == 0) & negative, jnp.asarray(-0.0, total.dtype), total)


@_restore_zero.defjvp
def _restore_zero_jvp(primals, tangents):
    return _restore_zero(*primals), tangents[0]


def _signed_sum(part):
    total = lax.psum(part, "x")

    def restore(value, summed):
        negative = lax.psum(((value == 0) & jnp.signbit(value)).astype(jnp.int32), "x") > 0
        return _restore_zero(summed, negative)

    if jnp.iscomplexobj(part):
        return lax.complex(restore(part.real, total.real), restore(part.imag, total.imag))
    return restore(part, total)


def packed_pieces(shapes, sizes):
    """Offsets and shapes in one packed carry row, in unpack order."""
    offset = 0
    for shape, size in zip(shapes, sizes):
        yield offset, tuple(int(v) for v in shape)
        offset += int(size)


def record_bytes(arrays):
    return sum(prod(a.shape) * jnp.dtype(jax.dtypes.canonicalize_dtype(a.dtype)).itemsize for a in arrays)


def chunk_shape(shape, dtype, record_size, x_axis):
    """Equal tiles: leading axis, then the first non-x trailing axis."""
    chunk = list(shape)
    limit = record_size // (40 * jnp.dtype(dtype).itemsize)
    axes = [0] + [a for a in range(1, len(shape)) if a != x_axis][:1]
    for axis in axes:
        other = prod(chunk[:axis] + chunk[axis + 1:])
        cap = max(1, min(chunk[axis], limit // max(1, other)))
        chunk[axis] = next(v for v in range(cap, 0, -1) if shape[axis] % v == 0)
    return tuple(chunk)


@lru_cache(maxsize=256)
def _join_program(kind, mesh, owners, shape, dtype, chunk, summed):
    """Cache only static layouts and programs, never carries or returned arrays.

Each piece is (packed offset, source shape, destination start, owned extent).
Snapshot source shapes include tail padding; their extents exclude it.
The chunk axes have identical extents on every owner (x is not chunked).
"""
    counts = tuple(s // c for s, c in zip(shape, chunk))
    branches = []
    for pieces in owners:
        def write(args, pieces=pieces):
            local, starts = args
            part = jnp.zeros(chunk, dtype)
            for offset, source_shape, destination, extent in pieces:
                if not prod(extent):
                    continue
                source = lax.slice_in_dim(local[0], offset, offset + prod(source_shape)).reshape(source_shape)
                sizes = tuple(min(c, e) for c, e in zip(chunk, extent))
                source_starts = tuple(start if c != s else 0 for start, c, s in zip(starts, chunk, shape))
                value = lax.dynamic_slice(source, source_starts, sizes)
                part = lax.dynamic_update_slice(part, value.astype(dtype), destination)
            return part
        branches.append(write)

    def join(local, *, rank):
        def body(index, result):
            starts = tuple((index // prod(counts[a + 1:]) % counts[a]) * c
                           for a, c in enumerate(chunk))
            part = lax.switch(rank, branches, (local, starts))
            total = lax.psum(part, "x") if summed else _signed_sum(part)
            return lax.dynamic_update_slice(result, total.astype(dtype), starts)
        return lax.fori_loop(0, prod(counts), body, jnp.zeros(shape, dtype))

    return jax.jit(rank_shard_map(join, mesh=mesh, in_specs=P("x"), out_specs=P(), check_rep=False))


def join_owned(buffer, owners, shape, dtype, *, mesh, kind, record_size, x_axis=1, summed=False):
    """Return one replicated array, reducing at most one tile at a time.

Moments use a true sum of the unpacked arrays; cell records preserve signed
zeros in each real/imaginary component, with the tangent of the floating sum.
"""
    shape = tuple(int(v) for v in shape)
    dtype = jnp.dtype(jax.dtypes.canonicalize_dtype(dtype))
    owners = tuple(tuple(pieces) for pieces in owners)
    chunk = chunk_shape(shape, dtype, record_size, x_axis)
    program = _join_program(kind, mesh, owners, shape, dtype, chunk, summed)
    return program(mesh_ranks(mesh), buffer.reshape((mesh.size, -1)))
