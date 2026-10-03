"""Mesh position is data, never an XLA partition-id (#1441).

Pass ``mesh_ranks(mesh)`` through the enclosing jit's arguments, then as the
first argument of each rank_shard_map. Capturing it as a compiled constant
instead lets SPMD partitioning reintroduce partition-id when slicing it.
"""
from functools import wraps

import jax
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P


def mesh_ranks(mesh):
    """One int32 mesh position per shard, also on non-addressable meshes."""
    sharding = NamedSharding(mesh, P("x"))
    return jax.make_array_from_callback(
        (mesh.size,), sharding,
        lambda index: np.arange(mesh.size, dtype=np.int32)[index])


def rank_shard_map(fun, *, mesh, in_specs, out_specs, check_rep=False):
    """shard_map with an explicit first rank-array input and scalar ``rank``."""
    @wraps(fun)
    def local(rank_shard, *args):
        return fun(*args, rank=rank_shard[0])

    # PartitionSpec is a tuple subclass, but denotes a single tree prefix.
    specs = in_specs if isinstance(in_specs, tuple) and not isinstance(in_specs, P) else (in_specs,)
    return shard_map(local, mesh=mesh, in_specs=(P("x"), *specs),
                     out_specs=out_specs, check_rep=check_rep)
