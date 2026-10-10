"""Mesh position is data, never an XLA partition-id (#1441).

Pass ``mesh_ranks(mesh)`` through the enclosing jit's arguments, then as the
first argument of each rank_shard_map. Capturing it as a compiled constant
instead lets SPMD partitioning reintroduce partition-id when slicing it.
An enclosing jit is refused on GPUs (tracker 1598).
"""
from functools import wraps
import os
import re

import jax
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P


OUTER_JIT_REFUSAL = (
    "rfx: a multi-device run on GPUs cannot be compiled inside an enclosing jax.jit "
    "(or lax.scan / jax.checkpoint): the compiler then encodes each device's position "
    "as an operation CUDA rejects ('Failed to add memset node to a CUDA graph', "
    "rfx tracker 1598). Either call run(devices=...) / forward(distributed=True) "
    "and jax.grad of it WITHOUT jax.jit — the runner compiles its own time loop — "
    "or start the process with XLA_FLAGS=--xla_gpu_enable_command_buffer= "
    "(set before importing jax)."
)


def _is_multi_device(mesh):
    return mesh.size > 1


def _is_gpu_mesh(mesh):
    """Unknown or mixed platforms conservatively count as GPU."""
    platforms = {getattr(device, "platform", None) for device in mesh.devices.flat}
    return None in platforms or len(platforms) != 1 or platforms == {"gpu"}


def _staged_by_enclosing_program():
    """Probe staging even when the enclosing program has no traced inputs."""
    try:
        return isinstance(jax.lax.add(np.float32(0), np.float32(0)), jax.core.Tracer)
    except Exception:
        return True


def _command_buffers_on():
    """Off only if the option is present and every occurrence has an empty value."""
    values = re.findall(r"(?:^|\s)--xla_gpu_enable_command_buffer=(\S*)",
                        os.environ.get("XLA_FLAGS", ""))
    return not values or any(values)


def mesh_ranks(mesh):
    """One int32 mesh position per shard, also on non-addressable meshes."""
    if (_is_multi_device(mesh) and _is_gpu_mesh(mesh)
            and _staged_by_enclosing_program() and _command_buffers_on()):
        raise NotImplementedError(OUTER_JIT_REFUSAL)
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
