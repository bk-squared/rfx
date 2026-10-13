"""Retain the existing eager shard-map callables without changing their bodies.

JAX's eager shard-map interpreter constructs HashablePartial and jit wrappers
for matching specs and applying individual primitives. Their weakly held
function identities otherwise expire between operations. Canonicalize those
static configurations and retain the jit wrappers in bounded caches; leave
the interpreter, primitive programs, and compilation options unchanged.
"""
from functools import lru_cache
from types import SimpleNamespace

from jax._src import shard_map


_partial = shard_map.HashablePartial
_jit = shard_map.api.jit


@lru_cache(maxsize=4096)
def _partial_for(function, *args, **kwargs):
    return _partial(function, *args, **kwargs)


@lru_cache(maxsize=4096)
def _jit_for(function, *args, **kwargs):
    return _jit(function, *args, **kwargs)


def retain_eager_callables():
    """Cache only the constructors used by JAX's eager shard-map interpreter."""
    shard_map.HashablePartial = _partial_for
    shard_map.api = SimpleNamespace(**{**vars(shard_map.api), 'jit': _jit_for})
