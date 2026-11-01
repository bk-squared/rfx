"""Static JAX metadata for host realization records."""
from contextvars import ContextVar
import weakref


_forward_geometry = ContextVar("forward_geometry", default=None)


def publish_forward_geometry(grid, record):
    """Bridge the graded runner result to its immediate legacy field packer."""
    _forward_geometry.set((id(grid), weakref.ref(grid.dx_arr), record))
    return record


def take_forward_geometry(grid):
    """Consume only this execution's record, without retaining its grid."""
    found = _forward_geometry.get()
    _forward_geometry.set(None)
    return (found[2] if found is not None and found[0] == id(grid)
            and found[1]() is grid.dx_arr else None)


class GeometryMetadata:
    """Keep execution attachments outside the historical record dictionary."""

    __slots__ = ("realization_table", "_packed_record", "_conductors",
                 "_report_grid", "_quantity_grid")


class _Metadata:
    def __init__(self, record):
        self.record = record

    def __hash__(self):
        return id(self.record)

    def __eq__(self, other):
        return isinstance(other, _Metadata) and self.record is other.record


def register_metadata(cls):
    """Host records are static auxiliary data, never differentiable leaves."""
    from jax.tree_util import register_pytree_node
    register_pytree_node(cls, lambda record: ((), _Metadata(record)),
                        lambda metadata, children: metadata.record)
