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

    __slots__ = ("realization_table", "_conductors",
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


def with_forward_geometry(base):
    """Attach host geometry without changing the legacy positional tuple."""
    import inspect
    from jax.tree_util import GetAttrKey, register_pytree_with_keys

    class Result(base):
        def __new__(cls, *args, realized_geometry=None, **kwargs):
            result = super().__new__(cls, *args, **kwargs)
            result._realized_geometry = realized_geometry
            return result

        @property
        def realized_geometry(self):
            return getattr(self, "_realized_geometry", None)

        def _replace(self, **kwargs):
            record = kwargs.pop("realized_geometry", self.realized_geometry)
            result = super()._replace(**kwargs)
            result._realized_geometry = record
            return result

    Result.__name__, Result.__qualname__, Result.__module__ = base.__name__, base.__qualname__, base.__module__
    Result.__doc__ = base.__doc__
    signature = inspect.signature(base)
    Result.__signature__ = signature.replace(parameters=[*signature.parameters.values(),
        inspect.Parameter("realized_geometry", inspect.Parameter.KEYWORD_ONLY, default=None)])
    register_pytree_with_keys(Result,
        lambda result: (tuple((GetAttrKey(name), value) for name, value in zip(base._fields, result)),
                        _Metadata(result.realized_geometry)),
        lambda metadata, values: Result(*values, realized_geometry=metadata.record))
    return Result
