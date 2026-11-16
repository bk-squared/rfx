"""Static JAX metadata for host realization records."""
from contextvars import ContextVar
from dataclasses import dataclass
import struct
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

    def __reduce__(self):
        attachments = {name: getattr(self, name) for name in self.__slots__
                       if hasattr(self, name)}
        # A borrowed kernel owner is execution-local, like table.prepared.
        if isinstance(attachments.get("_conductors"), weakref.ReferenceType):
            attachments.pop("_conductors")
        return _restore_geometry, (type(self), vars(self), attachments)


def _restore_geometry(cls, values, attachments):
    """Reconstruct frozen records through their keyword metadata constructor.

    Both pickle and deepcopy use this path; inherited slots cannot be restored
    by the default reducer's setattr on a frozen dataclass.
    """
    from dataclasses import fields
    names = {field.name for field in fields(cls)}
    record = cls(**{name: values[name] for name in names},
                 realization_table=attachments.get("realization_table"))
    for name, value in {**values, **attachments}.items():
        if name not in names and name != "realization_table":
            object.__setattr__(record, name, value)
    return record


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


class _StaticKey:
    """Hash host values once, preserving scalar types and floating-point bits."""

    def __init__(self, value):
        import numpy as np
        self.value = value
        if isinstance(value, np.ndarray):
            key = (value.dtype, value.shape, value.tobytes())
        elif isinstance(value, np.generic):
            key = (type(value), value.dtype, value.tobytes())
        elif type(value) in (str, bool, int):
            key = (type(value), value)
        elif type(value) is float:
            key = (float, struct.pack('!d', value))
        elif type(value) is complex:
            key = (complex, struct.pack('!dd', value.real, value.imag))
        else:
            key = (type(value), id(value))
        self.key = key
        self._hash = hash(key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, _StaticKey) and self.key == other.key

    def __reduce__(self):
        # Identity keys must refer to the restored object after copying.
        return type(self), (self.value,)


@dataclass(frozen=True)
class _FieldMetadata:
    definition: object
    dynamic: tuple
    static: tuple
    shapes: tuple
    direct: bool

    def children(self, value):
        leaves = self.definition.flatten_up_to(value)
        children = tuple(leaf for leaf, dynamic in zip(leaves, self.dynamic) if dynamic)
        return children[0] if self.direct else children

    def rebuild(self, children):
        children = iter((children,) if self.direct else children)
        static = iter(self.static)
        return self.definition.unflatten([
            next(children) if dynamic else next(static).value for dynamic in self.dynamic])


def _result_flatten(result):
    import jax
    from jax.tree_util import GetAttrKey, tree_flatten
    metadata = getattr(result, '_forward_aux', None)
    if metadata is None:
        fields = []
        for value in result:
            leaves, definition = tree_flatten(value)
            dynamic = tuple(isinstance(leaf, jax.Array) for leaf in leaves)
            fields.append(_FieldMetadata(
                definition, dynamic,
                tuple(_StaticKey(leaf) for leaf, flag in zip(leaves, dynamic) if not flag),
                tuple(leaf.shape for leaf, flag in zip(leaves, dynamic) if flag),
                isinstance(value, jax.Array)))
        metadata = (_Metadata(result.realized_geometry), tuple(fields))
    return tuple((GetAttrKey(name), field.children(value))
                 for name, field, value in zip(result._fields, metadata[1], result)), metadata


def forward_arrays_match(result):
    """Whether rebuilt children still have the shapes of the original record."""
    import jax
    import numpy as np
    metadata = getattr(result, '_forward_aux', None)
    if metadata is None:
        return True
    for field, value in zip(metadata[1], result):
        children = field.children(value)
        children = (children,) if field.direct else children
        if any(not isinstance(child, (jax.Array, np.ndarray)) or child.shape != shape
               for child, shape in zip(children, field.shapes)):
            return False
    return True


def with_forward_geometry(base):
    """Attach host geometry without changing the legacy positional tuple."""
    import inspect
    from jax.tree_util import register_pytree_with_keys

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
            result.__dict__.pop('_forward_aux', None)
            return result

    def unflatten(metadata, children):
        result = Result(*(field.rebuild(child) for field, child in zip(metadata[1], children)),
                        realized_geometry=metadata[0].record)
        result._forward_aux = metadata
        return result

    Result.__name__, Result.__qualname__, Result.__module__ = base.__name__, base.__qualname__, base.__module__
    Result.__doc__ = base.__doc__
    signature = inspect.signature(base)
    Result.__signature__ = signature.replace(parameters=[*signature.parameters.values(),
        inspect.Parameter("realized_geometry", inspect.Parameter.KEYWORD_ONLY, default=None)])
    register_pytree_with_keys(Result, _result_flatten, unflatten)
    return Result
