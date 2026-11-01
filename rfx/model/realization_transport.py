"""Static JAX metadata for host realization records."""


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
