"""Execution-local metadata transport across legacy forward result packers."""
from contextvars import ContextVar
from functools import wraps


def is_replay():
    return _replay.get()


_records = ContextVar("realization_records", default=None)
_replay = ContextVar("realization_replay", default=False)


def publish(record):
    records = _records.get()
    if records is not None:
        records.append(record)


def forward_metadata(entry):
    """Keep nested forwards and concurrent callers' geometry records isolated."""
    @wraps(entry)
    def wrapped(*args, **kwargs):
        records = []
        replay = _replay.set(_records.get() is not None or entry.__name__ != "forward")
        token = _records.set(records)
        try:
            result = entry(*args, **kwargs)
            from rfx.api._spec import ForwardResult
            if isinstance(result, ForwardResult) and result.realized_geometry is None and records:
                result = result._replace(realized_geometry=records[0])
            return result
        finally:
            _records.reset(token)
            _replay.reset(replay)
    return wrapped


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
