"""Retain distributed diagnostic arrays without pinning dense kernel products."""
from dataclasses import dataclass, fields, is_dataclass, replace
import zlib

import numpy as np


@dataclass(frozen=True)
class _PackedArray:
    dimensions: tuple
    dtype: str
    data: bytes


@dataclass(frozen=True)
class _PackedRecord:
    kind: type
    values: tuple


def _pack(value):
    if isinstance(value, np.ndarray):
        return _PackedArray(value.shape, value.dtype.str, zlib.compress(value.tobytes(), 1))
    if is_dataclass(value):
        return _PackedRecord(type(value), tuple((f.name, _pack(getattr(value, f.name)))
                                                for f in fields(value)))
    if isinstance(value, tuple):
        items = tuple(_pack(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    return value


def _unpack(value):
    if isinstance(value, _PackedArray):
        return np.frombuffer(zlib.decompress(value.data), dtype=value.dtype).reshape(value.dimensions)
    if isinstance(value, _PackedRecord):
        return value.kind(**{name: _unpack(item) for name, item in value.values})
    if isinstance(value, tuple):
        items = tuple(_unpack(item) for item in value)
        return type(value)(*items) if hasattr(value, "_fields") else items
    return value


@dataclass(frozen=True)
class DistributedSnapshot:
    grid: object
    packed: object
    table: object

    def restore(self):
        record = replace(_unpack(self.packed), realization_table=self.table)
        object.__setattr__(record, "_report_grid", self.grid)
        return record


def snapshot(sim, root, record):
    from rfx.realized_geometry import record_from_conductors
    dense = record_from_conductors(sim, root, lane=record.lane, compact=False)
    # Dataclass replacement drops the private strong reference to the root.
    packed = _pack(replace(dense, realization_table=None))
    return DistributedSnapshot(root.grid, packed, record.realization_table)
