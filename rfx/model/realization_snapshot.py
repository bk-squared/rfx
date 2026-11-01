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
    from rfx.core.jax_utils import is_tracer
    if is_tracer(value):
        return None
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return _PackedArray(value.shape, value.dtype.str, zlib.compress(value.tobytes(), 1))
    if isinstance(value, (_PackedArray, _PackedRecord)):
        return value
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
        from rfx.model.conductors import RealizedConductors
        record = replace(_unpack(self.packed), realization_table=self.table)
        object.__setattr__(record, "_report_grid", self.grid)
        # Diagnostic arrays belong to this requested view, never to the cache.
        view = RealizedConductors(
            lane="nonuniform" if hasattr(self.grid, "dx_arr") else "uniform",
            grid=self.grid, materials=record.materials, pec_cells=record.pec_mask,
            pec_edges=record.edge_masks, sheets=record.sheets, wires=record.wires,
            sheet_impedance=(), periodic=record.periodic, geometry_masks={},
            mode="audit", assembly_from_sim=False, realization_table=self.table)
        object.__setattr__(record, "_conductors", view)
        return record


def snapshot(sim, root, record):
    import copy
    from rfx.realized_geometry import _freeze
    grid = copy.copy(root.grid)
    # Grid metrics are small host arrays, never device-owned state.
    values = grid._asdict() if hasattr(grid, "_asdict") else vars(grid)
    updates = {name: _freeze(value) for name, value in values.items() if hasattr(value, "shape")}
    if hasattr(grid, "_replace"):
        grid = grid._replace(**updates)
    else:
        for name, value in updates.items():
            object.__setattr__(grid, name, value)
    packed = getattr(record, "_packed_record", None)
    if packed is None:
        packed = _pack(replace(record, realization_table=None))
    return DistributedSnapshot(grid, packed, record.realization_table)


def retain(record, entities, assembled):
    """Compress each existing collector once while the compact record is formed."""
    compact = replace(record, entities=tuple(entities), realization_table=None)
    object.__setattr__(record, "_packed_record", _pack(compact))
    return record


def finish(record, assembled, packed):
    """Capture final stage operands once, reusing the packed entity collectors."""
    updates = dict(lane=record.lane, dt=record.dt, ports=record.ports,
        sources=record.sources, probes=record.probes, limitations=record.limitations,
        edge_masks=assembled.pec_edges or (), pec_mask=assembled.pec_mask,
        sheets=assembled.sheets, wires=assembled.wires, materials=assembled.materials)
    updates = {name: _pack(value) for name, value in updates.items()}
    object.__setattr__(record, "_packed_record", _PackedRecord(packed.kind,
        tuple((name, updates.get(name, value)) for name, value in packed.values)))


def pack_array(value):
    return _pack(np.asarray(value))
