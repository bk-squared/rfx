"""Frozen dense body geometry and its build facts."""

from dataclasses import dataclass, field
import hashlib
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class CellGeometryTable:
    """Unclamped float64 edge lengths and face areas, in x/y/z order.

    Each array dimension is ``grid.shape[axis] + 2``: one low and one high owner
    halo. Formation samples one further trailing node to bound those owners.
    SC first takes ``sc_lattice_slice``, applies its rule, then takes
    ``sc_physical_slice``. The floor uses the whole lattice before taking
    ``floor_physical_slice``. Both consumers cast only after slicing.

    Every owner has count one, including fully metal owners. A count above
    one means "see part layer"; that layer is empty for body-only geometry.
    Part owners have columns (axis, i, j, k). Cut counts include the halos.
    The topology key hashes count arrays in edge/face axis order, followed
    by the empty part owner, index, and fraction arrays. Each array contributes
    its shape as little-endian int64, then its values in C byte order.
    """

    lengths: tuple[np.ndarray, np.ndarray, np.ndarray]
    areas: tuple[np.ndarray, np.ndarray, np.ndarray]
    edge_part_counts: tuple = field(init=False)
    face_part_counts: tuple = field(init=False)
    part_owners: np.ndarray = field(init=False)
    part_indices: np.ndarray = field(init=False)
    part_fractions: np.ndarray = field(init=False)
    cut_edges: tuple[int, int, int] = field(init=False)
    cut_faces: tuple[int, int, int] = field(init=False)
    topology_key: str = field(init=False)

    sc_lattice_slice: ClassVar[tuple] = (slice(None, -1),) * 3
    sc_physical_slice: ClassVar[tuple] = (slice(1, None),) * 3
    floor_physical_slice: ClassVar[tuple] = (slice(1, -1),) * 3

    def __post_init__(self):
        for name, arrays in (('edges', self.lengths), ('faces', self.areas)):
            counts = tuple(np.ones(a.shape, dtype=np.int8) for a in arrays)
            object.__setattr__(self, 'edge_part_counts' if name == 'edges' else 'face_part_counts', counts)
            object.__setattr__(self, f'cut_{name}', tuple(
                int(np.count_nonzero((a > 0) & (a < 1))) for a in arrays))
        object.__setattr__(self, 'part_owners', np.empty((0, 4), dtype=np.int32))
        object.__setattr__(self, 'part_indices', np.empty(0, dtype=np.int8))
        object.__setattr__(self, 'part_fractions', np.empty(0, dtype=np.float64))
        topology = (*self.edge_part_counts, *self.face_part_counts,
                    self.part_owners, self.part_indices, self.part_fractions)
        digest = hashlib.sha256()
        for array in topology:
            digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
            digest.update(array.tobytes())
        object.__setattr__(self, 'topology_key', digest.hexdigest())
        for array in (*self.lengths, *self.areas, *topology):
            array.setflags(write=False)
