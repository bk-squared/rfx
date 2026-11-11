"""Snapshot the arguments at the realized-integral capture boundary."""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np


@contextmanager
def record_cell_table_capture(grid, shapes):
    calls = []
    shape_indices = {id(shape): index for index, shape in enumerate(shapes)}

    def record(captured_grid, captured_shapes, fraction, owner, component, *, face=False):
        assert captured_grid is grid
        calls.append({
            'fraction': np.array(fraction, copy=True),
            'owner': np.array(owner, copy=True),
            'component': np.asarray(component),
            'face': np.asarray(face),
            # Object identities are local to a run; preserve their scene order.
            'shape_indices': np.array([shape_indices[id(s)] for s in captured_shapes],
                                      dtype=np.int64),
        })

    with patch('rfx.model.realization_integrals.capture', record):
        yield calls
