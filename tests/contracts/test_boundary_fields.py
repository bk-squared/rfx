"""Recorded distributed layouts select exactly the declared grid."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.contracts.boundary_fields import measured_fields


@pytest.mark.parametrize("nx", [5, 6])
def test_distributed_record_layout_selects_full_grid_or_two_slabs(nx):
    grid = SimpleNamespace(shape=(nx, 3, 4))
    full = np.arange(12 * 6 * nx * 3 * 4).reshape(12, 6, nx, 3, 4)
    record = dict(rank=-1, fields=full, psi=[], layout="full-grid")
    fields, _, _ = measured_fields([record], grid, "distributed")
    np.testing.assert_array_equal(fields, full)
    per_slab = (nx + 1) // 2
    # Distinct ghost/alignment sentinels must never appear in selected fields.
    interior = np.pad(full, ((0, 0), (0, 0), (0, 2 * per_slab - nx), (0, 0), (0, 0)),
                      constant_values=-123)
    slabs = [np.pad(interior[:, :, r * per_slab:(r + 1) * per_slab],
                    ((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)), constant_values=-456) for r in range(2)]
    record.update(fields=np.concatenate(slabs, axis=2), layout="two-ghosted-slabs")
    fields, _, _ = measured_fields([record], grid, "distributed")
    np.testing.assert_array_equal(fields, full)


def test_distributed_unknown_or_inconsistent_layout_stops():
    grid = SimpleNamespace(shape=(5, 3, 4))
    record = dict(rank=-1, fields=np.ones((12, 6, 5, 3, 4)), psi=[], layout="unknown")
    with pytest.raises(AssertionError, match="Unrecognized"):
        measured_fields([record], grid, "distributed")
    record["layout"] = "two-ghosted-slabs"
    with pytest.raises(AssertionError, match="slabs do not match"):
        measured_fields([record], grid, "distributed")
    record.update(layout="full-grid", fields=np.ones((12, 6, 6, 3, 4)))
    with pytest.raises(AssertionError, match="Selected fields"):
        measured_fields([record], grid, "distributed")
