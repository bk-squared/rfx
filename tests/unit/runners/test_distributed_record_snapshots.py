"""Snapshot timing, slices, full fields and public coordinate metadata."""
import json
from time import perf_counter

import jax
import numpy as np
import pytest

from rfx import SnapshotSpec
from tests.unit.runners.test_distributed_record_planes import model


@pytest.mark.parametrize("n_devices,boundary", [(2, "pec"), (3, "pec"), (2, "cpml"), (3, "cpml")])
@pytest.mark.parametrize("interval,axis", [(1, None), (3, None), (1, 0), (3, 0), (1, 1), (3, 1)])
def test_snapshots(n_devices, boundary, interval, axis, tmp_path):
    sim, grid, seam = model(boundary, n_devices)
    index = None if axis is None else seam if axis == 0 else grid.shape[1] // 2
    spec = SnapshotSpec(interval=interval, components=("ex", "ez", "hy"),
                        slice_axis=axis, slice_index=index)
    start = perf_counter()
    reference = sim.run(n_steps=11, snapshot=spec, skip_preflight=True)
    actual = sim.run(n_steps=11, snapshot=spec, devices=jax.devices("cpu")[:n_devices],
                     skip_preflight=True)
    jax.block_until_ready(actual.snapshots)
    differences = {}
    for comp, value in reference.snapshots.items():
        a, b = np.asarray(actual.snapshots[comp]), np.asarray(value)
        differences[comp] = dict(equal=bool(np.array_equal(a, b)),
                                 max_abs=float(np.max(np.abs(a - b))), peak=float(np.max(np.abs(b))))
        axes, expected = actual.snapshot_axes[comp], reference.snapshot_axes[comp]
        for name in vars(expected):
            if name == "coords":
                for coordinate in expected.coords:
                    np.testing.assert_array_equal(axes.coords[coordinate], expected.coords[coordinate])
            else:
                np.testing.assert_array_equal(getattr(axes, name), getattr(expected, name))
    report = dict(boundary=boundary, devices=n_devices, grid_shape=grid.shape, n_steps=11,
                  interval=interval, axis=axis, seconds=perf_counter() - start, snapshots=differences)
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)
    assert all(v["equal"] for v in differences.values()), ("snapshot array_equal", differences)


@pytest.mark.parametrize("mutation", ["one_step_late", "skip"])
def test_snapshot_mutations(mutation, monkeypatch, tmp_path):
    from rfx.runners._distributed_records import SnapshotRecords
    original = SnapshotRecords.update
    if mutation == "one_step_late":
        monkeypatch.setattr(SnapshotRecords, "update", lambda self, buffer, state, dt, step, **kw:
                            original(self, buffer, state, dt, step - 1, **kw))
    else:
        monkeypatch.setattr(SnapshotRecords, "update", lambda self, buffer, *a, **kw: buffer)
    with pytest.raises(AssertionError, match="snapshot array_equal") as failure:
        test_snapshots(2, "pec", 3, None, tmp_path)
    print(f"mutation={mutation}: RED {failure.value}")
