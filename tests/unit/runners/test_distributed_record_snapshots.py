"""Snapshot timing, slices, full fields and public coordinate metadata."""
import json
from time import perf_counter

import jax
from tests._multi_device_scene import host_devices
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
    actual = sim.run(n_steps=11, snapshot=spec, devices=host_devices(n_devices),
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


def _equations(value):
    """Visit shard maps, rank switches and nested jit wrappers in a scan body."""
    if hasattr(value, "jaxpr"):
        yield from _equations(value.jaxpr)
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            yield from _equations(equation.params)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _equations(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _equations(item)


@pytest.mark.parametrize("n_devices,interval", [(2, 1), (3, 3)])
def test_snapshot_scan_does_not_rebuild_history(n_devices, interval, monkeypatch, tmp_path):
    sim, grid, seam = model("pec", n_devices)
    frames = 19
    spec = SnapshotSpec(interval=interval, components=("ex", "ez", "hy"))
    # A local component history is much larger than all field/ghost temporaries.
    history_size = frames * seam * grid.shape[1] * grid.shape[2]
    original = jax.lax.scan
    reports = []

    def inspect_scan(body, initial, xs, *args, **kwargs):
        if isinstance(initial, dict) and "snapshots" in initial.get("records", {}):
            traced = jax.make_jaxpr(body)(initial, jax.tree.map(lambda a: a[0], xs))
            equations = list(_equations(traced))
            rebuilt = [(e.primitive.name, v.aval.shape) for e in equations
                       if e.primitive.name in ("concatenate", "pad") for v in e.outvars
                       if np.prod(getattr(v.aval, "shape", ())) >= history_size]
            updates = [v.aval.shape for e in equations if e.primitive.name == "dynamic_update_slice"
                       for v in e.outvars if np.prod(v.aval.shape) >= history_size]
            # A history passed through the rank conditional is copied by XLA
            # every step (measured: cost grew with frames x steps).
            through_cond = [(v.aval.shape) for e in equations if e.primitive.name == "cond"
                            for v in (*e.invars, *e.outvars)
                            if np.prod(getattr(v.aval, "shape", ())) >= history_size]
            report = dict(devices=n_devices, interval=interval, frames=frames,
                          local_history_elements=history_size, rebuilt=rebuilt, updates=updates,
                          through_cond=through_cond)
            reports.append(report)
            (tmp_path / "structure.json").write_text(json.dumps(report))
            assert not rebuilt, ("snapshot history rebuilt", rebuilt)
            assert not through_cond, ("snapshot history through the rank conditional", report)
            assert len(updates) == len(spec.components), report
        return original(body, initial, xs, *args, **kwargs)

    monkeypatch.setattr(jax.lax, "scan", inspect_scan)
    result = sim.run(n_steps=frames * interval + interval - 1, snapshot=spec,
                     devices=host_devices(n_devices), skip_preflight=True)
    jax.block_until_ready(result.snapshots)
    assert len(reports) == 1, "inspect the actual distributed scan body"
    print(reports[0])


def test_snapshot_packed_history_mutation(monkeypatch, tmp_path):
    from functools import partial
    from jax.sharding import PartitionSpec as P
    from rfx.runners._rank import rank_shard_map
    from rfx.runners._distributed_records import Packed, SnapshotRecords
    original = SnapshotRecords.update

    def repack(self, buffer, state, dt, step, *, ranks):
        updated = original(self, buffer, state, dt, step, ranks=ranks)
        shapes = [a.shape[1:] for a in buffer]
        packed = Packed(self.mesh, [shapes] * self.mesh.size, self.dtype)

        @partial(rank_shard_map, mesh=self.mesh, in_specs=(P("x"),), out_specs=P("x"))
        def rebuild(local, *, rank):
            history = packed.pack([a[0] for a in local])
            return tuple(a[None] for a in packed.unpack(history, 0))
        return rebuild(ranks, updated)

    monkeypatch.setattr(SnapshotRecords, "update", repack)
    with pytest.raises(AssertionError, match="snapshot history rebuilt") as failure:
        test_snapshot_scan_does_not_rebuild_history(2, 1, monkeypatch, tmp_path)
    (tmp_path / "mutation.json").write_text(json.dumps(dict(assertion=str(failure.value))))
    print(f"Packed history mutation: RED {failure.value}")


def test_snapshot_history_through_conditional_mutation(monkeypatch, tmp_path):
    from functools import partial
    from jax.sharding import PartitionSpec as P
    from rfx.runners._rank import rank_shard_map
    from rfx.runners._distributed_records import SnapshotRecords
    original = SnapshotRecords.update

    def through_switch(self, buffer, state, dt, step, *, ranks):
        updated = original(self, buffer, state, dt, step, ranks=ranks)

        @partial(rank_shard_map, mesh=self.mesh, in_specs=(P("x"),), out_specs=P("x"))
        def passthrough(local, *, rank):
            return jax.lax.switch(rank, [lambda h: h] * self.mesh.size, local)
        return passthrough(ranks, updated)

    monkeypatch.setattr(SnapshotRecords, "update", through_switch)
    with pytest.raises(AssertionError, match="through the rank conditional") as failure:
        test_snapshot_scan_does_not_rebuild_history(2, 1, monkeypatch, tmp_path)
    print(f"history-through-conditional mutation: RED {failure.value}")
