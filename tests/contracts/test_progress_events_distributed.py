"""Byte identity and ownership across progress chunks on real device meshes."""
import json
import logging
from time import perf_counter

import jax
import numpy as np
import pytest

from rfx import SnapshotSpec
from rfx.boundaries.spec import BoundarySpec
from rfx.progress import events
from tests.contracts._progress_scenes import add_records, arrays, compare, measure, scene, sequence, require_current_moment_determinism

pytestmark = [pytest.mark.distributed, pytest.mark.slow]


def devices(count):
    available = jax.devices()
    if len(available) < count:
        pytest.skip(f"requires {count} devices; set XLA_FLAGS=--xla_force_host_platform_device_count={count}")
    return available[:count]


@pytest.mark.parametrize("entry,graded,count,n_steps", [
    ("run", False, 2, 101), ("run", False, 2, 650),
    ("forward", False, 2, 101), ("forward", False, 2, 650),
    ("run", True, 2, 101), ("run", True, 2, 650),
    ("forward", True, 2, 101), ("forward", True, 2, 650),
    ("run", False, 3, 101),
])
def test_wrapped_records(entry, graded, count, n_steps, monkeypatch, tmp_path, caplog):
    require_current_moment_determinism()
    from rfx.stepping import sharded
    mesh_devices = devices(count)
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    sim = scene(graded=graded)
    add_records(sim)
    options = dict(n_steps=n_steps, devices=mesh_devices, skip_preflight=True)
    if entry == "forward":
        options.update(distributed=True, checkpoint=False)
    else:
        options["compute_s_params"] = False
        options["snapshot"] = SnapshotSpec(interval=7, components=("ex", "ez", "hy"))
    original = sharded.recorded_sharded_call
    calls, records, final = [], [], []
    chunk_layouts = []
    def layout(carry):
        return [(a.sharding, tuple(s.data.nbytes for s in a.addressable_shards))
                for a in jax.tree.leaves(carry)]
    def partitions_x(sharding):
        return any(axis == "x" or isinstance(axis, tuple) and "x" in axis
                   for axis in getattr(sharding, "spec", ()))
    def describe(array, leaf_layout):
        sharding, shard_bytes = leaf_layout
        return dict(shape=array.shape, dtype=str(array.dtype), sharding=str(sharding),
                    shard_bytes=list(shard_bytes))
    def observe(fn, carry, invariants, xs, ranks):
        chunked = isinstance(xs, tuple)
        before = layout(carry)
        cached_before = fn._cache_size()
        start = perf_counter()
        result = original(fn, carry, invariants, xs, ranks)
        jax.block_until_ready(result)
        seconds = perf_counter() - start
        after = layout(result[0])
        row = dict(seconds=seconds, cache_before=cached_before, cache_after=fn._cache_size(),
                   per_device_bytes=[list(b) for _, b in after])
        if chunked:
            if chunk_layouts:
                assert before == chunk_layouts[-1], "carry layout changed between chunks"
                assert after == chunk_layouts[0], "chunk output layout changed"
            else:
                row["input_layout_changes"] = [
                    dict(leaf_index=i, before=describe(a, left), after=describe(b, right))
                    for i, (a, b, left, right) in enumerate(zip(
                        jax.tree.leaves(carry), jax.tree.leaves(result[0]), before, after, strict=True))
                    if left != right]
            for i, (left, right) in enumerate(zip(before, after, strict=True)):
                if partitions_x(left[0]) or partitions_x(right[0]):
                    assert left == right, f"x-partitioned carry leaf {i} changed layout"
            chunk_layouts.append(after)
            indices, waves = xs
            row["indices"] = np.asarray(indices).tolist()
            row["wave_sharding"] = str(waves.sharding)
            assert waves.sharding.spec == jax.sharding.PartitionSpec(None, "x")
        calls.append(row)
        records.append(result[1])
        final[:] = [result[0]]
        return result
    monkeypatch.setattr(sharded, "recorded_sharded_call", observe)
    start = perf_counter()
    reference = getattr(sim, entry)(**options)
    off_final_layout = layout(final[0])
    off = dict(arrays((reference, final[0], records)))
    off_seconds = perf_counter() - start
    assert len(calls) == 1
    for name in ("dft_planes", "flux_monitors", "ntff_data", "current_moment_data"):
        assert getattr(reference, name) is not None, name
    if entry == "run":
        assert reference.snapshots
    calls.clear()
    records.clear()
    rows = []
    caplog.set_level(logging.WARNING, logger="jax._src.compiler")
    start = perf_counter()
    with jax.log_compiles(True), events(rows.append):
        actual = getattr(sim, entry)(**options)
    # Reconstruct the single per-step record tree before hashing every leaf.
    joined = jax.tree.map(lambda *parts: np.concatenate(parts, axis=0), *records)
    on = dict(arrays((actual, final[0], [joined])))
    on_seconds = perf_counter() - start
    lengths = [2] * 49 + [3] if n_steps == 101 else [7] * 92 + [6]
    assert [len(c["indices"]) for c in calls] == lengths
    assert [i for c in calls for i in c["indices"]] == list(range(n_steps))
    report = dict(entry=entry, graded=graded, devices=count, n_steps=n_steps,
                  cells=(sim._build_nonuniform_grid() if graded else sim._build_grid()).shape,
                  off_seconds=off_seconds, on_seconds=on_seconds, chunks=calls,
                  compile_log=[r.getMessage() for r in caplog.records
                               if "XLA compilation of jit(run_fn)" in r.getMessage()],
                  first_chunk_seconds=calls[0]["seconds"], tail_chunk_seconds=calls[-1]["seconds"])
    (tmp_path / "chunks.json").write_text(json.dumps(report, indent=2))
    assert layout(final[0]) == off_final_layout, "final events-on/off carry layouts differ"
    compare(on, off, report, tmp_path)
    sequence(rows, "multi-device")


def test_wrapped_ignored_report_every(monkeypatch):
    from rfx.errors import ConfigurationWarning
    from rfx.stepping import sharded
    selected = jax.devices("cpu")[:2]
    if len(selected) < 2:
        pytest.skip("requires 2 forced CPU devices")
    monkeypatch.delenv("RFX_EVENTS", raising=False)
    lengths = []
    original = sharded.recorded_sharded_call
    def observe(fn, carry, invariants, xs, ranks):
        lengths.append(xs[0].shape[0])
        return original(fn, carry, invariants, xs, ranks)
    monkeypatch.setattr(sharded, "recorded_sharded_call", observe)
    rows = []
    with events(rows.append), pytest.warns(ConfigurationWarning, match="report_every=100 is ignored"):
        scene().run(n_steps=401, devices=selected, report_every=100,
                    compute_s_params=False, skip_preflight=True)
    assert lengths == [5] * 79 + [6]


@pytest.mark.parametrize("n_steps", [101, 650])
def test_legacy_graded(n_steps, monkeypatch, tmp_path):
    sim = scene(graded=True, ports=0, boundary=BoundarySpec(x="pec", y="pmc", z="pec"))
    selected = devices(2)
    measure(lambda: sim.run(n_steps=n_steps, devices=selected, compute_s_params=False, skip_preflight=True),
            path="multi-device", ticks=False, tmp_path=tmp_path, monkeypatch=monkeypatch,
            metadata=dict(n_steps=n_steps, boundary="pec/pmc/pec", cells=sim._build_nonuniform_grid().shape))


@pytest.mark.parametrize("entry", ["run", "calculator"])
def test_two_port_passes(entry, monkeypatch, tmp_path):
    from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan
    sim = scene(ports=2, source=False, drive_all_ports=True)
    selected = devices(2)
    if entry == "run":
        call = lambda: sim.run(n_steps=101, devices=selected, compute_s_params=True,
                               s_param_freqs=np.array([3e9, 7e9]), skip_preflight=True)
        passes, labels = 3, ["main", "p1", "p2"]
    else:
        call = lambda: compute_lumped_wire_s_matrix_via_scan(
            sim, np.array([3e9, 7e9]), n_steps=101, devices=selected, skip_preflight=True)
        passes, labels = 2, ["p1", "p2"]
    rows = measure(call, path="multi-device", passes=passes, tmp_path=tmp_path, monkeypatch=monkeypatch)
    assert rows[0]["passes"] == labels


@pytest.mark.parametrize("options", [{"n_warmup": 2}, {"checkpoint_every": 3}])
def test_graded_controls_keep_single_call(options, monkeypatch, tmp_path):
    sim = scene(graded=True)
    selected = devices(2)
    measure(lambda: sim.forward(n_steps=8, distributed=True, devices=selected,
                                skip_preflight=True, **options),
            path="multi-device", ticks=False, tmp_path=tmp_path, monkeypatch=monkeypatch)
