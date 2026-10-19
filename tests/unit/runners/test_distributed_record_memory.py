"""Measure real record shards against independently counted owner shares."""
import gc
import json
from time import perf_counter

import jax
from tests._multi_device_scene import host_devices
import numpy as np
import pytest

from rfx import SnapshotSpec
from rfx.runners import _distributed_records as records
from tests.unit.runners.test_distributed_record_planes import model, add_planes


@pytest.mark.parametrize("n_devices", [2, 3])
def test_record_storage(n_devices, monkeypatch, tmp_path):
    sim, grid, seam = model("cpml", n_devices)
    add_planes(sim, grid, seam)
    sim.add_flux_monitor(axis="y", coordinate=4e-3, freqs=[3e9, 7e9])
    sim.add_current_moment_monitor((3e-3, 2e-3, 2e-3), (9e-3, 7e-3, 6e-3),
                                   block_size=2e-3, freqs=[3e9, 7e9])
    sim.add_ntff_box((3e-3, 2e-3, 2e-3), (9e-3, 7e-3, 6e-3), freqs=[3e9, 7e9])
    snapshot = SnapshotSpec(interval=3, components=("ex", "ez", "hy"))
    start = perf_counter()
    reference = sim.run(n_steps=11, snapshot=snapshot, skip_preflight=True)
    shares = {kind: [0] * n_devices for kind in ("planes", "flux", "snapshots", "moments", "ntff", "moment_previous_e")}
    for rank in range(n_devices):
        lo, hi = rank * seam, min((rank + 1) * seam, grid.shape[0])
        for probe in reference.dft_planes.values():
            a = np.asarray(probe.accumulator)
            shares["planes"][rank] += (a.nbytes if probe.index // seam == rank else 0) if probe.axis == 0 else a[:, lo:hi].nbytes
        for mon in reference.flux_monitors.values():
            shares["flux"][rank] += sum(np.asarray(a)[:, lo:hi].nbytes for a in mon[:4])
        shares["snapshots"][rank] = sum(np.asarray(a)[:, lo:hi].nbytes for a in reference.snapshots.values())
        shares["moments"][rank] = sum(np.asarray(a).nbytes for a in reference.current_moment_data)
        m = reference.current_moment_monitor
        width = max(0, min(m.i_hi, hi) - max(m.i_lo, lo))
        shares["moment_previous_e"][rank] = (width * (m.j_hi - m.j_lo)
            * (3 * (m.k_hi - m.k_lo) + 2) * np.dtype(reference.state.ex.dtype).itemsize)
        box = reference.ntff_box
        for index, a in enumerate(reference.ntff_data):
            a = np.asarray(a)
            face = index % 6
            if face < 2:
                shares["ntff"][rank] += a.nbytes if (box.i_lo, box.i_hi)[face] // seam == rank else 0
            else:
                left, right = max(box.i_lo, lo), min(box.i_hi, hi)
                shares["ntff"][rank] += a[:, max(0, left - box.i_lo):max(0, right - box.i_lo)].nbytes if right > left else 0
    del reference
    gc.collect()
    rows = {}
    original = records.SlabRecords.initial

    def observe(self):
        buffers = original(self)
        # Measure the actual compact pre-step E snapshot separately from the
        # scan carry. It is transient, but still belongs to the record budget.
        import inspect
        from rfx.runners._rank import mesh_ranks
        state = inspect.currentframe().f_back.f_locals["sharded_state"]
        moment = self.layouts["moments"]
        previous = moment.prepare(state, ranks=mesh_ranks(moment.mesh))
        measured = {**buffers, "moment_previous_e": previous}
        live_ids = {id(array) for array in jax.live_arrays()}
        for kind, buffer in measured.items():
            leaves = jax.tree.leaves(buffer)
            assert all(id(a) in live_ids for a in leaves), ("record missing from live_arrays", kind)
            actual = [sum(a.addressable_shards[rank].data.nbytes for a in leaves)
                      for rank in range(n_devices)]
            rows[kind] = dict(per_device_bytes=actual, owner_bytes=shares[kind],
                              largest_owner_bytes=max(shares[kind]))
            assert actual == [max(shares[kind])] * n_devices, ("record owner byte bound", kind, rows[kind])
        # All global templates and moment weights are host arrays during the scan.
        for layout in self.layouts.values():
            for name in ("templates", "monitor"):
                for value in jax.tree.leaves(getattr(layout, name, ())):
                    assert not isinstance(value, jax.Array), ("whole record template on device", name)
        return buffers

    monkeypatch.setattr(records.SlabRecords, "initial", observe)
    actual = sim.run(n_steps=11, snapshot=snapshot, devices=host_devices(n_devices), skip_preflight=True)
    jax.block_until_ready(actual.current_moment_data)
    report = dict(devices=n_devices, boundary="cpml", grid_shape=grid.shape,
                  n_steps=11, seconds=perf_counter() - start, memory=rows)
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)


@pytest.mark.parametrize("mutation", ["replicate", "whole_template"])
def test_memory_mutations(mutation, monkeypatch, tmp_path):
    if mutation == "replicate":
        original = records.Packed.initial

        def replicate(self):
            from jax.sharding import NamedSharding, PartitionSpec as P
            return jax.device_put(original(self), NamedSharding(self.mesh, P()))
        monkeypatch.setattr(records.Packed, "initial", replicate)
        message = "record owner byte bound"
    else:
        monkeypatch.setattr(records, "_host_arrays", lambda value: value)
        message = "whole record template on device"
    with pytest.raises(AssertionError, match=message) as failure:
        test_record_storage(2, monkeypatch, tmp_path)
    (tmp_path / "mutation.json").write_text(json.dumps(dict(mutation=mutation, assertion=str(failure.value))))
    print(f"mutation={mutation}: RED {failure.value}")
