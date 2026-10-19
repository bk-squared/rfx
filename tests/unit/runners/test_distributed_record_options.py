"""Uniform-only admission and named refusals for unimplemented records."""
import jax
import numpy as np
import pytest

from rfx import SnapshotSpec
from rfx.runners.distributed_v2 import run_distributed
from tests.unit.runners.test_distributed_record_planes import model


def test_global_zero_flux_is_native_only():
    sim, grid, _ = model("pec", 2)
    assert grid.pad_x_lo == 0
    sim.add_flux_monitor(axis="x", coordinate=0., freqs=[3e9], name="zero")
    result = sim.run(n_steps=4, skip_preflight=True)
    assert result.flux_monitors["zero"].index == 0
    print("A3: one-device x-normal flux at global x=0 admitted; 4 steps")
    with pytest.raises(NotImplementedError, match="x-normal flux at global x index 0"):
        sim.run(n_steps=4, devices=jax.devices("cpu")[:2], skip_preflight=True)


@pytest.mark.parametrize("entry,kind", [("run", "plane"), ("forward", "plane"), ("run", "snapshot")])
def test_graded_refusals_unchanged(entry, kind):
    sim, _, _ = model("pec", 2)
    sim._dz_profile = np.full(7, 1e-3)
    kwargs = dict(n_steps=4, devices=jax.devices("cpu")[:2], skip_preflight=True)
    if entry == "forward":
        kwargs["distributed"] = True
    if kind == "plane":
        sim.add_dft_plane_probe(axis="x", coordinate=6e-3, freqs=[3e9])
        expected = ("add_dft_plane_probe() is not supported on the distributed "
                    + ("multi-device run() path" if entry == "run" else "non-uniform forward path"))
    else:
        kwargs["snapshot"] = SnapshotSpec(interval=1)
        expected = "this lane records no field snapshots"
    with pytest.raises(NotImplementedError) as info:
        getattr(sim, entry)(**kwargs)
    assert expected in str(info.value), str(info.value)
    print(str(info.value))


@pytest.mark.parametrize("name", ["record_dft", "dft_time_records", "dft_planes", "flux_monitors", "current_moments"])
def test_direct_record_options_refused_by_name(name):
    sim, _, _ = model("pec", 2)
    with pytest.raises(NotImplementedError, match=name):
        run_distributed(sim, n_steps=4, devices=jax.devices("cpu")[:2], **{name: True})


def test_internal_plane_time_records_refused():
    sim, _, _ = model("pec", 2)
    sim.add_dft_plane_probe(axis="x", coordinate=6e-3, freqs=[3e9])
    sim.add_probe((6e-3, 4e-3, 3e-3), "ez")
    sim._internal_probe_indices = {0}
    with pytest.raises(NotImplementedError, match="dft_time_records.*record_dft"):
        sim.run(n_steps=4, devices=jax.devices("cpu")[:2], skip_preflight=True)


@pytest.mark.source_scan
@pytest.mark.parametrize("mutation", ["record_options", "global_zero"])
def test_admission_mutations(mutation, monkeypatch, tmp_path):
    from rfx.runners import _distributed_records as records
    if mutation == "record_options":
        monkeypatch.setattr(records, "refuse_record_options", lambda *a, **kw: None)
        run = lambda: test_direct_record_options_refused_by_name("record_dft")
    else:
        import inspect
        import textwrap
        source = textwrap.dedent(inspect.getsource(records.SlabRecords.__init__))
        source = source.replace("if any(m.axis == 0 and m.index == 0 for m in flux):", "if False:")
        namespace = dict(vars(records))
        exec(compile(source, "<allow-global-zero-flux>", "exec"), namespace)
        monkeypatch.setattr(records.SlabRecords, "__init__", namespace["__init__"])
        run = test_global_zero_flux_is_native_only
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE") as failure:
        run()
    (tmp_path / "mutation.txt").write_text(f"{mutation}: {failure.value}\n")
    print(f"mutation={mutation}: RED {failure.value}")
