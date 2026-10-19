"""Exact plane placement on both sides of seams and on padded slabs."""
import json
from time import perf_counter

import jax
from tests._multi_device_scene import host_devices
import numpy as np
import pytest

from rfx import Simulation


def model(boundary, n_devices):
    sim = Simulation(freq_max=15e9, domain=(12e-3, 9e-3, 7e-3), dx=1e-3,
                     boundary=boundary, cpml_layers=2)
    sim.add_source((5e-3, 5e-3, 4e-3), "ez", amplitude_kind="field")
    sim.add_source((7e-3, 4e-3, 3e-3), "ey", amplitude_kind="field")
    grid = sim._build_grid()
    seam = (grid.shape[0] + n_devices - 1) // n_devices
    return sim, grid, seam


def add_planes(sim, grid, seam):
    for axis, index in ((0, seam - 1), (0, seam), (1, grid.shape[1] // 2),
                        (2, grid.shape[2] // 2)):
        coordinate = (index - (grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo)[axis]) * grid.dx
        for component in ("ez", "hy"):
            sim.add_dft_plane_probe(axis="xyz"[axis], coordinate=coordinate,
                                    component=component, freqs=[3e9, 7e9],
                                    name=f"{axis}_{index}_{component}")


def compare(actual, reference, tmp_path, metadata, seconds):
    differences = {}
    for name, expected in reference.dft_planes.items():
        a, b = np.asarray(actual.dft_planes[name].accumulator), np.asarray(expected.accumulator)
        differences[name] = {"equal": bool(np.array_equal(a, b)),
                             "max_abs": float(np.max(np.abs(a - b))),
                             "peak": float(np.max(np.abs(b)))}
    report = dict(metadata, seconds=seconds, planes=differences)
    (tmp_path / "measurements.json").write_text(json.dumps(report))
    print(report)
    assert all(v["equal"] for v in differences.values()), ("DFT plane array_equal", differences)


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_planes(boundary, n_devices, tmp_path):
    sim, grid, seam = model(boundary, n_devices)
    assert grid.shape[0] % n_devices != 0
    add_planes(sim, grid, seam)
    start = perf_counter()
    reference = sim.run(n_steps=40, skip_preflight=True)
    actual = sim.run(n_steps=40, devices=host_devices(n_devices), skip_preflight=True)
    jax.block_until_ready(actual.dft_planes)
    compare(actual, reference, tmp_path,
            dict(boundary=boundary, devices=n_devices, grid_shape=grid.shape, n_steps=40),
            perf_counter() - start)


@pytest.mark.parametrize("mutation", ["owned_rows", "skip"])
def test_plane_mutations(mutation, monkeypatch, tmp_path):
    from rfx.runners import _distributed_records as records
    if mutation == "owned_rows":
        original = records._cut
        monkeypatch.setattr(records, "_cut", lambda axis, index, region, start, end, nx_per:
                            original(axis, index, region, start + (axis != 0), end, nx_per))
    else:
        monkeypatch.setattr(records.PlaneRecords, "update", lambda self, buffer, *a, **kw: buffer)
    with pytest.raises(AssertionError, match="DFT plane array_equal") as failure:
        test_planes("pec", 3, tmp_path)
    measured = json.loads((tmp_path / "measurements.json").read_text())["planes"]
    assert any(not row["equal"] for name, row in measured.items() if name.startswith("1_")), "y-normal plane must turn red"
    print(f"mutation={mutation}: RED {failure.value}")


def finite_model(boundary, n_devices):
    """Off-centre finite regions and unequal Yee products, following review B."""
    from rfx.geometry import Box
    sim = Simulation(freq_max=14e9, domain=(16e-3, 10e-3, 8e-3), dx=1e-3,
                     boundary=boundary, cpml_layers=3)
    sim.add_material("diel", eps_r=3.3, sigma=0.02)
    sim.add(Box((5.4e-3, 2.1e-3, 2.6e-3), (11.3e-3, 6.6e-3, 4.1e-3)), material="diel")
    for position, component in (((4.3e-3, 6.7e-3, 3.6e-3), "ez"),
                                 ((9.6e-3, 3.4e-3, 3.9e-3), "ex"),
                                 ((12.2e-3, 7.3e-3, 3.1e-3), "ey")):
        sim.add_source(position, component, amplitude_kind="field")
    grid = sim._build_grid()
    return sim, grid, (grid.shape[0] + n_devices - 1) // n_devices


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_plane_explicit_region(boundary, n_devices, tmp_path):
    sim, grid, seam = finite_model(boundary, n_devices)
    region = (2, grid.shape[0] - 3, 1, grid.shape[2] - 2)
    assert region[0] < seam < region[1]
    sim.add_dft_plane_probe(axis="y", coordinate=3.1e-3, component="hz",
                            freqs=[2.5e9, 6.1e9, 11.7e9], region=region, name="finite_y")
    start = perf_counter()
    reference = sim.run(n_steps=47, skip_preflight=True)
    actual = sim.run(n_steps=47, devices=host_devices(n_devices), skip_preflight=True)
    jax.block_until_ready(actual.dft_planes)
    assert np.max(np.abs(reference.dft_planes["finite_y"].accumulator)) > 0
    compare(actual, reference, tmp_path,
            dict(boundary=boundary, devices=n_devices, grid_shape=grid.shape,
                 n_steps=47, region=region, seam=seam), perf_counter() - start)
