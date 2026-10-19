"""Step 4u's exact NTFF bar, before changing the record adapter."""

import json
from time import perf_counter

import jax
from tests._multi_device_scene import host_devices
import numpy as np
import pytest

from tests.unit.runners.test_distributed_ntff import _model
from rfx.farfield import compute_far_field


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("boundary", ["cpml", "pec"])
def test_ntff_exact_one_device_baseline(n_devices, boundary, monkeypatch, tmp_path):
    """Use actual one-device output; no adapter helper supplies expectations."""
    devices = host_devices(n_devices)
    assert len(devices) == n_devices
    if boundary == "pec":
        # Match the existing NTFF test's accumulator-only PEC witness.
        monkeypatch.setattr("rfx.boundaries.features.refuse_ntff_reflecting_faces",
                            lambda *args, **kwargs: None)
    sim = _model("cpml" if boundary == "cpml" else "x_hi_padded", n_devices)
    started = perf_counter()
    reference = sim.run(n_steps=160, skip_preflight=True)
    actual = sim.run(n_steps=160, devices=devices, skip_preflight=True)
    jax.block_until_ready((reference.ntff_data, actual.ntff_data))
    measurements = {}
    theta, phi = np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37)
    expected_ff = compute_far_field(reference.ntff_data, reference.ntff_box, reference.grid, theta, phi)
    actual_ff = compute_far_field(actual.ntff_data, actual.ntff_box, actual.grid, theta, phi)
    comparisons = list(zip(reference.ntff_data._fields, actual.ntff_data, reference.ntff_data))
    comparisons += [("far_" + name, getattr(actual_ff, name), getattr(expected_ff, name))
                    for name in ("E_theta", "E_phi")]
    for name, value, expected in comparisons:
        value, expected = np.asarray(value), np.asarray(expected)
        delta = float(np.max(np.abs(value.astype(np.complex128)
                                    - expected.astype(np.complex128)), initial=0))
        peak = float(np.max(np.abs(expected), initial=0))
        measurements[name] = {
            "equal": bool(np.array_equal(value, expected)),
            "bytes_equal": value.tobytes() == expected.tobytes(),
            "unequal_cells": int(np.count_nonzero(value != expected)),
            "max_abs": delta,
            "reference_peak": peak,
            "relative": delta / peak if peak else delta,
        }
    report = {"boundary": boundary, "devices": len(devices),
              "n_steps": 160, "grid_shape": reference.grid.shape,
              "seconds": perf_counter() - started, "ntff": measurements}
    (tmp_path / "measurements.json").write_text(json.dumps(report, sort_keys=True))
    np.savez(tmp_path / "distributed-before.npz",
             **{name: np.asarray(value) for name, value in
                zip(actual.ntff_data._fields, actual.ntff_data)})
    print(json.dumps(report, sort_keys=True))
    assert all(row["equal"] for row in measurements.values()), (
        "4u NTFF np.array_equal bar", measurements)


@pytest.mark.parametrize("mutation", ["skip", "compensation"])
def test_ntff_exact_mutations(mutation, monkeypatch, tmp_path):
    from rfx.runners._distributed_ntff import SlabNTFF
    if mutation == "skip":
        monkeypatch.setattr(SlabNTFF, "update", lambda self, buffer, *a, **kw: buffer)
    else:
        original = SlabNTFF.assemble

        def drop(self, buffer):
            data = original(self, buffer)
            return data._replace(**{name: np.zeros_like(getattr(data, name)) for name in data._fields[6:]})
        monkeypatch.setattr(SlabNTFF, "assemble", drop)
    with pytest.raises(AssertionError, match="4u NTFF np.array_equal bar") as failure:
        test_ntff_exact_one_device_baseline(2, "cpml", monkeypatch, tmp_path)
    print(f"mutation={mutation}: RED {failure.value}")
