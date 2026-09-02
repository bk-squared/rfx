"""Tests for batch simulation and parameter sweep utilities."""

import os
import tempfile
import numpy as np

from rfx import Simulation, GaussianPulse
from rfx.batch import ParameterSweep, run_batch, SimulationDataset


def test_parameter_sweep_combinations():
    """Sweep should produce correct count and iterate all combos."""
    sweep = ParameterSweep(a=[1, 2, 3], b=[10, 20])
    assert sweep.total == 6
    assert sweep.keys == ["a", "b"]

    combos = list(sweep.combinations())
    assert len(combos) == 6
    assert combos[0] == {"a": 1.0, "b": 10.0}
    assert combos[-1] == {"a": 3.0, "b": 20.0}


def test_run_batch_produces_results():
    """Small batch run should produce correct number of results."""
    sweep = ParameterSweep(eps_r=[1.0, 4.0])

    def factory(eps_r):
        sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.02),
                         boundary="pec", dx=0.002)
        if eps_r > 1.0:
            from rfx import Box
            sim.add_material("mat", eps_r=eps_r)
            sim.add(Box((0.005, 0, 0), (0.015, 0.02, 0.02)), material="mat")
        sim.add_port((0.005, 0.01, 0.01), "ez", waveform=GaussianPulse(f0=3e9))
        return sim

    results = run_batch(factory, sweep, run_kwargs={"n_steps": 30})
    assert len(results) == 2
    assert results[0][0]["eps_r"] == 1.0
    assert results[1][0]["eps_r"] == 4.0
    # Both should have time_series
    assert results[0][1].time_series is not None
    assert results[1][1].time_series is not None


def test_dataset_export_numpy():
    """Dataset should produce correctly shaped X, Y arrays."""
    # Mock results
    results = [
        ({"w": 0.01, "e": 2.0}, type("R", (), {"s_params": np.array([[[0.1+0j, 0.2+0j]]])})()),
        ({"w": 0.02, "e": 4.0}, type("R", (), {"s_params": np.array([[[0.3+0j, 0.4+0j]]])})()),
    ]
    ds = SimulationDataset.from_results(
        results, input_keys=["w", "e"],
        output_fn=lambda r: np.abs(r.s_params[0, 0, :]),
    )
    X, Y = ds.to_numpy()
    assert X.shape == (2, 2)
    assert Y.shape == (2, 2)
    np.testing.assert_allclose(X[0], [0.01, 2.0])
    np.testing.assert_allclose(Y[0], [0.1, 0.2])


def test_dataset_export_hdf5():
    """HDF5 export should produce a readable file."""
    results = [
        ({"x": 1.0}, type("R", (), {"val": np.array([10.0, 20.0])})()),
        ({"x": 2.0}, type("R", (), {"val": np.array([30.0, 40.0])})()),
    ]
    ds = SimulationDataset.from_results(
        results, input_keys=["x"],
        output_fn=lambda r: r.val,
    )
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        ds.to_hdf5(path)
        import h5py
        with h5py.File(path, "r") as hf:
            assert hf["inputs"].shape == (2, 1)
            assert hf["outputs"].shape == (2, 2)
            assert list(hf.attrs["input_keys"]) == ["x"]
    finally:
        os.unlink(path)
