"""The split-axis advisory travels with results without changing the fields."""

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Simulation


def _model(*, x_long=False, graded=False):
    profile = np.array([1., .75, 1.25, 1., 1.25, .75, 1.]) * 1e-3
    sim = Simulation(
        freq_max=15e9, dx=1e-3, boundary="pec",
        domain=(15e-3, 7e-3, 7e-3) if x_long else (7e-3, 15e-3, 7e-3),
        **({"dx_profile": profile} if graded else {}),
    )
    sim.add_source((3e-3, 3e-3, 3e-3), "ez", amplitude_kind="field",
                   waveform=lambda t: jnp.ones_like(t))
    return sim


def _records(result):
    return [d for d in result.diagnostics if d.code == "multi_device_split_axis"]


def _assert_advisory(result, lane):
    grid = result.grid
    assert grid.ny > grid.nx
    records = _records(result)
    assert len(records) == 1
    record = records[0]
    assert record.severity == "advisory"
    assert record.subject == "device split axis"
    assert record.path == lane
    assert record.values == {
        "nx": grid.nx, "ny": grid.ny, "nz": grid.nz, "n_devices": 2,
        "longest_axis": "y" if grid.ny >= grid.nz else "z",
        "measured_slowdown_3_devices": "1.51-1.69",
    }
    assert "1.51-1.69x" in record.message
    assert "The run is not changed." in record.message


def test_split_axis_advisory():
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2, "requires the root conftest's two host devices"
    with jax.default_device(devices[0]):
        single = _model().run(n_steps=3, devices=devices[:1])
        multi = _model().run(n_steps=3, devices=devices)
        skipped = _model().run(n_steps=3, devices=devices, skip_preflight=True)
        x_long = _model(x_long=True).run(n_steps=3, devices=devices)
        graded = _model(graded=True).run(n_steps=3, devices=devices)

    _assert_advisory(multi, "run_distributed")
    _assert_advisory(skipped, "run_distributed")
    _assert_advisory(graded, "run_distributed_nu")
    assert not _records(single)
    assert x_long.grid.nx >= max(x_long.grid.ny, x_long.grid.nz)
    assert not _records(x_long)
    assert np.any(np.asarray(single.state.ez) != 0), "nonzero field witness"
    for result in (multi, skipped):
        assert result.grid.shape == single.grid.shape
        for field in ("ex", "ey", "ez", "hx", "hy", "hz"):
            got = np.asarray(getattr(result.state, field))
            expected = np.asarray(getattr(single.state, field))
            assert got.shape == expected.shape and got.dtype == expected.dtype
            assert np.isfinite(got).all()
            assert got.tobytes() == expected.tobytes(), field
