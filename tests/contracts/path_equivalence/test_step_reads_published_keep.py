"""The static conductor stage consumes the factor published before the scan."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation
from rfx.model import occupancy as occupancy_model


@pytest.mark.parametrize("lane", ["uniform", "graded"])
def test_step_reads_published_keep(lane, monkeypatch):
    kwargs = (dict(dy_profile=np.array([.001] + [.0009, .0011] * 7 + [.001]))
              if lane == "graded" else {})
    sim = Simulation(freq_max=15e9, domain=(.022, .016, .012), dx=.001,
                     boundary="pec", **kwargs)
    grid = sim._build_nonuniform_grid() if lane == "graded" else sim._build_grid()
    sim.add_source((.005, .007, .005), "ez", amplitude_kind="field")
    for position in ((.013, .008, .006), (.017, .010, .007)):
        sim.add_probe(position, "ez")
    cells_a = np.zeros(grid.shape, np.float32)
    cells_b = np.zeros(grid.shape, np.float32)
    cells_a[10:12, 7:9, 5:7] = .2
    cells_b[10:12, 7:9, 5:7] = .6

    # Independent float32 four-cell products; non-periodic outside cells are 0.
    def back(cells, axis):
        result = np.roll(cells, 1, axis=axis)
        face = [slice(None)] * 3
        face[axis] = 0
        result[tuple(face)] = 0
        return result

    keep_b = []
    for component in range(3):
        t1, t2 = (axis for axis in range(3) if axis != component)
        b1 = back(cells_b, t1)
        keep_b.append(jnp.asarray(
            ((1 - cells_b) * (1 - b1))
            * ((1 - back(cells_b, t2)) * (1 - back(b1, t2)))))

    def forward(cells):
        return np.asarray(sim.forward(
            n_steps=60, pec_occupancy_override=jnp.asarray(cells),
            checkpoint=False, skip_preflight=True).time_series)

    expected = forward(cells_b)
    calls = []

    def publish_b(cells, **kwargs):
        calls.append(np.asarray(cells))
        return tuple(keep_b)

    # Both setup and drive_table resolve the builder on this module.
    monkeypatch.setattr(occupancy_model, "build_edge_keep", publish_b)
    actual = forward(cells_a)
    assert len(calls) == 1
    assert np.array_equal(calls[0], cells_a)
    assert np.isfinite(expected).all() and np.any(expected != 0)
    assert np.array_equal(actual, expected)
