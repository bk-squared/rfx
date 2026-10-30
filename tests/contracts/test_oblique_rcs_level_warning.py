"""The mesh-level advisory is emitted once, before any solver steps (#1991)."""

import warnings

import pytest

from rfx.core.yee import init_materials
from rfx.errors import AccuracyWarning
from rfx.grid import Grid
from rfx.measurement.rcs import compute_rcs


@pytest.mark.parametrize("theta_inc, expected_count", [(20.0, 1), (0.0, 0)])
def test_oblique_rcs_level_warning_before_run(monkeypatch, theta_inc, expected_count):
    grid = Grid(freq_max=15e9, domain=(.054, .054, .054), dx=.003, cpml_layers=6)
    materials = init_materials(grid.shape)

    class BeforeStepping(Exception):
        pass

    def stop_before_stepping(*args, **kwargs):
        raise BeforeStepping

    monkeypatch.setattr("rfx.measurement.rcs.run", stop_before_stepping)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(BeforeStepping):
            compute_rcs(grid, materials, 1, f0=10e9, theta_inc=theta_inc,
                        cpml_layers=6, tfsf_margin=3)
    assert len(caught) == expected_count
    if expected_count:
        assert caught[0].category is AccuracyWarning
        assert "#1991" in str(caught[0].message)
        assert "mesh dependent" in str(caught[0].message)
