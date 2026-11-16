"""Concrete graded z declarations distinguish strict and declared placement."""
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.model.graded_admission import DeclaredZWarning


def test_concrete_z_mismatch_refuses_with_both_lengths():
    with pytest.raises(ValueError, match=r"0\.009.*0\.008.*rel.*snap="):
        Simulation(freq_max=1e9, domain=(.008, .008, .009), dx=.001,
                   boundary="pec", dz_profile=np.full(8, .001))


def test_declared_z_mismatch_warns_once_and_keeps_extent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=1e9, domain=(.008, .008, .009), dx=.001,
                         boundary="pec", dz_profile=np.full(8, .001), snap="declared")
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeclaredZWarning)
    assert "0.009" in str(caught[0].message) and "0.008" in str(caught[0].message)
    assert caught[0].filename == __file__
    assert sim._domain[2] == .009
    grid = sim._build_realized_grid()
    assert np.sum(grid.cells("z")[:-1]) == pytest.approx(.008, rel=1e-9)


def test_half_percent_mismatch_is_silent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=1e9, domain=(.008, .008, .008), dx=.001,
                         boundary="pec", dz_profile=np.full(8, .000995))
    assert not caught
    grid = sim._build_realized_grid()
    assert np.sum(grid.cells("z")[:-1]) == pytest.approx(.00796, rel=1e-9)
    assert sim._domain[2] == .008


def test_the_public_grid_factory_refuses_a_declared_z_its_profile_misses():
    from rfx import make_nonuniform_grid
    with pytest.raises(ValueError, match=r"0\.009.*0\.008"):
        make_nonuniform_grid((.008, .008), np.full(8, .001), .001,
                             domain=(.008, .008, .009))
    grid = make_nonuniform_grid((.008, .008), np.full(8, .001), .001,
                                domain=(.008, .008, .008))
    assert grid.domain[2] == .008
