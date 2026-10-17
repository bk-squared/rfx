"""``Simulation.validate_subgrid`` moved from rfx/api/__init__.py to
rfx/subgridding/validation.py and returns to the class by a class-body import.
A module-level ``def`` gets a bare ``__qualname__``; the module foot restores
the pre-move one. The expected strings were measured on the pre-move tree
(origin/main aa2137118) and are written here as literals.
"""
import pytest

from rfx import Simulation


def test_validate_subgrid_keeps_its_pre_move_qualname():
    assert Simulation.validate_subgrid.__qualname__ == "Simulation.validate_subgrid"


def test_a_bad_keyword_reports_the_pre_move_name():
    sim = Simulation(freq_max=1e10, domain=(.004, .004, .004), dx=.001,
                     boundary="pec")
    with pytest.raises(TypeError) as caught:
        sim.validate_subgrid(no_such_keyword=1)
    assert str(caught.value) == (
        "Simulation.validate_subgrid() got an unexpected keyword argument "
        "'no_such_keyword'")
