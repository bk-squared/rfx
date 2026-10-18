"""``Simulation.auto`` moved from rfx/api/__init__.py to rfx/auto_config.py as
``simulation_auto`` and returns to the class as ``auto = classmethod(simulation_auto)``.
A module-level ``def`` gets a bare ``__qualname__``; the module foot restores
the pre-move one. The expected strings were measured on the pre-move tree
(origin/main 153d4316f) and are written here as literals.
"""
import pytest

from rfx import Simulation


def test_auto_keeps_its_pre_move_qualname():
    assert Simulation.auto.__qualname__ == "Simulation.auto"


def test_a_missing_argument_reports_the_pre_move_name():
    with pytest.raises(TypeError) as caught:
        Simulation.auto()
    assert str(caught.value) == (
        "Simulation.auto() missing 1 required positional argument: 'freq_range'")
