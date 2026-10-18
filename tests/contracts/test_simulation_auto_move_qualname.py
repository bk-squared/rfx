"""``Simulation.auto`` moved from rfx/api/__init__.py to rfx/auto_config.py as
``simulation_auto`` and returns to the class as ``auto = classmethod(simulation_auto)``.
A module-level ``def`` gets its own bare ``__qualname__`` and ``__name__``; the
module foot restores the pre-move ones (pickle finds a bound classmethod on its
class by ``__name__``). The expected strings were measured on the pre-move tree
(origin/main 153d4316f) and are written here as literals.
"""
import pickle

import pytest

from rfx import Simulation


def test_auto_keeps_its_pre_move_qualname_and_name():
    assert Simulation.auto.__qualname__ == "Simulation.auto"
    assert Simulation.auto.__name__ == "auto"


def test_auto_still_pickles_as_a_bound_classmethod():
    assert pickle.loads(pickle.dumps(Simulation.auto)) == Simulation.auto


def test_a_missing_argument_reports_the_pre_move_name():
    with pytest.raises(TypeError) as caught:
        Simulation.auto()
    assert str(caught.value) == (
        "Simulation.auto() missing 1 required positional argument: 'freq_range'")
