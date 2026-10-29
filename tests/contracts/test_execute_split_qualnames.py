"""Block E1 moved two ``_ExecuteMixin`` methods to module-level functions in
``rfx/runners/adi.py`` and ``rfx/runners/uniform_forward.py``. A module-level
``def`` gets a bare ``__qualname__``, and ``rfx/api/__init__.py`` rewrites only
``<mixin>.<name>`` -> ``Simulation.<name>``, so without the restore at each
module's foot a bad keyword would stop naming ``Simulation...()`` in its
TypeError. The expected strings below were measured on the pre-move tree
(origin/main aa2137118) and are written here as literals.

The admission move expected strings were measured on the pre-move tree
(origin/main 1310cae617ef8989b67c027647a31edcd6dd711e).
"""
import pytest

from rfx import Simulation


@pytest.mark.parametrize("name, expected", [
    ("_run_adi_from_materials",
     "Simulation._run_adi_from_materials() got an unexpected keyword argument "
     "'no_such_keyword'"),
    ("_forward_from_materials",
     "_ExecuteMixin._forward_from_materials() got an unexpected keyword argument "
     "'no_such_keyword'"),
    ("_refuse_conformal_boundary",
     "Simulation._refuse_conformal_boundary() got an unexpected keyword argument "
     "'no_such_keyword'"),
    ("_reject_upml_on_nonuniform",
     "Simulation._reject_upml_on_nonuniform() got an unexpected keyword argument "
     "'no_such_keyword'"),
])
def test_a_bad_keyword_reports_the_pre_move_name(name, expected):
    sim = Simulation(freq_max=1e10, domain=(.004, .004, .004), dx=.001,
                     boundary="pec")
    with pytest.raises(TypeError) as caught:
        getattr(sim, name)(no_such_keyword=1)
    assert str(caught.value) == expected
