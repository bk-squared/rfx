"""The admission move expected strings were measured on the pre-move tree
(origin/main 1310cae617ef8989b67c027647a31edcd6dd711e).
"""
import pytest

from rfx import Simulation


@pytest.mark.parametrize("name, expected", [("_forward_from_materials",
     "_ExecuteMixin._forward_from_materials() got an unexpected keyword argument "
     "'no_such_keyword'"), ("_refuse_conformal_boundary",
     "Simulation._refuse_conformal_boundary() got an unexpected keyword argument "
     "'no_such_keyword'"), ("_reject_upml_on_nonuniform",
     "Simulation._reject_upml_on_nonuniform() got an unexpected keyword argument "
     "'no_such_keyword'")])
def test_a_bad_keyword_reports_the_pre_move_name(name, expected):
    sim = Simulation(freq_max=1e10, domain=(.004, .004, .004), dx=.001,
                     boundary="pec")
    with pytest.raises(TypeError) as caught:
        getattr(sim, name)(no_such_keyword=1)
    assert str(caught.value) == expected
