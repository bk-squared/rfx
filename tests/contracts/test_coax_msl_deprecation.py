"""The transition warns once at the caller before admission or simulation work."""
import warnings

import pytest

from rfx.api import Simulation


MESSAGE = (
    "compute_coax_msl_transition is not supported and will be removed after rfx 2.3: "
    "the coax-to-microstrip transition matrix it returns is not passive on the measured boards."
)


def test_transition_deprecation_precedes_admission_and_warns_once_per_call(monkeypatch):
    """Even refused calls warn once, with no wrapper frame hiding the caller."""
    admissions = []

    def refuse(entry, arguments):
        admissions.append(entry)
        raise ValueError("admission sentinel")

    monkeypatch.setattr("rfx.sparams.coax.admit_coax_entry", refuse)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            with pytest.raises(ValueError, match="admission sentinel"):
                Simulation.compute_coax_msl_transition(object(), junction_x=0.0)
    assert admissions == ["Simulation.compute_coax_msl_transition"] * 2
    assert len(caught) == 2
    assert all(w.category is DeprecationWarning for w in caught)
    assert all(str(w.message) == MESSAGE for w in caught)
    assert all(w.filename == __file__ for w in caught)


def test_transition_deprecation_as_error_prevents_admission(monkeypatch):
    """Warnings-as-errors stops the call before any admission work."""
    def forbidden(*args, **kwargs):
        pytest.fail("admission ran before the deprecation warning")

    monkeypatch.setattr("rfx.sparams.coax.admit_coax_entry", forbidden)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning) as raised:
            Simulation.compute_coax_msl_transition(object(), junction_x=0.0)
    assert str(raised.value) == MESSAGE
