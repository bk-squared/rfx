"""Unknown witnesses and real failures have distinct settling warning text."""

import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx.sparams._settling_warning import _warn_if_ringdown_truncated


def _memory_row():
    return dict(
        status="undetermined",
        reason=("identification refused: shorten the identification window, identify on "
                "fewer channels, or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES"),
        share_per_bin=np.zeros(1),
    )


def _message(values, rows):
    with pytest.warns(AccuracyWarning) as caught:
        _warn_if_ringdown_truncated(
            values, ("feed", "load"), n_steps=400, witnesses=rows,
            consequence="S-parameter truncation is suspect")
    assert len(caught) == 1
    return str(caught[0].message)


def test_memory_refused_only():
    message = _message([np.nan], [_memory_row()])
    assert message == (
        "ring-down settling witness cannot judge (identification not run) "
        "(pole-tail amplitude threshold -40 dB): port feed driven: undetermined "
        "(identification refused: shorten the identification window, identify on "
        "fewer channels, or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES). "
        "The n_steps=400 record has no passing value witness. "
        "Check identification before quoting any S value "
        "(see the result's settling_db and settling_witness fields).")
    assert "NOT PASSED" not in message
    assert "truncation is suspect" not in message
    assert "Increase n_steps" not in message


def test_real_fail():
    message = _message([-20.], [dict(status="fail", reason="pole tail exceeds threshold",
                                    share_per_bin=np.array([.1]))])
    assert message == (
        "ring-down settling witness NOT PASSED (pole-tail amplitude threshold -40 dB): "
        "port feed driven: -20.0 dB. The n_steps=400 record has no passing value witness. "
        "S-parameter truncation is suspect. "
        "A failing record-end witness means lengthen the record. "
        "Increase n_steps and check identification before quoting any S value "
        "(see the result's settling_db and settling_witness fields).")


def test_undetermined_for_another_reason():
    message = _message([np.nan], [dict(
        status="undetermined", reason="identification windows disagree at 1e+10 Hz",
        share_per_bin=np.zeros(1))])
    assert message == (
        "ring-down settling witness cannot judge (pole-tail amplitude threshold -40 dB): "
        "port feed driven: undetermined (identification windows disagree at 1e+10 Hz). "
        "The n_steps=400 record has no passing value witness. "
        "Increase n_steps and check identification before quoting any S value "
        "(see the result's settling_db and settling_witness fields).")
    assert "truncation is suspect" not in message


def test_mixed_unknown_and_real_fail():
    message = _message([np.nan, -20.], [
        _memory_row(), dict(status="fail", reason="pole tail exceeds threshold",
                            share_per_bin=np.array([.1]))])
    assert message == (
        "ring-down settling witness NOT PASSED (pole-tail amplitude threshold -40 dB): "
        "port feed driven: undetermined (identification refused: shorten the identification "
        "window, identify on fewer channels, or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES), "
        "port load driven: -20.0 dB. The n_steps=400 record has no passing value witness. "
        "S-parameter truncation is suspect. "
        "A failing record-end witness means lengthen the record. "
        "Check identification before quoting any S value "
        "(see the result's settling_db and settling_witness fields).")
