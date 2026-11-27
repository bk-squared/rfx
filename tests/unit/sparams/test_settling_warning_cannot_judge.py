"""Only memory-only undetermined witnesses change the settling warning."""

import numpy as np
import pytest

from rfx.errors import AccuracyWarning
from rfx.sparams._settling_warning import _warn_if_ringdown_truncated


STAGE1 = (
    "identification would need 0.02 GiB at stage 1 (pencil; "
    "n_win=16, Nd=16, C=1, L=8, k=8), above the 0.01 GiB bound; "
    "shorten the identification window, identify on fewer channels, "
    "or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES")
STAGE2 = (
    "identification would need 0.03 GiB at stage 2 (residue fit; "
    "n_win=16, Nd=16, C=1, L=8, rank=2), above the 0.01 GiB bound; "
    "shorten the identification window, identify on fewer channels, "
    "or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES")


def _row(reason, **fields):
    return dict(status="undetermined", reason=reason,
                share_per_bin=np.zeros(1), **fields)


def _message(values, rows):
    with pytest.warns(AccuracyWarning) as caught:
        _warn_if_ringdown_truncated(
            values, ("feed", "load"), n_steps=400, witnesses=rows,
            consequence="S-parameter truncation is suspect")
    assert len(caught) == 1
    return str(caught[0].message)


def _assert_main(message):
    assert message.startswith(
        "ring-down settling witness NOT PASSED (pole-tail amplitude threshold -40 dB): ")
    assert "S-parameter truncation is suspect." in message
    assert "cannot judge" not in message


def test_T1_one_stage1_refusal():
    message = _message([np.nan], [_row(STAGE1)])
    assert message == (
        "ring-down settling witness cannot judge (identification refused by the memory bound) "
        "(pole-tail amplitude threshold -40 dB): port feed driven: undetermined "
        "(identification would need 0.02 GiB at stage 1 (pencil; "
        "n_win=16, Nd=16, C=1, L=8, k=8), above the 0.01 GiB bound; "
        "shorten the identification window, identify on fewer channels, "
        "or raise rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES). "
        "The n_steps=400 record has no passing value witness. "
        "Check identification before quoting any S value "
        "(see the result's settling_db and settling_witness fields).")
    assert "NOT PASSED" not in message
    assert "truncation is suspect" not in message
    assert "Increase n_steps" not in message


def test_T2_two_rows_multiple_groups():
    message = _message([np.nan, np.nan], [
        _row("a, b: " + STAGE1 + "; c: " + STAGE2), _row("d: " + STAGE1)])
    assert message.startswith(
        "ring-down settling witness cannot judge (identification refused by the memory bound) "
        "(pole-tail")
    assert "truncation is suspect" not in message


@pytest.mark.parametrize('clause', [
    "the partial SVD reached the full SVD's operation count",
    "a partial-SVD block of b=64 exceeded the bound",
])
def test_partial_fallback_memory_refusal(clause):
    reason = STAGE1.replace('(pencil;', f'(pencil, full-SVD fallback after {clause};')
    message = _message([np.nan], [_row(reason)])
    assert message.startswith(
        "ring-down settling witness cannot judge (identification refused by the memory bound) ")
    assert reason in message
    assert "NOT PASSED" not in message
    assert "truncation is suspect" not in message


def test_T3_real_fail():
    row = _row("pole tail exceeds threshold")
    row['status'] = 'fail'
    message = _message([-20.], [row])
    assert message == (
        "ring-down settling witness NOT PASSED (pole-tail amplitude threshold -40 dB): "
        "port feed driven: -20.0 dB. The n_steps=400 record has no passing value witness. "
        "S-parameter truncation is suspect. "
        "A failing record-end witness means lengthen the record. "
        "Increase n_steps and check identification before quoting any S value "
        "(see the result's settling_db and settling_witness fields).")


def test_T4a_indicator_undetermined():
    message = _message([np.nan], [_row(
        "share 0.534477124", record_end_error_per_bin=[1.31313258],
        kappa_per_bin=[3.88037145])])
    _assert_main(message)
    assert "e_Q 1.3131326, kappa_Q 3.8803715" in message


def test_T4b_source_end_unavailable():
    _assert_main(_message([np.nan], [_row("source end is unavailable")]))


def test_T5_mixed_groups():
    _assert_main(_message([np.nan], [_row(
        "a: " + STAGE1 + "; b: identification windows disagree at 1e+10 Hz")]))


def test_T6_memory_other_memory():
    _assert_main(_message([np.nan], [_row(
        "a: " + STAGE1 + "; b: no kept poles identified; c: " + STAGE2)]))


def test_T7_memory_with_growing_poles():
    message = _message([np.nan], [_row(STAGE1, growing_poles=[dict(
        frequency_hz=1e9, growth_per_step=.01, e_folding_steps=100.,
        max_share=.1, rule="kept non-static growing")])])
    _assert_main(message)
    assert (
        "The recorded fields contain a term growing after the source ended. "
        "A passive structure cannot gain energy after its source ends. "
        "A longer record carries a larger error from this term. "
        "Growing poles (largest growth first): 1 GHz, e-folding 100 steps, max share 0.1. "
        "No passing value witness supports quoting any S value "
        "(see the result's settling_db and settling_witness fields).") in message


def test_T8_memory_beside_other_reason():
    _assert_main(_message([np.nan, np.nan], [
        _row(STAGE1), _row("source end is unavailable")]))


def test_T9_memory_beside_real_fail():
    row = _row("pole tail exceeds threshold")
    row['status'] = 'fail'
    _assert_main(_message([np.nan, -20.], [_row(STAGE1), row]))


def test_T10_invalid_bound():
    _assert_main(_message([np.nan], [_row(
        "a: rfx.ringdown.IDENTIFY_MEMORY_BOUND_BYTES must be a positive finite number or math.inf")]))


def test_absent_memory_row_keeps_main():
    row = _row(STAGE1)
    row['status'] = 'absent'
    _assert_main(_message([np.nan], [row]))


def test_nonhot_row_does_not_change_memory_heading():
    row = _row("source end is unavailable")
    row['status'] = 'pass'
    message = _message([np.nan, -60.], [_row(STAGE1), row])
    assert message.startswith(
        "ring-down settling witness cannot judge (identification refused by the memory bound) "
        "(pole-tail")
    assert "truncation is suspect" not in message
    assert "source end is unavailable" not in message
