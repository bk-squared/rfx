"""Aligned source-off rows (R5).

CPU budget: <60 s per file; three six-sample arrays, no mesh or field steps.
"""
import jax
import numpy as np
import pytest

from rfx.measurement.record_end import injection_end, source_end_step

FIRST = np.array([0., -2., 0., 1., 0., 0.])
LAST = np.array([0., 3.e-12, 0., 0., 0., 0.])


def _assert_concrete_row(row, raw):
    magnitudes = np.array([abs(float(value)) for value in raw])
    peak = max(magnitudes)
    on = [i for i, value in enumerate(magnitudes) if value > 1.e-6 * peak]
    np.testing.assert_array_equal(row[0], magnitudes)
    assert row[0].tobytes() == magnitudes.tobytes()
    assert row[1:] == (peak, on[-1] + 1 if on else 0)


def test_mixed_drives_keep_the_traced_slot():
    """Physical anchor: one unknown drive cannot erase the next drive's off time."""
    def objective(amplitude):
        value, rows = source_end_step(
            [(FIRST, 0.), (amplitude * FIRST, 0.), (LAST, 0.)], 6, .1,
            return_detail=True)
        assert value is None
        assert len(rows) == 3 and rows[1] is None
        _assert_concrete_row(rows[0], FIRST)
        _assert_concrete_row(rows[2], LAST)
        return amplitude * 0.
    jax.grad(objective)(1.)


def test_traced_dt_keeps_every_slot():
    """Physical anchor: off times cannot be judged without a concrete time step."""
    def objective(dt):
        value, rows = source_end_step([(FIRST, 0.), (FIRST, 0.), (LAST, 0.)],
                                      6, dt, return_detail=True)
        assert value is None and rows == [None, None, None]
        return dt * 0.
    jax.grad(objective)(.1)


@pytest.mark.parametrize("dt", [None, 0., -1., np.nan, np.inf])
def test_invalid_dt_keeps_every_slot(dt):
    """Physical anchor: missing or nonphysical time steps provide no off-time witness."""
    assert source_end_step([(FIRST, 0.), (LAST, 0.)], 6, dt,
                           return_detail=True) == (None, [None, None])


@pytest.mark.parametrize("missing", [None, np.ones(2)])
def test_unavailable_samples_keep_their_slot(missing):
    """Physical anchor: a missing drive record is not evidence of free ringing."""
    value, rows = source_end_step([(FIRST, 0.), (missing, 0.), (LAST, 0.)],
                                  6, .1, return_detail=True)
    assert value is None and len(rows) == 3 and rows[1] is None
    _assert_concrete_row(rows[0], FIRST)
    _assert_concrete_row(rows[2], LAST)


def test_concrete_end_and_rows_are_unchanged():
    """Physical anchor: the final active lobe fixes the source end at sample four."""
    value, rows = source_end_step([(FIRST, 0.), (LAST, 0.)], 6, .1, return_detail=True)
    assert value == 4
    _assert_concrete_row(rows[0], FIRST)
    _assert_concrete_row(rows[1], LAST)


def test_injection_end_keeps_the_traced_outcome():
    """Physical anchor: a traced source leaves the settling witness unavailable."""
    def objective(amplitude):
        result = injection_end(6, .1, sources=[(0, 0, 0, 'ez', amplitude * FIRST)],
                               return_detail=True)
        assert result == (None, "cannot judge: traced", [("source[0]", "traced", None)])
        return amplitude * 0.
    jax.grad(objective)(1.)
