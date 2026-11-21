"""The streamed face-read maxima and the level ring-down forms from them.

``face_read_maxima`` steps the closed-box incident line as its own program and
keeps only the per-step maxima of the operands the box faces read; the 3-D scan
reads ``record_face_values``' record or carries its own line, a different
compiled form. The two forms agree per step within 9 float32 ULP of each
kind's peak (the cross-trace bar), and at the late level the windows read,
where 9 ULP is as large as the 1e-6 tolerance, the window levels of the two
are in the same class and within a factor 2 of each other. Each stream is its
own N-step call, as the check makes it; the record side reads the prefixes of
one 32,000-step record.
"""
from functools import lru_cache

import numpy as np
import pytest

from tests._closed_box_plane_wave import (TOL, declare, level, line, pulse_end, recorded_maxima,
                                          same_level, windows)
from rfx.measurement.record_end import plane_wave_face_level
from rfx.stepping.incident_line import face_read_maxima

LONGEST = 32000
LENGTHS = (2000, 8000, 32000)
LINES = {"plus_x_ez": {}, "minus_x_ez": {"direction": "-x"}}


@lru_cache(maxsize=None)
def record(name):
    i_h, i_e, _ = recorded_maxima(declare(LINES[name]), LONGEST)
    return i_h, i_e


@lru_cache(maxsize=None)
def stream(name, n):
    cfg, state, grid = line(declare(LINES[name]), n)
    i_h, i_e = face_read_maxima(cfg, state, grid.dx, grid.dt)
    return np.asarray(i_h), np.asarray(i_e)


@pytest.mark.parametrize("name", list(LINES))
@pytest.mark.parametrize("n", LENGTHS)
def test_stream_equals_record_per_step_within_nine_ulp(name, n):
    rec_h, rec_e = record(name)
    got_h, got_e = stream(name, n)
    for kind, got, rec in (("E", got_e, rec_e[:n]), ("H", got_h, rec_h[:n])):
        assert got.shape == (n,) and got.dtype == np.float32
        bar = 9 * np.spacing(np.float32(np.max(rec)))
        diff = np.abs(got.astype(np.float64) - rec.astype(np.float64))
        worst = int(np.argmax(diff))
        assert np.all(diff <= bar), (
            f"{name} N={n} I_{kind}: step {worst} differs by {diff[worst]:.3e}, bar {bar:.3e}")


@pytest.mark.parametrize("name", list(LINES))
def test_stream_window_levels_equal_record_levels(name):
    rec_h, rec_e = record(name)
    long_h, long_e = stream(name, LONGEST)
    rec_end = pulse_end(level(rec_h, rec_e))
    stream_end = pulse_end(level(long_h, long_e))
    rec = windows(rec_h, rec_e, LENGTHS, rec_end)
    got = {}
    for n in LENGTHS:
        got.update(windows(*stream(name, n), (n,), stream_end))
    differ = {key: (got[key], rec[key]) for key in rec if not same_level(got[key], rec[key])}
    assert not differ, f"{name}: windows not at the record's level (stream, record): {differ}"


def test_level_is_each_kind_by_its_own_peak():
    i_h, i_e = record("plus_x_ez")
    i_h, i_e = i_h[:2000], i_e[:2000]
    e_term = i_e.astype(np.float64) / np.float64(np.max(i_e))
    h_term = i_h.astype(np.float64) / np.float64(np.max(i_h))
    assert np.any(h_term > e_term) and np.any(e_term > h_term), (
        "the line must have steps where each term sets the level")
    expected = np.maximum(e_term, h_term)
    got = plane_wave_face_level(i_h, i_e)
    assert got.dtype == np.float64
    assert got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("missing", ["H", "E"])
@pytest.mark.parametrize("form", ["no operand", "zero peak"])
def test_level_leaves_out_a_kind_without_operand_or_peak(missing, form):
    i_h, i_e = record("plus_x_ez")
    i_h, i_e = i_h[:2000], i_e[:2000]
    absent = np.zeros(0, np.float32) if form == "no operand" else np.zeros(2000, np.float32)
    kept = i_e if missing == "H" else i_h
    expected = kept.astype(np.float64) / np.float64(np.max(kept))
    got = (plane_wave_face_level(absent, i_e) if missing == "H"
           else plane_wave_face_level(i_h, absent))
    assert got.tobytes() == expected.tobytes()
    assert np.max(got) == 1.0 and np.max(got[1000:]) <= TOL
