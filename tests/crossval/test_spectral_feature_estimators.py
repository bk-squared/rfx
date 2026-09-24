"""Integrity tests for the shared sub-bin spectral-feature estimators (#812 P3).

The estimators are exercised on the Sheen low-pass filter's openEMS reference
record, whose stopband is a double transmission zero near 7 and 8 GHz -- the
structure that motivated the sub-bin work in the first place. Two things must
hold (the MSL notch filter's old gates, which shared these estimators, were
retired with that case on 2026-09-22; the Sheen case's own gates and its two
committed run legs left the same way on 2026-09-23):

 1. The refinement actually leaves the bin centre on a real curve, and the
    zero counter finds both members of the doublet and no ripple minimum.
    Otherwise the sub-bin machinery would be cosmetic.

 2. The half-grid witness is *structurally* unpassable by a bin-quantised
    estimator. That is the whole reason it can be used as an in-run proof of
    sub-bin resolution rather than an assertion of it.

Plus the closed-form checks the retired stopband-width gate rested on.

What left with the Sheen case's run records: the check that ``refined_extremum``
reproduces the Palace referee fixture's ``referee.fdtd_doublet_ghz`` to six
decimals. That comparison read the two committed FDTD legs
(``validation/crossval/_07_sheen_results/{rfx,openems}.json``) and cannot be
made without them.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "validation/crossval/comparators/spectral_features.py"
OPENEMS_RECORD = REPO_ROOT / "tests/crossval/sheen_lpf/reference/openems_sheen.json"
RECORD_RUNGS = ("stage_b_coarse", "stage_b_mid", "stage_b_fine")

_LOWER_WIN = (6.3, 7.5)
_UPPER_WIN = (7.5, 8.6)


@pytest.fixture(scope="module")
def sf():
    spec = importlib.util.spec_from_file_location("_sf", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rung(name):
    """One mesh rung of the openEMS reference record: (GHz, linear |S21|)."""
    d = json.loads(OPENEMS_RECORD.read_text())[name]
    return (np.asarray(d["freqs_ghz"], dtype=float),
            np.asarray(d["s21_mag"], dtype=float))


@pytest.mark.parametrize("rung", RECORD_RUNGS)
def test_refinement_actually_moves_off_the_bin(sf, rung):
    """If the vertex never left the bin centre the re-gate would be cosmetic."""
    f, s21 = _rung(rung)
    r = sf.refined_extremum(f, s21, *_UPPER_WIN)
    assert abs(r["sub_bin_shift"]) > 0.05
    assert r["refined_f"] != r["bin_f"]


@pytest.mark.parametrize("rung", RECORD_RUNGS)
def test_half_grid_witness_is_unpassable_by_a_quantised_estimator(sf, rung):
    """The two interleaved sub-grids are disjoint, so any estimator that
    returns a bin centre scores >= 1 full-grid bin here. This is the property
    that makes ``spread_bins < 1.0`` a proof rather than a claim."""
    f, s21 = _rung(rung)
    w = sf.half_grid_witness(f, s21, *_UPPER_WIN)
    assert w["argmin_spread_bins"] >= 1.0 - 1e-9
    assert w["spread_bins"] < 1.0


@pytest.mark.parametrize("rung", RECORD_RUNGS)
def test_transmission_zero_counter_finds_the_doublet_and_rejects_ripple(sf, rung):
    """Every rung of the record carries exactly two structural zeros in
    5-15 GHz; the shallow ripple minima above 12 GHz must not count."""
    f, s21 = _rung(rung)
    zeros = sf.transmission_zeros(f, s21, 5.0, 15.0,
                                  depth_db_max=-20.0, prominence_db=0.5)
    assert len(zeros) == 2, [z["refined_f"] for z in zeros]
    assert all(z["depth_db"] <= -20.0 for z in zeros)


def test_ideal_shunt_open_stub_bandwidth_closed_form(sf):
    """The retired stopband-width gate rested on |S21| = 2/(2 + j r tan theta):
    the -10 dB fractional bandwidth is (4/pi) atan(r/6). Check the estimator
    recovers it from a synthetic sweep of that exact model."""
    f0, r = 3.6424, 1.0
    f = np.linspace(2.0, 5.5, 4001)
    theta = 0.5 * np.pi * f / f0
    s21 = np.abs(2.0 / (2.0 + 1j * r * np.tan(theta)))
    i = int(np.argmin(s21))
    lo, hi, _ = sf.band_at_level(f, s21, -10.0, i)
    assert (hi - lo) / f0 == pytest.approx(4.0 / np.pi * np.arctan(r / 6.0),
                                           rel=2e-3)


def test_worst_sampled_notch_minimum_on_the_63_mhz_grid(sf):
    """The audit's finding, re-derived: on a 63.6364 MHz grid an ideal r=1
    stub's WORST sampled minimum is ~-31 dB, i.e. >20 dB inside the retired
    -10 dB depth gate, so that gate could not fail while a notch exists."""
    f0, h = 3.6424e9, 63.6364e6
    theta = 0.5 * np.pi * (1.0 + (h / 2.0) / f0)
    worst_db = 20.0 * np.log10(2.0 / np.sqrt(4.0 + np.tan(theta) ** 2))
    assert worst_db == pytest.approx(-31.23, abs=0.05)
    assert worst_db < -10.0 - 20.0


def test_half_grid_witness_is_unpassable_by_a_bare_argmin_on_a_float32_axis(sf):
    """#812 round-2 review (B3): with the spread measured in GLOBAL bins
    (``f[1]-f[0]``) a bare argmin on a 100-point float32 sweep read 0.99999 at
    dozens of bin positions and PASSED ``< 1.0``.  The witness now measures
    in the LOCAL bin between the two argmin bins, so adjacent argmin bins
    score exactly 1.0 -- numerator and denominator are the same float
    subtraction -- at every position.  The MSL notch filter's estimator re-gate
    fixture recorded the same sweep as
    ``case_D_quantised_estimator.float32_axis_sweep``; it left with that case on
    2026-09-22 and the sweep is rebuilt in this test instead."""
    f = np.linspace(0.7e9, 7.0e9, 100).astype(np.float32).astype(np.float64)
    n_global_below, n_local_below, n = 0, 0, 0
    for k in range(1, 99):
        mag = 1.0 - 0.9 * np.exp(-((f - f[k]) / (f[k + 1] - f[k])) ** 2)
        w = sf.half_grid_witness(f, mag)
        if w["argmin_index_gap"] != 1:
            continue
        n += 1
        n_global_below += int(w["argmin_spread"] / w["full_bin_width"] < 1.0)
        n_local_below += int(w["argmin_spread_bins"] < 1.0)
        assert w["argmin_spread_bins"] == 1.0, k
    assert n >= 90
    assert n_global_below > 0, "the defect this guards against must be reproducible"
    assert n_local_below == 0
