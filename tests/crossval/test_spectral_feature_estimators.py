"""Integrity tests for the shared sub-bin spectral-feature estimators (#812 P3).

Two things must hold for the cv06b/cv07 re-gate to mean anything:

 1. Factoring the log-parabolic vertex out of the two Palace referee producers
    changed NOTHING -- ``refined_extremum`` must reproduce the committed
    referee fixture's ``referee.fdtd_doublet_ghz`` for both FDTD solvers
    to the 6 decimals the fixture stores. Otherwise the new gates would be locking a different number
    than the fixture the evidence chain already carries.

 2. The half-grid witness is *structurally* unpassable by a bin-quantised
    estimator. That is the whole reason it can be used as an in-run proof of
    sub-bin resolution rather than an assertion of it.

Plus the closed-form checks the cv06b stopband-width gate rests on.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "validation/crossval/comparators/spectral_features.py"
REFEREE = REPO_ROOT / "tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json"
RFX_LEG = REPO_ROOT / "validation/crossval/_07_sheen_results/rfx.json"
OEMS_LEG = REPO_ROOT / "validation/crossval/_07_sheen_results/openems.json"
NOTCH_DX50 = REPO_ROOT / "tests/fixtures/msl_notch_e4/msl_stub_notch_rfx_dx50.json"

_LOWER_WIN = (6.3, 7.5)
_UPPER_WIN = (7.5, 8.6)


@pytest.fixture(scope="module")
def sf():
    spec = importlib.util.spec_from_file_location("_sf", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _leg(path):
    d = json.loads(path.read_text())
    return (np.asarray(d["freqs_hz"], dtype=float) / 1e9,
            np.asarray(d["s21_mag"], dtype=float))


@pytest.mark.parametrize("tag,path", [("rfx", RFX_LEG), ("openems", OEMS_LEG)])
def test_refined_vertex_reproduces_committed_referee_fixture(sf, tag, path):
    """The factored estimator IS the referee producer's estimator."""
    want = json.loads(REFEREE.read_text())["referee"]["fdtd_doublet_ghz"][tag]
    f, s21 = _leg(path)
    got_lo = sf.refined_extremum(f, s21, *_LOWER_WIN)["refined_f"]
    got_hi = sf.refined_extremum(f, s21, *_UPPER_WIN)["refined_f"]
    assert round(got_lo, 6) == want["lower_ghz"]
    assert round(got_hi, 6) == want["upper_ghz"]


@pytest.mark.parametrize("path", [RFX_LEG, OEMS_LEG])
def test_refinement_actually_moves_off_the_bin(sf, path):
    """If the vertex never left the bin centre the re-gate would be cosmetic."""
    f, s21 = _leg(path)
    r = sf.refined_extremum(f, s21, *_UPPER_WIN)
    assert abs(r["sub_bin_shift"]) > 0.05
    assert r["refined_f"] != r["bin_f"]


@pytest.mark.parametrize("path", [RFX_LEG, OEMS_LEG])
def test_half_grid_witness_is_unpassable_by_a_quantised_estimator(sf, path):
    """The two interleaved sub-grids are disjoint, so any estimator that
    returns a bin centre scores >= 1 full-grid bin here. This is the property
    that makes ``spread_bins < 1.0`` a proof rather than a claim."""
    f, s21 = _leg(path)
    w = sf.half_grid_witness(f, s21, *_UPPER_WIN)
    assert w["argmin_spread_bins"] >= 1.0 - 1e-9
    assert w["spread_bins"] < 1.0


def test_transmission_zero_counter_finds_the_doublet_and_rejects_ripple(sf):
    """Both committed legs carry exactly two structural zeros in 5-15 GHz;
    the shallow ~12.6-12.8 GHz ripple minimum must not count."""
    for path in (RFX_LEG, OEMS_LEG):
        f, s21 = _leg(path)
        zeros = sf.transmission_zeros(f, s21, 5.0, 15.0,
                                      depth_db_max=-20.0, prominence_db=0.5)
        assert len(zeros) == 2, [z["refined_f"] for z in zeros]
        assert all(z["depth_db"] <= -20.0 for z in zeros)


def test_ideal_shunt_open_stub_bandwidth_closed_form(sf):
    """cv06b's stopband-width gate rests on |S21| = 2/(2 + j r tan theta):
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


def test_committed_notch_fixture_matches_the_ideal_stub_bandwidth(sf):
    """Prior provenance for cv06b's T2 window: a real committed rfx run of the
    same open-stub notch (dx=50um sibling board) on the same 63.6364 MHz grid
    lands within 5% of the closed form, i.e. 4x inside the +-20% window."""
    d = json.loads(NOTCH_DX50.read_text())
    f = np.asarray(d["freqs_ghz"], dtype=float)
    s21 = np.asarray(d["s21_mag"], dtype=float)
    r = sf.refined_extremum(f, s21)
    lo, hi, _ = sf.band_at_level(f, s21, -10.0, r["index"])
    ratio = ((hi - lo) / r["refined_f"]) / (4.0 / np.pi * np.arctan(1.0 / 6.0))
    assert 0.80 < ratio < 1.20
    assert ratio == pytest.approx(0.9512, abs=0.005)


def test_worst_sampled_notch_minimum_on_the_cv06b_grid(sf):
    """The audit's finding, re-derived: on a 63.6364 MHz grid an ideal r=1
    stub's WORST sampled minimum is ~-31 dB, i.e. >20 dB inside cv06b's
    -10 dB depth gate, so that gate cannot fail while a notch exists."""
    f0, h = 3.6424e9, 63.6364e6
    theta = 0.5 * np.pi * (1.0 + (h / 2.0) / f0)
    worst_db = 20.0 * np.log10(2.0 / np.sqrt(4.0 + np.tan(theta) ** 2))
    assert worst_db == pytest.approx(-31.23, abs=0.05)
    assert worst_db < -10.0 - 20.0


def test_half_grid_witness_is_unpassable_by_a_bare_argmin_on_a_float32_axis(sf):
    """#812 round-2 review (B3): with the spread measured in GLOBAL bins
    (``f[1]-f[0]``) a bare argmin on cv06b's float32 sweep read 0.99999 at
    dozens of bin positions and PASSED ``< 1.0``.  The witness now measures
    in the LOCAL bin between the two argmin bins, so adjacent argmin bins
    score exactly 1.0 -- numerator and denominator are the same float
    subtraction -- at every position.  The committed cv06b artifact records
    the same sweep (``case_D_quantised_estimator.float32_axis_sweep``)."""
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
