"""Unit tests for the MSL broad-E5 envelope gate logic.

Synthesises ideal MSL S-matrix data and feeds it to ``_gate_thru`` /
``_gate_open_stub`` to verify the pass/fail criteria fire as expected.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "diagnostics"))
from build_msl_broad_e5_envelope import (  # type: ignore  # noqa: E402
    _gate_open_stub,
    _gate_thru,
)


def _ideal_thru_case_npz(
    n_freqs: int = 21,
    f_lo: float = 2e9,
    f_hi: float = 10e9,
    s11_mag: float = 0.02,
    s21_mag: float = 0.99,
    z0_ohm: complex = 50.0 + 0.0j,
    beta_eff: float = 100.0,
    dx: float = 200e-6,
):
    f = np.linspace(f_lo, f_hi, n_freqs)
    s = np.zeros((2, 2, n_freqs), dtype=complex)
    s[0, 0, :] = s11_mag
    s[1, 0, :] = s21_mag
    s[0, 1, :] = s21_mag
    s[1, 1, :] = s11_mag
    z0 = np.full((2, n_freqs), z0_ohm)
    beta = np.full(n_freqs, beta_eff + 0.0j)
    return dict(
        freqs_hz=f,
        s_matrix=s,
        z0_extracted=z0,
        beta_extracted=beta,
        dx=np.float64(dx),
    )


def _ideal_thru_summary(z0_target: float = 50.0, f_lo: float = 2e9, f_hi: float = 10e9):
    return dict(
        case_id="ideal-thru",
        geometry="thru",
        freq_lo_hz=f_lo,
        freq_hi_hz=f_hi,
        z0_target=z0_target,
    )


def test_thru_ideal_passes():
    npz = _ideal_thru_case_npz()
    res = _gate_thru(npz, _ideal_thru_summary())
    assert res["passed"], res
    assert all(res["gates"].values())


def test_thru_high_s11_fails():
    npz = _ideal_thru_case_npz(s11_mag=0.15)  # > 0.10 threshold
    res = _gate_thru(npz, _ideal_thru_summary())
    assert not res["passed"]
    assert res["gates"]["max_s11_lt_0_10"] is False
    assert res["gates"]["mean_s21_gt_0_95"] is True


def test_thru_low_s21_fails():
    npz = _ideal_thru_case_npz(s21_mag=0.90)  # < 0.95 threshold
    res = _gate_thru(npz, _ideal_thru_summary())
    assert not res["passed"]
    assert res["gates"]["mean_s21_gt_0_95"] is False


def test_thru_z0_far_from_target_fails():
    npz = _ideal_thru_case_npz(z0_ohm=60.0 + 0.0j)  # 20% off 50
    res = _gate_thru(npz, _ideal_thru_summary(z0_target=50.0))
    assert not res["passed"]
    assert res["gates"]["z0_rel_err_lt_5pct"] is False


def test_thru_q_amp_gt_1_fails():
    # q = exp(-j β Δ). For β = β_r + j β_i with β_i > 0:
    # |q| = exp(+β_i Δ). β_i = 5000, Δ = 200e-6 -> |q| = e^1 = 2.72 > 1.
    npz = _ideal_thru_case_npz(beta_eff=100.0 + 5000.0j, dx=200e-6)
    res = _gate_thru(npz, _ideal_thru_summary())
    assert not res["passed"]
    assert res["gates"]["max_q_lt_1"] is False


def _make_stub_case_npz(
    f_lo: float = 5e9,
    f_hi: float = 15e9,
    f_dip: float = 10e9,
    dip_db: float = -25.0,
    bandwidth: float = 0.5e9,
):
    """Synth a stub-resonator S21 with a Gaussian dip at f_dip."""
    n = 41
    f = np.linspace(f_lo, f_hi, n)
    # Background ~ 1, dip = 10**(-25/20) at f_dip with Gaussian width.
    dip_amp = 10 ** (dip_db / 20.0)
    s21 = 1.0 + (dip_amp - 1.0) * np.exp(-((f - f_dip) / bandwidth) ** 2)
    s = np.zeros((2, 2, n), dtype=complex)
    s[1, 0, :] = s21
    s[0, 1, :] = s21
    return dict(
        freqs_hz=f,
        s_matrix=s,
        z0_extracted=np.full((2, n), 50.0 + 0.0j),
        beta_extracted=np.full(n, 100.0 + 0.0j),
        dx=np.float64(200e-6),
    )


def _stub_summary(f_lo: float = 5e9, f_hi: float = 15e9):
    return dict(
        case_id="ideal-stub",
        geometry="open_stub",
        freq_lo_hz=f_lo,
        freq_hi_hz=f_hi,
        z0_target=50.0,
    )


def test_stub_on_target_passes():
    # Dip at band centre (10 GHz), depth -25 dB
    npz = _make_stub_case_npz(f_dip=10e9, dip_db=-30.0)
    res = _gate_open_stub(npz, _stub_summary())
    assert res["passed"], res


def test_stub_off_target_freq_fails():
    # Dip at 12 GHz vs target 10 GHz -> 20% error > 10% threshold
    npz = _make_stub_case_npz(f_dip=12e9, dip_db=-25.0)
    res = _gate_open_stub(npz, _stub_summary())
    assert not res["passed"]
    assert res["gates"]["freq_err_lt_10pct"] is False


def test_stub_over_unity_col_power_fails():
    """HIT-7a: an over-unity stub S-matrix (extraction/normalization artifact) must fail the
    new passivity ceiling even though freq + depth would pass — previously the stub gate
    checked only freq+depth, so a non-physical column power > 1 slipped through unflagged."""
    npz = _make_stub_case_npz(f_dip=10e9, dip_db=-30.0)
    npz["s_matrix"] = npz["s_matrix"] * 1.3          # column power ~1.69 >> 1.05
    res = _gate_open_stub(npz, _stub_summary())
    assert not res["passed"]
    assert res["gates"]["passive_col_power_le_1p05"] is False
    # the passivity gate is what fails — freq + depth are untouched (isolates the new gate)
    assert res["gates"]["freq_err_lt_10pct"] is True
    assert res["gates"]["depth_gt_15db"] is True


def test_stub_shallow_dip_fails():
    # Dip at right freq but only -10 dB below the band mean
    npz = _make_stub_case_npz(f_dip=10e9, dip_db=-5.0, bandwidth=2e9)
    res = _gate_open_stub(npz, _stub_summary())
    assert not res["passed"]
    assert res["gates"]["depth_gt_15db"] is False
