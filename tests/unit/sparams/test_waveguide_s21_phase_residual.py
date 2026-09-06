"""Falsifiers for the S21 phase residual vs -beta*L (post-v1.8 plan item 5).

The residual is the waveguide port's discretization witness (#894): unlike a
near-cutoff ``|S11|`` headline it does not pass through the absorber, and the
committed sweep artifact
(``tests/fixtures/waveguide_vi_envelope/s21_phase_residual_witness.json``,
replayed by ``tests/oracle/test_waveguide_vi_envelope_phase_witness.py``)
records it converging at order 2.00 at every band and moving by less than
0.2 % under 3x absorber, 2.5x record and a precision change.

This file does not re-measure that. It checks the ARITHMETIC of the reporter
that now carries the number on every waveguide S-matrix result, with planted
defects whose residual is known in closed form and no FDTD at all -- except
the last test, which runs the cheapest rung of the chain battery once to show
the field and the banner survive a real two-port call.

Nothing here is a gate: the library never compares this number with a
threshold, and neither does this file, beyond the sanity ceiling documented at
the run test.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api._sparams import (
    WAVEGUIDE_PHASE_BETA_CONVENTION,
    WAVEGUIDE_PHASE_MAG_FLOOR,
    s21_phase_residual_deg_rms,
)
from rfx.sources.waveguide_port import _compute_beta

from tests import _waveguide_chain_battery_fixture as F


C0 = 299_792_458.0

# One realistic set of beta ingredients, taken from the chain battery's coarse
# rung rather than invented: WR-90, dx = a/9, the band the battery measures.
FREQS = np.asarray(F.FREQS, dtype=float)
DX_M = F.DX_LADDER[0]
F_C_HZ = (C0 / (2.0 * F.A_M)) * (math.sin(math.pi / 18) / (math.pi / 18))
DT_S = 0.99 * DX_M / (C0 * math.sqrt(3.0))     # a plausible Yee step
L_M = F.REF_RIGHT_DEFAULT_M - F.REF_LEFT_DEFAULT_M


def _beta() -> np.ndarray:
    return np.real(np.asarray(
        _compute_beta(jnp.asarray(FREQS), F_C_HZ, dt=DT_S, dx=DX_M),
        dtype=complex,
    ))


def _two_port(s21: np.ndarray) -> np.ndarray:
    """A (2, 2, n) S-matrix carrying ``s21`` in the [1, 0] column."""
    s = np.zeros((2, 2, s21.size), dtype=complex)
    s[1, 0] = s21
    s[0, 1] = s21
    return s


def _residual(s21: np.ndarray, **kw):
    return s21_phase_residual_deg_rms(
        _two_port(s21), FREQS,
        f_cutoff_hz=F_C_HZ, dt=DT_S, dx=DX_M, length_m=L_M, **kw,
    )


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def test_a_perfect_thru_has_no_residual():
    """S21 = exp(-j*beta*L) built with the SAME beta the reporter uses: the
    residual is zero by construction, so anything above float noise is the
    reporter's own arithmetic being wrong."""
    rms, meta = _residual(np.exp(-1j * _beta() * L_M))
    assert rms == pytest.approx(0.0, abs=1e-9)
    assert meta["n_bins"] == FREQS.size
    assert meta["masked_bins"] == 0
    assert meta["beta_convention"] == WAVEGUIDE_PHASE_BETA_CONVENTION
    assert meta["f_cutoff_hz"] == pytest.approx(F_C_HZ, rel=1e-12)
    assert meta["L_m"] == pytest.approx(L_M, rel=1e-12)


def test_a_planted_half_percent_beta_error_reads_its_predicted_rms():
    """The falsifier with a closed form: a wave carrying 1.005*beta*L leaves
    exactly wrap(-0.005*beta*L) per bin, and the RMS of that is the number the
    reporter must print. This is what says the residual measures the phase
    slope and not some fitted constant."""
    beta = _beta()
    rms, meta = _residual(np.exp(-1j * 1.005 * beta * L_M))
    predicted = math.degrees(
        math.sqrt(float(np.mean(_wrap(-0.005 * beta * L_M) ** 2))))
    assert rms == pytest.approx(predicted, rel=1e-6)
    assert predicted > 1.0, predicted        # the planted defect is visible
    assert meta["masked_bins"] == 0


def test_a_planted_180_degree_flip_on_one_bin_raises_it_by_the_predicted_amount():
    """From zero, flipping one bin's sign plants exactly pi radians there, so
    the RMS over n bins becomes 180/sqrt(n) degrees."""
    s21 = np.exp(-1j * _beta() * L_M)
    s21[3] *= -1.0
    rms, meta = _residual(s21)
    assert rms == pytest.approx(180.0 / math.sqrt(FREQS.size), rel=1e-6)
    assert meta["n_bins"] == FREQS.size


def test_bins_below_the_phase_mask_floor_are_excluded():
    """A bin with no signal carries no phase. The masked bins here are given a
    deliberately huge phase error: if they entered the RMS the result would be
    nowhere near the clean bins' own number."""
    beta = _beta()
    s21 = np.exp(-1j * beta * L_M)
    weak = np.zeros(FREQS.size, dtype=bool)
    weak[[0, 1, -1]] = True
    s21[weak] = 0.5 * WAVEGUIDE_PHASE_MAG_FLOOR * np.exp(1j * 2.0)  # junk phase
    s21[~weak] *= np.exp(-1j * 0.005 * beta[~weak] * L_M)

    rms, meta = _residual(s21)
    predicted = math.degrees(math.sqrt(float(np.mean(
        _wrap(-0.005 * beta[~weak] * L_M) ** 2))))
    assert meta["masked_bins"] == int(weak.sum())
    assert meta["n_bins"] == int((~weak).sum())
    assert rms == pytest.approx(predicted, rel=1e-6)


def test_a_fully_masked_band_reports_none_not_a_vacuous_zero():
    """R5: a mask must not hide the absence of signal."""
    rms, meta = _residual(
        np.full(FREQS.size, 0.1 * WAVEGUIDE_PHASE_MAG_FLOOR, dtype=complex))
    assert rms is None
    assert meta["n_bins"] == 0
    assert "phase mask floor" in meta["reason"]


def test_a_result_that_is_not_a_two_port_reports_none():
    s = np.zeros((3, 3, FREQS.size), dtype=complex)
    rms, meta = s21_phase_residual_deg_rms(
        s, FREQS, f_cutoff_hz=F_C_HZ, dt=DT_S, dx=DX_M, length_m=L_M)
    assert rms is None
    assert "two-port" in meta["reason"]


def test_the_library_mask_floor_is_the_repos_phase_mask_floor():
    """One value, two homes: the library cannot import a diagnostics script,
    so the copy is pinned equal here instead of drifting silently."""
    import sys
    from pathlib import Path

    diagnostics = Path(__file__).resolve().parents[3] / "scripts" / "diagnostics"
    if str(diagnostics) not in sys.path:
        sys.path.insert(0, str(diagnostics))
    from build_waveguide_band_broad_e5_phase_envelope import (  # type: ignore
        PHASE_MAG_FLOOR,
    )
    assert WAVEGUIDE_PHASE_MAG_FLOOR == float(PHASE_MAG_FLOOR)


def test_a_real_two_port_run_carries_the_field_and_prints_the_banner(capsys):
    """One FDTD call, the cheapest rung of the chain battery (dx = a/9, thru).

    The 35-degree assertion is a SANITY CEILING, not a gate: the sweep's
    coarsest rung read 30.4 deg rms at the hardest band it measured
    (f/f_c = 1.010, ``s21_phase_residual_witness.json`` band R0, N = 9), and
    this fixture's band sits at 1.28, much further from cutoff. A number above
    that ceiling would mean the reporter is not measuring what the sweep
    measured. Nothing in the library compares this value with anything.
    """
    sim = F.build_simulation("thru", F.DX_LADDER[0])
    res = sim.compute_waveguide_s_matrix(num_periods=F.NUM_PERIODS)

    rms = res.s21_phase_residual_deg_rms
    meta = res.s21_phase_residual_meta
    assert isinstance(rms, float) and math.isfinite(rms), rms
    assert 0.0 < rms < 35.0, rms
    assert meta["n_bins"] == len(F.FREQS)
    assert meta["masked_bins"] == 0
    assert meta["beta_convention"] == WAVEGUIDE_PHASE_BETA_CONVENTION
    # L is read off the result's own reference planes, not re-derived.
    planes = np.asarray(res.reference_planes, dtype=float)
    assert meta["L_m"] == pytest.approx(abs(planes[1] - planes[0]), rel=1e-12)
    assert meta["L_m"] == pytest.approx(
        F.REF_RIGHT_DEFAULT_M - F.REF_LEFT_DEFAULT_M, rel=1e-9)

    banner = [ln for ln in capsys.readouterr().out.splitlines()
              if "S21 phase residual vs -beta*L" in ln]
    assert len(banner) == 1, banner
    assert "deg rms over" in banner[0]
    assert "not a gate" in banner[0]
    assert f"{rms:.4g}" in banner[0]
