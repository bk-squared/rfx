"""Regression coverage for issue #681: silent β-scan rail pinning.

The N-probe extractor's β estimate scans ``±_BETA_SCAN_FRAC`` (35%)
around the analytic Hammerstad-Jensen guess and returns the
residual-minimizing node after a 3-point parabolic refine.  When the
true β lies OUTSIDE that window (e.g. the real ε_eff is far from the HJ
estimate because ``eps_r_sub`` was wrong or the substrate was not
detected under the port), the argmin sits at the window edge and the
fp32 refine returns a value pinned at/near the rail — on ``b29f9de``
this was returned SILENTLY: no flag, no warning, ``Z0``/``beta`` were
the scan limit presented as a measurement.

Measured repro on b29f9de (the fixture below): true ε_eff 6.30 against
an HJ guess of 2.66 → raw argmin at edge node 40/40, returned
β = 1.315·β₀ = 0.974·(1.35·β₀ rail), residual 30× the in-window
control, zero diagnostics.

The fix: ``extract_msl_nprobe`` returns a per-bin ``beta_railed`` bool
(raw argmin at an edge node, or refined β within half a grid step of a
window limit — pure edge conditions, no tunable threshold), the drivers
propagate it next to ``reliable``, and one aggregate warning names the
affected ports/bins (mirroring ``_warn_msl_wave_split_unreliable``).

These tests FAIL on b29f9de: ``beta_railed`` is absent from the result
dict (KeyError) and the warning helper does not exist (ImportError).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.api._sparams import _warn_msl_beta_scan_railed
from rfx.api._spec import MSLSMatrixResult, MixedSMatrixResult
from rfx.probes.msl_wave_decomp import (
    _BETA_SCAN_FRAC,
    _BETA_SCAN_NODES,
    extract_msl_nprobe,
)

_C0 = 2.99792458e8


def _issue_681_fixture(eps_true: float, eps_hj: float = 2.66, f: float = 20e9):
    """The #681 repro geometry: 5 probes, 3 cells of 50 µm spacing.

    Returns (v, x, i1, beta0, beta_true).  Span = 600 µm ≈ 0.10 λ_g at
    the true ε_eff — the auto-resolved span the issue reports.
    """
    n_probes = 5
    spacing = 150e-6
    x = jnp.arange(n_probes, dtype=jnp.float32) * spacing
    beta0 = 2.0 * np.pi * f / _C0 * np.sqrt(eps_hj)
    beta_true = 2.0 * np.pi * f / _C0 * np.sqrt(eps_true)
    alpha = 1.0 + 0.0j
    gamma = 0.3 * np.exp(1j * 0.7)
    v = (
        alpha * np.exp(-1j * beta_true * np.asarray(x))
        + gamma * np.exp(1j * beta_true * np.asarray(x))
    )[None, :]
    i1 = jnp.asarray([(alpha - gamma) / 50.0])
    return jnp.asarray(v), x, i1, jnp.asarray([beta0]), beta_true


def test_out_of_window_beta_is_pinned_and_flagged():
    """The #681 silent-wrong class: β_true outside the ±35% scan window.

    True ε_eff 6.30 vs HJ guess 2.66 → β_true = 1.54·β₀, beyond the
    1.35·β₀ rail.  The extractor cannot reach it; the fitted β comes
    back pinned within one grid step of the rail (that part is the
    DOCUMENTED symptom, true before and after the fix) — and the result
    must now SAY so: ``beta_railed`` is True at that bin.

    On b29f9de this test fails at the ``res["beta_railed"]`` access —
    the key does not exist, which is exactly the silence being fixed.
    """
    v, x, i1, beta0, _ = _issue_681_fixture(eps_true=6.30)
    res = extract_msl_nprobe(v, x, i1, beta0, z0_hj=50.0)

    b0 = float(np.real(np.asarray(beta0))[0])
    rail_hi = b0 * (1.0 + _BETA_SCAN_FRAC)
    step = 2.0 * _BETA_SCAN_FRAC * b0 / (_BETA_SCAN_NODES - 1)
    b_fit = float(np.real(np.asarray(res["beta"]))[0])

    # The pinned return (the bug's symptom, kept as documentation): the
    # fitted β sits within ~two grid steps of the scan rail, nowhere
    # near the true β (which is ~11 steps beyond the rail).
    assert rail_hi - b_fit < 2.5 * step, (
        f"fixture no longer pins: β={b_fit:.2f}, rail={rail_hi:.2f}"
    )

    # The fix: the pin is no longer silent.
    railed = np.asarray(res["beta_railed"], dtype=bool)
    assert railed.shape == (1,)
    assert bool(railed[0]), "out-of-window β must set beta_railed"


def test_in_window_beta_is_not_flagged():
    """In-window β (the healthy class every pre-#681 test exercises)."""
    # ε_true = 1.06² · ε_hj → β_true = 1.06·β₀, well inside ±35%.
    v, x, i1, beta0, beta_true = _issue_681_fixture(
        eps_true=2.66 * 1.06**2
    )
    res = extract_msl_nprobe(v, x, i1, beta0, z0_hj=50.0)
    b_fit = float(np.real(np.asarray(res["beta"]))[0])
    assert abs(b_fit - beta_true) / beta_true < 5e-3
    assert not bool(np.asarray(res["beta_railed"])[0])


def test_rail_flag_is_traceable_and_batch_shaped():
    """The flag is JAX-traceable (jit) and per-bin over a frequency batch."""
    v_out, x, i1, beta0, _ = _issue_681_fixture(eps_true=6.30)
    v_in, _, _, _, _ = _issue_681_fixture(eps_true=2.66 * 1.06**2)
    v = jnp.concatenate([v_out, v_in], axis=0)          # (2, N)
    i1_b = jnp.concatenate([i1, i1])
    beta0_b = jnp.concatenate([beta0, beta0])

    fit = jax.jit(
        lambda vv: extract_msl_nprobe(vv, x, i1_b, beta0_b, z0_hj=50.0)
    )
    res = fit(v)
    railed = np.asarray(res["beta_railed"], dtype=bool)
    np.testing.assert_array_equal(railed, [True, False])


def test_warn_helper_aggregates_and_names_ports():
    freqs = np.array([1.0e9, 2.0e9, 3.0e9, 4.0e9])
    railed = np.array([
        [False, True, True, False],
        [False, False, False, False],
    ])
    with pytest.warns(UserWarning) as rec:
        _warn_msl_beta_scan_railed(railed, freqs, ("msl_0", "msl_1"))
    msgs = [str(w.message) for w in rec]
    assert len(msgs) == 1, f"expected ONE aggregate warning, got {msgs}"
    assert "pinned at its own window limit" in msgs[0]
    assert "2 bins in [2.0000, 3.0000] GHz" in msgs[0]
    assert "'msl_0'" in msgs[0] and "'msl_1'" not in msgs[0]


def test_warn_helper_silent_when_clean():
    freqs = np.array([1.0e9, 2.0e9])
    railed = np.zeros((2, 2), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_msl_beta_scan_railed(railed, freqs, ("msl_0", "msl_1"))


def test_result_dataclasses_carry_beta_railed():
    """Driver plumbing contract: both result types accept the mask.

    Fails on b29f9de with ``TypeError: unexpected keyword argument``.
    """
    freqs = np.array([1.0e9, 2.0e9])
    mask = np.array([[False, True]])
    msl = MSLSMatrixResult(
        S=np.zeros((1, 1, 2), dtype=complex),
        freqs=freqs,
        Z0=np.full((1, 2), 50.0 + 0j),
        beta=np.ones(2, dtype=complex),
        port_names=("msl_0",),
        beta_railed=mask,
    )
    np.testing.assert_array_equal(msl.beta_railed, mask)
    mixed = MixedSMatrixResult(
        S=np.zeros((1, 1, 2), dtype=complex),
        freqs=freqs,
        port_names=("msl_0",),
        port_families=("msl",),
        z0_ref=np.array([50.0]),
        beta_railed=mask,
    )
    np.testing.assert_array_equal(mixed.beta_railed, mask)
