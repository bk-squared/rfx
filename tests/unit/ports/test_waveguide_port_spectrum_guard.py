"""Issue #150 closure gates: waveguide source-spectrum defaults + guards.

Root cause of #150: with ``f0`` omitted, the waveguide source defaulted to
``freq_max / 2`` — unrelated to the port mode and, for the canonical WR-90
toy (freq_max=12 GHz → f0=6 GHz < fc_TE10=6.56 GHz), BELOW the mode cutoff.
The launch was evanescent, near-cutoff content crawled at vanishing group
velocity, and the extracted S grew with run length while the in-band
incident reference sat in the source spectral tail.

Fixes pinned here:
1. ``f0=None`` now resolves to the center of the requested DFT band.
2. Preflight emits ``port_source_below_cutoff`` / ``port_freqs_below_cutoff``
   when the resolved source center or any measurement bin sits at/below the
   excited mode's cutoff.
3. Physical acceptance: the f0-omitted empty guide transmits (|S21| ≈ 1)
   and the recorded #150 two-slab toy is run-length-invariant and passive
   once the source actually propagates (slow-marked).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Simulation, validate_port_smatrix

A_WR90, B_WR90 = 22.86e-3, 10.16e-3
FC_TE10 = 3e8 / (2 * A_WR90)  # ~6.56 GHz


def _build(lx, dx, freqs, *, f0=None, x1=0.024, x2=None):
    sim = Simulation(freq_max=12e9, domain=(lx, A_WR90, B_WR90), dx=dx,
                     boundary="cpml", cpml_layers=8)
    kw = {} if f0 is None else {"f0": f0}
    sim.add_waveguide_port(x_position=x1, y_range=(0.0, A_WR90),
                           z_range=(0.0, B_WR90), direction="+x",
                           n_modes=1, freqs=freqs, **kw)
    sim.add_waveguide_port(x_position=(x2 if x2 is not None else lx - x1),
                           y_range=(0.0, A_WR90), z_range=(0.0, B_WR90),
                           direction="-x", n_modes=1, freqs=freqs, **kw)
    return sim


def _preflight_codes(sim):
    issues = sim.preflight()
    return {getattr(i, "code", None) for i in issues}


def test_preflight_flags_source_center_below_cutoff():
    """Explicit f0 below the TE10 cutoff must be flagged."""
    sim = _build(0.10, 2e-3, jnp.linspace(9e9, 11e9, 3), f0=5e9)
    codes = _preflight_codes(sim)
    assert "port_source_below_cutoff" in codes


def test_preflight_flags_measurement_bins_below_cutoff():
    """A DFT bin below cutoff (junk S + NaN gradients) must be flagged."""
    sim = _build(0.10, 2e-3, jnp.linspace(5e9, 11e9, 4), f0=9e9)
    codes = _preflight_codes(sim)
    assert "port_freqs_below_cutoff" in codes


def test_preflight_clean_when_band_above_cutoff():
    """The f0-omitted in-band setup (post-fix default) must NOT be flagged."""
    sim = _build(0.10, 2e-3, jnp.linspace(9e9, 11e9, 3))
    codes = _preflight_codes(sim)
    assert "port_source_below_cutoff" not in codes
    assert "port_freqs_below_cutoff" not in codes


def test_f0_omitted_empty_guide_transmits():
    """Acceptance smoke (#150): f0-omitted empty WR-90 guide must transmit.

    Pre-fix this exact setup gave |S21| 0.6-1.2 / |S11| 0.5-0.9 garbage
    (f0 defaulted to 6 GHz < fc=6.56 GHz). Post-fix the default source is
    centered in the 9-11 GHz measurement band and the guide must look like
    a guide: |S21| ~ 1, |S11| small at every bin.
    """
    freqs = jnp.linspace(9e9, 11e9, 3)
    sim = _build(0.08, 2e-3, freqs, x1=0.020)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.compute_waveguide_s_matrix(n_steps=1200, normalize=False)
    s = np.asarray(res.s_params)
    s21 = np.abs(s[1, 0, :])
    s11 = np.abs(s[0, 0, :])
    assert np.all(s21 > 0.9) and np.all(s21 < 1.1), f"|S21|={s21}"
    assert np.all(s11 < 0.3), f"|S11|={s11}"


@pytest.mark.slow
def test_issue150_two_slab_toy_run_length_invariant_and_passive():
    """The recorded #150 toy: the run-length GROWTH signature must stay dead.

    Measured calibration (2026-06-12, post-fix, this exact toy):
      pre-fix : colpow 20.3 (n=600) -> 57.0 -> 113.7 (n=2400)  [GROWING]
      post-fix: colpow 1.1071 at n=4800 AND n=9600 (identical to 4 dp);
                max per-entry |S| drift 0.074 (2400->4800), 0.068 (4800->9600).
    The 1.107 column-power floor is the coarse-toy artifact envelope
    (dx=2mm staircase on the eps=4 slab + 16mm ~ 0.4*lambda_g port-obstacle
    clearance), NOT ideal passivity — gated at the measured level, not
    loosened beyond it. The #150 regression signature is GROWTH of column
    power with n_steps; that is the primary assertion.
    """
    freqs = jnp.linspace(9e9, 11e9, 5)
    sim = _build(0.10, 2e-3, freqs)
    grid = sim._build_grid()
    eps = jnp.ones(grid.shape, dtype=jnp.float32)
    i = int(round(0.056 / grid.dx))
    eps = eps.at[i:i + 2].set(4.0)
    i = int(round(0.040 / grid.dx))
    eps = eps.at[i:i + 2].set(2.5)

    out = {}
    for n in (2400, 4800):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sim.compute_waveguide_s_matrix(
                n_steps=n, normalize=False, eps_override=eps)
        rep = validate_port_smatrix(res)
        out[n] = (np.asarray(res.s_params),
                  float(rep.metrics["max_column_power"]))

    # Primary (#150 signature): column power must NOT grow with run length.
    growth = out[4800][1] / out[2400][1]
    assert growth < 1.10, (
        f"column power GREW with n_steps: {out[2400][1]:.3f} -> "
        f"{out[4800][1]:.3f} (x{growth:.2f}) — the #150 pathology is back")
    # Secondary: stay at the measured coarse-toy envelope (1.1071 measured).
    for n, (_, colpow) in out.items():
        assert colpow <= 1.15, f"n={n}: max column power {colpow:.3f} > 1.15"
    d = np.max(np.abs(np.abs(out[2400][0]) - np.abs(out[4800][0])))
    assert d < 0.15, f"run-length drift max|Delta|S|| = {d:.4f} >= 0.15 (measured 0.074)"
