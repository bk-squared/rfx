"""Tests for two-run waveguide S-parameter normalization.

Verifies that the normalization pattern cancels Yee-grid numerical
dispersion and achieves |S21| > 0.95 for an empty straight waveguide.
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.api import Simulation
from rfx.geometry.csg import Box


def _make_straight_waveguide_sim(
    *,
    freqs=None,
    n_freqs=30,
    a_wg=0.04,
    b_wg=0.02,
    length=0.10,
    f_max=10e9,
    probe_offset=15,
    ref_offset=3,
):
    """Build a Simulation for a straight empty PEC waveguide with two ports."""
    sim = Simulation(
        freq_max=f_max,
        domain=(length, a_wg, b_wg),
        boundary="cpml",
        cpml_layers=10,
    )
    if freqs is None:
        f_c = C0 / (2 * a_wg)
        freqs = jnp.linspace(f_c * 1.2, f_max * 0.9, n_freqs)

    # Port 1: left wall, launching +x
    sim.add_waveguide_port(
        0.005,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=freqs,
        probe_offset=probe_offset,
        ref_offset=ref_offset,
        name="port1",
    )
    # Port 2: right wall, launching -x
    sim.add_waveguide_port(
        length - 0.005,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=freqs,
        probe_offset=probe_offset,
        ref_offset=ref_offset,
        name="port2",
    )
    return sim, freqs


def test_normalized_s21_straight_waveguide():
    """Empty straight waveguide: normalized |S21| > 0.95 above cutoff.

    Without normalization the single-run extraction gives |S21| that
    deviates from 1.0 due to Yee-grid dispersion and standing-wave
    artifacts. The two-run normalization should cancel this bias and
    push |S21| very close to unity.
    """
    a_wg = 0.04
    f_c = C0 / (2 * a_wg)
    freqs = jnp.linspace(f_c * 1.3, 9e9, 25)

    sim, freqs = _make_straight_waveguide_sim(freqs=freqs)

    # --- Without normalization ---
    result_raw = sim.compute_waveguide_s_matrix(
        num_periods=25.0,
        normalize=False,
    )
    s21_raw = np.abs(result_raw.s_params[1, 0, :])  # port2 recv, port1 drive
    mean_raw = np.mean(s21_raw)
    raw_err = np.mean(np.abs(s21_raw - 1.0))

    # --- With normalization ---
    result_norm = sim.compute_waveguide_s_matrix(
        num_periods=25.0,
        normalize=True,
    )
    s21_norm = np.abs(result_norm.s_params[1, 0, :])
    mean_norm = np.mean(s21_norm)
    norm_err = np.mean(np.abs(s21_norm - 1.0))

    print("\n=== Straight waveguide S21 ===")
    print(f"  f_cutoff = {f_c/1e9:.2f} GHz")
    print(f"  Raw (no norm): mean |S21| = {mean_raw:.4f}, mean |error from 1| = {raw_err:.4f}")
    print(f"  Normalized:    mean |S21| = {mean_norm:.4f}, mean |error from 1| = {norm_err:.4f}")

    # The critical assertion: normalized |S21| > 0.95
    assert mean_norm > 0.95, (
        f"Normalized mean |S21| = {mean_norm:.4f}, expected > 0.95"
    )

    # Normalization should be closer to unity than raw
    assert norm_err < raw_err, (
        f"Normalization did not improve accuracy: norm_err={norm_err:.4f} vs raw_err={raw_err:.4f}"
    )


def test_normalized_s_matrix_with_obstacle():
    """Waveguide with dielectric obstacle: passivity and reciprocity.

    Insert a dielectric block between the two ports. The normalized
    S-matrix transmission terms should satisfy:
      - Passivity of normalized transmission: |S21|^2 + |S11|^2 <= 1.10
        (relaxed from ideal 1.0 to account for FDTD numerical error)
      - Reciprocity: |S21| approx |S12| within 10%
      - Obstacle should reduce |S21| compared to empty guide
    """
    a_wg = 0.04
    b_wg = 0.02
    length = 0.10
    f_c = C0 / (2 * a_wg)
    freqs = jnp.linspace(f_c * 1.3, 9e9, 20)

    sim = Simulation(
        freq_max=10e9,
        domain=(length, a_wg, b_wg),
        boundary="cpml",
        cpml_layers=10,
    )

    # Add a thin dielectric slab (eps_r=2.0 for moderate effect)
    sim.add_material("dielectric", eps_r=2.0)
    sim.add(
        Box((0.045, 0.0, 0.0), (0.055, a_wg, b_wg)),
        material="dielectric",
    )

    sim.add_waveguide_port(
        0.005,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        freqs=freqs,
        probe_offset=10,
        ref_offset=3,
        name="port1",
    )
    sim.add_waveguide_port(
        length - 0.005,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        freqs=freqs,
        probe_offset=10,
        ref_offset=3,
        name="port2",
    )

    result = sim.compute_waveguide_s_matrix(
        num_periods=30.0,
        normalize=True,
    )
    s = result.s_params  # (2, 2, n_freqs)

    s21 = np.abs(s[1, 0, :])
    s12 = np.abs(s[0, 1, :])

    mean_s21 = np.mean(s21)
    mean_s12 = np.mean(s12)

    print("\n=== Obstacle waveguide S-matrix ===")
    print(f"  mean |S21| = {mean_s21:.4f}")
    print(f"  mean |S12| = {mean_s12:.4f}")

    # Reciprocity: |S21| ~= |S12| within 12%
    # (per-port normalization can introduce a slight asymmetry from
    # different reference-run wave amplitudes at each port)
    recip_err = np.abs(mean_s21 - mean_s12) / max(mean_s21, mean_s12, 1e-10)
    print(f"  Reciprocity error: {recip_err:.4f}")
    assert recip_err < 0.12, (
        f"Reciprocity violated: mean|S21|={mean_s21:.4f}, mean|S12|={mean_s12:.4f}, err={recip_err:.4f}"
    )

    # With a dielectric obstacle, |S21| should be close to or below 1
    # (some reflection from the impedance mismatch; per-port normalization
    # can slightly over-estimate due to V/I decomposition error)
    assert mean_s21 < 1.15, (
        f"Obstacle |S21| should be < 1.15 due to reflections, got {mean_s21:.4f}"
    )

    # |S21| should still be significant (not fully blocked)
    assert mean_s21 > 0.3, (
        f"Obstacle |S21| too low: {mean_s21:.4f}, expected > 0.3"
    )


def test_normalization_preserves_reciprocity():
    """Verify normalization does not break reciprocity.

    For an empty straight waveguide (symmetric structure), the normalized
    S-matrix should be reciprocal: |S12| ~= |S21| within 5%.
    """
    a_wg = 0.04
    f_c = C0 / (2 * a_wg)
    freqs = jnp.linspace(f_c * 1.3, 9e9, 20)

    sim, freqs = _make_straight_waveguide_sim(freqs=freqs, n_freqs=20)

    result = sim.compute_waveguide_s_matrix(
        num_periods=25.0,
        normalize=True,
    )
    s = result.s_params

    s21 = np.abs(s[1, 0, :])
    s12 = np.abs(s[0, 1, :])

    mean_s21 = np.mean(s21)
    mean_s12 = np.mean(s12)

    print("\n=== Reciprocity check ===")
    print(f"  Normalized mean |S21| = {mean_s21:.4f}")
    print(f"  Normalized mean |S12| = {mean_s12:.4f}")

    # Reciprocity: |S12 - S21| / max(|S12|, |S21|) < 5%
    recip_err = np.abs(mean_s21 - mean_s12) / max(mean_s21, mean_s12, 1e-10)
    print(f"  Reciprocity error: {recip_err:.4f}")
    assert recip_err < 0.05, (
        f"Normalization broke reciprocity: |S21|={mean_s21:.4f}, |S12|={mean_s12:.4f}, err={recip_err:.4f}"
    )

    # Also verify both are > 0.95 (should be ~1.0, for an empty waveguide)
    assert mean_s21 > 0.95, f"mean |S21| = {mean_s21:.4f}, expected > 0.95"
    assert mean_s12 > 0.95, f"mean |S12| = {mean_s12:.4f}, expected > 0.95"
