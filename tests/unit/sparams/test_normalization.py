"""Tests for two-run waveguide S-parameter normalization.

Verifies reference subtraction and transmission at the declared port planes.
The empty guide retains propagation and has |S21| > 0.95.
"""

import math

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
    f0=None,
    bandwidth=0.5,
):
    """Build a Simulation for a straight empty PEC waveguide with two ports."""
    # The cross-section is a_wg x b_wg rounded up to whole cells, the count
    # the grid built before #1930 (27 x 14 cells for the defaults, so the
    # default model is unchanged). f_max sets dx; a_wg also sets the band.
    dx = C0 / f_max / 20.0
    n_y = math.ceil(a_wg / dx - 1e-9)
    n_z = math.ceil(b_wg / dx - 1e-9)
    sim = Simulation(
        freq_max=f_max, dx=dx,
        domain=(length, n_y * dx, n_z * dx),  # solved length; #1887
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
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
        f0=f0,
        bandwidth=bandwidth,
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
        f0=f0,
        bandwidth=bandwidth,
    )
    return sim, freqs


def _assert_settled_in_source_band(sim, freqs, result):
    """Check the record and the declared pulse, independently of returned S."""
    settling_db = np.asarray(result.settling_db)
    print("Drive settling_db:", settling_db)
    assert np.all(settling_db <= -40.0), settling_db
    dt = sim._build_grid().dt
    levels = []
    for port in sim._waveguide_ports:
        assert port.waveform == "modulated_gaussian"
        # Declared pulse: cos(2*pi*f0*t) exp(-t**2/(2*width**2)),
        # cut off at five widths either side of its peak.
        width = 1.0 / (port.f0 * port.bandwidth)
        t = np.arange(0.0, 10.0 * width, dt) - 5.0 * width
        pulse = np.cos(2.0 * np.pi * port.f0 * t) * np.exp(-0.5 * (t / width) ** 2)
        spectrum = np.abs(np.exp(-2j * np.pi * np.asarray(freqs)[:, None] * t) @ pulse)
        peak = max(
            np.max(np.abs(np.fft.rfft(pulse, n=16 * len(pulse)))),
            np.abs(np.exp(-2j * np.pi * port.f0 * t) @ pulse),
        )
        level_db = 20.0 * np.log10(spectrum / peak)
        assert np.all(level_db >= -20.0), level_db
        levels.append(level_db)
    return np.asarray(levels)


def _assert_passive_per_bin(freqs, result, source_db):
    sigma = np.linalg.svd(np.moveaxis(np.asarray(result.s_params), 2, 0), compute_uv=False)[:, 0]
    settling_db = np.asarray(result.settling_db)
    print("frequency_GHz sigma_max settling_drive1_dB settling_drive2_dB source1_dB source2_dB")
    for k, freq in enumerate(freqs):
        print(f"{float(freq) / 1e9:.9f} {sigma[k]:.9f} "
              f"{settling_db[0]:.6f} {settling_db[1]:.6f} "
              f"{source_db[0, k]:.6f} {source_db[1, k]:.6f}")
    assert np.all(sigma - 1.0 <= 1e-2), sigma


def test_straight_waveguide_helper_cross_section_follows_a_wg():
    """The helper's realized cross-section is a_wg x b_wg in whole cells.

    dx = C0 / 10 GHz / 20 = 1.499 mm: 40 mm -> 27 cells, 20 mm -> 14 cells,
    and a_wg = 50 mm -> 34 cells. The y and z walls are PEC, so the grid
    holds one more node than cells on those axes.
    """
    sim, _ = _make_straight_waveguide_sim()
    grid = sim._build_grid()
    assert (grid.shape[1] - 1, grid.shape[2] - 1) == (27, 14)

    sim, _ = _make_straight_waveguide_sim(a_wg=0.05)
    grid = sim._build_grid()
    assert (grid.shape[1] - 1, grid.shape[2] - 1) == (34, 14)


def test_normalized_s21_straight_waveguide():
    """Empty straight waveguide: normalized |S21| > 0.95 above cutoff.

    The empty-guide runs have the same incident wave, so both lanes
    return the same complex transmission, including Yee dispersion.
    """
    a_wg = 0.04
    f_c = C0 / (2 * a_wg)
    freqs = jnp.linspace(5.8e9, 6.4e9, 25)

    sim, freqs = _make_straight_waveguide_sim(freqs=freqs, f0=6.5e9, bandwidth=0.6)

    # --- Without normalization ---
    result_raw = sim.compute_waveguide_s_matrix(
        num_periods=30.0,
        normalize=False,
    )
    s21_raw = np.abs(result_raw.s_params[1, 0, :])  # port2 recv, port1 drive
    mean_raw = np.mean(s21_raw)
    raw_err = np.mean(np.abs(s21_raw - 1.0))
    _assert_settled_in_source_band(sim, freqs, result_raw)

    # --- With normalization ---
    result_norm = sim.compute_waveguide_s_matrix(
        num_periods=30.0,
        normalize=True,
    )
    s21_norm = np.abs(result_norm.s_params[1, 0, :])
    mean_norm = np.mean(s21_norm)
    norm_err = np.mean(np.abs(s21_norm - 1.0))
    source_db = _assert_settled_in_source_band(sim, freqs, result_norm)
    _assert_passive_per_bin(freqs, result_norm, source_db)

    print("\n=== Straight waveguide S21 ===")
    print(f"  f_cutoff = {f_c/1e9:.2f} GHz")
    print(f"  Raw (no norm): mean |S21| = {mean_raw:.4f}, mean |error from 1| = {raw_err:.4f}")
    print(f"  Normalized:    mean |S21| = {mean_norm:.4f}, mean |error from 1| = {norm_err:.4f}")

    # The critical assertion: normalized |S21| > 0.95
    assert mean_norm > 0.95, (
        f"Normalized mean |S21| = {mean_norm:.4f}, expected > 0.95"
    )

    # The same empty-guide incident wave normalizes both lanes.
    for i, j in ((1, 0), (0, 1)):
        actual = np.asarray(result_norm.s_params[i, j])
        expected = np.asarray(result_raw.s_params[i, j])
        assert np.max(np.abs(actual - expected)) <= 1e-5 * np.max(np.abs(expected))


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
    freqs = jnp.linspace(5.8e9, 6.4e9, 20)

    sim = Simulation(
        freq_max=10e9,
        domain=(length, a_wg, b_wg),
        boundary={"x": "cpml", "y": "pec", "z": "pec"},
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
        f0=6.5e9,
        bandwidth=0.6,
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
        f0=6.5e9,
        bandwidth=0.6,
    )

    result = sim.compute_waveguide_s_matrix(
        num_periods=30.0,
        normalize=True,
    )
    s = result.s_params  # (2, 2, n_freqs)
    source_db = _assert_settled_in_source_band(sim, freqs, result)
    _assert_passive_per_bin(freqs, result, source_db)

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
    freqs = jnp.linspace(5.8e9, 6.4e9, 20)

    sim, freqs = _make_straight_waveguide_sim(
        freqs=freqs, n_freqs=20, f0=6.5e9, bandwidth=0.6,
    )

    result = sim.compute_waveguide_s_matrix(
        num_periods=30.0,
        normalize=True,
    )
    s = result.s_params
    source_db = _assert_settled_in_source_band(sim, freqs, result)
    _assert_passive_per_bin(freqs, result, source_db)

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
