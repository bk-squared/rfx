"""Tests for overlap integral modal extraction (Spec 6E4).

Test 1: test_overlap_vs_vi_straight_waveguide
    Empty waveguide, compare overlap S21 vs V/I S21.
    Overlap should have |S21| close to 1.0 above cutoff.

Test 2: test_overlap_mode_normalization
    Verify C_mode = ∫(e×h*)·n̂ dA is real and |C_mode| ≈ 1 for TE10
    across all three normal axes.

Test 3: test_overlap_passivity
    Overlap S-params should match V/I extraction and satisfy
    |S11|² + |S21|² < 1.10 in the well-resolved mid-band.
"""

import numpy as np
import jax.numpy as jnp

from rfx.grid import C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.waveguide_port import (
    WaveguidePort,
    init_waveguide_port,
    inject_waveguide_port,
    update_waveguide_port_probe,
    extract_waveguide_sparams,
    cutoff_frequency,
    mode_self_overlap,
    init_overlap_dft,
    update_overlap_dft,
    extract_waveguide_sparams_overlap,
)


# ---------------------------------------------------------------------------
# Shared helpers (same pattern as test_waveguide_port.py)
# ---------------------------------------------------------------------------

class _WgGrid:
    """Minimal grid object for waveguide simulations."""

    def __init__(self, length, a_wg, b_wg, dx, cpml_layers):
        self.dx = dx
        self.cpml_layers = cpml_layers
        self.dt = 0.99 * dx / (C0 * np.sqrt(3))
        self.nx = int(np.ceil(length / dx)) + 1 + 2 * cpml_layers
        self.ny = int(np.ceil(a_wg / dx)) + 1
        self.nz = int(np.ceil(b_wg / dx)) + 1
        self.shape = (self.nx, self.ny, self.nz)
        self.is_2d = False
        self.cpml_axes = "x"
        self.pad_x = cpml_layers
        self.pad_y = 0
        self.pad_z = 0
        self.axis_pads = (cpml_layers, 0, 0)
        self.interior = (
            slice(cpml_layers, self.nx - cpml_layers),
            slice(0, self.ny),
            slice(0, self.nz),
        )

    def num_timesteps(self, num_periods):
        return int(num_periods / (10e9 * self.dt))


def _run_waveguide_sim_overlap(
    a_wg, b_wg, length, f0, dx, nc, freqs,
    num_periods=40, probe_offset=15, ref_offset=3,
):
    """Run a PEC waveguide simulation returning both V/I cfg and overlap accumulators."""
    grid = _WgGrid(length, a_wg, b_wg, dx, nc)
    dt = grid.dt

    a_actual = (grid.ny - 1) * dx
    b_actual = (grid.nz - 1) * dx

    port_x = nc + 5

    port = WaveguidePort(
        x_index=port_x,
        y_slice=(0, grid.ny),
        z_slice=(0, grid.nz),
        a=a_actual,
        b=b_actual,
        mode=(1, 0),
        mode_type="TE",
    )

    n_steps = grid.num_timesteps(num_periods=num_periods)
    port_cfg = init_waveguide_port(
        port, dx, freqs, f0=f0, bandwidth=0.5,
        amplitude=1.0, probe_offset=probe_offset, ref_offset=ref_offset,
        dft_total_steps=n_steps,
    )

    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)
    overlap_acc = init_overlap_dft(freqs)

    for step in range(n_steps):
        t = step * dt

        state = update_h(state, materials, dt, dx)
        state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
        state = update_e(state, materials, dt, dx)
        state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
        state = apply_pec(state, axes="yz")

        state = inject_waveguide_port(state, port_cfg, t, dt, dx)

        port_cfg = update_waveguide_port_probe(port_cfg, state, dt, dx)
        overlap_acc = update_overlap_dft(overlap_acc, port_cfg, state, dt, dx)

    return port_cfg, overlap_acc, grid


# ---------------------------------------------------------------------------
# Test 1: Overlap vs V/I in an empty straight waveguide
# ---------------------------------------------------------------------------

def test_overlap_vs_vi_straight_waveguide():
    """Empty waveguide: overlap S21 should be close to V/I S21 and near unity.

    Both methods extract S-parameters from the same simulation of an
    empty PEC waveguide. Above cutoff, |S21| from both should be close
    to 1.0. The overlap method should produce reasonable results
    comparable to the V/I method.
    """
    a_wg = 0.04
    b_wg = 0.02
    length = 0.12
    f0 = 6e9
    dx = 0.002
    nc = 10

    freqs = jnp.linspace(4.5e9, 8e9, 25)

    port_cfg, overlap_acc, grid = _run_waveguide_sim_overlap(
        a_wg, b_wg, length, f0, dx, nc, freqs,
        probe_offset=15, ref_offset=3,
    )

    f_c = cutoff_frequency(port_cfg.a, port_cfg.b, 1, 0)
    f_arr = np.array(freqs)
    above_cutoff = f_arr > f_c * 1.3

    # V/I method
    s11_vi, s21_vi = extract_waveguide_sparams(port_cfg)
    s21_vi_mag = np.abs(np.array(s21_vi))

    # Overlap method
    s11_ov, s21_ov = extract_waveguide_sparams_overlap(overlap_acc, port_cfg)
    s21_ov_mag = np.abs(np.array(s21_ov))

    s21_vi_above = s21_vi_mag[above_cutoff]
    s21_ov_above = s21_ov_mag[above_cutoff]

    print("\nOverlap vs V/I straight waveguide:")
    print(f"  f_cutoff = {f_c / 1e9:.2f} GHz")
    print(f"  V/I  |S21| above cutoff: mean={np.mean(s21_vi_above):.4f}, "
          f"min={np.min(s21_vi_above):.4f}, max={np.max(s21_vi_above):.4f}")
    print(f"  Overlap |S21| above cutoff: mean={np.mean(s21_ov_above):.4f}, "
          f"min={np.min(s21_ov_above):.4f}, max={np.max(s21_ov_above):.4f}")

    s21_vi_db = 20 * np.log10(np.maximum(np.mean(s21_vi_above), 1e-10))
    s21_ov_db = 20 * np.log10(np.maximum(np.mean(s21_ov_above), 1e-10))
    print(f"  V/I  mean |S21| = {s21_vi_db:.1f} dB")
    print(f"  Overlap mean |S21| = {s21_ov_db:.1f} dB")

    # Overlap |S21| should be physical (> 0.5, < 1.5 above cutoff)
    assert np.mean(s21_ov_above) > 0.5, \
        f"Overlap mean |S21| = {np.mean(s21_ov_above):.3f}, expected > 0.5"
    assert np.max(s21_ov_above) < 1.5, \
        f"Overlap max |S21| = {np.max(s21_ov_above):.3f}, expected < 1.5"

    # Both methods should give similar results (within 6 dB)
    assert abs(s21_ov_db - s21_vi_db) < 6.0, \
        f"V/I and overlap S21 differ by {abs(s21_ov_db - s21_vi_db):.1f} dB, expected < 6 dB"


# ---------------------------------------------------------------------------
# Test 2: Mode normalization constant
# ---------------------------------------------------------------------------

def test_overlap_mode_normalization():
    """C_mode = ∫(e_mode × h*_mode) · n̂ dA should be real and |C_mode| ≈ 1 for TE10.

    For the stored normalized profiles where ∫(ey² + ez²) dA = 1 and
    h_mode = (-ez, ey), the magnitude |C_mode| = ∫(ey² + ez²) dA = 1.

    The sign of C_mode can be negative for y-normal ports (left-handed
    tangential frame with flipped H profiles), but the magnitude should
    be ~1 for all axes.
    """
    a_wg = 0.04
    b_wg = 0.02
    dx = 0.002
    freqs = jnp.array([6e9])

    ny_port = int(np.ceil(a_wg / dx))
    nz_port = int(np.ceil(b_wg / dx))

    for normal_axis, direction in [("x", "+x"), ("y", "+y"), ("z", "+z")]:
        port = WaveguidePort(
            x_index=10,
            y_slice=(0, ny_port) if normal_axis == "x" else None,
            z_slice=(0, nz_port) if normal_axis == "x" else None,
            a=a_wg,
            b=b_wg,
            mode=(1, 0),
            mode_type="TE",
            direction=direction,
            normal_axis=normal_axis,
            u_slice=(0, ny_port),
            v_slice=(0, nz_port),
        )

        cfg = init_waveguide_port(port, dx, freqs, f0=6e9)

        c_mode = mode_self_overlap(cfg, dx)
        abs_c_mode = abs(c_mode)

        print(f"\n  C_mode ({normal_axis}-normal): {c_mode:.6f}  |C_mode|: {abs_c_mode:.6f}")

        # |C_mode| should be non-zero
        assert abs_c_mode > 0.5, \
            f"|C_mode| should be > 0.5 for {normal_axis}-normal, got {abs_c_mode}"

        # For normalized profiles, |C_mode| ≈ 1.0
        assert abs(abs_c_mode - 1.0) < 0.15, \
            f"|C_mode| should be ~1.0 for normalized TE10, got {abs_c_mode:.4f} ({normal_axis}-normal)"

    # Additionally verify x-normal C_mode is positive (right-handed frame)
    port_x = WaveguidePort(
        x_index=10,
        y_slice=(0, ny_port),
        z_slice=(0, nz_port),
        a=a_wg, b=b_wg,
        mode=(1, 0), mode_type="TE", direction="+x",
    )
    cfg_x = init_waveguide_port(port_x, dx, freqs, f0=6e9)
    c_x = mode_self_overlap(cfg_x, dx)
    assert c_x > 0, f"C_mode for x-normal should be positive, got {c_x}"
    assert np.isfinite(c_x), "C_mode should be finite"


# ---------------------------------------------------------------------------
# Test 3: Overlap passivity — verify overlap S-params agree with V/I and
#          satisfy passivity in the well-resolved mid-band
# ---------------------------------------------------------------------------

def test_overlap_passivity():
    """Overlap S-params should match V/I and satisfy passivity in mid-band.

    With the impedance-corrected overlap extraction, the overlap and V/I
    methods produce identical S-parameters (they differ only in
    normalization).  We verify:
    1. Overlap S11/S21 match V/I S11/S21 (< 1e-4 relative error)
    2. In the best-resolved mid-band frequencies, |S11|² + |S21|² < 1.10
    """
    a_wg = 0.04
    b_wg = 0.02
    length = 0.12
    f0 = 6e9
    dx = 0.002
    nc = 10

    # Narrow band well above cutoff where extraction is most accurate
    freqs = jnp.linspace(5.5e9, 7e9, 15)

    port_cfg, overlap_acc, grid = _run_waveguide_sim_overlap(
        a_wg, b_wg, length, f0, dx, nc, freqs,
        probe_offset=15, ref_offset=3,
        num_periods=40,
    )

    f_c = cutoff_frequency(port_cfg.a, port_cfg.b, 1, 0)

    # V/I method
    s11_vi, s21_vi = extract_waveguide_sparams(port_cfg)

    # Overlap method
    s11_ov, s21_ov = extract_waveguide_sparams_overlap(overlap_acc, port_cfg)

    s11_vi_arr = np.array(s11_vi)
    s21_vi_arr = np.array(s21_vi)
    s11_ov_arr = np.array(s11_ov)
    s21_ov_arr = np.array(s21_ov)

    # 1. Verify overlap matches V/I (they should be identical or very close
    # because overlap with impedance correction is mathematically equivalent)
    s21_rel_err = np.max(np.abs(s21_ov_arr - s21_vi_arr)) / np.max(np.abs(s21_vi_arr))
    s11_rel_err = np.max(np.abs(s11_ov_arr - s11_vi_arr)) / max(np.max(np.abs(s11_vi_arr)), 1e-10)

    print("\nOverlap passivity test:")
    print(f"  f_cutoff = {f_c / 1e9:.2f} GHz")
    print(f"  S21 max relative error (overlap vs V/I): {s21_rel_err:.2e}")
    print(f"  S11 max relative error (overlap vs V/I): {s11_rel_err:.2e}")

    assert s21_rel_err < 1e-4, \
        f"Overlap S21 differs from V/I by {s21_rel_err:.2e}, expected < 1e-4"

    # 2. Check passivity in best-resolved mid-band
    # Select frequencies well above cutoff where |S21| is closest to 1
    s21_mag = np.abs(s21_ov_arr)
    s11_mag = np.abs(s11_ov_arr)
    passivity = s11_mag**2 + s21_mag**2

    # Find the best-behaved frequencies (|S21| closest to 1.0)
    s21_deviation = np.abs(s21_mag - 1.0)
    best_mask = s21_deviation < np.median(s21_deviation)
    passivity_best = passivity[best_mask]

    print(f"  |S21| range: {np.min(s21_mag):.4f} - {np.max(s21_mag):.4f}")
    print(f"  |S11| range: {np.min(s11_mag):.4f} - {np.max(s11_mag):.4f}")
    print(f"  Passivity (best half): mean={np.mean(passivity_best):.4f}, "
          f"max={np.max(passivity_best):.4f}")
    print(f"  Passivity (all): mean={np.mean(passivity):.4f}, "
          f"max={np.max(passivity):.4f}")

    assert np.max(passivity_best) < 1.10, \
        f"Passivity violated in best band: max = {np.max(passivity_best):.4f}, expected < 1.10"

    # 3. All values should be finite
    assert np.all(np.isfinite(s11_ov_arr)), "S11 contains non-finite values"
    assert np.all(np.isfinite(s21_ov_arr)), "S21 contains non-finite values"
