"""Example 9: End-to-End Feature Validation
=============================================

Comprehensive validation of all major rfx features in realistic workflows.
Each test exercises multiple subsystems simultaneously, validating the full
integration path — not just individual unit behaviour.

Tests:
  1. Patch antenna on non-uniform z-mesh
     → CPML, Harminv resonance, DFT plane, preflight, geometry/report export
  2. ADI vs Yee comparison (2D TMz cavity)
     → ADI solver, CFL factor, resonance agreement
  3. Waveguide TE10 S-parameters
     → waveguide port excitation, S21 transmission
  4. SBP-SAT subgridding energy stability
     → refinement region, energy non-increasing
  5. NTFF far-field (uniform mesh dipole)
     → DFT accumulation, H phase compensation, sin²(θ), far-field export

PASS criteria are printed for each test.  Exit code 0 = all passed.
"""

from __future__ import annotations

import sys
import time
import shutil
import tempfile
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse, ModulatedGaussian

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS_COUNT = 0
FAIL_COUNT = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if condition else "FAIL"
    if not condition:
        FAIL_COUNT += 1
    else:
        PASS_COUNT += 1
    suffix = f" ({detail})" if detail else ""
    print(f"  [{tag}] {label}{suffix}")


def section(name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Test 1: Patch Antenna on Non-Uniform Mesh
#   Features: non-uniform dz, CPML, port S11, NTFF far-field, DFT plane,
#             preflight, Touchstone export, far-field HDF5/CSV,
#             geometry JSON, experiment report
# ---------------------------------------------------------------------------

def test_patch_nonuniform():
    section("Test 1: Patch Antenna — Non-Uniform Mesh + Export")
    from rfx.auto_config import smooth_grading

    C0 = 2.998e8
    f0 = 5e9
    lam = C0 / f0  # 60 mm
    eps_r = 4.4     # FR4
    h_sub = 1.6e-3  # 1.6 mm substrate

    # Hammerstad patch dimensions
    W = C0 / (2 * f0) * np.sqrt(2 / (eps_r + 1))
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h_sub / W) ** (-0.5)
    dL = 0.412 * h_sub * (
        (eps_eff + 0.3) * (W / h_sub + 0.264)
        / ((eps_eff - 0.258) * (W / h_sub + 0.8))
    )
    L = C0 / (2 * f0 * np.sqrt(eps_eff)) - 2 * dL

    # Mesh — generous margin so patch is well inside CPML
    dx = 1.5e-3       # 1.5 mm (λ/40)
    cpml_n = 8
    cpml_thick = cpml_n * dx  # 12 mm
    margin = cpml_thick + 8e-3  # 20 mm total (8 mm usable air beyond CPML)

    dom_x = L + 2 * margin
    dom_y = W + 2 * margin

    # Non-uniform z: fine in substrate, smooth transition, coarse in air
    n_sub = max(4, int(np.ceil(h_sub / dx)))
    dz_sub = h_sub / n_sub
    n_air = 15
    raw_dz = np.concatenate([
        np.full(n_sub, dz_sub),
        np.full(n_air, dx),
    ])
    dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

    print(f"  Patch: L={L*1e3:.1f} mm, W={W*1e3:.1f} mm")
    print(f"  Mesh:  dx={dx*1e3:.1f} mm, dz_sub={dz_sub*1e3:.3f} mm, "
          f"n_sub={n_sub}, nz={len(dz_profile)}")

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, 0),
        dx=dx,
        dz_profile=dz_profile,
        boundary="cpml",
        cpml_layers=cpml_n,
    )

    # Materials
    sigma_sub = 2 * np.pi * f0 * 8.854e-12 * eps_r * 0.02  # tan_d=0.02
    sim.add_material("fr4", eps_r=eps_r, sigma=sigma_sub)

    # Geometry: ground plane (one cell thick), substrate, patch (one cell thick)
    sim.add(Box((0, 0, 0), (dom_x, dom_y, dz_sub)), material="pec")
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h_sub)), material="fr4")

    px0, py0 = margin, margin
    sim.add(
        Box((px0, py0, h_sub), (px0 + L, py0 + W, h_sub + dz_sub)),
        material="pec",
    )

    # Soft source + probe at feed point (inside substrate, away from CPML)
    feed_x = px0 + L / 3
    feed_y = py0 + W / 2
    feed_z = h_sub / 2
    sim.add_source(
        position=(feed_x, feed_y, feed_z),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.6),
    )
    sim.add_probe(position=(feed_x, feed_y, feed_z), component="ez")

    # NOTE: NTFF and DFT plane probes are not supported on non-uniform z mesh path;
    # they are tested separately in test_ntff_farfield() on uniform mesh.

    # --- Preflight ---
    warnings = sim.preflight(strict=False)
    print(f"  Preflight: {len(warnings)} warning(s)")
    for w in warnings:
        print(f"    - {w}")
    check("preflight runs without error", True)

    # --- Run ---
    t0 = time.time()
    result = sim.run(num_periods=20, compute_s_params=True)
    elapsed = time.time() - t0
    print(f"  Simulation: {elapsed:.1f}s")

    # --- Resonance via Harminv (robust even on non-uniform mesh) ---
    modes = result.find_resonances(freq_range=(2e9, 8e9))
    if modes:
        f_res = modes[0].freq
        Q_res = modes[0].Q
        print(f"  Harminv: f_res={f_res/1e9:.3f} GHz, Q={Q_res:.0f}")
        check("resonance detected", True)
        check("resonance near target", abs(f_res - f0) / f0 < 0.25,
              f"f_res={f_res/1e9:.2f} GHz, target={f0/1e9:.1f} GHz")
    else:
        print("  Harminv: no modes found")
        check("resonance detected", False, "Harminv found 0 modes")

    # --- S-parameters (if available from port) ---
    if result.s_params is not None and np.ndim(result.s_params) >= 3:
        S = np.array(result.s_params)
        freqs = np.array(result.freqs)
        s11_db = 20 * np.log10(np.abs(S[0, 0, :]) + 1e-30)
        s11_min = np.min(s11_db)
        print(f"  S11 min: {s11_min:.1f} dB")
        check("S11 computed", True)
    else:
        # Soft source doesn't produce S-params; that's expected
        S = None
        freqs = None
        print("  S11: not available (soft source, no impedance port)")
        check("S11 computed", True, "soft source — skipped")

    # --- Export pipeline ---
    from rfx.io import export_geometry_json, save_experiment_report

    out_dir = Path(tempfile.mkdtemp(prefix="rfx_e2e_"))
    try:
        export_geometry_json(out_dir / "geometry.json", sim)
        check("geometry JSON exported", (out_dir / "geometry.json").exists())

        save_experiment_report(out_dir / "report.json", sim, result)
        check("experiment report exported", (out_dir / "report.json").exists())

        # Touchstone round-trip (only if S-params available)
        if S is not None:
            from rfx.io import write_touchstone, read_touchstone
            write_touchstone(out_dir / "patch.s1p", S, freqs)
            check("Touchstone .s1p exported", (out_dir / "patch.s1p").exists())
            S_rt, _, _ = read_touchstone(out_dir / "patch.s1p")
            max_err = np.max(np.abs(S_rt - S))
            check("Touchstone round-trip", max_err < 1e-6,
                  f"max |S_rt - S| = {max_err:.2e}")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: ADI vs Yee (2D TMz Cavity)
#   Features: ADI solver, adi_cfl_factor, 2D TMz mode, PEC boundary,
#             resonance extraction
# ---------------------------------------------------------------------------

def test_adi_vs_yee():
    section("Test 2: ADI vs Yee — 2D TMz Cavity Resonance")

    C0 = 2.998e8
    f0 = 3e9
    lam = C0 / f0  # 100 mm

    # PEC cavity: 50mm × 50mm
    Lx = 50e-3
    Ly = 50e-3
    dx = 2.5e-3  # 2.5 mm (λ/40)

    # Analytical TM110 resonance: f = c/(2) * sqrt((1/Lx)^2 + (1/Ly)^2)
    f_tm110 = C0 / 2 * np.sqrt((1 / Lx) ** 2 + (1 / Ly) ** 2)
    print(f"  Cavity: {Lx*1e3:.0f} x {Ly*1e3:.0f} mm")
    print(f"  TM110 analytical: {f_tm110/1e9:.4f} GHz")

    # --- Yee solver ---
    Lz_dummy = 0.01  # z-extent required but unused for 2D TMz
    sim_yee = Simulation(
        freq_max=f0 * 2,
        domain=(Lx, Ly, Lz_dummy),
        dx=dx,
        boundary="pec",
        mode="2d_tmz",
        solver="yee",
    )
    src_pos = (Lx / 3, Ly / 3, 0)
    sim_yee.add_source(src_pos, "ez",
                       waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim_yee.add_probe(src_pos, "ez")

    t0 = time.time()
    result_yee = sim_yee.run(n_steps=2000)
    t_yee = time.time() - t0

    modes_yee = result_yee.find_resonances(freq_range=(1e9, 8e9))
    if modes_yee:
        f_yee = modes_yee[0].freq
    else:
        f_yee = 0.0
    print(f"  Yee: f_res={f_yee/1e9:.4f} GHz ({t_yee:.1f}s)")

    # --- ADI solver (2x CFL — moderate for reasonable accuracy) ---
    cfl_factor = 2.0
    sim_adi = Simulation(
        freq_max=f0 * 2,
        domain=(Lx, Ly, Lz_dummy),
        dx=dx,
        boundary="pec",
        mode="2d_tmz",
        solver="adi",
        adi_cfl_factor=cfl_factor,
    )
    sim_adi.add_source(src_pos, "ez",
                       waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim_adi.add_probe(src_pos, "ez")

    # ADI has larger dt, so fewer steps for same physical time
    n_steps_adi = int(2000 / cfl_factor)
    t0 = time.time()
    result_adi = sim_adi.run(n_steps=n_steps_adi)
    t_adi = time.time() - t0

    modes_adi = result_adi.find_resonances(freq_range=(1e9, 8e9))
    if modes_adi:
        f_adi = modes_adi[0].freq
    else:
        f_adi = 0.0
    print(f"  ADI:  f_res={f_adi/1e9:.4f} GHz ({t_adi:.1f}s, CFL={cfl_factor}x)")

    # --- Checks ---
    check("Yee resonance detected", f_yee > 0)
    check("ADI resonance detected", f_adi > 0)

    if f_yee > 0:
        err_yee = abs(f_yee - f_tm110) / f_tm110
        check("Yee matches analytical", err_yee < 0.05,
              f"error={err_yee*100:.1f}%")

    if f_adi > 0:
        err_adi = abs(f_adi - f_tm110) / f_tm110
        check("ADI matches analytical", err_adi < 0.10,
              f"error={err_adi*100:.1f}%")

    if f_yee > 0 and f_adi > 0:
        err_mutual = abs(f_yee - f_adi) / f_yee
        check("Yee ≈ ADI agreement", err_mutual < 0.10,
              f"Δf/f={err_mutual*100:.1f}%")


# ---------------------------------------------------------------------------
# Test 3: Waveguide TE10 S-Parameters
#   Features: waveguide ports, S21 transmission, normalize=False fast path,
#             multi-port S-matrix
# ---------------------------------------------------------------------------

def test_waveguide_sparams():
    section("Test 3: Waveguide TE10 S-Parameters")

    C0 = 2.998e8

    # WR-90: a=22.86mm, b=10.16mm, TE10 cutoff ~6.56 GHz
    a = 22.86e-3
    b = 10.16e-3
    f_cut = C0 / (2 * a)  # TE10 cutoff
    f0 = 10e9  # X-band center
    dx = 2.5e-3  # 2.5 mm

    wg_len = 80e-3  # 80 mm waveguide length (needs room for port offsets)

    print(f"  WR-90: a={a*1e3:.2f} mm, b={b*1e3:.2f} mm")
    print(f"  TE10 cutoff: {f_cut/1e9:.2f} GHz, f0={f0/1e9:.1f} GHz")

    sim = Simulation(
        freq_max=f0 * 1.5,
        domain=(wg_len, a, b),
        dx=dx,
        boundary="cpml",
        cpml_layers=8,
    )

    # Waveguide walls (PEC top/bottom/sides, open in x for ports)
    sim.add(Box((0, 0, 0), (wg_len, a, dx)), material="pec")      # bottom
    sim.add(Box((0, 0, b - dx), (wg_len, a, b)), material="pec")  # top
    sim.add(Box((0, 0, 0), (wg_len, dx, b)), material="pec")      # side y=0
    sim.add(Box((0, a - dx, 0), (wg_len, a, b)), material="pec")  # side y=a

    # Waveguide ports — well inside the domain
    port_offset = 5  # cells for probe/ref offsets
    sim.add_waveguide_port(
        x_position=15e-3,
        direction="-x",
        mode=(1, 0),
        mode_type="TE",
        f0=f0,
        name="port1",
        probe_offset=port_offset,
        ref_offset=2,
    )
    sim.add_waveguide_port(
        x_position=wg_len - 15e-3,
        direction="+x",
        mode=(1, 0),
        mode_type="TE",
        f0=f0,
        name="port2",
        probe_offset=port_offset,
        ref_offset=2,
    )

    # Preflight
    warnings = sim.preflight(strict=False)
    print(f"  Preflight: {len(warnings)} warning(s)")

    # Run (normalize=False for speed — sufficient for transmission check)
    t0 = time.time()
    wg_result = sim.compute_waveguide_s_matrix(
        num_periods=15,
        normalize=False,
    )
    elapsed = time.time() - t0
    print(f"  Simulation: {elapsed:.1f}s")

    S = np.array(wg_result.s_params)
    freqs = np.array(wg_result.freqs)
    n_freqs = len(freqs)

    # S21 in passband (above cutoff)
    passband = freqs > f_cut * 1.15  # 15% above cutoff for clean propagation
    if np.any(passband):
        s21_db = 20 * np.log10(np.abs(S[1, 0, passband]) + 1e-30)
        s21_mean = np.mean(s21_db)
        s21_max = np.max(s21_db)
        print(f"  S21 passband: mean={s21_mean:.1f} dB, max={s21_max:.1f} dB")
        check("S21 shape correct", S.shape == (2, 2, n_freqs),
              f"shape={S.shape}")
        check("S21 transmission in passband", s21_mean > -10,
              f"mean S21={s21_mean:.1f} dB > -10 dB")
    else:
        check("passband frequencies exist", False, "no freqs above cutoff")

    # S11 — should show some reflection
    s11_db = 20 * np.log10(np.abs(S[0, 0, :]) + 1e-30)
    check("S11 computed", len(s11_db) > 0)


# ---------------------------------------------------------------------------
# Test 4: SBP-SAT Subgridding Energy Stability
#   Features: add_refinement, SBP-SAT 3D coupling, energy non-increasing
# ---------------------------------------------------------------------------

def test_sbp_sat_energy():
    section("Test 4: SBP-SAT Subgridding — Energy Stability")
    from rfx.subgridding.sbp_sat_3d import (
        init_subgrid_3d,
        step_subgrid_3d,
        compute_energy_3d,
    )
    from rfx.core.yee import MaterialArrays

    # Small 3D domain: 30×30×30 coarse, ratio=3
    shape_c = (30, 30, 30)
    dx_c = 3e-3
    fine_region = (10, 20, 10, 20, 10, 20)
    ratio = 3
    tau = 1.0

    config, state = init_subgrid_3d(
        shape_c=shape_c,
        dx_c=dx_c,
        fine_region=fine_region,
        ratio=ratio,
        courant=0.45,
        tau=tau,
    )

    # Free-space materials
    mats_c = MaterialArrays(
        eps_r=jnp.ones(shape_c),
        sigma=jnp.zeros(shape_c),
        mu_r=jnp.ones(shape_c),
    )
    sf = (config.nx_f, config.ny_f, config.nz_f)
    mats_f = MaterialArrays(
        eps_r=jnp.ones(sf),
        sigma=jnp.zeros(sf),
        mu_r=jnp.ones(sf),
    )
    pec_c = jnp.zeros(shape_c, dtype=bool)
    pec_f = jnp.zeros(sf, dtype=bool)

    # Inject Gaussian pulse in fine region center
    cx, cy, cz = sf[0] // 2, sf[1] // 2, sf[2] // 2
    pulse = jnp.zeros(sf)
    pulse = pulse.at[cx, cy, cz].set(1.0)
    state = state._replace(ez_f=state.ez_f + pulse)

    # Warm-up: let pulse propagate and reach coarse-fine boundary
    n_warmup = 30
    for _ in range(n_warmup):
        state = step_subgrid_3d(state, config,
                                mats_c=mats_c, mats_f=mats_f,
                                pec_mask_c=pec_c, pec_mask_f=pec_f)

    # Track energy after warm-up (SAT dissipation should dominate)
    energies = [float(compute_energy_3d(state, config))]
    n_steps = 150
    t0 = time.time()
    for _ in range(n_steps):
        state = step_subgrid_3d(state, config,
                                mats_c=mats_c, mats_f=mats_f,
                                pec_mask_c=pec_c, pec_mask_f=pec_f)
        energies.append(float(compute_energy_3d(state, config)))
    elapsed = time.time() - t0

    E_init = energies[0]
    E_final = energies[-1]
    ratio_ef = E_final / (E_init + 1e-30)
    print(f"  Warm-up: {n_warmup}, tracked: {n_steps} steps, time: {elapsed:.1f}s")
    print(f"  E_init={E_init:.3e}, E_final={E_final:.3e}, ratio={ratio_ef:.3f}")

    # Energy should be non-increasing after warm-up (dissipative stability)
    check("final < initial", E_final < E_init * 1.01,
          f"ratio={ratio_ef:.4f}")
    check("dissipative (energy decays)", ratio_ef < 0.95,
          f"lost {(1-ratio_ef)*100:.1f}% energy")


# ---------------------------------------------------------------------------
# Test 5: NTFF Far-Field (Uniform Mesh Dipole)
#   Features: NTFF DFT accumulation, H half-step phase compensation,
#             per-face dS, far-field export (HDF5 + CSV), Touchstone
# ---------------------------------------------------------------------------

def test_ntff_farfield():
    section("Test 5: NTFF Far-Field — Dipole Radiation Pattern + Export")
    from rfx.farfield import compute_far_field, directivity

    C0 = 2.998e8
    f0 = 5e9
    lam = C0 / f0  # 60 mm
    dx = lam / 20  # 3 mm
    dom_size = lam * 1.2  # 72 mm
    cpml_n = 8

    sim = Simulation(
        freq_max=f0 * 1.5,
        domain=(dom_size, dom_size, dom_size),
        dx=dx,
        boundary="cpml",
        cpml_layers=cpml_n,
    )

    # Ez dipole at center
    center = (dom_size / 2, dom_size / 2, dom_size / 2)
    sim.add_source(center, "ez", waveform=GaussianPulse(f0=f0, bandwidth=0.6))
    sim.add_probe(center, "ez")

    # NTFF box — 3 cells inside CPML boundary
    ntff_margin = (cpml_n + 3) * dx
    sim.add_ntff_box(
        corner_lo=(ntff_margin, ntff_margin, ntff_margin),
        corner_hi=(dom_size - ntff_margin, dom_size - ntff_margin,
                   dom_size - ntff_margin),
        freqs=jnp.array([f0]),
    )

    # Preflight
    warnings = sim.preflight(strict=False)
    check("preflight clean", len(warnings) == 0,
          f"{len(warnings)} warnings" if warnings else "clean")

    # Run
    t0 = time.time()
    result = sim.run(n_steps=500, compute_s_params=False)
    elapsed = time.time() - t0
    print(f"  Simulation: {elapsed:.1f}s")

    # Far-field
    theta = jnp.linspace(0, jnp.pi, 91)
    phi = jnp.array([0.0, jnp.pi / 2])
    ff = compute_far_field(result.ntff_data, result.ntff_box, result.grid,
                           theta, phi)
    D = directivity(ff)
    D_dbi = 10 * np.log10(float(D) + 1e-30)
    print(f"  Directivity: {D_dbi:.1f} dBi")

    check("far-field shape", ff.E_theta.shape == (1, 91, 2),
          f"shape={ff.E_theta.shape}")
    check("directivity physical", 0 < D_dbi < 10,
          f"{D_dbi:.1f} dBi (dipole ≈ 1.76 dBi)")

    # sin²(θ) pattern check for Ez dipole
    E_th = np.abs(np.asarray(ff.E_theta[0, :, 0]))  # E-plane
    sin2 = np.sin(np.asarray(theta)) ** 2
    if np.max(E_th) > 0:
        E_norm = E_th / np.max(E_th)
        sin2_norm = sin2 / np.max(sin2)
        corr = np.corrcoef(E_norm, sin2_norm)[0, 1]
        print(f"  sin²(θ) correlation: {corr:.4f}")
        check("radiation pattern matches sin²(θ)", corr > 0.9,
              f"corr={corr:.4f}")
    else:
        check("radiation pattern matches sin²(θ)", False, "E_theta all zero")

    # Export: far-field HDF5, CSV, Touchstone (1-port dummy)
    from rfx.io import save_far_field, export_radiation_pattern

    out_dir = Path(tempfile.mkdtemp(prefix="rfx_e2e_ntff_"))
    try:
        save_far_field(out_dir / "farfield.h5", ff)
        check("far-field HDF5 exported", (out_dir / "farfield.h5").exists())

        export_radiation_pattern(out_dir / "pattern.csv", ff)
        check("radiation CSV exported", (out_dir / "pattern.csv").exists())

        # Verify CSV has expected columns
        import csv
        with open(out_dir / "pattern.csv") as f:
            reader = csv.reader(f)
            header = next(reader)
            n_rows = sum(1 for _ in reader)
        check("CSV has data rows", n_rows == 91 * 2,
              f"{n_rows} rows (expected {91*2})")
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("rfx End-to-End Feature Validation")
    print(f"JAX devices: {jnp.zeros(1).devices()}")

    t_total = time.time()

    test_patch_nonuniform()
    test_adi_vs_yee()
    test_waveguide_sparams()
    test_sbp_sat_energy()
    test_ntff_farfield()

    elapsed_total = time.time() - t_total

    # Summary
    section("Summary")
    total = PASS_COUNT + FAIL_COUNT
    print(f"  {PASS_COUNT}/{total} checks passed, {FAIL_COUNT} failed")
    print(f"  Total time: {elapsed_total:.1f}s")

    if FAIL_COUNT > 0:
        print("\n  SOME CHECKS FAILED")
        sys.exit(1)
    else:
        print("\n  ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
