#!/usr/bin/env python3
"""Isolate whether the rfx vs Meep discrepancy is in the extractor or the FDTD engine.

Tests:

  A) Empty-guide sanity at the port-2 plane:
     - Drive port 1 in rfx empty guide.
     - Extract rfx's port-2 (a, b) = (incident, outgoing).
     - Clean source + clean CPML + clean extractor ⇒
         |outgoing_port2 / a_port1_ref| ≈ 1   (lossless transmission)
         |incident_port2| ≈ 0                 (no reverse incident)
         ∠(outgoing_port2 / a_port1_ref) ≈ −β(f)·L_ports

  B) If (A) deviates from the clean-guide analytic by 5-20%, the gap is
     in the extractor or CPML (not device-specific). If (A) matches within
     1%, the extractor is OK on empty guide and the slab case's
     ±20% col_power swing is about the device normalization path.

  C) Explicit reference-plane test:
     - Re-run slab device with user-specified reference_plane matching
       Meep's mon_left/mon_right absolute positions.
     - If rfx vs Meep phase offset disappears → offset was reference-plane
       convention, NOT an extractor bug.

Run: `python scripts/isolate_extractor_vs_engine.py`.
Outputs are plain text tables; no assertions, no pytest.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.waveguide_port import (
    extract_waveguide_port_waves,
    _compute_beta,
    _compute_mode_impedance,
    _co_located_current_spectrum,
)

C0 = 2.998e8

# Match the crossval/11 and Meep script geometry exactly.
A_WG = 0.02286
B_WG = 0.01016
DOMAIN_X = 0.200
PORT_LEFT_X = 0.030
PORT_RIGHT_X = 0.170
DX_M = 0.001
CPML_LAYERS = int(os.environ.get("RFX_CPML_LAYERS", "10"))
KAPPA_MAX = float(os.environ.get("RFX_KAPPA_MAX", "1.0"))
NUM_PERIODS = 50
F_CUTOFF_TE10 = C0 / (2.0 * A_WG)

FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0_HZ = float(FREQS_HZ.mean())
BANDWIDTH = 0.5


def _build_empty_sim():
    sim = Simulation(
        freq_max=FREQS_HZ[-1] * 1.1,
        domain=(DOMAIN_X, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML_LAYERS,
        cpml_kappa_max=KAPPA_MAX,
        dx=DX_M,
    )
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ), f0=F0_HZ, bandwidth=BANDWIDTH,
        waveform="modulated_gaussian", name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ), f0=F0_HZ, bandwidth=BANDWIDTH,
        waveform="modulated_gaussian", name="right",
    )
    return sim


def run_and_extract() -> dict:
    """Drive port 1 only in empty guide, extract raw (a,b) at both ports."""
    sim = _build_empty_sim()
    # Drive port 1 only. sim.compute_waveguide_s_matrix drives one port at a
    # time and normalizes; we skip normalization and just inspect the raw
    # final cfgs.
    # Path: run once with only port 1 amplitude active.
    from rfx.simulation import run as run_simulation
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        base_materials = base_materials._replace(
            sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
    materials = base_materials
    n_steps = grid.num_timesteps(num_periods=NUM_PERIODS)
    _, debye, lorentz = sim._init_dispersion(materials, grid.dt, debye_spec, lorentz_spec)

    freqs_arr = jnp.asarray(FREQS_HZ)
    entries = list(sim._waveguide_ports)
    raw_cfgs = [sim._build_waveguide_port_config(e, grid, freqs_arr, n_steps)
                for e in entries]
    cfgs_template = [c[0] if isinstance(c, list) else c for c in raw_cfgs]

    # Configure: only port 1 (left) driven
    driven = []
    for idx, cfg in enumerate(cfgs_template):
        zeros = jnp.zeros_like(cfg.v_probe_dft)
        driven.append(cfg._replace(
            src_amp=cfg.src_amp if idx == 0 else 0.0,
            v_probe_dft=zeros, v_ref_dft=zeros,
            i_probe_dft=zeros, i_ref_dft=zeros,
            v_inc_dft=zeros,
        ))

    result = run_simulation(
        grid, materials, n_steps,
        boundary="cpml", cpml_axes="x", pec_axes="yz",
        debye=debye, lorentz=lorentz,
        waveguide_ports=driven,
    )
    final_cfgs = result.waveguide_ports or ()
    if len(final_cfgs) != 2:
        raise RuntimeError(f"expected 2 port cfgs, got {len(final_cfgs)}")

    a1, b1 = extract_waveguide_port_waves(final_cfgs[0])
    a2, b2 = extract_waveguide_port_waves(final_cfgs[1])

    # Raw V, I at both ports' ref planes (to inspect Z = V/I directly).
    cfg1 = final_cfgs[0]
    cfg2 = final_cfgs[1]
    v1 = np.asarray(cfg1.v_ref_dft)
    i1 = np.asarray(_co_located_current_spectrum(cfg1, cfg1.i_ref_dft))
    v2 = np.asarray(cfg2.v_ref_dft)
    i2 = np.asarray(_co_located_current_spectrum(cfg2, cfg2.i_ref_dft))
    # What Z rfx FORMULA claims (used inside extract). If the actual Yee Z
    # matches the formula, V/I should equal Z_formula for a pure forward wave.
    z_formula = np.asarray(_compute_mode_impedance(
        jnp.asarray(FREQS_HZ), F_CUTOFF_TE10, "TE",
        dt=cfg1.dt, dx=cfg1.dx,
    ))

    return {
        "freqs_hz": FREQS_HZ.tolist(),
        "a1_ref": np.asarray(a1),
        "b1_ref": np.asarray(b1),
        "a2_ref": np.asarray(a2),
        "b2_ref": np.asarray(b2),
        "v1": v1, "i1": i1,
        "v2": v2, "i2": i2,
        "z_formula": z_formula,
    }


def _safe_ratio(num, den):
    d = np.where(np.abs(den) > 1e-30, den, 1.0 + 0j)
    return num / d


def _phase_deg(z):
    return float(np.angle(z) * 180 / np.pi)


def _load_meep():
    path = ("/root/workspace/byungkwan-workspace/research/microwave-energy/"
            "results/rfx_crossval_wr90_meep/wr90_meep_reference.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> int:
    print("=== Empty-guide raw (a,b) extraction in rfx ===")
    data = run_and_extract()
    freqs = np.asarray(data["freqs_hz"])
    a1 = data["a1_ref"]   # incident at port 1 (forward wave)
    b1 = data["b1_ref"]   # outgoing at port 1 (reflected wave in empty should be ~0)
    a2 = data["a2_ref"]   # incident at port 2 (should be ~0 in empty, drive left only)
    b2 = data["b2_ref"]   # outgoing at port 2 (transmitted wave ~= a1 × e^{-jβ·L})

    # Expected: in empty guide driven from left, port 2 "outgoing" (= +x wave
    # arriving at port 2 from source) should equal port 1 "incident"
    # (source wave) times e^{-jβ·(port2_ref − port1_ref)}.
    port_sep_m = (PORT_RIGHT_X - CPML_LAYERS*DX_M - 3*DX_M) - \
                 (PORT_LEFT_X + 3*DX_M)  # ref_x = port_x ± 3·dx inward
    # Actually ref_x for "+x" port is port_x + 3·dx; for "-x" port is port_x − 3·dx.
    port_sep_m = (PORT_RIGHT_X - 3*DX_M) - (PORT_LEFT_X + 3*DX_M)
    print(f"rfx reference-plane separation: {port_sep_m*1e3:.2f} mm")

    beta = np.asarray(_compute_beta(jnp.asarray(freqs), F_CUTOFF_TE10))
    expected_phase = -beta * port_sep_m * 180 / np.pi  # degrees
    transmission_rfx = _safe_ratio(b2, a1)

    print("\n --- rfx empty guide sanity (port 1 drives, port 2 reads) ---")
    header = (" f/GHz  |a1|_rfx  |b1|_rfx  |a2|_rfx  |b2|_rfx  |b2/a1|"
              "  ∠(b2/a1)   ∠expected  Δ∠")
    print(header)
    print("-" * len(header))
    for i, f in enumerate(freqs):
        row = (f" {f/1e9:5.2f}"
               f"  {abs(a1[i]):7.4f}  {abs(b1[i]):7.4f}"
               f"  {abs(a2[i]):7.4f}  {abs(b2[i]):7.4f}"
               f"  {abs(transmission_rfx[i]):7.4f}"
               f"  {_phase_deg(transmission_rfx[i]):8.2f}°"
               f"  {expected_phase[i]:8.2f}°"
               f"  {_phase_deg(transmission_rfx[i])-expected_phase[i]:+7.2f}°")
        print(row)

    # === Sanity bucket interpretation ===
    print("\n=== Interpretation ===")
    trans_mag = np.abs(transmission_rfx)
    print(f"  |b2/a1| (empty transmission): range [{trans_mag.min():.4f}, "
          f"{trans_mag.max():.4f}] mean {trans_mag.mean():.4f}")
    print(f"  Clean source + clean CPML + clean extractor ⇒ expect |b2/a1|=1.0")

    b1_over_a1 = np.abs(b1) / (np.abs(a1) + 1e-30)
    print(f"  |b1/a1| (port-1 reflection in empty): range [{b1_over_a1.min():.4f}, "
          f"{b1_over_a1.max():.4f}] mean {b1_over_a1.mean():.4f}")
    print(f"  Clean CPML ⇒ expect |b1/a1|~0 (no reflection in empty guide)")

    a2_over_a1 = np.abs(a2) / (np.abs(a1) + 1e-30)
    print(f"  |a2/a1| (incoming to port 2, empty): range [{a2_over_a1.min():.4f}, "
          f"{a2_over_a1.max():.4f}] mean {a2_over_a1.mean():.4f}")
    print(f"  drive-left-only empty ⇒ expect |a2/a1|~0 (no wave INTO port 2)")

    # === Compare with Meep ===
    meep = _load_meep()
    if meep is not None:
        for rk in ["r3", "r4"]:
            if rk not in meep:
                continue
            e = meep[rk].get("empty")
            if e is None:
                continue
            T_amp_meep = np.asarray(e["ref_transmission_abs"])
            print(f"\n  Meep@{rk}  |T_ref| range [{T_amp_meep.min():.4f}, "
                  f"{T_amp_meep.max():.4f}] mean {T_amp_meep.mean():.4f}  "
                  "(expect ≈1 for lossless empty)")

    print("\n=== Rule-of-thumb verdicts ===")
    if trans_mag.mean() > 0.95 and trans_mag.mean() < 1.05:
        print("  ✓ rfx source+engine+extractor produces empty-guide transmission ≈ 1.0 (within 5%).")
    else:
        print(f"  ✗ rfx empty-guide transmission = {trans_mag.mean():.3f}; OFF from 1.0.")
    if b1_over_a1.max() < 0.05:
        print("  ✓ port-1 CPML reflection in empty guide is <5%.")
    else:
        print(f"  ✗ port-1 reflection max = {b1_over_a1.max():.3f}; CPML leak OR source rear leakage.")

    # === Direct Z(actual) check at port 2 (far-field pure forward wave) ===
    # For a pure forward wave at port 2: V = A, I = A/Z_actual (after the
    # _co_located_current_spectrum fix). So V/I = Z_actual.
    print("\n=== Z (actual) from V/I at port 2 vs Z_formula ===")
    z_formula = data["z_formula"]
    v2 = data["v2"]; i2 = data["i2"]
    # Only use freqs where |i2| is non-trivial (source spectrum).
    # Raw |I2| is tiny (~1e-14) because source excitation amplitude is small
    # after bandpass filtering; ratio V/I is still well-defined.
    z_actual = np.where(np.abs(i2) > 1e-25, v2 / i2, np.nan + 0j)
    print("  f/GHz  |V2|       |I2|       Z_formula      Z_actual=V2/I2      ratio(|Zf/Za|)  ∠(Zf/Za)°")
    for idx in [0, 5, 10, 15, 20]:
        f = FREQS_HZ[idx]
        ratio = z_formula[idx] / z_actual[idx] if not np.isnan(z_actual[idx]).any() else np.nan+0j
        line = (f"   {f/1e9:5.2f}  {abs(v2[idx]):.3e}  {abs(i2[idx]):.3e}  "
                f"{z_formula[idx].real:8.2f}+{z_formula[idx].imag:+7.2f}j  "
                f"{z_actual[idx].real:8.2f}+{z_actual[idx].imag:+7.2f}j  "
                f"{abs(ratio):6.4f}  {np.angle(ratio)*180/np.pi:+7.2f}°")
        print(line)
    ratio_mean = np.mean(np.abs(z_formula / z_actual))
    print(f"\n  Mean |Z_formula / Z_actual| = {ratio_mean:.4f}")
    expected_b_over_a = abs(1 - ratio_mean) / abs(1 + ratio_mean)
    print(f"  Predicted |b/a| from Z mismatch: {expected_b_over_a:.4f}")
    print(f"  Observed |b/a| at port 2: {(np.abs(data['a2_ref'])/np.abs(data['b2_ref'])).mean():.4f}")
    print()
    # Phase of Zf/Za: for real Z (lossless), should be ~0. Non-zero phase
    # means there's an imaginary component in Z_actual — indicative of
    # time-stagger residual not being cancelled by _co_located_current_spectrum.
    phases = np.angle(z_formula / z_actual) * 180 / np.pi
    print(f"  ∠(Z_formula / Z_actual) range [{phases.min():.2f}°, {phases.max():.2f}°] mean {phases.mean():.2f}°")
    print("  For lossless guide, both Z values should be real ⇒ ∠(Zf/Za) ≈ 0°.")


if __name__ == "__main__":
    sys.exit(main() or 0)
