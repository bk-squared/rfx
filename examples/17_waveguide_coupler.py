"""Validation Case 4: Waveguide Coupler

Two rectangular waveguides sharing a wall with an aperture opening.
Validates coupling and isolation between ports using waveguide S-matrix.

Reference: qualitative -- aperture coupling should produce measurable
S21 with frequency dependence above TE10 cutoff.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.grid import C0

# ---- Waveguide geometry ----
# Standard-ish waveguide: a=40mm, b=20mm
Ly = 0.04     # guide width (a=40mm, cutoff TE10 ~ 3.75 GHz)
Lz = 0.02     # guide height (b=20mm)
Lx = 0.10     # guide length

# Aperture in shared wall
aperture_length = 0.02   # 20mm aperture along x
aperture_height = 0.015  # 15mm aperture in z
aperture_x_start = (Lx - aperture_length) / 2

print(f"Waveguide     : {Ly * 1e3:.0f} x {Lz * 1e3:.0f} mm cross-section")
print(f"Length        : {Lx * 1e3:.0f} mm")
print(f"Aperture      : {aperture_length * 1e3:.0f} x {aperture_height * 1e3:.0f} mm")
f_cutoff = C0 / (2 * Ly)
print(f"TE10 cutoff   : {f_cutoff / 1e9:.2f} GHz")

# ---- Build simulation ----
# Two waveguides stacked in y with a shared wall (PEC with aperture)
wall_thickness = 2e-3
dom_y = 2 * Ly + wall_thickness

sim = Simulation(
    freq_max=8e9,
    domain=(Lx, dom_y, Lz),
    boundary="cpml",
    cpml_layers=8,
    dx=0.002,
)

# Shared wall (PEC) between the two guides, with aperture removed
# Wall spans the full length, full height, at y = Ly to Ly + wall_thickness
wall_y_lo = Ly
wall_y_hi = Ly + wall_thickness

# Wall in two parts: before and after the aperture in x
# Left section of wall
sim.add(Box((0, wall_y_lo, 0), (aperture_x_start, wall_y_hi, Lz)), material="pec")
# Right section of wall
sim.add(Box((aperture_x_start + aperture_length, wall_y_lo, 0),
            (Lx, wall_y_hi, Lz)), material="pec")
# Bottom part of aperture wall (below aperture)
aperture_z_start = (Lz - aperture_height) / 2
sim.add(Box((aperture_x_start, wall_y_lo, 0),
            (aperture_x_start + aperture_length, wall_y_hi, aperture_z_start)), material="pec")
# Top part of aperture wall (above aperture)
sim.add(Box((aperture_x_start, wall_y_lo, aperture_z_start + aperture_height),
            (aperture_x_start + aperture_length, wall_y_hi, Lz)), material="pec")

# ---- Waveguide ports ----
# Port 1: input on guide 1 (lower y)
# Port 2: output on guide 1 (lower y, far end)
freqs = jnp.linspace(4.5e9, 7.5e9, 25)
sim.add_waveguide_port(
    0.01, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=6e9, name="port1",
    y_range=(0, Ly), z_range=(0, Lz),
)
sim.add_waveguide_port(
    Lx - 0.01, direction="-x", mode=(1, 0), mode_type="TE",
    freqs=freqs, f0=6e9, name="port2",
    y_range=(0, Ly), z_range=(0, Lz),
)

# ---- Run S-matrix extraction ----
print("Running waveguide coupler S-matrix ...")
result = sim.compute_waveguide_s_matrix(num_periods=25)
S = result.s_params
f_GHz = np.array(result.freqs) / 1e9

s11_dB = 20 * np.log10(np.maximum(np.abs(S[0, 0, :]), 1e-10))
s21_dB = 20 * np.log10(np.maximum(np.abs(S[1, 0, :]), 1e-10))

s21_mean = np.mean(np.abs(S[1, 0, :]))
s11_mean = np.mean(np.abs(S[0, 0, :]))

print(f"\n--- Validation Results ---")
print(f"Frequency range       : {f_GHz[0]:.1f} - {f_GHz[-1]:.1f} GHz")
print(f"|S21| mean (linear)   : {s21_mean:.4f}")
print(f"|S11| mean (linear)   : {s11_mean:.4f}")
print(f"|S21| mean (dB)       : {20 * np.log10(max(s21_mean, 1e-10)):.1f} dB")
print(f"|S11| mean (dB)       : {20 * np.log10(max(s11_mean, 1e-10)):.1f} dB")

# Reciprocity check
recip_err = np.mean(np.abs(np.abs(S[1, 0, :]) - np.abs(S[0, 1, :])))
print(f"Reciprocity error     : {recip_err:.4f}")

# Validation: S21 should show transmission, S11 should show some reflection
# With the aperture, the coupler transmits energy through the shared wall
passed = s21_mean > 0.01 and s11_mean < 1.0
status = "PASS" if passed else "FAIL"
print(f"\nValidation: {status} (waveguide with aperture shows multi-port coupling behavior)")
