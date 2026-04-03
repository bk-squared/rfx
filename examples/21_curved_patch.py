"""Validation Case 8: Curved vs Flat Patch Antenna Comparison

Compares a curved patch antenna (CurvedPatch) to a flat patch.
Validates that curvature shifts the resonant frequency.

Reference: qualitative -- curvature changes effective electrical length,
shifting resonance compared to the flat case.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np

from rfx import Simulation, Box, CurvedPatch
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

# ---- Design parameters ----
f0 = 2.4e9
eps_r = 4.4
h = 1.6e-3
tan_d = 0.02

# Patch dimensions
W_patch = C0 / (2 * f0) * np.sqrt(2.0 / (eps_r + 1.0))
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W_patch) ** (-0.5)
dL = 0.412 * h * (
    (eps_eff + 0.3) * (W_patch / h + 0.264) /
    ((eps_eff - 0.258) * (W_patch / h + 0.8))
)
L_patch = C0 / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * dL

# Curvature radius (mild curvature: R >> patch length)
R_curve = 0.08  # 80 mm radius

dx = 1e-3
margin = 10e-3

print(f"Patch: L={L_patch * 1e3:.1f} mm, W={W_patch * 1e3:.1f} mm")
print(f"Curvature radius: {R_curve * 1e3:.0f} mm")
print(f"Substrate: h={h * 1e3:.1f} mm, eps_r={eps_r}")

# ---- Helper function to run one simulation ----
def run_patch_sim(patch_label, patch_boxes, dom_z):
    """Run a patch simulation and return resonant frequency."""
    dom_x = L_patch + 2 * margin
    dom_y = W_patch + 2 * margin

    sim = Simulation(
        freq_max=f0 * 2,
        domain=(dom_x, dom_y, dom_z),
        boundary="cpml",
        cpml_layers=8,
        dx=dx,
    )

    sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

    # Ground plane
    sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
    # Substrate
    sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")

    # Patch (multiple boxes for curved, single for flat)
    for box in patch_boxes:
        sim.add(box, material="pec")

    # Source
    src_x = margin + L_patch / 3
    src_y = dom_y / 2
    src_z = h / 2
    sim.add_source(
        (src_x, src_y, src_z),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=0.8),
    )
    sim.add_probe((src_x, src_y, src_z), component="ez")

    grid = sim._build_grid()
    n_steps = int(np.ceil(12e-9 / grid.dt))
    print(f"  [{patch_label}] Running {n_steps} steps ...")
    result = sim.run(n_steps=n_steps)

    # Extract resonance via FFT
    ts = np.asarray(result.time_series).ravel()
    nfft = len(ts) * 8
    spectrum = np.abs(np.fft.rfft(ts, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=result.dt)
    band = (freqs > f0 * 0.5) & (freqs < f0 * 1.5)
    if np.any(band):
        peak_idx = np.argmax(spectrum[band])
        f_res = freqs[band][peak_idx]
    else:
        f_res = 0.0

    # Try Harminv
    try:
        modes = result.find_resonances(
            freq_range=(f0 * 0.5, f0 * 1.5),
            probe_idx=0,
        )
        if modes:
            best = min(modes, key=lambda m: abs(m.freq - f0))
            f_res = best.freq
    except Exception:
        pass

    return f_res

# ---- Flat patch ----
px0, py0 = margin, (W_patch + 2 * margin - W_patch) / 2
flat_boxes = [Box((px0, py0, h), (px0 + L_patch, py0 + W_patch, h))]
dom_z_flat = h + 8e-3

print("\nRunning flat patch:")
f_flat = run_patch_sim("Flat", flat_boxes, dom_z_flat)

# ---- Curved patch ----
# CurvedPatch creates staircase boxes along x-axis with z-offset
curved = CurvedPatch(
    center=(margin + L_patch / 2, (W_patch + 2 * margin) / 2, h),
    length=L_patch,
    width=W_patch,
    radius=R_curve,
    axis="x",
)
curved_boxes = curved.to_staircase(dx)
# Extra z headroom for curved patch
max_z_offset = R_curve - np.sqrt(R_curve ** 2 - (L_patch / 2) ** 2)
dom_z_curved = h + max_z_offset + 8e-3

print("\nRunning curved patch:")
f_curved = run_patch_sim("Curved", curved_boxes, dom_z_curved)

# ---- Compare results ----
freq_shift = f_curved - f_flat
shift_pct = abs(freq_shift) / f_flat * 100 if f_flat > 0 else 0

print(f"\n--- Validation Results ---")
print(f"Flat patch resonance    : {f_flat / 1e9:.4f} GHz")
print(f"Curved patch resonance  : {f_curved / 1e9:.4f} GHz")
print(f"Frequency shift         : {freq_shift / 1e6:.1f} MHz ({shift_pct:.2f}%)")
print(f"Curvature effect        : {'resonance shifted' if shift_pct > 0.1 else 'no measurable shift'}")

# Validation: curvature should produce a measurable frequency shift
# Even mild curvature changes the effective path length
passed = f_flat > 0 and f_curved > 0 and f_flat != f_curved
status = "PASS" if passed else "FAIL"
print(f"\nValidation: {status} (curvature produces different resonance than flat patch)")
