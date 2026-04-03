"""Example 4: 2.4 GHz Microstrip Patch Antenna on FR4

Showcase example demonstrating non-uniform mesh (graded dz for thin substrate),
analytical patch design via Hammerstad formula, Harminv resonance extraction,
and comprehensive 6-panel visualization.

Design flow:
  1. Hammerstad formula → patch length L, width W, effective eps_eff
  2. Simulation(dz_profile=...) for non-uniform z-grid (fine in substrate)
  3. add_source + add_probe for soft excitation and ring-down recording
  4. find_resonances() via Harminv for accurate frequency and Q extraction
  5. 6-panel figure: geometry (2 slices + PEC mask) + results (time, spectrum, annotation)

Saves: examples/04_patch_antenna.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

# ---- Physical constants ----
C0 = 2.998e8   # speed of light (m/s)

# ---- Design parameters ----
f0 = 2.4e9
eps_r = 4.4          # FR4 relative permittivity
tan_d = 0.02         # loss tangent
h = 1.6e-3           # substrate thickness

# ---- Hammerstad patch dimensions ----
W = C0 / (2 * f0) * np.sqrt(2.0 / (eps_r + 1.0))
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)
dL = 0.412 * h * (
    (eps_eff + 0.3) * (W / h + 0.264) /
    ((eps_eff - 0.258) * (W / h + 0.8))
)
L = C0 / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * dL

print(f"Patch dimensions : L={L * 1e3:.2f} mm  W={W * 1e3:.2f} mm")
print(f"Substrate        : h={h * 1e3:.1f} mm  eps_r={eps_r}  tan_d={tan_d}")
print(f"eps_eff          : {eps_eff:.4f}")

# ---- Mesh parameters ----
dx = 0.5e-3           # lateral cell size (0.5 mm)
margin = 15e-3        # air margin around patch
dom_x = L + 2 * margin
dom_y = W + 2 * margin

# ---- Non-uniform z-profile: fine cells in substrate, coarse in air ----
n_sub = max(4, int(np.ceil(h / dx)))   # at least 4 cells through substrate
dz_sub = h / n_sub                     # fine z cell
n_air = max(6, int(np.ceil(margin / dx)))
dz_profile = np.concatenate([
    np.full(n_sub, dz_sub),            # substrate region
    np.full(n_air, dx),                # air above substrate
])
dom_z_total = float(np.sum(dz_profile))

print(f"\nDomain           : {dom_x * 1e3:.0f} x {dom_y * 1e3:.0f} x {dom_z_total * 1e3:.1f} mm")
print(f"dz_sub           : {dz_sub * 1e3:.3f} mm  ({n_sub} cells in substrate)")
print(f"dx               : {dx * 1e3:.1f} mm")

# ---- Build simulation ----
sigma_sub = 2.0 * np.pi * f0 * 8.854e-12 * eps_r * tan_d

sim = Simulation(
    freq_max=f0 * 2.0,
    domain=(dom_x, dom_y, 0),   # z=0 sentinel; actual z from dz_profile
    dx=dx,
    dz_profile=dz_profile,
    cpml_layers=12,
)

# ---- Materials ----
sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)

# ---- Geometry ----
# Ground plane: bottom face of substrate (z=0 plane, 1 cell thick)
sim.add(Box((0, 0, 0), (dom_x, dom_y, 0)), material="pec")
# FR4 substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="substrate")
# Patch: top surface of substrate
px0, py0 = margin, margin
sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h)), material="pec")

# ---- Source: soft point source near feed point ----
# Feed at L/3 from edge, center in y — standard edge-feed offset
src_x = px0 + L / 3.0
src_y = py0 + W / 2.0
src_z = h / 2.0    # inside substrate

sim.add_source(
    (src_x, src_y, src_z),
    component="ez",
    waveform=GaussianPulse(f0=f0, bandwidth=0.8),
)
sim.add_probe((src_x, src_y, src_z), component="ez")

# ---- Run simulation ----
nu_grid = sim._build_nonuniform_grid()
# Run ~15 ns for good Harminv ring-down
n_steps = int(np.ceil(15e-9 / nu_grid.dt))
print(f"\nRunning {n_steps} steps  (dt={nu_grid.dt * 1e12:.3f} ps) ...")
result = sim.run(n_steps=n_steps)

# ---- Resonance extraction ----
modes = result.find_resonances(
    freq_range=(f0 * 0.5, f0 * 1.5),
    probe_idx=0,
)

if modes:
    best = min(modes, key=lambda m: abs(m.freq - f0))
    f_sim = best.freq
    Q_sim = best.Q
    print(f"Harminv modes found: {len(modes)}")
else:
    # FFT fallback
    ts_arr = np.asarray(result.time_series).ravel()
    spectrum_fb = np.abs(np.fft.rfft(ts_arr, n=len(ts_arr) * 8))
    freqs_fb = np.fft.rfftfreq(len(ts_arr) * 8, d=result.dt)
    band = (freqs_fb > f0 * 0.5) & (freqs_fb < f0 * 1.5)
    f_sim = freqs_fb[np.argmax(spectrum_fb * band)]
    Q_sim = float("nan")
    print("Harminv found no modes; using FFT peak")

err_pct = abs(f_sim - f0) / f0 * 100
print(f"\nDesign frequency : {f0 / 1e9:.4f} GHz")
print(f"Simulated        : {f_sim / 1e9:.4f} GHz")
print(f"Error            : {err_pct:.2f} %")
if not np.isnan(Q_sim):
    print(f"Q factor         : {Q_sim:.1f}")

# ---- Assemble materials for geometry visualization ----
materials_nu, pec_mask_nu = sim._assemble_materials_nu(nu_grid)
eps_r_arr = np.asarray(materials_nu.eps_r)
pec_arr = np.asarray(pec_mask_nu) if pec_mask_nu is not None else np.zeros(
    (nu_grid.nx, nu_grid.ny, nu_grid.nz), dtype=bool)

# Grid index helpers
cpml = nu_grid.cpml_layers
dz_np = np.array(nu_grid.dz)
z_cumsum = np.cumsum(dz_np)
z_cumsum = np.insert(z_cumsum, 0, 0.0)
z_offset = z_cumsum[cpml]
iz_sub_mid = cpml + int(np.argmin(np.abs(
    z_cumsum[cpml:] - z_offset - h / 2.0)))
ix_ctr = nu_grid.nx // 2
iy_ctr = nu_grid.ny // 2

ts_arr = np.asarray(result.time_series)
if ts_arr.ndim == 2:
    ts_probe = ts_arr[:, 0]
else:
    ts_probe = ts_arr.ravel()

# FFT spectrum
nfft = len(ts_probe) * 8
spectrum = np.abs(np.fft.rfft(ts_probe, n=nfft))
freqs_fft = np.fft.rfftfreq(nfft, d=result.dt) / 1e9   # GHz

# ---- 6-panel figure ----
fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    f"2.4 GHz Patch Antenna on FR4  "
    f"(L={L * 1e3:.1f} mm, W={W * 1e3:.1f} mm, h={h * 1e3:.1f} mm)",
    fontsize=13, fontweight="bold",
)
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

# Panel 1: eps_r slice at z = h/2 (substrate mid-plane) — top view
ax1 = fig.add_subplot(gs[0, 0])
eps_xy = eps_r_arr[:, :, iz_sub_mid]
im1 = ax1.imshow(eps_xy.T, origin="lower", cmap="viridis", aspect="equal")
fig.colorbar(im1, ax=ax1, label="eps_r")
ax1.set_xlabel("x (cells)")
ax1.set_ylabel("y (cells)")
ax1.set_title(f"eps_r  z=h/2 (xy view)")

# Panel 2: eps_r slice at y=center — side view showing layers
ax2 = fig.add_subplot(gs[0, 1])
eps_xz = eps_r_arr[:, iy_ctr, :]
im2 = ax2.imshow(eps_xz.T, origin="lower", cmap="viridis", aspect="auto")
fig.colorbar(im2, ax=ax2, label="eps_r")
ax2.axhline(iz_sub_mid, color="white", ls="--", lw=0.8, label=f"z=h/2 (k={iz_sub_mid})")
ax2.set_xlabel("x (cells)")
ax2.set_ylabel("z (cells)")
ax2.set_title("eps_r  y=center (xz view)")
ax2.legend(fontsize=7)

# Panel 3: PEC mask slice at z = h/2 showing ground + patch outline
ax3 = fig.add_subplot(gs[0, 2])
pec_xy = pec_arr[:, :, iz_sub_mid].astype(float)
ax3.imshow(pec_xy.T, origin="lower", cmap="binary", aspect="equal")
ax3.set_xlabel("x (cells)")
ax3.set_ylabel("y (cells)")
ax3.set_title("PEC mask  z=h/2 (patch)")

# Panel 4: Time-domain probe signal
ax4 = fig.add_subplot(gs[1, 0])
t_ns = np.arange(len(ts_probe)) * result.dt * 1e9
ax4.plot(t_ns, ts_probe, lw=0.6)
ax4.set_xlabel("Time (ns)")
ax4.set_ylabel("Ez amplitude")
ax4.set_title("Probe time series")
ax4.grid(True, alpha=0.3)

# Panel 5: Frequency spectrum with resonance marked
ax5 = fig.add_subplot(gs[1, 1])
spec_db = 20 * np.log10(np.maximum(spectrum / (spectrum.max() or 1.0), 1e-10))
band_mask = (freqs_fft > f0 * 0.4 / 1e9) & (freqs_fft < f0 * 1.6 / 1e9)
ax5.plot(freqs_fft[band_mask], spec_db[band_mask], lw=1.0)
ax5.axvline(f0 / 1e9, color="g", ls="--", lw=1.5,
            label=f"Design {f0 / 1e9:.2f} GHz")
ax5.axvline(f_sim / 1e9, color="r", ls=":", lw=1.5,
            label=f"Simulated {f_sim / 1e9:.3f} GHz")
ax5.set_xlabel("Frequency (GHz)")
ax5.set_ylabel("Normalized (dB)")
ax5.set_title("Frequency spectrum")
ax5.set_ylim(-60, 5)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Panel 6: Summary annotation
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
lines = [
    "Patch Antenna Summary",
    "─" * 28,
    f"Design freq  : {f0 / 1e9:.4f} GHz",
    f"Simulated    : {f_sim / 1e9:.4f} GHz",
    f"Error        : {err_pct:.3f} %",
]
if not np.isnan(Q_sim):
    lines.append(f"Q factor     : {Q_sim:.1f}")
lines += [
    "",
    f"L = {L * 1e3:.2f} mm",
    f"W = {W * 1e3:.2f} mm",
    f"h = {h * 1e3:.1f} mm  (FR4)",
    f"eps_eff = {eps_eff:.4f}",
    "",
    f"dx = {dx * 1e3:.1f} mm",
    f"dz_sub = {dz_sub * 1e3:.3f} mm",
    f"Steps = {n_steps}",
    f"dt = {result.dt * 1e12:.3f} ps",
]
ax6.text(0.05, 0.97, "\n".join(lines), transform=ax6.transAxes,
         va="top", ha="left", fontsize=9, family="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

out_path = "examples/04_patch_antenna.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nPlot saved: {out_path}")
