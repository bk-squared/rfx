"""Example 1: PEC Cavity TM110 Resonance

Simulates a rectangular PEC cavity and identifies the TM110 resonant
frequency using the high-level Simulation API and Harminv resonance
extraction.  Compares with the analytical eigenfrequency.

Expected output:
  Analytical TM110: 2.1213 GHz
  Simulated:        ~2.12 GHz (< 0.5% error)

Saves: examples/01_cavity_resonance.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rfx import Simulation
from rfx.grid import C0
from rfx.sources.sources import ModulatedGaussian

# ---- Cavity dimensions (metres) ----
a, b, d = 0.10, 0.10, 0.05

# ---- Analytical TM110 frequency ----
f_analytical = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)
print(f"Analytical TM110: {f_analytical / 1e9:.4f} GHz")

# ---- Build simulation via high-level API ----
# PEC boundary: no CPML padding, closed metallic box.
sim = Simulation(
    freq_max=5e9,
    domain=(a, b, d),
    boundary="pec",
    dx=0.001,
)

# Soft source (impedance=0) near one corner to excite TM110.
# ModulatedGaussian has zero DC content, preventing static charge
# accumulation on PEC surfaces.
src_waveform = ModulatedGaussian(f0=f_analytical, bandwidth=0.8)
sim.add_source((a / 3, b / 3, d / 2), component="ez", waveform=src_waveform)

# Probe at a spatially distinct location to record the ring-down.
sim.add_probe((2 * a / 3, 2 * b / 3, d / 2), component="ez")

# Run long enough for Harminv ring-down analysis (~80 periods at f_analytical).
n_steps = int(np.ceil(80.0 / (f_analytical * sim._build_grid().dt)))
print(f"Running {n_steps} steps ...")
result = sim.run(n_steps=n_steps)

# ---- Resonance extraction via Harminv ----
modes = result.find_resonances(
    freq_range=(f_analytical * 0.5, f_analytical * 1.5),
    probe_idx=0,
)

if modes:
    best = min(modes, key=lambda m: abs(m.freq - f_analytical))
    f_sim = best.freq
    Q_sim = best.Q
else:
    # Fall back to FFT peak if Harminv found nothing
    ts = np.asarray(result.time_series).ravel()
    spectrum = np.abs(np.fft.rfft(ts, n=len(ts) * 8))
    freqs_fft = np.fft.rfftfreq(len(ts) * 8, d=result.dt)
    mask = (freqs_fft > f_analytical * 0.5) & (freqs_fft < f_analytical * 1.5)
    f_sim = freqs_fft[np.argmax(spectrum * mask)]
    Q_sim = float("nan")

error_pct = abs(f_sim - f_analytical) / f_analytical * 100
print(f"Simulated TM110:  {f_sim / 1e9:.4f} GHz")
print(f"Error:            {error_pct:.2f}%")
if not np.isnan(Q_sim):
    print(f"Q factor:         {Q_sim:.1f}")

# ---- Build helper objects for visualization ----
grid = sim._build_grid()
state = result.state
ts_arr = np.asarray(result.time_series)
if ts_arr.ndim == 2:
    ts_probe = ts_arr[:, 0]
else:
    ts_probe = ts_arr.ravel()

# ---- FFT spectrum for panel 3 ----
nfft = len(ts_probe) * 8
spectrum = np.abs(np.fft.rfft(ts_probe, n=nfft))
freqs_fft = np.fft.rfftfreq(nfft, d=result.dt) / 1e9  # GHz

# ---- 4-panel figure ----
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("PEC Cavity TM110 Resonance", fontsize=14, fontweight="bold")

# Panel 1: Ez field slice at z = nz//2
ax = axes[0, 0]
ez_slice = np.asarray(state.ez)[:, :, grid.nz // 2]
vmax = float(np.max(np.abs(ez_slice))) or 1.0
im = ax.imshow(
    ez_slice.T, origin="lower", cmap="RdBu_r",
    vmin=-vmax, vmax=vmax, aspect="equal",
)
fig.colorbar(im, ax=ax, label="Ez (V/m)")
ax.set_xlabel("x (cells)")
ax.set_ylabel("y (cells)")
ax.set_title(f"Ez field slice  z={grid.nz // 2}")

# Panel 2: Time-domain probe signal
ax = axes[0, 1]
t_ns = np.arange(len(ts_probe)) * result.dt * 1e9
ax.plot(t_ns, ts_probe, lw=0.8)
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez amplitude")
ax.set_title("Probe time series")
ax.grid(True, alpha=0.3)

# Panel 3: Frequency spectrum with analytical line
ax = axes[1, 0]
spec_db = 20 * np.log10(np.maximum(spectrum / (spectrum.max() or 1.0), 1e-10))
band = (freqs_fft > 0.5) & (freqs_fft < 5.0)
ax.plot(freqs_fft[band], spec_db[band], lw=1.0)
ax.axvline(f_analytical / 1e9, color="r", ls="--", lw=1.5,
           label=f"Analytical {f_analytical / 1e9:.4f} GHz")
ax.axvline(f_sim / 1e9, color="g", ls=":", lw=1.5,
           label=f"Simulated {f_sim / 1e9:.4f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("Frequency spectrum")
ax.set_ylim(-60, 5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Text annotation
ax = axes[1, 1]
ax.axis("off")
lines = [
    "Cavity: {:.0f} x {:.0f} x {:.0f} mm".format(a * 1e3, b * 1e3, d * 1e3),
    "",
    f"Analytical TM110 : {f_analytical / 1e9:.4f} GHz",
    f"Simulated TM110  : {f_sim / 1e9:.4f} GHz",
    f"Frequency error  : {error_pct:.3f} %",
]
if not np.isnan(Q_sim):
    lines.append(f"Q factor         : {Q_sim:.1f}")
lines += [
    "",
    f"Grid dx : {grid.dx * 1e3:.2f} mm",
    f"Steps   : {n_steps}",
    f"dt      : {result.dt * 1e12:.2f} ps",
]
text = "\n".join(lines)
ax.text(0.05, 0.95, text, transform=ax.transAxes,
        va="top", ha="left", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
ax.set_title("Simulation summary")

plt.tight_layout()
out_path = "examples/01_cavity_resonance.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Plot saved: {out_path}")
