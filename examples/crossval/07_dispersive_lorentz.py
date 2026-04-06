"""Cross-validation: Lorentz dispersive material response.

Structure: Lorentz medium with known resonance, pulse excitation
Comparison: Transmission dip at resonance frequency matches analytical
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.materials.lorentz import LorentzPole
from rfx.grid import C0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Lorentz medium: eps(w) = eps_inf + De * w0^2 / (w0^2 - w^2 + j*delta*w)
eps_inf = 2.25  # high-freq permittivity (glass-like)
De = 1.0        # oscillator strength
f0_lorentz = 10e9  # resonance at 10 GHz
delta = 1e9     # damping (1 GHz linewidth)

print("=" * 60)
print("Cross-Validation: Lorentz Dispersive Material")
print("=" * 60)
print(f"eps_inf = {eps_inf}, De = {De}")
print(f"Resonance: {f0_lorentz/1e9:.1f} GHz, damping: {delta/1e9:.1f} GHz")
print()

dx = 1.0e-3  # 1 mm
domain_x = 0.15  # 150 mm
domain_yz = 3 * dx

sim = Simulation(
    freq_max=20e9,
    domain=(domain_x, domain_yz, domain_yz),
    boundary="cpml",
    cpml_layers=10,
    dx=dx,
)

# Lorentz slab in the middle
omega_0 = 2 * np.pi * f0_lorentz
kappa = De * omega_0 ** 2  # kappa = delta_eps * omega_0^2
sim.add_material("lorentz_medium", eps_r=eps_inf,
                 lorentz_poles=[LorentzPole(
                     omega_0=omega_0,
                     delta=delta,
                     kappa=kappa,
                 )])
slab_lo = domain_x * 0.35
slab_hi = domain_x * 0.65
sim.add(Box((slab_lo, 0, 0), (slab_hi, domain_yz, domain_yz)),
        material="lorentz_medium")

# Broadband source before slab
sim.add_source((domain_x * 0.15, domain_yz / 2, domain_yz / 2), "ez",
               waveform=GaussianPulse(f0=10e9, bandwidth=0.8))

# Probe after slab (transmitted)
sim.add_probe((domain_x * 0.85, domain_yz / 2, domain_yz / 2), "ez")
# Reference probe before slab
sim.add_probe((domain_x * 0.25, domain_yz / 2, domain_yz / 2), "ez")

grid = sim._build_grid()
n_steps = int(np.ceil(10e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, steps={n_steps}")

result = sim.run(n_steps=n_steps)

ts = np.array(result.time_series)
ts_trans = ts[:, 0] if ts.ndim == 2 else ts.ravel()
ts_ref = ts[:, 1] if ts.ndim == 2 and ts.shape[1] >= 2 else ts_trans

# FFT for transmission spectrum
nfft = len(ts_trans) * 4
spec_trans = np.abs(np.fft.rfft(ts_trans, n=nfft))
spec_ref = np.abs(np.fft.rfft(ts_ref, n=nfft))
freqs_hz = np.fft.rfftfreq(nfft, d=result.dt)
freqs_ghz = freqs_hz / 1e9

T = spec_trans / (spec_ref + 1e-30)
band = (freqs_ghz > 2) & (freqs_ghz < 18)

# Find transmission minimum (should be near resonance)
T_band = T[band]
f_band = freqs_ghz[band]
min_idx = np.argmin(T_band)
f_min = f_band[min_idx]
T_min = T_band[min_idx]

print(f"\nTransmission minimum: {T_min:.4f} at {f_min:.2f} GHz")
print(f"Expected resonance: {f0_lorentz/1e9:.1f} GHz")
f_err = abs(f_min - f0_lorentz / 1e9) / (f0_lorentz / 1e9) * 100
print(f"Frequency error: {f_err:.1f}%")

if f_err < 10 and T_min < 0.5:
    print("PASS: absorption dip at correct resonance")
else:
    print(f"FAIL: dip at {f_min:.1f} GHz (expect {f0_lorentz/1e9:.1f}), T_min={T_min:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Lorentz Dispersive Material Transmission", fontsize=13)

ax = axes[0]
t_ns = np.arange(len(ts_trans)) * result.dt * 1e9
ax.plot(t_ns, ts_trans, "b-", lw=0.8, label="Transmitted")
ax.plot(t_ns, ts_ref, "r--", lw=0.8, alpha=0.5, label="Reference")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Ez")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title("Time Domain")

ax = axes[1]
ax.plot(f_band, 20 * np.log10(T_band + 1e-30), "b-", lw=1.5)
ax.axvline(f0_lorentz / 1e9, color="r", ls="--", alpha=0.5,
           label=f"Resonance {f0_lorentz/1e9:.0f} GHz")
ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Transmission (dB)")
ax.set_title("Transmission Spectrum")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-30, 5)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "07_dispersive_lorentz.png")
plt.savefig(out, dpi=150)
plt.close(fig)
print(f"Plot saved: {out}")
