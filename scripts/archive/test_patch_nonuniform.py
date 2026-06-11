"""Test: patch antenna with non-uniform z mesh (bug #23 fixed).

Uses the rfx-ref ex04 proven setup: GaussianPulse, CPML-12, non-uniform z.
"""
import numpy as np
from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

f0 = 2.4e9
eps_r = 4.4
h = 1.6e-3
W = C0 / (2*f0) * np.sqrt(2.0 / (eps_r + 1.0))
eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h / W) ** (-0.5)
dL = 0.412 * h * ((eps_eff + 0.3) * (W / h + 0.264) / ((eps_eff - 0.258) * (W / h + 0.8)))
L = C0 / (2.0 * f0 * np.sqrt(eps_eff)) - 2.0 * dL

dx = 1e-3
margin = 15e-3

# Non-uniform z: 4 cells in substrate, coarse in air
dz_sub = h / 4  # 0.4mm
n_sub = 4
dz_air = dx
n_air = int(np.ceil(margin / dz_air))
dz_profile = [dz_sub] * n_sub + [dz_air] * n_air

dom_x = L + 2 * margin
dom_y = W + 2 * margin

print(f"Patch: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm")
print(f"Domain: {dom_x*1e3:.1f} x {dom_y*1e3:.1f} mm")
print(f"dz_profile: {n_sub} x {dz_sub*1e3:.2f}mm + {n_air} x {dz_air*1e3:.1f}mm")

sim = Simulation(
    freq_max=f0 * 2,
    domain=(dom_x, dom_y),
    dx=dx,
    boundary="pec",
    dz_profile=dz_profile,
    cpml_layers=0,
)

sigma_sub = 2 * np.pi * f0 * 8.854e-12 * eps_r * 0.02
sim.add_material("fr4", eps_r=eps_r, sigma=sigma_sub)

# Substrate
sim.add(Box((0, 0, 0), (dom_x, dom_y, h)), material="fr4")
# Patch (zero-thickness — tests the fix)
px0, py0 = margin, margin
sim.add(Box((px0, py0, h), (px0 + L, py0 + W, h)), material="pec")

# Source inside substrate
sim.add_source((px0 + L/3, py0 + W/2, h/2), "ez",
               waveform=GaussianPulse(f0=f0, bandwidth=0.8))
sim.add_probe((px0 + L/3, py0 + W/2, h/2), "ez")

grid = sim._build_nonuniform_grid()
n_steps = int(np.ceil(15e-9 / grid.dt))
print(f"Grid: {grid.nx}x{grid.ny}x{grid.nz}, dt={grid.dt*1e12:.2f}ps, steps={n_steps}")

result = sim.run(n_steps=n_steps)
print(f"Ran {n_steps} steps")

modes = result.find_resonances(freq_range=(f0 * 0.5, f0 * 1.5))
if modes:
    best = min(modes, key=lambda m: abs(m.freq - f0))
    err = abs(best.freq - f0) / f0 * 100
    print(f"Harminv: f={best.freq/1e9:.4f} GHz, Q={best.Q:.0f}, error={err:.2f}%")
else:
    ts = np.array(result.time_series).ravel()
    nfft = len(ts) * 8
    spec = np.abs(np.fft.rfft(ts, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=result.dt)
    band = (freqs > f0 * 0.5) & (freqs < f0 * 1.5)
    f_sim = freqs[np.argmax(spec * band)]
    err = abs(f_sim - f0) / f0 * 100
    print(f"FFT: f={f_sim/1e9:.4f} GHz, error={err:.2f}%")
