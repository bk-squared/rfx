"""Patch antenna: JIT path + profiling + error analysis.

Tests:
1. JIT-compiled path (Simulation.run) vs Python loop — speed comparison
2. Harminv resonance extraction from JIT-compiled run
3. Error source analysis: resolution, geometry, CPML
"""
import time
import numpy as np
import jax
import jax.numpy as jnp

from rfx import Simulation, Box, GaussianPulse
from rfx.simulation import SnapshotSpec
from rfx.harminv import harminv
from rfx.grid import C0

f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3
lam0 = C0 / f0
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f_design={f0/1e9:.3f}GHz")
print(f"Backend: {jax.default_backend()}")

margin = lam0 / 4
sub_margin = 5e-3

def make_sim(dx):
    """Create patch antenna simulation at given resolution."""
    dom_x = L + 2*margin
    dom_y = W + 2*margin
    dom_z = h + margin
    cpml_n = 12

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z),
                     boundary='cpml', cpml_layers=cpml_n, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)

    px0, py0 = margin, margin
    # Ground PEC + substrate + patch PEC
    sim.add(Box((0, 0, 0), (dom_x, dom_y, dx)), material='pec')
    sim.add(Box((px0-sub_margin, py0-sub_margin, 0),
                (px0+L+sub_margin, py0+W+sub_margin, h)), material='FR4')
    sim.add(Box((px0, py0, h), (px0+L, py0+W, h+dx)), material='pec')

    # Pure soft source (no port impedance loading)
    feed_x, feed_y = px0+L/3, py0+W/2
    sim.add_source(position=(feed_x, feed_y, h/2), component='ez',
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((feed_x, feed_y, h/2), 'ez')
    return sim

# ========================================================
# Test 1: JIT-compiled run at dx=0.5mm with profiling
# ========================================================
print("\n" + "="*60)
print("TEST 1: JIT-compiled run (dx=0.5mm)")
print("="*60)

dx = 0.5e-3
sim = make_sim(dx)
grid = sim._build_grid()
target_t = 10e-9
n_steps = int(np.ceil(target_t / grid.dt))

print(f"Grid: {grid.shape}, n_steps={n_steps}, dt={grid.dt:.3e}")
total_cells = grid.nx * grid.ny * grid.nz
print(f"Total cells: {total_cells/1e6:.2f}M")

# Warmup JIT
t0 = time.time()
result = sim.run(n_steps=100)
t_warmup = time.time() - t0
print(f"JIT warmup (100 steps): {t_warmup:.1f}s")

# Full run
t0 = time.time()
result = sim.run(n_steps=n_steps)
t_full = time.time() - t0
steps_per_sec = n_steps / t_full
mcells_per_sec = total_cells * n_steps / t_full / 1e6

print(f"Full run ({n_steps} steps): {t_full:.1f}s")
print(f"Performance: {steps_per_sec:.0f} steps/s, {mcells_per_sec:.0f} Mcells/s")

# Harminv on JIT result
ts = np.array(result.time_series).ravel()
t0_pulse = 3/(f0*0.8*np.pi)
start = int(2*t0_pulse/grid.dt)
windowed = ts[start:] - np.mean(ts[start:])
modes = harminv(windowed, grid.dt, 1.5e9, 3.5e9)

print(f"\nHarminv modes:")
for m in modes[:3]:
    err = abs(m.freq - f0)/f0*100
    print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

# FFT cross-check
windowed_h = windowed * np.hanning(len(windowed))
nfft = len(windowed_h)*8
spec = np.abs(np.fft.rfft(windowed_h, n=nfft))
fg = np.fft.rfftfreq(nfft, d=grid.dt)/1e9
band = (fg > 1.5) & (fg < 3.5)
f_fft = fg[band][np.argmax(spec[band])]
print(f"FFT peak: {f_fft:.4f} GHz (err={abs(f_fft-2.4)/2.4*100:.2f}%)")

# ========================================================
# Test 2: Resolution convergence (dx=0.5 vs lower if time)
# ========================================================
print("\n" + "="*60)
print("ERROR SOURCE ANALYSIS")
print("="*60)

# The remaining ~3% error could come from:
# 1. Grid resolution (h/dx = 3.2 cells)
# 2. Finite ground plane / substrate extent
# 3. CPML absorption at oblique angles
# 4. Grid dispersion
# 5. PEC staircasing at patch edges

h_dx = h / dx
print(f"Substrate resolution: h/dx = {h_dx:.1f} cells")
print(f"Patch length in cells: L/dx = {L/dx:.0f}")
print(f"Margin/lambda: {margin/lam0:.2f}")

# Grid dispersion estimate at 2.4 GHz
kd = 2*np.pi*f0*dx/C0  # k*dx (dimensionless)
dispersion_err = (kd**2) / 24 * 100  # % error from numerical dispersion
print(f"Grid dispersion error: ~{dispersion_err:.3f}% (negligible)")

# Substrate discretization error (h resolves to int cells)
h_actual = round(h/dx) * dx
print(f"Substrate: h_design={h*1e3:.3f}mm, h_grid={h_actual*1e3:.3f}mm, "
      f"delta={abs(h-h_actual)/h*100:.1f}%")

# Edge extension discretization
dL_actual = round(dL/dx) * dx
print(f"Edge ext: dL_design={dL*1e3:.3f}mm, dL_grid={dL_actual*1e3:.3f}mm")

# Effective L in grid
L_grid = round(L/dx) * dx
L_eff_grid = L_grid + 2*dL_actual
f_grid = C0 / (2*L_eff_grid*np.sqrt(eps_eff))
print(f"f from grid dimensions: {f_grid/1e9:.4f} GHz (err={abs(f_grid-f0)/f0*100:.2f}%)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if modes:
    f_best = modes[0].freq
    err_best = abs(f_best - f0)/f0*100
    print(f"Best Harminv: {f_best/1e9:.4f} GHz, err={err_best:.2f}%, Q={modes[0].Q:.0f}")
print(f"Performance: {mcells_per_sec:.0f} Mcells/s")
print(f"Grid dispersion: {dispersion_err:.3f}%")
print(f"Discretization shift: f_grid={f_grid/1e9:.4f} GHz ({abs(f_grid-f0)/f0*100:.2f}%)")
