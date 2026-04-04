"""Patch antenna resonance detection using point source (no port impedance).

This uses the low-level FDTD API to avoid port impedance loading
that masks the cavity resonance.
"""
import numpy as np
import jax.numpy as jnp
from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MaterialArrays
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.sources.sources import GaussianPulse, add_point_source

# ---- Design ----
f0 = 2.4e9
eps_r = 4.4; h = 1.6e-3
lam0 = C0 / f0
W = C0 / (2*f0) * np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0 / (2*f0*np.sqrt(eps_eff)) - 2*dL

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")

# ---- Grid ----
dx = 0.5e-3
margin = lam0 / 4  # ~31mm
dom_x = L + 2*margin
dom_y = W + 2*margin
dom_z = h + lam0 / 4
cpml_n = max(int(round(lam0/20/dx)), 8)

grid = Grid(freq_max=4e9, domain=(dom_x, dom_y, dom_z), dx=dx, cpml_layers=cpml_n)
print(f"Grid: {grid.shape}, dx={dx*1e3}mm, cpml={cpml_n}, dt={grid.dt:.3e}")

# ---- Materials + PEC mask (manual) ----
eps_r_arr = jnp.ones(grid.shape, dtype=jnp.float32)
sigma_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
mu_r_arr = jnp.ones(grid.shape, dtype=jnp.float32)
pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)

sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
px0, py0 = margin, margin

# Ground PEC (1 cell at z=0)
z_ground = grid.position_to_index((0, 0, 0))[2]
pec_mask = pec_mask.at[:, :, z_ground].set(True)

# Substrate FR4 (only under patch + 5mm)
sub_m = 5e-3
ix0 = grid.position_to_index((px0-sub_m, 0, 0))[0]
ix1 = grid.position_to_index((px0+L+sub_m, 0, 0))[0]
iy0 = grid.position_to_index((0, py0-sub_m, 0))[1]
iy1 = grid.position_to_index((0, py0+W+sub_m, 0))[1]
iz_sub_top = grid.position_to_index((0, 0, h))[2]
eps_r_arr = eps_r_arr.at[ix0:ix1, iy0:iy1, z_ground:iz_sub_top].set(eps_r)
sigma_arr = sigma_arr.at[ix0:ix1, iy0:iy1, z_ground:iz_sub_top].set(sigma_fr4)

# Patch PEC (1 cell at z=h)
ipx0 = grid.position_to_index((px0, 0, 0))[0]
ipx1 = grid.position_to_index((px0+L, 0, 0))[0]
ipy0 = grid.position_to_index((0, py0, 0))[1]
ipy1 = grid.position_to_index((0, py0+W, 0))[1]
iz_patch = grid.position_to_index((0, 0, h))[2]
pec_mask = pec_mask.at[ipx0:ipx1, ipy0:ipy1, iz_patch].set(True)

materials = MaterialArrays(eps_r=eps_r_arr, sigma=sigma_arr, mu_r=mu_r_arr)

n_pec = int(jnp.sum(pec_mask))
print(f"PEC cells: {n_pec}, substrate z: {z_ground} to {iz_sub_top}, patch z: {iz_patch}")

# ---- CPML ----
cpml_params, cpml_state = init_cpml(grid)

# ---- Source and probe ----
feed_x = px0 + L/3
feed_y = py0 + W/2
probe_idx = grid.position_to_index((feed_x, feed_y, h/2))
source_pos = (feed_x, feed_y, h/2)
pulse = GaussianPulse(f0=f0, bandwidth=0.8)

# ---- Time stepping ----
target_t = 10e-9
n_steps = int(np.ceil(target_t / grid.dt))
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")
print(f"Running...")

state = init_state(grid.shape)
ez_trace = np.zeros(n_steps)

import time
t0_wall = time.time()
for n in range(n_steps):
    t = n * grid.dt
    state = update_h(state, materials, grid.dt, grid.dx)
    state, cpml_state = apply_cpml_h(state, cpml_params, cpml_state, grid, "xyz")
    state = update_e(state, materials, grid.dt, grid.dx)
    state, cpml_state = apply_cpml_e(state, cpml_params, cpml_state, grid, "xyz")
    state = apply_pec(state)
    state = apply_pec_mask(state, pec_mask)
    state = add_point_source(state, grid, source_pos, 'ez', pulse(t))
    ez_trace[n] = float(state.ez[probe_idx])

    if n > 0 and n % 1000 == 0:
        rate = n / (time.time() - t0_wall)
        eta = (n_steps - n) / rate
        print(f"  step {n}/{n_steps} | {rate:.0f} steps/s | ETA {eta:.0f}s | max|Ez|={np.max(np.abs(ez_trace[:n+1])):.3e}")

elapsed = time.time() - t0_wall
print(f"Done in {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s)")

# ---- Spectral analysis ----
# Window: use only ring-down portion (after source decays) and remove DC
t0_pulse = 3.0 / (f0 * 0.8 * np.pi)
start_idx = int(np.ceil(2.0 * t0_pulse / grid.dt))  # start after 2*t0
windowed = ez_trace[start_idx:] - np.mean(ez_trace[start_idx:])  # remove DC offset
# Apply Hann window to suppress spectral leakage
windowed *= np.hanning(len(windowed))

nfft = len(windowed) * 8
spec = np.abs(np.fft.rfft(windowed, n=nfft))
fg = np.fft.rfftfreq(nfft, d=grid.dt) / 1e9
print(f"Analysis window: step {start_idx} to {n_steps} ({len(windowed)} samples)")

band = (fg > 1.5) & (fg < 4.0)
idx_peak = np.argmax(spec[band])
f_res = fg[band][idx_peak]
err = abs(f_res - f0/1e9) / (f0/1e9) * 100

print(f"\n=== Results ===")
print(f"Resonance PEAK: {f_res:.3f} GHz (design: {f0/1e9:.3f} GHz)")
print(f"Error: {err:.1f}%")

# All peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(spec[band], height=spec[band].max()*0.02, distance=10)
sorted_p = sorted(peaks, key=lambda p: spec[band][p], reverse=True)
print(f"Top spectral peaks (1-4 GHz):")
for i, p in enumerate(sorted_p[:5]):
    print(f"  {i+1}. {fg[band][p]:.3f} GHz (amp={spec[band][p]:.2e})")
