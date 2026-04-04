"""dx=0.25mm patch antenna with substrate edge taper for surface wave absorption.

The key insight: FR4 substrate on ground plane supports surface waves
that standard CPML can't absorb efficiently. Solution: add a lossy
taper zone at the substrate edges that gradually absorbs surface waves
before they reach the CPML boundary.
"""
import numpy as np
import jax.numpy as jnp
import time
from rfx import Simulation, Box, GaussianPulse
from rfx.harminv import harminv
from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MaterialArrays
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.sources.sources import GaussianPulse as GP, add_point_source

f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3; lam0 = C0/f0
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")

# Grid
dx = 0.25e-3
margin = lam0/3  # ~42mm
dom_x = L+2*margin; dom_y = W+2*margin; dom_z = h+margin
cpml_n = max(int(round(8e-3/dx)), 12)

grid = Grid(freq_max=4e9, domain=(dom_x, dom_y, dom_z), dx=dx, cpml_layers=cpml_n)
print(f"Grid: {grid.shape}, dx={dx*1e3}mm, cpml={cpml_n}")
print(f"h/dx={h/dx:.0f}, total cells={grid.nx*grid.ny*grid.nz/1e6:.1f}M")

# Build materials with substrate taper
eps_r_arr = jnp.ones(grid.shape, dtype=jnp.float32)
sigma_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)

px0, py0 = margin, margin
cpml = cpml_n

# Ground PEC at z=0
z_gnd = cpml  # physical z=0
pec_mask = pec_mask.at[:, :, z_gnd].set(True)

# Full-domain substrate with lossy taper at edges
iz_top = cpml + int(round(h/dx))
# Core substrate (under and around patch)
eps_r_arr = eps_r_arr.at[cpml:-cpml, cpml:-cpml, z_gnd:iz_top].set(eps_r)
sigma_arr = sigma_arr.at[cpml:-cpml, cpml:-cpml, z_gnd:iz_top].set(sigma_fr4)

# Lossy taper: gradually increase sigma at substrate edges
# This absorbs surface waves before they reach CPML
taper_cells = int(round(15e-3 / dx))  # 15mm taper zone
sigma_max = 2.0  # S/m — strong enough to damp surface waves

for axis in [0, 1]:  # x and y edges
    n = grid.shape[axis]
    for side in ['lo', 'hi']:
        if side == 'lo':
            ramp = jnp.linspace(sigma_max, 0, taper_cells)
            sl_start, sl_end = cpml, cpml + taper_cells
        else:
            ramp = jnp.linspace(0, sigma_max, taper_cells)
            sl_start, sl_end = n - cpml - taper_cells, n - cpml

        for k in range(z_gnd, iz_top):
            if axis == 0:
                sigma_arr = sigma_arr.at[sl_start:sl_end, cpml:-cpml, k].set(
                    jnp.maximum(sigma_arr[sl_start:sl_end, cpml:-cpml, k],
                                ramp[:, None] * jnp.ones(grid.ny - 2*cpml)[None, :]))
            else:
                sigma_arr = sigma_arr.at[cpml:-cpml, sl_start:sl_end, k].set(
                    jnp.maximum(sigma_arr[cpml:-cpml, sl_start:sl_end, k],
                                jnp.ones(grid.nx - 2*cpml)[:, None] * ramp[None, :]))

# Patch PEC
ipx0 = cpml + int(round(px0/dx))
ipx1 = cpml + int(round((px0+L)/dx))
ipy0 = cpml + int(round(py0/dx))
ipy1 = cpml + int(round((py0+W)/dx))
iz_patch = cpml + int(round(h/dx))
pec_mask = pec_mask.at[ipx0:ipx1, ipy0:ipy1, iz_patch].set(True)

materials = MaterialArrays(eps_r=eps_r_arr, sigma=sigma_arr,
                           mu_r=jnp.ones(grid.shape, dtype=jnp.float32))

n_pec = int(jnp.sum(pec_mask))
n_lossy = int(jnp.sum(sigma_arr > sigma_fr4 * 2))
print(f"PEC cells: {n_pec}, lossy taper cells: {n_lossy}")

# CPML
cpml_params, cpml_state = init_cpml(grid)

# Source + probe
feed_x, feed_y = px0+L/3, py0+W/2
probe_idx = grid.position_to_index((feed_x, feed_y, h/2))
pulse = GP(f0=f0, bandwidth=0.8)

target_t = 10e-9
n_steps = int(np.ceil(target_t / grid.dt))
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")
print("Running...")

state = init_state(grid.shape)
ez_trace = np.zeros(n_steps)
t0w = time.time()

for n in range(n_steps):
    t = n * grid.dt
    state = update_h(state, materials, grid.dt, grid.dx)
    state, cpml_state = apply_cpml_h(state, cpml_params, cpml_state, grid, "xyz")
    state = update_e(state, materials, grid.dt, grid.dx)
    state, cpml_state = apply_cpml_e(state, cpml_params, cpml_state, grid, "xyz")
    state = apply_pec(state)
    state = apply_pec_mask(state, pec_mask)
    state = add_point_source(state, grid, (feed_x, feed_y, h/2), 'ez', pulse(t))
    ez_trace[n] = float(state.ez[probe_idx])
    if n > 0 and n % 2000 == 0:
        rate = n/(time.time()-t0w)
        print(f"  step {n}/{n_steps} | {rate:.0f}/s | ETA {(n_steps-n)/rate:.0f}s | max|Ez|={np.max(np.abs(ez_trace[:n+1])):.3e}")

elapsed = time.time() - t0w
print(f"Done in {elapsed:.0f}s ({n_steps/elapsed:.0f} steps/s)")

# Analysis
t0_pulse = 3/(f0*0.8*np.pi)
start = int(2*t0_pulse/grid.dt)
windowed = ez_trace[start:] - np.mean(ez_trace[start:])

# Harminv (subsample)
max_h = 3000
step_ds = max(len(windowed)//max_h, 1)
w_ds = windowed[::step_ds][:max_h]
modes = harminv(w_ds, grid.dt*step_ds, 1.5e9, 3.5e9)

print(f"\nHarminv ({len(w_ds)} samples):")
for m in modes[:3]:
    print(f"  f={m.freq/1e9:.4f} GHz (err={abs(m.freq-f0)/f0*100:.2f}%), Q={m.Q:.0f}")

# FFT
wh = windowed * np.hanning(len(windowed))
nfft = len(wh)*8
spec = np.abs(np.fft.rfft(wh, n=nfft))
fg = np.fft.rfftfreq(nfft, d=grid.dt)/1e9
band = (fg>1.5)&(fg<3.5)
f_fft = fg[band][np.argmax(spec[band])]
print(f"FFT: {f_fft:.4f} GHz (err={abs(f_fft-2.4)/2.4*100:.2f}%)")

if modes:
    print(f"\n=== RESULT: {modes[0].freq/1e9:.4f} GHz, err={abs(modes[0].freq-f0)/f0*100:.2f}% ===")
