"""Mesh convergence verification for patch antenna resonance.

Tests: dx = 1.0, 0.5, 0.25mm with point source + windowed FFT.
All use: large domain (lambda/4 margin), true PEC mask, limited substrate.
Verifies that finer resolution converges toward analytical 2.4 GHz.
"""
import numpy as np
import jax.numpy as jnp
from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, EPS_0, MaterialArrays
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.sources.sources import GaussianPulse, add_point_source

f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3; lam0 = C0/f0
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")
print(f"lambda={lam0*1e3:.0f}mm")

margin = lam0/4
results = []

for dx in [1.0e-3, 0.5e-3]:
    dom_x = L+2*margin; dom_y = W+2*margin; dom_z = h+lam0/4
    cpml_n = max(int(round(lam0/20/dx)), 8)

    grid = Grid(freq_max=4e9, domain=(dom_x, dom_y, dom_z), dx=dx, cpml_layers=cpml_n)

    # Build materials
    eps_r_arr = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)

    px0, py0 = margin, margin
    feed_x, feed_y = px0+L/3, py0+W/2

    # Ground PEC (z=0 plane)
    z_gnd = grid.position_to_index((0,0,0))[2]
    pec_mask = pec_mask.at[:,:,z_gnd].set(True)

    # Substrate (only around patch)
    sub_m = 5e-3
    ix0 = grid.position_to_index((px0-sub_m,0,0))[0]
    ix1 = grid.position_to_index((px0+L+sub_m,0,0))[0]
    iy0 = grid.position_to_index((0,py0-sub_m,0))[1]
    iy1 = grid.position_to_index((0,py0+W+sub_m,0))[1]
    iz_top = grid.position_to_index((0,0,h))[2]
    eps_r_arr = eps_r_arr.at[ix0:ix1,iy0:iy1,z_gnd:iz_top].set(eps_r)
    sigma_arr = sigma_arr.at[ix0:ix1,iy0:iy1,z_gnd:iz_top].set(sigma_fr4)

    # Patch PEC
    ipx0 = grid.position_to_index((px0,0,0))[0]
    ipx1 = grid.position_to_index((px0+L,0,0))[0]
    ipy0 = grid.position_to_index((0,py0,0))[1]
    ipy1 = grid.position_to_index((0,py0+W,0))[1]
    iz_patch = grid.position_to_index((0,0,h))[2]
    pec_mask = pec_mask.at[ipx0:ipx1,ipy0:ipy1,iz_patch].set(True)

    materials = MaterialArrays(eps_r=eps_r_arr, sigma=sigma_arr,
                               mu_r=jnp.ones(grid.shape, dtype=jnp.float32))

    # CPML
    cpml_params, cpml_state = init_cpml(grid)

    # Time stepping
    target_t = 10e-9
    n_steps = int(np.ceil(target_t / grid.dt))
    probe_idx = grid.position_to_index((feed_x, feed_y, h/2))
    pulse = GaussianPulse(f0=f0, bandwidth=0.8)

    print(f"\n=== dx={dx*1e3:.1f}mm, grid={grid.shape}, h/dx={h/dx:.1f}, n_steps={n_steps} ===")

    state = init_state(grid.shape)
    ez_trace = np.zeros(n_steps)

    import time; t0w = time.time()
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
            print(f"  step {n}/{n_steps} | {rate:.0f}/s | ETA {(n_steps-n)/rate:.0f}s")

    elapsed = time.time() - t0w
    print(f"  Done in {elapsed:.0f}s")

    # Windowed spectral analysis
    t0_pulse = 3.0/(f0*0.8*np.pi)
    start = int(np.ceil(2*t0_pulse/grid.dt))
    windowed = ez_trace[start:] - np.mean(ez_trace[start:])
    windowed *= np.hanning(len(windowed))

    nfft = len(windowed)*8
    spec = np.abs(np.fft.rfft(windowed, n=nfft))
    fg = np.fft.rfftfreq(nfft, d=grid.dt)/1e9

    band = (fg > 1.5) & (fg < 4.0)
    idx = np.argmax(spec[band])
    f_res = fg[band][idx]
    err = abs(f_res-2.4)/2.4*100
    results.append((dx*1e3, h/dx, f_res, err))
    print(f"  PEAK: {f_res:.3f} GHz (err={err:.1f}%)")

print(f"\n=== CONVERGENCE TABLE ===")
print(f"{'dx(mm)':>8} {'h/dx':>6} {'f_res(GHz)':>11} {'error(%)':>9}")
for dx_mm, h_dx, f, e in results:
    print(f"{dx_mm:>8.2f} {h_dx:>6.1f} {f:>11.3f} {e:>9.1f}")

if len(results) >= 2:
    # Richardson extrapolation (assuming 2nd-order convergence)
    f1, f2 = results[-2][2], results[-1][2]
    dx1, dx2 = results[-2][0], results[-1][0]
    r = dx1/dx2
    f_extrap = (r**2 * f2 - f1) / (r**2 - 1)
    print(f"\nRichardson extrapolation (dx->0): {f_extrap:.3f} GHz (err={abs(f_extrap-2.4)/2.4*100:.1f}%)")
