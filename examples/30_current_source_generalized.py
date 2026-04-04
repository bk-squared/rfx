"""Generalized <1% test with proper current source normalization.

Uses make_current_source (Taflove/Meep-style: E += dt/eps × I/dV)
for resolution-independent power injection. Tests all 4 patch designs.
"""
import numpy as np
import jax.numpy as jnp
import jax
import time
from rfx.nonuniform import (
    make_nonuniform_grid, run_nonuniform, make_current_source,
)
from rfx.core.yee import MaterialArrays, EPS_0
from rfx.sources.sources import GaussianPulse
from rfx.harminv import harminv
from rfx.grid import C0


def design_patch(f0, eps_r, h):
    W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
    eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
    dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
    L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
    return L, W, eps_eff


def simulate(f0, eps_r, h, tan_d, dx):
    L, W, eps_eff = design_patch(f0, eps_r, h)
    sigma_sub = 2*np.pi*f0*8.854e-12*eps_r*tan_d
    lam0 = C0/f0
    margin = lam0/4
    cpml = 12

    # Non-uniform z: snap h with at least 4 cells
    n_sub = max(4, int(np.ceil(h/dx)))
    dz_sub = h / n_sub  # exact snap
    dz_air = np.ones(max(1, int(round(margin/dx)))) * dx
    dz_profile = np.concatenate([np.ones(n_sub)*dz_sub, dz_air])

    grid = make_nonuniform_grid((L+2*margin, W+2*margin), dz_profile, dx, cpml)
    shape = (grid.nx, grid.ny, grid.nz)

    eps_arr = jnp.ones(shape, dtype=jnp.float32)
    sigma_arr = jnp.zeros(shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(shape, dtype=jnp.bool_)

    pec_mask = pec_mask.at[cpml:-cpml, cpml:-cpml, cpml].set(True)
    iz_top = cpml + n_sub
    eps_arr = eps_arr.at[cpml:-cpml, cpml:-cpml, cpml:iz_top].set(eps_r)
    sigma_arr = sigma_arr.at[cpml:-cpml, cpml:-cpml, cpml:iz_top].set(sigma_sub)

    px0 = int(round(margin/dx)) + cpml
    py0 = px0
    px1 = px0 + int(round(L/dx))
    py1 = py0 + int(round(W/dx))
    pec_mask = pec_mask.at[px0:px1, py0:py1, iz_top].set(True)

    materials = MaterialArrays(eps_r=eps_arr, sigma=sigma_arr,
                               mu_r=jnp.ones(shape, dtype=jnp.float32))

    fi = px0 + int(round(L/3/dx))
    fj = py0 + int(round(W/2/dx))
    fk = cpml + n_sub // 2

    n_steps = int(np.ceil(10e-9 / grid.dt))
    pulse = GaussianPulse(f0=f0, bandwidth=0.8)

    # Current source: properly normalized by cell volume
    source = make_current_source(grid, (fi, fj, fk), 'ez',
                                  pulse, n_steps, materials)
    probes = [(fi, fj, fk, 'ez')]

    t0 = time.time()
    r = run_nonuniform(grid, materials, n_steps, pec_mask=pec_mask,
                       sources=[source], probes=probes)
    elapsed = time.time() - t0

    ts = np.array(r['time_series']).ravel()
    t0p = 3/(f0*0.8*np.pi)
    start = int(2*t0p/grid.dt)
    w = ts[start:] - np.mean(ts[start:])

    # Try direct Harminv
    modes = harminv(w, grid.dt, f0*0.5, f0*1.5)
    if not modes:
        # Bandpass fallback
        fft_data = np.fft.rfft(w)
        fft_freqs = np.fft.rfftfreq(len(w), d=grid.dt)
        bp = (fft_freqs >= f0*0.5) & (fft_freqs <= f0*1.5)
        w_bp = np.fft.irfft(fft_data*bp, n=len(w))
        modes = harminv(w_bp[:3000], grid.dt, f0*0.5, f0*1.5)

    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f0))
        err = abs(best.freq - f0)/f0*100
        return best.freq, err, best.Q, elapsed, n_sub, int(round(L/dx))
    return 0, 100, 0, elapsed, n_sub, int(round(L/dx))


# ============================================================
designs = [
    ("2.4GHz FR4",    2.4e9,   4.4,  1.6e-3,   0.02,  0.5e-3),
    ("5.8GHz FR4",    5.8e9,   4.4,  0.8e-3,   0.02,  0.5e-3),
    ("1.575GHz GPS",  1.575e9, 2.2,  3.175e-3, 0.001, 0.5e-3),
    ("3.5GHz Rogers", 3.5e9,   3.55, 1.524e-3, 0.0027, 0.5e-3),
]

print("="*70)
print("CURRENT SOURCE GENERALIZED VALIDATION")
print("make_current_source: E += (dt/eps) * I / dV")
print("="*70)

results = []
for name, f0, eps_r, h, tan_d, dx in designs:
    L, W, _ = design_patch(f0, eps_r, h)
    print(f"\n--- {name} ---")
    print(f"  f0={f0/1e9:.3f}GHz, L={L*1e3:.1f}mm, dx={dx*1e3:.1f}mm")

    f_res, err, Q, elapsed, n_sub, L_dx = simulate(f0, eps_r, h, tan_d, dx)
    results.append((name, f0, f_res, err, Q, elapsed, n_sub, L_dx))

    if f_res > 0:
        print(f"  Result: {f_res/1e9:.4f} GHz, err={err:.2f}%, Q={Q:.0f}, "
              f"{elapsed:.1f}s (h/dz={n_sub}, L/dx={L_dx})")
    else:
        print(f"  No modes ({elapsed:.1f}s)")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Design':>18} {'f0':>6} {'f_res':>7} {'err%':>6} {'Q':>5} {'h/dz':>5} {'L/dx':>5} {'time':>5}")
for name, f0, f_res, err, Q, t, n_sub, L_dx in results:
    print(f"{name:>18} {f0/1e9:>6.3f} {f_res/1e9:>7.4f} {err:>6.2f} {Q:>5.0f} {n_sub:>5} {L_dx:>5} {t:>4.0f}s")

n_pass = sum(1 for r in results if r[3] < 1.0)
print(f"\n<1% achieved: {n_pass}/{len(results)} designs")
