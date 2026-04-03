"""Generalized validation: multiple frequencies and substrate designs.

Tests rfx non-uniform Yee + CPML on patch antennas at different
frequencies and substrates to confirm <1% is not design-specific.
"""
import numpy as np
import jax.numpy as jnp
import jax
import time
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform
from rfx.core.yee import MaterialArrays, EPS_0
from rfx.sources.sources import GaussianPulse
from rfx.harminv import harminv
from rfx.grid import C0


def design_patch(f0, eps_r, h, tan_d=0.02):
    """Analytical patch antenna dimensions."""
    W = C0 / (2*f0) * np.sqrt(2/(eps_r+1))
    eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
    dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
    L = C0 / (2*f0*np.sqrt(eps_eff)) - 2*dL
    sigma = 2*np.pi*f0*8.854e-12*eps_r*tan_d
    return L, W, eps_eff, sigma


def simulate_patch(f0, eps_r, h, tan_d=0.02, dx=0.5e-3):
    """Run patch antenna simulation with non-uniform Yee + CPML."""
    L, W, eps_eff, sigma_sub = design_patch(f0, eps_r, h, tan_d)
    lam0 = C0/f0
    margin = lam0/4
    cpml = 12

    # Choose dz that snaps to h
    n_sub = max(2, int(round(h / dx)))
    if abs(n_sub * dx - h) / h > 0.01:
        # dx doesn't divide h well — find better dz
        for n in range(2, 20):
            dz_try = h / n
            if dz_try <= dx:
                n_sub = n
                break
    dz_sub = h / n_sub

    dz_profile = np.concatenate([
        np.ones(n_sub) * dz_sub,
        np.ones(max(1, int(round(margin / dx)))) * dx,
    ])

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

    materials = MaterialArrays(
        eps_r=eps_arr, sigma=sigma_arr,
        mu_r=jnp.ones(shape, dtype=jnp.float32))

    fi = px0 + int(round(L/3/dx))
    fj = py0 + int(round(W/2/dx))
    fk = cpml + n_sub // 2

    pulse = GaussianPulse(f0=f0, bandwidth=0.8)
    n_steps = int(np.ceil(10e-9 / grid.dt))
    times = jnp.arange(n_steps, dtype=jnp.float32) * grid.dt
    wf = jax.vmap(pulse)(times)

    t0 = time.time()
    r = run_nonuniform(grid, materials, n_steps, pec_mask=pec_mask,
                       sources=[(fi, fj, fk, 'ez', np.array(wf))],
                       probes=[(fi, fj, fk, 'ez')])
    elapsed = time.time() - t0

    ts = np.array(r['time_series']).ravel()
    t0p = 3/(f0*0.8*np.pi)
    start = int(2*t0p/grid.dt)
    w = ts[start:] - np.mean(ts[start:])

    # Try direct Harminv, then bandpass
    modes = harminv(w, grid.dt, f0*0.5, f0*1.5)
    if not modes:
        fft_data = np.fft.rfft(w)
        fft_freqs = np.fft.rfftfreq(len(w), d=grid.dt)
        bp = (fft_freqs >= f0*0.5) & (fft_freqs <= f0*1.5)
        w_bp = np.fft.irfft(fft_data*bp, n=len(w))
        modes = harminv(w_bp[:3000], grid.dt, f0*0.5, f0*1.5)

    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f0))
        err = abs(best.freq - f0) / f0 * 100
        return best.freq, err, best.Q, elapsed, n_sub, dz_sub
    return 0, 100, 0, elapsed, n_sub, dz_sub


# ============================================================
# Test suite: multiple designs
# ============================================================
designs = [
    # (name, f0_GHz, eps_r, h_mm, tan_d)
    ("2.4GHz FR4",     2.4e9, 4.4,  1.6e-3, 0.02),
    ("5.8GHz FR4",     5.8e9, 4.4,  0.8e-3, 0.02),
    ("1.575GHz GPS",   1.575e9, 2.2, 3.175e-3, 0.001),
    ("3.5GHz Rogers",  3.5e9, 3.55, 1.524e-3, 0.0027),
]

print("=" * 70)
print("GENERALIZED PATCH ANTENNA VALIDATION")
print("Non-uniform Yee + CPML, raw source, Harminv")
print("=" * 70)

results = []
for name, f0, eps_r, h, tan_d in designs:
    L, W, _, _ = design_patch(f0, eps_r, h, tan_d)
    print(f"\n--- {name} ---")
    print(f"  f0={f0/1e9:.3f}GHz, eps_r={eps_r}, h={h*1e3:.3f}mm, L={L*1e3:.1f}mm")

    f_res, err, Q, elapsed, n_sub, dz = simulate_patch(f0, eps_r, h, tan_d)
    results.append((name, f0, f_res, err, Q, elapsed, n_sub))

    if f_res > 0:
        print(f"  Result: {f_res/1e9:.4f} GHz, err={err:.2f}%, Q={Q:.0f}, {elapsed:.1f}s (h/dz={n_sub})")
    else:
        print(f"  No modes detected ({elapsed:.1f}s)")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Design':>18} {'f0(GHz)':>8} {'f_res':>8} {'err%':>6} {'Q':>6} {'time':>6} {'h/dz':>5}")
for name, f0, f_res, err, Q, t, n_sub in results:
    print(f"{name:>18} {f0/1e9:>8.3f} {f_res/1e9:>8.4f} {err:>6.2f} {Q:>6.0f} {t:>5.1f}s {n_sub:>5}")

n_pass = sum(1 for _, _, _, err, _, _, _ in results if err < 1.0)
print(f"\n<1% achieved: {n_pass}/{len(results)} designs")
