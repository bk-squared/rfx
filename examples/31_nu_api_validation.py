"""Validate non-uniform mesh via Simulation API across patch designs.

Uses the integrated Simulation(dz_profile=...) path with
auto-derived non-uniform z grids. Tests both standard and
finer resolution for designs that failed at dx=0.5mm.
"""
import numpy as np
import time
from rfx.api import Simulation
from rfx.geometry.csg import Box
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0


def design_patch(f0, eps_r, h):
    W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
    eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
    dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
    L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
    return L, W, eps_eff


def run_patch_nu(name, f0, eps_r, h, tan_d, dx):
    """Run patch antenna using Simulation API with non-uniform dz."""
    L, W, eps_eff = design_patch(f0, eps_r, h)
    sigma_sub = 2*np.pi*f0*8.854e-12*eps_r*tan_d
    lam0 = C0/f0
    margin = lam0/4

    # Non-uniform z: snap substrate with at least 4 cells
    n_sub = max(4, int(np.ceil(h/dx)))
    dz_sub = h / n_sub
    n_air = max(1, int(round(margin/dx)))
    dz_profile = np.concatenate([np.ones(n_sub)*dz_sub, np.ones(n_air)*dx])

    sim = Simulation(
        freq_max=f0*2,
        domain=(L + 2*margin, W + 2*margin, 0),  # z from dz_profile
        dx=dx, dz_profile=dz_profile, cpml_layers=12,
    )

    # Materials
    sim.add_material("substrate", eps_r=eps_r, sigma=sigma_sub)
    sim.add_material("pec", eps_r=1.0, sigma=1e10)

    # Geometry: ground plane + substrate + patch
    sim.add(Box((0, 0, 0), (L+2*margin, W+2*margin, 0)), material="pec")       # ground
    sim.add(Box((0, 0, 0), (L+2*margin, W+2*margin, h)), material="substrate")  # substrate
    sim.add(Box((margin, margin, h), (margin+L, margin+W, h)), material="pec")   # patch

    # Source: offset from edge for better mode excitation
    src_x = margin + L/3
    src_y = margin + W/2
    src_z = h/2
    sim.add_source((src_x, src_y, src_z), "ez",
                    waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((src_x, src_y, src_z), "ez")

    n_steps = int(np.ceil(10e-9 / sim._build_nonuniform_grid().dt))

    t0 = time.time()
    result = sim.run(n_steps=n_steps)
    elapsed = time.time() - t0

    # Resonance extraction
    modes = result.find_resonances(freq_range=(f0*0.5, f0*1.5))
    if modes:
        best = min(modes, key=lambda m: abs(m.freq - f0))
        err = abs(best.freq - f0)/f0*100
        return best.freq, err, best.Q, elapsed, n_sub, int(round(L/dx))
    return 0, 100, 0, elapsed, n_sub, int(round(L/dx))


# ============================================================
designs = [
    # name,            f0,       eps_r, h,         tan_d,  dx
    ("2.4GHz FR4",     2.4e9,    4.4,   1.6e-3,    0.02,   0.5e-3),
    ("5.8GHz FR4",     5.8e9,    4.4,   0.8e-3,    0.02,   0.25e-3),  # finer dx
    ("1.575GHz GPS",   1.575e9,  2.2,   3.175e-3,  0.001,  0.5e-3),
    ("3.5GHz Rogers",  3.5e9,    3.55,  1.524e-3,  0.0027, 0.5e-3),
]

print("="*70)
print("NON-UNIFORM MESH API VALIDATION")
print("Simulation(dz_profile=...) + make_current_source")
print("="*70)

results = []
for name, f0, eps_r, h, tan_d, dx in designs:
    L, W, _ = design_patch(f0, eps_r, h)
    print(f"\n--- {name} ---")
    print(f"  f0={f0/1e9:.3f}GHz, L={L*1e3:.1f}mm, dx={dx*1e3:.2f}mm")

    f_res, err, Q, elapsed, n_sub, L_dx = run_patch_nu(name, f0, eps_r, h, tan_d, dx)
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
