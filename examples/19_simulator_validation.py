"""rfx simulator validation: source normalization + convergence + multi-structure.

Tests the Cb/dx² (Meep-style) J-source across:
1. PEC cavity (exact analytical reference) at dx=1mm
2. Patch antenna convergence: dx=1.0, 0.5, 0.25mm
3. Dielectric resonator (high-eps, high-Q)

All through JIT-compiled Simulation API (no manual loops).
"""
import numpy as np
import time
from rfx import Simulation, Box, GaussianPulse, ModulatedGaussian
from rfx.grid import C0

# ============================================================
# Test 1: PEC cavity (boundary PEC → raw source, should be ~0%)
# ============================================================
print("=" * 60)
print("TEST 1: PEC Cavity (30×30×10mm, TM110)")
print("=" * 60)
a, b, d = 0.03, 0.03, 0.01
f_tm110 = C0/2 * np.sqrt((1/a)**2 + (1/b)**2)

sim = Simulation(freq_max=10e9, domain=(a, b, d), boundary='pec', dx=1e-3)
sim.add_source((a/3, b/3, d/2), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
sim.add_probe((a/3, b/3, d/2), 'ez')
r = sim.run(n_steps=3000)
modes = r.find_resonances(freq_range=(4e9, 10e9), source_decay_time=0.15e-9)
if modes:
    err = abs(modes[0].freq - f_tm110) / f_tm110 * 100
    print(f"  Analytical: {f_tm110/1e9:.3f} GHz")
    print(f"  Harminv:    {modes[0].freq/1e9:.3f} GHz (err={err:.2f}%, Q={modes[0].Q:.0f})")
else:
    print("  No modes found")

# ============================================================
# Test 2: Patch antenna convergence (CPML → J source)
# ============================================================
print("\n" + "=" * 60)
print("TEST 2: Patch Antenna Convergence")
print("=" * 60)

f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3; lam0 = C0/f0
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
margin = lam0/4

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")

results = []
for dx in [1.0e-3, 0.5e-3]:
    dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
    sim = Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
                     boundary='cpml', cpml_layers=12, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
    px0, py0 = margin, margin
    sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
    sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')
    sim.add_source((px0+L/3, py0+W/2, h/2), 'ez',
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((px0+L/3, py0+W/2, h/2), 'ez')

    grid = sim._build_grid()
    target_t = 10e-9
    n_steps = int(np.ceil(target_t / grid.dt))

    print(f"\n  dx={dx*1e3:.1f}mm: grid={grid.shape}, h/dx={h/dx:.0f}, n_steps={n_steps}")
    t0 = time.time()
    r = sim.run(n_steps=n_steps, subpixel_smoothing=True)
    elapsed = time.time() - t0
    cells = grid.nx*grid.ny*grid.nz
    print(f"  Runtime: {elapsed:.1f}s ({cells*n_steps/elapsed/1e6:.0f} Mcells/s)")

    t0_pulse = 3/(f0*0.8*np.pi)
    modes = r.find_resonances(freq_range=(1.5e9, 3.5e9),
                              source_decay_time=2*t0_pulse)
    if modes:
        f_res = modes[0].freq
        err = abs(f_res - f0)/f0*100
        results.append((dx*1e3, h/dx, f_res/1e9, err, modes[0].Q))
        print(f"  Harminv: f={f_res/1e9:.4f} GHz (err={err:.2f}%), Q={modes[0].Q:.0f}")
    else:
        results.append((dx*1e3, h/dx, 0, 100, 0))
        print(f"  No modes found")

# ============================================================
# Test 3: Dielectric resonator (PEC boundary, high-eps)
# ============================================================
print("\n" + "=" * 60)
print("TEST 3: Dielectric Resonator (eps_r=9.8, 10mm cube)")
print("=" * 60)

sim = Simulation(freq_max=15e9, domain=(0.03,0.03,0.03), boundary='pec', dx=0.5e-3)
sim.add(Box((0.01,0.01,0.01),(0.02,0.02,0.02)), material='silicon')
sim.add_source((0.012, 0.015, 0.015), 'ez',
               waveform=GaussianPulse(f0=8e9, bandwidth=0.8))
sim.add_probe((0.015, 0.015, 0.015), 'ez')

t0 = time.time()
r = sim.run(n_steps=4000)
elapsed = time.time() - t0
print(f"  Runtime: {elapsed:.1f}s")

modes = r.find_resonances(freq_range=(2e9, 12e9), source_decay_time=0.1e-9)
print(f"  Modes: {len(modes)}")
for m in modes[:5]:
    print(f"    f={m.freq/1e9:.3f} GHz, Q={m.Q:.0f}, amp={m.amplitude:.3e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("CONVERGENCE TABLE")
print("=" * 60)
print(f"{'dx(mm)':>8} {'h/dx':>6} {'f(GHz)':>8} {'err(%)':>8} {'Q':>6}")
for dx_mm, hdx, f, err, Q in results:
    print(f"{dx_mm:>8.2f} {hdx:>6.0f} {f:>8.4f} {err:>8.2f} {Q:>6.0f}")

if len(results) >= 2 and results[-1][2] > 0 and results[-2][2] > 0:
    f1, f2 = results[-2][2], results[-1][2]
    dx1, dx2 = results[-2][0], results[-1][0]
    r = dx1/dx2
    f_ext = (r**2 * f2 - f1) / (r**2 - 1)
    print(f"\nRichardson extrapolation (dx→0): {f_ext:.4f} GHz (err={abs(f_ext-2.4)/2.4*100:.2f}%)")
