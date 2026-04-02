"""Test ModulatedGaussian source (Meep-style, zero DC).

Validates:
1. PEC cavity resonance — should give 0% error (no DC distortion)
2. Patch antenna dx=0.5mm — convergence baseline
3. Patch antenna dx=0.25mm — should work without bandpass hack
"""
import numpy as np
import time
from rfx import Simulation, Box, GaussianPulse
from rfx.harminv import harminv_from_probe
from rfx.grid import C0

f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3; lam0 = C0/f0
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02

# Test 1: PEC cavity
print("="*50)
print("TEST 1: PEC cavity (J source)")
print("="*50)
a,b,d = 0.03, 0.03, 0.01
f_tm110 = C0/2*np.sqrt((1/a)**2+(1/b)**2)
sim = Simulation(freq_max=10e9, domain=(a,b,d), boundary='pec', dx=1e-3)
sim.add_source((a/3,b/3,d/2), 'ez', waveform=GaussianPulse(f0=6e9, bandwidth=0.8))
sim.add_probe((a/3,b/3,d/2), 'ez')
r = sim.run(n_steps=3000)
modes = r.find_resonances(freq_range=(4e9,10e9), source_decay_time=0.2e-9)
if modes:
    print(f"TM110: {modes[0].freq/1e9:.3f} GHz (analytical={f_tm110/1e9:.3f}), err={abs(modes[0].freq-f_tm110)/f_tm110*100:.2f}%")

# Test 2 & 3: Patch antenna at two resolutions
for label, dx in [("0.5mm", 0.5e-3), ("0.25mm", 0.25e-3)]:
    print(f"\n{'='*50}")
    print(f"TEST: Patch antenna dx={label} (J source)")
    print(f"{'='*50}")

    margin = lam0/4
    dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin

    sim = Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
                     boundary='cpml', cpml_layers=12, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
    sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
    sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
    px0,py0 = margin, margin
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')

    sim.add_source((px0+L/3, py0+W/2, h/2), 'ez',
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((px0+L/3, py0+W/2, h/2), 'ez')

    grid = sim._build_grid()
    target_t = 10e-9
    n_steps = int(np.ceil(target_t / grid.dt))
    print(f"Grid: {grid.shape}, h/dx={h/dx:.0f}, n_steps={n_steps}")

    t0 = time.time()
    r = sim.run(n_steps=n_steps, subpixel_smoothing=True)
    elapsed = time.time()-t0
    total_cells = grid.nx*grid.ny*grid.nz
    print(f"Runtime: {elapsed:.1f}s ({total_cells*n_steps/elapsed/1e6:.0f} Mcells/s)")

    t0_pulse = 3/(f0*0.8*np.pi)
    modes = r.find_resonances(freq_range=(1.5e9,3.5e9),
                              source_decay_time=2*t0_pulse)
    print(f"Harminv: {len(modes)} modes")
    for m in modes[:3]:
        err = abs(m.freq-f0)/f0*100
        print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

print("\n" + "="*50)
print("DONE")
