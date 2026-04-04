"""Patch antenna via JIT-compiled SBP-SAT subgridding.

Coarse: dx=2mm (air region, CPML boundary)
Fine: dx=0.25mm (substrate + near-field, ratio=8)
→ h/dx_fine = 6.4 cells across substrate

This is the proper way to achieve <1% accuracy: fine resolution
where needed, coarse elsewhere, all JIT-compiled.
"""
import numpy as np
import time
from rfx import Simulation, Box, GaussianPulse
from rfx.grid import C0

f0=2.4e9; eps_r=4.4; h=1.6e-3; lam0=C0/f0
W=C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff=(eps_r+1)/2+(eps_r-1)/2*(1+12*h/W)**(-0.5)
dL=0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L=C0/(2*f0*np.sqrt(eps_eff))-2*dL
sigma_fr4=2*np.pi*f0*8.854e-12*eps_r*0.02

# Coarse grid: 2mm (air, CPML)
dx_c = 2e-3
ratio = 8
dx_f = dx_c / ratio  # 0.25mm

margin = lam0/4  # 31mm — proper CPML margin
dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
px0, py0 = margin, margin

sim = Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
                 boundary='cpml', cpml_layers=6, dx=dx_c)
sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)

# Geometry (PEC thickness = dx_c = 1 coarse cell)
sim.add(Box((0,0,0),(dom_x,dom_y,dx_c)), material='pec')
sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx_c)), material='pec')

# Soft source (J-source, no port loading)
sim.add_source((px0+L/3, py0+W/2, h/2), 'ez',
               waveform=GaussianPulse(f0=f0, bandwidth=0.8))
sim.add_probe((px0+L/3, py0+W/2, h/2), 'ez')

# Subgridding: fine grid around substrate
sim.add_refinement(z_range=(0, h+3e-3), ratio=ratio)

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")
print(f"Coarse dx={dx_c*1e3}mm, Fine dx={dx_f*1e3}mm (ratio={ratio})")
print(f"h/dx_fine={h/dx_f:.0f} cells")

target_t = 10e-9
sg_dt = 0.45 * dx_f / (C0 * np.sqrt(3))
n_steps = int(np.ceil(target_t / sg_dt))
print(f"n_steps={n_steps}, T_sim={n_steps*sg_dt*1e9:.1f}ns")

print("\nRunning JIT subgridded...")
t0 = time.time()
result = sim.run(n_steps=n_steps)
elapsed = time.time() - t0
print(f"Done: {elapsed:.1f}s")

# Resonance extraction
t0_pulse = 3/(f0*0.8*np.pi)
modes = result.find_resonances(freq_range=(1.5e9, 3.5e9),
                               source_decay_time=2*t0_pulse)
print(f"\nHarminv: {len(modes)} modes")
for m in modes[:5]:
    err = abs(m.freq-f0)/f0*100
    print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

if modes:
    best = modes[0]
    err = abs(best.freq-f0)/f0*100
    print(f"\n=== SUBGRID RESULT: {best.freq/1e9:.4f} GHz, err={err:.2f}% ===")
    print(f"Target: <1%")
