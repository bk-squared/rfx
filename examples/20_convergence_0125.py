"""dx=0.125mm patch antenna — verify <1% convergence path.

h/dx = 12.8 cells (excellent substrate resolution).
Uses J source (Cb, CPML auto-select) + GaussianPulse.
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

dx = 0.125e-3
margin = lam0/6  # ~21mm (minimal to fit GPU memory)
cpml_n = 16  # 2mm physical
dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin

sim = Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
                 boundary='cpml', cpml_layers=cpml_n, dx=dx)
sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
px0, py0 = margin, margin

sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')

sim.add_source((px0+L/3, py0+W/2, h/2), 'ez',
               waveform=GaussianPulse(f0=f0, bandwidth=0.8))
sim.add_probe((px0+L/3, py0+W/2, h/2), 'ez')

grid = sim._build_grid()
total = grid.nx*grid.ny*grid.nz
target_t = 10e-9
n_steps = int(np.ceil(target_t / grid.dt))
mem_gb = total * 6 * 4 / 1e9  # 6 fields × float32

print(f"dx={dx*1e3}mm, grid={grid.shape}")
print(f"Cells: {total/1e6:.1f}M, est field memory: {mem_gb:.1f}GB")
print(f"h/dx={h/dx:.0f}, L/dx={L/dx:.0f}")
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")
print(f"Estimated time: {total*n_steps/2e9:.0f}s at 2000 Mcells/s")

if mem_gb > 10:
    print(f"WARNING: {mem_gb:.1f}GB may OOM on 24GB GPU")

print("Running...")
t0 = time.time()
result = sim.run(n_steps=n_steps)  # no subpixel — save memory for fine grid
elapsed = time.time() - t0
print(f"Done: {elapsed:.0f}s ({total*n_steps/elapsed/1e6:.0f} Mcells/s)")

# Resonance extraction
t0_pulse = 3/(f0*0.8*np.pi)
modes = result.find_resonances(freq_range=(1.5e9, 3.5e9),
                               source_decay_time=2*t0_pulse)
print(f"\nHarminv: {len(modes)} modes")
for m in modes[:3]:
    err = abs(m.freq-f0)/f0*100
    print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

# Convergence table (include previous results for reference)
print(f"\n=== CONVERGENCE ===")
print(f"dx=1.0mm → TBD (from separate run)")
print(f"dx=0.5mm → 4.89% (VESSL #369367231522)")
print(f"dx=0.25mm → 3.78% (VESSL #369367231522)")
if modes:
    err = abs(modes[0].freq-f0)/f0*100
    print(f"dx=0.125mm → {err:.2f}% (this run)")
    print(f"\nTarget: <1%")
