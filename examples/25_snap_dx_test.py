"""Test: does snapping dx to substrate thickness eliminate discretization error?

h=1.6mm. Test dx values where h/dx is exact integer vs non-integer:
- dx=0.5mm: h/dx=3.2 (rounded to 3, 6.25% h error)
- dx=0.4mm: h/dx=4.0 (exact, 0% h error)
- dx=0.32mm: h/dx=5.0 (exact, 0% h error)
"""
import numpy as np
import time
from rfx import Simulation, Box, GaussianPulse
from rfx.harminv import harminv
from rfx.grid import C0

f0=2.4e9; eps_r=4.4; h=1.6e-3; lam0=C0/f0
W=C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff=(eps_r+1)/2+(eps_r-1)/2*(1+12*h/W)**(-0.5)
dL=0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L=C0/(2*f0*np.sqrt(eps_eff))-2*dL
sigma_fr4=2*np.pi*f0*8.854e-12*eps_r*0.02
margin=lam0/4

results=[]
for label, dx in [("0.50mm (h/dx=3.2)", 0.5e-3),
                   ("0.40mm (h/dx=4.0)", 0.4e-3),
                   ("0.32mm (h/dx=5.0)", 0.32e-3)]:
    h_cells=h/dx; h_grid=round(h/dx)*dx; h_err=abs(h_grid-h)/h*100
    dom_x=L+2*margin;dom_y=W+2*margin;dom_z=h+margin;px0,py0=margin,margin

    sim=Simulation(freq_max=4e9,domain=(dom_x,dom_y,dom_z),
                   boundary='cpml',cpml_layers=12,dx=dx)
    sim.add_material('FR4',eps_r=eps_r,sigma=sigma_fr4)
    sim.add(Box((0,0,0),(dom_x,dom_y,dx)),material='pec')
    sim.add(Box((0,0,0),(dom_x,dom_y,h)),material='FR4')
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)),material='pec')
    sim.add_source((px0+L/3,py0+W/2,h/2),'ez',
                   waveform=GaussianPulse(f0=f0,bandwidth=0.8))
    sim.add_probe((px0+L/3,py0+W/2,h/2),'ez')

    grid=sim._build_grid()
    n=int(np.ceil(10e-9/grid.dt))
    cells=grid.nx*grid.ny*grid.nz

    print(f"\n=== {label} ===")
    print(f"  h/dx={h_cells:.1f}, h_grid={h_grid*1e3:.3f}mm, h_err={h_err:.1f}%")
    print(f"  grid={grid.shape}, cells={cells/1e6:.1f}M, n_steps={n}")

    t0=time.time()
    r=sim.run(n_steps=n)
    elapsed=time.time()-t0

    ts=np.array(r.time_series).ravel()
    t0p=3/(f0*0.8*np.pi);start=int(2*t0p/grid.dt)
    w=ts[start:]-np.mean(ts[start:])
    modes=harminv(w,grid.dt,1.5e9,3.5e9)
    if modes:
        best=min(modes,key=lambda m:abs(m.freq-f0))
        err=abs(best.freq-f0)/f0*100
        results.append((label,dx*1e3,h_cells,h_err,best.freq/1e9,err,best.Q,elapsed))
        print(f"  Resonance: {best.freq/1e9:.4f} GHz (err={err:.2f}%), Q={best.Q:.0f}, {elapsed:.1f}s")
    else:
        results.append((label,dx*1e3,h_cells,h_err,0,100,0,elapsed))
        print(f"  No modes found, {elapsed:.1f}s")

print(f"\n{'='*70}")
print(f"SNAP-DX CONVERGENCE TABLE")
print(f"{'='*70}")
print(f"{'dx':>8} {'h/dx':>6} {'h_err%':>7} {'f(GHz)':>8} {'err%':>7} {'Q':>6} {'time':>6}")
for label,dx_mm,h_dx,h_err,freq,err,Q,t in results:
    print(f"{dx_mm:>7.2f}  {h_dx:>5.1f}  {h_err:>6.1f}  {freq:>7.4f}  {err:>6.2f}  {Q:>5.0f}  {t:>5.1f}s")
