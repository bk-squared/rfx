"""Patch antenna targeting <1% error.

dx=0.125mm (h/dx=13 cells), proper λ/4 margin, J source, long ring-down.
Uses A6000 (48GB) for the larger grid.
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

dx=0.125e-3  # 0.125mm → h/dx=12.8
margin=lam0/4  # proper λ/4 = 31mm
cpml_n=12  # 12*0.125=1.5mm physical CPML

dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
px0,py0=margin,margin

sim=Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
               boundary='cpml', cpml_layers=cpml_n, dx=dx)
sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')

# J source (auto-selected for CPML boundary)
sim.add_source((px0+L/3, py0+W/2, h/2), 'ez',
               waveform=GaussianPulse(f0=f0, bandwidth=0.8))
sim.add_probe((px0+L/3, py0+W/2, h/2), 'ez')

grid=sim._build_grid()
total=grid.nx*grid.ny*grid.nz
mem_gb=total*6*4/1e9

# Long run: 15ns for full ring-down
target_t=15e-9
n_steps=int(np.ceil(target_t/grid.dt))

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")
print(f"dx={dx*1e3}mm, h/dx={h/dx:.0f}, L/dx={L/dx:.0f}")
print(f"Grid: {grid.shape}, cells={total/1e6:.1f}M, mem={mem_gb:.1f}GB")
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")
print(f"Estimated: {total*n_steps/2e9:.0f}s at 2000 Mcells/s")

print("\nRunning with subpixel smoothing...")
t0=time.time()
result=sim.run(n_steps=n_steps, subpixel_smoothing=True)
elapsed=time.time()-t0
mcells=total*n_steps/elapsed/1e6
print(f"Done: {elapsed:.0f}s ({mcells:.0f} Mcells/s)")

# Analysis
ts=np.array(result.time_series).ravel()
t0_pulse=3/(f0*0.8*np.pi)
start=int(2*t0_pulse/grid.dt)
w=ts[start:]-np.mean(ts[start:])
print(f"Analysis window: {len(w)} samples ({len(w)*grid.dt*1e9:.1f}ns)")

# Harminv with bandpass (auto for CPML)
modes=result.find_resonances(freq_range=(1.5e9, 3.5e9),
                             source_decay_time=2*t0_pulse)
print(f"\nHarminv: {len(modes)} modes")
for m in modes[:5]:
    err=abs(m.freq-f0)/f0*100
    print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

# Also try direct Harminv without bandpass
modes2=result.find_resonances(freq_range=(1.5e9, 3.5e9),
                              source_decay_time=2*t0_pulse,
                              bandpass=False)
print(f"\nDirect Harminv (no bandpass): {len(modes2)} modes")
for m in modes2[:5]:
    err=abs(m.freq-f0)/f0*100
    print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

# FFT cross-check
wh=w*np.hanning(len(w))
nfft=len(wh)*8
spec=np.abs(np.fft.rfft(wh,n=nfft))
fg=np.fft.rfftfreq(nfft,d=grid.dt)/1e9
band=(fg>1.5)&(fg<3.5)
f_fft=fg[band][np.argmax(spec[band])]
print(f"\nFFT peak: {f_fft:.4f} GHz (err={abs(f_fft-2.4)/2.4*100:.2f}%)")

if modes:
    best=min(modes, key=lambda m: abs(m.freq-f0))
    err=abs(best.freq-f0)/f0*100
    print(f"\n{'='*50}")
    print(f"RESULT: {best.freq/1e9:.4f} GHz, err={err:.2f}%")
    print(f"Target: <1%  {'ACHIEVED!' if err < 1 else 'NOT YET'}")
    print(f"{'='*50}")
