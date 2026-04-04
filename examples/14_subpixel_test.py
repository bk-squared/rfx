"""Test subpixel smoothing impact on patch antenna resonance accuracy."""
import numpy as np
from rfx import Simulation, Box, GaussianPulse
from rfx.harminv import harminv
from rfx.grid import C0

f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3; lam0 = C0/f0
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
margin = lam0/4; sub_m = 5e-3

import time

for label, use_sp in [("No smoothing", False), ("Subpixel smoothing", True)]:
    dx = 0.5e-3
    dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
    px0, py0 = margin, margin

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z),
                     boundary='cpml', cpml_layers=12, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
    sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
    sim.add(Box((px0-sub_m,py0-sub_m,0),(px0+L+sub_m,py0+W+sub_m,h)), material='FR4')
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')
    sim.add_source(position=(px0+L/3,py0+W/2,h/2), component='ez',
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((px0+L/3,py0+W/2,h/2), 'ez')

    grid = sim._build_grid()
    target_t = 10e-9
    n_steps = int(np.ceil(target_t / grid.dt))

    print(f"\n=== {label} (dx={dx*1e3}mm, n_steps={n_steps}) ===")
    t0 = time.time()
    result = sim.run(n_steps=n_steps, subpixel_smoothing=use_sp)
    elapsed = time.time() - t0
    print(f"Runtime: {elapsed:.1f}s")

    ts = np.array(result.time_series).ravel()
    t0_pulse = 3/(f0*0.8*np.pi)
    start = int(2*t0_pulse/grid.dt)
    windowed = ts[start:] - np.mean(ts[start:])

    modes = harminv(windowed, grid.dt, 1.5e9, 3.5e9)
    if modes:
        f_res = modes[0].freq
        err = abs(f_res - f0)/f0*100
        print(f"Harminv: f={f_res/1e9:.4f} GHz, err={err:.2f}%, Q={modes[0].Q:.0f}")
    else:
        windowed_h = windowed * np.hanning(len(windowed))
        nfft = len(windowed_h)*8
        spec = np.abs(np.fft.rfft(windowed_h, n=nfft))
        fg = np.fft.rfftfreq(nfft, d=grid.dt)/1e9
        band = (fg>1.5)&(fg<3.5)
        f_res = fg[band][np.argmax(spec[band])]
        err = abs(f_res-2.4)/2.4*100
        print(f"FFT: f={f_res:.4f} GHz, err={err:.2f}%")

print("\n=== COMPARISON ===")
print("Subpixel smoothing compensates for substrate boundary discretization")
print("by computing effective permittivity at interface cells (Farjadpour 2006)")
