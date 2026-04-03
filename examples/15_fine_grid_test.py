"""dx=0.25mm patch antenna with subpixel smoothing — target <2% error."""
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

margin=lam0/3; sub_m=5e-3; dx=0.25e-3
cpml_n=max(int(round(8e-3/dx)), 12)  # at least 8mm physical thickness
dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
px0,py0=margin,margin

sim=Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
               boundary='cpml', cpml_layers=cpml_n, dx=dx)
sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')  # full-domain substrate
sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')
# Use moderate-impedance port (Z0=500Ω) to:
# 1. Selectively damp substrate surface waves (they couple to port)
# 2. Still allow patch cavity to resonate (Q_loaded ≈ Q/2 at 500Ω)
sim.add_port(position=(px0+L/3,py0+W/2,0), component='ez',
             impedance=500.0,
             waveform=GaussianPulse(f0=f0, bandwidth=0.8),
             extent=h)
sim.add_probe((px0+L/3,py0+W/2,h/2), 'ez')

grid=sim._build_grid()
target_t=10e-9
n_steps=int(np.ceil(target_t/grid.dt))
total_cells=grid.nx*grid.ny*grid.nz
mem_est=total_cells*6*4/1e9  # 6 fields, 4 bytes each

print(f"dx={dx*1e3}mm, grid={grid.shape}")
print(f"Cells: {total_cells/1e6:.1f}M, est memory: {mem_est:.1f}GB")
print(f"h/dx={h/dx:.0f}, L/dx={L/dx:.0f}")
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")

# Run with subpixel smoothing
print(f"\nRunning with subpixel smoothing...")
t0=time.time()
result=sim.run(n_steps=n_steps, subpixel_smoothing=True)
elapsed=time.time()-t0
mcells=total_cells*n_steps/elapsed/1e6
print(f"Runtime: {elapsed:.1f}s ({mcells:.0f} Mcells/s)")

ts=np.array(result.time_series).ravel()
t0_pulse=3/(f0*0.8*np.pi)
start=int(2*t0_pulse/grid.dt)
windowed=ts[start:]-np.mean(ts[start:])

# Limit to 3000 samples for Harminv speed (SVD scales as N^3)
max_harminv = 3000
if len(windowed) > max_harminv:
    step = len(windowed) // max_harminv
    windowed_ds = windowed[::step][:max_harminv]
    dt_ds = grid.dt * step
else:
    windowed_ds = windowed
    dt_ds = grid.dt
modes=harminv(windowed_ds, dt_ds, 1.5e9, 3.5e9)
print(f"\nHarminv modes:")
for m in modes[:3]:
    err=abs(m.freq-f0)/f0*100
    print(f"  f={m.freq/1e9:.4f} GHz (err={err:.2f}%), Q={m.Q:.0f}")

# FFT cross-check
windowed_h=windowed*np.hanning(len(windowed))
nfft=len(windowed_h)*8
spec=np.abs(np.fft.rfft(windowed_h, n=nfft))
fg=np.fft.rfftfreq(nfft, d=grid.dt)/1e9
band=(fg>1.5)&(fg<3.5)
f_fft=fg[band][np.argmax(spec[band])]
print(f"FFT: {f_fft:.4f} GHz (err={abs(f_fft-2.4)/2.4*100:.2f}%)")

if modes:
    f_best=modes[0].freq
    print(f"\n=== RESULT: {f_best/1e9:.4f} GHz, err={abs(f_best-f0)/f0*100:.2f}%, Q={modes[0].Q:.0f} ===")
