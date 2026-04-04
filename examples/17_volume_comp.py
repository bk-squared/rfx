"""dx=0.25mm with volume-compensated source + highpass filter.

Root cause: point source injects energy ∝ dx³. At finer resolution,
cavity mode is weaker relative to static PEC surface charge.
Fix: amplitude ∝ (dx_ref/dx)³ + highpass filter to remove DC.
"""
import numpy as np
import jax.numpy as jnp
import time
from rfx.grid import Grid, C0
from rfx.core.yee import init_state, update_e, update_h, EPS_0, MaterialArrays
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.sources.sources import GaussianPulse, add_point_source
from rfx.harminv import harminv

f0=2.4e9; eps_r=4.4; h=1.6e-3; lam0=C0/f0
W=C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff=(eps_r+1)/2+(eps_r-1)/2*(1+12*h/W)**(-0.5)
dL=0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L=C0/(2*f0*np.sqrt(eps_eff))-2*dL
sigma_fr4=2*np.pi*f0*8.854e-12*eps_r*0.02

dx=0.25e-3; margin=lam0/4; cpml_n=12
dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
grid=Grid(freq_max=4e9, domain=(dom_x,dom_y,dom_z), dx=dx, cpml_layers=cpml_n)
print(f"Grid: {grid.shape}, cells={grid.nx*grid.ny*grid.nz/1e6:.1f}M")

# Materials (manual, proven pattern)
eps_arr=jnp.ones(grid.shape,dtype=jnp.float32)
sigma_arr=jnp.zeros(grid.shape,dtype=jnp.float32)
pec_mask=jnp.zeros(grid.shape,dtype=jnp.bool_)
px0,py0=margin,margin
cpml=cpml_n
z_gnd=grid.position_to_index((0,0,0))[2]
iz_top=grid.position_to_index((0,0,h))[2]
# ONLY set PEC inside physical domain (exclude CPML cells!)
pec_mask=pec_mask.at[cpml:-cpml,cpml:-cpml,z_gnd].set(True)
eps_arr=eps_arr.at[cpml:-cpml,cpml:-cpml,z_gnd:iz_top].set(eps_r)
sigma_arr=sigma_arr.at[cpml:-cpml,cpml:-cpml,z_gnd:iz_top].set(sigma_fr4)
ipx0=grid.position_to_index((px0,0,0))[0]
ipx1=grid.position_to_index((px0+L,0,0))[0]
ipy0=grid.position_to_index((0,py0,0))[1]
ipy1=grid.position_to_index((0,py0+W,0))[1]
iz_patch=grid.position_to_index((0,0,h))[2]
pec_mask=pec_mask.at[ipx0:ipx1,ipy0:ipy1,iz_patch].set(True)
materials=MaterialArrays(eps_r=eps_arr,sigma=sigma_arr,mu_r=jnp.ones(grid.shape,dtype=jnp.float32))

cpml_params,cpml_state=init_cpml(grid)

# Normal amplitude (1.0) — use bandpass filter instead of volume scaling
pulse=GaussianPulse(f0=f0,bandwidth=0.8)
probe_idx=grid.position_to_index((px0+L/3,py0+W/2,h/2))
print(f"Source amp=1.0 (bandpass filter for cavity mode extraction), h/dx={h/dx:.0f}")

n_steps=5000
state=init_state(grid.shape)
ez=np.zeros(n_steps)
t0w=time.time()
for n in range(n_steps):
    t=n*grid.dt
    state=update_h(state,materials,grid.dt,grid.dx)
    state,cpml_state=apply_cpml_h(state,cpml_params,cpml_state,grid,"xyz")
    state=update_e(state,materials,grid.dt,grid.dx)
    state,cpml_state=apply_cpml_e(state,cpml_params,cpml_state,grid,"xyz")
    state=apply_pec(state)
    state=apply_pec_mask(state,pec_mask)
    state=add_point_source(state,grid,(px0+L/3,py0+W/2,h/2),'ez',pulse(t))
    ez[n]=float(state.ez[probe_idx])
    if n>0 and n%1000==0:
        rate=n/(time.time()-t0w)
        print(f"  step {n}/{n_steps} | {rate:.0f}/s | max|Ez|={np.max(np.abs(ez[:n+1])):.3e}")

print(f"Done in {time.time()-t0w:.0f}s")

# Windowing
t0_p=3/(f0*0.8*np.pi); start=int(2*t0_p/grid.dt)
w=ez[start:]-np.mean(ez[start:])
print(f"Ring-down: std={np.std(w):.3e}, peak={np.max(np.abs(w)):.3e}")

# FFT-based bandpass filter (1.5-3.5 GHz) — robust unlike IIR at high Nyquist
fft_data = np.fft.rfft(w)
fft_freqs = np.fft.rfftfreq(len(w), d=grid.dt)
bp_mask = (fft_freqs >= 1.5e9) & (fft_freqs <= 3.5e9)
w_hp = np.fft.irfft(fft_data * bp_mask, n=len(w))
print(f"After FFT bandpass: std={np.std(w_hp):.3e}, peak={np.max(np.abs(w_hp)):.3e}")

# Harminv
modes=harminv(w_hp[:3000],grid.dt,1.5e9,3.5e9)
print(f"\nHarminv: {len(modes)} modes")
for m in modes[:3]:
    print(f"  f={m.freq/1e9:.4f} GHz (err={abs(m.freq-f0)/f0*100:.2f}%), Q={m.Q:.0f}")

# FFT
wh=w_hp*np.hanning(len(w_hp)); nfft=len(wh)*8
spec=np.abs(np.fft.rfft(wh,n=nfft))
fg=np.fft.rfftfreq(nfft,d=grid.dt)/1e9
band=(fg>1.5)&(fg<3.5)
if np.any(band) and np.max(spec[band])>0:
    f_fft=fg[band][np.argmax(spec[band])]
    print(f"FFT: {f_fft:.4f} GHz (err={abs(f_fft-2.4)/2.4*100:.2f}%)")

if modes:
    print(f"\n=== RESULT: {modes[0].freq/1e9:.4f} GHz, err={abs(modes[0].freq-f0)/f0*100:.2f}% ===")
