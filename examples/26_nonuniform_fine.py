"""Non-uniform mesh: fine lateral (dx=0.25mm) + snapped substrate (dz=0.4mm).

Combines both accuracy improvements:
1. h/dz=4 exact (no substrate discretization error)
2. dx=dy=0.25mm (L/dx=118 cells, fine lateral resolution)
3. dz=0.5mm in air (coarse, saves cells)
"""
import numpy as np, jax.numpy as jnp, jax, time
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform
from rfx.core.yee import MaterialArrays, EPS_0
from rfx.sources.sources import GaussianPulse
from rfx.harminv import harminv
from rfx.grid import C0

f0=2.4e9;eps_r=4.4;h=1.6e-3
W=C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff=(eps_r+1)/2+(eps_r-1)/2*(1+12*h/W)**(-0.5)
dL=0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L=C0/(2*f0*np.sqrt(eps_eff))-2*dL
sigma_fr4=2*np.pi*f0*8.854e-12*eps_r*0.02

dx=0.25e-3  # fine lateral
margin=C0/f0/4; cpml=12

# Z-profile: 4×0.4mm substrate + air at 0.5mm
dz_sub=np.ones(4)*0.4e-3  # h=1.6mm exact
dz_air=np.ones(int(round(margin/0.5e-3)))*0.5e-3
dz_profile=np.concatenate([dz_sub, dz_air])

grid=make_nonuniform_grid((L+2*margin,W+2*margin), dz_profile, dx, cpml)
shape=(grid.nx,grid.ny,grid.nz)
total=grid.nx*grid.ny*grid.nz
print(f"dx={dx*1e3}mm, dz_sub=0.4mm, dz_air=0.5mm")
print(f"Grid: {shape}, cells={total/1e6:.1f}M")
print(f"h snapped: {np.sum(np.ones(4)*0.4e-3)*1e3:.3f}mm")

eps_arr=jnp.ones(shape,dtype=jnp.float32);sigma_arr=jnp.zeros(shape,dtype=jnp.float32)
pec_mask=jnp.zeros(shape,dtype=jnp.bool_)
pec_mask=pec_mask.at[cpml:-cpml,cpml:-cpml,cpml].set(True)
iz_top=cpml+4
eps_arr=eps_arr.at[cpml:-cpml,cpml:-cpml,cpml:iz_top].set(eps_r)
sigma_arr=sigma_arr.at[cpml:-cpml,cpml:-cpml,cpml:iz_top].set(sigma_fr4)
px0=int(round(margin/dx))+cpml;py0=px0
px1=px0+int(round(L/dx));py1=py0+int(round(W/dx))
pec_mask=pec_mask.at[px0:px1,py0:py1,iz_top].set(True)
materials=MaterialArrays(eps_r=eps_arr,sigma=sigma_arr,mu_r=jnp.ones(shape,dtype=jnp.float32))

fi=px0+int(round(L/3/dx));fj=py0+int(round(W/2/dx));fk=cpml+2
dz_local=float(grid.dz[fk])
eps_s=float(eps_arr[fi,fj,fk])*EPS_0;sig_s=float(sigma_arr[fi,fj,fk])
loss=sig_s*grid.dt/(2*eps_s);cb=(grid.dt/eps_s)/(1+loss)
pulse=GaussianPulse(f0=f0,bandwidth=0.8)
n_steps=min(20000,int(np.ceil(10e-9/grid.dt)))
times=jnp.arange(n_steps,dtype=jnp.float32)*grid.dt
wf=jax.vmap(pulse)(times)  # RAW source — resolution-independent
print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")

print("Running...")
t0=time.time()
r=run_nonuniform(grid,materials,n_steps,pec_mask=pec_mask,
                 sources=[(fi,fj,fk,'ez',np.array(wf))],probes=[(fi,fj,fk,'ez')])
elapsed=time.time()-t0
print(f"Done: {elapsed:.1f}s ({total*n_steps/elapsed/1e6:.0f} Mcells/s)")

ts=np.array(r['time_series']).ravel()
t0p=3/(f0*0.8*np.pi);start=int(2*t0p/grid.dt)
w=ts[start:]-np.mean(ts[start:])
modes=harminv(w,grid.dt,1.5e9,3.5e9)
if modes:
    best=min(modes,key=lambda m:abs(m.freq-f0))
    print(f"\nHarminv: {best.freq/1e9:.4f} GHz, err={abs(best.freq-f0)/f0*100:.2f}%, Q={best.Q:.0f}")
else:
    # Bandpass fallback
    fft_data=np.fft.rfft(w);fft_freqs=np.fft.rfftfreq(len(w),d=grid.dt)
    bp=(fft_freqs>=1.5e9)&(fft_freqs<=3.5e9)
    w_bp=np.fft.irfft(fft_data*bp,n=len(w))
    modes=harminv(w_bp[:3000],grid.dt,1.5e9,3.5e9)
    if modes:
        best=min(modes,key=lambda m:abs(m.freq-f0))
        print(f"\nBP Harminv: {best.freq/1e9:.4f} GHz, err={abs(best.freq-f0)/f0*100:.2f}%, Q={best.Q:.0f}")
    else:
        wh=w*np.hanning(len(w));nfft=len(wh)*8
        spec=np.abs(np.fft.rfft(wh,n=nfft))
        fg=np.fft.rfftfreq(nfft,d=grid.dt)/1e9
        band=(fg>1.5)&(fg<3.5)
        f_fft=fg[band][np.argmax(spec[band])]
        print(f"\nFFT: {f_fft:.4f} GHz, err={abs(f_fft-2.4)/2.4*100:.2f}%")
