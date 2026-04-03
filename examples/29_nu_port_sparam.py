"""Non-uniform mesh + wire port S-param extraction.

Uses lumped port (Z0=50Ω) for proper S11 extraction instead of
point source + Harminv. S11 minimum = resonance frequency.
"""
import numpy as np
import jax.numpy as jnp
import jax
import time
from rfx.nonuniform import make_nonuniform_grid, run_nonuniform
from rfx.core.yee import MaterialArrays, EPS_0
from rfx.sources.sources import GaussianPulse
from rfx.grid import C0

f0=2.4e9; eps_r=4.4; h=1.6e-3
W=C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff=(eps_r+1)/2+(eps_r-1)/2*(1+12*h/W)**(-0.5)
dL=0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L=C0/(2*f0*np.sqrt(eps_eff))-2*dL
sigma_fr4=2*np.pi*f0*8.854e-12*eps_r*0.02

dx=0.5e-3; margin=C0/f0/4; cpml=12
dz_sub=np.ones(4)*0.4e-3  # h=1.6mm exact
dz_air=np.ones(int(round(margin/0.5e-3)))*0.5e-3
grid=make_nonuniform_grid((L+2*margin,W+2*margin),
                           np.concatenate([dz_sub,dz_air]), dx, cpml)
shape=(grid.nx, grid.ny, grid.nz)
print(f"Grid: {shape}, dt={grid.dt:.3e}")

# Materials + PEC
eps_arr=jnp.ones(shape,dtype=jnp.float32)
sigma_arr=jnp.zeros(shape,dtype=jnp.float32)
pec_mask=jnp.zeros(shape,dtype=jnp.bool_)
pec_mask=pec_mask.at[cpml:-cpml,cpml:-cpml,cpml].set(True)
iz_top=cpml+4
eps_arr=eps_arr.at[cpml:-cpml,cpml:-cpml,cpml:iz_top].set(eps_r)
sigma_arr=sigma_arr.at[cpml:-cpml,cpml:-cpml,cpml:iz_top].set(sigma_fr4)
px0=int(round(margin/dx))+cpml; py0=px0
px1=px0+int(round(L/dx)); py1=py0+int(round(W/dx))
pec_mask=pec_mask.at[px0:px1,py0:py1,iz_top].set(True)

# Wire port: add impedance loading to substrate
Z0=50.0
fi=px0+int(round(L/3/dx)); fj=py0+int(round(W/2/dx))
# Wire cells: from ground (cpml+1) to patch (iz_top-1)
wire_cells = list(range(cpml+1, iz_top))
n_cells = len(wire_cells)
sigma_port_per_cell = n_cells / (Z0 * dx)
for k in wire_cells:
    sigma_arr = sigma_arr.at[fi, fj, k].add(sigma_port_per_cell)
    pec_mask = pec_mask.at[fi, fj, k].set(False)

materials=MaterialArrays(eps_r=eps_arr, sigma=sigma_arr,
                         mu_r=jnp.ones(shape,dtype=jnp.float32))

# Source waveform (Cb-corrected for port, like make_port_source)
fk_mid = cpml + 2  # midpoint
eps_s=float(eps_arr[fi,fj,fk_mid])*EPS_0
sig_s=float(sigma_arr[fi,fj,fk_mid])
loss=sig_s*grid.dt/(2*eps_s)
cb=(grid.dt/eps_s)/(1+loss)
pulse=GaussianPulse(f0=f0, bandwidth=0.8)
n_steps=int(np.ceil(10e-9/grid.dt))
times=jnp.arange(n_steps,dtype=jnp.float32)*grid.dt
# Distribute source across wire cells
wf_per_cell = (cb/dx)*jax.vmap(pulse)(times)/n_cells
sources = [(fi, fj, k, 'ez', np.array(wf_per_cell)) for k in wire_cells]

# Probes + wire port spec for S-param
probes = [(fi, fj, fk_mid, 'ez')]
wire_port_specs = [{
    'mid_i': fi, 'mid_j': fj, 'mid_k': fk_mid,
    'component': 'ez', 'impedance': Z0,
}]
sp_freqs = np.linspace(1.5e9, 3.5e9, 100)

print(f"Wire port: {n_cells} cells, Z0={Z0}Ω, n_steps={n_steps}")
print("Running...")

t0=time.time()
r=run_nonuniform(grid, materials, n_steps, pec_mask=pec_mask,
                 sources=sources, probes=probes,
                 wire_ports=wire_port_specs, s_param_freqs=sp_freqs)
elapsed=time.time()-t0
print(f"Done: {elapsed:.1f}s")

if "s_params" in r:
    S = r["s_params"]
    freqs_GHz = r["s_param_freqs"] / 1e9
    s11_dB = 20*np.log10(np.maximum(np.abs(S[0,0,:]), 1e-10))

    # Find resonance: minimum S11
    idx_min = np.argmin(s11_dB)
    f_res = freqs_GHz[idx_min]
    s11_min = s11_dB[idx_min]
    err = abs(f_res - f0/1e9) / (f0/1e9) * 100

    print(f"\n=== S11 Results ===")
    print(f"Resonance (S11 min): {f_res:.4f} GHz")
    print(f"S11 at resonance: {s11_min:.1f} dB")
    print(f"Error: {err:.2f}%")
    print(f"Design: {f0/1e9:.3f} GHz")

    # Also try Harminv on time series
    from rfx.harminv import harminv
    ts = np.array(r['time_series']).ravel()
    t0p=3/(f0*0.8*np.pi); start=int(2*t0p/grid.dt)
    w=ts[start:]-np.mean(ts[start:])
    modes=harminv(w, grid.dt, 1.5e9, 3.5e9)
    if modes:
        best=min(modes, key=lambda m:abs(m.freq-f0))
        print(f"Harminv: {best.freq/1e9:.4f} GHz (err={abs(best.freq-f0)/f0*100:.2f}%), Q={best.Q:.0f}")
else:
    print("No S-params extracted")
