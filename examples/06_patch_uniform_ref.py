"""Diagnostic: Uniform dx=0.25mm patch antenna reference (no subgridding)."""
import numpy as np
import jax.numpy as jnp
from rfx import Simulation, Box, GaussianPulse

f0 = 2.4e9; C0 = 3e8; eps_r = 4.4; h = 1.6e-3
W = C0 / (2*f0) * np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2 * (1+12*h/W)**(-0.5)
dL = 0.412*h * ((eps_eff+0.3)*(W/h+0.264) / ((eps_eff-0.258)*(W/h+0.8)))
L = C0 / (2*f0*np.sqrt(eps_eff)) - 2*dL

dx = 0.25e-3; margin = 15e-3
dom_x = L + 2*margin; dom_y = W + 2*margin; dom_z = h + 15e-3
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
px0, py0 = margin, margin
feed_x, feed_y = px0 + L/3, py0 + W/2

sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z),
                 boundary='cpml', cpml_layers=8, dx=dx)
sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
sim.add(Box((0,0,0), (dom_x, dom_y, dx)), material='pec')
sim.add(Box((0,0,0), (dom_x, dom_y, h)), material='FR4')
sim.add(Box((px0, py0, h), (px0+L, py0+W, h+dx)), material='pec')
sim.add_port(position=(feed_x, feed_y, 0), component='ez',
             waveform=GaussianPulse(f0=f0, bandwidth=0.8), extent=h)
sim.add_probe((feed_x, feed_y, h/2), 'ez')

grid = sim._build_grid()
print(f'Grid: {grid.shape}, substrate={h/dx:.0f} cells, dt={grid.dt:.3e}')
print(f'Running 4000 steps...')
result = sim.run(n_steps=4000)
ts = np.array(result.time_series).ravel()
print(f'Max|Ez|={np.max(np.abs(ts)):.4e}, NaN={np.any(np.isnan(ts))}')

pulse = GaussianPulse(f0=f0, bandwidth=0.8)
times = np.arange(4000) * grid.dt
src = np.array([float(pulse(t)) for t in times])
nfft = len(ts) * 4
sp = np.abs(np.fft.rfft(ts, n=nfft))
ss = np.abs(np.fft.rfft(src, n=nfft))
fg = np.fft.rfftfreq(nfft, d=grid.dt) / 1e9
safe = np.where(ss > ss.max()*1e-3, ss, ss.max()*1e-3)
s11 = 20 * np.log10(sp / safe)
band = (fg > 1.5) & (fg < 3.5)
idx = np.argmin(s11[band])
f_res = fg[band][idx]
print(f'RESULT: f_res={f_res:.3f}GHz err={abs(f_res-2.4)/2.4*100:.1f}% dip={s11[band][idx]:.1f}dB')
