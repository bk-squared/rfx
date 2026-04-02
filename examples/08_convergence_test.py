"""Convergence test: uniform vs subgridded, peak finding for resonance."""
import numpy as np
import jax.numpy as jnp
from rfx import Simulation, Box, GaussianPulse

f0 = 2.4e9; C0 = 3e8; eps_r = 4.4; h = 1.6e-3
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f_analytical={f0/1e9:.3f}GHz")

sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
# Margin >= lambda/4 for reasonable CPML absorption
lam0 = C0 / f0  # 125mm at 2.4 GHz
margin = lam0 / 4  # quarter-wavelength (~31mm)
px0, py0 = margin, margin
print(f"lambda={lam0*1e3:.0f}mm, margin={margin*1e3:.0f}mm (={margin/lam0:.1f}*lambda)")
feed_x, feed_y = px0+L/3, py0+W/2

def run_test(label, dx, n_steps, use_subgrid=False, ratio=8):
    dom_x = L+2*margin; dom_y = W+2*margin; dom_z = h+lam0/4
    dx_f = dx/ratio if use_subgrid else dx
    cpml_n = max(int(round(lam0/20 / dx)), 8)  # ~lambda/20

    sim = Simulation(freq_max=4e9, domain=(dom_x, dom_y, dom_z),
                     boundary='cpml', cpml_layers=cpml_n, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)

    pec_t = dx  # PEC thickness = cell size
    # Ground plane: full domain
    sim.add(Box((0,0,0),(dom_x,dom_y,pec_t)), material='pec')
    # Substrate: only around the patch (not full domain)
    sub_m = 5e-3
    sim.add(Box((px0-sub_m,py0-sub_m,0),(px0+L+sub_m,py0+W+sub_m,h)), material='FR4')
    # Patch
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+pec_t)), material='pec')

    sim.add_port(position=(feed_x,feed_y,pec_t), component='ez',
                 waveform=GaussianPulse(f0=f0, bandwidth=0.8),
                 extent=h-2*pec_t)
    sim.add_probe((feed_x,feed_y,h/2), 'ez')

    if use_subgrid:
        sim.add_refinement(z_range=(0, h+3e-3), ratio=ratio)

    grid = sim._build_grid()
    dt = 0.45*dx_f/(C0*np.sqrt(3)) if use_subgrid else grid.dt

    print(f"\n=== {label}: dx={'subgrid '+str(dx*1e3)+'/'+str(dx_f*1e3)+'mm' if use_subgrid else str(dx*1e3)+'mm'}, {n_steps} steps ===")
    print(f"  dt={dt:.3e}s, T_sim={n_steps*dt*1e9:.2f}ns, h/dx_f={h/dx_f:.0f} cells")

    result = sim.run(n_steps=n_steps)
    ts = np.array(result.time_series).ravel()

    nfft = len(ts)*8
    spec = np.abs(np.fft.rfft(ts, n=nfft))
    fg = np.fft.rfftfreq(nfft, d=dt)/1e9
    band = (fg > 1.0) & (fg < 4.0)
    idx = np.argmax(spec[band])
    f_peak = fg[band][idx]
    err = abs(f_peak-2.4)/2.4*100
    print(f"  PEAK: {f_peak:.3f} GHz (err={err:.1f}%)")
    print(f"  max|Ez|={np.max(np.abs(ts)):.3e}, NaN={np.any(np.isnan(ts))}")
    return f_peak

# Ensure all runs have T_sim >= 10ns for proper ring-down
target_t = 10e-9  # 10ns

for label, dx in [("1.0mm", 1.0e-3), ("0.5mm", 0.5e-3)]:
    dt_est = dx / (C0 * np.sqrt(3)) * 0.99
    n_steps = int(np.ceil(target_t / dt_est))
    run_test(f"Uniform {label}", dx=dx, n_steps=n_steps)
