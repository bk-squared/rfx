"""Diagnostic: Find spectral peaks (resonance) in patch antenna probe signal."""
import numpy as np
import jax.numpy as jnp
from rfx import Simulation, Box, GaussianPulse

f0 = 2.4e9; C0 = 3e8; eps_r = 4.4; h = 1.6e-3
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL

for label, dx in [("0.5mm", 0.5e-3), ("0.25mm", 0.25e-3)]:
    margin = 15e-3
    dom_x = L+2*margin; dom_y = W+2*margin; dom_z = h+10e-3
    sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02
    px0, py0 = margin, margin
    feed_x, feed_y = px0+L/3, py0+W/2

    sim = Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
                     boundary='cpml', cpml_layers=8, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
    sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
    sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')
    sim.add_port(position=(feed_x,feed_y,0), component='ez',
                 waveform=GaussianPulse(f0=f0, bandwidth=0.8), extent=h)
    sim.add_probe((feed_x,feed_y,h/2), 'ez')

    grid = sim._build_grid()
    n_steps = 4000
    print(f"\n=== dx={label}, grid={grid.shape}, h/dx={h/dx:.1f} cells ===")
    result = sim.run(n_steps=n_steps)
    ts = np.array(result.time_series).ravel()

    nfft = len(ts)*8  # extra zero-padding for frequency resolution
    spec = np.abs(np.fft.rfft(ts, n=nfft))
    fg = np.fft.rfftfreq(nfft, d=grid.dt)/1e9
    band = (fg > 1.0) & (fg < 4.0)

    # Find peak in probe spectrum (resonance = peak for interior probe)
    idx_peak = np.argmax(spec[band])
    f_peak = fg[band][idx_peak]
    print(f"  Spectral PEAK: {f_peak:.3f} GHz (err={abs(f_peak-2.4)/2.4*100:.1f}%)")

    # Find top-3 peaks
    spec_band = spec[band]
    fg_band = fg[band]
    from scipy.signal import find_peaks as _fp
    peaks, _ = _fp(spec_band, height=spec_band.max()*0.05, distance=20)
    sorted_p = sorted(peaks, key=lambda p: spec_band[p], reverse=True)
    for i, p in enumerate(sorted_p[:5]):
        print(f"  Peak {i+1}: {fg_band[p]:.3f} GHz (amp={spec_band[p]:.3e})")

    # Also check source-normalized min and max
    pulse = GaussianPulse(f0=f0, bandwidth=0.8)
    times = np.arange(n_steps)*grid.dt
    src = np.array([float(pulse(t)) for t in times])
    ss = np.abs(np.fft.rfft(src, n=nfft))
    safe = np.where(ss > ss.max()*1e-3, ss, ss.max()*1e-3)
    norm = 20*np.log10(spec/safe)
    print(f"  Norm MIN: {fg[band][np.argmin(norm[band])]:.3f} GHz")
    print(f"  Norm MAX: {fg[band][np.argmax(norm[band])]:.3f} GHz")
