"""Cross-validation: same patch antenna in rfx, Meep, and OpenEMS.

All three simulators run the identical 2.4 GHz rectangular patch antenna:
- FR4 substrate (eps_r=4.4, tan_d=0.02), h=1.6mm
- L=29.4mm, W=38.0mm (analytical design)
- Probe feed at L/3 from edge
- PEC ground + patch
- dx=0.5mm, CPML/PML boundaries

Compares resonance frequency, runtime, and methodology.
"""
import numpy as np
import time

C0 = 299792458.0
f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3
W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02

print(f"Design: L={L*1e3:.1f}mm, W={W*1e3:.1f}mm, f0={f0/1e9:.3f}GHz")
print(f"eps_eff={eps_eff:.3f}, dL={dL*1e3:.3f}mm")

dx = 0.5e-3  # 0.5mm for all simulators
margin = C0/f0/4  # lambda/4
results = {}

# ============================================================
# 1. rfx
# ============================================================
print("\n" + "="*60)
print("rfx (JAX FDTD)")
print("="*60)
try:
    from rfx import Simulation, Box, GaussianPulse
    from rfx.harminv import harminv

    dom_x=L+2*margin; dom_y=W+2*margin; dom_z=h+margin
    px0,py0=margin,margin

    sim = Simulation(freq_max=4e9, domain=(dom_x,dom_y,dom_z),
                     boundary='cpml', cpml_layers=12, dx=dx)
    sim.add_material('FR4', eps_r=eps_r, sigma=sigma_fr4)
    sim.add(Box((0,0,0),(dom_x,dom_y,dx)), material='pec')
    sim.add(Box((0,0,0),(dom_x,dom_y,h)), material='FR4')
    sim.add(Box((px0,py0,h),(px0+L,py0+W,h+dx)), material='pec')
    sim.add_source((px0+L/3,py0+W/2,h/2), 'ez',
                   waveform=GaussianPulse(f0=f0, bandwidth=0.8))
    sim.add_probe((px0+L/3,py0+W/2,h/2), 'ez')

    grid = sim._build_grid()
    n_steps = int(np.ceil(10e-9 / grid.dt))
    print(f"Grid: {grid.shape}, n_steps={n_steps}")

    t0 = time.time()
    r = sim.run(n_steps=n_steps)
    t_rfx = time.time() - t0

    # Harminv on ring-down
    ts = np.array(r.time_series).ravel()
    t0_pulse = 3/(f0*0.8*np.pi)
    start = int(2*t0_pulse/grid.dt)
    w = ts[start:] - np.mean(ts[start:])
    modes = harminv(w, grid.dt, 1.5e9, 3.5e9)

    if modes:
        # Find mode closest to 2.4 GHz
        best = min(modes, key=lambda m: abs(m.freq - f0))
        err = abs(best.freq - f0)/f0*100
        results['rfx'] = {'freq': best.freq, 'err': err, 'Q': best.Q, 'time': t_rfx}
        print(f"Resonance: {best.freq/1e9:.4f} GHz (err={err:.2f}%), Q={best.Q:.0f}")
    else:
        results['rfx'] = {'freq': 0, 'err': 100, 'Q': 0, 'time': t_rfx}
        print("No modes found")
    print(f"Runtime: {t_rfx:.1f}s")
except Exception as e:
    print(f"rfx failed: {e}")
    results['rfx'] = {'freq': 0, 'err': 100, 'Q': 0, 'time': 0}

# ============================================================
# 2. Meep
# ============================================================
print("\n" + "="*60)
print("Meep")
print("="*60)
try:
    import meep as mp

    # Meep uses its own unit system: lengths in μm, freq in c/μm
    # Convert to Meep units: 1 unit = 1mm
    unit = 1e-3  # 1 Meep unit = 1mm
    a_m = 1/unit  # conversion factor

    resolution = int(1/(dx*a_m))  # pixels per Meep unit

    sx = (L + 2*margin)*a_m
    sy = (W + 2*margin)*a_m
    sz = (h + margin)*a_m
    pml_thick = 6*dx*a_m

    cell = mp.Vector3(sx, sy, sz)
    pml_layers = [mp.PML(pml_thick)]

    # Materials
    fr4 = mp.Medium(epsilon=eps_r, D_conductivity=2*np.pi*f0*0.02/eps_r)

    px0_m = margin*a_m
    py0_m = margin*a_m
    L_m = L*a_m
    W_m = W*a_m
    h_m = h*a_m
    dx_m = dx*a_m

    geometry = [
        # Ground plane
        mp.Block(center=mp.Vector3(0, 0, -sz/2+dx_m/2),
                 size=mp.Vector3(sx, sy, dx_m),
                 material=mp.perfect_electric_conductor),
        # Substrate
        mp.Block(center=mp.Vector3(0, 0, -sz/2+h_m/2),
                 size=mp.Vector3(sx, sy, h_m),
                 material=fr4),
        # Patch
        mp.Block(center=mp.Vector3(px0_m-sx/2+L_m/2, py0_m-sy/2+W_m/2, -sz/2+h_m+dx_m/2),
                 size=mp.Vector3(L_m, W_m, dx_m),
                 material=mp.perfect_electric_conductor),
    ]

    # Source: Gaussian pulse centered at f0
    fcen = f0 * unit / C0  # frequency in Meep units
    df = fcen * 0.8
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(px0_m-sx/2+L_m/3, 0, -sz/2+h_m/2),
        )
    ]

    sim_meep = mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    # Harminv monitor
    probe_pt = mp.Vector3(px0_m-sx/2+L_m/3, 0, -sz/2+h_m/2)
    h_mon = mp.Harminv(mp.Ez, probe_pt, fcen, df)

    t0 = time.time()
    sim_meep.run(
        mp.after_sources(h_mon),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, probe_pt, 1e-3)
    )
    t_meep = time.time() - t0

    print(f"Meep Harminv modes: {len(h_mon.modes)}")
    for m in h_mon.modes[:5]:
        f_hz = m.freq * C0 / unit
        Q = -m.freq / (2*m.decay) if m.decay != 0 else float('inf')
        err = abs(f_hz - f0)/f0*100
        print(f"  f={f_hz/1e9:.4f} GHz (err={err:.2f}%), Q={Q:.0f}")

    if h_mon.modes:
        best_m = min(h_mon.modes, key=lambda m: abs(m.freq*C0/unit - f0))
        f_best = best_m.freq * C0 / unit
        Q_best = -best_m.freq/(2*best_m.decay) if best_m.decay != 0 else 0
        err_best = abs(f_best-f0)/f0*100
        results['meep'] = {'freq': f_best, 'err': err_best, 'Q': Q_best, 'time': t_meep}
        print(f"Best: {f_best/1e9:.4f} GHz (err={err_best:.2f}%)")
    else:
        results['meep'] = {'freq': 0, 'err': 100, 'Q': 0, 'time': t_meep}
    print(f"Runtime: {t_meep:.1f}s")
    sim_meep.reset_meep()
except Exception as e:
    print(f"Meep failed: {e}")
    import traceback; traceback.print_exc()
    results['meep'] = {'freq': 0, 'err': 100, 'Q': 0, 'time': 0}

# ============================================================
# 3. OpenEMS
# ============================================================
print("\n" + "="*60)
print("OpenEMS")
print("="*60)
try:
    import CSXCAD
    from openEMS import openEMS

    unit_oe = 1e-3  # mm
    FDTD = openEMS(NrTS=50000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, f0*0.4)
    BC = ['PML_8','PML_8','PML_8','PML_8','PML_8','PML_8']
    FDTD.SetBoundaryCond(BC)

    CSX = CSXCAD.ContinuousStructure()
    FDTD.SetCSX(CSX)

    max_res = dx / unit_oe  # mesh resolution in mm

    # Grid
    margin_oe = margin/unit_oe
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit_oe)

    x_min, x_max = 0, (L+2*margin)/unit_oe
    y_min, y_max = 0, (W+2*margin)/unit_oe
    z_min, z_max = 0, (h+margin)/unit_oe

    mesh.AddLine('x', np.arange(x_min, x_max+max_res/2, max_res))
    mesh.AddLine('y', np.arange(y_min, y_max+max_res/2, max_res))
    mesh.AddLine('z', np.arange(z_min, z_max+max_res/2, max_res))

    px0_oe = margin_oe
    py0_oe = margin_oe
    L_oe = L/unit_oe
    W_oe = W/unit_oe
    h_oe = h/unit_oe
    dx_oe = dx/unit_oe

    # Ground
    gnd = CSX.AddMetal('ground')
    gnd.AddBox([x_min, y_min, 0], [x_max, y_max, dx_oe], priority=10)

    # Substrate
    sub = CSX.AddMaterial('FR4', epsilon=eps_r, kappa=sigma_fr4)
    sub.AddBox([x_min, y_min, 0], [x_max, y_max, h_oe], priority=0)

    # Patch
    patch = CSX.AddMetal('patch')
    patch.AddBox([px0_oe, py0_oe, h_oe],
                 [px0_oe+L_oe, py0_oe+W_oe, h_oe+dx_oe], priority=10)

    # Feed (lumped port)
    feed_x = px0_oe + L_oe/3
    feed_y = py0_oe + W_oe/2
    port = FDTD.AddLumpedPort(1, 50, [feed_x, feed_y, dx_oe],
                               [feed_x, feed_y, h_oe], 'z', 1.0)

    import tempfile, os
    sim_path = tempfile.mkdtemp(prefix='openems_patch_')

    t0 = time.time()
    FDTD.Run(sim_path, verbose=0)
    t_oems = time.time() - t0

    # Extract port data
    port.CalcPort(sim_path, f0*0.5, f0*1.5, 501)
    Zin = port.uf_tot / port.if_tot
    s11 = (Zin - 50) / (Zin + 50)
    freqs_oe = port.f

    # Find resonance (minimum |S11|)
    idx_min = np.argmin(np.abs(s11))
    f_res_oe = freqs_oe[idx_min]
    s11_min = 20*np.log10(np.abs(s11[idx_min]))
    err_oe = abs(f_res_oe - f0)/f0*100

    results['openems'] = {'freq': f_res_oe, 'err': err_oe, 'Q': 0, 'time': t_oems,
                          's11_min': s11_min}
    print(f"Resonance: {f_res_oe/1e9:.4f} GHz (err={err_oe:.2f}%)")
    print(f"S11 min: {s11_min:.1f} dB")
    print(f"Runtime: {t_oems:.1f}s")

    # Cleanup
    import shutil
    shutil.rmtree(sim_path, ignore_errors=True)
except Exception as e:
    print(f"OpenEMS failed: {e}")
    import traceback; traceback.print_exc()
    results['openems'] = {'freq': 0, 'err': 100, 'Q': 0, 'time': 0}

# ============================================================
# Comparison Table
# ============================================================
print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)
print(f"Analytical: {f0/1e9:.3f} GHz")
print(f"{'Simulator':>10} {'f(GHz)':>8} {'err(%)':>8} {'Q':>8} {'time(s)':>8}")
for name in ['rfx', 'meep', 'openems']:
    r = results.get(name, {})
    f = r.get('freq', 0)
    e = r.get('err', 100)
    q = r.get('Q', 0)
    t = r.get('time', 0)
    print(f"{name:>10} {f/1e9:>8.4f} {e:>8.2f} {q:>8.0f} {t:>8.1f}")
