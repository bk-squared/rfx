"""Validation suite: auto-config + Harminv + field animation.

Tests rfx on three canonical structures:
1. PEC cavity (exact analytical resonance)
2. Dielectric resonator (high-eps, tests material + PEC)
3. Patch antenna (full antenna workflow)

Each test uses Simulation.auto() for zero-config setup and
Result.find_resonances() for Harminv-based mode extraction.
"""
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")

from rfx import Simulation, Box, GaussianPulse
from rfx.simulation import SnapshotSpec
from rfx.sources.sources import add_point_source
from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h, MaterialArrays
from rfx.boundaries.pec import apply_pec, apply_pec_mask
from rfx.boundaries.cpml import init_cpml, apply_cpml_h, apply_cpml_e
from rfx.harminv import harminv


def test_pec_cavity():
    """Test 1: PEC box cavity — exact analytical resonance."""
    print("\n" + "="*60)
    print("TEST 1: PEC Cavity (30×30×10 mm)")
    print("="*60)

    a, b, d = 0.03, 0.03, 0.01
    f_tm110 = C0/2 * np.sqrt((1/a)**2 + (1/b)**2)
    print(f"Analytical TM110: {f_tm110/1e9:.3f} GHz")

    dx = 1e-3
    grid = Grid(freq_max=10e9, domain=(a, b, d), dx=dx, cpml_layers=0)
    materials = init_materials(grid.shape)
    pulse = GaussianPulse(f0=6e9, bandwidth=0.8)
    probe_idx = grid.position_to_index((a/3, b/3, d/2))

    state = init_state(grid.shape)
    n_steps = 3000
    ez_trace = np.zeros(n_steps)
    for n in range(n_steps):
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = add_point_source(state, grid, (a/3, b/3, d/2), 'ez', pulse(n*grid.dt))
        ez_trace[n] = float(state.ez[probe_idx])

    # Harminv
    t0 = 3/(6e9*0.8*np.pi)
    modes = harminv(ez_trace[int(2*t0/grid.dt):] - np.mean(ez_trace[int(2*t0/grid.dt):]),
                    grid.dt, 4e9, 10e9)

    if modes:
        f_found = modes[0].freq
        err = abs(f_found - f_tm110) / f_tm110 * 100
        print(f"Harminv TM110: {f_found/1e9:.3f} GHz (err={err:.1f}%)")
        print(f"Q = {modes[0].Q:.0f}")
        return err
    else:
        print("No modes found!")
        return 100.0


def test_dielectric_resonator():
    """Test 2: Dielectric resonator in PEC cavity."""
    print("\n" + "="*60)
    print("TEST 2: Dielectric Resonator (eps_r=9.8, 10mm cube in 30mm PEC box)")
    print("="*60)

    # High-eps cube inside PEC box — resonance shifted down by sqrt(eps_r)
    a = 0.03; d_res = 0.01
    eps_r = 9.8

    # Approximate: f ≈ f_pec_cavity / sqrt(eps_r) for dominant mode in dielectric
    f_approx = C0 / (2*d_res*np.sqrt(eps_r)) / np.sqrt(3)  # rough TE111 estimate
    print(f"Approximate resonance: ~{f_approx/1e9:.1f} GHz")

    dx = 0.5e-3
    grid = Grid(freq_max=15e9, domain=(a, a, a), dx=dx, cpml_layers=0)

    eps_r_arr = jnp.ones(grid.shape, dtype=jnp.float32)
    # Place dielectric cube at center
    c_lo = grid.position_to_index((0.01, 0.01, 0.01))
    c_hi = grid.position_to_index((0.02, 0.02, 0.02))
    eps_r_arr = eps_r_arr.at[c_lo[0]:c_hi[0], c_lo[1]:c_hi[1], c_lo[2]:c_hi[2]].set(eps_r)
    materials = MaterialArrays(eps_r=eps_r_arr,
                               sigma=jnp.zeros(grid.shape, dtype=jnp.float32),
                               mu_r=jnp.ones(grid.shape, dtype=jnp.float32))

    pulse = GaussianPulse(f0=8e9, bandwidth=0.8)
    probe_idx = grid.position_to_index((0.015, 0.015, 0.015))

    state = init_state(grid.shape)
    n_steps = 4000
    ez_trace = np.zeros(n_steps)
    for n in range(n_steps):
        state = update_h(state, materials, grid.dt, grid.dx)
        state = update_e(state, materials, grid.dt, grid.dx)
        state = apply_pec(state)
        state = add_point_source(state, grid, (0.012, 0.015, 0.015), 'ez', pulse(n*grid.dt))
        ez_trace[n] = float(state.ez[probe_idx])

    # Harminv
    t0 = 3/(8e9*0.8*np.pi)
    start = int(2*t0/grid.dt)
    windowed = ez_trace[start:] - np.mean(ez_trace[start:])
    modes = harminv(windowed, grid.dt, 2e9, 12e9)

    print(f"Modes found: {len(modes)}")
    for m in modes[:5]:
        print(f"  f={m.freq/1e9:.3f} GHz, Q={m.Q:.0f}, amp={m.amplitude:.3e}")

    if modes:
        f1 = modes[0].freq
        # The dominant mode should be lower than the empty cavity TM110 (7.07 GHz)
        # due to the high-eps filling
        f_empty = C0/2 * np.sqrt(2) / a
        ratio = f1 / f_empty
        print(f"f_mode/f_empty = {ratio:.3f} (expected ~1/sqrt(eps_r) ≈ {1/np.sqrt(eps_r):.3f})")
        return modes[0]
    return None


def test_patch_antenna():
    """Test 3: Patch antenna — full workflow with auto-config."""
    print("\n" + "="*60)
    print("TEST 3: Patch Antenna (2.4 GHz, auto-configured)")
    print("="*60)

    f0 = 2.4e9; eps_r = 4.4; h = 1.6e-3
    W = C0/(2*f0)*np.sqrt(2/(eps_r+1))
    eps_eff = (eps_r+1)/2 + (eps_r-1)/2*(1+12*h/W)**(-0.5)
    dL = 0.412*h*((eps_eff+0.3)*(W/h+0.264)/((eps_eff-0.258)*(W/h+0.8)))
    L = C0/(2*f0*np.sqrt(eps_eff)) - 2*dL
    sigma_fr4 = 2*np.pi*f0*8.854e-12*eps_r*0.02

    # Use low-level API with point source (no port loading)
    margin = C0/f0/4  # lambda/4
    dx = 0.5e-3
    dom_x = L+2*margin; dom_y = W+2*margin; dom_z = h+margin
    cpml_n = 12

    grid = Grid(freq_max=4e9, domain=(dom_x, dom_y, dom_z), dx=dx, cpml_layers=cpml_n)
    print(f"Grid: {grid.shape}, dx={dx*1e3}mm")

    # Materials + PEC mask
    eps_r_arr = jnp.ones(grid.shape, dtype=jnp.float32)
    sigma_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
    pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)

    px0, py0 = margin, margin
    z_gnd = grid.position_to_index((0,0,0))[2]
    pec_mask = pec_mask.at[:,:,z_gnd].set(True)

    sub_m = 5e-3
    ix0 = grid.position_to_index((px0-sub_m,0,0))[0]
    ix1 = grid.position_to_index((px0+L+sub_m,0,0))[0]
    iy0 = grid.position_to_index((0,py0-sub_m,0))[1]
    iy1 = grid.position_to_index((0,py0+W+sub_m,0))[1]
    iz_top = grid.position_to_index((0,0,h))[2]
    eps_r_arr = eps_r_arr.at[ix0:ix1,iy0:iy1,z_gnd:iz_top].set(eps_r)
    sigma_arr = sigma_arr.at[ix0:ix1,iy0:iy1,z_gnd:iz_top].set(sigma_fr4)

    ipx0 = grid.position_to_index((px0,0,0))[0]
    ipx1 = grid.position_to_index((px0+L,0,0))[0]
    ipy0 = grid.position_to_index((0,py0,0))[1]
    ipy1 = grid.position_to_index((0,py0+W,0))[1]
    iz_patch = grid.position_to_index((0,0,h))[2]
    pec_mask = pec_mask.at[ipx0:ipx1,ipy0:ipy1,iz_patch].set(True)

    materials = MaterialArrays(eps_r=eps_r_arr, sigma=sigma_arr,
                               mu_r=jnp.ones(grid.shape, dtype=jnp.float32))

    cpml_params, cpml_state = init_cpml(grid)
    feed_x, feed_y = px0+L/3, py0+W/2
    probe_idx = grid.position_to_index((feed_x, feed_y, h/2))
    pulse = GaussianPulse(f0=f0, bandwidth=0.8)

    target_t = 10e-9
    n_steps = int(np.ceil(target_t / grid.dt))
    print(f"n_steps={n_steps}, T_sim={n_steps*grid.dt*1e9:.1f}ns")

    state = init_state(grid.shape)
    ez_trace = np.zeros(n_steps)
    import time; t0w = time.time()
    for n in range(n_steps):
        t = n * grid.dt
        state = update_h(state, materials, grid.dt, grid.dx)
        state, cpml_state = apply_cpml_h(state, cpml_params, cpml_state, grid, "xyz")
        state = update_e(state, materials, grid.dt, grid.dx)
        state, cpml_state = apply_cpml_e(state, cpml_params, cpml_state, grid, "xyz")
        state = apply_pec(state)
        state = apply_pec_mask(state, pec_mask)
        state = add_point_source(state, grid, (feed_x, feed_y, h/2), 'ez', pulse(t))
        ez_trace[n] = float(state.ez[probe_idx])
        if n > 0 and n % 2000 == 0:
            rate = n/(time.time()-t0w)
            print(f"  step {n}/{n_steps} | {rate:.0f}/s | ETA {(n_steps-n)/rate:.0f}s")

    print(f"  Done in {time.time()-t0w:.0f}s")

    # Harminv
    t0_pulse = 3/(f0*0.8*np.pi)
    start = int(2*t0_pulse/grid.dt)
    windowed = ez_trace[start:] - np.mean(ez_trace[start:])
    modes = harminv(windowed, grid.dt, 1.5e9, 3.5e9)

    print(f"\nModes found: {len(modes)}")
    for m in modes[:5]:
        err_pct = abs(m.freq - f0)/f0*100
        print(f"  f={m.freq/1e9:.3f} GHz (err={err_pct:.1f}%), Q={m.Q:.0f}, amp={m.amplitude:.3e}")

    if modes:
        return abs(modes[0].freq - f0)/f0*100
    return 100.0


# ---- Run all tests ----
print("rfx Validation Suite")
print("=" * 60)

err1 = test_pec_cavity()
mode2 = test_dielectric_resonator()
err3 = test_patch_antenna()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"PEC cavity TM110:    {err1:.1f}% error")
if mode2:
    print(f"Dielectric resonator: {mode2.freq/1e9:.3f} GHz, Q={mode2.Q:.0f}")
print(f"Patch antenna:       {err3:.1f}% error")
print(f"\nTarget: <3% for all. Patch <1% needs dx=0.25mm or subgridding.")
