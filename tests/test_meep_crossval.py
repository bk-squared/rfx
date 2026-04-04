"""Cross-validation: rfx vs Meep for PEC rectangular cavity.

Compares TM110 resonant frequency from both simulators against
the analytical solution.

Requires: meep (pip install meep)
"""

import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


def fft_peak_freq(time_series: np.ndarray, dt: float,
                  f_lo: float, f_hi: float) -> float:
    """Find peak frequency using zero-padded FFT + parabolic interpolation.

    Zero-pads to 8x length for finer bin spacing, then uses 3-point
    parabolic interpolation around the peak for sub-bin accuracy.
    """
    n = len(time_series)
    n_pad = n * 8  # zero-pad for finer interpolation
    spectrum = np.abs(np.fft.rfft(time_series, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt)

    mask = (freqs >= f_lo) & (freqs <= f_hi)
    masked = np.where(mask, spectrum, 0.0)
    peak_idx = np.argmax(masked)

    # Parabolic interpolation around peak
    if 0 < peak_idx < len(spectrum) - 1:
        alpha = spectrum[peak_idx - 1]
        beta = spectrum[peak_idx]
        gamma = spectrum[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-30:
            p = 0.5 * (alpha - gamma) / denom
            return freqs[peak_idx] + p * (freqs[1] - freqs[0])

    return freqs[peak_idx]

# Physical cavity dimensions (meters)
CAVITY_A = 0.1
CAVITY_B = 0.1
CAVITY_D = 0.05

# Analytical TM110
F_ANALYTICAL = (C0 / 2.0) * np.sqrt((1 / CAVITY_A) ** 2 + (1 / CAVITY_B) ** 2)


def run_rfx_cavity() -> float:
    """Run rfx PEC cavity and return peak frequency via FFT."""
    # Match Meep resolution (dx=1mm) for fair comparison
    grid = Grid(freq_max=5e9, domain=(CAVITY_A, CAVITY_B, CAVITY_D),
                dx=0.001, cpml_layers=0)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=F_ANALYTICAL, bandwidth=0.8)
    src_i = grid.nx // 3
    src_j = grid.ny // 3
    src_k = grid.nz // 2

    probe_i = 2 * grid.nx // 3
    probe_j = 2 * grid.ny // 3
    probe_k = grid.nz // 2

    # 120 periods for sufficient FFT frequency resolution
    num_steps = grid.num_timesteps(num_periods=120)
    dt = grid.dt
    dx = grid.dx

    time_series = np.zeros(num_steps)
    for n in range(num_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        ez = state.ez.at[src_i, src_j, src_k].add(pulse(t))
        state = state._replace(ez=ez)
        time_series[n] = float(state.ez[probe_i, probe_j, probe_k])

    return fft_peak_freq(time_series, dt, F_ANALYTICAL * 0.5, F_ANALYTICAL * 1.5)


def run_meep_cavity() -> float:
    """Run Meep PEC cavity and return resonant frequency via Harminv."""
    import meep as mp

    unit = 0.01  # 1 meep unit = 1 cm
    Lx = CAVITY_A / unit
    Ly = CAVITY_B / unit
    Lz = CAVITY_D / unit

    fcen = F_ANALYTICAL * unit / C0
    df = fcen * 0.8

    cell = mp.Vector3(Lx, Ly, Lz)

    sim = mp.Simulation(
        cell_size=cell,
        resolution=10,
        boundary_layers=[],  # PEC walls
        default_material=mp.Medium(epsilon=1),
    )

    src_pt = mp.Vector3(Lx / 3 - Lx / 2, Ly / 3 - Ly / 2, 0)
    sim.sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=src_pt,
        )
    ]

    probe_pt = mp.Vector3(2 * Lx / 3 - Lx / 2, 2 * Ly / 3 - Ly / 2, 0)
    h = mp.Harminv(mp.Ez, probe_pt, fcen, df)

    sim.run(mp.after_sources(h), until_after_sources=200 / fcen)

    # Extract strongest mode frequency
    if not h.modes:
        pytest.fail("Meep Harminv found no modes")

    best = max(h.modes, key=lambda m: abs(m.amp))
    f_meep_si = best.freq * C0 / unit
    return f_meep_si


@pytest.mark.slow
def test_rfx_vs_meep_cavity():
    """rfx and Meep should agree on TM110 resonance within 0.5%."""
    f_rfx = run_rfx_cavity()
    f_meep = run_meep_cavity()

    err_rfx = abs(f_rfx - F_ANALYTICAL) / F_ANALYTICAL
    err_meep = abs(f_meep - F_ANALYTICAL) / F_ANALYTICAL
    err_cross = abs(f_rfx - f_meep) / f_meep

    print(f"\n{'='*50}")
    print("PEC Cavity TM110 Cross-Validation")
    print(f"{'='*50}")
    print(f"Analytical:  {F_ANALYTICAL / 1e9:.6f} GHz")
    print(f"rfx:         {f_rfx / 1e9:.6f} GHz  (err: {err_rfx*100:.3f}%)")
    print(f"Meep:        {f_meep / 1e9:.6f} GHz  (err: {err_meep*100:.3f}%)")
    print(f"rfx vs Meep: {err_cross*100:.3f}%")
    print(f"{'='*50}")

    assert err_rfx < 0.005, f"rfx error {err_rfx*100:.2f}% exceeds 0.5%"
    assert err_meep < 0.005, f"Meep error {err_meep*100:.2f}% exceeds 0.5%"
    assert err_cross < 0.01, f"Cross-validation gap {err_cross*100:.2f}% exceeds 1%"


@pytest.mark.slow
def test_rfx_vs_meep_dielectric_loaded():
    """Cross-validate with a dielectric slab inside the cavity.

    Insert εr=4 slab in center third of x-axis.
    Both simulators should find the same shifted resonance.
    """
    import meep as mp

    eps_slab = 4.0

    # --- rfx ---
    # Match Meep resolution: dx=1mm for fair comparison
    grid = Grid(freq_max=5e9, domain=(CAVITY_A, CAVITY_B, CAVITY_D),
                dx=0.001, cpml_layers=0)
    materials = init_materials(grid.shape)

    # Dielectric slab: x in [a/3, 2a/3], snapped to grid
    slab_lo_m = CAVITY_A / 3.0
    slab_hi_m = 2.0 * CAVITY_A / 3.0
    slab_lo = round(slab_lo_m / grid.dx)
    slab_hi = round(slab_hi_m / grid.dx)
    eps_r = materials.eps_r.at[slab_lo:slab_hi, :, :].set(eps_slab)
    materials = materials._replace(eps_r=eps_r)

    state = init_state(grid.shape)
    pulse = GaussianPulse(f0=1.5e9, bandwidth=1.0)
    src_i, src_j, src_k = grid.nx // 4, grid.ny // 4, grid.nz // 2
    probe_i, probe_j, probe_k = 3 * grid.nx // 4, 3 * grid.ny // 4, grid.nz // 2

    num_steps = grid.num_timesteps(num_periods=120)
    dt, dx = grid.dt, grid.dx

    time_series = np.zeros(num_steps)
    for n in range(num_steps):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        ez = state.ez.at[src_i, src_j, src_k].add(pulse(t))
        state = state._replace(ez=ez)
        time_series[n] = float(state.ez[probe_i, probe_j, probe_k])

    f_rfx = fft_peak_freq(time_series, dt, 0.5e9, F_ANALYTICAL)

    # --- Meep ---
    unit = 0.01
    Lx = CAVITY_A / unit
    Ly = CAVITY_B / unit
    Lz = CAVITY_D / unit

    # Dielectric block in center third of x
    geometry = [
        mp.Block(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(Lx / 3, Ly, Lz),
            material=mp.Medium(epsilon=eps_slab),
        )
    ]

    fcen = 1.5e9 * unit / C0
    df = fcen * 1.0

    sim = mp.Simulation(
        cell_size=mp.Vector3(Lx, Ly, Lz),
        resolution=10,
        boundary_layers=[],
        geometry=geometry,
        default_material=mp.Medium(epsilon=1),
        eps_averaging=False,  # disable subpixel smoothing for fair comparison
    )

    src_pt = mp.Vector3(Lx / 4 - Lx / 2, Ly / 4 - Ly / 2, 0)
    sim.sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=src_pt,
        )
    ]

    probe_pt = mp.Vector3(3 * Lx / 4 - Lx / 2, 3 * Ly / 4 - Ly / 2, 0)
    h = mp.Harminv(mp.Ez, probe_pt, fcen, df)

    sim.run(mp.after_sources(h), until_after_sources=300 / fcen)

    if not h.modes:
        pytest.fail("Meep Harminv found no modes for dielectric-loaded cavity")

    best = max(h.modes, key=lambda m: abs(m.amp))
    f_meep = best.freq * C0 / unit

    err_cross = abs(f_rfx - f_meep) / f_meep

    print(f"\n{'='*50}")
    print("Dielectric-Loaded Cavity Cross-Validation")
    print(f"{'='*50}")
    print(f"rfx:         {f_rfx / 1e9:.6f} GHz")
    print(f"Meep:        {f_meep / 1e9:.6f} GHz")
    print(f"rfx vs Meep: {err_cross*100:.3f}%")
    print(f"{'='*50}")

    # With matched resolution (dx=1mm), gap should be under 1%
    # (dielectric interface effects make this harder to match than the empty cavity)
    assert err_cross < 0.01, f"Cross-validation gap {err_cross*100:.2f}% exceeds 1%"
