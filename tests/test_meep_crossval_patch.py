"""Cross-validation: rfx vs Meep for dielectric-loaded PEC cavity.

Tests a substrate-thickness PEC cavity (simplified patch model) that
isolates FDTD accuracy from analytical formula error. Both simulators
use identical geometry — any disagreement is a true FDTD difference.

Uses 2D TMz mode for fast CPU execution (~seconds each).
"""

import numpy as np
import pytest

C0 = 299792458.0


def analytical_tm_2d(a, b, eps_r, m, n):
    """Analytical TM_mn resonance for rectangular cavity with uniform eps_r."""
    return (C0 / (2 * np.sqrt(eps_r))) * np.sqrt((m / a) ** 2 + (n / b) ** 2)


# --- Test cases ---
# Use square-ish cavities filled with substrate materials.
# This tests that rfx and Meep agree on dielectric-loaded cavity physics.
# TM11 mode: f = C0/(2*sqrt(eps_r)) * sqrt((1/a)^2 + (1/b)^2)

# dx=1mm for all cases (fair comparison, same grid for both simulators)
_DX = 1e-3

CASE_FR4 = {
    "a": 50e-3, "b": 40e-3, "eps_r": 4.4, "sigma": 0.0,
    "mode": (1, 1), "dx": _DX, "desc": "FR4 dielectric cavity"
}

CASE_ROGERS = {
    "a": 40e-3, "b": 30e-3, "eps_r": 3.55, "sigma": 0.0,
    "mode": (1, 1), "dx": _DX, "desc": "Rogers dielectric cavity"
}

CASE_GPS = {
    "a": 80e-3, "b": 60e-3, "eps_r": 2.2, "sigma": 0.0,
    "mode": (1, 1), "dx": _DX, "desc": "GPS dielectric cavity"
}


def run_rfx_2d_cavity(a, b, eps_r, sigma, m, n, dx=None):
    """Run rfx 2D TMz cavity and return resonant frequency."""
    from rfx.grid import Grid
    from rfx.core.yee import init_state, update_e, update_h, MaterialArrays
    from rfx.boundaries.pec import apply_pec
    from rfx.sources.sources import GaussianPulse
    import jax.numpy as jnp

    f_est = analytical_tm_2d(a, b, eps_r, m, n)
    if dx is None:
        # Use fine dx for substrate: at least 4 cells across b
        dx = min(b / 4, C0 / f_est / (20 * np.sqrt(eps_r)))

    grid = Grid(freq_max=f_est * 2, domain=(a, b, dx),
                dx=dx, cpml_layers=0, mode="2d_tmz")

    # Fill with dielectric
    eps_arr = jnp.ones(grid.shape, dtype=jnp.float32) * eps_r
    sigma_arr = jnp.ones(grid.shape, dtype=jnp.float32) * sigma
    materials = MaterialArrays(
        eps_r=eps_arr,
        sigma=sigma_arr,
        mu_r=jnp.ones(grid.shape, dtype=jnp.float32),
    )

    state = init_state(grid.shape)
    pulse = GaussianPulse(f0=f_est, bandwidth=0.8)

    src_i = grid.nx // 3
    src_j = grid.ny // 3
    src_k = grid.nz // 2
    probe_i = 2 * grid.nx // 3
    probe_j = 2 * grid.ny // 3
    probe_k = src_k

    num_steps = grid.num_timesteps(num_periods=150)
    dt = grid.dt
    time_series = np.zeros(num_steps)

    for step in range(num_steps):
        t = step * dt
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        ez = state.ez.at[src_i, src_j, src_k].add(pulse(t))
        state = state._replace(ez=ez)
        time_series[step] = float(state.ez[probe_i, probe_j, probe_k])

    # FFT peak with parabolic interpolation
    n_pad = len(time_series) * 8
    spectrum = np.abs(np.fft.rfft(time_series, n=n_pad))
    freqs = np.fft.rfftfreq(n_pad, d=dt)
    mask = (freqs >= f_est * 0.5) & (freqs <= f_est * 1.5)
    masked = np.where(mask, spectrum, 0.0)
    peak_idx = np.argmax(masked)
    if 0 < peak_idx < len(spectrum) - 1:
        alpha, beta, gamma = spectrum[peak_idx - 1], spectrum[peak_idx], spectrum[peak_idx + 1]
        denom = alpha - 2 * beta + gamma
        if abs(denom) > 1e-30:
            p = 0.5 * (alpha - gamma) / denom
            return freqs[peak_idx] + p * (freqs[1] - freqs[0])
    return freqs[peak_idx]


def run_meep_2d_cavity(a, b, eps_r, sigma, m, n, dx=None):
    """Run Meep 2D TMz cavity and return resonant frequency."""
    import meep as mp

    f_est = analytical_tm_2d(a, b, eps_r, m, n)
    if dx is None:
        dx = min(b / 4, C0 / f_est / (20 * np.sqrt(eps_r)))

    unit = 1e-3  # 1 Meep unit = 1mm
    resolution = max(1, int(round(1 / (dx / unit))))

    Lx = a / unit
    Ly = b / unit
    fcen = f_est * unit / C0
    df = fcen * 0.8

    medium = mp.Medium(
        epsilon=eps_r,
        D_conductivity=2 * np.pi * f_est * sigma / eps_r if sigma > 0 else 0,
    )

    cell = mp.Vector3(Lx, Ly, 0)  # 2D
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=[],
        default_material=medium,
        dimensions=2,
    )

    src_pt = mp.Vector3(Lx / 3 - Lx / 2, Ly / 3 - Ly / 2)
    sim.sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=src_pt,
        )
    ]

    probe_pt = mp.Vector3(2 * Lx / 3 - Lx / 2, 2 * Ly / 3 - Ly / 2)
    h = mp.Harminv(mp.Ez, probe_pt, fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=200 / fcen)

    if not h.modes:
        return 0.0

    best = max(h.modes, key=lambda mode: abs(mode.amp))
    return best.freq * C0 / unit


@pytest.mark.parametrize("case", [CASE_FR4, CASE_ROGERS, CASE_GPS],
                         ids=["FR4", "Rogers", "GPS"])
def test_rfx_vs_meep_dielectric_cavity(case):
    """rfx and Meep should agree within 0.5% on substrate-filled cavity."""
    pytest.importorskip("meep")

    a, b, eps_r = case["a"], case["b"], case["eps_r"]
    sigma = case["sigma"]
    m, n = case["mode"]
    dx = case.get("dx")
    desc = case["desc"]

    f_analytical = analytical_tm_2d(a, b, eps_r, m, n)
    f_rfx = run_rfx_2d_cavity(a, b, eps_r, sigma, m, n, dx=dx)
    f_meep = run_meep_2d_cavity(a, b, eps_r, sigma, m, n, dx=dx)

    err_rfx = abs(f_rfx - f_analytical) / f_analytical
    err_meep = abs(f_meep - f_analytical) / f_analytical
    err_cross = abs(f_rfx - f_meep) / max(f_meep, 1.0)

    print(f"\n{'='*55}")
    print(f"{desc} (TM{m}{n}, eps_r={eps_r})")
    print(f"{'='*55}")
    print(f"  Analytical: {f_analytical/1e9:.4f} GHz")
    print(f"  rfx:        {f_rfx/1e9:.4f} GHz  (err: {err_rfx*100:.3f}%)")
    print(f"  Meep:       {f_meep/1e9:.4f} GHz  (err: {err_meep*100:.3f}%)")
    print(f"  rfx vs Meep: {err_cross*100:.3f}%")

    assert err_rfx < 0.01, f"rfx error {err_rfx*100:.2f}% exceeds 1.0%"
    assert err_meep < 0.01, f"Meep error {err_meep*100:.2f}% exceeds 1.0%"
    assert err_cross < 0.005, f"Cross-val gap {err_cross*100:.3f}% exceeds 0.5%"
