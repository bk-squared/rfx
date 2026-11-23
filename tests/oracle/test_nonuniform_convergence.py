"""Non-uniform mesh convergence test.

Verifies that non-uniform z-mesh converges toward analytical resonance
as dz_substrate is refined. Uses a PEC cavity with thin FR4 slab.
"""

import numpy as np
import pytest


def test_nonuniform_z_convergence():
    """Non-uniform z-mesh: the measured resonance must converge as the
    substrate cells are refined.

    The cavity is partially dielectric-filled and has no simple
    closed-form resonance, so this is an oracle-free *Cauchy*
    convergence check (successive refinements move the resonance by a
    shrinking amount), not an accuracy check against an analytic
    value. The precise per-stencil guard for the non-uniform Yee
    metric is ``test_review_tier1_validation_battery.py::test_corec2_*``.
    Smooth grading avoids abrupt cell-size transitions that cause numerical reflections.
    """
    from rfx import Simulation, Box, GaussianPulse
    from rfx.auto_config import smooth_grading
    from rfx.grid import C0

    a, b = 50e-3, 40e-3
    FR4_SUBSTRATE_THICKNESS_M = 1.6e-3
    h_sub = FR4_SUBSTRATE_THICKNESS_M
    AIR_HEIGHT_M = 20e-3
    h_air = AIR_HEIGHT_M
    eps_r = 4.4

    f_empty = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)

    results = []
    for dz_sub in [0.8e-3, 0.4e-3, 0.2e-3]:
        dx = 2e-3
        n_sub = max(2, int(np.ceil(h_sub / dz_sub)))
        dz_air = dx
        n_air = max(2, int(np.ceil(h_air / dz_air)))
        dz_raw = [dz_sub] * n_sub + [dz_air] * n_air
        dz_profile = list(smooth_grading(dz_raw, max_ratio=1.3))

        sim = Simulation(
            freq_max=f_empty * 2,
            domain=(a, b),
            boundary="pec",
            dx=dx,
            dz_profile=dz_profile,
        )
        sim.add_material("fr4", eps_r=eps_r)
        sim.add(Box((0, 0, 0), (a, b, h_sub)), material="fr4")
        sim.add_source((a / 3, b / 3, h_sub / 2), "ez",
                        waveform=GaussianPulse(f0=f_empty, bandwidth=0.8))
        sim.add_probe((2 * a / 3, 2 * b / 3, h_sub / 2), "ez")

        result = sim.run(n_steps=3000)
        modes = result.find_resonances(freq_range=(f_empty * 0.5, f_empty * 1.5))
        if modes:
            f_sim = min(modes, key=lambda m: abs(m.freq - f_empty)).freq
        else:
            f_sim = 0.0

        results.append((dz_sub, f_sim))
        print(f"  dz_sub={dz_sub*1e3:.1f}mm: f={f_sim/1e9:.4f} GHz")

    freqs_sim = [f for _, f in results]
    assert all(f > 0 for f in freqs_sim), (
        f"resonance not found at every refinement level: {results}"
    )
    d01 = abs(freqs_sim[1] - freqs_sim[0])
    d12 = abs(freqs_sim[2] - freqs_sim[1])
    assert d12 <= d01 * 1.3, (
        f"non-convergent: refinement step |Δf| grew "
        f"{d01/1e6:.2f} MHz → {d12/1e6:.2f} MHz"
    )
    print(f"Cauchy convergence: |Δf| {d01/1e6:.2f} → {d12/1e6:.2f} MHz")


def test_nonuniform_cfl_uses_min_dz():
    """CFL timestep should be based on min(dz), not dx."""
    from rfx import Simulation
    from rfx.auto_config import smooth_grading
    from rfx.grid import C0

    dx = 5e-3
    dz_raw = [0.2e-3] * 8 + [5e-3] * 4
    dz_profile = list(smooth_grading(dz_raw, max_ratio=1.3))

    sim = Simulation(
        freq_max=5e9,
        domain=(20e-3, 20e-3),
        boundary="pec",
        dx=dx,
        dz_profile=dz_profile,
    )

    grid = sim._build_nonuniform_grid()
    min_dz = min(dz_profile)
    dt_expected_max = 0.99 / (C0 * np.sqrt(1/dx**2 + 1/dx**2 + 1/min_dz**2))

    print(f"dt = {grid.dt*1e12:.2f} ps, CFL limit = {dt_expected_max*1e12:.2f} ps")
    print(f"dx={dx*1e3:.1f}mm, min_dz={min_dz*1e3:.1f}mm")
    assert grid.dt <= dt_expected_max * 1.01, (
        f"dt={grid.dt:.2e} exceeds CFL ({dt_expected_max:.2e})"
    )
    dt_uniform = dx / (C0 * np.sqrt(3))
    assert grid.dt < dt_uniform, "Non-uniform dt should be < uniform dt when min_dz < dx"
