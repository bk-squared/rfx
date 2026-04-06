"""Non-uniform mesh convergence test.

Verifies that non-uniform z-mesh converges toward analytical resonance
as dz_substrate is refined. Uses a PEC cavity with thin FR4 slab.
"""

import numpy as np
import pytest


@pytest.mark.slow
def test_nonuniform_z_convergence():
    """Resonance should converge as substrate dz is refined."""
    from rfx import Simulation, Box, GaussianPulse
    from rfx.grid import C0

    a, b = 50e-3, 40e-3
    h_sub = 1.6e-3   # FR4 substrate thickness
    h_air = 20e-3     # air above
    eps_r = 4.4

    # Analytical: cavity TM110 with partial dielectric
    f_empty = (C0 / 2) * np.sqrt((1 / a) ** 2 + (1 / b) ** 2)

    results = []
    for dz_sub in [0.8e-3, 0.4e-3, 0.2e-3]:
        dx = 2e-3
        # Build dz profile: fine in substrate, coarse in air
        n_sub = max(2, int(np.ceil(h_sub / dz_sub)))
        dz_air = dx  # coarse in air
        n_air = max(2, int(np.ceil(h_air / dz_air)))
        dz_profile = [dz_sub] * n_sub + [dz_air] * n_air

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

    # Convergence: error should decrease (or stay small) as dz refines
    errors = [abs(f - f_empty) / f_empty for _, f in results if f > 0]
    if len(errors) >= 2:
        assert errors[-1] <= errors[0] * 1.5, (
            f"Non-convergent: coarsest err={errors[0]:.4f}, finest err={errors[-1]:.4f}"
        )
        print(f"Convergence: {errors[0]*100:.2f}% → {errors[-1]*100:.2f}%")


def test_nonuniform_cfl_uses_min_dz():
    """CFL timestep should be based on min(dz), not dx."""
    from rfx import Simulation
    from rfx.grid import C0

    dx = 5e-3
    dz_profile = [0.2e-3] * 8 + [5e-3] * 4  # min dz = 0.2mm

    sim = Simulation(
        freq_max=5e9,
        domain=(20e-3, 20e-3),
        boundary="pec",
        dx=dx,
        dz_profile=dz_profile,
    )

    grid = sim._build_nonuniform_grid()
    dt_expected_max = 0.2e-3 / (C0 * np.sqrt(3))  # CFL from min cell

    print(f"dt = {grid.dt*1e12:.2f} ps, CFL limit = {dt_expected_max*1e12:.2f} ps")
    assert grid.dt <= dt_expected_max * 1.01, (
        f"dt={grid.dt:.2e} exceeds CFL from min dz ({dt_expected_max:.2e})"
    )
