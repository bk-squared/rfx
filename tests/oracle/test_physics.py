"""CPML grazing-incidence and late-time numerical stability tests."""

from rfx.geometry.rasterize_grid import unpadded_high_walls

import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, EPS_0, MU_0
from rfx.stepping.yee import update_e, update_h
from rfx.boundaries.pec import apply_pec
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.sources import (
    GaussianPulse,
)


# =========================================================================
# Test 2: CPML grazing-incidence reflection benchmark
# =========================================================================

@pytest.mark.slow_physics
def test_cpml_grazing_incidence():
    """CPML reflection at near-grazing incidence should stay below -20 dB.

    A point source near a CPML boundary generates waves at all angles
    including grazing. Compare energy after CPML absorption to initial energy.
    Tighter than normal CPML test: the source is placed close to one face
    to maximize grazing-angle content.
    """
    grid = Grid(freq_max=5e9, domain=(0.10, 0.10, 0.04), dx=0.002, cpml_layers=12)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cp, cs = init_cpml(grid)
    nc = grid.cpml_layers
    dt, dx = grid.dt, grid.dx

    # Source placed very close to x-lo CPML face (2 cells from CPML interior edge)
    # This maximizes grazing-angle energy hitting that face
    src = (nc + 2, grid.ny // 2, grid.nz // 2)
    pulse = GaussianPulse(f0=3e9, bandwidth=0.8, amplitude=10.0)

    def em_energy(s):
        interior = (slice(nc, -nc), slice(nc, -nc), slice(nc, -nc))
        return float(
            0.5 * EPS_0 * (s.ex[interior]**2 + s.ey[interior]**2 + s.ez[interior]**2).sum()
            + 0.5 * MU_0 * (s.hx[interior]**2 + s.hy[interior]**2 + s.hz[interior]**2).sum()
        )

    # Inject source
    peak_energy = 0.0
    for n in range(300):
        t = n * dt
        state = update_h(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state, cs = apply_cpml_h(state, cp, cs, grid)
        state = update_e(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state, cs = apply_cpml_e(state, cp, cs, grid)
        ez = state.ez.at[src].add(pulse(t))
        state = state._replace(ez=ez)

        if n % 50 == 49:
            e = em_energy(state)
            peak_energy = max(peak_energy, e)

    # Let CPML absorb — run until energy decays
    for n in range(1500):
        state = update_h(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state, cs = apply_cpml_h(state, cp, cs, grid)
        state = update_e(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state, cs = apply_cpml_e(state, cp, cs, grid)

    energy_final = em_energy(state)
    assert peak_energy > 0, "No energy injected"

    reflection_db = 10 * np.log10(max(energy_final / peak_energy, 1e-30))

    print("\nCPML grazing incidence:")
    print(f"  Peak energy:  {peak_energy:.4e}")
    print(f"  Final energy: {energy_final:.4e}")
    print(f"  Reflection: {reflection_db:.1f} dB")

    # CPML should achieve < -20 dB even at grazing incidence
    assert reflection_db < -20, \
        f"CPML grazing reflection {reflection_db:.1f} dB exceeds -20 dB"
    assert not np.isnan(energy_final)


# =========================================================================
# Test 4: Late-time numerical stability
# =========================================================================

def test_late_time_stability():
    """Lossless PEC cavity: no NaN/Inf/blowup over thousands of timesteps.

    Uses a strong source to keep energy well above float32 noise floor.
    In a lossless PEC cavity the scheme is energy-conserving; any
    exponential growth would indicate instability.
    """
    grid = Grid(freq_max=3e9, domain=(11 * (C0 / 3e9 / 20.0), 11 * (C0 / 3e9 / 20.0), 11 * (C0 / 3e9 / 20.0)), cpml_layers=0)  # main's solved length; #1887
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)

    pulse = GaussianPulse(f0=2e9, bandwidth=0.5, amplitude=1e3)
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    dt, dx = grid.dt, grid.dx

    def em_energy(s):
        return float(
            0.5 * EPS_0 * (s.ex**2 + s.ey**2 + s.ez**2).sum()
            + 0.5 * MU_0 * (s.hx**2 + s.hy**2 + s.hz**2).sum()
        )

    # Inject source for 150 steps
    for n in range(150):
        t = n * dt
        state = update_h(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state = update_e(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)
        ez = state.ez.at[cx, cy, cz].add(pulse(t))
        state = state._replace(ez=ez)

    # Let settle for 50 steps (no source)
    for _ in range(50):
        state = update_h(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state = update_e(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)

    energy_ref = em_energy(state)
    assert not np.isnan(energy_ref), "NaN after source"
    assert energy_ref > 0, "No energy injected"

    # Run 5000 more steps — check for stability
    for n in range(5000):
        state = update_h(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state = update_e(state, materials, dt, dx, high_wall=unpadded_high_walls(grid))
        state = apply_pec(state)

        if n % 1000 == 999:
            e = em_energy(state)
            assert not np.isnan(e), f"NaN at step {200 + n + 1}"
            assert not np.isinf(e), f"Inf at step {200 + n + 1}"

    energy_final = em_energy(state)
    drift = abs(energy_final - energy_ref) / energy_ref

    print("\nLate-time stability (5000 steps, dt=0.99*CFL):")
    print(f"  Energy reference: {energy_ref:.4e}")
    print(f"  Energy final:     {energy_final:.4e}")
    print(f"  Drift: {drift * 100:.4f}%")

    assert drift < 0.05, f"Energy drift {drift*100:.2f}% exceeds 5%"
    assert not np.isnan(energy_final)
    assert not np.isinf(energy_final)
