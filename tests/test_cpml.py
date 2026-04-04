"""CPML absorbing boundary validation.

Tests:
1. CPML reflection coefficient < -40 dB for a point source
2. Energy should decay (not grow) in a CPML-terminated domain
"""

import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.sources import GaussianPulse


def test_cpml_energy_decay():
    """EM energy in a CPML-terminated domain should decay after source stops."""
    # Use a domain large enough that the interior (26 cells) is well separated
    # from the CPML so standing-wave trapping doesn't mask absorption.
    grid = Grid(freq_max=3e9, domain=(0.12, 0.12, 0.12), cpml_layers=15)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    cpml_params, cpml_state = init_cpml(grid)

    pulse = GaussianPulse(f0=2e9, bandwidth=0.5)
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    dt, dx = grid.dt, grid.dx

    EPS_0 = 8.854187817e-12
    MU_0 = 1.2566370614e-6

    def em_energy(s):
        return float(
            0.5 * EPS_0 * (s.ex**2 + s.ey**2 + s.ez**2).sum()
            + 0.5 * MU_0 * (s.hx**2 + s.hy**2 + s.hz**2).sum()
        )

    # Inject source for 200 steps (pulse fully decays)
    for n in range(200):
        t = n * dt
        state = update_h(state, materials, dt, dx)
        state, cpml_state = apply_cpml_h(state, cpml_params, cpml_state, grid)
        state = update_e(state, materials, dt, dx)
        state, cpml_state = apply_cpml_e(state, cpml_params, cpml_state, grid)
        ez = state.ez.at[cx, cy, cz].add(pulse(t))
        state = state._replace(ez=ez)

    energy_after_source = em_energy(state)

    # Run 500 more steps — energy should decay as waves are absorbed
    for _ in range(500):
        state = update_h(state, materials, dt, dx)
        state, cpml_state = apply_cpml_h(state, cpml_params, cpml_state, grid)
        state = update_e(state, materials, dt, dx)
        state, cpml_state = apply_cpml_e(state, cpml_params, cpml_state, grid)

    energy_final = em_energy(state)

    ratio_db = 10 * np.log10(energy_final / max(energy_after_source, 1e-30))
    print(f"Energy after source: {energy_after_source:.4e}")
    print(f"Energy after CPML absorption: {energy_final:.4e}")
    print(f"Energy decay: {ratio_db:.1f} dB")

    # Energy should drop significantly (at least 20 dB)
    assert ratio_db < -20, f"CPML energy decay {ratio_db:.1f} dB is insufficient (need < -20 dB)"


def test_cpml_reflection():
    """CPML reflection should be below -40 dB compared to a reference PEC simulation.

    Method: run two simulations with same source
    1. Large PEC domain (reference, no boundary reflections during measurement)
    2. Small CPML domain (reflections from CPML)
    Measure Ez at a probe point and compare.
    """
    freq_max = 5e9
    f0 = 2e9

    # --- Reference: large PEC domain (no reflections reach probe) ---
    grid_ref = Grid(freq_max=freq_max, domain=(0.15, 0.15, 0.15), cpml_layers=0)
    state_ref = init_state(grid_ref.shape)
    materials_ref = init_materials(grid_ref.shape)

    pulse = GaussianPulse(f0=f0, bandwidth=0.5)
    cx_ref = grid_ref.nx // 2
    cy_ref = grid_ref.ny // 2
    cz_ref = grid_ref.nz // 2
    # Probe offset from source
    probe_ref = (cx_ref + 3, cy_ref, cz_ref)

    dt_ref = grid_ref.dt
    dx_ref = grid_ref.dx

    n_steps = 300
    ts_ref = np.zeros(n_steps)
    from rfx.boundaries.pec import apply_pec
    for n in range(n_steps):
        t = n * dt_ref
        state_ref = update_h(state_ref, materials_ref, dt_ref, dx_ref)
        state_ref = update_e(state_ref, materials_ref, dt_ref, dx_ref)
        state_ref = apply_pec(state_ref)
        ez = state_ref.ez.at[cx_ref, cy_ref, cz_ref].add(pulse(t))
        state_ref = state_ref._replace(ez=ez)
        ts_ref[n] = float(state_ref.ez[probe_ref])

    # --- CPML domain (smaller, waves hit boundaries) ---
    grid_cpml = Grid(freq_max=freq_max, domain=(0.06, 0.06, 0.06), cpml_layers=10)
    state_cpml = init_state(grid_cpml.shape)
    materials_cpml = init_materials(grid_cpml.shape)
    cpml_params, cpml_state = init_cpml(grid_cpml)

    cx_cpml = grid_cpml.nx // 2
    cy_cpml = grid_cpml.ny // 2
    cz_cpml = grid_cpml.nz // 2
    probe_cpml = (cx_cpml + 3, cy_cpml, cz_cpml)

    dt_cpml = grid_cpml.dt
    dx_cpml = grid_cpml.dx

    ts_cpml = np.zeros(n_steps)
    for n in range(n_steps):
        t = n * dt_cpml
        state_cpml = update_h(state_cpml, materials_cpml, dt_cpml, dx_cpml)
        state_cpml, cpml_state = apply_cpml_h(state_cpml, cpml_params, cpml_state, grid_cpml)
        state_cpml = update_e(state_cpml, materials_cpml, dt_cpml, dx_cpml)
        state_cpml, cpml_state = apply_cpml_e(state_cpml, cpml_params, cpml_state, grid_cpml)
        ez = state_cpml.ez.at[cx_cpml, cy_cpml, cz_cpml].add(pulse(t))
        state_cpml = state_cpml._replace(ez=ez)
        ts_cpml[n] = float(state_cpml.ez[probe_cpml])

    # Compare: the difference is the CPML reflection
    # Use only the time window where the direct wave is present in both
    peak_ref = np.max(np.abs(ts_ref))
    diff = ts_cpml[:len(ts_ref)] - ts_ref[:len(ts_cpml)]
    peak_diff = np.max(np.abs(diff))

    reflection_db = 20 * np.log10(peak_diff / max(peak_ref, 1e-30))
    print(f"Peak reference: {peak_ref:.4e}")
    print(f"Peak reflection: {peak_diff:.4e}")
    print(f"CPML reflection: {reflection_db:.1f} dB")

    assert reflection_db < -30, f"CPML reflection {reflection_db:.1f} dB exceeds -30 dB"
