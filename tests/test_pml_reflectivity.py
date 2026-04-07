"""PML broadband reflectivity must be < -40 dB.

Uses the reference-comparison method: run the same source/probe in a large
PEC domain (where boundary reflections cannot reach the probe during the
measurement window) and in a smaller CPML domain. The difference is the
PML reflection artifact.

Validates the standard CPML defaults (kappa_max=1.0, 8 layers).
Evidence: 56-run sweep (2026-04-07) showed standard CPML >= CFS-CPML,
and 8 layers achieves -49.7 dB at 1 GHz (worst case).
"""
import numpy as np

from rfx.grid import Grid
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse


import pytest


class TestPMLReflectivity:
    @pytest.mark.slow
    def test_broadband_reflectivity_below_minus_40db(self):
        """CPML reflection < -40 dB using large-PEC reference comparison.

        Uses kappa_max=1.0 and 8 layers — the unified defaults.
        """
        freq_max = 5e9
        f0 = 2e9
        n_steps = 400

        pulse = GaussianPulse(f0=f0, bandwidth=0.5)

        # --- Reference: large PEC domain (no reflections reach probe) ---
        grid_ref = Grid(freq_max=freq_max, domain=(0.20, 0.20, 0.20),
                        cpml_layers=0)
        state_ref = init_state(grid_ref.shape)
        materials_ref = init_materials(grid_ref.shape)

        cx_r = grid_ref.nx // 2
        cy_r = grid_ref.ny // 2
        cz_r = grid_ref.nz // 2
        probe_ref = (cx_r + 3, cy_r, cz_r)
        dt_r, dx_r = grid_ref.dt, grid_ref.dx

        ts_ref = np.zeros(n_steps)
        for n in range(n_steps):
            t = n * dt_r
            state_ref = update_h(state_ref, materials_ref, dt_r, dx_r)
            state_ref = update_e(state_ref, materials_ref, dt_r, dx_r)
            state_ref = apply_pec(state_ref)
            ez = state_ref.ez.at[cx_r, cy_r, cz_r].add(pulse(t))
            state_ref = state_ref._replace(ez=ez)
            ts_ref[n] = float(state_ref.ez[probe_ref])

        # --- CPML domain: standard CPML (kappa_max=1.0, 8 layers) ---
        grid_cpml = Grid(freq_max=freq_max, domain=(0.06, 0.06, 0.06),
                         cpml_layers=8)
        state_cpml = init_state(grid_cpml.shape)
        materials_cpml = init_materials(grid_cpml.shape)
        cpml_params, cpml_state = init_cpml(grid_cpml)

        cx_c = grid_cpml.nx // 2
        cy_c = grid_cpml.ny // 2
        cz_c = grid_cpml.nz // 2
        probe_cpml = (cx_c + 3, cy_c, cz_c)
        dt_c, dx_c = grid_cpml.dt, grid_cpml.dx

        ts_cpml = np.zeros(n_steps)
        for n in range(n_steps):
            t = n * dt_c
            state_cpml = update_h(state_cpml, materials_cpml, dt_c, dx_c)
            state_cpml, cpml_state = apply_cpml_h(
                state_cpml, cpml_params, cpml_state, grid_cpml)
            state_cpml = update_e(state_cpml, materials_cpml, dt_c, dx_c)
            state_cpml, cpml_state = apply_cpml_e(
                state_cpml, cpml_params, cpml_state, grid_cpml)
            ez = state_cpml.ez.at[cx_c, cy_c, cz_c].add(pulse(t))
            state_cpml = state_cpml._replace(ez=ez)
            ts_cpml[n] = float(state_cpml.ez[probe_cpml])

        # The difference between CPML and reference is the PML reflection
        peak_ref = np.max(np.abs(ts_ref))
        diff = ts_cpml[:n_steps] - ts_ref[:n_steps]
        peak_diff = np.max(np.abs(diff))

        reflectivity_db = 20 * np.log10(peak_diff / max(peak_ref, 1e-30))
        print(f"Peak reference: {peak_ref:.4e}")
        print(f"Peak reflection: {peak_diff:.4e}")
        print(f"CPML reflectivity: {reflectivity_db:.1f} dB")

        assert reflectivity_db < -40, (
            f"PML reflectivity {reflectivity_db:.1f} dB > -40 dB"
        )
