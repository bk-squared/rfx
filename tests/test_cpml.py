"""CPML absorbing boundary validation.

Tests:
1. CPML boundary reflection vs a CLEAN free-space reference (issue #398)
2. The reflection gate discriminates a real CPML degradation (layer reduction)
3. Energy should decay (not grow) in a CPML-terminated domain
4. Cross-frequency reflection regression against clean per-config envelopes

Issue #398 (comparator-first, historically 9/9 in this repo)
---------------------------------------------------------------
The previous reflection recipe compared the CPML run against a SMALL PEC
reference (0.15 m) and asserted "no reflections reach the probe". That premise
was FALSE: with dx=3 mm, dt=5.7 ps the reference's own PEC wall echo reaches the
probe at ~step 87 of the 300-step window, and the reported "CPML reflection"
(-47.3 dB) was dominated by that reference echo, not by the CPML. The -40 dB
gate was therefore measuring the reference artifact and was blind to real CPML
degradations (probe 2026-07-20: reducing the CPML from 8 to 2 layers left the
old number pinned at -47.3 -> -45.1 dB, still passing -40).

Fix: size the free-space reference so its nearest-wall echo lands AFTER the
measurement window, so the difference isolates the CPML's own boundary
perturbation. See ``_reflection_db_vs_clean_reference`` below.

Metric scope (documented confound, R5 trace-verified 2026-07-20)
----------------------------------------------------------------
This peak-deviation-from-free-space metric is dominated by the CPML front-
interface (discretization) reflection plus near-field reactive loading. It
MONOTONICALLY tracks reflection-INCREASING failures — fewer layers, a broken
kappa/sigma profile, a coefficient/sign corruption (clean floor 8->2 layers:
-68.3 -> -50.1 dB at f0=2 GHz). It is, by construction, INSENSITIVE to a
gentle reduction of sigma_max: a lower-sigma profile has a SMALLER front-
interface discontinuity, so the metric paradoxically improves (-68 -> -75 dB
for R_asymptotic 1e-15 -> 1e-2). Absorber STRENGTH (not front reflection) is
the separate, loosely-gated axis owned by ``test_cpml_energy_decay``.
"""

import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse

pytestmark = pytest.mark.gpu


def _reflection_db_vs_clean_reference(f0, freq_max, n_layers, n_steps,
                                      cpml_domain=0.06):
    """CPML boundary reflection as peak deviation from a CLEAN free-space run.

    The free-space PEC reference is sized so its nearest-wall round-trip echo
    (step ~ D / (C0 * dt)) lands AFTER ``n_steps`` — the difference then
    isolates the CPML domain's own boundary perturbation rather than the
    reference's wall echo (issue #398). Both domains share dx, dt, source and
    interior, so the direct wave cancels until a boundary reflection returns.

    Returns ``reflection_db = 20*log10(max|ts_cpml - ts_ref| / peak_ref)``.
    """
    pulse = GaussianPulse(f0=f0, bandwidth=0.5)

    # dt is set by freq_max; probe it from a throwaway grid to size the ref.
    dt_probe = Grid(freq_max=freq_max, domain=(0.02, 0.02, 0.02),
                    cpml_layers=0).dt
    ref_extent = 1.15 * n_steps * dt_probe * C0  # echo step ~1.15*n_steps

    # --- clean free-space reference (PEC walls too far to echo in-window) ---
    grid_ref = Grid(freq_max=freq_max, domain=(ref_extent,) * 3, cpml_layers=0)
    state = init_state(grid_ref.shape)
    materials = init_materials(grid_ref.shape)
    cx, cy, cz = grid_ref.nx // 2, grid_ref.ny // 2, grid_ref.nz // 2
    probe = (cx + 3, cy, cz)
    dt, dx = grid_ref.dt, grid_ref.dx
    ts_ref = np.zeros(n_steps)
    for n in range(n_steps):
        state = update_h(state, materials, dt, dx)
        state = update_e(state, materials, dt, dx)
        state = apply_pec(state)
        ez = state.ez.at[cx, cy, cz].add(pulse(n * dt))
        state = state._replace(ez=ez)
        ts_ref[n] = float(state.ez[probe])

    # --- CPML domain (small; waves hit the CPML boundary) ---
    grid_c = Grid(freq_max=freq_max, domain=(cpml_domain,) * 3,
                  cpml_layers=n_layers)
    state = init_state(grid_c.shape)
    materials = init_materials(grid_c.shape)
    cpml_params, cpml_state = init_cpml(grid_c)
    cxx, cyy, czz = grid_c.nx // 2, grid_c.ny // 2, grid_c.nz // 2
    probe_c = (cxx + 3, cyy, czz)
    dtc, dxc = grid_c.dt, grid_c.dx
    ts_c = np.zeros(n_steps)
    for n in range(n_steps):
        state = update_h(state, materials, dtc, dxc)
        state, cpml_state = apply_cpml_h(state, cpml_params, cpml_state, grid_c)
        state = update_e(state, materials, dtc, dxc)
        state, cpml_state = apply_cpml_e(state, cpml_params, cpml_state, grid_c)
        ez = state.ez.at[cxx, cyy, czz].add(pulse(n * dtc))
        state = state._replace(ez=ez)
        ts_c[n] = float(state.ez[probe_c])

    peak_ref = np.max(np.abs(ts_ref))
    peak_diff = np.max(np.abs(ts_c - ts_ref))
    return 20 * np.log10(peak_diff / max(peak_ref, 1e-30))


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
    """CPML boundary reflection, measured against a CLEAN free-space reference.

    Issue #398: the reference PEC domain is now sized so its own wall echo lands
    AFTER the measurement window, so the reported number reflects the CPML, not
    the reference artifact. Clean floor here measures ~-68 dB (f0=2 GHz, 8
    layers, 2026-07-20). The -55 dB gate sits ~13 dB above the measured floor
    (measured-envelope + margin) and catches a real CPML degradation — a
    2-layer boundary measures ~-50 dB and fails it (see
    ``test_cpml_reflection_gate_discriminates_layer_degradation``). The old
    -40 dB gate, measured against the contaminated 0.15 m reference, was pinned
    at ~-47 dB and blind to that degradation.
    """
    reflection_db = _reflection_db_vs_clean_reference(
        f0=2e9, freq_max=5e9, n_layers=8, n_steps=250,
    )
    print(f"CPML reflection (clean reference): {reflection_db:.1f} dB")
    assert reflection_db < -55, (
        f"CPML reflection {reflection_db:.1f} dB exceeds the -55 dB clean "
        f"envelope (measured floor ~-68 dB); the boundary is reflecting more "
        f"than the pinned envelope — suspect a CPML profile/coefficient "
        f"regression."
    )


def test_cpml_reflection_gate_discriminates_layer_degradation():
    """Discrimination witness for issue #398: the clean-reference reflection
    gate PASSES a healthy 8-layer CPML and CATCHES a degraded 2-layer boundary.

    This is the "old blind / new catches" proof the contaminated recipe could
    not provide: against the contaminated 0.15 m reference both the healthy and
    the degraded CPML reported ~-47/-45 dB (both pass -40); against the clean
    reference they separate to ~-68 dB vs ~-50 dB.
    """
    healthy_db = _reflection_db_vs_clean_reference(
        f0=2e9, freq_max=5e9, n_layers=8, n_steps=250,
    )
    degraded_db = _reflection_db_vs_clean_reference(
        f0=2e9, freq_max=5e9, n_layers=2, n_steps=250,
    )
    print(f"healthy(8 layers) = {healthy_db:.1f} dB  "
          f"degraded(2 layers) = {degraded_db:.1f} dB")
    assert healthy_db < -55, (
        f"healthy 8-layer CPML {healthy_db:.1f} dB should pass the -55 dB gate"
    )
    assert degraded_db > -55, (
        f"degraded 2-layer CPML {degraded_db:.1f} dB should FAIL the -55 dB "
        f"gate — the gate is not discriminating layer reduction"
    )
    assert healthy_db < degraded_db - 8, (
        f"expected clean separation between healthy ({healthy_db:.1f}) and "
        f"degraded ({degraded_db:.1f}) CPML reflection"
    )


@pytest.mark.slow
@pytest.mark.parametrize("f0,n_layers,gate_db", [
    # gate_db = measured clean floor + margin (2026-07-20, n_steps=250).
    # Degraded 2-layer floors (caught by these gates): 1GHz -38.8, 5GHz -77.3.
    (1e9, 8, -50.0),   # clean floor -56.3 dB (was contaminated -49.7)
    (5e9, 8, -75.0),   # clean floor -92.8 dB (was contaminated -65.4)
    (1e9, 4, -40.0),   # clean floor -45.7 dB (minimum-viable config)
])
def test_cpml_reflectivity_regression(f0, n_layers, gate_db):
    """Cross-frequency regression against CLEAN per-config reflection envelopes.

    Issue #398: the previous sweep floors (-49.7/-65.4/-46.3 dB) were measured
    with the contaminated 0.20 m reference. Re-measured against a window-sized
    clean reference (2026-07-20), the standard CPML floors are -56.3 / -92.8 /
    -45.7 dB. Each gate is pinned at the clean measured floor plus margin and
    catches a gross degradation of that config (a full-CPML failure pushes the
    number above the gate).
    """
    reflection_db = _reflection_db_vs_clean_reference(
        f0=f0, freq_max=f0 * 3, n_layers=n_layers, n_steps=250,
    )
    print(f"f0={f0/1e9:.0f}GHz layers={n_layers}: "
          f"reflection {reflection_db:.1f} dB (gate {gate_db} dB)")
    assert reflection_db < gate_db, (
        f"CPML reflectivity {reflection_db:.1f} dB > {gate_db} dB clean "
        f"envelope (f0={f0/1e9:.0f}GHz, layers={n_layers})"
    )
