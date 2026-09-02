"""UPML absorbing-boundary physics oracle (issue #403).

Gap (physics audit 2026-07-20): UPML is in the documented boundary baseline
(``docs/guides/support_matrix.md``: pec / cpml / upml) yet had NO reflection or
absorption physics gate in pytest. ``test_pml_reflectivity.py`` is CPML-only
(and gpu/slow), and the only UPML coverage lived OUTSIDE pytest (crossval
``01_waveguide_bend.py`` Meep T-match, weekly lane). No committed pytest oracle
could fail on wrong UPML physics.

This module adds two oracles, both built on the issue-#398 clean-reference
discipline (size the free-space reference so its own wall echo lands AFTER the
measurement window, isolating the boundary's OWN perturbation instead of the
reference's wall echo).

Oracle 1 — ``test_upml_reflection_floor_regression`` (a REGRESSION LOCK)
-----------------------------------------------------------------------
Peak deviation of the UPML-domain probe from a window-sized clean free-space run.

SCOPE / CONFOUND (measured on current main 2026-07-20, R5 trace-verified): for
UPML this peak-deviation metric is dominated by the front-interface
(discretization) reflection and is ANTI-correlated with absorber strength.
Measured (f0=2 GHz, 8 layers, 250 steps):

    sigma x0.2 -> -45.3 dB   sigma x1 -> -43.0   sigma x2 -> -42.3
    sigma x5   -> -41.4       sigma x10 -> -40.8   sigma x20 -> -40.4
    sigma x0 (dead) -> -63.6  |  8->2 layers moves it only 1.4 dB

i.e. a STRONGER sigma RAISES the number and a weaker/dead absorber LOWERS it, and
layer count barely matters. The metric is therefore a one-sided regression lock on
the front-interface envelope, NOT a falsifiable absorber-strength gate. Absorber
strength is owned by Oracle 2 (issue #403's explicit "delegate absorber-strength
to an energy-decay check"). The healthy 8-layer floor is -43.0 dB; CPML at the
identical config measures -68.3 dB (``test_cpml.py``), so rfx's UPML reflects
~25 dB more than its CPML here — a real characterization, consistent with the
UPML design note (the PML-parallel E component is attenuated only indirectly via
curl coupling, not by direct field stretching; see ``rfx/boundaries/upml.py``).

Oracle 2 — ``test_upml_interior_energy_decay`` + the dead-absorber discriminator
--------------------------------------------------------------------------------
The FALSIFIABLE absorber oracle: EM energy in the INTERIOR (non-PML) cells must
decay after the source stops, because outgoing waves leave the domain through the
UPML. A healthy 12-layer UPML drains the interior to about -17 dB; a loss-disabled
("dead", sigma=0) UPML and a PEC box trap the energy (~0 dB) and FAIL the gate.
That ~17 dB separation is the "old blind / new catches" proof that the gate is not
vacuous — a UPML regression that stopped absorbing would land near the dead number
and fail.
"""
import numpy as np
import pytest

from rfx.grid import Grid, C0
from rfx.core.yee import init_state, init_materials, update_e, update_h
from rfx.boundaries.upml import init_upml, apply_upml_e, apply_upml_h
from rfx.boundaries.pec import apply_pec
from rfx.sources.sources import GaussianPulse

# Free-space permittivity/permeability for the energy integral (SI).
_EPS_0 = 8.854187817e-12
_MU_0 = 1.2566370614e-6


# --------------------------------------------------------------------------- #
# Oracle 1 helper: reflection vs a CLEAN, window-sized free-space reference.
# --------------------------------------------------------------------------- #
def _upml_reflection_db_vs_clean_reference(f0, freq_max, n_layers, n_steps,
                                           upml_domain=0.06):
    """UPML boundary reflection as peak deviation from a CLEAN free-space run.

    Mirrors ``test_cpml.py::_reflection_db_vs_clean_reference`` but the small
    absorbing domain is stepped with ``apply_upml_h`` / ``apply_upml_e`` (which
    REPLACE the Yee update — they compute the curl themselves and, where sigma=0
    in the interior, reduce to the standard leapfrog, so the direct wave cancels
    against the reference until a boundary reflection returns).

    The free-space PEC reference is sized (``ref_extent``) so its nearest-wall
    round-trip echo lands AFTER ``n_steps`` (issue #398): the difference then
    isolates the UPML domain's own boundary perturbation, not the reference echo.
    """
    pulse = GaussianPulse(f0=f0, bandwidth=0.5)

    # dt is fixed by freq_max; probe it to size the reference domain.
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

    # --- small UPML domain (waves hit the UPML boundary) ---
    grid_u = Grid(freq_max=freq_max, domain=(upml_domain,) * 3,
                  cpml_layers=n_layers)
    state = init_state(grid_u.shape)
    materials = init_materials(grid_u.shape)
    coeffs = init_upml(grid_u, materials, axes="xyz")
    cxx, cyy, czz = grid_u.nx // 2, grid_u.ny // 2, grid_u.nz // 2
    probe_u = (cxx + 3, cyy, czz)
    dtu = grid_u.dt
    ts_u = np.zeros(n_steps)
    for n in range(n_steps):
        state = apply_upml_h(state, coeffs)
        state = apply_upml_e(state, coeffs)
        ez = state.ez.at[cxx, cyy, czz].add(pulse(n * dtu))
        state = state._replace(ez=ez)
        ts_u[n] = float(state.ez[probe_u])

    peak_ref = np.max(np.abs(ts_ref))
    peak_diff = np.max(np.abs(ts_u - ts_ref))
    return 20 * np.log10(peak_diff / max(peak_ref, 1e-30))


# --------------------------------------------------------------------------- #
# Oracle 2 helper: INTERIOR (non-PML) energy decay after source-off.
# --------------------------------------------------------------------------- #
def _upml_interior_energy_decay_db(n_layers, *, boundary="upml", f0=2e9,
                                   freq_max=3e9, domain=0.12, n_src=200,
                                   n_relax=600):
    """Energy in the interior (non-PML) cells: ratio of final to post-source.

    A working absorber lets outgoing waves LEAVE the domain, so interior energy
    decays. A reflecting boundary (``boundary='pec'``, or a sigma=0 "dead" UPML
    via monkeypatch) traps the energy in the interior, so it does not decay.
    Returns ``10*log10(E_interior_final / E_interior_after_source)`` in dB.
    """
    pulse = GaussianPulse(f0=f0, bandwidth=0.5)
    grid = Grid(freq_max=freq_max, domain=(domain,) * 3, cpml_layers=n_layers)
    state = init_state(grid.shape)
    materials = init_materials(grid.shape)
    coeffs = init_upml(grid, materials, axes="xyz") if boundary == "upml" else None
    cx, cy, cz = grid.nx // 2, grid.ny // 2, grid.nz // 2
    dt, dx = grid.dt, grid.dx

    # Interior = the non-PML core. For the PEC control keep a 2-cell margin so
    # the two boundaries integrate a comparable interior volume.
    lo = n_layers if boundary == "upml" else 2
    hi = grid.nx - lo
    sl = (slice(lo, hi),) * 3

    def e_interior(s):
        return float(
            0.5 * _EPS_0 * (s.ex[sl] ** 2 + s.ey[sl] ** 2 + s.ez[sl] ** 2).sum()
            + 0.5 * _MU_0 * (s.hx[sl] ** 2 + s.hy[sl] ** 2 + s.hz[sl] ** 2).sum()
        )

    def _step(s):
        if boundary == "upml":
            s = apply_upml_h(s, coeffs)
            s = apply_upml_e(s, coeffs)
            return s
        s = update_h(s, materials, dt, dx)
        s = update_e(s, materials, dt, dx)
        return apply_pec(s)

    for n in range(n_src):
        state = _step(state)
        ez = state.ez.at[cx, cy, cz].add(pulse(n * dt))
        state = state._replace(ez=ez)
    e_after = e_interior(state)

    for _ in range(n_relax):
        state = _step(state)
    e_final = e_interior(state)

    return 10 * np.log10(e_final / max(e_after, 1e-30))


# --------------------------------------------------------------------------- #
# Oracle 1: reflection-floor REGRESSION LOCK.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_upml_reflection_floor_regression():
    """UPML front-interface reflection stays at/below its measured envelope.

    Measured clean floor (current main, 2026-07-20): -43.0 dB (f0=2 GHz, 8
    layers, 250 steps). The -38 dB gate sits ~5 dB above that floor. This is a
    ONE-SIDED regression lock: per the module docstring the metric is front-
    interface dominated and anti-correlated with absorber strength, so it catches
    a gross reflection blow-up (solver/passivity regression) but NOT a subtle
    absorber degradation — that falsifier lives on the energy oracle below.
    """
    reflection_db = _upml_reflection_db_vs_clean_reference(
        f0=2e9, freq_max=5e9, n_layers=8, n_steps=250,
    )
    print(f"UPML reflection (clean reference): {reflection_db:.1f} dB "
          f"(CPML same config: -68.3 dB)")
    assert reflection_db < -38, (
        f"UPML reflection {reflection_db:.1f} dB exceeds the -38 dB envelope "
        f"(measured floor ~-43 dB); the boundary is reflecting more than the "
        f"pinned front-interface envelope — suspect a UPML coefficient / "
        f"passivity regression."
    )


# --------------------------------------------------------------------------- #
# Oracle 2: interior energy-decay absorber gate + its non-vacuity falsifier.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_upml_interior_energy_decay():
    """A healthy UPML drains interior energy after the source stops.

    Measured on current main (2026-07-20): 12-layer -17.4 dB, 8-layer -19.5 dB.
    The -12 dB gate sits ~5 dB above the 12-layer floor. The falsifier that this
    gate is not vacuous is ``..._discriminates_dead_absorber`` below (a sigma=0
    UPML and a PEC box stay near 0 dB and fail this same gate).
    """
    decay_db = _upml_interior_energy_decay_db(n_layers=12)
    print(f"UPML interior energy decay (12 layers): {decay_db:.1f} dB")
    assert decay_db < -12, (
        f"UPML interior energy decay {decay_db:.1f} dB is insufficient "
        f"(need < -12 dB); the boundary is not absorbing outgoing waves — "
        f"suspect a UPML sigma/coefficient regression."
    )


@pytest.mark.slow
def test_upml_energy_decay_gate_discriminates_dead_absorber(monkeypatch):
    """Non-vacuity witness (issue #398 discipline): the interior energy-decay
    gate PASSES a healthy UPML and CATCHES a broken (loss-disabled) absorber.

    A sigma=0 "dead" UPML (graded loss profile zeroed) and a PEC box both trap
    the outgoing wave in the interior (~0 dB decay) and FAIL the -12 dB gate the
    healthy 12-layer UPML clears (~-17 dB). This is the discrimination the
    reflection metric cannot provide for UPML (it is anti-correlated with
    absorber strength — see module docstring).
    """
    healthy_db = _upml_interior_energy_decay_db(n_layers=12)

    # PEC box: a fully reflecting boundary — the limiting "absorber does nothing".
    pec_db = _upml_interior_energy_decay_db(n_layers=12, boundary="pec")

    # Dead UPML: keep the full UPML update path but zero the graded sigma so the
    # boundary is lossless (a coefficient/profile regression that stops absorbing).
    def _zero_sigma(n, dt, dx, order=2, R_asymptotic=1e-15):
        z = np.zeros(n, dtype=np.float64)
        return z, z
    monkeypatch.setattr("rfx.boundaries.upml._sigma_profile_1d", _zero_sigma)
    dead_db = _upml_interior_energy_decay_db(n_layers=12)

    print(f"healthy(12-layer UPML) = {healthy_db:.1f} dB  "
          f"dead(sigma=0 UPML) = {dead_db:.1f} dB  "
          f"PEC box = {pec_db:.1f} dB")
    assert healthy_db < -12, (
        f"healthy 12-layer UPML {healthy_db:.1f} dB should pass the -12 dB gate"
    )
    assert dead_db > -6, (
        f"dead (sigma=0) UPML {dead_db:.1f} dB should FAIL the -12 dB gate — the "
        f"gate is not discriminating a non-absorbing boundary"
    )
    assert pec_db > -6, (
        f"PEC box {pec_db:.1f} dB should FAIL the -12 dB gate (fully reflecting)"
    )
    assert healthy_db < dead_db - 8, (
        f"expected clean separation between a healthy UPML ({healthy_db:.1f} dB) "
        f"and a dead absorber ({dead_db:.1f} dB)"
    )
