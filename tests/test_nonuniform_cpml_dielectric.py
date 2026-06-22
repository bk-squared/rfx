"""Regression: non-uniform CPML must be material-aware (issue #205, nonuniform path).

The non-uniform scan body (rfx/nonuniform.py step_fn) called apply_cpml_h/e
WITHOUT the materials= argument, so the CPML fell back to free-space eps_0/mu_0
coefficients (rfx/boundaries/cpml.py). Inside a dielectric those coefficients are
eps_r-times too strong; when the dielectric fills the absorber region the CPML
update is unstable and the field diverges to NaN -- the same mechanism as the
uniform-path bug #203/#204, here on the non-uniform mesh.

Witnessed: a non-uniform (dz_profile) sim whose dielectric fills the whole
domain (so every CPML face is dielectric) diverges to all-NaN pre-fix at
eps_r=4 and eps_r=10, and becomes finite + cleanly absorbing (tail/peak ~1e-4)
once materials= is threaded into the two CPML calls (matching the production
uniform scan, rfx/simulation.py:1021-1023/1107-1109).

The production single-device uniform scan was already material-aware; this closes
the non-uniform sibling. (vmap-sweep and subgrid scan bodies carry the same
omission and remain tracked in #205 -- the vmap path needs separate vmap-shape
verification before the same change is applied there.)
"""

import numpy as np

from rfx import Box, Simulation
from rfx.sources.sources import GaussianPulse


def _nu_full_dielectric_sim(eps_r):
    """Non-uniform z-mesh, CPML boundary, dielectric filling the entire domain
    (every CPML cell is dielectric) -- the geometry that forces all absorption
    through dielectric-filled CPML faces."""
    dz_profile = np.concatenate([
        np.full(8, 0.5e-3), np.full(8, 1.0e-3), np.full(8, 0.5e-3),
    ])
    lz = float(dz_profile.sum())
    sim = Simulation(
        freq_max=10e9, domain=(0.03, 0.03, lz), dx=1.0e-3,
        boundary="cpml", cpml_layers=6, dz_profile=dz_profile,
    )
    sim.add_material("diel", eps_r=eps_r)
    sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="diel")  # fills domain + CPML
    sim.add_source((0.015, 0.015, lz / 2), "ez",
                   waveform=GaussianPulse(f0=4e9, bandwidth=0.7))
    sim.add_probe((0.022, 0.015, lz / 2), "ez")
    return sim


def test_nonuniform_cpml_dielectric_stable_and_absorbing():
    """NU CPML in a dielectric-filled domain must stay finite and absorb.

    Pre-fix (free-space CPML coefficients in the dielectric) this diverged to
    all-NaN; the materials-aware fix makes it finite and cleanly absorbing.
    """
    for eps_r in (4.0, 10.0):
        res = _nu_full_dielectric_sim(eps_r).run(n_steps=2500)
        ts = np.abs(np.asarray(res.time_series).reshape(-1))
        assert ts.size > 0
        assert np.all(np.isfinite(ts)), \
            f"NU CPML diverged (non-finite) in eps_r={eps_r} dielectric (issue #205)"
        peak = float(ts.max())
        tail = float(ts[-200:].max())
        # A working absorber drives the late-time field far below the peak.
        assert tail <= 0.05 * peak, \
            f"NU CPML did not absorb (eps_r={eps_r}): tail/peak={tail / max(peak, 1e-30):.3e}"
