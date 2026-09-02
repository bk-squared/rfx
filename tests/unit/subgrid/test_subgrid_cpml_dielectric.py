"""Regression: material-aware CPML on the subgrid scan bodies (issue #205, last leg).

The subgrid coarse-grid scan bodies (``rfx/subgridding/jit_runner.py`` — the
``Simulation.add_refinement`` API lane — and the eager
``rfx/subgridding/runner.py``) called ``apply_cpml_h/e`` WITHOUT ``materials=``,
so the absorber used free-space eps_0/mu_0 coefficients regardless of the local
dielectric. When a dielectric overlaps the CPML region the free-space
coefficient is eps_r-times too strong and the scan diverges to NaN — the same
mechanism as the uniform extractor bug (#203/#204), the non-uniform scan body
(#208), the vmap sweep (#224), and the three distributed runners (#227-#229).
This was the last remaining CPML scan body without ``materials=``.

WHY ``validation="research"``: the production subgrid envelope structurally
rejects every CPML-adjacent configuration (measured 2026-07-03:
``z_slab_requires_guarded_boundary`` for centered slabs,
``boundary_terminated_requires_pec_no_cpml`` + ``subgrid_overlaps_absorber``
for boundary-touching slabs — even all-vacuum with x/y-only CPML). The buggy
code path is therefore reachable only through the research lane, which this
test exercises deliberately; it is a divergence regression, not a
physics-accuracy claim for the experimental subgrid lane.

Witness measured on the pre-fix tree (research lane, dielectric filling the
domain incl. all x/y CPML faces, z=PEC): eps_r=4 and eps_r=10 both diverge to
all-NaN, while the vacuum control stays finite and absorbing
(tail/peak ~1e-2). Post-fix all three are finite and absorbing; the vacuum
control changes only at float32 round-off (the materials-aware coefficient
path with all-vacuum arrays differs from the materials=None fallback by
~6e-8 absolute on ~1e-3-scale fields).
"""

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse


def _subgrid_dielectric_cpml_sim(eps_r):
    """Subgrid (z-slab refinement) sim with x/y CPML, z PEC, and a dielectric
    filling the entire domain — every x/y CPML cell is dielectric, forcing
    absorption through dielectric-filled CPML on the coarse grid."""
    lz = 0.016
    sim = Simulation(
        freq_max=10e9, domain=(0.02, 0.02, lz), dx=1.0e-3,
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="cpml", hi="cpml"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=5,
    )
    sim.add_refinement((0.0, 0.006), ratio=2, validation="research")
    if eps_r is not None:
        sim.add_material("diel", eps_r=eps_r)
        sim.add(Box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)), material="diel")
    sim.add_source((0.010, 0.010, 0.003), "ez",
                   waveform=GaussianPulse(f0=4e9, bandwidth=0.7))
    sim.add_probe((0.015, 0.010, 0.003), "ez")
    return sim


def test_subgrid_cpml_dielectric_stable_and_absorbing():
    """Subgrid coarse-grid CPML in a dielectric-filled domain must stay finite
    and absorb. Pre-fix (free-space CPML coefficients in the dielectric) this
    diverged to all-NaN at eps_r=4 and eps_r=10 (issue #205)."""
    for eps_r in (4.0, 10.0):
        res = _subgrid_dielectric_cpml_sim(eps_r).run(n_steps=1500)
        ts = np.abs(np.asarray(res.time_series).reshape(-1))
        assert ts.size > 0
        assert np.all(np.isfinite(ts)), \
            f"subgrid CPML diverged (non-finite) in eps_r={eps_r} dielectric (issue #205)"
        peak = float(ts.max())
        tail = float(ts[-150:].max())
        assert tail <= 0.05 * peak, \
            f"subgrid CPML did not absorb (eps_r={eps_r}): tail/peak={tail / max(peak, 1e-30):.3e}"


def test_subgrid_cpml_vacuum_control_still_absorbing():
    """Vacuum control: the materials-aware coefficient path with all-vacuum
    arrays must behave like the old materials=None fallback (same finite,
    absorbing run — only float32 round-off moves)."""
    res = _subgrid_dielectric_cpml_sim(None).run(n_steps=1500)
    ts = np.abs(np.asarray(res.time_series).reshape(-1))
    assert np.all(np.isfinite(ts))
    peak = float(ts.max())
    tail = float(ts[-150:].max())
    assert tail <= 0.05 * peak, \
        f"subgrid vacuum CPML control regressed: tail/peak={tail / max(peak, 1e-30):.3e}"
