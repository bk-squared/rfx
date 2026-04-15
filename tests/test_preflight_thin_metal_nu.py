"""Issue #48: preflight must warn when a thin PEC sits on a NU axis
without symmetric neighbouring cells (Meep/OpenEMS convention)."""

from __future__ import annotations

import math
import warnings as _w
import numpy as np

from rfx import Simulation, Box


def _has(issues, substring):
    return any(substring in i for i in issues)


def _build(dz_profile):
    h_sub = 1.5e-3
    sim = Simulation(
        freq_max=4e9, domain=(0.08, 0.075, 0), dx=1e-3,
        dz_profile=dz_profile, cpml_layers=8,
    )
    sim.add_material("fr4", eps_r=4.3)
    z_gnd_lo = 12e-3 - 0.25e-3
    z_sub_lo = 12e-3
    z_sub_hi = 12e-3 + h_sub
    z_patch_lo = z_sub_hi
    z_patch_hi = z_sub_hi + 0.25e-3
    sim.add(Box((0.010, 0.010, z_gnd_lo), (0.070, 0.065, z_sub_lo)),
            material="pec")
    sim.add(Box((0.010, 0.010, z_sub_lo), (0.070, 0.065, z_sub_hi)),
            material="fr4")
    sim.add(Box((0.025, 0.018, z_patch_lo), (0.054, 0.057, z_patch_hi)),
            material="pec")
    return sim


def test_asymmetric_metal_on_nu_triggers_warning():
    # Raw profile with sharp 1mm → 0.25mm → 1mm transitions. Metal planes
    # sit in cells with 4x larger neighbours — should warn.
    dz = np.concatenate([np.full(12, 1e-3), np.full(6, 0.25e-3),
                         np.full(25, 1e-3)])
    sim = _build(dz)
    issues = sim.preflight()
    assert _has(issues, "issue #48"), (
        f"expected issue #48 warning, got: {issues!r}"
    )


def test_symmetric_metal_on_nu_is_silent():
    # All-uniform 0.25mm z profile. Metal cells have symmetric neighbours.
    dz = np.full(60, 0.25e-3)
    sim = _build(dz)
    issues = sim.preflight()
    assert not _has(issues, "issue #48"), (
        f"uniform-dz profile triggered the asymmetric-metal warning; "
        f"issues: {issues!r}"
    )
