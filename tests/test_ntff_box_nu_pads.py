"""The NU-lane NTFF box must take its per-face CPML depths from the grid (#743).

A non-uniform grid carries no `face_layers`, so both the direct box
construction and `NTFFBox.from_grid` fall back to the scalar
`cpml_layers` on every face. That is wrong whenever the pads are
asymmetric — which they are for any non-absorbing face — and the error is
a rigid displacement of every NTFF face coordinate, i.e. a wrong phase
reference, returned with no warning.
"""
from __future__ import annotations

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import BoundarySpec
from rfx.sources import GaussianPulse


def _sim(z_lo: str):
    sim = Simulation(freq_max=40e9, domain=(6e-3, 6e-3, 6e-3), dx=300e-6,
                     boundary=BoundarySpec(x="cpml", y="cpml",
                                           z={"lo": z_lo, "hi": "cpml"}),
                     cpml_layers=6, dz_profile=np.full(20, 300e-6))
    sim.add(Box((2.7e-3, 2.7e-3, 2.7e-3), (3.3e-3, 3.3e-3, 3.3e-3)),
            material="pec")
    sim.add_source(position=(3e-3, 3e-3, 2.1e-3), component="ez",
                   amplitude_kind="current",
                   waveform=GaussianPulse(f0=30e9, bandwidth=0.5))
    sim.add_ntff_box((1.5e-3, 1.5e-3, 1.5e-3), (4.5e-3, 4.5e-3, 4.5e-3),
                     freqs=[30e9])
    return sim


def test_box_cpml_offsets_match_the_grids_pads():
    """A PEC z_lo face has pad_z_lo = 0; the box must say 0, not the scalar 6."""
    sim = _sim("pec")
    grid = sim._build_nonuniform_grid()
    assert int(grid.pad_z_lo) == 0 and int(grid.pad_z_hi) == 6, (
        "fixture assumption: a non-absorbing z_lo face carries no pad")
    res = sim.run(n_steps=120)
    box = res.ntff_box
    assert int(box.cpml_lo_z) == int(grid.pad_z_lo), (
        f"box.cpml_lo_z = {int(box.cpml_lo_z)} but the grid's z_lo pad is "
        f"{int(grid.pad_z_lo)} — face coordinates would be displaced by "
        f"{abs(int(box.cpml_lo_z) - int(grid.pad_z_lo))} cells in z")
    assert int(box.cpml_lo_x) == int(grid.pad_x_lo)
    assert int(box.cpml_lo_y) == int(grid.pad_y_lo)


def test_symmetric_case_is_unchanged():
    """With CPML on every face the pads equal the scalar, so nothing moves."""
    sim = _sim("cpml")
    grid = sim._build_nonuniform_grid()
    assert int(grid.pad_z_lo) == int(getattr(grid, "cpml_layers", 0))
    res = sim.run(n_steps=120)
    box = res.ntff_box
    assert int(box.cpml_lo_z) == int(grid.pad_z_lo) == 6
