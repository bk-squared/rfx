"""Eigenmode witness for the lattice ownership contract: volume vs sheet (#931).

WHAT THIS LOCK IS FOR
---------------------
The contract's own battery
(``tests/contracts/test_lattice_ownership_contract.py``) is edge-set level:
it reads ``realized_pec_edge_masks`` and compares boolean arrays. Nothing
in it steps a field, so nothing in it would notice if the realized edges
were right and the solver applied a different set. The only end-to-end
witness the tree ever had for WHERE a conductor's electrical wall lands
was ``TestEigenmodeWitness`` in ``tests/locks/test_two_plane_pec_slab.py``,
which #931 deleted along with the ``two_plane`` flag it guarded. This file
is that witness, rebuilt on the contract instead of on the flag.

THE MEASUREMENT
---------------
A parallel-plate cavity between two z-normal conductors inside a PEC box.
The lowest Ex mode is ``f = c/2 * sqrt((1/Lxy)^2 + (1/Lz')^2)`` with
``Lxy`` the lateral wall spacing and ``Lz'`` the PLATE SPACING — the
distance between the two facing electric walls. ``Lz'`` is the whole
question, and the two declarations answer it differently by exactly one
cell:

* **volume** (``sim.add(Box(one cell thick), material="pec")``, §1.2) — a
  body realizes tangential walls at BOTH bounding node planes and shorts
  the normal edge between them. The two slabs occupy cells ``k1`` and
  ``k2``; the cavity runs from the upper wall of the first (``k1+1``) to
  the lower wall of the second (``k2``), so ``Lz' = (k2 - k1 - 1)*dx``.
* **sheet** (``sim.add_thin_conductor(Box(zero extent))``, §1.3) — a foil
  is one node plane with its normal edge live. The planes are ``k1`` and
  ``k2``, so ``Lz' = (k2 - k1)*dx``.

One cell of plate spacing out of sixteen is a 4.7 % move in the mode, so
the two hypotheses are far apart compared with the 1 % gates: each leg is
pinned to its OWN prediction within 1 % and REQUIRED to be more than 3 %
from the other. A rule change that quietly moved a wall by one plane
would have to move BOTH legs the same way to survive, which the volume and
sheet paths do not share code to do.

MEASURED (this fixture, grid (33,33,25), dx = 200 um, slabs at z-cells
{3, 20}, 8000 steps, harminv over the [2000:] window, JAX cpu float32):

  volume : dominant 52.3341 GHz (Q 1.88e5) vs pred(16 cells) 52.3716
           -> -0.07 %; distance to pred(17) 49.9223 is 4.83 %
  sheet  : dominant 49.8924 GHz (Q 8.68e4) vs pred(17 cells) 49.9223
           -> -0.06 %; distance to pred(16) 52.3716 is 4.73 %

The sheet leg reproduces the DELETED lock's one-plane number to all four
recorded digits (49.8924 GHz), which is the evidence that a declared sheet
realizes what the old rule realized for a foil — the contract renamed that
behaviour and made it explicit, it did not move it.

WHAT ELSE THE REDRAW FIXED. The deleted fixture had to read its plate
indices back out of the raster because "this fixture's second sheet lands
one cell high of the naive coordinate arithmetic, measured": node
half-open sampling put a face-registered slab's cell one plane up when the
float arithmetic drifted by an ulp. Cell-CENTRE sampling (§1.1) puts a
face half a cell from the decision point, so the same coordinates now land
where they read: cells {3, 20}, asserted below as a fixed expectation
rather than read back.

Runtime ~6 s per leg on an idle CPU (measured 93 s / 102 s on a loaded
shared pod).
"""

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived (analytic parallel-plate ladder; the legs are FDTD + harminv)",
    "commit": "900cd79c",
    "date": "2026-09-07",
    "run_id": "local",
    "host": "shared pod, JAX_PLATFORMS=cpu float32",
    "pinned_until": "2027-03-07",
}

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation, harminv
from tests._realized_geometry import assert_sheet_planes, assert_wall_planes, realized

C0 = 299792458.0
DX = 0.2e-3
GAP_CELLS = 16
LAT_CELLS = 32

#: The two slab cells, fixed (not read back off the raster — see the module
#: docstring's note on centre sampling).
K1, K2 = 3, 20

#: Measured dominant modes, GHz. Re-derive by running this module, never by
#: editing toward a failing leg.
MEASURED_VOLUME_GHZ = 52.3341
MEASURED_SHEET_GHZ = 49.8924

_Z1_LO = K1 * DX
_Z1_HI = _Z1_LO + DX
_Z2_LO = _Z1_HI + GAP_CELLS * DX
_Z2_HI = _Z2_LO + DX
_LAT = LAT_CELLS * DX
_LZ = _Z2_HI + 3 * DX


def _build(kind: str) -> Simulation:
    """The same cavity twice; only the DECLARATION of the metal differs."""
    sim = Simulation(60e9, (_LAT, _LAT, _LZ), dx=DX, boundary="pec")
    if kind == "volume":
        sim.add(Box((0, 0, _Z1_LO), (_LAT, _LAT, _Z1_HI)), material="pec")
        sim.add(Box((0, 0, _Z2_LO), (_LAT, _LAT, _Z2_HI)), material="pec")
    elif kind == "sheet":
        # Zero-extent Box = "a sheet on this plane" (§1.3). The lo face of
        # each slab, so the two legs' metal starts at the same coordinate.
        sim.add_thin_conductor(Box((0, 0, _Z1_LO), (_LAT, _LAT, _Z1_LO)),
                               sigma_bulk=5.8e7)
        sim.add_thin_conductor(Box((0, 0, _Z2_LO), (_LAT, _LAT, _Z2_LO)),
                               sigma_bulk=5.8e7)
    else:
        raise AssertionError(kind)
    sim.add_source((11.1 * DX, 13.3 * DX, _Z1_HI + GAP_CELLS * DX * 0.31),
                   "ex", waveform=GaussianPulse(f0=50e9, bandwidth=0.5),
                   amplitude_kind="current")
    sim.add_probe((21.7 * DX, 8.9 * DX, _Z1_HI + GAP_CELLS * DX * 0.62), "ex")
    return sim


def _pred_ghz(sim, plate_cells: int) -> float:
    grid = sim._build_grid()
    lxy = (grid.shape[0] - 1) * DX      # tangential-E walls at planes 0 and n-1
    lz = plate_cells * DX
    return C0 / 2 * np.sqrt((1 / lxy) ** 2 + (1 / lz) ** 2) / 1e9


def _dominant_ghz(sim) -> float:
    res = sim.run(n_steps=8000)
    ts = np.asarray(res.time_series)[:, 0].ravel()
    modes = harminv(ts[2000:], float(res.dt), f_min=35e9, f_max=60e9)
    cand = sorted([m for m in modes if m.Q > 100],
                  key=lambda m: -abs(m.amplitude))
    assert cand, "harminv found no mode above Q=100 in [35, 60] GHz"
    return cand[0].freq / 1e9


# --------------------------------------------------------------- build-time --

def test_volume_slabs_realize_walls_on_both_drawn_faces():
    """No solve: the two 1-cell Boxes are volumes with four walls (§1.2)."""
    sim = _build("volume")
    assert_wall_planes(sim, 2, expected_planes=[K1, K1 + 1, K2, K2 + 1],
                       what="one-cell PEC slabs")
    rz = realized(sim)
    assert rz.sheet_planes == {}, (
        "a PEC Box through add() is a volume; it must declare no sheet")
    assert int(np.asarray(rz.pec_mask).sum()) > 0


def test_declared_sheets_realize_one_plane_each_and_own_no_cell():
    """No solve: the two zero-extent Boxes are sheets with two walls (§1.3)."""
    sim = _build("sheet")
    assert_sheet_planes(sim, 2, expected_planes=[K1, K2], what="foil sheets")
    assert_wall_planes(sim, 2, expected_planes=[K1, K2], what="foil sheets")
    rz = realized(sim)
    assert rz.pec_mask is None or not bool(np.asarray(rz.pec_mask).any()), (
        "a sheet owns no cell (§1.3), so the volume mask must stay empty")


def test_the_two_declarations_differ_by_exactly_one_plate_cell():
    """The whole point of making the choice explicit, at build time.

    Both legs declare metal starting at the same two coordinates. The
    volume leg's cavity is one cell SHORTER because the lower slab's far
    face is a wall. If this ever reads zero, the two declarations have
    collapsed into one and the physics legs below are measuring the same
    thing twice.
    """
    v = assert_wall_planes(_build("volume"), 2,
                           expected_planes=[K1, K1 + 1, K2, K2 + 1])
    s = assert_wall_planes(_build("sheet"), 2, expected_planes=[K1, K2])
    gap_volume = v[2] - v[1]        # k2 - (k1+1)
    gap_sheet = s[1] - s[0]         # k2 - k1
    assert gap_sheet - gap_volume == 1, (gap_volume, gap_sheet)
    assert gap_volume == GAP_CELLS


# ------------------------------------------------------------------ physics --

def test_volume_leg_sits_on_the_two_wall_ladder():
    """A 1-cell PEC Box is a filled slab: plate spacing (k2-k1-1)*dx."""
    sim = _build("volume")
    assert_wall_planes(sim, 2, expected_planes=[K1, K1 + 1, K2, K2 + 1])
    f = _dominant_ghz(sim)
    f_two = _pred_ghz(sim, K2 - K1 - 1)
    f_one = _pred_ghz(sim, K2 - K1)
    assert abs(f - MEASURED_VOLUME_GHZ) / MEASURED_VOLUME_GHZ < 0.002, f
    assert abs(f - f_two) / f_two < 0.01, (f, f_two)
    assert abs(f - f_one) / f_one > 0.03, (f, f_one)


def test_sheet_leg_sits_on_the_one_plane_ladder():
    """A declared sheet is one plane: plate spacing (k2-k1)*dx."""
    sim = _build("sheet")
    assert_sheet_planes(sim, 2, expected_planes=[K1, K2])
    f = _dominant_ghz(sim)
    f_one = _pred_ghz(sim, K2 - K1)
    f_two = _pred_ghz(sim, K2 - K1 - 1)
    assert abs(f - MEASURED_SHEET_GHZ) / MEASURED_SHEET_GHZ < 0.002, f
    assert abs(f - f_one) / f_one < 0.01, (f, f_one)
    assert abs(f - f_two) / f_two > 0.03, (f, f_two)
